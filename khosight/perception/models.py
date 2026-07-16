"""Detection + tracking + pose wrapper.

Uses Ultralytics YOLO-pose with ByteTrack/BoT-SORT. Imports are lazy so the
rules layer and tests never require torch. Swap `model_name` for a fine-tuned
checkpoint at M1 (`scripts/train_detector.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# COCO keypoint indices
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_ANKLE, KP_R_ANKLE = 15, 16
KP_CONF_MIN = 0.35


# classes for the fine-tuned role detector (M1) — must match dataset data.yaml
ROLE_CLASSES = ("chaser-seated", "chaser-active", "runner", "cone")


@dataclass
class PersonDetection:
    """One tracked person in one frame, in image pixels."""

    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    keypoints: Optional[np.ndarray] = None   # (17, 3) x, y, conf
    role_class: Optional[str] = None         # from the fine-tuned role detector

    def _kp(self, idx: int) -> Optional[tuple[float, float]]:
        if self.keypoints is None or self.keypoints[idx, 2] < KP_CONF_MIN:
            return None
        return float(self.keypoints[idx, 0]), float(self.keypoints[idx, 1])

    def ankles(self) -> list[tuple[float, float]]:
        return [p for p in (self._kp(KP_L_ANKLE), self._kp(KP_R_ANKLE)) if p]

    def wrists(self) -> list[tuple[float, float]]:
        return [p for p in (self._kp(KP_L_WRIST), self._kp(KP_R_WRIST)) if p]

    def shoulders_mid(self) -> Optional[tuple[float, float]]:
        pts = [p for p in (self._kp(KP_L_SHOULDER), self._kp(KP_R_SHOULDER)) if p]
        return tuple(np.mean(pts, axis=0)) if pts else None

    def hips_mid(self) -> Optional[tuple[float, float]]:
        pts = [p for p in (self._kp(KP_L_HIP), self._kp(KP_R_HIP)) if p]
        return tuple(np.mean(pts, axis=0)) if pts else None

    def ground_point(self) -> tuple[float, float]:
        """Best estimate of ground contact in pixels: mean ankle, else bbox bottom-centre."""
        ankles = self.ankles()
        if ankles:
            return tuple(np.mean(ankles, axis=0))
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    def torso_crop_box(self) -> tuple[int, int, int, int]:
        """Central torso region for jersey-colour team classification."""
        x1, y1, x2, y2 = self.bbox
        w, h = x2 - x1, y2 - y1
        return (
            int(x1 + 0.25 * w), int(y1 + 0.2 * h),
            int(x2 - 0.25 * w), int(y1 + 0.55 * h),
        )


def bbox_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def assign_roles(
    detections: list["PersonDetection"],
    role_boxes: list[tuple[tuple[float, float, float, float], str]],
    iou_min: float = 0.4,
) -> None:
    """Attach fine-tuned role-detector classes to tracked persons by best IoU."""
    for det in detections:
        best_iou, best_cls = iou_min, None
        for bbox, cls in role_boxes:
            if cls == "cone":
                continue  # cones are static court furniture, not players
            i = bbox_iou(det.bbox, bbox)
            if i > best_iou:
                best_iou, best_cls = i, cls
        det.role_class = best_cls


@dataclass
class PerceptionModel:
    """YOLO-pose + tracker. `track_frame` returns per-frame PersonDetections
    with persistent track ids.

    `role_model_name` optionally adds a fine-tuned Kho-Kho role detector
    (classes: ROLE_CLASSES, trained via notebooks/train_khosight_colab.ipynb).
    Its classes ride along on each detection and replace jersey-colour
    clustering for team assignment.
    """

    model_name: str = "yolo11n-pose.pt"
    role_model_name: Optional[str] = None  # e.g. "khosight_roles_best.pt" (M1)
    tracker: str = "bytetrack.yaml"  # or "botsort.yaml" (ReID, more robust to occlusion)
    conf: float = 0.25
    role_conf: float = 0.35
    device: Optional[str] = None     # None = auto
    _model: Any = field(default=None, repr=False)
    _role_model: Any = field(default=None, repr=False)

    def _ensure_model(self) -> None:
        if self._model is None:
            from ultralytics import YOLO  # lazy: keeps rules layer torch-free

            self._model = YOLO(self.model_name)
        if self._role_model is None and self.role_model_name:
            from ultralytics import YOLO

            self._role_model = YOLO(self.role_model_name)

    def track_frame(self, frame_bgr: np.ndarray) -> list[PersonDetection]:
        self._ensure_model()
        results = self._model.track(
            frame_bgr,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            classes=[0],  # person
            device=self.device,
            verbose=False,
        )[0]
        detections: list[PersonDetection] = []
        if results.boxes is None or results.boxes.id is None:
            return detections
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()
        kps = (
            results.keypoints.data.cpu().numpy()
            if results.keypoints is not None
            else [None] * len(ids)
        )
        for bbox, tid, c, kp in zip(boxes, ids, confs, kps):
            detections.append(
                PersonDetection(
                    track_id=int(tid),
                    bbox=tuple(map(float, bbox)),
                    confidence=float(c),
                    keypoints=kp,
                )
            )
        if self._role_model is not None:
            r = self._role_model.predict(
                frame_bgr, conf=self.role_conf, device=self.device, verbose=False
            )[0]
            role_boxes = [
                (tuple(map(float, b)), r.names[int(k)])
                for b, k in zip(
                    r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()
                )
            ]
            assign_roles(detections, role_boxes)
        return detections
