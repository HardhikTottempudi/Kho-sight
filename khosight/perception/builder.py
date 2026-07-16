"""Turns pixel-space detections into court-space FrameObservations.

Responsibilities: project to metres via the calibration homography, split teams
by jersey colour, classify posture, derive roles from game state (the chasing
team is the one seated along the centre line — rule 4.1.1), and infer seated
chasers' facing directions (rule 4.1.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..calibration.court import Court, CourtCalibration
from ..rules.events import FrameObservation, PlayerObservation, Posture, Team
from .models import PersonDetection


def classify_posture(det: PersonDetection) -> Posture:
    """Seated vs standing from pose geometry (image space).

    Heuristic for M0: hip-to-ankle vertical extent relative to bbox height.
    Seated players on the floor have hips near ankle level and squat bboxes.
    Replace with a small classifier trained on labelled Kho-Kho poses at M1.
    """
    x1, y1, x2, y2 = det.bbox
    h = max(y2 - y1, 1.0)
    aspect = (x2 - x1) / h
    hips = det.hips_mid()
    ankles = det.ankles()
    if hips and ankles:
        ankle_y = float(np.mean([a[1] for a in ankles]))
        hip_drop = (ankle_y - hips[1]) / h  # legs' share of body height
        if hip_drop < 0.22:
            return Posture.SEATED
        if hip_drop > 0.38:
            return Posture.STANDING
        return Posture.TRANSITION
    # keypoint-free fallback on bbox shape
    if aspect > 1.1:
        return Posture.SEATED
    if aspect < 0.75:
        return Posture.STANDING
    return Posture.TRANSITION


@dataclass
class TeamColourModel:
    """2-means over torso hue histograms; assigns each track to team 0/1 and
    smooths with a per-track vote so single-frame errors don't flip identity."""

    _centroids: Optional[np.ndarray] = None
    _votes: dict[int, list[int]] = field(default_factory=dict)

    @staticmethod
    def _torso_feature(frame_bgr: np.ndarray, det: PersonDetection) -> Optional[np.ndarray]:
        import cv2

        x1, y1, x2, y2 = det.torso_crop_box()
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        crop = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([crop], [0, 1], None, [12, 4], [0, 180, 0, 256])
        hist = hist.flatten()
        return hist / max(hist.sum(), 1e-6)

    def fit_or_update(
        self, frame_bgr: np.ndarray, detections: list[PersonDetection]
    ) -> dict[int, int]:
        feats, tids = [], []
        for d in detections:
            f = self._torso_feature(frame_bgr, d)
            if f is not None:
                feats.append(f)
                tids.append(d.track_id)
        if len(feats) < 4:
            return {t: self._majority(t) for t in tids}
        X = np.asarray(feats)
        if self._centroids is None:
            # k-means++ style init, k=2, few Lloyd iterations
            rng = np.random.default_rng(0)
            c0 = X[rng.integers(len(X))]
            d2 = ((X - c0) ** 2).sum(axis=1)
            c1 = X[int(np.argmax(d2))]
            self._centroids = np.stack([c0, c1])
        for _ in range(5):
            d = ((X[:, None, :] - self._centroids[None]) ** 2).sum(axis=2)
            lab = d.argmin(axis=1)
            for k in (0, 1):
                if (lab == k).any():
                    self._centroids[k] = 0.9 * self._centroids[k] + 0.1 * X[lab == k].mean(axis=0)
        d = ((X[:, None, :] - self._centroids[None]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        for tid, lab in zip(tids, labels):
            self._votes.setdefault(tid, []).append(int(lab))
            self._votes[tid] = self._votes[tid][-60:]
        return {t: self._majority(t) for t in tids}

    def _majority(self, tid: int) -> int:
        v = self._votes.get(tid, [0])
        return int(round(sum(v) / len(v)))


@dataclass
class ObservationBuilder:
    calibration: CourtCalibration
    court: Court = field(default_factory=Court)
    colours: TeamColourModel = field(default_factory=TeamColourModel)
    chase_cluster: Optional[int] = None  # which colour cluster is the chasing team
    _facing: dict[int, int] = field(default_factory=dict)

    def build(
        self,
        t: float,
        frame_idx: int,
        frame_bgr: np.ndarray,
        detections: list[PersonDetection],
        kho_call: bool = False,
    ) -> FrameObservation:
        cluster_of = self.colours.fit_or_update(frame_bgr, detections)
        players: list[PlayerObservation] = []
        provisional: list[tuple[PersonDetection, int, PlayerObservation]] = []

        for det in detections:
            (cx, cy) = self.calibration.image_to_court([det.ground_point()])[0]
            if not self.court.in_bounds((cx, cy), margin=2.0):
                continue  # spectators / bench
            feet = self.calibration.image_to_court(det.ankles()) if det.ankles() else []
            wrists = self.calibration.image_to_court(det.wrists()) if det.wrists() else []
            lean_x = 0.0
            sh, hp = det.shoulders_mid(), det.hips_mid()
            if sh and hp:
                sh_c, hp_c = self.calibration.image_to_court([sh, hp])
                lean_x = sh_c[0] - hp_c[0]
            obs = PlayerObservation(
                track_id=det.track_id,
                team=Team.RUN,  # provisional; fixed below
                pos=(cx, cy),
                posture=classify_posture(det),
                feet=feet,
                wrists=wrists,
                lean_x=lean_x,
                confidence=det.confidence,
            )
            provisional.append((det, cluster_of.get(det.track_id, 0), obs))

        self._resolve_chase_cluster(provisional)
        for det, cluster, obs in provisional:
            if det.role_class in ("chaser-seated", "chaser-active"):
                obs.team = Team.CHASE  # fine-tuned role detector wins (M1)
            elif det.role_class == "runner":
                obs.team = Team.RUN
            else:
                obs.team = Team.CHASE if cluster == self.chase_cluster else Team.RUN
            if obs.team == Team.CHASE and obs.posture == Posture.SEATED:
                obs.facing = self._infer_facing(obs)
            players.append(obs)

        return FrameObservation(
            t=t, frame_idx=frame_idx, players=players, kho_call=kho_call
        )

    def _resolve_chase_cluster(
        self, provisional: list[tuple[PersonDetection, int, PlayerObservation]]
    ) -> None:
        """The chasing team is the colour cluster with more players seated near
        the centre line (rule 4.1.1). Locked in once confident."""
        if self.chase_cluster is not None:
            return
        seated_near_centre = {0: 0, 1: 0}
        for _, cluster, obs in provisional:
            if (
                obs.posture == Posture.SEATED
                and abs(obs.pos[1] - self.court.centre_y) < 1.0
            ):
                seated_near_centre[cluster] += 1
        best = max(seated_near_centre, key=seated_near_centre.get)
        if seated_near_centre[best] >= 5:  # majority of the 8 seats visible
            self.chase_cluster = best

    def _infer_facing(self, obs: PlayerObservation) -> int:
        """Seat facing (rule 4.1.3): persist per track; infer from lean/wrists
        relative to the centre line, else alternate from neighbours at M1."""
        if obs.track_id in self._facing:
            return self._facing[obs.track_id]
        if obs.wrists:
            wy = float(np.mean([w[1] for w in obs.wrists]))
            facing = 1 if wy > self.court.centre_y else -1
            self._facing[obs.track_id] = facing
            return facing
        return 0
