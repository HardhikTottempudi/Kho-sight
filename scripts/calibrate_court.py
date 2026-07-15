"""Interactive court calibration: click known court landmarks on a video frame.

Usage:  python scripts/calibrate_court.py --video match.mp4 --out court_calib.json

You'll be shown a frame and a sequence of named court landmarks; click each in
turn (right-click to skip one you can't see). Needs >= 4 clicked points; more
points = a better-conditioned homography. Re-run whenever the camera moves.
"""

from __future__ import annotations

import argparse

# (name, court x, court y) — see khosight/calibration/court.py for the frame
LANDMARKS = [
    ("corner: end A / side y=0", 0.0, 0.0),
    ("corner: end A / side y=9", 0.0, 9.0),
    ("corner: end B / side y=0", 18.0, 0.0),
    ("corner: end B / side y=9", 18.0, 9.0),
    ("free-zone line A / side y=0", 2.0, 0.0),
    ("free-zone line A / side y=9", 2.0, 9.0),
    ("free-zone line B / side y=0", 16.0, 0.0),
    ("free-zone line B / side y=9", 16.0, 9.0),
    ("cone A (centre line x=2)", 2.0, 4.5),
    ("cone B (centre line x=16)", 16.0, 4.5),
    ("centre line / end line A", 0.0, 4.5),
    ("centre line / end line B", 18.0, 4.5),
]


def run_calibration(video: str, out: str, frame_idx: int = 0) -> None:
    import cv2

    from khosight.calibration.court import CourtCalibration

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"cannot read frame {frame_idx} of {video}")

    clicks: list[tuple[float, float] | None] = []
    current = {"i": 0}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and current["i"] < len(LANDMARKS):
            clicks.append((float(x), float(y)))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            current["i"] += 1
        elif event == cv2.EVENT_RBUTTONDOWN and current["i"] < len(LANDMARKS):
            clicks.append(None)  # skip this landmark
            current["i"] += 1

    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibrate", on_mouse)
    while current["i"] < len(LANDMARKS):
        disp = frame.copy()
        name = LANDMARKS[current["i"]][0]
        cv2.putText(disp, f"Click: {name}  (right-click = skip, q = abort)",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("calibrate", disp)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            raise SystemExit("aborted")
    cv2.destroyAllWindows()

    img_pts, court_pts = [], []
    for click, (_, cx, cy) in zip(clicks, LANDMARKS):
        if click is not None:
            img_pts.append(click)
            court_pts.append((cx, cy))
    if len(img_pts) < 4:
        raise SystemExit(f"need >= 4 points, got {len(img_pts)}")

    calib = CourtCalibration.from_correspondences(
        img_pts, court_pts, image_size=(frame.shape[1], frame.shape[0])
    )
    calib.save(out)

    # report reprojection sanity
    back = calib.image_to_court(img_pts)
    errs = [((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
            for a, b in zip(back, court_pts)]
    print(f"Saved {out} using {len(img_pts)} points; "
          f"max residual {max(errs):.3f} m, mean {sum(errs)/len(errs):.3f} m")
    if max(errs) > 0.15:
        print("WARNING: residual > 15 cm — re-click points or add more landmarks.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="court_calib.json")
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()
    run_calibration(args.video, args.out, args.frame)
