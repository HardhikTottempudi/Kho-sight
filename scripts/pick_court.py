"""Click the court outline once per match video -> court_regions.json.

The camera is fixed within a match, so the court occupies the same pixels in
every frame from that video. This tool shows one frame per match (grouped by
frame filename prefix) and lets you click the court's corners; the polygons are
saved and used by auto_label.py / the Colab autolabel notebook to drop
detections of people OUTSIDE the court (spectators, waiting batches, officials).

Usage:
  python scripts/pick_court.py --frames frames --out court_regions.json

Controls: left-click = add corner (4+ points, go around the court boundary in
order), u = undo, n = done with this match / next, q = abort.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict


def video_prefix(stem: str) -> str:
    """frames are named <video-stem>_<frameidx>; strip the trailing index."""
    return stem.rsplit("_", 1)[0]


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="court_regions.json")
    args = ap.parse_args()

    by_video: dict[str, list[pathlib.Path]] = defaultdict(list)
    for p in sorted(pathlib.Path(args.frames).iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            by_video[video_prefix(p.stem)].append(p)
    if not by_video:
        raise SystemExit(f"no frames found in {args.frames}")

    regions: dict[str, list[list[float]]] = {}
    if pathlib.Path(args.out).exists():
        regions = json.load(open(args.out))
        print(f"loaded {len(regions)} existing regions from {args.out}")

    for prefix, files in by_video.items():
        if prefix in regions:
            print(f"skip {prefix} (already picked)")
            continue
        frame = cv2.imread(str(files[len(files) // 2]))  # mid-match frame
        pts: list[list[float]] = []

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pts.append([float(x), float(y)])

        win = "pick court corners"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)
        print(f"\n{prefix}: click the court corners in order "
              "(4+ points), then press n. u = undo, q = abort.")
        while True:
            disp = frame.copy()
            cv2.putText(disp, f"{prefix}  |  {len(pts)} pts  |  n=next u=undo q=quit",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            for i, (x, y) in enumerate(pts):
                cv2.circle(disp, (int(x), int(y)), 6, (0, 0, 255), -1)
                if i:
                    cv2.line(disp, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                             (int(x), int(y)), (0, 255, 0), 2)
            if len(pts) >= 3:
                cv2.line(disp, (int(pts[-1][0]), int(pts[-1][1])),
                         (int(pts[0][0]), int(pts[0][1])), (0, 255, 0), 1)
            cv2.imshow(win, disp)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("n") and len(pts) >= 4:
                regions[prefix] = pts
                break
            if k == ord("u") and pts:
                pts.pop()
            if k == ord("q"):
                cv2.destroyAllWindows()
                raise SystemExit("aborted")
        cv2.destroyAllWindows()
        json.dump(regions, open(args.out, "w"), indent=2)  # save as we go

    print(f"\nSaved {len(regions)} court regions to {args.out}")
    print("Use with:  python scripts/auto_label.py ... --regions", args.out)


if __name__ == "__main__":
    main()
