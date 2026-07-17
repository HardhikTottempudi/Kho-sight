"""Bootstrap annotations: pre-label extracted frames with a pretrained model.

Runs a large pretrained YOLO on your frames and writes YOLO-format label files
guessing the role of each person from scene context (seated near other people
in a row = chaser-seated, etc.). Import frames+labels into Roboflow/CVAT and
CORRECT them — reviewing pre-drawn boxes is 5-10x faster than drawing from
scratch.

Usage (also runnable in Colab, see notebooks/train_khosight_colab.ipynb):
  python scripts/auto_label.py --frames dataset/frames --out dataset/labels \\
      --model yolo11x.pt

Classes written (ids match ROLE_CLASSES order):
  0 chaser-seated  1 chaser-active  2 runner  3 cone
The person-role guesses are heuristic (aspect ratio for seated vs standing);
expect to fix ~20-30% of them — that is the point of the human pass.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from ultralytics import YOLO

SEATED_ASPECT = 0.9  # w/h above this => probably seated/crouched


def point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin:
                inside = not inside
    return inside


def on_court(box_xyxy, poly: list[list[float]], margin_px: float) -> bool:
    """Keep a detection if its feet (bbox bottom-centre) are inside the court
    polygon, with a pixel margin so boundary-straddling players survive."""
    x1, y1, x2, y2 = box_xyxy
    fx, fy = (x1 + x2) / 2.0, y2
    if point_in_polygon(fx, fy, poly):
        return True
    return any(
        point_in_polygon(fx + dx, fy + dy, poly)
        for dx in (-margin_px, 0, margin_px)
        for dy in (-margin_px, 0, margin_px)
    )


def video_prefix(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo11x.pt", help="big model = better pre-labels")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--regions", default=None,
                    help="court_regions.json from scripts/pick_court.py; drops "
                    "detections outside the court polygon (spectators etc.)")
    ap.add_argument("--margin-px", type=float, default=15.0,
                    help="court-polygon tolerance so boundary players are kept")
    args = ap.parse_args()

    regions = json.load(open(args.regions)) if args.regions else {}
    fine_tuned = "khosight" in args.model or "best" in args.model
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    frames = sorted(
        p for p in pathlib.Path(args.frames).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    dropped = 0
    for i, img in enumerate(frames):
        poly = regions.get(video_prefix(img.stem))
        if args.regions and poly is None:
            print(f"WARNING: no court region for {video_prefix(img.stem)}; not filtering")
        # COCO bootstrap: persons only; fine-tuned model: all 4 classes
        classes = None if fine_tuned else [0]
        res = model.predict(str(img), conf=args.conf, classes=classes, verbose=False)[0]
        h, w = res.orig_shape
        lines = []
        for box, kcls in zip(
            res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()
        ):
            x1, y1, x2, y2 = box
            if poly is not None and not on_court(box, poly, args.margin_px):
                dropped += 1
                continue
            bw, bh = x2 - x1, y2 - y1
            if fine_tuned:
                cls = int(kcls)
            else:
                cls = 0 if (bw / max(bh, 1)) > SEATED_ASPECT else 2  # seated/runner guess
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        (out / f"{img.stem}.txt").write_text("\n".join(lines))
        if i % 100 == 0:
            print(f"{i}/{len(frames)} labelled")
    # data.yaml so Roboflow maps class ids -> names correctly on import
    if fine_tuned:
        names = [model.names[k] for k in sorted(model.names)]
    else:
        names = ["chaser-seated", "chaser-active", "runner", "cone"]
    (out.parent / "data.yaml").write_text(
        f"train: images\nval: images\nnc: {len(names)}\nnames: {names}\n"
    )
    print(f"Wrote {len(frames)} label files to {out}"
          + (f" (dropped {dropped} off-court detections)" if args.regions else "")
          + f" and {out.parent / 'data.yaml'}."
          " Import into Roboflow/CVAT and correct classes/boxes.")


if __name__ == "__main__":
    main()
