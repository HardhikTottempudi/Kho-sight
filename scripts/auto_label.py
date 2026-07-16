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
import pathlib

from ultralytics import YOLO

SEATED_ASPECT = 0.9  # w/h above this => probably seated/crouched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo11x.pt", help="big model = better pre-labels")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    frames = sorted(
        p for p in pathlib.Path(args.frames).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for i, img in enumerate(frames):
        res = model.predict(str(img), conf=args.conf, classes=[0], verbose=False)[0]
        h, w = res.orig_shape
        lines = []
        for box in res.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            cls = 0 if (bw / max(bh, 1)) > SEATED_ASPECT else 2  # seated vs runner guess
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        (out / f"{img.stem}.txt").write_text("\n".join(lines))
        if i % 100 == 0:
            print(f"{i}/{len(frames)} labelled")
    print(f"Wrote {len(frames)} label files to {out}. "
          "Import into Roboflow/CVAT and correct classes/boxes.")


if __name__ == "__main__":
    main()
