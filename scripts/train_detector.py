"""Fine-tune the detector/pose model on annotated Kho-Kho frames (M1).

Expects a YOLO-format dataset (from CVAT/Label Studio/Roboflow export):

  dataset/
    data.yaml        # names: [person-chaser-seated, person-chaser-active,
                     #         person-runner, cone]
    images/{train,val}/...
    labels/{train,val}/...

Usage:
  python scripts/train_detector.py --data dataset/data.yaml --base yolo11m-pose.pt \\
      --epochs 80 --imgsz 1280

Notes:
- Start from the pose model so keypoints stay available; a detect-only model
  loses posture/lean/wrist signals the rule engine depends on.
- imgsz 1280: seated chasers are small at court-camera distance.
- After training, pass the best.pt path as --model to `khosight analyze/live`.
"""

from __future__ import annotations

import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="yolo11m-pose.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        degrees=5, scale=0.3, mosaic=0.5,  # court cameras are static; mild aug
        project="runs/khosight",
    )
    metrics = model.val()
    print(metrics)
