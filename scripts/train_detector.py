"""Fine-tune the Kho-Kho role detector on annotated frames (M1).

This trains the 4-class role detector used via `--role-model`:
  classes: chaser-seated, chaser-active, runner, cone
The pretrained YOLO-pose model keeps providing keypoints; the role detector
fixes the two things COCO models get wrong on Kho-Kho footage: missing seated
chasers, and telling the teams/roles apart.

Prefer running this in Google Colab (free GPU):
  notebooks/train_khosight_colab.ipynb  — same training, zero local setup.

Local usage (needs a CUDA GPU):
  python scripts/train_detector.py --data dataset/data.yaml --epochs 100

Expected dataset (YOLO format, e.g. Roboflow export):
  dataset/
    data.yaml            # names: [chaser-seated, chaser-active, runner, cone]
    train/images  train/labels
    valid/images  valid/labels
"""

from __future__ import annotations

import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="yolo11s.pt",
                    help="yolo11s balances accuracy/speed; yolo11m if GPU allows")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="large imgsz: seated chasers are small at court distance")
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto-fit to GPU memory")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=25,
        degrees=5, scale=0.3, mosaic=0.5,  # court cameras are static; mild aug
        project="runs/khosight",
        name="roles",
    )
    metrics = model.val()
    print(metrics)
    print("\nBest weights: runs/khosight/roles/weights/best.pt")
    print("Use with:  python -m khosight analyze ... --role-model <best.pt>")
