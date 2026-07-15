"""Mine training data from your match videos (M1).

1) Frames for detector/pose fine-tuning:
     python scripts/prepare_dataset.py frames --videos dir_of_matches/ --out dataset/frames \\
         --every-s 2.0
   Then annotate in CVAT / Label Studio / Roboflow with classes:
     person-chaser-seated, person-chaser-active, person-runner, cone
   Export YOLO format and train with scripts/train_detector.py.

2) Audio clips for the "kho" keyword spotter:
     python scripts/prepare_dataset.py audio --videos dir_of_matches/ --out dataset/audio
   Extracts 16 kHz mono WAVs; label 1-second windows containing "kho" calls.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


def extract_frames(videos_dir: str, out_dir: str, every_s: float) -> None:
    import cv2

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for vid in sorted(pathlib.Path(videos_dir).iterdir()):
        if vid.suffix.lower() not in VIDEO_EXTS:
            continue
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * every_s)))
        i = saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                cv2.imwrite(str(out / f"{vid.stem}_{i:07d}.jpg"), frame)
                saved += 1
            i += 1
        cap.release()
        print(f"{vid.name}: saved {saved} frames")


def extract_audio(videos_dir: str, out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for vid in sorted(pathlib.Path(videos_dir).iterdir()):
        if vid.suffix.lower() not in VIDEO_EXTS:
            continue
        wav = out / f"{vid.stem}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(vid), "-ac", "1", "-ar", "16000", str(wav)],
            check=True, capture_output=True,
        )
        print(f"{vid.name} -> {wav.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames")
    f.add_argument("--videos", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--every-s", type=float, default=2.0)
    a = sub.add_parser("audio")
    a.add_argument("--videos", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "frames":
        extract_frames(args.videos, args.out, args.every_s)
    else:
        extract_audio(args.videos, args.out)
