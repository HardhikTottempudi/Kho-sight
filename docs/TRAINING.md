# Training Kho-Sight on your match videos

End-to-end guide from "folder of match videos" to "fine-tuned model in the
pipeline". Only **step 4 needs a GPU**, and it runs on free Google Colab —
your laptop just extracts frames and runs the finished model.

## The loop at a glance

```
match videos ─► extract frames ─► pre-label (auto) ─► correct in Roboflow
       ▲                                                     │
       │                                                     ▼
  add failure frames ◄── run pipeline, review ◄── train in Colab (GPU)
```

## Step 1 — extract frames (laptop, minutes)

```bash
python scripts/prepare_dataset.py frames --videos matches/ --out frames/ --every-s 2.0
```

Targets: **1,500–3,000 frames** to start. Spread them across as many different
*matches, venues, kits and camera angles* as you have — 100 frames each from 20
matches beats 2,000 frames from one match. If your laptop is too slow even for
this, Step 2b of the Colab notebook does it in the cloud from Google Drive.

## Step 2 — pre-label automatically (optional, saves ~80% of annotation time)

```bash
python scripts/auto_label.py --frames frames/ --out labels/ --model yolo11x.pt
```

A big pretrained model draws person boxes and guesses classes; you only correct
mistakes instead of drawing everything. Runs fine in Colab too (same script,
Step 2b in the notebook) if your laptop can't handle `yolo11x`.

## Step 3 — annotate in Roboflow (human, the real work)

1. Create a free [Roboflow](https://roboflow.com) **Object Detection** project.
2. Classes — exactly these names (the pipeline expects them):
   `chaser-seated`, `chaser-active`, `runner`, `cone`
3. Upload frames (with the pre-labels from step 2 if you made them) and correct.
4. Labelling rules:
   - `chaser-seated`: any chasing-team player seated/crouched on the centre line
     — box the whole body. This is the class COCO models miss; it matters most.
   - `chaser-active`: the one standing/running chaser (rule 4.1.4 — there is
     only ever one, except mid-kho swaps: label both as active during a swap).
   - `runner`: on-court runners only; ignore waiting batches beside the referee.
   - `cone`: both cones, tight boxes.
   - Box partially occluded players too (estimate the hidden extent).
5. Generate a dataset version: **split by match, not randomly** if possible
   (Roboflow: assign train/valid at upload time using filename prefixes) —
   random splits leak near-duplicate frames and inflate your metrics.
   Suggested split: ~85% train / 15% valid, with 2+ whole matches held out for
   validation. Skip Roboflow's augmentations (training does its own).
6. Export as **YOLOv11** format.

## Step 4 — train on Google Colab (GPU, ~2 hours)

Open [`notebooks/train_khosight_colab.ipynb`](../notebooks/train_khosight_colab.ipynb)
in Colab (GitHub → Open in Colab, or upload it), set **Runtime → T4 GPU**, and
run top to bottom. It pulls the dataset straight from Roboflow, trains
`yolo11s` at `imgsz=1280`, validates, and saves `khosight_roles_best.pt` to
your Google Drive.

What "good" looks like: **mAP50 ≥ 0.85 per class**. Watch `chaser-seated`
specifically — if it lags, add more seated-row frames (crowded, occluded ones).

## Step 5 — plug it into the pipeline (laptop, CPU is fine)

```bash
python -m khosight analyze --video match.mp4 --calib court_calib.json \
    --team-a Lions --team-b Tigers --half Lions:0:180 --half Tigers:240:420 \
    --role-model khosight_roles_best.pt --out report
```

With `--role-model`, team/role assignment comes from your fine-tuned detector
instead of jersey-colour clustering; the pretrained pose model keeps providing
keypoints for postures, tags and line calls.

## Step 6 — iterate (this is where accuracy comes from)

1. Analyse a match, read `report.md`, and note wrong/missed events.
2. Grab frames around each mistake, add them to the Roboflow project, correct.
3. Retrain (the notebook again — subsequent runs are faster with `patience`).
4. Repeat 2–3 times. Then start calibrating rule thresholds
   (`RuleEngineConfig`: `tag_radius`, `backtrack_tolerance`, …) against clips
   where you know the correct call.

## Later (M1+, same pattern)

- **"Kho" audio spotter**: `python scripts/prepare_dataset.py audio --videos matches/`
  extracts WAVs; label 1 s windows containing the call; train a small
  keyword-spotting CNN (Colab again).
- **Pose fine-tuning**: only if wrist/ankle precision limits tag detection —
  keypoint annotation is expensive, so exhaust the role-detector loop first.

## Colab practicalities

- Free tier gives a T4 for a few hours at a time — enough for each training
  run here. Keep the tab open; if it disconnects mid-train, re-run the train
  cell with `resume=True`.
- Everything worth keeping (weights, dataset zips) goes to Google Drive in the
  notebook — Colab's local disk is wiped between sessions.
- Don't commit `.pt` weights to git (already in `.gitignore`); keep them in
  Drive or GitHub Releases.
