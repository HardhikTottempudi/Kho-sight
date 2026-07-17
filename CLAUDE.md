# CLAUDE.md — Kho-Sight project context

Claude Code reads this file automatically. It carries the full context of the
project so any session can pick up where the last one left off.

## What this project is

Computer-vision officiating assistant for **Kho-Kho** (Indian tag sport), built
against the **KPL Official Rulebook v3.0** (18×9 m court, 2 m free zones each
end, 8 chasers seated on the centre line + 1 active chaser, runners out = +1 pt,
chasing fouls = −0.5 pt).

- **Phase 1 (built):** offline pipeline — match video in → scoresheet out
  (every out/foul with timestamp, rule citation, confidence, `needs_review` flag).
- **Phase 2 (built, untested):** same rule engine on a live camera feed with
  referee alert sinks.

Architecture: perception (YOLO-pose + ByteTrack + homography + role classes)
converts pixels → court-space observations in metres; a deterministic **rule
engine** (`khosight/rules/engine.py`) applies the rulebook (fouls 8.1–8.10,
outs 7.1–7.3, kho validity 4.2.x, phase-of-play 9.1–9.4). The engine never sees
pixels — it is fully unit-tested with synthetic trajectories (`tests/`,
26 tests, `python -m pytest tests/ -q`, no GPU/torch needed).

Key docs: `PLAN.md` (roadmap), `docs/RESEARCH.md` (CV technique survey),
`docs/RULEBOOK_MAPPING.md` (rule → detector mapping), `docs/TRAINING.md`
(training workflow).

## Owner's environment

- Windows laptop, project at `C:\Projects\Kho-sight`, Python 3.12.4.
  **Use `python -m pip`, never bare `pip`** (their pip.exe launcher is broken).
- No usable local GPU — **all training runs on Google Colab** (free T4) via
  `notebooks/train_khosight_colab.ipynb` and `notebooks/autolabel_colab.ipynb`.
- Annotation in **Roboflow**: workspace `workspace-gggd2`, project `kho-sight`,
  Object Detection, classes exactly: `chaser-active`, `chaser-seated`, `cone`,
  `runner` (**alphabetical = the class-id order in trained weights and YOLO
  label files — preserve this order in any data.yaml**).
- Owner is not an ML expert — explain things plainly, give exact commands.
- Commits are authored as the repo owner; do NOT add Claude attribution,
  co-author trailers, or model names to commits/PRs (explicit owner request).

## Dataset status (as of 2026-07-17)

- 6 match videos (~6 min each, KPL fixtures, fixed wide camera, 1080p) in
  `C:\Users\thard\Downloads\kho-matches`.
- 973 frames extracted at 1 fps into `frames/` (script: `scripts/prepare_dataset.py`).
- **Hold-out match** (never annotated/trained, reserved for end-to-end testing):
  `Nottingham-Nemesis-Chasing-V-Loughboroug_Media_sW6b1H9-WjM_001_1080p` —
  its ~192 frames live in `holdout/`, the video stays out of all training.
- ~13 frames hand-annotated → Roboflow **version 1** → bootstrap model trained
  in Colab (yolo11s, imgsz 1280): chaser-seated mAP50 0.90 / cone 0.99 /
  chaser-active 0.33 / runner 0.44 on a 1-image val set (bootstrap only).
  Weights: `khosight_roles_best.pt` (owner's Drive `MyDrive/khosight/` + local Downloads).
- `scripts/pick_court.py` was used to click per-match court polygons →
  `court_regions.json`; `scripts/auto_label.py` then pre-labelled all 973
  frames with the bootstrap model, dropping 2,139 off-court detections.

## Where the work currently is (next actions)

1. Owner uploads `frames/` + `labels/` + `data.yaml` (4 classes, alphabetical
   order) into Roboflow — frames must arrive as "Annotated".
2. Owner reviews/corrects pre-labels in Roboflow (focus: `runner`,
   `chaser-active` — the bootstrap model's weak classes).
3. Generate Roboflow **version 2**: ~85/15 train/valid split (v1 had NO valid
   split — that broke training once), REMOVE the default 640 resize
   (players are small; train at imgsz 1280), no augmentations.
4. Retrain in Colab with `project.version(2)`. Target mAP50 ≥ 0.85 per class.
5. Court calibration on hold-out match:
   `python -m khosight calibrate --video <holdout.mp4> --out court_calib.json`
6. End-to-end test:
   `python -m khosight analyze --video <holdout.mp4> --calib court_calib.json
   --team-a "Nottingham Nemesis" --team-b "Loughborough Falcons"
   --half "Nottingham Nemesis:0:180" --half "Loughborough Falcons:<t2start>:<t2end>"
   --role-model khosight_roles_best.pt --out report`
   (`--half` = CHASING_TEAM:START_S:END_S of each half within the video).
7. Compare `report.md` against the real match; split errors into perception
   (→ more annotation) vs rule-engine calls (→ tune `RuleEngineConfig`
   thresholds: `tag_radius`, `backtrack_tolerance`, `centre_cross_margin`, …).
8. Later: audio "Kho" spotter (M1), realtime Phase 2 trial
   (`python -m khosight live`).

## Gotchas already hit (don't repeat)

- Newer ultralytics nests `save_dir` (e.g. `/content/runs/detect/runs/khosight/roles-N/`)
  — always locate weights with `glob('/content/**/weights/best.pt', recursive=True)`.
  The notebooks already do this.
- Roboflow "Version number N is not found" = no dataset version generated yet.
- A Roboflow version with 0 valid images breaks training
  ("missing path .../valid/images") — check the split before generating.
- YOLO label class ids are meaningless without data.yaml; class order must be
  the alphabetical order above or imports/training silently mislabel.
- Roboflow's default preprocessing adds Resize 640 — remove it every version.
- The owner's Roboflow API key was pasted into chats/notebooks; remind them to
  rotate it and never commit it.
- Tests import numpy only; perception deps (ultralytics/cv2) are lazy imports —
  keep it that way so `pytest` stays fast and torch-free.

## Rule-engine specifics worth remembering

- Court frame: x ∈ [0,18] along length, y ∈ [0,9], centre line y=4.5, free
  zones x<2 and x>16, 8 seats at x = 2 + i·14/9, cones at (2,4.5) and (16,4.5).
- Tags (7.1) are provisional `TagEvent`s; they only become `OutEvent`s when the
  phase of play ends without a chasing foul (rules 9.1/9.4 — a foul voids tags,
  and the taint persists until a valid kho). Phase ends on valid kho or
  marker-line crossing (9.3).
- Kho detection is visual (sit/stand swap) + optional audio flag
  (`FrameObservation.kho_call`); without audio, kho events are `needs_review`.
- Every threshold lives in `RuleEngineConfig` (`khosight/rules/engine.py`) —
  tune there, not inline.
