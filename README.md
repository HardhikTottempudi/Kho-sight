# Kho-Sight

Computer-vision officiating assistant for **Kho-Kho**, built against the Kho
Premier League Official Rulebook v3.0 (June 2026).

- **Phase 1 (offline):** feed it a match video → get a scoresheet: every out
  (+1 pt, rule 6.1), every chasing-team foul (−0.5 pt, rule 6.2), phase-of-play
  adjudication (rules 9.1–9.4), final score and winner.
- **Phase 2 (realtime):** the same rule engine on a live camera feed, pushing
  advisory alerts to referees within ~1 second.

See [PLAN.md](PLAN.md) for the roadmap, [docs/RESEARCH.md](docs/RESEARCH.md)
for the CV technique survey, and
[docs/RULEBOOK_MAPPING.md](docs/RULEBOOK_MAPPING.md) for exactly how each
rulebook clause is detected.

## How it works

```
video ─► YOLO-pose detection ─► ByteTrack tracking ─► homography (px → metres)
      ─► team/role/posture classification ─► RULE ENGINE ─► events ─► report/alerts
```

Perception converts pixels into court-space observations (positions in metres,
postures, wrist/ankle keypoints). A deterministic **rule engine** then applies
the rulebook: fouls 8.1–8.10, outs 7.1–7.3, kho validity 4.2.x, phase of play
9.x. Every event carries a timestamp, a rule citation, a confidence, and a
`needs_review` flag — the system assists referees, it never overrules them
(rule 10.2).

## Quick start

```bash
pip install -r requirements.txt

# 1. one-off per camera setup: click court landmarks on a frame
python -m khosight calibrate --video match.mp4 --out court_calib.json

# 2. Phase 1: analyse a match (teams swap chasing roles at half time)
python -m khosight analyze --video match.mp4 --calib court_calib.json \
    --team-a Lions --team-b Tigers \
    --half Lions:0:180 --half Tigers:240:420 \
    --out report        # writes report.json + report.md

# 3. Phase 2: live referee alerts from a camera / RTSP stream
python -m khosight live --calib court_calib.json --source rtsp://cam/feed \
    --webhook http://scoreboard.local/alerts --log alerts.jsonl
```

## Repository layout

```
khosight/
  calibration/   court geometry (rule 1) + image→court homography
  perception/    YOLO-pose + tracker wrapper; team/role/posture classification
  rules/         observation/event model + the rule engine (the rulebook, as code)
  audio/         "Kho" call spotting (rule 4.2.1) — stub, trained model at M1
  pipeline/      Phase 1 offline analysis; Phase 2 realtime referee
  alerts/        console / JSONL / webhook alert sinks
  report/        scoresheet generation (JSON + Markdown)
scripts/         court calibration UI, dataset mining, detector fine-tuning
tests/           rule engine tested on synthetic trajectories (no GPU needed)
docs/            research survey + rulebook→detector mapping
```

## Training on your match videos (next step — M1)

The pipeline runs today on pretrained COCO models; accuracy on real matches
comes from fine-tuning a role detector (`chaser-seated`, `chaser-active`,
`runner`, `cone`) on your footage. **Full walkthrough:
[docs/TRAINING.md](docs/TRAINING.md)** — no GPU laptop needed, training runs on
free Google Colab via
[notebooks/train_khosight_colab.ipynb](notebooks/train_khosight_colab.ipynb).

```bash
# 1. extract frames (laptop)          2. pre-label automatically (optional)
python scripts/prepare_dataset.py frames --videos matches/ --out frames/
python scripts/auto_label.py --frames frames/ --out labels/

# 3. correct labels in Roboflow  →  4. train in Colab (notebook above)

# 5. use the trained weights
python -m khosight analyze ... --role-model khosight_roles_best.pt
```

## Tests

```bash
python -m pytest tests/ -q
```

The rule engine is exercised end-to-end with synthetic court-space
trajectories — backtracking, centre-line crossing, kho validity, tag-then-foul
phase adjudication, batch timing, and scoring — with no model downloads needed.
