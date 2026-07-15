# Kho-Sight — Project Plan

Computer-vision officiating assistant for Kho-Kho (Kho Premier League rules, v3.0 June 2026).

## Goal

**Phase 1 (offline):** input a full match video → output a match report: final score,
every out (+1 pt) and every chasing-team foul (−0.5 pt), with timestamps, rule
references, and confidence, applying phase-of-play logic (rules 9.1–9.4).

**Phase 2 (realtime):** the same analysis running live on a camera feed, raising
alerts to referees within ~1 second of an incident.

## Architecture

The system is split into two halves so that models can improve independently of the
rules:

```
video/camera
   │
   ▼
┌──────────────────── PERCEPTION ────────────────────┐
│ 1. Person detection        (YOLO, fine-tuned)      │
│ 2. Multi-object tracking   (ByteTrack / BoT-SORT)  │
│ 3. 2D pose estimation      (YOLO-pose / RTMPose)   │
│ 4. Court registration      (homography → metres)   │
│ 5. Team & role classify    (jersey colour + state) │
│ 6. Audio "Kho" spotting    (optional keyword det.) │
└────────────────────────────────────────────────────┘
   │  per-frame FrameObservation (court-space, metres)
   ▼
┌──────────────────── RULE ENGINE ───────────────────┐
│ Deterministic detectors, one per rulebook clause:  │
│  fouls 8.1–8.10, outs 7.1–7.4, kho validity 4.2,   │
│  phase of play 9.1–9.4, scoring 6.1–6.3            │
└────────────────────────────────────────────────────┘
   │  events (Foul / Out / Kho / Tag / PhaseEnd)
   ▼
Phase 1: report generator (JSON + Markdown scoresheet)
Phase 2: alert sinks (console, sound, webhook/WebSocket to ref devices)
```

Key design decision: **the rule engine consumes only court-space observations**
(positions in metres, postures, keypoints), never pixels. This makes it:
- unit-testable with synthetic trajectories (no GPU needed),
- reusable unchanged between Phase 1 and Phase 2,
- auditable — every event cites the rulebook clause that fired.

## Milestones

### M0 — Foundations (this commit)
- [x] Court model with exact KPL geometry (18×9 m, free zones, seat slots at 14/9 m)
- [x] Homography calibration (image → court metres) + interactive calibration script
- [x] Full rule engine with per-clause detectors and phase-of-play state machine
- [x] Offline pipeline skeleton (detect → track → pose → project → engine → report)
- [x] Realtime pipeline skeleton with alert sinks
- [x] Unit tests for every rule detector using synthetic data

### M1 — Perception quality (needs your training videos)
- [ ] Annotate ~2–5k frames (players, roles, cones) — CVAT / Label Studio / Roboflow
- [ ] Fine-tune the detector on Kho-Kho footage (`scripts/train_detector.py`)
- [ ] Fit posture classifier (seated / standing / transition) on pose keypoints
- [ ] Calibrate thresholds (tag radius, backtrack tolerance) against labelled clips
- [ ] Train "Kho" audio keyword spotter from match audio

### M2 — Phase 1 hardening
- [ ] Evaluate on held-out matches vs. official scoresheets (target: exact score match)
- [ ] Multi-camera fusion (two ends of court) for occlusion robustness
- [ ] Review UI: event timeline with video snippets for human verification

### M3 — Phase 2 deployment
- [ ] Latency budget: ≤ 66 ms/frame perception (15 fps min) on a single GPU
- [ ] TensorRT / ONNX export of fine-tuned models
- [ ] Referee alert device integration (WebSocket + buzzer/watch)
- [ ] On-field trial alongside human referees; measure precision/recall per rule

## Honest limitations

- Tag detection (rule 7.1, hand-touch) at fingertip precision is the hardest problem;
  we flag low-confidence tags `needs_review` rather than silently deciding.
- "Kho" is an *audio* event (4.2.1); vision only sees the sit/stand swap. Reliable
  8.3 calls need the audio keyword spotter (M1) or ref confirmation.
- Single wide-angle camera will lose players in occlusion pile-ups; M2 adds a second
  camera before we claim referee-grade accuracy.
