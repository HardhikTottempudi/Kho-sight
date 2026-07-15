# CV Techniques Research — Kho-Sight

Survey of techniques evaluated for each perception sub-problem, with the choice made
and why. Current as of mid-2026.

## 1. Person detection

| Option | Notes |
|---|---|
| **YOLO (v8/11, Ultralytics)** ✅ | Real-time (100+ fps on GPU), easy fine-tuning API, built-in tracking integration. The de-facto standard for sports player tracking. |
| RT-DETR | Transformer detector, no NMS, similar speed at higher accuracy on crowded scenes — good candidate for M1 A/B test since Kho-Kho has a dense centre-line cluster. |
| SAM 2 | Promptable segmentation; overkill for boxes, useful later for pixel-accurate foot-on-line calls. |

Choice: **YOLO** pretrained on COCO `person` for M0; fine-tune on Kho-Kho frames in M1
(camera angle, seated players and Indian outdoor/indoor court conditions are out of
COCO distribution — seated chasers in a row are the main failure mode to train for).

## 2. Multi-object tracking

Production trackers in 2026 reduce to two choices ([Forasoft survey](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/multi-object-tracking-deepsort-bytetrack-ocsort),
[Encord top-10](https://encord.com/blog/video-object-tracking-algorithms/)):

- **ByteTrack** ✅ — motion-only, fastest, keeps low-confidence boxes in association,
  which matters because seated chasers are low-confidence detections.
- **BoT-SORT-ReID** — appearance features survive 5 s+ occlusions; Ultralytics default.
  Kho-Kho chases cause frequent player crossovers, so we make the tracker pluggable
  and A/B both (`tracker:` config key).
- DeepSORT — obsolete in 2026; every use case is served better by the two above.
- Graph/hierarchy trackers (SoccerNet-style long-term ReID) — M2 option for
  full-match identity persistence across halves.

Identity persistence matters: rule 8.3 ("kho to a chaser already passed") and batch
tracking (7.3/7.4) both need stable IDs. Runner jersey numbers + OCR is the M2
fallback for re-identification after ID switches.

## 3. Pose estimation

Needed for: posture (seated vs standing — rules 4.2.4, 8.4, 8.6, 8.10), shoulder
lean (backtracking 8.1), wrist positions (hand-tag 7.1, hand-across-line 8.8),
foot placement (all line rules).

- **YOLO-pose** ✅ M0 — one model does detection+17 COCO keypoints, single pass, fast.
- RTMPose / ViTPose — top-down, more accurate; drop-in upgrade at M1 if wrist/ankle
  precision limits tag detection ([systematic review of HPE in sport](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12696263/)).
- 3D pose (MotionBERT etc.) — only if 2D lean detection proves insufficient for 8.1.

## 4. Court registration (image → metres)

All line-based rules (bounds 1.5/8.5, centre line 8.2/8.8, free-zone marker
4.4.3/8.9) require projecting feet/hands into court coordinates.

- **Fixed-camera homography from operator-clicked keypoints** ✅ M0 — matches are
  filmed from tripods; a one-off 4+ point calibration per camera setup is exact and
  free. DLT with normalisation, no learned model needed.
- Automatic registration (keypoint/line CNNs: [PnLCalib](https://arxiv.org/html/2404.08401v4),
  ["No Bells, Just Whistles"](https://arxiv.org/html/2404.08401v2),
  segmentation-based [racket-court registration](https://openreview.net/pdf/01b2e7445170ebd4328ed615e1196d4ce6b880ef.pdf)) —
  M2, for broadcast/moving cameras; train a court-keypoint head on Kho-Kho lines.
- Per-frame refinement (evolution-strategy pose refinement) — only for PTZ cameras.

Foot contact point: bottom-centre of bbox is biased under lean; we use ankle
keypoints when confidence allows, bbox fallback otherwise.

## 5. Team & role classification

- Team split: HSV jersey-colour k-means (k=2) on torso crops ✅ — robust, no training.
  Fallback: fine-tuned classifier head or SigLIP embeddings if kits clash.
- Roles derived from state, not appearance: the chasing team is the one with ~8
  players seated near the centre line; the **active chaser is the standing member of
  the chasing team**; runners are the ≤3 on-court members of the other team.

## 6. Action/event recognition

Two strategies for tag ("touch") and kho detection:

1. **Geometric heuristics on pose + tracks** ✅ M0 — wrist-to-runner distance for tags,
   sit/stand swap detection for kho. Transparent, tunable, no data needed.
2. Learned temporal models (M1+): skeleton-based ST-GCN / PoseConv3D on 2-second
   keypoint windows, or video models (VideoMAE, X3D) for tag/no-tag classification.
   Given "loads of training videos", the pragmatic path is: run M0 heuristics to
   *mine candidate moments*, human-verify them in the review UI, and use the verified
   clips as training data (active learning loop).

## 7. Audio "Kho" keyword spotting

Rule 4.2.1 makes the kho an audible call. Plan: band-energy voice-activity gate (M0
stub) → small keyword-spotting model (1D-CNN / conformer on log-mel spectrograms,
trained on "kho" utterances mined from match audio) at M1. Audio-visual fusion
(call + swap must coincide) gives 8.3/8.10 their timestamps.

## 8. Realtime engineering (Phase 2)

- Same rule engine; perception swapped to smaller model (YOLO-n), frame-skipping
  with tracker interpolation, ONNX/TensorRT export.
- Budget: 15 fps floor → ≤66 ms/frame; alert latency target ≤1 s end-to-end.
- Alerts carry rule reference + court location + confidence; low-confidence events
  are advisory (assistant-ref style "raise hand"), never auto-whistle. Rule 10.2
  keeps the human main referee as final authority — the system is an assistant.

## Sources

- [Ultralytics multi-object tracking docs](https://docs.ultralytics.com/modes/track)
- [MOT in production 2026: DeepSORT/ByteTrack/OC-SORT](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/multi-object-tracking-deepsort-bytetrack-ocsort)
- [Top 10 video object tracking algorithms 2026](https://encord.com/blog/video-object-tracking-algorithms/)
- [Football players tracking, YOLOv8 + ByteTrack](https://github.com/Darkmyter/Football-Players-Tracking)
- [SoccerNet 2023 challenges results](https://arxiv.org/pdf/2309.06006)
- [Long-term player tracking with graph hierarchies](https://arxiv.org/pdf/2502.21242)
- [PnLCalib: field registration via points and lines](https://arxiv.org/html/2404.08401v4)
- [No Bells, Just Whistles: sports field registration](https://arxiv.org/html/2404.08401v2)
- [Racket-sports court registration framework](https://openreview.net/pdf/01b2e7445170ebd4328ed615e1196d4ce6b880ef.pdf)
- [Deep learning in sports survey](https://arxiv.org/pdf/2307.03353)
- [HPE in sport, systematic review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12696263/)
- [Best AI models for player tracking](https://www.sportsfirst.net/post/best-ai-models-for-player-tracking-and-ball-tracking-in-usa-sports)
