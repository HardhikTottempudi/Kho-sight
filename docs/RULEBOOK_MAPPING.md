# Rulebook → Detector Mapping

How each clause of the KPL Official Rulebook v3.0 is (or will be) detected.
Signals: **T** = tracks (court-space positions), **P** = pose keypoints,
**A** = audio, **H** = human review needed.

## Court model (rule 1)

18 m × 9 m; free zones x∈[0,2] and x∈[16,18]; main area x∈[2,16]; centre line
y = 4.5; 8 seat slots on the centre line at x = 2 + i·14/9, i = 1..8; cones at
(2, 4.5) and (16, 4.5). Implemented in `khosight/calibration/court.py`.

## Fouls (rule 8, −0.5 pt each, rule 6.2)

| Rule | Description | Detector | Signals |
|------|-------------|----------|---------|
| 8.1 | Active chaser backtracks (feet or shoulder lean) | `BacktrackDetector`: committed direction from sustained x-motion; foul on reverse x-displacement > tolerance, or shoulder-behind-hip lean while moving | T, P |
| 8.2 | Active chaser crosses/steps on centre line in main area | `CentreLineDetector`: foot y crosses 4.5 (with hysteresis) while 2 < x < 16 | T |
| 8.3 | Kho without foot behind seat / kho to chaser already passed | `KhoDetector`: at swap moment check giver foot within behind-zone of seat; seat id ∉ passed-set of current run | T, P, A |
| 8.4 | Seated chaser stands before receiving valid kho | `PostureMonitor`: non-active chaser posture STANDING outside a swap window | P |
| 8.5 | Chaser fully outside court boundary | `BoundsDetector`: all feet outside court polygon | T, P |
| 8.6 | More than one chaser standing | `PostureMonitor`: standing count > 1 sustained beyond swap window | P |
| 8.7 | Chaser touches/kicks a cone | `ConeDetector`: chaser foot/hand within radius of cone (M1: cone displacement check) | T, P |
| 8.8 | Hand beyond centre line when pivoting (unless both feet in free zone) | `PivotHandDetector`: wrist projected across y=4.5 while feet not both inside free zone | P |
| 8.9 | Front foot touches/crosses free-zone marker line when giving kho to end seat | `KhoDetector`: at swap to seat 1 or 8, front-foot x vs marker line | T, P |
| 8.10 | Kho to a chaser not fully seated | `KhoDetector`: target posture ≠ SEATED at kho moment | P, A |

## Ways out (rule 7, +1 pt each, rule 6.1)

| Rule | Description | Detector | Signals |
|------|-------------|----------|---------|
| 7.1 | Active chaser touches runner with hands | `TagDetector`: active-chaser wrist within radius of runner body; confidence-scored; confirmed only by phase logic | T, P, H |
| 7.2 | Runner steps out of bounds (per 1.5: foot fully beyond line) | `BoundsDetector` on runners | T, P |
| 7.3 | Runner fails to enter within 3 s of batch out | `BatchMonitor`: timer from batch-cleared to next runner on court | T |
| 7.4 | Runner enters with wrong batch | `BatchMonitor`: track-id ↔ batch registry (needs stable IDs / jersey OCR) | T, H |

## Kho validity (rule 4.2) — inputs to 8.3/8.9/8.10

Vision detects the *swap*: a seated chaser stands while the active chaser sits at
that seat within a short window. Audio keyword spotting timestamps the call itself
(4.2.1). 4.2.2 (wrong side) checked via giver side vs seat facing.

## Phase of play (rule 9)

`PhaseTracker`: a phase ends when (9.3.1) a valid kho completes (new chaser stood)
or (9.3.2) the active chaser crosses a free-zone marker line (x=2 or x=16), either
direction. Tags collected within a phase are **confirmed as outs only if the phase
contains no chasing foul** (9.1, 9.4); otherwise the tag is voided and the foul
stands. Fouls after the phase ended leave the out standing (9.2).

## Scoring (rule 6)

score(chasing team, half) = confirmed_outs × 1.0 − fouls × 0.5. Teams swap roles at
half time (3.2); the report maps halves to team names. Rule 10.5 misconduct
deductions are out of scope for CV (human decision) but representable in the report.

## Deliberately out of scope for automation

- 2.x squad/substitution admin, 3.4 added time, 10.5 misconduct — human decisions.
- 10.x whistle/hand signals are *outputs* we assist, not inputs we detect
  (though detecting the assistant referee's raised hand is a possible M2 cross-check).
