"""Phase 1: offline match analysis — video in, scoresheet out."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Iterator, Optional

from ..calibration.court import CourtCalibration
from ..perception.builder import ObservationBuilder
from ..perception.models import PerceptionModel
from ..report.generator import MatchReport, build_report
from ..rules.engine import RuleEngine, RuleEngineConfig
from ..rules.events import Event


@dataclass
class HalfSpec:
    """One half of a match inside a video file (times in seconds)."""

    chasing_team: str
    start_s: float
    end_s: Optional[float] = None  # None = until video end


@dataclass
class AnalysisConfig:
    stride: int = 1                 # process every Nth frame (2-3 fine for Phase 1)
    engine: RuleEngineConfig = field(default_factory=RuleEngineConfig)
    model: PerceptionModel = field(default_factory=PerceptionModel)


def _iter_frames(video_path: str, start_s: float, end_s: Optional[float], stride: int):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    idx = int(start_s * fps)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if end_s is not None and t > end_s:
            break
        if (idx - int(start_s * fps)) % stride == 0:
            yield t - start_s, idx, frame
        idx += 1
    cap.release()


def analyze_half(
    video_path: str,
    calibration: CourtCalibration,
    half: HalfSpec,
    config: AnalysisConfig | None = None,
    progress: bool = True,
) -> list[Event]:
    """Run perception + rules over one half; returns the event list."""
    cfg = config or AnalysisConfig()
    builder = ObservationBuilder(calibration=calibration)
    engine = RuleEngine(cfg.engine)

    for t, idx, frame in _iter_frames(video_path, half.start_s, half.end_s, cfg.stride):
        detections = cfg.model.track_frame(frame)
        obs = builder.build(t, idx, frame, detections)
        engine.update(obs)
        if progress and idx % 300 == 0:
            print(f"  t={t:6.1f}s  players={len(obs.players)}  "
                  f"events={len(engine.events)}", file=sys.stderr)
    engine.finish()
    return engine.events


def analyze_match(
    video_path: str,
    calibration_path: str,
    team_a: str,
    team_b: str,
    halves: list[HalfSpec],
    config: AnalysisConfig | None = None,
) -> MatchReport:
    """Full Phase-1 entry point. `halves` carries who chases when (teams swap
    at half time, rule 3.2; extra-time periods are additional entries, 3.6/3.7)."""
    calibration = CourtCalibration.load(calibration_path)
    half_events: list[tuple[str, list[Event]]] = []
    for i, half in enumerate(halves, 1):
        print(f"Analysing half {i} ({half.chasing_team} chasing)...", file=sys.stderr)
        events = analyze_half(video_path, calibration, half, config)
        half_events.append((half.chasing_team, events))
    return build_report(team_a, team_b, half_events, video=video_path)
