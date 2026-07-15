"""Phase 2: realtime referee assistant.

Same perception + rule engine as Phase 1, driven by a live source (webcam id,
RTSP URL, or capture card), pushing events to alert sinks as they happen.

Latency levers (see PLAN.md M3):
  - smaller model (yolo11n-pose) and/or TensorRT export,
  - `stride` frame skipping — the tracker bridges skipped frames,
  - `max_fps` guard so a slow model degrades to lower rate, not lag build-up.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from ..alerts.sinks import AlertSink, ConsoleAlertSink, dispatch
from ..calibration.court import CourtCalibration
from ..perception.builder import ObservationBuilder
from ..perception.models import PerceptionModel
from ..rules.engine import RuleEngine, RuleEngineConfig


@dataclass
class RealtimeReferee:
    calibration: CourtCalibration
    sinks: list[AlertSink] = field(default_factory=lambda: [ConsoleAlertSink()])
    model: PerceptionModel = field(
        default_factory=lambda: PerceptionModel(model_name="yolo11n-pose.pt")
    )
    engine_config: RuleEngineConfig = field(default_factory=RuleEngineConfig)
    stride: int = 2

    def run(self, source: int | str = 0, display: bool = False) -> RuleEngine:
        import cv2

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"cannot open source: {source}")
        builder = ObservationBuilder(calibration=self.calibration)
        engine = RuleEngine(self.engine_config)
        t0 = time.monotonic()
        idx = 0
        print("Realtime referee running — Ctrl-C to stop.", file=sys.stderr)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % self.stride:
                    continue
                t = time.monotonic() - t0
                detections = self.model.track_frame(frame)
                obs = builder.build(t, idx, frame, detections)
                events = engine.update(obs)
                dispatch(events, self.sinks)
                if display:
                    self._draw(frame, obs, engine)
                    cv2.imshow("kho-sight", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
        dispatch(engine.finish(), self.sinks)
        s = engine.score()
        print(f"\nSession score (chasing team): {s['points']:g} "
              f"({s['outs']} outs, {s['fouls']} fouls)", file=sys.stderr)
        return engine

    @staticmethod
    def _draw(frame, obs, engine) -> None:
        import cv2

        for p in obs.players:
            colour = (0, 200, 255) if p.team.value == "chase" else (0, 255, 0)
            if p.track_id == engine.active_chaser_id:
                colour = (0, 0, 255)
            cv2.putText(
                frame,
                f"{p.track_id}:{p.posture.value[:4]}",
                (20, 20 + 18 * (p.track_id % 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
            )
        s = engine.score()
        cv2.putText(
            frame,
            f"pts {s['points']:g} | outs {s['outs']} | fouls {s['fouls']}",
            (10, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
