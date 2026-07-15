"""Kho-Sight command line.

  python -m khosight calibrate --video match.mp4 --out court_calib.json
  python -m khosight analyze --video match.mp4 --calib court_calib.json \\
      --team-a Lions --team-b Tigers --half Lions:0:180 --half Tigers:240:420 \\
      --out report
  python -m khosight live --calib court_calib.json --source rtsp://... \\
      --webhook http://scoreboard.local/alerts
"""

from __future__ import annotations

import argparse
import sys


def _parse_half(spec: str):
    from .pipeline.offline import HalfSpec

    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            "half spec is CHASING_TEAM:START_S[:END_S], e.g. Lions:0:180"
        )
    return HalfSpec(
        chasing_team=parts[0],
        start_s=float(parts[1]),
        end_s=float(parts[2]) if len(parts) == 3 else None,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="khosight", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="click court landmarks to build the homography")
    c.add_argument("--video", required=True)
    c.add_argument("--out", default="court_calib.json")
    c.add_argument("--frame", type=int, default=0, help="frame to calibrate on")

    a = sub.add_parser("analyze", help="Phase 1: full-match analysis to a report")
    a.add_argument("--video", required=True)
    a.add_argument("--calib", required=True)
    a.add_argument("--team-a", required=True)
    a.add_argument("--team-b", required=True)
    a.add_argument("--half", action="append", required=True, type=_parse_half,
                   metavar="TEAM:START[:END]", help="repeat per half/extra-time period")
    a.add_argument("--out", default="report", help="basename for .json/.md outputs")
    a.add_argument("--stride", type=int, default=2)
    a.add_argument("--model", default="yolo11m-pose.pt")

    l = sub.add_parser("live", help="Phase 2: realtime referee alerts")
    l.add_argument("--calib", required=True)
    l.add_argument("--source", default="0", help="webcam index or RTSP/file URL")
    l.add_argument("--model", default="yolo11n-pose.pt")
    l.add_argument("--stride", type=int, default=2)
    l.add_argument("--webhook", default=None)
    l.add_argument("--log", default=None, help="JSONL audit log path")
    l.add_argument("--display", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "calibrate":
        from scripts.calibrate_court import run_calibration  # interactive, needs GUI

        run_calibration(args.video, args.out, args.frame)
        return 0

    if args.cmd == "analyze":
        from .perception.models import PerceptionModel
        from .pipeline.offline import AnalysisConfig, analyze_match

        cfg = AnalysisConfig(stride=args.stride, model=PerceptionModel(model_name=args.model))
        report = analyze_match(
            args.video, args.calib, args.team_a, args.team_b, args.half, cfg
        )
        with open(f"{args.out}.json", "w") as f:
            f.write(report.to_json())
        with open(f"{args.out}.md", "w") as f:
            f.write(report.to_markdown())
        totals = report.totals()
        w = report.winner()
        print(f"\n{args.team_a} {totals[args.team_a]:g} — "
              f"{totals[args.team_b]:g} {args.team_b}  "
              f"({'draw' if w is None else w + ' win'})")
        print(f"Wrote {args.out}.json and {args.out}.md")
        return 0

    if args.cmd == "live":
        from .alerts.sinks import ConsoleAlertSink, JsonlAlertSink, WebhookAlertSink
        from .calibration.court import CourtCalibration
        from .perception.models import PerceptionModel
        from .pipeline.realtime import RealtimeReferee

        sinks = [ConsoleAlertSink()]
        if args.webhook:
            sinks.append(WebhookAlertSink(args.webhook))
        if args.log:
            sinks.append(JsonlAlertSink(args.log))
        source = int(args.source) if args.source.isdigit() else args.source
        ref = RealtimeReferee(
            calibration=CourtCalibration.load(args.calib),
            sinks=sinks,
            model=PerceptionModel(model_name=args.model),
            stride=args.stride,
        )
        ref.run(source=source, display=args.display)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
