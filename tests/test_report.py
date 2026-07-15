import json

from khosight.report.generator import build_report
from khosight.rules.events import FoulEvent, OutEvent


def _out(t, runner_id):
    return OutEvent(t=t, frame_idx=int(t * 30), rule="7.1", runner_id=runner_id,
                    description="Runner tagged by active chaser")


def _foul(t, rule="8.1"):
    return FoulEvent(t=t, frame_idx=int(t * 30), rule=rule,
                     description="test foul", track_id=100)


def test_totals_and_winner():
    # half 1: Lions chase, 3 outs 1 foul = 2.5; half 2: Tigers chase, 2 outs = 2.0
    report = build_report(
        "Lions", "Tigers",
        [
            ("Lions", [_out(10, 201), _out(50, 202), _out(90, 203), _foul(120)]),
            ("Tigers", [_out(30, 101), _out(70, 102)]),
        ],
    )
    totals = report.totals()
    assert totals["Lions"] == 2.5
    assert totals["Tigers"] == 2.0
    assert report.winner() == "Lions"


def test_draw_is_valid_result_rule_3_5():
    report = build_report(
        "Lions", "Tigers",
        [("Lions", [_out(10, 201)]), ("Tigers", [_out(30, 101)])],
    )
    assert report.winner() is None


def test_json_and_markdown_outputs():
    report = build_report(
        "Lions", "Tigers",
        [("Lions", [_out(10, 201), _foul(20, "8.2")]), ("Tigers", [])],
    )
    data = json.loads(report.to_json())
    assert data["totals"]["Lions"] == 0.5
    assert data["halves"][0]["outs"] == 1
    assert data["halves"][0]["fouls"] == 1
    md = report.to_markdown()
    assert "Lions 0.5" in md
    assert "| OUT | 7.1 |" in md.replace("  ", " ")
    assert "FOUL" in md and "8.2" in md
