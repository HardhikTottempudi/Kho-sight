"""Match report: turns rule-engine events into the final scoresheet (rule 6)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..rules.events import Event, FoulEvent, KhoEvent, OutEvent, TagEvent


@dataclass
class HalfResult:
    half: int                      # 1 or 2 (extra time: 3, 4, ...)
    chasing_team: str              # team name chasing this half (teams swap, rule 3.2)
    events: list[Event] = field(default_factory=list)

    @property
    def outs(self) -> list[OutEvent]:
        return [e for e in self.events if isinstance(e, OutEvent)]

    @property
    def fouls(self) -> list[FoulEvent]:
        return [e for e in self.events if isinstance(e, FoulEvent)]

    @property
    def points(self) -> float:
        """Points earned by the chasing team this half (rules 6.1, 6.2)."""
        return len(self.outs) * 1.0 - len(self.fouls) * 0.5


@dataclass
class MatchReport:
    team_a: str
    team_b: str
    halves: list[HalfResult]
    video: Optional[str] = None

    def totals(self) -> dict[str, float]:
        totals = {self.team_a: 0.0, self.team_b: 0.0}
        for h in self.halves:
            totals[h.chasing_team] += h.points
        return totals

    def winner(self) -> Optional[str]:
        t = self.totals()
        if t[self.team_a] > t[self.team_b]:
            return self.team_a
        if t[self.team_b] > t[self.team_a]:
            return self.team_b
        return None  # draw (valid in league fixtures, rule 3.5)

    # ------------------------------------------------------------------ output

    def to_json(self) -> str:
        def enc(e: Event) -> dict:
            d = asdict(e)
            d["type"] = type(e).__name__
            return d

        return json.dumps(
            {
                "teams": [self.team_a, self.team_b],
                "video": self.video,
                "totals": self.totals(),
                "winner": self.winner(),
                "halves": [
                    {
                        "half": h.half,
                        "chasing_team": h.chasing_team,
                        "outs": len(h.outs),
                        "fouls": len(h.fouls),
                        "points": h.points,
                        "events": [enc(e) for e in h.events],
                    }
                    for h in self.halves
                ],
            },
            indent=2,
        )

    def to_markdown(self) -> str:
        t = self.totals()
        w = self.winner()
        lines = [
            f"# Match Report: {self.team_a} vs {self.team_b}",
            "",
            f"**Final score:** {self.team_a} {t[self.team_a]:g} — "
            f"{t[self.team_b]:g} {self.team_b}",
            f"**Result:** {'Draw' if w is None else f'{w} win'}",
            "",
        ]
        for h in self.halves:
            lines += [
                f"## Half {h.half} — {h.chasing_team} chasing",
                "",
                f"Outs: {len(h.outs)} (+{len(h.outs):g}) | "
                f"Fouls: {len(h.fouls)} (−{len(h.fouls) * 0.5:g}) | "
                f"**Points: {h.points:g}**",
                "",
                "| Time | Event | Rule | Detail | Conf. | Review |",
                "|------|-------|------|--------|-------|--------|",
            ]
            for e in sorted(h.events, key=lambda e: e.t):
                if isinstance(e, OutEvent):
                    kind, rule, detail = "OUT", e.rule, e.description
                elif isinstance(e, FoulEvent):
                    kind, rule, detail = "FOUL", e.rule, e.description
                elif isinstance(e, KhoEvent):
                    kind, rule = "KHO", "4.2"
                    detail = f"seat {e.seat_index + 1}, {'valid' if e.valid else 'INVALID'}"
                elif isinstance(e, TagEvent):
                    kind, rule, detail = "tag", "7.1", f"runner {e.runner_id} (provisional)"
                else:
                    continue  # phase ends omitted from the scoresheet table
                mins, secs = divmod(e.t, 60)
                flag = "⚠" if e.needs_review else ""
                lines.append(
                    f"| {int(mins):02d}:{secs:04.1f} | {kind} | {rule} | "
                    f"{detail} | {e.confidence:.2f} | {flag} |"
                )
            lines.append("")
        lines.append(
            "_Events marked ⚠ are below the confidence threshold and should be "
            "verified against video. The main referee has the final say (rule 10.2)._"
        )
        return "\n".join(lines)


def build_report(
    team_a: str,
    team_b: str,
    half_events: list[tuple[str, list[Event]]],
    video: Optional[str] = None,
) -> MatchReport:
    """half_events: list of (chasing_team_name, events) in half order."""
    halves = [
        HalfResult(half=i + 1, chasing_team=name, events=evts)
        for i, (name, evts) in enumerate(half_events)
    ]
    return MatchReport(team_a=team_a, team_b=team_b, halves=halves, video=video)
