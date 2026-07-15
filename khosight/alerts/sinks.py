"""Referee alert sinks for Phase 2 (realtime).

Alerts are advisory — the assistant-referee hand-signal of rules 10.3.1/10.3.2,
never an auto-whistle. High-confidence events alert immediately; low-confidence
ones are marked ADVISORY so the referee treats them as "have a look".
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from typing import Protocol

from ..rules.events import Event, FoulEvent, KhoEvent, OutEvent, TagEvent


def _summarise(event: Event) -> str | None:
    if isinstance(event, FoulEvent):
        loc = f" at ({event.location[0]:.1f}, {event.location[1]:.1f}) m" if event.location else ""
        return f"FOUL rule {event.rule}: {event.description}{loc} → −0.5 pt"
    if isinstance(event, OutEvent):
        who = f"runner #{event.runner_id}" if event.runner_id is not None else "runner"
        return f"OUT rule {event.rule}: {who} — {event.description} → +1 pt"
    if isinstance(event, TagEvent):
        return f"TAG (provisional): chaser #{event.chaser_id} on runner #{event.runner_id}"
    if isinstance(event, KhoEvent):
        return None if event.valid else f"INVALID KHO at seat {event.seat_index + 1}"
    return None


class AlertSink(Protocol):
    def send(self, event: Event, message: str) -> None: ...


class ConsoleAlertSink:
    """Terminal alerts with a bell for high-confidence incidents."""

    def send(self, event: Event, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        prefix = "ADVISORY" if event.needs_review else "ALERT"
        bell = "" if event.needs_review else "\a"
        print(f"{bell}[{stamp}] {prefix} (t={event.t:6.1f}s conf={event.confidence:.2f}) {message}",
              file=sys.stderr, flush=True)


class JsonlAlertSink:
    """Append-only JSONL log — the audit trail of everything alerted."""

    def __init__(self, path: str):
        self._f = open(path, "a")

    def send(self, event: Event, message: str) -> None:
        rec = asdict(event)
        rec["type"] = type(event).__name__
        rec["message"] = message
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()


class WebhookAlertSink:
    """POSTs each alert as JSON — point it at a referee device / scoreboard app.
    Swap for a persistent WebSocket at M3 for lower latency."""

    def __init__(self, url: str, timeout_s: float = 1.0):
        self.url = url
        self.timeout_s = timeout_s

    def send(self, event: Event, message: str) -> None:
        import urllib.request

        rec = asdict(event)
        rec["type"] = type(event).__name__
        rec["message"] = message
        req = urllib.request.Request(
            self.url,
            data=json.dumps(rec).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=self.timeout_s)
        except Exception as exc:  # alerts must never kill the pipeline
            print(f"[alerts] webhook failed: {exc}", file=sys.stderr)


def dispatch(events: list[Event], sinks: list[AlertSink]) -> None:
    for e in events:
        msg = _summarise(e)
        if msg is None:
            continue
        for s in sinks:
            s.send(e, msg)
