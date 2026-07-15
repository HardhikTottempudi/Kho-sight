"""Observation and event data model shared by perception and the rule engine.

Perception produces `FrameObservation`s in COURT coordinates (metres); the rule
engine consumes them and emits `Event`s. Nothing here touches pixels or models,
so the whole rules layer is unit-testable with synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

Point = tuple[float, float]


class Team(str, Enum):
    CHASE = "chase"
    RUN = "run"


class Posture(str, Enum):
    STANDING = "standing"
    SEATED = "seated"
    TRANSITION = "transition"  # mid sit/stand, or classifier unsure
    UNKNOWN = "unknown"


@dataclass
class PlayerObservation:
    """One player in one frame, in court space (metres)."""

    track_id: int
    team: Team
    pos: Point  # ground contact point (ankle-derived, bbox fallback)
    posture: Posture = Posture.UNKNOWN
    feet: list[Point] = field(default_factory=list)  # per-foot points if available
    wrists: list[Point] = field(default_factory=list)
    lean_x: float = 0.0  # shoulder-midpoint minus hip-midpoint, court-x metres (8.1)
    facing: int = 0  # seated chasers: +1/-1 = y-direction faced (4.1.3), 0 unknown
    confidence: float = 1.0

    def foot_points(self) -> list[Point]:
        return self.feet if self.feet else [self.pos]


@dataclass
class FrameObservation:
    t: float  # seconds since half start
    frame_idx: int
    players: list[PlayerObservation]
    kho_call: bool = False  # audio keyword spotter fired this frame (4.2.1)

    def by_team(self, team: Team) -> list[PlayerObservation]:
        return [p for p in self.players if p.team == team]

    def get(self, track_id: int) -> Optional[PlayerObservation]:
        for p in self.players:
            if p.track_id == track_id:
                return p
        return None


# --------------------------------------------------------------------------- events


@dataclass
class Event:
    t: float
    frame_idx: int
    confidence: float = 1.0
    needs_review: bool = False
    note: str = ""


@dataclass
class FoulEvent(Event):
    """Chasing-team foul: -0.5 pt (rule 6.2)."""

    rule: str = ""  # e.g. "8.1"
    description: str = ""
    track_id: Optional[int] = None
    location: Optional[Point] = None


@dataclass
class TagEvent(Event):
    """Provisional touch (rule 7.1) — becomes an OutEvent only if its phase is clean."""

    chaser_id: int = -1
    runner_id: int = -1
    location: Optional[Point] = None


@dataclass
class OutEvent(Event):
    """Runner given out: +1 pt (rule 6.1)."""

    rule: str = "7.1"
    runner_id: Optional[int] = None
    description: str = ""


@dataclass
class KhoEvent(Event):
    """A kho was given (valid or not; invalid ones also raise FoulEvents)."""

    giver_id: int = -1
    receiver_id: int = -1
    seat_index: int = -1
    valid: bool = True


@dataclass
class PhaseEndEvent(Event):
    """Phase of play ended (rule 9.3)."""

    reason: str = ""  # "valid_kho" | "marker_line_crossed"
