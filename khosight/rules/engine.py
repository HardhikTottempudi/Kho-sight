"""Deterministic Kho-Kho rule engine (KPL rulebook v3.0).

Consumes court-space `FrameObservation`s, maintains game state, and emits events
with rulebook citations. Perception uncertainty is surfaced via `confidence` and
`needs_review` on events — the engine assists referees (rule 10.2), it does not
replace them.

Implemented clauses: fouls 8.1–8.10, outs 7.1–7.3 (7.4 flagged for review),
kho validity 4.2.x, phase of play 9.1–9.4, scoring 6.1–6.2.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from ..calibration.court import Court
from .events import (
    Event,
    FoulEvent,
    FrameObservation,
    KhoEvent,
    OutEvent,
    PhaseEndEvent,
    PlayerObservation,
    Posture,
    TagEvent,
    Team,
)

Point = tuple[float, float]


@dataclass
class RuleEngineConfig:
    # geometry tolerances (metres) — calibrate on labelled clips at M1
    line_eps: float = 0.05          # "touching a line" tolerance
    centre_cross_margin: float = 0.10  # 8.2: foot beyond centre line by this = cross
    backtrack_tolerance: float = 0.30  # 8.1: reverse displacement allowed as noise
    direction_commit_dist: float = 0.50  # 4.3.1: movement to commit a direction
    lean_back_threshold: float = 0.20  # 8.1: shoulders behind hips (metres, court x)
    kho_reach: float = 0.80         # 4.2.1: giver foot within this of seat = "behind"
    tag_radius: float = 0.45        # 7.1: wrist-to-runner distance for a touch
    cone_radius: float = 0.20       # 8.7
    hand_over_line_margin: float = 0.10  # 8.8
    out_of_bounds_margin: float = 0.05   # 1.5 / 7.2 / 8.5

    # timing
    batch_entry_seconds: float = 3.0  # 5.3 / 7.3
    kho_swap_window_s: float = 1.2   # sit/stand swap tolerance around a kho
    seat_settle_s: float = 0.8      # 8.10: time fully seated before kho-eligible
    tag_cooldown_s: float = 2.0
    foul_cooldown_s: float = 2.0

    # debouncing (frames)
    posture_frames: int = 3         # sustained frames before posture-based fouls
    lean_frames: int = 3


@dataclass
class _PhaseState:
    """One phase of play (rule 9.3)."""

    index: int = 0
    started_t: float = 0.0
    tags: list[TagEvent] = field(default_factory=list)
    fouls: list[FoulEvent] = field(default_factory=list)


class _TrackHistory:
    """Short per-player motion history for direction / crossing logic."""

    def __init__(self, maxlen: int = 90):
        self.pos: Deque[tuple[float, Point]] = deque(maxlen=maxlen)

    def push(self, t: float, p: Point) -> None:
        self.pos.append((t, p))

    @property
    def last(self) -> Optional[Point]:
        return self.pos[-1][1] if self.pos else None

    def prev(self) -> Optional[Point]:
        return self.pos[-2][1] if len(self.pos) >= 2 else None


class RuleEngine:
    def __init__(self, config: RuleEngineConfig | None = None, court: Court | None = None):
        self.cfg = config or RuleEngineConfig()
        self.court = court or Court()
        self.events: list[Event] = []

        # chase-side state
        self.active_chaser_id: Optional[int] = None
        self.run_direction: int = 0          # committed +1/-1 along x (4.3.1)
        self._chaser_side: Optional[int] = None  # side of centre line (4.3.3)
        self._direction_anchor_x: Optional[float] = None
        self._max_progress_x: Optional[float] = None  # furthest point in direction
        self.passed_seats: set[int] = set()  # seats passed in current run (8.3)
        self._last_kho_t: float = -1e9
        self._last_kho_receiver: Optional[int] = None
        self._seat_settled_since: dict[int, float] = {}   # track_id -> seated since t
        self._settled_at_rise: dict[int, bool] = {}       # settle state when they rose
        self._lean_count = 0

        # phase of play (rule 9)
        self.phase = _PhaseState(index=0)
        self._tainted_by_foul = False        # 9.4: fouls void tags until a valid kho

        # runners / batches (rules 5, 7)
        self.out_runner_ids: set[int] = set()
        self._batch_cleared_t: Optional[float] = None
        self._batch_count = 1

        # bookkeeping
        self._hist: dict[int, _TrackHistory] = {}
        self._standing_count_frames = 0
        self._early_stand_frames: dict[int, int] = {}
        self._cooldowns: dict[str, float] = {}
        self._last_t = 0.0
        self._prev_frame: Optional[FrameObservation] = None

    # ------------------------------------------------------------------ public

    def update(self, frame: FrameObservation) -> list[Event]:
        """Process one frame; returns events emitted for this frame."""
        out: list[Event] = []
        self._last_t = frame.t
        for p in frame.players:
            self._hist.setdefault(p.track_id, _TrackHistory()).push(frame.t, p.pos)
        self._update_seat_settle(frame)

        self._update_active_chaser(frame, out)
        active = frame.get(self.active_chaser_id) if self.active_chaser_id is not None else None

        if active is not None:
            self._update_direction(active)
            self._update_passed_seats(active)
            self._check_backtrack(frame, active, out)          # 8.1
            self._check_centre_line(frame, active, out)        # 8.2
            self._check_pivot_hand(frame, active, out)         # 8.8
            self._check_marker_crossing(frame, active, out)    # 9.3.2 phase end
            self._check_tags(frame, active, out)               # 7.1 (provisional)

        self._check_kho_swap(frame, out)                       # 4.2 / 8.3 / 8.9 / 8.10
        self._check_postures(frame, out)                       # 8.4 / 8.6
        self._check_bounds(frame, out)                         # 8.5 / 7.2
        self._check_cones(frame, out)                          # 8.7
        self._check_batches(frame, out)                        # 5.3 / 7.3

        for e in out:
            if isinstance(e, FoulEvent):
                self.phase.fouls.append(e)
                self._tainted_by_foul = True
        self.events.extend(out)
        self._prev_frame = frame
        return out

    def finish(self) -> list[Event]:
        """End of half/video: resolve the open phase (rule 9.2)."""
        out = self._end_phase(self._last_t, -1, reason="end_of_play")
        self.events.extend(out)
        return out

    def score(self) -> dict:
        """Chasing-team score for the processed half (rules 6.1, 6.2)."""
        outs = sum(1 for e in self.events if isinstance(e, OutEvent))
        fouls = sum(1 for e in self.events if isinstance(e, FoulEvent))
        return {
            "outs": outs,
            "fouls": fouls,
            "points": outs * 1.0 - fouls * 0.5,
        }

    # ------------------------------------------------------- helpers / detectors

    def _cooldown_ok(self, key: str, t: float, seconds: float) -> bool:
        if t - self._cooldowns.get(key, -1e9) >= seconds:
            self._cooldowns[key] = t
            return True
        return False

    def _update_seat_settle(self, frame: FrameObservation) -> None:
        for p in frame.by_team(Team.CHASE):
            if p.posture == Posture.SEATED:
                self._seat_settled_since.setdefault(p.track_id, frame.t)
            elif p.track_id in self._seat_settled_since:
                # remember the settle state at the moment they rose, so a kho
                # given as they stand is judged on their pre-rise state (8.10)
                since = self._seat_settled_since.pop(p.track_id)
                self._settled_at_rise[p.track_id] = (
                    frame.t - since
                ) >= self.cfg.seat_settle_s

    def _is_settled(self, track_id: int, t: float) -> bool:
        since = self._seat_settled_since.get(track_id)
        if since is not None:
            return (t - since) >= self.cfg.seat_settle_s
        return self._settled_at_rise.get(track_id, False)

    def _update_active_chaser(self, frame: FrameObservation, out: list[Event]) -> None:
        standing = [
            p for p in frame.by_team(Team.CHASE) if p.posture == Posture.STANDING
        ]
        if self.active_chaser_id is None and len(standing) == 1:
            self._set_active(standing[0].track_id, frame.t)
        elif self.active_chaser_id is not None and frame.get(self.active_chaser_id) is None:
            # active chaser track lost; adopt the sole standing chaser if unambiguous
            if len(standing) == 1:
                self._set_active(standing[0].track_id, frame.t)

    def _set_active(self, track_id: int, t: float) -> None:
        self.active_chaser_id = track_id
        self.run_direction = 0
        self._chaser_side = None
        self._direction_anchor_x = None
        self._max_progress_x = None
        self.passed_seats = set()
        self._lean_count = 0

    def _update_direction(self, active: PlayerObservation) -> None:
        """Commit a run direction after sustained movement in the main area (4.3.1);
        the free zone lifts all direction restrictions (4.4.1)."""
        x = active.pos[0]
        if self.court.in_free_zone(active.pos):
            # free zone lifts direction rules and is the legal way to switch
            # sides of the centre line (4.4.1, 4.4.4)
            self.run_direction = 0
            self._chaser_side = None
            self._direction_anchor_x = None
            self._max_progress_x = None
            return
        if self._direction_anchor_x is None:
            self._direction_anchor_x = x
        if self.run_direction == 0:
            dx = x - self._direction_anchor_x
            if abs(dx) >= self.cfg.direction_commit_dist:
                self.run_direction = 1 if dx > 0 else -1
                self._max_progress_x = x
        elif self._max_progress_x is not None:
            if (x - self._max_progress_x) * self.run_direction > 0:
                self._max_progress_x = x

    def _update_passed_seats(self, active: PlayerObservation) -> None:
        if self.run_direction == 0:
            return
        for i, (sx, _sy) in enumerate(self.court.seat_positions):
            if (active.pos[0] - sx) * self.run_direction > self.cfg.kho_reach:
                self.passed_seats.add(i)

    # ---- 8.1 backtracking
    def _check_backtrack(
        self, frame: FrameObservation, active: PlayerObservation, out: list[Event]
    ) -> None:
        if self.run_direction == 0 or self._max_progress_x is None:
            return
        if not self.court.in_main_area(active.pos):
            return
        regress = (self._max_progress_x - active.pos[0]) * self.run_direction
        feet_back = regress > self.cfg.backtrack_tolerance
        if active.lean_x * self.run_direction < -self.cfg.lean_back_threshold:
            self._lean_count += 1
        else:
            self._lean_count = 0
        lean_back = self._lean_count >= self.cfg.lean_frames
        if (feet_back or lean_back) and self._cooldown_ok(
            "8.1", frame.t, self.cfg.foul_cooldown_s
        ):
            out.append(
                FoulEvent(
                    t=frame.t, frame_idx=frame.frame_idx, rule="8.1",
                    description="Active chaser backtracked"
                    + (" (feet)" if feet_back else " (shoulder lean)"),
                    track_id=active.track_id, location=active.pos,
                    confidence=0.9 if feet_back else 0.7,
                    needs_review=not feet_back,
                )
            )
            # re-anchor so one incident fires once
            self._max_progress_x = active.pos[0]
            self._lean_count = 0

    # ---- 8.2 centre line
    def _check_centre_line(
        self, frame: FrameObservation, active: PlayerObservation, out: list[Event]
    ) -> None:
        if not self.court.in_main_area(active.pos):
            return
        # lock in the chaser's side while they are clearly on it, so a later
        # crossing is judged against where they came from, not where they are
        side_now = self.court.side_of_centre(active.pos, eps=0.3)
        if self._chaser_side is None:
            if side_now != 0:
                self._chaser_side = side_now
            return
        for foot in active.foot_points():
            if not self.court.in_main_area(foot):
                continue
            toward_other = (foot[1] - self.court.centre_y) * (-self._chaser_side)
            crossed = toward_other > self.cfg.centre_cross_margin
            if crossed and self._cooldown_ok("8.2", frame.t, self.cfg.foul_cooldown_s):
                out.append(
                    FoulEvent(
                        t=frame.t, frame_idx=frame.frame_idx, rule="8.2",
                        description="Active chaser crossed/stepped on centre line in main area",
                        track_id=active.track_id, location=foot,
                    )
                )
                self._chaser_side = None  # re-lock to whichever side they settle on
                return

    # ---- 8.8 hand beyond centre line while pivoting
    def _check_pivot_hand(
        self, frame: FrameObservation, active: PlayerObservation, out: list[Event]
    ) -> None:
        if not active.wrists:
            return
        both_feet_in_fz = all(
            self.court.in_free_zone(f) for f in active.foot_points()
        )
        if both_feet_in_fz:  # explicitly allowed (4.4.5)
            return
        # only meaningful near the free zone / cone pivot
        near_pivot = active.pos[0] < 3.5 or active.pos[0] > self.court.length - 3.5
        if not near_pivot:
            return
        my_side = self.court.side_of_centre(active.pos)
        if my_side == 0:
            return
        for w in active.wrists:
            over = (self.court.centre_y - w[1]) * my_side
            if over > self.cfg.hand_over_line_margin and not self.court.in_free_zone(w):
                if self._cooldown_ok("8.8", frame.t, self.cfg.foul_cooldown_s):
                    out.append(
                        FoulEvent(
                            t=frame.t, frame_idx=frame.frame_idx, rule="8.8",
                            description="Hand placed beyond centre line while pivoting",
                            track_id=active.track_id, location=w,
                            confidence=0.6, needs_review=True,
                        )
                    )
                return

    # ---- 9.3.2 marker-line crossing ends the phase
    def _check_marker_crossing(
        self, frame: FrameObservation, active: PlayerObservation, out: list[Event]
    ) -> None:
        h = self._hist.get(active.track_id)
        prev = h.prev() if h else None
        if prev is None:
            return
        for mx in self.court.marker_lines_x:
            if (prev[0] - mx) * (active.pos[0] - mx) < 0:  # sign change = crossed
                out.extend(
                    self._end_phase(frame.t, frame.frame_idx, reason="marker_line_crossed")
                )
                return

    # ---- 7.1 tags (provisional; resolved by phase logic 9.1/9.4)
    def _check_tags(
        self, frame: FrameObservation, active: PlayerObservation, out: list[Event]
    ) -> None:
        touch_points = active.wrists if active.wrists else [active.pos]
        for runner in frame.by_team(Team.RUN):
            if runner.track_id in self.out_runner_ids:
                continue
            d = min(
                ((tp[0] - runner.pos[0]) ** 2 + (tp[1] - runner.pos[1]) ** 2) ** 0.5
                for tp in touch_points
            )
            if d <= self.cfg.tag_radius and self._cooldown_ok(
                f"tag:{runner.track_id}", frame.t, self.cfg.tag_cooldown_s
            ):
                conf = max(0.3, 1.0 - d / self.cfg.tag_radius) * (
                    1.0 if active.wrists else 0.6
                )
                tag = TagEvent(
                    t=frame.t, frame_idx=frame.frame_idx,
                    chaser_id=active.track_id, runner_id=runner.track_id,
                    location=runner.pos, confidence=conf,
                    needs_review=conf < 0.6,
                    note="" if active.wrists else "no wrist keypoints; proximity only",
                )
                self.phase.tags.append(tag)
                out.append(tag)

    # ---- 4.2 kho detection + validity (8.3, 8.9, 8.10)
    def _check_kho_swap(self, frame: FrameObservation, out: list[Event]) -> None:
        """A kho shows as a swap: a settled seated chaser rises while the active
        chaser is within reach of that seat."""
        if self.active_chaser_id is None or self._prev_frame is None:
            return
        active_prev = self._prev_frame.get(self.active_chaser_id)
        active_now = frame.get(self.active_chaser_id)
        if active_prev is None or active_now is None:
            return
        for p in frame.by_team(Team.CHASE):
            if p.track_id == self.active_chaser_id:
                continue
            prev = self._prev_frame.get(p.track_id)
            if prev is None:
                continue
            rising = prev.posture in (Posture.SEATED, Posture.TRANSITION) and p.posture in (
                Posture.TRANSITION, Posture.STANDING
            )
            if not rising or prev.posture == p.posture:
                continue
            seat_idx, seat_d = self.court.nearest_seat(p.pos)
            if seat_d > 1.0:
                continue
            giver_d = min(
                ((f[0] - p.pos[0]) ** 2 + (f[1] - p.pos[1]) ** 2) ** 0.5
                for f in active_now.foot_points()
            )
            if giver_d > self.cfg.kho_reach * 2.5:
                continue  # rising far from the active chaser: handled by 8.4 monitor
            if not self._cooldown_ok("kho", frame.t, self.cfg.kho_swap_window_s):
                return
            self._emit_kho(frame, active_now, p, seat_idx, giver_d, out)
            return

    def _emit_kho(
        self,
        frame: FrameObservation,
        giver: PlayerObservation,
        receiver: PlayerObservation,
        seat_idx: int,
        giver_foot_dist: float,
        out: list[Event],
    ) -> None:
        cfg, court = self.cfg, self.court
        valid = True

        # 8.3a — foot not behind the seated chaser when kho given
        behind_ok = giver_foot_dist <= cfg.kho_reach
        if behind_ok and receiver.facing != 0:
            giver_side = court.side_of_centre(giver.pos)
            if giver_side == receiver.facing:  # at the chaser's front / wrong side (4.2.2)
                behind_ok = False
        if not behind_ok:
            valid = False
            out.append(
                FoulEvent(
                    t=frame.t, frame_idx=frame.frame_idx, rule="8.3",
                    description="Kho given without a foot behind the seated chaser",
                    track_id=giver.track_id, location=receiver.pos,
                    confidence=0.75, needs_review=True,
                )
            )
        # 8.3b — kho to a chaser already passed
        if seat_idx in self.passed_seats:
            valid = False
            out.append(
                FoulEvent(
                    t=frame.t, frame_idx=frame.frame_idx, rule="8.3",
                    description=f"Kho given to already-passed chaser (seat {seat_idx + 1})",
                    track_id=giver.track_id, location=receiver.pos,
                )
            )
        # 8.10 — receiver had not fully returned to a seated position
        if not self._is_settled(receiver.track_id, frame.t):
            valid = False
            out.append(
                FoulEvent(
                    t=frame.t, frame_idx=frame.frame_idx, rule="8.10",
                    description="Kho given to a chaser not fully seated",
                    track_id=giver.track_id, location=receiver.pos,
                    confidence=0.7, needs_review=True,
                )
            )
        # 8.9 — front foot on/over free-zone marker line at an end seat
        if seat_idx in (0, len(court.seat_positions) - 1):
            mx = court.nearest_marker_line_x(receiver.pos[0])
            inward = 1 if mx == court.marker_lines_x[0] else -1  # +x into main area
            for f in giver.foot_points():
                if (f[0] - mx) * inward < cfg.line_eps:  # touching or beyond the line
                    valid = False
                    out.append(
                        FoulEvent(
                            t=frame.t, frame_idx=frame.frame_idx, rule="8.9",
                            description="Front foot touched/crossed free-zone marker "
                            "line giving kho to end chaser",
                            track_id=giver.track_id, location=f,
                        )
                    )
                    break

        out.append(
            KhoEvent(
                t=frame.t, frame_idx=frame.frame_idx,
                giver_id=giver.track_id, receiver_id=receiver.track_id,
                seat_index=seat_idx, valid=valid,
                confidence=0.8 if frame.kho_call else 0.65,
                needs_review=not frame.kho_call,
                note="" if frame.kho_call else "no audio confirmation of the call",
            )
        )
        self._last_kho_t = frame.t
        self._last_kho_receiver = receiver.track_id
        # control passes regardless; validity affects phase reset (9.4)
        self._set_active(receiver.track_id, frame.t)
        if valid:
            out.extend(self._end_phase(frame.t, frame.frame_idx, reason="valid_kho"))

    # ---- 8.4 / 8.6 posture fouls
    def _check_postures(self, frame: FrameObservation, out: list[Event]) -> None:
        cfg = self.cfg
        in_swap_window = (frame.t - self._last_kho_t) <= cfg.kho_swap_window_s
        chasers = frame.by_team(Team.CHASE)
        standing = [p for p in chasers if p.posture == Posture.STANDING]

        # 8.4 — a seated chaser stands early (the giver sitting down after a kho
        # is covered by the swap window, not a foul)
        for p in chasers:
            if p.track_id == self.active_chaser_id:
                continue
            if in_swap_window:
                continue
            if p.posture == Posture.STANDING and self.court.in_main_area(p.pos):
                n = self._early_stand_frames.get(p.track_id, 0) + 1
                self._early_stand_frames[p.track_id] = n
                if n == cfg.posture_frames and self._cooldown_ok(
                    f"8.4:{p.track_id}", frame.t, cfg.foul_cooldown_s
                ):
                    out.append(
                        FoulEvent(
                            t=frame.t, frame_idx=frame.frame_idx, rule="8.4",
                            description="Seated chaser stood before receiving a valid kho",
                            track_id=p.track_id, location=p.pos,
                        )
                    )
            else:
                self._early_stand_frames.pop(p.track_id, None)

        # 8.6 — more than one chaser standing (outside a legitimate swap)
        if len(standing) > 1 and not in_swap_window:
            self._standing_count_frames += 1
            if self._standing_count_frames == cfg.posture_frames and self._cooldown_ok(
                "8.6", frame.t, cfg.foul_cooldown_s
            ):
                out.append(
                    FoulEvent(
                        t=frame.t, frame_idx=frame.frame_idx, rule="8.6",
                        description=f"{len(standing)} chasers standing simultaneously",
                        track_id=None,
                    )
                )
        else:
            self._standing_count_frames = 0

    # ---- 8.5 chasers / 7.2 runners out of bounds
    def _check_bounds(self, frame: FrameObservation, out: list[Event]) -> None:
        m = self.cfg.out_of_bounds_margin
        for p in frame.players:
            fully_out = all(
                not self.court.in_bounds(f, margin=m) for f in p.foot_points()
            )
            if not fully_out:
                continue
            if p.team == Team.CHASE:
                if self._cooldown_ok(f"8.5:{p.track_id}", frame.t, self.cfg.foul_cooldown_s):
                    out.append(
                        FoulEvent(
                            t=frame.t, frame_idx=frame.frame_idx, rule="8.5",
                            description="Chaser stepped fully outside the court",
                            track_id=p.track_id, location=p.pos,
                        )
                    )
            elif p.track_id not in self.out_runner_ids:
                self.out_runner_ids.add(p.track_id)
                out.append(
                    OutEvent(
                        t=frame.t, frame_idx=frame.frame_idx, rule="7.2",
                        runner_id=p.track_id,
                        description="Runner stepped out of bounds",
                    )
                )

    # ---- 8.7 cone touch
    def _check_cones(self, frame: FrameObservation, out: list[Event]) -> None:
        for p in frame.by_team(Team.CHASE):
            pts = p.foot_points() + p.wrists
            for cone in self.court.cone_positions:
                if any(
                    ((q[0] - cone[0]) ** 2 + (q[1] - cone[1]) ** 2) ** 0.5
                    <= self.cfg.cone_radius
                    for q in pts
                ):
                    if self._cooldown_ok(f"8.7:{p.track_id}", frame.t, self.cfg.foul_cooldown_s):
                        out.append(
                            FoulEvent(
                                t=frame.t, frame_idx=frame.frame_idx, rule="8.7",
                                description="Chaser touched the cone",
                                track_id=p.track_id, location=cone,
                                confidence=0.6, needs_review=True,
                                note="proximity-based; verify contact on video",
                            )
                        )

    # ---- 5.3 / 7.3 batch entry
    def _check_batches(self, frame: FrameObservation, out: list[Event]) -> None:
        on_court = [
            r for r in frame.by_team(Team.RUN)
            if r.track_id not in self.out_runner_ids and self.court.in_bounds(r.pos, 0.2)
        ]
        if not on_court:
            if self._batch_cleared_t is None:
                self._batch_cleared_t = frame.t
            elif frame.t - self._batch_cleared_t > self.cfg.batch_entry_seconds:
                if self._cooldown_ok("7.3", frame.t, 5.0):
                    out.append(
                        OutEvent(
                            t=frame.t, frame_idx=frame.frame_idx, rule="7.3",
                            description=f"Batch {self._batch_count + 1} failed to enter "
                            "within 3 seconds",
                            confidence=0.7, needs_review=True,
                        )
                    )
        elif self._batch_cleared_t is not None:
            self._batch_cleared_t = None
            self._batch_count += 1

    # ---- rule 9: phase resolution
    def _end_phase(self, t: float, frame_idx: int, reason: str) -> list[Event]:
        events: list[Event] = []
        phase = self.phase
        clean = not phase.fouls and not self._tainted_by_foul
        for tag in phase.tags:
            if clean:
                if tag.runner_id not in self.out_runner_ids:
                    self.out_runner_ids.add(tag.runner_id)
                    events.append(
                        OutEvent(
                            t=tag.t, frame_idx=tag.frame_idx, rule="7.1",
                            runner_id=tag.runner_id,
                            description="Runner tagged by active chaser",
                            confidence=tag.confidence, needs_review=tag.needs_review,
                        )
                    )
            else:
                events.append(
                    PhaseEndEvent(
                        t=t, frame_idx=frame_idx, reason=reason,
                        note=f"tag on runner {tag.runner_id} voided by foul in the "
                        "same phase (rules 9.1/9.4)",
                    )
                )
        events.append(PhaseEndEvent(t=t, frame_idx=frame_idx, reason=reason))
        # 9.4: only a valid kho clears the foul taint
        if reason in ("valid_kho", "end_of_play"):
            self._tainted_by_foul = False
        self.phase = _PhaseState(index=phase.index + 1, started_t=t)
        return events
