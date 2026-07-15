"""Rule-engine tests on synthetic court-space trajectories (no models needed).

Conventions: 30 fps synthetic frames; chase team = track ids 1..8 (seated, on
their rulebook seat slots) + id 100 (active chaser); runners = ids 200+.
"""

from khosight.calibration.court import Court
from khosight.rules.engine import RuleEngine
from khosight.rules.events import (
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

FPS = 30.0
COURT = Court()
ACTIVE = 100


def seated_chasers(exclude=()):
    players = []
    for i, seat in enumerate(COURT.seat_positions):
        tid = i + 1
        if tid in exclude:
            continue
        players.append(
            PlayerObservation(
                track_id=tid, team=Team.CHASE, pos=seat,
                posture=Posture.SEATED, facing=1 if i % 2 == 0 else -1,
            )
        )
    return players


def runner(tid=200, pos=(12.0, 2.0)):
    return PlayerObservation(track_id=tid, team=Team.RUN, pos=pos,
                             posture=Posture.STANDING)


def active(pos, posture=Posture.STANDING, wrists=(), lean_x=0.0, tid=ACTIVE):
    return PlayerObservation(track_id=tid, team=Team.CHASE, pos=pos,
                             posture=posture, wrists=list(wrists), lean_x=lean_x)


def run_frames(engine, frames):
    for i, players in enumerate(frames):
        engine.update(FrameObservation(t=i / FPS, frame_idx=i, players=players))
    return engine


def fouls(engine, rule=None):
    return [e for e in engine.events
            if isinstance(e, FoulEvent) and (rule is None or e.rule == rule)]


def outs(engine, rule=None):
    return [e for e in engine.events
            if isinstance(e, OutEvent) and (rule is None or e.rule == rule)]


def lerp(a, b, n):
    return [a + (b - a) * k / (n - 1) for k in range(n)]


# --------------------------------------------------------------- 8.1 backtrack


def test_backtrack_feet_foul_8_1():
    frames = []
    # commit direction +x: 5.0 -> 6.2, then regress to 5.6 (0.6 m > 0.3 tol)
    xs = lerp(5.0, 6.2, 20) + lerp(6.2, 5.6, 15)
    for x in xs:
        frames.append(seated_chasers() + [runner(), active((x, 3.6))])
    engine = run_frames(RuleEngine(), frames)
    assert len(fouls(engine, "8.1")) == 1
    f = fouls(engine, "8.1")[0]
    assert "feet" in f.description


def test_natural_reach_not_a_foul_4_3_4():
    # small oscillation within tolerance must not fire
    xs = lerp(5.0, 6.2, 20) + lerp(6.2, 6.0, 10) + lerp(6.0, 7.0, 10)
    frames = [seated_chasers() + [runner(), active((x, 3.6))] for x in xs]
    engine = run_frames(RuleEngine(), frames)
    assert fouls(engine, "8.1") == []


def test_backtrack_lean_foul_8_1():
    frames = []
    for x in lerp(5.0, 6.2, 20):
        frames.append(seated_chasers() + [runner(), active((x, 3.6))])
    for _ in range(6):  # sustained backwards shoulder lean, feet planted
        frames.append(seated_chasers() + [runner(), active((6.2, 3.6), lean_x=-0.3)])
    engine = run_frames(RuleEngine(), frames)
    lean_fouls = [f for f in fouls(engine, "8.1") if "lean" in f.description]
    assert len(lean_fouls) == 1
    assert lean_fouls[0].needs_review


# ------------------------------------------------------------- 8.2 centre line


def test_centre_line_cross_foul_8_2():
    ys = lerp(3.6, 3.6, 10) + lerp(3.6, 4.75, 15)  # settle on a side, then cross
    frames = [seated_chasers() + [runner(), active((8.0, y))] for y in ys]
    engine = run_frames(RuleEngine(), frames)
    assert len(fouls(engine, "8.2")) == 1


def test_no_centre_line_foul_when_staying_on_side():
    ys = lerp(3.6, 4.42, 25)  # approaches the line but stays on own side
    frames = [seated_chasers() + [runner(), active((8.0, y))] for y in ys]
    engine = run_frames(RuleEngine(), frames)
    assert fouls(engine, "8.2") == []


# ------------------------------------------------------------------- 4.2 kho


def kho_swap_frames(receiver_tid=3, giver_start_x=5.2, settle_frames=40):
    """Active chaser runs to a seat and gives kho: receiver rises, giver sits."""
    seat_x, seat_y = COURT.seat_positions[receiver_tid - 1]
    frames = []
    for _ in range(settle_frames):  # everyone settled (8.10 eligibility)
        frames.append(seated_chasers() + [runner(), active((giver_start_x, 3.9))])
    for x in lerp(giver_start_x, seat_x - 0.3, 10):  # approach behind the seat
        frames.append(seated_chasers() + [runner(), active((x, 3.9))])
    # swap: receiver transitions to standing, giver to seated
    for k, (rec_p, giv_p) in enumerate(
        [(Posture.TRANSITION, Posture.TRANSITION),
         (Posture.STANDING, Posture.SEATED),
         (Posture.STANDING, Posture.SEATED)]
    ):
        others = seated_chasers(exclude=(receiver_tid,))
        rec = PlayerObservation(track_id=receiver_tid, team=Team.CHASE,
                                pos=(seat_x, seat_y), posture=rec_p, facing=1)
        giv = active((seat_x - 0.3, 3.9), posture=giv_p)
        frames.append(others + [rec, giv, runner()])
    return frames


def test_valid_kho_transfers_chase_and_ends_phase():
    engine = run_frames(RuleEngine(), kho_swap_frames(receiver_tid=3))
    khos = [e for e in engine.events if isinstance(e, KhoEvent)]
    assert len(khos) == 1
    assert khos[0].valid
    assert khos[0].receiver_id == 3
    assert engine.active_chaser_id == 3
    assert any(isinstance(e, PhaseEndEvent) and e.reason == "valid_kho"
               for e in engine.events)
    assert fouls(engine) == []


def test_kho_to_passed_chaser_is_foul_8_3():
    # active runs past seat 2 (committing +x), then seat 2 occupant rises
    seat2_x = COURT.seat_positions[1][0]
    frames = []
    for _ in range(40):
        frames.append(seated_chasers() + [runner(), active((3.7, 3.9))])
    for x in lerp(3.7, seat2_x + 1.1, 25):  # passes seat 2 by > kho_reach
        frames.append(seated_chasers() + [runner(), active((x, 3.9))])
    others = seated_chasers(exclude=(2,))
    for rec_p in (Posture.TRANSITION, Posture.STANDING):
        rec = PlayerObservation(track_id=2, team=Team.CHASE,
                                pos=COURT.seat_positions[1], posture=rec_p, facing=1)
        frames.append(others + [rec, active((seat2_x + 1.1, 3.9)), runner()])
    engine = run_frames(RuleEngine(), frames)
    khos = [e for e in engine.events if isinstance(e, KhoEvent)]
    assert len(khos) == 1 and not khos[0].valid
    assert any(f.rule == "8.3" for f in fouls(engine))


# ------------------------------------------------- 7.1 tags + rule 9 phases


def tag_then_cross_frames(with_foul=False):
    """Active tags a runner, then crosses the free-zone marker line (ends phase)."""
    frames = []
    for _ in range(5):
        frames.append(seated_chasers() + [runner(210, (14.6, 3.4)), active((13.0, 3.6))])
    for x in lerp(13.0, 14.5, 10):  # commit +x and close on the runner
        frames.append(seated_chasers() + [runner(210, (14.6, 3.4)), active((x, 3.6))])
    # touch: wrist right next to the runner
    frames.append(seated_chasers()
                  + [runner(210, (14.6, 3.4)),
                     active((14.5, 3.6), wrists=[(14.58, 3.42)])])
    if with_foul:  # backtrack after the tag, same phase
        for x in lerp(14.5, 13.9, 10):
            frames.append(seated_chasers() + [runner(210, (10.0, 1.5)), active((x, 3.6))])
        run_out = lerp(13.9, 16.3, 12)
    else:
        run_out = lerp(14.5, 16.3, 12)
    for x in run_out:  # cross marker line x=16 -> phase ends (9.3.2)
        frames.append(seated_chasers() + [runner(210, (10.0, 1.5)), active((x, 3.6))])
    return frames


def test_tag_confirmed_as_out_when_phase_clean():
    engine = run_frames(RuleEngine(), tag_then_cross_frames(with_foul=False))
    assert any(isinstance(e, TagEvent) for e in engine.events)
    assert len(outs(engine, "7.1")) == 1
    assert outs(engine, "7.1")[0].runner_id == 210
    assert engine.score()["points"] == 1.0


def test_foul_in_same_phase_voids_tag_9_1():
    engine = run_frames(RuleEngine(), tag_then_cross_frames(with_foul=True))
    assert len(fouls(engine, "8.1")) == 1
    assert outs(engine, "7.1") == []  # runner returns to play (9.1)
    assert any("voided" in e.note for e in engine.events
               if isinstance(e, PhaseEndEvent))
    assert engine.score()["points"] == -0.5


def test_end_of_play_resolves_open_phase_9_2():
    frames = []
    for _ in range(5):
        frames.append(seated_chasers() + [runner(210, (14.6, 3.4)), active((13.0, 3.6))])
    for x in lerp(13.0, 14.5, 10):
        frames.append(seated_chasers() + [runner(210, (14.6, 3.4)), active((x, 3.6))])
    frames.append(seated_chasers()
                  + [runner(210, (14.6, 3.4)),
                     active((14.5, 3.6), wrists=[(14.58, 3.42)])])
    engine = run_frames(RuleEngine(), frames)
    assert outs(engine, "7.1") == []   # phase still open
    engine.finish()
    assert len(outs(engine, "7.1")) == 1


# ------------------------------------------------------------ bounds & batches


def test_runner_out_of_bounds_7_2_and_chaser_out_8_5():
    frames = []
    for _ in range(5):
        frames.append(seated_chasers()
                      + [runner(220, (9.0, -0.5)),           # fully outside
                         active((-0.4, 6.0))])               # chaser fully outside
    engine = run_frames(RuleEngine(), frames)
    assert len(outs(engine, "7.2")) == 1
    assert outs(engine, "7.2")[0].runner_id == 220
    assert len(fouls(engine, "8.5")) == 1


def test_batch_entry_timeout_7_3():
    frames = []
    n_active = 20  # runner on court ~0.7 s
    n_empty = 110  # then absent for ~3.7 s > 3 s limit
    for i in range(n_active + n_empty):
        players = seated_chasers() + [active((8.0, 3.6))]
        if i < n_active:
            players.append(runner(230, (12.0, 2.0)))
        frames.append(players)
    engine = run_frames(RuleEngine(), frames)
    late = outs(engine, "7.3")
    assert len(late) == 1
    assert late[0].needs_review


# ------------------------------------------------------------- posture fouls


def test_early_stand_foul_8_4_and_multiple_standing_8_6():
    frames = []
    for i in range(15):
        others = seated_chasers(exclude=(6,))
        # seat 6 (far from the active chaser) stands with no kho
        p6 = PlayerObservation(
            track_id=6, team=Team.CHASE, pos=COURT.seat_positions[5],
            posture=Posture.SEATED if i < 5 else Posture.STANDING, facing=-1,
        )
        frames.append(others + [p6, active((3.0, 3.6)), runner()])
    engine = run_frames(RuleEngine(), frames)
    assert len(fouls(engine, "8.4")) == 1
    assert len(fouls(engine, "8.6")) == 1


# ----------------------------------------------------------------- 8.7 / 8.8


def test_cone_touch_foul_8_7():
    frames = [seated_chasers() + [runner(), active((15.9, 4.44))] for _ in range(5)]
    engine = run_frames(RuleEngine(), frames)
    assert len(fouls(engine, "8.7")) == 1


def test_pivot_hand_over_centre_line_8_8():
    frames = [
        seated_chasers()
        + [runner(), active((2.5, 3.8), wrists=[(2.6, 4.7)])]
        for _ in range(5)
    ]
    engine = run_frames(RuleEngine(), frames)
    assert len(fouls(engine, "8.8")) == 1


def test_hand_over_centre_allowed_with_both_feet_in_free_zone_4_4_5():
    frames = [
        seated_chasers()
        + [runner(), active((1.5, 3.8), wrists=[(1.6, 4.9)])]
        for _ in range(5)
    ]
    engine = run_frames(RuleEngine(), frames)
    assert fouls(engine, "8.8") == []


# ---------------------------------------------------------------- score maths


def test_score_composition_rule_6():
    engine = run_frames(RuleEngine(), tag_then_cross_frames(with_foul=False))
    s = engine.score()
    assert s == {"outs": 1, "fouls": 0, "points": 1.0}
