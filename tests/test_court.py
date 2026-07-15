import math

from khosight.calibration.court import Court, CourtCalibration, estimate_homography


def test_seat_positions_match_rule_4_1_2():
    court = Court()
    seats = court.seat_positions
    assert len(seats) == 8
    gap = 14.0 / 9.0
    # first seat one gap inside the free-zone line, even spacing, symmetric
    assert math.isclose(seats[0][0], 2.0 + gap)
    assert math.isclose(seats[-1][0], 16.0 - gap)
    for a, b in zip(seats, seats[1:]):
        assert math.isclose(b[0] - a[0], gap)
    assert all(y == 4.5 for _, y in seats)


def test_zone_predicates():
    court = Court()
    assert court.in_free_zone((1.0, 4.0))
    assert court.in_free_zone((17.5, 8.0))
    assert not court.in_free_zone((2.0, 4.0))  # marker line is not free zone (1.6)
    assert court.in_main_area((2.0, 4.0))
    assert court.in_main_area((9.0, 0.0))
    assert not court.in_main_area((1.9, 4.0))
    assert court.in_bounds((0.0, 0.0))
    assert not court.in_bounds((-0.1, 5.0))
    assert court.side_of_centre((5.0, 6.0)) == 1
    assert court.side_of_centre((5.0, 3.0)) == -1
    assert court.side_of_centre((5.0, 4.5)) == 0


def test_homography_roundtrip():
    # synthetic camera: projective map from court metres to "pixels"
    import numpy as np

    H_true = np.array([[80.0, 12.0, 100.0], [4.0, -60.0, 700.0], [0.001, 0.002, 1.0]])

    def project(p):
        v = H_true @ np.array([p[0], p[1], 1.0])
        return (v[0] / v[2], v[1] / v[2])

    court_pts = [(0, 0), (18, 0), (18, 9), (0, 9), (2, 4.5), (16, 4.5), (9, 0)]
    img_pts = [project(p) for p in court_pts]
    H = estimate_homography(img_pts, court_pts)
    calib = CourtCalibration(H)
    for cp, ip in zip(court_pts, img_pts):
        rx, ry = calib.image_to_court([ip])[0]
        assert abs(rx - cp[0]) < 1e-6 and abs(ry - cp[1]) < 1e-6


def test_calibration_save_load(tmp_path):
    court_pts = [(0, 0), (18, 0), (18, 9), (0, 9)]
    img_pts = [(10, 500), (900, 480), (860, 60), (60, 40)]
    calib = CourtCalibration.from_correspondences(img_pts, court_pts, (960, 540))
    path = str(tmp_path / "calib.json")
    calib.save(path)
    loaded = CourtCalibration.load(path)
    a = calib.image_to_court([(400, 300)])[0]
    b = loaded.image_to_court([(400, 300)])[0]
    assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9
