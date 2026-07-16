from khosight.perception.models import PersonDetection, assign_roles, bbox_iou


def test_bbox_iou():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_assign_roles_matches_by_overlap_and_ignores_cones():
    dets = [
        PersonDetection(track_id=1, bbox=(100, 100, 140, 200), confidence=0.9),
        PersonDetection(track_id=2, bbox=(300, 120, 360, 180), confidence=0.9),
        PersonDetection(track_id=3, bbox=(500, 100, 540, 200), confidence=0.9),
    ]
    role_boxes = [
        ((102, 98, 138, 202), "chaser-active"),   # overlaps det 1
        ((295, 118, 358, 182), "chaser-seated"),  # overlaps det 2
        ((498, 102, 542, 198), "cone"),           # overlaps det 3 but is a cone
        ((900, 900, 950, 950), "runner"),         # overlaps nothing
    ]
    assign_roles(dets, role_boxes)
    assert dets[0].role_class == "chaser-active"
    assert dets[1].role_class == "chaser-seated"
    assert dets[2].role_class is None  # cones never label people
