import cv2
import numpy as np
import pytest

from idv.face.align import ARCFACE_TEMPLATE, align, similarity_transform
from idv.face.pairs import make_pairs
from idv.face.scrfd import anchor_centers, distance_to_box, distance_to_points, nms


def _similarity(angle_deg, scale, tx, ty):
    a = np.deg2rad(angle_deg)
    return np.array(
        [[scale * np.cos(a), -scale * np.sin(a), tx], [scale * np.sin(a), scale * np.cos(a), ty]]
    )


def _apply(m, pts):
    return pts @ m[:, :2].T + m[:, 2]


def test_similarity_transform_recovers_known_pose():
    src_from_template = _similarity(20, 2.5, 140, 60)
    landmarks = _apply(src_from_template, ARCFACE_TEMPLATE)
    m = similarity_transform(landmarks)
    np.testing.assert_allclose(_apply(m, landmarks), ARCFACE_TEMPLATE, atol=1e-3)


def test_align_moves_landmark_pixels_onto_template():
    m_src = _similarity(-15, 3.0, 200, 120)
    landmarks = _apply(m_src, ARCFACE_TEMPLATE)
    img = np.zeros((600, 600, 3), np.uint8)
    for x, y in landmarks:
        cv2.circle(img, (round(x), round(y)), 6, (255, 255, 255), -1)

    out = align(img, landmarks)
    assert out.shape == (112, 112, 3)
    for x, y in ARCFACE_TEMPLATE:
        assert out[round(y), round(x)].min() == 255


def test_anchor_centers_layout():
    c = anchor_centers(2, 3, stride=8, num_anchors=2)
    assert c.shape == (12, 2)
    np.testing.assert_array_equal(c[:4], [[0, 0], [0, 0], [8, 0], [8, 0]])
    np.testing.assert_array_equal(c[-1], [16, 8])


def test_distance_decoding():
    centers = np.array([[100.0, 50.0]])
    box = distance_to_box(centers, np.array([[10.0, 5.0, 20.0, 15.0]]))
    np.testing.assert_array_equal(box, [[90, 45, 120, 65]])
    pts = distance_to_points(centers, np.array([[1.0, 2.0, -3.0, -4.0]]))
    np.testing.assert_array_equal(pts, [[[101, 52], [97, 46]]])


def test_nms_drops_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    scores = np.array([0.8, 0.9, 0.7])
    assert nms(boxes, scores, iou=0.4) == [1, 2]


@pytest.fixture
def cohort():
    rng = np.random.default_rng(0)
    videos = {f"s{i:03d}": [f"s{i:03d}_v{j}" for j in range(rng.integers(1, 6))] for i in range(60)}
    keys = {s: ("F" if i % 2 else "M", "18-30" if i % 3 else "31-45") for i, s in enumerate(videos)}
    return videos, keys


def test_pairs_genuine_and_impostor_rules(cohort):
    videos, keys = cohort
    pairs = make_pairs(videos, keys, selfies_per_subject=2, impostors_per_subject=5, seed=0)
    id_video = {}
    for p in pairs:
        # One ID photo per subject across every pair it appears in.
        assert id_video.setdefault(p.id_subject, p.id_video) == p.id_video
        assert p.id_video in videos[p.id_subject]
        assert p.selfie_video in videos[p.selfie_subject]
        if p.genuine:
            assert p.selfie_video != p.id_video
        else:
            assert keys[p.selfie_subject] == keys[p.id_subject]
    assert len(set(pairs)) == len(pairs)


def test_single_video_subjects_get_no_genuine_pair(cohort):
    videos, keys = cohort
    pairs = make_pairs(videos, keys, seed=0)
    singles = {s for s, v in videos.items() if len(v) == 1}
    assert singles
    assert not any(p.genuine and p.id_subject in singles for p in pairs)


def test_pairs_deterministic(cohort):
    videos, keys = cohort
    assert make_pairs(videos, keys, seed=5) == make_pairs(videos, keys, seed=5)
    assert make_pairs(videos, keys, seed=5) != make_pairs(videos, keys, seed=6)
