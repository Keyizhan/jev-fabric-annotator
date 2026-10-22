import cv2, numpy as np
from jevfabric.geometry import chirality, clean_segment, family_features, geometry_vote


def b_like(size=300):
    """Block body with its bottom-LEFT corner cut off and a thin horn rising from the top: the 'B' layout (thin part up)."""
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [np.array([(100, 150), (200, 150), (200, 250), (140, 250), (100, 210)])], 255)   # body, BL corner missing
    cv2.rectangle(m, (140, 80), (152, 150), 255, -1)                                                   # thin horn
    return m


def rotate(mask, deg):
    big = cv2.copyMakeBorder(mask, 150, 150, 150, 150, cv2.BORDER_CONSTANT)
    M = cv2.getRotationMatrix2D((big.shape[1] / 2, big.shape[0] / 2), deg, 1.0)
    return (cv2.warpAffine(big, M, big.shape[::-1], flags=cv2.INTER_NEAREST) > 0).astype(np.uint8) * 255


def test_b_layout_is_left():
    assert chirality(b_like())["side"] == "L"


def test_mirror_swaps_side():
    assert chirality(b_like()[:, ::-1].copy())["side"] == "R"


def test_rotation_invariant():
    for deg in (0, 37, 90, 145, 200, 290):
        assert chirality(rotate(b_like(), deg))["side"] == "L", deg
        assert chirality(rotate(b_like()[:, ::-1].copy(), deg))["side"] == "R", deg


def crop_with(mask):
    bgr = np.full((275, 330, 3), 30, np.uint8); m = cv2.resize(mask, (330, 275)) if mask.shape != (275, 330) else mask
    bgr[m > 0] = (190, 205, 215)
    return bgr


def test_wedge_is_a():
    m = np.zeros((275, 330), np.uint8); cv2.fillPoly(m, [np.array([(60, 150), (240, 130), (250, 165), (70, 190)])], 255)
    asp, sol = family_features(m); assert asp >= 2.2 and sol >= 0.82
    assert geometry_vote(crop_with(m)) == "A"


def test_block_with_horn_is_b_and_mirror_is_e():
    m = np.zeros((275, 330), np.uint8); m[:, :300] = b_like()[:275, :]
    assert geometry_vote(crop_with(m)) == "B"
    mirrored = np.zeros((275, 330), np.uint8); mirrored[:, :300] = m[:, :300][:, ::-1]
    assert geometry_vote(crop_with(mirrored)) == "E"


def test_empty_mat_has_no_piece():
    assert clean_segment(np.full((275, 330, 3), 30, np.uint8)) is None
