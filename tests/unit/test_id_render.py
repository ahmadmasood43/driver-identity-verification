import json

import cv2
import numpy as np
import pytest

from idv.data.midv import load_annotations
from idv.face.align import ARCFACE_TEMPLATE
from idv.synth.id_render import grabcut_matte, is_greyscale, render


def _via(regions):
    return {
        "_via_img_metadata": {
            "00.jpg1": {
                "filename": "00.jpg",
                "regions": [
                    {
                        "shape_attributes": shape,
                        "region_attributes": {"field_name": name, "value": value},
                    }
                    for name, value, shape in regions
                ],
            }
        }
    }


@pytest.mark.parametrize(
    ("raw", "gender"),
    [
        ("M", "M"),
        ("F", "F"),
        ("K/M", "M"),
        ("Ž / F", "F"),
        ("ЖЕН.", "F"),
        ("МУЖ.", "M"),
        ("", None),
    ],
)
def test_template_gender(tmp_path, raw, gender):
    rect = {"name": "rect", "x": 1, "y": 2, "width": 3, "height": 4}
    p = tmp_path / "xx_id.json"
    p.write_text(json.dumps(_via([("gender", raw, rect)])))
    assert load_annotations(p, tmp_path)[0].gender == gender


def test_annotation_shapes(tmp_path):
    rect = {"name": "rect", "x": 10, "y": 20, "width": 30, "height": 40}
    poly = {"name": "polygon", "all_points_x": [0, 5, 4], "all_points_y": [1, 2, 9]}
    p = tmp_path / "esp_id.json"
    p.write_text(json.dumps(_via([("photo", "", rect), ("doc", "", poly)])))
    t = load_annotations(p, tmp_path / "images")[0]
    assert t.doc_type == "esp_id"
    assert t.image == tmp_path / "images" / "00.jpg"
    assert t.fields["photo"].box == (10, 20, 30, 40)
    assert t.fields["doc"].box == (0, 1, 5, 8)


def _scene():
    """A bright 'face' disc with dark landmark dots on a mid-grey background."""
    face = np.full((400, 400, 3), 90, np.uint8)
    lm = ARCFACE_TEMPLATE * 2.4 + np.float32([65, 60])
    cv2.ellipse(face, (200, 190), (110, 140), 0, 0, 360, (220, 200, 180), -1)
    for px, py in lm:
        cv2.circle(face, (round(px), round(py)), 5, (0, 0, 0), -1)
    return face, lm


def test_render_places_landmarks_and_leaves_rest_untouched():
    face, lm = _scene()
    rng = np.random.default_rng(0)
    template = rng.integers(0, 255, (300, 500, 3), dtype=np.uint8)
    box = (300, 40, 150, 200)
    target = ARCFACE_TEMPLATE * 1.2 + np.float32([box[0] + 8, box[1] + 30])
    matte = np.ones(face.shape[:2], np.float32)

    out = render(template, box, target, face, lm, matte)
    assert out.shape == template.shape

    # Landmark dots land at the target positions (darkest pixels in their neighbourhood).
    grey = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    for tx, ty in target:
        tx, ty = round(tx), round(ty)
        patch = grey[ty - 6 : ty + 7, tx - 6 : tx + 7]
        dy, dx = np.unravel_index(np.argmin(patch), patch.shape)
        assert abs(dx - 6) <= 2 and abs(dy - 6) <= 2

    # Outside the photo box plus its feather margin, the document is unchanged.
    edge = max(2, box[2] // 60)
    keep = np.ones(template.shape[:2], bool)
    keep[box[1] - edge : box[1] + box[3] + edge, box[0] - edge : box[0] + box[2] + edge] = False
    assert np.array_equal(out[keep], template[keep])


def test_render_follows_template_colour_mode():
    face, lm = _scene()
    box = (20, 20, 150, 200)
    target = ARCFACE_TEMPLATE * 1.2 + np.float32([box[0] + 8, box[1] + 30])
    matte = np.ones(face.shape[:2], np.float32)
    x, y, w, h = box

    grey_doc = np.full((260, 200, 3), 200, np.uint8)
    out = render(grey_doc, box, target, face, lm, matte)
    assert is_greyscale(out[y : y + h, x : x + w])

    # A colour print: saturated, with some texture so it has contrast to match.
    colour_doc = grey_doc.copy()
    ramp = np.linspace(0.6, 1.0, w)[None, :, None]
    colour_doc[y : y + h, x : x + w] = (np.array([60, 120, 220]) * ramp).astype(np.uint8)
    colour_face = face.copy()
    colour_face[face[..., 0] == 220] = (40, 120, 230)
    out = render(colour_doc, box, target, colour_face, lm, matte)
    assert not is_greyscale(out[y : y + h, x : x + w])


def test_grabcut_matte_separates_subject_from_plain_background():
    face, _ = _scene()
    box = np.array([90, 50, 310, 330], float)
    m = grabcut_matte(face, box)
    assert m.shape == face.shape[:2]
    assert m[190, 200] > 0.9  # face centre
    assert m[10, 10] < 0.1  # corner
