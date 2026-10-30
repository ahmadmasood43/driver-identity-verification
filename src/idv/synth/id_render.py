"""Put a real person's face into a MIDV-2020 template so it reads like an ID photo.

The face is aligned to the landmarks of the template's own (synthetic) portrait, cut out
with a person matte, composited on a plain background, then made to match the print:
greyscale where the template prints greyscale, the template's brightness and contrast,
and a slight print blur. The output is a clean document image; capture effects (blur,
glare, perspective) are applied later as perturbations.
"""

import cv2
import numpy as np

GREYSCALE_SATURATION = 30  # mean HSV saturation below this counts as a greyscale print


def _crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return img[y : y + h, x : x + w]


def portrait_landmarks(
    detector, template_bgr: np.ndarray, photo_box: tuple[int, int, int, int]
) -> np.ndarray:
    """Five landmarks of the template's own portrait, in document coordinates.

    The crop is padded first: unpadded, the face fills the frame and is larger than
    SCRFD's biggest anchors, so detection fails on more than half the templates.
    """
    x, y, w, h = photo_box
    p = max(w, h) // 2
    crop = cv2.copyMakeBorder(_crop(template_bgr, photo_box), p, p, p, p, cv2.BORDER_REPLICATE)
    det = detector.largest(crop)
    if det is None:
        raise ValueError("no face found in template portrait")
    return det.landmarks + np.float32([x - p, y - p])


def is_greyscale(region_bgr: np.ndarray) -> bool:
    return cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)[..., 1].mean() < GREYSCALE_SATURATION


def grabcut_matte(image: np.ndarray, face_box: np.ndarray, iters: int = 5) -> np.ndarray:
    """Soft person matte in [0, 1] from GrabCut seeded by the detected face box.

    Face core is certain foreground, an enlarged head-and-shoulders region probable
    foreground, the image border certain background.
    """
    h, w = image.shape[:2]
    scale = min(1.0, 480 / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    x0, y0, x1, y1 = np.asarray(face_box, float) * scale
    fw, fh = x1 - x0, y1 - y0
    sh, sw = small.shape[:2]

    mask = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    hx0, hx1 = int(max(0, x0 - 0.6 * fw)), int(min(sw, x1 + 0.6 * fw))
    mask[int(max(0, y0 - 0.7 * fh)) :, hx0:hx1] = cv2.GC_PR_FGD
    mask[int(y0 + 0.15 * fh) : int(y1 - 0.1 * fh), int(x0 + 0.2 * fw) : int(x1 - 0.2 * fw)] = (
        cv2.GC_FGD
    )
    b = max(2, sw // 50)
    mask[:b] = mask[:, :b] = mask[:, -b:] = cv2.GC_BGD

    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(small, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.float32)
    fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(fg, (0, 0), sigmaX=max(1.0, 0.01 * max(h, w)))


def _match_luminance(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_l = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    lum = lab[..., 0]
    lab[..., 0] = (lum - lum.mean()) / (lum.std() + 1e-6) * ref_l.std() + ref_l.mean()
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _border_colour(region: np.ndarray) -> np.ndarray:
    h, w = region.shape[:2]
    b = max(2, min(h, w) // 20)
    ring = np.concatenate(
        [region[:b].reshape(-1, 3), region[-b:].reshape(-1, 3), region[:, :b].reshape(-1, 3)]
    )
    return np.median(ring, axis=0)


def render(
    template_bgr: np.ndarray,
    photo_box: tuple[int, int, int, int],
    target_landmarks: np.ndarray,
    face_bgr: np.ndarray,
    face_landmarks: np.ndarray,
    matte: np.ndarray,
) -> np.ndarray:
    """Return a copy of the template with the portrait in `photo_box` replaced.

    `target_landmarks` are the template portrait's five landmarks in document coordinates;
    `face_landmarks` and `matte` belong to `face_bgr`.
    """
    x, y, w, h = photo_box
    original = _crop(template_bgr, photo_box)

    m, _ = cv2.estimateAffinePartial2D(
        np.asarray(face_landmarks, np.float32),
        np.asarray(target_landmarks, np.float32) - np.float32([x, y]),
        method=cv2.LMEDS,
    )
    if m is None:
        raise ValueError("could not align face to template landmarks")
    face = cv2.warpAffine(
        face_bgr, m, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE
    )
    alpha = cv2.warpAffine(matte.astype(np.float32), m, (w, h), borderValue=0.0)[..., None]

    background = np.broadcast_to(_border_colour(original), face.shape)
    portrait = (alpha * face + (1 - alpha) * background).astype(np.uint8)
    if is_greyscale(original):
        portrait = cv2.cvtColor(cv2.cvtColor(portrait, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    portrait = _match_luminance(portrait, original)
    portrait = cv2.GaussianBlur(portrait, (0, 0), sigmaX=max(0.5, w / 600))

    # Feather outwards into the document, never inwards: blending inside the box would let
    # the old portrait (its hair, its background) bleed through at the edges.
    edge = max(2, w // 60)
    H, W = template_bgr.shape[:2]
    x0, y0, x1, y1 = max(0, x - edge), max(0, y - edge), min(W, x + w + edge), min(H, y + h + edge)
    patch = cv2.copyMakeBorder(
        portrait, y - y0, y1 - y - h, x - x0, x1 - x - w, cv2.BORDER_REPLICATE
    ).astype(np.float32)
    feather = np.zeros((y1 - y0, x1 - x0), np.float32)
    feather[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = 1
    feather = np.maximum(feather, cv2.GaussianBlur(feather, (0, 0), sigmaX=edge / 2))[..., None]

    out = template_bgr.copy()
    region = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (feather * patch + (1 - feather) * region).astype(np.uint8)
    return out
