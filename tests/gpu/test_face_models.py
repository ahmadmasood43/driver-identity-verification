"""Real SCRFD + ArcFace weights on CUDA. Expects models/ populated per models/manifest.csv."""

from pathlib import Path

import numpy as np
import pytest

from idv.data.manifest import verify

pytestmark = pytest.mark.gpu

MODELS = Path(__file__).parents[2] / "models"


@pytest.fixture(scope="module")
def models():
    if not (MODELS / "manifest.csv").exists() or verify(MODELS, MODELS / "manifest.csv"):
        pytest.skip("model weights missing or not matching models/manifest.csv")
    from idv.face.arcface import ArcFace
    from idv.face.scrfd import SCRFD

    return SCRFD(MODELS / "det_10g.onnx"), ArcFace(MODELS / "w600k_r50.onnx")


def test_detector_runs_on_cuda_and_finds_nothing_in_noise(models):
    det, _ = models
    noise = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    assert det.detect(noise) == []


def test_embedder_unit_norm_and_self_similarity(models):
    _, arc = models
    crops = list(np.random.default_rng(0).integers(0, 255, (3, 112, 112, 3), dtype=np.uint8))
    e = arc.embed(crops)
    assert e.shape == (3, 512)
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(arc.embed(crops[:1])[0] @ e[0], 1, atol=1e-5)
