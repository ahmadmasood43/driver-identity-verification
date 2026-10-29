"""Frozen ArcFace embedder (insightface ONNX export). Embeddings are never persisted."""

from pathlib import Path

import cv2
import numpy as np

from idv.face.onnx import session


class ArcFace:
    def __init__(self, model: Path, device: str = "cuda"):
        self.sess = session(model, device)
        self.input_name = self.sess.get_inputs()[0].name

    def embed(self, aligned_bgr: list[np.ndarray]) -> np.ndarray:
        """L2-normalised embeddings for 112x112 aligned crops, one row per crop."""
        blob = cv2.dnn.blobFromImages(aligned_bgr, 1 / 127.5, (112, 112), 127.5, swapRB=True)
        e = self.sess.run(None, {self.input_name: blob})[0]
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of already-normalised embeddings."""
    return np.sum(a * b, axis=-1)
