"""SCRFD face detector (ONNX export from insightface), decoding done here."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from idv.face.onnx import session


@dataclass(frozen=True)
class Detection:
    box: np.ndarray  # x1, y1, x2, y2
    score: float
    landmarks: np.ndarray  # 5 x 2


def anchor_centers(height: int, width: int, stride: int, num_anchors: int) -> np.ndarray:
    ys, xs = np.mgrid[:height, :width]
    centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32) * stride
    return np.repeat(centers, num_anchors, axis=0)


def distance_to_box(centers: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.concatenate([centers - d[:, :2], centers + d[:, 2:4]], axis=1)


def distance_to_points(centers: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.repeat(centers[:, None, :], d.shape[1] // 2, axis=1) + d.reshape(len(d), -1, 2)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        w = np.maximum(0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        h = np.maximum(0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = w * h
        order = rest[inter / (areas[i] + areas[rest] - inter) <= iou]
    return keep


class SCRFD:
    strides = (8, 16, 32)
    num_anchors = 2

    def __init__(
        self,
        model: Path,
        device: str = "cuda",
        input_size: int = 640,
        score_threshold: float = 0.5,
        nms_iou: float = 0.4,
    ):
        self.sess = session(model, device)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = input_size
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        if len(self.sess.get_outputs()) != 9:
            raise ValueError("expected an SCRFD export with keypoints (9 outputs)")

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        scale = self.size / max(h, w)
        resized = cv2.resize(image, (round(w * scale), round(h * scale)))
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        return canvas, scale

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        canvas, scale = self._letterbox(image_bgr)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128, (self.size, self.size), 127.5, swapRB=True)
        outs = [o[0] if o.ndim == 3 else o for o in self.sess.run(None, {self.input_name: blob})]

        n = len(self.strides)
        boxes, scores, points = [], [], []
        for k, stride in enumerate(self.strides):
            s, b, p = outs[k].ravel(), outs[k + n] * stride, outs[k + 2 * n] * stride
            centers = anchor_centers(self.size // stride, self.size // stride, stride, 2)
            keep = s >= self.score_threshold
            scores.append(s[keep])
            boxes.append(distance_to_box(centers[keep], b[keep]))
            points.append(distance_to_points(centers[keep], p[keep]))

        scores_a = np.concatenate(scores)
        if scores_a.size == 0:
            return []
        boxes_a = np.concatenate(boxes) / scale
        points_a = np.concatenate(points) / scale
        return [
            Detection(boxes_a[i], float(scores_a[i]), points_a[i])
            for i in nms(boxes_a, scores_a, self.nms_iou)
        ]

    def largest(self, image_bgr: np.ndarray) -> Detection | None:
        dets = self.detect(image_bgr)
        if not dets:
            return None
        return max(dets, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
