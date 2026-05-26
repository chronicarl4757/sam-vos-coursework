from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PromptSpec:
    points: list[tuple[float, float]]
    labels: list[int]
    box: tuple[float, float, float, float] | None = None

    def as_json(self) -> dict:
        return {
            "points": [list(p) for p in self.points],
            "labels": list(self.labels),
            "box": list(self.box) if self.box is not None else None,
        }


def load_rgb_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_csv_floats(text: str, expected_len: int | None = None) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"expected {expected_len} comma-separated values, got {len(values)}: {text!r}")
    return values


def parse_points(text: str | None) -> list[tuple[float, float]]:
    if not text:
        return []
    points: list[tuple[float, float]] = []
    for item in text.split(";"):
        x, y = parse_csv_floats(item, expected_len=2)
        points.append((x, y))
    return points


def parse_labels(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_box(text: str | None) -> tuple[float, float, float, float] | None:
    if not text:
        return None
    x0, y0, x1, y1 = parse_csv_floats(text, expected_len=4)
    return x0, y0, x1, y1


def build_prompt(points: str | None, labels: str | None, box: str | None) -> PromptSpec:
    parsed_points = parse_points(points)
    parsed_labels = parse_labels(labels)
    parsed_box = parse_box(box)
    if parsed_points and parsed_labels and len(parsed_points) != len(parsed_labels):
        raise ValueError("points and labels must have the same length")
    if parsed_points and not parsed_labels:
        parsed_labels = [1] * len(parsed_points)
    if parsed_labels and not parsed_points and parsed_box is None:
        raise ValueError("labels were provided without points or box")
    return PromptSpec(points=parsed_points, labels=parsed_labels, box=parsed_box)


def to_point_arrays(prompt: PromptSpec) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not prompt.points:
        return None, None
    point_coords = np.asarray(prompt.points, dtype=np.float32)
    point_labels = np.asarray(prompt.labels, dtype=np.int32)
    return point_coords, point_labels


def binary_mask_from_masks(masks: np.ndarray, scores: Sequence[float] | np.ndarray | None = None) -> np.ndarray:
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"expected masks with shape [N, H, W] or [N, 1, H, W], got {masks.shape}")
    if scores is None:
        return masks[0] > 0
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    scores_arr = np.asarray(scores)
    best_index = int(scores_arr.argmax())
    return masks[best_index] > 0


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 255, 0), alpha: float = 0.5) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be RGB with shape [H, W, 3]")
    mask_bool = mask.astype(bool)
    overlay = image.copy().astype(np.float32)
    color_arr = np.asarray(color, dtype=np.float32)
    overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + color_arr * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)
