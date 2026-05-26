from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), rgb)


def load_video_frames(video_path: Path) -> list[np.ndarray]:
    if video_path.is_dir():
        frame_paths = sorted(
            [
                path
                for path in video_path.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ],
            key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem),
        )
        if not frame_paths:
            raise FileNotFoundError(f"no readable frames found in directory: {video_path}")
        return [np.asarray(Image.open(path).convert("RGB")) for path in frame_paths]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def save_video(path: Path, frames: list[np.ndarray], fps: float = 12.0) -> None:
    if not frames:
        raise ValueError("cannot save empty video")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
