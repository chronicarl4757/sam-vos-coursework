from __future__ import annotations

from dataclasses import dataclass
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .media import PromptSpec, binary_mask_from_masks, to_point_arrays


@dataclass(frozen=True)
class Sam2Config:
    model_cfg: str
    checkpoint: str
    device: str = "cuda"
    use_hf: bool = False
    hf_model_id: str = "facebook/sam2-hiera-large"


def _normalize_model_cfg(config_path: str) -> str:
    path = Path(config_path)
    if not path.is_absolute():
        return path.as_posix()

    try:
        import sam2
    except Exception:
        return path.as_posix()

    package_root = Path(sam2.__file__).resolve().parent
    try:
        return path.relative_to(package_root).as_posix()
    except ValueError:
        if "configs" in path.parts:
            return Path(*path.parts[path.parts.index("configs") :]).as_posix()
        return path.as_posix()


def _import_image_predictor():
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "SAM2 is not installed. Install the official sam2 package first."
        ) from exc
    return build_sam2, SAM2ImagePredictor


def _import_video_predictor():
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "SAM2 is not installed. Install the official sam2 package first."
        ) from exc
    return build_sam2_video_predictor


def load_image_predictor(cfg: Sam2Config):
    build_sam2, predictor_cls = _import_image_predictor()
    if cfg.use_hf:
        return predictor_cls.from_pretrained(cfg.hf_model_id)
    model_cfg = _normalize_model_cfg(cfg.model_cfg)
    try:
        model = build_sam2(model_cfg, cfg.checkpoint, device=cfg.device)
    except TypeError:
        model = build_sam2(model_cfg, cfg.checkpoint)
    return predictor_cls(model)


def load_video_predictor(cfg: Sam2Config):
    build_sam2_video_predictor = _import_video_predictor()
    if cfg.use_hf:
        try:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "SAM2 is not installed. Install the official sam2 package first."
            ) from exc
        return SAM2VideoPredictor.from_pretrained(cfg.hf_model_id)
    model_cfg = _normalize_model_cfg(cfg.model_cfg)
    try:
        return build_sam2_video_predictor(model_cfg, cfg.checkpoint, device=cfg.device)
    except TypeError:
        return build_sam2_video_predictor(model_cfg, cfg.checkpoint)


def predict_image_mask(predictor, image: np.ndarray, prompt: PromptSpec, *, multimask_output: bool = True) -> np.ndarray:
    predictor.set_image(image)
    point_coords, point_labels = to_point_arrays(prompt)
    kwargs: dict = {}
    if point_coords is not None:
        kwargs["point_coords"] = point_coords
        kwargs["point_labels"] = point_labels
    if prompt.box is not None:
        kwargs["box"] = np.asarray(prompt.box, dtype=np.float32)

    masks, scores, _ = predictor.predict(
        point_coords=kwargs.get("point_coords"),
        point_labels=kwargs.get("point_labels"),
        box=kwargs.get("box"),
        multimask_output=multimask_output,
    )
    return binary_mask_from_masks(masks, scores)


def init_video_state(
    predictor,
    video_path: Path,
    *,
    offload_video_to_cpu: bool = False,
    offload_state_to_cpu: bool = False,
    async_loading_frames: bool = False,
):
    source_path = video_path
    if video_path.is_file():
        suffix = video_path.suffix.lower()
        if suffix in {".mp4", ".mov", ".mkv", ".avi"}:
            import cv2

            frame_dir = Path(tempfile.mkdtemp(prefix=f"{video_path.stem}_frames_"))
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise FileNotFoundError(f"cannot open video: {video_path}")
            frame_idx = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_path = frame_dir / f"{frame_idx:05d}.jpg"
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(frame_path)
                frame_idx += 1
            capture.release()
            source_path = frame_dir
    return predictor.init_state(
        str(source_path),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
        async_loading_frames=async_loading_frames,
    )


def add_prompt_to_video(predictor, state, prompt: PromptSpec, frame_idx: int, object_id: int = 0):
    point_coords, point_labels = to_point_arrays(prompt)
    kwargs: dict = {
        "inference_state": state,
        "frame_idx": frame_idx,
        "obj_id": object_id,
    }
    if point_coords is not None:
        kwargs["points"] = point_coords
        kwargs["labels"] = point_labels
    if prompt.box is not None:
        kwargs["box"] = np.asarray(prompt.box, dtype=np.float32)
    return predictor.add_new_points_or_box(**kwargs)
