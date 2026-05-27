#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def normalize_model_cfg(config_path: str) -> str:
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


def frame_paths(video_dir: Path) -> list[Path]:
    paths = sorted(
        [
            *video_dir.glob("*.jpg"),
            *video_dir.glob("*.jpeg"),
            *video_dir.glob("*.png"),
        ],
        key=lambda p: int(p.stem),
    )
    if not paths:
        raise FileNotFoundError(f"no frames found under {video_dir}")
    return paths


def find_first_annotation(annotation_dir: Path, first_frame_stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = annotation_dir / f"{first_frame_stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no first-frame annotation found under {annotation_dir}")


def read_index_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.uint8, copy=False)


def object_ids_from_mask(mask: np.ndarray) -> list[int]:
    return sorted(int(value) for value in np.unique(mask) if int(value) != 0)


def compose_index_mask(
    object_ids: list[int],
    mask_logits,
    height: int,
    width: int,
) -> np.ndarray:
    output = np.zeros((height, width), dtype=np.uint8)
    for index, object_id in enumerate(object_ids):
        mask = mask_logits[index]
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask)
        mask = np.squeeze(mask) > 0
        output[mask] = object_id
    return output


def overlay_index_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colors = np.array(
        [
            [0, 0, 0],
            [255, 64, 64],
            [64, 160, 255],
            [80, 220, 120],
            [255, 200, 64],
            [190, 110, 255],
            [255, 120, 210],
            [90, 230, 230],
        ],
        dtype=np.float32,
    )
    output = image.astype(np.float32).copy()
    for object_id in object_ids_from_mask(mask):
        color = colors[object_id % len(colors)]
        region = mask == object_id
        output[region] = output[region] * (1.0 - alpha) + color * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def build_video_predictor(model_cfg: str, checkpoint: str, device: str):
    from sam2.build_sam import build_sam2_video_predictor

    model_cfg = normalize_model_cfg(model_cfg)
    try:
        return build_sam2_video_predictor(
            model_cfg,
            checkpoint,
            device=device,
            apply_postprocessing=False,
        )
    except TypeError:
        return build_sam2_video_predictor(model_cfg, checkpoint, device=device)


def maybe_local_copy(video_dir: Path, local_root: Path | None) -> tuple[Path, Path | None]:
    if local_root is None:
        return video_dir, None
    local_root.mkdir(parents=True, exist_ok=True)
    target = local_root / video_dir.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(video_dir, target)
    return target, target


def run_video(
    predictor,
    *,
    video_dir: Path,
    annotation_dir: Path,
    submission_dir: Path,
    overlay_dir: Path | None,
    overlay_stride: int,
    local_root: Path | None,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    async_loading_frames: bool,
) -> dict:
    frames = frame_paths(video_dir)
    first_annotation = find_first_annotation(annotation_dir, frames[0].stem)
    first_mask = read_index_mask(first_annotation)
    object_ids = object_ids_from_mask(first_mask)
    if not object_ids:
        raise ValueError(f"first-frame annotation has no foreground objects: {first_annotation}")

    run_video_dir, copied_dir = maybe_local_copy(video_dir, local_root)
    try:
        state = predictor.init_state(
            video_path=str(run_video_dir),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
        )

        for object_id in object_ids:
            object_mask = first_mask == object_id
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=object_id,
                mask=object_mask,
            )

        submission_dir.mkdir(parents=True, exist_ok=True)
        if overlay_dir is not None:
            overlay_dir.mkdir(parents=True, exist_ok=True)
        frame_name_by_idx = {idx: frame.stem for idx, frame in enumerate(frames)}
        frame_path_by_idx = {idx: frame for idx, frame in enumerate(frames)}
        height, width = first_mask.shape
        written = 0
        for frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            frame_stem = frame_name_by_idx.get(frame_idx)
            if frame_stem is None:
                continue
            ordered_logits = []
            for object_id in object_ids:
                if object_id not in out_obj_ids:
                    ordered_logits.append(np.zeros((height, width), dtype=bool))
                    continue
                index = list(out_obj_ids).index(object_id)
                ordered_logits.append(out_mask_logits[index])
            index_mask = compose_index_mask(object_ids, ordered_logits, height, width)
            Image.fromarray(index_mask).save(submission_dir / f"{frame_stem}.png")
            if overlay_dir is not None and frame_idx % max(1, overlay_stride) == 0:
                frame_image = np.asarray(Image.open(frame_path_by_idx[frame_idx]).convert("RGB"))
                overlay = overlay_index_mask(frame_image, index_mask)
                Image.fromarray(overlay).save(overlay_dir / f"{frame_stem}.png")
            written += 1
    finally:
        if copied_dir is not None and copied_dir.exists():
            shutil.rmtree(copied_dir)

    return {
        "video": video_dir.name,
        "frames": len(frames),
        "written": written,
        "object_ids": object_ids,
    }


def zip_submission(submission_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(submission_dir.rglob("*.png")):
            archive.write(path, path.relative_to(submission_dir.parent))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAM2 video benchmark inference from first-frame annotations."
    )
    parser.add_argument("--dataset-root", required=True, help="Directory containing JPEGImages/ and Annotations/.")
    parser.add_argument("--output-root", default="output/sam_benchmark", help="Output root.")
    parser.add_argument("--checkpoint", default="/content/sam2-checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--model-cfg", default="/content/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N videos. 0 means all videos.")
    parser.add_argument("--root-name", default="sample_submission")
    parser.add_argument("--save-overlays", action="store_true", help="Save visual overlays outside the submission folder.")
    parser.add_argument("--overlay-stride", type=int, default=25, help="Save every N-th overlay frame.")
    parser.add_argument("--copy-to-local", action="store_true")
    parser.add_argument("--local-root", default=None)
    parser.add_argument("--offload-video-to-cpu", action="store_true", default=True)
    parser.add_argument("--offload-state-to-cpu", action="store_true", default=True)
    parser.add_argument("--async-loading-frames", action="store_true", default=True)
    parser.add_argument("--no-zip", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    jpeg_root = dataset_root / "JPEGImages"
    annotation_root = dataset_root / "Annotations"
    output_root = Path(args.output_root).expanduser().resolve()
    submission_root = output_root / args.root_name
    overlay_root = output_root / "overlays" if args.save_overlays else None

    if not jpeg_root.is_dir():
        raise FileNotFoundError(f"missing JPEGImages directory: {jpeg_root}")
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"missing Annotations directory: {annotation_root}")

    videos = sorted(path for path in jpeg_root.iterdir() if path.is_dir())
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        raise FileNotFoundError(f"no video directories found under {jpeg_root}")

    if args.device == "cuda" and torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    predictor = build_video_predictor(args.model_cfg, args.checkpoint, args.device)
    local_root = None
    temp_dir = None
    if args.copy_to_local:
        local_root = Path(args.local_root).expanduser().resolve() if args.local_root else Path(
            tempfile.mkdtemp(prefix="sam_benchmark_frames_")
        )
        if args.local_root is None:
            temp_dir = local_root

    manifests = []
    try:
        for index, video_dir in enumerate(videos, start=1):
            print(f"[{index}/{len(videos)}] {video_dir.name}", flush=True)
            manifests.append(
                run_video(
                    predictor,
                    video_dir=video_dir,
                    annotation_dir=annotation_root / video_dir.name,
                    submission_dir=submission_root / video_dir.name,
                    overlay_dir=overlay_root / video_dir.name if overlay_root is not None else None,
                    overlay_stride=args.overlay_stride,
                    local_root=local_root,
                    offload_video_to_cpu=args.offload_video_to_cpu,
                    offload_state_to_cpu=args.offload_state_to_cpu,
                    async_loading_frames=args.async_loading_frames,
                )
            )
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps({"videos": manifests}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not args.no_zip:
        zip_submission(submission_root, output_root / "submission.zip")
        print(f"Submission zip written to: {output_root / 'submission.zip'}")
    print(f"Submission directory written to: {submission_root}")
    if overlay_root is not None:
        print(f"Overlay directory written to: {overlay_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
