from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .media import build_prompt, ensure_dir, load_rgb_image, overlay_mask
from .sam2_backend import (
    Sam2Config,
    add_prompt_to_video,
    init_video_state,
    load_image_predictor,
    load_video_predictor,
    predict_image_mask,
)
from .visualize import load_video_frames, save_image, save_video


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_image(args: argparse.Namespace) -> int:
    output_dir = ensure_dir(Path(args.output_dir))
    prompt = build_prompt(args.points, args.labels, args.box)
    _write_json(output_dir / "prompt.json", prompt.as_json())

    cfg = Sam2Config(
        model_cfg=args.model_cfg,
        checkpoint=args.checkpoint,
        device=args.device,
        use_hf=args.use_hf,
        hf_model_id=args.hf_model_id,
    )
    predictor = load_image_predictor(cfg)
    image = load_rgb_image(Path(args.image))
    mask = predict_image_mask(predictor, image, prompt, multimask_output=not args.single_mask)

    overlay = overlay_mask(image, mask)
    save_image(output_dir / "overlay.png", overlay)
    save_image(output_dir / "mask.png", np.stack([mask * 255] * 3, axis=-1).astype(np.uint8))
    np.save(output_dir / "mask.npy", mask.astype(np.uint8))
    return 0


def run_video(args: argparse.Namespace) -> int:
    output_dir = ensure_dir(Path(args.output_dir))
    frames = load_video_frames(Path(args.video))
    if not frames:
        raise ValueError("video contains no readable frames")
    prompt = build_prompt(args.points, args.labels, args.box)
    _write_json(output_dir / "prompt.json", prompt.as_json())

    cfg = Sam2Config(
        model_cfg=args.model_cfg,
        checkpoint=args.checkpoint,
        device=args.device,
        use_hf=args.use_hf,
        hf_model_id=args.hf_model_id,
    )
    predictor = load_video_predictor(cfg)
    state = init_video_state(
        predictor,
        Path(args.video),
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
        async_loading_frames=args.async_loading_frames,
    )

    add_prompt_to_video(predictor, state, prompt, frame_idx=args.frame_index, object_id=args.object_id)

    frame_out_dir = ensure_dir(output_dir / "frames")
    mask_out_dir = ensure_dir(output_dir / "masks")
    overlay_frames: list[np.ndarray] = []
    frame_map = {idx: frame for idx, frame in enumerate(frames)}

    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        if frame_idx % max(1, args.frame_step) != 0:
            continue
        frame = frame_map.get(frame_idx)
        if frame is None:
            continue
        union_mask = None
        for mask_tensor in masks:
            mask = mask_tensor[0].detach().cpu().numpy() > 0
            union_mask = mask if union_mask is None else (union_mask | mask)
        if union_mask is None:
            continue
        overlay = overlay_mask(frame, union_mask)
        overlay_frames.append(overlay)
        save_image(frame_out_dir / f"frame_{frame_idx:05d}.png", frame)
        save_image(mask_out_dir / f"mask_{frame_idx:05d}.png", np.stack([union_mask * 255] * 3, axis=-1).astype(np.uint8))

    if overlay_frames:
        save_video(output_dir / "overlay.mp4", overlay_frames, fps=args.fps / max(1, args.frame_step))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sam-vos", description="SAM2 coursework utilities")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-hf", action="store_true", help="load SAM2 from Hugging Face instead of local checkpoint")
    parser.add_argument("--hf-model-id", default="facebook/sam2-hiera-large")

    subparsers = parser.add_subparsers(dest="command", required=True)

    image = subparsers.add_parser("image", help="run image segmentation")
    image.add_argument("--image", required=True)
    image.add_argument("--checkpoint", required=True)
    image.add_argument("--model-cfg", required=True)
    image.add_argument("--points", default=None, help="semicolon-separated x,y pairs, e.g. 120,80;160,140")
    image.add_argument("--labels", default=None, help="comma-separated labels, e.g. 1,1")
    image.add_argument("--box", default=None, help="x0,y0,x1,y1")
    image.add_argument("--output-dir", required=True)
    image.add_argument("--single-mask", action="store_true", help="use the top-ranked mask only")
    image.set_defaults(func=run_image)

    video = subparsers.add_parser("video", help="run video segmentation")
    video.add_argument("--video", required=True)
    video.add_argument("--checkpoint", required=True)
    video.add_argument("--model-cfg", required=True)
    video.add_argument("--points", default=None, help="semicolon-separated x,y pairs, e.g. 120,80;160,140")
    video.add_argument("--labels", default=None, help="comma-separated labels, e.g. 1,1")
    video.add_argument("--box", default=None, help="x0,y0,x1,y1")
    video.add_argument("--frame-index", type=int, default=0)
    video.add_argument("--object-id", type=int, default=0)
    video.add_argument("--frame-step", type=int, default=1, help="frame sampling step when exporting visualizations")
    video.add_argument("--fps", type=float, default=12.0)
    video.add_argument("--offload-video-to-cpu", action="store_true", help="keep video frames on CPU to reduce VRAM use")
    video.add_argument("--offload-state-to-cpu", action="store_true", help="keep SAM2 tracking state on CPU to reduce VRAM use")
    video.add_argument("--async-loading-frames", action="store_true", help="load JPEG frames asynchronously")
    video.add_argument("--output-dir", required=True)
    video.set_defaults(func=run_video)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
