#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_sam_benchmark_from_annotations import build_video_predictor, run_video  # noqa: E402


def zip_submission_direct(submission_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(submission_dir.rglob("*.png")):
            archive.write(path, path.relative_to(submission_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAM2 on MOSEv2-style data and build a Codabench submission zip."
    )
    parser.add_argument("--dataset-root", required=True, help="Directory containing JPEGImages/ and Annotations/.")
    parser.add_argument("--output-root", default="output/mosev2", help="Output root.")
    parser.add_argument("--checkpoint", default="/content/sam2-checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--model-cfg", default="/content/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N videos. 0 means all videos.")
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
    submission_root = output_root / "submission"
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
            tempfile.mkdtemp(prefix="mosev2_frames_")
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
        zip_submission_direct(submission_root, output_root / "submission.zip")
        print(f"Submission zip written to: {output_root / 'submission.zip'}")
    print(f"Submission directory written to: {submission_root}")
    if overlay_root is not None:
        print(f"Overlay directory written to: {overlay_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
