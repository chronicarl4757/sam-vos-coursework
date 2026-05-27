#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_frame_name(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("mask_"):
        stem = stem.removeprefix("mask_")
    return f"{int(stem):05d}.png"


def normalize_mask(image: Image.Image) -> Image.Image:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = (arr[..., :3] > 0).any(axis=2).astype(np.uint8)
    elif arr.max(initial=0) == 255 and set(np.unique(arr).tolist()) <= {0, 255}:
        arr = (arr > 0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8, copy=False)
    return Image.fromarray(arr, mode="L")


def fix_zip(input_zip: Path, output_zip: Path, root_name: str) -> tuple[int, int]:
    videos: set[str] = set()
    frames = 0
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_zip) as source, zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as target:
        for info in source.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/") or not info.filename.endswith(".png"):
                continue
            parts = Path(info.filename).parts
            if len(parts) < 3:
                continue
            video_id = parts[-2]
            frame_name = normalize_frame_name(parts[-1])
            with source.open(info) as file_obj:
                image = Image.open(file_obj).copy()
            output_image = normalize_mask(image)
            buffer = BytesIO()
            output_image.save(buffer, format="PNG")
            target.writestr(f"{root_name}/{video_id}/{frame_name}", buffer.getvalue())
            videos.add(video_id)
            frames += 1
    return len(videos), frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize a MOSEv2 submission zip for Codabench.")
    parser.add_argument("--input", required=True, help="Input zip generated from sam-vos masks.")
    parser.add_argument("--output", required=True, help="Fixed output zip.")
    parser.add_argument("--root-name", default="sample_submission", help="Top-level folder name inside the zip.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    videos, frames = fix_zip(Path(args.input).expanduser(), Path(args.output).expanduser(), args.root_name)
    print(f"wrote {args.output}")
    print(f"videos={videos} frames={frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
