#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_bin="$repo_root/.venv/bin"
sam_vos_bin="${SAM_VOS_BIN:-$venv_bin/sam-vos}"
python_bin="${PYTHON_BIN:-$venv_bin/python}"

checkpoint_dir="${SAM2_CHECKPOINT_DIR:-/media/chronicarl/Data/sam2-checkpoints}"
sam2_repo_dir="${SAM2_REPO_DIR:-/media/chronicarl/Data/sam2}"
checkpoint="${SAM2_CHECKPOINT:-$checkpoint_dir/sam2.1_hiera_tiny.pt}"
model_cfg="${SAM2_MODEL_CFG:-$sam2_repo_dir/sam2/configs/sam2.1/sam2.1_hiera_t.yaml}"
device="${SAM2_DEVICE:-cpu}"
output_root="${SAM2_OUTPUT_DIR:-$repo_root/output/sam2_smoke}"
work_dir="${SAM2_WORK_DIR:-$repo_root/tmp/sam2_smoke_inputs}"
frame_dir="$work_dir/video_frames"

if [[ ! -x "$sam_vos_bin" ]]; then
  echo "sam-vos binary not found at $sam_vos_bin" >&2
  echo "Activate the virtual environment or set SAM_VOS_BIN." >&2
  exit 1
fi

if [[ ! -f "$checkpoint" ]]; then
  echo "checkpoint not found: $checkpoint" >&2
  exit 1
fi

if [[ ! -f "$model_cfg" ]]; then
  echo "model config not found: $model_cfg" >&2
  exit 1
fi

mkdir -p "$work_dir" "$output_root" "$frame_dir"

"$python_bin" - "$work_dir" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

work_dir = Path(sys.argv[1])
work_dir.mkdir(parents=True, exist_ok=True)
frame_dir = work_dir / "video_frames"
frame_dir.mkdir(parents=True, exist_ok=True)

image = np.full((128, 128, 3), 240, dtype=np.uint8)
image[30:98, 28:96] = (30, 70, 220)
image[34:54, 78:98] = (255, 255, 255)
from PIL import Image

Image.fromarray(image).save(work_dir / "image.png")

for idx in range(8):
    frame = np.full((128, 128, 3), 245, dtype=np.uint8)
    left = 16 + idx * 5
    top = 36
    frame[top : top + 44, left : left + 56] = (40, 180, 80)
    frame[top + 10 : top + 26, left + 28 : left + 44] = (255, 255, 255)
    Image.fromarray(frame).save(frame_dir / f"{idx:05d}.jpg")
PY

image_out="$output_root/image"
video_out="$output_root/video"

"$sam_vos_bin" --device "$device" image \
  --image "$work_dir/image.png" \
  --checkpoint "$checkpoint" \
  --model-cfg "$model_cfg" \
  --box 24,24,104,104 \
  --single-mask \
  --output-dir "$image_out"

"$sam_vos_bin" --device "$device" video \
  --video "$frame_dir" \
  --checkpoint "$checkpoint" \
  --model-cfg "$model_cfg" \
  --frame-index 0 \
  --box 16,36,72,80 \
  --frame-step 1 \
  --fps 6 \
  --offload-video-to-cpu \
  --offload-state-to-cpu \
  --output-dir "$video_out"

echo "SAM2 smoke test finished."
echo "Image output: $image_out"
echo "Video output: $video_out"
