#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_jpeg_dataset_colab.sh --video-root PATH [--output-root DIR] [--box x0,y0,x1,y1]

Expected input layout:
  /video_dataset/JPEGImages/video_id/frame_id.jpg
  /video_dataset/JPEGImages/video_id/frame_id.png

Options:
  --video-root PATH    Root directory that contains per-video frame folders. Required.
  --output-root DIR    Output directory. Default: output/colab_jpeg_dataset
  --checkpoint PATH   SAM2 checkpoint. Default: /content/sam2-checkpoints/sam2.1_hiera_tiny.pt
  --model-cfg PATH    SAM2 config. Default: /content/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml
  --device DEVICE     Device to use. Default: cuda
  --box ...           Bounding box prompt. Required.
  --frame-index N     Prompt frame index. Default: 0
  --frame-step N      Export every N-th frame. Default: 1
  --fps N             Output FPS. Default: 12
  --offload-video-to-cpu   Keep frames on CPU to save VRAM.
  --offload-state-to-cpu   Keep state on CPU to save VRAM.
  --async-loading-frames   Use async JPEG loading inside SAM2.
  --copy-to-local     Copy each video folder to local /content before running.
  --local-root DIR    Local cache directory used with --copy-to-local.
  --make-preview      Re-encode overlay.mp4 files to h264 preview videos.
  -h, --help          Show this help message
EOF
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sam_vos_bin="${SAM_VOS_BIN:-$repo_dir/.venv/bin/sam-vos}"
preview_script="${PREVIEW_SCRIPT:-$repo_dir/scripts/reencode_preview_videos.sh}"

video_root=""
output_root="$repo_dir/output/colab_jpeg_dataset"
checkpoint="${SAM2_CHECKPOINT:-/content/sam2-checkpoints/sam2.1_hiera_tiny.pt}"
model_cfg="${SAM2_MODEL_CFG:-/content/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml}"
device="${SAM2_DEVICE:-cuda}"
points=""
labels=""
box=""
frame_index=0
frame_step=1
fps=12
offload_video_to_cpu=1
offload_state_to_cpu=1
async_loading_frames=1
copy_to_local=0
local_root="/content/sam_jpeg_cache"
make_preview=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video-root) video_root="${2:?missing value for --video-root}"; shift 2 ;;
    --output-root) output_root="${2:?missing value for --output-root}"; shift 2 ;;
    --checkpoint) checkpoint="${2:?missing value for --checkpoint}"; shift 2 ;;
    --model-cfg) model_cfg="${2:?missing value for --model-cfg}"; shift 2 ;;
    --device) device="${2:?missing value for --device}"; shift 2 ;;
    --points) points="${2:?missing value for --points}"; shift 2 ;;
    --labels) labels="${2:?missing value for --labels}"; shift 2 ;;
    --box) box="${2:?missing value for --box}"; shift 2 ;;
    --frame-index) frame_index="${2:?missing value for --frame-index}"; shift 2 ;;
    --frame-step) frame_step="${2:?missing value for --frame-step}"; shift 2 ;;
    --fps) fps="${2:?missing value for --fps}"; shift 2 ;;
    --offload-video-to-cpu) offload_video_to_cpu=1; shift ;;
    --offload-state-to-cpu) offload_state_to_cpu=1; shift ;;
    --async-loading-frames) async_loading_frames=1; shift ;;
    --copy-to-local) copy_to_local=1; shift ;;
    --local-root) local_root="${2:?missing value for --local-root}"; shift 2 ;;
    --make-preview) make_preview=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$video_root" ]]; then
  echo "--video-root is required" >&2
  usage >&2
  exit 1
fi

if [[ -z "$box" && -z "$points" ]]; then
  echo "You must provide --box or --points/--labels." >&2
  usage >&2
  exit 1
fi

if [[ ! -x "$sam_vos_bin" ]]; then
  echo "sam-vos binary not found: $sam_vos_bin" >&2
  exit 1
fi

video_root="$(realpath "$video_root")"
mkdir -p "$output_root"

shopt -s nullglob
video_dirs=("$video_root"/*)
shopt -u nullglob

if [[ ${#video_dirs[@]} -eq 0 ]]; then
  echo "No video directories found under $video_root" >&2
  exit 1
fi

for video_dir in "${video_dirs[@]}"; do
  [[ -d "$video_dir" ]] || continue
  video_id="$(basename "$video_dir")"
  out_dir="$output_root/$video_id"
  mkdir -p "$out_dir"

  run_input="$video_dir"
  if [[ "$copy_to_local" -eq 1 ]]; then
    mkdir -p "$local_root"
    local_video_dir="$local_root/$video_id"
    rm -rf "$local_video_dir"
    cp -a "$video_dir" "$local_video_dir"
    run_input="$local_video_dir"
  fi

  cmd=(
    "$sam_vos_bin" --device "$device" video
    --video "$run_input"
    --checkpoint "$checkpoint"
    --model-cfg "$model_cfg"
    --output-dir "$out_dir"
    --frame-index "$frame_index"
    --frame-step "$frame_step"
    --fps "$fps"
  )

  if [[ -n "$points" ]]; then
    cmd+=(--points "$points")
  fi
  if [[ -n "$labels" ]]; then
    cmd+=(--labels "$labels")
  fi
  if [[ -n "$box" ]]; then
    cmd+=(--box "$box")
  fi
  if [[ "$offload_video_to_cpu" -eq 1 ]]; then
    cmd+=(--offload-video-to-cpu)
  fi
  if [[ "$offload_state_to_cpu" -eq 1 ]]; then
    cmd+=(--offload-state-to-cpu)
  fi
  if [[ "$async_loading_frames" -eq 1 ]]; then
    cmd+=(--async-loading-frames)
  fi

  echo "Running $video_id"
  "${cmd[@]}"

done

if [[ "$make_preview" -eq 1 ]]; then
  if [[ ! -x "$preview_script" ]]; then
    echo "preview script not found: $preview_script" >&2
    exit 1
  fi
  bash "$preview_script" --input "$output_root" --output-dir "$output_root/preview"
fi

echo "Dataset results written to: $output_root"
