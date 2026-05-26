#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_video_colab.sh --video PATH [--output-dir DIR] [--box x0,y0,x1,y1] [--frame-index N]

Options:
  --video PATH        Input video path or extracted frame directory. Required.
  --output-dir DIR    Output directory. Default: output/colab_video
  --checkpoint PATH   SAM2 checkpoint. Default: /content/sam2-checkpoints/sam2.1_hiera_tiny.pt
  --model-cfg PATH    SAM2 config. Default: /content/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml
  --device DEVICE     Device to use. Default: cuda
  --box ...           Bounding box prompt.
  --frame-index N     Prompt frame index. Default: 0
  --frame-step N      Export every N-th frame. Default: 1
  --fps N             Output FPS. Default: 12
  --offload-video-to-cpu   Keep frames on CPU to save VRAM.
  --offload-state-to-cpu   Keep state on CPU to save VRAM.
  -h, --help          Show this help message
EOF
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sam_vos_bin="${SAM_VOS_BIN:-$repo_dir/.venv/bin/sam-vos}"

video=""
output_dir="$repo_dir/output/colab_video"
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) video="${2:?missing value for --video}"; shift 2 ;;
    --output-dir) output_dir="${2:?missing value for --output-dir}"; shift 2 ;;
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
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$video" ]]; then
  echo "--video is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -x "$sam_vos_bin" ]]; then
  echo "sam-vos binary not found: $sam_vos_bin" >&2
  exit 1
fi

mkdir -p "$output_dir"

cmd=(
  "$sam_vos_bin" --device "$device" video
  --video "$video"
  --checkpoint "$checkpoint"
  --model-cfg "$model_cfg"
  --output-dir "$output_dir"
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

"${cmd[@]}"

echo "Video result written to: $output_dir"
