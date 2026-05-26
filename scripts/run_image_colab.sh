#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_image_colab.sh --image PATH [--output-dir DIR] [--box x0,y0,x1,y1] [--points x,y;...] [--labels 1,1] [--single-mask]

Options:
  --image PATH        Input image path. Required.
  --output-dir DIR    Output directory. Default: output/colab_image
  --checkpoint PATH   SAM2 checkpoint. Default: /content/sam2-checkpoints/sam2.1_hiera_tiny.pt
  --model-cfg PATH    SAM2 config. Default: /content/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml
  --device DEVICE     Device to use. Default: cuda
  --box ...           Bounding box prompt.
  --points ...        Point prompt list, e.g. 120,80;160,140
  --labels ...        Label list, e.g. 1,1
  --single-mask       Use the top-ranked mask only.
  -h, --help          Show this help message
EOF
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sam_vos_bin="${SAM_VOS_BIN:-$repo_dir/.venv/bin/sam-vos}"

image=""
output_dir="$repo_dir/output/colab_image"
checkpoint="${SAM2_CHECKPOINT:-/content/sam2-checkpoints/sam2.1_hiera_tiny.pt}"
model_cfg="${SAM2_MODEL_CFG:-/content/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml}"
device="${SAM2_DEVICE:-cuda}"
points=""
labels=""
box=""
single_mask=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) image="${2:?missing value for --image}"; shift 2 ;;
    --output-dir) output_dir="${2:?missing value for --output-dir}"; shift 2 ;;
    --checkpoint) checkpoint="${2:?missing value for --checkpoint}"; shift 2 ;;
    --model-cfg) model_cfg="${2:?missing value for --model-cfg}"; shift 2 ;;
    --device) device="${2:?missing value for --device}"; shift 2 ;;
    --points) points="${2:?missing value for --points}"; shift 2 ;;
    --labels) labels="${2:?missing value for --labels}"; shift 2 ;;
    --box) box="${2:?missing value for --box}"; shift 2 ;;
    --single-mask) single_mask=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$image" ]]; then
  echo "--image is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -x "$sam_vos_bin" ]]; then
  echo "sam-vos binary not found: $sam_vos_bin" >&2
  exit 1
fi

mkdir -p "$output_dir"

cmd=(
  "$sam_vos_bin" --device "$device" image
  --image "$image"
  --checkpoint "$checkpoint"
  --model-cfg "$model_cfg"
  --output-dir "$output_dir"
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
if [[ "$single_mask" -eq 1 ]]; then
  cmd+=(--single-mask)
fi

"${cmd[@]}"

echo "Image result written to: $output_dir"
