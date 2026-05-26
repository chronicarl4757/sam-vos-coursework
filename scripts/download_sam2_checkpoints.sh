#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_sam2_checkpoints.sh [--dir DIR] [--model NAME ...] [--source SOURCE]

Options:
  --dir DIR         Output directory for checkpoints. Default: checkpoints
  --model NAME      Download only the named checkpoint. Can be repeated.
                    Supported names:
                      sam2.1_hiera_tiny
                      sam2.1_hiera_small
                      sam2.1_hiera_base_plus
                      sam2.1_hiera_large
  --source SOURCE   Download source: auto, official, hf. Default: auto
  -h, --help        Show this help message

Examples:
  bash scripts/download_sam2_checkpoints.sh
  bash scripts/download_sam2_checkpoints.sh --dir assets/checkpoints
  bash scripts/download_sam2_checkpoints.sh --model sam2.1_hiera_large
  HTTP_PROXY=http://127.0.0.1:7890 \
  HTTPS_PROXY=http://127.0.0.1:7890 \
  ALL_PROXY=socks5://127.0.0.1:7890 \
  bash scripts/download_sam2_checkpoints.sh --source auto
EOF
}

out_dir="checkpoints"
source_mode="auto"
models=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      out_dir="${2:?missing value for --dir}"
      shift 2
      ;;
    --model)
      models+=("${2:?missing value for --model}")
      shift 2
      ;;
    --source)
      source_mode="${2:?missing value for --source}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$out_dir"

if command -v wget >/dev/null 2>&1; then
  downloader() {
    wget -nc -O "$1" "$2"
  }
elif command -v curl >/dev/null 2>&1; then
  downloader() {
    curl -L --fail --retry 3 --retry-delay 2 -o "$1" "$2"
  }
else
  echo "Please install wget or curl to download the checkpoints." >&2
  exit 1
fi

base_url="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

declare -A urls=(
  ["sam2.1_hiera_tiny"]="${base_url}/sam2.1_hiera_tiny.pt"
  ["sam2.1_hiera_small"]="${base_url}/sam2.1_hiera_small.pt"
  ["sam2.1_hiera_base_plus"]="${base_url}/sam2.1_hiera_base_plus.pt"
  ["sam2.1_hiera_large"]="${base_url}/sam2.1_hiera_large.pt"
)

declare -A hf_repos=(
  ["sam2.1_hiera_tiny"]="facebook/sam2-hiera-tiny"
  ["sam2.1_hiera_small"]="facebook/sam2-hiera-small"
  ["sam2.1_hiera_base_plus"]="facebook/sam2-hiera-base-plus"
  ["sam2.1_hiera_large"]="facebook/sam2-hiera-large"
)

if [[ ${#models[@]} -eq 0 ]]; then
  models=(
    "sam2.1_hiera_tiny"
    "sam2.1_hiera_small"
    "sam2.1_hiera_base_plus"
    "sam2.1_hiera_large"
  )
fi

for model in "${models[@]}"; do
  if [[ -z "${urls[$model]:-}" ]]; then
    echo "Unknown model name: $model" >&2
    exit 1
  fi
  dst="$out_dir/${model}.pt"
  if [[ "$source_mode" == "hf" ]]; then
    echo "Downloading ${model}.pt from Hugging Face -> ${dst}"
    python - "$model" "$dst" "${hf_repos[$model]}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except Exception as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "huggingface_hub is required for --source hf. Install it with: uv pip install huggingface_hub"
    ) from exc

model_name = sys.argv[1]
dst = Path(sys.argv[2])
repo_id = sys.argv[3]

local_dir = dst.with_suffix("")
snapshot_download(
    repo_id=repo_id,
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
)

candidate_names = [
    f"{model_name}.pt",
    "model.pt",
    "checkpoint.pt",
]

for name in candidate_names:
    candidate = local_dir / name
    if candidate.exists():
        dst.write_bytes(candidate.read_bytes())
        break
else:
    matches = sorted(local_dir.rglob("*.pt"))
    if not matches:
        raise SystemExit(f"No .pt file found in {local_dir}")
    dst.write_bytes(matches[0].read_bytes())
PY
  else
    echo "Downloading ${model}.pt -> ${dst}"
    if ! downloader "$dst" "${urls[$model]}"; then
      if [[ "$source_mode" == "official" ]]; then
        echo "Official download failed for ${model}." >&2
        exit 1
      fi
      echo "Official download failed for ${model}, falling back to Hugging Face."
      python - "$model" "$dst" "${hf_repos[$model]}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except Exception as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "huggingface_hub is required for automatic fallback. Install it with: uv pip install huggingface_hub"
    ) from exc

model_name = sys.argv[1]
dst = Path(sys.argv[2])
repo_id = sys.argv[3]

local_dir = dst.with_suffix("")
snapshot_download(
    repo_id=repo_id,
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
)

candidate_names = [
    f"{model_name}.pt",
    "model.pt",
    "checkpoint.pt",
]

for name in candidate_names:
    candidate = local_dir / name
    if candidate.exists():
        dst.write_bytes(candidate.read_bytes())
        break
else:
    matches = sorted(local_dir.rglob("*.pt"))
    if not matches:
        raise SystemExit(f"No .pt file found in {local_dir}")
    dst.write_bytes(matches[0].read_bytes())
PY
    fi
  fi
done

echo "All requested checkpoints are downloaded successfully."
