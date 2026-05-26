#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/colab_bootstrap.sh [--repo-dir DIR] [--sam2-dir DIR] [--checkpoints-dir DIR] [--run-smoke]

Options:
  --repo-dir DIR          Path to this coursework repo. Default: current directory
  --sam2-dir DIR          Path to the official SAM2 repo. Default: /content/sam2
  --checkpoints-dir DIR   Path to SAM2 checkpoints. Default: /content/sam2-checkpoints
  --run-smoke             Run a small GPU smoke test after setup.
  -h, --help              Show this help message

Example in Colab:
  !git clone https://github.com/chronicarl4757/sam-vos-coursework.git
  %cd sam-vos-coursework
  !bash scripts/colab_bootstrap.sh --run-smoke
EOF
}

repo_dir="$(pwd)"
sam2_dir="/content/sam2"
checkpoints_dir="/content/sam2-checkpoints"
run_smoke=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      repo_dir="${2:?missing value for --repo-dir}"
      shift 2
      ;;
    --sam2-dir)
      sam2_dir="${2:?missing value for --sam2-dir}"
      shift 2
      ;;
    --checkpoints-dir)
      checkpoints_dir="${2:?missing value for --checkpoints-dir}"
      shift 2
      ;;
    --run-smoke)
      run_smoke=1
      shift
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

if [[ ! -d "$repo_dir" ]]; then
  echo "Repo directory not found: $repo_dir" >&2
  exit 1
fi

cd "$repo_dir"

export DEBIAN_FRONTEND=noninteractive

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git ffmpeg curl python3-pip python3-venv
fi

python -m pip install --upgrade pip setuptools wheel

python -m pip install -e .

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

HTTP_PROXY="${HTTP_PROXY:-}" \
HTTPS_PROXY="${HTTPS_PROXY:-}" \
ALL_PROXY="${ALL_PROXY:-}" \
bash scripts/install_sam2_official.sh --dir "$sam2_dir"

HTTP_PROXY="${HTTP_PROXY:-}" \
HTTPS_PROXY="${HTTPS_PROXY:-}" \
ALL_PROXY="${ALL_PROXY:-}" \
bash scripts/download_sam2_checkpoints.sh --dir "$checkpoints_dir"

mkdir -p output report

if [[ "$run_smoke" -eq 1 ]]; then
  SAM2_REPO_DIR="$sam2_dir" \
  SAM2_CHECKPOINT_DIR="$checkpoints_dir" \
  SAM2_DEVICE="${SAM2_DEVICE:-cuda}" \
  bash scripts/smoke_sam2.sh
fi

cat <<EOF
Colab bootstrap finished.

Repo: $repo_dir
SAM2: $sam2_dir
Checkpoints: $checkpoints_dir

Next:
  source .venv/bin/activate
  SAM2_DEVICE=cuda sam-vos image ...
  SAM2_DEVICE=cuda sam-vos video ...
EOF
