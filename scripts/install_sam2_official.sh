#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/install_sam2_official.sh [--dir DIR] [--branch BRANCH]

Options:
  --dir DIR         Target directory for the official SAM2 repo. Default: third_party/sam2
  --branch BRANCH   Git branch to clone. Default: main
  -h, --help        Show this help message

Examples:
  HTTP_PROXY=http://127.0.0.1:7890 \
  HTTPS_PROXY=http://127.0.0.1:7890 \
  ALL_PROXY=socks5://127.0.0.1:7890 \
  bash scripts/install_sam2_official.sh
EOF
}

repo_dir="third_party/sam2"
branch="main"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      repo_dir="${2:?missing value for --dir}"
      shift 2
      ;;
    --branch)
      branch="${2:?missing value for --branch}"
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

if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 1
fi

mkdir -p "${UV_CACHE_DIR:-$PWD/tmp/uvcache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/tmp/uvcache}"

mkdir -p "$(dirname "$repo_dir")"

if [[ ! -d "$repo_dir/.git" ]]; then
  git clone --branch "$branch" --depth 1 https://github.com/facebookresearch/sam2.git "$repo_dir"
else
  git -C "$repo_dir" fetch --depth 1 origin "$branch"
  git -C "$repo_dir" checkout "$branch"
  git -C "$repo_dir" pull --ff-only origin "$branch"
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$(command -v python)" -e "$repo_dir"
elif python -m pip --version >/dev/null 2>&1; then
  python -m pip install -e "$repo_dir"
else
  echo "Neither uv nor pip is available for installing the official SAM2 repo." >&2
  exit 1
fi

echo "SAM2 official repository installed at $repo_dir"
