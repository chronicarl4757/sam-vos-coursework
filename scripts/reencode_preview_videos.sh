#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reencode_preview_videos.sh --input FILE_OR_DIR [--output-dir DIR]

Options:
  --input PATH      Single mp4 file or a directory containing mp4 files.
  --output-dir DIR   Output directory for re-encoded preview videos.
                    Default: <input parent>/preview or <input>/preview
  -h, --help        Show this help message

Examples:
  bash scripts/reencode_preview_videos.sh --input output/user_runs_gpu/videos/vedio01/overlay.mp4
  bash scripts/reencode_preview_videos.sh --input output/user_runs_gpu/videos --output-dir output/user_runs_gpu/preview
EOF
}

input_path=""
output_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      input_path="${2:?missing value for --input}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:?missing value for --output-dir}"
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

if [[ -z "$input_path" ]]; then
  echo "--input is required" >&2
  usage >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required." >&2
  exit 1
fi

input_abs="$(realpath "$input_path")"
if [[ -d "$input_abs" ]]; then
  if [[ -z "$output_dir" ]]; then
    output_dir="$input_abs/preview"
  fi
  mkdir -p "$output_dir"
  mapfile -t files < <(find "$input_abs" -type f -name '*.mp4' | sort)
else
  if [[ -z "$output_dir" ]]; then
    output_dir="$(dirname "$input_abs")/preview"
  fi
  mkdir -p "$output_dir"
  files=("$input_abs")
fi

for src in "${files[@]}"; do
  base="$(basename "$src")"
  parent="$(basename "$(dirname "$src")")"
  dst="$output_dir/${parent}_${base}"
  echo "Re-encoding $src -> $dst"
  ffmpeg -y -i "$src" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    -an \
    "$dst"
done

echo "Preview videos written to: $output_dir"
