#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/package_submission.sh --report REPORT.pdf --media MEDIA_DIR --output OUT.zip

Required:
  --report REPORT.pdf   Final report PDF
  --media MEDIA_DIR     Directory with images, videos, masks, screenshots
  --output OUT.zip      Output zip path

Optional:
  --code CODE_DIR       Code directory to include. Default: current repo root
  -h, --help            Show this help message
EOF
}

report=""
media_dir=""
output=""
code_dir="$(pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report)
      report="${2:?missing value for --report}"
      shift 2
      ;;
    --media)
      media_dir="${2:?missing value for --media}"
      shift 2
      ;;
    --output)
      output="${2:?missing value for --output}"
      shift 2
      ;;
    --code)
      code_dir="${2:?missing value for --code}"
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

if [[ -z "$report" || -z "$media_dir" || -z "$output" ]]; then
  usage >&2
  exit 1
fi

python - "$report" "$media_dir" "$output" "$code_dir" <<'PY'
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

report = Path(sys.argv[1]).expanduser().resolve()
media_dir = Path(sys.argv[2]).expanduser().resolve()
output = Path(sys.argv[3]).expanduser().resolve()
code_dir = Path(sys.argv[4]).expanduser().resolve()

if not report.exists():
    raise SystemExit(f"Report not found: {report}")
if not media_dir.exists():
    raise SystemExit(f"Media directory not found: {media_dir}")
if not code_dir.exists():
    raise SystemExit(f"Code directory not found: {code_dir}")

output.parent.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    tmp_root = Path(tmp)
    bundle = tmp_root / "submission"
    bundle.mkdir()
    shutil.copy2(report, bundle / report.name)
    shutil.copytree(media_dir, bundle / "media")

    code_target = bundle / "code"
    code_target.mkdir()
    for name in [
        "README.md",
        "pyproject.toml",
        "src",
        "docs",
        "scripts",
        "tests",
    ]:
        src = code_dir / name
        if src.exists():
            dst = code_target / name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    shutil.make_archive(str(output.with_suffix("")), "zip", root_dir=bundle)

print(f"Wrote {output}")
PY
