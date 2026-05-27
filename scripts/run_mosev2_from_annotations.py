#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ObjectPrompt:
    object_id: int
    box: tuple[int, int, int, int]


def read_index_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def object_prompts_from_mask(mask: np.ndarray, margin: int = 0) -> list[ObjectPrompt]:
    if mask.ndim == 3:
        mask = mask[..., 0]
    height, width = mask.shape
    prompts: list[ObjectPrompt] = []
    for object_id in sorted(int(v) for v in np.unique(mask) if int(v) != 0):
        ys, xs = np.where(mask == object_id)
        if len(xs) == 0:
            continue
        x0 = max(0, int(xs.min()) - margin)
        y0 = max(0, int(ys.min()) - margin)
        x1 = min(width - 1, int(xs.max()) + margin)
        y1 = min(height - 1, int(ys.max()) + margin)
        prompts.append(ObjectPrompt(object_id=object_id, box=(x0, y0, x1, y1)))
    return prompts


def find_annotation(annotation_dir: Path, frame_stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = annotation_dir / f"{frame_stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no annotation found for first frame {frame_stem!r} under {annotation_dir}")


def frame_paths(video_dir: Path) -> list[Path]:
    paths = sorted(
        [
            *video_dir.glob("*.jpg"),
            *video_dir.glob("*.jpeg"),
            *video_dir.glob("*.png"),
        ]
    )
    if not paths:
        raise FileNotFoundError(f"no frames found under {video_dir}")
    return paths


def run_video(
    *,
    repo_dir: Path,
    video_dir: Path,
    annotation_dir: Path,
    raw_output_dir: Path,
    submission_video_dir: Path,
    checkpoint: Path,
    model_cfg: Path,
    device: str,
    frame_step: int,
    margin: int,
    keep_raw: bool,
) -> dict:
    frames = frame_paths(video_dir)
    first_frame = frames[0]
    first_mask = read_index_mask(find_annotation(annotation_dir, first_frame.stem))
    prompts = object_prompts_from_mask(first_mask, margin=margin)
    if not prompts:
        raise ValueError(f"no foreground object found in {annotation_dir}")

    raw_output_dir.mkdir(parents=True, exist_ok=True)
    object_output_dirs: list[Path] = []
    sam_vos = repo_dir / ".venv" / "bin" / "sam-vos"
    if not sam_vos.exists():
        sam_vos = Path("sam-vos")

    for prompt in prompts:
        object_dir = raw_output_dir / f"object_{prompt.object_id}"
        object_output_dirs.append(object_dir)
        box_text = ",".join(str(v) for v in prompt.box)
        cmd = [
            str(sam_vos),
            "--device",
            device,
            "video",
            "--video",
            str(video_dir),
            "--checkpoint",
            str(checkpoint),
            "--model-cfg",
            str(model_cfg),
            "--output-dir",
            str(object_dir),
            "--frame-index",
            "0",
            "--object-id",
            str(prompt.object_id),
            "--box",
            box_text,
            "--frame-step",
            str(frame_step),
            "--fps",
            "12",
            "--offload-video-to-cpu",
            "--offload-state-to-cpu",
            "--async-loading-frames",
        ]
        subprocess.run(cmd, check=True)

    submission_video_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames[:: max(1, frame_step)]:
        output = np.zeros_like(first_mask, dtype=np.uint8)
        for prompt, object_dir in zip(prompts, object_output_dirs, strict=True):
            mask_path = object_dir / "masks" / f"mask_{int(frame.stem):05d}.png"
            if not mask_path.exists():
                continue
            mask = np.asarray(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]
            output[mask > 0] = prompt.object_id
        Image.fromarray(output).save(submission_video_dir / f"{frame.stem}.png")

    manifest = {
        "video": video_dir.name,
        "frames": len(frames),
        "exported_frames": len(frames[:: max(1, frame_step)]),
        "objects": [{"object_id": p.object_id, "box": list(p.box)} for p in prompts],
    }
    (raw_output_dir / "mosev2_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not keep_raw:
        for object_dir in object_output_dirs:
            overlay = object_dir / "overlay.mp4"
            if overlay.exists():
                overlay.unlink()

    return manifest


def zip_submission(submission_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(submission_dir.rglob("*.png")):
            archive.write(path, path.relative_to(submission_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SAM2 on MOSEv2-style data and build a Codabench zip.")
    parser.add_argument("--dataset-root", required=True, help="Directory containing JPEGImages/ and Annotations/.")
    parser.add_argument("--output-root", default="output/mosev2", help="Output root for raw outputs and submission files.")
    parser.add_argument("--checkpoint", default="/content/sam2-checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--model-cfg", default="/content/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-step", type=int, default=1, help="Use 1 for official submission; larger values only for dry runs.")
    parser.add_argument("--box-margin", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N videos. 0 means all videos.")
    parser.add_argument("--keep-raw", action="store_true", help="Keep per-object overlays in the raw output directory.")
    parser.add_argument("--no-zip", action="store_true", help="Do not create submission.zip.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_dir = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    jpeg_root = dataset_root / "JPEGImages"
    annotation_root = dataset_root / "Annotations"
    output_root = Path(args.output_root).expanduser().resolve()
    raw_root = output_root / "raw"
    submission_root = output_root / "submission"

    if not jpeg_root.is_dir():
        raise FileNotFoundError(f"missing JPEGImages directory: {jpeg_root}")
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"missing Annotations directory: {annotation_root}")

    videos = sorted(path for path in jpeg_root.iterdir() if path.is_dir())
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        raise FileNotFoundError(f"no video directories found under {jpeg_root}")

    manifests = []
    for index, video_dir in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video_dir.name}", flush=True)
        manifests.append(
            run_video(
                repo_dir=repo_dir,
                video_dir=video_dir,
                annotation_dir=annotation_root / video_dir.name,
                raw_output_dir=raw_root / video_dir.name,
                submission_video_dir=submission_root / video_dir.name,
                checkpoint=Path(args.checkpoint).expanduser(),
                model_cfg=Path(args.model_cfg).expanduser(),
                device=args.device,
                frame_step=args.frame_step,
                margin=args.box_margin,
                keep_raw=args.keep_raw,
            )
        )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps({"videos": manifests}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not args.no_zip:
        zip_submission(submission_root, output_root / "submission.zip")
        print(f"Submission zip written to: {output_root / 'submission.zip'}")
    print(f"Submission directory written to: {submission_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
