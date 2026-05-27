# MOSEv2 Benchmark

Codabench page: https://www.codabench.org/competitions/10062/

For a full Colab notebook-style workflow, see [MOSEv2 Colab Notebook Blocks](./mosev2_colab_notebook.md).

## Submission Format

MOSEv2 expects a `.zip` file. The root of the zip must contain video folders directly. Each video folder contains predicted indexed-mask PNG files whose filenames match the image frame names:

```text
submission.zip
├── video_name_1/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── video_name_2/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
└── ...
```

The official page says the layout follows DAVIS2017 style and recommends checking the sample submission from the dataset download link.

## Dataset Layout

The local runner expects:

```text
MOSEv2/
├── JPEGImages
│   └── video_id
│       ├── 00000.jpg
│       └── ...
└── Annotations
    └── video_id
        ├── 00000.png
        └── ...
```

It reads the first-frame annotation, uses each object mask as a SAM2 `add_new_mask` prompt, propagates all objects through the video, and writes indexed PNG masks for submission.

## Colab Usage

After repository setup:

```bash
bash scripts/colab_bootstrap.sh

python scripts/run_mosev2_from_annotations.py \
  --dataset-root /content/MOSEv2/valid \
  --output-root /content/drive/MyDrive/mosev2_sam2_tiny \
  --checkpoint /content/sam2-checkpoints/sam2.1_hiera_tiny.pt \
  --model-cfg /content/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda \
  --copy-to-local \
  --local-root /content/mosev2_frames_cache \
  --save-overlays \
  --overlay-stride 25
```

The upload file is:

```text
/content/drive/MyDrive/mosev2_sam2_tiny/submission.zip
```

The standard submission folder is:

```text
/content/drive/MyDrive/mosev2_sam2_tiny/submission/
```

Visual overlays are stored separately:

```text
/content/drive/MyDrive/mosev2_sam2_tiny/overlays/
```

For a short dry run:

```bash
python scripts/run_mosev2_from_annotations.py \
  --dataset-root /content/MOSEv2/valid \
  --output-root /content/drive/MyDrive/mosev2_dry_run \
  --limit 1 \
  --device cuda \
  --no-zip
```

Do not use frame skipping for a real Codabench submission. The runner writes every frame by default.

## Fix Existing sam-vos Mask Zips

If Codabench reports an error like:

```text
invalid literal for int() with base 10: 'mask_00127'
```

the submitted PNG filenames still contain the `mask_` prefix. Normalize the zip before uploading:

```bash
python scripts/fix_mosev2_submission_zip.py \
  --input /content/drive/MyDrive/sam_vos_final_submission.zip \
  --output /content/drive/MyDrive/mosev2_submission_fixed.zip
```

The fixed zip uses:

```text
sample_submission/video_id/00000.png
```

This helper is only for repairing old binary-mask zips. It is not the recommended MOSEv2 runner, because it cannot recover object ids that were lost in an old binary mask.

## Report Notes

Record these items in the report:

- Codabench username.
- Submission id and screenshot.
- Metrics: `J`, `F`, `J&F`, `Ḟ`, `J&Ḟ`, plus disappear/reappear metrics if available.
- Whether the run used `sam2.1_hiera_tiny`, `small`, `base_plus`, or `large`.
- Failure analysis: disappearance, reappearance, occlusion, small objects, similar distractors, and mask drift.
