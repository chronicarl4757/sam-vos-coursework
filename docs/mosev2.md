# MOSEv2 Benchmark

Codabench page: https://www.codabench.org/competitions/10062/

## Submission Format

MOSEv2 expects a `.zip` file. The root of the zip must contain video folders directly. Each video folder contains predicted indexed-mask PNG files whose filenames match the image frame names:

```text
submission.zip
├── video_name_1
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── video_name_2
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

It reads the first-frame annotation, extracts one box prompt for each object id, runs SAM2 video propagation object by object, and writes indexed PNG masks for submission.

## Colab Usage

After repository setup:

```bash
bash scripts/colab_bootstrap.sh

python scripts/run_mosev2_from_annotations.py \
  --dataset-root /content/MOSEv2/valid \
  --output-root /content/drive/MyDrive/mosev2_sam2_tiny \
  --checkpoint /content/sam2-checkpoints/sam2.1_hiera_tiny.pt \
  --model-cfg /content/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda
```

The upload file is:

```text
/content/drive/MyDrive/mosev2_sam2_tiny/submission.zip
```

For a short dry run:

```bash
python scripts/run_mosev2_from_annotations.py \
  --dataset-root /content/MOSEv2/valid \
  --output-root /content/drive/MyDrive/mosev2_dry_run \
  --limit 1 \
  --frame-step 5 \
  --device cuda
```

Use `--frame-step 1` for a real Codabench submission.

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

It also converts RGB binary masks to single-channel indexed PNG masks.

## Report Notes

Record these items in the report:

- Codabench username.
- Submission id and screenshot.
- Metrics: `J`, `F`, `J&F`, `Ḟ`, `J&Ḟ`, plus disappear/reappear metrics if available.
- Whether the run used `sam2.1_hiera_tiny`, `small`, `base_plus`, or `large`.
- Failure analysis: disappearance, reappearance, occlusion, small objects, similar distractors, and mask drift.
