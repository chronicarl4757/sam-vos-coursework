# SAM Benchmark Submission

This page follows the reference notebook at `~/Downloads/sam2.ipynb`.

The key point is to use the first-frame annotation as a mask prompt:

```python
video_predictor.add_new_mask(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=object_id,
    mask=first_frame_mask == object_id,
)
```

Do not submit `sam-vos video` visualization masks directly. Those files are named like `mask_00031.png` and usually contain only a binary foreground mask. The benchmark expects indexed masks named like `00031.png`.

## Expected Dataset Layout

```text
dataset/
├── JPEGImages
│   └── video_id
│       ├── 00000.jpg
│       └── ...
└── Annotations
    └── video_id
        └── 00000.png
```

The first annotation frame must be an indexed PNG:

```text
0 = background
1 = object 1
2 = object 2
...
```

## Colab Run

```bash
git pull
bash scripts/colab_bootstrap.sh

python scripts/run_sam_benchmark_from_annotations.py \
  --dataset-root /content/video_dataset \
  --output-root /content/drive/MyDrive/sam_benchmark_submit \
  --checkpoint /content/sam2-checkpoints/sam2.1_hiera_tiny.pt \
  --model-cfg /content/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda \
  --copy-to-local \
  --local-root /content/sam_frames_cache \
  --save-overlays \
  --overlay-stride 25
```

Upload:

```text
/content/drive/MyDrive/sam_benchmark_submit/submission.zip
```

Standard submission folder:

```text
/content/drive/MyDrive/sam_benchmark_submit/sample_submission/
```

Overlay folder:

```text
/content/drive/MyDrive/sam_benchmark_submit/overlays/
```

## Dry Run

```bash
python scripts/run_sam_benchmark_from_annotations.py \
  --dataset-root /content/video_dataset \
  --output-root /content/drive/MyDrive/sam_benchmark_dry_run \
  --limit 1 \
  --device cuda \
  --no-zip
```

## Common Failure

If Codabench reports:

```text
invalid literal for int() with base 10: 'mask_00127'
```

the submitted filenames are wrong. The expected filename is `00127.png`, not `mask_00127.png`.
