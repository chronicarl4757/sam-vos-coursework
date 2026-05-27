# MOSEv2 Colab Notebook Blocks

下面代码块按 Colab notebook 单元组织。假设 MOSEv2 数据压缩包已经放在 Google Drive，例如：

```text
/content/drive/MyDrive/MOSEv2.zip
```

如果你的压缩包文件名不同，只需要修改 `MOSE_ZIP`。

## 1. 挂载云盘

```python
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
```

## 2. 克隆项目并配置环境

```bash
%cd /content
!rm -rf sam-vos-coursework
!git clone https://github.com/chronicarl4757/sam-vos-coursework.git
%cd /content/sam-vos-coursework
!bash scripts/colab_bootstrap.sh --run-smoke
```

## 3. 设置路径

```python
from pathlib import Path

REPO = Path("/content/sam-vos-coursework")
DRIVE = Path("/content/drive/MyDrive")

MOSE_ZIP = DRIVE / "MOSEv2.zip"
MOSE_ROOT = Path("/content/MOSEv2")
OUTPUT_ROOT = DRIVE / "mosev2_sam2_tiny"
CACHE_ROOT = Path("/content/mosev2_frames_cache")

CHECKPOINT = Path("/content/sam2-checkpoints/sam2.1_hiera_tiny.pt")
MODEL_CFG = Path("/content/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml")

print("MOSE_ZIP:", MOSE_ZIP)
print("MOSE_ROOT:", MOSE_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
```

## 4. 解压 MOSEv2 数据

```bash
!rm -rf /content/MOSEv2
!mkdir -p /content/MOSEv2
!unzip -q "$MOSE_ZIP" -d /content/MOSEv2
!find /content/MOSEv2 -maxdepth 3 -type d | head -40
```

## 5. 自动定位数据根目录

数据根目录必须直接包含 `JPEGImages/` 和 `Annotations/`。如果压缩包解压后多了一层目录，下面代码会自动查找。

```python
from pathlib import Path

candidates = [p for p in MOSE_ROOT.rglob("JPEGImages") if (p.parent / "Annotations").is_dir()]
if not candidates:
    raise FileNotFoundError("Cannot find a directory containing both JPEGImages/ and Annotations/.")

DATASET_ROOT = candidates[0].parent
print("DATASET_ROOT:", DATASET_ROOT)
print("video count:", len([p for p in (DATASET_ROOT / "JPEGImages").iterdir() if p.is_dir()]))
```

## 6. 检查首帧标注与对象 id

```python
from PIL import Image
import numpy as np

video_dirs = sorted([p for p in (DATASET_ROOT / "JPEGImages").iterdir() if p.is_dir()])
first_video = video_dirs[0]
first_frame = sorted(first_video.glob("*.jpg"))[0]
first_ann = DATASET_ROOT / "Annotations" / first_video.name / f"{first_frame.stem}.png"

mask = np.asarray(Image.open(first_ann))
object_ids = sorted(int(x) for x in np.unique(mask) if int(x) != 0)

print("first video:", first_video.name)
print("first frame:", first_frame.name)
print("first annotation:", first_ann)
print("object ids:", object_ids)
print("mask shape:", mask.shape)
```

## 7. Dry Run：先跑 1 个视频

```bash
%cd /content/sam-vos-coursework
!rm -rf /content/drive/MyDrive/mosev2_dry_run
!mkdir -p /content/mosev2_frames_cache

!python scripts/run_mosev2_from_annotations.py \
  --dataset-root "$DATASET_ROOT" \
  --output-root /content/drive/MyDrive/mosev2_dry_run \
  --checkpoint /content/sam2-checkpoints/sam2.1_hiera_tiny.pt \
  --model-cfg /content/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda \
  --limit 1 \
  --copy-to-local \
  --local-root /content/mosev2_frames_cache \
  --save-overlays \
  --overlay-stride 25 \
  --no-zip
```

## 8. 检查 Dry Run 输出

```python
from pathlib import Path

dry = Path("/content/drive/MyDrive/mosev2_dry_run")
print("submission pngs:", len(list((dry / "submission").rglob("*.png"))))
print("overlay pngs:", len(list((dry / "overlays").rglob("*.png"))))
print("manifest:", dry / "manifest.json")
```

## 9. 正式运行 MOSEv2

```bash
%cd /content/sam-vos-coursework
!rm -rf /content/drive/MyDrive/mosev2_sam2_tiny
!mkdir -p /content/mosev2_frames_cache

!python scripts/run_mosev2_from_annotations.py \
  --dataset-root "$DATASET_ROOT" \
  --output-root /content/drive/MyDrive/mosev2_sam2_tiny \
  --checkpoint /content/sam2-checkpoints/sam2.1_hiera_tiny.pt \
  --model-cfg /content/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda \
  --copy-to-local \
  --local-root /content/mosev2_frames_cache \
  --save-overlays \
  --overlay-stride 25
```

正式输出：

```text
/content/drive/MyDrive/mosev2_sam2_tiny/submission/
/content/drive/MyDrive/mosev2_sam2_tiny/overlays/
/content/drive/MyDrive/mosev2_sam2_tiny/manifest.json
/content/drive/MyDrive/mosev2_sam2_tiny/submission.zip
```

## 10. 检查提交 zip 结构

MOSEv2 的 zip 根目录应该直接是视频文件夹，不能再包一层 `sample_submission/`。

```bash
!unzip -l /content/drive/MyDrive/mosev2_sam2_tiny/submission.zip | head -80
```

```python
import zipfile
from pathlib import Path
from PIL import Image
import numpy as np

zip_path = Path("/content/drive/MyDrive/mosev2_sam2_tiny/submission.zip")

with zipfile.ZipFile(zip_path) as zf:
    names = [n for n in zf.namelist() if n.endswith(".png")]
    roots = sorted(set(Path(n).parts[0] for n in names))
    bad_depth = [n for n in names if len(Path(n).parts) != 2]
    bad_mask_prefix = [n for n in names if Path(n).stem.startswith("mask_")]
    bad_int = []
    for n in names:
        try:
            int(Path(n).stem)
        except ValueError:
            bad_int.append(n)

    with zf.open(names[0]) as f:
        img = Image.open(f).copy()
    arr = np.asarray(img)

print("zip:", zip_path)
print("png count:", len(names))
print("video count:", len(roots))
print("bad depth:", len(bad_depth))
print("bad mask_ prefix:", len(bad_mask_prefix))
print("bad integer stems:", len(bad_int))
print("sample mode:", img.mode)
print("sample shape:", arr.shape)
print("sample unique ids:", np.unique(arr)[:20])
```

## 11. 下载或上传 Codabench

上传文件：

```text
/content/drive/MyDrive/mosev2_sam2_tiny/submission.zip
```

提交后在报告里记录：

```text
Codabench username:
Submission ID:
Submit time:
J:
F:
J&F:
J&F_new:
Screenshot path:
```

