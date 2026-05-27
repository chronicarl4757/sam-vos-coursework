# 基于 SAM2 的视频目标分割课程作业

这是一个面向“计算机视觉大作业：基于 SAM 的视频目标分割”的最小可运行项目骨架。

## 目标

- 跑通 `SAM2` 单帧图像分割
- 跑通 `SAM2` 视频目标分割
- 输出适合写报告的可视化结果
- 为 Codabench / MOSEv2 / MeViS 的后续实验留出接口

## 目录结构

```text
src/sam_vos/      核心代码
tests/            最小单元测试
docs/             文档入口、模板、运行说明
scripts/          安装、下载、smoke test 和打包脚本
```

## 环境

建议使用 Python 3.10+。基础依赖：

- `numpy`
- `pillow`
- `opencv-python`

若要真正运行分割，还需要安装官方 `SAM2` 代码和对应权重。

推荐先创建虚拟环境，再安装本项目：

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

### 安装 SAM2

按照官方仓库安装：

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

如果你想直接用本仓库里的脚本：

```bash
bash scripts/install_sam2_official.sh
```

官方加载方式见其 README：

- 图像：`build_sam2` + `SAM2ImagePredictor`
- 视频：`build_sam2_video_predictor`

如果你已经把官方仓库放到数据盘，这个仓库默认参考的是：

- `/media/chronicarl/Data/sam2`
- `/media/chronicarl/Data/sam2-checkpoints`

### 下载权重

官方仓库提供了 `sam2.1` checkpoints 的下载地址。你也可以直接在本仓库里运行：

```bash
bash scripts/download_sam2_checkpoints.sh
```

默认会下载以下四个权重到 `checkpoints/`：

- `sam2.1_hiera_tiny.pt`
- `sam2.1_hiera_small.pt`
- `sam2.1_hiera_base_plus.pt`
- `sam2.1_hiera_large.pt`

如果只想下载单个模型：

```bash
bash scripts/download_sam2_checkpoints.sh --model sam2.1_hiera_large
```

如果你在大陆网络环境下下载，建议先走本机代理：

```bash
HTTP_PROXY=http://127.0.0.1:7890 \
HTTPS_PROXY=http://127.0.0.1:7890 \
ALL_PROXY=socks5://127.0.0.1:7890 \
bash scripts/download_sam2_checkpoints.sh
```

如果官方源不稳定，可以切到 Hugging Face：

```bash
uv pip install -e '.[hf]'
bash scripts/download_sam2_checkpoints.sh --source hf
```

如果你在中国大陆环境下使用 Hugging Face 镜像，可以额外设置：

```bash
HF_ENDPOINT=https://hf-mirror.com
```

## 快速开始

建议先跑 smoke test，确认整条链路可用：

```bash
bash scripts/smoke_sam2.sh
```

如果你在 Colab 上跑，可以直接用一键启动脚本：

```bash
bash scripts/colab_bootstrap.sh --run-smoke
```

Colab 中的推荐步骤：

```bash
git clone https://github.com/chronicarl4757/sam-vos-coursework.git
cd sam-vos-coursework
bash scripts/colab_bootstrap.sh --run-smoke
```

如果已经完成环境安装，可以分别跑图像或视频：

```bash
bash scripts/run_image_colab.sh --image /content/your_image.jpg --box 120,80,640,520
bash scripts/run_video_colab.sh --video /content/your_video.mp4 --box 120,80,640,520
```

如果你的视频已经解压成帧，目录结构是 `video_dataset/JPEGImages/video_id/frame_id.jpg`，可以直接跑整个数据集：

```bash
bash scripts/run_jpeg_dataset_colab.sh \
  --video-root /content/video_dataset/JPEGImages \
  --box 120,80,640,520 \
  --make-preview
```

如果你的数据放在 Google Drive 或挂载盘上，建议加上本地拷贝，通常会快很多：

```bash
bash scripts/run_jpeg_dataset_colab.sh \
  --video-root /content/video_dataset/JPEGImages \
  --box 120,80,640,520 \
  --copy-to-local \
  --local-root /content/sam_jpeg_cache
```

基础 SAM Benchmark 不要直接提交 `sam-vos video` 的 `masks/mask_*.png`。参考 notebook 的正确做法是用首帧 annotation 作为 mask prompt：

```bash
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

标准提交文件夹是 `sam_benchmark_submit/sample_submission/`，overlay 单独放在 `sam_benchmark_submit/overlays/`，上传文件是 `sam_benchmark_submit/submission.zip`。具体见 [`docs/sam_benchmark.md`](docs/sam_benchmark.md)。

MOSEv2 进阶提交可以用首帧标注自动生成 SAM2 提示，并打包 Codabench zip：

```bash
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

标准提交文件夹是 `mosev2_sam2_tiny/submission/`，overlay 单独放在 `mosev2_sam2_tiny/overlays/`，上传文件是 `mosev2_sam2_tiny/submission.zip`。具体格式见 [`docs/mosev2.md`](docs/mosev2.md)。

更完整的文档入口见：

- [`docs/README.md`](docs/README.md)
- [`docs/setup.md`](docs/setup.md)
- [`docs/usage.md`](docs/usage.md)

### 1. 单帧图像分割

```bash
sam-vos image \
  --image path/to/image.jpg \
  --checkpoint path/to/sam2.1_hiera_large.pt \
  --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --points 520,380 \
  --labels 1 \
  --output-dir output/image_demo
```

也可以用框：

```bash
sam-vos image \
  --image path/to/image.jpg \
  --checkpoint path/to/sam2.1_hiera_large.pt \
  --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --box 120,80,640,520 \
  --output-dir output/image_demo
```

输出文件：

- `overlay.png`
- `mask.png`
- `mask.npy`
- `prompt.json`

### 2. 视频目标分割

```bash
sam-vos video \
  --video path/to/video.mp4 \
  --checkpoint path/to/sam2.1_hiera_large.pt \
  --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --frame-index 0 \
  --box 120,80,640,520 \
  --output-dir output/video_demo
```

输出文件：

- `overlay.mp4`
- `masks/`
- `frames/`
- `prompt.json`

### 3. 抽帧做轻量实验

如果视频较长，先降采样：

```bash
sam-vos video \
  --video path/to/video.mp4 \
  --checkpoint path/to/sam2.1_hiera_large.pt \
  --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --frame-index 0 \
  --box 120,80,640,520 \
  --output-dir output/video_demo \
  --frame-step 5
```

## 报告建议结构

1. 作业概述与实验环境
2. `SAM2` baseline 方法说明
3. 单帧图像分割结果
4. 视频目标分割结果
5. Benchmark 提交与得分
6. 失败案例分析与改进方向
7. 进阶实验：`MOSEv2` / `MeViS` / baseline 改进

## 说明

- 本仓库只负责推理、结果导出和报告整理，不包含 SAM2 训练代码。
- 真实提交前请把姓名、学号和 Codabench 用户名写进报告。

## 最终提交打包

先把报告 PDF 和媒体结果整理好，再执行：

```bash
bash scripts/package_submission.sh \
  --report path/to/report.pdf \
  --media path/to/media_dir \
  --output submission.zip
```
