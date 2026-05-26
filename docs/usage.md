# 运行指南

## 1. Smoke test

先跑一个最小闭环，确认 `SAM2` 的安装、权重和命令入口都可用：

```bash
bash scripts/smoke_sam2.sh
```

默认会在 `output/sam2_smoke/` 下生成：

- `image/overlay.png`
- `image/mask.png`
- `image/mask.npy`
- `video/overlay.mp4`
- `video/frames/`
- `video/masks/`

## 2. 单帧图像分割

```bash
source .venv/bin/activate
sam-vos image \
  --image path/to/image.jpg \
  --checkpoint /media/chronicarl/Data/sam2-checkpoints/sam2.1_hiera_tiny.pt \
  --model-cfg /media/chronicarl/Data/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
  --box 120,80,640,520 \
  --output-dir output/image_demo
```

如果只想用点提示：

```bash
sam-vos image \
  --image path/to/image.jpg \
  --checkpoint /media/chronicarl/Data/sam2-checkpoints/sam2.1_hiera_tiny.pt \
  --model-cfg /media/chronicarl/Data/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
  --points 520,380 \
  --labels 1 \
  --single-mask \
  --output-dir output/image_demo
```

## 3. 视频目标分割

`--video` 可以直接接：

- 一个 `mp4`
- 一个按帧编号命名的 JPEG/PNG 目录，例如 `00000.jpg`, `00001.jpg`

```bash
source .venv/bin/activate
sam-vos video \
  --video path/to/video.mp4 \
  --checkpoint /media/chronicarl/Data/sam2-checkpoints/sam2.1_hiera_tiny.pt \
  --model-cfg /media/chronicarl/Data/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
  --frame-index 0 \
  --box 120,80,640,520 \
  --offload-video-to-cpu \
  --offload-state-to-cpu \
  --output-dir output/video_demo
```

如果视频较长，可以用抽帧导出减小可视化成本：

```bash
sam-vos video \
  --video path/to/video.mp4 \
  --checkpoint /media/chronicarl/Data/sam2-checkpoints/sam2.1_hiera_tiny.pt \
  --model-cfg /media/chronicarl/Data/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
  --frame-index 0 \
  --box 120,80,640,520 \
  --frame-step 5 \
  --offload-video-to-cpu \
  --offload-state-to-cpu \
  --output-dir output/video_demo
```

## 4. 最终交付流程

1. 跑 `image` 和 `video` 结果
2. 记录失败案例和成功案例
3. 组织 Codabench 提交截图
4. 填写 [实验记录模板](./experiment_log_template.md)
5. 按 [报告模板](./report_template.md) 输出 PDF
