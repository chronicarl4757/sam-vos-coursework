# 环境与安装

## 1. 基础环境

- Python 3.10+
- 可用的 `uv` 或 `pip`
- `numpy`
- `pillow`
- `opencv-python`

建议先创建虚拟环境：

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

## 2. 安装官方 SAM2

推荐直接使用仓库脚本：

```bash
bash scripts/install_sam2_official.sh --dir /media/chronicarl/Data/sam2
```

如果大陆网络环境下访问 GitHub 不稳，建议配合本机代理：

```bash
HTTP_PROXY=http://127.0.0.1:7890 \
HTTPS_PROXY=http://127.0.0.1:7890 \
ALL_PROXY=socks5://127.0.0.1:7890 \
bash scripts/install_sam2_official.sh --dir /media/chronicarl/Data/sam2
```

## 3. 下载 checkpoints

```bash
bash scripts/download_sam2_checkpoints.sh --dir /media/chronicarl/Data/sam2-checkpoints
```

只下载轻量模型可用：

```bash
bash scripts/download_sam2_checkpoints.sh \
  --dir /media/chronicarl/Data/sam2-checkpoints \
  --model sam2.1_hiera_tiny
```

如果官方源失败，可以切换到 Hugging Face：

```bash
uv pip install -e '.[hf]'
bash scripts/download_sam2_checkpoints.sh --source hf
```

## 4. 官方配置路径

SAM2.1 常用配置文件位置：

- `/media/chronicarl/Data/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml`
- `/media/chronicarl/Data/sam2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml`
- `/media/chronicarl/Data/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml`
- `/media/chronicarl/Data/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml`

## 5. 建议的权重

优先从小模型开始验证：

- `sam2.1_hiera_tiny.pt`
- `sam2.1_hiera_small.pt`

在确定流程正确后再切换到更大的模型。
