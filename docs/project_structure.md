# 项目结构说明

## 代码层

- `src/sam_vos/cli.py`
  - 命令行入口
  - 分发 `image` / `video` 子命令
- `src/sam_vos/sam2_backend.py`
  - 适配官方 `SAM2` API
  - 负责加载模型、预测图像、初始化视频状态
- `src/sam_vos/media.py`
  - 处理提示解析、图片读取、mask 处理
- `src/sam_vos/visualize.py`
  - 负责保存图片和视频

## 文档层

- `docs/README.md`
  - 文档总入口
- `docs/setup.md`
  - 安装与下载
- `docs/usage.md`
  - 实际运行命令
- `docs/report_outline.md`
  - 报告章节结构
- `docs/report_template.md`
  - 可直接填写的报告模板

## 输出层

- `output/`
  - 所有实验结果都放在这里
  - 建议按 `image/`、`video/`、`benchmark/` 分类

## 参考资产

- 官方 SAM2 仓库已安装在 `/media/chronicarl/Data/sam2`
- 权重已下载到 `/media/chronicarl/Data/sam2-checkpoints`
