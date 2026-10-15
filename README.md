# DDRNet Cityscapes 语义分割 — AX650 量化部署

将 DDRNet（DDRNet-23 / DDRNet-23-slim，Cityscapes 19 类）从 ONNX 经
Pulsar2 量化编译为 `compiled.axmodel`，并在 AX650A 上完成全量 val 精度验证。

## 1. 参考项目

- **DDRNet**（TPAMI 2023）: *Deep Dual-Resolution Networks for Real-Time
  and Accurate Semantic Segmentation of Traffic Scenes*
  - 官方实现: <https://github.com/ydhongHIT/DDRNet>
- **MMSegmentation**（OpenMMLab）: DDRNet Cityscapes 训练配置与权重
 （模型由此工程导出，导出脚本不在本目录）
  - <https://github.com/open-mmlab/mmsegmentation>（`configs/ddrnet`）
- **AXERA Pulsar2** v7.0 量化编译工具链 + **axengine** 运行时（AX650A）
  - <https://pulsar2-docs.readthedocs.io/en/latest/pulsar2/introduction.html>

## 2. 模型（`onnx/`）

| 文件 | 网络 | 参数量 | 输入 (NCHW) | 大小 |
| --- | --- | --- | --- | --- |
| `ddrnet23_cityscapes-2048x1024.onnx` | DDRNet-23 | 20.15M | 1x3x**1024x2048** | 80.6 MB |
| `ddrnet23-slim_cityscapes-2048x1024.onnx` | DDRNet-23-slim | 5.70M | 1x3x**1024x2048** | 22.9 MB |
| `ddrnet23_cityscapes-1024x1024.onnx` | DDRNet-23 | 20.15M | 1x3x1024x1024 | 80.6 MB |
| `ddrnet23-slim_cityscapes-1024x1024.onnx` | DDRNet-23-slim | 5.70M | 1x3x1024x1024 | 22.9 MB |

- 文件名中的 `WxH` 表示图像宽高；张量一律为 NCHW，**官方输入为
  1x3x1024x2048**（本文档默认使用该尺寸）。
- 输入约定：FP32、**RGB**、按 `(x - mean) / std` 归一化，图内不含归一化
  节点；mean/std 为 ImageNet（RGB）：
  - mean = [123.675, 116.28, 103.53]
  - std  = [58.395, 57.12, 57.375]
- 量化部署使用 **DDRNet-23-slim**（`ddrnet23_cityscapes-2048x1024.onnx`
  仅用于 ONNX 侧功能核对/对比）。

## 3. 校准数据集准备（`ddrnet_make_calib.py`）

数据源：`../dataset/cityscapes/val`（500 张，frankfurt / lindau / munster）。
预处理与 `ddrnet_onnx_eval.py:preprocess` 一致：`cv2.resize`（INTER_LINEAR
到目标 H W）→ BGR→RGB → `(x - mean) / std`。所有产物归档为**扁平 tar**
（条目 `./<stem>.bin|png`），保存在 `../dataset/cityscapes/`。

```bash
cd DDRNet

# 1) 已归一化 float32 bin（[1,3,1024,1024]，值域约 [-2.1, 2.6]）
python ddrnet_make_calib.py

# 2) 已归一化 float32 bin（[1,3,1024,2048]，值域约 [-2.1, 2.6]）
python ddrnet_make_calib.py --input-size 1024 2048

# 3) 仅 resize 的 PNG（1024x2048，BGR，0-255），供 Image 校准
python ddrnet_make_calib.py --input-size 1024 2048 --save-images

# 4) 仅 resize 的 float32 bin（0-255），辅助模式
python ddrnet_make_calib.py --input-size 1024 2048 --resize-only
```

已生成的校准集：

| 产物（`../dataset/cityscapes/`） | 内容 | 用途 |
| --- | --- | --- |
| `ddrnet_calib_1024x1024-bin(.tar)` | 500 x float32 [1,3,1024,1024]，已归一化 | 1024x1024 模型的 Binary 校准 |
| `ddrnet_calib_1024x2048-bin(.tar)` | 500 x float32 [1,3,1024,2048]，已归一化 | `config_siluInU8_ptq.json`（Binary 校准） |
| `ddrnet_calib_1024x2048-resize-img(.tar)` | 500 x PNG，仅 resize | `config_siluInU8_ptq-preprocess.json`（Image 校准，归一化在板上完成） |

说明：`calibration_size: 32` 时 pulsar2 只从 tar 中取前 32 张；脚本生成时
已内嵌首张数值一致性自检（对 `preprocess` 输出 `assert_allclose` / 图像逐
位比对）。

## 4. 模型转换（配置与方法）

工具链：Pulsar2 v7.0（Docker 内运行），目标平台 **AX650A**；最终构建使用
NPU3 模式（`npu_mode: "NPU3"`）。

```bash
cd DDRNet
pulsar2 build --config config/config_siluInU8_ptq-preprocess.json
```

量化配置（`config/`）：

| 配置项 | `config_siluInU8_ptq.json` | `config_siluInU8_ptq-preprocess.json` |
| --- | --- | --- |
| input | `onnx/ddrnet23-slim_cityscapes-2048x1024.onnx` | 同左 |
| 校准数据 | `ddrnet_calib_1024x2048-bin.tar`（Binary） | `ddrnet_calib_1024x2048-resize-img.tar`（Image） |
| 校准 mean/std | 0 / 1（数据已归一化） | ImageNet RGB mean/std |
| 输入端处理 | 无（NCHW/FP32） | `src_format: BGR, src_layout: NHWC, src_dtype: U8` → `RGB/NCHW` + mean/std |
| Silu 精度 | U8 | U8 |
| conv bias | FP32 | FP32 |
| 精度分析 | EndToEnd / NPUBackend | 同左 |
| `compiler.check` | 2 | 0 |
| `npu_mode` | NPU3（当前文件） | NPU3 |
| output_dir | `output_ddrnet23-slim-ptq` | `output_ddrnet23-slim-ptq-preprocess-npu3` |

> `config_siluInU8_ptq.json` 为初版配置，当前文件已改为 NPU3；对应初版
> 构建（见下表）当时使用的是 NPU1。

> `config/config_siluInU16_ptq.json` 为 RTMDet（U16/S16 + smooth_quant）
> 参考模板，与 DDRNet 无关。

构建产出（每个 `output_*` 目录）：

- `compiled.axmodel` — 部署产物（含嵌入式输入预处理）
- `frontend/optimized.onnx` — 前端优化图
- `quant/quant_axmodel.onnx`（+ `.data/.json`）— 量化图
- `quant/dataset/input`、`quant/debug/precision_analysis_table.txt` —
  校准样本与逐层量化精度表（Cosine / MSE）

三次构建记录：

| 输出目录 | 校准方式 | npu_mode | toolchain | 说明 |
| --- | --- | --- | --- | --- |
| `output_ddrnet23-slim-ptq` | Binary（预归一化 bin） | NPU1 | v7.0 (`00a4a752`) | 初版，`compiler.check=2` |
| `output_ddrnet23-slim-ptq-preprocess` | Image（resize PNG） | NPU1 | v7.0 (`8d1bbdc2`) | 归一化移至板上 input_processors |
| `output_ddrnet23-slim-ptq-preprocess-npu3` | Image | NPU3 | v7.0 (`8d1bbdc2`) | **最终版本，上板验证用** |

量化精度（`output_ddrnet23-slim-ptq-preprocess-npu3/quant/debug/precision_analysis_table.txt`）：
Silu→U8、整网 U8；绝大部分中间层 Cosine > 0.97，末端输出层
Cosine 0.99421 / MSE 0.05802。

## 5. 上板精度（AX650A）

- 模型：`output_ddrnet23-slim-ptq-preprocess-npu3/compiled.axmodel`
  （axengine，`AxEngineExecutionProvider`）
- 数据：Cityscapes val **全量 500 张**，输入 1x3x1024x2048；
  GT 为 `*_gtFine_labelTrainIds.png`（ignore_index=255，19 类）
- 结果：**mIoU 77.79 / aAcc 95.92**，参考 mIoU 77.84（**仅降 0.05**）
- 速度：平均 **118.9 ms/帧**（500 张平均，meta.json）

| 指标 | 值 |
| --- | --- |
| mIoU | 77.79 |
| aAcc | 95.92 |
| 参考 mIoU | 77.84（diff -0.05） |
| 平均推理耗时 | 118.94 ms |

逐类 IoU 见 `axera_pred-slim_vs_gt/metrics.csv`（节选：road 97.85 /
building 92.44 / vegetation 92.20 / sky 94.83 / car 94.79；
bicycle 76.17 / motorcycle 59.30 / pole 64.01）。

结果文件：

- `axera_pred-slim/meta.json` — 上板推理元数据（500 张、118.94 ms）
- `axera_pred-slim/pred/<city>/*_pred.png` — 全量预测标签
- `axera_pred-slim_vs_gt/{metrics.json, metrics.csv, confusion_matrix.csv}`
  — 与 GT 对比的官方指标及混淆矩阵

参照（本机 ONNX Runtime，CPU FP32，非部署路径）：slim 111.7 ms/帧、
DDRNet-23 164.1 ms/帧（`onnx-vis/meta.json`）。

## 6. 运行脚本

| 脚本 | 作用 | 典型命令 |
| --- | --- | --- |
| `ddrnet_make_calib.py` | 生成校准集（bin / 图片，见第 3 节） | `python ddrnet_make_calib.py --input-size 1024 2048` |
| `ddrnet_onnx_eval.py` | ONNX FP32 精度/可视化（CPU/CUDA） | `python ddrnet_onnx_eval.py --model onnx/ddrnet23_cityscapes-2048x1024.onnx` |
| `ddrnet_axera_eval.py` | axmodel 上板评估（axengine） | `python ddrnet_axera_eval.py` |

- `ddrnet_onnx_eval.py`：默认模型 `onnx/ddrnet23_cityscapes-2048x1024.onnx`，
  默认只跑前 5 张（`--limit`），输出 `onnx-vis/`；给 `--gt-dir` 且存在
  `gtFine` 时计算 mIoU/aAcc，全量评估用 `--limit 0`。
- `ddrnet_axera_eval.py`：默认模型
  `output_ddrnet23-slim-ptq-preprocess-npu3/compiled.axmodel`，默认全量
  500 张，输出 `axera_pred-slim/`；输入为 **U8 NHWC**（脚本仅
  resize + BGR→RGB，归一化由 axmodel 内嵌 input_processors 完成），
  因此无需在本机安装 onnxruntime，改为 `import axengine`。
- 环境依赖：`python3`、`opencv-python`、`numpy`；ONNX 评估另需
  `onnxruntime`；上板评估需按 AXERA 工具链安装 `axengine`；模型转换需
  Pulsar2 v7.0 运行环境（Docker）。

## 7. 目录结构

```
DDRNet/
├── README.md
├── config/                                  # pulsar2 量化配置
│   ├── config_siluInU8_ptq.json
│   ├── config_siluInU8_ptq-preprocess.json
│   └── config_siluInU16_ptq.json            # RTMDet 参考模板
├── onnx/                                    # 待转换 ONNX（4 个尺寸/主干组合）
├── ddrnet_make_calib.py                     # 校准集生成
├── ddrnet_onnx_eval.py                      # ONNX FP32 评估
├── ddrnet_axera_eval.py                     # compiled.axmodel 上板评估
├── output_ddrnet23-slim-ptq/                # 转换产出（Binary 校准初版）
├── output_ddrnet23-slim-ptq-preprocess/     # 转换产出（Image 校准）
├── output_ddrnet23-slim-ptq-preprocess-npu3/# 转换产出（最终，上板使用）
├── onnx-vis/                                # ONNX 评估输出（pred/vis/meta.json）
├── axera_pred-slim/                         # 上板预测输出（pred/meta.json）
└── axera_pred-slim_vs_gt/                   # 上板预测 vs GT（metrics.*/混淆矩阵）
```
