# DDRNet Cityscapes 语义分割 — AX650 量化部署

将 DDRNet（DDRNet-23 / DDRNet-23-slim，Cityscapes 19 类）从 ONNX 经
Pulsar2 量化编译为 `compiled.axmodel`，并在 AX650A 上完成全量 val 精度验证。

## 1. 模型精度对比

Cityscapes val 全量 500 张，19 类，ignore_index=255；指标为官方口径的
mIoU / aAcc。

| 模型 | 精度模式 | mIoU | aAcc | ΔmIoU（量化 vs 浮点） | 平均推理耗时（ms） |
| --- | --- | --- | --- | --- | --- |
| DDRNet-23 | FP32 ONNX | 79.99 | 96.36 | / | / |
| DDRNet-23 | [W8A8](./config/config_u8_ptq-preprocess.json) | 78.76 | 96.09 | -1.23 | / |
| DDRNet-23 | [W8A16](./config/config_u16_ptq-preprocess.json) | 79.95 | 96.35 | -0.04 | / |
| DDRNet-23-slim | FP32 ONNX | 77.97 | 95.98 | / | / |
| DDRNet-23-slim | [W8A8](./config/config_u8_ptq-preprocess.json) | 77.79 | 95.92 | -0.18 | / |

## 2. 校准集准备

1）数据下载（Cityscapes 官方 zip 的 HuggingFace 镜像）：

```bash
mkdir -p ../dataset/cityscapes && cd ../dataset/cityscapes

# 来源：HuggingFace 镜像 mkhan91/Cityscapes（官方原始 zip，支持断点续传）
curl -sL -C - -o gtFine_trainvaltest.zip \
  "https://huggingface.co/datasets/mkhan91/Cityscapes/resolve/main/gtFine_trainvaltest.zip"
curl -sL -C - -o leftImg8bit.zip \
  "https://huggingface.co/datasets/mkhan91/Cityscapes/resolve/main/leftImg8bit.zip"

# 校验并解压（leftImg8bit 约 11.6GB，gtFine 约 241MB）
unzip -tq gtFine_trainvaltest.zip && unzip -tq leftImg8bit.zip
unzip -q gtFine_trainvaltest.zip && unzip -q leftImg8bit.zip
```

解压后得到 `leftImg8bit/{train,val,test}` 与 `gtFine/{train,val,test}`；
本校准集使用 `leftImg8bit/val` 500 张。（路径按本包部署布局
`../dataset/cityscapes` 书写；本机上本包嵌套在 `DDRNet/` 下，实际数据在
`../../dataset/cityscapes`，以下命令同理调整。）

2）生成校准集：

- 数据源：Cityscapes val 500 张（frankfurt / lindau / munster），
  部署布局下位于 `../dataset/cityscapes/leftImg8bit/val`。
- 工具：本包 `tools/make_calib.py`（自含预处理，不依赖其他脚本，仅需
  `opencv-python` + `numpy`），完成后自动归档为**扁平 tar**
  （条目 `./<stem>.png|bin`）。

```bash
# Image 校准（当前 U8/U16 配置实际使用）：resize 至 1024x2048 的 PNG
python tools/make_calib.py --img-dir ../dataset/cityscapes/leftImg8bit/val \
    --input-size 1024 2048 --save-images
```

默认输出并归档到 `../dataset/cityscapes/ddrnet_calib_<H>x<W>[-resize]-img|bin`
（`--img-dir` 按实际图像目录指定，脚本默认 `../dataset/cityscapes/val`）。
已生成的校准集：

| 校准集（`../dataset/cityscapes/`） | 内容 | 用途 |
| --- | --- | --- |
| `ddrnet_calib_1024x2048-resize-img(.tar)` | 500 x PNG，仅 resize（BGR 0-255） | 当前 `config_u8/u16_ptq-preprocess.json`（Image 校准，归一化在板上完成） |

## 3. 模型转换

工具链：[Pulsar2-modelScope](https://www.modelscope.cn/models/AXERA-TECH/Pulsar2/files) / [Pulsar2-HF](https://huggingface.co/AXERA-TECH/Pulsar2)，目标平台 **AX650A**，NPU3 模式。

```bash
# 在 Pulsar2 环境中、以本包目录为工作目录执行
pulsar2 build --config config/config_u8_ptq-preprocess.json    # U8
pulsar2 build --config config/config_u16_ptq-preprocess.json   # U16
```

## 4. 精度验证

流程：GT 转换 → ONNX FP32 评估 → axmodel 评估 → 预测对比（数据下载见第 2 节）。

1）GT 转换（labelIds → labelTrainIds）：镜像包不含
`*_gtFine_labelTrainIds.png`，需转换后才能算官方指标：

```bash
# 从 polygons 栅格化（边界像素更严格；依赖 pip install cityscapesscripts mmengine）
python tools/cityscapes.py ../dataset/cityscapes --nproc 32
```

校验：`leftImg8bit/val` 500 张图，`gtFine/val` 500 个
`*_gtFine_labelTrainIds.png`。评估时 `--img-dir` 指向图像目录
（下载布局为 `leftImg8bit/val`）、`--gt-dir` 指向`gtFine/val`。

2）ONNX FP32 评估 `eval_onnx.py`（依赖 onnxruntime）：

```bash
# 快速抽查（默认前 5 张，输出 onnx-vis/）
python eval_onnx.py

# 全量 val + GT 指标
python eval_onnx.py \
    --model onnx/ddrnet23_cityscapes-2048x1024.onnx \
    --img-dir ../dataset/cityscapes/leftImg8bit/val \
    --gt-dir ../dataset/cityscapes/gtFine/val \
    --limit 0 --device cpu
```

3）axmodel 评估 `eval_axera.py`（依赖 axengine，`AxEngineExecutionProvider`）：

```bash
# 默认：U16 量化模型，全量 500 张，输出 axera_pred-u16/
python eval_axera.py

# 指定其他 axmodel（如 U8 / slim）
python eval_axera.py \
    --model output_ddrnet23-ptq-preprocess-npu3/compiled.axmodel \
    --out-dir axera_pred --gt-dir ../dataset/cityscapes/gtFine/val
```

- 该脚本只做 resize + BGR→RGB，输入 **U8 NHWC**；mean/std 归一化由
  axmodel 内嵌 `input_processors` 自动完成，与量化的预处理保持一致。
- 输出：`<out-dir>/pred/<city>/*_pred.png`、`meta.json`（含平均耗时），
  有 GT 时另有 `metrics.json/csv`。

4）预测对比 `tools/compare.py`：将任意预测目录与 GT 复核，生成官方指标与
混淆矩阵（可选误差可视化）。适用于无 GT 时先跑推理、后统一对比的场景。

```bash
# 复核 axmodel U16 预测（输出 compare/axera_pred-u16_vs_gt/）
python tools/compare.py \
    --pred-dir axera_pred-u16/pred \
    --gt-dir ../dataset/cityscapes/gtFine/val \
    --ref-miou 79.99

# 可选：为前 5 对保存 (GT | 预测 | 误差) 三联图
python tools/compare.py \
    --pred-dir axera_pred-u16/pred \
    --gt-dir ../dataset/cityscapes/gtFine/val \
    --out-dir compare/axera_pred-u16_vs_gt --save-diff 5
```

- 输出：`metrics.json`、`metrics.csv`（逐类 IoU/Acc）、`confusion_matrix.csv`；
  `--save-diff N` 时另有 `diff/<city>/*_diff.png`（绿=正确，红=错误）。

指标为 19 类 IoU/Acc、mIoU、aAcc（ignore_index=255），与官方 Cityscapes
评测口径一致。

## 5. 参考项目

- **DDRNet**（TPAMI 2023）: *Deep Dual-Resolution Networks for Real-Time
  and Accurate Semantic Segmentation of Traffic Scenes*
  - 官方实现: <https://github.com/ydhongHIT/DDRNet>
- **MMSegmentation**（OpenMMLab）: DDRNet Cityscapes 训练配置与权重
 （模型由此工程导出，导出脚本不在本目录）
  - <https://github.com/open-mmlab/mmsegmentation>（`configs/ddrnet`）
- **AXERA Pulsar2** v7.0 量化编译工具链 + **axengine** 运行时（AX650A）
  - <https://pulsar2-docs.readthedocs.io/en/latest/pulsar2/introduction.html>
