# Dental AI Assistant

口腔AI多病灶智能检测系统（Windows 桌面软件）。输入一张口腔照片，自动完成
**牙菌斑 / 牙结石 / 龋齿** 三类病灶检测 → 结果可视化 → 风险评级 → 干预建议 →
检测报告（HTML + PDF），并提供一键 **Demo 展示模式**。

```
口腔图片输入 → 图像预处理/质量门 → AI推理(ONNX) → 牙菌斑/牙结石/龋齿检测
→ 结果可视化(蓝/黄/红框+mask) → 风险评级 + Oral Health Score → 干预建议 → 报告
```

## 模型（复用本工程最佳生产模型，已导出 ONNX）

| 任务 | 模型 | 权重来源 | 关键指标 | 推理配置 |
|---|---|---|---|---|
| 牙齿分割（前置） | YOLO11m-seg @896 | `outputs/iop_toothseg` | mask mAP50 0.990 | conf 0.30 / iou 0.70 |
| 牙结石 | v5 两级级联 S1+S2+EffNetV2 verifier+M3 融合+FB | `outputs/calculus_v5_fix7` | 实例 F1 0.462 / 图像 AUROC 0.915 | conf 0.005 → 0.05 |
| 龋齿 | YOLO11m det @960 | `outputs/caries_c2` | test F1 0.887 | conf 0.55 / iou 0.60 |
| 牙菌斑 | YOLO11s-cls @224（整牙 crop 二分类） | `outputs/plaque_bin_v1` | test AUROC 0.895 | 阈值 0.92 |

运行时只依赖 **onnxruntime**（无需 PyTorch / ultralytics / CUDA）。
CUDA 可选：设环境变量 `DENTALAI_EP=cuda` 自动启用（需安装 onnxruntime-gpu）。

## 目录结构

```
app/
├── main.py            # GUI 入口（python -m app.main）
├── headless.py        # 无界面入口（CLI/测试共用管线）
├── export_onnx.py     # 训练权重 → ONNX 导出（在训练环境运行一次）
├── qt_compat.py       # PySide6 优先、PyQt5 回退
├── core/              # 推理核心（GUI 无关，可独立测试）
│   ├── backend.py     #   letterbox / NMS / mask 后处理（与 ultralytics 对齐）
│   ├── models_onnx.py #   四模型封装 + 牙结石两级级联（v5-fix7/FB 完整复刻）
│   ├── pipeline.py    #   统一管线：质量门→牙齿→牙结石→龋齿→菌斑→关联
│   ├── risk.py        #   风险评级 + 评分 + 干预建议
│   ├── visualize.py   #   蓝(菌斑)/黄(牙石)/红(龋齿) 框 + 半透明 mask
│   └── report.py      #   HTML + PDF 报告（内嵌 NotoSansSC）
├── gui/               # PySide6/PyQt5 界面（主窗口 / Demo / 推理线程）
├── models/            # *.onnx + calculus_fusion.npz（打包收集）
├── fonts/             # NotoSansSC.ttf（由 assets/fonts 的 OTF 转换）
├── demo/              # Demo 四案例图片
├── DentalAI.spec      # PyInstaller 打包配置
└── requirements_app.txt
```

## 本机运行（开发）

```bash
python -m venv appenv
appenv/bin/pip install -r app/requirements_app.txt pytest
# GUI（需要显示器；无显示器可 QT_QPA_PLATFORM=offscreen）
PYTHONPATH=. appenv/bin/python -m app.main
# 无界面批处理
PYTHONPATH=. appenv/bin/python -m app.headless --fast demo_images/intraoral_163.jpg
```

结果输出到 `results/`：

```
results/
├── result_image/   xxx_detected.jpg        # AI 标注图
├── reports/        xxx_report.html/.pdf/.json
└── demo/           demo 案例标注图
```

## Windows 打包（生成 DentalAI.exe）

在 Windows 机器（Python 3.10）上：

```bat
python -m venv venv && venv\Scripts\activate
pip install -r app\requirements_app.txt pyinstaller
pyinstaller app\DentalAI.spec --noconfirm
```

产物 `dist/DentalAI/` 整个目录免安装分发：双击 `DentalAI.exe` 即用，
用户无需 Python / CUDA / PyTorch。`--headless` 参数可复用同一 exe 做批处理。

## 性能（实测：10 核服务器 CPU / onnxruntime CPU EP）

| 模式 | 612×459 图 | 说明 |
|---|---|---|
| 标准模式 | ~18 s | 与训练侧生产管线**逐位一致**（含 S2 翻转 TTA、2×2 切片、FB 兜底） |
| 快速模式 | ~7 s | 跳过 TTA/切片（设置里可勾选），灵敏度略降 |
| GPU (CUDA EP) | <2 s | `DENTALAI_EP=cuda` |

普通 i5 桌面 CPU 建议勾选快速模式（约 3-6 s/图）。进一步压到 CPU<5s 的
后续选项：静态 int8 量化（需重新验证级联一致性）或捆绑 onnxruntime-gpu。

## 与生产管线的等价性验证

`scripts/verify_onnx_consistency.py` 三阶段对比（torch 基准 / ONNX / compare）：

- 牙齿分割：实例数、box IoU=1.0000、conf 差 0.0000（4 个域逐位一致）
- 龋齿检测：检出数一致
- 牙菌斑：P(plaque) 差 0.0000
- 牙结石级联：检出实例 box IoU≥0.88、mask IoU≥0.91；少数阈值边缘实例因
  OpenCV 4.x/5.x 插值噪声存在 1 个实例级差异（标准模式下）

ONNX 导出注意：det/seg 用 `dynamic=True`（ultralytics 预处理是 auto
letterbox 最小矩形）；cls 模型导出自带 softmax，不能二次 softmax；
预处理 `letterbox` 后必须转 RGB（ultralytics `im[..., ::-1]`）。

## 测试

```bash
# 桌面应用侧（appenv）
PYTHONPATH=. QT_QPA_PLATFORM=offscreen appenv/bin/pytest tests/test_desktop_app.py -v
# 训练工程侧（训练环境，自动跳过桌面用例）
pytest tests/ -q
```
