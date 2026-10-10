# OralAI-DentalHealth

基于普通 RGB 口内照片的**牙齿级口腔健康 AI 筛查系统**。
用 4 个独立模型在单张照片上完成**牙齿分割 → 牙结石 / 龋齿 / 牙菌斑检测 → 病灶与牙齿关联 → 结构化 JSON**。

> ### ⚠️ 免责声明（请先阅读）
>
> 本项目输出为 **AI 辅助筛查 / 病灶候选检测**，
> **不是医学诊断，不是治疗建议，不能替代执业牙医的临床判断。**
>
> 本项目**未通过任何医疗器械注册，未做前瞻性临床验证**，
> 禁止直接用于患者诊疗决策。研究与教学用途。

---

## 目录

- [1. 能做什么](#1-能做什么)
- [2. 性能指标](#2-性能指标)
- [3. 快速开始](#3-快速开始)
- [4. 模型权重下载](#4-模型权重下载)
- [5. 数据集准备](#5-数据集准备)
- [6. 目录结构](#6-目录结构)
- [7. 训练与评估](#7-训练与评估)
- [8. 桌面应用](#8-桌面应用)
- [9. 输出 JSON 格式](#9-输出-json-格式)
- [10. 已知限制](#10-已知限制)
- [11. 许可证与版权](#11-许可证与版权)
- [12. 引用](#12-引用)

---

## 1. 能做什么

| 任务 | 模型 | 输入 | 输出语义 |
|---|---|---|---|
| 图像质量门控 | 规则式 | 整图 | 可用性 + 模糊/过曝/高光比 |
| 牙齿实例分割 | YOLO11m-seg @896 | 整图 | 每颗牙 mask + 局部 ID |
| **牙结石**病灶分割 | YOLO11m-seg 两级级联 + 验证器 | 整图 | 病灶 mask + 实例级置信度 |
| **龋齿**检测 | YOLO11m det @960 | 整图 | 病灶 bbox |
| **牙菌斑**逐牙分类 | YOLO11s-cls @224 | 逐牙 crop | P(plaque) + 判定 |
| 病灶-牙齿关联 | 几何 | — | 每个病灶归属到具体牙齿 |

推理链路：

```
质量门控 → 牙齿分割 → 牙结石级联 → 龋齿检测 → 菌斑逐牙分类 → 病灶关联 → 统一 JSON
```

**关键设计：训练与推理使用完全相同的裁剪协议**（分割 → 整牙 crop → 分类），
消除 train/serve skew。实测该路径的判别力优于直接用标注框裁剪
（AUROC 0.916 vs 0.868）。

---

## 2. 性能指标

均为**患者级隔离**的官方 test 集结果，非训练集拟合。

| 任务 | 指标 | 评测集 |
|---|---|---|
| 牙齿实例分割 | Box P 0.984 / R 0.983 / mAP50 0.990<br>Mask mAP50 0.990 | IOP-Compass test 250 图 / 4,449 颗牙<br>**4,373 颗正确检出（98.3%）** |
| 龋齿检测 | **F1 0.887**（P 0.872 / R 0.903）<br>mAP50 0.924 | 6313 Intraoral test |
| 牙菌斑（逐牙） | **AUROC 0.916**（牙位 crop 路径） | Mendeley 官方患者级 test |
| 牙结石（实例级） | F1 0.462（P 0.425 / R 0.505）<br>large-lesion R 0.571 | Oral-Mamba 系 test 700 图 |
| 牙结石（图像级筛查） | **AUROC 0.915** / Sens 0.696 / Spec 0.940 | 同上 |

> 牙结石为四类中表现最弱的一项，属小目标 + 标签噪声问题，
> 瓶颈已收敛到 stage1 漏框而非阈值。详见 [`docs/reports/`](docs/reports/)。

---

## 3. 快速开始

### 环境要求

- Python 3.10 ~ 3.11
- 训练：PyTorch + CUDA（V100 级别；默认 AMP/FP16）
- 仅推理：只需 `onnxruntime`，**无需 PyTorch / CUDA**

### 安装

```bash
git clone https://github.com/PjjPjjPjj/AI_DentalHealth.git
cd OralAI-DentalHealth

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 验证管线（无需权重）

```bash
# 干跑，验证管线结构与 JSON schema
python demo.py --dry-run --image <你的口内照片.jpg>
```

### 单元测试

```bash
pip install pytest imagehash
python -m pytest tests/ -q
```

---

## 4. 模型权重下载

**本仓库不包含权重**（`.pt` / `.onnx` 体积过大，且需与源码同协议分发）。

权重通过 GitHub Releases 提供，见
[Releases 页面](https://github.com/PjjPjjPjj/AI_DentalHealth/releases)。

下载后放到仓库根目录（路径与 `configs/pipeline.yaml` 对应）：

```text
outputs/iop_toothseg/weights/best.pt          牙齿分割
outputs/calculus_v5_stage1/weights/best.pt    牙结石 stage1
outputs/calculus_v3b_stage2/weights/best.pt   牙结石 stage2
outputs/calculus_v5_fix7/artifacts/           牙结石验证器 + 融合参数
outputs/caries_c2/weights/best.pt             龋齿
outputs/plaque_bin_v2_sano/weights/best.pt    牙菌斑
```

桌面端另需 ONNX 版本，放到 `app/models/`：

```bash
python app/export_onnx.py     # 在装有 torch + ultralytics 的环境中运行
```

### 自行训练

不想下载权重也可以自己训练——见 [第 7 节](#7-训练与评估)：

```bash
./scripts/train_caries.sh     # 建议从 Phase 2 开始，先验证训练环境
./scripts/train_calculus.sh
./scripts/train_plaque.sh
./scripts/train_tooth.sh
```

---

## 5. 数据集准备

**本仓库不含任何数据集。** 所有数据需自行从原始出处获取，
并自行确认其许可条款。完整来源与合规状态见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 第 3 节。

数据放置位置（`datasets/raw/` 永远只读，代码不会修改）：

```text
datasets/raw/
├── IOP_Compass/       牙齿实例分割
├── oral_mamba/        牙结石病灶 mask
├── Plaque/            牙菌斑（Mendeley 多模态集）
├── caries6313/        龋齿（Zenodo record 14827784）
├── Lesi_Intraoral/    跨域验证
└── eka_gingivitis_hires/  跨域验证
```

数据审计（**Gate 1：审计未通过禁止进入训练**）：

```bash
./scripts/audit_all.sh
# 输出：outputs/reports/data_audit_report.{json,md}
```

审计涵盖：图片可读性、标签合法性、bbox/mask 越界、类别统计、
重复检测（md5 + pHash）、split 完整性、**患者级泄漏**。

---

## 6. 目录结构

```text
.
├── configs/              任务配置 + pipeline.yaml（部署唯一入口）
├── src/
│   ├── data/             审计、split、数据转换器
│   ├── train/            4 个训练入口 + 公共训练核心
│   ├── eval/             4 个评估 + tooth-level 指标
│   ├── inference/        质量检查、模型封装、关联、Pipeline、Schema
│   ├── annotation/       SAM2 辅助 + 导出 + 审核队列
│   └── utils/            logger / config / reproducibility / metrics / paths
├── scripts/              174 个数据构建、训练、评估、审计脚本
├── tests/                31 个单元测试
├── docs/reports/         48 份实验与审计报告（已脱敏）
├── app/                  Windows 桌面应用（PySide6 + ONNX 运行时）
├── demo.py               单图全管线推理
├── demo_plaque.py        菌斑专项 demo
└── requirements.txt
```

---

## 7. 训练与评估

### 训练

```bash
./scripts/train_caries.sh     # 龋齿
./scripts/train_calculus.sh   # 牙结石
./scripts/train_plaque.sh     # 牙菌斑
./scripts/train_tooth.sh      # 牙齿分割
```

所有超参在 `configs/*.yaml`；每次训练自动记录到 `outputs/experiments.csv`。
默认 `imgsz=640`（plaque=224）、`epochs=100`、`patience=20`、`seed=42`。

> Ultralytics 的数据集 `path` 按**当前工作目录**解析，
> 请务必在仓库根目录执行训练命令。

### 评估

```bash
python -m src.eval.evaluate_caries
python -m src.eval.evaluate_calculus
python -m src.eval.evaluate_plaque
python -m src.eval.evaluate_tooth
```

指标口径：检测/分割报 P / R / mAP50 / mAP50-95，分割另算 Dice / IoU；
分类**以 Sensitivity 优先于 Accuracy**，并报 Spec / F1 / AUROC。

### 单图推理

```bash
python demo.py --image <image.jpg>
# 输出：outputs/demo/<name>_result.json + <name>_visualization.jpg
```

---

## 8. 桌面应用

`app/` 是独立的 Windows 桌面程序：输入照片 → 推理 → 可视化 →
风险评级 → 干预建议 → HTML/PDF 报告。运行时**只依赖 onnxruntime**，
无需 PyTorch / CUDA。

```bash
pip install -r app/requirements_app.txt
python -m app.main                    # GUI
python -m app.headless --fast <img>   # 无界面批处理
```

打包为 Windows exe：

```bash
pip install pyinstaller
pyinstaller app/DentalAI.spec --noconfirm    # -> dist/DentalAI/DentalAI.exe
```

详见 [`app/README.md`](app/README.md)。

---

## 9. 输出 JSON 格式

```json
{
  "image_id": "test_0001.jpg",
  "quality": {"usable": true, "blur_score": 0.0, "brightness": 0.0, "highlight_ratio": 0.0},
  "teeth": [
    {
      "tooth_id": 1,
      "bbox": [120, 82, 246, 381],
      "plaque":   {"score": 0.50, "confidence": 0.85},
      "calculus": {"present": true, "confidence": 0.88, "area_ratio": 0.061},
      "caries":   {"present": false, "confidence": 0.09}
    }
  ],
  "unassigned_lesions": []
}
```

> `tooth_id` 是**图内局部序号（1..N）**，**不是 FDI 牙位编号**。
> V1 明确不实现 FDI 自动编号，避免给出错误临床信息。

---

## 10. 已知限制

1. **牙结石实例级 F1 仅 0.462** —— 远低于其他三类，是当前主要短板。
2. **菌斑特异度受限** —— test 集干净阴性仅 56 个框，Spec 置信区间宽。
   0.746 部署档 test Sens 0.792 / Spec 0.978。
3. **正畸托槽是已知盲区** —— 分割在正畸图上定性通过，但托槽遮挡区域未专项训练。
4. **牙结石 vs 色素沉着、龋齿 vs 窝沟染色** —— 混淆对需真实数据进一步分析。
5. **未做跨数据集量化验证** —— 跨域目前仅定性检查。
6. **无 FDI 牙位编号**，`tooth_id` 仅为局部序号。
7. **模型未做临床验证**，禁止用于诊疗决策。

---

## 11. 许可证与版权

### 本项目

**AGPL-3.0-or-later** —— 完整协议见 [`LICENSE`](LICENSE)。

本项目基于 Ultralytics YOLO11 构建。YOLO11 采用 **AGPL-3.0**，
其第 13 条要求网络服务场景下必须向使用者提供源码。
因此本项目整体适用 AGPL-3.0，**包括随附的模型权重**。

> 若你希望采用 MIT / Apache-2.0 等宽松协议对外分发，
> 必须先取得 [Ultralytics Enterprise 商业许可](https://www.ultralytics.com/license)。

### 第三方组件

全部依赖的许可证、数据集来源与合规状态见
**[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)**。要点：

- **数据集未包含在本仓库中** —— 需自行获取并自行确认许可
- **患者影像未包含在本仓库中** —— 隐私保护，已主动移除全部影像
- **模型权重未包含在本仓库中** —— 通过 Releases 分发
- **字体（Noto Sans SC）未包含** —— SIL OFL-1.1，需自行下载

### 源码头

全部 237 个 Python 源文件带 SPDX 标识：

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 <版权持有人>
```

---

## 12. 引用

如果你在研究中使用了本项目，请引用：

```bibtex
@software{oralai_dentalhealth_2026,
  title  = {OralAI-DentalHealth: Tooth-level Oral Lesion Screening from RGB Intraoral Photos},
  author = {Pengjunjie},
  year   = {2026},
  url    = {https://github.com/PjjPjjPjj/AI_DentalHealth}
}
```

使用本项目的指标或数据集审计结果时，请同时引用原始数据集的发表论文。
