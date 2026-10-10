# 第三方组件与数据集声明（THIRD-PARTY NOTICES）

本文件是本仓库的一部分，依据 AGPL-3.0 第 5(d) 条列出全部第三方依赖与数据来源。

> **本仓库不包含任何原始数据集、患者影像或模型权重文件。**
> 所有数据与权重均需使用者自行从原始出处获取，并自行承担其许可与合规责任。

---

## 1. 第三方软件依赖

### 1.1 Ultralytics YOLO11 —— 决定本仓库许可证的关键依赖

| 项 | 内容 |
|---|---|
| 名称 | Ultralytics YOLO11 / `ultralytics` Python 包 |
| 版本 | 8.4.129（开发环境） |
| 许可证 | **AGPL-3.0**（作者 Ultralytics） |
| 主页 | https://github.com/ultralytics/ultralytics |
| 版权 | Copyright (C) 2018-2025 Ultralytics. All Rights Reserved. |

**为什么本仓库必须用 AGPL-3.0：**
Ultralytics YOLO11 采用 AGPL-3.0，其第 13 条明确要求：用户通过网络与软件交互时，
必须向这些交互用户提供对应源码。本项目的**训练代码、推理代码、模型权重**
均直接链接或派生自 YOLO11，因此在对外分发时，本项目整体适用 AGPL-3.0。
若你希望改用 MIT / Apache-2.0 等宽松协议，必须先取得
[Ultralytics Enterprise 商业许可](https://www.ultralytics.com/license)。

> 完整协议文本见本仓库 `LICENSE` 文件。Ultralytics 亦提供
> [开源许可说明](https://docs.ultralytics.com/license/)。

### 1.2 其他依赖

| 组件 | 许可证 | 用途 | 主页 |
|---|---|---|---|
| PyTorch | BSD-3-Clause | 深度学习框架 | https://github.com/pytorch/pytorch |
| torchvision | BSD-3-Clause | 视觉算子与预训练骨干 | https://github.com/pytorch/vision |
| OpenCV | Apache-2.0 | 图像处理 | https://opencv.org |
| NumPy | BSD-3-Clause | 数值计算 | https://numpy.org |
| pandas | BSD-3-Clause | 表格数据处理 | https://pandas.pydata.org |
| scikit-learn | BSD-3-Clause | 指标与评估 | https://scikit-learn.org |
| Matplotlib | PSF-based (Matplotlib License) | 绘图 | https://matplotlib.org |
| PyYAML | MIT | 配置解析 | https://pyyaml.org |
| tqdm | MPL-2.0 / MIT | 进度条 | https://tqdm.github.io |
| Pillow | MIT-CMU | 图像 IO | https://python-pillow.org |
| ImageHash | BSD-2-Clause | 感知哈希去重 | https://github.com/JohannesBuchner/ImageHash |
| onnxruntime | MIT | 桌面端 ONNX 推理 | https://onnxruntime.ai |
| PySide6 | LGPL-3.0 | GUI（Windows 默认） | https://doc.qt.io/qtforpython/ |
| PyQt5 | GPL-3.0 / 商业双许可 | GUI 回退（Linux/macOS） | https://www.riverbankcomputing.com |
| reportlab | BSD-3-Clause | PDF 报告生成 | https://www.reportlab.com |
| PyInstaller | GPL-2.0（附例外条款） | 打包为 Windows exe | https://pyinstaller.org |

> `app/qt_compat.py` 的设计是 PySide6 优先、PyQt5 回退。
> 若你将本项目整体闭源分发，请注意 **PyQt5 本身是 GPL-3.0**，
> 建议改用 PySide6（LGPL-3.0）以降低义务。

---

## 2. 字体

本项目的中文图表与报告使用 **Noto Sans SC / Noto Sans CJK SC**。

| 项 | 内容 |
|---|---|
| 名称 | Noto Sans SC (Source Han Sans SC) |
| 版权 | Copyright 2014-2021 Adobe (http://www.adobe.com/)，以 Noto 项目名义发布 |
| 许可证 | **SIL Open Font License 1.1**（OFL-1.1） |
| 完整协议 | https://scripts.sil.org/OFL |
| 主页 | https://github.com/notofonts/noto-cjk |

**OFL-1.1 关键义务（本项目已履行）：**
字体文件**未随本仓库分发**（`assets/fonts/` 已在 `.gitignore` 中排除）。
使用者需自行从 Noto 官方仓库下载。若你选择分发字体文件本身，
必须：(a) 附带 OFL-1.1 全文；(b) 保留版权与许可声明；(c) 不得单独售卖字体；
(d) 不得使用 Reserved Font Name。

字体可转换为其他格式（如本项目在桌面端所做的 TTF 转换），
但转换后的文件仍受 OFL-1.1 约束，且**不得**使用 OFL 保留名称。

---

## 3. 训练数据集来源与归因

以下数据集**均未包含在本仓库中**。本节仅记录训练过程中使用过的来源，
以满足学术归因要求并提醒使用者核查许可。

> ⚠️ **重要**：下表中标注 `许可待确认` 的数据集，其再分发条款并不明确。
> 即使你已在本地获得数据，**也不代表你可以重新分发它**。
> 公开仓库中请勿上传这些数据集的任何子集。

| 数据集 | 用于任务 | 来源 | 许可状态 | 备注 |
|---|---|---|---|---|
| **IOP-Compass** | 牙齿实例分割 | IOP-Compass 发布包（1,000 患者 × 5 视角） | 许可待确认 | 患者级隔离切分 |
| **Oral-Mamba** | 牙结石病灶分割 | Oral-Mamba 发布包 | 许可待确认 | mask 语义见 `docs/reports/mask_semantics.md` |
| **Mendeley 多模态牙菌斑集**<br>*"A new multi-modal dataset for Dental Plaque Diagnosis"* | 牙菌斑检测 | Mendeley Data | 学术数据集，含 `legal_documents/`（306 份伦理批件 + 知情同意）与 `clinical_records/`（148 份） | **合规链完整**；含患者临床记录，禁止外传 |
| **SANO 集** | 菌斑阴性样本 | 独立临床摄影集（Canon EOS R 系列） | 许可待确认 | 意大利语 "SANO" = 健康；全部为健康牙面 |
| **6313 Intraoral Caries** | 龋齿检测 | Zenodo record 14827784 | 见 Zenodo 记录 | 1.5 GB，见 `scripts/dl_zenodo_parallel.py` |
| **Lesi_Intraoral** | 跨域验证 | 发布包 | 许可待确认 | Calculus / Caries / Gingivitis 三类 |
| **EKA 牙龈炎** | 跨域验证 | 发布包（6.52 GB，1,090 图） | 许可待确认 | 高分辨率 7286 px |
| **"Teeth or Dental Image Dataset"** | 硬负样本挖掘 | 疑似 Kaggle 同名集 | **无 README / 无 license / 无伦理文件** | 视频抽帧来源；出处存疑 |
| **Kaggle 口腔疾病 7 类集** | 外部鲁棒性测试 | Kaggle archive | 许可待确认 | 与现有训练集 md5 重叠 4,708 张 |
| **Roboflow 集**（tzg18 / rc8qm） | 牙结石 / 龋齿补充 | Roboflow Universe | 依各数据集页面条款 | 社区标注数据 |
| **AIRC-LABDEN** | 菌斑区域分类 | 发布包 | 许可待确认 | 记录为未获取（BLOCKED_EXTERNAL） |
| **IO150K / TeethSEG** | 牙齿分割（备选） | 发布包 | 许可待确认 | 未进入 V1 训练 |

各数据集的详细审计结论（规模、画质、重复率、与本项目数据的重叠情况）见
[`docs/reports/`](docs/reports/)。

### 3.1 数据合规声明

1. **患者隐私**：所用数据集包含可识别的临床口腔影像。本项目**已主动从公开仓库中
   移除全部患者影像**（`demo_images/`、报告中的照片墙与蒙太奇图均未上传）。
2. **伦理批件**：Mendeley 牙菌斑数据集自带 306 份伦理批件与知情同意文件，
   说明原始采集已获伦理批准。**但伦理批准不等于允许再分发。**
3. **标识符处理**：`docs/reports/` 中出现的数据集内部编号（如 `patient0058`）为
   公开数据集的原始命名，已移除其中的拍摄日期字段，仅用于复现说明，
   不含姓名、住院号、联系方式等直接身份标识。
4. **数据许可需你自行确认**：本节记录的是"本项目训练时使用了什么"，
   **不代表"你可以使用或再分发"**。使用前请逐一查阅原始发布页面的许可条款。

---

## 4. 模型权重

本仓库**不包含任何模型权重**。权重通过 GitHub Releases 分发，
在 Release 说明中适用与本仓库相同的 AGPL-3.0 协议。

| 任务 | 架构 | 底模来源 |
|---|---|---|
| 牙齿实例分割 | YOLO11m-seg | Ultralytics COCO 预训练权重（AGPL-3.0） |
| 牙结石检测（级联两级） | YOLO11m-seg | 同上 |
| 龋齿检测 | YOLO11m（det） | 同上 |
| 牙菌斑分类 | YOLO11s-cls | 同上 |
| 牙结石验证器 / 图像分类 | EfficientNetV2 / 自建 | ImageNet 预训练（各自许可） |

> 预训练权重继承 Ultralytics / 原发布方的许可条款，分发时须一并遵守。

---

## 5. 本项目自身的版权

源码版权归原作者所有，以 **AGPL-3.0-or-later** 授权。
全部 237 个 Python 源文件顶部带有 SPDX 标识：

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 <版权持有人>
# This file is part of OralAI-DentalHealth.
```

---

## 6. 医学免责声明

本项目输出为 **AI 辅助筛查 / 病灶候选检测**，
**不是医学诊断，不是治疗建议，不能替代执业牙医的临床判断**。

模型未通过任何医疗器械注册或前瞻性临床验证，禁止直接用于患者诊疗决策。
详见仓库根目录 `README.md` 的"免责声明"章节。
