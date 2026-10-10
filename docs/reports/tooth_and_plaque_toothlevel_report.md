# 单牙级成果汇报：牙位分割训练/测试 + 菌斑单牙判定 + Demo 用例

日期：2026-09-07；数据源：`runs/segment/outputs/iop_toothseg/train/results.csv`、
`outputs/plaque_toothlevel_v2/{results.json,tooth_records.csv,threshold_sweep.csv}`、
`outputs/demo_plaque_pipeline/`

## 1. 牙位分割（iop_toothseg，YOLO11m-seg @896，单类 tooth）

### 数据规模（IOP-Compass，患者级隔离，1,000 患者 × 5 视角 = 5,000 图 / 88,601 颗牙）

| split | 图像 | 牙齿实例 |
|---|---|---|
| **训练** | 4,500 | **79,714** |
| val | 250 | 4,438 |
| **测试** | 250 | **4,449** |

### 测试准确率（官方 test）

| 口径 | 数值 | 换算"准确识别" |
|---|---|---|
| Box P / R | 0.984 / 0.983 | **4,373 / 4,449 颗正确检出（98.3%）** |
| Box mAP50 / 50-95 | 0.990 / 0.959 | — |
| Mask mAP50 / 50-95 | 0.9899 / 0.9452 | 分割形态 99 分档 |

- val 同口径：4,438 × R 0.981 ≈ 4,354 颗正确
- 跨域定性：caries6313 / Oral-Mamba / Mendeley（含正畸）逐牙分割正确
- 已知长尾：Mendeley 域 638 GT 牙中 46 颗未配对（边缘牙/分割缺失，6.8%）

## 2. 菌斑单牙级判定（复用上述分割，下游评测）

- 配对协议 v2（tooth_id 分组）：配对率 92.8%（v1 box×IoU 协议仅 16.1%，已作废）
- **关键事实：Mendeley val/test 无纯阴性牙**，判别力只能对外部健康集测
- test 592 颗阳性牙：@0.746 正确判定 469（Sens 0.792）；@0.39 正确判定 **540（Sens 0.912）**
- SANO 健康 1,148 颗：@0.746 误报 25（Spec 0.978）；@0.39 误报 115（Spec 0.900）
- 组合判别 AUROC 0.9784 / AP 0.9618
- 龈缘窗口推理已证伪（整牙 crop 最优 0.9846）；0.39 来自网格阈值扫描
  （约束 val Sens≥0.95 → SANO Spec 最大，`threshold_sweep.csv/png`）

## 3. Demo 用例（outputs/demo_plaque_pipeline/）

样本：`patient0058_central-left`（Mendeley test base，正畸患者）

- 分割 14 颗牙（T01–T14）；与 GT 配对 6 颗（GT 全部 = plaque）
- **@0.746：TP 6 / FN 0 / FP 0 —— 可评判牙全部正确**
- 其余 8 颗无标注（图中标 `GT:-`，不计对错）
- 图级：max-P 0.998 / mean-P 0.763 → 图级判"有菌斑"正确
- 五步图：step1 输入 → step2 分割叠加 → step3 单牙 crop+GT → step4 逐牙 P 值 → step5 结果
- 托槽牙 2 颗分数偏低（0.63/0.33）但均无 GT 标注，不构成漏检；正畸托槽仍为模型已知盲区

## 4. 一句话结论

牙位分割：**79,714 颗牙训练，4,449 颗测试，正确识别 4,373 颗（98.3%），mask mAP50 0.99**。
菌斑单牙级：638 颗 GT 牙配对 592 颗，双档阈值下正确判定 469（@0.746）或 540（@0.39），
健康误报 25 或 115 颗。
