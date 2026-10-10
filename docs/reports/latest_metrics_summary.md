# 全链路最新一次实验数据汇总（2026-09-02）

三种病灶均为各自独立 test 集的最新一次评测；数据源文件逐项核对无误。

## 1. 牙结石（最新最优配置 = fix7-FB，stage1-mask 兜底，已打包进 pipeline）

评测集：Oral-Mamba 系 test 700 张（患者级隔离），实例匹配 IoU≥0.5

**实例级**（官方冻结评测，对外口径 = final_fix7 config B）：
- Precision 0.4251 / Recall 0.5053 / **F1 0.4618**
- 分尺寸召回：small 0.4367 / medium 0.600 / **large 0.5714**
- TP Dice 0.817（检出即准）｜FP 0.274/图
- 打包 pipeline 复测：F1 0.4560（P 0.4204 / R 0.4982 / r_large 0.5714）；同口径 fix4 复测 F1 0.4525
  → promote 后 P/R/F1 同升、大病灶 +2 例，非以精度换召回

**图像级**（三流集成 A+B+C，thr 0.38）：
- Accuracy 0.8857 / Sensitivity 0.6959 / Specificity 0.9402
- Precision 0.7537 / F1 0.7254 / **AUROC 0.9151** / PR-AUC 0.7303

数据文件：`outputs/calculus_v5_fix7/final_fix7_metrics.json`（对外口径 B）、
`outputs/calculus_pipeline_eval/pipeline_calculus_eval.json`（打包复测）、
`outputs/reports/calculus_v5_fix7.md`（实验/promote 记录）

## 2. 牙菌斑（plaque_bin_v1 框级二分类，YOLO11s-cls@224）

评测集：Mendeley 官方患者级 split；test 干净 base 框 2,472 阳 / 56 阴（扩展阴 420）
阈值 thr=0.92（val 上 Sens≥0.95 约束下 Spec 最大，筛查优先）

- **AUROC：val 0.8676 / test 0.8950**（AP 0.9967）
- thr 0.92 下 test：Sens 0.9397 / Spec 0.5536 / Precision 0.9894 / F1 0.9639
- 接入牙齿分割的牙位 crop 路径：**AUROC 0.9164**（val，n=282 配对）——高于 GT 框直推
- 已知限制：阴性标注稀缺（test 56 个），Spec 置信区间宽

数据文件：`outputs/plaque_bin_v1/eval_plaque_bin.json`、
`outputs/pipeline_v2_validation/plaque_integration_loss.json`

## 3. 龋齿（C2 = 合并数据 YOLO11m@960）

评测集：6313 Intraoral test，conf=0.55（val 冻结），IoU≥0.5

- **Precision 0.8722 / Recall 0.9028 / F1 0.8872**
- mAP50 0.9243 / mAP50-95 0.7158
- pipeline 路径回归验证：逐位复现 0.8872（零整合损失）

数据文件：`outputs/caries_c2/eval_c2.json`、
`outputs/pipeline_v2_validation/caries_regression.json`

## 4. Pipeline 支撑组件（非病灶）

- 牙齿实例分割（IOP-Compass 11m-seg@896，官方 test 250 图/4449 牙）：
  Box P 0.984 / R 0.983 / mAP50 0.99 / mAP50-95 0.959；Mask mAP50 0.9899 / mAP50-95 0.9452
- 跨域定性：caries6313 / Oral-Mamba / Mendeley（含正畸）逐牙分割正确
- 统一 JSON schema 校验零违例；30/30 单元测试通过

## 5. 一句话结论

龋齿（F1 0.887）> 牙菌斑（AUROC 0.90，Spec 受数据限制）> 牙结石（实例 F1 0.462，
图像级 AUROC 0.915）。牙结石等新数据；fix7-FB（stage1-mask 兜底）已把大病灶 r_large
0.50→0.571 且精度无损；剩余大病灶瓶颈收敛到 stage1 漏框（特写 film 型 + 标签噪声），
需真实数据/标注修正，阈值与边界形态学均已证伪。
