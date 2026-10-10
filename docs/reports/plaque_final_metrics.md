# 牙菌斑检测最终指标总表（收口版）

**部署配置**：plaque_bin_v2_sano（YOLO11s-cls @224）+ Platt 校准，**thr = 0.746**（校准刻度 ≡ raw 0.70）
训练数据：plaque_bin_v1 官方划分（患者级隔离）+ SANO 健康阴性 3,522 crop；v1→v2 低 lr 微调
日期：2026-09-03；所有数字可由 `outputs/` 下评估产物复现

---

## A. 框级（逐标注框/逐牙 crop）

### A1. Mendeley 官方 test（主口径，患者级隔离，与训练零重叠）

| 指标 | n | v1（旧部署 @0.92） | **v2（现部署 @0.746）** |
|---|---|---|---|
| AUROC（clean） | 2,528 | 0.8950 | **0.8958** |
| AP（clean） | 2,528 | 0.9967 | **0.9973** |
| Sens（extended） | 2,892 | 0.940 | **0.937** |
| Spec（extended，420 阴性） | 2,892 | 0.543 | **0.560** |
| Prec（extended） | 2,892 | 0.924 | **0.926** |
| F1（extended） | 2,892 | 0.932 | **0.932** |
| Spec（clean，56 阴性） | 2,528 | 0.554 | **0.571** |

### A2. Mendeley 官方 val（参考）

| 指标 | n | v2 @0.746 |
|---|---|---|
| Sens（extended） | 2,015（1,919+/96-） | 0.926 |
| Spec（extended） | 2,015 | 0.469 |
| Spec（clean，11 阴性） | 1,930 | 0.364（n 过小仅参考） |

### A3. SANO 健康牙 crop（外部阴性，跨域特异度）

| 指标 | n | v1 | **v2 @0.746** |
|---|---|---|---|
| Spec | 1,148（全阴性） | 0.348 | **0.978**（CI95 0.965~0.989，图级 bootstrap） |
| FP 数 | 1,148 | ~750 | **25** |
| P(plaque) 中位 | 1,148 | 0.984 | **0.023** |

### A4. 阈值画像（test extended，校准刻度）

| thr | 0.5 | 0.6 | **0.746** | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|
| Sens | 0.976 | 0.965 | **0.937** | 0.920 | 0.853 | 0.772 |
| Spec | 0.302 | 0.393 | **0.560** | 0.624 | 0.795 | 0.876 |

## B. 图级（整图筛查：牙齿分割→逐牙 crop→聚合）

| 口径 | 集合（图） | 阈值 | Sens | Spec |
|---|---|---|---|---|
| max-P（最差牙决定） | Mendeley val（157+） | 0.952ᵖ | 0.949 | —（无阴性图） |
| max-P | Mendeley test（205+） | 0.952ᵖ | **0.980** | — |
| max-P | SANO 健康（46-） | 0.952ᵖ | — | **0.978**（1 误报） |
| **mean-P（推荐筛查）** | Mendeley val（157+） | 0.523ᵖ | 0.949 | — |
| **mean-P（推荐筛查）** | Mendeley test（205+） | 0.523ᵖ | 0.966 | — |
| **mean-P（推荐筛查）** | SANO 健康（46-） | 0.523ᵖ | — | **1.000**（0 误报，CI95 下界 0.92） |

ᵖ 图级阈值 = Mendeley val 阳性图分数 P5 分位（规则性 Sens≥95%）；图级阈值独立于框级 0.746，
**不可混用**（校准分在 max 聚合下 0.746 只对应 SANO Spec 0.630）。

## C. 关键说明

1. **数据事实**：Mendeley val/test 图级全部阳性（每图≥1 菌斑框，数据集设计），图级 Spec 只能由外部集度量
2. **框级 Spec 0.56 已判定接近标注协议信息上限**（三轮实验证伪：托槽挖掘/推理归一化/曝光增广，
   详见 `plaque_exposure_experiments.md`）；420 阴性复审：~75% 明确干净、~4% 疑似错标
3. **校准**：Platt logit(P')=0.4695·logit(P)+0.6801，ECE 0.040→0.012；分数为跨域可比概率
4. 已知盲区：正畸托槽牙（语义冲突，需子区域级标注才能解决）
5. 回滚：`weights.plaque → outputs/plaque_bin_v1/weights/best.pt` + thr 0.92 + 删 calibration 行

## D. 产物索引

| 类别 | 路径 |
|---|---|
| 部署权重/校准 | `outputs/plaque_bin_v2_sano/weights/best.pt`、`calibration_platt.npz` |
| pipeline 配置 | `configs/pipeline.yaml`（inference.plaque + weights.plaque） |
| 框级评估 | `outputs/plaque_bin_v2_sano/{eval_v1_vs_v2,boxlevel_retest_calibrated}.{json,md}` |
| 图级评估 | `outputs/plaque_image_level/{eval_image_level.json,md,image_scores.csv}` |
| 曝光/挖掘实验（阴性结果） | `outputs/plaque_exposure_sweep/`、`outputs/plaque_bin_v3_mined/` |
| SANO 数据与审计 | `datasets/{raw,processed}/plaque_sano/`、`plaque_sano_dataset_audit.md` |
| 复核队列 | `annotation_queue/{sano_eval_review,test_neg_review,mined_neg_review}/` |
