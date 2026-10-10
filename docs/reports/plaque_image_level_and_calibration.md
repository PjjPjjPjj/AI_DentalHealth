# 牙菌斑：图级筛查口径 + Platt 校准接入（迭代 3）

日期：2026-09-03；模型：v2_sano（部署中）；新增：校准分数 + 图级聚合评估
结论先行：**两项均无需补充数据**，已用现有资产完成；图级聚合让业务口径的特异度大幅提升

---

## 1. Platt 校准已接入 pipeline

- v2 专属映射（v3c 的参数不可混用）：**logit(P') = 0.4695·logit(P) + 0.6801**
- calib 集 = Mendeley extended val（2,015）+ 503 SANO val 阴性（均未参与训练）
- **ECE 0.0402 → 0.0118**（isotonic 可到 0 但 Platt 可移植性更好）
- 代码：`src/inference/models.py` `PlaqueModel(calibration=...)`，向后兼容（不传=原行为）
- 配置：`configs/pipeline.yaml` `inference.plaque.calibration` 已指向
  `outputs/plaque_bin_v2_sano/calibration_platt.npz`；conf_threshold 换算为校准刻度
  **0.746**（≡ raw 0.70，部署行为不变，分数含义更规范）

## 2. 图级筛查评估（业务真实口径）

### 2.0 关键数据事实
**Mendeley val/test 的 base 图在图级全部为阳性**（每图≥1 个菌斑框，数据集设计如此）
→ 图级特异度**只能**在外部健康集上度量（当前：SANO 46 图；n 小，CI 宽）

### 2.1 聚合器对比（thr = val 阳性图分数 P5，规则性 Sens≥95%）

| 聚合器 | 图分定义 | thr | test Sens | SANO Spec（46 图） |
|---|---|---|---|---|
| max-P | 最差牙决定 | 0.952 | **0.980** | 0.978 |
| **mean-P（推荐）** | 全口平均 | 0.523 | 0.966 | **1.000**（CI95 下界 0.92） |
| top3-P | 前3牙均值 | 0.895 | 0.956 | 0.978 |

- **mean-P 完胜**：菌斑阳性口内有大量高分行拉高均值，而健康口整口低分——
  单牙误报被全口平均稀释，这正是筛查业务想要的性质
- SANO 误报图：max-P 1/46 → **mean-P 0/46**
- test Sens 0.966 vs max-P 0.980：损失 1.4pt 换取跨域零误报，筛查场景划算

### 2.2 固定阈值参考（max-P）
| thr | 0.5 | 0.7 | 0.9 |
|---|---|---|---|
| SANO Spec | 0.413 | 0.587 | 0.870 |

## 3. 产物

| 产物 | 路径 |
|---|---|
| 校准参数（v2） | `outputs/plaque_bin_v2_sano/calibration_platt.npz` |
| 图级评估 json/md | `outputs/plaque_image_level/eval_image_level.{json,md}` |
| 每图分数明细 | `outputs/plaque_image_level/image_scores.csv`（415 图 × 3 聚合器） |
| 评估脚本 | `scripts/eval_plaque_image_level.py` |

## 4. 待办（需要决策，不需要数据）

1. **图级判定进 pipeline JSON**：tooth 级已有 plaque.present；建议加 `image.plaque_screening`
   （mean-P + thr 0.523）——涉及 `src/inference/pipeline.py` + `schemas.py` 小改，等产品确认聚合器选择
2. SANO n=46 的 CI 偏宽：若需更硬的图级 Spec 结论，扩充外部健康集（如视频帧集中清晰度
   top 分位人工筛 100~200 张）——这是唯一"可选"的数据补充，非必需
3. 正畸托槽盲区不变（需子区域级标注协议升级才能解决）
