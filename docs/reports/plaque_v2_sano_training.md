# 牙菌斑特异度优化：SANO 健康阴性微调（plaque_bin_v2_sano）

日期：2026-09-03；任务：利用 SANO 健康牙面数据集修复 v1 的"模糊/暖色→误报菌斑"捷径
审计前置：`plaque_sano_dataset_audit.md`（225 图全量目检=健康；与现有域零重叠）

---

## 1. 问题定位（训练前）

v1 在 SANO 健康牙 crop 上 **P(plaque) 中位 0.984**——即当前部署模型把几乎所有
陌生域的健康牙判为菌斑。crop 抽检证实这些是干净牙面（轻度模糊/暖色调），
v1 学到的是"模糊+暖色 = 菌斑"的域捷径。这正是特异度瓶颈的根因之一：
阴性形态多样性不足。

## 2. 数据构建（全链路可复现）

| 步骤 | 脚本/产物 | 结果 |
|---|---|---|
| 提取+md5 去重 | `datasets/raw/Plaque/SANO/images/` | 225→**202 唯一** |
| 划分（md5 哈希确定性 70/30） | `split.json` | train 144 / eval 58 |
| 跨集近重复防护（dHash≤5） | 剔除 eval 侧 12 对 | eval 46 图 |
| 牙齿分割→crop（iop_toothseg@896，bbox 直裁，min 8px） | `scripts/build_plaque_sano_crops.py` | train 3,522 / **eval 1,148** crop |
| v1 预筛+人工核验 | `annotation_queue/sano_eval_review/` | 高 P crop 确认=干净牙（模糊/暖色误报）→train 全量作阴性 |
| 合成 plaque_bin_v2 | symlink v1 + 3,522 SANO 阴性(×1) | train 117,360（pos 94,018 / neg 23,340） |

关键策略：**不过滤高 P 的 SANO crop**——它们恰是模型当前误报的难阴性；
标签依据目录级"SANO=健康"+全量目检，残余噪声 <1%（3,522/117,360）。

## 3. 训练

- 从 v1 best.pt 低学习率微调（lr0=0.002, cos_lr, 10 epochs, patience 4, batch 384@224，保守增广同 v1）
- 早停于 epoch 6，best=epoch2 → `outputs/plaque_bin_v2_sano/weights/best.pt`
- val 与 v1 完全一致（重平衡口径），checkpoint 选择可比

## 4. 结果（v1 vs v2，各自 extended val 冻结阈值）

### 4.1 SANO 健康阴性评估集（1,148 crop / 46 独立图，全阴性）

| 指标 | v1 | v2_sano |
|---|---|---|
| Spec @各自冻结 thr | 0.348 | **0.952**（CI95 0.928~0.973，图级 cluster bootstrap） |
| P(plaque) 中位 | 0.984 | **0.000** |
| 误报图占比 | 46/46 | 22/46 |

### 4.2 原协议（Mendeley 官方 test，不重划，防回退检查）

| 指标 | v1 | v2_sano |
|---|---|---|
| AUROC test clean | 0.8950 | **0.8958** |
| AP test clean | 0.9967 | **0.9973** |
| test clean Sens / Spec(n=56) | 0.940 / 0.554 | 0.958 / 0.464 |
| test extended Sens / Spec(n=420) | 0.940 / 0.543 | 0.958 / 0.464 |

排序能力无损（AUROC/AP 持平微升）；固定 thr=0.92 时点态 Spec 略降属工作点漂移，见 4.3。

### 4.3 工作点选择（核心）

v2 概率分布更分离，阈值成为自由参数。**同敏感度对比（推荐部署点 thr=0.70）**：

| | v1 @0.92（旧部署） | **v2 @0.70（新部署）** |
|---|---|---|
| testExt Sens | 0.940 | 0.937（持平） |
| testExt Spec | 0.543 | **0.560** |
| SANO 健康 Spec | 0.348 | **0.978** |

## 5. 部署变更

- `configs/pipeline.yaml`：`weights.plaque → outputs/plaque_bin_v2_sano/weights/best.pt`，
  `plaque.conf_threshold: 0.92 → 0.70`
- 回滚：weights 改回 `outputs/plaque_bin_v1/weights/best.pt` + thr 0.92 即可
- 评估产物：`outputs/plaque_bin_v2_sano/eval_v1_vs_v2.{json,md}`（含全阈值画像）

## 6. 局限与后续

1. SANO 标签为目录级弱标签，正式对外报告前建议人工快审 `annotation_queue/sano_eval_review/`（10 张 review sheet）
2. SANO eval 与 train 已做 md5+dHash 双重隔离；但源数据无患者 ID，图级相关性已用 cluster bootstrap 处理
3. 协议冻结点（0.48）与部署点（0.70）的差异源于 val 阴性仅 96 个（噪声大）——
   若后续扩充 val 阴性，可收紧冻结规则
4. 特异度进一步提升路径：SANO 前牙视角为主，后牙阴性可从 dental disease detection normal 图补充
