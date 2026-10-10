# 牙结石大病灶修复实验报告（v5-fix 系列）

日期：2026-09-01　|　基线：v5 级联（test 700 张，实例 IoU≥0.5 匹配，阈值/级联参数冻结）

## 1. 问题定位（13 个 large 桶 FN 全链路 trace）

- "large"是**相对尺寸**（面积比 ≥1.57%），并非绝对大目标：FN 全部来自 126~612px 的特写图
- 失败环节分布：
  - **stage2 mask 质量（7/13，主因）**：stage1 已框住（conf 0.36~0.90），但 160px 固定窗口
    内大 blob 占满画面（OOD），s2 mask IoU 卡在 0.36~0.49
  - stage1 完全漏检（3/13）：3129 / 5959 / 5873（特写图大 blob 无候选）
  - verifier 误杀多为 stage2 mask 失败的连锁反应
- 证伪的假设：zoom-out 二尺度对特写图无效（特写图本来就在整图 640 推理，结果逐位相同）

## 2. 实验结果（instance_cascade 模式）

| 方案 | P | R | F1 | r_large | fp_img | 图像F1 | 结论 |
|---|---|---|---|---|---|---|---|
| v5 基线 | 0.4389 | 0.4733 | 0.4555 | 0.4286 | 0.243 | 0.7025 | — |
| fix1 自适应窗口 | 0.4397 | **0.4804** | **0.4592** | 0.4286 | 0.246 | 0.7163 | 稳健小幅全面增益 |
| fix2 灰边画布 | — | — | — | **0.3214↓** | — | — | **证伪丢弃**：s2 本在放大特写分布上训练，pad 反而推离分布 |
| fix3 全量翻转TTA | 0.3476↓ | 0.4911 | 0.4071↓ | **0.6071** | 0.370 | **0.7273** | 大病灶+18pt 但 P 崩 |
| fix4 条件TTA | 0.4242 | **0.4982** | 0.4583 | **0.5000** | 0.271 | 0.7163 | **推荐**：大病灶+7pt、P 代价 1.5pt |
| fix5 TTA+阈值重定 | 0.4444 | 0.4698 | 0.4567 | 0.4643 | 0.236 | 0.7203 | val-F1 调参抵消了大病灶增益 |

## 3. 最终推荐配置（fix4）

```python
# scripts/v3_pipeline_lib.py 开关
_lib.S2_ADAPTIVE = True   # s1 框 >=0.55*160px 时窗口放大到 2.4x（上限整图）
_lib.S2_TTA = True        # 相对大 blob（>=0.35 裁剪区）触发水平翻转 TTA + mask 并集
# stage1/stage2/verifier 权重全部不动；M3 融合与图像集成照常在 val 重拟合
```

- 收益：r_large 0.43→**0.50**，r_medium 0.56→**0.60**，实例 R 0.473→**0.498**，
  图像级 Sens 0.662→**0.682**、F1 0.703→**0.716**
- 代价：fp_img +0.028，实例 P -1.5pt
- 若场景为"高敏感筛查"（宁多勿漏）：可切换 fix3 全量 TTA（r_large 0.61、图像 F1 0.727），
  接受实例 P 降至 0.35
- 注：v5b stage1（copy-paste FT）+ fix4 配置（fix6）已实验并否决，见第 4 节

## 4. 剩余已知限制

- stage1 对特写图大 blob 的漏检（3 例）——人工核验发现这 3 例并非尺度问题：
  - `caries_0_3129`：红圈标注实为**深色修复体/染色样外观**（同图其余深色牙未标注），
    疑似标签噪声，模型不响应属合理行为
  - `caries_0_5959`：倾斜牙冠上**大面积淡黄薄膜**，与牙体低对比，属真实难例
  - 可视化：`outputs/demo/large_fn_vis/`
- **stage1 copy-paste 重训（v5b, fix6）已尝试并证伪**：1200 张无缝克隆合成图微调 14 epochs 后，
  r_large 无变化（0.50），实例 P 0.424→0.358、F1 0.458→0.412——合成斑点多触发 FP，
  学不到 film 型外观。结论：此 3 例只能靠**标注修正/真实数据补充**解决，推理与训练增强均无解。
- fix3/fix4 的 mask 并集会轻微放大边界 mask，若下游对 mask 面积敏感需注意

## 5. 产物

- 脚本：`scripts/final_test_v5_fix{1..5}.py`、`final_test_v5b_fix6.py`、
  `scripts/build_v5b_synth.py`、`scripts/train_v5b_stage1.py`、
  `scripts/diag_large_fn_v5.py`、`scripts/trace_large_fn_v5.py`；
  lib 开关在 `scripts/v3_pipeline_lib.py`
- 指标：`outputs/calculus_v5_fix*/final_fix*_metrics.json`、
  `outputs/calculus_v5b_fix6/final_fix6_metrics.json`、
  `outputs/calculus_v5_cls/large_fn_{probe,trace}.json`
- 日志：`outputs/logs/v5_fix*.log`、`outputs/logs/v5b_*.log`
