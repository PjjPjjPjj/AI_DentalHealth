# 牙结石大病灶 Recall 实验报告（v5-fix7：stage1 mask 兜底 + 边界闭运算）

日期：2026-09-02　|　基线：v5-fix4（已打包进 pipeline，test 700 张，实例 IoU≥0.5 匹配，权重/阈值冻结）
状态：**已 promote 到线上（配置 FB），打包 pipeline 复测通过**；回滚见 §6。

## 1. 动机（对 v5-fix4 的 16 个 large GT FN 逐例归因）

阈值扫描已证明阈值不是瓶颈（`outputs/calculus_pipeline_eval/thr_large_sweep.json`：
`thr_large` 0.45→0.25 只把 r_large 抬到 0.571 却把实例 P 打到 0.349）。逐例 trace
（`outputs/calculus_v5_cls/large_fn_trace.json`）显示 28 个 large GT 的 14 个 FN 里：

| 失败环节 | 占比 | 可否推理端修 |
|---|---|---|
| stage2 mask-off（s1 已框住，s2 无任何覆盖 GT 的输出） | 4 | **可**（s1 mask 现被丢弃） |
| stage2 near-miss（s2 mask IoU 0.34–0.49，差临门一脚） | 5 | 可尝试（边界精修） |
| stage1 漏框（特写 film 型 / 疑标签噪声 3129） | 5 | 否（需真实数据/标注） |
| 贪婪 1-to-1 匹配吞并（预测本已选中仍判 FN） | — | 评测侧，另议 |

关键观察：级联里 **stage1 的 retina mask（`msk`，原图分辨率，从未过 s2 小窗）只当融合特征用，
从不作为预测输出**（`scripts/v3_pipeline_lib.py::collect_split`、`src/inference/models.py::CalculusCascadeModel`）。
mask-off / 部分 near-miss 恰恰是"s1 找到、s2 弄丢"，s1 mask 本身可能就够了。

## 2. 方法（两个纯推理 hook，s1/s2/verifier 权重全部不动，候选仍走同一 verifier+M3 打分以保精度）

- **FB（方向 1，s1 mask 兜底）**：对 large 框（s1 mask 全图面积比 ≥ `large_edge`=0.01565）
  追加一个候选 = 闭运算后的 s1 裁剪 mask（proxy s2_conf = s1_conf）。mask-NMS 会自动把它和
  s2 已占据同区域的实例去重，因此它只在 **s2 欠覆盖时存活** → 直击 mask-off。
- **CLOSE（方向 2，边界闭运算）**：对 large 框的 s2 实例 mask 在 640 空间做椭圆核闭运算，
  平滑锯齿/侵蚀边界，试图把 near-miss 的 IoU 推过 0.5。

消融 4 组：A(FB0 CL0=fix4 复现校验)、B(FB1 CL0)、C(FB1 CL1)、D(FB0 CL1)。
M3 与图像集成每组在 val 上重拟合（与 fix1–5 同协议，数字可比）。

## 3. 结果（test 700，instance_cascade 为对外口径）

| 方案 | 实例 P | 实例 R | 实例 F1 | **r_large** | r_med | fp/图 | TP-Dice | 图 Sens | 图 F1 | 图 AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| A = fix4 复现 | 0.4242 | 0.4982 | 0.4583 | 0.500 (14/28) | 0.60 | 0.271 | 0.817 | 0.682 | 0.7163 | 0.9134 |
| **B 仅FB（推荐）** | **0.4251** | **0.5053** | **0.4618** | **0.571 (16/28)** | 0.60 | 0.274 | 0.8166 | 0.696 | 0.7254 | 0.9151 |
| C FB+CLOSE | 0.4239 | 0.5053 | 0.4610 | 0.571 (16/28) | 0.60 | 0.276 | 0.8165 | 0.696 | 0.7254 | 0.9149 |
| D 仅CLOSE | 0.4217 | 0.4982 | 0.4568 | 0.500 (14/28) | 0.60 | 0.274 | 0.8169 | 0.682 | 0.7163 | 0.9134 |

- **A 与线上 fix4 逐位一致**（P/R/F1/r_large/fp/dice 全等）→ 本实验脚手架（缓存+聚合）忠实复现，数字可信。
- **FB 有效且精度安全**：r_large 0.500→**0.571**（+2 例 large TP），实例 R/F1 升，
  **实例 P 不降反微升**（0.4242→0.4251），fp/图仅 +0.003，TP-Dice 基本持平（0.817→0.8166，非用精度换召回）。
- **CLOSE 证伪**：叠加在 FB 上零增量（B≡C 的 r_large/召回一致，P 略降），单独用（D）也无收益且略伤——
  印证 near-miss 的 0.34–0.49 并非靠形态学闭运算能翻过 0.5（与 v5 报告"边界并集会放大 mask"的教训一致）。
  → 推荐配置为 **B（FB only）**。

## 4. 逐例验证（large FN 归因，`fix7_large_fn_attribution.json`）

- FB 救回的 2 例：`caries_0_2747`(451px)、`caries_0_835`(189px)——均为 trace 里的 **s1_cover≈0.83–0.99 但 s2 mask-off（IoU=0）**，机制完全吻合。
- FB **未引入任何新 FN**（B 相对 A 无回归，`NEW: NONE`），CLOSE 相对 FB 亦零变化。
- 剩余未解（符合预期，需数据/标注，非推理端）：stage1 漏框 5 例（216/3129/5873/5959/7630，含标签噪声 3129、
  film 型低对比 5959）+ 少数 s1 mask 本身就没框准的 near-miss/mask-off（74/837、caries_0_115/277 等）。

## 5. 结论与后续

- 采用 **FB（方向 1）** 作为 fix7：当前 28 large GT 中 16 命中（0.571），大病灶瓶颈从"s1 找到但 s2 弄丢"
  进一步收敛到"s1 本身就漏"（特写 film 型 + 标签噪声），后者只能靠**真实数据补充/标注修正**解决。
- FB 与已有 fix1（自适应窗）、fix4（条件 TTA）**正交且叠加**，无需动权重，属纯推理增益。

## 6. Promote 落地与回滚（已完成）

已将 **FB（方向 1）** promote 为线上默认；CLOSE 未采纳。

**改动文件（均有 `.pre_fix7bak` 备份）**：
- `src/inference/models.py::CalculusCascadeModel`：新增 `fb/fb_min_px/close_k` 参数（`fb` 默认 True），
  predict 内 large 框追加 s1-mask 兜底候选；新增 `_morph_close` helper。
- `scripts/v3_pipeline_lib.py::collect_split`：新增 **gated**（`FB_ENABLE` 默认 False，保护归档复现）的同款兜底钩子，
  供需要 FB 口径的离线标定/评测显式开启。
- `configs/pipeline.yaml`：`inference.calculus.fb: true`(+`fb_min_px/close_k`)；
  `weights.calculus.artifacts: outputs/calculus_v5_fix7/artifacts`（含 FB 重拟合的 `fusion_v3.npz`，
  verifier/imgclf 权重原样复制自 fix4，未变）。

**FB 重拟合的 M3**：`python -m scripts.final_test_v5_fix7 fitfb` → 用 val 上带 FB 的候选集重拟合，
写入 artifacts/fusion_v3.npz（生产加载它，故 FB 候选的打分口径与训练一致）。

**打包 pipeline 复测（apples-to-apples，同一 imgclf rescue 路径）** `scripts/eval_pipeline_calculus.py`：

| 配置 | 实例P | 实例R | 实例F1 | r_large | fp/图 | TP-Dice |
|---|---|---|---|---|---|---|
| packaged fix4 (fb=off) | 0.4195 | 0.4911 | 0.4525 | 0.500 (14/28) | 0.273 | 0.8181 |
| **packaged fix7-B (fb=on)** | **0.4204** | **0.4982** | **0.4560** | **0.5714 (16/28)** | 0.276 | 0.8176 |

- packaged fix4 F1=0.4525 与 `latest_metrics_summary.md` 归档打包值**逐位一致** → 复测口径可信。
- promote 后 P/R/F1 **同升**，大病灶 r_large 0.500→0.5714（+2 例），fp/图 +0.003、dice 持平——
  非以精度换召回。offline 与 packaged 数值差 ~0.5pt 属已知的 rescue 分类器差异（effnet@448 vs 打包 imgclf@384），
  两个基线各自一致，不影响增益结论。
- 回归：`tests/` 30 单测全过。

**回滚（一步、零残留）**：
```bash
# 方式一：只回退配置，保留实验产物（models.py 的 fb 默认值会被 pipeline.yaml 覆盖回 false）
cp configs/pipeline.yaml.pre_fix7bak configs/pipeline.yaml
# 方式二：连同代码完全复原
cp src/inference/models.py.pre_fix7bak src/inference/models.py
cp scripts/v3_pipeline_lib.py.pre_fix7bak scripts/v3_pipeline_lib.py
# 实验/产物目录可随意删除，与线上无关：
rm -rf outputs/calculus_v5_fix7 scripts/final_test_v5_fix7.py /tmp/opencode/packaged_ab.py
```

## 7. 产物

- 脚本（自包含，含可续跑缓存构建）：`scripts/final_test_v5_fix7.py`
  - 缓存（config 无关的 s1/s2 前向，1408 张，可续跑）：`outputs/calculus_v5_fix7/cache/*.pkl`
  - 集成特征缓存：`outputs/calculus_v5_fix7/ensemble_cache.npz`
- 指标：`outputs/calculus_v5_fix7/final_fix7_metrics.json`
- 大病灶 FN 归因：`outputs/calculus_v5_fix7/fix7_large_fn_attribution.json`
- 复跑命令：
  ```bash
  python -m scripts.final_test_v5_fix7 build val 708    # 一次性（已缓存则自动跳过）
  python -m scripts.final_test_v5_fix7 build test 700
  python -m scripts.final_test_v5_fix7 agg A,B,C,D      # 出全部消融
  python -m scripts.final_test_v5_fix7 status
  ```
