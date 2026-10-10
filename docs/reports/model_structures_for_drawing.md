# 三种病灶检测：原理与模型结构（绘图资料）

生成：2026-09-02。所有数字均提取自真实权重文件。
逐层完整明细（含参数量）见 `outputs/reports/model_layers_dump.txt`。

---

## 0. 总览（画图先看这张表）

| 任务 | 形式 | 模型 | 规模 | 输入 | 输出 |
|---|---|---|---|---|---|
| 龋齿 | 单阶段目标检测 | **YOLO11m-det** (DetectionModel) | 20.05M | 整图 960×960 | 龋损 bbox + conf |
| 牙结石 | **两阶段级联** | S1: YOLO11m-seg 22.36M → S2: YOLO11m-seg 22.36M → Verifier: EffNet-V2-S 20.18M → M3: LogisticRegression | 级联 | S1 整图/分块 640；S2 自适应窗口 crop→640 | 结石 mask + 融合置信度 |
| 牙菌斑 | 牙位级二分类 | **YOLO11s-cls** (ClassificationModel) | 5.45M | 逐牙 crop 224×224 | P(plaque)/P(normal) |

注意：牙结石权重文件名带 "s" 是历史遗留，实际为 **m 规模**（22.36M = YOLO11m-seg 官方参数量）。

---

## 1. 共享骨干（三个模型同源，先画懂这一块）

三个 YOLO 模型的 backbone 完全相同（YOLO11 系列）：

```
输入 640×640×3 (caries 为 960×960)
 ├─ [0] Conv  3×3 s2   64     → 320×320   (下采样×2)
 ├─ [1] Conv  3×3 s2   128    → 160×160
 ├─ [2] C3k2  r=2      256
 ├─ [3] Conv  3×3 s2   256    → 80×80    ← P3 特征 (stride 8)
 ├─ [4] C3k2  r=2      512
 ├─ [5] Conv  3×3 s2   512    → 40×40    ← P4 特征 (stride 16)
 ├─ [6] C3k2  r=2      512 (C3k=True 大核)
 ├─ [7] Conv  3×3 s2   1024   → 20×20    ← P5 特征 (stride 32)
 ├─ [8] C3k2  r=2      1024 (C3k=True)
 ├─ [9] SPPF  5×5 MaxPool×3 并联 1024      ← 多尺度感受野聚合
 └─ [10] C2PSA 1024                        ← 注意力特征(PSA)
```

- **C3k2**：CSP 式双分支特征块（Ultralytics YOLO11 新模块）
- **SPPF**：串联 3 个 5×5 MaxPool 等效多尺度池化
- **C2PSA**：位置敏感注意力块
- **neck（只用于 det/seg）**：PAN-FPN 双向 —— 自顶向下 Upsample+Concat（P5→P4→P3），再自底向上 Conv+Concat（P3→P4→P5），输出 [16]=P3/80×80、[19]=P4/40×40、[22]=P5/20×20 三尺度

## 2. 龋齿检测：YOLO11m-det 单阶段检测

**原理**：整图一次前向 → 三尺度 Anchor-free 解耦头（Detect）→ 每个位置回归 (bbox 4 + cls 1) → 训练用 TAL 任务对齐分配，推理 NMS。960 输入对小龙齲损友好。

```
Backbone(11m) → PAN-FPN neck → Detect([16,19,22]) → boxes/scores → conf≥0.55 → 龋损框
```

**Detect 头**：三尺度各接 2 个分支 —— box 分支 (4×reg_max=DFl 8) 与 cls 分支；训练时 one2many + TAL，推理解码后 NMS。

**关键层**（m.model 逐层，完整表见 dump 文件）：
`[23] Detect from=[16,19,22] params=1.41M`，其余见第 1 节 backbone。

---

## 3. 牙结石：两阶段级联（最复杂，重点画）

**原理**：小病灶多 → 单模型直检召回不够。S1 分块滑窗（宽图 320px 2×2 tile，特写整图）**高召回找候选**；S2 用候选中心 160px（大框自适应放大，上限整图）裁剪 4× 放大做**精细分割**；再由 Verifier + M3 逻辑回归对候选**二次判伪**，按病灶尺寸用不同阈值筛选，加翻转 TTA mask 并集与图像级救援。

```
                    ┌─ 宽图(max边≥500): 2×2 tile 320px
整图 → [S1 m-seg] ──┤
                    └─ 特写: 整图 640
         ↓ 候选框 conf≥0.005 (类别0=calculus)
每候选: 以框中心裁 SIDE=160px (框≥88px 时放大到 2.4×框, ≤整图)
         ↓ resize 640×640
      [S2 m-seg] → 实例 mask(640²) → NMS(0.5) → 放回原图坐标
         ↓ 特征8维: s1_conf, s2_conf, verifier_prob, s1_area_ratio,
                    s2_area_ratio, s1s2_iou, bbox_area_ratio, bbox_aspect
      [Verifier: EffNet-V2-S 4ch] ← 输入 = RGB crop(224) + mask 通道 (验证"这真是结石吗")
         ↓ verifier_prob
      [M3: LogisticRegression] → fused = σ(w·x + b)
         ↓ 尺寸条件阈值: s2_area_ratio≥1.565% → 0.45, 否则 0.6; 且 s1_conf≥0.02
         ↓ (全空且图像级 EffNet≥0.4 时: 救援阈值 0.25)
      翻转 TTA mask 并集(仅相对大病灶) → mask NMS → 输出实例
```

**S1/S2 网络结构**（同一 yaml，YOLO11m-seg，22.36M）：

```
backbone: 同第1节 (m 规模: 末段通道×1.5, max_channels=768)
head (PAN-FPN + Segment):
  H12: [16,19,22] → Segment(nc=2, mask_coef=32, proto=256)
Segment 头 = Detect 检测分支 + Proto 分支:
  proto: P5(20×20×768) → Conv上采样×4 → 32×160×160 原型图
  每实例输出 32 个系数 → mask = Σ coefᵢ·protoᵢ → sigmoid
```

**Verifier**（验证器，20.18M）：
`EfficientNet-V2-S`，唯一改动：首个 Conv2d 由 3ch 扩到 **4ch**（第 4 通道 = 候选 mask，让网络看"RGB+候选区域"），分类头替换为 `Linear(1280→1)` → sigmoid。推理：crop resize 224 + mask 通道 NEAREST 224。

**M3 融合**：LogisticRegression，8 维输入（上列特征），coef≈[-0.32, 1.02, 2.05, -0.44, 1.87, 0.61, 0.38, 0.12]（示例，精确值见 fusion_v3.npz / model_layers_dump.txt 末尾）。

---

## 4. 牙菌斑：YOLO11s-cls 牙位级二分类

**原理**：不做分割——菌斑弥散无像素边界、标注本身就是"区域级 0/1"。牙齿分割提供牙位框 → 逐牙 crop → 分类器输出 P(plaque) → 阈值 0.92 判定（Sens≥0.95 约束下 val 上最大 Spec）。

```
牙位 crop 224×224×3
 ├─ [0] Conv  64  s2
 ├─ [1] Conv  128 s2
 ├─ [2] C3k2  256
 ├─ [3] Conv  256 s2
 ├─ [4] C3k2  512
 ├─ [5] Conv  512 s2
 ├─ [6] C3k2  512
 ├─ [7] Conv  1024 s2
 ├─ [8] C3k2  1024
 ├─ [9] C2PSA 1024          ← 分类版无 SPPF、无 PAN-FPN、无多尺度
 └─ [10] Classify            ← Conv1×1(1280) → GAP → Dropout → Linear(1280→2)
     → Softmax → P(normal), P(plaque)
```

5.45M 参数（s 规模）；训练：阴性 ×8 过采样 + val 1:1 重平衡（防 best 塌缩）。

---

## 5. forward() 真实源码（ultralytics 8.4.129）

### 层循环与跨层路由（所有模型共用）`BaseModel._predict_once`
```python
def _predict_once(self, x, profile=False, embed=None):
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:
        if m.f != -1:  # 非顺序层: 按 from 索引取历史特征
            x = y[m.f] if isinstance(m.f, int) else \
                [x if j == -1 else y[j] for j in m.f]
        x = m(x)                                   # 执行该层
        y.append(x if m.i in self.save else None)  # 需要跨层 Concat 的存下来
    return x
```
（画图要点：neck 里 `Concat from=[-1,6]` 就是把 backbone P4 与上采样特征拼接——这是 U 形连接）

### Segment 头（牙结石）
```python
def forward(self, x):
    outputs = super().forward(x)          # Detect 检测分支
    proto = self.proto(x[0])              # P5 → 原型图 32×160×160
    if self.training: return preds
    return (outputs[0], proto), preds     # 推理: (检测, 原型) → 系数组合成 mask
```

### Detect 头（龋齿）
```python
def forward(self, x):
    preds = self.forward_head(x, **self.one2many)   # 三尺度 box+cls
    if self.training: return preds
    y = self._inference(preds)                      # 解码: DFL→xyxy + NMS
    return (y, preds)
```

### Classify 头（牙菌斑）
```python
def forward(self, x):
    x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
    y = x.softmax(1)                       # → P(normal), P(plaque)
    return (y, x)
```

---

## 6. 绘图建议

1. 三张图统一风格：backbone 画一次，det/seg/cls 用不同颜色高亮各自差异（neck+三尺度头 / +proto 分支 / 无 neck 单头）
2. 牙结石图重点画"级联+验证器+融合"的系统结构，网络细节可缩为一个块
3. 尺寸标注建议：P3/P4/P5 (stride 8/16/32) 与 Detect/Segment 的 from 索引 [16,19,22]
4. 逐层参数量可直接标在层框下方（数据在 model_layers_dump.txt）
