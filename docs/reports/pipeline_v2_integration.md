# 横向整合报告：龋齿 + 牙菌斑 统一 Pipeline v2

日期：2026-09-01　|　牙结石：留接口（等待新数据），calculus 字段保持占位

## 0. 结论

四个验证全部通过，统一 pipeline 已可用：

| 验证 | 结果 |
|---|---|
| 1 牙齿分割跨域 | **通过**：IOP-Compass 训练的 yolo11m-seg@896 在 caries6313 真实照片、
Oral-Mamba 特写、Mendeley（含正畸托槽）上均逐牙正确分割，conf 0.53~0.96 |
| 2 龋齿回归 | **通过**：pipeline 路径精确复现 C2 test 指标（P 0.8722 / R 0.9028 / F1 0.8872，零损失） |
| 3 牙菌斑整合损失 | **无损失反有增益**：牙位 crop 路径 AUROC **0.9164** vs GT 框直推 0.8676
（整牙 crop 比 GT 小条带更可分）；多窗口无增益（0.9101），默认回退整牙 crop |
| 4 端到端 smoke | **通过**：真实照片 12 颗牙 + 2 个龋齿框（conf 0.83/0.89，均落在真实龋坏磨牙上，
全部成功关联），schema 零违例；30/30 测试通过 |

## 1. 配置（configs/pipeline.yaml）

```
tooth:   outputs/iop_toothseg/weights/best.pt   @896, conf 0.30, iou 0.70
caries:  outputs/caries_c2/weights/best.pt      @960, conf 0.55(冻结), 整图一次
plaque:  outputs/plaque_bin_v1/weights/best.pt  @224, thr 0.92(冻结, Sens优先), 整牙crop
calculus: 无权重 → 自动禁用，JSON 字段保留占位
```

## 2. 数据流

```
image → 质量门(Laplacian/亮度/高光, 阈值全配置化)
      → 牙齿实例(896 seg, retina masks)
      → 龋齿(960 整图检测, conf 0.55)
      → 牙菌斑(逐牙 crop → cls prob → thr 0.92)
      → 关联(caries: 中心∈mask → 最大重叠 → unassigned)
      → V1 JSON(teeth[] + unassigned_lesions) + 可视化
```

## 3. 已知限制 / 注意事项

1. **plaque 阈值 0.92 是 Sens 优先工作点**（Sens 0.94 / Spec 0.55）：demo 中大多数牙
   plaque present——属设计行为；产品化时按场景调 `inference.plaque.conf_threshold`
2. **质量门偏严**：浅景深特写常被标记 blurry（usable=false 仅为告警不阻断，结果照常输出）
3. demo_images/test.jpg 是卡通占位图（0 牙检测正确）；真实演示图已补
   demo_images/intraoral_real.jpg
4. plaque 模型训练域为 Mendeley 特写，pipeline 牙位 crop 的跨域标定仅在该 val 上验证
   （n=282 配对）；更大规模外部验证待做
5. 牙位 ID 为局部序号（1..N），非 FDI（V1 边界）

## 4. 产物

- 代码：`src/inference/pipeline.py`（calculus 禁用逻辑、plaque 多窗口/阈值、caries 原因字段）、
  `src/inference/models.py`（imgsz/max_det/windows 参数化）
- 配置：`configs/pipeline.yaml`
- 验证脚本：`scripts/verify{1..4}_*.py`
- 证据：`outputs/pipeline_v2_validation/`（tooth 跨域可视化 18 张、caries_regression.json、
  plaque_integration_loss.json、demo json/jpg、smoke_summary.json）
- 测试：`tests/test_pipeline_smoke.py` 新增真实权重 e2e 用例（缺权重自动 skip）

## 5. 后续（V2 待办）

- 新牙结石数据到位后接入 calculus 模型（配置加回 weights.calculus 即可）
- tooth-level 聚合指标（tooth_level_metrics.py 已有框架）接入 pipeline 输出
- plaque G/I/M/D 区域几何（V1 规划中预留的接口）
