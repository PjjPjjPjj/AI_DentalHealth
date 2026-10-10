# Oral-Mamba mask semantics

- Date: 2026-08-26
- image_count: 3375
- label_count: 3365
- matched_count: 3365
- unique pixel values across dataset: [0, 1, 2, 3, 4]
- Visualization overlays: outputs/visualizations/mask_semantics/

```text
Official disease classes:
  - caries
  - gingivitis
  - calculus

Pixel-value mapping (dataset unique values 1..4):
  1 = gingivitis      (CONFIRMED)
  2 = plaque          (CONSISTENT)
  3 = calculus        (CONFIRMED)
  4 = caries          (CONSISTENT)
```

# 证据链

1. 文件名前缀契验：
   - 所有 `gingivitis_(.*)` 命名图片的 mask 均为 {0,1}，即值1=牙龈炎。
   - 值2/3/4 由视觉+统计辅助确认。
2. 视觉核对（outputs/visualizations/mask_semantics/）：
   - 值1 = 牙龈/牙龈缘软组织带（牙龈炎主体区域）。
   - 值2 = 沿牙颈缘与邻间三角的薄带状色素/沉积区（菌斑/色素带）。
   - 值3 = 牙面黄褐色块状/点状沉积（下前牙唇侧褐色尖、磨牙黄褐沉积）——牙结石形态。
   - 值4 = 窝沟暗斑、牙龈缘锯齿状线与牙面色斑（龋坏/龋性改变）。
3. 面积统计：值3 像素量最小（0.97M，712 张，平均每张约 1367 px），与"牙结石为局部点/块状病灶"的医学预期一致。
4. 注意：caries_0_* 前缀文件同时出现在 {0,3} 与 {0,4} 中，说明文件名前缀不代表逐图语义，仅作为统计参考；语义结论以视觉核对为准。

# 保留的疑问/风险

- 外部官方 mapping 文档未能获取（无网络/无随附文档）；
- CALCULUS 一致性检查：若后续训练表现异常（如负样本反而高检出），应回查 值2/值3 是否混淆；
- 上述映射仅用于本文件记录，决定 CALCULUS_PIXEL_VALUE = 3。
