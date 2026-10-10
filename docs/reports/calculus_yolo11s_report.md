# 牙结石分割实验报告（Oral-Mamba → YOLO11s-seg）

- 生成日期: 2026-08-26
- 数据集: Oral-Mamba（calculus 子标签）
- 模型: YOLO11s-seg 预训练 yolo11s-seg.pt（COCO 预训练）
- 最终冻结置信度阈值: 0.31（max F1 on validation set）

## 1. 数据集
- 原始图像数: 3375（10 缺标签，不参与）
- 匹配 label 数: 3365（image-label 一一对应）
- mask 尺寸与图像一致: 0 处不一致
- 全库像素值: 0(背景) + [0, 1, 2, 3, 4]

## 2. 数据集划分（STEP 7）
- 无 patient_id 可用 → 采用 seed=42 随机分层划分（按 calculus 阴阳性），train:val:test = 6:2:2
- **train**: 2018 张（阳性 427，阴性 1591，阳性率 21.2%）
- **val**: 672 张（阳性 142，阴性 530，阳性率 21.1%）
- **test**: 675 张（阳性 143，阴性 532，阳性率 21.2%）
- 划分清单: datasets/processed/calculus_seg/split_manifest.csv

## 3. Mask 语义映射（STEP 3-4）
```text
1 = gingivitis（CONFIRMED，文件名前缀 100% 一致）
2 = plaque（CONSISTENT，视觉+解剖位置）
3 = calculus（CONFIRMED，视觉形态+面积统计）
4 = caries（CONSISTENT）
```
- 证据链: outputs/reports/mask_semantics.md；可视化: outputs/visualizations/mask_semantics/

## 4. 阳性/阴性分布
- 阳性图像: 712 / 3365（21.2%）
- 阴性图像: 2653（78.8%）—— 负样本全部保留，未仅用阳性图
- 病灶像素中位数: 852.5 像素（平均 1368）
- 小病灶(<30px): 0 张；病灶占比中位数: 0.99%
- 连通域数量: 平均 0.440，最大 10

## 5. 标签质量检查（STEP 5-6）
- 尺寸不符: 0；异常值: 0；不可读: 0

## 6. Polygon 转换质量（STEP 8-9）
- 抽检 100 张，IoU(原始mask, 重建polygon) mean=0.9999 median=1.0000 min=0.9960
- <0.95 数量: 0 → 转换合格，未触发停止

## 7. Sanity check（STEP 10）
- 单张正样本重复 30 次，关闭所有增强，60 epochs：seg_loss 收敛至 0.003、box_loss 0.23 → 数据管道/标签/loss 正常
- 40 张混合子集 30 epochs：mAP≈0（样本过小+epoch 少，非数据问题）

## 8. 训练配置
- imgsz=640, epochs=100, patience=20, batch=48, seed=42, amp=true(FP16)
- 增强: degrees=5.0, translate=0.05, scale=0.2, shear=2.0, fliplr=0, mosaic=0.5, mixup=0, copy_paste=0, HSV 保守(h/s/v=0.01/0.25/0.25)
- 硬件: NVIDIA Tesla V100-32GB / CUDA 12.3 / PyTorch torch 2.5.1+cu121 / ultralytics 8.4.129
- 预训练权重: yolo11s-seg.pt

## 9-12. Test 指标（冻结模型，单次评估）
| 指标 | 值 | 95% CI |
|---|---|---|
| Precision (pixel) | 0.4038 | - |
| Recall/Sensitivity (pixel) | 0.4359 | - |
| F1 (pixel) | 0.4193 | - |
| Dice | 0.8178 | 0.791–0.845 |
| IoU | 0.8000 | 0.771–0.829 |
| Dice (positive images) | 0.4196 | - |
| IoU (positive images) | 0.3355 | - |
| Mask mAP50 | 0.3622 | - |
| Mask mAP50-95 | 0.1322 | - |
| Image-level Sensitivity | 0.6643 | 0.587–0.741 |
| Image-level Specificity | 0.9248 | 0.902–0.946 |
| Image-level Precision | 0.7037 | - |
| Image-level F1 | 0.6835 | 0.617–0.740 |
| Image-level Accuracy | 0.8696 | - |
| Image-level AUROC | 0.8252 | - |

- 冻结阈值 conf=0.31，来自验证集最大 F1；test 未参与任何阈值/超参选择

## 13. 与验证集对比
- val AUC=0.8756，test AUROC=0.8252
- val（conf=0.31）: 灵敏度 0.690 特异度 0.926 F1 0.703

## 14. 病灶大小分层分析（STEP 23）
| 分层 | n | 检出率(图级Recall) | Dice | IoU |
|---|---|---|---|---|
| small | 48 | 0.500 | 0.270 | 0.220 |
| medium | 47 | 0.745 | 0.512 | 0.410 |
| large | 48 | 0.750 | 0.479 | 0.378 |

> 结论：小病灶较中/大病灶更易漏检（见 error_analysis）

## 15. 错误分析汇总（STEP 27-28）
| 类别 | 数量 | Dice(中位) | max_conf(中位) |
|---|---|---|---|
| false_positive | 40 | 0.000 | 0.595 |
| false_negative | 48 | 0.000 | 0.000 |
| low_dice | 10 | 0.000 | 0.438 |
| high_confidence_error | 1 | 0.000 | 0.840 |

- 错误案例图: outputs/calculus_yolo11s/error_analysis/

### 需要人工复核的主要混淆（推荐方向）
- 牙结石 vs 色素沉着（外源性染色，颜色形似）
- 牙结石 vs 荧光剂/高光/阴影（唾液高光易误标）
- 小病灶（< 三分位面积）漏检率需关注

## 16. 示例预测
- outputs/calculus_yolo11s/test_visualizations/ ≥ 50 张（Original | GT | Pred | Overlay + conf/Dice/IoU）

## 17. 已知限制
- 仅 calculus 单任务；caries/gingivitis/plaque 区域一律视为背景（医学上非正常，只是本任务定义）
- 仅在 V100 单卡训练一次，未做超参搜索
- 标签像素语义以可视化+文件名统计确认，无官方 mapping 文档佐证（置信度高但非 100%）
- 未含患者级划分（无 patient_id）

## 18. imgsz×2 对照实验（YOLO11s-seg imgsz=960）
| 指标 | 640（baseline） | 960 |
|---|---|---|
| 图级 Sensitivity | 0.664 | 0.559 |
| 图级 Specificity | 0.925 | 0.913 |
| 图级 F1 | 0.683 | 0.595 |
| AUROC | 0.825 | 0.784 |
| Mask mAP50 | 0.362 | 0.299 |
| Mask mAP50-95 | 0.132 | 0.106 |
| Dice（阳性图） | 0.420 | 0.330 |
| IoU（阳性图） | 0.336 | 0.259 |

- 分层检出率 small: 640=0.500 / 960=0.438；medium: 0.745 / 0.532

**结论**：原始图像中位分辨率仅 ~612×408，imgsz=640 等价于原生分辨率采样；
960 为纯上采样（无新增信息，插值模糊破坏牙结石颗粒状纹理），全面劣化。
640 为当前最优配置，回退 baseline。

## 19. 建议下一步（一次只改一个变量）
- 回退 imgsz=640；继续只改一个变量
- 方案 B（推荐）：两阶段——在 640 上检测小病灶区域 → 局部裁剪放大 → 细分割；
  病灶原生分辨率仅为图像 1-2%，局部放大比全图放大更有效
- 方案 C：YOLO11m-seg @ imgsz=640（验证容量是否受限；预计提升有限）
- 方案 D：保留全部阴性 + 高反光/染色难例加权；检查患者级划分数据来源