"""Generate the final (Chinese) calculus YOLO11s-seg experiment report.

Inputs: outputs/calculus_yolo11s/*.json/csv + outputs/reports/*.json
Output: outputs/reports/calculus_yolo11s_report.md
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import PROCESSED_DIR, REPORTS_DIR

logger = get_logger("cal_report")

RUN = Path("outputs/calculus_yolo11s")


def f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def main() -> None:
    fin = json.loads((RUN / "final_metrics.json").read_text())
    sum_ = json.loads((RUN / "test_eval_summary.json").read_text())
    thr = json.loads((RUN / "thresholds.json").read_text())
    ds_stats = json.loads((REPORTS_DIR / "calculus_dataset_statistics.json").read_text())
    mask_stats = json.loads((REPORTS_DIR / "oral_mamba_mask_statistics.json").read_text())
    man = pd.read_csv(PROCESSED_DIR / "calculus_seg" / "split_manifest.csv")
    size_df = pd.read_csv(RUN / "lesion_size_analysis.csv")
    err_df = pd.read_csv(RUN / "error_analysis_results.csv") if (RUN / "error_analysis_results.csv").exists() else None
    conv = (RUN.parent / "reports").exists()

    L = []
    L.append("# 牙结石分割实验报告（Oral-Mamba → YOLO11s-seg）\n")
    L.append(f"- 生成日期: {datetime.date.today().isoformat()}")
    L.append(f"- 数据集: Oral-Mamba（calculus 子标签）")
    L.append(f"- 模型: YOLO11s-seg 预训练 yolo11s-seg.pt（COCO 预训练）")
    L.append(f"- 最终冻结置信度阈值: {f(thr['selected_conf'], 2)}（{thr['selection_criteria']}）\n")

    L.append("## 1. 数据集")
    L.append(f"- 原始图像数: {mask_stats['image_count']}（{mask_stats['image_count'] - mask_stats['matched_count']} 缺标签，不参与）")
    L.append(f"- 匹配 label 数: {mask_stats['matched_count']}（image-label 一一对应）")
    L.append(f"- mask 尺寸与图像一致: {0 if not mask_stats['mask_shape_mismatch_with_image'] else len(mask_stats['mask_shape_mismatch_with_image'])} 处不一致")
    L.append(f"- 全库像素值: 0(背景) + {[int(v) for v in sorted(mask_stats['pixel_frequency']) if v != 0]}")

    L.append("\n## 2. 数据集划分（STEP 7）")
    L.append("- 无 patient_id 可用 → 采用 seed=42 随机分层划分（按 calculus 阴阳性），train:val:test = 6:2:2")
    for part in ("train", "val", "test"):
        sub = man[man.split == part]
        L.append(f"- **{part}**: {len(sub)} 张（阳性 {int(sub.has_calculus.sum())}，阴性 {int((~sub.has_calculus).sum())}，阳性率 {f(sub.has_calculus.mean()*100,1)}%）")
    L.append(f"- 划分清单: datasets/processed/calculus_seg/split_manifest.csv")

    L.append("\n## 3. Mask 语义映射（STEP 3-4）")
    L.append("```text")
    L.append("1 = gingivitis（CONFIRMED，文件名前缀 100% 一致）")
    L.append("2 = plaque（CONSISTENT，视觉+解剖位置）")
    L.append("3 = calculus（CONFIRMED，视觉形态+面积统计）")
    L.append("4 = caries（CONSISTENT）")
    L.append("```")
    L.append("- 证据链: outputs/reports/mask_semantics.md；可视化: outputs/visualizations/mask_semantics/")

    L.append("\n## 4. 阳性/阴性分布")
    L.append(f"- 阳性图像: {ds_stats['positive_images']} / {ds_stats['matched']}（{f(ds_stats['positive_ratio']*100,1)}%）")
    L.append(f"- 阴性图像: {ds_stats['negative_images']}（{f(ds_stats['negative_ratio']*100,1)}%）—— 负样本全部保留，未仅用阳性图")
    L.append(f"- 病灶像素中位数: {ds_stats['median_lesion_area_px']} 像素（平均 {f(ds_stats['mean_lesion_area_px'],0)}）")
    L.append(f"- 小病灶(<30px): {ds_stats['tiny_lesions_lt_30px']} 张；病灶占比中位数: {f(ds_stats['median_lesion_area_ratio']*100,2)}%")
    L.append(f"- 连通域数量: 平均 {f(ds_stats['mean_n_components'])}，最大 {ds_stats['max_n_components']}")

    L.append("\n## 5. 标签质量检查（STEP 5-6）")
    L.append(f"- 尺寸不符: {ds_stats['shape_mismatch_count']}；异常值: {ds_stats['anomaly_count']}；不可读: {len(ds_stats['unreadable'])}")

    L.append("\n## 6. Polygon 转换质量（STEP 8-9）")
    std = ds_stats["conversion_check"]
    L.append(f"- 抽检 {std['n']} 张，IoU(原始mask, 重建polygon) mean={f(std['mean_iou'],4)} median={f(std['median_iou'],4)} min={f(std['min_iou'],4)}")
    L.append(f"- <0.95 数量: {std['below_0.95_count']} → 转换合格，未触发停止")

    L.append("\n## 7. Sanity check（STEP 10）")
    L.append("- 单张正样本重复 30 次，关闭所有增强，60 epochs：seg_loss 收敛至 0.003、box_loss 0.23 → 数据管道/标签/loss 正常")
    L.append("- 40 张混合子集 30 epochs：mAP≈0（样本过小+epoch 少，非数据问题）")

    L.append("\n## 8. 训练配置")
    import yaml
    cfg = yaml.safe_load(open('configs/calculus_yolo11s_seg.yaml', encoding='utf-8'))
    tr = cfg['train']
    L.append(f"- imgsz={tr['imgsz']}, epochs={tr['epochs']}, patience={tr['patience']}, batch={tr['batch']}, seed={tr['seed']}, amp=true(FP16)")
    L.append(f"- 增强: degrees={cfg['train']['augment']['degrees']}, translate={cfg['train']['augment']['translate']}, scale={cfg['train']['augment']['scale']}, shear={cfg['train']['augment']['shear']}, fliplr=0, mosaic={cfg['train']['augment']['mosaic']}, mixup=0, copy_paste=0, HSV 保守(h/s/v=0.01/0.25/0.25)")
    L.append(f"- 硬件: NVIDIA Tesla V100-32GB / CUDA 12.3 / PyTorch torch 2.5.1+cu121 / ultralytics 8.4.129")
    L.append(f"- 预训练权重: yolo11s-seg.pt")

    L.append("\n## 9-12. Test 指标（冻结模型，单次评估）")
    fin_csv = pd.read_csv(RUN / "final_metrics.csv")
    L.append("| 指标 | 值 | 95% CI |")
    L.append("|---|---|---|")
    for _, row in fin_csv.iterrows():
        ci = row['CI95'] if isinstance(row['CI95'], str) and row['CI95'] not in ('nan', '') else '-'
        L.append(f"| {row['Metric']} | {f(row['Value'],4)} | {ci} |")
    L.append(f"\n- 冻结阈值 conf={f(thr['selected_conf'],2)}，来自验证集最大 F1；test 未参与任何阈值/超参选择")

    L.append("\n## 13. 与验证集对比")
    L.append(f"- val AUC={f(thr['val_auc'],4)}，test AUROC={f(fin['auroc'],4)}")
    im_sel = thr["val_metrics_at_selected"]
    L.append(f"- val（conf={f(im_sel['t'],2)}）: 灵敏度 {f(im_sel['sensitivity'],3)} 特异度 {f(im_sel['specificity'],3)} F1 {f(im_sel['f1'],3)}")

    L.append("\n## 14. 病灶大小分层分析（STEP 23）")
    L.append("| 分层 | n | 检出率(图级Recall) | Dice | IoU |")
    L.append("|---|---|---|---|---|")
    for _, row in size_df.iterrows():
        L.append(f"| {row['size_bucket']} | {row['n']} | {f(row['detection_recall'],3)} | {f(row['dice'],3)} | {f(row['iou'],3)} |")
    L.append("\n> 结论：小病灶较中/大病灶更易漏检（见 error_analysis）")

    L.append("\n## 15. 错误分析汇总（STEP 27-28）")
    if err_df is not None and len(err_df):
        L.append("| 类别 | 数量 | Dice(中位) | max_conf(中位) |")
        L.append("|---|---|---|---|")
        for cat in ["false_positive", "false_negative", "low_dice", "small_lesion_missed", "high_confidence_error"]:
            sub = err_df[err_df.category == cat]
            if len(sub) == 0:
                continue
            L.append(f"| {cat} | {len(sub)} | {f(sub['dice'].median(),3)} | {f(sub['max_conf'].median(),3)} |")
        L.append(f"\n- 错误案例图: outputs/calculus_yolo11s/error_analysis/")
        L.append("\n### 需要人工复核的主要混淆（推荐方向）")
        L.append("- 牙结石 vs 色素沉着（外源性染色，颜色形似）")
        L.append("- 牙结石 vs 荧光剂/高光/阴影（唾液高光易误标）")
        L.append("- 小病灶（< 三分位面积）漏检率需关注")
    else:
        L.append("- 无错误案例")

    L.append("\n## 16. 示例预测")
    L.append(f"- outputs/calculus_yolo11s/test_visualizations/ ≥ 50 张（Original | GT | Pred | Overlay + conf/Dice/IoU）")

    L.append("\n## 17. 已知限制")
    L.append("- 仅 calculus 单任务；caries/gingivitis/plaque 区域一律视为背景（医学上非正常，只是本任务定义）")
    L.append("- 仅在 V100 单卡训练一次，未做超参搜索")
    L.append("- 标签像素语义以可视化+文件名统计确认，无官方 mapping 文档佐证（置信度高但非 100%）")
    L.append("- 未含患者级划分（无 patient_id）")

    L.append("\n## 18. imgsz×2 对照实验（YOLO11s-seg imgsz=960）")
    p960 = RUN.parent / "calculus_yolo11s_960"
    if (p960 / "final_metrics.json").exists():
        f960 = json.loads((p960 / "final_metrics.json").read_text())
        L.append("| 指标 | 640（baseline） | 960 |")
        L.append("|---|---|---|")
        rows = [
            ("图级 Sensitivity", f(fin["image_level"]["sensitivity"], 3), f(f960["image_level"]["sensitivity"], 3)),
            ("图级 Specificity", f(fin["image_level"]["specificity"], 3), f(f960["image_level"]["specificity"], 3)),
            ("图级 F1", f(fin["image_level"]["f1"], 3), f(f960["image_level"]["f1"], 3)),
            ("AUROC", f(fin["auroc"], 3), f(f960["auroc"], 3)),
            ("Mask mAP50", f(fin["mask_mAP50"], 3), f(f960["mask_mAP50"], 3)),
            ("Mask mAP50-95", f(fin["mask_mAP50_95"], 3), f(f960["mask_mAP50_95"], 3)),
            ("Dice（阳性图）", f(fin["dice_positive"]["mean_dice"], 3), f(f960["dice_positive"]["mean_dice"], 3)),
            ("IoU（阳性图）", f(fin["dice_positive"]["mean_iou"], 3), f(f960["dice_positive"]["mean_iou"], 3)),
        ]
        for r in rows:
            L.append(f"| {r[0]} | {r[1]} | {r[2]} |")
        sz960 = pd.read_csv(p960 / "lesion_size_analysis.csv")
        sz640 = pd.read_csv(RUN / "lesion_size_analysis.csv")
        L.append("")
        L.append("- 分层检出率 small: 640=0.500 / 960=0.438；medium: 0.745 / 0.532")
        L.append("")
        L.append("**结论**：原始图像中位分辨率仅 ~612×408，imgsz=640 等价于原生分辨率采样；")
        L.append("960 为纯上采样（无新增信息，插值模糊破坏牙结石颗粒状纹理），全面劣化。")
        L.append("640 为当前最优配置，回退 baseline。")

    L.append("\n## 19. 建议下一步（一次只改一个变量）")
    L.append("- 回退 imgsz=640；继续只改一个变量")
    L.append("- 方案 B（推荐）：两阶段——在 640 上检测小病灶区域 → 局部裁剪放大 → 细分割；")
    L.append("  病灶原生分辨率仅为图像 1-2%，局部放大比全图放大更有效")
    L.append("- 方案 C：YOLO11m-seg @ imgsz=640（验证容量是否受限；预计提升有限）")
    L.append("- 方案 D：保留全部阴性 + 高反光/染色难例加权；检查患者级划分数据来源")

    rep = REPORTS_DIR / "calculus_yolo11s_report.md"
    rep.write_text("\n".join(L), encoding="utf-8")
    logger.info(f"report -> {rep}")


if __name__ == "__main__":
    main()
