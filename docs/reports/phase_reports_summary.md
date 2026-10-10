# OralAI-V1 Phase Reports

Generated: 2026-08-26T14:21:26

## Phase 0: 工程初始化
Status: environment.json + phase_0_report.md
Completed:
 - 目录骨架
 - .gitignore
 - requirements.txt
 - 环境报告

## Phase 1: 数据审计
Status: raw 目录为空 => 训练侧 BLOCKED_EXTERNAL
Completed:
 - audit.py 全项检查
 - 重复检测(md5+pHash)
 - missing_resources.md
 - Gate1 规则
 - datasets.csv 头

## Phase 2: Caries Detection
Status: 等待 6313 数据即可训练
Completed:
 - convert_caries6313.py
 - train_caries.py
 - evaluate_caries.py
 - configs/caries.yaml
 - TP/FP/FN 导出

## Phase 3: Calculus Segmentation
Status: 等待 Oral-Mamba 数据
Completed:
 - convert_oral_mamba.py
 - train_calculus.py
 - evaluate_calculus.py
 - configs/calculus.yaml

## Phase 4: Plaque Classification
Status: 等待 AIRC-LABDEN 数据
Completed:
 - convert_airc.py(患者级split优先官方文件)
 - train_plaque.py
 - evaluate_plaque.py(整体+分区域)
 - plaque_crops.csv

## Phase 5: Tooth Segmentation
Status: 等待 IO150K 数据
Completed:
 - convert_io150k.py
 - train_tooth.py
 - evaluate_tooth.py

## Phase 6: 图像质量检查
Status: 完成
Completed:
 - image_quality.py(Laplacian/亮度/高光)
 - 阈值全部走 pipeline.yaml

## Phase 7: 病灶-牙齿关联
Status: 完成
Completed:
 - associate.py(calculus overlap 规则/caries 中心规则/unassigned)
 - tooth_level_metrics.py

## Phase 8: 统一 Pipeline
Status: 完成
Completed:
 - pipeline.py 整图推理(非 per-tooth fan-out)
 - schemas.py 统一 JSON schema

## Phase 9: Demo
Status: 完整权重路径待训练后跑
Completed:
 - demo.py --dry-run 验证通过
 - outputs/demo/test_result.json + visualization

## Phase 10: SAM2 标注辅助
Status: sam2 包未安装(按需安装)
Completed:
 - sam2_helper.py(仅几何)
 - export_masks.py
 - review_queue.py
 - annotation_queue/
