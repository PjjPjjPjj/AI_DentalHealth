# Missing Resources (BLOCKED_EXTERNAL)

Generated: 2026-08-26T14:14:05

The following items block only their own training phase. All code, configs, converters, eval and tests are implemented so that once data is placed into `datasets/raw/<dataset>/`, the pipeline can run end to end.

## AIRC-LABDEN - missing
- Task: `plaque_cls`
- Needed: images, region bbox labels, patient_id in Data_splits.csv
- Supports: Plaque classification training + patient-level split
- Expected location: `datasets/raw/AIRC-LABDEN/`

## 6313 Intraoral Caries - missing
- Task: `caries_det`
- Needed: images, caries bbox labels (roLabelImg/VOC/CSV)
- Supports: Caries detection baseline
- Expected location: `datasets/raw/6313 Intraoral Caries/`

## Oral-Mamba - missing
- Task: `calculus_seg`
- Needed: RGB images, calculus masks
- Supports: Calculus segmentation baseline
- Expected location: `datasets/raw/Oral-Mamba/`

## IO150K / TeethSEG - missing
- Task: `tooth_seg`
- Needed: images, tooth instance masks, domain note (real RGB vs rendered)
- Supports: Tooth instance segmentation baseline
- Expected location: `datasets/raw/IO150K /`
