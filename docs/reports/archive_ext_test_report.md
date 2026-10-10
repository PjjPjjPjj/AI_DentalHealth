# Kaggle archive（7 类口腔疾病）外部测试报告

日期：2026-09-02　|　数据：datasets/raw/archive（246MB，13,873 图 + YOLO 标注子集）

## 0. 结论（先说）

**不适合作为性能测试基准，适合当 V2 鲁棒性训练素材。** 两个硬伤：

1. **数据泄漏**：与我们现有数据 md5 重叠 **4,708 张**（其中直接打进 calculus_v3 的
   test/val split 313 张）——该包就是 calculus_v3 数据集（caries_0/gingivitis/other 域）的
   上游来源。不做过滤直接测 = 自我抄袭。
2. **图像质量域差距悬殊**：过滤后剩余 ~5,000 张真正新图上，模型全面失效（见下）。
   人工核验：消费级照片（笑脸自拍、拼图、旋转/镜像、运动涂抹、强压缩伪影），
   与临床拍摄基准图完全不同域。

## 1. 审计要点

- 7 个文件夹：Calculus 1296 / Data caries 2601 / Gingivitis 2349 / Mouth Ulcer 2806 /
  Tooth Discoloration 2017 / hypodontia 1251 / YOLO 标注子集 1510（4 类框：
  Caries 2086 / Ulcer 369 / Discoloration 333 / Gingivitis 759，**无 calculus 框标注**）
- 0 张损坏；数据集内部 md5 重复组 2,046（原始/增强子目录互相复制）
- 与我们数据重叠分布：Calculus 660、Data caries 1749、Gingivitis 1364、YOLO 子集 1237
  （Mouth Ulcer 0、Tooth Discoloration 3、hypodontia 9）
- 过滤后真正新图：Calculus **466**（内部去重后）/ Mouth Ulcer 2806 / Discoloration 2014 /
  hypodontia 1242 / Gingivitis 985 / Data caries 850

## 2. 外部测试结果（全部基于过滤后的新图）

### A. 牙结石图像级筛查（Calculus+466 vs 其他文件夹负样本 3,432）

| 流 | AUROC | 正样本均分 | 负样本均分 |
|---|---|---|---|
| B: EffNet imgclf@384 | **0.180**（反向） | 0.048 | 0.197 |
| C: yolocls@448 | **0.252**（反向） | 0.066 | 0.175 |

- AUROC < 0.5 = 系统性反向：负样本（Tooth Discoloration 0.232 / Mouth Ulcer 0.231 /
  Data caries 0.276）得分反而高于 Calculus 正样本
- 人工核验（outputs/archive_ext_test/calculus_pos_thumbs.jpg）：正样本多为自拍/拼图/
  旋转图，结石表现为深色染色；模型训练域（Oral-Mamba 临床特写、明黄龈上结石）
  对此完全失配
- **结论**：当前筛查模型只在临床拍摄域有效，对消费级照片零泛化（负结果，重要）

### B. 龋齿检测（141 张带 class-0 框的新图，247 GT 框，conf=0.55 冻结）

- P 0.269 / **R 0.028** / F1 0.051（域内 0.887 → 近乎全失）
- 人工核验（caries_ext_*.jpg）：图源为镜像/涂抹/低分辨率强压缩图，与 6313 基准图
  质量差距悬殊；标注协议亦不同
- **结论**：质量域失效，非指标问题

### C. 牙齿分割（定性）

- 牙龈炎/变色文件夹的正常口内照分割良好（16/23、11/7 颗）
- Mouth Ulcer 样本 0 牙（黏膜溃疡特写，无牙可见，行为合理）

## 3. 建议用途

1. **V2 鲁棒性训练素材**（hard negative + 质量域适应）：过滤重叠后按文件夹作弱标签，
   训练时加质量增强（旋转/拼图/压缩/低光模拟）
2. 筛查产品若面向手机自拍场景，需要**专门的拍摄引导 + 质量门**，
   或采集/构造消费级域的训练数据（本数据集是现成来源）
3. 不建议用于报告指标；已在本报告存档负结果

## 4. 产物

- 审计：`outputs/reports/archive_kaggle_audit.{json,md}`
- 测试：`outputs/archive_ext_test/`（ext_test_summary.json、正样本缩略图墙、
  caries 可视化 4 张、tooth 定性 6 张）
- 脚本：`scripts/audit_archive_kaggle.py`、`scripts/ext_test_archive.py`
