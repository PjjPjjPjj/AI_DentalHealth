# Mendeley 牙菌斑数据集审计报告

- 官方 split 行数: 10450；图片总数: 10450（base 1319 / 增强变体 9131）
- 图片-split 覆盖: 缺 csv 0，缺文件 0
- 患者跨 split 泄漏: 0 名
- split 分布(全部): {'train': 7538, 'test': 1646, 'validation': 1266}
- split 分布(仅 base): {'test': 207, 'train': 950, 'validation': 162}
- 无法解码图片: []
- 无标签图片: 0
- 总框数: 133824；均值/图: 12.81
- plaque=1 框占比 per split: {'train': 0.9771, 'validation': 0.9941, 'test': 0.9801}
- (image,tooth) 实例: 33456，组内标签冲突: 2016
- md5 跨 split 重复组: 0；dHash 跨 split 近重复组: 4
- 牙位 ID 使用数: 20（top: [(1, 19920), (6, 19848), (16, 14616), (11, 13872), (3, 9000), (8, 8112)]）
- 类别失衡说明: 数据集按框标注，plaque=1 框占比约 98%，负样本极少；训练需对 normal 类过采样，评估以 Sens/Spec/F1/AUROC 为主，不使用 Accuracy。
