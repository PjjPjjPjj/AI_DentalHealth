# GitHub 上传指南

本指南说明如何把本仓库发布到你自己的 GitHub。

> 本仓库已完成**许可证选择、版权标注、路径可移植化、隐私脱敏**四项准备工作。
> 你只需要建仓、推送、传权重三步。

---

## 0. 上传前检查清单

| # | 检查项 | 状态 | 说明 |
|---|---|---|---|
| 1 | 许可证已选定为 AGPL-3.0 | ✅ | `LICENSE`（Ultralytics YOLO11 要求） |
| 2 | 第三方声明已写全 | ✅ | `THIRD_PARTY_NOTICES.md`：16 个依赖 + 字体 + 12 个数据集 |
| 3 | 源码头 SPDX 标识 | ✅ | **240 / 240** 个 `.py` 全部带 `AGPL-3.0-or-later` |
| 4 | 无硬编码绝对路径 | ✅ | 74 个文件（`.py` / `.sh` / `.yaml` / 报告）已改为项目相对路径 |
| 5 | 无患者影像 | ✅ | `demo_images/` 真实照片已剔除，改为程序生成合成示意图 |
| 6 | 无数据集 | ✅ | 176 GB 数据集全部排除 |
| 7 | 无模型权重 | ✅ | 507 MB ONNX + 100 MB+ `.pt` 全部排除，走 Releases |
| 8 | 无备份/缓存文件 | ✅ | 3 个 `.pre_fix7bak` 旧备份已排除 |
| 9 | 测试可运行 | ✅ | `25 passed, 6 skipped`（跳过项需真实权重/字体/显示器） |
| 10 | 语法校验 | ✅ | 240 个 `.py` 零语法错误；11 个 `.sh` 通过 `bash -n`；11 个 YAML 有效 |
| 11 | 仓库体积 | ✅ | **2.5 MB**（git 压缩后约 500 KB），最大单文件 55 KB |

### ⚠️ 上传前你必须自己改的 4 个文件

代码里用占位符标记未填信息，**忘记替换会让版权声明失真**。
先定位所有占位符：

```bash
cd github_upload
grep -rn "COPYRIGHT HOLDER\|<你的用户名>\|<你的名字>\|YOUR_GITHUB_ID\|YOUR_NAME\|YOUR_GIVEN_NAME" \
  --include="*.md" --include="*.cff" --include="LICENSE" .
```

| 文件 | 占位符 | 改成 |
|---|---|---|
| `LICENSE` | `<COPYRIGHT HOLDER>` | 你的名字或组织名 |
| `README.md`（3 处） | `<你的用户名>` | 你的 GitHub 用户名 |
| `README.md`（1 处） | `<你的名字>` | 你的名字 |
| `CITATION.cff` | `YOUR_NAME` / `YOUR_GIVEN_NAME` / `YOUR_GITHUB_ID` | 你的真实信息 |
| 全部 240 个 `.py` | `Copyright (C) 2026 OralAI-DentalHealth contributors` | 你的名字或组织名 |

一键替换（把 `YOUR_NAME` 换成你的名字，`YOUR_GITHUB_ID` 换成你的用户名）：

```bash
cd github_upload

# 1. LICENSE 的版权持有人
sed -i 's|<COPYRIGHT HOLDER>|你的名字|g' LICENSE

# 2. README 的用户名与姓名
sed -i 's|<你的用户名>|YOUR_GITHUB_ID|g; s|<你的名字>|你的名字|g' README.md

# 3. CITATION.cff
sed -i 's|YOUR_GITHUB_ID|YOUR_GITHUB_ID|g; s|YOUR_GIVEN_NAME|YOUR_GIVEN_NAME|g; s|YOUR_NAME|YOUR_NAME|g' CITATION.cff

# 4. 全部 Python 源文件的版权头
grep -rl "OralAI-DentalHealth contributors" --include="*.py" . \
  | xargs sed -i 's|OralAI-DentalHealth contributors|YOUR_NAME|g'

# 验证：应无输出
grep -rn "COPYRIGHT HOLDER\|<你的用户名>\|<你的名字>\|OralAI-DentalHealth contributors" \
  --include="*.md" --include="*.cff" --include="LICENSE" --include="*.py" .
```

---

## 1. 准备 Git 环境

```bash
git --version          # 需要 >= 2.0
git config --global user.name  "你的名字"
git config --global user.email "你的邮箱@example.com"
```

---

## 2. 在 GitHub 上创建空仓库

1. 打开 <https://github.com/new>
2. **Repository name**：例如 `OralAI-DentalHealth`
3. **Public**（公开）—— AGPL-3.0 本身就是要求源码公开的协议
4. ⚠️ **不要勾选** "Add a README file"、"Add .gitignore"、"Choose a license"
   （本仓库已经自带，手动添加会产生冲突）
5. 点 **Create repository**

---

## 3. 推送代码

```bash
cd github_upload

git init                      # git < 2.28 只能这样写，见下方说明
git symbolic-ref HEAD refs/heads/main
git add .
git status                    # 确认没有大文件、权重、图片混进来
git commit -m "Initial release: tooth-level oral lesion screening (AGPL-3.0)"
```

> **git 版本兼容**：`git init -b main` 需要 git ≥ 2.28。
> 如果报 `unknown switch 'b'`，改用上面两行写法
> （`git init` + `git symbolic-ref`），效果完全相同。
> 查看版本：`git --version`。

### 推送前务必核对 staged 内容

```bash
# 318 个文件，约 2.5 MB，压缩后约 500 KB
git ls-files --cached | wc -l

# 以下命令应全部输出「0 命中」——任何一项命中都说明混入了不该上传的文件
git ls-files --cached | grep -E '\.(pt|onnx|pkl|npz|zip|7z|otf|ttf)$'
git ls-files --cached | grep -E '^datasets/|^runs/|^app/models/|^appenv/|^dist/'
git ls-files --cached | grep -E '\.bak$|fix7bak|__pycache__'

# 图片应只有 1 张合成示意图
git ls-files --cached | grep -E '\.(jpg|jpeg|png)$'
```

预期结果：前三条无输出，最后一条只输出
`demo_images/synthetic_schematic.jpg`。

### 推送到 GitHub

```bash
git remote add origin https://github.com/YOUR_GITHUB_ID/AI_DentalHealth.git
git push -u origin main
```

> 如果需要用 SSH：`git remote add origin git@github.com:YOUR_GITHUB_ID/AI_DentalHealth.git`

---

## 4. 上传模型权重到 Releases

权重不适合放进 git（体积大 + 需与源码同协议分发），走 Releases。

### 4.1 准备要上传的权重

```bash
cd /root/AI_DentalHealth   # 你的工作目录

# 训练用权重（.pt，约 100 MB~1 GB）
mkdir -p /tmp/release/pt
cp outputs/iop_toothseg/weights/best.pt          /tmp/release/pt/tooth_seg.pt
cp outputs/calculus_v5_stage1/weights/best.pt     /tmp/release/pt/calculus_s1.pt
cp outputs/calculus_v3b_stage2/weights/best.pt    /tmp/release/pt/calculus_s2.pt
cp outputs/caries_c2/weights/best.pt              /tmp/release/pt/caries_det.pt
cp outputs/plaque_bin_v2_sano/weights/best.pt     /tmp/release/pt/plaque_cls.pt
cp -r outputs/calculus_v5_fix7/artifacts          /tmp/release/pt/calculus_artifacts

# 桌面端 ONNX（约 507 MB）
mkdir -p /tmp/release/onnx
cp app/models/*.onnx app/models/*.npz /tmp/release/onnx/
```

### 4.2 打包（可选，便于一次上传）

```bash
cd /tmp/release && zip -r oralai_weights_v1.zip pt onnx
ls -lh oralai_weights_v1.zip
```

### 4.3 创建 Release

1. 打开仓库的 <https://github.com/YOUR_GITHUB_ID/AI_DentalHealth/releases>
2. 点 **Draft a new release**
3. **Choose a tag** → 填 `v1.0.0` → **Create new tag on publish**
4. **Title**：`v1.0.0 — 牙齿级口腔病灶筛查模型权重`
5. **Release notes** 建议写：

```markdown
## 模型权重 v1.0.0

本 Release 的权重与仓库源码适用**相同的 AGPL-3.0 协议**
（Ultralytics YOLO11 派生权重，见 THIRD_PARTY_NOTICES.md 第 4 节）。

### 下载后放置路径

| 文件 | 放置位置 | 任务 |
|---|---|---|
| `pt/tooth_seg.pt` | `outputs/iop_toothseg/weights/best.pt` | 牙齿实例分割 |
| `pt/calculus_s1.pt` | `outputs/calculus_v5_stage1/weights/best.pt` | 牙结石 stage1 |
| `pt/calculus_s2.pt` | `outputs/calculus_v3b_stage2/weights/best.pt` | 牙结石 stage2 |
| `pt/calculus_artifacts/` | `outputs/calculus_v5_fix7/artifacts/` | 验证器 + 融合参数 |
| `pt/caries_det.pt` | `outputs/caries_c2/weights/best.pt` | 龋齿检测 |
| `pt/plaque_cls.pt` | `outputs/plaque_bin_v2_sano/weights/best.pt` | 牙菌斑分类 |
| `onnx/*.onnx` | `app/models/` | 桌面端运行时 |

### 性能（患者级隔离的官方 test 集）

| 任务 | 指标 |
|---|---|
| 牙齿分割 | 4,373/4,449 颗正确检出（98.3%） |
| 龋齿检测 | F1 0.887 |
| 牙菌斑 | AUROC 0.916 |
| 牙结石 | 图像级 AUROC 0.915 / 实例级 F1 0.462 |

### ⚠️ 使用限制

- 权重**仅供研究、教学与内部评估**使用
- **不是医疗器械**，未通过临床验证，**禁止用于患者诊疗决策**
- 禁止用于任何临床收费服务
- 衍生模型同样适用 AGPL-3.0
```

6. 上传 `oralai_weights_v1.zip`
7. 点 **Publish release**

---

## 5. 提交 Issue 模板与 PR 模板（可选但推荐）

```bash
cd github_upload && mkdir -p .github
```

`.github/PULL_REQUEST_TEMPLATE.md`：

```markdown
## 变更内容
## 关联 Issue
## 测试情况
- [ ] `python -m pytest tests/ -q` 通过
- [ ] 新增/修改代码含 SPDX 版权头
- [ ] 未引入新的第三方依赖（或已更新 THIRD_PARTY_NOTICES.md）
- [ ] 未提交任何数据集、患者影像、模型权重
```

---

## 6. 发布后自查

```bash
# 1. 仓库体积应 < 10 MB
git count-objects -vH | grep size-pack

# 2. 确认没有大文件
git ls-files | xargs -I{} du -k {} 2>/dev/null | sort -rn | head -5

# 3. 确认许可证被 GitHub 识别
#    打开仓库页面，右侧应显示 "AGPL-3.0 license"
```

在 GitHub 仓库主页确认：
- [ ] 右侧栏显示 **AGPL-3.0 license**
- [ ] 代码浏览页每个文件顶部有 SPDX 头
- [ ] `docs/reports/` 正常显示
- [ ] Releases 页面权重可下载

---

## 7. 常见问题

### push 被拒绝：文件超过 100 MB

GitHub 硬限制单文件 100 MB。本仓库已排除所有权重，正常不会触发。
若误提交：

```bash
git filter-repo --path app/models --invert-paths
git push --force
```

> `filter-repo` 需先装：`pip install git-filter-repo`

### 别人报 "该仓库使用了 AGPL 代码但没有提供源码"

AGPL 第 13 条要求：提供**网络服务**时必须向服务使用者提供源码。
你的 GitHub 仓库本身就是源码，符合要求。需确保：
- 不要把核心代码移到私有仓库
- Release 中的权重对应的源码也必须公开

### 权重被下架（DMCA）

数据集来源不明的部分（如"Teeth or Dental Image Dataset"、Kaggle 归档集）
存在版权风险。若收到下架通知：
1. 立即从 Releases 删除相关权重
2. 保留 `THIRD_PARTY_NOTICES.md` 的审计记录（这本身是尽责证据）
3. 改用许可清晰的公开数据集重训

### 想改成闭源

必须先取得 [Ultralytics Enterprise 商业许可](https://www.ultralytics.com/license)，
否则 YOLO 衍生权重与代码的 AGPL 义务依然存在。

---

## 附：文件清单速查

| 目录 | 文件数 | 体积 | 是否上传 |
|---|---|---|---|
| `src/` | 55 | 384 KB | ✅ 核心库代码 |
| `scripts/` | 174 | 1.5 MB | ✅ 数据/训练/评估脚本 |
| `tests/` | 5 | 32 KB | ✅ 单元测试 |
| `configs/` | 11 | 44 KB | ✅ 任务与管线配置 |
| `docs/reports/` | 42 | 240 KB | ✅ 已脱敏报告 |
| `app/` | 22 | 152 KB | ✅ 桌面应用代码（无权重/字体） |
| `demo.py` `demo_plaque.py` | 2 | 24 KB | ✅ 推理入口 |
| `demo_images/synthetic_schematic.jpg` | 1 | 30 KB | ✅ 合成示意图（非患者） |
| `LICENSE` | 1 | 34 KB | ✅ AGPL-3.0 全文 |
| `THIRD_PARTY_NOTICES.md` | 1 | 12 KB | ✅ 第三方与数据声明 |
| **合计** | **~314** | **2.5 MB** | ✅ |

**不上传的内容：**

| 内容 | 体积 | 原因 |
|---|---|---|
| `datasets/` | 176 GB | 数据集，禁止再分发 |
| `runs/` | 5.3 GB | 训练输出 |
| `outputs/` 其余部分 | 4.3 GB | 中间产物、缓存、权重 |
| `app/models/` | 507 MB | ONNX 权重 → Releases |
| `*.pt` | 100 MB+ | 训练权重 → Releases |
| `assets/fonts/` | 16 MB | Noto OFL-1.1，需自行下载 |
| `dist/` `build/` `appenv/` | 1.5 GB | 桌面端构建产物 |
| `demo_images/` 真实照片 | 284 KB | 患者影像，隐私 |
| 报告中的照片墙/蒙太奇 | 6.3 MB | 患者影像，隐私 |
