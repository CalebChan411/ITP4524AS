# Part A2：Wine Quality（葡萄酒质量预测）

本项目使用老师提供的 `winequality_with_missing.csv`，按教材方法完成：

- 红酒、白酒分别使用 **多元线性回归 LinearRegression** 预测 `quality`。
- 红酒、白酒分别使用 **决策树 DecisionTreeClassifier** 预测好酒（`quality >= 7`）。
- 共保存 **4 个已训练模型**，并提交原始数据、清洗数据、训练/测试数据、预测结果、评价指标和英文报告。

## 1. 最快测试：检查已提交的结果

推荐使用 **Python 3.12**（本次验证版本为 3.12.14；依赖要求至少 Python 3.11）。

从 GitHub 下载整个仓库，解压后进入包含根目录 `README.md`、`part_a/` 的文件夹，在该目录打开终端。以下命令用 `python`；macOS/Linux 如需使用 `python3`，请替换命令开头。

建议先建立独立虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux 激活：

```bash
source .venv/bin/activate
```

如 Windows 不允许激活脚本，也可以直接用 `.\.venv\Scripts\python.exe` 代替后续命令中的 `python`。

安装已验证版本的依赖：

```bash
python -m pip install -r part_a/a2/requirements.txt
```

验证已上传的模型、数据划分、预测和指标：

```bash
python part_a/a2/wine_quality.py --verify
```

预期最后显示：

```text
red: saved split order, models, predictions and metrics verified
white: saved split order, models, predictions and metrics verified
Verification passed.
```

这个命令只加载模型并重新计算测试集结果，不重新训练或覆盖报告。

## 2. 从原始数据完整运行

```bash
python part_a/a2/wine_quality.py
```

程序会依次进行：读取原始文件 → 统计及清洗 → 按颜色和行序划分 → 训练 → 保存并重载模型 → 评价和绘图 → 生成英文 Markdown/Word 报告 → 验证保存的结果。

可在 `part_a/a2/` 目录内运行 `python wine_quality.py`；默认路径依据脚本位置计算，不依赖终端所在目录。

完整运行会更新本目录的生成数据、模型、图表及报告。也可以把一次复现实验写到另一个目录，便于保留已有结果：

```bash
python part_a/a2/wine_quality.py --output-dir a2_reproduction
python part_a/a2/wine_quality.py --verify --output-dir a2_reproduction
```

## 3. 自动测试

从仓库根目录运行：

```bash
python -m unittest discover -s part_a/a2/tests -v
```

预期 **8 项测试通过**，最后显示 `OK`。测试覆盖原始文件 SHA-256、清洗数量、保留行序、红白酒划分、6/7 分界、输入特征不含答案、4 个模型重新加载、保存预测及所有要求的评价指标。

A1 可独立验证：

```bash
python -m unittest discover -s part_a/a1/tests -v
```

## 4. 数据处理规则及已验证数量

原文件实际以 **分号 `;`** 分隔，列名包含空格（例如 `fixed acidity`）。读取后仅把列名规范为教材/题目形式（例如 `fixed_acidity`），不改变记录顺序。原始文件本身保持字节不变，包含原始 CRLF 换行；Git 属性设置保证上传和下载时不会改写原始 CSV。

原始文件 SHA-256：

```text
3bccfeab023bee1a43b1b697a387eab12b145b5aa3c36adbcc31cf0649925361
```

| 清洗统计 | 数量 |
| --- | ---: |
| 原始记录 | 6,697 |
| 重复额外记录（保留第一次出现） | 1,177 |
| 原始缺失单元格 | 407 |
| 原始含缺失值的记录 | 200 |
| 去重后记录 | 5,520 |
| 去重后再删除的缺失记录 | 200 |
| 最终清洗记录 | 5,320 |

先统计原始重复和缺失，再 `drop_duplicates(keep="first")`，再 `dropna()`；没有填补缺失值、随机打乱、排序或重新分层抽样。两种删除可能在一般数据中重叠，因此报告分别记录原始数量和分阶段删除数量。

| 颜色 | 清洗后 | 前 floor(80%) 训练 | 剩余测试 | 训练好酒 | 测试好酒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| red | 1,359 | 1,087 | 272 | 161 | 23 |
| white | 3,961 | 3,168 | 793 | 694 | 131 |

`source_row` 是从 0 开始的原始数据记录编号（不包括 CSV 标题）；它在清洗后导出时才添加，用于追溯记录，不参与去重或建模。`is_good` 是从质量生成的 0/1 标签。模型只接受 11 个理化特征，排除 `quality`、`is_good`、`color` 和 `source_row`。

## 5. 模型选择与教材对应

| 内容 | 教材依据 |
| --- | --- |
| Pandas/NumPy、索引与表格处理 | Python Foundations 2 |
| 函数、循环、模块和文件操作 | Python Foundations 1–3 |
| 多元线性回归、MAE/MSE/R² | Lecture 5 |
| 决策树、过拟合及分类评价 | Lecture 6 |

回归使用默认普通最小二乘；输出保留连续预测值，不把回归结果四舍五入成分类标签。

决策树参数固定为 `max_depth=5`、`min_samples_leaf=5`、`random_state=42`，用于限制模型复杂度。红白酒使用相同设置；没有根据测试集表现反复挑选参数。这里没有额外的验证集调参步骤。

Pandas 的 CSV 读取/删除接口、joblib 模型持久化、命令行参数、自动测试、Matplotlib 绘图及 python-docx 报告生成属于实现与提交所需的工程补充；核心预测方法来自教材。SciPy 是 scikit-learn 的数值计算依赖。

## 6. 实际测试集结果

| 颜色 | MAE | MSE | R² | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| red | 0.526939 | 0.484720 | 0.266576 | 0.889706 | 0.360000 | 0.391304 | 0.375000 |
| white | 0.524701 | 0.456466 | 0.214626 | 0.788146 | 0.370629 | 0.404580 | 0.386861 |

分类评价以“好酒=1”为正类，Precision/Recall/F1 是该正类指标，不是加权或宏平均。混淆矩阵行表示真实类别、列表示预测类别，顺序均为 `[0, 1]`：

```text
red:   [[233, 16],
        [ 14,  9]]

white: [[572, 90],
        [ 78, 53]]
```

报告如实说明效果限制：回归 R² 约 0.27/0.21；决策树对好酒的 Recall 约 0.39/0.40。测试集中非好酒较多，全部预测为训练集多数类“非好酒”的 Accuracy 分别约为 0.9154/0.8348，均高于决策树 Accuracy，但这个简单基准不能识别好酒。不能仅凭 Accuracy 评价分类质量。

## 7. 提交材料位置

| 路径 | 内容 |
| --- | --- |
| `wine_quality.py` | 清洗、分割、训练、评价、保存和验证 |
| `build_report.py` | 从本次实验产物生成英文报告 |
| `requirements.txt` | 验证过的依赖版本 |
| `data/raw/winequality_with_missing.csv` | 老师原始文件 |
| `data/cleaned_winequality.csv` | 清洗后数据及追溯列 |
| `data/splits/` | 四份训练/测试 CSV |
| `models/` | 四个 `.joblib` 模型和元数据 |
| `results/cleaning_summary.json` | 完整清洗统计及数据指纹 |
| `results/split_summary.csv` | 划分与好酒数量 |
| `results/metrics.csv` / `metrics.json` | 全部要求的评价指标及基准结果 |
| `results/confusion_*.csv` | 两个混淆矩阵 |
| `results/predictions_*.csv` | 每条测试记录的真实值及预测 |
| `results/linear_coefficients.csv` | 回归系数与截距 |
| `results/decision_tree_rules_*.txt` | 决策树规则 |
| `results/plots/` | 分布、回归及混淆矩阵图 |
| `report.md` / `report.docx` | 英文 A2 报告 |

GitHub 下载后请保留目录结构，尤其是数据和模型子目录。A1 与 A2 独立，不需要先运行 A1。
