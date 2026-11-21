# Part A2：Wine Quality

仅保留作业要求的清洗、顺序划分、回归、分类、指标和模型保存。

## 运行

使用 Python 3.12。下载最新版仓库后解压到新文件夹，进入 `part_a/a2` 目录，在终端运行：

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 wine_quality.py
```

macOS/Linux 可将 `py -3.12` 换为指向 Python 3.12 的 `python3`。
从仓库根目录也可以运行 `python part_a/a2/wine_quality.py`。

## 核对结果

- 原始 6,697 行；重复额外记录 1,177 行。
- 缺失单元格 407 个，分布在 200 行。
- 先去重、再删除含缺失值的行，剩余 5,320 行。
- 红酒：1,087 行训练、272 行测试。
- 白酒：3,168 行训练、793 行测试。
- 结果写入 `results/metrics.csv`，混淆矩阵写入 `results/confusion_red.csv` 和 `confusion_white.csv`。
- 混淆矩阵的行是真实类别，列是预测类别；0 表示非好酒，1 表示好酒。

## 文件与课程依据

| 文件/内容 | 用途或教材依据 |
| --- | --- |
| `wine_quality.py` | 唯一的 A2 程序；执行清洗、训练、评价和保存 |
| `data/raw/winequality_with_missing.csv` | 老师提供的原文件，使用分号分隔 |
| `data/splits/` | 红/白酒各自的训练集与测试集，共四份 CSV |
| `models/` | 红/白酒各自的回归与分类模型，共四个文件 |
| `results/` | 清洗统计、要求的七项指标和两个混淆矩阵 |
| `report.docx` | 独立的简短报告，运行程序不会生成或修改它；请阅读并核对分析 |
| Pandas、条件筛选及索引 | Python Foundations 2；CSV 读写接口用于落实题目要求 |
| `LinearRegression`、MAE/MSE/R² | Lecture 5 |
| `DecisionTreeClassifier`、Accuracy/Precision/Recall/F1/Confusion Matrix | Lecture 6 |

`is_good` 是由 `quality >= 7` 生成的分类标签。模型输入只使用十一项理化特征，排除 `quality`、`is_good` 和每组内恒定的 `color`。模型按课程的 `fit`、`predict` 流程训练与评价。

分类树使用固定的深度及叶节点样本数限制，没有按测试集结果调参。回归和分类算法与评价来自教材；`joblib` 仅用于保存老师要求提交的已训练模型，`pathlib` 用于定位文件。

## 报告要求

报告说明重复/缺失数量、模型选择及理由，并列出所有指定指标、混淆矩阵和结果观察。模型效果有限：好酒 Recall 约为 39%/40%，不能只凭 Accuracy 判断表现。
