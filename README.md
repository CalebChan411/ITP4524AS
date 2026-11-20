# ITP4524 Assignment

本仓库的作业以已提供的 ITP4524 教材为基础；必要的教材外补充会单独说明。

## Part A1: Route Finding

- Python 程序：[route_finding.py](part_a/a1/route_finding.py)
- 英文报告：[report.md](part_a/a1/report.md) / [report.docx](part_a/a1/report.docx)
- 中文运行与学习说明：[README.md](part_a/a1/README.md)
- 实际运行结果：[results](part_a/a1/results/)

在仓库根目录运行（只需要 Python 3，不需安装第三方库）：

```bash
python3 part_a/a1/route_finding.py
python3 -m unittest discover -s part_a/a1/tests -v
```

## Part A2: Wine Quality

- Python 程序：[wine_quality.py](part_a/a2/wine_quality.py)
- 英文报告：[report.md](part_a/a2/report.md) / [report.docx](part_a/a2/report.docx)
- 中文安装、运行与测试：[README.md](part_a/a2/README.md)
- 老师原始数据：[winequality_with_missing.csv](part_a/a2/data/raw/winequality_with_missing.csv)
- 训练/测试集：[data/splits](part_a/a2/data/splits/)
- 已训练的四个模型：[models](part_a/a2/models/)
- 指标、预测和图表：[results](part_a/a2/results/)

Python 3.12 环境中，从仓库根目录运行：

```bash
python -m pip install -r part_a/a2/requirements.txt
python part_a/a2/wine_quality.py --verify
python -m unittest discover -s part_a/a2/tests -v
```

完整重跑：`python part_a/a2/wine_quality.py`。
