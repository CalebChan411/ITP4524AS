# Part A1：路线搜索

使用课程中的 Uniform Cost Search（UCS，无信息搜索）和 A*（启发式搜索），计算作业 Figure 1 中 CW 到 TSW 的最短距离路线。

## 运行

A1 只使用 Python 标准库，**不需要安装依赖**，也不需要 `requirements.txt` 或安装脚本。已在 Python 3.12 和 3.13 上测试。

在本文件夹 `part_a/a1` 中运行：

```powershell
python route_finding.py
```

也可以在编辑器里直接运行 `route_finding.py`。如果当前在整个仓库的根目录，命令为 `python part_a/a1/route_finding.py`。Windows 也可以用 `py` 代替 `python`。

程序显示两种算法的路线、距离和节点展开顺序，并自动更新本目录中的 `results/comparison.csv`。Word 报告是独立文档，运行程序不会改写报告。

## 结果核对

两种算法都应输出：

```text
CW -> MH -> SYP -> TY -> KF -> TSW
Distance: 54.7 km
```

| 算法 | 距离 | 展开节点数（不包含终点） |
| --- | ---: | ---: |
| UCS | 54.7 km | 17 |
| A* | 54.7 km | 5 |

展开指取出节点后检查其相邻节点。终点取出后就结束，因此不计入展开数。这些数量比较本题的搜索工作量，不等同于实测运行时间。

## 对应教材与题目

- 图和邻接表：`ITP4524 Primer on Artificial Intelligence`，slides 4–9。
- UCS 按累计距离 `g(n)` 选择节点：同一教材 slides 16–25。
- A* 按 `g(n) + h(n)` 选择节点：同一教材 slides 32–47。
- 字典、列表、循环和函数：`Python Foundations 1–3`。
- `heapq`、`csv`、`pathlib` 是实现优先队列和保存文件的 Python 标准库工具，属于实现细节；没有调用现成的最短路径算法。

程序按原图保存 19 个节点、33 条无向边，只在 A* 中使用老师给的原始启发值。内部以 0.1 km 为整数单位计算，输出还原为 km。两种算法都在终点从优先队列有效取出时停止。A* 遇到更低成本会更新并允许再次展开，因为原始启发值有不一致的地方。UCS 同优先级按节点名称排序；A* 同优先级先比较累计成本，再比较节点名称。

## 文件与提交

| 文件 | 用途 |
| --- | --- |
| `route_finding.py` | 地图、UCS、A* 和结果输出 |
| `report.docx` | 第一人称英文分析：算法选择、结果比较、时间复杂度和路径最优性 |
| `results/comparison.csv` | 程序实际运行的比较结果，可作为附件 |
| `README.md` | 安装与运行说明 |

A1 原文要求 Python 实现和比较分析，未单独指定分析报告的文件格式。建议提交 `route_finding.py` 和 `report.docx`；另外两个文件是运行说明和结果附件。课程材料以外的计时实验、详细审计文件和 JSON 输出已移除。
