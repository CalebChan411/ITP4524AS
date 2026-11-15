# Part A1：路线搜索

本部分使用教材中的 Uniform Cost Search（UCS，无信息搜索）和 A*（启发式搜索），寻找作业 Figure 1 中从 CW 到 TSW 的最短距离路线。

## 文件与运行

核心程序只使用 Python 标准库，没有需要安装的第三方依赖。已在 Python 3.12.14 上验证。

从仓库根目录运行：

```bash
python3 part_a/a1/route_finding.py
```

也可以进入本目录，运行 `python3 route_finding.py`。Windows 如果没有 `python3` 命令，可以使用 `python` 或 `py`。结果始终写入本文件所在目录下的 `results/`，运行时会更新这些生成文件。

| 文件 | 用途 |
| --- | --- |
| `route_finding.py` | 地图、原始启发值、UCS、A*、结果输出与计时 |
| `report.md` / `report.docx` | 相同内容的英文分析报告；Word 版适合继续编辑与整理提交 |
| `assets/figure1.png` | 从用户提供的 assignment 文件提取的原始地图，用于核对 |
| `tests/test_route_finding.py` | 7 项正确性测试，包括独立回溯枚举和 A* 重新展开的例子 |
| `results/comparison.csv` | 路线、距离、节点数、边检查次数及队列统计 |
| `results/ucs_trace.csv` | UCS 每次有效取出节点的过程 |
| `results/astar_trace.csv` | A* 每次有效取出节点的 g、h、f |
| `results/heuristic_audit.csv` | 19 个节点的原始 h 与实际最短剩余距离 |
| `results/heuristic_inconsistencies.csv` | 6 个不满足一致性条件的有向检查 |
| `results/search_results.json` | 完整搜索记录 |
| `results/runtime.csv` | 交替运行各 1,000 次的实测耗时（微秒，中位数及四分位数） |

## 已验证的结果

```text
CW -> MH -> SYP -> TY -> KF -> TSW
10.5 + 2.1 + 13.0 + 3.5 + 25.6 = 54.7 km
```

| 指标 | UCS | A* |
| --- | ---: | ---: |
| 有效取出节点数，包含终点 | 18 | 6 |
| 展开次数，不包含终点 | 17 | 5 |
| 检查相邻边次数 | 62 | 23 |
| 队列条目峰值，包含旧条目 | 8 | 12 |

本题中没有实际重新展开同一个节点，但 A* 保留了这一机制。耗时会随机器负载变化；不能把一次运行的数值当作普遍性能保证。

## 代码怎么对应教材

1. **图表示**：字典中的字典表示邻接关系，对应 Primer 的 graph / adjacency list。
2. **UCS**：按累计距离 `g(n)` 取出节点，对应 Primer p16–25。
3. **A***：按 `f(n) = g(n) + h(n)` 取出节点，对应 Primer p32–47。
4. **函数、字典、列表、循环与文件输出**：对应 Python Foundations 1–3。
5. `heapq` 是 Python 标准库中实现优先队列的工具；`csv`、`json`、`pathlib`、`perf_counter_ns` 是保存结果和计时所用的标准库接口。这些是工程补充，不是额外的搜索算法。

为便于看清两种算法的区别，代码分别实现 `uniform_cost_search()` 和 `a_star_search()`。UCS 函数既不接收也不读取启发值。没有调用现成的最短路径求解库。

内部距离用 0.1 km 为一个整数单位，避免十进制浮点数造成比较误差；输入列表和报告仍使用原图公里数。优先级相同时，UCS 按节点名字排序，A* 先按较小 g，再按节点名字排序。

## 本题需要特别注意的地方

- 原图没有箭头，按无向图处理：19 个节点、33 条边。没有标记节点的交叉线不代表路口。
- HW–KB 经放大核对为 **9.9 km**。左侧外框线是 TSW–KF，距离 25.6 km。
- 两种算法都在“终点作为当前最小优先级的有效条目被取出”时结束，不在首次发现终点时结束。
- 原始 h 都没有高估实际剩余距离，但有 6 处不一致。例如 `h(TY)=29 > 3.5+h(KF)=28.5`。
- 因此 A* 记录最低 g，发现更短路径时更新并重新放入队列；不使用永久关闭节点的写法。原始 h 没有被修改。
- 旧的较高成本条目会留在堆里，取出时跳过。这与“重新展开一个更低成本的节点”是不同情况。

## 验证

从仓库根目录运行：

```bash
python3 -m unittest discover -s part_a/a1/tests -v
```

测试独立使用小图上的回溯枚举核对最短距离，还验证：所有节点到 TSW 的最优性、启发值的可采纳性、A* 重新展开、首次发现终点时不提前结束、旧条目跳过、起点等于终点和无路可达。

## 提交与报告使用

A1 的核心提交材料是 Python 源码与算法比较分析。可提交 `route_finding.py` 和 `report.docx`，并附 `results/` 作为结果证据；原始地图和测试也可以一并附上。本目录不包含 A2 的内容。报告的理论和实现请结合教材阅读，确保能够解释每一步。
