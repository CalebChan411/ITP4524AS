"""ITP4524 A1: UCS and A* on the assignment's Figure 1.

Run: python3 part_a/a1/route_finding.py
Only Python's standard library is required.
"""

import csv
import heapq
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns


# Figure 1 has no arrows: every edge is treated as undirected.
# Crossed lines without a labelled node do not create an intersection.
EDGE_DISTANCES_KM = [
    ("TSW", "TW", 23.5),
    ("TSW", "KF", 25.6),  # The outer line on the left of the figure.
    ("TSW", "TM", 21.8),
    ("TW", "KF", 3.9),
    ("TW", "TM", 19.6),
    ("KF", "TY", 3.5),
    ("KF", "TM", 21.7),
    ("TY", "KC", 3.4),
    ("TY", "SYP", 13.0),
    ("TY", "MH", 17.9),
    ("KC", "TM", 21.7),
    ("KC", "HW", 5.8),
    ("TM", "MSTI", 7.9),
    ("TM", "ST", 29.5),
    ("HW", "ST", 11.2),
    ("HW", "KB", 9.9),  # Enlarged original image confirms 9.9, not 9.8.
    ("HW", "KT", 14.5),
    ("HW", "MH", 12.6),
    ("ST", "KB", 12.3),
    ("KB", "KT", 3.3),
    ("KT", "LW", 7.6),
    ("KT", "MH", 11.1),
    ("KT", "CW", 10.3),
    ("LW", "YCM", 2.6),
    ("LW", "CW", 12.3),
    ("YCM", "CW", 12.1),
    ("MH", "SYP", 2.1),
    ("MH", "CB", 2.8),
    ("MH", "WCH", 5.0),
    ("MH", "CW", 10.5),
    ("CB", "CW", 8.7),
    ("SYP", "CCI", 5.2),
    ("CCI", "WCH", 4.7),
]

# Original table, unchanged. Only A* uses these estimates for node selection.
HEURISTIC_KM = {
    "TSW": 0, "TM": 21, "TW": 23, "KF": 25, "TY": 29,
    "MSTI": 29, "KC": 32, "HW": 38, "SYP": 42, "MH": 44,
    "CB": 47, "CCI": 47, "KB": 48, "WCH": 49, "ST": 49,
    "KT": 51, "CW": 54, "LW": 59, "YCM": 61,
}


def build_graph():
    """Build an adjacency dictionary; internal costs are tenths of a km.

    Integer arithmetic avoids floating-point errors in cost comparisons.
    For example, 10.5 km is stored as 105. Displayed results use km.
    """
    graph = {}
    for first, second, distance in EDGE_DISTANCES_KM:
        cost = int(round(distance * 10))
        if first not in graph:
            graph[first] = {}
        if second not in graph:
            graph[second] = {}
        graph[first][second] = cost
        graph[second][first] = cost
    return graph


def new_result(algorithm):
    return {
        "algorithm": algorithm,
        "path": [],
        "distance_km": None,
        "selected_order": [],
        "expanded_order": [],
        "edge_checks": 0,
        "heap_pushes": 1,
        "stale_pops": 0,
        "peak_frontier_entries": 1,
        "trace": [],
    }


def finish_result(result, parents, goal, cost):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parents[node]
    path.reverse()
    result["path"] = path
    result["distance_km"] = cost / 10
    return result


def uniform_cost_search(graph, start, goal):
    """Uninformed search: select the smallest accumulated distance g(n).

    graph contains nonnegative integer costs in tenths of a kilometre.
    No heuristic is passed to, or read by, this algorithm.
    """
    frontier = [(0, start)]
    best_cost = {start: 0}
    parents = {start: None}
    result = new_result("UCS")

    while frontier:
        cost, node = heapq.heappop(frontier)

        # An improved path may have left an older entry in the heap.
        if cost != best_cost[node]:
            result["stale_pops"] += 1
            continue

        result["selected_order"].append(node)
        result["trace"].append({
            "step": len(result["selected_order"]), "node": node,
            "g_km": cost / 10, "h_km": "", "priority_km": cost / 10,
            "action": "goal" if node == goal else "expand",
        })

        # Stop when the goal is removed, not when it is first discovered.
        if node == goal:
            return finish_result(result, parents, goal, cost)

        result["expanded_order"].append(node)
        for neighbour, edge_cost in graph[node].items():
            result["edge_checks"] += 1
            candidate = cost + edge_cost
            if candidate < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = candidate
                parents[neighbour] = node
                heapq.heappush(frontier, (candidate, neighbour))
                result["heap_pushes"] += 1
                result["peak_frontier_entries"] = max(
                    result["peak_frontier_entries"], len(frontier)
                )

    return result  # An empty path and None distance mean no route exists.


def a_star_search(graph, start, goal, heuristic):
    """Informed search: select the smallest f(n) = g(n) + h(n).

    Both graph costs and heuristic values are in tenths of a kilometre.
    A* assumes nonnegative edges, admissible h, and h(goal) == 0.
    Better paths can reopen nodes because the supplied h is inconsistent.
    """
    # Ties: smaller g first, then alphabetical node name.
    frontier = [(heuristic[start], 0, start)]
    best_cost = {start: 0}
    parents = {start: None}
    result = new_result("A*")

    while frontier:
        priority, cost, node = heapq.heappop(frontier)
        if cost != best_cost[node]:
            result["stale_pops"] += 1
            continue

        result["selected_order"].append(node)
        result["trace"].append({
            "step": len(result["selected_order"]), "node": node,
            "g_km": cost / 10, "h_km": heuristic[node] / 10,
            "priority_km": priority / 10,
            "action": "goal" if node == goal else "expand",
        })
        if node == goal:
            return finish_result(result, parents, goal, cost)

        result["expanded_order"].append(node)
        for neighbour, edge_cost in graph[node].items():
            result["edge_checks"] += 1
            candidate = cost + edge_cost
            if candidate < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = candidate
                parents[neighbour] = node
                new_priority = candidate + heuristic[neighbour]
                heapq.heappush(frontier, (new_priority, candidate, neighbour))
                result["heap_pushes"] += 1
                result["peak_frontier_entries"] = max(
                    result["peak_frontier_entries"], len(frontier)
                )

    return result


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check_heuristic(graph, heuristic, goal):
    """Audit original h against exact UCS distances; never change h."""
    rows = []
    for node in HEURISTIC_KM:
        distance = uniform_cost_search(graph, node, goal)["distance_km"]
        rows.append({
            "node": node, "h_km": heuristic[node] / 10,
            "shortest_distance_km": distance,
            "admissible": heuristic[node] <= round(distance * 10),
        })

    violations = []
    for node, neighbours in graph.items():
        for neighbour, edge_cost in neighbours.items():
            if heuristic[node] > edge_cost + heuristic[neighbour]:
                violations.append({
                    "from": node, "to": neighbour,
                    "h_from_km": heuristic[node] / 10,
                    "edge_km": edge_cost / 10,
                    "h_to_km": heuristic[neighbour] / 10,
                    "edge_plus_h_to_km": (edge_cost + heuristic[neighbour]) / 10,
                })
    return rows, violations


def benchmark(graph, heuristic, runs=1000):
    """Time search calls with identical tracing enabled, excluding file I/O.

    Alternate which algorithm runs first. The graph is constructed once.
    Results describe this run and machine; they are not universal speedups.
    """
    functions = {
        "UCS": lambda: uniform_cost_search(graph, "CW", "TSW"),
        "A*": lambda: a_star_search(graph, "CW", "TSW", heuristic),
    }
    for function in functions.values():
        function()  # Warm up both functions before collecting measurements.
    times = {"UCS": [], "A*": []}
    for iteration in range(runs):
        order = ["UCS", "A*"] if iteration % 2 == 0 else ["A*", "UCS"]
        for name in order:
            begin = perf_counter_ns()
            functions[name]()
            times[name].append((perf_counter_ns() - begin) / 1000)
    rows = []
    for name, values in times.items():
        ordered = sorted(values)
        middle = len(ordered) // 2
        rows.append({
            "algorithm": name, "runs": runs,
            "median_us": round(median(values), 3),
            "q1_us": round(median(ordered[:middle]), 3),
            "q3_us": round(median(ordered[middle:]), 3),
        })
    return rows


def main():
    graph = build_graph()
    heuristic = {}
    for node, distance in HEURISTIC_KM.items():
        heuristic[node] = distance * 10

    results = [uniform_cost_search(graph, "CW", "TSW"),
               a_star_search(graph, "CW", "TSW", heuristic)]
    audit, violations = check_heuristic(graph, heuristic, "TSW")
    if not all(row["admissible"] for row in audit):
        raise ValueError("The heuristic overestimates a remaining distance.")
    if results[0]["distance_km"] != results[1]["distance_km"]:
        raise AssertionError("UCS and A* disagree on the shortest distance.")

    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    summaries = []
    for result, filename in zip(results, ["ucs_trace.csv", "astar_trace.csv"]):
        print(f'\n{result["algorithm"]}')
        print("Route:", " -> ".join(result["path"]))
        print(f'Distance: {result["distance_km"]:.1f} km')
        print("Selected:", " -> ".join(result["selected_order"]))
        print("Expanded nodes (excluding goal):", len(result["expanded_order"]))
        print("Edge checks:", result["edge_checks"])
        write_csv(output / filename, result["trace"],
                  ["step", "node", "g_km", "h_km", "priority_km", "action"])
        summaries.append({
            "algorithm": result["algorithm"],
            "path": " -> ".join(result["path"]),
            "distance_km": result["distance_km"],
            "selected_nodes": len(result["selected_order"]),
            "expanded_nodes": len(result["expanded_order"]),
            "edge_checks": result["edge_checks"],
            "heap_pushes": result["heap_pushes"],
            "stale_pops": result["stale_pops"],
            "peak_frontier_entries": result["peak_frontier_entries"],
        })
    write_csv(output / "comparison.csv", summaries, list(summaries[0]))
    write_csv(output / "heuristic_audit.csv", audit, list(audit[0]))
    write_csv(output / "heuristic_inconsistencies.csv", violations,
              ["from", "to", "h_from_km", "edge_km", "h_to_km", "edge_plus_h_to_km"])
    (output / "search_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    timing = benchmark(graph, heuristic)
    write_csv(output / "runtime.csv", timing, list(timing[0]))
    print(f"\nGraph: {len(graph)} nodes, {len(EDGE_DISTANCES_KM)} undirected edges")
    print(f"Heuristic: all admissible; {len(violations)} directed consistency violations")
    for row in timing:
        print(f'{row["algorithm"]}: median {row["median_us"]} us '
              f'(Q1 {row["q1_us"]}, Q3 {row["q3_us"]}; {row["runs"]} runs)')
    print("Results saved to:", output)


if __name__ == "__main__":
    main()
