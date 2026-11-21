"""ITP4524 A1: UCS and A* on the assignment's Figure 1.

Run: python3 part_a/a1/route_finding.py
Only Python's standard library is required.
"""

import csv
import heapq
from pathlib import Path


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
    ("HW", "KB", 9.9),
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
        "expanded_order": [],
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
            continue

        # Stop when the goal is removed, not when it is first discovered.
        if node == goal:
            return finish_result(result, parents, goal, cost)

        result["expanded_order"].append(node)
        for neighbour, edge_cost in graph[node].items():
            candidate = cost + edge_cost
            if candidate < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = candidate
                parents[neighbour] = node
                heapq.heappush(frontier, (candidate, neighbour))

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
        _, cost, node = heapq.heappop(frontier)
        if cost != best_cost[node]:
            continue

        if node == goal:
            return finish_result(result, parents, goal, cost)

        result["expanded_order"].append(node)
        for neighbour, edge_cost in graph[node].items():
            candidate = cost + edge_cost
            if candidate < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = candidate
                parents[neighbour] = node
                new_priority = candidate + heuristic[neighbour]
                heapq.heappush(frontier, (new_priority, candidate, neighbour))

    return result


def main():
    graph = build_graph()
    heuristic = {}
    for node, distance in HEURISTIC_KM.items():
        heuristic[node] = distance * 10

    results = [uniform_cost_search(graph, "CW", "TSW"),
               a_star_search(graph, "CW", "TSW", heuristic)]
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    summaries = []
    for result in results:
        print(f'\n{result["algorithm"]}')
        print("Route:", " -> ".join(result["path"]))
        print(f'Distance: {result["distance_km"]:.1f} km')
        print("Expanded nodes (excluding goal):", len(result["expanded_order"]))
        print("Expansion order:", " -> ".join(result["expanded_order"]))
        summaries.append({
            "algorithm": result["algorithm"],
            "path": " -> ".join(result["path"]),
            "distance_km": result["distance_km"],
            "expanded_nodes": len(result["expanded_order"]),
            "expanded_order": " -> ".join(result["expanded_order"]),
        })
    result_file = output / "comparison.csv"
    with result_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    print("\nComparison saved to:", result_file)


if __name__ == "__main__":
    main()
