"""Behaviour checks; no third-party test runner is required."""

import unittest

from part_a.a1.route_finding import (
    EDGE_DISTANCES_KM, HEURISTIC_KM, a_star_search, build_graph,
    check_heuristic, uniform_cost_search,
)


def exhaustive_shortest_cost(graph, start, goal):
    """Independent reference: enumerate simple paths with backtracking.

    Positive edge costs allow pruning a branch once its cost reaches the
    best complete route found so far. This is only a small-graph test oracle.
    """
    def visit(node, seen, cost, best):
        if cost >= best:
            return best
        if node == goal:
            return cost
        for neighbour, weight in graph[node].items():
            if neighbour not in seen:
                best = visit(neighbour, seen | {neighbour}, cost + weight, best)
        return best

    return visit(start, {start}, 0, float("inf"))


class RouteFindingTests(unittest.TestCase):
    def setUp(self):
        self.graph = build_graph()
        self.heuristic = {node: h * 10 for node, h in HEURISTIC_KM.items()}

    def test_assignment_route_and_independent_optimality(self):
        expected_path = ["CW", "MH", "SYP", "TY", "KF", "TSW"]
        reference = exhaustive_shortest_cost(self.graph, "CW", "TSW")
        self.assertEqual(reference, 547)
        for result in [uniform_cost_search(self.graph, "CW", "TSW"),
                       a_star_search(self.graph, "CW", "TSW", self.heuristic)]:
            with self.subTest(algorithm=result["algorithm"]):
                self.assertEqual(result["path"], expected_path)
                self.assertEqual(result["distance_km"], reference / 10)
                path_cost = sum(self.graph[a][b] for a, b in
                                zip(result["path"], result["path"][1:]))
                self.assertEqual(path_cost, reference)

    def test_graph_transcription_and_symmetry(self):
        self.assertEqual(len(self.graph), 19)
        self.assertEqual(len(EDGE_DISTANCES_KM), 33)
        self.assertEqual(sum(map(len, self.graph.values())), 66)
        self.assertEqual(set(self.graph), set(self.heuristic))
        self.assertEqual(self.graph["HW"]["KB"], 99)
        self.assertEqual(self.graph["TSW"]["KF"], 256)
        for first, neighbours in self.graph.items():
            for second, cost in neighbours.items():
                self.assertGreater(cost, 0)
                self.assertEqual(self.graph[second][first], cost)

    def test_heuristic_against_independent_costs_for_every_node(self):
        for start in self.graph:
            with self.subTest(start=start):
                exact = exhaustive_shortest_cost(self.graph, start, "TSW")
                self.assertLessEqual(self.heuristic[start], exact)
                for result in [uniform_cost_search(self.graph, start, "TSW"),
                               a_star_search(self.graph, start, "TSW", self.heuristic)]:
                    self.assertEqual(round(result["distance_km"] * 10), exact)
        audit, violations = check_heuristic(self.graph, self.heuristic, "TSW")
        self.assertTrue(all(row["admissible"] for row in audit))
        self.assertEqual({(row["from"], row["to"]) for row in violations}, {
            ("TY", "KF"), ("HW", "KC"), ("MSTI", "TM"),
            ("KB", "HW"), ("LW", "KT"), ("CB", "MH"),
        })

    def test_astar_reopens_expanded_node_when_better_path_arrives(self):
        # A is expanded with cost 30 before B, then reopened with cost 20.
        graph = {"S": {"A": 30, "B": 10}, "A": {"G": 20},
                 "B": {"A": 10}, "G": {}}
        h = {"S": 0, "A": 0, "B": 30, "G": 0}
        result = a_star_search(graph, "S", "G", h)
        self.assertEqual(result["path"], ["S", "B", "A", "G"])
        self.assertEqual(result["distance_km"], 4.0)
        self.assertEqual(result["expanded_order"].count("A"), 2)

    def test_goal_discovery_does_not_stop_search(self):
        graph = {"S": {"G": 100, "A": 10}, "A": {"G": 10}, "G": {}}
        h = {node: 0 for node in graph}
        for result in [uniform_cost_search(graph, "S", "G"),
                       a_star_search(graph, "S", "G", h)]:
            self.assertEqual(result["path"], ["S", "A", "G"])
            self.assertEqual(result["distance_km"], 2.0)

    def test_stale_entries_are_not_expanded(self):
        graph = {"S": {"A": 50, "B": 10}, "B": {"A": 10},
                 "A": {}, "G": {}}
        h = {node: 0 for node in graph}
        for result in [uniform_cost_search(graph, "S", "G"),
                       a_star_search(graph, "S", "G", h)]:
            self.assertEqual(result["path"], [])
            self.assertIsNone(result["distance_km"])
            self.assertEqual(result["stale_pops"], 1)
            self.assertEqual(result["expanded_order"].count("A"), 1)

    def test_start_equals_goal(self):
        for result in [uniform_cost_search(self.graph, "TSW", "TSW"),
                       a_star_search(self.graph, "TSW", "TSW", self.heuristic)]:
            self.assertEqual(result["path"], ["TSW"])
            self.assertEqual(result["distance_km"], 0.0)
            self.assertEqual(result["expanded_order"], [])


if __name__ == "__main__":
    unittest.main()
