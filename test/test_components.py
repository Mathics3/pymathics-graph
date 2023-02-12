# -*- coding: utf-8 -*-
"""
Unit tests for pymathics.graph.components
"""
from test.helper import check_evaluation, evaluate, evaluate_value


def setup_module(module):
    """Load pymathics.graph"""
    assert evaluate_value('LoadModule["pymathics.graph"]') == "pymathics.graph"
    evaluate("SortList[list_] := Sort[Map[Sort, list]]")


def test_connected_components():
    for str_expr, str_expected in [
        ("PlanarGraphQ[Graph[{}]]", "False"),
        ("g = Graph[{1 -> 2, 2 -> 3, 3 <-> 4}];", "Null"),
        ("SortList[ConnectedComponents[g]]", "{{1}, {2}, {3, 4}}"),
        ("g = Graph[{1 -> 2, 2 -> 3, 3 -> 1}];", "Null"),
        ("SortList[ConnectedComponents[g]]", "{{1, 2, 3}}"),
    ]:
        check_evaluation(str_expr, str_expected)
