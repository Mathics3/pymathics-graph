# -*- coding: utf-8 -*-
"""
Unit tests for mathics.builtins.numbers.algebra
"""
from test.helper import check_evaluation, evaluate, evaluate_value


def setup_module(module):
    """Load pymathics.graph"""
    assert evaluate_value('LoadModule["pymathics.graph"]') == "pymathics.graph"
    evaluate("SortList[list_] := Sort[Map[Sort, list]]")


def test_connected_components():
    for str_expr, str_expected in [
        ("g = Graph[{1 -> 2, 2 -> 3, 3 <-> 4}];", "Null"),
        ("SortList[ConnectedComponents[g]]", "{{1}, {2}, {3, 4}}"),
        ("g = Graph[{1 -> 2, 2 -> 3, 3 -> 1}];", "Null"),
        ("SortList[ConnectedComponents[g]]", "{{1, 2, 3}}"),
    ]:
        check_evaluation(str_expr, str_expected)


def test_graph_distance():
    for str_expr, str_expected in [
        ("GraphDistance[{1 <-> 2, 2 <-> 3, 3 <-> 4, 2 <-> 4, 4 -> 5}, 1, 5]", "3"),
        ("GraphDistance[{1 <-> 2, 2 <-> 3, 3 <-> 4, 4 -> 2, 4 -> 5}, 1, 5]", "4"),
        # ("GraphDistance[{1 <-> 2, 2 <-> 3, 4 -> 3, 4 -> 2, 4 -> 5}, 1, 5]", "Infinity"),
        (
            "Sort[GraphDistance[{1 <-> 2, 2 <-> 3, 3 <-> 4, 2 <-> 4, 4 -> 5}, 3]]",
            "{0, 1, 1, 2, 2}",
        ),
    ]:
        check_evaluation(str_expr, str_expected)
