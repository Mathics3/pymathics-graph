# -*- coding: utf-8 -*-
"""
Unit tests for pymathics.graph.parametric
"""
from test.helper import check_evaluation, evaluate, evaluate_value


def setup_module(module):
    """Load pymathics.graph"""
    assert evaluate_value('LoadModule["pymathics.graph"]') == "pymathics.graph"
    evaluate("SortList[list_] := Sort[Map[Sort, list]]")


def test_complete_graph():
    for str_expr, str_expected, mess in [
        (
            "CompleteGraph[0]",
            "CompleteGraph[0]",
            ["Expected a positive integer at position 1 in CompleteGraph[0]."],
        ),
    ]:
        check_evaluation(str_expr, str_expected, expected_messages=mess)
