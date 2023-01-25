# -*- coding: utf-8 -*-
"""
Unit tests for pymathics.graph.tree
"""
from test.helper import check_evaluation, evaluate_value


def setup_module(module):
    """Load pymathics.graph"""
    assert evaluate_value('LoadModule["pymathics.graph"]') == "pymathics.graph"


def test_tree():
    for str_expr, str_expected in [
        ("TreeGraphQ[StarGraph[3]]", "True"),
        ("TreeGraphQ[CompleteGraph[0]]", "False"),
        ("TreeGraphQ[CompleteGraph[1]]", "True"),
        ("TreeGraphQ[CompleteGraph[2]]", "True"),
        ("TreeGraphQ[CompleteGraph[3]]", "False"),
    ]:
        check_evaluation(str_expr, str_expected)
