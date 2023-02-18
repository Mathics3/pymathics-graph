# -*- coding: utf-8 -*-
"""
Unit tests for pymathics.graph.base
"""
from test.helper import check_evaluation, evaluate, evaluate_value

import pytest


def setup_module(module):
    """Load pymathics.graph"""
    assert evaluate_value('LoadModule["pymathics.graph"]') == "pymathics.graph"
    evaluate("SortList[list_] := Sort[Map[Sort, list]]")


@pytest.mark.parametrize(
    ("str_expr", "str_expected", "msg"),
    [
        (None, None, None),
        (
            "g = Graph[{a -> b, b <-> c, d -> c, d -> a, e <-> c, d -> b}];",
            "Null",
            "Intialize graph",
        ),
        ("Sort[DegreeCentrality[g]]", "{2, 2, 3, 4, 5}", None),
        ('Sort[DegreeCentrality[g, "In"]]', "{0, 1, 1, 3, 3}", None),
        ('Sort[DegreeCentrality[g, "Out"]]', "{1, 1, 1, 2, 3}", None),
        (None, None, None),
    ],
)
def test_degree_centrality(str_expr, str_expected, msg):
    check_evaluation(str_expr, str_expected, failure_message=msg)


@pytest.mark.parametrize(
    ("str_expr", "str_expected", "msg"),
    [
        (None, None, None),
        (
            "Length[EdgeList[EdgeDelete[{a -> b, b -> c, c -> d}, b -> c]]]",
            "2",
            None,
        ),
        # ("Length[EdgeList[EdgeDelete[{a -> b, b -> c, c -> b, c -> d}, b <-> c]]]", "4", None),
        ("Length[EdgeList[EdgeDelete[{a -> b, b <-> c, c -> d}, b -> c]]]", "3", None),
        ("Length[EdgeList[EdgeDelete[{a -> b, b <-> c, c -> d}, c -> b]]]", "3", None),
        ("Length[EdgeList[EdgeDelete[{a -> b, b <-> c, c -> d}, b <-> c]]]", "2", None),
        (None, None, None),
    ],
)
def test_edge_delete(str_expr, str_expected, msg):
    check_evaluation(str_expr, str_expected, failure_message=msg)
