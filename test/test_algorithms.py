# -*- coding: utf-8 -*-
"""
Unit tests for pymathics.graph.algorithms
"""
from test.helper import check_evaluation, evaluate, evaluate_value
import pytest

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

@pytest.mark.parametrize(
    ("str_expr", "str_expected","msg"),
    [
        (None, None, None),
        ('g = Graph[{a -> b, b <-> c, d -> c, d -> a, e <-> c, d -> b}];', "Null", "Intialize graph"),
        ("Sort[DegreeCentrality[g]]", "{2, 2, 3, 4, 5}", None),
        ('Sort[DegreeCentrality[g, "In"]]', "{0, 1, 1, 3, 3}", None),
        ('Sort[DegreeCentrality[g, "Out"]]','{1, 1, 1, 2, 3}', None),
        (None, None, None),
    ]
)
def test_degree_centrality(str_expr, str_expected, msg):
    check_evaluation(str_expr, str_expected, failure_message=msg)
    
        
def test_graph_distance():
    for str_expr, str_expected, mess in [
        (
            "GraphDistance[{1 <-> 2, 2 <-> 3, 3 <-> 4, 2 <-> 4, 4 -> 5}, 1, 5]",
            "3",
            None,
        ),
        ("GraphDistance[{1 <-> 2, 2 <-> 3, 3 <-> 4, 4 -> 2, 4 -> 5}, 1, 5]", "4", None),
        # ("GraphDistance[{1 <-> 2, 2 <-> 3, 4 -> 3, 4 -> 2, 4 -> 5}, 1, 5]", "Infinity"),
        (
            "Sort[GraphDistance[{1 <-> 2, 2 <-> 3, 3 <-> 4, 2 <-> 4, 4 -> 5}, 3]]",
            "{0, 1, 1, 2, 2}",
            None,
        ),
        (
            "GraphDistance[{}, 1, 1]",
            "GraphDistance[{}, 1, 1]",
            [
                "The vertex at position 2 in GraphDistance[{}, 1, 1] does not belong to "
                "the graph at position 1."
            ],
        ),
        (
            "GraphDistance[{1 -> 2}, 3, 4]",
            "GraphDistance[{1 -> 2}, 3, 4]",
            [
                "The vertex at position 2 in GraphDistance[{1 -> 2}, 3, 4] does not belong "
                "to the graph at position 1."
            ],
        ),
    ]:
        check_evaluation(str_expr, str_expected, expected_messages=mess)
