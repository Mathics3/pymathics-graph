"""
Unit tests for pymathics.graph.measures_and_metrics
"""

from test.helper import check_evaluation


def test_graph_distance():
    for str_expr, str_expected, mess in [
        (
            "GraphDistance[{1 <-> 2, 2 <-> 3, 3 <-> 4, 2 <-> 4, 4 -> 5}, 1, 5]",
            "3",
            None,
        ),
        ("GraphDistance[{1 <-> 2, 2 <-> 3, 3 <-> 4, 4 -> 2, 4 -> 5}, 1, 5]", "4", None),
        (
            "GraphDistance[{1 <-> 2, 2 <-> 3, 4 -> 3, 4 -> 2, 4 -> 5}, 1, 5]",
            "Infinity",
            None,
        ),
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
