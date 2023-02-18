"""
Unit tests for pymathics.graph.measures_and_metrics
"""

from test.helper import check_evaluation, evaluate_value


def setup_module(module):
    """Load pymathics.graph"""
    assert evaluate_value('LoadModule["pymathics.graph"]') == "pymathics.graph"


def test_AcyclicQ():
    for str_expr, str_expected, mess in [
        (
            "AcyclicGraphQ[Graph[{1 -> 2, 2 -> 3, 5 -> 2, 3 -> 4, 5 -> 3}]]",
            "True",
            None,
        ),
        (
            "AcyclicGraphQ[Graph[{1 -> 2, 2 -> 3, 5 -> 2, 3 -> 4, 5 <-> 3}]]",
            "False",
            None,
        ),
        (
            "AcyclicGraphQ[Graph[{1 <-> 2, 2 <-> 3, 5 <-> 2, 3 <-> 4, 5 <-> 3}]]",
            "False",
            None,
        ),
        # (
        #     "AcyclicGraphQ[Graph[{}]]",
        #     "False",
        #     None,
        # ),
        (
            'AcyclicGraphQ["abc"]',
            "False",
            ["Expected a graph at position 1 in AcyclicGraphQ[abc]."],
        ),
    ]:
        check_evaluation(str_expr, str_expected, expected_messages=mess)
