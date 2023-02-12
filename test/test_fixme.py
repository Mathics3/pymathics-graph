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



@pytest.mark.skip("Wrong result. Investigate me.")
def test_EdgeList():
    check_evaluation(
     "EdgeList[{1 -> 2, 2 <-> 3}]",
     "{DirectedEdge[1, 2], UndirectedEdge[2, 3]}"
     )


    
@pytest.mark.skip("Wrong result. Investigate me.")
def test_FindShortestPath():
     check_evaluation(
         (
             "g = Graph[{1 -> 2, 2 -> 3, 1 -> 3}, EdgeWeight -> {0.5, a, 3}];"
             "a = 0.5; FindShortestPath[g, 1, 3]'"),
         "{1, 2, 3}")
     check_evaluation("a = 10; FindShortestPath[g, 1, 3]", "{1, 3}")
     check_evaluation("a = .;", "Null")




@pytest.mark.skip("This finds d<->a in the position 4 instead 2.")
def test_EdgeIndex():
    check_evaluation("EdgeIndex[{c <-> d, d <-> a, a -> e}, d <-> a]",
                     "2")

    
@pytest.mark.parametrize(
    ("str_expr", "str_expect"),
    [
        (
            ("g = Graph[{1 -> 2, 2 -> 3}, DirectedEdges -> True];"
             "EdgeCount[g, _DirectedEdge]"), 
            "2"
        ),
        (
            (
                "g = Graph[{1 -> 2, 2 -> 3}, DirectedEdges -> False];"
                "EdgeCount[g, _DirectedEdge]"),
            "0"
        ),
        ("EdgeCount[g, _UndirectedEdge]","2")
    ]
)
@pytest.mark.skip("This finds d<->a in the position 4 instead 2.")
def test_edgecount(str_expr, str_expect):
    check_evaluation(str_expr, str_expect)

    
@pytest.mark.skip("This finds d<->a in the position 4 instead 2.")
def test_EdgeIndex():
    check_evaluation("EdgeIndex[{c <-> d, d <-> a, a -> e}, d <-> a]",
                     "2")

@pytest.mark.skip("This is not properly evaluated. Investigate me")
def test_HITSCentrality():
     check_evaluation("g = Graph[{a -> d, b -> c, d -> c, d -> a, e -> c}]; HITSCentrality[g]",
     "{{0.292893, 0., 0., 0.707107, 0.}, {0., 1., 0.707107, 0., 0.707107}}")

@pytest.mark.skip("Investigate me.")
def test_EdgeRules():
    check_evaluation(
        "EdgeRules[{1 <-> 2, 2 -> 3, 3 <-> 4}]",
     "{1 -> 2, 2 -> 3, 3 -> 4}"
    )
     
