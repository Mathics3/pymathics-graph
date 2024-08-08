# -*- coding: utf-8 -*-
"""
Graph Components and Connectivity
"""

from typing import Optional

import networkx as nx
from mathics.core.convert.expression import to_mathics_list
from mathics.core.evaluation import Evaluation
from mathics.core.list import ListExpression

from pymathics.graph.base import _NetworkXBuiltin


class ConnectedComponents(_NetworkXBuiltin):
    """
    <url>
    :Strongly connected components:
    https://en.wikipedia.org/wiki/Strongly_connected_component</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/algorithms/\
generated/networkx.algorithms.components.strongly_connected_components.html</url>, <url>
:WMA:https://reference.wolfram.com/language/ref/ConnectedComponents.html</url>)

    <dl>
      <dt>'ConnectedComponents'[$g$]
      <dd> gives the connected components of the graph $g$.
    </dl>

    >> g = Graph[{1 -> 2, 2 -> 3, 3 <-> 4}, VertexLabels->True]
     = -Graph-

    >> ConnectedComponents[g]
     = ...

    >> g = Graph[{1 -> 2, 2 -> 3, 3 -> 1}, VertexLabels->True]
     = -Graph-

    >> ConnectedComponents[g]
     = ...

    >> g = Graph[{1 <-> 2, 2 <-> 3, 3 -> 4, 4 <-> 5}, VertexLabels->True]
     = -Graph-

    >> ConnectedComponents[g]
     = ...
    """

    summary_text = "list the connected components"

    def eval(
        self, graph, expression, evaluation: Evaluation, options: dict
    ) -> Optional[ListExpression]:
        "ConnectedComponents[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            connect_fn = (
                nx.strongly_connected_components
                if graph.G.is_directed()
                else nx.connected_components
            )
            components = [to_mathics_list(*sorted(c)) for c in connect_fn(graph.G)]
            return ListExpression(*components)


# This goes in Path, Cycles, Flows
# class FindHamiltonianPath(_NetworkXBuiltin):
#     """
#     <dl>
#       <dt>'FindHamiltonianPath[$g$]'
#       <dd>returns a Hamiltonian path in the given tournament graph.
#       </dl>
#
#     """
#     def eval_(self, graph, expression, evaluation: Evaluation, options):
#         "FindHamiltonianPath[graph_, OptionsPattern[FindHamiltonPath]]"

#         graph = self._build_graph(graph, evaluation: Evaluation, options, expression)
#         if graph:
#             # FIXME: for this to work we need to fill in all O(n^2) edges as an adjacency matrix?
#             path = nx.algorithms.tournament.hamiltonian_path(graph.G)
#             if path:
#                 # int_path = map(Integer, path)
#                 return to_mathics_list(*path)


class WeaklyConnectedComponents(_NetworkXBuiltin):
    """
    <url>
    :Weak components:
    https://en.wikipedia.org/wiki/Weak_component</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/algorithms/\
generated/networkx.algorithms.components.weakly_connected_components.html</url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/WeaklyConnectedComponents.html</url>)

    <dl>
      <dt>'WeaklyConnectedComponents'[$g$]
      <dd> gives the weakly connected components of the graph $g$.
    </dl>

    >> g = Graph[{1 -> 2, 2 -> 3, 3 <-> 4}, VertexLabels->True]
     = -Graph-

    >> WeaklyConnectedComponents[g]
     = {{1, 2, 3, 4}}

    >> g = Graph[{1 -> 2, 2 -> 3, 3 -> 1}, VertexLabels->True]
     = -Graph-

    >> WeaklyConnectedComponents[g]
     = {{1, 2, 3}}

    >> g = Graph[{1 <-> 2, 2 <-> 3, 3 -> 4, 4 <-> 5, 6 <-> 7, 7 <-> 8}, VertexLabels->True]
     = -Graph-

    >> WeaklyConnectedComponents[g]
     = {{1, 2, 3, 4, 5}, {6, 7, 8}}
    """

    summary_text = "list the weakly connected components"

    def eval(self, graph, expression, evaluation: Evaluation, options):
        "WeaklyConnectedComponents[graph_, OptionsPattern[WeaklyConnectedComponents]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            components = nx.connected_components(graph.G.to_undirected())
            result = []
            for component in components:
                result.append(sorted(component))
            return to_mathics_list(*result)
