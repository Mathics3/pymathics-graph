# -*- coding: utf-8 -*-
"""
Algorithms on Graphs.

networkx does all the heavy lifting.
"""

from mathics.core.convert.expression import to_mathics_list
from mathics.core.convert.python import from_python
from mathics.core.expression import Expression
from mathics.core.list import ListExpression
from mathics.core.symbols import SymbolFalse
from mathics.core.systemsymbols import SymbolDirectedInfinity

from pymathics.graph.__main__ import (
    DEFAULT_GRAPH_OPTIONS,
    SymbolDirectedEdge,
    SymbolUndirectedEdge,
    _create_graph,
    _NetworkXBuiltin,
    nx,
)


class ConnectedComponents(_NetworkXBuiltin):
    """
    ## >> g = Graph[{1 -> 2, 2 -> 3, 3 <-> 4}]; ConnectedComponents[g]
    ##  = {{4, 3}, {2}, {1}}

    ## >> g = Graph[{1 -> 2, 2 -> 3, 3 -> 1}]; ConnectedComponents[g]
    ## = {{1, 2, 3}}

    ## >> g = Graph[{1 <-> 2, 2 <-> 3, 3 -> 4, 4 <-> 5}]; ConnectedComponents[g]
    ##  = {{4, 5}, {1, 2, 3}}
    """

    def apply(self, graph, expression, evaluation, options):
        "ConnectedComponents[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            connect_fn = (
                nx.strongly_connected_components
                if graph.G.is_directed()
                else nx.connected_components
            )
            components = [to_mathics_list(*c) for c in connect_fn(graph.G)]
            return ListExpression(*components)


# class FindHamiltonianPath(_NetworkXBuiltin):
#     """
#     <dl>
#       <dt>'FindHamiltonianPath[$g$]'
#       <dd>returns a Hamiltonian path in the given tournament graph.
#       </dl>
#     """
#     def apply_(self, graph, expression, evaluation, options):
#         "%(name)s[graph_, OptionsPattern[%(name)s]]"

#         graph = self._build_graph(graph, evaluation, options, expression)
#         if graph:
#             # FIXME: for this to work we need to fill in all O(n^2) edges as an adjacency matrix?
#             path = nx.algorithms.tournament.hamiltonian_path(graph.G)
#             if path:
#                 # int_path = map(Integer, path)
#                 return to_mathics_list(*path)


class GraphDistance(_NetworkXBuiltin):
    """
    <dl>
      <dt>'GraphDistance[$g$, $s$, $t$]'
      <dd>returns the distance from source vertex $s$ to target vertex $t$ in the graph $g$.
    </dl>

    <dl>
      <dt>'GraphDistance[$g$, $s$]'
      <dd>returns the distance from source vertex $s$ to all vertices in the graph $g$.
    </dl>

    <dl>
      <dt>'GraphDistance[{$v$->$w$, ...}, ...]'
      <dd>use rules $v$->$w$ to specify the graph $g$
    </dl>

    >> GraphDistance[{1 <-> 2, 2 <-> 3, 3 <-> 4, 2 <-> 4, 4 -> 5}, 1, 5]
     = 3

    >> GraphDistance[{1 <-> 2, 2 <-> 3, 3 <-> 4, 4 -> 2, 4 -> 5}, 1, 5]
     = 4

    >> GraphDistance[{1 <-> 2, 2 <-> 3, 4 -> 3, 4 -> 2, 4 -> 5}, 1, 5]
     = Infinity

    >> GraphDistance[{1 <-> 2, 2 <-> 3, 3 <-> 4, 2 <-> 4, 4 -> 5}, 3]
     = {2, 1, 0, 1, 2}

    >> GraphDistance[{1 <-> 2, 3 <-> 4}, 3]
     = {Infinity, Infinity, 0, 1}

    #> GraphDistance[{}, 1, 1]
     : The vertex at position 2 in GraphDistance[{}, 1, 1] does not belong to the graph at position 1.
     = GraphDistance[{}, 1, 1]
    #> GraphDistance[{1 -> 2}, 3, 4]
     : The vertex at position 2 in GraphDistance[{1 -> 2}, 3, 4] does not belong to the graph at position 1.
     = GraphDistance[{1 -> 2}, 3, 4]
    """

    def apply_s(self, graph, s, expression, evaluation, options):
        "%(name)s[graph_, s_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            weight = graph.update_weights(evaluation)
            d = nx.shortest_path_length(graph.G, source=s, weight=weight)
            inf = Expression(SymbolDirectedInfinity, 1)
            return to_mathics_list(*[d.get(v, inf) for v in graph.vertices])

    def apply_s_t(self, graph, s, t, expression, evaluation, options):
        "%(name)s[graph_, s_, t_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if not graph:
            return
        G = graph.G
        if not G.has_node(s):
            self._not_a_vertex(expression, 2, evaluation)
        elif not G.has_node(t):
            self._not_a_vertex(expression, 3, evaluation)
        else:
            try:
                weight = graph.update_weights(evaluation)
                return from_python(
                    nx.shortest_path_length(graph.G, source=s, target=t, weight=weight)
                )
            except nx.exception.NetworkXNoPath:
                return Expression(SymbolDirectedInfinity, 1)


class FindSpanningTree(_NetworkXBuiltin):
    """
    <dl>
      <dt>'FindSpanningTree[$g$]'
      <dd>finds a spanning tree of the graph $g$.
    </dl>

    >> FindSpanningTree[CycleGraph[4]]
    """

    options = DEFAULT_GRAPH_OPTIONS

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            graph.update_weights(evaluation)
            SymbolDirectedEdge if graph.G.is_directed() else SymbolUndirectedEdge
            # FIXME: put in edge to Graph conversion function?
            edges = [
                Expression("UndirectedEdge", u, v)
                for u, v in nx.minimum_spanning_edges(graph.G, data=False)
            ]
            g = _create_graph(edges, [None] * len(edges), options)
            if not hasattr(g.G, "graph_layout"):
                if hasattr(graph.G, "graph_layout"):
                    g.G.graph_layout = graph.G.graph_layout
                else:
                    g.G.graph_layout = "tree"
            return g


class PlanarGraphQ(_NetworkXBuiltin):
    """
    <dl>
      <dd>PlanarGraphQ[g]
      <dd>Returns True if g is a planar graph and False otherwise.
    </dl>

    >> PlanarGraphQ[CycleGraph[4]]
    = True
    >> PlanarGraphQ[CompleteGraph[5]]
    = False
    """

    options = DEFAULT_GRAPH_OPTIONS

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if not graph:
            return SymbolFalse
        is_planar, _ = nx.check_planarity(graph.G)
        return from_python(is_planar)


class WeaklyConnectedComponents(_NetworkXBuiltin):
    """
    >> g = Graph[{1 -> 2, 2 -> 3, 3 <-> 4}]; WeaklyConnectedComponents[g]
     = {{1, 2, 3, 4}}

    >> g = Graph[{1 -> 2, 2 -> 3, 3 -> 1}]; WeaklyConnectedComponents[g]
     = {{1, 2, 3}}

    >> g = Graph[{1 <-> 2, 2 <-> 3, 3 -> 4, 4 <-> 5, 6 <-> 7, 7 <-> 8}]; WeaklyConnectedComponents[g]
     = {{1, 2, 3, 4, 5}, {6, 7, 8}}
    """

    def apply(self, graph, expression, evaluation, options):
        "WeaklyConnectedComponents[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            components = nx.connected_components(graph.G.to_undirected())
            result = []
            for component in components:
                result.append(sorted(component))
            return to_mathics_list(*result)
