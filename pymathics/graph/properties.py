"""
Graph Properties and Measurements
"""

import networkx as nx
from mathics.core.convert.python import from_python
from mathics.core.evaluation import Evaluation
from mathics.core.symbols import SymbolFalse, SymbolTrue

from pymathics.graph.base import (
    DEFAULT_GRAPH_OPTIONS,
    Graph,
    _NetworkXBuiltin,
    is_connected,
)


class AcyclicGraphQ(_NetworkXBuiltin):
    """
    >> g = Graph[{1 -> 2, 2 -> 3}]; AcyclicGraphQ[g]
     = True

    >> g = Graph[{1 -> 2, 2 -> 3, 5 -> 2, 3 -> 4, 3 -> 5}]; AcyclicGraphQ[g]
     = False

    #> g = Graph[{1 -> 2, 2 -> 3, 5 -> 2, 3 -> 4, 5 -> 3}]; AcyclicGraphQ[g]
     = True

    #> g = Graph[{1 -> 2, 2 -> 3, 5 -> 2, 3 -> 4, 5 <-> 3}]; AcyclicGraphQ[g]
     = False

    #> g = Graph[{1 <-> 2, 2 <-> 3, 5 <-> 2, 3 <-> 4, 5 <-> 3}]; AcyclicGraphQ[g]
     = False

    #> g = Graph[{}]; AcyclicGraphQ[{}]
     = False

    #> AcyclicGraphQ["abc"]
     = False
     : Expected a graph at position 1 in AcyclicGraphQ[abc].
    """

    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=False)
        if not graph or graph.empty():
            return SymbolFalse

        try:
            cycles = nx.find_cycle(graph.G)
        except nx.exception.NetworkXNoCycle:
            return SymbolTrue
        return from_python(not cycles)


class ConnectedGraphQ(_NetworkXBuiltin):
    """
    >> g = Graph[{1 -> 2, 2 -> 3}]; ConnectedGraphQ[g]
     = False

    >> g = Graph[{1 -> 2, 2 -> 3, 3 -> 1}]; ConnectedGraphQ[g]
     = True

    #> g = Graph[{1 -> 2, 2 -> 3, 2 -> 3, 3 -> 1}]; ConnectedGraphQ[g]
     = True

    #> g = Graph[{1 -> 2, 2 -> 3}]; ConnectedGraphQ[g]
     = False

    >> g = Graph[{1 <-> 2, 2 <-> 3}]; ConnectedGraphQ[g]
     = True

    >> g = Graph[{1 <-> 2, 2 <-> 3, 4 <-> 5}]; ConnectedGraphQ[g]
     = False

    #> ConnectedGraphQ[Graph[{}]]
     = True

    #> ConnectedGraphQ["abc"]
     = False
    """

    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            return from_python(is_connected(graph.G))
        else:
            return SymbolFalse


class DirectedGraphQ(_NetworkXBuiltin):
    """
    <dl>
      <dt>'DirectedGraphQ'[$graph$]
      <dd>True if $graph$ is a 'Graph' and all the edges are directed.
    </dl>

    >> g = Graph[{1 -> 2, 2 -> 3}]; DirectedGraphQ[g]
     = True

    >> g = Graph[{1 -> 2, 2 <-> 3}]; DirectedGraphQ[g]
     = False

    #> g = Graph[{}]; DirectedGraphQ[{}]
     = False

    #> DirectedGraphQ["abc"]
     = False
    """

    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            return from_python(graph.is_directed())
        else:
            return SymbolFalse


class LoopFreeGraphQ(_NetworkXBuiltin):
    """
    >> g = Graph[{1 -> 2, 2 -> 3}]; LoopFreeGraphQ[g]
     = True

    >> g = Graph[{1 -> 2, 2 -> 3, 1 -> 1}]; LoopFreeGraphQ[g]
     = False

    #> g = Graph[{}]; LoopFreeGraphQ[{}]
     = False

    #> LoopFreeGraphQ["abc"]
     = False
    """

    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if not graph or graph.empty():
            return SymbolFalse

        return from_python(graph.is_loop_free())


class MixedGraphQ(_NetworkXBuiltin):
    """
    >> g = Graph[{1 -> 2, 2 -> 3}]; MixedGraphQ[g]
     = False

    # Seems to not be implemented...
    # >> g = Graph[{1 -> 2, 2 <-> 3}]; MixedGraphQ[g]
    # = True

    #> g = Graph[{}]; MixedGraphQ[g]
     = False

    #> MixedGraphQ["abc"]
     = False

    # #> g = Graph[{1 -> 2, 2 -> 3}]; MixedGraphQ[g]
    #  = False
    # #> g = EdgeAdd[g, a <-> b]; MixedGraphQ[g]
    #  = True
    # #> g = EdgeDelete[g, a <-> b]; MixedGraphQ[g]
    # = False
    """

    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            return from_python(graph.is_mixed_graph())
        return SymbolFalse


class MultigraphQ(_NetworkXBuiltin):
    """
    >> g = Graph[{1 -> 2, 2 -> 3}]; MultigraphQ[g]
     = False

    >> g = Graph[{1 -> 2, 2 -> 3, 1 -> 2}]; MultigraphQ[g]
     = True

    #> g = Graph[{}]; MultigraphQ[g]
     = False

    #> MultigraphQ["abc"]
     = False
    """

    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            return from_python(graph.is_multigraph())
        else:
            return SymbolFalse


class PathGraphQ(_NetworkXBuiltin):
    """
    >> PathGraphQ[Graph[{1 -> 2, 2 -> 3}]]
     = True
    #> PathGraphQ[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]
     = True
    #> PathGraphQ[Graph[{1 <-> 2, 2 <-> 3}]]
     = True
    >> PathGraphQ[Graph[{1 -> 2, 2 <-> 3}]]
     = False
    >> PathGraphQ[Graph[{1 -> 2, 3 -> 2}]]
     = False
    >> PathGraphQ[Graph[{1 -> 2, 2 -> 3, 2 -> 4}]]
     = False
    >> PathGraphQ[Graph[{1 -> 2, 3 -> 2, 2 -> 4}]]
     = False

    #> PathGraphQ[Graph[{}]]
     = False
    #> PathGraphQ[Graph[{1 -> 2, 3 -> 4}]]
     = False
    #> PathGraphQ[Graph[{1 -> 2, 2 -> 1}]]
     = True
    >> PathGraphQ[Graph[{1 -> 2, 2 -> 3, 2 -> 3}]]
     = False
    #> PathGraphQ[Graph[{}]]
     = False
    #> PathGraphQ["abc"]
     = False
    #> PathGraphQ[{1 -> 2, 2 -> 3}]
     = False
    """

    def eval(self, graph, expression, evaluation, options):
        "PathGraphQ[graph_, OptionsPattern[%(name)s]]"
        if not isinstance(graph, Graph) or graph.empty():
            return SymbolFalse

        G = graph.G

        if G.is_directed():
            connected = nx.is_semiconnected(G)
        else:
            connected = nx.is_connected(G)

        if connected:
            is_path = all(d <= 2 for _, d in G.degree(graph.vertices))
        else:
            is_path = False

        return from_python(is_path)


class PlanarGraphQ(_NetworkXBuiltin):
    """
    <url>
    :Planar Graph:
    <url>https://en.wikipedia.org/wiki/Planar_graph</url> testing

    <dl>
      <dd>'PlanarGraphQ'[$g$]
      <dd>Returns True if $g$ is a planar graph and False otherwise.
    </dl>

    >> PlanarGraphQ[CycleGraph[4]]
    = True

    >> PlanarGraphQ[CompleteGraph[5]]
    = False

    >> PlanarGraphQ[CompleteGraph[4]]
     = True

    >> PlanarGraphQ[CompleteGraph[5]]
     = False

    #> PlanarGraphQ[Graph[{}]]
     = False


    >> PlanarGraphQ["abc"]
     : Expected a graph at position 1 in PlanarGraphQ[abc].
     = False
    """

    options = DEFAULT_GRAPH_OPTIONS

    def eval(self, graph, expression, evaluation: Evaluation, options: dict):
        "Pymathics`PlanarGraphQ[graph_, OptionsPattern[PlanarGraphQ]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if not graph or graph.empty():
            return SymbolFalse
        is_planar, _ = nx.check_planarity(graph.G)
        return from_python(is_planar)


class SimpleGraphQ(_NetworkXBuiltin):
    """
    >> g = Graph[{1 -> 2, 2 -> 3, 3 <-> 4}]; SimpleGraphQ[g]
     = True

    >> g = Graph[{1 -> 2, 2 -> 3, 1 -> 1}]; SimpleGraphQ[g]
     = False

    >> g = Graph[{1 -> 2, 2 -> 3, 1 -> 2}]; SimpleGraphQ[g]
     = False

    #> SimpleGraphQ[Graph[{}]]
     = True

    #> SimpleGraphQ["abc"]
     = False
    """

    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            if graph.empty():
                return SymbolTrue
            else:
                simple = graph.is_loop_free() and not graph.is_multigraph()
                return from_python(simple)
        else:
            return SymbolFalse
