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
    <url>
    :Acyclic graph:
    https://en.wikipedia.org/wiki/Acyclic_graph</url> test (<url>
    :NetworkX:
    https://networkx.org/documentation/stable/reference/algorithms\
/generated/networkx.algorithms.cycles.find_cycle.html
    </url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/AcyclicGraphQ.html</url>)

    <dl>
      <dt>'AcyclicGraphQ'[$graph$]
      <dd>check if $graph$ is an acyclic graph.
    </dl>


    Create a directed graph with a cycle in it:

    >> g = Graph[{1 -> 2, 2 -> 3, 5 -> 2, 3 -> 4, 3 -> 5}, VertexLabels->True]
     = -Graph-

    >> AcyclicGraphQ[g]
     = False

    Remove a cycle edge:

    >> g = EdgeDelete[g, 5 -> 2]; EdgeList[g]
     = {{1, 2}, {2, 3}, {3, 4}, {3, 5}}

    >> AcyclicGraphQ[g]
     = True
    """

    summary_text = "test if is a graph is acyclic"

    def eval(self, graph, expression, evaluation, options):
        "AcyclicGraphQ[graph_, OptionsPattern[AcyclicGraphQ]]"
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
    <url>
    :Connected graph:
    https://en.wikipedia.org/wiki/Connectivity_(graph_theory)\
#Connected_vertices_and_graphs
    </url> test (<url>
    :NetworkX:
https://networkx.org/documentation/networkx-2.8.8/reference/algorithms\
/generated/networkx.algorithms.components.is_connected.html
    </url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/ConnectedGraphQ.html</url>)

    <dl>
      <dt>'ConnectedGraphQ'[$graph$]
      <dd>check if $graph$ is a connected graph.
    </dl>

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

    summary_text = "test if a graph is a connected"

    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            return from_python(is_connected(graph.G))
        else:
            return SymbolFalse


class DirectedGraphQ(_NetworkXBuiltin):
    """
    <url>
    :Directed graph:
    https://en.wikipedia.org/wiki/Directed_graph
    </url> test (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference\
/generated/networkx.classes.function.is_directed.html
    </url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/DirectedGraphQ.html</url>)

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

    summary_text = "test if a graph is directed"

    def eval(self, graph, expression, evaluation, options):
        "DirectedGraphQ[graph_, OptionsPattern[DirectedGraphQ]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            return from_python(graph.is_directed())
        else:
            return SymbolFalse


class LoopFreeGraphQ(_NetworkXBuiltin):
    """
    <url>
    :Loop-Free graph:
    https://en.wikipedia.org/wiki/Loop_(graph_theory)
    </url> test (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/\
generated/networkx.classes.function.nodes_with_selfloops.html
    </url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/LoopFreeGraphQ.html</url>)

    <dl>
      <dt>'LoopFreeGraphQ'[$graph$]
      <dd>True if $graph$ is a 'Graph' and the edges do not close any loop.
    </dl>

    >> g = Graph[{1 -> 2, 2 -> 3}]; LoopFreeGraphQ[g]
     = True

    >> g = Graph[{1 -> 2, 2 -> 3, 1 -> 1}]; LoopFreeGraphQ[g]
     = False

    #> g = Graph[{}]; LoopFreeGraphQ[{}]
     = False

    #> LoopFreeGraphQ["abc"]
     = False
    """

    summary_text = "test if a graph is loop free"

    def eval(self, graph, expression, evaluation, options):
        "LoopFreeGraphQ[graph_, OptionsPattern[LoopFreeGraphQ]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if not graph or graph.empty():
            return SymbolFalse

        return from_python(graph.is_loop_free())


class MixedGraphQ(_NetworkXBuiltin):
    """
    <url>
    :Mixed Graph:
    https://en.wikipedia.org/wiki/Mixed_graph</url> test (<url>:WMA:
    https://reference.wolfram.com/language/ref/MixedGraphQ.html</url>)

    <dl>
      <dt>'MixedGraphQ'[$graph$]
      <dd>returns 'True' if $graph$ is a 'Graph' with both directed and undirected edges, \
          and 'False' otherwise.
    </dl>

    >> MixedGraphQ[Graph[{1 -> 2, 2 -> 3}]]
     = False

    >> MixedGraphQ[Graph[{1 -> 2, 2 <-> 3}]]
     = True

    >> MixedGraphQ[Graph[{}]]
     = False

    >> MixedGraphQ["abc"]
     = False

    ## Add as pytests
    ## > g = Graph[{1 -> 2, 2 -> 3}]; MixedGraphQ[g]
    ##  = False
    ## > g = EdgeAdd[g, a <-> b]; MixedGraphQ[g]
    ##   = True
    ## > g = EdgeDelete[g, a <-> b]; MixedGraphQ[g]
    ## = False
    """

    summary_text = "test if a graph has directed and undirected edges"

    def eval(self, graph, expression, evaluation, options):
        "MixedGraphQ[graph_, OptionsPattern[MixedGraphQ]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            return from_python(graph.is_mixed_graph())
        return SymbolFalse


class MultigraphQ(_NetworkXBuiltin):
    """
    <url>
    :Multigraph:
    https://en.wikipedia.org/wiki/Multigraph</url> test (<url>
    :NetworkX:
https://networkx.org/documentation/networkx-2.8.8/reference/classes/multigraph.html</url>, \
    <url>
    :WMA:
    https://reference.wolfram.com/language/ref/MulitGraphQ.html</url>)


    <dl>
      <dt>'MultigraphQ'[$graph$]
      <dd>True if $graph$ is a 'Graph' and there vertices connected by more than \
    one edge.
    </dl>

    >> g = Graph[{1 -> 2, 2 -> 3}]; MultigraphQ[g]
     = False

    >> g = Graph[{1 -> 2, 2 -> 3, 1 -> 2}]; MultigraphQ[g]
     = True

    #> g = Graph[{}]; MultigraphQ[g]
     = False

    #> MultigraphQ["abc"]
     = False
    """

    summary_text = "test if a graph is a multi graph"

    def eval(self, graph, expression, evaluation, options):
        "MultigraphQ[graph_, OptionsPattern[MultigraphQ]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            return from_python(graph.is_multigraph())
        else:
            return SymbolFalse


class PathGraphQ(_NetworkXBuiltin):
    """
    <url>
    :Path graph:
    https://en.wikipedia.org/wiki/Path_graph
    </url> test (<url>
    :WMA:
    https://reference.wolfram.com/language/ref/PathGraphQ.html</url>)

    <dl>
      <dt>'LoopFreeGraphQ'[$graph$]
      <dd>True if $graph$ is a 'Graph' and it becomes disconnected by removing \
    a single edge.
    </dl>


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

    summary_text = "test if a graph is a path-like graph"

    def eval(self, graph, expression, evaluation, options):
        "PathGraphQ[graph_, OptionsPattern[PathGraphQ]]"
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
        https://en.wikipedia.org/wiki/Planar_graph</url> test (<url>
        :NetworkX:
        https://networkx.org/documentation/networkx-2.8.8/reference/algorithms/
    generated/networkx.algorithms.planarity.check_planarity.html</url>, <url>
        :WMA:
        https://reference.wolfram.com/language/ref/PlanaGraphQ.html</url>)


        <dl>
          <dt>'PlanarGraphQ'[$g$]
          <dd>Returns True if $g$ is a planar graph and False otherwise.
        </dl>

        >> PlanarGraphQ[CycleGraph[4]]
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
    summary_text = "test if a graph is planar"

    def eval(self, graph, expression, evaluation: Evaluation, options: dict):
        "Pymathics`PlanarGraphQ[graph_, OptionsPattern[PlanarGraphQ]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if not graph or graph.empty():
            return SymbolFalse
        is_planar, _ = nx.check_planarity(graph.G)
        return from_python(is_planar)


class SimpleGraphQ(_NetworkXBuiltin):
    """
    Simple (not multigraph) <url>
    :graph:
    https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)#Graph
    </url> test (<url>
    :WMA:
    https://reference.wolfram.com/language/ref/SimpleGraphQ.html</url>)

    <dl>
      <dt>'SimpleGraphQ'[$graph$]
      <dd>True if $graph$ is a 'Graph', loop-free and each pair of \
          vertices are connected at most by a single edge.
    </dl>

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

    summary_text = "test if a graph is simple (not multigraph)"

    def eval(self, graph, expression, evaluation, options):
        "SimpleGraphQ[graph_, OptionsPattern[LoopFreeGraphQ]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            if graph.empty():
                return SymbolTrue
            else:
                simple = graph.is_loop_free() and not graph.is_multigraph()
                return from_python(simple)
        else:
            return SymbolFalse
