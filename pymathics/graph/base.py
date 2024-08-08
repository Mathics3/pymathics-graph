# -*- coding: utf-8 -*-

"""
Core routines for working with Graphs.
"""

# uses networkx

from collections import defaultdict
from inspect import isgenerator
from typing import Callable, Optional, Union

from mathics.core.builtin import AtomBuiltin, Builtin
from mathics.core.atoms import Atom, Integer, Integer0, Integer1, Integer2, String
from mathics.core.convert.expression import ListExpression, from_python, to_mathics_list
from mathics.core.element import BaseElement
from mathics.core.expression import Expression
from mathics.core.pattern import pattern_objects
from mathics.core.symbols import Symbol, SymbolList, SymbolTrue
from mathics.core.systemsymbols import (
    SymbolBlank,
    SymbolCases,
    SymbolFailed,
    SymbolMissing,
    SymbolRGBColor,
    SymbolRule,
)
from mathics.eval.patterns import Matcher

from pymathics.graph.boxes import GraphBox
from pymathics.graph.collections import _EdgeCollection
from pymathics.graph.graphsymbols import (
    SymbolDirectedEdge,
    SymbolGraph,
    SymbolTwoWayRule,
    SymbolUndirectedEdge,
)


WL_MARKER_TO_NETWORKX = {
    "Circle": "o",
    "Diamond": "D",
    "Square": "s",
    "Star": "*",
    "Pentagon": "p",
    "Octagon": "8",
    "Hexagon": "h",
    "Triangle": "^",
    # And many others. Is there a list somewhere?
}

WL_COLOR_TO_NETWORKX = {
    "Green": "g",
    "Blue": "b",
    # And many others. Is there a list somewhere?
}

WL_LAYOUT_TO_NETWORKX = {
    "CircularEmbedding": "circular",
    "SpiralEmbedding": "spiral",
    "SpectralEmbedding": "spectral",
    "SpringEmbedding": "spring",
    # And many others. Is there a list somewhere?
}

DEFAULT_GRAPH_OPTIONS = {
    "DirectedEdges": "False",
    "EdgeStyle": "{}",
    "EdgeWeight": "{}",
    "GraphLayout": "Null",
    "PlotLabel": "Null",
    "VertexLabels": "False",
    "VertexSize": "{}",
    "VertexShape": '"Circle"',
    "VertexStyle": "{}",
}

import networkx as nx


def graph_helper(
    graph_generator_func: Callable,
    options: dict,
    can_digraph: bool,
    graph_layout: Optional[str],
    evaluation,
    root: Optional[int] = None,
    *args,
    **kwargs,
) -> Optional["Graph"]:
    """
    This is a wrapping function for a NetworkX function of some sort indicated in ``graph_generator``,
    e.g. ``nx.complete_graph``.

    ``args`` and ``kwargs`` are passed to the NetworkX function, while ``can_digraph`` determines
    whether ``create_using=nx.DiGraph``is added to the NetworkX graph-generation function call.

    Parameter ``options`` and ``graph_layout`` are Mathics3 Graph
    options; ``graph_layout`` has a ``String()`` wrapped around it
    ``root`` is used when the graph is a tree.
    """
    should_digraph = can_digraph and has_directed_option(options)
    try:
        G = (
            graph_generator_func(*args, create_using=nx.DiGraph, **kwargs)
            if should_digraph
            else graph_generator_func(*args, **kwargs)
        )
    except MemoryError:
        evaluation.message("Graph", "mem", evaluation)
        return None
    if (
        graph_layout and not options["System`GraphLayout"].get_string_value()
        if "System`GraphLayout" in options
        else False
    ):
        options["System`GraphLayout"] = String(graph_layout)

    g = Graph(G)
    _process_graph_options(g, options)

    if root is not None:
        G.root = g.root = root
    return g


def has_directed_option(options: dict) -> bool:
    return options.get("System`DirectedEdges", False)


def _process_graph_options(g, options: dict) -> None:
    """
    Handle common graph-related options like VertexLabels, PlotLabel, VertexShape, etc.
    """
    # FIXME: for now we are adding both to both g and g.G.
    # g is where it is used in format. However we should wrap this as our object.
    # Access in G which might be better, currently isn't used.
    g.G.vertex_labels = g.vertex_labels = (
        options["System`VertexLabels"].to_python()
        if "System`VertexLabels" in options
        else False
    )
    shape = (
        options["System`VertexShape"].get_string_value()
        if "System`VertexShape" in options
        else "Circle"
    )

    g.G.node_shape = g.node_shape = WL_MARKER_TO_NETWORKX.get(shape, shape)

    color = (
        options["System`VertexStyle"].get_string_value()
        if "System`VertexStyle" in options
        else "Blue"
    )

    g.graph_layout = (
        options["System`GraphLayout"].get_string_value()
        if "System`GraphLayout" in options
        else ""
    )

    g.G.graph_layout = g.graph_layout = WL_LAYOUT_TO_NETWORKX.get(
        g.graph_layout, g.graph_layout
    )

    g.G.node_color = g.node_color = WL_COLOR_TO_NETWORKX.get(color, color)

    g.G.title = g.title = (
        options["System`PlotLabel"].get_string_value()
        if "System`PlotLabel" in options
        else None
    )


def _circular_layout(G):
    return nx.drawing.circular_layout(G, scale=1)


def _spectral_layout(G):
    return nx.drawing.spectral_layout(G, scale=2)


def _shell_layout(G):
    return nx.drawing.shell_layout(G, scale=2)


def _generic_layout(G, warn):
    return nx.nx_pydot.graphviz_layout(G, prog="dot")


def _path_layout(G, root):
    v = root
    x = 0
    y = 0

    k = 0
    d = 0

    pos = {}
    neighbors = G.neighbors(v)

    for _ in range(len(G)):
        pos[v] = (x, y)

        if not neighbors:
            break
        try:
            v = next(neighbors) if isgenerator(neighbors) else neighbors[0]
        except StopIteration:
            break
        neighbors = G.neighbors(v)

        if k == 0:
            if d < 1 or neighbors:
                d += 1
            x += d
        elif k == 1:
            y += d
        elif k == 2:
            if neighbors:
                d += 1
            x -= d
        elif k == 3:
            y -= d

        k = (k + 1) % 4

    return pos


def _auto_layout(G, warn):
    path_root = None

    for v, d in G.degree(G.nodes):
        if d == 1 and G.neighbors(v):
            path_root = v
        elif d > 2:
            path_root = None
            break

    if path_root is not None:
        return _path_layout(G, path_root)
    else:
        return _generic_layout(G, warn)


def _convert_networkx_graph(G, options):
    mapping = dict((v, Integer(i)) for i, v in enumerate(G.nodes))
    G = nx.relabel_nodes(G, mapping)
    [Expression(SymbolUndirectedEdge, u, v) for u, v in G.edges]
    return Graph(
        G,
        **options,
    )


_default_minimum_distance = 0.3


def _vertex_style(expr):
    return expr


def _edge_style(expr):
    return expr


def _parse_property(expr, attr_dict=None):
    if expr.has_form("Rule", 2):
        name, value = expr.elements
        if isinstance(name, Symbol):
            if attr_dict is None:
                attr_dict = {}
            attr_dict[name.get_name()] = value
    elif expr.has_form("List", None):
        for element in expr.elements:
            attr_dict = _parse_property(element, attr_dict)
    return attr_dict


class _NetworkXBuiltin(Builtin):
    requires = ("networkx",)

    options = DEFAULT_GRAPH_OPTIONS

    messages = {
        "graph": "Expected a graph at position 1 in ``.",
        "inv": "The `1` at position `2` in `3` does not belong to the graph at position 1.",
    }

    def _not_a_vertex(self, expression, pos, evaluation):
        evaluation.message(self.get_name(), "inv", "vertex", pos, expression)

    def _not_an_edge(self, expression, pos, evaluation):
        evaluation.message(self.get_name(), "inv", "edge", pos, expression)

    def _build_graph(
        self,
        graph: Union["Graph", Expression],
        evaluation,
        options: dict,
        expr,
        quiet=False,
    ) -> Optional["Graph"]:
        head = graph.get_head()
        if head is SymbolGraph and isinstance(graph, Atom) and hasattr(graph, "G"):
            return graph
        elif head is SymbolList:
            return _graph_from_list(graph.elements, options)
        elif not quiet:
            evaluation.message(self.get_name(), "graph", expr)

    def _evaluate_atom(self, graph, options, compute):
        head = graph.head
        if head is SymbolGraph:
            return compute(graph)
        elif head is SymbolList:
            return compute(_graph_from_list(graph.elements, options))

    def __str__(self):
        return "-Graph-"

    def get_sort_key(self, pattern_sort=False) -> tuple:
        """
        Returns a particular encoded list (which should be a tuple) that is used
        in ``Sort[]`` comparisons and in the ordering that occurs
        in an M-Expression which has the ``Orderless`` property.

        See the docstring for element.get_sort_key() for more detail.
        """

        if pattern_sort:
            return super(_NetworkXBuiltin, self).get_sort_key(True)
        else:
            # Return a sort_key tuple.
            # but with a `2` instead of `1` in the 5th position,
            # and adding two extra fields: the length in the 5th position,
            # and a hash in the 6th place.
            return [
                1,
                3,
                self.class_head_name,
                tuple(),
                2,
                len(self.vertices),
                hash(self),
            ]
            return hash(self)


class _FullGraphRewrite(Exception):
    pass


def _normalize_edges(edges):
    for edge in edges:
        if edge.has_form("System`Property", 2):
            expr, prop = edge.elements
            yield Expression(edge.get_head(), list(_normalize_edges([expr]))[0], prop)
        elif edge.get_head_name() == "System`Rule":
            yield Expression(SymbolDirectedEdge, *edge.elements)
        else:
            yield edge


def is_connected(G):
    if len(G) == 0:  # empty graph?
        return True
    elif G.is_directed():
        return nx.is_strongly_connected(G)
    else:
        return nx.is_connected(G)


def _edge_weights(options):
    expr = options.get("Pymathics`EdgeWeight")
    if expr is None:
        return []
    if not expr.has_form("List", None):
        return []
    return expr.elements


class _GraphParseError(Exception):
    def __init__(self, msg=""):
        self.msg = msg

    def __repr__(self):
        if self.msg:
            return "GraphParseError: " + self.msg
        else:
            return "GraphParseError."


class Graph(Atom):
    class_head_name = "Pymathics`Graph"

    options = DEFAULT_GRAPH_OPTIONS

    def __init__(self, Gr, **kwargs):
        super(Graph, self).__init__()
        self.G = Gr
        self.mixed = kwargs.get("mixed", False)

        # Here we define types that appear on some, but not all
        # graphs. So these are optional, which we given an initial
        # value of None

        # Some graphs has a second int parameter like HknHarary
        self.n: Optional[int] = None

        # Trees have a root
        self.root: Optional[int] = None

    def __hash__(self):
        return hash(("Pymathics`Graph", self.G))

    def __str__(self):
        return "-Graph-"

    def atom_to_boxes(self, f, evaluation) -> "GraphBox":
        return GraphBox(self.G)

    def add_edges(self, new_edges, new_edge_properties):
        G = self.G.copy()
        mathics_new_edges = list(_normalize_edges(new_edges))
        return _create_graph(
            mathics_new_edges, new_edge_properties, options={}, from_graph=G
        )

    def add_vertices(self, *vertices_to_add):
        G = self.G.copy()
        G.add_nodes_from(vertices_to_add)
        return Graph(G)

    def coalesced_graph(self, evaluation):
        if not isinstance(self.G, (nx.MultiDiGraph, nx.MultiGraph)):
            return self.G, "WEIGHT"

        new_edges = defaultdict(lambda: 0)
        for u, v, w in self.G.edges.data("Pymathics`EdgeWeight", default=None):
            if w is not None:
                w = float(w.evaluate(evaluation).to_mpmath())
            else:
                w = 1
            new_edges[(u, v)] += w

        if self.G.is_directed():
            new_graph = nx.DiGraph()
        else:
            new_graph = nx.Graph()

        new_graph.add_edges_from(
            ((u, v, {"WEIGHT": w}) for (u, v), w in new_edges.items())
        )

        return new_graph, "WEIGHT"

    def delete_edges(self, edges_to_delete) -> "Graph":
        G = self.G.copy()
        directed = G.is_directed()

        normalized_edges_to_delete = list(_normalize_edges(edges_to_delete))

        for edge in normalized_edges_to_delete:
            if edge.has_form("DirectedEdge", 2):
                if directed:
                    u, v = edge.elements
                    G.remove_edge(u, v)
            elif edge.has_form("UndirectedEdge", 2):
                u, v = edge.elements
                if directed:
                    G.remove_edge(u, v)
                    G.remove_edge(v, u)
                else:
                    G.remove_edge(u, v)

        return Graph(G)

    def delete_vertices(self, vertices_to_delete):
        G = self.G.copy()
        for n in vertices_to_delete:
            G.remove_node(n)
        return Graph(G)

    def default_format(self, evaluation, form):
        return "-Graph-"

    def do_format(self, evaluation, form):
        return self

    @property
    def edges(self) -> tuple:
        # TODO: check if this should not return expressions
        return self.G.edges

    def empty(self):
        return len(self.G) == 0

    @property
    def head(self):
        return SymbolGraph

    def is_directed(self):
        if self.G.is_directed():
            return not self.mixed
        return False

    def is_loop_free(self):
        return not any(True for _ in nx.nodes_with_selfloops(self.G))

    # networkx graphs can't be used for mixed
    def is_mixed_graph(self):
        return self.mixed
        # return self.edges. ... is_mixed()

    def is_multigraph(self):
        return isinstance(self.G, (nx.MultiDiGraph, nx.MultiGraph))

    def get_sort_key(self, pattern_sort=False) -> tuple:
        """
        Returns a particular encoded list (which should be a tuple) that is used
        in ``Sort[]`` comparisons and in the ordering that occurs
        in an M-Expression which has the ``Orderless`` property.

        See the docstring for element.get_sort_key() for more detail.
        """

        if pattern_sort:
            return super(Graph, self).get_sort_key(True)
        else:
            # Return a sort_key tuple.
            # but with a `2` instead of `1` in the 5th position,
            # and adding two extra fields: the length in the 5th position,
            # and a hash in the 6th place.
            return [
                1,
                3,
                self.class_head_name,
                tuple(),
                2,
                len(self.vertices),
                hash(self),
            ]

    def sort_vertices(self, vertices):
        return sorted(vertices)

    def update_weights(self, evaluation):
        weights = None
        G = self.G

        if self.is_multigraph():
            for u, v, k, w in G.edges.data(
                "Pymathics`EdgeWeight", default=None, keys=True
            ):
                data = G.get_edge_data(u, v, key=k)
                w = data.get()
                if w is not None:
                    w = w.evaluate(evaluation).to_mpmath()
                    G[u][v][k]["WEIGHT"] = w
                    weights = "WEIGHT"
        else:
            for u, v, w in G.edges.data("Pymathics`EdgeWeight", default=None):
                if w is not None:
                    w = w.evaluate(evaluation).to_mpmath()
                    G[u][v]["WEIGHT"] = w
                    weights = "WEIGHT"

        return weights

    @property
    def value(self):
        return self.G

    @property
    def vertices(self):
        return self.G.nodes


def _parse_item(x, attr_dict=None):
    if x.has_form("Property", 2):
        expr, prop = x.elements
        attr_dict = _parse_property(prop, attr_dict)
        return _parse_item(expr, attr_dict)
    else:
        return x, attr_dict


def _graph_from_list(rules, options, new_vertices=None):
    if not rules:
        return Graph(nx.Graph())
    else:
        new_edges, new_edge_properties = zip(*[_parse_item(x) for x in rules])
        return _create_graph(
            new_edges, new_edge_properties, options=options, new_vertices=new_vertices
        )


def _create_graph(
    new_edges, new_edge_properties, options, from_graph=None, new_vertices=None
):
    vertices_dict = {}
    # Classification of vertex and edges
    known_vertices = set()
    vertices = []
    vertex_properties = []

    def add_vertex(x, attr_dict=None):
        if x.has_form("Property", 2):
            expr, prop = x.elements
            attr_dict = _parse_property(prop, attr_dict)
            return add_vertex(expr, attr_dict)
        elif x not in known_vertices:
            known_vertices.add(x)
            vertices.append(x)
            vertex_properties.append(attr_dict)
        return x

    directed_edges = []
    undirected_edges = []

    if from_graph is not None:
        old_vertices = dict(from_graph.nodes.data())
        vertices += old_vertices
        edges = list(from_graph.edges.data())

        for edge, attr_dict in edges:
            u, v = edge.elements
            if edge.head in (SymbolDirectedEdge, SymbolRule):
                directed_edges.append((u, v, attr_dict))
            else:
                undirected_edges.append((u, v, attr_dict))

        multigraph = [from_graph.is_multigraph()]
    else:
        edges = []
        edge_properties = []

        multigraph = [False]

    if new_vertices is not None:
        for v in new_vertices:
            add_vertex(v)

    def add_vertex(x, attr_dict=None):
        if attr_dict is None:
            attr_dict = {}
        if x.has_form("Pymathics`Property", 2):
            expr, prop = x.elements
            attr_dict.update(_parse_property(prop, attr_dict))
            return add_vertex(expr, attr_dict)
        elif x not in known_vertices:
            known_vertices.add(x)
            vertices.append(x)
            vertex_properties.append(attr_dict)
            vertices_dict[x] = attr_dict
        else:
            vertices_dict[x].update(attr_dict)
        return x

    if new_vertices is not None:
        for v in new_vertices:
            add_vertex(v)

    known_edges = set(edges)
    # It is simpler to just recompute this than change the above to work
    # incrementally
    known_vertices = set(vertices)

    def track_edges(*edges):
        if multigraph[0]:
            return
        previous_n_edges = len(known_edges)
        for edge in edges:
            known_edges.add(edge)
        if len(known_edges) < previous_n_edges + len(edges):
            multigraph[0] = True

    edge_weights = _edge_weights(options)
    use_directed_edges = options.get("System`DirectedEdges", SymbolTrue) is SymbolTrue

    directed_edge_head = (
        SymbolDirectedEdge if use_directed_edges else SymbolUndirectedEdge
    )

    def parse_edge(r, attr_dict=None):
        if attr_dict is None:
            attr_dict = {}

        if isinstance(r, Atom):
            raise _GraphParseError(
                msg=f"{r} is an atom, and hence does not define an edge."
            )

        if r.has_form("Pymathics`Property", None):
            expr, prop = r.elements
            attr_dict.update(_parse_property(prop, attr_dict))
            return parse_edge(expr, attr_dict)

        if r.head not in (
            SymbolRule,
            SymbolDirectedEdge,
            SymbolTwoWayRule,
            SymbolUndirectedEdge,
        ):
            raise _GraphParseError(msg=f"{r} is not an edge description.")

        r_head = r.head
        elements = r.elements

        if len(elements) != 2:
            raise _GraphParseError(
                msg=f"{r} does not have 2 elements, so it is not an edge."
            )

        u, v = elements
        assert isinstance(u, BaseElement) and isinstance(v, BaseElement)
        u = add_vertex(u)
        v = add_vertex(v)

        if r_head in (SymbolRule, SymbolDirectedEdge):
            edges_container = directed_edges
            head = directed_edge_head
            track_edges((u, v))
        elif r_head in (SymbolTwoWayRule, SymbolUndirectedEdge):
            edges_container = undirected_edges
            head = SymbolUndirectedEdge
            track_edges((u, v), (v, u))
        else:
            raise _GraphParseError(msg=f"{r_head} is an unknown kind of edge.")

        if r_head is head:
            edges.append(r)
        else:
            edges.append(Expression(head, u, v))
        edge_properties.append(attr_dict)

        edges_container.append((u, v, attr_dict))

    try:

        def full_new_edge_properties(new_edge_style):
            for i, (attr_dict, w) in enumerate(zip(new_edge_properties, edge_weights)):
                attr_dict = {} if attr_dict is None else attr_dict.copy()
                attr_dict["Pymathics`EdgeWeight"] = from_python(w)
                yield attr_dict
            # FIXME: figure out what to do here. Color is a mess.
            # for i, (attr_dict, s) in enumerate(zip(new_edge_style, new_edge_style)):
            #     attr_dict = {} if attr_dict is None else attr_dict.copy()
            #     attr_dict["Pymathics`EdgeStyle"] = s
            #     yield attr_dict
            for attr_dict in new_edge_properties[len(edge_weights) :]:
                yield attr_dict

        if "Pymathics`EdgeStyle" in options:
            # FIXME: Figure out what to do here:
            # Color is a f-ing mess.
            # edge_options = options["Pymathics`EdgeStyle"].to_python()
            edge_options = []
        else:
            edge_options = []
        edge_properties = list(full_new_edge_properties(edge_options))
        for edge, attr_dict in zip(new_edges, edge_properties):
            parse_edge(edge, attr_dict)
    except _GraphParseError:
        return None

    empty_dict = {}
    mixed = False
    if directed_edges:
        G = nx.MultiDiGraph() if multigraph[0] else nx.DiGraph()
        nodes_seen = set()
        for u, v, attr_dict in directed_edges:
            attr_dict = attr_dict or empty_dict
            G.add_edge(u, v, **attr_dict)
            nodes_seen.add(u)
            nodes_seen.add(v)

        unseen_vertices = set(vertices) - nodes_seen
        for v in unseen_vertices:
            G.add_node(v)

        if undirected_edges:
            mixed = True
        for u, v, attr_dict in undirected_edges:
            attr_dict = attr_dict or empty_dict
            G.add_edge(u, v, **attr_dict)
            G.add_edge(v, u, **attr_dict)
    else:
        G = nx.MultiGraph() if multigraph[0] else nx.Graph()
        for u, v, attr_dict in undirected_edges:
            attr_dict = attr_dict or empty_dict
            G.add_edge(u, v, **attr_dict)

    # For what is this?
    _EdgeCollection(
        edges,
        edge_properties,
        n_directed=len(directed_edges),
        n_undirected=len(undirected_edges),
    )

    g = Graph(G, mixed=mixed)
    _process_graph_options(g, options)
    return g


class _PatternList(_NetworkXBuiltin):
    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return ListExpression(*(from_python(q) for q in self._items(graph)))

    def eval_patt(self, graph, patt, expression, evaluation, options):
        "%(name)s[graph_, patt_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return Expression(SymbolCases, ListExpression(*self._items(graph)), patt)


class AdjacencyList(_NetworkXBuiltin):
    """
    <url>
    :Adjacency list:
    https://en.wikipedia.org/wiki/Adjacency_list</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/readwrite/adjlist.html</url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/AdjacencyList.html</url>)

    <dl>
      <dt>'AdjacencyList'[$graph$, $v$]
      <dd>gives a list of vertices adjacent to $v$ in a 'Graph' \
          or a list of edges $g$.

      <dt>'AdjacencyList'[$graph$, $pattern$]
      <dd>gives a list of vertices adjacent to vertices \
          matching $pattern$.
    </dl>

    >> AdjacencyList[{1 -> 2, 2 -> 3}, 3]
     = {2}

    >> AdjacencyList[{1 -> 2, 2 -> 3}, _?EvenQ]
     = {1, 3}

    >> AdjacencyList[{x -> 2, x -> 3, x -> 4, 2 -> 10, 2 -> 11, 4 -> 20, 4 -> 21, 10 -> 100}, 10, 2]
     = {2, 11, 100, x}
    """

    summary_text = "list the adjacent vertices"

    def _retrieve(self, graph, what, neighbors, expression, evaluation):
        if what.get_head_name() in pattern_objects:
            collected = set()
            match = Matcher(what).match
            for v in graph.G.nodes:
                if match(v, evaluation):
                    collected.update(neighbors(v))
            return ListExpression(*sorted(collected))
        elif graph.G.has_node(what):
            return ListExpression(*sorted(neighbors(what)))
        else:
            self._not_a_vertex(expression, 2, evaluation)

    def eval(self, graph, what, expression, evaluation, options):
        "%(name)s[graph_, what_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            G = graph.G.to_undirected()  # FIXME inefficient
            return self._retrieve(
                graph, what, lambda v: G.neighbors(v), expression, evaluation
            )

    def eval_d(self, graph, what, d, expression, evaluation, options):
        "%(name)s[graph_, what_, d_, OptionsPattern[%(name)s]]"
        py_d = d.to_mpmath()
        if py_d is None:
            return

        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            G = graph.G

            def neighbors(v):
                return nx.ego_graph(
                    G, v, radius=py_d, undirected=True, center=False
                ).nodes()

            return self._retrieve(graph, what, neighbors, expression, evaluation)


class DirectedEdge(Builtin):
    """
    Edge of a <url>
    :Directed graph:
    https://en.wikipedia.org/wiki/Directed_graph</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/classes/digraph.html</url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/DirectedEdge.html</url>)


    <dl>
      <dt>'DirectedEdge[$u$, $v$]'
      <dd>create a directed edge from $u$ to $v$.
    </dl>
    """

    summary_text = "make a directed graph edge"


class EdgeConnectivity(_NetworkXBuiltin):
    """
    <url>
    :Edge connectivity:
    https://en.wikipedia.org/wiki/Directed_graph#Directed_graph_connectivity</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/algorithms/\
generated/networkx.algorithms.connectivity.connectivity.edge_connectivity.html</url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/EdgeConnectivity.html</url>)

    <dl>
      <dt>'EdgeConnectivity[$g$]'
      <dd>gives the edge connectivity of the graph $g$.
    </dl>

    >> EdgeConnectivity[{1 <-> 2, 2 <-> 3}]
     = 1

    >> EdgeConnectivity[{1 -> 2, 2 -> 3}]
     = 0

    >> EdgeConnectivity[{1 -> 2, 2 -> 3, 3 -> 1}]
     = 1

    >> EdgeConnectivity[{1 <-> 2, 2 <-> 3, 1 <-> 3}]
     = 2

    >> EdgeConnectivity[{1 <-> 2, 3 <-> 4}]
     = 0

    #> EdgeConnectivity[Graph[{}]]
     = EdgeConnectivity[-Graph-]
    """

    summary_text = "edge connectivity of a graph"

    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph and not graph.empty():
            return Integer(nx.edge_connectivity(graph.G))

    def eval_st(self, graph, s, t, expression, evaluation, options):
        "%(name)s[graph_, s_, t_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph and not graph.empty():
            return Integer(nx.edge_connectivity(graph.G, s, t))


class EdgeIndex(_NetworkXBuiltin):
    """
    <url>
    :WMA link:
    https://reference.wolfram.com/language/ref/EdgeIndex.html</url>

    <dl>
    <dt>'EdgeIndex['graph', 'edge']'
    <dd>gives the position of the 'edge' in the list of edges associated \
    to the graph.
    </dl>
    """

    summary_text = "find the position of an edge"

    def eval(self, graph, v, expression, evaluation, options):
        "%(name)s[graph_, v_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            # FIXME: check if directionality must be considered or not.
            try:
                i = list(graph.edges).index(v.elements)
            except Exception:
                self._not_an_edge(expression, Integer2, evaluation)
                return
            return Integer(i + 1)


class EdgeList(_PatternList):
    """
    <url>
    :WMA link:
    https://reference.wolfram.com/language/ref/EdgeList.html</url>

    <dl>
      <dt>'EdgeList'[$g$]
      <dd>gives the list of edges that defines $g$
    </dl>
    """

    summary_text = "list the edges of a graph"

    def _items(self, graph):
        return graph.edges


class EdgeRules(_NetworkXBuiltin):
    """
    <url>
    :WMA link:
    https://reference.wolfram.com/language/ref/EdgeRules.html</url>

    <dl>
      <dt>'EdgeRules'[$g$]
      <dd> gives the list of edge rules for the graph $g$.
    </dl>
    """

    summary_text = "list the edge as rules"

    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:

            def rules():
                for edge in graph.edges:
                    u, v = edge
                    yield Expression(SymbolRule, u, v)

            return ListExpression(*list(rules()))


class FindShortestPath(_NetworkXBuiltin):
    """

    <url>
    :Shortest path problem:
    https://en.wikipedia.org/wiki/Shortest_path_problem</url> (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/algorithms\
    /generated/networkx.algorithms.shortest_paths.generic.shortest_path.html</url>, <url>
    :WMA:
    https://reference.wolfram.com/language/ref/FindShortestPath.html</url>)

    <dl>
      <dt>'FindShortestPath'[$g$, $src$, $tgt$]
      <dd>List the vertices in the shortest path connecting the source $src$ with the target $tgt$ \
          in the graph $g$.
    </dl>


    >> FindShortestPath[{1 <-> 2, 2 <-> 3, 3 <-> 4, 2 <-> 4, 4 -> 5}, 1, 5]
     = {1, 2, 4, 5}

    >> FindShortestPath[{1 <-> 2, 2 <-> 3, 3 <-> 4, 4 -> 2, 4 -> 5}, 1, 5]
     = {1, 2, 3, 4, 5}

    >> FindShortestPath[{1 <-> 2, 2 <-> 3, 4 -> 3, 4 -> 2, 4 -> 5}, 1, 5]
     = {}

    >> g = Graph[{1 -> 2, 2 -> 3, 1 -> 3}, EdgeWeight -> {0.5, a, 3}];

    #> FindShortestPath[{}, 1, 2]
     : The vertex at position 2 in FindShortestPath[{}, 1, 2] does not belong to the graph at position 1.
     = FindShortestPath[{}, 1, 2]

    #> FindShortestPath[{1 -> 2}, 1, 3]
     : The vertex at position 3 in FindShortestPath[{1 -> 2}, 1, 3] does not belong to the graph at position 1.
     = FindShortestPath[{1 -> 2}, 1, 3]
    """

    summary_text = "find the shortest path between two vertices"

    def eval_s_t(self, graph, s, t, expression, evaluation, options):
        "FindShortestPath[graph_, s_, t_, OptionsPattern[FindShortestPath]]"
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
                return ListExpression(
                    *list(nx.shortest_path(G, source=s, target=t, weight=weight)),
                )
            except nx.exception.NetworkXNoPath:
                return ListExpression()


class FindVertexCut(_NetworkXBuiltin):
    """
    <url>:Minimum cut:https://en.wikipedia.org/wiki/Minimum_cut</url>\
    (<url>
    :NetworkX:
    https://networkx.org/documentation/networkx-2.8.8/reference/algorithms/\
generated/networkx.algorithms.connectivity.cuts.minimum_node_cut.html
    </url>, \
    <url>
    :WMA:
    https://reference.wolfram.com/language/ref/FindVertexCut.html</url>)

    <dl>
    <dt>'FindVertexCut[$g$]'
        <dd>finds a set of vertices of minimum cardinality that, if removed, renders $g$ disconnected.
    <dt>'FindVertexCut[$g$, $s$, $t$]'
        <dd>finds a vertex cut that disconnects all paths from $s$ to $t$.
    </dl>

    >> g = Graph[{1 -> 2, 2 -> 3}]; FindVertexCut[g]
     = {}

    >> g = Graph[{1 <-> 2, 2 <-> 3}]; FindVertexCut[g]
     = {2}

    >> g = Graph[{1 <-> x, x <-> 2, 1 <-> y, y <-> 2, x <-> y}]; FindVertexCut[g]
     = {x, y}

    #> FindVertexCut[Graph[{}]]
     = {}
    #> FindVertexCut[Graph[{}], 1, 2]
     : The vertex at position 2 in FindVertexCut[-Graph-, 1, 2] does not belong to the graph at position 1.
     = FindVertexCut[-Graph-, 1, 2]
    """

    summary_text = "find the vertex cuts"

    def eval(self, graph, expression, evaluation, options):
        "FindVertexCut[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            if graph.empty() or not is_connected(graph.G):
                return ListExpression()
            else:
                return ListExpression(
                    *graph.sort_vertices(nx.minimum_node_cut(graph.G))
                )

    def eval_st(self, graph, s, t, expression, evaluation, options):
        "FindVertexCut[graph_, s_, t_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if not graph:
            return

        G = graph.G
        if not G.has_node(s):
            self._not_a_vertex(expression, 2, evaluation)
        elif not G.has_node(t):
            self._not_a_vertex(expression, 3, evaluation)
        elif graph.empty() or not is_connected(graph.G):
            return ListExpression()
        else:
            return ListExpression(*graph.sort_vertices(nx.minimum_node_cut(G, s, t)))


class GraphAtom(AtomBuiltin):
    """
    <url>:Graph:https://en.wikipedia.org/wiki/graph</url> (<url>:WMA:
    https://reference.wolfram.com/language/ref/Graph.html</url>)
    <dl>
      <dt>'Graph[{$e1, $e2, ...}]'
      <dd>returns a graph with edges $e_j$.
    </dl>

    <dl>
      <dt>'Graph[{v1, v2, ...}, {$e1, $e2, ...}]'
      <dd>returns a graph with vertices $v_i$ and edges $e_j$.
    </dl>

    >> Graph[{1->2, 2->3, 3->1}]
     = -Graph-

    #>> Graph[{1->2, 2->3, 3->1}, EdgeStyle -> {Red, Blue, Green}]
    # = -Graph-

    >> Graph[{1->2, Property[2->3, EdgeStyle -> Thick], 3->1}]
     = -Graph-

    # >> Graph[{1->2, 2->3, 3->1}, VertexStyle -> {1 -> Green, 3 -> Blue}]
    # = -Graph-

    >> Graph[x]
     = Graph[x]

    >> Graph[{1}]
     = Graph[{1}]

    >> Graph[{{1 -> 2}}]
     = Graph[{{1 -> 2}}]

    """

    # requires = ("networkx",)
    name = "Graph"
    options = DEFAULT_GRAPH_OPTIONS
    summary_text = "build a graph"

    def eval(self, graph, evaluation, options):
        "Pymathics`Graph[graph_List, OptionsPattern[%(name)s]]"
        return _graph_from_list(graph.elements, options)

    def eval_1(self, vertices, edges, evaluation, options):
        "Pymathics`Graph[vertices_List, edges_List, OptionsPattern[%(name)s]]"
        return _graph_from_list(
            edges.elements, options=options, new_vertices=vertices.elements
        )


class HighlightGraph(_NetworkXBuiltin):
    """
    <url>
    :WMA link:
    https://reference.wolfram.com/language/ref/HighlightGraph.html</url>

    <dl>
    <dt>'HighlightGraph'[$graph$, $what$]
    <dd>highlight in $graph$ the elements enumerated in $what$ by adding style marks.
    </dl>
    """

    summary_text = "highlight elements in a graph"

    def eval(self, graph, what, expression, evaluation, options):
        "HighlightGraph[graph_, what_List, OptionsPattern[%(name)s]]"
        default_highlight = [Expression(SymbolRGBColor, Integer1, Integer0, Integer0)]

        def parse(item):
            if item.get_head_name() == "System`Rule":
                return Expression(SymbolDirectedEdge, *item.elements)
            else:
                return item

        rules = []
        for element in what.elements:
            if element.get_head_name() == "System`Style":
                if len(element.elements) >= 2:
                    rules.append((parse(element.elements[0]), element.elements[1:]))
            else:
                rules.append((parse(element), default_highlight))

        rules.append((Expression(SymbolBlank), Expression(SymbolMissing)))

        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            rule_exprs = ListExpression(*[Expression(SymbolRule, *r) for r in rules])
            return graph.with_highlight(rule_exprs)


class Property(Builtin):
    """
    <url>
    :WMA link:
    https://reference.wolfram.com/language/ref/Property.html</url>

    <dl>
      <dt>'Property'[$item$, {$name$, $val$}]
      <dd>associate a property called $name$ with value $val$ to $item$.
    </dl>

    """

    summary_text = "assign a property"


class PropertyValue(Builtin):
    """
    <url>
    :WMA link:
    https://reference.wolfram.com/language/ref/PropertyValue.html</url>

    <dl>
      <dt>'PropertyValue'[{$obj$, $item$}, $name$]
      <dd>gives the value of a property associated with the name  $name$
          for $item$ in the object $obj$.
    </dl>


    >> g = Graph[{a <-> b, Property[b <-> c, SomeKey -> 123]}];
    >> PropertyValue[{g, b <-> c}, SomeKey]
     = 123
    >> PropertyValue[{g, b <-> c}, SomeUnknownKey]
     = $Failed
    """

    requires = ("networkx",)
    summary_text = "retrieve the value of a property"

    def eval(self, graph, item, name, evaluation):
        "PropertyValue[{graph_Graph, item_}, name_Symbol]"
        name_str = name.get_name()
        if isinstance(graph, Graph):
            if (
                item.has_form("Rule", 2)
                or item.has_form("DirectedEdge", 2)
                or item.has_form("UndirectedEdge", 2)
            ):
                item_g = graph.G.edges().get(item.elements)
            else:
                item_g = graph.G.nodes().get(item)

            if item_g is None:
                return SymbolFailed

            value = item_g.get(name_str, SymbolFailed)
            return value


class VertexAdd(_NetworkXBuiltin):
    """
    <url>
    :WMA link:
    https://reference.wolfram.com/language/ref/VertexAdd.html</url>

    <dl>
      <dt>'VertexAdd'[$g$, $ver$]
      <dd>create a new graph from $g$, by adding the vertex $ver$.
    </dl>

    >> g1 = Graph[{1 -> 2, 2 -> 3}];
    >> g2 = VertexAdd[g1, 4]
     = -Graph-
    >> g3 = VertexAdd[g2, {5, 10}]
     = -Graph-
    >> VertexAdd[{a -> b}, c]
     = -Graph-
    """

    summary_text = "add a vertex"

    def eval(self, graph: Expression, what, expression, evaluation, options):
        "VertexAdd[graph_, what_, OptionsPattern[VertexAdd]]"
        mathics_graph = self._build_graph(graph, evaluation, options, expression)
        if mathics_graph:
            if what.get_head_name() == "System`List":
                return mathics_graph.add_vertices(
                    *zip(*[_parse_item(x) for x in what.elements])
                )
            else:
                return mathics_graph.add_vertices(*zip(*[_parse_item(what)]))


class VertexConnectivity(_NetworkXBuiltin):
    """
    <url>
    :WMA link:
    https://reference.wolfram.com/language/ref/VertexConnectivity.html</url>

    <dl>
      <dt>'VertexConnectivity'[$g$]
      <dd>gives the vertex connectivity of the graph $g$.
    </dl>

    >> VertexConnectivity[{1 <-> 2, 2 <-> 3}]
     = 1

    >> VertexConnectivity[{1 -> 2, 2 -> 3}]
     = 0

    >> VertexConnectivity[{1 -> 2, 2 -> 3, 3 -> 1}]
     = 1

    >> VertexConnectivity[{1 <-> 2, 2 <-> 3, 1 <-> 3}]
     = 2

    >> VertexConnectivity[{1 <-> 2, 3 <-> 4}]
     = 0

    #> VertexConnectivity[Graph[{}]]
     = VertexConnectivity[-Graph-]
    """

    summary_text = "vertex connectivity"

    def eval(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph and not graph.empty():
            if not is_connected(graph.G):
                return Integer(0)
            else:
                return Integer(nx.node_connectivity(graph.G))

    def eval_st(self, graph, s, t, expression, evaluation, options):
        "%(name)s[graph_, s_, t_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph and not graph.empty():
            if not is_connected(graph.G):
                return Integer(0)
            else:
                return Integer(nx.node_connectivity(graph.G, s, t))


class VertexDelete(_NetworkXBuiltin):
    """
    <url>
    :WMA link:
    https://reference.wolfram.com/language/ref/VertexDelete.html</url>

    <dl>
      <dt>'VertexDelete'[$g$, $vert$]
      <dd>remove the vertex $vert$ and their associated edges.
    </dl>

    >> g1 = Graph[{1 -> 2, 2 -> 3, 3 -> 4}];
    >> VertexDelete[g1, 3]
     = -Graph-
    >> VertexDelete[{a -> b, b -> c, c -> d, d -> a}, {a, c}]
     = -Graph-
    >> VertexDelete[{1 -> 2, 2 -> 3, 3 -> 4, 4 -> 6, 6 -> 8, 8 -> 2}, _?OddQ]
     = -Graph-
    """

    summary_text = "remove a vertex"

    def eval(self, graph, what, expression, evaluation, options) -> Optional[Graph]:
        "VertexDelete[graph_, what_, OptionsPattern[VertexDelete]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            head_name = what.get_head_name()
            if head_name in pattern_objects:
                cases = Expression(
                    SymbolCases, ListExpression(*graph.vertices), what
                ).evaluate(evaluation)
                if cases.get_head_name() == "System`List":
                    return graph.delete_vertices(cases.elements)
            elif head_name == "System`List":
                return graph.delete_vertices(what.elements)
            else:
                return graph.delete_vertices([what])


class VertexIndex(_NetworkXBuiltin):
    """
    <url>
    :WMA link:
    https://reference.wolfram.com/language/ref/VertexIndex.html</url>
    <dl>
      <dt>'VertexIndex'['g', 'v']
      <dd> gives the integer index of the vertex 'v' in the\
       graph 'g'.
    </dl>

    >> a=.;
    >> VertexIndex[{c <-> d, d <-> a}, a]
     = 3
    """

    summary_text = "find the position of a vertex"

    def eval(self, graph, v, expression, evaluation, options):
        "%(name)s[graph_, v_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            try:
                return Integer(list(graph.vertices).index(v) + 1)
            except ValueError:
                self._not_a_vertex(expression, 2, evaluation)
        return None


class VertexList(_PatternList):
    """
    <url>
    :WMA link:
    https://reference.wolfram.com/language/ref/VertexList.html</url>
    <dl>
      <dt>'VertexList[$edgelist$]'
      <dd>list the vertices from a list of directed edges.
    </dl>

    >> a=.;
    >> VertexList[{1 -> 2, 2 -> 3}]
     = {1, 2, 3}

    >> VertexList[{a -> c, c -> b}]
     = {a, c, b}

    >> VertexList[{a -> c, 5 -> b}, _Integer -> 10]
     = {10}
    """

    summary_text = "list the vertices of a graph"

    def _items(self, graph):
        return graph.vertices


class UndirectedEdge(Builtin):
    """
    <url>
    :WMA link:
    https://reference.wolfram.com/language/ref/UndirectedEdge.html</url>

    <dl>
      <dt>'UndirectedEdge[$u$, $v$]'
      <dd>create an undirected edge between $u$ and $v$.
    </dl>

    >> a <-> b
     = UndirectedEdge[a, b]

    >> (a <-> b) <-> c
     = UndirectedEdge[UndirectedEdge[a, b], c]

    >> a <-> (b <-> c)
     = UndirectedEdge[a, UndirectedEdge[b, c]]
    """

    summary_text = "undirected graph edge"
    pass


# class EdgeAdd(_NetworkXBuiltin):
#     """
#     >> EdgeAdd[{1->2,2->3},3->1]
#      = -Graph-
#     """

#     def eval(self, graph: Expression, what, expression, evaluation, options):
#         "%(name)s[graph_, what_, OptionsPattern[%(name)s]]"
#         mathics_graph = self._build_graph(graph, evaluation, options, expression)
#         if mathics_graph:
#             if what.get_head_name() == "System`List":
#                 return mathics_graph.add_edges(*zip(*[_parse_item(x) for x in what.elements]))
#             else:
#                 return mathics_graph.add_edges(*zip(*[_parse_item(what)]))


class EdgeDelete(_NetworkXBuiltin):
    """
    Delete an Edge (<url>
    :WMA:
    https://reference.wolfram.com/language/ref/EdgeDelete.html</url>)

    <dl>
      <dt>'EdgeDelete'[$g$, $edge$]
      <dd>remove the edge $edge$.
    </dl>

    >> g = Graph[{1 -> 2, 2 -> 3, 3 -> 1}, VertexLabels->True]
     = -Graph-

    >> EdgeList[EdgeDelete[g, 2 -> 3]]
     = {{1, 2}, {3, 1}}

    ## >> g = Graph[{4<->5,5<->7,7<->9,9<->5,2->4,4->6,6->2}, VertexLabels->True]
    ##  = -Graph-

    ## >> EdgeDelete[g, _UndirectedEdge]
    ##  = -Graph-
    """

    summary_text = "remove an edge"

    def eval(self, graph, what, expression, evaluation, options) -> Optional[Graph]:
        "EdgeDelete[graph_, what_, OptionsPattern[EdgeDelete]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            head_name = what.get_head_name()
            if head_name in pattern_objects:
                cases = Expression(
                    SymbolCases, to_mathics_list(*graph.edges), what
                ).evaluate(evaluation)
                if cases.get_head_name() == "System`List":
                    return graph.delete_edges(cases.elements)
            elif head_name == "System`List":
                return graph.delete_edges(what.elements)
            else:
                return graph.delete_edges([what])
