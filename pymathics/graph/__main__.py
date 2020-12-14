# -*- coding: utf-8 -*-

"""
Core routines for working with Graphs.
A Graph is a tuple of a set of Nodes and Edges.

networkx does all the heavy lifting.
"""

# uses networkx

from mathics.builtin.base import Builtin, AtomBuiltin
from mathics.builtin.graphics import GraphicsBox
from mathics.core.expression import (
    Atom,
    Expression,
    Integer,
    Real,
    String,
    Symbol,
)
from mathics.builtin.patterns import Matcher

from inspect import isgenerator

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


def has_directed_option(options: dict) -> bool:
    return options.get("System`DirectedEdges", False).to_python()


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
    return nx.drawing.circular_layout(G, scale=1.5)


def _spectral_layout(G):
    return nx.drawing.spectral_layout(G, scale=2.0)


def _shell_layout(G):
    return nx.drawing.shell_layout(G, scale=2.0)


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


def _components(G):
    if G.is_directed():
        return nx.strongly_connected_components(G)
    else:
        return nx.connected_components(G)


_default_minimum_distance = 0.3


def _vertex_style(expr):
    return expr


def _edge_style(expr):
    return expr


def _parse_property(expr, attr_dict=None):
    if expr.has_form("Rule", 2):
        name, value = expr.leaves
        if isinstance(name, Symbol):
            if attr_dict is None:
                attr_dict = {}
            attr_dict[name.get_name()] = value
    elif expr.has_form("List", None):
        for item in expr.leaves:
            attr_dict = _parse_property(item, attr_dict)
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

    def _build_graph(self, graph, evaluation, options, expr, quiet=False):
        head = graph.get_head_name()
        if head == "System`Graph" and isinstance(graph, Atom) and hasattr(graph, "G"):
            return graph
        elif head == "System`List":
            return _graph_from_list(graph.leaves, options)
        elif not quiet:
            evaluation.message(self.get_name(), "graph", expr)

    def _evaluate_atom(self, graph, options, compute):
        head = graph.get_head_name()
        if head == "System`Graph":
            return compute(graph)
        elif head == "System`List":
            return compute(_graph_from_list(graph.leaves, options))

    def __str__(self):
        return "-Graph-"


class GraphBox(GraphicsBox):
    def _graphics_box(self, leaves, options):
        evaluation = options["evaluation"]
        graph, form = leaves
        primitives = graph._layout(evaluation)
        graphics = Expression("Graphics", primitives)
        graphics_box = Expression("MakeBoxes", graphics, form).evaluate(evaluation)
        return graphics_box

    def boxes_to_text(self, leaves, **options):
        return "-Graph-"

    def boxes_to_xml(self, leaves, **options):
        # Figure out what to do here.
        return "-Graph-XML-"

    def boxes_to_tex(self, leaves, **options):
        # Figure out what to do here.
        return "-Graph-TeX-"


class _Collection(object):
    def __init__(self, expressions, properties=None, index=None):
        self.expressions = expressions
        self.properties = properties if properties else None
        self.index = index

    def clone(self):
        properties = self.properties
        return _Collection(
            self.expressions[:], properties[:] if properties else None, None
        )

    def filter(self, expressions):
        index = self.get_index()
        return [expr for expr in expressions if expr in index]

    def extend(self, expressions, properties):
        if properties:
            if self.properties is None:
                self.properties = [None] * len(self.expressions)
            self.properties.extend(properties)
        self.expressions.extend(expressions)
        self.index = None
        return expressions

    def delete(self, expressions):
        index = self.get_index()
        trash = set(index[x] for x in expressions)
        deleted = [self.expressions[i] for i in trash]
        self.expressions = [x for i, x in enumerate(self.expressions) if i not in trash]
        self.properties = [x for i, x in enumerate(self.properties) if i not in trash]
        self.index = None
        return deleted

    def data(self):
        return self.expressions, list(self.get_properties())

    def get_index(self):
        index = self.index
        if index is None:
            index = dict((v, i) for i, v in enumerate(self.expressions))
            self.index = index
        return index

    def get_properties(self):
        if self.properties:
            for p in self.properties:
                yield p
        else:
            for _ in range(len(self.expressions)):
                yield None

    def get_sorted(self):
        index = self.get_index()
        return lambda c: sorted(c, key=lambda v: index[v])

    def get_property(self, item, name):
        properties = self.properties
        if properties is None:
            return None
        index = self.get_index()
        i = index.get(item)
        if i is None:
            return None
        p = properties[i]
        if p is None:
            return None
        return p.get(name)


def _count_edges(counts, edges, sign):
    n_directed, n_undirected = counts
    for edge in edges:
        if edge.get_head_name() == "System`DirectedEdge":
            n_directed += sign
        else:
            n_undirected += sign
    return n_directed, n_undirected


class _EdgeCollection(_Collection):
    def __init__(
        self, expressions, properties=None, index=None, n_directed=0, n_undirected=0
    ):
        super(_EdgeCollection, self).__init__(expressions, properties, index)
        self.counts = (n_directed, n_undirected)

    def is_mixed(self):
        n_directed, n_undirected = self.counts
        return n_directed > 0 and n_undirected > 0

    def clone(self):
        properties = self.properties
        n_directed, n_undirected = self.counts
        return _EdgeCollection(
            self.expressions[:],
            properties[:] if properties else None,
            None,  # index
            n_directed,
            n_undirected,
        )

    def extend(self, expressions, properties):
        added = super(_EdgeCollection, self).extend(expressions, properties)
        self.counts = _count_edges(self.counts, added, 1)
        return added

    def delete(self, expressions):
        deleted = super(_EdgeCollection, self).delete(expressions)
        self.counts = _count_edges(self.counts, deleted, -1)
        return deleted


class _FullGraphRewrite(Exception):
    pass


def _normalize_edges(edges):
    for edge in edges:
        if edge.has_form("Property", 2):
            expr, prop = edge.leaves
            yield Expression(edge.get_head(), list(_normalize_edges([expr]))[0], prop)
        elif edge.get_head_name() == "System`Rule":
            yield Expression("System`DirectedEdge", *edge.leaves)
        else:
            yield edge


class Graph(Atom):

    options = DEFAULT_GRAPH_OPTIONS

    def __init__(self, G, **kwargs):
        super(Graph, self).__init__()
        self.G = G

    @property
    def edges(self):
        return self.G.edges

    @property
    def vertices(self):
        return self.G.nodes

    def empty(self):
        return len(self.G) == 0

    # networkx graphs can't be for mixed
    def is_mixed_graph(self):
        return False
        # return self.edges. ... is_mixed()

    def is_multigraph(self):
        return isinstance(self.G, (nx.MultiDiGraph, nx.MultiGraph))

    def is_loop_free(self):
        return not any(True for _ in nx.nodes_with_selfloops(self.G))

    def __str__(self):
        return "-Graph-"

    def do_copy(self):
        return Graph(self.G)

    def get_sort_key(self, pattern_sort=False):
        if pattern_sort:
            return super(Graph, self).get_sort_key(True)
        else:
            return hash(self)

    def default_format(self, evaluation, form):
        return "-Graph-"

    def same(self, other):
        return isinstance(other, Graph) and self.G == other.G
        # FIXME
        # self.properties == other.properties
        # self.options == other.options
        # self.highlights == other.highlights

    def to_python(self, *args, **kwargs):
        return self.G

    def __hash__(self):
        return hash(("Graph", self.G))  # FIXME self.properties, ...

    def atom_to_boxes(self, form, evaluation):
        return Expression("GraphBox", self, form)

    def boxes_to_xml(self, **options):
        # Figure out what to do here.
        return "-Graph-XML-"

    def get_property(self, item, name):
        if item.get_head_name() in ("System`DirectedEdge", "System`UndirectedEdge"):
            x = self.edges.get_property(item, name)
        if x is None:
            x = self.vertices.get_property(item, name)
        return x

    def delete_edges(self, edges_to_delete):
        G = self.G.copy()
        directed = G.is_directed()

        edges_to_delete = list(_normalize_edges(edges_to_delete))
        # FIXME: edges_to_delete is needs to be a tuple. tuples
        # are edges in networkx
        edges_to_delete = [edge for edge in self.edges if edge in edges_to_delete]

        for edge in edges_to_delete:
            if edge.has_form("DirectedEdge", 2):
                if directed:
                    u, v = edge.leaves
                    G.remove_edge(u, v)
            elif edge.has_form("UndirectedEdge", 2):
                u, v = edge.leaves
                if directed:
                    G.remove_edge(u, v)
                    G.remove_edge(v, u)
                else:
                    G.remove_edge(u, v)

        edges = self.edges.clone()
        edges.delete(edges_to_delete)

        return Graph(G)

    def update_weights(self, evaluation):
        weights = None
        G = self.G

        if self.is_multigraph():
            for u, v, k, w in G.edges.data(
                "System`EdgeWeight", default=None, keys=True
            ):
                data = G.get_edge_data(u, v, key=k)
                w = data.get()
                if w is not None:
                    w = w.evaluate(evaluation).to_mpmath()
                    G[u][v][k]["WEIGHT"] = w
                    weights = "WEIGHT"
        else:
            for u, v, w in G.edges.data("System`EdgeWeight", default=None):
                if w is not None:
                    w = w.evaluate(evaluation).to_mpmath()
                    G[u][v]["WEIGHT"] = w
                    weights = "WEIGHT"

        return weights


def _is_connected(G):
    if len(G) == 0:  # empty graph?
        return True
    elif G.is_directed():
        return nx.is_strongly_connected(G)
    else:
        return nx.is_connected(G)


def _edge_weights(options):
    expr = options.get("System`EdgeWeight")
    if expr is None:
        return []
    if not expr.has_form("List", None):
        return []
    return expr.leaves


class _GraphParseError(Exception):
    pass


def _parse_item(x, attr_dict=None):
    if x.has_form("Property", 2):
        expr, prop = x.leaves
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

    known_vertices = set()
    vertices = []
    vertex_properties = []

    def add_vertex(x, attr_dict=None):
        if x.has_form("Property", 2):
            expr, prop = x.leaves
            attr_dict = _parse_property(prop, attr_dict)
            return add_vertex(expr, attr_dict)
        elif x not in known_vertices:
            known_vertices.add(x)
            vertices.append(x)
            vertex_properties.append(attr_dict)
        return x

    directed_edges = []
    undirected_edges = []

    if new_vertices is not None:
        vertices = [add_vertex(v) for v in new_vertices]

    if from_graph is not None:
        old_vertices, vertex_properties = from_graph.vertices.data()
        vertices += old_vertices
        edges, edge_properties = from_graph.edges.data()

        for edge, attr_dict in zip(edges, edge_properties):
            u, v = edge.leaves
            if edge.get_head_name() == "System`DirectedEdge":
                directed_edges.append((u, v, attr_dict))
            else:
                undirected_edges.append((u, v, attr_dict))

        multigraph = [from_graph.is_multigraph()]
    else:
        edges = []
        edge_properties = []

        multigraph = [False]

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
    use_directed_edges = options.get("System`DirectedEdges", Symbol("True")).is_true()

    directed_edge_head = Symbol(
        "DirectedEdge" if use_directed_edges else "UndirectedEdge"
    )
    undirected_edge_head = Symbol("UndirectedEdge")

    def parse_edge(r, attr_dict):
        if r.is_atom():
            raise _GraphParseError

        name = r.get_head_name()
        leaves = r.leaves

        if len(leaves) != 2:
            raise _GraphParseError

        u, v = leaves

        u = add_vertex(u)
        v = add_vertex(v)

        if name in ("System`Rule", "System`DirectedEdge"):
            edges_container = directed_edges
            head = directed_edge_head
            track_edges((u, v))
        elif name == "System`UndirectedEdge":
            edges_container = undirected_edges
            head = undirected_edge_head
            track_edges((u, v), (v, u))
        elif name == "PyMathics`Property":
            for prop in edge.leaves:
                prop_str = str(prop.head)
                if prop_str in ("System`Rule", "System`DirectedEdge"):
                    edges_container = directed_edges
                    head = directed_edge_head
                    track_edges((u, v))
                elif prop_str == "System`UndirectedEdge":
                    edges_container = undirected_edges
                    head = undirected_edge_head
                else:
                    pass
            pass
        else:
            raise _GraphParseError

        if head.get_name() == name:
            edges.append(r)
        else:
            edges.append(Expression(head, u, v))
        edge_properties.append(attr_dict)

        edges_container.append((u, v, attr_dict))

    try:

        def full_new_edge_properties(new_edge_style):
            for i, (attr_dict, w) in enumerate(zip(new_edge_properties, edge_weights)):
                attr_dict = {} if attr_dict is None else attr_dict.copy()
                attr_dict["System`EdgeWeight"] = w
                yield attr_dict
            # FIXME: figure out what to do here. Color is a mess.
            # for i, (attr_dict, s) in enumerate(zip(new_edge_style, new_edge_style)):
            #     attr_dict = {} if attr_dict is None else attr_dict.copy()
            #     attr_dict["System`EdgeStyle"] = s
            #     yield attr_dict
            for attr_dict in new_edge_properties[len(edge_weights) :]:
                yield attr_dict

        if "System`EdgeStyle" in options:
            # FIXME: Figure out what to do here:
            # Color is a f-ing mess.
            # edge_options = options["System`EdgeStyle"].to_python()
            edge_options = []
        else:
            edge_options = []
        edge_properties = list(full_new_edge_properties(edge_options))
        for edge, attr_dict in zip(new_edges, edge_properties):
            parse_edge(edge, attr_dict)
    except _GraphParseError:
        return

    empty_dict = {}
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

        for u, v, attr_dict in undirected_edges:
            attr_dict = attr_dict or empty_dict
            G.add_edge(u, v, **attr_dict)
            G.add_edge(v, u, **attr_dict)
    else:
        G = nx.MultiGraph() if multigraph[0] else nx.Graph()
        for u, v, attr_dict in undirected_edges:
            attr_dict = attr_dict or empty_dict
            G.add_edge(u, v, **attr_dict)

    edge_collection = _EdgeCollection(
        edges,
        edge_properties,
        n_directed=len(directed_edges),
        n_undirected=len(undirected_edges),
    )

    g = Graph(G)
    _process_graph_options(g, options)
    return g


class Property(Builtin):
    pass


class PropertyValue(Builtin):
    """
    >> g = Graph[{a <-> b, Property[b <-> c, SomeKey -> 123]}];
    >> PropertyValue[{g, b <-> c}, SomeKey]
     = 123
    >> PropertyValue[{g, b <-> c}, SomeUnknownKey]
     = $Failed
    """

    requires = ("networkx",)

    def apply(self, graph, item, name, evaluation):
        "PropertyValue[{graph_Graph, item_}, name_Symbol]"
        value = graph.get_property(item, name.get_name())
        if value is None:
            return Symbol("$Failed")
        return value


class DirectedEdge(Builtin):
    """
    <dl>
    <dt>'DirectedEdge[$u$, $v$]'
      <dd>a directed edge from $u$ to $v$.
    </dl>
    """

    pass


class UndirectedEdge(Builtin):
    """
    <dl>
      <dt>'UndirectedEdge[$u$, $v$]'
      <dd>an undirected edge between $u$ and $v$.
    </dl>

    >> a <-> b
     = UndirectedEdge[a, b]

    >> (a <-> b) <-> c
     = UndirectedEdge[UndirectedEdge[a, b], c]

    >> a <-> (b <-> c)
     = UndirectedEdge[a, UndirectedEdge[b, c]]
    """

    pass


class GraphAtom(AtomBuiltin):
    """
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

    #>> Graph[{1->2, 2->3, 3->1}, VertexStyle -> {1 -> Green, 3 -> Blue}]
    #= -Graph-

    >> Graph[x]
     = Graph[x]

    >> Graph[{1}]
     = Graph[{1}]

    >> Graph[{{1 -> 2}}]
     = Graph[{{1 -> 2}}]

    >> g = Graph[{1 -> 2, 2 -> 3}, DirectedEdges -> True];
    >> EdgeCount[g, _DirectedEdge]
     = 2
    >> g = Graph[{1 -> 2, 2 -> 3}, DirectedEdges -> False];
    >> EdgeCount[g, _DirectedEdge]
     = 0
    >> EdgeCount[g, _UndirectedEdge]
     = 2
    """

    requires = ("networkx",)

    options = DEFAULT_GRAPH_OPTIONS

    def apply(self, graph, evaluation, options):
        "Graph[graph_List, OptionsPattern[%(name)s]]"
        return _graph_from_list(graph.leaves, options)

    def apply_1(self, vertices, edges, evaluation, options):
        "Graph[vertices_List, edges_List, OptionsPattern[%(name)s]]"
        return _graph_from_list(
            edges.leaves, options=options, new_vertices=vertices.leaves
        )


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

    def apply(self, graph, expression, evaluation, options):
        "PathGraphQ[graph_, OptionsPattern[%(name)s]]"
        if not isinstance(graph, Graph):
            return Symbol("False")

        if graph.empty():
            is_path = False
        else:
            G = graph.G

            if G.is_directed():
                connected = nx.is_semiconnected(G)
            else:
                connected = nx.is_connected(G)

            if connected:
                is_path = all(d <= 2 for _, d in G.degree(graph.vertices))
            else:
                is_path = False

        return Symbol("True" if is_path else "False")


class MixedGraphQ(_NetworkXBuiltin):
    """
    >> g = Graph[{1 -> 2, 2 -> 3}]; MixedGraphQ[g]
     = False

    >> g = Graph[{1 -> 2, 2 <-> 3}]; MixedGraphQ[g]
     = True

    #> g = Graph[{}]; MixedGraphQ[g]
     = False

    #> MixedGraphQ["abc"]
     = False

    #> g = Graph[{1 -> 2, 2 -> 3}]; MixedGraphQ[g]
     = False
    #> g = EdgeAdd[g, a <-> b]; MixedGraphQ[g]
     = True
    #> g = EdgeDelete[g, a <-> b]; MixedGraphQ[g]
     = False
    """

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            return Symbol("True" if graph.is_mixed_graph() else "False")
        else:
            return Symbol("False")


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

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            return Symbol("True" if graph.is_multigraph() else "False")
        else:
            return Symbol("False")


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
    """

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph and not graph.empty():
            try:
                cycles = nx.find_cycle(graph.G)
            except nx.exception.NetworkXNoCycle:
                cycles = None
            return Symbol("True" if not cycles else "False")
        else:
            return Symbol("False")


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

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            if graph.empty():
                return Symbol("False")
            else:
                return Symbol("True" if graph.is_loop_free() else "False")
        else:
            return Symbol("False")


class DirectedGraphQ(_NetworkXBuiltin):
    """
    >> g = Graph[{1 -> 2, 2 -> 3}]; DirectedGraphQ[g]
     = True

    >> g = Graph[{1 -> 2, 2 <-> 3}]; DirectedGraphQ[g]
     = False

    #> g = Graph[{}]; DirectedGraphQ[{}]
     = False

    #> DirectedGraphQ["abc"]
     = False
    """

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            directed = graph.G.is_directed() and not graph.is_mixed_graph()
            return Symbol("True" if directed else "False")
        else:
            return Symbol("False")


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

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            return Symbol("True" if _is_connected(graph.G) else "False")
        else:
            return Symbol("False")


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

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            if graph.empty():
                return Symbol("True")
            else:
                simple = graph.is_loop_free() and not graph.is_multigraph()
                return Symbol("True" if simple else "False")
        else:
            return Symbol("False")


class PlanarGraphQ(_NetworkXBuiltin):
    """
    # see https://en.wikipedia.org/wiki/Planar_graph

    >> PlanarGraphQ[CompleteGraph[4]]
     = True

    >> PlanarGraphQ[CompleteGraph[5]]
     = False

    #> PlanarGraphQ[Graph[{}]]
     = False

    #> PlanarGraphQ["abc"]
     = False
    """

    requires = _NetworkXBuiltin.requires + ("planarity",)

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression, quiet=True)
        if graph:
            if graph.empty():
                return Symbol("False")
            else:
                import planarity

                return Symbol("True" if planarity.is_planar(graph.G) else "False")
        else:
            return Symbol("False")


class FindVertexCut(_NetworkXBuiltin):
    """
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

    def apply(self, graph, expression, evaluation, options):
        "FindVertexCut[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            if graph.empty() or not _is_connected(graph.G):
                return Expression("List")
            else:
                return Expression(
                    "List", *graph.sort_vertices(nx.minimum_node_cut(graph.G))
                )

    def apply_st(self, graph, s, t, expression, evaluation, options):
        "FindVertexCut[graph_, s_, t_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if not graph:
            return

        G = graph.G
        if not G.has_node(s):
            self._not_a_vertex(expression, 2, evaluation)
        elif not G.has_node(t):
            self._not_a_vertex(expression, 3, evaluation)
        elif graph.empty() or not _is_connected(graph.G):
            return Expression("List")
        else:
            return Expression(
                "List", *graph.sort_vertices(nx.minimum_node_cut(G, s, t))
            )


class HighlightGraph(_NetworkXBuiltin):
    """"""

    def apply(self, graph, what, expression, evaluation, options):
        "HighlightGraph[graph_, what_List, OptionsPattern[%(name)s]]"
        default_highlight = [Expression("RGBColor", 1, 0, 0)]

        def parse(item):
            if item.get_head_name() == "System`Rule":
                return Expression("DirectedEdge", *item.leaves)
            else:
                return item

        rules = []
        for item in what.leaves:
            if item.get_head_name() == "System`Style":
                if len(item.leaves) >= 2:
                    rules.append((parse(item.leaves[0]), item.leaves[1:]))
            else:
                rules.append((parse(item), default_highlight))

        rules.append((Expression("Blank"), Expression("Missing")))

        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            rule_exprs = Expression("List", *[Expression("Rule", *r) for r in rules])
            return graph.with_highlight(rule_exprs)


class _PatternList(_NetworkXBuiltin):
    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return Expression("List", *self._items(graph))

    def apply_patt(self, graph, patt, expression, evaluation, options):
        "%(name)s[graph_, patt_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return Expression("Cases", Expression("List", *self._items(graph)), patt)


class _PatternCount(_NetworkXBuiltin):
    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return Integer(len(self._items(graph)))

    def apply_patt(self, graph, patt, expression, evaluation, options):
        "%(name)s[graph_, patt_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return Expression(
                "Length",
                Expression("Cases", Expression("List", *self._items(graph)), patt),
            )


class VertexCount(_PatternCount):
    """
    >> VertexCount[{1 -> 2, 2 -> 3}]
     = 3

    >> VertexCount[{1 -> x, x -> 3}, _Integer]
     = 2
    """

    def _items(self, graph):
        return graph.vertices.expressions


class VertexList(_PatternList):
    """
    >> VertexList[{1 -> 2, 2 -> 3}]
     = {1, 2, 3}

    >> VertexList[{a -> c, c -> b}]
     = {a, c, b}

    >> VertexList[{a -> c, 5 -> b}, _Integer -> 10]
     = {10}
    """

    def _items(self, graph):
        return graph.vertices


class EdgeCount(_PatternCount):
    """
    >> EdgeCount[{1 -> 2, 2 -> 3}]
     = 2
    """

    def _items(self, graph):
        return graph.G.edges


class EdgeList(_PatternList):
    """
    >> EdgeList[{1 -> 2, 2 <-> 3}]
     = {DirectedEdge[1, 2], UndirectedEdge[2, 3]}
    """

    def _items(self, graph):
        return graph.edges


class EdgeRules(_NetworkXBuiltin):
    """
    >> EdgeRules[{1 <-> 2, 2 -> 3, 3 <-> 4}]
     = {1 -> 2, 2 -> 3, 3 -> 4}
    """

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:

            def rules():
                for expr in graph.edges.expressions:
                    u, v = expr.leaves
                    yield Expression("Rule", u, v)

            return Expression("List", *list(rules()))


class AdjacencyList(_NetworkXBuiltin):
    """
    >> AdjacencyList[{1 -> 2, 2 -> 3}, 3]
     = {2}

    >> AdjacencyList[{1 -> 2, 2 -> 3}, _?EvenQ]
     = {1, 3}

    >> AdjacencyList[{x -> 2, x -> 3, x -> 4, 2 -> 10, 2 -> 11, 4 -> 20, 4 -> 21, 10 -> 100}, 10, 2]
     = {x, 2, 11, 100}
    """

    def _retrieve(self, graph, what, neighbors, expression, evaluation):
        from mathics.builtin import pattern_objects

        if what.get_head_name() in pattern_objects:
            collected = set()
            match = Matcher(what).match
            for v in graph.G.nodes:
                if match(v, evaluation):
                    collected.update(neighbors(v))
            return Expression("List", *graph.sort_vertices(list(collected)))
        elif graph.G.has_node(what):
            return Expression("List", *graph.sort_vertices(neighbors(what)))
        else:
            self._not_a_vertex(expression, 2, evaluation)

    def apply(self, graph, what, expression, evaluation, options):
        "%(name)s[graph_, what_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            G = graph.G.to_undirected()  # FIXME inefficient
            return self._retrieve(
                graph, what, lambda v: G.neighbors(v), expression, evaluation
            )

    def apply_d(self, graph, what, d, expression, evaluation, options):
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


class VertexIndex(_NetworkXBuiltin):
    """
    >> VertexIndex[{c <-> d, d <-> a}, a]
     = 3
    """

    def apply(self, graph, v, expression, evaluation, options):
        "%(name)s[graph_, v_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            i = graph.vertices.get_index().get(v)
            if i is None:
                self._not_a_vertex(expression, 2, evaluation)
            else:
                return Integer(i + 1)


class EdgeIndex(_NetworkXBuiltin):
    """
    >> EdgeIndex[{c <-> d, d <-> a, a -> e}, d <-> a]
     = 2
    """

    def apply(self, graph, v, expression, evaluation, options):
        "%(name)s[graph_, v_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            i = graph.edges.get_index().get(v)
            if i is None:
                self._not_an_edge(expression, 2, evaluation)
            else:
                return Integer(i + 1)


class EdgeConnectivity(_NetworkXBuiltin):
    """
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

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph and not graph.empty():
            return Integer(nx.edge_connectivity(graph.G))

    def apply_st(self, graph, s, t, expression, evaluation, options):
        "%(name)s[graph_, s_, t_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph and not graph.empty():
            return Integer(nx.edge_connectivity(graph.G, s, t))


class VertexConnectivity(_NetworkXBuiltin):
    """
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

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph and not graph.empty():
            if not _is_connected(graph.G):
                return Integer(0)
            else:
                return Integer(nx.node_connectivity(graph.G))

    def apply_st(self, graph, s, t, expression, evaluation, options):
        "%(name)s[graph_, s_, t_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph and not graph.empty():
            if not _is_connected(graph.G):
                return Integer(0)
            else:
                return Integer(nx.node_connectivity(graph.G, s, t))


class _Centrality(_NetworkXBuiltin):
    pass


class BetweennessCentrality(_Centrality):
    """
    >> g = Graph[{a -> b, b -> c, d -> c, d -> a, e -> c, d -> b}]; BetweennessCentrality[g]
     = {0., 1., 0., 0., 0.}

    >> g = Graph[{a -> b, b -> c, c -> d, d -> e, e -> c, e -> a}]; BetweennessCentrality[g]
     = {3., 3., 6., 6., 6.}
    """

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            weight = graph.update_weights(evaluation)
            centrality = nx.betweenness_centrality(
                graph.G, normalized=False, weight=weight
            )
            return Expression(
                "List",
                *[Real(centrality.get(v, 0.0)) for v in graph.vertices.expressions],
            )


class ClosenessCentrality(_Centrality):
    """
    >> g = Graph[{a -> b, b -> c, d -> c, d -> a, e -> c, d -> b}]; ClosenessCentrality[g]
     = {0.666667, 1., 0., 1., 1.}

    >> g = Graph[{a -> b, b -> c, c -> d, d -> e, e -> c, e -> a}]; ClosenessCentrality[g]
     = {0.4, 0.4, 0.4, 0.5, 0.666667}
    """

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            weight = graph.update_weights(evaluation)
            G = graph.G
            if G.is_directed():
                G = G.reverse()
            centrality = nx.closeness_centrality(G, distance=weight, wf_improved=False)
            return Expression(
                "List",
                *[Real(centrality.get(v, 0.0)) for v in graph.vertices.expressions],
            )


class DegreeCentrality(_Centrality):
    """
    >> g = Graph[{a -> b, b <-> c, d -> c, d -> a, e <-> c, d -> b}]; DegreeCentrality[g]
     = {2, 4, 5, 3, 2}

    >> g = Graph[{a -> b, b <-> c, d -> c, d -> a, e <-> c, d -> b}]; DegreeCentrality[g, "In"]
     = {1, 3, 3, 0, 1}

    >> g = Graph[{a -> b, b <-> c, d -> c, d -> a, e <-> c, d -> b}]; DegreeCentrality[g, "Out"]
     = {1, 1, 2, 3, 1}
    """

    def _from_dict(self, graph, centrality):
        s = len(graph.G) - 1  # undo networkx's normalization
        return Expression(
            "List",
            *[Integer(s * centrality.get(v, 0)) for v in graph.vertices.expressions],
        )

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return self._from_dict(graph, nx.degree_centrality(graph.G))

    def apply_in(self, graph, expression, evaluation, options):
        '%(name)s[graph_, "In", OptionsPattern[%(name)s]]'
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return self._from_dict(graph, nx.in_degree_centrality(graph.G))

    def apply_out(self, graph, expression, evaluation, options):
        '%(name)s[graph_, "Out", OptionsPattern[%(name)s]]'
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return self._from_dict(graph, nx.out_degree_centrality(graph.G))


class _ComponentwiseCentrality(_Centrality):
    def _centrality(self, g, weight):
        raise NotImplementedError

    def _compute(self, graph, evaluation, reverse=False, normalized=True, **kwargs):
        vertices = graph.vertices.expressions
        G, weight = graph.coalesced_graph(evaluation)
        if reverse:
            G = G.reverse()

        components = list(_components(G))
        components = [c for c in components if len(c) > 1]

        result = [0] * len(vertices)
        for bunch in components:
            g = G.subgraph(bunch)
            centrality = self._centrality(g, weight, **kwargs)
            values = [centrality.get(v, 0) for v in vertices]
            if normalized:
                s = sum(values) * len(components)
            else:
                s = 1
            if s > 0:
                for i, x in enumerate(values):
                    result[i] += x / s

        return Expression("List", *[Real(x) for x in result])


class EigenvectorCentrality(_ComponentwiseCentrality):
    """
    >> g = Graph[{a -> b, b -> c, c -> d, d -> e, e -> c, e -> a}]; EigenvectorCentrality[g, "In"]
     = {0.16238, 0.136013, 0.276307, 0.23144, 0.193859}

    >> EigenvectorCentrality[g, "Out"]
     = {0.136013, 0.16238, 0.193859, 0.23144, 0.276307}

    >> g = Graph[{a <-> b, b <-> c, c <-> d, d <-> e, e <-> c, e <-> a}]; EigenvectorCentrality[g]
     = {0.162435, 0.162435, 0.240597, 0.193937, 0.240597}

    >> g = Graph[{a <-> b, b <-> c, a <-> c, d <-> e, e <-> f, f <-> d, e <-> d}]; EigenvectorCentrality[g]
     = {0.166667, 0.166667, 0.166667, 0.183013, 0.183013, 0.133975}

    #> g = Graph[{a -> b, b -> c, c -> d, b -> e, a -> e}]; EigenvectorCentrality[g]
     = {0., 0., 0., 0., 0.}

    >> g = Graph[{a -> b, b -> c, c -> d, b -> e, a -> e, c -> a}]; EigenvectorCentrality[g]
     = {0.333333, 0.333333, 0.333333, 0., 0.}
    """

    def _centrality(self, g, weight):
        return nx.eigenvector_centrality(g, max_iter=10000, tol=1.0e-7, weight=weight)

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return self._compute(graph, evaluation)

    def apply_in_out(self, graph, dir, expression, evaluation, options):
        "%(name)s[graph_, dir_String, OptionsPattern[%(name)s]]"
        py_dir = dir.get_string_value()
        if py_dir not in ("In", "Out"):
            return
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            return self._compute(graph, evaluation, py_dir == "Out")


class KatzCentrality(_ComponentwiseCentrality):
    """
    >> g = Graph[{a -> b, b -> c, c -> d, d -> e, e -> c, e -> a}]; KatzCentrality[g, 0.2]
     = {1.25202, 1.2504, 1.5021, 1.30042, 1.26008}

    >> g = Graph[{a <-> b, b <-> c, a <-> c, d <-> e, e <-> f, f <-> d, e <-> d}]; KatzCentrality[g, 0.1]
     = {1.25, 1.25, 1.25, 1.41026, 1.41026, 1.28205}
    """

    rules = {
        "KatzCentrality[g_, alpha_]": "KatzCentrality[g, alpha, 1]",
    }

    def _centrality(self, g, weight, alpha, beta):
        return nx.katz_centrality(
            g, alpha=alpha, beta=beta, normalized=False, weight=weight
        )

    def apply(self, graph, alpha, beta, expression, evaluation, options):
        "%(name)s[graph_, alpha_, beta_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            py_alpha = alpha.to_mpmath()
            py_beta = beta.to_mpmath()
            if py_alpha is None or py_beta is None:
                return
            return self._compute(
                graph, evaluation, normalized=False, alpha=py_alpha, beta=py_beta
            )


class PageRankCentrality(_Centrality):
    """
    >> g = Graph[{a -> d, b -> c, d -> c, d -> a, e -> c, d -> c}]; PageRankCentrality[g, 0.2]
     = {0.184502, 0.207565, 0.170664, 0.266605, 0.170664}
    """

    def apply_alpha_beta(self, graph, alpha, expression, evaluation, options):
        "%(name)s[graph_, alpha_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            py_alpha = alpha.to_mpmath()
            if py_alpha is None:
                return
            G, weight = graph.coalesced_graph(evaluation)
            centrality = nx.pagerank(G, alpha=py_alpha, weight=weight, tol=1.0e-7)
            return Expression(
                "List",
                *[Real(centrality.get(v, 0)) for v in graph.vertices.expressions],
            )


class HITSCentrality(_Centrality):
    """
    >> g = Graph[{a -> d, b -> c, d -> c, d -> a, e -> c}]; HITSCentrality[g]
     = {{0.292893, 0., 0., 0.707107, 0.}, {0., 1., 0.707107, 0., 0.707107}}
    """

    def apply(self, graph, expression, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            G, _ = graph.coalesced_graph(evaluation)  # FIXME warn if weight > 1

            tol = 1.0e-14
            _, a = nx.hits(G, normalized=True, tol=tol)
            h, _ = nx.hits(G, normalized=False, tol=tol)

            def _crop(x):
                return 0 if x < tol else x

            vertices = graph.vertices.expressions
            return Expression(
                "List",
                Expression("List", *[Real(_crop(a.get(v, 0))) for v in vertices]),
                Expression("List", *[Real(_crop(h.get(v, 0))) for v in vertices]),
            )


class VertexDegree(_Centrality):
    """
    >> VertexDegree[{1 <-> 2, 2 <-> 3, 2 <-> 4}]
     = {1, 3, 1, 1}
    """

    def apply(self, graph, evaluation, options):
        "%(name)s[graph_, OptionsPattern[%(name)s]]"

        def degrees(graph):
            degrees = dict(list(graph.G.degree(graph.vertices)))
            return Expression(
                "List", *[Integer(degrees.get(v, 0)) for v in graph.vertices]
            )

        return self._evaluate_atom(graph, options, degrees)


class FindShortestPath(_NetworkXBuiltin):
    """
    >> FindShortestPath[{1 <-> 2, 2 <-> 3, 3 <-> 4, 2 <-> 4, 4 -> 5}, 1, 5]
     = {1, 2, 4, 5}

    >> FindShortestPath[{1 <-> 2, 2 <-> 3, 3 <-> 4, 4 -> 2, 4 -> 5}, 1, 5]
     = {1, 2, 3, 4, 5}

    >> FindShortestPath[{1 <-> 2, 2 <-> 3, 4 -> 3, 4 -> 2, 4 -> 5}, 1, 5]
     = {}

    >> g = Graph[{1 -> 2, 2 -> 3, 1 -> 3}, EdgeWeight -> {0.5, a, 3}];
    >> a = 0.5; FindShortestPath[g, 1, 3]
     = {1, 2, 3}
    >> a = 10; FindShortestPath[g, 1, 3]
     = {1, 3}

    #> FindShortestPath[{}, 1, 2]
     : The vertex at position 2 in FindShortestPath[{}, 1, 2] does not belong to the graph at position 1.
     = FindShortestPath[{}, 1, 2]

    #> FindShortestPath[{1 -> 2}, 1, 3]
     : The vertex at position 3 in FindShortestPath[{1 -> 2}, 1, 3] does not belong to the graph at position 1.
     = FindShortestPath[{1 -> 2}, 1, 3]
    """

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
                return Expression(
                    "List",
                    *list(nx.shortest_path(G, source=s, target=t, weight=weight)),
                )
            except nx.exception.NetworkXNoPath:
                return Expression("List")


def _convert_networkx_graph(G, options):
    mapping = dict((v, Integer(i)) for i, v in enumerate(G.nodes))
    G = nx.relabel_nodes(G, mapping)
    edges = [Expression("System`UndirectedEdge", u, v) for u, v in G.edges]
    return Graph(
        G,
        **options,
    )


class VertexAdd(_NetworkXBuiltin):
    """
    >> g1 = Graph[{1 -> 2, 2 -> 3}];
    >> g2 = VertexAdd[g1, 4]
     = -Graph-
    >> g3 = VertexAdd[g2, {5, 10}]
     = -Graph-
    >> VertexAdd[{a -> b}, c]
     = -Graph-
    """

    def apply(self, graph, what, expression, evaluation, options):
        "%(name)s[graph_, what_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            if what.get_head_name() == "System`List":
                return graph.add_vertices(*zip(*[_parse_item(x) for x in what.leaves]))
            else:
                return graph.add_vertices(*zip(*[_parse_item(what)]))


class VertexDelete(_NetworkXBuiltin):
    """
    >> g1 = Graph[{1 -> 2, 2 -> 3, 3 -> 4}];
    >> VertexDelete[g1, 3]
     = -Graph-
    >> VertexDelete[{a -> b, b -> c, c -> d, d -> a}, {a, c}]
     = -Graph-
    >> VertexDelete[{1 -> 2, 2 -> 3, 3 -> 4, 4 -> 6, 6 -> 8, 8 -> 2}, _?OddQ]
     = -Graph-
    """

    def apply(self, graph, what, expression, evaluation, options):
        "%(name)s[graph_, what_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            from mathics.builtin import pattern_objects

            head_name = what.get_head_name()
            if head_name in pattern_objects:
                cases = Expression(
                    "Cases", Expression("List", *graph.vertices.expressions), what
                ).evaluate(evaluation)
                if cases.get_head_name() == "System`List":
                    return graph.delete_vertices(cases.leaves)
            elif head_name == "System`List":
                return graph.delete_vertices(what.leaves)
            else:
                return graph.delete_vertices([what])


class EdgeAdd(_NetworkXBuiltin):
    """
    >> EdgeAdd[{1->2,2->3},3->1]
     = -Graph-
    """

    def apply(self, graph, what, expression, evaluation, options):
        "%(name)s[graph_, what_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            if what.get_head_name() == "System`List":
                return graph.add_edges(*zip(*[_parse_item(x) for x in what.leaves]))
            else:
                return graph.add_edges(*zip(*[_parse_item(what)]))


class EdgeDelete(_NetworkXBuiltin):
    """
    >> Length[EdgeList[EdgeDelete[{a -> b, b -> c, c -> d}, b -> c]]]
     = 2

    >> Length[EdgeList[EdgeDelete[{a -> b, b -> c, c -> b, c -> d}, b <-> c]]]
     = 4

    >> Length[EdgeList[EdgeDelete[{a -> b, b <-> c, c -> d}, b -> c]]]
     = 3

    >> Length[EdgeList[EdgeDelete[{a -> b, b <-> c, c -> d}, c -> b]]]
     = 3

    >> Length[EdgeList[EdgeDelete[{a -> b, b <-> c, c -> d}, b <-> c]]]
     = 2

    >> EdgeDelete[{4<->5,5<->7,7<->9,9<->5,2->4,4->6,6->2}, _UndirectedEdge]
     = -Graph-
    """

    def apply(self, graph, what, expression, evaluation, options):
        "%(name)s[graph_, what_, OptionsPattern[%(name)s]]"
        graph = self._build_graph(graph, evaluation, options, expression)
        if graph:
            from mathics.builtin import pattern_objects

            head_name = what.get_head_name()
            if head_name in pattern_objects:
                cases = Expression(
                    "Cases", Expression("List", *graph.edges.expressions), what
                ).evaluate(evaluation)
                if cases.get_head_name() == "System`List":
                    return graph.delete_edges(cases.leaves)
            elif head_name == "System`List":
                return graph.delete_edges(what.leaves)
            else:
                return graph.delete_edges([what])
