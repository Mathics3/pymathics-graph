"""
Curated Graphs
"""

import networkx as nx
from typing import Callable, Dict, Optional, Tuple
from mathics.core.evaluation import Evaluation

from pymathics.graph.base import Graph, _NetworkXBuiltin, graph_helper


class GraphData(_NetworkXBuiltin):
    """
    <url>
    :WMA link:https://reference.wolfram.com/language/ref/GraphData.html</url>

    <dl>
      <dt>'GraphData[$name$]'
      <dd>Returns a graph with the specified name.
    </dl>

    >> GraphData["PappusGraph"]
     = -Graph-
    """

    summary_text = "create a graph by name"

    def eval(
        self, name, expression, evaluation: Evaluation, options: dict
    ) -> Optional[Graph]:
        "GraphData[name_String, OptionsPattern[GraphData]]"
        py_name = name.get_string_value()
        fn, layout = WL_TO_NETWORKX_FN.get(py_name, (None, None))
        if not fn:
            if not py_name.endswith("_graph"):
                py_name += "_graph"
            if py_name in ("LCF_graph", "make_small_graph"):
                # These graphs require parameters
                return
            import inspect

            fn = dict(inspect.getmembers(nx, inspect.isfunction)).get(py_name, None)
            # parameters = inspect.signature(nx.diamond_graph).parameters.values()
            # if len([p for p in list(parameters) if p.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]]) != 0:
            #     return
        if fn:
            g = graph_helper(fn, options, False, layout, evaluation)
            if g is not None:
                g.G.name = py_name
            return g


WL_TO_NETWORKX_FN: Dict[str, Tuple[Callable, Optional[str]]] = {
    "DodecahedralGraph": (nx.dodecahedral_graph, None),
    "DiamondGraph": (nx.diamond_graph, "spring"),
    "PappusGraph": (nx.pappus_graph, "circular"),
    "IsohedralGraph": (nx.icosahedral_graph, "spring"),
    "PetersenGraph": (nx.petersen_graph, None),
}

# TODO: ExampleData
