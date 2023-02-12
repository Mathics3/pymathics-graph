import networkx as nx
from mathics.core.symbols import SymbolConstant, SymbolFalse, SymbolTrue

from pymathics.graph.base import Graph


def eval_TreeGraphQ(g: Graph) -> SymbolConstant:
    """
    Returns SymbolTrue if g is a (networkx) tree and SymbolFalse
    otherwise.
    """
    if not isinstance(g, Graph):
        return SymbolFalse
    return SymbolTrue if nx.is_tree(g.G) else SymbolFalse
