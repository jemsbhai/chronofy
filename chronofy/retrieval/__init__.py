"""Retrieval layer components for Chronofy Layer 2.

Core exports (no optional dependencies):
- EpistemicFilter: τ-threshold filtering for temporal validity
- TemporalTriple: timestamped KG triple data model
- TemporalKnowledgeGraph: indexed triple store (directed + undirected)

Graph extras (requires ``networkx`` — install via ``pip install chronofy[graph]``):
- TemporalRule: abstracted temporal rule with statistics
- RuleMiner: Apriori-style temporal rule extraction
- TemporalRuleGraph: MDL-optimized decay-weighted PageRank graph
"""

from chronofy.retrieval.filter import EpistemicFilter
from chronofy.retrieval.triples import TemporalKnowledgeGraph, TemporalTriple

__all__ = [
    "EpistemicFilter",
    "TemporalTriple",
    "TemporalKnowledgeGraph",
]

# Graph components require networkx — graceful degradation
try:
    from chronofy.retrieval.graph import TemporalRuleGraph
    from chronofy.retrieval.rules import RuleMiner, TemporalRule

    __all__ += ["TemporalRule", "RuleMiner", "TemporalRuleGraph"]
except ImportError:
    pass
