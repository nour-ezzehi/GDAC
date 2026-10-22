"""Semantic Code Search Tool for GDAC Agents."""

from __future__ import annotations

import json
from typing import Any, Dict, List
import networkx as nx

from gdac.services.cpg_rag import find_semantic_entry_points


def semantic_code_search(
    graph: nx.MultiDiGraph,
    query: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Performs semantic similarity search against all repository methods in the CPG."""
    results = find_semantic_entry_points(graph, query, top_k=top_k)
    formatted = []
    for r in results:
        formatted.append({
            "qualified_name": r.get("qualified_name"),
            "node_id": r.get("node_id"),
            "line": r.get("line_number"),
            "similarity_score": r.get("similarity_score"),
            "docstring": r.get("docstring", "")[:120],
        })
    return {"query": query, "top_matches": formatted}
