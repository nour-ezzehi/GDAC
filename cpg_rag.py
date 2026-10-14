#!/usr/bin/env python3
"""Hybrid GraphRAG: Semantic Entry Point Search for Code Property Graph.

Indexes CPG methods, signatures, class hierarchies, and docstrings using:
  1. Dense Embeddings: sentence-transformers/all-MiniLM-L6-v2 (when available in GPU/Colab)
  2. Lexical Fallback: BM25 / TF-IDF token matching (offline/pure Python fallback)

Matches failing test reports and bug symptoms against repository methods to eliminate
blind node guessing and inject high-confidence entry points into the agent context.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from cpg_bridge import find_nodes, traverse_ast


class CPGSemanticIndex:
    """Indexes CPG methods and performs semantic similarity search against bug queries."""

    def __init__(self, graph: nx.MultiDiGraph, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.graph = graph
        self.model_name = model_name
        self.documents: List[Dict[str, Any]] = []
        self._dense_model: Any = None
        self._dense_embeddings: Any = None
        self._use_dense: bool = False
        self._build_index()

    def _build_index(self) -> None:
        """Extracts non-compiler METHOD nodes, classes, and docstrings from CPG."""
        for m in find_nodes(self.graph, label="METHOD"):
            d = self.graph.nodes[m]
            code = str(d.get("code", "")).strip()

            if not code or code.startswith("<") or code.endswith("<metaClassAdapter>"):
                continue

            # Determine parent class
            class_name = ""
            for p, _, ed in self.graph.in_edges(m, data=True):
                if ed.get("label") == "AST":
                    pdata = self.graph.nodes[p]
                    if pdata.get("label") == "TYPE_DECL":
                        class_name = pdata.get("code", "")
                    elif pdata.get("code") == "<body>":
                        for pp, _, eed in self.graph.in_edges(p, data=True):
                            if eed.get("label") == "AST" and self.graph.nodes[pp].get("label") == "TYPE_DECL":
                                class_name = self.graph.nodes[pp].get("code", "")

            is_test = class_name.startswith("Test") or code.startswith("test_")
            line_no = d.get("line_number")

            # Extract docstrings from AST literal children
            docstring = ""
            try:
                ast_nodes = traverse_ast(self.graph, root_node_id=m, max_depth=3)
                doc_nodes = [
                    str(n.get("code", ""))
                    for n in ast_nodes
                    if n.get("label") == "LITERAL" and ('"""' in str(n.get("code", "")) or "'''" in str(n.get("code", "")))
                ]
                if doc_nodes:
                    docstring = doc_nodes[0].replace('"""', "").replace("'''", "").strip()
            except Exception:
                pass

            qualified_name = f"{class_name}.{code}" if class_name else code
            search_text = f"{class_name} {code} {qualified_name} {docstring}"

            self.documents.append({
                "node_id": m,
                "class_name": class_name,
                "method_name": code,
                "qualified_name": qualified_name,
                "line_number": line_no,
                "docstring": docstring,
                "is_test": is_test,
                "search_text": search_text,
            })

        # Try initializing dense embeddings
        self._init_dense_embeddings()

    def _init_dense_embeddings(self) -> None:
        """Tries to initialize SentenceTransformer; falls back to lexical if unavailable."""
        try:
            from sentence_transformers import SentenceTransformer
            self._dense_model = SentenceTransformer(self.model_name)
            doc_texts = [d["search_text"] for d in self.documents]
            self._dense_embeddings = self._dense_model.encode(doc_texts, convert_to_tensor=True)
            self._use_dense = True
        except Exception:
            self._use_dense = False

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Finds top_k method entry points matching query using dense or lexical search."""
        if not self.documents:
            return []

        if self._use_dense and self._dense_model is not None:
            return self._dense_search(query, top_k)
        return self._lexical_search(query, top_k)

    def _dense_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Dense vector search using sentence-transformers and cosine similarity."""
        from sentence_transformers import util

        query_emb = self._dense_model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_emb, self._dense_embeddings)[0]

        scored = []
        for idx, score in enumerate(cos_scores):
            doc = dict(self.documents[idx])
            s = float(score)
            # Prioritize implementation methods over unit tests
            if not doc["is_test"]:
                s += 0.15
            doc["similarity_score"] = round(s, 4)
            scored.append((s, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def _tokenize(self, text: str) -> List[str]:
        """Extracts sub-tokens, handles camelCase/snake_case, and strips simple suffixes."""
        # Split on non-alphanumeric and underscores
        raw_words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        tokens: List[str] = []
        for w in raw_words:
            if len(w) >= 3:
                tokens.append(w)
                # Handle common verb/plural inflection suffixes
                if w.endswith("ing") and len(w) > 5:
                    tokens.append(w[:-3])
                elif w.endswith("ed") and len(w) > 4:
                    tokens.append(w[:-2])
                elif w.endswith("s") and len(w) > 4:
                    tokens.append(w[:-1])
        return tokens

    def _lexical_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """TF-IDF / Sub-Token Overlap search fallback with discriminative weighting."""
        q_tokens = Counter(self._tokenize(query))
        if not q_tokens:
            return [dict(d) for d in self.documents[:top_k]]

        # Compute document frequencies across indexed methods
        total_docs = len(self.documents)
        df_counter: Counter[str] = Counter()
        doc_token_sets = []
        for doc in self.documents:
            t_set = set(self._tokenize(doc["search_text"]))
            doc_token_sets.append(t_set)
            for t in t_set:
                df_counter[t] += 1

        def idf(t: str) -> float:
            freq = df_counter.get(t, 0)
            return math.log(1.0 + (total_docs - freq + 0.5) / (freq + 0.5))

        scored = []
        for doc, d_tokens in zip(self.documents, doc_token_sets):
            common_tokens = set(q_tokens.keys()) & d_tokens
            if not common_tokens:
                score = 0.0
            else:
                # TF-IDF sum normalized by doc length
                raw_score = sum(q_tokens[t] * idf(t) for t in common_tokens)
                score = raw_score / (math.sqrt(len(d_tokens)) + 1e-5)

                # Prioritize non-test implementation methods
                if not doc["is_test"]:
                    score *= 1.25

                # Exact method/class name presence bonuses
                if doc["method_name"].lower() in query.lower():
                    score += 5.0
                if doc["class_name"] and doc["class_name"].lower() in query.lower():
                    score += 2.0

            doc_copy = dict(doc)
            doc_copy["similarity_score"] = round(float(score), 4)
            scored.append((score, doc_copy))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


def find_semantic_entry_points(
    graph: nx.MultiDiGraph,
    query_text: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Convenience helper to extract top semantic entry points for a query."""
    index = CPGSemanticIndex(graph)
    return index.search(query_text, top_k=top_k)


def format_entry_points_markdown(entry_points: List[Dict[str, Any]]) -> str:
    """Formats top entry points as a markdown guidance block for the system prompt."""
    if not entry_points:
        return ""

    lines = [
        "### Recommended Semantic Entry Points (Hybrid GraphRAG):",
        "Based on semantic similarity with the bug report, the most relevant methods in the CPG are:",
    ]
    for idx, ep in enumerate(entry_points, 1):
        doc = f" - *{ep['docstring'][:65]}...*" if ep.get("docstring") else ""
        lines.append(
            f"{idx}. `{ep['qualified_name']}` (Node #{ep['node_id']}, Line {ep.get('line_number')}) [score: {ep['similarity_score']}]{doc}"
        )
    lines.append("Suggested starting point: Begin your Graph of Thoughts analysis by inspecting one of these methods.")
    return "\n".join(lines)
