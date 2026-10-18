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
    """Indexes CPG methods and performs Asymmetric Field-Weighted Hybrid GraphRAG search
    (fusing Dense Semantic Embeddings, Lexical BM25/TF-IDF, and State-Mutation `writes` Overlap).
    """

    def __init__(self, graph: nx.MultiDiGraph, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.graph = graph
        self.model_name = model_name
        self.documents: List[Dict[str, Any]] = []
        self._dense_model: Any = None
        self._dense_embeddings: Any = None
        self._use_dense: bool = False
        self._build_index()

    def _build_index(self) -> None:
        """Extracts non-compiler METHOD nodes, classes, docstrings, and data-flow summaries from CPG."""
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
            df_summary = d.get("data_flow_summary") or {
                "reads": d.get("reads", []),
                "writes": d.get("writes", []),
                "calls": d.get("calls", []),
            }
            writes_list = list(df_summary.get("writes", [])) if isinstance(df_summary, dict) else []
            reads_list = list(df_summary.get("reads", [])) if isinstance(df_summary, dict) else []
            calls_list = list(df_summary.get("calls", [])) if isinstance(df_summary, dict) else []

            df_tokens = " ".join(reads_list + writes_list + calls_list)
            # Keep clean semantic signature + docstring for dense encoder so long `reads` lists do not dilute embeddings
            semantic_text = f"{class_name} {code} {qualified_name} {docstring} {' '.join(writes_list)}".strip()
            search_text = f"{class_name} {code} {qualified_name} {docstring} {df_tokens}".strip()

            self.documents.append({
                "node_id": m,
                "class_name": class_name,
                "method_name": code,
                "qualified_name": qualified_name,
                "line_number": line_no,
                "docstring": docstring,
                "data_flow_summary": df_summary,
                "writes": writes_list,
                "reads": reads_list,
                "calls": calls_list,
                "is_test": is_test,
                "semantic_text": semantic_text,
                "search_text": search_text,
            })

        # Try initializing dense embeddings
        self._init_dense_embeddings()

    _SHARED_DENSE_MODELS: Dict[str, Any] = {}

    def _init_dense_embeddings(self) -> None:
        """Tries to initialize SentenceTransformer on CPU; falls back to lexical + mutation scoring if unavailable."""
        try:
            from sentence_transformers import SentenceTransformer
            if self.model_name not in self._SHARED_DENSE_MODELS:
                self._SHARED_DENSE_MODELS[self.model_name] = SentenceTransformer(
                    self.model_name, device="cpu"
                )
            self._dense_model = self._SHARED_DENSE_MODELS[self.model_name]
            doc_texts = [d["semantic_text"] for d in self.documents]
            self._dense_embeddings = self._dense_model.encode(doc_texts, convert_to_tensor=True)
            self._use_dense = True
        except Exception:
            self._use_dense = False

    def _tokenize(self, text: str) -> List[str]:
        """Extracts sub-tokens, properly splitting camelCase, PascalCase, and snake_case before lowercasing."""
        if not text:
            return []
        # Split camelCase and PascalCase boundaries BEFORE lowercasing
        de_camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        de_camel = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", de_camel)
        raw_words = re.findall(r"[a-zA-Z0-9]+", de_camel.lower())
        tokens: List[str] = []
        for w in raw_words:
            if len(w) >= 2:
                tokens.append(w)
                # Handle common verb/plural inflection suffixes
                if w.endswith("ing") and len(w) > 5:
                    tokens.append(w[:-3])
                elif w.endswith("ed") and len(w) > 4:
                    tokens.append(w[:-2])
                elif w.endswith("s") and len(w) > 4:
                    tokens.append(w[:-1])
        return tokens

    def _extract_variable_tokens(self, exprs: List[str]) -> Set[str]:
        """Extracts normalized full expressions, tail field names, and sub-tokens from variable expressions."""
        out: Set[str] = set()
        for expr in exprs:
            clean = expr.strip().lower()
            if not clean:
                continue
            out.add(clean)
            # Strip `self.` / `this->` prefixes for direct field matching
            for prefix in ("self.", "this->", "this."):
                if clean.startswith(prefix):
                    out.add(clean[len(prefix):])
            # Add member/field tail (e.g. `item.price` -> `price`, `acct->balance` -> `balance`)
            parts = re.split(r"(?:\.|->|\[|\])+", clean)
            for p in parts:
                if p and p not in ("self", "this") and len(p) >= 2:
                    out.add(p)
            for tok in self._tokenize(expr):
                if tok not in ("self", "this"):
                    out.add(tok)
        return out

    def _extract_query_variable_candidates(self, query: str) -> Set[str]:
        """Extracts code identifiers, dotted/arrow expressions, and domain tokens from the bug query ($V_{query}$)."""
        candidates: Set[str] = set()
        # 1. Explicit dotted/arrow/indexed expressions in query (e.g. `item.price`, `acct->balance`, `summary[cat]`)
        for m in re.finditer(r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:(?:\.|->)[a-zA-Z_][a-zA-Z0-9_]*)+\b", query):
            expr = m.group(0).lower()
            candidates.add(expr)
            for part in re.split(r"(?:\.|->)", expr):
                if part and part not in ("self", "this") and len(part) >= 2:
                    candidates.add(part)
        # 2. Snake_case or camelCase identifiers in query
        for m in re.finditer(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", query):
            ident = m.group(0)
            if "_" in ident or any(c.isupper() for c in ident[1:]):
                candidates.add(ident.lower())
        # 3. Sub-tokens (excluding generic test/error stopwords)
        stopwords = {
            "test", "tests", "failed", "failure", "error", "assert", "assertion",
            "assertionerror", "expected", "result", "resulted", "instead", "when",
            "with", "from", "that", "this", "self", "true", "false", "none",
        }
        for tok in self._tokenize(query):
            if tok not in stopwords and len(tok) >= 3:
                candidates.add(tok)
        return candidates

    def _compute_mutation_and_matched_vars(
        self, doc: Dict[str, Any], query: str, q_var_candidates: Set[str]
    ) -> Tuple[float, List[str]]:
        """Computes Asymmetric State-Mutation score boost:
        $\\gamma \\cdot |V_{writes}(m) \\cap V_{query}| / (|V_{query}| + \\epsilon)$
        and extracts matched `target_vars` for Stage 2 program slicing handoff.
        """
        if not q_var_candidates:
            return 0.0, []

        writes_list: List[str] = doc.get("writes", [])
        reads_list: List[str] = doc.get("reads", [])
        calls_list: List[str] = doc.get("calls", [])

        v_writes = self._extract_variable_tokens(writes_list)
        v_reads = self._extract_variable_tokens(reads_list)
        v_calls = self._extract_variable_tokens(calls_list)

        writes_overlap = v_writes & q_var_candidates
        reads_overlap = v_reads & q_var_candidates
        calls_overlap = v_calls & q_var_candidates

        denom = len(q_var_candidates) + 1e-5
        # Asymmetric weighting: state mutations (`writes`) receive strong priority over passive `reads`
        write_ratio = len(writes_overlap) / denom
        read_ratio = len(reads_overlap) / denom
        call_ratio = len(calls_overlap) / denom

        # Exact dotted/arrow expression write bonus (e.g., `item.price` or `acct->balance` in query AND writes)
        query_lower = query.lower()
        exact_write_hit = any(
            w.lower() in query_lower or w.lower().removeprefix("self.") in query_lower
            for w in writes_list
            if len(w) >= 3
        )

        mutation_boost = (
            1.8 * write_ratio
            + (0.75 if exact_write_hit else 0.0)
            + 0.45 * call_ratio
            + 0.25 * read_ratio
        )

        # Determine concrete `target_vars` in this method that matched the query for Stage 2 slicing
        matched_vars: List[str] = []
        seen_vars: Set[str] = set()
        for expr in writes_list + reads_list:
            expr_tokens = self._extract_variable_tokens([expr])
            if expr_tokens & q_var_candidates:
                # Normalize `self.foo` -> `foo` or keep clean identifier
                clean_var = expr.strip()
                if clean_var not in seen_vars and clean_var != "self":
                    seen_vars.add(clean_var)
                    matched_vars.append(clean_var)

        return mutation_boost, matched_vars[:4]

    def _compute_lexical_scores(self, query: str) -> List[float]:
        """Computes TF-IDF / BM25-style lexical scores across all indexed documents."""
        q_tokens = Counter(self._tokenize(query))
        if not q_tokens:
            return [0.0 for _ in self.documents]

        total_docs = len(self.documents)
        df_counter: Counter[str] = Counter()
        doc_token_counts: List[Counter[str]] = []
        for doc in self.documents:
            t_counts = Counter(self._tokenize(doc["search_text"]))
            doc_token_counts.append(t_counts)
            for t in t_counts.keys():
                df_counter[t] += 1

        def idf(t: str) -> float:
            freq = df_counter.get(t, 0)
            return math.log(1.0 + (total_docs - freq + 0.5) / (freq + 0.5))

        scores: List[float] = []
        for doc, d_counts in zip(self.documents, doc_token_counts):
            common_tokens = set(q_tokens.keys()) & set(d_counts.keys())
            if not common_tokens:
                score = 0.0
            else:
                # Sub-linear TF saturation (BM25-style) weighted by IDF
                raw_score = sum(
                    q_tokens[t] * idf(t) * (1.0 + math.log(1.0 + d_counts[t]))
                    for t in common_tokens
                )
                score = raw_score / (math.sqrt(sum(d_counts.values())) + 1e-5)

            # Exact method/class name presence bonuses
            if doc["method_name"].lower() in query.lower():
                score += 3.5
            if doc["class_name"] and doc["class_name"].lower() in query.lower():
                score += 1.5

            scores.append(score)
        return scores

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Finds top_k method entry points using Asymmetric Field-Weighted Hybrid Scoring:
        $S_{entry}(m, Q) = \\alpha \\cdot CosSim(E(m_{doc}), E(Q)) + \\beta \\cdot BM25(m_{text}, Q) + \\gamma \\cdot MutationOverlap(m, Q)$
        """
        if not self.documents:
            return []

        q_var_candidates = self._extract_query_variable_candidates(query)
        lexical_scores = self._compute_lexical_scores(query)

        dense_scores: List[float] = [0.0 for _ in self.documents]
        if self._use_dense and self._dense_model is not None:
            try:
                from sentence_transformers import util
                query_emb = self._dense_model.encode(query, convert_to_tensor=True)
                cos_scores = util.cos_sim(query_emb, self._dense_embeddings)[0]
                dense_scores = [float(s) for s in cos_scores]
            except Exception:
                dense_scores = [0.0 for _ in self.documents]

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for idx, doc in enumerate(self.documents):
            doc_copy = dict(doc)
            lex_s = lexical_scores[idx]
            dense_s = dense_scores[idx]
            mut_boost, matched_vars = self._compute_mutation_and_matched_vars(
                doc_copy, query, q_var_candidates
            )

            if self._use_dense:
                total_score = 0.45 * dense_s + 0.35 * lex_s + 0.50 * mut_boost
            else:
                total_score = 0.65 * lex_s + 0.75 * mut_boost

            # Prioritize implementation methods over unit tests
            if not doc_copy["is_test"]:
                total_score = total_score * 1.25 + (0.15 if self._use_dense else 0.0)

            doc_copy["similarity_score"] = round(float(total_score), 4)
            doc_copy["mutation_boost"] = round(float(mut_boost), 4)
            doc_copy["matched_target_vars"] = matched_vars
            scored.append((total_score, doc_copy))

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
    """Formats top entry points as a markdown guidance block for the system prompt,
    including Stage 1 -> Stage 2 `matched_target_vars` slicing hints.
    """
    if not entry_points:
        return ""

    lines = [
        "### Recommended Semantic Entry Points (Hybrid GraphRAG):",
        "Based on semantic similarity and state-mutation (`writes`) analysis, the most relevant methods in the CPG are:",
    ]
    for idx, ep in enumerate(entry_points, 1):
        doc = f" - *{ep['docstring'][:65]}...*" if ep.get("docstring") else ""
        df_sum = ep.get("data_flow_summary") or {}
        df_str = ""
        if df_sum and any(df_sum.values()):
            df_str = f" | writes={df_sum.get('writes', [])}, calls={df_sum.get('calls', [])}"
        matched_vars = ep.get("matched_target_vars") or []
        slice_hint = (
            f" | matched_target_vars={matched_vars}"
            if matched_vars
            else ""
        )
        lines.append(
            f"{idx}. `{ep['qualified_name']}` (Node #{ep['node_id']}, Line {ep.get('line_number')}) [score: {ep['similarity_score']}]{doc}{df_str}{slice_hint}"
        )
    top_ep = entry_points[0]
    top_vars = top_ep.get("matched_target_vars") or (top_ep.get("data_flow_summary") or {}).get("writes", [])[:2]
    if top_vars:
        lines.append(
            f"Suggested starting point: Inspect `{top_ep['qualified_name']}` (Node #{top_ep['node_id']}) "
            f"or slice directly with `condense_method_slice(target=\"{top_ep['node_id']}\", target_vars={top_vars})`."
        )
    else:
        lines.append("Suggested starting point: Begin your Graph of Thoughts analysis by inspecting one of these methods.")
    return "\n".join(lines)
