"""CPG Traversal and Slicing Tools for GDAC Agents."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from gdac.services.cpg_parser import (
    find_nodes,
    get_node_label_summary,
    trace_variable_to_sink,
    traverse_ast,
    traverse_cfg,
    traverse_data_flow,
)
from gdac.services.cpg_condenser import (
    condense_method_slice as svc_condense_method_slice,
    resolve_method_id,
)
from gdac.services.test_runner import get_project_root

_SUPPORTED_SOURCE_EXTENSIONS = (
    ".py", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp",
    ".java", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs",
    ".cs", ".rb", ".php", ".kt", ".swift", ".scala",
)


def resolve_target_node(graph: nx.MultiDiGraph, target: Any) -> Optional[int]:
    """Dynamically resolves a target to a CPG node ID.

    Supports:
      1. Direct integer node ID: 107374182415 or '#107374182415'
      2. Qualified names: 'InventoryManager.apply_discount' or 'TestInventoryManager.test_apply_discount'
      3. Simple method/symbol names: 'apply_discount'
    Prioritizes real implementation bodies over compiler adapters (<metaClassAdapter>).
    """
    if target is None:
        return None

    # Case 1: Numeric ID
    if isinstance(target, int):
        return target if target in graph else None
    target_str = str(target).strip()
    clean_id = target_str.lstrip("#")
    if clean_id.isdigit():
        nid = int(clean_id)
        if nid in graph:
            return nid

    # Case 2: Dot-separated qualified name e.g. InventoryManager.apply_discount
    if "." in target_str:
        parts = target_str.split(".")
        meth_part = parts[-1].strip()
        cls_part = parts[-2].strip().lower() if len(parts) > 1 else ""
        candidates = find_nodes(graph, label="METHOD", code_contains=meth_part)
        filtered = []
        for c in candidates:
            matched = False
            for p, _, ed in graph.in_edges(c, data=True):
                if ed.get("label") == "AST":
                    pdata = graph.nodes[p]
                    if cls_part and cls_part in str(pdata.get("code", "")).lower():
                        matched = True
                        break
                    elif pdata.get("code") == "<body>":
                        for pp, _, eed in graph.in_edges(p, data=True):
                            if eed.get("label") == "AST" and cls_part in str(graph.nodes[pp].get("code", "")).lower():
                                matched = True
                                break
            if matched or not cls_part:
                filtered.append(c)
        candidates = filtered if filtered else candidates
    else:
        candidates = find_nodes(graph, label="METHOD", code_contains=target_str)
        if not candidates:
            candidates = find_nodes(graph, label="METHOD", name=target_str)

    # Filter out compiler-generated adapter wrappers (<metaClassAdapter>)
    real_nodes = [
        n for n in candidates
        if "<metaClassAdapter>" not in str(graph.nodes[n].get("code", ""))
        and "<metaClassAdapter>" not in str(graph.nodes[n].get("name", ""))
    ]
    if real_nodes:
        return real_nodes[0]
    return candidates[0] if candidates else None


def resolve_method_node(graph: nx.MultiDiGraph, method_name: str) -> Optional[int]:
    """Backward compatible wrapper around resolve_target_node."""
    return resolve_target_node(graph, method_name)


def resolve_method_source_file(
    graph: nx.MultiDiGraph,
    method_id: int,
    repo_root: str | Path = "stackflow",
) -> Optional[Path]:
    """Resolves a CPG METHOD node to its physical source file on disk across any language."""
    base_dir = Path(repo_root)
    if not base_dir.is_absolute():
        project_root = get_project_root()
        base_dir = project_root / base_dir

    if method_id not in graph:
        return None

    def _try_resolve_relpath(raw_path: str) -> Optional[Path]:
        if not raw_path or raw_path == "empty" or raw_path.startswith("<"):
            return None
        clean_p = raw_path.split(":")[0].strip()
        if not clean_p or not any(clean_p.endswith(ext) for ext in _SUPPORTED_SOURCE_EXTENSIONS):
            return None
        project_root = get_project_root()
        for root_cand in (base_dir, base_dir.parent, project_root):
            cand = root_cand / clean_p
            if cand.is_file():
                return cand
        return None

    node_data = graph.nodes[method_id]
    # 1. Direct filename or full_name attribute on the METHOD node or its AST ancestors
    curr_check: Optional[int] = method_id
    visited_anc: Set[int] = set()
    while curr_check is not None and curr_check not in visited_anc:
        visited_anc.add(curr_check)
        cdata = graph.nodes.get(curr_check, {})
        for key in ("filename", "file", "filepath", "full_name", "code"):
            resolved = _try_resolve_relpath(str(cdata.get(key) or ""))
            if resolved is not None:
                return resolved
        next_parent = None
        for p, _, ed in graph.in_edges(curr_check, data=True):
            if ed.get("label") == "AST":
                next_parent = p
                break
        curr_check = next_parent

    # 2. Match method name, parent class/struct/type, and line number in repo_root across languages
    method_name = str(node_data.get("name") or node_data.get("code") or "").strip()
    if "::" in method_name:
        method_name = method_name.split("::")[-1].strip()
    if "." in method_name:
        method_name = method_name.split(".")[-1].strip()
    m_line = node_data.get("line_number")

    parent_class = None
    for p, _, ed in graph.in_edges(method_id, data=True):
        if ed.get("label") == "AST":
            pdata = graph.nodes[p]
            pcode = str(pdata.get("code") or pdata.get("name") or "")
            if pdata.get("label") == "TYPE_DECL" and not pcode.endswith("<metaClassAdapter>"):
                parent_class = pcode
                break
            elif pcode == "<body>":
                for pp, _, eed in graph.in_edges(p, data=True):
                    if eed.get("label") == "AST" and graph.nodes[pp].get("label") == "TYPE_DECL":
                        ppcode = str(graph.nodes[pp].get("code") or graph.nodes[pp].get("name") or "")
                        if not ppcode.endswith("<metaClassAdapter>"):
                            parent_class = ppcode
                            break

    if not base_dir.exists():
        return None

    source_files = sorted(
        f for f in base_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in _SUPPORTED_SOURCE_EXTENSIONS
    )
    best_file: Optional[Path] = None
    best_score = -1

    def_pattern = (
        re.compile(
            rf"(?:^\s*(?:(?:pub|public|private|protected|internal|static|virtual|override|async|inline|constexpr|extern)\s+)*"
            rf"(?:def|fn|func|function|fun)\s+{re.escape(method_name)}\s*[\(<]"
            rf"|^\s*[\w:<>\[\]\*&\s]+?\b{re.escape(method_name)}\s*\()"
        )
        if method_name
        else None
    )
    cls_pattern = (
        re.compile(rf"\b(?:class|struct|interface|impl|type|trait|enum|module)\s+{re.escape(parent_class)}\b")
        if parent_class
        else None
    )

    for src_file in source_files:
        try:
            lines = src_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        score = 0
        if cls_pattern and any(cls_pattern.search(line) for line in lines):
            score += 10

        for idx, line in enumerate(lines, start=1):
            if def_pattern and def_pattern.search(line):
                curr_score = score + 20
                if isinstance(m_line, int):
                    dist = abs(idx - m_line)
                    if dist == 0:
                        curr_score += 50
                    elif dist <= 5:
                        curr_score += max(0, 30 - dist * 5)
                if curr_score > best_score:
                    best_score = curr_score
                    best_file = src_file

    return best_file


def materialize_source_window(
    file_path: Path,
    line_number: int,
    window_size: int = 15,
) -> Dict[str, Any]:
    """Extracts ONLY the surrounding `window_size` lines (default 15: +/- 7 lines) from disk."""
    lines = file_path.read_text(encoding="utf-8").splitlines()
    total_lines = len(lines)
    half = window_size // 2
    center = max(1, min(int(line_number), total_lines))
    start_line = max(1, center - half)
    end_line = min(total_lines, start_line + window_size - 1)
    if end_line - start_line + 1 < window_size:
        start_line = max(1, end_line - window_size + 1)

    window_lines = lines[start_line - 1 : end_line]
    numbered_window = "\n".join(
        f"L{ln}: {text}" for ln, text in zip(range(start_line, end_line + 1), window_lines)
    )
    raw_window = "\n".join(window_lines)
    return {
        "file_path": str(file_path),
        "start_line": start_line,
        "end_line": end_line,
        "target_line": center,
        "raw_window": raw_window,
        "numbered_window": numbered_window,
        "window_lines": window_lines,
    }


def _normalize_expr_for_match(expr: str) -> str:
    """Normalizes whitespace and redundant grouping parentheses for CPG-to-disk matching."""
    return re.sub(r"[\s()]+", "", expr.strip())
