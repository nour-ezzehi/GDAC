#!/usr/bin/env python3
"""CPG Python Bridge: NetworkX Loader and Graph Traversals for Joern CPG14.

Loads Joern-exported CPG14 files (JSON or DOT) into a NetworkX MultiDiGraph and provides
three distinct traversal functions:
  1. AST Traversal: Syntactic structure exploration (AST hierarchy).
  2. CFG Traversal: Control flow & execution path tracing.
  3. Data Flow Traversal: Reaching definitions & variable taint propagation (DDG).
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

import networkx as nx


# ==============================================================================
# 1. Graph Loading & Construction
# ==============================================================================

def _normalize_node(raw_node: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """Normalizes raw node dictionary into (node_id, attributes)."""
    node_id = int(raw_node.get("id", raw_node.get("_id", 0)))
    attrs: Dict[str, Any] = {
        "id": node_id,
        "label": raw_node.get("label", raw_node.get("_label", "UNKNOWN")),
    }

    properties = raw_node.get("properties", {})
    if isinstance(properties, dict):
        for k, v in properties.items():
            attrs[k.lower()] = v
    for k, v in raw_node.items():
        if k not in ("id", "_id", "label", "_label", "properties"):
            attrs[k.lower()] = v

    if "name" not in attrs and "name" in raw_node:
        attrs["name"] = raw_node["name"]
    if "code" not in attrs and "code" in raw_node:
        attrs["code"] = raw_node["code"]
    if "line_number" not in attrs:
        attrs["line_number"] = attrs.get("linenumber") or attrs.get("line_number")

    return node_id, attrs


def _normalize_edge(raw_edge: Dict[str, Any]) -> Tuple[int, int, Dict[str, Any]]:
    """Normalizes raw edge dictionary into (src_id, dst_id, attributes)."""
    src_id = int(raw_edge.get("out", raw_edge.get("source", raw_edge.get("src", 0))))
    dst_id = int(raw_edge.get("in", raw_edge.get("target", raw_edge.get("dst", 0))))

    edge_label = raw_edge.get("label", raw_edge.get("_label", "UNKNOWN")).upper()
    edge_attrs: Dict[str, Any] = {"label": edge_label}

    for k, v in raw_edge.items():
        if k not in ("out", "in", "source", "target", "src", "dst", "label", "_label"):
            edge_attrs[k.lower()] = v

    return src_id, dst_id, edge_attrs


def _parse_dot_file(fpath: Path, graph: nx.MultiDiGraph) -> None:
    """Parses Joern CPG14 dot file into NetworkX graph."""
    node_pattern = re.compile(r"\"(\d+)\"\s*\[\s*label\s*=\s*<(.*?)>\s*\]", re.DOTALL)
    edge_pattern = re.compile(r"\"(\d+)\"\s*->\s*\"(\d+)\"\s*\[\s*label\s*=\s*\"([^\"]*)\"\s*\]")

    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    for m in node_pattern.finditer(content):
        nid = int(m.group(1))
        raw_label = html.unescape(m.group(2))
        parts = [p.strip() for p in raw_label.split("<BR/>") if p.strip()]
        header = parts[0] if parts else ""
        hdr_parts = [h.strip() for h in header.split(",", 1)]
        label = hdr_parts[0] if hdr_parts else "NODE"
        line = int(hdr_parts[1]) if len(hdr_parts) > 1 and hdr_parts[1].isdigit() else None
        code = parts[1] if len(parts) > 1 else ""

        graph.add_node(
            nid,
            id=nid,
            label=label,
            code=code,
            name=code.split()[0] if code else label,
            line_number=line,
        )

    for m in edge_pattern.finditer(content):
        src, dst = int(m.group(1)), int(m.group(2))
        raw_elbl = html.unescape(m.group(3)).strip()
        edge_parts = raw_elbl.split(":", 1)
        etype = edge_parts[0].strip().upper()
        var = edge_parts[1].strip() if len(edge_parts) > 1 else None

        edge_attrs: Dict[str, Any] = {"label": etype}
        if var:
            edge_attrs["variable"] = var
        graph.add_edge(src, dst, **edge_attrs)


def load_cpg_json(path_or_dir: str | Path) -> nx.MultiDiGraph:
    """Loads Joern CPG14 export (JSON or directory of files) into a NetworkX MultiDiGraph.

    Supports:
      - A single .json file (e.g. cpg_json/cpg.json)
      - A directory containing cpg.json or *.json files
      - A directory of exported Joern *.dot files
    """
    path = Path(path_or_dir)
    if not path.exists():
        raise FileNotFoundError(f"CPG path '{path}' does not exist.")

    graph = nx.MultiDiGraph()

    # Case 1: Directory with cpg.json
    consolidated_json = path / "cpg.json" if path.is_dir() else None
    if consolidated_json and consolidated_json.exists():
        with open(consolidated_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for n in data.get("nodes", []):
            nid, attrs = _normalize_node(n)
            graph.add_node(nid, **attrs)
        for e in data.get("edges", []):
            src, dst, attrs = _normalize_edge(e)
            graph.add_edge(src, dst, **attrs)
        return graph

    # Case 2: Individual JSON files
    json_files = [path] if path.is_file() and path.suffix == ".json" else sorted(path.glob("*.json"))
    if json_files:
        for file_path in json_files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                raw_nodes = data.get("nodes", [])
                raw_edges = data.get("edges", [])
            elif isinstance(data, list):
                raw_nodes = [item for item in data if "out" not in item and "source" not in item]
                raw_edges = [item for item in data if "out" in item or "source" in item]
            else:
                continue
            for n in raw_nodes:
                nid, attrs = _normalize_node(n)
                graph.add_node(nid, **attrs)
            for e in raw_edges:
                src, dst, attrs = _normalize_edge(e)
                graph.add_edge(src, dst, **attrs)
        return graph

    # Case 3: DOT files from joern-export
    dot_files = [path] if path.is_file() and path.suffix == ".dot" else sorted(path.glob("*.dot"))
    if dot_files:
        for fpath in dot_files:
            _parse_dot_file(fpath, graph)
        return graph

    raise ValueError(f"No CPG JSON or DOT files found at '{path}'.")


# ==============================================================================
# 2. Node Inspection Helpers
# ==============================================================================

def find_nodes(
    graph: nx.MultiDiGraph,
    label: Optional[str] = None,
    name: Optional[str] = None,
    code_contains: Optional[str] = None,
) -> List[int]:
    """Finds node IDs matching given label, name, or code substring."""
    matches = []
    for node_id, data in graph.nodes(data=True):
        if label and data.get("label", "").upper() != label.upper():
            continue
        if name and name not in str(data.get("name", "")):
            continue
        if code_contains and code_contains not in str(data.get("code", "")):
            continue
        matches.append(node_id)
    return matches


def get_node_label_summary(graph: nx.MultiDiGraph, node_id: int) -> str:
    """Returns a concise human-readable description of a CPG node."""
    data = graph.nodes.get(node_id, {})
    label = data.get("label", "NODE")
    code = data.get("code", "") or data.get("name", "")
    line = data.get("line_number", "")
    line_str = f"L{line}: " if line else ""

    if len(code) > 50:
        code = code[:47] + "..."
    return f"[{label} #{node_id}] {line_str}{code}".strip()


# ==============================================================================
# 3. Three Distinct Graph Traversal Functions
# ==============================================================================

def traverse_ast(
    graph: nx.MultiDiGraph,
    root_node_id: int,
    filter_labels: Optional[List[str]] = None,
    max_depth: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Traversal 1: AST (Abstract Syntax Tree) Hierarchy Traversal.

    Explores syntactic tree structure starting from `root_node_id` by traversing
    edges with label == 'AST'. Useful for inspecting function bodies,
    nested expressions, parameters, and AST node types.
    """
    if root_node_id not in graph:
        raise ValueError(f"Root node {root_node_id} not found in graph.")

    allowed_labels = {lbl.upper() for lbl in filter_labels} if filter_labels else None
    results: List[Dict[str, Any]] = []
    visited: Set[int] = set()

    def dfs(curr_id: int, depth: int) -> None:
        if curr_id in visited:
            return
        visited.add(curr_id)

        node_data = graph.nodes.get(curr_id, {})
        node_label = node_data.get("label", "").upper()

        if allowed_labels is None or node_label in allowed_labels:
            results.append({
                "node_id": curr_id,
                "depth": depth,
                "label": node_label,
                "name": node_data.get("name"),
                "code": node_data.get("code"),
                "line_number": node_data.get("line_number"),
                "data": node_data,
            })

        if max_depth is not None and depth >= max_depth:
            return

        for _, neighbor, edge_data in graph.out_edges(curr_id, data=True):
            if edge_data.get("label") == "AST":
                dfs(neighbor, depth + 1)

    dfs(root_node_id, depth=0)
    return results


def traverse_cfg(
    graph: nx.MultiDiGraph,
    start_node_id: int,
    target_node_id: Optional[int] = None,
    max_paths: int = 10,
    max_depth: int = 50,
) -> List[List[int]]:
    """Traversal 2: Control Flow Graph (CFG) Execution Path Traversal.

    Follows directed 'CFG' edges to trace program execution paths, conditional
    branches, and reachability.
    """
    if start_node_id not in graph:
        raise ValueError(f"Start node {start_node_id} not found in graph.")

    cfg_edges = [
        (u, v)
        for u, v, data in graph.edges(data=True)
        if data.get("label") == "CFG"
    ]
    cfg_graph = nx.DiGraph()
    cfg_graph.add_nodes_from(graph.nodes(data=True))
    cfg_graph.add_edges_from(cfg_edges)

    if target_node_id is not None:
        if target_node_id not in cfg_graph:
            return []
        paths: List[List[int]] = []
        try:
            for p in nx.all_simple_paths(
                cfg_graph, source=start_node_id, target=target_node_id, cutoff=max_depth
            ):
                paths.append(p)
                if len(paths) >= max_paths:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return paths

    paths = []

    def explore(curr_path: List[int]) -> None:
        if len(paths) >= max_paths:
            return
        curr = curr_path[-1]
        successors = list(cfg_graph.successors(curr))
        valid_succs = [s for s in successors if s not in curr_path]

        if not valid_succs or len(curr_path) >= max_depth:
            paths.append(list(curr_path))
            return

        for s in valid_succs:
            explore(curr_path + [s])

    explore([start_node_id])
    return paths


def traverse_data_flow(
    graph: nx.MultiDiGraph,
    source_node_id: int,
    var_name: Optional[str] = None,
    sink_node_id: Optional[int] = None,
    max_hops: int = 30,
) -> List[List[Dict[str, Any]]]:
    """Traversal 3: Data Flow & Reaching Definitions (DDG / Taint) Traversal.

    Traces data propagation across DDG and REACHING_DEF edges.
    Tracks how a value (e.g., parameter or variable assignment) flows
    through code to downstream uses or sinks.
    """
    if source_node_id not in graph:
        raise ValueError(f"Source node {source_node_id} not found in graph.")

    df_edge_labels = {"REACHING_DEF", "DDG", "ARGUMENT", "REF"}
    discovered_traces: List[List[int]] = []

    def explore(curr_path: List[int]) -> None:
        curr_id = curr_path[-1]

        if sink_node_id is not None and curr_id == sink_node_id:
            discovered_traces.append(list(curr_path))
            return

        if len(curr_path) >= max_hops:
            discovered_traces.append(list(curr_path))
            return

        next_nodes = []
        for _, dst, edge_data in graph.out_edges(curr_id, data=True):
            label = edge_data.get("label", "").upper()
            if label in df_edge_labels:
                edge_var = edge_data.get("variable") or edge_data.get("name")
                if var_name and edge_var and edge_var != var_name:
                    continue
                if dst not in curr_path:
                    next_nodes.append(dst)

        if not next_nodes:
            if sink_node_id is None and len(curr_path) > 1:
                discovered_traces.append(list(curr_path))
            return

        for nxt in next_nodes:
            explore(curr_path + [nxt])

    explore([source_node_id])

    formatted_traces: List[List[Dict[str, Any]]] = []
    for path in discovered_traces:
        trace = []
        for nid in path:
            ndata = graph.nodes.get(nid, {})
            trace.append({
                "node_id": nid,
                "label": ndata.get("label", "NODE"),
                "name": ndata.get("name"),
                "code": ndata.get("code"),
                "line_number": ndata.get("line_number"),
            })
        formatted_traces.append(trace)

    return formatted_traces


# ==============================================================================
# 4. CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load Joern CPG14 into NetworkX and run graph traversals."
    )
    parser.add_argument(
        "--cpg-path",
        type=str,
        default="cpg_json",
        help="Path to CPG JSON or directory of exported files (default: cpg_json)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Target method name or substring to inspect (e.g. apply_discount)",
    )

    args = parser.parse_args()

    cpg_path = Path(args.cpg_path)
    if not cpg_path.exists():
        print(f"[-] Error: CPG path '{cpg_path}' not found.", file=sys.stderr)
        print("    Run `python joern_extraction.py` first to generate it.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading CPG from '{cpg_path}'...")
    graph = load_cpg_json(cpg_path)
    print(f"[+] Loaded Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.\n")

    target_method = args.method or "apply_discount"
    print(f"[*] Inspecting target method: '{target_method}'...")
    method_nodes = find_nodes(graph, label="METHOD", code_contains=target_method)
    if not method_nodes:
        method_nodes = find_nodes(graph, label="METHOD")

    if not method_nodes:
        print(f"[-] No METHOD nodes found in graph.")
        return

    m_id = method_nodes[0]
    m_info = graph.nodes[m_id]
    print(f"[+] Method Node #{m_id}: {m_info.get('code') or m_info.get('name')}")

    # 1. AST Traversal
    print("\n" + "=" * 60)
    print("--- [1] AST Traversal (Top 10 Syntactic Subtree Elements) ---")
    ast_nodes = traverse_ast(graph, root_node_id=m_id, max_depth=3)
    for n in ast_nodes[:10]:
        indent = "  " * n["depth"]
        print(f"{indent}- [{n['label']}] #{n['node_id']} L{n.get('line_number')}: {n.get('code')}")

    # 2. CFG Traversal
    print("\n" + "=" * 60)
    print("--- [2] CFG Traversal (Execution Paths from Method Entry) ---")
    cfg_paths = traverse_cfg(graph, start_node_id=m_id, max_paths=3)
    for idx, path in enumerate(cfg_paths, 1):
        print(f"Execution Path #{idx}:")
        for nid in path[:6]:
            print(f"    -> {get_node_label_summary(graph, nid)}")
        if len(path) > 6:
            print(f"    -> ... ({len(path) - 6} more steps)")

    # 3. Data Flow Traversal
    print("\n" + "=" * 60)
    print("--- [3] Data Flow / Reaching Definitions Traversal ---")
    df_traces = traverse_data_flow(graph, source_node_id=m_id, max_hops=10)
    if df_traces:
        for idx, trace in enumerate(df_traces[:2], 1):
            print(f"Data Flow Trace #{idx}:")
            for step in trace[:5]:
                print(f"    ~> [{step['label']} #{step['node_id']}] L{step.get('line_number')}: {step.get('code')}")
            if len(trace) > 5:
                print(f"    ~> ... ({len(trace) - 5} more steps)")
    else:
        print("  (No direct data flow edges from entry node; check parameter or assignment nodes)")

    print("\n" + "=" * 60)
    print("[+] All traversals completed successfully on real CPG!")


if __name__ == "__main__":
    main()
