#!/usr/bin/env python3
"""CPG Python Bridge: NetworkX Loader and Graph Traversals for Joern CPG14.

Loads Joern-exported CPG14 JSON files into a NetworkX MultiDiGraph and provides
three distinct traversal functions:
  1. AST Traversal: Syntactic structure exploration (AST hierarchy).
  2. CFG Traversal: Control flow & execution path tracing.
  3. Data Flow Traversal: Reaching definitions & variable taint propagation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

import networkx as nx


# ==============================================================================
# 1. Graph Loading & Construction
# ==============================================================================

def _normalize_node(raw_node: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """Normalizes raw Joern node into (node_id, attributes)."""
    node_id = int(raw_node.get("id", raw_node.get("_id", 0)))
    attrs: Dict[str, Any] = {
        "id": node_id,
        "label": raw_node.get("label", raw_node.get("_label", "UNKNOWN")),
    }

    # Extract properties (Joern may nest under 'properties' or provide flat)
    properties = raw_node.get("properties", {})
    if isinstance(properties, dict):
        for k, v in properties.items():
            attrs[k.lower()] = v
    for k, v in raw_node.items():
        if k not in ("id", "_id", "label", "_label", "properties"):
            attrs[k.lower()] = v

    # Normalize common code property graph fields
    if "name" not in attrs and "name" in raw_node:
        attrs["name"] = raw_node["name"]
    if "code" not in attrs and "code" in raw_node:
        attrs["code"] = raw_node["code"]
    if "line_number" not in attrs:
        attrs["line_number"] = attrs.get("linenumber") or attrs.get("line_number")

    return node_id, attrs


def _normalize_edge(raw_edge: Dict[str, Any]) -> Tuple[int, int, Dict[str, Any]]:
    """Normalizes raw Joern edge into (src_id, dst_id, attributes)."""
    src_id = int(
        raw_edge.get("out", raw_edge.get("source", raw_edge.get("src", 0)))
    )
    dst_id = int(
        raw_edge.get("in", raw_edge.get("target", raw_edge.get("dst", 0)))
    )

    edge_label = raw_edge.get("label", raw_edge.get("_label", "UNKNOWN")).upper()
    edge_attrs: Dict[str, Any] = {"label": edge_label}

    for k, v in raw_edge.items():
        if k not in ("out", "in", "source", "target", "src", "dst", "label", "_label"):
            edge_attrs[k.lower()] = v

    return src_id, dst_id, edge_attrs


def load_cpg_json(path_or_dir: str | Path) -> nx.MultiDiGraph:
    """Loads Joern CPG14 JSON export (file or directory of JSONs) into a NetworkX MultiDiGraph.

    Args:
        path_or_dir: Path to a single CPG JSON file or a directory containing
          CPG JSON files (e.g. `cpg_json/`).

    Returns:
        nx.MultiDiGraph: MultiDiGraph where nodes contain CPG properties and
        directed edges represent AST, CFG, REACHING_DEF, etc.
    """
    path = Path(path_or_dir)
    if not path.exists():
        raise FileNotFoundError(f"CPG path '{path}' does not exist.")

    json_files = [path] if path.is_file() else sorted(path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found at '{path}'.")

    graph = nx.MultiDiGraph()

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle various Joern export shapes:
        # Shape A: {"nodes": [...], "edges": [...]}
        # Shape B: list of items where each has label and out/in or properties
        if isinstance(data, dict):
            raw_nodes = data.get("nodes", [])
            raw_edges = data.get("edges", [])
        elif isinstance(data, list):
            raw_nodes = [item for item in data if "out" not in item and "source" not in item]
            raw_edges = [item for item in data if "out" in item or "source" in item]
        else:
            continue

        for n in raw_nodes:
            node_id, attrs = _normalize_node(n)
            graph.add_node(node_id, **attrs)

        for e in raw_edges:
            src, dst, edge_attrs = _normalize_edge(e)
            graph.add_edge(src, dst, **edge_attrs)

    return graph


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
        if name and data.get("name") != name:
            continue
        if code_contains and code_contains not in str(data.get("code", "")):
            continue
        matches.append(node_id)
    return matches


def get_node_label_summary(graph: nx.MultiDiGraph, node_id: int) -> str:
    """Returns a concise human-readable description of a CPG node."""
    data = graph.nodes.get(node_id, {})
    label = data.get("label", "NODE")
    name = data.get("name", "")
    code = data.get("code", "")
    line = data.get("line_number", "")
    line_str = f"L{line}: " if line else ""

    content = name or code or ""
    if len(content) > 40:
        content = content[:37] + "..."
    return f"[{label} #{node_id}] {line_str}{content}".strip()


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

    Explores the syntactic tree structure under `root_node_id` by traversing
    edges with label == 'AST'. Useful for inspecting function bodies,
    nested expressions, parameters, and AST node types.

    Args:
        graph: The CPG MultiDiGraph.
        root_node_id: Starting node (e.g. a METHOD node).
        filter_labels: Optional list of node labels to filter (e.g. ['CALL', 'IDENTIFIER']).
        max_depth: Maximum tree depth to traverse.

    Returns:
        List of dicts representing traversed AST nodes with depth and properties.
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

        # Find outgoing AST edges
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

    Follows directed 'CFG' edges to trace program execution flow, conditional
    branches, and reachability.

    Args:
        graph: The CPG MultiDiGraph.
        start_node_id: Starting CFG node (e.g. method entry point or statement).
        target_node_id: Optional destination node. If provided, finds all simple
          execution paths from start to target. If None, returns all reachable
          paths up to `max_depth`.
        max_paths: Maximum number of execution paths to return.
        max_depth: Maximum path length to prevent cycles in loops.

    Returns:
        List of paths, where each path is an ordered list of node IDs.
    """
    if start_node_id not in graph:
        raise ValueError(f"Start node {start_node_id} not found in graph.")

    # Build CFG-only view
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

    # Explore all forward execution paths from start_node_id
    paths = []

    def explore(curr_path: List[int]) -> None:
        if len(paths) >= max_paths:
            return
        curr = curr_path[-1]
        successors = list(cfg_graph.successors(curr))

        # Filter out nodes already in path to break infinite cycles/loops
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
    """Traversal 3: Data Flow & Reaching Definitions (Taint) Traversal.

    Traces data propagation across REACHING_DEF, DDG, and ARGUMENT edges.
    Tracks how a value (e.g., parameter or variable assignment) flows
    through the code to downstream uses or sinks.

    Args:
        graph: The CPG MultiDiGraph.
        source_node_id: Starting node (e.g. PARAMETER or ASSIGNMENT node).
        var_name: Optional variable name filter (e.g. 'discount_percent').
        sink_node_id: Optional sink node to target.
        max_hops: Maximum traversal depth.

    Returns:
        List of data flow traces, where each trace is a list of node metadata dicts.
    """
    if source_node_id not in graph:
        raise ValueError(f"Source node {source_node_id} not found in graph.")

    # Data flow edge labels in Joern CPG
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

        # Check outgoing data flow edges
        next_nodes = []
        for _, dst, edge_data in graph.out_edges(curr_id, data=True):
            label = edge_data.get("label", "").upper()
            if label in df_edge_labels:
                # If variable filter specified, check edge variable property if present
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

    # Convert node ID paths into detailed step dictionaries
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
# 4. Mock CPG Generator & Demo
# ==============================================================================

def generate_sample_cpg() -> nx.MultiDiGraph:
    """Generates a representative sample CPG representing stackflow code.

    Includes `apply_discount` and `generate_category_summary` with AST, CFG,
    and REACHING_DEF edges so traversals can be tested immediately.
    """
    g = nx.MultiDiGraph()

    # --- Method 1: apply_discount ---
    nodes_m1 = [
        (100, {"label": "METHOD", "name": "apply_discount", "code": "def apply_discount(self, sku, discount_percent):", "line_number": 78}),
        (101, {"label": "METHOD_PARAMETER_IN", "name": "sku", "code": "sku: str", "line_number": 78}),
        (102, {"label": "METHOD_PARAMETER_IN", "name": "discount_percent", "code": "discount_percent: float", "line_number": 78}),
        (103, {"label": "CONTROL_STRUCTURE", "code": "if not (0.0 <= discount_percent <= 100.0):", "line_number": 83}),
        (104, {"label": "CALL", "name": "<operator>.throw", "code": "raise ValueError(...)", "line_number": 84}),
        (105, {"label": "CALL", "name": "get_item", "code": "item = self.get_item(sku)", "line_number": 85}),
        (106, {"label": "CALL", "name": "<operator>.assignment", "code": "item.price = round(item.price * (discount_percent / 100.0), 2)", "line_number": 86}),
        (107, {"label": "RETURN", "code": "return item", "line_number": 95}),
    ]
    for nid, attrs in nodes_m1:
        g.add_node(nid, **attrs)

    # AST edges for apply_discount
    for child in [101, 102, 103, 105, 106, 107]:
        g.add_edge(100, child, label="AST")
    g.add_edge(103, 104, label="AST")

    # CFG edges for apply_discount
    g.add_edge(100, 103, label="CFG")
    g.add_edge(103, 104, label="CFG")  # Branch 1: raise
    g.add_edge(103, 105, label="CFG")  # Branch 2: normal execution
    g.add_edge(105, 106, label="CFG")
    g.add_edge(106, 107, label="CFG")

    # Data Flow (REACHING_DEF) for discount_percent
    g.add_edge(102, 103, label="REACHING_DEF", variable="discount_percent")
    g.add_edge(103, 106, label="REACHING_DEF", variable="discount_percent")

    # --- Method 2: generate_category_summary ---
    nodes_m2 = [
        (200, {"label": "METHOD", "name": "generate_category_summary", "code": "def generate_category_summary(self):", "line_number": 27}),
        (201, {"label": "CALL", "name": "<operator>.assignment", "code": "summary = {}", "line_number": 32}),
        (202, {"label": "CONTROL_STRUCTURE", "code": "for item in self.manager.list_all_items():", "line_number": 33}),
        (203, {"label": "CALL", "name": "<operator>.assignment", "code": "item_total = item.price * item.quantity", "line_number": 34}),
        (204, {"label": "CALL", "name": "<operator>.assignment", "code": "summary[item.category] = round(item_total, 2)", "line_number": 35}),
        (205, {"label": "RETURN", "code": "return summary", "line_number": 36}),
    ]
    for nid, attrs in nodes_m2:
        g.add_node(nid, **attrs)

    for child in [201, 202, 205]:
        g.add_edge(200, child, label="AST")
    g.add_edge(202, 203, label="AST")
    g.add_edge(202, 204, label="AST")

    g.add_edge(200, 201, label="CFG")
    g.add_edge(201, 202, label="CFG")
    g.add_edge(202, 203, label="CFG")
    g.add_edge(203, 204, label="CFG")
    g.add_edge(204, 202, label="CFG")  # Loop back
    g.add_edge(202, 205, label="CFG")  # Loop exit

    g.add_edge(201, 204, label="REACHING_DEF", variable="summary")
    g.add_edge(204, 205, label="REACHING_DEF", variable="summary")

    return g


def run_demo() -> None:
    """Runs a demonstration of all 3 graph traversals on sample CPG."""
    print("=" * 60)
    print("  CPG Bridge Demonstration (Sample CPG)")
    print("=" * 60)
    g = generate_sample_cpg()
    print(f"[+] Loaded Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges\n")

    # 1. AST Traversal
    print("--- [1] AST Traversal (Method: apply_discount) ---")
    ast_nodes = traverse_ast(g, root_node_id=100)
    for n in ast_nodes:
        indent = "  " * n["depth"]
        print(f"{indent}- [{n['label']}] (ID: {n['node_id']}) {n['code']}")

    # 2. CFG Traversal
    print("\n--- [2] CFG Traversal (Execution Paths from Method Entry) ---")
    cfg_paths = traverse_cfg(g, start_node_id=100, max_paths=5)
    for idx, path in enumerate(cfg_paths, 1):
        step_strs = [get_node_label_summary(g, nid) for nid in path]
        print(f"Path #{idx}:")
        for s in step_strs:
            print(f"    -> {s}")

    # 3. Data Flow Traversal
    print("\n--- [3] Data Flow Traversal (Tracing 'discount_percent') ---")
    df_traces = traverse_data_flow(g, source_node_id=102, var_name="discount_percent")
    for idx, trace in enumerate(df_traces, 1):
        print(f"Data Flow #{idx}:")
        for step in trace:
            print(f"    ~> [{step['label']} #{step['node_id']}] L{step.get('line_number')}: {step.get('code')}")
    print("\n" + "=" * 60)


# ==============================================================================
# 5. CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load Joern CPG14 JSON into NetworkX and run graph traversals."
    )
    parser.add_argument(
        "--cpg-path",
        type=str,
        default=None,
        help="Path to CPG JSON file or directory (e.g. cpg_json/)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Target method name to inspect (e.g. apply_discount)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run self-contained demo illustrating all 3 traversals",
    )

    args = parser.parse_args()

    if args.demo or (not args.cpg_path and not args.method):
        run_demo()
        return

    cpg_path = Path(args.cpg_path)
    if not cpg_path.exists():
        print(f"[-] Error: CPG path '{cpg_path}' not found.", file=sys.stderr)
        print("    Run `python joern_extraction.py` first, or use `--demo`.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading CPG from '{cpg_path}'...")
    graph = load_cpg_json(cpg_path)
    print(f"[+] Successfully loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

    if args.method:
        method_nodes = find_nodes(graph, label="METHOD", name=args.method)
        if not method_nodes:
            print(f"[-] No METHOD node found with name '{args.method}'.")
            return
        m_id = method_nodes[0]
        print(f"\n[+] Found method '{args.method}' (Node ID: {m_id})")

        print("\n--- AST Traversal ---")
        for item in traverse_ast(graph, root_node_id=m_id, max_depth=3):
            indent = "  " * item["depth"]
            print(f"{indent}- {get_node_label_summary(graph, item['node_id'])}")

        print("\n--- CFG Execution Paths ---")
        for p in traverse_cfg(graph, start_node_id=m_id, max_paths=3):
            steps = " -> ".join(f"#{nid}" for nid in p)
            print(f"  Path: {steps}")


if __name__ == "__main__":
    main()
