#!/usr/bin/env python3
"""Tool Binding: JSON Schema definitions and dispatcher for CPG graph traversals.

Formats the three CPG traversals (AST, CFG, Data Flow) from cpg_bridge as standard
JSON Schema tools for HuggingFace / OpenAI tool-calling models, with an execution
dispatcher to invoke them on a loaded NetworkX MultiDiGraph.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

from cpg_bridge import (
    find_nodes,
    get_node_label_summary,
    trace_variable_to_sink,
    traverse_ast,
    traverse_cfg,
    traverse_data_flow,
)


# ==============================================================================
# 1. JSON Schema Tool Definitions
# ==============================================================================

TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "traverse_ast",
            "description": (
                "Traverses the Abstract Syntax Tree (AST) of a target method or code block. "
                "Use this to understand the syntactic hierarchy, find declared parameters, "
                "expression statements, and child function calls without control flow."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "method_name": {
                        "type": "string",
                        "description": "Name or substring of the method to inspect (e.g. 'apply_discount' or 'generate_category_summary').",
                    },
                    "node_id": {
                        "type": "integer",
                        "description": "Optional specific AST node ID to start traversal from. If omitted, starts from method entry.",
                    },
                    "filter_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of AST node labels to filter (e.g. ['CALL', 'IDENTIFIER', 'PARAM', 'RETURN', 'LITERAL']).",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum tree depth to explore (default: 4).",
                    },
                },
                "required": ["method_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "traverse_cfg",
            "description": (
                "Traverses the Control Flow Graph (CFG) of a method to trace actual runtime execution paths, "
                "branching conditions (if/else), loops, and return points. Use this to determine which branch "
                "executes and how control reaches an error or return."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "method_name": {
                        "type": "string",
                        "description": "Name or substring of the method to inspect (e.g. 'apply_discount').",
                    },
                    "start_node_id": {
                        "type": "integer",
                        "description": "Optional specific CFG node ID to start from. If omitted, starts from method entry.",
                    },
                    "target_node_id": {
                        "type": "integer",
                        "description": "Optional destination node ID to find execution paths leading specifically to that node.",
                    },
                    "max_paths": {
                        "type": "integer",
                        "description": "Maximum number of execution paths to return (default: 3).",
                    },
                },
                "required": ["method_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "traverse_data_flow",
            "description": (
                "Traces data propagation and reaching definitions (DDG) across variables and expressions. "
                "Use this to see how a parameter or variable value is calculated, modified, or overwritten, "
                "and whether it flows into an incorrect mathematical formula or return statement."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "method_name": {
                        "type": "string",
                        "description": "Name or substring of the method containing the variable (e.g. 'apply_discount').",
                    },
                    "var_name": {
                        "type": "string",
                        "description": "Optional name of the variable or parameter to trace (e.g. 'discount_percent' or 'summary').",
                    },
                    "source_node_id": {
                        "type": "integer",
                        "description": "Optional specific source node ID. If omitted, automatically selects the parameter/definition matching var_name or method entry.",
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Maximum number of data flow steps to follow (default: 15).",
                    },
                },
                "required": ["method_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trace_variable_to_sink",
            "description": (
                "Macro-Tool: Algorithmic multi-hop data flow traversal. Uses NetworkX graph algorithms to automatically "
                "trace a variable or parameter from its origin through all intermediate mutations/calculations to its terminal sinks "
                "(return statements, assignments, or consumers). Returns a clean, human-readable summary of the entire propagation path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "method_name": {
                        "type": "string",
                        "description": "Name or substring of the method containing the variable (e.g. 'apply_discount' or 'generate_category_summary').",
                    },
                    "var_name": {
                        "type": "string",
                        "description": "Optional name of the variable or parameter to trace (e.g. 'discount_percent', 'item_total', or 'summary').",
                    },
                    "start_node_id": {
                        "type": "integer",
                        "description": "Optional specific starting node ID. If omitted, automatically resolves variable origin.",
                    },
                },
                "required": ["method_name"],
            },
        },
    },
]


# ==============================================================================
# 2. Tool Execution Dispatcher
# ==============================================================================

def resolve_method_node(graph: nx.MultiDiGraph, method_name: str) -> Optional[int]:
    """Finds the most specific METHOD node matching method_name, prioritizing real implementation bodies over compiler adapters."""
    nodes = find_nodes(graph, label="METHOD", code_contains=method_name)
    if not nodes:
        nodes = find_nodes(graph, label="METHOD", name=method_name)

    # Filter out compiler-generated adapter wrappers (<metaClassAdapter>)
    real_nodes = [
        n for n in nodes
        if "<metaClassAdapter>" not in str(graph.nodes[n].get("code", ""))
        and "<metaClassAdapter>" not in str(graph.nodes[n].get("name", ""))
    ]
    if real_nodes:
        return real_nodes[0]
    return nodes[0] if nodes else None


class ToolMemoryBuffer:
    """Deterministic stateful memory buffer preventing repetitive tool invocation loops."""

    def __init__(self) -> None:
        self.visited_signatures: Set[Tuple[str, ...]] = set()
        self.visited_nodes: Set[Tuple[str, int]] = set()
        self.history: List[Dict[str, Any]] = []

    def check_and_record(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        resolved_node_id: Optional[int] = None,
    ) -> Optional[str]:
        """Checks if a tool invocation or target node was already explored.

        Returns an interception error message if repeated, or None if new.
        """
        method_name = str(arguments.get("method_name", "")).strip().lower()
        var_name = str(arguments.get("var_name", "")).strip().lower()
        node_id_arg = arguments.get("node_id") or arguments.get("start_node_id") or arguments.get("source_node_id")

        canonical_sig = (
            tool_name,
            method_name,
            var_name,
            str(node_id_arg or ""),
            str(arguments.get("filter_labels") or ""),
        )

        if canonical_sig in self.visited_signatures:
            return "System Error: You have already inspected this node. Review your previous thoughts and explore a different execution path."

        if resolved_node_id is not None:
            node_key = (tool_name, resolved_node_id)
            if node_key in self.visited_nodes and not var_name:
                return "System Error: You have already inspected this node. Review your previous thoughts and explore a different execution path."
            self.visited_nodes.add(node_key)

        self.visited_signatures.add(canonical_sig)
        self.history.append({"tool": tool_name, "args": arguments, "node_id": resolved_node_id})
        return None

    def clear(self) -> None:
        """Clears memory for a fresh debugging session."""
        self.visited_signatures.clear()
        self.visited_nodes.clear()
        self.history.clear()


GLOBAL_TOOL_MEMORY = ToolMemoryBuffer()


def reset_tool_memory() -> None:
    """Resets global tool execution memory."""
    GLOBAL_TOOL_MEMORY.clear()


def dispatch_tool(
    graph: nx.MultiDiGraph,
    tool_name: str,
    arguments: Dict[str, Any],
    memory: Optional[ToolMemoryBuffer] = None,
) -> Dict[str, Any]:
    """Executes a requested tool call against the loaded NetworkX CPG graph."""
    method_name = arguments.get("method_name", "").strip()
    method_id = resolve_method_node(graph, method_name) if method_name else None

    # Handle wildcards, class names, or missing methods gracefully by listing methods
    if method_name in ("*", "all", "list", "") or (method_name and method_id is None):
        all_methods = []
        seen = set()
        for n in find_nodes(graph, label="METHOD"):
            m_code = graph.nodes[n].get("code", "")
            if m_code and not m_code.startswith("<") and not m_code.endswith("<metaClassAdapter>"):
                if m_code not in seen:
                    seen.add(m_code)
                    all_methods.append({
                        "node_id": n,
                        "name": m_code,
                        "line": graph.nodes[n].get("line_number"),
                    })

        matching = [
            m for m in all_methods
            if method_name == "*" or method_name.lower() in str(m["name"]).lower()
        ]
        if matching and len(matching) == 1:
            method_id = matching[0]["node_id"]
        elif matching:
            return {
                "info": f"Found {len(matching)} methods related to '{method_name}':",
                "available_methods": [m["name"] for m in matching[:15]],
                "hint": "Please call the tool specifying one of the method names listed above.",
            }
        else:
            return {
                "error": f"Method or class '{method_name}' was not found in the Code Property Graph.",
                "available_methods": [m["name"] for m in all_methods[:15]],
                "suggestion": "Select one of the available methods listed above.",
            }

    # Anti-looping interceptor: check deterministic memory buffer
    mem = memory if memory is not None else GLOBAL_TOOL_MEMORY
    loop_error = mem.check_and_record(tool_name, arguments, resolved_node_id=method_id)
    if loop_error:
        return {"error": loop_error}

    try:
        if tool_name == "traverse_ast":
            root_id = arguments.get("node_id") or method_id
            filter_labels = arguments.get("filter_labels")
            max_depth = int(arguments.get("max_depth", 4))

            nodes = traverse_ast(
                graph, root_node_id=root_id, filter_labels=filter_labels, max_depth=max_depth
            )
            compact_nodes = [
                {
                    "node_id": n["node_id"],
                    "depth": n["depth"],
                    "label": n["label"],
                    "line": n.get("line_number"),
                    "code": n.get("code") or n.get("name"),
                }
                for n in nodes[:25]
            ]
            return {
                "tool": "traverse_ast",
                "method": method_name,
                "root_node_id": root_id,
                "total_nodes_found": len(nodes),
                "ast_nodes": compact_nodes,
            }

        elif tool_name == "traverse_cfg":
            start_id = arguments.get("start_node_id") or method_id
            target_id = arguments.get("target_node_id")
            max_paths = int(arguments.get("max_paths", 3))

            paths = traverse_cfg(
                graph, start_node_id=start_id, target_node_id=target_id, max_paths=max_paths
            )
            formatted_paths = []
            for idx, p in enumerate(paths, 1):
                steps = [get_node_label_summary(graph, nid) for nid in p[:12]]
                if len(p) > 12:
                    steps.append(f"... ({len(p) - 12} more steps)")
                formatted_paths.append({"path_index": idx, "length": len(p), "steps": steps})

            return {
                "tool": "traverse_cfg",
                "method": method_name,
                "start_node_id": start_id,
                "paths_count": len(paths),
                "execution_paths": formatted_paths,
            }

        elif tool_name == "traverse_data_flow":
            var_name = arguments.get("var_name")
            source_id = arguments.get("source_node_id")

            # If no source_id provided, look for matching parameter or assignment
            if source_id is None and method_id is not None:
                if var_name:
                    var_nodes = [
                        n for n in find_nodes(graph, code_contains=var_name)
                        if n in nx.descendants(graph, method_id) or n == method_id
                    ]
                    param_nodes = [n for n in var_nodes if graph.nodes[n].get("label") == "PARAM"]
                    assign_nodes = [n for n in var_nodes if "assignment" in str(graph.nodes[n].get("label", ""))]
                    if param_nodes:
                        source_id = param_nodes[0]
                    elif assign_nodes:
                        source_id = assign_nodes[0]
                    elif var_nodes:
                        source_id = var_nodes[0]
                    else:
                        source_id = method_id
                else:
                    source_id = method_id

            max_hops = int(arguments.get("max_hops", 15))
            traces = traverse_data_flow(
                graph, source_node_id=source_id, var_name=var_name, max_hops=max_hops
            )

            formatted_traces = []
            for idx, trace in enumerate(traces[:3], 1):
                steps = [
                    f"[{s['label']} #{s['node_id']}] L{s.get('line_number')}: {s.get('code')}"
                    for s in trace[:10]
                ]
                formatted_traces.append({"trace_index": idx, "steps": steps})

            return {
                "tool": "traverse_data_flow",
                "method": method_name,
                "source_node_id": source_id,
                "tracked_variable": var_name,
                "traces_found": len(traces),
                "data_flow_traces": formatted_traces,
            }

        elif tool_name == "trace_variable_to_sink":
            var_name = arguments.get("var_name")
            start_id = arguments.get("start_node_id")
            result = trace_variable_to_sink(
                graph,
                start_node_id=start_id,
                method_name=method_name,
                var_name=var_name,
            )
            return {
                "tool": "trace_variable_to_sink",
                "method": method_name,
                "variable": result.get("variable"),
                "origin": result.get("origin"),
                "terminal_sink": result.get("terminal_sink"),
                "total_paths_found": result.get("total_paths_found"),
                "summary": result.get("summary"),
                "propagation_path": result.get("detailed_path"),
            }

        else:
            return {"error": f"Unknown tool: '{tool_name}'"}

    except Exception as exc:
        return {"error": f"Execution failed for tool '{tool_name}': {str(exc)}"}
