"""GDAC Tool Registry: JSON Schema definitions, LangChain StructuredTools, and execution dispatcher."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import networkx as nx

try:
    from langchain_core.tools import StructuredTool, tool
except ImportError:
    StructuredTool = None  # type: ignore
    tool = None  # type: ignore

from gdac.tools.schemas import (
    TraverseASTInput,
    TraverseCFGInput,
    TraverseDataFlowInput,
    TraceVariableToSinkInput,
    CondenseMethodSliceInput,
    VerifyPatchInput,
    SemanticSearchInput,
    CalculateExpressionInput,
)
from gdac.tools.cpg_tools import (
    resolve_target_node,
    resolve_method_node,
    resolve_method_source_file,
    materialize_source_window,
)
from gdac.tools.math_tools import calculate_expression
from gdac.tools.patch_tools import verify_patch, _extract_implicated_variables
from gdac.tools.search_tools import semantic_code_search
from gdac.services.cpg_parser import (
    find_nodes,
    get_node_label_summary,
    trace_variable_to_sink,
    traverse_ast,
    traverse_cfg,
    traverse_data_flow,
)
from gdac.services.cpg_condenser import condense_method_slice
from gdac.services.cpg_rag import find_semantic_entry_points


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
                    "target": {
                        "type": "string",
                        "description": "Universal anchor: exact integer Node ID (e.g. '100000000001' from semantic entry points), qualified name ('ClassName.method_name'), or method name ('method_name').",
                    },
                    "method_name": {
                        "type": "string",
                        "description": "Optional legacy alias for target method name.",
                    },
                    "node_id": {
                        "type": "integer",
                        "description": "Optional specific AST node ID to start traversal from.",
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
                    "target": {
                        "type": "string",
                        "description": "Universal anchor: exact integer Node ID (e.g. '100000000001'), qualified name ('ClassName.method_name'), or method name.",
                    },
                    "method_name": {
                        "type": "string",
                        "description": "Optional legacy alias for target method name.",
                    },
                    "start_node_id": {
                        "type": "integer",
                        "description": "Optional specific CFG node ID to start from. If omitted, starts from target entry.",
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
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "traverse_data_flow",
            "description": (
                "Traces data propagation and reaching definitions (DDG) across variables and expressions. "
                "Use this to see how a variable value is calculated, modified, or overwritten, and identify "
                "the origin of a bad value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Universal anchor: exact integer Node ID (e.g. '100000000001'), qualified name ('ClassName.method_name'), or method name.",
                    },
                    "method_name": {
                        "type": "string",
                        "description": "Optional legacy alias for target method name.",
                    },
                    "var_name": {
                        "type": "string",
                        "description": "Optional name of the variable or parameter to trace (e.g. 'result' or 'total').",
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
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trace_variable_to_sink",
            "description": (
                "Macro-Tool: Algorithmic multi-hop data flow traversal. Uses NetworkX graph algorithms to automatically "
                "trace a variable or parameter from its origin through all intermediate mutations/calculations to its terminal sinks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Universal anchor: exact integer Node ID (e.g. '100000000001'), qualified name ('ClassName.method_name'), or method name.",
                    },
                    "var_name": {
                        "type": "string",
                        "description": "Optional name of the variable or parameter to trace (e.g. 'result', 'total', or 'data').",
                    },
                    "start_node_id": {
                        "type": "integer",
                        "description": "Optional specific starting node ID. If omitted, automatically resolves variable origin.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "condense_method_slice",
            "description": (
                "Program Slicer & Condenser: Computes the backward and forward slice of `target_vars` inside a method "
                "using REACHING_DEF/DDG and CFG edges, folding non-causal blocks into single-line "
                "`# [condensed block: N statements]` markers within `token_budget`, and returns the method's "
                "data-flow summary (reads, writes, calls)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Universal anchor: exact integer Node ID (e.g. '107374182442'), qualified name ('ClassName.method_name'), or method name.",
                    },
                    "target_vars": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of variable names to slice on. If omitted, automatically slices on all variables written by the method.",
                    },
                    "method_id": {
                        "type": "integer",
                        "description": "Optional alias for target method node ID.",
                    },
                    "token_budget": {
                        "type": "integer",
                        "description": "Maximum token budget for the condensed slice (default: 200).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_patch",
            "description": (
                "Blind-Locate, Surgical-Patch (BLSP) Verifier: Extracts the surrounding 15 lines from disk at "
                "`(target, line_number)`, applies a surgical `old_string -> new_string` diff, executes `pytest` "
                "via subprocess, and returns `status='SOLVED'` (returncode 0) or parses the new `AssertionError` "
                "traceback and implicated variables to route back to `condense_method_slice`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Universal anchor: exact integer Node ID (e.g. '100000000001'), qualified name ('ClassName.method_name'), or method name.",
                    },
                    "line_number": {
                        "type": "integer",
                        "description": "Exact 1-indexed statement line number (e.g. 84 or 34) from condense_method_slice.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement statement or line that fixes the bug.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Optional existing statement to replace. If omitted, replaces the line at line_number directly.",
                    },
                    "test_path": {
                        "type": "string",
                        "description": "Optional test target path.",
                    },
                },
                "required": ["target", "line_number", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_expression",
            "description": (
                "Safely evaluates an arithmetic or Python math expression (e.g. '100.0 * (1 - 20.0 / 100.0)' or 'round(100.0 * 0.8, 2)' or '2000 + 500 + 450'). "
                "Use this to test and verify your calculation formula against the expected test assertion before calling verify_patch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A single-line arithmetic or Python math expression.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


# ==============================================================================
# 2. Stateful Memory & Loop Interception
# ==============================================================================

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
        target_str = str(arguments.get("target") or arguments.get("method_name") or "").strip().lower()
        var_name = str(
            arguments.get("var_name")
            or arguments.get("target_vars")
            or arguments.get("new_string")
            or ""
        ).strip().lower()
        node_id_arg = (
            arguments.get("node_id")
            or arguments.get("start_node_id")
            or arguments.get("source_node_id")
            or arguments.get("line_number")
        )

        canonical_sig = (
            tool_name,
            target_str,
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
        self.visited_signatures.clear()
        self.visited_nodes.clear()
        self.history.clear()

    def allow_reslice(self, resolved_node_id: Optional[int] = None, target_str: str = "") -> None:
        target_low = target_str.strip().lower()
        self.visited_signatures = {
            s for s in self.visited_signatures
            if not (s[0] == "condense_method_slice" and (not target_low or s[1] == target_low))
        }
        if resolved_node_id is not None:
            self.visited_nodes.discard(("condense_method_slice", resolved_node_id))


GLOBAL_TOOL_MEMORY = ToolMemoryBuffer()


def reset_tool_memory() -> None:
    """Resets global tool execution memory."""
    GLOBAL_TOOL_MEMORY.clear()


# ==============================================================================
# 3. Tool Dispatcher
# ==============================================================================

def dispatch_tool(
    graph: nx.MultiDiGraph,
    tool_name: str,
    arguments: Dict[str, Any],
    memory: Optional[ToolMemoryBuffer] = None,
) -> Dict[str, Any]:
    """Executes a requested tool call against the loaded NetworkX CPG graph."""
    if tool_name == "calculate_expression":
        expr_val = str(arguments.get("expression") or arguments.get("expr") or "").strip()
        return calculate_expression(expr_val)

    if tool_name == "semantic_code_search":
        q = str(arguments.get("query") or "")
        k = int(arguments.get("top_k", 3))
        return semantic_code_search(graph, q, top_k=k)

    target_raw = (
        arguments.get("target")
        or arguments.get("method_name")
        or arguments.get("node_id")
        or arguments.get("method_id")
    )
    target_display = str(target_raw) if target_raw is not None else ""
    method_id = resolve_target_node(graph, target_raw)

    if method_id is None:
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
            if target_display == "*" or target_display.lower() in str(m["name"]).lower()
        ]
        if matching and len(matching) == 1:
            method_id = matching[0]["node_id"]
        elif matching:
            return {
                "info": f"Found {len(matching)} methods related to '{target_display}':",
                "available_methods": [m["name"] for m in matching[:15]],
                "hint": "Please call the tool specifying one of the method names or node IDs listed above.",
            }
        else:
            return {
                "error": f"Method or class '{target_display}' was not found in the Code Property Graph.",
                "available_methods": [m["name"] for m in all_methods[:15]],
                "suggestion": "Select one of the available methods listed above.",
            }

    if tool_name == "condense_method_slice":
        tv_existing = arguments.get("target_vars") or arguments.get("var_name")
        if not tv_existing and method_id is not None and method_id in graph:
            df_sum = graph.nodes[method_id].get("data_flow_summary") or {}
            auto_vars = list(df_sum.get("writes") or []) + list(df_sum.get("reads") or [])[:2]
            arguments = dict(arguments)
            arguments["target_vars"] = [v for v in auto_vars if v not in ("self", "this")]

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
            resolved_desc = get_node_label_summary(graph, root_id) if root_id in graph else None
            out_dict: Dict[str, Any] = {
                "tool": "traverse_ast",
                "target": target_display,
                "resolved_target": resolved_desc,
                "root_node_id": root_id,
                "total_nodes_found": len(nodes),
                "ast_nodes": compact_nodes,
            }
            if root_id in graph and "data_flow_summary" in graph.nodes[root_id]:
                out_dict["data_flow_summary"] = graph.nodes[root_id]["data_flow_summary"]
            return out_dict

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

            resolved_desc = get_node_label_summary(graph, start_id) if start_id in graph else None
            return {
                "tool": "traverse_cfg",
                "target": target_display,
                "resolved_target": resolved_desc,
                "start_node_id": start_id,
                "paths_count": len(paths),
                "execution_paths": formatted_paths,
            }

        elif tool_name == "traverse_data_flow":
            var_name = arguments.get("var_name")
            source_id = arguments.get("source_node_id")

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

            resolved_desc = get_node_label_summary(graph, source_id) if source_id in graph else None
            return {
                "tool": "traverse_data_flow",
                "target": target_display,
                "resolved_target": resolved_desc,
                "source_node_id": source_id,
                "tracked_variable": var_name,
                "traces_found": len(traces),
                "data_flow_traces": formatted_traces,
            }

        elif tool_name == "trace_variable_to_sink":
            var_name = arguments.get("var_name")
            start_id = arguments.get("start_node_id") or method_id
            result = trace_variable_to_sink(
                graph,
                start_node_id=start_id,
                method_name=target_display if not method_id else None,
                var_name=var_name,
            )
            resolved_desc = get_node_label_summary(graph, method_id) if method_id in graph else None
            return {
                "tool": "trace_variable_to_sink",
                "target": target_display,
                "resolved_target": resolved_desc,
                "variable": result.get("variable"),
                "origin": result.get("origin"),
                "terminal_sink": result.get("terminal_sink"),
                "total_paths_found": result.get("total_paths_found"),
                "summary": result.get("summary"),
                "propagation_path": result.get("detailed_path"),
            }

        elif tool_name == "condense_method_slice":
            target_vars_arg = arguments.get("target_vars") or arguments.get("var_name") or ""
            if not target_vars_arg and method_id is not None and method_id in graph:
                df_sum = graph.nodes[method_id].get("data_flow_summary") or {}
                auto_vars = list(df_sum.get("writes") or []) + list(df_sum.get("reads") or [])[:2]
                target_vars_arg = [v for v in auto_vars if v not in ("self", "this")]
            token_budget = int(arguments.get("token_budget", 200))
            slice_res = condense_method_slice(
                graph,
                method_id=method_id if method_id is not None else target_display,
                target_vars=target_vars_arg,
                token_budget=token_budget,
            )
            return {
                "tool": "condense_method_slice",
                "target": target_display,
                "method_id": slice_res["method_id"],
                "target_vars": slice_res["target_vars"],
                "slice_lines": slice_res["slice_text"].splitlines(),
                "slice_text": slice_res["slice_text"],
                "method_summary": slice_res["method_summary"],
                "interprocedural_links": slice_res.get("interprocedural_links", []),
                "token_stats": slice_res["token_stats"],
            }

        elif tool_name == "verify_patch":
            line_number = int(arguments.get("line_number", 1))
            old_string = str(arguments.get("old_string", ""))
            new_string = str(arguments.get("new_string", ""))
            test_path = arguments.get("test_path")
            keep_patch = bool(arguments.get("keep_patch", False))
            return verify_patch(
                graph,
                target=method_id if method_id is not None else target_display,
                line_number=line_number,
                old_string=old_string,
                new_string=new_string,
                test_path=test_path,
                keep_patch=keep_patch,
            )

        elif tool_name == "calculate_expression":
            expr_val = str(arguments.get("expression") or arguments.get("expr") or "").strip()
            return calculate_expression(expr_val)

        else:
            return {"error": f"Unknown tool: '{tool_name}'"}

    except Exception as exc:
        return {"error": f"Execution failed for tool '{tool_name}': {str(exc)}"}


# ==============================================================================
# 4. LangChain Tool Factory
# ==============================================================================

def create_cpg_langchain_tools(
    graph: nx.MultiDiGraph,
    memory: Optional[ToolMemoryBuffer] = None,
) -> List[Any]:
    """Instantiates and returns the 8 LangChain StructuredTools bound to the given CPG graph."""
    if StructuredTool is None:
        raise ImportError(
            "LangChain is not installed. Run: pip install langchain langchain-core"
        )

    def _traverse_ast(
        target: str,
        node_id: Optional[int] = None,
        filter_labels: Optional[List[str]] = None,
        max_depth: int = 4,
    ) -> str:
        res = dispatch_tool(
            graph,
            "traverse_ast",
            {
                "target": target,
                "node_id": node_id,
                "filter_labels": filter_labels,
                "max_depth": max_depth,
            },
            memory=memory,
        )
        return json.dumps(res, indent=2)

    def _traverse_cfg(
        target: str,
        start_node_id: Optional[int] = None,
        target_node_id: Optional[int] = None,
        max_paths: int = 3,
    ) -> str:
        res = dispatch_tool(
            graph,
            "traverse_cfg",
            {
                "target": target,
                "start_node_id": start_node_id,
                "target_node_id": target_node_id,
                "max_paths": max_paths,
            },
            memory=memory,
        )
        return json.dumps(res, indent=2)

    def _traverse_data_flow(
        target: str,
        var_name: Optional[str] = None,
        source_node_id: Optional[int] = None,
        max_hops: int = 15,
    ) -> str:
        res = dispatch_tool(
            graph,
            "traverse_data_flow",
            {
                "target": target,
                "var_name": var_name,
                "source_node_id": source_node_id,
                "max_hops": max_hops,
            },
            memory=memory,
        )
        return json.dumps(res, indent=2)

    def _trace_variable_to_sink(
        target: str,
        var_name: Optional[str] = None,
        start_node_id: Optional[int] = None,
    ) -> str:
        res = dispatch_tool(
            graph,
            "trace_variable_to_sink",
            {
                "target": target,
                "var_name": var_name,
                "start_node_id": start_node_id,
            },
            memory=memory,
        )
        return json.dumps(res, indent=2)

    def _condense_method_slice(
        target: Optional[str] = None,
        target_vars: Optional[str | List[str]] = None,
        method_id: Optional[int | str] = None,
        token_budget: int = 200,
    ) -> str:
        effective_target = target or (str(method_id) if method_id is not None else "")
        res = dispatch_tool(
            graph,
            "condense_method_slice",
            {
                "target": effective_target,
                "target_vars": target_vars,
                "token_budget": token_budget,
            },
            memory=memory,
        )
        return json.dumps(res, indent=2)

    def _verify_patch(
        target: str,
        line_number: int,
        new_string: str,
        old_string: Optional[str] = None,
        test_path: Optional[str] = None,
    ) -> str:
        res = dispatch_tool(
            graph,
            "verify_patch",
            {
                "target": target,
                "line_number": line_number,
                "old_string": old_string or "",
                "new_string": new_string,
                "test_path": test_path,
            },
            memory=memory,
        )
        return json.dumps(res, indent=2)

    def _semantic_code_search(query: str, top_k: int = 3) -> str:
        res = semantic_code_search(graph, query, top_k=top_k)
        return json.dumps(res, indent=2)

    def _calculate_expression(expression: str) -> str:
        res = dispatch_tool(
            graph,
            "calculate_expression",
            {"expression": expression},
            memory=memory,
        )
        return json.dumps(res, indent=2)

    tool_ast = StructuredTool.from_function(
        func=_traverse_ast,
        name="traverse_ast",
        description=(
            "Traverses the Abstract Syntax Tree (AST) of a target method or code block. "
            "Use this to inspect syntactic hierarchy, parameter declarations, "
            "statements, and child function calls."
        ),
        args_schema=TraverseASTInput,
    )

    tool_cfg = StructuredTool.from_function(
        func=_traverse_cfg,
        name="traverse_cfg",
        description=(
            "Traverses the Control Flow Graph (CFG) of a method to trace actual runtime execution paths, "
            "branching conditions (if/else), loops, and return points."
        ),
        args_schema=TraverseCFGInput,
    )

    tool_df = StructuredTool.from_function(
        func=_traverse_data_flow,
        name="traverse_data_flow",
        description=(
            "Traces data propagation and reaching definitions (DDG) across variables and expressions. "
            "Use this to see how a variable value is calculated, modified, or overwritten."
        ),
        args_schema=TraverseDataFlowInput,
    )

    tool_sink = StructuredTool.from_function(
        func=_trace_variable_to_sink,
        name="trace_variable_to_sink",
        description=(
            "Macro-Tool: Algorithmic multi-hop data flow traversal. Uses NetworkX graph algorithms to automatically "
            "trace a variable or parameter from its origin through all intermediate mutations/calculations to its terminal sinks."
        ),
        args_schema=TraceVariableToSinkInput,
    )

    tool_slice = StructuredTool.from_function(
        func=_condense_method_slice,
        name="condense_method_slice",
        description=(
            "Program Slicer & Condenser: Computes the backward and forward slice of `target_vars` inside a method "
            "using REACHING_DEF/DDG and CFG edges, folding non-causal blocks into single-line "
            "`# [condensed block: N statements]` markers within `token_budget`, and returns the method's "
            "data-flow summary (reads, writes, calls)."
        ),
        args_schema=CondenseMethodSliceInput,
    )

    tool_patch = StructuredTool.from_function(
        func=_verify_patch,
        name="verify_patch",
        description=(
            "Blind-Locate, Surgical-Patch (BLSP) Verifier: Extracts the surrounding 15 lines from disk at "
            "`(target, line_number)`, applies a surgical `old_string -> new_string` diff, executes `pytest` "
            "via subprocess, and returns `status='SOLVED'` (returncode 0) or parses the new `AssertionError` "
            "traceback and implicated variables to route back to `condense_method_slice`."
        ),
        args_schema=VerifyPatchInput,
    )

    tool_rag = StructuredTool.from_function(
        func=_semantic_code_search,
        name="semantic_code_search",
        description=(
            "Performs semantic similarity search against all repository methods in the CPG. "
            "Use this dynamically mid-investigation to find related helper classes, utility functions, or callee methods."
        ),
        args_schema=SemanticSearchInput,
    )

    tool_calc = StructuredTool.from_function(
        func=_calculate_expression,
        name="calculate_expression",
        description=(
            "Safely evaluates an arithmetic or Python math expression (e.g. '100.0 * (1 - 20.0 / 100.0)' or 'round(100.0 * 0.8, 2)' or '2000 + 500 + 450'). "
            "Use this to test and verify your calculation formula against the expected test assertion before calling verify_patch."
        ),
        args_schema=CalculateExpressionInput,
    )

    return [tool_ast, tool_cfg, tool_df, tool_sink, tool_slice, tool_patch, tool_rag, tool_calc]
