#!/usr/bin/env python3
"""LangChain Structured Tools for Code Property Graph (CPG) Static Analysis.

Defines Pydantic schemas and StructuredTools for:
  1. traverse_ast: AST syntactic structure and declarations
  2. traverse_cfg: CFG execution branching and paths
  3. traverse_data_flow: DDG data propagation across variables
  4. trace_variable_to_sink: NetworkX multi-hop data flow macro-tool
  5. semantic_code_search: Dynamic Hybrid GraphRAG entry point search
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import networkx as nx
from pydantic import BaseModel, Field

try:
    from langchain_core.tools import StructuredTool, tool
except ImportError:
    StructuredTool = None  # type: ignore
    tool = None  # type: ignore

from agent_tools import dispatch_tool, ToolMemoryBuffer
from cpg_rag import find_semantic_entry_points


# ==============================================================================
# 1. Pydantic Input Schemas
# ==============================================================================

class TraverseASTInput(BaseModel):
    """Input schema for AST syntactic tree traversal."""
    target: str = Field(
        ...,
        description="Universal anchor: exact integer Node ID (e.g. '100000000001' from semantic entry points), qualified name ('ClassName.method_name'), or method name ('method_name').",
    )
    node_id: Optional[int] = Field(
        None,
        description="Optional specific AST node ID to start traversal from.",
    )
    filter_labels: Optional[List[str]] = Field(
        None,
        description="Optional list of AST node labels to filter (e.g. ['CALL', 'IDENTIFIER', 'PARAM', 'RETURN', 'LITERAL']).",
    )
    max_depth: int = Field(
        4,
        description="Maximum tree depth to explore (default: 4).",
    )


class TraverseCFGInput(BaseModel):
    """Input schema for Control Flow Graph (CFG) runtime path traversal."""
    target: str = Field(
        ...,
        description="Universal anchor: exact integer Node ID (e.g. '100000000001'), qualified name ('ClassName.method_name'), or method name.",
    )
    start_node_id: Optional[int] = Field(
        None,
        description="Optional specific CFG node ID to start from. If omitted, starts from target entry.",
    )
    target_node_id: Optional[int] = Field(
        None,
        description="Optional destination node ID to find execution paths leading specifically to that node.",
    )
    max_paths: int = Field(
        3,
        description="Maximum number of execution paths to return (default: 3).",
    )


class TraverseDataFlowInput(BaseModel):
    """Input schema for Data Flow Graph (DDG) reaching definition traversal."""
    target: str = Field(
        ...,
        description="Universal anchor: exact integer Node ID (e.g. '100000000001'), qualified name ('ClassName.method_name'), or method name.",
    )
    var_name: Optional[str] = Field(
        None,
        description="Optional name of the variable or parameter to trace (e.g. 'result' or 'total').",
    )
    source_node_id: Optional[int] = Field(
        None,
        description="Optional specific source node ID. If omitted, automatically selects the parameter/definition matching var_name or method entry.",
    )
    max_hops: int = Field(
        15,
        description="Maximum number of data flow steps to follow (default: 15).",
    )


class TraceVariableToSinkInput(BaseModel):
    """Input schema for Macro-Tool multi-hop data flow traversal to terminal sinks."""
    target: str = Field(
        ...,
        description="Universal anchor: exact integer Node ID (e.g. '100000000001'), qualified name ('ClassName.method_name'), or method name.",
    )
    var_name: Optional[str] = Field(
        None,
        description="Optional name of the variable or parameter to trace (e.g. 'result', 'total', or 'data').",
    )
    start_node_id: Optional[int] = Field(
        None,
        description="Optional specific starting node ID. If omitted, automatically resolves variable origin.",
    )


class SemanticSearchInput(BaseModel):
    """Input schema for dynamic Hybrid GraphRAG search."""
    query: str = Field(
        ...,
        description="Search query describing the bug symptom, method responsibility, or functionality to locate in the CPG.",
    )
    top_k: int = Field(
        3,
        description="Number of top method matches to return (default: 3).",
    )


class CondenseMethodSliceInput(BaseModel):
    """Input schema for program slicing and non-causal block folding."""
    target: Optional[str] = Field(
        None,
        description="Universal anchor: exact integer Node ID (e.g. '107374182442'), qualified name ('ClassName.method_name'), or method name.",
    )
    target_vars: Optional[str | List[str]] = Field(
        None,
        description="Variable name, list of variable names, or comma-separated list of variable names to slice on. If omitted, automatically slices on all variables written by the method.",
    )
    method_id: Optional[int | str] = Field(
        None,
        description="Optional alias for target method node ID.",
    )
    token_budget: int = Field(
        200,
        description="Maximum token budget for the condensed slice (default: 200).",
    )


class VerifyPatchInput(BaseModel):
    """Input schema for Blind-Locate, Surgical-Patch (BLSP) runtime verification."""
    target: str = Field(
        ...,
        description="Universal anchor: exact integer Node ID (e.g. '100000000001'), qualified name ('ClassName.method_name'), or method name.",
    )
    line_number: int = Field(
        ...,
        description="Target 1-indexed line number identified during condense_method_slice inspection.",
    )
    old_string: str = Field(
        ...,
        description="Exact existing statement or line to replace within the 15-line materialized source window.",
    )
    new_string: str = Field(
        ...,
        description="Replacement statement or line that fixes the bug.",
    )
    test_path: Optional[str] = Field(
        None,
        description="Optional pytest target path (default: automatically resolves the corresponding test file under stackflow/tests/).",
    )


# ==============================================================================
# 2. LangChain Tool Builder
# ==============================================================================

def create_cpg_langchain_tools(
    graph: nx.MultiDiGraph,
    memory: Optional[ToolMemoryBuffer] = None,
) -> List[Any]:
    """Instantiates and returns the 7 LangChain StructuredTools bound to the given CPG graph."""
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
        old_string: str,
        new_string: str,
        test_path: Optional[str] = None,
    ) -> str:
        res = dispatch_tool(
            graph,
            "verify_patch",
            {
                "target": target,
                "line_number": line_number,
                "old_string": old_string,
                "new_string": new_string,
                "test_path": test_path,
            },
            memory=memory,
        )
        return json.dumps(res, indent=2)

    def _semantic_code_search(query: str, top_k: int = 3) -> str:
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
        return json.dumps({"query": query, "top_matches": formatted}, indent=2)

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

    return [tool_ast, tool_cfg, tool_df, tool_sink, tool_slice, tool_patch, tool_rag]
