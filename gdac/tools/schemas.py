"""Pydantic Input Schemas for GDAC Tools."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


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
        description="Exact 1-indexed statement line number (e.g. 84 or 34) from condense_method_slice.",
    )
    new_string: str = Field(
        ...,
        description="Replacement statement or line that fixes the bug.",
    )
    old_string: Optional[str] = Field(
        None,
        description="Optional existing statement to replace. If omitted, replaces the line at line_number directly.",
    )
    test_path: Optional[str] = Field(
        None,
        description="Optional test target path.",
    )


class CalculateExpressionInput(BaseModel):
    """Input schema for evaluating mathematical or arithmetic expressions."""
    expression: str = Field(
        ...,
        description="A single-line arithmetic or Python math expression (e.g. '100.0 * (1 - 20.0 / 100.0)' or 'round(100.0 * 0.8, 2)' or '2000 + 500 + 450').",
    )
