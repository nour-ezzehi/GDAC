#!/usr/bin/env python3
"""Tool Binding: JSON Schema definitions and dispatcher for CPG graph traversals.

Formats the three CPG traversals (AST, CFG, Data Flow) from cpg_bridge as standard
JSON Schema tools for HuggingFace / OpenAI tool-calling models, with an execution
dispatcher to invoke them on a loaded NetworkX MultiDiGraph.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
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
from cpg_condenser import condense_method_slice


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
                "Use this to see how a parameter or variable value is calculated, modified, or overwritten, "
                "and whether it flows into an incorrect mathematical formula or return statement."
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
                "trace a variable or parameter from its origin through all intermediate mutations/calculations to its terminal sinks "
                "(return statements, assignments, or consumers). Returns a clean, human-readable summary of the entire propagation path."
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
                "using REACHING_DEF/DDG and CFG edges, folding non-causal statement blocks into single-line "
                "`# [condensed block: N statements]` markers within `token_budget`, and returns the method's "
                "data-flow summary (reads, writes, calls)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Universal anchor: exact integer Node ID, qualified name ('ClassName.method_name'), or method name.",
                    },
                    "target_vars": {
                        "type": "string",
                        "description": "Variable name or comma-separated list of variable names to slice on (e.g. 'discount_percent' or 'item_total,summary').",
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
                "Blind-Locate, Surgical-Patch (BLSP) Verifier: Materializes the 15-line source window around "
                "(target, line_number), applies a surgical old_string -> new_string replacement on disk, "
                "executes pytest via subprocess, and returns status='SOLVED' (returncode 0) or parses the "
                "new AssertionError traceback and implicated variables for re-slicing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Universal anchor: exact integer Node ID, qualified name ('ClassName.method_name'), or method name.",
                    },
                    "line_number": {
                        "type": "integer",
                        "description": "Target 1-indexed line number identified during condense_method_slice inspection.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact existing line or statement block to replace within the 15-line materialized window.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement line or statement block that fixes the bug.",
                    },
                    "test_path": {
                        "type": "string",
                        "description": "Optional pytest path (default: automatically resolves the corresponding test file under stackflow/tests/).",
                    },
                },
                "required": ["target", "line_number", "old_string", "new_string"],
            },
        },
    },
]


# ==============================================================================
# 2. Tool Execution Dispatcher & BLSP Runtime Verifier
# ==============================================================================

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


_SUPPORTED_SOURCE_EXTENSIONS = (
    ".py", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp",
    ".java", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs",
    ".cs", ".rb", ".php", ".kt", ".swift", ".scala",
)


def resolve_method_source_file(
    graph: nx.MultiDiGraph,
    method_id: int,
    repo_root: str | Path = "stackflow",
) -> Optional[Path]:
    """Resolves a CPG METHOD node to its physical source file on disk across any language.

    Checks CPG node attributes (`filename`, `file`, `filepath`, `full_name`) and ancestor
    AST nodes first, then falls back to scanning repository source files across all
    supported programming languages by matching method name, enclosing type/class name,
    and definition line number.
    """
    base_dir = Path(repo_root)
    if not base_dir.is_absolute():
        project_root = Path(__file__).resolve().parent
        base_dir = project_root / base_dir

    if method_id not in graph:
        return None

    def _try_resolve_relpath(raw_path: str) -> Optional[Path]:
        if not raw_path or raw_path == "empty" or raw_path.startswith("<"):
            return None
        clean_p = raw_path.split(":")[0].strip()
        if not clean_p or not any(clean_p.endswith(ext) for ext in _SUPPORTED_SOURCE_EXTENSIONS):
            return None
        for root_cand in (base_dir, base_dir.parent, Path(__file__).resolve().parent):
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

    # Language-agnostic function/method declaration pattern (Python, Rust, Go, JS/TS, C/C++, Java, C#, Kotlin, Ruby, PHP)
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


def _extract_implicated_variables(
    traceback_text: str,
    graph: Optional[nx.MultiDiGraph] = None,
    method_id: Optional[int] = None,
) -> List[str]:
    """Extracts newly implicated variables from a test runner failure / assertion output in any language."""
    candidates: List[str] = []
    seen: Set[str] = set()

    # Check method reads/writes from condensed CPG summary if available
    method_vars: Set[str] = set()
    if graph is not None and method_id is not None and method_id in graph:
        summary = graph.nodes[method_id].get("data_flow_summary", {})
        for v in summary.get("writes", []) + summary.get("reads", []):
            method_vars.add(str(v))

    # Extract identifiers from assertion/failure lines across Python, C/C++, Java, Go, Rust, JS/TS
    important_lines = [
        line for line in traceback_text.splitlines()
        if line.strip().startswith(("E ", "> ", "assert ", "ASSERT_", "EXPECT_", "--- FAIL"))
        or any(tok in line for tok in ("AssertionError", "AssertionFailed", "panicked at", "expected", "actual", "Check failed"))
    ]
    search_blob = "\n".join(important_lines) if important_lines else traceback_text

    # Match dotted attributes (`item.price`) and C/C++ pointer dereferences (`acct->balance`)
    var_Regex = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*(?:(?:\.|->)[a-zA-Z_][a-zA-Z0-9_]*)?)\b")
    stop_words = {
        "assert", "AssertionError", "def", "fn", "func", "function", "return", "in", "is", "not", "and", "or",
        "True", "False", "true", "false", "None", "null", "nullptr", "nil", "self", "this", "pytest", "approx",
        "test", "where", "float", "double", "int", "long", "void", "char", "bool", "str", "string", "round",
        "len", "get", "Exception", "Error", "FAILED", "FAIL", "E",
    }

    # First prioritize variables known in the method's data_flow_summary
    for mv in sorted(method_vars):
        base_mv = re.split(r"\.|->", mv)[-1]
        if mv in traceback_text or base_mv in traceback_text:
            if mv not in seen and mv not in ("self", "this"):
                seen.add(mv)
                candidates.append(mv)

    for m in var_Regex.finditer(search_blob):
        tok = m.group(1)
        if tok in stop_words or tok.startswith("test_") or tok.startswith("Test") or tok.endswith("Test"):
            continue
        if "." in tok or "->" in tok:
            lhs, rhs = re.split(r"\.|->", tok, maxsplit=1)
            if lhs in stop_words or rhs in ("py", "c", "cpp", "java", "rs", "go", "js", "ts", "approx", "get"):
                continue
        if tok not in seen:
            seen.add(tok)
            candidates.append(tok)

    # Also ensure all method writes are included if fallback needed
    if graph is not None and method_id is not None and method_id in graph:
        summary = graph.nodes[method_id].get("data_flow_summary", {})
        for w in summary.get("writes", []):
            if w not in seen and w not in ("self", "this"):
                seen.add(w)
                candidates.append(w)

    return candidates[:6]


def verify_patch(
    graph: nx.MultiDiGraph,
    target: Any,
    line_number: int,
    old_string: str,
    new_string: str,
    test_path: Optional[str] = None,
    repo_root: str | Path = "stackflow",
    keep_patch: bool = False,
) -> Dict[str, Any]:
    """Executes the Blind-Locate, Surgical-Patch (BLSP) verification sequence.

    1. Context Materialization: Resolves `(target, line_number)` to its physical file
       and extracts ONLY the surrounding 15 lines (+/- 7 lines) from disk.
    2. Surgical Diff: Replaces `old_string -> new_string` within the localized window
       (handling indentation and Joern CPG parenthesis normalization cleanly).
    3. Subprocess Execution: Runs `pytest` via `subprocess.run()`.
    4. Feedback Routing:
       - If `returncode == 0`, returns `status="SOLVED"`.
       - If `returncode != 0`, rolls back the failed edit, parses the new
         `AssertionError` / traceback, extracts `implicated_variables`, and returns
         `status="FAILED"` to route the agent back to `condense_method_slice`.
    """
    method_id = resolve_target_node(graph, target)
    if method_id is None:
        return {
            "tool": "verify_patch",
            "status": "ERROR",
            "error": f"Could not resolve target method '{target}' in CPG.",
        }

    file_path = resolve_method_source_file(graph, method_id, repo_root=repo_root)
    if file_path is None or not file_path.is_file():
        return {
            "tool": "verify_patch",
            "status": "ERROR",
            "error": f"Could not resolve physical source file for method '{target}' (#{method_id}).",
        }

    # Step 1: Context Materialization (15-line window around line_number)
    window_info = materialize_source_window(file_path, line_number, window_size=15)
    original_content = file_path.read_text(encoding="utf-8")
    all_lines = original_content.splitlines()

    old_clean = re.sub(r"^L\d+\s*:\s*", "", (old_string or "").strip())
    new_clean = re.sub(r"^L\d+\s*:\s*", "", (new_string or "").strip())
    if not new_clean:
        return {
            "tool": "verify_patch",
            "status": "MATERIALIZED",
            "target": str(target),
            "method_id": method_id,
            "file_path": str(file_path),
            "line_number": line_number,
            "materialized_window": window_info["numbered_window"],
            "hint": "Provide line_number and new_string to replace the target line in the 15-line materialized window.",
        }

    # When old_string is omitted, default to the exact line at line_number on disk
    if not old_clean and 1 <= line_number <= len(all_lines):
        old_clean = all_lines[line_number - 1].strip()
        old_string = old_clean

    if old_clean == new_clean:
        return {
            "tool": "verify_patch",
            "status": "FAILED",
            "returncode": -1,
            "target": str(target),
            "method_id": method_id,
            "file_path": str(file_path),
            "line_number": line_number,
            "materialized_window": window_info["numbered_window"],
            "error": (
                f"Invalid no-op patch: `old_string` and `new_string` are identical ({old_clean!r}). "
                f"Inspect `materialized_window` (L{window_info['start_line']}-L{window_info['end_line']}), "
                f"select the exact buggy statement line number inside the method body, and provide a modified `new_string`."
            ),
            "implicated_variables": _extract_implicated_variables(old_clean, graph, method_id),
        }

    # Step 2: Surgical Diff within the 15-line materialized window
    w_start = window_info["start_line"]
    w_end = window_info["end_line"]
    patched_content: Optional[str] = None
    applied_line: Optional[int] = None

    # 2a. Exact multi-line or single-line match within the window slice
    window_text = "\n".join(all_lines[w_start - 1 : w_end])
    if old_string in window_text:
        replaced_window = window_text.replace(old_string, new_string, 1)
        new_all_lines = (
            all_lines[: w_start - 1]
            + replaced_window.splitlines()
            + all_lines[w_end:]
        )
        patched_content = "\n".join(new_all_lines) + ("\n" if original_content.endswith("\n") else "")
        applied_line = line_number
    else:
        # 2b. Line-by-line match inside the 15-line window (preserving indentation)
        # Prioritizes line_number first, then scans the 15-line window
        candidate_indices = [line_number - 1] + [
            i for i in range(w_start - 1, w_end) if i != line_number - 1
        ]
        norm_old = _normalize_expr_for_match(old_clean)
        for idx in candidate_indices:
            if 0 <= idx < len(all_lines):
                disk_line = all_lines[idx]
                disk_stripped = disk_line.strip()
                if not disk_stripped or disk_stripped.startswith("#"):
                    continue
                if disk_stripped == old_clean or _normalize_expr_for_match(disk_stripped) == norm_old:
                    indent = disk_line[: len(disk_line) - len(disk_line.lstrip())]
                    replacement_lines = [
                        (indent + nl.lstrip()) if nl.strip() else ""
                        for nl in new_clean.splitlines()
                    ]
                    new_all_lines = (
                        all_lines[:idx] + replacement_lines + all_lines[idx + 1 :]
                    )
                    patched_content = "\n".join(new_all_lines) + (
                        "\n" if original_content.endswith("\n") else ""
                    )
                    applied_line = idx + 1
                    break

    if patched_content is None:
        return {
            "tool": "verify_patch",
            "status": "FAILED",
            "returncode": -1,
            "target": str(target),
            "method_id": method_id,
            "file_path": str(file_path),
            "line_number": line_number,
            "materialized_window": window_info["numbered_window"],
            "error": (
                f"Surgical replacement failed: `old_string` ({old_clean!r}) did not match any line "
                f"in the 15-line window (L{w_start}-L{w_end}). Inspect `materialized_window` and retry."
            ),
            "implicated_variables": _extract_implicated_variables(old_clean, graph, method_id),
        }

    # Resolve test target and language-appropriate test runner command
    project_root = Path(__file__).resolve().parent
    ext = file_path.suffix.lower()
    resolved_test: Any = test_path or ""
    cmd: List[str] = []

    if ext == ".py":
        if test_path:
            resolved_test_path = Path(test_path)
            if not resolved_test_path.is_absolute():
                resolved_test_path = project_root / resolved_test_path
        else:
            module_test = file_path.parent.parent / "tests" / f"test_{file_path.stem}.py"
            if module_test.is_file():
                resolved_test_path = module_test
            else:
                resolved_test_path = file_path.parent.parent / "tests"
        resolved_test = str(resolved_test_path)
        cmd = [sys.executable, "-m", "pytest", resolved_test, "-v", "--tb=short"]
    elif ext == ".go":
        resolved_test = test_path or "./..."
        cmd = ["go", "test", "-v", str(resolved_test)]
    elif ext == ".rs":
        resolved_test = test_path or "cargo test"
        cmd = ["cargo", "test"] + ([str(test_path)] if test_path else [])
    elif ext in (".js", ".jsx", ".ts", ".tsx"):
        resolved_test = test_path or "npm test"
        cmd = ["npm", "test"] + ([str(test_path)] if test_path else [])
    elif ext == ".java":
        resolved_test = test_path or "mvn test"
        if (project_root / "gradlew").is_file() or (project_root / "build.gradle").is_file():
            cmd = ["gradle", "test"]
        else:
            cmd = ["mvn", "test"]
    else:
        # C / C++ / generic build systems (CTest / Make)
        resolved_test = test_path or "ctest"
        if (project_root / "CMakeLists.txt").is_file():
            cmd = ["ctest", "--output-on-failure"]
        else:
            cmd = ["make", "test"]

    # Step 3: Subprocess Execution with atomic rollback guarantee
    returncode = 1
    stdout_text = ""
    stderr_text = ""
    try:
        file_path.write_text(patched_content, encoding="utf-8")
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        returncode = proc.returncode
        stdout_text = proc.stdout
        stderr_text = proc.stderr
    finally:
        if returncode != 0 or not keep_patch:
            file_path.write_text(original_content, encoding="utf-8")

    combined_output = (stdout_text + "\n" + stderr_text).strip()

    # Step 4: Feedback Routing
    if returncode == 0:
        return {
            "tool": "verify_patch",
            "status": "SOLVED",
            "returncode": 0,
            "target": str(target),
            "method_id": method_id,
            "file_path": str(file_path),
            "applied_line": applied_line,
            "old_string": old_clean,
            "new_string": new_clean,
            "materialized_window": window_info["numbered_window"],
            "test_target": str(resolved_test),
            "pytest_output": combined_output,
        }

    # Extract concise assertion / failure block across test runners
    failure_lines = []
    capture = False
    for line in combined_output.splitlines():
        if (
            line.startswith(("___", "E   ", "--- FAIL"))
            or any(k in line for k in ("AssertionError", "FAILED", "FAIL:", "panicked at", "ASSERT_", "EXPECT_"))
        ):
            capture = True
        if capture:
            failure_lines.append(line)
    parsed_traceback = "\n".join(failure_lines[-25:]) if failure_lines else combined_output[-1000:]
    implicated_vars = _extract_implicated_variables(parsed_traceback, graph, method_id)

    return {
        "tool": "verify_patch",
        "status": "FAILED",
        "returncode": returncode,
        "target": str(target),
        "method_id": method_id,
        "file_path": str(file_path),
        "applied_line": applied_line,
        "old_string": old_clean,
        "new_string": new_clean,
        "materialized_window": window_info["numbered_window"],
        "test_target": str(resolved_test),
        "traceback": parsed_traceback,
        "implicated_variables": implicated_vars,
        "feedback_directive": (
            f"Patch verification FAILED (returncode={returncode}). "
            f"Loop back and call `condense_method_slice` on `{target}` targeting "
            f"implicated variables {implicated_vars} to inspect the causal chain before retrying `verify_patch`."
        ),
    }


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
        """Clears memory for a fresh debugging session."""
        self.visited_signatures.clear()
        self.visited_nodes.clear()
        self.history.clear()

    def allow_reslice(self, resolved_node_id: Optional[int] = None, target_str: str = "") -> None:
        """Clears prior condense_method_slice entries so BLSP retry loops can re-slice the target method."""
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


def dispatch_tool(
    graph: nx.MultiDiGraph,
    tool_name: str,
    arguments: Dict[str, Any],
    memory: Optional[ToolMemoryBuffer] = None,
) -> Dict[str, Any]:
    """Executes a requested tool call against the loaded NetworkX CPG graph."""
    target_arg = (
        arguments.get("target")
        or arguments.get("method_name")
        or arguments.get("method_id")
        or arguments.get("node_id")
        or arguments.get("start_node_id")
        or arguments.get("source_node_id")
    )
    method_id = resolve_target_node(graph, target_arg) if target_arg is not None else None
    target_display = str(target_arg or "")

    # Handle wildcards, class names, or missing methods gracefully by listing methods
    if target_display in ("*", "all", "list", "") or (target_display and method_id is None):
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

    # If condense_method_slice is called without target_vars, default to the method's writes + reads
    if tool_name == "condense_method_slice":
        tv_existing = arguments.get("target_vars") or arguments.get("var_name")
        if not tv_existing and method_id is not None and method_id in graph:
            df_sum = graph.nodes[method_id].get("data_flow_summary") or {}
            auto_vars = list(df_sum.get("writes") or []) + list(df_sum.get("reads") or [])[:2]
            arguments = dict(arguments)
            arguments["target_vars"] = [v for v in auto_vars if v not in ("self", "this")]

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

        else:
            return {"error": f"Unknown tool: '{tool_name}'"}

    except Exception as exc:
        return {"error": f"Execution failed for tool '{tool_name}': {str(exc)}"}
