#!/usr/bin/env python3
"""Language-Agnostic CPG Condenser: Statement Contraction, Data-Flow Summaries, and Slicing.

Operates strictly on universal Joern CPG14 graph properties (`METHOD`, `PARAM`, `BLOCK`,
`CONTROL_STRUCTURE`, `RETURN`, `CALL`, `<operator>.*`, `AST`, `CFG`, `DDG`, `REACHING_DEF`, `CDG`)
so that C/C++, Java, JavaScript/TypeScript, Go, Python, Rust, C#, PHP, and Ruby CPGs are
condensed and sliced with the exact same graph-theoretic rules.

Core functions:
  1. `condense_cpg(G: nx.MultiDiGraph) -> nx.MultiDiGraph`:
     Collapses sub-expression AST nodes into statement-level CFG/DDG nodes and computes
     a fast Node Data-Flow Summary (`reads`, `writes`, `calls`) for every `METHOD` node.
  2. `condense_method_slice(G, method_id, target_vars, token_budget)`:
     Uses `REACHING_DEF`/`DDG` and `CFG`/`CDG` edges to retain only statements lying on
     the backward/forward slice of `target_vars`, folding non-causal blocks into
     single-line `# [condensed block: N statements]` markers within `token_budget`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx

from cpg_bridge import (
    find_nodes,
    get_node_label_summary,
    load_cpg_json,
    trace_variable_to_sink,
    traverse_ast,
    traverse_cfg,
    traverse_data_flow,
)


# ==============================================================================
# 1. Universal CPG14 Constants & Predicates (Language-Agnostic)
# ==============================================================================

# Universal compiler-generated temporary patterns across Joern frontends
# (e.g., `tmp0`, `_tmp_1`, `manager_tmp0`, `$t0`, `__newInstance`, `<iterator>0`)
_COMPILER_TEMP_RE = re.compile(r"^(?:[a-zA-Z0-9_]*_?tmp_?\d+|\$t\d+|__newInstance|<.*>)$")
_IDENT_TOKEN_RE = re.compile(r"\b[a-zA-Z_$][a-zA-Z0-9_$]*\b")

_STRUCTURAL_LABELS: Set[str] = {
    "METHOD",
    "PARAM",
    "METHOD_RETURN",
    "TYPE_DECL",
    "MEMBER",
}

_NON_EXECUTABLE_AST_LABELS: Set[str] = {
    "",
    "LOCAL",
    "TYPE_REF",
    "METHOD_REF",
    "MODIFIER",
    "NAMESPACE_BLOCK",
    "JUMP_TARGET",
    "COMMENT",
    "ANNOTATION",
    "BINDING",
    "IDENTIFIER",
    "FIELD_IDENTIFIER",
    "LITERAL",
}

_OPERATOR_PREFIX = "<operator>."

# Universal CPG14 assignment and mutation operators across all languages
_BINARY_ASSIGNMENT_OPERATORS: Set[str] = {
    "<operator>.assignment",
    "<operator>.assignmentPlus",
    "<operator>.assignmentMinus",
    "<operator>.assignmentMultiplication",
    "<operator>.assignmentDivision",
    "<operator>.assignmentModulo",
    "<operator>.assignmentAnd",
    "<operator>.assignmentOr",
    "<operator>.assignmentXor",
    "<operator>.assignmentShiftLeft",
    "<operator>.assignmentArithmeticShiftRight",
    "<operator>.assignmentLogicalShiftRight",
    "<operator>.assignmentExponentiation",
}

_COMPOUND_ASSIGNMENT_OPERATORS: Set[str] = _BINARY_ASSIGNMENT_OPERATORS - {
    "<operator>.assignment"
}

_UNARY_MUTATION_OPERATORS: Set[str] = {
    "<operator>.preIncrement",
    "<operator>.postIncrement",
    "<operator>.preDecrement",
    "<operator>.postDecrement",
}

_ALL_MUTATION_OPERATORS: Set[str] = (
    _BINARY_ASSIGNMENT_OPERATORS | _UNARY_MUTATION_OPERATORS
)

# Universal CPG14 member/index access operators (`.` / `->` / `[]`)
_ACCESS_OPERATORS: Set[str] = {
    "<operator>.fieldAccess",
    "<operator>.indirectFieldAccess",
    "<operator>.indexAccess",
    "<operator>.indirectIndexAccess",
    "<operator>.pointerShift",
}

# Universal CPG14 infix operator symbols for reconstructing truncated `.dot` expressions
_BINARY_OP_SYMBOLS: Dict[str, str] = {
    "<operator>.assignment": "=",
    "<operator>.assignmentPlus": "+=",
    "<operator>.assignmentMinus": "-=",
    "<operator>.assignmentMultiplication": "*=",
    "<operator>.assignmentDivision": "/=",
    "<operator>.addition": "+",
    "<operator>.subtraction": "-",
    "<operator>.multiplication": "*",
    "<operator>.division": "/",
    "<operator>.modulo": "%",
    "<operator>.equals": "==",
    "<operator>.notEquals": "!=",
    "<operator>.lessThan": "<",
    "<operator>.lessEqualsThan": "<=",
    "<operator>.greaterThan": ">",
    "<operator>.greaterEqualsThan": ">=",
    "<operator>.logicalAnd": "&&",
    "<operator>.logicalOr": "||",
    "<operator>.and": "&",
    "<operator>.or": "|",
}

# Universal CPG14 internal runtime/lowering helper calls to exclude from user-level call summaries
_INTERNAL_LOWERING_CALLS: Set[str] = {
    "__iter__",
    "__next__",
    "__enter__",
    "__exit__",
    "<init>",
    "<clinit>",
    "<fakeNew>",
    "<metaClassCallHandler>",
}


def _is_compiler_temp(name: Optional[str]) -> bool:
    """Returns True if `name` is a synthetic compiler temporary or builtin namespace."""
    if not name:
        return True
    clean = name.strip()
    if not clean or _COMPILER_TEMP_RE.match(clean):
        return True
    if clean.startswith(("__builtins__", "__builtin")) or "_tmp" in clean or ".__next__()" in clean:
        return True
    return False


def _estimate_tokens(text: str) -> int:
    """Estimates LLM token count using ~4 chars/token bounded by word/punctuation tokens."""
    if not text:
        return 0
    char_est = math.ceil(len(text) / 4.0)
    word_punct = len(re.findall(r"\w+|[^\w\s]", text))
    return max(1, min(char_est, word_punct))


def resolve_method_id(G: nx.MultiDiGraph, method_id_or_name: Union[int, str]) -> int:
    """Resolves a `METHOD` node ID from an integer ID or method/qualified name."""
    if isinstance(method_id_or_name, int):
        if method_id_or_name in G:
            return method_id_or_name
        raise ValueError(f"Method node ID {method_id_or_name} not found in graph.")

    target_str = str(method_id_or_name).strip().lstrip("#")
    if target_str.isdigit() and int(target_str) in G:
        return int(target_str)

    meth_part = target_str.split(".")[-1].strip()
    cls_part = target_str.split(".")[-2].strip().lower() if "." in target_str else ""

    candidates = find_nodes(G, label="METHOD", code_contains=meth_part)
    if not candidates:
        candidates = find_nodes(G, label="METHOD", name=meth_part)

    if cls_part and candidates:
        filtered = []
        for c in candidates:
            for p, _, ed in G.in_edges(c, data=True):
                if ed.get("label") == "AST":
                    pdata = G.nodes[p]
                    if cls_part in str(pdata.get("code", "")).lower():
                        filtered.append(c)
                        break
                    if pdata.get("code") == "<body>":
                        for pp, _, eed in G.in_edges(p, data=True):
                            if eed.get("label") == "AST" and cls_part in str(G.nodes[pp].get("code", "")).lower():
                                filtered.append(c)
                                break
        if filtered:
            candidates = filtered

    real_methods = [
        n for n in candidates
        if "<metaClassAdapter>" not in str(G.nodes[n].get("code", ""))
        and "<metaClassAdapter>" not in str(G.nodes[n].get("name", ""))
    ]
    chosen = real_methods[0] if real_methods else (candidates[0] if candidates else None)
    if chosen is None:
        raise ValueError(f"Method '{method_id_or_name}' not found in CPG.")
    return chosen


# ==============================================================================
# 2. Universal AST Subtree & Data-Flow Summary (`reads`, `writes`, `calls`)
# ==============================================================================

def _build_ast_children_map(G: nx.MultiDiGraph) -> Dict[int, List[int]]:
    """Builds a deduplicated adjacency list of AST children for fast single-pass traversals."""
    ast_children: Dict[int, List[int]] = {}
    seen_pairs: Set[Tuple[int, int]] = set()
    for u, v, d in G.edges(data=True):
        if d.get("label") == "AST" and (u, v) not in seen_pairs:
            seen_pairs.add((u, v))
            ast_children.setdefault(u, []).append(v)
    return ast_children


def _collect_ast_subtree(
    root_id: int,
    ast_children: Dict[int, List[int]],
    stop_nodes: Optional[Set[int]] = None,
) -> List[int]:
    """Collects all AST nodes in the subtree rooted at `root_id`, stopping at `stop_nodes`."""
    visited: List[int] = []
    seen: Set[int] = set()
    stack = [root_id]
    while stack:
        curr = stack.pop()
        if curr in seen:
            continue
        seen.add(curr)
        visited.append(curr)
        for child in reversed(ast_children.get(curr, [])):
            if stop_nodes and child in stop_nodes:
                continue
            if child not in seen:
                stack.append(child)
    return visited


def _discover_temp_aliases(
    G: nx.MultiDiGraph,
    node_set: Set[int],
    ast_children: Dict[int, List[int]],
) -> Dict[str, str]:
    """Finds compiler-lowered temporary aliases (`tmp0 = expr`) across any Joern frontend."""
    temp_alias_map: Dict[str, str] = {}
    for nid in sorted(node_set, key=lambda x: (G.nodes[x].get("line_number") or 0, x)):
        ndata = G.nodes.get(nid, {})
        lbl = str(ndata.get("label", ""))
        if lbl == "<operator>.assignment":
            ch = ast_children.get(nid, [])
            if len(ch) >= 2:
                lhs_code = str(G.nodes.get(ch[0], {}).get("code", "")).strip()
                rhs_code = str(G.nodes.get(ch[1], {}).get("code", "")).strip()
                if _COMPILER_TEMP_RE.match(lhs_code) and rhs_code:
                    # Resolve chained temp references (e.g. `tmp2 = tmp3.foo()`)
                    for k, v in temp_alias_map.items():
                        rhs_code = re.sub(rf"\b{re.escape(k)}\b", v, rhs_code)
                    if not _COMPILER_TEMP_RE.match(rhs_code) and "\n" not in rhs_code and not rhs_code.endswith("..."):
                        temp_alias_map[lhs_code] = rhs_code
    return temp_alias_map


def _substitute_temp_aliases(text: str, temp_alias_map: Dict[str, str]) -> str:
    """Replaces compiler-generated temporary identifiers with their underlying expressions."""
    if not text:
        return text
    out = " ".join(text.strip().split())
    # Collapse `tmp0 = {} tmp0` or `tmp0 = [] tmp0` lowered block expressions
    out = re.sub(r"\b_?tmp_?\d+\s*=\s*(\{\}|\[\])\s+_?tmp_?\d+\b", r"\1", out)
    # Collapse inline `tmp0 = expr` assignment prefixes inside lowered expressions
    out = re.sub(r"\b_?tmp_?\d+\s*=\s*[^\s]+\s+", "", out)
    for k, v in temp_alias_map.items():
        out = re.sub(rf"\b{re.escape(k)}\b", v, out)
    return out.strip()


def _extract_node_data_flow_summary(
    G: nx.MultiDiGraph,
    subtree_nodes: Iterable[int],
    ast_children: Dict[int, List[int]],
    include_params: bool = False,
) -> Dict[str, List[str]]:
    """Computes language-agnostic `reads`, `writes`, and `calls` over a set of CPG AST nodes."""
    reads: List[str] = []
    writes: List[str] = []
    calls: List[str] = []

    reads_set: Set[str] = set()
    writes_set: Set[str] = set()
    calls_set: Set[str] = set()

    node_set = set(subtree_nodes)
    temp_alias_map = _discover_temp_aliases(G, node_set, ast_children)

    def _add_read(val: str) -> None:
        v = _substitute_temp_aliases(val, temp_alias_map)
        if v and not _is_compiler_temp(v) and v not in reads_set:
            reads_set.add(v)
            reads.append(v)

    def _add_write(val: str) -> None:
        v = _substitute_temp_aliases(val, temp_alias_map)
        if v and not _is_compiler_temp(v) and v not in writes_set:
            writes_set.add(v)
            writes.append(v)

    def _add_call(val: str) -> None:
        v = _substitute_temp_aliases(val, temp_alias_map)
        if v and not _is_compiler_temp(v) and v not in _INTERNAL_LOWERING_CALLS and v not in calls_set:
            calls_set.add(v)
            calls.append(v)

    pure_lhs_ids: Set[int] = set()
    callee_ident_ids: Set[int] = set()

    # Pass 1: Inspect parameters, mutation operators (`writes`), and call nodes (`calls`)
    for nid in sorted(node_set, key=lambda x: (G.nodes[x].get("line_number") or 0, x)):
        ndata = G.nodes.get(nid, {})
        lbl = str(ndata.get("label", ""))
        code = str(ndata.get("code", "")).strip()

        if include_params and lbl == "PARAM":
            if code and code != "cls":
                _add_read(code)
            continue

        if lbl in _ALL_MUTATION_OPERATORS:
            ch = ast_children.get(nid, [])
            if ch:
                lhs_id = ch[0]
                lhs_data = G.nodes.get(lhs_id, {})
                lhs_lbl = str(lhs_data.get("label", ""))
                lhs_code = str(lhs_data.get("code", "")).strip()

                if lhs_lbl == "IDENTIFIER":
                    if lbl == "<operator>.assignment":
                        pure_lhs_ids.add(lhs_id)
                    else:
                        # Compound assignment (`+=`) or unary increment (`++`) both reads and writes LHS
                        _add_read(lhs_code)
                    if not _is_compiler_temp(lhs_code):
                        _add_write(lhs_code)
                elif lhs_lbl in _ACCESS_OPERATORS:
                    if lbl == "<operator>.assignment":
                        pure_lhs_ids.add(lhs_id)
                    else:
                        _add_read(lhs_code)
                    _add_write(lhs_code)
                    # Base object/pointer and index expressions of an access LHS are read
                    lhs_ch = ast_children.get(lhs_id, [])
                    if lhs_ch:
                        base_code = str(G.nodes.get(lhs_ch[0], {}).get("code", "")).strip()
                        if not _is_compiler_temp(base_code):
                            _add_read(base_code)

        elif (
            lbl not in _STRUCTURAL_LABELS
            and lbl not in _NON_EXECUTABLE_AST_LABELS
            and lbl not in ("BLOCK", "CONTROL_STRUCTURE", "RETURN", "UNKNOWN")
            and not lbl.startswith(_OPERATOR_PREFIX)
        ):
            # Universal CPG14 CALL node
            if lbl in _INTERNAL_LOWERING_CALLS or lbl.endswith("<metaClassAdapter>"):
                continue
            ch = ast_children.get(nid, [])
            call_name = lbl
            if ch:
                first_ch = G.nodes.get(ch[0], {})
                first_lbl = str(first_ch.get("label", ""))
                first_code = str(first_ch.get("code", "")).strip()
                if first_lbl in _ACCESS_OPERATORS:
                    if first_code:
                        call_name = first_code
                    for sub_c in ast_children.get(ch[0], []):
                        if G.nodes.get(sub_c, {}).get("label") == "FIELD_IDENTIFIER":
                            callee_ident_ids.add(sub_c)
                elif first_lbl == "IDENTIFIER":
                    callee_ident_ids.add(ch[0])
                    if first_code and not _is_compiler_temp(first_code):
                        call_name = first_code
            _add_call(call_name)

    # Pass 2: Collect variable and field/index `reads`
    for nid in sorted(node_set, key=lambda x: (G.nodes[x].get("line_number") or 0, x)):
        if nid in pure_lhs_ids or nid in callee_ident_ids:
            continue
        ndata = G.nodes.get(nid, {})
        lbl = str(ndata.get("label", ""))
        code = str(ndata.get("code", "")).strip()

        if lbl in _ACCESS_OPERATORS:
            resolved_code = _substitute_temp_aliases(code, temp_alias_map)
            if (
                resolved_code
                and resolved_code not in calls_set
                and not any(resolved_code.endswith(f".{ic}") for ic in _INTERNAL_LOWERING_CALLS)
            ):
                _add_read(resolved_code)
        elif lbl == "IDENTIFIER":
            if not _is_compiler_temp(code) and code not in calls_set:
                _add_read(code)

    return {
        "reads": reads,
        "writes": writes,
        "calls": calls,
    }


# ==============================================================================
# 3. Universal Statement Contraction (`condense_cpg`)
# ==============================================================================

def _reconstruct_statement_code(
    G: nx.MultiDiGraph,
    stmt_id: int,
    ast_children: Dict[int, List[int]],
    temp_alias_map: Dict[str, str],
) -> str:
    """Reconstructs clean, untruncated statement code from CPG AST nodes in a language-agnostic way."""
    s_data = G.nodes.get(stmt_id, {})
    s_lbl = str(s_data.get("label", ""))
    raw_code = str(s_data.get("code", "")).strip()

    # If Joern truncated the node's `code` with `...`, reconstruct from direct AST children
    if raw_code.endswith("..."):
        ch_list = ast_children.get(stmt_id, [])
        if s_lbl in _BINARY_OP_SYMBOLS and len(ch_list) >= 2:
            lhs_c = _reconstruct_statement_code(G, ch_list[0], ast_children, temp_alias_map)
            rhs_c = _reconstruct_statement_code(G, ch_list[1], ast_children, temp_alias_map)
            op_sym = _BINARY_OP_SYMBOLS[s_lbl]
            if lhs_c and rhs_c and not rhs_c.endswith("..."):
                raw_code = f"{lhs_c} {op_sym} {rhs_c}"
        elif s_lbl == "<operator>.logicalNot" and ch_list:
            inner = _reconstruct_statement_code(G, ch_list[0], ast_children, temp_alias_map)
            if inner and not inner.endswith("..."):
                raw_code = f"!({inner})"
        elif s_lbl == "BLOCK" and ch_list:
            # Find non-temp child expression inside lowered block
            non_temp_parts = []
            for c_id in ch_list:
                c_lbl = str(G.nodes.get(c_id, {}).get("label", ""))
                c_code = str(G.nodes.get(c_id, {}).get("code", "")).strip()
                if c_lbl == "<operator>.assignment" and "=" in c_code:
                    lhs_p = c_code.split("=", 1)[0].strip()
                    if _COMPILER_TEMP_RE.match(lhs_p):
                        continue
                reconstructed = _reconstruct_statement_code(G, c_id, ast_children, temp_alias_map)
                if reconstructed:
                    non_temp_parts.append(reconstructed)
            if non_temp_parts:
                raw_code = " ".join(non_temp_parts)
    elif s_lbl in _BINARY_OP_SYMBOLS and raw_code:
        # Even if not ending with `...`, check if child is a BLOCK that was lowered
        ch_list = ast_children.get(stmt_id, [])
        if len(ch_list) == 2 and str(G.nodes.get(ch_list[1], {}).get("label", "")) == "BLOCK":
            lhs_c = str(G.nodes.get(ch_list[0], {}).get("code", "")).strip()
            rhs_sub = [
                str(G.nodes.get(sub_c, {}).get("code", "")).strip()
                for sub_c in ast_children.get(ch_list[1], [])
                if str(G.nodes.get(sub_c, {}).get("label", "")) not in ("IDENTIFIER", "LOCAL")
            ]
            if lhs_c and rhs_sub:
                raw_code = f"{lhs_c} {_BINARY_OP_SYMBOLS[s_lbl]} {rhs_sub[-1]}"

    return _substitute_temp_aliases(raw_code or s_lbl, temp_alias_map)


def _extract_method_statements(
    G: nx.MultiDiGraph,
    method_id: int,
    ast_children: Dict[int, List[int]],
    temp_alias_map: Dict[str, str],
) -> Tuple[List[int], Dict[int, int], Dict[int, str], List[Tuple[int, int]]]:
    """Identifies statement-level AST nodes inside `method_id` using universal CPG14 structure,
    maps every sub-expression node (and pure compiler temp assignment) to its enclosing
    statement representative, and records structural `CDG` edges `(ctrl_id, body_stmt_id)`.
    """
    statement_ids: List[int] = []
    node_to_rep: Dict[int, int] = {method_id: method_id}
    stmt_clean_code: Dict[int, str] = {}
    cdg_edges: List[Tuple[int, int]] = []

    for ch in ast_children.get(method_id, []):
        clbl = str(G.nodes.get(ch, {}).get("label", ""))
        if clbl in ("PARAM", "METHOD_RETURN"):
            node_to_rep[ch] = ch

    def _is_temp_assignment(nid: int) -> bool:
        ndata = G.nodes.get(nid, {})
        if str(ndata.get("label", "")) != "<operator>.assignment":
            return False
        ch = ast_children.get(nid, [])
        if not ch:
            return False
        lhs_code = str(G.nodes.get(ch[0], {}).get("code", "")).strip()
        return bool(_COMPILER_TEMP_RE.match(lhs_code))

    def _walk_children_list(
        exec_children: List[int],
        enclosing_ctrl_id: Optional[int] = None,
        inherited_temp_subnodes: Optional[List[int]] = None,
    ) -> None:
        pending_temp_subnodes: List[int] = list(inherited_temp_subnodes or [])

        for ch in exec_children:
            cdata = G.nodes.get(ch, {})
            clbl = str(cdata.get("label", ""))

            # If this node is a pure compiler-temp assignment (`tmp0 = expr`), queue its subtree
            # so it contracts into its consumer statement rather than severing data-flow links.
            if _is_temp_assignment(ch):
                pending_temp_subnodes.extend(_collect_ast_subtree(ch, ast_children))
                continue

            # Case A: Lowered nested BLOCK
            if clbl == "BLOCK":
                sub_children = [
                    sc for sc in ast_children.get(ch, [])
                    if str(G.nodes.get(sc, {}).get("label", "")) not in _NON_EXECUTABLE_AST_LABELS
                ]
                non_temp_stmts = [sc for sc in sub_children if not _is_temp_assignment(sc)]
                temp_stmts = [sc for sc in sub_children if _is_temp_assignment(sc)]

                if len(non_temp_stmts) == 1 and temp_stmts:
                    main_stmt = non_temp_stmts[0]
                    main_lbl = str(G.nodes.get(main_stmt, {}).get("label", ""))
                    if main_lbl == "CONTROL_STRUCTURE":
                        prefix_nodes = list(pending_temp_subnodes) + [ch]
                        pending_temp_subnodes.clear()
                        for ts in temp_stmts:
                            prefix_nodes.extend(_collect_ast_subtree(ts, ast_children))
                        _register_control_structure(
                            main_stmt,
                            prefix_nodes=prefix_nodes,
                            enclosing_ctrl_id=enclosing_ctrl_id,
                        )
                    else:
                        statement_ids.append(main_stmt)
                        if enclosing_ctrl_id is not None:
                            cdg_edges.append((enclosing_ctrl_id, main_stmt))
                        stmt_clean_code[main_stmt] = _reconstruct_statement_code(
                            G, main_stmt, ast_children, temp_alias_map
                        )
                        for sub_nid in pending_temp_subnodes:
                            node_to_rep[sub_nid] = main_stmt
                        pending_temp_subnodes.clear()
                        for sub_nid in _collect_ast_subtree(ch, ast_children):
                            node_to_rep[sub_nid] = main_stmt
                else:
                    _walk_block(
                        ch,
                        enclosing_ctrl_id=enclosing_ctrl_id,
                        inherited_temp_subnodes=pending_temp_subnodes,
                    )
                    pending_temp_subnodes.clear()
                continue

            # Case B: CONTROL_STRUCTURE (IF, WHILE, FOR, SWITCH, TRY, etc.)
            if clbl == "CONTROL_STRUCTURE":
                prefix_nodes = list(pending_temp_subnodes) if pending_temp_subnodes else None
                pending_temp_subnodes.clear()
                _register_control_structure(
                    ch,
                    prefix_nodes=prefix_nodes,
                    enclosing_ctrl_id=enclosing_ctrl_id,
                )
                continue

            # Case C: Standard statement node
            statement_ids.append(ch)
            if enclosing_ctrl_id is not None:
                cdg_edges.append((enclosing_ctrl_id, ch))
            stmt_clean_code[ch] = _reconstruct_statement_code(
                G, ch, ast_children, temp_alias_map
            )
            for sub_nid in pending_temp_subnodes:
                node_to_rep[sub_nid] = ch
            pending_temp_subnodes.clear()
            for sub_nid in _collect_ast_subtree(ch, ast_children):
                node_to_rep[sub_nid] = ch

        # Attach any trailing temp nodes to the last statement in scope
        if pending_temp_subnodes and statement_ids:
            last_stmt = statement_ids[-1]
            for sub_nid in pending_temp_subnodes:
                node_to_rep[sub_nid] = last_stmt

    def _walk_block(
        block_id: int,
        enclosing_ctrl_id: Optional[int] = None,
        inherited_temp_subnodes: Optional[List[int]] = None,
    ) -> None:
        node_to_rep.setdefault(block_id, enclosing_ctrl_id if enclosing_ctrl_id is not None else method_id)
        children = ast_children.get(block_id, [])

        exec_children = []
        for ch in children:
            clbl = str(G.nodes.get(ch, {}).get("label", ""))
            if clbl in _NON_EXECUTABLE_AST_LABELS or clbl in _STRUCTURAL_LABELS:
                node_to_rep[ch] = method_id
                continue
            exec_children.append(ch)

        _walk_children_list(
            exec_children,
            enclosing_ctrl_id=enclosing_ctrl_id,
            inherited_temp_subnodes=inherited_temp_subnodes,
        )

    def _register_control_structure(
        ctrl_id: int,
        prefix_nodes: Optional[List[int]] = None,
        enclosing_ctrl_id: Optional[int] = None,
    ) -> None:
        ctrl_data = G.nodes.get(ctrl_id, {})
        ctrl_children = ast_children.get(ctrl_id, [])

        cond_node = (
            ctrl_children[0]
            if ctrl_children and str(G.nodes.get(ctrl_children[0], {}).get("label", "")) != "BLOCK"
            else None
        )
        body_blocks = [c for c in ctrl_children if c != cond_node]

        statement_ids.append(ctrl_id)
        node_to_rep[ctrl_id] = ctrl_id
        if enclosing_ctrl_id is not None:
            cdg_edges.append((enclosing_ctrl_id, ctrl_id))

        if prefix_nodes:
            for pn in prefix_nodes:
                node_to_rep[pn] = ctrl_id

        if cond_node is not None:
            for sub_nid in _collect_ast_subtree(cond_node, ast_children):
                node_to_rep[sub_nid] = ctrl_id
            cond_str = _reconstruct_statement_code(G, cond_node, ast_children, temp_alias_map)
            if cond_str == "iteratorNonEmptyOrException" and prefix_nodes:
                # Universal iterator loop header reconstruction
                iter_calls = [
                    _reconstruct_statement_code(G, pn, ast_children, temp_alias_map)
                    for pn in prefix_nodes
                    if str(G.nodes.get(pn, {}).get("label", ""))
                    not in ("BLOCK", "IDENTIFIER", "FIELD_IDENTIFIER", "<operator>.assignment")
                    and not str(G.nodes.get(pn, {}).get("label", "")).startswith(_OPERATOR_PREFIX)
                    and str(G.nodes.get(pn, {}).get("label", "")) not in _INTERNAL_LOWERING_CALLS
                ]
                iter_expr = iter_calls[0] if iter_calls else "iterable"
                stmt_clean_code[ctrl_id] = f"LOOP ({iter_expr})"
            else:
                ctrl_type = str(ctrl_data.get("code", "")).splitlines()[0].strip() or "CONTROL"
                stmt_clean_code[ctrl_id] = f"{ctrl_type} ({cond_str})"
        else:
            stmt_clean_code[ctrl_id] = _substitute_temp_aliases(
                str(ctrl_data.get("code", "") or "CONTROL_STRUCTURE"), temp_alias_map
            )

        for b_id in body_blocks:
            blbl = str(G.nodes.get(b_id, {}).get("label", ""))
            if blbl == "BLOCK":
                b_children = [
                    sc for sc in ast_children.get(b_id, [])
                    if str(G.nodes.get(sc, {}).get("label", "")) not in _NON_EXECUTABLE_AST_LABELS
                ]
                # Fold compiler-generated iterator fetch `item = tmp1.__next__()` into the loop header node
                if b_children and ".__next__()" in str(G.nodes.get(b_children[0], {}).get("code", "")):
                    next_assign = b_children[0]
                    for sub_nid in _collect_ast_subtree(next_assign, ast_children):
                        node_to_rep[sub_nid] = ctrl_id
                    assign_ch = ast_children.get(next_assign, [])
                    if assign_ch:
                        loop_var = str(G.nodes.get(assign_ch[0], {}).get("code", "")).strip()
                        if loop_var:
                            stmt_clean_code[ctrl_id] = f"LOOP ({loop_var} : {stmt_clean_code[ctrl_id][5:-1]})"
                    _walk_children_list(b_children[1:], enclosing_ctrl_id=ctrl_id)
                else:
                    _walk_block(b_id, enclosing_ctrl_id=ctrl_id)
            elif blbl == "CONTROL_STRUCTURE":
                _register_control_structure(b_id, enclosing_ctrl_id=ctrl_id)
            else:
                _walk_children_list([b_id], enclosing_ctrl_id=ctrl_id)

    for ch in ast_children.get(method_id, []):
        if str(G.nodes.get(ch, {}).get("label", "")) == "BLOCK":
            _walk_block(ch, enclosing_ctrl_id=None)

    return statement_ids, node_to_rep, stmt_clean_code, cdg_edges


def condense_cpg(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Language-agnostic CPG condenser: collapses sub-expression AST nodes into statement-level
    CFG/DDG nodes and computes a fast Node Data-Flow Summary (`reads`, `writes`, `calls`)
    for every `METHOD` node without mutating the input `G`.
    """
    ast_children = _build_ast_children_map(G)
    condensed = nx.MultiDiGraph()

    global_node_to_rep: Dict[int, int] = {}
    node_temp_alias_map: Dict[int, Dict[str, str]] = {}
    all_structural_cdg_edges: List[Tuple[int, int]] = []
    method_nodes = find_nodes(G, label="METHOD")

    for nid, ndata in G.nodes(data=True):
        lbl = str(ndata.get("label", ""))
        if lbl in ("TYPE_DECL", "MEMBER"):
            condensed.add_node(nid, **dict(ndata))
            global_node_to_rep[nid] = nid

    for m_id in method_nodes:
        m_data = dict(G.nodes[m_id])
        m_code = str(m_data.get("code", "")).strip()

        nested_boundaries: Set[int] = set()
        stack = list(ast_children.get(m_id, []))
        visited_ast: Set[int] = {m_id}
        while stack:
            curr = stack.pop()
            if curr in visited_ast:
                continue
            visited_ast.add(curr)
            clbl = str(G.nodes.get(curr, {}).get("label", ""))
            if clbl in ("METHOD", "TYPE_DECL"):
                nested_boundaries.add(curr)
                continue
            stack.extend(ast_children.get(curr, []))

        method_subtree = _collect_ast_subtree(m_id, ast_children, stop_nodes=nested_boundaries)
        temp_alias_map = _discover_temp_aliases(G, set(method_subtree), ast_children)
        method_summary = _extract_node_data_flow_summary(
            G, method_subtree, ast_children, include_params=True
        )

        # Write summary only to `m_data` for `condensed.nodes[m_id]` (do NOT mutate input `G`)
        m_data["data_flow_summary"] = method_summary
        m_data["reads"] = method_summary["reads"]
        m_data["writes"] = method_summary["writes"]
        m_data["calls"] = method_summary["calls"]

        is_compiler_wrapper = (
            "<metaClassAdapter>" in m_code
            or m_code in _INTERNAL_LOWERING_CALLS
        )

        condensed.add_node(m_id, **m_data)
        global_node_to_rep[m_id] = m_id
        node_temp_alias_map[m_id] = temp_alias_map

        for ch in ast_children.get(m_id, []):
            clbl = str(G.nodes.get(ch, {}).get("label", ""))
            if clbl in ("PARAM", "METHOD_RETURN"):
                condensed.add_node(ch, **dict(G.nodes[ch]))
                condensed.add_edge(m_id, ch, label="AST")
                global_node_to_rep[ch] = ch
                node_temp_alias_map[ch] = temp_alias_map
            elif clbl == "BLOCK":
                for bch in ast_children.get(ch, []):
                    if str(G.nodes.get(bch, {}).get("label", "")) == "LITERAL":
                        c_str = str(G.nodes.get(bch, {}).get("code", ""))
                        if '"""' in c_str or "'''" in c_str or "&quot;&quot;&quot;" in c_str:
                            condensed.add_node(bch, **dict(G.nodes[bch]))
                            condensed.add_edge(m_id, bch, label="AST")

        if is_compiler_wrapper:
            continue

        stmt_ids, node_to_rep, stmt_clean_code, cdg_edges = _extract_method_statements(
            G, m_id, ast_children, temp_alias_map
        )
        global_node_to_rep.update(node_to_rep)
        all_structural_cdg_edges.extend(cdg_edges)

        rep_counts: Dict[int, int] = {}
        for sub_n, rep_n in node_to_rep.items():
            rep_counts[rep_n] = rep_counts.get(rep_n, 0) + 1

        for s_id in stmt_ids:
            s_data = dict(G.nodes[s_id])
            s_subnodes = [sub_n for sub_n, r_n in node_to_rep.items() if r_n == s_id]
            s_summary = _extract_node_data_flow_summary(
                G, s_subnodes, ast_children, include_params=False
            )
            if s_id in stmt_clean_code and stmt_clean_code[s_id]:
                s_data["code"] = stmt_clean_code[s_id]
            s_data["is_condensed_statement"] = True
            s_data["collapsed_subnodes_count"] = rep_counts.get(s_id, 1)
            s_data["data_flow_summary"] = s_summary
            s_data["reads"] = s_summary["reads"]
            s_data["writes"] = s_summary["writes"]
            s_data["calls"] = s_summary["calls"]

            condensed.add_node(s_id, **s_data)
            condensed.add_edge(m_id, s_id, label="AST")
            node_temp_alias_map[s_id] = temp_alias_map

    for u, v, d in G.edges(data=True):
        if d.get("label") == "AST":
            if u in condensed and v in condensed and not condensed.has_edge(u, v):
                ulbl = str(condensed.nodes[u].get("label", ""))
                vlbl = str(condensed.nodes[v].get("label", ""))
                if ulbl == "TYPE_DECL" or vlbl in ("TYPE_DECL", "METHOD", "MEMBER"):
                    condensed.add_edge(u, v, label="AST")

    seen_edges: Set[Tuple[int, int, str, str]] = set()

    # Add structural CDG edges `(ctrl_id, body_stmt_id, label="CDG")` so enclosing IF/LOOP
    # headers are connected to their body statements even when the frontend omits CDG edges.
    for ctrl_u, body_v in all_structural_cdg_edges:
        if ctrl_u in condensed and body_v in condensed and ctrl_u != body_v:
            edge_key = (ctrl_u, body_v, "CDG", "")
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                condensed.add_edge(ctrl_u, body_v, label="CDG")

    for u, v, d in G.edges(data=True):
        elbl = str(d.get("label", "")).upper()
        if elbl == "AST":
            continue

        rep_u = global_node_to_rep.get(u)
        rep_v = global_node_to_rep.get(v)
        if rep_u is None or rep_v is None or rep_u == rep_v:
            continue
        if rep_u not in condensed or rep_v not in condensed:
            continue

        var_attr = str(d.get("variable") or d.get("name") or "").strip()
        if var_attr:
            alias_map = node_temp_alias_map.get(rep_u) or node_temp_alias_map.get(rep_v)
            if alias_map:
                var_attr = _substitute_temp_aliases(var_attr, alias_map)

        if elbl in ("DDG", "REACHING_DEF") and var_attr and _COMPILER_TEMP_RE.match(var_attr):
            # Only drop unresolved compiler-temp edges to/from structural boundary nodes (e.g. METHOD_RETURN);
            # preserve inter-statement edges so multi-statement lowered blocks keep their data-flow link.
            if not (
                condensed.nodes[rep_u].get("is_condensed_statement")
                and condensed.nodes[rep_v].get("is_condensed_statement")
            ):
                continue

        edge_key = (rep_u, rep_v, elbl, var_attr)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        edge_attrs: Dict[str, Any] = {"label": elbl}
        if var_attr:
            edge_attrs["variable"] = var_attr
        condensed.add_edge(rep_u, rep_v, **edge_attrs)

    return condensed


# ==============================================================================
# 4. Language-Agnostic Program Slicing (`condense_method_slice`)
# ==============================================================================

class CondensedSliceResult(dict):
    """Dictionary result from `condense_method_slice` that also behaves like a string (`slice_text`)."""

    def __str__(self) -> str:
        return str(self.get("slice_text", ""))

    def __contains__(self, item: object) -> bool:
        if super().__contains__(item):
            return True
        if isinstance(item, str):
            return item in str(self.get("slice_text", ""))
        return False

    def splitlines(self, keepends: bool = False) -> List[str]:
        return str(self.get("slice_text", "")).splitlines(keepends=keepends)


def _var_matches_target(candidate: str, target_vars: Set[str]) -> bool:
    """Checks if a variable/member expression matches or references any symbol in `target_vars`."""
    if not candidate:
        return False
    cand_clean = candidate.strip()
    if cand_clean in target_vars:
        return True
    tokens = set(_IDENT_TOKEN_RE.findall(cand_clean))
    return bool(tokens & target_vars)


def _extract_root_symbol(expr: str) -> str:
    """Extracts the root identifier from a C/Java/Python/Go member or index expression (`a->b[i].c` -> `a`)."""
    m = _IDENT_TOKEN_RE.search(expr)
    return m.group(0) if m else expr.strip()


def _is_non_defining_void_call(cond_G: nx.MultiDiGraph, s_id: int, target_set: Set[str]) -> bool:
    """Graph-theoretic check: returns True if `s_id` defines/mutates no variables (`writes == []`),
    is not a control/return/throw node, does not read any variable in `target_set` (which would
    make it a data-flow sink), and is not a directly targeted function call.
    """
    s_data = cond_G.nodes.get(s_id, {})
    s_lbl = str(s_data.get("label", ""))
    if (
        s_lbl in ("RETURN", "CONTROL_STRUCTURE", "<operator>.raise", "<operator>.throw")
        or s_lbl in _ALL_MUTATION_OPERATORS
    ):
        return False
    if s_data.get("writes"):
        return False
    s_reads = s_data.get("reads", [])
    if any(_var_matches_target(r, target_set) for r in s_reads):
        return False
    s_calls = s_data.get("calls", [])
    if any(_var_matches_target(c, target_set) for c in s_calls):
        return False
    return True


def _get_method_parent_class(G: nx.MultiDiGraph, m_id: int) -> str:
    """Returns the enclosing `TYPE_DECL` class name of `m_id`, if any."""
    for p, _, ed in G.in_edges(m_id, data=True):
        if ed.get("label") == "AST":
            pdata = G.nodes.get(p, {})
            if pdata.get("label") == "TYPE_DECL":
                return str(pdata.get("code", "")).strip()
            if pdata.get("code") == "<body>":
                for pp, _, eed in G.in_edges(p, data=True):
                    if eed.get("label") == "AST" and G.nodes.get(pp, {}).get("label") == "TYPE_DECL":
                        return str(G.nodes[pp].get("code", "")).strip()
    return ""


def _resolve_interprocedural_links(
    cond_G: nx.MultiDiGraph,
    m_id: int,
    kept_records: List[Dict[str, Any]],
    target_set: Set[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Resolves inter-procedural Callee and Sibling State-Mutator pointers for kept statements
    so the agent can hop directly across method boundaries using exact `METHOD #node_id` anchors.
    """
    caller_cls = _get_method_parent_class(cond_G, m_id)

    # Catalog repository methods in `cond_G`
    repo_methods: List[Dict[str, Any]] = []
    for cand_id in find_nodes(cond_G, label="METHOD"):
        if cand_id == m_id:
            continue
        cdata = cond_G.nodes[cand_id]
        m_code = str(cdata.get("code") or cdata.get("name") or "").strip()
        if (
            not m_code
            or m_code.startswith("<")
            or m_code.endswith("<metaClassAdapter>")
            or m_code in _INTERNAL_LOWERING_CALLS
        ):
            continue
        cls_name = _get_method_parent_class(cond_G, cand_id)
        if cls_name.startswith("Test") or m_code.startswith("test_"):
            continue
        qual_name = f"{cls_name}.{m_code}" if cls_name else m_code
        df_sum = cdata.get("data_flow_summary") or {}
        repo_methods.append({
            "node_id": cand_id,
            "class_name": cls_name,
            "method_name": m_code,
            "qualified_name": qual_name,
            "line_number": cdata.get("line_number"),
            "reads": list(df_sum.get("reads", cdata.get("reads", []))),
            "writes": list(df_sum.get("writes", cdata.get("writes", []))),
            "calls": list(df_sum.get("calls", cdata.get("calls", []))),
        })

    links: List[Dict[str, Any]] = []
    formatted_lines: List[str] = []
    seen_link_keys: Set[Tuple[str, int]] = set()

    # 1. Resolve Callee Pointers for calls invoked in kept statements
    kept_calls: List[str] = []
    for rec in kept_records:
        for c in rec.get("calls", []):
            if c and c not in kept_calls:
                kept_calls.append(c)

    for call_expr in kept_calls:
        parts = [p.strip() for p in re.split(r"(?:\.|->)", call_expr) if p.strip()]
        if not parts:
            continue
        short_name = parts[-1]
        recv_hint = parts[-2].lower() if len(parts) >= 2 else ""

        matches = [m for m in repo_methods if m["method_name"] == short_name]
        if not matches:
            continue

        def _rank_callee(cand: Dict[str, Any]) -> Tuple[int, int]:
            c_cls = cand["class_name"].lower()
            if recv_hint in ("self", "this") and caller_cls and cand["class_name"] == caller_cls:
                return (3, -int(cand["node_id"]))
            if recv_hint and c_cls and (recv_hint in c_cls or c_cls in recv_hint):
                return (2, -int(cand["node_id"]))
            if caller_cls and cand["class_name"] == caller_cls:
                return (1, -int(cand["node_id"]))
            return (0, -int(cand["node_id"]))

        matches.sort(key=_rank_callee, reverse=True)
        best = matches[0]
        link_key = ("callee", best["node_id"])
        if link_key not in seen_link_keys:
            seen_link_keys.add(link_key)
            links.append({
                "type": "callee",
                "call_expr": call_expr,
                "qualified_name": best["qualified_name"],
                "method_id": best["node_id"],
                "line_number": best["line_number"],
                "writes": best["writes"],
                "reads": best["reads"],
            })
            formatted_lines.append(
                f"  -> Callee `{best['qualified_name']}` (METHOD #{best['node_id']}): "
                f"writes={best['writes'][:4]}, reads={best['reads'][:4]}"
            )

    # 2. Resolve Sibling State-Mutator Pointers for read attributes (`self.x`, `this->x`, `obj.field`)
    read_attrs: List[str] = []
    for rec in kept_records:
        for r in rec.get("reads", []):
            if ("." in r or "->" in r) and "[" not in r and r not in read_attrs:
                read_attrs.append(r)

    for attr_expr in read_attrs:
        attr_parts = [p.strip() for p in re.split(r"(?:\.|->)", attr_expr) if p.strip()]
        if len(attr_parts) < 2:
            continue
        field_tail = attr_parts[-1]
        is_self_attr = attr_parts[0] in ("self", "this")

        mutator_cands = []
        for cand in repo_methods:
            if is_self_attr and caller_cls and cand["class_name"] != caller_cls:
                continue
            cand_writes = cand["writes"]
            if any(
                w == attr_expr
                or w.endswith(f".{field_tail}")
                or w.endswith(f"->{field_tail}")
                or w.startswith(f"{attr_expr}[")
                for w in cand_writes
            ):
                mutator_cands.append(cand)

        mutator_cands.sort(key=lambda c: (0 if c["method_name"] != "__init__" else 1, c["node_id"]))
        for mut in mutator_cands[:2]:
            link_key = ("mutator", mut["node_id"])
            if link_key not in seen_link_keys:
                seen_link_keys.add(link_key)
                links.append({
                    "type": "sibling_mutator",
                    "attribute": attr_expr,
                    "qualified_name": mut["qualified_name"],
                    "method_id": mut["node_id"],
                    "line_number": mut["line_number"],
                    "writes": mut["writes"],
                })
                formatted_lines.append(
                    f"  -> Sibling Mutator of `{attr_expr}`: `{mut['qualified_name']}` "
                    f"(METHOD #{mut['node_id']}): writes={mut['writes'][:4]}"
                )

    return links, formatted_lines


def condense_method_slice(
    G: nx.MultiDiGraph,
    method_id: Union[int, str],
    target_vars: Union[Sequence[str], str],
    token_budget: int = 200,
) -> CondensedSliceResult:
    """Language-agnostic backward + forward program slice for `target_vars` inside `method_id`,
    folding non-causal statement blocks into `# [condensed block: N statements]` markers,
    appending Inter-Procedural Boundary Pointers, and enforcing `token_budget`.
    """
    m_id = resolve_method_id(G, method_id)

    if isinstance(target_vars, str):
        target_set = {v.strip() for v in target_vars.split(",") if v.strip()}
    else:
        target_set = {str(v).strip() for v in target_vars if str(v).strip()}

    ast_children = _build_ast_children_map(G)
    direct_stmts = [
        ch for ch in ast_children.get(m_id, [])
        if G.nodes.get(ch, {}).get("is_condensed_statement")
    ]

    if direct_stmts:
        cond_G = G
        stmt_ids = sorted(
            direct_stmts,
            key=lambda nid: (cond_G.nodes[nid].get("line_number") or 0, nid),
        )
        method_summary = cond_G.nodes[m_id].get("data_flow_summary") or {
            "reads": cond_G.nodes[m_id].get("reads", []),
            "writes": cond_G.nodes[m_id].get("writes", []),
            "calls": cond_G.nodes[m_id].get("calls", []),
        }
    else:
        cond_G = condense_cpg(G)
        cond_ast = _build_ast_children_map(cond_G)
        stmt_ids = sorted(
            [
                ch for ch in cond_ast.get(m_id, [])
                if cond_G.nodes.get(ch, {}).get("is_condensed_statement")
            ],
            key=lambda nid: (cond_G.nodes[nid].get("line_number") or 0, nid),
        )
        method_summary = cond_G.nodes[m_id].get(
            "data_flow_summary", {"reads": [], "writes": [], "calls": []}
        )

    stmt_set = set(stmt_ids)

    # 1. Identify Seed Statements that define, mutate, or read `target_vars`
    seed_stmts: Set[int] = set()
    for s_id in stmt_ids:
        if _is_non_defining_void_call(cond_G, s_id, target_set):
            continue
        s_data = cond_G.nodes[s_id]
        s_reads = s_data.get("reads", [])
        s_writes = s_data.get("writes", [])
        s_code = str(s_data.get("code", ""))

        if any(_var_matches_target(r, target_set) for r in s_reads) or any(
            _var_matches_target(w, target_set) for w in s_writes
        ):
            seed_stmts.add(s_id)
        elif not target_set or any(
            re.search(rf"\b{re.escape(tv)}\b", s_code) for tv in target_set
        ):
            seed_stmts.add(s_id)

    if not seed_stmts and stmt_ids:
        seed_stmts.add(stmt_ids[0])

    # 2. Build Statement-Level Dependency Graph (REACHING_DEF / DDG + CDG + Def-Use chains)
    dep_graph = nx.DiGraph()
    cfg_graph = nx.DiGraph()
    cdg_graph = nx.DiGraph()
    for s_id in stmt_ids:
        dep_graph.add_node(s_id)
        cfg_graph.add_node(s_id)
        cdg_graph.add_node(s_id)

    for u, v, d in cond_G.edges(data=True):
        if u in stmt_set and v in stmt_set and u != v:
            elbl = str(d.get("label", "")).upper()
            if elbl == "CDG":
                cdg_graph.add_edge(u, v)
                if not _is_non_defining_void_call(cond_G, v, target_set):
                    dep_graph.add_edge(u, v, label=elbl)
            elif elbl in ("REACHING_DEF", "DDG", "REF", "ARGUMENT"):
                if not _is_non_defining_void_call(cond_G, v, target_set):
                    dep_graph.add_edge(u, v, label=elbl, variable=d.get("variable"))
            elif elbl == "CFG":
                cfg_graph.add_edge(u, v)

    # Identify loop scopes (loop header + CDG-dominated body statements) for loop-carried Def-Use
    loop_scopes: List[Set[int]] = []
    for s_id in stmt_ids:
        s_data = cond_G.nodes[s_id]
        if str(s_data.get("label", "")) == "CONTROL_STRUCTURE":
            s_code = str(s_data.get("code", "")).strip()
            body_desc = nx.descendants(cdg_graph, s_id)
            is_loop_ctrl = bool(
                re.match(r"^(?:LOOP|FOR|WHILE|DO)\b", s_code, re.IGNORECASE)
                or any(nx.has_path(cfg_graph, b_node, s_id) for b_node in body_desc)
            )
            if is_loop_ctrl and body_desc:
                loop_scopes.append({s_id} | body_desc)

    # Supplement explicit DDG edges with statement-level Def-Use (`writes` -> `reads`) edges,
    # including loop-carried back-edges (`j < i` inside the same loop body or reachable via `cfg_graph`).
    for i, u in enumerate(stmt_ids):
        u_writes = cond_G.nodes[u].get("writes", [])
        u_base_writes = {_extract_root_symbol(w) for w in u_writes if w}
        if not u_base_writes:
            continue
        for j, v in enumerate(stmt_ids):
            if u == v or _is_non_defining_void_call(cond_G, v, target_set):
                continue
            is_forward = j > i
            is_loop_or_cfg_reachable = (
                nx.has_path(cfg_graph, u, v)
                or any(u in scope and v in scope for scope in loop_scopes)
            )
            if not (is_forward or is_loop_or_cfg_reachable):
                continue

            v_reads = cond_G.nodes[v].get("reads", [])
            v_writes = cond_G.nodes[v].get("writes", [])
            if is_forward:
                v_base_refs = {
                    _extract_root_symbol(r) for r in (list(v_reads) + list(v_writes)) if r
                }
            else:
                v_base_refs = {_extract_root_symbol(r) for r in v_reads if r}
            if u_base_writes & v_base_refs:
                dep_graph.add_edge(u, v, label="REACHING_DEF")

    # 3. Compute Backward and Forward Slice from Seeds
    backward_slice: Set[int] = set(seed_stmts)
    forward_slice: Set[int] = set(seed_stmts)

    for s in seed_stmts:
        if s in dep_graph:
            backward_slice |= nx.ancestors(dep_graph, s)
            forward_slice |= nx.descendants(dep_graph, s)

    causal_stmts: Set[int] = {
        s for s in (backward_slice | forward_slice)
        if not _is_non_defining_void_call(cond_G, s, target_set)
    }

    # Assign hop distance from `seed_stmts` for token-budget prioritization
    stmt_priority: Dict[int, Tuple[int, int]] = {}
    undirected_dep = dep_graph.to_undirected()
    for s_id in causal_stmts:
        s_lbl = str(cond_G.nodes[s_id].get("label", ""))
        is_direct_mutation = (
            s_id in seed_stmts
            and (s_lbl in _ALL_MUTATION_OPERATORS or s_lbl == "RETURN")
        )
        min_dist = 0 if s_id in seed_stmts else 99
        if s_id not in seed_stmts:
            for seed in seed_stmts:
                if nx.has_path(undirected_dep, seed, s_id):
                    d_len = nx.shortest_path_length(undirected_dep, seed, s_id)
                    min_dist = min(min_dist, d_len)
        stmt_priority[s_id] = (0 if is_direct_mutation else 1, min_dist)

    # 4. Render Language-Agnostic Method Header & Statements
    m_name = str(cond_G.nodes[m_id].get("code") or cond_G.nodes[m_id].get("name") or f"method_{m_id}")
    m_line = cond_G.nodes[m_id].get("line_number")
    params = [
        str(cond_G.nodes[ch].get("code", "")).strip()
        for ch in _build_ast_children_map(cond_G).get(m_id, [])
        if str(cond_G.nodes.get(ch, {}).get("label", "")) == "PARAM"
    ]
    header_line = (
        f"[METHOD] {m_name}({', '.join(params)})  # L{m_line}"
        if m_line
        else f"[METHOD] {m_name}({', '.join(params)})"
    )

    def _render_slice(
        kept_set: Set[int], include_header: bool = True
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        lines: List[str] = [header_line] if include_header else []
        indent = "  " if include_header else ""
        kept_records: List[Dict[str, Any]] = []
        folded_records: List[Dict[str, Any]] = []

        non_causal_run: List[int] = []

        def _flush_folded() -> None:
            if not non_causal_run:
                return
            count = len(non_causal_run)
            marker = f"{indent}# [condensed block: {count} statements]"
            lines.append(marker)
            folded_records.append({
                "marker": marker.strip(),
                "statement_count": count,
                "node_ids": list(non_causal_run),
                "start_line": cond_G.nodes[non_causal_run[0]].get("line_number"),
                "end_line": cond_G.nodes[non_causal_run[-1]].get("line_number"),
            })
            non_causal_run.clear()

        for s_id in stmt_ids:
            if s_id in kept_set:
                _flush_folded()
                s_data = cond_G.nodes[s_id]
                s_line = s_data.get("line_number")
                s_code = str(s_data.get("code", "")).strip()
                line_prefix = f"L{s_line}: " if s_line else ""
                lines.append(f"{indent}{line_prefix}{s_code}")
                kept_records.append({
                    "node_id": s_id,
                    "line_number": s_line,
                    "label": s_data.get("label"),
                    "code": s_code,
                    "reads": s_data.get("reads", []),
                    "writes": s_data.get("writes", []),
                    "calls": s_data.get("calls", []),
                })
            else:
                non_causal_run.append(s_id)

        _flush_folded()
        return "\n".join(lines), kept_records, folded_records

    raw_ast_nodes = traverse_ast(G, root_node_id=m_id, max_depth=6)
    raw_serialized = "\n".join(
        f"[{n['label']} #{n['node_id']}] L{n.get('line_number')}: {n.get('code')}"
        for n in raw_ast_nodes
    )
    raw_tokens = _estimate_tokens(raw_serialized)

    # 5. Enforce `token_budget`
    current_kept = set(causal_stmts)
    slice_text, kept_list, folded_list = _render_slice(current_kept, include_header=True)
    used_tokens = _estimate_tokens(slice_text)

    if token_budget > 0 and used_tokens > token_budget and len(current_kept) > 1:
        removable = sorted(
            current_kept,
            key=lambda nid: (stmt_priority.get(nid, (1, 99))[0], stmt_priority.get(nid, (1, 99))[1], nid),
            reverse=True,
        )
        for drop_id in removable:
            if len(current_kept) <= 1:
                break
            current_kept.remove(drop_id)
            slice_text, kept_list, folded_list = _render_slice(current_kept, include_header=True)
            used_tokens = _estimate_tokens(slice_text)
            if used_tokens <= token_budget:
                break

    if token_budget > 0 and used_tokens > token_budget:
        slice_text, kept_list, folded_list = _render_slice(current_kept, include_header=False)
        used_tokens = _estimate_tokens(slice_text)

    # 6. Resolve Inter-Procedural Callee & Sibling State-Mutator Pointers
    interproc_links, interproc_lines = _resolve_interprocedural_links(
        cond_G, m_id, kept_list, target_set
    )
    if interproc_lines:
        footer_header = f"\n[Inter-Procedural Links for target_vars={sorted(target_set)}]:\n"
        # Add lines one by one if they fit within `token_budget`
        added_lines: List[str] = []
        for l_str in interproc_lines:
            candidate_text = slice_text + footer_header + "\n".join(added_lines + [l_str])
            if token_budget <= 0 or _estimate_tokens(candidate_text) <= token_budget:
                added_lines.append(l_str)
            else:
                break
        if added_lines:
            slice_text = slice_text + footer_header + "\n".join(added_lines)
            used_tokens = _estimate_tokens(slice_text)

    reduction_pct = round((1.0 - (used_tokens / max(1, raw_tokens))) * 100.0, 2)

    return CondensedSliceResult({
        "method_id": m_id,
        "method_name": m_name,
        "target_vars": sorted(target_set),
        "slice_text": slice_text,
        "kept_statements": kept_list,
        "folded_blocks": folded_list,
        "method_summary": method_summary,
        "interprocedural_links": interproc_links,
        "token_stats": {
            "used_tokens": used_tokens,
            "raw_tokens": raw_tokens,
            "token_budget": token_budget,
            "within_budget": used_tokens <= token_budget,
            "token_reduction_pct": reduction_pct,
            "total_statements": len(stmt_ids),
            "kept_statements_count": len(kept_list),
            "folded_statements_count": sum(b["statement_count"] for b in folded_list),
        },
    })


# ==============================================================================
# 5. Optimization Benchmark Comparison (Old Raw CPG vs. New Condensed CPG)
# ==============================================================================

def compare_cpg_optimization(
    raw_graph: nx.MultiDiGraph,
    target_methods: Optional[List[Tuple[str, str]]] = None,
    token_budget: int = 120,
) -> Dict[str, Any]:
    """Benchmarks the Old (Raw CPG) method against the New (Condensed CPG + Method Slicing)
    method across graph size, traversal hops, token payload, and execution time.
    """
    if target_methods is None:
        target_methods = [
            ("apply_discount", "discount_percent"),
            ("generate_category_summary", "item_total"),
        ]

    t0 = time.perf_counter()
    condensed_graph = condense_cpg(raw_graph)
    condense_time_ms = (time.perf_counter() - t0) * 1000.0

    raw_nodes = raw_graph.number_of_nodes()
    raw_edges = raw_graph.number_of_edges()
    cond_nodes = condensed_graph.number_of_nodes()
    cond_edges = condensed_graph.number_of_edges()

    node_reduction_pct = round((1.0 - cond_nodes / max(1, raw_nodes)) * 100.0, 2)
    edge_reduction_pct = round((1.0 - cond_edges / max(1, raw_edges)) * 100.0, 2)

    method_benchmarks: List[Dict[str, Any]] = []

    for m_name, var_name in target_methods:
        m_id = resolve_method_id(raw_graph, m_name)

        # --- OLD METHOD: Raw CPG Traversals ---
        t_old_start = time.perf_counter()
        old_ast = traverse_ast(raw_graph, root_node_id=m_id, max_depth=6)
        old_cfg = traverse_cfg(raw_graph, start_node_id=m_id, max_paths=3)
        old_sink = trace_variable_to_sink(raw_graph, start_node_id=m_id, var_name=var_name)
        old_time_ms = (time.perf_counter() - t_old_start) * 1000.0

        old_payload_str = json.dumps({
            "ast": old_ast,
            "cfg": old_cfg,
            "sink": old_sink,
        })
        old_tokens = _estimate_tokens(old_payload_str)
        old_cfg_hops = max((len(p) for p in old_cfg), default=0)
        old_ddg_hops = len(old_sink.get("detailed_path", []))

        # --- NEW METHOD: Condensed CPG + Method Slice ---
        t_new_start = time.perf_counter()
        new_ast = traverse_ast(condensed_graph, root_node_id=m_id, max_depth=6)
        new_cfg = traverse_cfg(condensed_graph, start_node_id=m_id, max_paths=3)
        new_sink = trace_variable_to_sink(condensed_graph, start_node_id=m_id, var_name=var_name)
        slice_res = condense_method_slice(
            condensed_graph, method_id=m_id, target_vars=[var_name], token_budget=token_budget
        )
        new_time_ms = (time.perf_counter() - t_new_start) * 1000.0

        new_payload_str = json.dumps({
            "summary": slice_res["method_summary"],
            "slice": slice_res["slice_text"],
            "sink": new_sink.get("summary"),
        })
        new_tokens = _estimate_tokens(new_payload_str)
        new_cfg_hops = max((len(p) for p in new_cfg), default=0)
        new_ddg_hops = len(new_sink.get("detailed_path", []))

        method_benchmarks.append({
            "method": m_name,
            "target_var": var_name,
            "old_ast_nodes": len(old_ast),
            "new_ast_nodes": len(new_ast),
            "ast_node_reduction_pct": round((1.0 - len(new_ast) / max(1, len(old_ast))) * 100.0, 2),
            "old_cfg_hops": old_cfg_hops,
            "new_cfg_hops": new_cfg_hops,
            "old_ddg_hops": old_ddg_hops,
            "new_ddg_hops": new_ddg_hops,
            "old_payload_tokens": old_tokens,
            "new_payload_tokens": new_tokens,
            "token_reduction_pct": round((1.0 - new_tokens / max(1, old_tokens)) * 100.0, 2),
            "slice_tokens": slice_res["token_stats"]["used_tokens"],
            "old_query_time_ms": round(old_time_ms, 3),
            "new_query_time_ms": round(new_time_ms, 3),
            "speedup_factor": round(old_time_ms / max(0.001, new_time_ms), 2),
            "slice_text": slice_res["slice_text"],
            "method_summary": slice_res["method_summary"],
        })

    return {
        "graph_metrics": {
            "raw_nodes": raw_nodes,
            "condensed_nodes": cond_nodes,
            "node_reduction_pct": node_reduction_pct,
            "raw_edges": raw_edges,
            "condensed_edges": cond_edges,
            "edge_reduction_pct": edge_reduction_pct,
            "condense_build_time_ms": round(condense_time_ms, 2),
        },
        "method_benchmarks": method_benchmarks,
    }


# ==============================================================================
# 6. CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Condense Joern CPG14 into statement-level nodes and benchmark slice optimization."
    )
    parser.add_argument(
        "--cpg-path",
        type=str,
        default="cpg_json",
        help="Path to CPG JSON or directory of exported files (default: cpg_json)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=120,
        help="Token budget for condense_method_slice (default: 120)",
    )
    args = parser.parse_args()

    cpg_path = Path(args.cpg_path)
    if not cpg_path.exists():
        print(f"[-] Error: CPG path '{cpg_path}' not found.", file=sys.stderr)
        sys.exit(1)

    raw_graph = load_cpg_json(cpg_path)
    report = compare_cpg_optimization(raw_graph, token_budget=args.budget)
    gm = report["graph_metrics"]

    print("=" * 75)
    print("  CPG CONDENSER OPTIMIZATION BENCHMARK (OLD RAW CPG vs. NEW CONDENSED CPG)")
    print("=" * 75)
    print(f"  Global Graph Nodes : {gm['raw_nodes']:>5} (Old) -> {gm['condensed_nodes']:>5} (New)  [-{gm['node_reduction_pct']}%]")
    print(f"  Global Graph Edges : {gm['raw_edges']:>5} (Old) -> {gm['condensed_edges']:>5} (New)  [-{gm['edge_reduction_pct']}%]")
    print(f"  Condensation Time  : {gm['condense_build_time_ms']} ms")
    print("-" * 75)

    for mb in report["method_benchmarks"]:
        print(f"\n[*] Method: `{mb['method']}` (Target Variable: `{mb['target_var']}`)")
        print(f"    - Method AST Nodes   : {mb['old_ast_nodes']:>4} -> {mb['new_ast_nodes']:>4}  (-{mb['ast_node_reduction_pct']}%)")
        print(f"    - Max CFG Path Hops  : {mb['old_cfg_hops']:>4} -> {mb['new_cfg_hops']:>4}")
        print(f"    - DDG Trace Hops     : {mb['old_ddg_hops']:>4} -> {mb['new_ddg_hops']:>4}")
        print(f"    - Context Tokens     : {mb['old_payload_tokens']:>4} -> {mb['new_payload_tokens']:>4}  (-{mb['token_reduction_pct']}%)")
        print(f"    - Query Latency      : {mb['old_query_time_ms']} ms -> {mb['new_query_time_ms']} ms ({mb['speedup_factor']}x speedup)")
        print(f"    - Data-Flow Summary  : reads={mb['method_summary']['reads']}, writes={mb['method_summary']['writes']}, calls={mb['method_summary']['calls']}")
        print(f"    - Condensed Slice ({mb['slice_tokens']} tokens):\n")
        for line in mb["slice_text"].splitlines():
            print(f"      {line}")

    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
