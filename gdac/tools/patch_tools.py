"""BLSP (Blind-Locate, Surgical-Patch) Verification Tool for GDAC Agents."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from gdac.tools.cpg_tools import (
    materialize_source_window,
    resolve_method_source_file,
    resolve_target_node,
    _normalize_expr_for_match,
)
from gdac.services.test_runner import get_project_root, resolve_test_cmd


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
    var_regex = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*(?:(?:\.|->)[a-zA-Z_][a-zA-Z0-9_]*)?)\b")
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

    for m in var_regex.finditer(search_blob):
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
    2. Surgical Diff: Replaces `old_string -> new_string` within the localized window.
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

    cmd, resolved_test, project_root = resolve_test_cmd(file_path, test_path)

    if old_clean == new_clean:
        tb_hint = ""
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            combined = (proc.stdout + "\n" + proc.stderr).strip()
            f_lines = [
                l.strip()
                for l in combined.splitlines()
                if any(k in l for k in ("AssertionError", "FAILED", "FAIL:", "E   ", "assert", "expected"))
            ]
            if f_lines:
                tb_hint = "\n  -> Failing Test Output: " + " | ".join(f_lines[-4:])
        except Exception:
            pass

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
                f"You submitted the existing buggy line as the fix! This line causes the test failure:{tb_hint}\n"
                f"You must provide a modified `new_string` that fixes the calculation (e.g. use `calculate_expression` to verify your formula before patching)."
            ),
            "implicated_variables": _extract_implicated_variables(old_clean, graph, method_id),
        }

    # Step 2: Surgical Diff within the 15-line materialized window
    w_start = window_info["start_line"]
    w_end = window_info["end_line"]
    patched_content: Optional[str] = None
    applied_line: Optional[int] = None

    # 2a. Exact match within window slice
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
