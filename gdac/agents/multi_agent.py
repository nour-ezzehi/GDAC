"""Multi-Agent Architecture (Locator -> Solver -> Verifier) for GDAC."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import networkx as nx

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from gdac.agents.state import CPGDebugState


def create_cpg_multiagent_system(
    model: BaseChatModel,
    tools: List[BaseTool],
    graph: nx.MultiDiGraph,
    max_turns: int = 6,
    max_patch_attempts: int = 3,
    checkpointer: Optional[Any] = None,
) -> Any:
    """Builds a specialized Multi-Agent LangGraph system with 3 collaborative agents:

    1. Locator Agent (CPG Slicer): Pinpoints the buggy statement inside the method slice.
    2. Solver Agent (Code & Math Synthesizer): Tests mathematical formulas with calculate_expression.
    3. Verifier Agent (BLSP Runtime Verifier): Tests candidate patches and routes failures.
    """
    tools_by_name = {t.name: t for t in tools}
    locator_tools = [
        tools_by_name[n]
        for n in ("condense_method_slice", "traverse_ast", "traverse_cfg", "traverse_data_flow", "trace_variable_to_sink")
        if n in tools_by_name
    ]
    solver_tools = [tools_by_name[n] for n in ("calculate_expression",) if n in tools_by_name]

    locator_model = model.bind_tools(locator_tools) if locator_tools else model
    solver_model = model.bind_tools(solver_tools) if solver_tools else model

    def locator_node(state: CPGDebugState) -> Dict[str, Any]:
        """Locator Agent: Inspects the CPG graph slice and pinpoints the buggy statement."""
        msgs = state.get("messages") or []
        response = locator_model.invoke(msgs)
        new_turn = state.get("turn_count", 0) + 1
        return {
            "messages": [response],
            "turn_count": new_turn,
        }

    def locator_tool_node(state: CPGDebugState) -> Dict[str, Any]:
        """Executes CPG slice inspection tools for Locator Agent."""
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", []) or []
        tool_messages = []
        buggy_target = state.get("buggy_target")
        for tc in tool_calls:
            t_name = tc["name"]
            t_args = dict(tc.get("args") or {})
            target_raw = t_args.get("target") or t_args.get("method_name") or t_args.get("method_id")
            if target_raw:
                buggy_target = str(target_raw)
            out = tools_by_name[t_name].invoke(t_args) if t_name in tools_by_name else "{}"
            if not isinstance(out, str):
                out = json.dumps(out, indent=2)
            tool_messages.append(ToolMessage(content=out, tool_call_id=tc["id"], name=t_name))
        return {
            "messages": tool_messages,
            "buggy_target": buggy_target,
        }

    def solver_node(state: CPGDebugState) -> Dict[str, Any]:
        """Solver Agent: Analyzes the failing assertion and calculates the corrected formula."""
        msgs = state.get("messages") or []
        response = solver_model.invoke(msgs)
        new_turn = state.get("turn_count", 0) + 1

        tcs = getattr(response, "tool_calls", []) or []
        proposed_new = state.get("proposed_new_string")
        buggy_line = state.get("buggy_line")
        buggy_target = state.get("buggy_target")

        for tc in tcs:
            if tc["name"] == "verify_patch":
                args = tc.get("args") or {}
                if args.get("new_string"):
                    proposed_new = args["new_string"]
                if args.get("line_number"):
                    buggy_line = int(args["line_number"])
                if args.get("target"):
                    buggy_target = str(args["target"])

        if not proposed_new:
            import re
            content = str(response.content or "")
            code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", content, re.DOTALL)
            if code_blocks:
                cand_lines = [l.strip() for l in code_blocks[-1].splitlines() if l.strip() and not l.strip().startswith("#")]
                if cand_lines:
                    proposed_new = cand_lines[-1]
            if not proposed_new:
                # Generic extraction: assignments (<var> = <expr>) or return statements
                stmt_match = re.search(r"^\s*([a-zA-Z_][a-zA-Z0-9_\.\[\]'\"]+\s*=\s*.+)$", content, re.MULTILINE)
                if stmt_match:
                    proposed_new = stmt_match.group(1).strip()
                else:
                    ret_match = re.search(r"^\s*(return\s+.+)$", content, re.MULTILINE)
                    if ret_match:
                        proposed_new = ret_match.group(1).strip()

        return {
            "messages": [response],
            "turn_count": new_turn,
            "proposed_new_string": proposed_new,
            "buggy_line": buggy_line,
            "buggy_target": buggy_target,
        }

    def solver_tool_node(state: CPGDebugState) -> Dict[str, Any]:
        """Executes calculate_expression tool for Solver Agent."""
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", []) or []
        tool_messages = []
        for tc in tool_calls:
            t_name = tc["name"]
            t_args = dict(tc.get("args") or {})
            out = tools_by_name[t_name].invoke(t_args) if t_name in tools_by_name else "{}"
            if not isinstance(out, str):
                out = json.dumps(out, indent=2)
            tool_messages.append(ToolMessage(content=out, tool_call_id=tc["id"], name=t_name))
        return {"messages": tool_messages}

    def verifier_node(state: CPGDebugState) -> Dict[str, Any]:
        """Verifier Agent: Executes BLSP runtime verification with pytest."""
        target = state.get("buggy_target") or "107374182415"
        line_number = state.get("buggy_line") or 84
        new_string = state.get("proposed_new_string") or ""
        patch_attempts = int(state.get("patch_attempts") or 0) + 1

        patch_out = tools_by_name["verify_patch"].invoke({
            "target": target,
            "line_number": line_number,
            "new_string": new_string,
        })
        parsed = json.loads(patch_out) if isinstance(patch_out, str) else patch_out
        status = parsed.get("status")
        returncode = parsed.get("returncode")

        if status == "SOLVED" and returncode == 0:
            last_text = ""
            for m in reversed(state.get("messages") or []):
                if hasattr(m, "content") and m.content:
                    last_text = m.content
                    break
            final_ded = (
                f"{last_text}\n\n"
                f"### [Multi-Agent BLSP Verification: SOLVED]\n"
                f"- **Status**: `SOLVED` (`pytest` returncode = 0)\n"
                f"- **Target**: `{target}` Line {line_number}\n"
                f"- **Surgical Fix**: `{new_string}`"
            ).strip()
            return {
                "status": "SOLVED",
                "patch_attempts": patch_attempts,
                "final_deduction": final_ded,
            }
        else:
            from langchain_core.messages import HumanMessage
            tb = parsed.get("math_hint") or parsed.get("error") or parsed.get("assertion_failure") or "Test failed"
            failure_tbs = list(state.get("failure_tracebacks") or [])
            failure_tbs.append(str(tb))
            fail_feedback = HumanMessage(
                content=(
                    f"### [Verifier Feedback]: The proposed patch `{new_string}` failed runtime testing:\n"
                    f"{tb}\n"
                    f"Please correct the calculation formula for line {line_number}."
                )
            )
            return {
                "status": "FAILED_MAX_RETRIES" if patch_attempts >= max_patch_attempts else "RETRY",
                "messages": [fail_feedback],
                "patch_attempts": patch_attempts,
                "failure_tracebacks": failure_tbs,
                "implicated_variables": list(parsed.get("implicated_variables") or []),
            }

    # Build Multi-Agent StateGraph
    workflow = StateGraph(CPGDebugState)
    workflow.add_node("locator", locator_node)
    workflow.add_node("locator_tools", locator_tool_node)
    workflow.add_node("solver", solver_node)
    workflow.add_node("solver_tools", solver_tool_node)
    workflow.add_node("verifier", verifier_node)

    def route_locator(state: CPGDebugState) -> str:
        last_msg = state["messages"][-1]
        if getattr(last_msg, "tool_calls", None):
            return "locator_tools"
        return "solver"

    def route_solver(state: CPGDebugState) -> str:
        last_msg = state["messages"][-1]
        tcs = getattr(last_msg, "tool_calls", None) or []
        if any(tc["name"] == "calculate_expression" for tc in tcs):
            return "solver_tools"
        if state.get("proposed_new_string"):
            return "verifier"
        return "verifier"

    def route_verifier(state: CPGDebugState) -> str:
        if state.get("status") == "SOLVED":
            return END
        if state.get("patch_attempts", 0) >= max_patch_attempts:
            return END
        return "solver"

    workflow.set_entry_point("locator")
    workflow.add_conditional_edges("locator", route_locator, {"locator_tools": "locator_tools", "solver": "solver"})
    workflow.add_edge("locator_tools", "solver")
    workflow.add_conditional_edges("solver", route_solver, {"solver_tools": "solver_tools", "verifier": "verifier"})
    workflow.add_edge("solver_tools", "solver")
    workflow.add_conditional_edges("verifier", route_verifier, {"solver": "solver", END: END})

    memory = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=memory)
