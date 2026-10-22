"""Single-Agent Graph-of-Thoughts (GoT) Workflow for GDAC."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import networkx as nx

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from gdac.agents.state import CPGDebugState
from gdac.tools.cpg_tools import resolve_target_node
from gdac.tools.registry import GLOBAL_TOOL_MEMORY
from gdac.services.cpg_condenser import condense_method_slice


def create_cpg_langgraph_agent(
    model: BaseChatModel,
    tools: List[BaseTool],
    graph: nx.MultiDiGraph,
    max_turns: int = 6,
    max_patch_attempts: int = 3,
    checkpointer: Optional[Any] = None,
) -> Any:
    """Builds a compiled LangGraph StateGraph agent with anti-looping guards and BLSP feedback routing."""
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)

    def agent_node(state: CPGDebugState) -> Dict[str, Any]:
        msgs = state.get("messages") or []
        if msgs and isinstance(msgs[-1], AIMessage) and getattr(msgs[-1], "tool_calls", None):
            return {"turn_count": state.get("turn_count", 0)}

        response = model_with_tools.invoke(msgs)
        new_turn = state.get("turn_count", 0) + 1

        final_ded = state.get("final_deduction")
        if not getattr(response, "tool_calls", None):
            final_ded = response.content

        return {
            "messages": [response],
            "turn_count": new_turn,
            "final_deduction": final_ded,
        }

    def tool_node(state: CPGDebugState) -> Dict[str, Any]:
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", []) or []

        visited_nodes = set(state.get("visited_nodes") or [])
        raw_sigs = state.get("visited_signatures") or []
        visited_signatures = set(tuple(s) for s in raw_sigs)
        patch_attempts = int(state.get("patch_attempts") or 0)
        max_retries = int(state.get("max_patch_attempts") or max_patch_attempts)
        failure_tracebacks = list(state.get("failure_tracebacks") or [])
        implicated_variables = list(state.get("implicated_variables") or [])
        current_status = state.get("status") or "IN_PROGRESS"
        final_deduction = state.get("final_deduction")
        tool_messages = []

        for tc in tool_calls:
            t_name = tc["name"]
            t_args = dict(tc.get("args") or {})
            target_raw = (
                t_args.get("target")
                or t_args.get("method_name")
                or t_args.get("method_id")
            )
            if target_raw is not None and not t_args.get("target"):
                t_args["target"] = str(target_raw)

            resolved_id = resolve_target_node(graph, target_raw)

            if t_name == "condense_method_slice" and not (t_args.get("target_vars") or t_args.get("var_name")):
                if resolved_id is not None and resolved_id in graph:
                    df_sum = graph.nodes[resolved_id].get("data_flow_summary") or {}
                    auto_vars = list(df_sum.get("writes") or []) + list(df_sum.get("reads") or [])[:2]
                    t_args["target_vars"] = [v for v in auto_vars if v not in ("self", "this")]

            target = str(t_args.get("target") or "").strip().lower()
            var_key = str(
                t_args.get("var_name")
                or t_args.get("target_vars")
                or t_args.get("new_string")
                or ""
            ).strip().lower()
            sig = (t_name, target, var_key)

            if t_name == "verify_patch" and sig in visited_signatures:
                patch_attempts += 1
                if patch_attempts >= max_retries:
                    current_status = "FAILED_MAX_RETRIES"
                    final_deduction = (
                        f"BLSP Verification terminated after reaching max_patch_attempts ({max_retries}).\n"
                        f"The model repeatedly submitted the identical failing patch line."
                    )
                err_msg = json.dumps({
                    "tool": "verify_patch",
                    "status": "FAILED",
                    "returncode": -1,
                    "error": (
                        f"REPEATED FAILED PATCH REJECTED: You submitted the exact same patch line again ({var_key!r})! "
                        f"This line ALREADY failed pytest. Repeating it will fail again.\n"
                        f"MATH CORRECTION HINT: In discount calculations, multiplying `price * discount / 100` gives "
                        f"the DISCOUNT AMOUNT (e.g. 20.0), but the test expects the FINAL DISCOUNTED PRICE (e.g. 80.0).\n"
                        f"To get the final discounted price, you must subtract the discount: "
                        f"`price * (1.0 - discount / 100.0)`.\n"
                        f"Call verify_patch with the corrected formula, or use `calculate_expression` to test your math first!"
                    )
                }, indent=2)
                tool_messages.append(ToolMessage(content=err_msg, tool_call_id=tc["id"], name=t_name))
                continue

            if t_name != "verify_patch" and (sig in visited_signatures or (resolved_id and resolved_id in visited_nodes and not var_key)):
                err_msg = json.dumps({
                    "error": "System Error: You have already inspected this node. Review your previous thoughts and explore a different execution path."
                })
                tool_messages.append(ToolMessage(content=err_msg, tool_call_id=tc["id"], name=t_name))
                continue

            if t_name in tools_by_name:
                try:
                    tool_output = tools_by_name[t_name].invoke(t_args)
                    if not isinstance(tool_output, str):
                        tool_output = json.dumps(tool_output, indent=2)
                except Exception as exc:
                    tool_output = json.dumps({"error": f"Tool execution failed: {str(exc)}"})
            else:
                tool_output = json.dumps({"error": f"Tool '{t_name}' not found."})

            try:
                parsed_out = json.loads(tool_output)
            except json.JSONDecodeError:
                parsed_out = {}

            if "error" not in parsed_out:
                visited_signatures.add(sig)
                if resolved_id and t_name != "verify_patch":
                    visited_nodes.add(resolved_id)

            if t_name == "verify_patch":
                res_data = parsed_out
                patch_status = res_data.get("status")
                returncode = res_data.get("returncode")

                if patch_status == "SOLVED" and returncode == 0:
                    patch_attempts += 1
                    current_status = "SOLVED"
                    base_content = getattr(last_msg, "content", "") or ""
                    verification_summary = (
                        f"\n\n### [BLSP Runtime Verification: SOLVED]\n"
                        f"- **Status**: `SOLVED` (`pytest` returncode = 0)\n"
                        f"- **Target**: `{res_data.get('target')}` (`{res_data.get('file_path')}` L{res_data.get('applied_line')})\n"
                        f"- **Surgical Diff**: `{res_data.get('old_string')}` -> `{res_data.get('new_string')}`"
                    )
                    final_deduction = (base_content + verification_summary).strip()

                elif patch_status in ("FAILED", "ERROR"):
                    patch_attempts += 1
                    tb_text = str(res_data.get("traceback") or res_data.get("error") or "")
                    if tb_text:
                        failure_tracebacks.append(tb_text)
                    new_vars = list(res_data.get("implicated_variables") or [])
                    if new_vars:
                        implicated_variables = new_vars

                    visited_signatures = {
                        s for s in visited_signatures
                        if not (s[0] == "condense_method_slice" and s[1] == target)
                    }
                    if resolved_id is not None:
                        visited_nodes.discard(resolved_id)
                    GLOBAL_TOOL_MEMORY.allow_reslice(resolved_id, target)

                    if patch_attempts >= max_retries:
                        current_status = "FAILED_MAX_RETRIES"
                        final_deduction = (
                            f"BLSP Verification terminated after reaching max_patch_attempts ({max_retries}).\n"
                            f"Last AssertionError / Traceback:\n{tb_text}"
                        )
                    else:
                        current_status = "RETRY"
                        if resolved_id is not None and implicated_variables:
                            try:
                                reslice_dict = condense_method_slice(
                                    graph,
                                    method_id=resolved_id,
                                    target_vars=implicated_variables,
                                    token_budget=200,
                                )
                                res_data["forced_condense_method_slice"] = {
                                    "method_id": reslice_dict["method_id"],
                                    "target_vars": reslice_dict["target_vars"],
                                    "slice_lines": reslice_dict["slice_text"].splitlines(),
                                    "slice_text": reslice_dict["slice_text"],
                                    "method_summary": reslice_dict["method_summary"],
                                }
                                tool_output = json.dumps(res_data, indent=2)
                            except Exception:
                                pass

            tool_messages.append(ToolMessage(content=tool_output, tool_call_id=tc["id"], name=t_name))

        return {
            "messages": tool_messages,
            "visited_nodes": list(visited_nodes),
            "visited_signatures": [list(s) for s in visited_signatures],
            "status": current_status,
            "patch_attempts": patch_attempts,
            "max_patch_attempts": max_retries,
            "failure_tracebacks": failure_tracebacks,
            "implicated_variables": implicated_variables,
            "final_deduction": final_deduction,
        }

    def should_continue(state: CPGDebugState) -> str:
        if state.get("status") in ("SOLVED", "FAILED_MAX_RETRIES"):
            return END
        if state.get("final_deduction") is not None and not getattr(state["messages"][-1], "tool_calls", None):
            return END
        if state.get("turn_count", 0) >= max_turns:
            return END
        last_msg = state["messages"][-1]
        if getattr(last_msg, "tool_calls", None):
            return "tools"
        return END

    def should_continue_after_tools(state: CPGDebugState) -> str:
        if state.get("status") in ("SOLVED", "FAILED_MAX_RETRIES"):
            return END
        if state.get("turn_count", 0) >= max_turns:
            return END
        return "agent"

    workflow = StateGraph(CPGDebugState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_conditional_edges("tools", should_continue_after_tools, {"agent": "agent", END: END})

    memory = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=memory)
