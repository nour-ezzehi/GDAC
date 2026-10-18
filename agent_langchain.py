#!/usr/bin/env python3
"""LangGraph Stateful Agent Architecture for Code Property Graph (CPG) Debugging.

Implements a production-grade cyclical agent using LangGraph and LangChain:
  1. CPGDebugState: Typed state tracking messages, visited CPG nodes, and tool signatures.
  2. Anti-Looping Interception: Built directly into the StateGraph ToolNode.
  3. Model-Agnostic: Works seamlessly with local quantized HuggingFace models in Colab,
     as well as cloud providers (OpenAI, Gemini, Anthropic, Ollama).
  4. MemorySaver: Persistent checkpointer for turn-by-turn state history.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypedDict

import networkx as nx
from pydantic import BaseModel, ConfigDict

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agent_prompt import GRAPH_OF_THOUGHTS_SYSTEM_PROMPT, format_user_problem_prompt
from agent_langchain_tools import create_cpg_langchain_tools
from agent_tools import resolve_target_node
from cpg_rag import find_semantic_entry_points, format_entry_points_markdown


# ==============================================================================
# 1. LangGraph State Definition
# ==============================================================================

class CPGDebugState(TypedDict, total=False):
    """The central state of the Graph-of-Thoughts debugging agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    visited_nodes: Set[int]
    visited_signatures: Set[Tuple[str, str, str]]
    entry_points: List[Dict[str, Any]]
    final_deduction: Optional[str]
    turn_count: int
    status: Optional[str]
    patch_attempts: int
    max_patch_attempts: int
    failure_tracebacks: List[str]
    implicated_variables: List[str]


# ==============================================================================
# 2. Local Quantized HuggingFace ChatModel Adapter (Colab / T4 GPU)
# ==============================================================================

class LocalHuggingFaceChatModel(BaseChatModel):
    """Wraps a PyTorch/Transformers causal language model with LangChain's ChatModel interface."""

    model: Any
    tokenizer: Any
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    bound_tools: List[Dict[str, Any]] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "local_huggingface_chat"

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        **kwargs: Any,
    ) -> BaseChatModel:
        """Binds tool schemas to the model."""
        formatted_tools = []
        for t in tools:
            if isinstance(t, BaseTool):
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.args,
                    },
                })
            elif isinstance(t, dict):
                formatted_tools.append(t)
        return self.__class__(
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            bound_tools=formatted_tools,
        )

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Converts LangChain messages to ChatML format for tokenizer."""
        chatml_msgs = []
        for m in messages:
            if isinstance(m, SystemMessage):
                chatml_msgs.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                chatml_msgs.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                chatml_msgs.append({"role": "assistant", "content": m.content})
            elif isinstance(m, ToolMessage):
                chatml_msgs.append({
                    "role": "tool",
                    "name": getattr(m, "name", "tool"),
                    "content": m.content,
                })

        try:
            return self.tokenizer.apply_chat_template(
                chatml_msgs,
                tools=self.bound_tools if self.bound_tools else None,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            # Fallback text rendering if template fails on tools
            rendered = []
            for m in chatml_msgs:
                rendered.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
            rendered.append("<|im_start|>assistant\n")
            return "\n".join(rendered)

    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extracts JSON tool calls from assistant text."""
        tool_calls = []
        # Pattern A: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        qwen_pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
        for m in qwen_pattern.finditer(text):
            try:
                data = json.loads(m.group(1))
                name = data.get("name")
                args = data.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                if name:
                    tool_calls.append({
                        "name": name,
                        "args": args,
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "tool_call",
                    })
            except json.JSONDecodeError:
                pass

        # Pattern B: ```json\n{"name": ..., "arguments": ...}\n```
        if not tool_calls:
            block_pattern = re.compile(
                r"```(?:json)?\s*(\{\s*\"(?:tool|name)\":.*?)\s*```", re.DOTALL
            )
            for m in block_pattern.finditer(text):
                try:
                    data = json.loads(m.group(1))
                    name = data.get("name") or data.get("tool")
                    args = data.get("arguments") or data.get("parameters", {})
                    if name:
                        tool_calls.append({
                            "name": name,
                            "args": args,
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "tool_call",
                        })
                except json.JSONDecodeError:
                    pass

        return tool_calls

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import torch

        prompt_text = self._convert_messages_to_prompt(messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        clean_text = raw_text.replace("<|im_end|>", "").strip()

        tool_calls = self._extract_tool_calls(clean_text)
        message = AIMessage(content=clean_text, tool_calls=tool_calls)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])


# ==============================================================================
# 3. Mock Model for Local Dry-Run Simulations
# ==============================================================================

class MockGoTChatModel(BaseChatModel):
    """Simulates multi-turn GoT reasoning steps including BLSP verify_patch execution."""

    problem_type: str = "discount"
    call_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "mock_got_chat"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_index += 1

        if self.problem_type == "discount":
            if self.call_index == 1:
                content = (
                    "### [Hypothesis]\n"
                    "The discount calculation appears flawed in `InventoryManager.apply_discount`.\n\n"
                    "### [Action]\n"
                    "Computing the causal program slice of `InventoryManager.apply_discount` on `discount_percent` and `item.price`."
                )
                tool_calls = [{
                    "name": "condense_method_slice",
                    "args": {
                        "target": "InventoryManager.apply_discount",
                        "target_vars": ["discount_percent", "item.price"],
                    },
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }]
            elif self.call_index == 2:
                content = (
                    "### [Observation]\n"
                    "Line 84 computes `item.price = round(item.price * discount_percent / 100.0, 2)`.\n\n"
                    "### [Synthesis & Deduction]\n"
                    "The formula multiplies by `discount_percent / 100.0` ($20.0) instead of subtracting the discount (`1 - discount_percent / 100.0`, $80.0).\n\n"
                    "### [Verification / Conclusion]\n"
                    "- Bug Location: `stackflow/inventory.py` line 84 in `apply_discount`\n"
                    "- Root Cause: Formula multiplies instead of `item.price * (1 - discount_percent / 100.0)`\n"
                    "- Exact Fix:\n"
                    "```python\n"
                    "item.price = round(item.price * (1 - discount_percent / 100.0), 2)\n"
                    "```"
                )
                tool_calls = [{
                    "name": "verify_patch",
                    "args": {
                        "target": "InventoryManager.apply_discount",
                        "line_number": 84,
                        "old_string": "item.price = round(item.price * (discount_percent / 100.0), 2)",
                        "new_string": "item.price = round(item.price * (1 - discount_percent / 100.0), 2)",
                    },
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }]
            else:
                content = (
                    "### [Conclusion]\n"
                    "- Bug Location: `stackflow/inventory.py` line 84 in `apply_discount`\n"
                    "- Exact Fix:\n"
                    "```python\n"
                    "item.price = round(item.price * (1 - discount_percent / 100.0), 2)\n"
                    "```"
                )
                tool_calls = []

        elif self.problem_type == "discount_retry":
            # Simulates a failed patch attempt -> forced re-slice on implicated vars -> successful patch
            if self.call_index == 1:
                content = (
                    "### [Hypothesis]\n"
                    "Attempting an initial patch on line 84 of `InventoryManager.apply_discount`."
                )
                tool_calls = [{
                    "name": "verify_patch",
                    "args": {
                        "target": "InventoryManager.apply_discount",
                        "line_number": 84,
                        "old_string": "item.price = round(item.price * (discount_percent / 100.0), 2)",
                        "new_string": "item.price = round(item.price - (discount_percent / 100.0), 2)",
                    },
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }]
            elif self.call_index == 2:
                content = (
                    "### [Observation]\n"
                    "Previous patch failed pytest with `AssertionError: assert 99.8 == 80.0`. "
                    "Looping back to `condense_method_slice` on the newly implicated variables."
                )
                tool_calls = [{
                    "name": "condense_method_slice",
                    "args": {
                        "target": "InventoryManager.apply_discount",
                        "target_vars": ["item.price", "discount_percent"],
                    },
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }]
            else:
                content = (
                    "### [Synthesis & Deduction]\n"
                    "Re-slicing confirms percentage multiplier `1 - discount_percent / 100.0` is required.\n\n"
                    "### [Verification / Conclusion]\n"
                    "```python\n"
                    "item.price = round(item.price * (1 - discount_percent / 100.0), 2)\n"
                    "```"
                )
                tool_calls = [{
                    "name": "verify_patch",
                    "args": {
                        "target": "InventoryManager.apply_discount",
                        "line_number": 84,
                        "old_string": "item.price = round(item.price * (discount_percent / 100.0), 2)",
                        "new_string": "item.price = round(item.price * (1 - discount_percent / 100.0), 2)",
                    },
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }]

        else:
            # Summary bug scenario
            if self.call_index == 1:
                content = (
                    "### [Hypothesis]\n"
                    "Category aggregation bug in `InventoryReporter.generate_category_summary`.\n\n"
                    "### [Action]\n"
                    "Computing the causal program slice of `InventoryReporter.generate_category_summary` on `summary` and `item_total`."
                )
                tool_calls = [{
                    "name": "condense_method_slice",
                    "args": {
                        "target": "InventoryReporter.generate_category_summary",
                        "target_vars": ["summary", "item_total"],
                    },
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }]
            elif self.call_index == 2:
                content = (
                    "### [Observation]\n"
                    "Line 34 inside the loop overwrites `summary[item.category] = round(item_total, 2)`.\n\n"
                    "### [Synthesis & Deduction]\n"
                    "The dictionary assignment overwrites rather than accumulating values (`summary.get(category, 0.0) + item_total`).\n\n"
                    "### [Verification / Conclusion]\n"
                    "- Bug Location: `stackflow/reports.py` line 34 in `generate_category_summary`\n"
                    "- Root Cause: `summary[item.category] = round(item_total, 2)` overwrites instead of accumulating `summary.get(category, 0.0) + item_total`\n"
                    "- Exact Fix:\n"
                    "```python\n"
                    "summary[item.category] = round(summary.get(item.category, 0.0) + item_total, 2)\n"
                    "```"
                )
                tool_calls = [{
                    "name": "verify_patch",
                    "args": {
                        "target": "InventoryReporter.generate_category_summary",
                        "line_number": 34,
                        "old_string": "summary[item.category] = round(item_total, 2)",
                        "new_string": "summary[item.category] = round(summary.get(item.category, 0.0) + item_total, 2)",
                    },
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }]
            else:
                content = (
                    "### [Conclusion]\n"
                    "- Bug Location: `stackflow/reports.py` in `generate_category_summary`\n"
                    "- Exact Fix:\n"
                    "```python\n"
                    "summary[category] = summary.get(category, 0.0) + item_total\n"
                    "```"
                )
                tool_calls = []

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content, tool_calls=tool_calls))])


# ==============================================================================
# 4. LangGraph Agent Construction
# ==============================================================================

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
        """Calls the model with current messages and increments turn counter."""
        msgs = state.get("messages") or []
        if msgs and isinstance(msgs[-1], AIMessage) and getattr(msgs[-1], "tool_calls", None):
            return {"turn_count": state.get("turn_count", 0)}

        response = model_with_tools.invoke(msgs)
        new_turn = state.get("turn_count", 0) + 1

        final_ded = state.get("final_deduction")
        # If model produced no tool calls, it has finished reasoning
        if not getattr(response, "tool_calls", None):
            final_ded = response.content

        return {
            "messages": [response],
            "turn_count": new_turn,
            "final_deduction": final_ded,
        }

    def tool_node(state: CPGDebugState) -> Dict[str, Any]:
        """Executes tool calls with state-level anti-looping interceptors and BLSP feedback routing."""
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", []) or []

        visited_nodes = set(state.get("visited_nodes") or set())
        visited_signatures = set(state.get("visited_signatures") or set())
        patch_attempts = int(state.get("patch_attempts") or 0)
        max_retries = int(state.get("max_patch_attempts") or max_patch_attempts)
        failure_tracebacks = list(state.get("failure_tracebacks") or [])
        implicated_variables = list(state.get("implicated_variables") or [])
        current_status = state.get("status") or "IN_PROGRESS"
        final_deduction = state.get("final_deduction")
        tool_messages = []

        for tc in tool_calls:
            t_name = tc["name"]
            t_args = tc.get("args") or {}
            target = str(t_args.get("target") or t_args.get("method_name") or "").strip().lower()
            var_key = str(
                t_args.get("var_name")
                or t_args.get("target_vars")
                or t_args.get("new_string")
                or ""
            ).strip().lower()
            sig = (t_name, target, var_key)

            # Resolve target node in CPG
            resolved_id = resolve_target_node(graph, t_args.get("target") or t_args.get("method_name"))

            # Anti-Looping Interception
            if sig in visited_signatures or (resolved_id and resolved_id in visited_nodes and not var_key):
                err_msg = json.dumps({
                    "error": "System Error: You have already inspected this node. Review your previous thoughts and explore a different execution path."
                })
                tool_messages.append(ToolMessage(content=err_msg, tool_call_id=tc["id"], name=t_name))
                continue

            # Record visit in state
            visited_signatures.add(sig)
            if resolved_id and t_name != "verify_patch":
                visited_nodes.add(resolved_id)

            # Execute tool
            if t_name in tools_by_name:
                try:
                    tool_output = tools_by_name[t_name].invoke(t_args)
                    if not isinstance(tool_output, str):
                        tool_output = json.dumps(tool_output, indent=2)
                except Exception as exc:
                    tool_output = json.dumps({"error": f"Tool execution failed: {str(exc)}"})
            else:
                tool_output = json.dumps({"error": f"Tool '{t_name}' not found."})

            # BLSP Feedback Routing for `verify_patch`
            if t_name == "verify_patch":
                try:
                    res_data = json.loads(tool_output)
                except json.JSONDecodeError:
                    res_data = {}

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

                    # Clear prior condense_method_slice signatures for this target so re-slicing is allowed
                    visited_signatures = {
                        s for s in visited_signatures
                        if not (s[0] == "condense_method_slice" and s[1] == target)
                    }

                    if patch_attempts >= max_retries:
                        current_status = "FAILED_MAX_RETRIES"
                        final_deduction = (
                            f"BLSP Verification terminated after reaching max_patch_attempts ({max_retries}).\n"
                            f"Last AssertionError / Traceback:\n{tb_text}"
                        )
                    else:
                        current_status = "RETRY"
                        # Force an immediate condense_method_slice on the newly implicated variables
                        if "condense_method_slice" in tools_by_name and implicated_variables:
                            try:
                                reslice_raw = tools_by_name["condense_method_slice"].invoke({
                                    "target": t_args.get("target"),
                                    "target_vars": implicated_variables,
                                    "token_budget": 200,
                                })
                                res_data["forced_condense_method_slice"] = json.loads(reslice_raw)
                                tool_output = json.dumps(res_data, indent=2)
                            except Exception:
                                pass

            tool_messages.append(ToolMessage(content=tool_output, tool_call_id=tc["id"], name=t_name))

        return {
            "messages": tool_messages,
            "visited_nodes": visited_nodes,
            "visited_signatures": visited_signatures,
            "status": current_status,
            "patch_attempts": patch_attempts,
            "max_patch_attempts": max_retries,
            "failure_tracebacks": failure_tracebacks,
            "implicated_variables": implicated_variables,
            "final_deduction": final_deduction,
        }

    def should_continue(state: CPGDebugState) -> str:
        """Determines whether to execute tools or end the loop after agent_node."""
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
        """Routes after tool_node: terminates on SOLVED or FAILED_MAX_RETRIES, otherwise loops back to agent."""
        if state.get("status") in ("SOLVED", "FAILED_MAX_RETRIES"):
            return END
        if state.get("turn_count", 0) >= max_turns:
            return END
        return "agent"

    # Build Graph
    workflow = StateGraph(CPGDebugState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_conditional_edges("tools", should_continue_after_tools, {"agent": "agent", END: END})

    memory = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=memory)


# ==============================================================================
# 5. High-Level LangGraph Execution Loop
# ==============================================================================

def run_langgraph_agent_loop(
    model: BaseChatModel,
    graph: nx.MultiDiGraph,
    problem_prompt: str,
    max_turns: int = 6,
    max_patch_attempts: int = 3,
    thread_id: str = "got_debug_session",
) -> str:
    """Executes the autonomous debugging loop using LangGraph."""
    # Step 0: Hybrid GraphRAG entry points
    if "Recommended Semantic Entry Points" not in problem_prompt:
        entry_points = find_semantic_entry_points(graph, problem_prompt, top_k=2)
        ep_block = format_entry_points_markdown(entry_points)
        if ep_block:
            problem_prompt = f"{problem_prompt}\n\n{ep_block}"
    else:
        entry_points = []

    # Initialize tools and state graph
    tools = create_cpg_langchain_tools(graph)
    checkpointer = MemorySaver()
    app = create_cpg_langgraph_agent(
        model,
        tools,
        graph,
        max_turns=max_turns,
        max_patch_attempts=max_patch_attempts,
        checkpointer=checkpointer,
    )

    initial_messages: List[BaseMessage] = [
        SystemMessage(content=GRAPH_OF_THOUGHTS_SYSTEM_PROMPT),
        HumanMessage(content=problem_prompt),
    ]

    initial_state: CPGDebugState = {
        "messages": initial_messages,
        "visited_nodes": set(),
        "visited_signatures": set(),
        "entry_points": entry_points,
        "final_deduction": None,
        "turn_count": 0,
        "status": "IN_PROGRESS",
        "patch_attempts": 0,
        "max_patch_attempts": max_patch_attempts,
        "failure_tracebacks": [],
        "implicated_variables": [],
    }

    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 60)
    print("  Beginning LangGraph Autonomous Graph of Thoughts Loop")
    print("=" * 60)

    final_result = None
    for event in app.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "agent":
                turn = node_output.get("turn_count", 0)
                last_msg = node_output["messages"][-1]
                print(f"\n--- [Turn {turn}/{max_turns}] Generating reasoning & actions ---")
                print(f"[Model Output]:\n{last_msg.content}\n")

                if node_output.get("final_deduction"):
                    print("[+] Model finished reasoning and provided final deduction.")
                    final_result = node_output["final_deduction"]

            elif node_name == "tools":
                for tm in node_output.get("messages", []):
                    lines = len(tm.content.splitlines())
                    print(f"[*] Executed Graph Tool: `{tm.name}` -> Returned {lines} lines.")
                if node_output.get("status") == "SOLVED" and node_output.get("final_deduction"):
                    print("[+] BLSP Runtime Verification succeeded (status='SOLVED'). Terminating graph.")
                    final_result = node_output["final_deduction"]
                elif node_output.get("status") == "FAILED_MAX_RETRIES" and node_output.get("final_deduction"):
                    print("[!] Reached maximum BLSP patch retry attempts. Terminating graph.")
                    final_result = node_output["final_deduction"]

    if not final_result:
        # Extract from the latest checkpoint
        final_state = app.get_state(config).values
        if final_state.get("final_deduction"):
            final_result = final_state["final_deduction"]
        elif final_state.get("messages"):
            final_result = final_state["messages"][-1].content
        else:
            final_result = "No deduction produced."

    return final_result
