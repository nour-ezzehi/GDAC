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
from agent_tools import GLOBAL_TOOL_MEMORY, reset_tool_memory, resolve_target_node
from cpg_condenser import condense_method_slice
from cpg_rag import find_semantic_entry_points, format_entry_points_markdown


# ==============================================================================
# 1. LangGraph State Definition
# ==============================================================================

class CPGDebugState(TypedDict, total=False):
    """The central state of the Graph-of-Thoughts debugging agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    visited_nodes: Sequence[int]
    visited_signatures: Sequence[Tuple[str, str, str]]
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

_KNOWN_CPG_TOOLS = (
    "condense_method_slice",
    "verify_patch",
    "trace_variable_to_sink",
    "traverse_ast",
    "traverse_cfg",
    "traverse_data_flow",
    "semantic_code_search",
)


from langchain_core.utils.json import parse_json_markdown, parse_partial_json


def strip_hallucinated_observation(text: str) -> str:
    """Truncates assistant output right after `</tool_call>` or before any self-generated `[Observation]` block."""
    if "</tool_call>" in text:
        idx = text.rfind("</tool_call>") + len("</tool_call>")
        return text[:idx].strip()

    for marker in ("**[Observation]**", "### [Observation", "[Observation]:", "**[Observation]:"):
        if marker in text:
            return text[: text.index(marker)].strip()
    return text.strip()


def _try_parse_json_dict(raw_str: str) -> Optional[Dict[str, Any]]:
    """Uses LangChain's partial JSON parser to parse tool-call dicts without brittle regexes."""
    s = raw_str.strip()
    if "{" not in s:
        return None
    s = s[s.find("{") :]
    # Strip any trailing xml or markdown tags/fences
    s = re.sub(r"</?(?:tool_call|xml)>.*$", "", s, flags=re.DOTALL).strip()
    s = re.sub(r"```.*$", "", s, flags=re.DOTALL).strip()

    # If a small model omitted the closing quote before the final `}}`, trim trailing braces so parse_partial_json closes both
    if s.count('"') % 2 == 1:
        s = s.rstrip("} \t\r\n")
    try:
        parsed = parse_json_markdown(s, parser=parse_partial_json)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def parse_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """Extracts tool calls from assistant output using LangChain's partial JSON parser."""
    cleaned = strip_hallucinated_observation(text)
    tool_calls: List[Dict[str, Any]] = []

    # Extract candidate chunks from <tool_call>...</tool_call> and ```tool_call ... ```
    chunks: List[str] = []
    for m in re.finditer(r"<tool_call>(.*?)(?:</tool_call>|$)", cleaned, re.DOTALL):
        chunks.append(m.group(1).strip())
    for m in re.finditer(r"```(?:tool_call|json)?\s*\n(.*?)(?:```|$)", cleaned, re.DOTALL):
        chunks.append(m.group(1).strip())

    if not chunks:
        chunks = (
            cleaned.split("tool_call>")[1:]
            if "tool_call>" in cleaned
            else [cleaned]
        )

    seen_calls = set()
    for chunk in chunks:
        data = _try_parse_json_dict(chunk)
        if isinstance(data, dict):
            name = data.get("name") or data.get("tool")
            args = data.get("arguments") or data.get("args") or data.get("parameters") or {}
            if isinstance(args, str):
                args = _try_parse_json_dict(args) or {}
            if name in _KNOWN_CPG_TOOLS:
                call_sig = (name, json.dumps(args, sort_keys=True))
                if call_sig not in seen_calls:
                    seen_calls.add(call_sig)
                    tool_calls.append({
                        "name": name,
                        "args": args if isinstance(args, dict) else {},
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "tool_call",
                    })

    # Fallback for natural-language `[Action]: <tool_name> on <target>` lines
    if not tool_calls and "Action]" in cleaned:
        action_tail = cleaned.split("Action]")[-1]
        for tname in _KNOWN_CPG_TOOLS:
            if tname in action_tail:
                tokens = [
                    tok.strip("`(),.:;\"'")
                    for tok in action_tail.split()
                    if "." in tok and not tok.startswith("http")
                ]
                if tokens:
                    tool_calls.append({
                        "name": tname,
                        "args": {"target": tokens[0]},
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "tool_call",
                    })
                break

    return tool_calls



class LocalHuggingFaceChatModel(BaseChatModel):
    """Wraps a PyTorch/Transformers causal or multimodal language model (Gemma, Qwen, Llama) with LangChain's ChatModel interface."""

    model: Any
    tokenizer: Optional[Any] = None
    processor: Optional[Any] = None
    max_new_tokens: int = 384
    temperature: float = 0.1
    top_p: float = 0.9
    bound_tools: List[Dict[str, Any]] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.tokenizer is None and self.processor is not None:
            self.tokenizer = self.processor

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
            processor=self.processor,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            bound_tools=formatted_tools,
        )

    def _format_tools_prompt_block(self) -> str:
        """Creates a compact tool schema block for models whose tokenizer template does not inject `tools` (e.g., Gemma)."""
        if not self.bound_tools:
            return ""
        lines = [
            "### AVAILABLE GRAPH TOOLS (Invoke using `<tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>`):"
        ]
        for t in self.bound_tools:
            fn = t.get("function", t)
            name = fn.get("name", "")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            props = list(params.get("properties", {}).keys()) if isinstance(params, dict) else list(params.keys())
            lines.append(f"- `{name}({', '.join(props)})`: {desc}")
        return "\n".join(lines)

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Converts LangChain messages into a prompt compatible with both Gemma (strict user/model alternation) and ChatML."""
        raw_msgs: List[Dict[str, str]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                raw_msgs.append({"role": "system", "content": str(m.content)})
            elif isinstance(m, HumanMessage):
                raw_msgs.append({"role": "user", "content": str(m.content)})
            elif isinstance(m, AIMessage):
                content = str(m.content or "")
                tcs = getattr(m, "tool_calls", None) or []
                if tcs and "<tool_call>" not in content:
                    tc_blocks = [
                        f"<tool_call>\n{json.dumps({'name': tc['name'], 'arguments': tc.get('args', {})})}\n</tool_call>"
                        for tc in tcs
                    ]
                    content = (content + "\n" + "\n".join(tc_blocks)).strip()
                raw_msgs.append({"role": "assistant", "content": content})
            elif isinstance(m, ToolMessage):
                t_name = getattr(m, "name", "tool")
                raw_msgs.append({
                    "role": "tool",
                    "name": t_name,
                    "content": str(m.content),
                })

        tok_obj = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        template_str = str(
            getattr(self.tokenizer, "chat_template", "")
            or getattr(tok_obj, "chat_template", "")
            or ""
        )
        tok_name = str(
            getattr(self.tokenizer, "name_or_path", "")
            or getattr(tok_obj, "name_or_path", "")
            or ""
        ).lower()
        is_gemma = "gemma" in tok_name or "<start_of_turn>" in template_str

        # Path 1: Native tool-aware ChatML template (e.g., Qwen2.5)
        if not is_gemma and "tools" in template_str and "system" in template_str:
            try:
                return self.tokenizer.apply_chat_template(
                    raw_msgs,
                    tools=self.bound_tools if self.bound_tools else None,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                pass

        # Path 2: Gemma / Generic strict alternating user/assistant turns with explicit tool-schema injection
        tools_block = self._format_tools_prompt_block()
        system_parts: List[str] = []
        alternating_msgs: List[Dict[str, str]] = []

        for m in raw_msgs:
            role = m["role"]
            content = m["content"]
            if role == "system":
                system_parts.append(content)
                continue
            elif role == "tool":
                t_name = m.get("name", "tool")
                parsed_tool = _try_parse_json_dict(content) or {}
                extra_code_block = ""
                follow_up = (
                    "First write 1-2 sentences of **[Synthesis & Deduction]** identifying the exact buggy statement "
                    "line `L<num>` inside the method body (do NOT use the `[METHOD]` header line!) and the corrected "
                    "statement, then emit `<tool_call>...</tool_call>` for `verify_patch`."
                )
                if t_name == "condense_method_slice" and parsed_tool.get("slice_text"):
                    extra_code_block = (
                        f"### Condensed Method Slice (`{parsed_tool.get('target', '')}`):\n"
                        f"```text\n{parsed_tool['slice_text']}\n```\n\n"
                    )
                elif t_name == "verify_patch" and parsed_tool.get("status") != "SOLVED":
                    mat_win = parsed_tool.get("materialized_window") or ""
                    err_reason = parsed_tool.get("error") or "test assertion failed"
                    if mat_win:
                        extra_code_block = (
                            f"### Materialized Source Window (`{parsed_tool.get('file_path', '')}`):\n"
                            f"```text\n{mat_win}\n```\n\n"
                        )
                    follow_up = (
                        f"The previous `verify_patch` failed ({err_reason}). "
                        f"Inspect the numbered lines above, write 1-2 sentences of **[Synthesis & Deduction]**, "
                        f"and emit a corrected `<tool_call>...</tool_call>` for `verify_patch` with the exact buggy "
                        f"`line_number`, `old_string`, and a DIFFERENT `new_string`."
                    )

                user_chunk = (
                    f"{extra_code_block}"
                    f"### [Observation from `{t_name}`]:\n```json\n{content}\n```\n\n"
                    f"{follow_up}"
                )
                role = "user"
                content = user_chunk

            if system_parts and role == "user":
                prefix = "\n\n".join(system_parts)
                if tools_block:
                    prefix = f"{prefix}\n\n{tools_block}"
                content = f"{prefix}\n\n---\n\n{content}"
                system_parts = []
                tools_block = ""

            if alternating_msgs and alternating_msgs[-1]["role"] == role:
                alternating_msgs[-1]["content"] += f"\n\n{content}"
            else:
                alternating_msgs.append({"role": role, "content": content})

        if not alternating_msgs and system_parts:
            prefix = "\n\n".join(system_parts)
            if tools_block:
                prefix = f"{prefix}\n\n{tools_block}"
            alternating_msgs.append({"role": "user", "content": prefix})

        try:
            return self.tokenizer.apply_chat_template(
                alternating_msgs,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            try:
                multimodal_msgs = [
                    {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                    for m in alternating_msgs
                ]
                return self.tokenizer.apply_chat_template(
                    multimodal_msgs,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                if is_gemma:
                    rendered = []
                    for m in alternating_msgs:
                        g_role = "model" if m["role"] == "assistant" else "user"
                        rendered.append(f"<start_of_turn>{g_role}\n{m['content']}<end_of_turn>")
                    rendered.append("<start_of_turn>model\n")
                    return "\n".join(rendered)
                rendered = []
                for m in raw_msgs:
                    rendered.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
                rendered.append("<|im_start|>assistant\n")
                return "\n".join(rendered)

    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extracts tool calls from assistant output."""
        return parse_tool_calls_from_text(text)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import gc
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        prompt_text = self._convert_messages_to_prompt(messages)
        try:
            inputs = self.tokenizer(prompt_text, return_tensors="pt")
        except TypeError:
            inputs = self.tokenizer(text=prompt_text, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        if hasattr(inputs, "input_ids"):
            prompt_len = inputs.input_ids.shape[1]
        elif isinstance(inputs, dict) and "input_ids" in inputs:
            prompt_len = inputs["input_ids"].shape[-1]
        else:
            prompt_len = len(inputs[0])

        tok_obj = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(tok_obj, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None) or getattr(tok_obj, "eos_token_id", None)

        class _ToolCallStoppingCriteria(StoppingCriteria):
            """Halts generation immediately after `</tool_call>`, closing ```, or before a hallucinated `[Observation]` block."""

            def __call__(self, input_ids: Any, scores: Any, **kw: Any) -> bool:
                gen_ids = input_ids[0][prompt_len:]
                if len(gen_ids) < 4:
                    return False
                tail_text = tok_obj.decode(gen_ids[-24:], skip_special_tokens=False)
                if "</tool_call>" in tail_text:
                    return True
                if "```" in tail_text and "tool_call" in tail_text and tail_text.rstrip().endswith("```"):
                    return True
                if "Observation]" in tail_text or "**[Observation" in tail_text:
                    return True
                return False

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": True,
            "pad_token_id": pad_id,
            "stopping_criteria": StoppingCriteriaList([_ToolCallStoppingCriteria()]),
        }
        if hasattr(self.model, "generation_config") and getattr(self.model.generation_config, "max_length", None) is not None:
            self.model.generation_config.max_length = None

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][prompt_len:]
        raw_text = tok_obj.decode(new_tokens, skip_special_tokens=False)


        # Free GPU tensors immediately to prevent multi-turn VRAM fragmentation/OOM
        del inputs, outputs, new_tokens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        clean_text = (
            raw_text.replace("<|im_end|>", "")
            .replace("<end_of_turn>", "")
            .replace("<eos>", "")
            .replace("<|eot_id|>", "")
            .strip()
        )
        clean_text = strip_hallucinated_observation(clean_text)

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

            # Resolve target node in CPG
            resolved_id = resolve_target_node(graph, target_raw)

            # If condense_method_slice is invoked without target_vars, default to the method's writes + reads
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

            # Anti-Looping Interception
            if sig in visited_signatures or (resolved_id and resolved_id in visited_nodes and not var_key):
                err_msg = json.dumps({
                    "error": "System Error: You have already inspected this node. Review your previous thoughts and explore a different execution path."
                })
                tool_messages.append(ToolMessage(content=err_msg, tool_call_id=tc["id"], name=t_name))
                continue

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

            # Only record visit in state if tool succeeded without error
            try:
                parsed_out = json.loads(tool_output)
            except json.JSONDecodeError:
                parsed_out = {}

            if "error" not in parsed_out:
                visited_signatures.add(sig)
                if resolved_id and t_name != "verify_patch":
                    visited_nodes.add(resolved_id)

            # BLSP Feedback Routing for `verify_patch`
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

                    # Clear prior condense_method_slice signatures and node memory so re-slicing is allowed
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
                        # Force an immediate condense_method_slice directly on the graph without polluting ToolMemoryBuffer
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
    reset_tool_memory()

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
        "visited_nodes": [],
        "visited_signatures": [],
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
