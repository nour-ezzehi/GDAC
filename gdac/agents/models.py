"""ChatModel Adapters and Model Utilities for GDAC Agents."""

from __future__ import annotations

import gc
import json
import re
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
from langchain_core.utils.json import parse_json_markdown, parse_partial_json
from pydantic import ConfigDict

_KNOWN_CPG_TOOLS = (
    "condense_method_slice",
    "verify_patch",
    "calculate_expression",
    "trace_variable_to_sink",
    "traverse_ast",
    "traverse_cfg",
    "traverse_data_flow",
    "semantic_code_search",
)


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
    """Wraps a PyTorch/Transformers causal or multimodal language model with LangChain's ChatModel interface."""

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


def load_quantized_model(
    model_name: str = "google/gemma-2-2b-it",
    quantization: str = "4bit",
    device_map: str = "auto",
) -> Tuple[Any, Any]:
    """Loads a lightweight coding/instruction model with 4-bit or 8-bit quantization."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(f"Missing required ML packages: {exc}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    compute_dtype = torch.float16
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8 and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16

    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    candidate_names = [model_name]
    if model_name.startswith("google/gemma"):
        candidate_names.append(model_name.replace("google/", "unsloth/", 1))

    last_err: Optional[Exception] = None
    for cand_name in candidate_names:
        try:
            tokenizer = AutoTokenizer.from_pretrained(cand_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                cand_name,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=compute_dtype,
                trust_remote_code=True,
            )
            return model, tokenizer
        except Exception as err:
            last_err = err
            continue

    raise RuntimeError(f"Could not load model '{model_name}': {last_err}")
