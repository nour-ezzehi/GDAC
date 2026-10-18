#!/usr/bin/env python3
"""Autonomous Graph of Thoughts Debugging Agent.

Loads a 4-bit/8-bit quantized LLM (e.g. Qwen2.5-Coder-7B-Instruct or Llama-3-8B-Instruct)
in Google Colab / local environment, binds CPG graph traversal tools, and executes
the Graph-of-Thoughts reasoning loop to find and diagnose bugs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from agent_prompt import GRAPH_OF_THOUGHTS_SYSTEM_PROMPT, format_user_problem_prompt
from agent_tools import TOOLS_SCHEMA, dispatch_tool, reset_tool_memory
from cpg_bridge import load_cpg_json
from cpg_condenser import condense_cpg, condense_method_slice
from cpg_rag import find_semantic_entry_points, format_entry_points_markdown

try:
    from agent_langchain import (
        LocalHuggingFaceChatModel,
        MockGoTChatModel,
        parse_tool_calls_from_text,
        run_langgraph_agent_loop,
        strip_hallucinated_observation,
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# ==============================================================================
# 1. Model Deployment (4-bit & 8-bit Quantization)
# ==============================================================================

def load_quantized_model(
    model_name: str = "google/gemma-2-2b-it",
    quantization: str = "4bit",
    device_map: str = "auto",
) -> Tuple[Any, Any]:
    """Loads a lightweight coding/instruction model (default: Gemma) with 4-bit or 8-bit quantization.

    Optimized for Google Colab (T4 / A100 GPU) to fit comfortably in VRAM.
    Automatically frees stale CUDA cache and falls back to ungated `unsloth/`
    mirrors if a `google/gemma-*` repo is gated without an active HF token.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        print(f"[-] Missing required ML packages: {exc}", file=sys.stderr)
        print("    In Colab, install with: pip install torch transformers bitsandbytes accelerate", file=sys.stderr)
        sys.exit(1)

    # Free any stale tensors / cached blocks before allocating a new model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[*] Loading model '{model_name}' with {quantization} quantization...")

    # Use float16 on T4 GPUs (compute capability < 8.0), bfloat16 only on Ampere+ (A100/L4)
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
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            print(f"[+] Model '{cand_name}' successfully loaded in {quantization} mode.")
            return model, tokenizer
        except Exception as exc:
            last_err = exc
            if cand_name != candidate_names[-1]:
                print(
                    f"[!] Could not load '{cand_name}' ({exc}). "
                    f"Falling back to ungated mirror '{candidate_names[-1]}'..."
                )

    raise RuntimeError(f"Failed to load model '{model_name}': {last_err}") from last_err


# ==============================================================================
# 2. Tool Call Parsing Helpers
# ==============================================================================

def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extracts tool calls from model output supporting Gemma, Qwen2.5, and standard formats."""
    if LANGCHAIN_AVAILABLE:
        parsed = parse_tool_calls_from_text(text)
        return [{"name": tc["name"], "arguments": tc.get("args", {})} for tc in parsed]

    tool_calls = []
    qwen_pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    for m in qwen_pattern.finditer(text):
        try:
            data = json.loads(m.group(1))
            name = data.get("name")
            args = data.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                tool_calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            pass
    return tool_calls


# ==============================================================================
# 3. Agent Execution Loop
# ==============================================================================

def run_agent_loop(
    model: Any,
    tokenizer: Any,
    graph: Any,
    problem_prompt: str,
    max_turns: int = 6,
) -> str:
    """Executes the multi-turn Graph of Thoughts debugging agent loop (compatible with Gemma & Qwen)."""
    if LANGCHAIN_AVAILABLE:
        chat_model = LocalHuggingFaceChatModel(model=model, tokenizer=tokenizer)
        return run_langgraph_agent_loop(chat_model, graph, problem_prompt, max_turns=max_turns)

    import torch

    # Reset stateful tool execution memory buffer
    reset_tool_memory()

    # Automatically index CPG and inject top semantic entry points if not already present
    if "Recommended Semantic Entry Points" not in problem_prompt:
        entry_points = find_semantic_entry_points(graph, problem_prompt, top_k=2)
        ep_block = format_entry_points_markdown(entry_points)
        if ep_block:
            problem_prompt = f"{problem_prompt}\n\n{ep_block}"

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": f"{GRAPH_OF_THOUGHTS_SYSTEM_PROMPT}\n\n---\n\n{problem_prompt}"},
    ]

    print("\n" + "=" * 60)
    print("  Beginning Graph of Thoughts Autonomous Agent Loop")
    print("=" * 60)

    for turn in range(1, max_turns + 1):
        print(f"\n--- [Turn {turn}/{max_turns}] Generating reasoning & actions ---")

        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=384,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][prompt_len:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        del inputs, outputs, new_tokens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        clean_response = (
            response_text.replace("<|im_end|>", "")
            .replace("<end_of_turn>", "")
            .replace("<eos>", "")
            .strip()
        )
        if LANGCHAIN_AVAILABLE:
            clean_response = strip_hallucinated_observation(clean_response)
        print(f"[Model Output]:\n{clean_response}\n")

        messages.append({"role": "assistant", "content": clean_response})

        tool_calls = extract_tool_calls(clean_response)

        if not tool_calls:
            print("[+] Model finished reasoning and provided final deduction.")
            return clean_response

        obs_chunks = []
        for tc in tool_calls:
            t_name = tc["name"]
            t_args = tc.get("arguments", {})
            print(f"[*] Executing Graph Tool: `{t_name}` with args: {t_args}")

            result = dispatch_tool(graph, t_name, t_args)
            result_str = json.dumps(result, indent=2)

            line_count = len(result_str.splitlines())
            print(f"    -> Returned {line_count} lines of graph properties.")
            obs_chunks.append(f"### [Observation from `{t_name}`]:\n```json\n{result_str}\n```")

        messages.append({
            "role": "user",
            "content": "\n\n".join(obs_chunks),
        })

    return messages[-1]["content"]


# ==============================================================================
# 4. Mock / Dry-Run Mode (Run Locally Without GPU)
# ==============================================================================

def run_mock_agent(graph: Any, problem_type: str = "discount") -> None:
    """Demonstrates the complete Graph of Thoughts loop locally without needing a GPU."""
    reset_tool_memory()

    print("=" * 65)
    print("  Graph of Thoughts Agent (Dry-Run Simulation with Upgrades)")
    print("  - Macro-Tools (Algorithmic Offloading via NetworkX)")
    print("  - Hybrid GraphRAG Semantic Entry Points")
    print("  - Stateful Anti-Looping Memory Buffer")
    print("=" * 65)

    if problem_type == "discount":
        target_method = "apply_discount"
        target_var = "discount_percent"
        failing_test = (
            "FAILED tests/test_inventory.py::TestInventoryManager::test_apply_discount\n"
            "AssertionError: 20.0 != 80.0\n"
            "Notes: Applying a 20% discount on $100 resulted in $20.00 instead of $80.00."
        )
    else:
        target_method = "generate_category_summary"
        target_var = "item_total"
        failing_test = (
            "FAILED tests/test_reports.py::TestInventoryReporter::test_generate_category_summary_aggregates_multiple_items\n"
            "AssertionError: 250.0 != 1000.0\n"
            "Notes: Total category value only shows the last item's value ($250) instead of sum ($1000)."
        )

    print(f"\n[Problem]:\n{failing_test}\n")

    # Step 0: Hybrid GraphRAG Entry Points
    print("--- [GoT Node 0: Semantic Entry Point Discovery (Hybrid GraphRAG)] ---")
    entry_points = find_semantic_entry_points(graph, failing_test, top_k=2)
    print(format_entry_points_markdown(entry_points))

    # Step 1: Hypothesis
    print("\n--- [GoT Node 1: Hypothesis Formulation] ---")
    print(f"Agent Hypothesis: Based on semantic entry points, focusing on `{target_method}`. Suspected variable: `{target_var}`.")

    # Step 2: AST Traversal
    print(f"\n--- [GoT Node 2: Syntactic Discovery (AST)] ---")
    ast_res = dispatch_tool(graph, "traverse_ast", {"method_name": target_method, "max_depth": 3})
    print(f"Tool Call: traverse_ast(method_name='{target_method}')")
    print(f"Observations: Found {ast_res.get('total_nodes_found')} AST nodes in `{target_method}`:")
    for n in ast_res.get("ast_nodes", [])[:6]:
        indent = "  " * n["depth"]
        print(f"  {indent}- [{n['label']}] L{n.get('line')}: {n.get('code')}")

    # Step 3: Anti-Looping Interception Test
    print(f"\n--- [GoT Node 3: Anti-Looping Deterministic Memory Test] ---")
    print(f"Attempting duplicate call: traverse_ast(method_name='{target_method}')")
    dup_res = dispatch_tool(graph, "traverse_ast", {"method_name": target_method, "max_depth": 3})
    print(f"Interceptor Response: {dup_res}")

    # Step 4: Program Slicing & Macro-Tool Algorithmic Offloading
    print(f"\n--- [GoT Node 4: Program Slicing (`condense_method_slice`) & Macro-Tool (`trace_variable_to_sink`)] ---")
    slice_res = dispatch_tool(
        graph, "condense_method_slice", {"target": target_method, "target_vars": target_var, "token_budget": 120}
    )
    print(f"Tool Call: condense_method_slice(target='{target_method}', target_vars='{target_var}')")
    print(f"[Data-Flow Summary]: {slice_res.get('method_summary')}")
    print(f"[Condensed Slice ({slice_res.get('token_stats', {}).get('used_tokens')} tokens)]:\n{slice_res.get('slice_text')}\n")

    macro_res = dispatch_tool(
        graph, "trace_variable_to_sink", {"method_name": target_method, "var_name": target_var}
    )
    print(f"Tool Call: trace_variable_to_sink(method_name='{target_method}', var_name='{target_var}')")
    print(f"[Algorithmic NetworkX Summary]:\n  {macro_res.get('summary')}\n")
    print(f"Origin: {macro_res.get('origin')}")
    print(f"Terminal Sink: {macro_res.get('terminal_sink')}")
    print("Propagation Path:")
    for step in macro_res.get("propagation_path", []):
        line_str = f"L{step.get('line')}: " if step.get("line") else ""
        print(f"    ~> [{step.get('label')}] {line_str}{step.get('code')}")

    # Step 5: Final Deduction & Patch
    print(f"\n--- [GoT Node 5: Synthesis & Root Cause Deduction] ---")
    if problem_type == "discount":
        print("""[Agent Deduction]:
1. Root Cause:
   In `apply_discount`, the mathematical calculation is:
     item.price = round(item.price * (discount_percent / 100.0), 2)
   This computes the DISCOUNT AMOUNT ($20), rather than subtracting it from the original price ($80).

2. Exact Fix:
   - File: stackflow/inventory.py
   - Method: InventoryManager.apply_discount
   - Replace:
       item.price = round(item.price * (discount_percent / 100.0), 2)
     With:
       item.price = round(item.price * (1.0 - (discount_percent / 100.0)), 2)
""")
    else:
        print("""[Agent Deduction]:
1. Root Cause:
   In `generate_category_summary`, the dictionary assignment inside the loop is:
     summary[item.category] = round(item_total, 2)
   This overwrites previous values for the category instead of accumulating the running total.

2. Exact Fix:
   - File: stackflow/reports.py
   - Method: InventoryReporter.generate_category_summary
   - Replace:
       summary[item.category] = round(item_total, 2)
     With:
       summary[item.category] = round(summary.get(item.category, 0.0) + item_total, 2)
""")
    print("=" * 65)


# ==============================================================================
# 5. CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Graph of Thoughts debugging agent using quantized LLMs & CPG tools."
    )
    parser.add_argument(
        "--cpg-path",
        type=str,
        default="cpg_json",
        help="Path to Joern CPG export directory (default: cpg_json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b-it",
        help="HuggingFace model ID (default: google/gemma-2-2b-it)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="4bit",
        choices=["4bit", "8bit", "none"],
        help="Quantization level (default: 4bit)",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="langchain" if LANGCHAIN_AVAILABLE else "native",
        choices=["langchain", "native"],
        help="Agent framework: 'langchain' (LangGraph stateful agent) or 'native' (original raw loop).",
    )
    parser.add_argument(
        "--no-condense",
        action="store_true",
        help="Disable CPG statement-level condensation and use the raw sub-expression CPG.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run local simulation without loading PyTorch LLM weights (useful for verifying flow)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="discount",
        choices=["discount", "summary"],
        help="Bug scenario to test in dry-run mode (discount or summary)",
    )

    args = parser.parse_args()

    cpg_path = Path(args.cpg_path)
    if not cpg_path.exists():
        print(f"[-] CPG path '{cpg_path}' does not exist. Run joern_extraction.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading CPG graph from '{cpg_path}'...")
    raw_graph = load_cpg_json(cpg_path)
    print(f"[+] Loaded Raw Graph: {raw_graph.number_of_nodes()} nodes, {raw_graph.number_of_edges()} edges.")

    if args.no_condense:
        graph = raw_graph
    else:
        graph = condense_cpg(raw_graph)
        print(
            f"[+] Condensed CPG (Statement-Level): {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges."
        )

    if args.dry_run or "--dry-run" in sys.argv:
        if args.framework == "langchain" and LANGCHAIN_AVAILABLE:
            failing_test = (
                "FAILED tests/test_inventory.py::TestInventoryManager::test_apply_discount\n"
                "AssertionError: 20.0 != 80.0\n"
                "Notes: Applying a 20% discount on $100 resulted in $20.00 instead of $80.00."
                if args.target == "discount"
                else (
                    "FAILED tests/test_reports.py::TestInventoryReporter::test_generate_category_summary_aggregates_multiple_items\n"
                    "AssertionError: 250.0 != 1000.0\n"
                    "Notes: Total category value only shows the last item's value ($250) instead of sum ($1000)."
                )
            )
            mock_llm = MockGoTChatModel(problem_type=args.target)
            run_langgraph_agent_loop(mock_llm, graph, failing_test, max_turns=4)
        else:
            run_mock_agent(graph, problem_type=args.target)
        return

    # Real LLM execution (for Colab / GPU environment)
    model, tokenizer = load_quantized_model(
        model_name=args.model,
        quantization=args.quantization,
    )

    sample_problem = format_user_problem_prompt(
        failing_test_output=(
            "FAILED tests/test_inventory.py::TestInventoryManager::test_apply_discount\n"
            "AssertionError: 20.0 != 80.0\n"
            "Applying 20% discount on $100 item resulted in $20.00 instead of $80.00."
        ),
        bug_description="The inventory discount operation fails its unit test.",
    )

    if args.framework == "langchain":
        if not LANGCHAIN_AVAILABLE:
            print("[-] LangChain is not installed. Install via: pip install langchain langgraph pydantic", file=sys.stderr)
            sys.exit(1)
        langchain_chat = LocalHuggingFaceChatModel(model=model, tokenizer=tokenizer)
        run_langgraph_agent_loop(langchain_chat, graph, sample_problem)
    else:
        run_agent_loop(model, tokenizer, graph, sample_problem)


if __name__ == "__main__":
    main()
