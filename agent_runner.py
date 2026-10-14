#!/usr/bin/env python3
"""Autonomous Graph of Thoughts Debugging Agent.

Loads a 4-bit/8-bit quantized LLM (e.g. Qwen2.5-Coder-7B-Instruct or Llama-3-8B-Instruct)
in Google Colab / local environment, binds CPG graph traversal tools, and executes
the Graph-of-Thoughts reasoning loop to find and diagnose bugs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_prompt import GRAPH_OF_THOUGHTS_SYSTEM_PROMPT, format_user_problem_prompt
from agent_tools import TOOLS_SCHEMA, dispatch_tool
from cpg_bridge import load_cpg_json


# ==============================================================================
# 1. Model Deployment (4-bit & 8-bit Quantization)
# ==============================================================================

def load_quantized_model(
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    quantization: str = "4bit",
    device_map: str = "auto",
) -> Tuple[Any, Any]:
    """Loads a state-of-the-art coding model with 4-bit or 8-bit quantization.

    Optimized for Google Colab (T4 / A100 GPU) to fit comfortably in VRAM.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        print(f"[-] Missing required ML packages: {exc}", file=sys.stderr)
        print("    In Colab, install with: pip install torch transformers bitsandbytes accelerate", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading model '{model_name}' with {quantization} quantization...")

    bnb_config = None
    if quantization == "4bit":
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    print(f"[+] Model '{model_name}' successfully loaded in {quantization} mode.")
    return model, tokenizer


# ==============================================================================
# 2. Tool Call Parsing Helpers
# ==============================================================================

def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extracts tool calls from model output supporting Qwen2.5 and standard formats."""
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
                tool_calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            pass

    # Pattern B: ```json\n{"name": ..., "parameters": ...}\n```
    if not tool_calls:
        block_pattern = re.compile(r"```(?:json)?\s*(\{\s*\"(?:tool|name)\":.*?)\s*```", re.DOTALL)
        for m in block_pattern.finditer(text):
            try:
                data = json.loads(m.group(1))
                name = data.get("name") or data.get("tool")
                args = data.get("arguments") or data.get("parameters", {})
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
    """Executes the multi-turn Graph of Thoughts debugging agent loop."""
    import torch

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": GRAPH_OF_THOUGHTS_SYSTEM_PROMPT},
        {"role": "user", "content": problem_prompt},
    ]

    print("\n" + "=" * 60)
    print("  Beginning Graph of Thoughts Autonomous Agent Loop")
    print("=" * 60)

    for turn in range(1, max_turns + 1):
        print(f"\n--- [Turn {turn}/{max_turns}] Generating reasoning & actions ---")

        # Format input using ChatML with tools
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tools=TOOLS_SCHEMA,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

        # Clean display
        clean_response = response_text.replace("<|im_end|>", "").strip()
        print(f"[Model Output]:\n{clean_response}\n")

        messages.append({"role": "assistant", "content": clean_response})

        # Detect tool calls
        tool_calls = extract_tool_calls(clean_response)

        if not tool_calls:
            print("[+] Model finished reasoning and provided final deduction.")
            return clean_response

        # Execute all tool calls against the Code Property Graph
        for tc in tool_calls:
            t_name = tc["name"]
            t_args = tc.get("arguments", {})
            print(f"[*] Executing Graph Tool: `{t_name}` with args: {t_args}")

            result = dispatch_tool(graph, t_name, t_args)
            result_str = json.dumps(result, indent=2)

            # Summarize output for terminal
            line_count = len(result_str.splitlines())
            print(f"    -> Returned {line_count} lines of graph properties.")

            messages.append({
                "role": "tool",
                "name": t_name,
                "content": result_str,
            })

    return messages[-1]["content"]


# ==============================================================================
# 4. Mock / Dry-Run Mode (Run Locally Without GPU)
# ==============================================================================

def run_mock_agent(graph: Any, problem_type: str = "discount") -> None:
    """Demonstrates the complete Graph of Thoughts loop locally without needing a GPU."""
    print("=" * 65)
    print("  Graph of Thoughts Agent (Dry-Run Simulation)")
    print("=" * 65)

    if problem_type == "discount":
        target_method = "apply_discount"
        target_var = "discount_percent"
        failing_test = (
            "FAILED tests/test_inventory.py::test_apply_discount\n"
            "AssertionError: 20.0 != 80.0\n"
            "Notes: Applying a 20% discount on $100 resulted in $20.00 instead of $80.00."
        )
    else:
        target_method = "generate_category_summary"
        target_var = "summary"
        failing_test = (
            "FAILED tests/test_reports.py::test_generate_category_summary_aggregates_multiple_items\n"
            "AssertionError: 250.0 != 1000.0\n"
            "Notes: Total category value only shows the last item's value ($250) instead of sum ($1000)."
        )

    print(f"\n[Problem]:\n{failing_test}\n")

    # Step 1: Hypothesis
    print("--- [GoT Node 1: Hypothesis Formulation] ---")
    print(f"Agent Hypothesis: The failure points directly to `{target_method}`. Suspected variable: `{target_var}`.")

    # Step 2: AST Traversal
    print(f"\n--- [GoT Node 2: Syntactic Discovery (AST)] ---")
    ast_res = dispatch_tool(graph, "traverse_ast", {"method_name": target_method, "max_depth": 3})
    print(f"Tool Call: traverse_ast(method_name='{target_method}')")
    print(f"Observations: Found {ast_res.get('total_nodes_found')} AST nodes in `{target_method}`:")
    for n in ast_res.get("ast_nodes", [])[:6]:
        indent = "  " * n["depth"]
        print(f"  {indent}- [{n['label']}] L{n.get('line')}: {n.get('code')}")

    # Step 3: CFG Traversal
    print(f"\n--- [GoT Node 3: Execution Path Tracing (CFG)] ---")
    cfg_res = dispatch_tool(graph, "traverse_cfg", {"method_name": target_method, "max_paths": 2})
    print(f"Tool Call: traverse_cfg(method_name='{target_method}')")
    print(f"Observations: Traced {cfg_res.get('paths_count')} execution path(s):")
    for p in cfg_res.get("execution_paths", []):
        print(f"  Execution Path #{p['path_index']} ({p['length']} steps):")
        for s in p["steps"][:4]:
            print(f"    -> {s}")

    # Step 4: Data Flow Traversal
    print(f"\n--- [GoT Node 4: Data Flow & Taint Tracking] ---")
    df_res = dispatch_tool(graph, "traverse_data_flow", {"method_name": target_method, "var_name": target_var})
    print(f"Tool Call: traverse_data_flow(method_name='{target_method}', var_name='{target_var}')")
    print(f"Observations: Found {df_res.get('traces_found')} data flow propagation trace(s):")
    for t in df_res.get("data_flow_traces", []):
        for s in t["steps"][:5]:
            print(f"    ~> {s}")

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
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-Coder-7B-Instruct)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="4bit",
        choices=["4bit", "8bit", "none"],
        help="Quantization level (default: 4bit)",
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
    graph = load_cpg_json(cpg_path)
    print(f"[+] Loaded Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

    if args.dry_run or "--dry-run" in sys.argv:
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

    run_agent_loop(model, tokenizer, graph, sample_problem)


if __name__ == "__main__":
    main()
