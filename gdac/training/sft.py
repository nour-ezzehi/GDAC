"""Supervised Fine-Tuning (SFT) with LoRA for CPG Debugging Models."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List


def create_training_dataset() -> List[Dict[str, Any]]:
    """Builds gold-standard CPG debugging trajectories for SFT and RL warm-up."""
    trajectories = [
        # Scenario 1: Discount Calculation Bug (apply_discount)
        {
            "prompt": (
                "Investigate and fix the following software bug:\n\n"
                "### Test Failure:\n"
                "```text\n"
                "FAILED tests/test_inventory.py::TestInventoryManager::test_apply_discount\n"
                "AssertionError: 20.0 != 80.0\n"
                "Applying a 20% discount on $100 item resulted in $20.00 instead of $80.00.\n"
                "```\n\n"
                "### Recommended Semantic Entry Points:\n"
                "1. `InventoryManager.apply_discount` (Node #107374182415)\n\n"
                "Start by inspecting the recommended method using `condense_method_slice`."
            ),
            "completion": (
                "```tool_call\n"
                '{"name": "condense_method_slice", "arguments": {"target": "InventoryManager.apply_discount"}}\n'
                "```"
            ),
        },
        {
            "prompt": (
                "### Condensed Method Slice (`InventoryManager.apply_discount`):\n"
                "```text\n"
                "[METHOD] apply_discount(self, sku, discount_percent)\n"
                "  L81: IF (!(0.0 <= discount_percent && discount_percent <= 100.0))\n"
                "  L82: raise ValueError(\"Discount percentage must be between 0 and 100.\")\n"
                "  L83: item = self.get_item(sku)\n"
                "  L84: item.price = round(item.price * discount_percent / 100.0, 2)\n"
                "  L85: self.history.append(Transaction(sku = sku, action = 'DISCOUNT'))\n"
                "  L94: return item\n"
                "```\n\n"
                "First write 1-2 sentences of **[Synthesis & Deduction]** identifying the exact buggy statement "
                "line `L<num>` and the corrected statement, then emit `<tool_call>...</tool_call>` for `verify_patch`."
            ),
            "completion": (
                "The bug is on statement `L84: item.price = round(item.price * discount_percent / 100.0, 2)`. "
                "The code computes only the discount amount ($20.00) instead of reducing the price to $80.00 ($100 * (1 - 20/100)).\n\n"
                "```tool_call\n"
                '{"name": "verify_patch", "arguments": {"target": "InventoryManager.apply_discount", "line_number": 84, "new_string": "item.price = round(item.price * (1.0 - discount_percent / 100.0), 2)"}}\n'
                "```"
            ),
        },
        # Scenario 2: Aggregation Defect (generate_category_summary)
        {
            "prompt": (
                "Investigate and fix the following software bug:\n\n"
                "### Test Failure:\n"
                "```text\n"
                "FAILED tests/test_reports.py::TestInventoryReporter::test_generate_category_summary_aggregates_multiple_items\n"
                "AssertionError: 250.0 != 1000.0\n"
                "Category summary only shows the last product value ($250) instead of aggregate total ($1000).\n"
                "```\n\n"
                "### Recommended Semantic Entry Points:\n"
                "1. `InventoryReporter.generate_category_summary` (Node #107374182442)\n\n"
                "Start by inspecting the recommended method using `condense_method_slice`."
            ),
            "completion": (
                "```tool_call\n"
                '{"name": "condense_method_slice", "arguments": {"target": "InventoryReporter.generate_category_summary"}}\n'
                "```"
            ),
        },
        {
            "prompt": (
                "### Condensed Method Slice (`InventoryReporter.generate_category_summary`):\n"
                "```text\n"
                "[METHOD] generate_category_summary(self)\n"
                "  L31: summary = {}\n"
                "  L32: LOOP (item : (self.manager.list_all_items())\n"
                "  L33: item_total = item.price * item.quantity\n"
                "  L34: summary[item.category] = round(item_total, 2)\n"
                "  L35: return summary\n"
                "```\n\n"
                "First write 1-2 sentences of **[Synthesis & Deduction]** identifying the exact buggy statement "
                "line `L<num>` and the corrected statement, then emit `<tool_call>...</tool_call>` for `verify_patch`."
            ),
            "completion": (
                "The bug is on statement `L34: summary[item.category] = round(item_total, 2)`. "
                "The loop assignment overwrites previous category totals with only the last item's value ($250) "
                "instead of accumulating all items to $1000.\n\n"
                "```tool_call\n"
                '{"name": "verify_patch", "arguments": {"target": "InventoryReporter.generate_category_summary", "line_number": 34, "new_string": "summary[item.category] = round(summary.get(item.category, 0.0) + item_total, 2)"}}\n'
                "```"
            ),
        },
    ]
    return trajectories


def run_sft_training(
    model_id: str = "google/gemma-2-2b-it",
    output_dir: str = "./gemma_cpg_lora",
    epochs: int = 3,
):
    """Runs Supervised Fine-Tuning (SFT) using Hugging Face TRL & PEFT LoRA."""
    print(f"=== Starting SFT Training for {model_id} ===")
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError:
        print("Please install TRL and PEFT: pip install trl peft datasets accelerate bitsandbytes")
        return

    trajectories = create_training_dataset()
    dataset = Dataset.from_list(trajectories)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model_id,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="completion",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"=== SFT Training complete! Adapter saved to {output_dir} ===")
