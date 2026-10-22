"""Verifiable Reinforcement Learning (GRPO) for CPG Debugging Models."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from gdac.agents.models import parse_tool_calls_from_text


def compute_rl_reward(completion_text: str, test_runner_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None) -> float:
    """Verifiable Reward Function for Reinforcement Learning (GRPO / PPO).

    Evaluates:
      1. Syntax & Schema: Balanced JSON, no unclosed quotes or hallucinated observations.
      2. Non-Trivial Patch: Rejects no-op submissions where new_string == old_string.
      3. Deterministic Ground Truth: Executes pytest via test_runner_fn.
    """
    reward = 0.0
    calls = parse_tool_calls_from_text(completion_text)

    # 1. Syntax Reward
    if not calls:
        return -1.0  # Failed to produce valid tool call
    reward += 0.3

    call = calls[0]
    tool_name = call.get("name")
    args = call.get("args") or {}

    # 2. Action Quality Reward
    if tool_name == "condense_method_slice":
        if args.get("target"):
            reward += 0.5

    elif tool_name == "calculate_expression":
        if args.get("expression"):
            reward += 0.5

    elif tool_name == "verify_patch":
        new_string = str(args.get("new_string") or "")
        old_string = str(args.get("old_string") or "")

        # Heavily penalize verbatim copy-paste of the buggy line
        if old_string and new_string.strip() == old_string.strip():
            return -1.0

        if not new_string:
            return -0.5
        reward += 0.5

        # 3. Execution Reward (Ground truth unit test verification)
        if test_runner_fn is not None:
            try:
                res = test_runner_fn(args)
                if res.get("status") == "SOLVED" and res.get("returncode") == 0:
                    reward += 2.0  # Maximum reward for verified patch!
                else:
                    reward -= 0.5
            except Exception:
                reward -= 0.5

    return reward
