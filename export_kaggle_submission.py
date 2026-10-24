#!/usr/bin/env python3
"""Export Generator for Google - The Gemma 4 Developer Agent Kaggle Competition.

Packages a strictly compliant Google ADK (Agent Development Kit) submission containing:
- agent.yaml (Root ADK AgentConfig referencing sub_agents via config_path)
- eval_config.yaml (Trajectory, tool call, and timeout budgets)
- configs/sampling.yaml (Generation, temperature, and thinking budget)
- prompts/
  - system.md (Root Manager / Orchestrator prompt with SWE-bench golden rules)
  - locator.md (Locator sub-agent prompt)
  - solver.md (Solver & math synthesis prompt)
  - verifier.md (Verifier sub-agent prompt for diff auditing & regression)
- sub_agents/
  - locator.yaml (Code Graph Navigation Specialist)
  - solver.yaml (Surgical Patch & Math Specialist)
  - verifier.yaml (Regression & Diff Verifier Specialist)
- skills/
  - cpg_navigation/SKILL.md (Code Property Graph Slicing Skill)
"""

import os
from pathlib import Path
import sys
import zipfile
import argparse


def generate_submission(output_dir: Path, model_name: str = "gemma-4-31b-it-qat-w4a16-ct") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "submission.zip"
    submission_dir = output_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    manager_system_prompt = """You are the Lead Autonomous Software Engineer coordinating a Graph-of-Thoughts debugging workflow.
Your objective: reproduce the reported issue, coordinate codebase navigation with code graphs, supervise surgical patching in /workspace, verify with regression tests, and submit the patch.

### CRITICAL OPERATIONAL CONSTRAINTS (ZERO-TOLERANCE RULES):
1. **NEVER TOUCH TEST FILES**: NEVER edit, create, or modify files in `tests/`, `testing/`, `test/`, or any file matching `*test*.py`. In SWE-bench and Kaggle evaluation, touching a test file instantly invalidates your solution with 0 points. You must ONLY edit production source code in the repository package.
2. **CALL `submit_patch` IMMEDIATELY UPON PASSING**: Once `run_command` confirms `pytest <test_path>` passes and `git diff` confirms clean, minimal changes to production code, invoke `submit_patch` IMMEDIATELY. Do NOT continue exploring or running redundant commands.
3. **SURGICAL DIFF LIMIT**: Keep patches minimal (1–5 lines). Never rewrite entire functions, reformat whitespace, or touch unrelated lines.

### Available Tools:
1. Code Navigation & Graph Traversal:
   - `get_code_subgraph`: Retrieve semantic dependency slices and callers/callees.
   - `get_code_neighbors`: Trace reaching definitions, data-flow, and references.
   - `search_similar_code`: Semantic entry point lookup across functions/classes.
   - `read_file`: Read specific lines of source files.
2. Modification & Execution:
   - `edit_file` / `write_file`: Apply minimal surgical edits to production code.
   - `run_command`: Run regression tests (e.g. `pytest <test_file> -k <test_name>`) and check `git diff`.
   - `get_status`: Check git status and modified paths.
   - `submit_patch`: Finish the task when the test suite passes.

### Sub-Agent Coordination:
- Delegate file and statement localization to `locator_subagent`.
- Delegate code synthesis, formula verification, and patching to `solver_subagent`.
- Delegate regression testing, git diff validation, and patch finalization to `verifier_subagent`.

### 4-Step Test-Driven Verification Loop:
1. **Step 1 - Reproduce**: Run the specific failing test first via `run_command` (e.g. `pytest <test_path> -k <test_name>`). Inspect the exact failure, traceback, and assertion values.
2. **Step 2 - Localize**: Pinpoint the causal production method and statement using `search_similar_code`, `get_code_subgraph`, and `get_code_neighbors`. Do not dump whole files.
3. **Step 3 - Patch**:
   - Verify mathematical logic (e.g., `price * (1.0 - discount / 100.0)` for discounted prices; `summary[k] = round(summary.get(k, 0.0) + val, 2)` for accumulations).
   - Apply a 1–5 line surgical edit to the production file using `edit_file`.
4. **Step 4 - Verify & Submit**:
   - Re-run the exact test command: `pytest <test_path> -k <test_name>`.
   - Inspect `git diff` to guarantee:
     - Zero test files were modified.
     - Only the intended causal lines changed.
   - Call `submit_patch` IMMEDIATELY.
"""

    locator_prompt = """You are the Codebase Navigation Specialist.
Your single goal: pinpoint the exact physical file and line number containing the root cause of the reported issue.

### Guidelines:
1. Use `search_similar_code` to find entry points matching the test failure or error traceback.
2. Use `get_code_subgraph` and `get_code_neighbors` to inspect caller/callee relations and variable data dependencies.
3. Use `read_file` to inspect the 15 lines surrounding the suspected line.
4. TARGET PRODUCTION CODE ONLY: Never target or inspect test files (`tests/`, `*_test.py`). The bug is in the package source code.
5. Do NOT edit files. Output the exact file path and line number with 1 sentence of causal justification.
"""

    solver_prompt = """You are the Code & Math Synthesis Specialist.
Given a located bug line and the failing test assertion:

### Guidelines:
1. **Analyze Mathematical Logic**:
   - For discounts/percentages: `price * (1.0 - discount / 100.0)` gives discounted price (`price * discount / 100.0` is only the discount amount).
   - For aggregations: `summary[k] = round(summary.get(k, 0.0) + val, 2)` preserves previous loop items.
2. **Production Code Only**: NEVER edit or touch test files (`tests/`, `*_test.py`). Apply minimal surgical edits (1–5 lines) exclusively to production source code using `edit_file`.
3. **Verify with Regression Test**: Run the failing test using `run_command` (e.g. `pytest <test_path> -k <test_name>`).
4. **Immediate Submission Trigger**: Inspect `git diff` via `run_command`. If the test passes and the diff is clean, call `submit_patch` IMMEDIATELY. Do not run further commands.
"""

    verifier_prompt = """You are the Regression & Diff Verifier Specialist.
Your single goal: ensure code safety, verify test suite completion, and submit the patch.

### Verification Protocol:
1. **Audit Git Diff**: Run `run_command` with `git diff`.
   - Ensure ZERO files under `tests/`, `testing/`, or `*test*.py` have been touched.
   - Ensure the diff is minimal (1–5 lines), with no extraneous debug prints or syntax errors.
2. **Execute Regression Test**: Run `run_command` with `pytest <test_path> -k <test_name>` (or full test suite if fast).
3. **Submit Patch**: As soon as tests pass and the diff is confirmed clean, call `submit_patch` IMMEDIATELY.
"""

    cpg_skill_md = """---
name: cpg_navigation
description: Autonomous Code Property Graph (CPG) program slicing and statement-level navigation.
---

# Code Property Graph Slicing Skill

Enables the agent to inspect large codebases without exceeding token limits by computing
backward and forward program slices along Control Flow (CFG) and Data Dependency (DDG) edges.

## Slicing Principles
- **Backward Slicing**: Traces all variables affecting a failing sink expression back to definitions.
- **Forward Slicing**: Traces mutations of an input variable forward to output returns or state changes.
- **Statement-Level Condensation**: Collapses sub-expression AST nodes into complete statement lines.
"""

    agent_yaml = f"""# yaml-language-server: $schema=https://raw.githubusercontent.com/google/adk-python/refs/heads/main/src/google/adk/agents/config_schemas/AgentConfig.json
name: gdac_developer_agent
model: {model_name}
description: Autonomous Graph-of-Thoughts developer agent using Code Property Graphs and BLSP runtime verification.
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
sub_agents:
  - config_path: sub_agents/locator.yaml
  - config_path: sub_agents/solver.yaml
  - config_path: sub_agents/verifier.yaml
generate_content_config: !include configs/sampling.yaml
"""

    locator_yaml = f"""# yaml-language-server: $schema=https://raw.githubusercontent.com/google/adk-python/refs/heads/main/src/google/adk/agents/config_schemas/AgentConfig.json
name: locator_subagent
model: {model_name}
description: Navigates code graphs and data-flow dependencies to pinpoint root cause bug locations.
instruction: !include ../prompts/locator.md
tools:
  - get_code_subgraph
  - get_code_neighbors
  - search_similar_code
  - read_file
generate_content_config: !include ../configs/sampling.yaml
"""

    solver_yaml = f"""# yaml-language-server: $schema=https://raw.githubusercontent.com/google/adk-python/refs/heads/main/src/google/adk/agents/config_schemas/AgentConfig.json
name: solver_subagent
model: {model_name}
description: Performs rigorous arithmetic verification and applies minimal surgical patches.
instruction: !include ../prompts/solver.md
tools:
  - edit_file
  - write_file
  - read_file
  - run_command
  - submit_patch
generate_content_config: !include ../configs/sampling.yaml
"""

    verifier_yaml = f"""# yaml-language-server: $schema=https://raw.githubusercontent.com/google/adk-python/refs/heads/main/src/google/adk/agents/config_schemas/AgentConfig.json
name: verifier_subagent
model: {model_name}
description: Audits git diffs for test isolation and minimal footprint, verifies regression tests, and triggers submit_patch.
instruction: !include ../prompts/verifier.md
tools:
  - run_command
  - read_file
  - submit_patch
generate_content_config: !include ../configs/sampling.yaml
"""

    sampling_yaml = """temperature: 0.1
top_p: 0.95
max_output_tokens: 8192
thinking_config:
  thinking_budget: 4096
  include_thoughts: true
"""

    eval_config_yaml = """evaluation:
  timeout_seconds: 300
  verification_timeout_seconds: 300
  max_tool_calls: 40
  max_time_minutes: 10
  max_turns: 40
"""

    package_files = {
        "agent.yaml": agent_yaml,
        "eval_config.yaml": eval_config_yaml,
        "configs/sampling.yaml": sampling_yaml,
        "prompts/system.md": manager_system_prompt,
        "prompts/locator.md": locator_prompt,
        "prompts/solver.md": solver_prompt,
        "prompts/verifier.md": verifier_prompt,
        "sub_agents/locator.yaml": locator_yaml,
        "sub_agents/solver.yaml": solver_yaml,
        "sub_agents/verifier.yaml": verifier_yaml,
        "skills/cpg_navigation/SKILL.md": cpg_skill_md,
    }

    # Write uncompressed submission directory
    for rel_path, content in package_files.items():
        file_path = submission_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content.strip() if rel_path.endswith(".md") else content, encoding="utf-8")

    # Write compressed submission.zip archive
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path, content in sorted(package_files.items()):
            archive.writestr(rel_path, content.strip() if rel_path.endswith(".md") else content)

    print(f"[+] Successfully generated directory: {submission_dir}")
    print(f"[+] Successfully generated archive:   {zip_path}")
    print(f"[+] Files packaged ({len(package_files)}):")
    for k in sorted(package_files.keys()):
        print(f"    - {k}")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package GDAC agent for Kaggle Gemma 4 competition.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("GEMMA_OUTPUT_ROOT", "/kaggle/working" if Path("/kaggle/working").exists() else "."),
        help="Target directory for submission.zip (default: current directory or /kaggle/working)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-4-31b-it-qat-w4a16-ct",
        choices=[
            "gemma-4-31b-it-qat-w4a16-ct",
            "gemma-4-12b-it-qat-w4a16-ct",
            "gemma-4-e4b-it-qat-w4a16-ct",
            "gemma-4-e2b-it-qat-w4a16-ct",
        ],
        help="Gemma 4 model variant to target (default: gemma-4-31b-it-qat-w4a16-ct)",
    )
    args = parser.parse_args()
    generate_submission(Path(args.output_dir), model_name=args.model)


if __name__ == "__main__":
    main()
