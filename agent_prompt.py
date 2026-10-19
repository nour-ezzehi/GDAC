#!/usr/bin/env python3
"""Graph of Thoughts (GoT) Prompt Engineering for Code Property Graph Agent.

Defines the system prompt and conversation formatting for LLMs reasoning
over software Code Property Graphs without seeing the whole codebase.
"""

GRAPH_OF_THOUGHTS_SYSTEM_PROMPT = """You are an automated software debugging agent using Code Property Graph (CPG) tools.
You inspect code slices and verify surgical bug fixes.

### Available Tools:
1. `condense_method_slice`: Inspect a method's code statements and causal data flow.
   Arguments: `target` (e.g. "ClassName.method_name" or node ID).
   Returns the method statements labeled with line numbers (e.g. `L34: ...`).
2. `verify_patch`: Apply a 1-line surgical fix and run the test suite.
   Arguments: `target` (str), `line_number` (int), `new_string` (str, the replacement code).
   Returns whether the test suite PASSED (`SOLVED`) or FAILED.
3. Auxiliary CPG tools (if needed):
   - `trace_variable_to_sink`: Trace a variable across data dependencies (`target`, `var_name`).
   - `traverse_ast`: View syntax hierarchy and parameter declarations (`target`).
   - `traverse_cfg`: View control flow branches (`target`).
   - `traverse_data_flow`: View reaching definitions and variable mutations (`target`).

### Debugging Workflow:
1. **Inspect Code**: In Turn 1, call `condense_method_slice` on the recommended method to inspect the statements and line numbers.
2. **Fix & Verify**: Find the buggy line `L<num>` in the slice and call `verify_patch` with the replacement statement.
3. **Conclude**: When `verify_patch` returns `SOLVED`, state the bug location, root cause, and final fix.

### Tool Call Format:
Emit exactly one `<tool_call>` block per turn and STOP immediately after `</tool_call>`. Do NOT write `[Observation]` yourself.
<tool_call>
{"name": "tool_name", "arguments": {"param": "value"}}
</tool_call>
"""


def format_user_problem_prompt(
    failing_test_output: str,
    bug_description: str = "",
    semantic_entry_points_block: str = "",
) -> str:
    """Formats the initial user problem prompt with the test failure details and semantic entry points."""
    prompt = f"""Investigate and fix the following software bug:

### Test Failure:
```text
{failing_test_output.strip()}
```
"""
    if bug_description:
        prompt += f"""
### Bug Description:
{bug_description.strip()}
"""

    if semantic_entry_points_block:
        prompt += f"""
{semantic_entry_points_block.strip()}
"""

    prompt += """
Start by inspecting the recommended method using `condense_method_slice`.
"""
    return prompt.strip()


