"""GDAC Tools: Structured Pydantic schemas, LangChain tool wrappers, and execution dispatcher."""

from gdac.tools.schemas import (
    CalculateExpressionInput,
    CondenseMethodSliceInput,
    SemanticSearchInput,
    TraceVariableToSinkInput,
    TraverseASTInput,
    TraverseCFGInput,
    TraverseDataFlowInput,
    VerifyPatchInput,
)
from gdac.tools.math_tools import calculate_expression
from gdac.tools.patch_tools import verify_patch
from gdac.tools.search_tools import semantic_code_search
from gdac.tools.cpg_tools import (
    materialize_source_window,
    resolve_method_node,
    resolve_method_source_file,
    resolve_target_node,
)
from gdac.tools.registry import (
    GLOBAL_TOOL_MEMORY,
    TOOLS_SCHEMA,
    ToolMemoryBuffer,
    create_cpg_langchain_tools,
    dispatch_tool,
    reset_tool_memory,
)

__all__ = [
    "TraverseASTInput",
    "TraverseCFGInput",
    "TraverseDataFlowInput",
    "TraceVariableToSinkInput",
    "CondenseMethodSliceInput",
    "VerifyPatchInput",
    "SemanticSearchInput",
    "CalculateExpressionInput",
    "calculate_expression",
    "verify_patch",
    "semantic_code_search",
    "resolve_target_node",
    "resolve_method_node",
    "resolve_method_source_file",
    "materialize_source_window",
    "TOOLS_SCHEMA",
    "ToolMemoryBuffer",
    "GLOBAL_TOOL_MEMORY",
    "reset_tool_memory",
    "dispatch_tool",
    "create_cpg_langchain_tools",
]
