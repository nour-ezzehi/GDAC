"""GDAC Services: CPG parsing, graph condensation, GraphRAG semantic search, and test runner."""

from gdac.services.cpg_parser import (
    find_nodes,
    get_node_label_summary,
    load_cpg_json,
    trace_variable_to_sink,
    traverse_ast,
    traverse_cfg,
    traverse_data_flow,
)
from gdac.services.cpg_condenser import (
    compare_cpg_optimization,
    condense_cpg,
    condense_method_slice,
    resolve_method_id,
)
from gdac.services.cpg_rag import (
    CPGSemanticIndex,
    find_semantic_entry_points,
    format_entry_points_markdown,
)
from gdac.services.test_runner import (
    get_project_root,
    resolve_test_cmd,
    run_test_cmd,
)

__all__ = [
    "load_cpg_json",
    "find_nodes",
    "get_node_label_summary",
    "traverse_ast",
    "traverse_cfg",
    "traverse_data_flow",
    "trace_variable_to_sink",
    "condense_cpg",
    "condense_method_slice",
    "resolve_method_id",
    "compare_cpg_optimization",
    "CPGSemanticIndex",
    "find_semantic_entry_points",
    "format_entry_points_markdown",
    "get_project_root",
    "resolve_test_cmd",
    "run_test_cmd",
]
