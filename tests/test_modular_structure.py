"""Tests for GDAC modular package architecture (agents, tools, services, prompts, training)."""

from pathlib import Path
import pytest
import networkx as nx

# 1. Test Modular Imports
import gdac
from gdac.agents import (
    CPGDebugState,
    LocalHuggingFaceChatModel,
    MockGoTChatModel,
    create_cpg_langgraph_agent,
    create_cpg_multiagent_system,
    run_langgraph_agent_loop,
)
from gdac.tools import (
    CalculateExpressionInput,
    CondenseMethodSliceInput,
    SemanticSearchInput,
    TraceVariableToSinkInput,
    TraverseASTInput,
    TraverseCFGInput,
    TraverseDataFlowInput,
    VerifyPatchInput,
    calculate_expression,
    create_cpg_langchain_tools,
    dispatch_tool,
    reset_tool_memory,
    verify_patch,
)
from gdac.services import (
    CPGSemanticIndex,
    compare_cpg_optimization,
    condense_cpg,
    condense_method_slice,
    find_nodes,
    find_semantic_entry_points,
    format_entry_points_markdown,
    get_project_root,
    load_cpg_json,
    resolve_method_id,
    resolve_test_cmd,
)
from gdac.prompts import (
    GRAPH_OF_THOUGHTS_SYSTEM_PROMPT,
    format_user_problem_prompt,
)
from gdac.training import (
    compute_rl_reward,
    create_training_dataset,
)



@pytest.fixture(scope="module")
def sample_graph() -> nx.MultiDiGraph:
    cpg_path = Path(__file__).parent.parent / "cpg_json"
    if not cpg_path.exists():
        pytest.skip("CPG JSON directory missing")
    return condense_cpg(load_cpg_json(cpg_path))


def test_package_structure():
    """Verifies that gdac exports all modular subpackages."""
    assert hasattr(gdac, "agents")
    assert hasattr(gdac, "tools")
    assert hasattr(gdac, "services")
    assert hasattr(gdac, "prompts")
    assert hasattr(gdac, "training")


def test_modular_services(sample_graph: nx.MultiDiGraph):
    """Verifies service functions work correctly when imported from gdac.services."""
    assert sample_graph.number_of_nodes() > 0
    methods = find_nodes(sample_graph, label="METHOD")
    assert len(methods) > 0

    root = get_project_root()
    assert (root / "cpg_json").is_dir()

    slice_res = condense_method_slice(
        sample_graph,
        method_id="apply_discount",
        target_vars=["item.price"],
        token_budget=200,
    )
    assert slice_res["method_id"] is not None
    assert "apply_discount" in slice_res["slice_text"]
    assert "item.price" in slice_res["method_summary"]["writes"]


def test_modular_tools(sample_graph: nx.MultiDiGraph):
    """Verifies tool functions and schemas work correctly from gdac.tools."""
    calc_res = calculate_expression("round(100.0 * (1 - 20.0 / 100.0), 2)")
    assert calc_res["result"] == 80.0

    tools = create_cpg_langchain_tools(sample_graph)
    tool_names = {t.name for t in tools}
    assert "calculate_expression" in tool_names
    assert "condense_method_slice" in tool_names
    assert "verify_patch" in tool_names


def test_modular_agents(sample_graph: nx.MultiDiGraph):
    """Verifies running the multi-agent system using modular imports."""
    mock_model = MockGoTChatModel(problem_type="discount")
    tools = create_cpg_langchain_tools(sample_graph)
    app = create_cpg_multiagent_system(
        mock_model,
        tools,
        sample_graph,
        max_turns=6,
        max_patch_attempts=3,
    )
    assert app is not None

    result = run_langgraph_agent_loop(
        model=mock_model,
        graph=sample_graph,
        problem_prompt="Test failure in apply_discount",
        max_turns=6,
        use_multiagent=True,
    )
    assert "SOLVED" in result


def test_modular_training():
    """Verifies dataset creation and RL reward function from gdac.training."""
    trajectories = create_training_dataset()
    assert len(trajectories) >= 2

    good_patch = '```tool_call\n{"name": "verify_patch", "arguments": {"target": "apply_discount", "line_number": 84, "new_string": "item.price = round(item.price * (1 - discount_percent / 100.0), 2)"}}\n```'
    reward_good = compute_rl_reward(good_patch)
    assert reward_good > 0.0

    noop_patch = '```tool_call\n{"name": "verify_patch", "arguments": {"target": "apply_discount", "line_number": 84, "old_string": "foo", "new_string": "foo"}}\n```'
    reward_noop = compute_rl_reward(noop_patch)
    assert reward_noop < 0.0
