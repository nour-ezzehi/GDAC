#!/usr/bin/env python3
"""Automated Unit Tests for LangChain & LangGraph CPG Autonomous Debugging Agent."""

import json
from pathlib import Path
import pytest
import networkx as nx

from agent_langchain_tools import (
    TraverseASTInput,
    TraverseCFGInput,
    TraverseDataFlowInput,
    TraceVariableToSinkInput,
    CondenseMethodSliceInput,
    VerifyPatchInput,
    SemanticSearchInput,
    create_cpg_langchain_tools,
)
from agent_langchain import (
    CPGDebugState,
    MockGoTChatModel,
    create_cpg_langgraph_agent,
    run_langgraph_agent_loop,
)
from agent_tools import reset_tool_memory
from cpg_bridge import load_cpg_json
from cpg_condenser import condense_cpg


@pytest.fixture(autouse=True)
def _clear_memory():
    """Clears global tool memory before each test."""
    reset_tool_memory()


@pytest.fixture(scope="module")
def sample_graph() -> nx.MultiDiGraph:
    """Loads and condenses the test CPG graph from cpg_json/."""
    cpg_path = Path(__file__).parent.parent / "cpg_json"
    if not cpg_path.exists():
        pytest.skip(f"CPG directory not found at {cpg_path}")
    return condense_cpg(load_cpg_json(cpg_path))


class TestLangChainTools:
    """Verifies Pydantic schema validation and tool execution."""

    def test_pydantic_schemas(self):
        """Tests that Pydantic input schemas enforce types and defaults."""
        ast_in = TraverseASTInput(target="InventoryManager.apply_discount")
        assert ast_in.target == "InventoryManager.apply_discount"
        assert ast_in.max_depth == 4
        assert ast_in.filter_labels is None

        sink_in = TraceVariableToSinkInput(target="apply_discount", var_name="discount_percent")
        assert sink_in.var_name == "discount_percent"

        slice_in = CondenseMethodSliceInput(target="apply_discount", target_vars=["discount_percent"])
        assert slice_in.target == "apply_discount"
        assert slice_in.token_budget == 200

        patch_in = VerifyPatchInput(
            target="InventoryManager.apply_discount",
            line_number=84,
            old_string="item.price = round(item.price * (discount_percent / 100.0), 2)",
            new_string="item.price = round(item.price * (1 - discount_percent / 100.0), 2)",
        )
        assert patch_in.line_number == 84

        search_in = SemanticSearchInput(query="discount error")
        assert search_in.top_k == 3

    def test_tools_creation(self, sample_graph):
        """Verifies that all 7 structured tools are properly instantiated."""
        tools = create_cpg_langchain_tools(sample_graph)
        assert len(tools) == 7
        tool_names = {t.name for t in tools}
        assert "traverse_ast" in tool_names
        assert "traverse_cfg" in tool_names
        assert "traverse_data_flow" in tool_names
        assert "trace_variable_to_sink" in tool_names
        assert "condense_method_slice" in tool_names
        assert "verify_patch" in tool_names
        assert "semantic_code_search" in tool_names

    def test_traverse_ast_tool(self, sample_graph):
        """Executes traverse_ast tool with Pydantic validation."""
        tools = {t.name: t for t in create_cpg_langchain_tools(sample_graph)}
        res_str = tools["traverse_ast"].invoke({"target": "InventoryManager.apply_discount"})
        data = json.loads(res_str)
        assert "ast_nodes" in data
        assert data["total_nodes_found"] > 0
        assert "data_flow_summary" in data

    def test_trace_variable_to_sink_tool(self, sample_graph):
        """Executes macro-tool trace_variable_to_sink."""
        tools = {t.name: t for t in create_cpg_langchain_tools(sample_graph)}
        res_str = tools["trace_variable_to_sink"].invoke({
            "target": "InventoryManager.apply_discount",
            "var_name": "discount_percent",
        })
        data = json.loads(res_str)
        assert "propagation_path" in data
        assert len(data["propagation_path"]) >= 1
        assert "summary" in data

    def test_condense_method_slice_tool(self, sample_graph):
        """Executes condense_method_slice tool with Pydantic validation."""
        tools = {t.name: t for t in create_cpg_langchain_tools(sample_graph)}
        res_str = tools["condense_method_slice"].invoke({
            "target": "InventoryManager.apply_discount",
            "target_vars": ["discount_percent", "new_price"],
            "token_budget": 200,
        })
        data = json.loads(res_str)
        assert "slice_text" in data
        assert "method_summary" in data
        assert "token_stats" in data
        assert len(data["slice_text"]) > 0

    def test_verify_patch_tool_solved_and_failed(self, sample_graph):
        """Tests BLSP 15-line context materialization, surgical diff, pytest run, and rollback."""
        tools = {t.name: t for t in create_cpg_langchain_tools(sample_graph)}
        inv_file = Path(__file__).parent.parent / "stackflow" / "stackflow" / "inventory.py"
        original_disk = inv_file.read_text(encoding="utf-8")

        # 1. Failing surgical patch -> returncode != 0, parses AssertionError & implicated_variables
        fail_str = tools["verify_patch"].invoke({
            "target": "InventoryManager.apply_discount",
            "line_number": 84,
            "old_string": "item.price = round(item.price * discount_percent / 100.0, 2)",
            "new_string": "item.price = round(item.price - discount_percent / 100.0, 2)",
        })
        fail_data = json.loads(fail_str)
        assert fail_data["status"] == "FAILED"
        assert fail_data["returncode"] != 0
        assert "AssertionError" in fail_data["traceback"] or "FAILED" in fail_data["traceback"]
        assert len(fail_data["implicated_variables"]) > 0
        assert len(fail_data["materialized_window"].splitlines()) == 15
        # Verify file was rolled back cleanly
        assert inv_file.read_text(encoding="utf-8") == original_disk

        # 2. Passing surgical patch (even with CPG-normalized old_string!) -> returncode == 0, status == SOLVED
        pass_str = tools["verify_patch"].invoke({
            "target": "InventoryManager.apply_discount",
            "line_number": 84,
            "old_string": "item.price = round(item.price * discount_percent / 100.0, 2)",
            "new_string": "item.price = round(item.price * (1 - discount_percent / 100.0), 2)",
        })
        pass_data = json.loads(pass_str)
        assert pass_data["status"] == "SOLVED"
        assert pass_data["returncode"] == 0
        assert len(pass_data["materialized_window"].splitlines()) == 15
        # Verify file remains clean after verification when keep_patch=False
        assert inv_file.read_text(encoding="utf-8") == original_disk

    def test_semantic_code_search_tool(self, sample_graph):
        """Executes dynamic semantic code search tool."""
        tools = {t.name: t for t in create_cpg_langchain_tools(sample_graph)}
        res_str = tools["semantic_code_search"].invoke({"query": "percentage discount price", "top_k": 2})
        data = json.loads(res_str)
        assert "top_matches" in data
        assert len(data["top_matches"]) > 0


class TestLangGraphAgent:
    """Verifies LangGraph StateGraph, BLSP feedback routing, and anti-looping guards."""

    def test_state_graph_discount_simulation(self, sample_graph):
        """Tests complete BLSP state transition on discount bug terminating with status='SOLVED'."""
        mock_llm = MockGoTChatModel(problem_type="discount")
        deduction = run_langgraph_agent_loop(
            mock_llm,
            sample_graph,
            "FAILED test_apply_discount: 20.0 != 80.0",
            max_turns=4,
            thread_id="test_discount_session",
        )
        assert "apply_discount" in deduction
        assert "round(item.price * (1 - discount_percent / 100.0), 2)" in deduction
        assert "SOLVED" in deduction

    def test_state_graph_summary_simulation(self, sample_graph):
        """Tests complete BLSP state transition on summary bug terminating with status='SOLVED'."""
        mock_llm = MockGoTChatModel(problem_type="summary")
        deduction = run_langgraph_agent_loop(
            mock_llm,
            sample_graph,
            "FAILED test_generate_category_summary: 250.0 != 1000.0",
            max_turns=4,
            thread_id="test_summary_session",
        )
        assert "generate_category_summary" in deduction
        assert "summary.get(category, 0.0) + item_total" in deduction
        assert "SOLVED" in deduction

    def test_blsp_retry_loop_to_solved(self, sample_graph):
        """Verifies failed patch -> traceback parsing + condense_method_slice feedback loop -> SOLVED."""
        mock_llm = MockGoTChatModel(problem_type="discount_retry")
        tools = create_cpg_langchain_tools(sample_graph)
        app = create_cpg_langgraph_agent(
            mock_llm, tools, sample_graph, max_turns=6, max_patch_attempts=3
        )
        config = {"configurable": {"thread_id": "test_blsp_retry"}}
        final_state = app.invoke(
            {
                "messages": [],
                "visited_nodes": set(),
                "visited_signatures": set(),
                "entry_points": [],
                "final_deduction": None,
                "turn_count": 0,
                "status": "IN_PROGRESS",
                "patch_attempts": 0,
                "max_patch_attempts": 3,
                "failure_tracebacks": [],
                "implicated_variables": [],
            },
            config=config,
        )
        assert final_state["status"] == "SOLVED"
        assert final_state["patch_attempts"] == 2
        assert len(final_state["failure_tracebacks"]) == 1
        assert len(final_state["implicated_variables"]) >= 1
        assert "SOLVED" in (final_state["final_deduction"] or "")

    def test_blsp_max_retry_cap_termination(self, sample_graph):
        """Verifies that exceeding max_patch_attempts terminates the graph with FAILED_MAX_RETRIES."""
        mock_llm = MockGoTChatModel(problem_type="discount_retry")
        tools = create_cpg_langchain_tools(sample_graph)
        # Set max_patch_attempts=1 so the first failed patch immediately triggers the hard retry cap
        app = create_cpg_langgraph_agent(
            mock_llm, tools, sample_graph, max_turns=6, max_patch_attempts=1
        )
        config = {"configurable": {"thread_id": "test_blsp_cap"}}
        final_state = app.invoke(
            {
                "messages": [],
                "visited_nodes": set(),
                "visited_signatures": set(),
                "entry_points": [],
                "final_deduction": None,
                "turn_count": 0,
                "status": "IN_PROGRESS",
                "patch_attempts": 0,
                "max_patch_attempts": 1,
                "failure_tracebacks": [],
                "implicated_variables": [],
            },
            config=config,
        )
        assert final_state["status"] == "FAILED_MAX_RETRIES"
        assert final_state["patch_attempts"] == 1
        assert "max_patch_attempts" in (final_state["final_deduction"] or "")

    def test_anti_looping_state_interception(self, sample_graph):
        """Verifies that LangGraph tool_node intercepts repeated tool calls."""
        from langchain_core.messages import AIMessage
        tools = create_cpg_langchain_tools(sample_graph)
        app = create_cpg_langgraph_agent(MockGoTChatModel(), tools, sample_graph)

        # Create state with duplicate tool call
        call_id_1 = "call_abc1"
        call_id_2 = "call_abc2"
        state = {
            "messages": [
                AIMessage(
                    content="Testing looping",
                    tool_calls=[
                        {"name": "traverse_ast", "args": {"target": "apply_discount"}, "id": call_id_1, "type": "tool_call"},
                        {"name": "traverse_ast", "args": {"target": "apply_discount"}, "id": call_id_2, "id": call_id_2, "type": "tool_call"},
                    ],
                )
            ],
            "visited_nodes": set(),
            "visited_signatures": set(),
            "entry_points": [],
            "final_deduction": None,
            "turn_count": 1,
        }

        # Invoke tool node directly through step
        result = app.invoke(state, config={"configurable": {"thread_id": "test_loop"}})
        # Second tool message should contain the interception error
        tool_msgs = [m for m in result["messages"] if hasattr(m, "tool_call_id")]
        assert len(tool_msgs) >= 2
        assert "System Error: You have already inspected this node" in tool_msgs[1].content

    def test_gemma_hallucinated_observation_stripping_and_action_parsing(self):
        """Verifies that Gemma outputs with natural-language [Action] and hallucinated [Observation] are cleaned and parsed."""
        from agent_langchain import parse_tool_calls_from_text, strip_hallucinated_observation

        raw_gemma_output = (
            "Okay, let's begin.\n\n"
            "**[Hypothesis]**: The `apply_discount` method is likely the source of the incorrect discount calculation.\n\n"
            "**[Action]**: `traverse_ast` on `InventoryManager.apply_discount` (Node #107374182415, Line 76).\n\n"
            "**[Observation]**:\n"
            "```json\n"
            "{\"method_name\": \"InventoryManager.apply_discount\", \"ast\": []}\n"
            "```\n\n"
            "**[Action]**: `traverse_cfg`"
        )
        cleaned = strip_hallucinated_observation(raw_gemma_output)
        assert "**[Observation]**" not in cleaned
        assert "discounted_price" not in cleaned

        calls = parse_tool_calls_from_text(raw_gemma_output)
        assert len(calls) == 1
        assert calls[0]["name"] == "traverse_ast"
        assert calls[0]["args"]["target"] == "InventoryManager.apply_discount"

    def test_scenario2_entry_point_excludes_tests_and_condense_default_vars(self, sample_graph):
        """Verifies that pytest tracebacks rank production methods #1 (excluding tests) and condense_method_slice works without target_vars."""
        from agent_tools import reset_tool_memory
        from cpg_rag import find_semantic_entry_points
        reset_tool_memory()

        scenario2_query = (
            "A pytest suite is failing with the following error:\n"
            "FAILED tests/test_reports.py::TestInventoryReporter::test_generate_category_summary_aggregates_multiple_items\n"
            "AssertionError: Expected 'Electronics' total value to be 2950.0 (2000 + 500 + 450), "
            "but `generate_category_summary()` returned 450.0 (only the last item's value!)."
        )
        entry_points = find_semantic_entry_points(sample_graph, scenario2_query, top_k=3)
        assert len(entry_points) >= 1
        assert entry_points[0]["qualified_name"] == "InventoryReporter.generate_category_summary"
        assert all(not ep["qualified_name"].startswith("Test") for ep in entry_points)

        tools = create_cpg_langchain_tools(sample_graph)
        tools_by_name = {t.name: t for t in tools}
        # Invoke condense_method_slice with method_id and WITHOUT target_vars (as Gemma did in Turn 3)
        raw_out = tools_by_name["condense_method_slice"].invoke(
            {"target": "InventoryReporter.generate_category_summary", "method_id": entry_points[0]["node_id"]}
        )
        parsed = json.loads(raw_out)
        assert "error" not in parsed
        assert "slice_text" in parsed
        assert "method_summary" in parsed
        assert "generate_category_summary" in parsed["slice_text"]

