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
    SemanticSearchInput,
    create_cpg_langchain_tools,
)
from agent_langchain import (
    CPGDebugState,
    MockGoTChatModel,
    create_cpg_langgraph_agent,
    run_langgraph_agent_loop,
)
from cpg_bridge import load_cpg_json


@pytest.fixture(scope="module")
def sample_graph() -> nx.MultiDiGraph:
    """Loads the test CPG graph from cpg_json/."""
    cpg_path = Path(__file__).parent.parent / "cpg_json"
    if not cpg_path.exists():
        pytest.skip(f"CPG directory not found at {cpg_path}")
    return load_cpg_json(cpg_path)


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

        search_in = SemanticSearchInput(query="discount error")
        assert search_in.top_k == 3

    def test_tools_creation(self, sample_graph):
        """Verifies that all 5 structured tools are properly instantiated."""
        tools = create_cpg_langchain_tools(sample_graph)
        assert len(tools) == 5
        tool_names = {t.name for t in tools}
        assert "traverse_ast" in tool_names
        assert "traverse_cfg" in tool_names
        assert "traverse_data_flow" in tool_names
        assert "trace_variable_to_sink" in tool_names
        assert "semantic_code_search" in tool_names

    def test_traverse_ast_tool(self, sample_graph):
        """Executes traverse_ast tool with Pydantic validation."""
        tools = {t.name: t for t in create_cpg_langchain_tools(sample_graph)}
        res_str = tools["traverse_ast"].invoke({"target": "InventoryManager.apply_discount"})
        data = json.loads(res_str)
        assert "ast_nodes" in data
        assert data["total_nodes_found"] > 0

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

    def test_semantic_code_search_tool(self, sample_graph):
        """Executes dynamic semantic code search tool."""
        tools = {t.name: t for t in create_cpg_langchain_tools(sample_graph)}
        res_str = tools["semantic_code_search"].invoke({"query": "percentage discount price", "top_k": 2})
        data = json.loads(res_str)
        assert "top_matches" in data
        assert len(data["top_matches"]) > 0


class TestLangGraphAgent:
    """Verifies LangGraph StateGraph, state tracking, and anti-looping guards."""

    def test_state_graph_discount_simulation(self, sample_graph):
        """Tests complete 3-turn state transition on discount bug."""
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

    def test_state_graph_summary_simulation(self, sample_graph):
        """Tests complete 3-turn state transition on summary bug."""
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
                        {"name": "traverse_ast", "args": {"target": "apply_discount"}, "id": call_id_2, "type": "tool_call"},
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
