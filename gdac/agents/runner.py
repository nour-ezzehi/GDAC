"""High-Level Execution Runner for GDAC LangGraph Agents."""

from __future__ import annotations

from typing import Any, List, Optional
import networkx as nx

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from gdac.agents.state import CPGDebugState
from gdac.agents.single_agent import create_cpg_langgraph_agent
from gdac.agents.multi_agent import create_cpg_multiagent_system
from gdac.prompts.got_prompts import GRAPH_OF_THOUGHTS_SYSTEM_PROMPT
from gdac.services.cpg_rag import find_semantic_entry_points, format_entry_points_markdown
from gdac.tools.registry import create_cpg_langchain_tools, reset_tool_memory


def run_langgraph_agent_loop(
    model: BaseChatModel,
    graph: nx.MultiDiGraph,
    problem_prompt: str,
    max_turns: int = 6,
    max_patch_attempts: int = 3,
    thread_id: str = "got_debug_session",
    use_multiagent: bool = True,
) -> str:
    """Executes the autonomous debugging loop using LangGraph."""
    reset_tool_memory()

    # Step 0: Hybrid GraphRAG entry points
    if "Recommended Semantic Entry Points" not in problem_prompt:
        entry_points = find_semantic_entry_points(graph, problem_prompt, top_k=2)
        ep_block = format_entry_points_markdown(entry_points)
        if ep_block:
            problem_prompt = f"{problem_prompt}\n\n{ep_block}"
    else:
        entry_points = []

    # Initialize tools and state graph
    tools = create_cpg_langchain_tools(graph)
    checkpointer = MemorySaver()
    if use_multiagent:
        app = create_cpg_multiagent_system(
            model,
            tools,
            graph,
            max_turns=max_turns,
            max_patch_attempts=max_patch_attempts,
            checkpointer=checkpointer,
        )
    else:
        app = create_cpg_langgraph_agent(
            model,
            tools,
            graph,
            max_turns=max_turns,
            max_patch_attempts=max_patch_attempts,
            checkpointer=checkpointer,
        )

    initial_messages: List[BaseMessage] = [
        SystemMessage(content=GRAPH_OF_THOUGHTS_SYSTEM_PROMPT),
        HumanMessage(content=problem_prompt),
    ]

    initial_state: CPGDebugState = {
        "messages": initial_messages,
        "visited_nodes": [],
        "visited_signatures": [],
        "entry_points": entry_points,
        "final_deduction": None,
        "turn_count": 0,
        "status": "IN_PROGRESS",
        "patch_attempts": 0,
        "max_patch_attempts": max_patch_attempts,
        "failure_tracebacks": [],
        "implicated_variables": [],
    }

    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 60)
    print("  Beginning LangGraph Autonomous Graph of Thoughts Loop")
    print("=" * 60)

    final_result = None
    for event in app.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name in ("agent", "locator", "solver"):
                turn = node_output.get("turn_count", 0)
                msgs = node_output.get("messages") or []
                if msgs:
                    last_msg = msgs[-1]
                    print(f"\n--- [Turn {turn}/{max_turns}] Generating reasoning & actions ---")
                    print(f"[Model Output]:\n{last_msg.content}\n")

                if node_output.get("final_deduction"):
                    print("[+] Model finished reasoning and provided final deduction.")
                    final_result = node_output["final_deduction"]

            elif node_name in ("tools", "locator_tools", "solver_tools"):
                for tm in node_output.get("messages", []):
                    lines = len(tm.content.splitlines())
                    print(f"[*] Executed Graph Tool: `{tm.name}` -> Returned {lines} lines.")
                if node_output.get("status") == "SOLVED" and node_output.get("final_deduction"):
                    print("[+] BLSP Runtime Verification succeeded (status='SOLVED'). Terminating graph.")
                    final_result = node_output["final_deduction"]
                elif node_output.get("status") == "FAILED_MAX_RETRIES" and node_output.get("final_deduction"):
                    print("[!] Reached maximum BLSP patch retry attempts. Terminating graph.")
                    final_result = node_output["final_deduction"]

            elif node_name == "verifier":
                if node_output.get("status") == "SOLVED" and node_output.get("final_deduction"):
                    print("[+] BLSP Runtime Verification succeeded (status='SOLVED'). Terminating graph.")
                    final_result = node_output["final_deduction"]
                elif node_output.get("status") == "FAILED_MAX_RETRIES":
                    print("[!] Reached maximum BLSP patch retry attempts. Terminating graph.")
                    final_result = node_output.get("final_deduction")

    if not final_result:
        final_state = app.get_state(config).values
        if final_state.get("final_deduction"):
            final_result = final_state["final_deduction"]
        elif final_state.get("messages"):
            final_result = final_state["messages"][-1].content
        else:
            final_result = "No deduction produced."

    return final_result
