"""LangGraph State Definition for GDAC Autonomous Debugging Agents."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class CPGDebugState(TypedDict, total=False):
    """The central state of the Graph-of-Thoughts and Multi-Agent debugging workflows."""
    messages: Annotated[List[BaseMessage], add_messages]
    visited_nodes: Sequence[int]
    visited_signatures: Sequence[Tuple[str, str, str]]
    entry_points: List[Dict[str, Any]]
    final_deduction: Optional[str]
    turn_count: int
    status: Optional[str]
    patch_attempts: int
    max_patch_attempts: int
    failure_tracebacks: List[str]
    implicated_variables: List[str]
    buggy_target: Optional[str]
    buggy_line: Optional[int]
    buggy_old_string: Optional[str]
    proposed_new_string: Optional[str]
