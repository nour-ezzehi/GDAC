"""GDAC: Graph-Driven Autonomous Debugging & CPG Analysis Framework.

A modular framework combining Code Property Graph (CPG) static analysis,
Language-Agnostic CPG Condensation, Graph of Thoughts (GoT) reasoning,
multi-agent collaborative debugging, and BLSP runtime verification.
"""

from gdac import agents, prompts, services, tools, training

__version__ = "0.2.0"

__all__ = [
    "agents",
    "tools",
    "services",
    "prompts",
    "training",
]
