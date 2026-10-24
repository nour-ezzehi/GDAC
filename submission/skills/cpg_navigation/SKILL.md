---
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