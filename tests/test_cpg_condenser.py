#!/usr/bin/env python3
"""Automated Unit & Optimization Comparison Tests for `cpg_condenser.py`.

Tests:
  1. `condense_cpg(G: nx.MultiDiGraph) -> nx.MultiDiGraph`:
     - Collapses sub-expression AST nodes into statement-level CFG/DDG nodes.
     - Computes fast Node Data-Flow Summary (`reads`, `writes`, `calls`) for every `METHOD` node.
     - Preserves backward compatibility with `cpg_bridge` traversals (`traverse_ast`, `traverse_cfg`,
       `traverse_data_flow`, `trace_variable_to_sink`).
  2. `condense_method_slice(G, method_id, target_vars, token_budget)`:
     - Retains only statements lying on the backward/forward slice of `target_vars`.
     - Folds non-causal blocks into single-line `# [condensed block: N statements]` markers.
     - Strictly enforces `token_budget`.
  3. Optimization Comparison Benchmark (Old Raw CPG vs. New Condensed CPG & Method Slicing):
     - Compares global & per-method node/edge reduction, CFG/DDG hop reduction,
       context token reduction, and traversal query speedup.
"""

from __future__ import annotations

import sys
from pathlib import Path
import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gdac.services import (
    compare_cpg_optimization,
    condense_cpg,
    condense_method_slice,
    find_nodes,
    load_cpg_json,
    resolve_method_id,
    trace_variable_to_sink,
    traverse_ast,
    traverse_cfg,
    traverse_data_flow,
)


@pytest.fixture(scope="module")
def raw_graph() -> nx.MultiDiGraph:
    """Loads the raw Joern CPG14 graph from `cpg_json/`."""
    cpg_path = Path(__file__).parent.parent / "cpg_json"
    if not cpg_path.exists():
        pytest.skip(f"CPG directory not found at {cpg_path}")
    return load_cpg_json(cpg_path)


@pytest.fixture(scope="module")
def condensed_graph(raw_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Returns the condensed CPG produced by `condense_cpg(raw_graph)`."""
    return condense_cpg(raw_graph)


class TestCondenseCPG:
    """Verifies `condense_cpg` AST sub-expression collapsing and METHOD data-flow summaries."""

    def test_condense_cpg_collapses_subexpressions(
        self, raw_graph: nx.MultiDiGraph, condensed_graph: nx.MultiDiGraph
    ) -> None:
        """Verifies significant node and edge reduction while preserving all METHOD nodes."""
        assert isinstance(condensed_graph, nx.MultiDiGraph)

        raw_nodes = raw_graph.number_of_nodes()
        cond_nodes = condensed_graph.number_of_nodes()
        raw_edges = raw_graph.number_of_edges()
        cond_edges = condensed_graph.number_of_edges()

        # Sub-expression contraction must reduce graph nodes & edges by >50%
        assert cond_nodes < raw_nodes * 0.5, f"Expected >50% node reduction, got {cond_nodes}/{raw_nodes}"
        assert cond_edges < raw_edges * 0.5, f"Expected >50% edge reduction, got {cond_edges}/{raw_edges}"

        # All METHOD nodes must be preserved
        raw_methods = set(find_nodes(raw_graph, label="METHOD"))
        cond_methods = set(find_nodes(condensed_graph, label="METHOD"))
        assert raw_methods == cond_methods

        # Sub-expression leaf labels like IDENTIFIER and FIELD_IDENTIFIER should be collapsed into statements
        cond_identifiers = find_nodes(condensed_graph, label="IDENTIFIER")
        assert len(cond_identifiers) == 0, "Sub-expression IDENTIFIER nodes should be collapsed into statements"

    def test_method_data_flow_summaries_populated(self, condensed_graph: nx.MultiDiGraph) -> None:
        """Verifies that every METHOD node has a populated `data_flow_summary` (`reads`, `writes`, `calls`)."""
        method_nodes = find_nodes(condensed_graph, label="METHOD")
        assert len(method_nodes) > 0

        for m_id in method_nodes:
            m_data = condensed_graph.nodes[m_id]
            assert "data_flow_summary" in m_data
            summary = m_data["data_flow_summary"]
            assert isinstance(summary, dict)
            assert set(summary.keys()) == {"reads", "writes", "calls"}
            assert m_data["reads"] == summary["reads"]
            assert m_data["writes"] == summary["writes"]
            assert m_data["calls"] == summary["calls"]

            # Ensure no compiler temporary variables (`tmp0`, `tmp1`, etc.) leak into summaries
            for key in ("reads", "writes", "calls"):
                for item in summary[key]:
                    assert not item.startswith("tmp"), f"Compiler temp '{item}' leaked into {key} of method #{m_id}"

        # Check specific accuracy on `apply_discount`
        discount_id = resolve_method_id(condensed_graph, "apply_discount")
        d_sum = condensed_graph.nodes[discount_id]["data_flow_summary"]
        assert "discount_percent" in d_sum["reads"]
        assert "sku" in d_sum["reads"]
        assert any("item.price" in w for w in d_sum["writes"])
        assert "item" in d_sum["writes"]
        assert any("get_item" in c for c in d_sum["calls"])
        assert "round" in d_sum["calls"]

        # Check specific accuracy on `generate_category_summary`
        summary_id = resolve_method_id(condensed_graph, "generate_category_summary")
        s_sum = condensed_graph.nodes[summary_id]["data_flow_summary"]
        assert "item_total" in s_sum["writes"]
        assert any("summary" in w for w in s_sum["writes"])
        assert any("item.price" in r for r in s_sum["reads"])
        assert any("item.quantity" in r for r in s_sum["reads"])
        assert "round" in s_sum["calls"]


class TestCondenseMethodSlice:
    """Verifies `condense_method_slice` backward/forward slicing, block folding, and token budget."""

    def test_slice_apply_discount_folds_non_causal_blocks(
        self, condensed_graph: nx.MultiDiGraph
    ) -> None:
        """Slices `apply_discount` on `discount_percent` and verifies non-causal blocks are folded."""
        discount_id = resolve_method_id(condensed_graph, "apply_discount")
        res = condense_method_slice(
            condensed_graph,
            method_id=discount_id,
            target_vars=["discount_percent"],
            token_budget=200,
        )

        slice_text = str(res)
        # Must retain the buggy discount price assignment and return statement
        assert "item.price = round(item.price * discount_percent / 100.0, 2)" in slice_text
        assert "return item" in slice_text

        # `self.history.append(...)` reads `discount_percent` (retained as a sink per Fix 4),
        # while `self.storage.save(self._items)` is non-causal and folded into `# [condensed block: 1 statements]`
        assert "# [condensed block:" in slice_text
        assert len(res["folded_blocks"]) >= 1
        assert res["token_stats"]["folded_statements_count"] >= 1
        assert res["token_stats"]["within_budget"] is True

    def test_slice_generate_category_summary(
        self, condensed_graph: nx.MultiDiGraph
    ) -> None:
        """Slices `generate_category_summary` on `item_total` and verifies causal statements are kept."""
        summary_id = resolve_method_id(condensed_graph, "generate_category_summary")
        res = condense_method_slice(
            condensed_graph,
            method_id=summary_id,
            target_vars=["item_total"],
            token_budget=200,
        )

        slice_text = str(res)
        assert "item_total = item.price * item.quantity" in slice_text
        assert "summary[item.category] = round(item_total, 2)" in slice_text
        assert res["token_stats"]["within_budget"] is True

    def test_strict_token_budget_pruning(
        self, condensed_graph: nx.MultiDiGraph
    ) -> None:
        """Verifies that a tight `token_budget` prunes distant statements into condensed markers."""
        discount_id = resolve_method_id(condensed_graph, "apply_discount")
        tight_budget = 45
        res = condense_method_slice(
            condensed_graph,
            method_id=discount_id,
            target_vars=["discount_percent"],
            token_budget=tight_budget,
        )

        assert res["token_stats"]["used_tokens"] <= tight_budget
        assert res["token_stats"]["within_budget"] is True
        # Core mutation statement on `discount_percent` should still be preserved as highest priority
        assert "discount_percent" in str(res)
        assert "# [condensed block:" in str(res)


class TestOptimizationComparisonOldVsNew:
    """Compares Old (Raw CPG) vs. New (Condensed CPG & Method Slice) across optimization metrics."""

    def test_optimization_comparison_metrics(self, raw_graph: nx.MultiDiGraph) -> None:
        """Benchmarks Raw CPG vs. Condensed CPG and asserts concrete optimization gains."""
        report = compare_cpg_optimization(raw_graph, token_budget=120)
        gm = report["graph_metrics"]

        # 1. Global Graph Size Optimization
        assert gm["node_reduction_pct"] >= 60.0, (
            f"Expected >=60% global node reduction, got {gm['node_reduction_pct']}%"
        )
        assert gm["edge_reduction_pct"] >= 60.0, (
            f"Expected >=60% global edge reduction, got {gm['edge_reduction_pct']}%"
        )

        # 2. Per-Method Optimization on both benchmark methods
        for mb in report["method_benchmarks"]:
            # AST nodes per method should drop dramatically (e.g. 69 -> ~12)
            assert mb["new_ast_nodes"] < mb["old_ast_nodes"]
            assert mb["ast_node_reduction_pct"] >= 65.0

            # CFG & DDG path hops should be shorter (statement hops instead of sub-expression hops)
            assert mb["new_cfg_hops"] < mb["old_cfg_hops"]
            assert mb["new_ddg_hops"] <= mb["old_ddg_hops"]

            # Context token payload should be significantly smaller (>65% token savings)
            assert mb["new_payload_tokens"] < mb["old_payload_tokens"]
            assert mb["token_reduction_pct"] >= 65.0

        # Print comparison summary when run with `pytest -s`
        print("\n" + "=" * 72)
        print("  OPTIMIZATION COMPARISON: OLD (RAW CPG) vs. NEW (CONDENSED CPG)")
        print("=" * 72)
        print(
            f"  Global Nodes : {gm['raw_nodes']} -> {gm['condensed_nodes']} "
            f"(-{gm['node_reduction_pct']}%)"
        )
        print(
            f"  Global Edges : {gm['raw_edges']} -> {gm['condensed_edges']} "
            f"(-{gm['edge_reduction_pct']}%)"
        )
        for mb in report["method_benchmarks"]:
            print(
                f"  [{mb['method']}] AST Nodes: {mb['old_ast_nodes']} -> {mb['new_ast_nodes']} "
                f"(-{mb['ast_node_reduction_pct']}%) | "
                f"CFG Hops: {mb['old_cfg_hops']} -> {mb['new_cfg_hops']} | "
                f"DDG Hops: {mb['old_ddg_hops']} -> {mb['new_ddg_hops']} | "
                f"Tokens: {mb['old_payload_tokens']} -> {mb['new_payload_tokens']} "
                f"(-{mb['token_reduction_pct']}%) | "
                f"Speedup: {mb['speedup_factor']}x"
            )
        print("=" * 72)

    def test_language_agnostic_c_cpp_cpg_condensation_and_slice(self) -> None:
        """Verifies that `condense_cpg` and `condense_method_slice` work on a C/C++ Joern CPG
        using pointer field accesses (`acct->balance`), compound assignments (`-=`),
        post-increment (`tx_count++`), and void calls (`printf`).
        """
        G = nx.MultiDiGraph()
        # Method: `double charge_fee(Account *acct, double fee)`
        G.add_node(1, id=1, label="METHOD", code="charge_fee", name="charge_fee", line_number=10)
        G.add_node(2, id=2, label="PARAM", code="acct", name="acct", line_number=10)
        G.add_node(3, id=3, label="PARAM", code="fee", name="fee", line_number=10)
        G.add_node(4, id=4, label="BLOCK", code="", line_number=10)
        G.add_node(5, id=5, label="METHOD_RETURN", code="double", line_number=10)

        # Statement 1 (L11): `tx_count++` (<operator>.postIncrement)
        G.add_node(10, id=10, label="<operator>.postIncrement", code="tx_count++", line_number=11)
        G.add_node(11, id=11, label="IDENTIFIER", code="tx_count", line_number=11)

        # Statement 2 (L12): `printf("Charging fee\\n")` (non-causal void logging call)
        G.add_node(20, id=20, label="printf", code='printf("Charging fee\\n")', line_number=12)
        G.add_node(21, id=21, label="LITERAL", code='"Charging fee\\n"', line_number=12)

        # Statement 3 (L13): `acct->balance -= fee` (<operator>.assignmentMinus + <operator>.indirectFieldAccess)
        G.add_node(30, id=30, label="<operator>.assignmentMinus", code="acct->balance -= fee", line_number=13)
        G.add_node(31, id=31, label="<operator>.indirectFieldAccess", code="acct->balance", line_number=13)
        G.add_node(32, id=32, label="IDENTIFIER", code="acct", line_number=13)
        G.add_node(33, id=33, label="FIELD_IDENTIFIER", code="balance", line_number=13)
        G.add_node(34, id=34, label="IDENTIFIER", code="fee", line_number=13)

        # Statement 4 (L14): `return acct->balance;`
        G.add_node(40, id=40, label="RETURN", code="return acct->balance;", line_number=14)
        G.add_node(41, id=41, label="<operator>.indirectFieldAccess", code="acct->balance", line_number=14)
        G.add_node(42, id=42, label="IDENTIFIER", code="acct", line_number=14)
        G.add_node(43, id=43, label="FIELD_IDENTIFIER", code="balance", line_number=14)

        # AST Edges
        for ch in (2, 3, 4, 5):
            G.add_edge(1, ch, label="AST")
        for st in (10, 20, 30, 40):
            G.add_edge(4, st, label="AST")
        G.add_edge(10, 11, label="AST")
        G.add_edge(20, 21, label="AST")
        G.add_edge(30, 31, label="AST")
        G.add_edge(30, 34, label="AST")
        G.add_edge(31, 32, label="AST")
        G.add_edge(31, 33, label="AST")
        G.add_edge(40, 41, label="AST")
        G.add_edge(41, 42, label="AST")
        G.add_edge(41, 43, label="AST")

        # Fine-grained CFG & DDG Edges
        G.add_edge(1, 11, label="CFG")
        G.add_edge(11, 10, label="CFG")
        G.add_edge(10, 21, label="CFG")
        G.add_edge(21, 20, label="CFG")
        G.add_edge(20, 32, label="CFG")
        G.add_edge(32, 33, label="CFG")
        G.add_edge(33, 31, label="CFG")
        G.add_edge(31, 34, label="CFG")
        G.add_edge(34, 30, label="CFG")
        G.add_edge(30, 42, label="CFG")
        G.add_edge(42, 43, label="CFG")
        G.add_edge(43, 41, label="CFG")
        G.add_edge(41, 40, label="CFG")
        G.add_edge(40, 5, label="CFG")

        G.add_edge(2, 32, label="DDG", variable="acct")
        G.add_edge(3, 34, label="DDG", variable="fee")
        G.add_edge(31, 30, label="DDG", variable="acct->balance")
        G.add_edge(34, 30, label="DDG", variable="fee")
        G.add_edge(30, 41, label="REACHING_DEF", variable="acct->balance")
        G.add_edge(41, 40, label="DDG", variable="acct->balance")

        cond = condense_cpg(G)
        summary = cond.nodes[1]["data_flow_summary"]

        # Verify C/C++ reads, writes, and calls
        assert "acct->balance" in summary["writes"]
        assert "tx_count" in summary["writes"]
        assert "fee" in summary["reads"]
        assert "acct->balance" in summary["reads"]
        assert "printf" in summary["calls"]

        # Verify slicing on `fee` keeps `acct->balance -= fee` and `return acct->balance;`
        # while folding `tx_count++` and `printf(...)` into `# [condensed block: 2 statements]`
        slice_res = condense_method_slice(cond, method_id=1, target_vars=["fee"], token_budget=200)
        slice_str = str(slice_res)
        assert "# [condensed block: 2 statements]" in slice_str
        assert "acct->balance -= fee" in slice_str
        assert "return acct->balance;" in slice_str
        assert "printf" not in slice_str


class TestRegressionFixes:
    """Verifies fixes for the 5 edge-case bugs in `cpg_condenser.py`."""

    def test_raw_graph_not_mutated_in_place(self, raw_graph: nx.MultiDiGraph) -> None:
        """Fix 5: Verifies `condense_cpg(G)` does not mutate `G.nodes[m_id]` in place."""
        fresh_raw = load_cpg_json(Path(__file__).parent.parent / "cpg_json")
        m_id = resolve_method_id(fresh_raw, "apply_discount")
        assert "data_flow_summary" not in fresh_raw.nodes[m_id]
        assert "reads" not in fresh_raw.nodes[m_id]

        cond = condense_cpg(fresh_raw)
        assert "data_flow_summary" not in fresh_raw.nodes[m_id]
        assert "reads" not in fresh_raw.nodes[m_id]
        assert "data_flow_summary" in cond.nodes[m_id]
        assert "reads" in cond.nodes[m_id]

    def test_temp_edges_resolved_and_multi_stmt_lowered_block_preserved(self) -> None:
        """Fix 1: Verifies that `tmp*` assignments in multi-statement blocks contract into their
        consumer statement and `DDG`/`REACHING_DEF` edges via `tmp0` are resolved rather than dropped.
        """
        G = nx.MultiDiGraph()
        G.add_node(1, id=1, label="METHOD", code="compute", name="compute", line_number=1)
        G.add_node(2, id=2, label="PARAM", code="x", name="x", line_number=1)
        G.add_node(3, id=3, label="BLOCK", code="", line_number=1)
        G.add_node(4, id=4, label="METHOD_RETURN", code="int", line_number=1)

        # Statement A (L2): `a = x + 1`
        G.add_node(10, id=10, label="<operator>.assignment", code="a = x + 1", line_number=2)
        G.add_node(11, id=11, label="IDENTIFIER", code="a", line_number=2)
        G.add_node(12, id=12, label="IDENTIFIER", code="x", line_number=2)

        # Lowered temp assignment (L3): `tmp0 = a`
        G.add_node(20, id=20, label="<operator>.assignment", code="tmp0 = a", line_number=3)
        G.add_node(21, id=21, label="IDENTIFIER", code="tmp0", line_number=3)
        G.add_node(22, id=22, label="IDENTIFIER", code="a", line_number=3)

        # Consumer statement B (L3): `b = tmp0 * 2`
        G.add_node(30, id=30, label="<operator>.assignment", code="b = tmp0 * 2", line_number=3)
        G.add_node(31, id=31, label="IDENTIFIER", code="b", line_number=3)
        G.add_node(32, id=32, label="IDENTIFIER", code="tmp0", line_number=3)

        # Return (L4): `return b`
        G.add_node(40, id=40, label="RETURN", code="return b", line_number=4)
        G.add_node(41, id=41, label="IDENTIFIER", code="b", line_number=4)

        for ch in (2, 3, 4):
            G.add_edge(1, ch, label="AST")
        for st in (10, 20, 30, 40):
            G.add_edge(3, st, label="AST")
        G.add_edge(10, 11, label="AST")
        G.add_edge(10, 12, label="AST")
        G.add_edge(20, 21, label="AST")
        G.add_edge(20, 22, label="AST")
        G.add_edge(30, 31, label="AST")
        G.add_edge(30, 32, label="AST")
        G.add_edge(40, 41, label="AST")

        # Data flow goes from 10 -> 20 via `a`, and 20 -> 30 via `tmp0`, and 30 -> 40 via `b`
        G.add_edge(10, 20, label="REACHING_DEF", variable="a")
        G.add_edge(20, 30, label="REACHING_DEF", variable="tmp0")
        G.add_edge(30, 40, label="REACHING_DEF", variable="b")

        cond = condense_cpg(G)
        # Verify edge between 10 and 30 is preserved in condensed graph
        assert cond.has_edge(10, 30)
        slice_res = condense_method_slice(cond, method_id=1, target_vars=["b"], token_budget=200)
        assert "a = x + 1" in str(slice_res)
        assert "b = a * 2" in str(slice_res)

    def test_enclosing_if_and_loop_headers_retained_via_cdg(self) -> None:
        """Fix 2: Verifies that `condense_cpg` adds `(ctrl_id, body_stmt_id, label="CDG")` edges
        so slicing on a statement inside an IF/LOOP branch retains its enclosing header.
        """
        G = nx.MultiDiGraph()
        G.add_node(1, id=1, label="METHOD", code="check_limit", name="check_limit", line_number=1)
        G.add_node(2, id=2, label="PARAM", code="flag", name="flag", line_number=1)
        G.add_node(3, id=3, label="BLOCK", code="", line_number=1)

        # Control structure `if (flag)` that does NOT mention `secret_val` in its condition
        G.add_node(10, id=10, label="CONTROL_STRUCTURE", code="if (flag)", line_number=2)
        G.add_node(11, id=11, label="IDENTIFIER", code="flag", line_number=2)
        G.add_node(12, id=12, label="BLOCK", code="", line_number=2)

        # Body statement inside `if (flag)`: `secret_val = 42`
        G.add_node(20, id=20, label="<operator>.assignment", code="secret_val = 42", line_number=3)
        G.add_node(21, id=21, label="IDENTIFIER", code="secret_val", line_number=3)

        G.add_edge(1, 2, label="AST")
        G.add_edge(1, 3, label="AST")
        G.add_edge(3, 10, label="AST")
        G.add_edge(10, 11, label="AST")
        G.add_edge(10, 12, label="AST")
        G.add_edge(12, 20, label="AST")
        G.add_edge(20, 21, label="AST")

        # Note: No explicit CDG edge in raw graph `G`
        cond = condense_cpg(G)
        cdg_edges = [
            (u, v) for u, v, d in cond.edges(data=True) if d.get("label") == "CDG"
        ]
        assert (10, 20) in cdg_edges

        # Slicing on `secret_val` must retain the enclosing `if (flag)` header via CDG
        slice_res = condense_method_slice(cond, method_id=1, target_vars=["secret_val"], token_budget=200)
        assert "if (flag)" in str(slice_res)
        assert "secret_val = 42" in str(slice_res)

    def test_loop_back_edges_in_def_use_slicing(self) -> None:
        """Fix 3: Verifies that a mutation at the bottom of a loop body links backward via Def-Use
        to a read at the top of the loop body on the next iteration.
        """
        G = nx.MultiDiGraph()
        G.add_node(1, id=1, label="METHOD", code="accumulate", name="accumulate", line_number=1)
        G.add_node(2, id=2, label="BLOCK", code="", line_number=1)

        # Loop header (L2): `WHILE (running)`
        G.add_node(10, id=10, label="CONTROL_STRUCTURE", code="WHILE", line_number=2)
        G.add_node(11, id=11, label="IDENTIFIER", code="running", line_number=2)
        G.add_node(12, id=12, label="BLOCK", code="", line_number=2)

        # Top of loop body (L3): `result = acc * 2` (reads `acc`, writes `result`)
        G.add_node(20, id=20, label="<operator>.assignment", code="result = acc * 2", line_number=3)
        G.add_node(21, id=21, label="IDENTIFIER", code="result", line_number=3)
        G.add_node(22, id=22, label="IDENTIFIER", code="acc", line_number=3)

        # Bottom of loop body (L4): `acc = step + 1` (writes `acc`, reads `step`)
        G.add_node(30, id=30, label="<operator>.assignment", code="acc = step + 1", line_number=4)
        G.add_node(31, id=31, label="IDENTIFIER", code="acc", line_number=4)
        G.add_node(32, id=32, label="IDENTIFIER", code="step", line_number=4)

        G.add_edge(1, 2, label="AST")
        G.add_edge(2, 10, label="AST")
        G.add_edge(10, 11, label="AST")
        G.add_edge(10, 12, label="AST")
        G.add_edge(12, 20, label="AST")
        G.add_edge(12, 30, label="AST")
        G.add_edge(20, 21, label="AST")
        G.add_edge(20, 22, label="AST")
        G.add_edge(30, 31, label="AST")
        G.add_edge(30, 32, label="AST")

        cond = condense_cpg(G)
        # Slicing on `result` (L3) must include `acc = step + 1` (L4) via loop-carried Def-Use back-edge
        slice_res = condense_method_slice(cond, method_id=1, target_vars=["result"], token_budget=200)
        assert "result = acc * 2" in str(slice_res)
        assert "acc = step + 1" in str(slice_res)

    def test_data_flow_sink_calls_retained_when_reading_target_var(
        self, condensed_graph: nx.MultiDiGraph
    ) -> None:
        """Fix 4: Verifies that void calls like `self.storage.save(self._items)` are retained
        as data-flow sinks when their read arguments match `target_vars`.
        """
        discount_id = resolve_method_id(condensed_graph, "apply_discount")
        # Slicing on `_items` should retain `self.storage.save(self._items)` as a sink call
        res = condense_method_slice(
            condensed_graph,
            method_id=discount_id,
            target_vars=["_items"],
            token_budget=200,
        )
        assert "self.storage.save(self._items)" in str(res)

    def test_bottleneck_a_asymmetric_mutation_scoring_and_camelcase(self) -> None:
        """Bottleneck A: Verifies that `CPGSemanticIndex` prioritizes methods that mutate (`writes`)
        a corrupted state variable over passive getters/readers with verbose docstrings, and
        properly splits camelCase identifiers.
        """
        from gdac.services import CPGSemanticIndex

        G = nx.MultiDiGraph()
        # Method 1: Verbose passive getter that reads `item.price` and mentions `price` many times
        G.add_node(
            1,
            id=1,
            label="METHOD",
            code="getItemPriceDisplay",
            name="getItemPriceDisplay",
            line_number=10,
            reads=["self", "item", "item.price", "a", "b", "c", "d", "e"],
            writes=[],
            calls=["format"],
            data_flow_summary={
                "reads": ["self", "item", "item.price", "a", "b", "c", "d", "e"],
                "writes": [],
                "calls": ["format"],
            },
        )
        G.add_node(
            2,
            id=2,
            label="LITERAL",
            code='"""Returns item.price formatted string for item.price display."""',
            line_number=11,
        )
        G.add_edge(1, 2, label="AST")

        # Method 2: Concise mutator that writes `item.price` without mentioning `discount` in its name
        G.add_node(
            10,
            id=10,
            label="METHOD",
            code="recomputeValuation",
            name="recomputeValuation",
            line_number=30,
            reads=["self", "item"],
            writes=["item.price"],
            calls=["round"],
            data_flow_summary={
                "reads": ["self", "item"],
                "writes": ["item.price"],
                "calls": ["round"],
            },
        )

        idx = CPGSemanticIndex(G)
        # Verify camelCase tokenization works (`recomputeValuation` -> `recompute`, `valuation`)
        assert "recompute" in idx._tokenize("recomputeValuation")
        assert "valuation" in idx._tokenize("recomputeValuation")

        # Query only mentions the corrupted state variable (`AssertionError: item.price == 20.0 != 80.0`)
        matches = idx.search("AssertionError: item.price == 20.0 != 80.0", top_k=2)
        assert len(matches) == 2
        # Mutator of `item.price` (`recomputeValuation`) must outrank passive reader (`getItemPriceDisplay`)
        assert matches[0]["method_name"] == "recomputeValuation"
        assert "item.price" in matches[0]["matched_target_vars"]

    def test_bottleneck_b_interprocedural_callee_and_mutator_links(
        self, condensed_graph: nx.MultiDiGraph
    ) -> None:
        """Bottleneck B: Verifies that `condense_method_slice` resolves Inter-Procedural Links
        (Callee `METHOD #id` and Sibling State-Mutator `METHOD #id`) and appends them to the slice.
        """
        discount_id = resolve_method_id(condensed_graph, "apply_discount")
        res = condense_method_slice(
            condensed_graph,
            method_id=discount_id,
            target_vars=["discount_percent"],
            token_budget=250,
        )
        assert "interprocedural_links" in res
        links = res["interprocedural_links"]
        assert len(links) >= 1

        # Must resolve `self.get_item` to `InventoryManager.get_item` with its exact METHOD node_id
        callee_names = {l["qualified_name"] for l in links if l["type"] == "callee"}
        assert "InventoryManager.get_item" in callee_names
        assert "[Inter-Procedural Links" in str(res)
        assert "Callee `InventoryManager.get_item` (METHOD #" in str(res)

        # Must also resolve sibling mutators of `self._items` when `_items` is read
        res_items = condense_method_slice(
            condensed_graph,
            method_id=discount_id,
            target_vars=["_items"],
            token_budget=250,
        )
        mutator_names = {
            l["qualified_name"]
            for l in res_items["interprocedural_links"]
            if l["type"] == "sibling_mutator"
        }
        assert "InventoryManager.add_item" in mutator_names
        assert "Sibling Mutator of `self._items`" in str(res_items)

    def test_bottleneck_c_stage1_to_stage2_target_vars_handoff(
        self, condensed_graph: nx.MultiDiGraph
    ) -> None:
        """Bottleneck C: Verifies that Stage 1 (`find_semantic_entry_points` &
        `format_entry_points_markdown`) extracts `matched_target_vars` and emits a direct
        `condense_method_slice` handoff hint.
        """
        from gdac.services import find_semantic_entry_points, format_entry_points_markdown

        eps = find_semantic_entry_points(
            condensed_graph,
            "FAILED test_apply_discount: discount_percent calculation on item.price",
            top_k=2,
        )
        assert len(eps) >= 1
        assert eps[0]["qualified_name"] == "InventoryManager.apply_discount"
        assert len(eps[0]["matched_target_vars"]) >= 1

        md = format_entry_points_markdown(eps)
        assert "matched_target_vars=" in md
        assert "condense_method_slice(" in md


