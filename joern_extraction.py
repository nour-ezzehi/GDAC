#!/usr/bin/env python3
"""Joern Extraction Pipeline.

Runs joern-parse on the target repository and exports the Code Property Graph (CPG14)
using joern-export. Also compiles a consolidated JSON representation for downstream bridges.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def get_joern_env() -> Dict[str, str]:
    """Ensures JAVA_HOME and PATH are properly configured for Joern."""
    env = os.environ.copy()

    # Prefer user-space JRE if present
    default_jre = Path("/home/aaaa/.local/lib/jre21")
    if "JAVA_HOME" not in env and default_jre.exists():
        env["JAVA_HOME"] = str(default_jre)

    java_bin = f"{env.get('JAVA_HOME', '')}/bin"
    joern_bin = "/home/aaaa/.local/share/joern/joern-cli"
    local_bin = "/home/aaaa/.local/bin"

    current_path = env.get("PATH", "")
    env["PATH"] = f"{local_bin}:{joern_bin}:{java_bin}:{current_path}"
    return env


def check_tool(tool_name: str, env: Dict[str, str]) -> str:
    """Checks if a required tool exists in PATH, returning its path."""
    path = shutil.which(tool_name, path=env.get("PATH"))
    if not path:
        print(f"[-] ERROR: Required executable '{tool_name}' not found in PATH.", file=sys.stderr)
        print("    Please ensure Joern and Java are installed and on PATH.", file=sys.stderr)
        sys.exit(1)
    return path


def run_command(cmd: List[str], description: str, env: Dict[str, str]) -> None:
    """Runs a subprocess command and handles errors."""
    print(f"[*] {description}...")
    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if result.returncode != 0:
        print(f"[-] Command failed with exit code {result.returncode}", file=sys.stderr)
        if result.stdout:
            print(f"--- STDOUT ---\n{result.stdout.strip()}", file=sys.stderr)
        if result.stderr:
            print(f"--- STDERR ---\n{result.stderr.strip()}", file=sys.stderr)
        sys.exit(result.returncode)
    for line in result.stdout.strip().splitlines()[-3:]:
        if line.strip():
            print(f"    {line.strip()}")


def convert_dot_to_json(dot_dir: Path) -> Path:
    """Compiles exported CPG14 dot files into a consolidated JSON file (cpg_json/cpg.json)."""
    node_pattern = re.compile(r"\"(\d+)\"\s*\[\s*label\s*=\s*<(.*?)>\s*\]", re.DOTALL)
    edge_pattern = re.compile(r"\"(\d+)\"\s*->\s*\"(\d+)\"\s*\[\s*label\s*=\s*\"([^\"]*)\"\s*\]")

    all_nodes: Dict[int, Dict[str, Any]] = {}
    all_edges: List[Dict[str, Any]] = []

    dot_files = sorted(dot_dir.glob("*.dot"))
    for fpath in dot_files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        for m in node_pattern.finditer(content):
            nid = int(m.group(1))
            raw_label = html.unescape(m.group(2))
            parts = [p.strip() for p in raw_label.split("<BR/>") if p.strip()]
            header = parts[0] if parts else ""
            hdr_parts = [h.strip() for h in header.split(",", 1)]
            label = hdr_parts[0] if hdr_parts else "NODE"
            line = int(hdr_parts[1]) if len(hdr_parts) > 1 and hdr_parts[1].isdigit() else None
            code = parts[1] if len(parts) > 1 else ""

            if nid not in all_nodes:
                all_nodes[nid] = {
                    "id": nid,
                    "label": label,
                    "code": code,
                    "line_number": line,
                }

        for m in edge_pattern.finditer(content):
            src, dst, raw_elbl = int(m.group(1)), int(m.group(2)), html.unescape(m.group(3)).strip()
            edge_parts = raw_elbl.split(":", 1)
            etype = edge_parts[0].strip().upper()
            var = edge_parts[1].strip() if len(edge_parts) > 1 else None

            edge_dict = {"out": src, "in": dst, "label": etype}
            if var:
                edge_dict["variable"] = var
            all_edges.append(edge_dict)

    json_path = dot_dir / "cpg.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"nodes": list(all_nodes.values()), "edges": all_edges}, f, indent=2)

    print(f"[+] Consolidated CPG JSON written to: {json_path}")
    print(f"    Total unique nodes: {len(all_nodes)}, Total edges: {len(all_edges)}")
    return json_path


def extract_cpg(
    repo_path: str | Path = "stackflow",
    cpg_bin: str | Path = "cpg.bin",
    out_dir: str | Path = "cpg_json",
) -> Path:
    """Extracts CPG from repository and exports to CPG14 format."""
    repo_path = Path(repo_path).resolve()
    cpg_bin = Path(cpg_bin).resolve()
    out_dir = Path(out_dir).resolve()

    if not repo_path.exists():
        raise FileNotFoundError(f"Target repository '{repo_path}' does not exist.")

    env = get_joern_env()
    joern_parse = check_tool("joern-parse", env)
    joern_export = check_tool("joern-export", env)

    # 1. Run joern-parse
    parse_cmd = [joern_parse, str(repo_path), "--output", str(cpg_bin)]
    run_command(parse_cmd, f"Running joern-parse on '{repo_path.name}'", env)

    # 2. Clean previous export directory if exists
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # 3. Run joern-export
    export_cmd = [
        joern_export,
        str(cpg_bin),
        "--repr",
        "cpg14",
        "--out",
        str(out_dir),
    ]
    run_command(export_cmd, f"Running joern-export --repr cpg14 --out '{out_dir.name}'", env)

    # 4. Generate consolidated JSON for JSON consumers
    convert_dot_to_json(out_dir)

    print(f"[+] Real CPG extraction complete! Output in: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract real CPG from repository using joern-parse and joern-export."
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="stackflow",
        help="Path to repository to parse (default: stackflow)",
    )
    parser.add_argument(
        "--cpg-bin",
        type=str,
        default="cpg.bin",
        help="Output path for CPG binary (default: cpg.bin)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="cpg_json",
        help="Output directory for CPG14 files (default: cpg_json)",
    )

    args = parser.parse_args()
    extract_cpg(repo_path=args.repo, cpg_bin=args.cpg_bin, out_dir=args.out)


if __name__ == "__main__":
    main()
