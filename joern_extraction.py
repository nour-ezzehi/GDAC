#!/usr/bin/env python3
"""Joern Extraction Pipeline.

Runs joern-parse on the target codebase and exports
the Code Property Graph "CPG" into CPG14 JSON format via joern-export.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_tool(tool_name: str) -> str:
    """Checks if a required tool exists in PATH, returning its path."""
    path = shutil.which(tool_name)
    if not path:
        print(f"[-] ERROR: Required executable '{tool_name}' not found in PATH.", file=sys.stderr)
        print("    If Joern is installed in a custom directory, add it to your PATH:", file=sys.stderr)
        print("      export PATH=\"$HOME/bin:$PATH\"", file=sys.stderr)
        print("    To install Joern on Linux, run:", file=sys.stderr)
        print("      curl -L 'https://github.com/joernio/joern/releases/latest/download/joern-install.sh' | bash", file=sys.stderr)
        sys.exit(1)
    return path


def run_command(cmd: list[str], description: str) -> None:
    """Runs a subprocess command and handles errors."""
    print(f"[*] {description}...")
    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[-] Command failed with exit code {result.returncode}", file=sys.stderr)
        if result.stdout:
            print(f"--- STDOUT ---\n{result.stdout.strip()}", file=sys.stderr)
        if result.stderr:
            print(f"--- STDERR ---\n{result.stderr.strip()}", file=sys.stderr)
        sys.exit(result.returncode)
    if result.stdout.strip():
        print(f"    {result.stdout.strip().splitlines()[-1]}")


def extract_cpg(
    repo_path: str | Path = "stackflow",
    cpg_bin: str | Path = "cpg.bin",
    out_dir: str | Path = "cpg_json",
) -> Path:
    """Extracts CPG from repository and exports to CPG14 JSON directory."""
    repo_path = Path(repo_path).resolve()
    cpg_bin = Path(cpg_bin).resolve()
    out_dir = Path(out_dir).resolve()

    if not repo_path.exists():
        raise FileNotFoundError(f"Target repository '{repo_path}' does not exist.")

    # 1. Verify joern-parse and joern-export exist
    joern_parse = check_tool("joern-parse")
    joern_export = check_tool("joern-export")

    # 2. Run joern-parse
    parse_cmd = [joern_parse, str(repo_path), "--output", str(cpg_bin)]
    run_command(parse_cmd, f"Running joern-parse on '{repo_path.name}'")

    # 3. Clean previous export directory if exists
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # 4. Run joern-export
    export_cmd = [
        joern_export,
        str(cpg_bin),
        "--repr",
        "cpg14",
        "--out",
        str(out_dir),
    ]
    run_command(export_cmd, f"Exporting CPG14 JSON to '{out_dir.name}'")

    print(f"[+] Successfully exported CPG14 JSON to: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CPG from repository using joern-parse and joern-export."
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
        help="Output directory for CPG14 JSON files (default: cpg_json)",
    )

    args = parser.parse_args()
    extract_cpg(repo_path=args.repo, cpg_bin=args.cpg_bin, out_dir=args.out)


if __name__ == "__main__":
    main()
