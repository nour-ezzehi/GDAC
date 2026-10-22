"""BLSP Test Runner Service: Test command resolution and subprocess execution."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


def get_project_root() -> Path:
    """Finds the root repository directory of GDAC."""
    # Look for markers like pyproject.toml, cpg_json, or .git
    curr = Path(__file__).resolve().parent
    for p in [curr] + list(curr.parents):
        if (p / "cpg_json").is_dir() or (p / ".git").is_dir() or (p / "stackflow").is_dir():
            return p
    return Path(__file__).resolve().parents[2]


def resolve_test_cmd(
    file_path: Path,
    test_path: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> Tuple[List[str], str, Path]:
    """Resolves test runner command and target across multiple programming languages."""
    if project_root is None:
        project_root = get_project_root()

    ext = file_path.suffix.lower()
    resolved_test: Any = test_path or ""
    cmd: List[str] = []

    if ext == ".py":
        if test_path:
            resolved_test_path = Path(test_path)
            if not resolved_test_path.is_absolute():
                resolved_test_path = project_root / resolved_test_path
        else:
            module_test = file_path.parent.parent / "tests" / f"test_{file_path.stem}.py"
            if module_test.is_file():
                resolved_test_path = module_test
            else:
                resolved_test_path = file_path.parent.parent / "tests"
        resolved_test = str(resolved_test_path)
        cmd = [sys.executable, "-m", "pytest", resolved_test, "-v", "--tb=short"]
    elif ext == ".go":
        resolved_test = test_path or "./..."
        cmd = ["go", "test", "-v", str(resolved_test)]
    elif ext == ".rs":
        resolved_test = test_path or "cargo test"
        cmd = ["cargo", "test"] + ([str(test_path)] if test_path else [])
    elif ext in (".js", ".jsx", ".ts", ".tsx"):
        resolved_test = test_path or "npm test"
        cmd = ["npm", "test"] + ([str(test_path)] if test_path else [])
    elif ext == ".java":
        resolved_test = test_path or "mvn test"
        if (project_root / "gradlew").is_file() or (project_root / "build.gradle").is_file():
            cmd = ["gradle", "test"]
        else:
            cmd = ["mvn", "test"]
    else:
        # C / C++ / generic build systems (CTest / Make)
        resolved_test = test_path or "ctest"
        if (project_root / "CMakeLists.txt").is_file():
            cmd = ["ctest", "--output-on-failure"]
        else:
            cmd = ["make", "test"]

    return cmd, str(resolved_test), project_root


def run_test_cmd(
    cmd: List[str],
    cwd: Path,
    timeout: int = 15,
) -> Tuple[int, str]:
    """Runs a test command in a subprocess and returns (returncode, combined_output)."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        combined = (proc.stdout + "\n" + proc.stderr).strip()
        return proc.returncode, combined
    except subprocess.TimeoutExpired:
        return -1, f"TimeoutExpired: Test runner exceeded {timeout}s"
    except Exception as exc:
        return -1, f"ExecutionError: {exc}"
