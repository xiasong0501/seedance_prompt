from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from agents import function_tool
except ImportError:  # pragma: no cover - runtime dependency
    def function_tool(func):  # type: ignore[misc]
        return func


def _resolve_repo_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if PROJECT_ROOT not in candidate.parents and candidate != PROJECT_ROOT:
        raise ValueError(f"路径超出项目根目录：{candidate}")
    return candidate


def _repo_python_bin() -> Path:
    preferred = PROJECT_ROOT / ".venv" / "bin" / "python"
    return preferred if preferred.exists() else Path(sys.executable).resolve()


@function_tool
def read_repo_file(path: str) -> str:
    target = _resolve_repo_path(path)
    if not target.exists():
        return f"[missing] {target}"
    return target.read_text(encoding="utf-8")


@function_tool
def write_repo_file(path: str, content: str) -> str:
    target = _resolve_repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content.rstrip() + "\n", encoding="utf-8")
    return str(target)


@function_tool
def list_series_scripts(series_dir: str) -> list[str]:
    directory = _resolve_repo_path(series_dir)
    if not directory.exists():
        return []
    return [str(path) for path in sorted(directory.glob("*.md"))]


@function_tool
def resolve_series_paths(series_name: str, episode_id: str, assets_suffix: str = "-gpt") -> dict[str, Any]:
    return {
        "script_dir": str((PROJECT_ROOT / "script" / series_name).resolve()),
        "analysis_dir": str((PROJECT_ROOT / "analysis" / series_name).resolve()),
        "outputs_episode_dir": str((PROJECT_ROOT / "outputs" / series_name / episode_id).resolve()),
        "assets_dir": str((PROJECT_ROOT / "assets" / f"{series_name}{assets_suffix}").resolve()),
        "director_analysis_path": str((PROJECT_ROOT / "outputs" / series_name / episode_id / "01-director-analysis.md").resolve()),
        "storyboard_path": str((PROJECT_ROOT / "outputs" / series_name / episode_id / "02-seedance-prompts.md").resolve()),
        "character_prompts_path": str((PROJECT_ROOT / "assets" / f"{series_name}{assets_suffix}" / "character-prompts.md").resolve()),
        "scene_prompts_path": str((PROJECT_ROOT / "assets" / f"{series_name}{assets_suffix}" / "scene-prompts.md").resolve()),
    }


@function_tool
def run_openai_repo_pipeline(config_path: str = "config/openai_agent_flow.local.json") -> str:
    config = _resolve_repo_path(config_path)
    command = [
        str(_repo_python_bin()),
        str((PROJECT_ROOT / "scripts" / "run_openai_agent_flow.py").resolve()),
        "--config",
        str(config),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or "OpenAI repo pipeline finished."
