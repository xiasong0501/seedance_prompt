from __future__ import annotations

from pathlib import Path

from prompt_utils import load_prompt
from skill_utils import load_skill
from .tools import (
    list_series_scripts,
    read_repo_file,
    resolve_series_paths,
    run_openai_repo_pipeline,
    write_repo_file,
)

try:
    from agents import Agent, Runner
except ImportError:  # pragma: no cover - runtime dependency
    Agent = None  # type: ignore[assignment]
    Runner = None  # type: ignore[assignment]


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _prompt(name: str) -> str:
    return load_prompt(Path("agents") / name)


def _skill(name: str) -> str:
    return load_skill(Path("openai_flow") / name)


def _template(name: str) -> str:
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8")


def _compose_instructions(base_prompt: str, extras: list[str]) -> str:
    chunks = [base_prompt.strip()]
    for item in extras:
        if item.strip():
            chunks.append(item.strip())
    return "\n\n".join(chunks).strip()


def build_agents(model: str = "gpt-5.4") -> dict[str, Agent]:
    if Agent is None:
        raise RuntimeError(
            "未安装 OpenAI Agents SDK。请先安装相关依赖后再运行 OpenAI agent flow。"
        )

    common_docs = [
        _skill("common/path_scope.md"),
        _skill("common/output_contracts.md"),
        _skill("common/review_rubric.md"),
    ]
    explosive_agent = Agent(
        name="Explosive Agent",
        model=model,
        instructions=_compose_instructions(
            _prompt("explosive.md"),
            common_docs + [_skill("explosive_workflow.md")],
        ),
        tools=[list_series_scripts, read_repo_file, write_repo_file, resolve_series_paths],
    )
    director_agent = Agent(
        name="Director Agent",
        model=model,
        instructions=_compose_instructions(
            _prompt("director.md"),
            common_docs + [_skill("director_workflow.md"), _template("director_analysis_contract.md")],
        ),
        tools=[read_repo_file, write_repo_file, resolve_series_paths],
    )
    art_agent = Agent(
        name="Art Agent",
        model=model,
        instructions=_compose_instructions(
            _prompt("art.md"),
            common_docs + [_skill("art_workflow.md")],
        ),
        tools=[read_repo_file, write_repo_file, resolve_series_paths],
    )
    storyboard_agent = Agent(
        name="Storyboard Agent",
        model=model,
        instructions=_compose_instructions(
            _prompt("storyboard.md"),
            common_docs + [_skill("storyboard_workflow.md"), _template("seedance_prompts_contract.md")],
        ),
        tools=[read_repo_file, write_repo_file, resolve_series_paths],
    )
    producer_agent = Agent(
        name="Producer Agent",
        model=model,
        instructions=_compose_instructions(
            _prompt("producer.md"),
            common_docs + [_skill("producer_workflow.md")],
        ),
        tools=[list_series_scripts, read_repo_file, write_repo_file, resolve_series_paths, run_openai_repo_pipeline],
        handoffs=[explosive_agent, director_agent, art_agent, storyboard_agent],
    )
    return {
        "producer": producer_agent,
        "explosive": explosive_agent,
        "director": director_agent,
        "art": art_agent,
        "storyboard": storyboard_agent,
    }


def run_openai_flow(task: str, model: str = "gpt-5.4") -> str:
    if Runner is None:
        raise RuntimeError(
            "未安装 OpenAI Agents SDK。请先安装相关依赖后再运行 OpenAI agent flow。"
        )
    agents = build_agents(model=model)
    result = Runner.run_sync(agents["producer"], task)
    return getattr(result, "final_output", str(result))
