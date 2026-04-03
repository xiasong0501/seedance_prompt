from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_telemetry import TelemetryRecorder
from genre_routing import load_genre_playbook_library
import run_video_pipeline as single_runner
from providers.base import build_provider_model_tag, derive_series_folder_name, save_json_file, save_text_file
from seedance_learning import build_series_purpose_libraries, is_seedance_learning_enabled


DEFAULT_CONFIG_PATH = Path("config/series_pipeline.local.json")
GENRE_SKILL_ROOT = PROJECT_ROOT / "skills" / "production" / "video-script-reconstruction-skill"
GENRE_DRAFT_ROOT = GENRE_SKILL_ROOT / "genres" / "__drafts__"
LEGACY_PLAYBOOK_DRAFT_ROOT = GENRE_SKILL_ROOT / "playbooks" / "__drafts__"


def parse_bool_flag(raw: str | None) -> bool | None:
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"无法解析布尔值：{raw}")


def print_status(message: str) -> None:
    print(f"[series-pipeline] {message}", flush=True)


def prompt_free_text(prompt: str, default_value: str = "") -> str:
    while True:
        raw = input(f"{prompt}{f'（默认 {default_value}）' if default_value else ''}：").strip()
        if raw:
            return raw
        if default_value:
            return default_value
        print("输入不能为空，请重新输入。")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _known_genre_keys() -> list[str]:
    return [
        str(item.get("genre_key", "")).strip()
        for item in load_genre_playbook_library()
        if str(item.get("genre_key", "")).strip()
    ]


def _sanitize_genre_slug(label: str) -> str:
    clean = re.sub(r"\s+", "_", str(label).strip())
    clean = re.sub(r"[\\\\/:*?\"<>|（）()\[\]{}，,。；;！!？?]+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean[:48] or "genre_draft"


def _split_genres_by_library(genres: list[str]) -> tuple[list[str], list[str]]:
    known = set(_known_genre_keys())
    library_keys: list[str] = []
    custom_tokens: list[str] = []
    for genre in genres:
        text = str(genre).strip()
        if not text:
            continue
        if text in known:
            if text not in library_keys:
                library_keys.append(text)
        elif text not in custom_tokens:
            custom_tokens.append(text)
    return library_keys[:3], custom_tokens[:3]


def scaffold_genre_drafts(genres: list[str], *, episode_id: str, reason: str) -> list[str]:
    created_paths: list[str] = []
    for genre in genres:
        label = str(genre).strip()
        if not label:
            continue
        slug = _sanitize_genre_slug(label)
        package_dir = GENRE_DRAFT_ROOT / slug
        playbook_json_path = package_dir / "playbook.json"
        skill_md_path = package_dir / "skill.md"
        legacy_json_path = LEGACY_PLAYBOOK_DRAFT_ROOT / f"{slug}.json"
        if not playbook_json_path.exists():
            save_json_file(
                playbook_json_path,
                {
                    "genre_key": label,
                    "aliases": [label],
                    "core_audience_promises": [],
                    "script_hooks": [],
                    "character_design_focus": [],
                    "scene_design_focus": [],
                    "storyboard_focus": [],
                    "_draft_meta": {
                        "source_episode": episode_id,
                        "reason": reason,
                        "status": "draft_pending_human_review",
                    },
                },
            )
            created_paths.append(str(playbook_json_path))
        if not skill_md_path.exists():
            save_text_file(
                skill_md_path,
                (
                    f"# 题材补充 Skill：{label}\n\n"
                    f"> 草稿来源：{episode_id}\n"
                    f"> 状态：draft_pending_human_review\n\n"
                    "请根据该剧的真实优点与玩法，补充：\n\n"
                    "- 这个题材最该抓的开头钩子\n"
                    "- 剧本重建时应强化的节奏\n"
                    "- 人物设计时最应突出的气质与身份\n"
                    "- 场景设计时最应保留的空间压强\n"
                    "- 分镜时最该抓的传播点与卡点\n"
                ),
            )
            created_paths.append(str(skill_md_path))
        if not legacy_json_path.exists():
            save_json_file(
                legacy_json_path,
                {
                    "genre_key": label,
                    "aliases": [label],
                    "_draft_meta": {
                        "source_episode": episode_id,
                        "reason": reason,
                        "status": "draft_pending_human_review",
                    },
                },
            )
            created_paths.append(str(legacy_json_path))
    return created_paths


def apply_series_genre_hints(
    config: dict[str, Any],
    *,
    library_keys: list[str],
    custom_tokens: list[str],
    source: str,
) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    updated.setdefault("series", {})
    updated["series"]["genre_hints"] = {
        "library_keys": library_keys,
        "custom_tokens": custom_tokens,
        "ai_suggested_keys": list(updated.get("series", {}).get("genre_hints", {}).get("ai_suggested_keys", [])),
        "source": source,
    }
    return updated


def maybe_handle_genre_override(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    override = dict(result.get("genre_override_request", {}))
    proposed = [
        str(item).strip()
        for item in [
            override.get("proposed_primary_genre", ""),
            *override.get("proposed_secondary_genres", []),
            *override.get("proposed_new_genres", []),
        ]
        if str(item).strip()
    ]
    if not override.get("needs_user_confirmation") or not proposed:
        return config

    draft_paths = scaffold_genre_drafts(
        list(override.get("proposed_new_genres", [])),
        episode_id=str(result.get("episode_id", "")),
        reason=str(override.get("reason", "")),
    )
    print_status("AI 提出了额外题材建议：" + "、".join(proposed))
    print_status("当前说明：" + str(override.get("reason", "")))
    if draft_paths:
        print_status("已为新题材生成草稿包：")
        for path in draft_paths:
            print_status(f"  - {path}")

    if not bool(config.get("run", {}).get("prompt_on_genre_override", False)):
        return config

    run_state = config.setdefault("run", {})
    existing_policy = str(run_state.get("_genre_override_followup_policy", "")).strip()
    if existing_policy == "keep_current":
        return config
    if existing_policy == "accept_ai":
        library_keys, custom_tokens = _split_genres_by_library(proposed[:3])
        updated = apply_series_genre_hints(
            config,
            library_keys=library_keys,
            custom_tokens=custom_tokens,
            source="interactive_ai_override_accepted",
        )
        updated.setdefault("run", {})
        updated["run"]["_genre_override_followup_policy"] = existing_policy
        print_status(
            "沿用本轮题材处理选择，自动采用 AI 建议更新后续集数题材："
            + ("、".join(library_keys) if library_keys else "自定义 " + "、".join(custom_tokens))
        )
        return updated
    if existing_policy == "manual_fixed":
        return config

    print_status("请选择后续集数的题材处理方式：")
    print("  1. 保持当前用户确认题材")
    print("  2. 采用 AI 建议覆盖后续集数题材")
    print("  3. 手动重新输入后续集数题材")
    while True:
        choice = input("请输入序号（默认 1）：").strip() or "1"
        if choice in {"1", "2", "3"}:
            break
        print("输入无效，请重新输入。")

    updated = copy.deepcopy(config)
    updated.setdefault("run", {})
    if choice == "1":
        updated["run"]["_genre_override_followup_policy"] = "keep_current"
        return updated

    if choice == "2":
        library_keys, custom_tokens = _split_genres_by_library(proposed[:3])
        updated = apply_series_genre_hints(
            updated,
            library_keys=library_keys,
            custom_tokens=custom_tokens,
            source="interactive_ai_override_accepted",
        )
        updated["run"]["_genre_override_followup_policy"] = "accept_ai"
        print_status(
            "后续集数题材已更新为："
            + ("、".join(library_keys) if library_keys else "自定义 " + "、".join(custom_tokens))
        )
        return updated

    manual_raw = prompt_free_text("请输入新的题材（空格隔开，例如：重生 替嫁 爱情）")
    manual_genres = [item.strip() for item in manual_raw.split() if item.strip()][:3]
    library_keys, custom_tokens = _split_genres_by_library(manual_genres)
    draft_paths = scaffold_genre_drafts(
        custom_tokens,
        episode_id=str(result.get("episode_id", "")),
        reason="用户手动确认的新题材",
    )
    updated = apply_series_genre_hints(
        updated,
        library_keys=library_keys,
        custom_tokens=custom_tokens,
        source="interactive_manual_override",
    )
    updated["run"]["_genre_override_followup_policy"] = "manual_fixed"
    if draft_paths:
        print_status("已为手动新增题材生成草稿包：")
        for path in draft_paths:
            print_status(f"  - {path}")
    print_status(
        "后续集数题材已更新为："
        + ("、".join(library_keys) if library_keys else "自定义 " + "、".join(custom_tokens))
    )
    return updated


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_batch_config(path: str | Path) -> dict[str, Any]:
    batch_config = load_json(path)
    base_path = batch_config.get("base_config")
    if base_path:
        base_config = load_json(base_path)
        return deep_merge(base_config, batch_config)
    return batch_config


def extract_episode_number(file_name: str, custom_regex: str | None = None) -> int | None:
    patterns: list[re.Pattern[str]] = []
    if custom_regex:
        patterns.append(re.compile(custom_regex, re.IGNORECASE))

    patterns.extend(
        [
            re.compile(r"^0*(\d+)(?:[._\-\s]|$)", re.IGNORECASE),
            re.compile(r"第\s*0*(\d+)\s*[集话回]", re.IGNORECASE),
            re.compile(r"\bep(?:isode)?\s*0*(\d+)\b", re.IGNORECASE),
            re.compile(r"\be\s*0*(\d+)\b", re.IGNORECASE),
            re.compile(r"[\(\[【]\s*0*(\d+)\s*[\)\]】]", re.IGNORECASE),
            re.compile(r"(?<!\d)0*(\d+)(?!\d)", re.IGNORECASE),
        ]
    )

    for pattern in patterns:
        match = pattern.search(file_name)
        if not match:
            continue
        try:
            return int(match.group(1))
        except (IndexError, ValueError):
            continue
    return None


def discover_episode_files(config: dict[str, Any]) -> list[dict[str, Any]]:
    series_config = config["series"]
    video_dir = Path(series_config["video_dir"]).expanduser().resolve()
    if not video_dir.exists():
        raise FileNotFoundError(f"视频目录不存在：{video_dir}")

    file_extensions = {
        item.lower()
        for item in series_config.get("file_extensions", [".mp4", ".mov", ".mkv"])
    }
    custom_regex = series_config.get("episode_number_regex")
    start_episode = int(series_config["start_episode"])
    end_episode = int(series_config["end_episode"])
    if start_episode > end_episode:
        raise ValueError("start_episode 不能大于 end_episode。")

    episodes: list[dict[str, Any]] = []
    for path in sorted(video_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in file_extensions:
            continue
        episode_number = extract_episode_number(path.stem, custom_regex)
        if episode_number is None:
            continue
        if episode_number < start_episode or episode_number > end_episode:
            continue
        episodes.append(
            {
                "episode_number": episode_number,
                "path": path.resolve(),
            }
        )

    episodes.sort(key=lambda item: item["episode_number"])
    return episodes


def build_episode_title(series_config: dict[str, Any], episode_number: int, video_path: Path) -> str:
    title_template = series_config.get("title_template", "第{num}集")
    return title_template.format(
        num=episode_number,
        number=episode_number,
        file_stem=video_path.stem,
        file_name=video_path.name,
    )


def build_episode_id(series_config: dict[str, Any], episode_number: int) -> str:
    prefix = series_config.get("episode_id_prefix", "ep")
    padding = int(series_config.get("episode_id_padding", 2))
    return f"{prefix}{episode_number:0{padding}d}"


def build_episode_runtime_config(config: dict[str, Any], episode_number: int, video_path: Path) -> dict[str, Any]:
    runtime_config = copy.deepcopy(config)
    run_config = runtime_config.setdefault("run", {})
    series_config = runtime_config["series"]
    genre_hints = copy.deepcopy(series_config.get("genre_hints", {}))

    run_config["episode_id"] = build_episode_id(series_config, episode_number)
    run_config["title"] = build_episode_title(series_config, episode_number, video_path)
    run_config["video_path"] = str(video_path)
    run_config["series_name"] = series_config.get("series_name") or derive_series_folder_name(video_path=video_path)
    run_config["genre_hints"] = genre_hints

    pipeline_config = runtime_config.get("pipeline", {})
    if "temperature" in pipeline_config:
        run_config["temperature"] = pipeline_config["temperature"]

    return runtime_config


def render_metrics_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 单集统计报告",
        "",
        f"- run_name：{report.get('run_name', '')}",
        f"- final_status：{report.get('context', {}).get('final_status', '')}",
        f"- episode_id：{report.get('context', {}).get('episode_id', '')}",
        f"- provider/model：{report.get('context', {}).get('provider', '')}/{report.get('context', {}).get('model', '')}",
        "",
        "## 总计",
        "",
        f"- steps：{report.get('totals', {}).get('step_count', 0)}",
        f"- duration_seconds：{report.get('totals', {}).get('duration_seconds', 0)}",
        f"- input_tokens：{report.get('totals', {}).get('input_tokens', 0)}",
        f"- output_tokens：{report.get('totals', {}).get('output_tokens', 0)}",
        f"- total_tokens：{report.get('totals', {}).get('total_tokens', 0)}",
        "",
        "## 阶段汇总",
        "",
        "| 阶段 | 步骤数 | 耗时(秒) | 输入tokens | 输出tokens | 总tokens |",
        "|------|--------|---------:|-----------:|-----------:|---------:|",
    ]
    for stage, totals in report.get("stage_totals", {}).items():
        lines.append(
            f"| {stage} | {totals.get('step_count', 0)} | {totals.get('duration_seconds', 0)} | "
            f"{totals.get('input_tokens', 0)} | {totals.get('output_tokens', 0)} | {totals.get('total_tokens', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 细粒度步骤",
            "",
            "| Step ID | 阶段 | 名称 | 状态 | 耗时(秒) | 输入tokens | 输出tokens | 总tokens |",
            "|---------|------|------|------|---------:|-----------:|-----------:|---------:|",
        ]
    )
    for step in report.get("steps", []):
        lines.append(
            f"| {step.get('step_id', '')} | {step.get('stage', '')} | {step.get('name', '')} | {step.get('status', '')} | "
            f"{step.get('duration_seconds', 0)} | {step.get('input_tokens', 0)} | {step.get('output_tokens', 0)} | {step.get('total_tokens', 0)} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def metrics_paths(
    runtime_config: dict[str, Any],
    *,
    provider: str,
    model: str,
    episode_id: str,
) -> tuple[Path, Path]:
    series_name = runtime_config["run"].get("series_name") or derive_series_folder_name(
        video_path=runtime_config["run"].get("video_path")
    )
    tag = build_provider_model_tag(provider, model)
    base_dir = (
        Path(runtime_config["run"].get("analysis_root", "analysis"))
        / series_name
        / episode_id
        / "metrics"
    ).expanduser().resolve()
    return (
        base_dir / f"episode_metrics__{tag}.json",
        base_dir / f"episode_metrics__{tag}.md",
    )


def save_episode_metrics(recorder: TelemetryRecorder, json_path: Path, md_path: Path) -> None:
    report = recorder.to_dict()
    save_json_file(json_path, report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_metrics_markdown(report), encoding="utf-8")


def run_single_episode(runtime_config: dict[str, Any], telemetry: TelemetryRecorder | None = None) -> dict[str, Any]:
    provider, model = single_runner.configure_provider_env(runtime_config)
    print_status(
        f"开始处理 {runtime_config['run']['episode_id']} | provider={provider} | model={model}"
    )
    preprocess_manifest: dict[str, Any] | None = None
    bundle = None
    try:
        preprocess_manifest = single_runner.preprocess_if_needed(runtime_config, telemetry=telemetry)
        if runtime_config["run"].get("only_preprocess", False):
            return {
                "status": "preprocess_only_completed",
                "episode_id": runtime_config["run"]["episode_id"],
                "provider": provider,
                "model": model,
                "preprocess_manifest": preprocess_manifest,
            }

        bundle = single_runner.build_bundle(runtime_config, preprocess_manifest)
        summary = single_runner.run_pipeline(runtime_config, provider, model, bundle, telemetry=telemetry)
        summary["status"] = "completed"
        summary["episode_id"] = runtime_config["run"]["episode_id"]
        summary["video_path"] = runtime_config["run"]["video_path"]
        summary["genre_override_request"] = dict(summary.get("genre_debug_summary", {}).get("genre_override_request", {}))
        return summary
    except Exception as exc:
        if isinstance(exc, getattr(single_runner, "PipelineExecutionError")):
            raise
        report, json_path, md_path = single_runner.save_failure_report(
            runtime_config,
            provider,
            model,
            preprocess_manifest,
            bundle,
            exc,
        )
        raise single_runner.PipelineExecutionError(
            f"{report['error_message']} | 调试报告：{md_path}",
            report=report,
            json_path=json_path,
            markdown_path=md_path,
        ) from exc


def build_batch_summary_path(config: dict[str, Any], episode_items: list[dict[str, Any]]) -> Path:
    selected_provider = config["run"]["selected_provider"]
    model = config["providers"][selected_provider]["model"]
    output_tag = build_provider_model_tag(selected_provider, model)
    series_name = config["series"].get("series_name") or derive_series_folder_name(
        video_path=config["series"]["video_dir"]
    )
    first_ep = build_episode_id(config["series"], episode_items[0]["episode_number"])
    last_ep = build_episode_id(config["series"], episode_items[-1]["episode_number"])
    return (
        Path(config["run"].get("analysis_root", "analysis"))
        / series_name
        / "batch_runs"
        / f"batch_summary__{output_tag}__{first_ep}-{last_ep}.json"
    )


def build_metrics_summary_paths(config: dict[str, Any], episode_items: list[dict[str, Any]]) -> tuple[Path, Path]:
    selected_provider = config["run"]["selected_provider"]
    model = config["providers"][selected_provider]["model"]
    output_tag = build_provider_model_tag(selected_provider, model)
    series_name = config["series"].get("series_name") or derive_series_folder_name(
        video_path=config["series"]["video_dir"]
    )
    first_ep = build_episode_id(config["series"], episode_items[0]["episode_number"])
    last_ep = build_episode_id(config["series"], episode_items[-1]["episode_number"])
    base = (
        Path(config["run"].get("analysis_root", "analysis"))
        / series_name
        / "batch_runs"
        / f"metrics_summary__{output_tag}__{first_ep}-{last_ep}"
    )
    return Path(f"{base}.json"), Path(f"{base}.md")


def render_batch_metrics_markdown(summary: dict[str, Any]) -> str:
    totals = summary.get("totals", {})
    lines = [
        "# 批量统计报告",
        "",
        f"- 剧名：{summary.get('series_name', '')}",
        f"- provider/model：{summary.get('selected_provider', '')}/{summary.get('model', '')}",
        f"- 集数范围：{summary.get('start_episode', '')}-{summary.get('end_episode', '')}",
        "",
        "## 总计",
        "",
        f"- episode_count：{totals.get('episode_count', 0)}",
        f"- succeeded：{totals.get('succeeded', 0)}",
        f"- failed：{totals.get('failed', 0)}",
        f"- duration_seconds：{totals.get('duration_seconds', 0)}",
        f"- input_tokens：{totals.get('input_tokens', 0)}",
        f"- output_tokens：{totals.get('output_tokens', 0)}",
        f"- total_tokens：{totals.get('total_tokens', 0)}",
        "",
        "## 分集汇总",
        "",
        "| 集数 | 状态 | 耗时(秒) | 输入tokens | 输出tokens | 总tokens | 报告 |",
        "|------|------|---------:|-----------:|-----------:|---------:|------|",
    ]
    for item in summary.get("episodes", []):
        lines.append(
            f"| {item.get('episode_id', '')} | {item.get('status', '')} | {item.get('duration_seconds', 0)} | "
            f"{item.get('input_tokens', 0)} | {item.get('output_tokens', 0)} | {item.get('total_tokens', 0)} | "
            f"{item.get('metrics_json_path', '')} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_genre_override_report_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    selected_provider = config["run"]["selected_provider"]
    model = config["providers"][selected_provider]["model"]
    output_tag = build_provider_model_tag(selected_provider, model)
    series_name = config["series"].get("series_name") or derive_series_folder_name(
        video_path=config["series"]["video_dir"]
    )
    base = (
        Path(config["run"].get("analysis_root", "analysis"))
        / series_name
        / f"genre_override_request__{output_tag}"
    )
    return Path(f"{base}.json"), Path(f"{base}.md")


def build_series_genre_override_report(config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    series_name = config["series"].get("series_name") or derive_series_folder_name(
        video_path=config["series"]["video_dir"]
    )
    selected_provider = config["run"]["selected_provider"]
    model = config["providers"][selected_provider]["model"]
    confirmed_genres = list(config.get("series", {}).get("genre_hints", {}).get("library_keys", [])) + list(
        config.get("series", {}).get("genre_hints", {}).get("custom_tokens", [])
    )
    episode_items: list[dict[str, Any]] = []
    aggregated_new_genres: list[str] = []
    aggregated_proposed_labels: list[str] = []
    for item in results:
        debug_summary = dict(item.get("genre_debug_summary", {}))
        override = dict(item.get("genre_override_request", {}))
        proposed_new = [str(value).strip() for value in override.get("proposed_new_genres", []) if str(value).strip()]
        proposed_secondary = [
            str(value).strip() for value in override.get("proposed_secondary_genres", []) if str(value).strip()
        ]
        proposed_primary = str(override.get("proposed_primary_genre", "")).strip()
        for value in [proposed_primary, *proposed_secondary, *proposed_new]:
            if value and value not in aggregated_proposed_labels:
                aggregated_proposed_labels.append(value)
        for value in proposed_new:
            if value not in aggregated_new_genres:
                aggregated_new_genres.append(value)
        episode_items.append(
            {
                "episode_id": item.get("episode_id", ""),
                "current_primary_genre": debug_summary.get("primary_genre", ""),
                "current_secondary_genres": debug_summary.get("secondary_genres", []),
                "confirmed_user_genres": debug_summary.get("confirmed_user_genres", []),
                "genre_resolution_mode": debug_summary.get("genre_resolution_mode", ""),
                "needs_user_confirmation": bool(override.get("needs_user_confirmation", False)),
                "proposed_primary_genre": proposed_primary,
                "proposed_secondary_genres": proposed_secondary,
                "proposed_new_genres": proposed_new,
                "reason": str(override.get("reason", "")).strip(),
                "genre_debug_markdown_path": item.get("genre_debug_paths", {}).get("markdown_path", ""),
            }
        )
    return {
        "series_name": series_name,
        "provider": selected_provider,
        "model": model,
        "confirmed_genres": confirmed_genres,
        "needs_user_confirmation": any(item.get("needs_user_confirmation", False) for item in episode_items),
        "aggregated_proposed_labels": aggregated_proposed_labels,
        "aggregated_new_genres": aggregated_new_genres,
        "episodes": episode_items,
    }


def render_series_genre_override_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 整剧题材修正建议汇总",
        "",
        f"- 剧名：{report.get('series_name', '')}",
        f"- provider/model：{report.get('provider', '')}/{report.get('model', '')}",
        f"- 用户已确认题材：{'、'.join(report.get('confirmed_genres', [])) or '无'}",
        f"- 是否存在 AI 修正建议：{report.get('needs_user_confirmation', False)}",
        f"- 全剧汇总建议标签：{'、'.join(report.get('aggregated_proposed_labels', [])) or '无'}",
        f"- 全剧建议新增题材：{'、'.join(report.get('aggregated_new_genres', [])) or '无'}",
        "",
        "## 分集明细",
        "",
    ]
    for item in report.get("episodes", []):
        lines.extend(
            [
                f"### {item.get('episode_id', '')}",
                "",
                f"- 当前锁定主题材：{item.get('current_primary_genre', '')}",
                f"- 当前锁定副题材：{'、'.join(item.get('current_secondary_genres', [])) or '无'}",
                f"- 用户确认题材：{'、'.join(item.get('confirmed_user_genres', [])) or '无'}",
                f"- 决议模式：{item.get('genre_resolution_mode', '')}",
                f"- AI 是否请求修正：{item.get('needs_user_confirmation', False)}",
                f"- AI 建议主题材：{item.get('proposed_primary_genre', '') or '无'}",
                f"- AI 建议副题材：{'、'.join(item.get('proposed_secondary_genres', [])) or '无'}",
                f"- AI 建议新增题材：{'、'.join(item.get('proposed_new_genres', [])) or '无'}",
                f"- 原因：{item.get('reason', '') or '无'}",
                f"- 调试报告：{item.get('genre_debug_markdown_path', '') or '无'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def save_series_genre_override_report(config: dict[str, Any], results: list[dict[str, Any]]) -> tuple[Path, Path]:
    report = build_series_genre_override_report(config, results)
    json_path, md_path = build_genre_override_report_paths(config)
    save_json_file(json_path, report)
    save_text_file(md_path, render_series_genre_override_markdown(report))
    return json_path, md_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run video-to-script pipeline for a whole series range.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--collect-metrics", choices=["true", "false"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print_status(f"加载批量配置：{args.config}")
    config = load_batch_config(args.config)
    collect_metrics_override = parse_bool_flag(args.collect_metrics)
    if collect_metrics_override is not None:
        config.setdefault("run", {})
        config["run"]["collect_metrics"] = collect_metrics_override
    episode_items = discover_episode_files(config)
    if not episode_items:
        raise RuntimeError("没有在指定目录下找到符合范围和命名规则的视频文件。")

    print_status(
        f"发现 {len(episode_items)} 集待处理：{episode_items[0]['episode_number']} -> {episode_items[-1]['episode_number']}"
    )

    if config["run"].get("dry_run", False):
        preview = [
            {
                "episode_number": item["episode_number"],
                "episode_id": build_episode_id(config["series"], item["episode_number"]),
                "video_path": str(item["path"]),
                "title": build_episode_title(config["series"], item["episode_number"], item["path"]),
                "genre_hints": config["series"].get("genre_hints", {}),
            }
            for item in episode_items
        ]
        print(json.dumps({"status": "dry_run", "episodes": preview}, ensure_ascii=False, indent=2))
        return

    results: list[dict[str, Any]] = []
    continue_on_error = bool(config["run"].get("continue_on_error", False))
    collect_metrics = bool(config["run"].get("collect_metrics", False))
    metrics_results: list[dict[str, Any]] = []

    for item in episode_items:
        runtime_config = build_episode_runtime_config(config, item["episode_number"], item["path"])
        selected_provider = runtime_config["run"]["selected_provider"]
        selected_model = runtime_config["providers"][selected_provider]["model"]
        recorder: TelemetryRecorder | None = None
        metrics_json_path: Path | None = None
        metrics_md_path: Path | None = None
        if collect_metrics:
            recorder = TelemetryRecorder(
                run_name=f"series-pipeline-{runtime_config['run']['episode_id']}",
                context={
                    "series_name": runtime_config["run"].get("series_name", ""),
                    "episode_id": runtime_config["run"]["episode_id"],
                    "video_path": runtime_config["run"]["video_path"],
                    "provider": selected_provider,
                    "model": selected_model,
                },
            )
            metrics_json_path, metrics_md_path = metrics_paths(
                runtime_config,
                provider=selected_provider,
                model=selected_model,
                episode_id=runtime_config["run"]["episode_id"],
            )
        try:
            result = run_single_episode(runtime_config, telemetry=recorder)
            if recorder and metrics_json_path and metrics_md_path:
                recorder.context["final_status"] = result["status"]
                save_episode_metrics(recorder, metrics_json_path, metrics_md_path)
                metrics_report = recorder.to_dict()
                result["metrics_json_path"] = str(metrics_json_path)
                result["metrics_markdown_path"] = str(metrics_md_path)
                result["duration_seconds"] = metrics_report["totals"]["duration_seconds"]
                result["input_tokens"] = metrics_report["totals"]["input_tokens"]
                result["output_tokens"] = metrics_report["totals"]["output_tokens"]
                result["total_tokens"] = metrics_report["totals"]["total_tokens"]
                metrics_results.append(
                    {
                        "episode_id": result["episode_id"],
                        "status": result["status"],
                        "duration_seconds": result["duration_seconds"],
                        "input_tokens": result["input_tokens"],
                        "output_tokens": result["output_tokens"],
                        "total_tokens": result["total_tokens"],
                        "metrics_json_path": str(metrics_json_path),
                    }
                )
                print_status(
                    f"{result['episode_id']} 统计：耗时 {result['duration_seconds']}s | "
                    f"tokens in/out/total = {result['input_tokens']}/{result['output_tokens']}/{result['total_tokens']}"
                )
            results.append(result)
            config = maybe_handle_genre_override(config, result)
        except Exception as exc:
            error_item = {
                "status": "failed",
                "episode_id": runtime_config["run"]["episode_id"],
                "video_path": runtime_config["run"]["video_path"],
                "error": str(exc),
            }
            failure_report_path = ""
            failure_report_json_path = ""
            if isinstance(exc, getattr(single_runner, "PipelineExecutionError")):
                failure_report_path = str(getattr(exc, "markdown_path", "") or "")
                failure_report_json_path = str(getattr(exc, "json_path", "") or "")
                if failure_report_path:
                    error_item["failure_report_markdown_path"] = failure_report_path
                if failure_report_json_path:
                    error_item["failure_report_json_path"] = failure_report_json_path
                report = dict(getattr(exc, "report", {}) or {})
                if report:
                    error_item["failure_debug_summary"] = {
                        "error_type": report.get("error_type", ""),
                        "frame_count": report.get("input_summary", {}).get("frame_count", 0),
                        "transcript_chars": report.get("input_summary", {}).get("transcript_chars", 0),
                        "ocr_chars": report.get("input_summary", {}).get("ocr_chars", 0),
                        "timeout_seconds": report.get("runtime_config", {}).get("timeout_seconds", 0),
                    }
            if recorder and metrics_json_path and metrics_md_path:
                recorder.context["final_status"] = "failed"
                recorder.context["failure_message"] = str(exc)
                save_episode_metrics(recorder, metrics_json_path, metrics_md_path)
                metrics_report = recorder.to_dict()
                error_item["metrics_json_path"] = str(metrics_json_path)
                error_item["metrics_markdown_path"] = str(metrics_md_path)
                error_item["duration_seconds"] = metrics_report["totals"]["duration_seconds"]
                error_item["input_tokens"] = metrics_report["totals"]["input_tokens"]
                error_item["output_tokens"] = metrics_report["totals"]["output_tokens"]
                error_item["total_tokens"] = metrics_report["totals"]["total_tokens"]
                metrics_results.append(
                    {
                        "episode_id": error_item["episode_id"],
                        "status": "failed",
                        "duration_seconds": error_item["duration_seconds"],
                        "input_tokens": error_item["input_tokens"],
                        "output_tokens": error_item["output_tokens"],
                        "total_tokens": error_item["total_tokens"],
                        "metrics_json_path": str(metrics_json_path),
                    }
                )
            results.append(error_item)
            print_status(f"{runtime_config['run']['episode_id']} 失败：{exc}")
            if failure_report_path:
                print_status(f"{runtime_config['run']['episode_id']} 调试报告：{failure_report_path}")
            if not continue_on_error:
                break

    summary = {
        "selected_provider": config["run"]["selected_provider"],
        "model": config["providers"][config["run"]["selected_provider"]]["model"],
        "video_dir": str(Path(config["series"]["video_dir"]).expanduser().resolve()),
        "series_name": config["series"].get("series_name") or derive_series_folder_name(video_path=config["series"]["video_dir"]),
        "start_episode": int(config["series"]["start_episode"]),
        "end_episode": int(config["series"]["end_episode"]),
        "collect_metrics": collect_metrics,
        "results": results,
    }

    summary_path = build_batch_summary_path(config, episode_items)
    if collect_metrics:
        totals = {
            "episode_count": len(metrics_results),
            "succeeded": sum(1 for item in metrics_results if item["status"] in {"completed", "preprocess_only_completed"}),
            "failed": sum(1 for item in metrics_results if item["status"] not in {"completed", "preprocess_only_completed"}),
            "duration_seconds": round(sum(float(item.get("duration_seconds", 0)) for item in metrics_results), 3),
            "input_tokens": sum(int(item.get("input_tokens", 0)) for item in metrics_results),
            "output_tokens": sum(int(item.get("output_tokens", 0)) for item in metrics_results),
            "total_tokens": sum(int(item.get("total_tokens", 0)) for item in metrics_results),
        }
        metrics_summary = {
            "series_name": summary["series_name"],
            "selected_provider": summary["selected_provider"],
            "model": summary["model"],
            "start_episode": summary["start_episode"],
            "end_episode": summary["end_episode"],
            "totals": totals,
            "episodes": metrics_results,
        }
        metrics_summary_json_path, metrics_summary_md_path = build_metrics_summary_paths(config, episode_items)
        save_json_file(metrics_summary_json_path, metrics_summary)
        metrics_summary_md_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_summary_md_path.write_text(render_batch_metrics_markdown(metrics_summary), encoding="utf-8")
        summary["metrics_summary_json_path"] = str(metrics_summary_json_path)
        summary["metrics_summary_markdown_path"] = str(metrics_summary_md_path)
        print_status(
            f"批量统计：总耗时 {totals['duration_seconds']}s | "
            f"tokens in/out/total = {totals['input_tokens']}/{totals['output_tokens']}/{totals['total_tokens']}"
        )
        print_status(f"统计报告已写入：{metrics_summary_json_path}")
    if is_seedance_learning_enabled(config):
        try:
            library_artifacts = build_series_purpose_libraries(
                project_root=PROJECT_ROOT,
                series_name=summary["series_name"],
                config=config,
            )
            if library_artifacts:
                summary["seedance_learning_library"] = library_artifacts
                print_status(f"Seedance 技能库已写入：{library_artifacts['skill_library_json_path']}")
                print_status(f"Seedance 模板库已写入：{library_artifacts['template_library_json_path']}")
                if library_artifacts.get("prompt_library_root"):
                    print_status(
                        f"Prompt Library 已导出：{library_artifacts['prompt_library_root']}"
                        + (
                            f"｜模板数 {library_artifacts.get('prompt_library_template_count', 0)}"
                            if library_artifacts.get("prompt_library_template_count") is not None
                            else ""
                        )
                    )
                if library_artifacts.get("prompt_library_index_markdown_path"):
                    print_status(
                        f"Prompt Library 检索索引已写入：{library_artifacts['prompt_library_index_markdown_path']}"
                    )
        except Exception as exc:  # pragma: no cover - learning library is best-effort
            summary["seedance_learning_library_error"] = str(exc)
            print_status(f"Seedance 学习库聚合失败，但不影响主流程：{exc}")
    genre_override_json_path, genre_override_md_path = save_series_genre_override_report(config, results)
    summary["genre_override_report_json_path"] = str(genre_override_json_path)
    summary["genre_override_report_markdown_path"] = str(genre_override_md_path)
    save_json_file(summary_path, summary)
    print_status(f"批量结果已写入：{summary_path}")
    print_status(f"整剧题材修正建议已写入：{genre_override_md_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
