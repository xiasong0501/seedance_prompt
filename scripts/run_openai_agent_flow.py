from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from openai_agents.runtime_utils import build_episode_ids, deep_merge, load_runtime_config
from pipeline_telemetry import TelemetryRecorder, telemetry_span
from genre_reference_bundle import bundle_paths, load_or_build_genre_reference_bundle
from providers.base import build_provider_model_tag, save_json_file, utc_timestamp
from generate_director_analysis import (
    choose_script_path,
    resolve_series_name as resolve_director_series_name,
    run_pipeline as run_director_pipeline,
)
from generate_art_assets import run_pipeline as run_art_pipeline
from generate_explosive_rewrites import run_pipeline as run_explosive_pipeline
from generate_seedance_prompt_refine import run_pipeline as run_seedance_prompt_refine_pipeline
from generate_seedance_prompts import run_pipeline as run_storyboard_pipeline


DEFAULT_CONFIG_PATH = Path("config/openai_agent_flow.local.json")


def print_status(message: str) -> None:
    print(f"[openai-agent-flow] {message}", flush=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_bool_flag(raw: str | None) -> bool | None:
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"无法解析布尔值：{raw}")


def resolve_series_name(config: Mapping[str, Any]) -> str:
    explicit = str(config.get("series", {}).get("series_name") or "").strip()
    if explicit:
        return explicit
    script_series_dir = str(config.get("series", {}).get("script_series_dir") or "").strip()
    if script_series_dir:
        return Path(script_series_dir).expanduser().resolve().name
    return resolve_director_series_name(
        {
            "series": {"series_name": ""},
            "script": {
                "series_dir": str(config.get("series", {}).get("script_series_dir") or ""),
                "script_path": str(config.get("source", {}).get("script_path_override") or ""),
            },
        }
    )


def episode_ids(config: Mapping[str, Any]) -> list[str]:
    return build_episode_ids(config.get("series", {}))


def episode_number(episode_id: str) -> int:
    match = re.search(r"(\d+)$", episode_id)
    if not match:
        raise ValueError(f"无法从集数标识提取数字：{episode_id}")
    return int(match.group(1))


def selected_script_preview(config: Mapping[str, Any]) -> dict[str, str]:
    selection_config = {
        "series": {"series_name": config.get("series", {}).get("series_name", "")},
        "script": {
            "series_dir": config.get("series", {}).get("script_series_dir", ""),
            "script_path": config.get("source", {}).get("script_path_override", ""),
            "episode_id": "",
            "preferred_filename_suffixes": config.get("source", {}).get("preferred_filename_suffixes", []),
        },
    }
    result: dict[str, str] = {}
    for episode_id in episode_ids(config):
        result[episode_id] = str(choose_script_path(selection_config, episode_id))
    return result


def outputs_root(config: Mapping[str, Any]) -> Path:
    root = Path(config.get("output", {}).get("outputs_root", "outputs")).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    return root


def output_series_name(config: Mapping[str, Any], series_name: str) -> str:
    output_cfg = config.get("output", {})
    explicit = str(output_cfg.get("outputs_series_name") or "").strip()
    suffix = str(output_cfg.get("outputs_series_suffix") or "-gpt").strip()
    return explicit or f"{series_name}{suffix}"


def assets_dir(config: Mapping[str, Any], series_name: str) -> Path:
    explicit_assets_series_name = str(config.get("quality", {}).get("assets_series_name") or "").strip()
    assets_series_suffix = str(config.get("quality", {}).get("assets_series_suffix") or "-gpt").strip()
    target_series_name = explicit_assets_series_name or f"{series_name}{assets_series_suffix}"
    return (PROJECT_ROOT / "assets" / target_series_name).resolve()


def explosive_output_path(config: Mapping[str, Any], series_name: str, episode_id: str) -> Path:
    model = str(config.get("provider", {}).get("model", "gpt-5.4")).strip()
    suffix = str(config.get("explosive", {}).get("rewrite_filename_suffix") or "__explosive").strip()
    script_series_dir = Path(config["series"]["script_series_dir"]).expanduser().resolve()
    return script_series_dir / f"{episode_id}__openai__{model}{suffix}.md"


def has_existing_explosive_variant(config: Mapping[str, Any], episode_id: str) -> bool:
    script_series_dir = Path(config["series"]["script_series_dir"]).expanduser().resolve()
    if not script_series_dir.exists():
        return False
    for candidate in sorted(script_series_dir.glob(f"{episode_id}*.md")):
        if candidate.name.endswith("__explosive.md"):
            return True
    return False


def director_output_path(config: Mapping[str, Any], series_name: str, episode_id: str) -> Path:
    return outputs_root(config) / output_series_name(config, series_name) / episode_id / "01-director-analysis.md"


def storyboard_output_path(config: Mapping[str, Any], series_name: str, episode_id: str) -> Path:
    return outputs_root(config) / output_series_name(config, series_name) / episode_id / "02-seedance-prompts.md"


def storyboard_pending_marker_path(config: Mapping[str, Any], series_name: str, episode_id: str) -> Path:
    return outputs_root(config) / output_series_name(config, series_name) / episode_id / "02-seedance-prompts.review-pending"


def storyboard_json_path(config: Mapping[str, Any], series_name: str, episode_id: str) -> Path | None:
    episode_dir = outputs_root(config) / output_series_name(config, series_name) / episode_id
    preferred_tag = build_provider_model_tag("openai", str(config.get("provider", {}).get("model", "gpt-5.4")).strip())
    preferred = episode_dir / f"02-seedance-prompts__{preferred_tag}.json"
    if preferred.exists():
        return preferred
    candidates = sorted(episode_dir.glob("02-seedance-prompts__*.json"))
    return candidates[-1] if candidates else None


def seedance_prompt_refine_stamp_path(config: Mapping[str, Any], series_name: str, episode_id: str) -> Path:
    episode_dir = outputs_root(config) / output_series_name(config, series_name) / episode_id
    provider_tag = build_provider_model_tag("openai", str(config.get("provider", {}).get("model", "gpt-5.4")).strip())
    return episode_dir / f"02-seedance-prompts.refine__{provider_tag}.json"


def episode_block_exists(path: Path, episode_id: str) -> bool:
    if not path.exists():
        return False
    marker = f"<!-- episode: {episode_id} start -->"
    return marker in path.read_text(encoding="utf-8")


def find_missing_explosive_episodes(config: Mapping[str, Any], series_name: str) -> list[str]:
    return [episode_id for episode_id in episode_ids(config) if not has_existing_explosive_variant(config, episode_id)]


def find_missing_director_episodes(config: Mapping[str, Any], series_name: str) -> list[str]:
    return [episode_id for episode_id in episode_ids(config) if not director_output_path(config, series_name, episode_id).exists()]


def find_missing_art_episodes(config: Mapping[str, Any], series_name: str) -> list[str]:
    character_path = assets_dir(config, series_name) / "character-prompts.md"
    scene_path = assets_dir(config, series_name) / "scene-prompts.md"
    missing: list[str] = []
    for episode_id in episode_ids(config):
        if not episode_block_exists(character_path, episode_id) or not episode_block_exists(scene_path, episode_id):
            missing.append(episode_id)
    return missing


def find_missing_storyboard_episodes(config: Mapping[str, Any], series_name: str) -> list[str]:
    missing: list[str] = []
    for episode_id in episode_ids(config):
        storyboard_path = storyboard_output_path(config, series_name, episode_id)
        pending_marker = storyboard_pending_marker_path(config, series_name, episode_id)
        if (not storyboard_path.exists()) or pending_marker.exists():
            missing.append(episode_id)
    return missing


def ready_storyboard_episodes(config: Mapping[str, Any], series_name: str) -> list[str]:
    missing = set(find_missing_storyboard_episodes(config, series_name))
    return [episode_id for episode_id in episode_ids(config) if episode_id not in missing]


def find_missing_seedance_prompt_refine_episodes(config: Mapping[str, Any], series_name: str) -> list[str]:
    missing: list[str] = []
    for episode_id in ready_storyboard_episodes(config, series_name):
        if not seedance_prompt_refine_stamp_path(config, series_name, episode_id).exists():
            missing.append(episode_id)
    return missing


def contiguous_episode_ranges(target_episode_ids: list[str]) -> list[tuple[int, int]]:
    if not target_episode_ids:
        return []
    numbers = sorted(episode_number(item) for item in target_episode_ids)
    ranges: list[tuple[int, int]] = []
    start = numbers[0]
    end = numbers[0]
    for value in numbers[1:]:
        if value == end + 1:
            end = value
            continue
        ranges.append((start, end))
        start = end = value
    ranges.append((start, end))
    return ranges


def resolve_stage_runtime_config(config_data: Mapping[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config_data))
    base_path = str(resolved.get("base_config") or "").strip()
    if not base_path:
        return resolved
    base_file = Path(base_path).expanduser()
    if not base_file.is_absolute():
        base_file = (PROJECT_ROOT / base_file).resolve()
    return deep_merge(load_json(base_file), resolved)


def build_explosive_config(
    master: Mapping[str, Any], *, series_name: str, start_episode: int, end_episode: int
) -> dict[str, Any]:
    return {
        "base_config": "config/video_pipeline.local.json",
        "script": {
            "series_dir": str(master["series"]["script_series_dir"]),
            "series_name": series_name,
        },
        "series": {
            "series_name": series_name,
            "start_episode": start_episode,
            "end_episode": end_episode,
            "episode_id_prefix": master["series"].get("episode_id_prefix", "ep"),
            "episode_id_padding": master["series"].get("episode_id_padding", 2),
        },
        "provider": {
            "openai": {
                "api_key": master.get("provider", {}).get("api_key", ""),
                "model": master.get("provider", {}).get("model", "gpt-5.4"),
            }
        },
        "quality": {
            "target_audience": master.get("explosive", {}).get(
                "target_audience",
                "女频古言漫剧用户，偏爱高钩子、高反差、高情绪兑现与强卡点",
            ),
            "extra_rules": master.get("explosive", {}).get("extra_rules", []),
        },
        "output": {
            "analysis_dir_name": master.get("explosive", {}).get("analysis_dir_name", "explosive-actor-gpt"),
            "script_series_suffix": "",
            "rewrite_filename_suffix": master.get("explosive", {}).get(
                "rewrite_filename_suffix", "__explosive"
            ),
        },
        "run": {
            "temperature": master.get("runtime", {}).get("temperature", 0.3),
            "timeout_seconds": master.get("runtime", {}).get("timeout_seconds", 600),
            "dry_run": master.get("runtime", {}).get("dry_run", True),
        },
    }


def build_director_config(
    master: Mapping[str, Any], *, series_name: str, start_episode: int, end_episode: int
) -> dict[str, Any]:
    return {
        "base_config": "config/video_pipeline.local.json",
        "script": {
            "series_dir": str(master["series"]["script_series_dir"]),
            "script_path": str(master.get("source", {}).get("script_path_override") or ""),
            "episode_id": "",
            "preferred_filename_suffixes": master.get("source", {}).get("preferred_filename_suffixes", []),
        },
        "sources": {
            "genre_reference_bundle_path": str(master.get("genre_reference", {}).get("bundle_path") or ""),
            "seedance_purpose_skill_library_path": "",
        },
        "series": {
            "series_name": series_name,
            "start_episode": start_episode,
            "end_episode": end_episode,
            "episode_id_prefix": master["series"].get("episode_id_prefix", "ep"),
            "episode_id_padding": master["series"].get("episode_id_padding", 2),
        },
        "provider": {
            "openai": {
                "api_key": master.get("provider", {}).get("api_key", ""),
                "model": master.get("provider", {}).get("model", "gpt-5.4"),
            }
        },
        "quality": {
            "visual_style": master.get("quality", {}).get(
                "visual_style",
                "真人写实，移动端9:16竖屏电影感构图，贴近高质量古装漫剧前期开发工作流",
            ),
            "target_medium": master.get("quality", {}).get("target_medium", "漫剧"),
            "frame_orientation": master.get("quality", {}).get("frame_orientation", "9:16竖屏"),
            "extra_rules": master.get("quality", {}).get("director_extra_rules", []),
            "director_scene_budget": master.get("quality", {}).get("director_scene_budget", {}),
        },
        "output": {
            "outputs_root": master.get("output", {}).get("outputs_root", "outputs"),
            "outputs_series_name": master.get("output", {}).get("outputs_series_name", ""),
            "outputs_series_suffix": master.get("output", {}).get("outputs_series_suffix", "-gpt"),
            "assets_series_suffix": master.get("quality", {}).get("assets_series_suffix", "-gpt"),
            "assets_series_name": master.get("quality", {}).get("assets_series_name", ""),
        },
        "run": {
            "temperature": master.get("runtime", {}).get("temperature", 0.25),
            "timeout_seconds": master.get("runtime", {}).get("timeout_seconds", 600),
            "enable_review_pass": master.get("runtime", {}).get("enable_review_pass", True),
            "skip_existing_output": master.get("runtime", {}).get("skip_existing_outputs", True),
            "dry_run": master.get("runtime", {}).get("dry_run", True),
        },
    }


def build_art_config(
    master: Mapping[str, Any], *, series_name: str, start_episode: int, end_episode: int
) -> dict[str, Any]:
    return {
        "base_config": "config/video_pipeline.local.json",
        "script": {
            "script_path": str(master.get("source", {}).get("script_path_override") or ""),
            "episode_id": "",
        },
        "series": {
            "series_name": series_name,
            "start_episode": start_episode,
            "end_episode": end_episode,
            "episode_id_prefix": master["series"].get("episode_id_prefix", "ep"),
            "episode_id_padding": master["series"].get("episode_id_padding", 2),
        },
        "provider": {
            "selected_provider": master.get("quality", {}).get("art_selected_provider", "openai"),
            "openai": {
                "api_key": master.get("provider", {}).get("api_key", ""),
                "model": master.get("provider", {}).get("model", "gpt-5.4"),
            },
            "gemini": {
                "api_key": master.get("quality", {}).get("gemini_api_key", ""),
                "model": master.get("quality", {}).get("art_gemini_model", "gemini-3-pro-preview"),
            },
        },
        "sources": {
            "analysis_provider": master.get("quality", {}).get("analysis_provider", "gemini"),
            "analysis_model": master.get("quality", {}).get("analysis_model", "gemini-3-pro-preview"),
            "analysis_path": "",
            "director_analysis_path": "",
            "director_outputs_root": master.get("output", {}).get("outputs_root", "outputs"),
            "outputs_series_name": master.get("output", {}).get("outputs_series_name", ""),
            "outputs_series_suffix": master.get("output", {}).get("outputs_series_suffix", "-gpt"),
            "genre_reference_bundle_path": str(master.get("genre_reference", {}).get("bundle_path") or ""),
        },
        "quality": {
            "visual_style": master.get("quality", {}).get(
                "visual_style",
                "真人写实，移动端9:16竖屏电影感构图，适合后续 Nano Banana / Seedance 参考图工作流",
            ),
            "target_medium": master.get("quality", {}).get("target_medium", "漫剧"),
            "frame_orientation": master.get("quality", {}).get("frame_orientation", "9:16竖屏"),
            "extra_rules": master.get("quality", {}).get("art_extra_rules", []),
        },
        "output": {
            "assets_series_name": master.get("quality", {}).get("assets_series_name", ""),
            "assets_series_suffix": master.get("quality", {}).get("assets_series_suffix", "-gpt"),
        },
        "run": {
            "analysis_root": master.get("output", {}).get("analysis_root", "analysis"),
            "temperature": master.get("runtime", {}).get("temperature", 0.25),
            "timeout_seconds": master.get("runtime", {}).get("timeout_seconds", 600),
            "enable_review_pass": master.get("runtime", {}).get("enable_review_pass", True),
            "dry_run": master.get("runtime", {}).get("dry_run", True),
        },
    }


def build_storyboard_config(
    master: Mapping[str, Any], *, series_name: str, start_episode: int, end_episode: int
) -> dict[str, Any]:
    return {
        "base_config": "config/video_pipeline.local.json",
        "script": {
            "series_dir": str(master["series"]["script_series_dir"]),
            "script_path": str(master.get("source", {}).get("script_path_override") or ""),
            "episode_id": "",
        },
        "series": {
            "series_name": series_name,
            "start_episode": start_episode,
            "end_episode": end_episode,
            "episode_id_prefix": master["series"].get("episode_id_prefix", "ep"),
            "episode_id_padding": master["series"].get("episode_id_padding", 2),
        },
        "provider": {
            "openai": {
                "api_key": master.get("provider", {}).get("api_key", ""),
                "model": master.get("provider", {}).get("model", "gpt-5.4"),
            }
        },
        "sources": {
            "director_analysis_path": "",
            "genre_reference_bundle_path": str(master.get("genre_reference", {}).get("bundle_path") or ""),
            "seedance_purpose_skill_library_path": "",
            "seedance_purpose_template_library_path": "",
        },
        "quality": {
            "visual_style": master.get("quality", {}).get(
                "visual_style",
                "真人写实，移动端9:16竖屏电影感构图，适合 Seedance 2.0 动态生成",
            ),
            "target_medium": master.get("quality", {}).get("target_medium", "漫剧"),
            "frame_orientation": master.get("quality", {}).get("frame_orientation", "9:16竖屏"),
            "extra_rules": master.get("quality", {}).get("storyboard_extra_rules", []),
        },
        "output": {
            "outputs_root": master.get("output", {}).get("outputs_root", "outputs"),
            "outputs_series_name": master.get("output", {}).get("outputs_series_name", ""),
            "outputs_series_suffix": master.get("output", {}).get("outputs_series_suffix", "-gpt"),
            "assets_series_name": master.get("quality", {}).get("assets_series_name", ""),
            "assets_series_suffix": master.get("quality", {}).get("assets_series_suffix", "-gpt"),
        },
        "run": {
            "temperature": master.get("runtime", {}).get("temperature", 0.2),
            "timeout_seconds": master.get("runtime", {}).get("timeout_seconds", 600),
            "enable_review_pass": master.get("runtime", {}).get("enable_review_pass", True),
            "storyboard_profile": master.get("runtime", {}).get("storyboard_profile", "normal"),
            "write_storyboard_metrics": master.get("runtime", {}).get("write_storyboard_metrics", True),
            "dry_run": master.get("runtime", {}).get("dry_run", True),
        },
    }


def build_seedance_prompt_refine_config(
    master: Mapping[str, Any], *, series_name: str, start_episode: int, end_episode: int
) -> dict[str, Any]:
    return {
        "base_config": "config/video_pipeline.local.json",
        "script": {
            "series_dir": str(master["series"]["script_series_dir"]),
            "script_path": str(master.get("source", {}).get("script_path_override") or ""),
            "episode_id": "",
            "preferred_filename_suffixes": master.get("source", {}).get("preferred_filename_suffixes", []),
        },
        "series": {
            "series_name": series_name,
            "start_episode": start_episode,
            "end_episode": end_episode,
            "episode_id_prefix": master["series"].get("episode_id_prefix", "ep"),
            "episode_id_padding": master["series"].get("episode_id_padding", 2),
        },
        "provider": {
            "openai": {
                "api_key": master.get("provider", {}).get("api_key", ""),
                "model": master.get("provider", {}).get("model", "gpt-5.4"),
            }
        },
        "sources": {
            "seedance_prompt_refine_techniques_path": str(
                master.get("sources", {}).get("seedance_prompt_refine_techniques_path") or ""
            ),
        },
        "quality": {
            "visual_style": master.get("quality", {}).get(
                "visual_style",
                "真人写实，移动端9:16竖屏电影感构图，适合 Seedance 2.0 动态生成",
            ),
            "target_medium": master.get("quality", {}).get("target_medium", "漫剧"),
            "frame_orientation": master.get("quality", {}).get("frame_orientation", "9:16竖屏"),
            "assets_series_name": master.get("quality", {}).get("assets_series_name", ""),
            "assets_series_suffix": master.get("quality", {}).get("assets_series_suffix", "-gpt"),
        },
        "output": {
            "outputs_root": master.get("output", {}).get("outputs_root", "outputs"),
            "outputs_series_name": master.get("output", {}).get("outputs_series_name", ""),
            "outputs_series_suffix": master.get("output", {}).get("outputs_series_suffix", "-gpt"),
        },
        "run": {
            "temperature": master.get("runtime", {}).get("seedance_prompt_refine_temperature", 0.35),
            "timeout_seconds": master.get("runtime", {}).get(
                "seedance_prompt_refine_timeout_seconds",
                master.get("runtime", {}).get("timeout_seconds", 600),
            ),
            "write_refine_metrics": master.get("runtime", {}).get("write_seedance_prompt_refine_metrics", True),
            "storyboard_profile": master.get("runtime", {}).get("storyboard_profile", "normal"),
            "dry_run": master.get("runtime", {}).get("dry_run", True),
        },
    }


def dry_run_summary(config: Mapping[str, Any], *, series_name: str) -> dict[str, Any]:
    resolved_outputs_root = outputs_root(config)
    analysis_root = Path(config.get("output", {}).get("analysis_root", "analysis")).expanduser()
    if not analysis_root.is_absolute():
        analysis_root = (PROJECT_ROOT / analysis_root).resolve()
    resolved_assets_dir = assets_dir(config, series_name)
    dry_run_mode = bool(config.get("runtime", {}).get("dry_run", True))
    stage_missing = {
        "explosive_missing": find_missing_explosive_episodes(config, series_name),
        "director_missing": find_missing_director_episodes(config, series_name),
        "art_missing": find_missing_art_episodes(config, series_name),
        "storyboard_missing": find_missing_storyboard_episodes(config, series_name),
        "seedance_prompt_refine_missing": find_missing_seedance_prompt_refine_episodes(config, series_name),
    }
    stage_existing = {
        "existing_explosive": [episode_id for episode_id in episode_ids(config) if episode_id not in stage_missing["explosive_missing"]],
        "existing_director": [episode_id for episode_id in episode_ids(config) if episode_id not in stage_missing["director_missing"]],
        "existing_art": [episode_id for episode_id in episode_ids(config) if episode_id not in stage_missing["art_missing"]],
        "existing_storyboard": [episode_id for episode_id in episode_ids(config) if episode_id not in stage_missing["storyboard_missing"]],
        "existing_seedance_prompt_refine": [
            episode_id
            for episode_id in ready_storyboard_episodes(config, series_name)
            if episode_id not in stage_missing["seedance_prompt_refine_missing"]
        ],
    }
    genre_bundle_json_path, genre_bundle_md_path = bundle_paths(PROJECT_ROOT, config, series_name)
    source_mode = str(config.get("source", {}).get("interactive_source_mode") or "").strip()
    if source_mode == "explosive_only":
        source_mode_note = "你在交互中选择了“仅爆改版”，因此每集只会选择 __explosive.md 版本。"
    elif source_mode == "explosive_first":
        source_mode_note = "你在交互中选择了“爆改版优先”，存在 __explosive.md 时会优先使用。"
    elif source_mode == "base_first":
        source_mode_note = "你在交互中选择了“原始基础版优先”，会优先使用基础稿。"
    else:
        source_mode_note = "未记录到交互来源策略，按 preferred_filename_suffixes 进行选择。"
    if dry_run_mode:
        summary_note = "当前是 dry_run 预演模式：本次不会真正生成 outputs/assets 文件。stage_missing 表示“如果现在正式运行，还需要生成哪些文件”，不是报错。"
    else:
        summary_note = "当前是正式运行模式：stage_missing 表示本次运行前仍缺失的阶段产物。"
    return {
        "series_name": series_name,
        "episode_ids": episode_ids(config),
        "dry_run_mode": dry_run_mode,
        "summary_note": summary_note,
        "selected_source_mode": source_mode or "config_only",
        "selected_source_note": source_mode_note,
        "selected_scripts": selected_script_preview(config),
        "outputs_root": str(resolved_outputs_root / output_series_name(config, series_name)),
        "analysis_root": str(analysis_root / series_name),
        "genre_reference_bundle_json": str(genre_bundle_json_path),
        "genre_reference_bundle_md": str(genre_bundle_md_path),
        "assets_dir": str(resolved_assets_dir),
        "path_exists": {
            "outputs_root_exists": (resolved_outputs_root / output_series_name(config, series_name)).exists(),
            "analysis_root_exists": (analysis_root / series_name).exists(),
            "genre_reference_bundle_json_exists": genre_bundle_json_path.exists(),
            "genre_reference_bundle_md_exists": genre_bundle_md_path.exists(),
            "assets_dir_exists": resolved_assets_dir.exists(),
        },
        "stages": config.get("stages", {}),
        "collect_metrics": bool(config.get("runtime", {}).get("collect_metrics", False)),
        "storyboard_profile": str(config.get("runtime", {}).get("storyboard_profile", "normal")),
        "stage_missing": stage_missing,
        "stage_existing": stage_existing,
        "generated_at": utc_timestamp(),
    }


def should_skip_existing(config: Mapping[str, Any]) -> bool:
    return bool(config.get("runtime", {}).get("skip_existing_outputs", True))


def render_metrics_markdown(report: dict[str, Any]) -> str:
    context = report.get("context", {}) or {}
    totals = report.get("totals", {}) or {}
    failure_hints = list(context.get("failure_hints", []) or [])
    lines = [
        "# OpenAI Agent Flow 统计报告",
        "",
        f"- run_name：{report.get('run_name', '')}",
        f"- final_status：{context.get('final_status', '')}",
        f"- series_name：{context.get('series_name', '')}",
        f"- episode_range：{context.get('episode_range', '')}",
        f"- model：{context.get('model', '')}",
        "",
        "## 总计",
        "",
        f"- steps：{totals.get('step_count', 0)}",
        f"- wall_clock_duration_seconds：{totals.get('duration_seconds', 0)}",
        f"- summed_step_duration_seconds：{totals.get('summed_step_duration_seconds', 0)}",
        f"- input_tokens：{totals.get('input_tokens', 0)}",
        f"- output_tokens：{totals.get('output_tokens', 0)}",
        f"- total_tokens：{totals.get('total_tokens', 0)}",
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
    failure_message = str(context.get("failure_message", "")).strip()
    if failure_message:
        lines.extend(["", "## 失败原因", "", f"- {failure_message}"])
    if failure_hints:
        lines.extend(["", "## 建议", ""])
        lines.extend([f"- {hint}" for hint in failure_hints])
    return "\n".join(lines).rstrip() + "\n"


def metrics_summary_paths(config: Mapping[str, Any], series_name: str) -> tuple[Path, Path]:
    ids = episode_ids(config)
    first_episode = ids[0]
    last_episode = ids[-1]
    output_tag = build_provider_model_tag("openai", str(config.get("provider", {}).get("model", "gpt-5.4")))
    base = (
        Path(config.get("output", {}).get("analysis_root", "analysis"))
        / series_name
        / "openai_agent_flow"
        / f"metrics_summary__{output_tag}__{first_episode}-{last_episode}"
    ).expanduser().resolve()
    return Path(f"{base}.json"), Path(f"{base}.md")


def save_metrics(recorder: TelemetryRecorder, json_path: Path, md_path: Path) -> dict[str, Any]:
    report = recorder.to_dict()
    started_at = str(report.get("started_at", "")).strip()
    finished_at = str(report.get("finished_at", "")).strip()
    wall_clock_duration_seconds = 0.0
    if started_at and finished_at:
        started_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finished_dt = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        wall_clock_duration_seconds = round(max(0.0, (finished_dt - started_dt).total_seconds()), 3)
    summed_step_duration_seconds = round(
        sum(float(step.get("duration_seconds", 0) or 0) for step in report.get("steps", [])),
        3,
    )
    report.setdefault("totals", {})
    report["totals"]["summed_step_duration_seconds"] = summed_step_duration_seconds
    report["totals"]["duration_seconds"] = wall_clock_duration_seconds
    save_json_file(json_path, report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_metrics_markdown(report), encoding="utf-8")
    return report


def derive_failure_hints(exc: Exception, recorder: TelemetryRecorder | None, config: Mapping[str, Any]) -> list[str]:
    message = str(exc).strip()
    lowered = message.lower()
    hints: list[str] = []
    if "timed out" in lowered or isinstance(exc, TimeoutError):
        current_profile = str(config.get("runtime", {}).get("storyboard_profile", "normal")).strip() or "normal"
        timeout_seconds = int(config.get("runtime", {}).get("timeout_seconds", 600) or 600)
        failed_step = None
        if recorder:
            for step in reversed(recorder.steps):
                if str(step.get("status")) == "failed":
                    failed_step = step
                    break
        prompt_chars = None
        if failed_step:
            prompt_chars = (failed_step.get("metadata") or {}).get("prompt_chars")
            if prompt_chars is None:
                for step in reversed(recorder.steps):
                    if step.get("stage") == str(failed_step.get("stage")) and (step.get("metadata") or {}).get("prompt_chars") is not None:
                        prompt_chars = (step.get("metadata") or {}).get("prompt_chars")
                        break
        if failed_step and str(failed_step.get("stage")) == "storyboard":
            if current_profile != "fast":
                hints.append("本次超时发生在 storyboard 阶段，建议优先改用 `fast`（极速版）模式后重试。")
            hints.append(f"当前 `runtime.timeout_seconds={timeout_seconds}`，建议至少提高到 `600`；超长分镜集可继续上调。")
            if prompt_chars:
                hints.append(f"本次超时前的 storyboard prompt 体积约为 `{prompt_chars}` 字符，属于偏大输入。")
        else:
            hints.append(f"当前 `runtime.timeout_seconds={timeout_seconds}`，可尝试提高到 `600` 后重试。")
    return hints


def telemetry_totals_snapshot(recorder: TelemetryRecorder | None) -> dict[str, int]:
    if recorder is None:
        return {
            "duration_seconds": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "step_count": 0,
        }
    totals = recorder.totals()
    return {
        "duration_seconds": int(round(float(totals.get("duration_seconds", 0) or 0))),
        "input_tokens": int(totals.get("input_tokens", 0) or 0),
        "output_tokens": int(totals.get("output_tokens", 0) or 0),
        "total_tokens": int(totals.get("total_tokens", 0) or 0),
        "step_count": int(totals.get("step_count", 0) or 0),
    }


def telemetry_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        "duration_seconds": max(0, int(after.get("duration_seconds", 0)) - int(before.get("duration_seconds", 0))),
        "input_tokens": max(0, int(after.get("input_tokens", 0)) - int(before.get("input_tokens", 0))),
        "output_tokens": max(0, int(after.get("output_tokens", 0)) - int(before.get("output_tokens", 0))),
        "total_tokens": max(0, int(after.get("total_tokens", 0)) - int(before.get("total_tokens", 0))),
        "step_count": max(0, int(after.get("step_count", 0)) - int(before.get("step_count", 0))),
    }


def print_stage_metrics(stage_label: str, delta: dict[str, int]) -> None:
    print_status(
        f"{stage_label} 统计：耗时 {delta['duration_seconds']}s | "
        f"tokens in/out/total = {delta['input_tokens']}/{delta['output_tokens']}/{delta['total_tokens']} | "
        f"新增步骤 {delta['step_count']}"
    )


def skipped_stage_summary(
    *,
    config: Mapping[str, Any],
    stage_enabled: bool,
    stage_label: str,
    stage_name: str,
    reason: str,
    message: str,
    telemetry: TelemetryRecorder | None = None,
) -> dict[str, Any]:
    summary = {
        "enabled": bool(stage_enabled),
        "target_episode_ids": episode_ids(config),
        "missing_episode_ids": [],
        "executed_ranges": [],
        "status": reason,
        "force_rerun": False,
        "message": message,
    }
    print_status(f"跳过 {stage_label}：{message}")
    with telemetry_span(
        telemetry,
        stage="openai_flow",
        name=f"skip_{stage_name}_stage",
        metadata={"reason": reason, "stage_label": stage_label, "message": message},
    ) as step:
        step["status"] = "skipped"
    return summary


def run_stage_for_missing_ranges(
    *,
    config: Mapping[str, Any],
    series_name: str,
    stage_enabled: bool,
    stage_label: str,
    stage_name: str,
    missing_episode_ids: list[str],
    config_builder,
    runner,
    telemetry: TelemetryRecorder | None = None,
    force_rerun: bool = False,
    target_episode_ids_override: list[str] | None = None,
) -> dict[str, Any]:
    target_episode_ids = target_episode_ids_override or episode_ids(config)
    stage_summary: dict[str, Any] = {
        "enabled": bool(stage_enabled),
        "target_episode_ids": target_episode_ids,
        "missing_episode_ids": missing_episode_ids,
        "executed_ranges": [],
        "status": "pending",
        "force_rerun": bool(force_rerun),
    }
    if not stage_enabled:
        print_status(f"跳过 {stage_label}：配置中未启用。")
        stage_summary["status"] = "disabled"
        with telemetry_span(
            telemetry,
            stage="openai_flow",
            name=f"skip_{stage_name}_stage",
            metadata={"reason": "disabled", "stage_label": stage_label},
        ) as step:
            step["status"] = "skipped"
        return stage_summary

    if should_skip_existing(config) and not missing_episode_ids and not force_rerun:
        print_status(f"跳过 {stage_label}：目标集数产物已存在。")
        stage_summary["status"] = "skipped_existing"
        with telemetry_span(
            telemetry,
            stage="openai_flow",
            name=f"skip_{stage_name}_stage",
            metadata={"reason": "outputs_exist", "stage_label": stage_label},
        ) as step:
            step["status"] = "skipped"
        return stage_summary

    target_ids = target_episode_ids if (force_rerun or not should_skip_existing(config)) else missing_episode_ids
    stage_accumulated = {
        "duration_seconds": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "step_count": 0,
    }
    for start_episode, end_episode in contiguous_episode_ranges(target_ids):
        print_status(f"{stage_label} 运行区间：ep{start_episode:02d}-ep{end_episode:02d}")
        runtime_config = resolve_stage_runtime_config(
            config_builder(config, series_name=series_name, start_episode=start_episode, end_episode=end_episode)
        )
        before = telemetry_totals_snapshot(telemetry)
        try:
            with telemetry_span(
                telemetry,
                stage="openai_flow",
                name=f"run_{stage_name}_range",
                metadata={
                    "stage_label": stage_label,
                    "start_episode": start_episode,
                    "end_episode": end_episode,
                },
            ):
                pipeline_results = runner(runtime_config, telemetry=telemetry) or []
            if "episode_results" not in stage_summary:
                stage_summary["episode_results"] = []
            if isinstance(pipeline_results, list):
                stage_summary["episode_results"].extend(pipeline_results)
        except Exception as exc:
            after = telemetry_totals_snapshot(telemetry)
            delta = telemetry_delta(before, after)
            print_stage_metrics(stage_label, delta)
            for key in stage_accumulated:
                stage_accumulated[key] += delta[key]
            stage_summary["executed_ranges"].append(f"ep{start_episode:02d}-ep{end_episode:02d}")
            stage_summary["metrics"] = stage_accumulated
            stage_summary["status"] = "failed"
            stage_summary["error_message"] = str(exc)
            stage_summary["failed_range"] = f"ep{start_episode:02d}-ep{end_episode:02d}"
            print_status(f"{stage_label} 失败，但流程将继续：{exc}")
            return stage_summary
        after = telemetry_totals_snapshot(telemetry)
        delta = telemetry_delta(before, after)
        print_stage_metrics(stage_label, delta)
        for key in stage_accumulated:
            stage_accumulated[key] += delta[key]
        stage_summary["executed_ranges"].append(f"ep{start_episode:02d}-ep{end_episode:02d}")
    stage_summary["metrics"] = stage_accumulated
    stage_summary["status"] = "completed"
    return stage_summary


def prompt_seedance_prompt_refine_after_pipeline(*, series_name: str, ready_episode_ids: list[str]) -> bool:
    ready_label = "、".join(ready_episode_ids)
    print_status(f"{series_name} 已就绪可 refine 的分镜集数：{ready_label}")
    while True:
        raw = input("是否继续执行 Seedance prompt refine（仅优化提示词质量，不改剧情；1=是，0=否，默认 0）：").strip()
        if not raw:
            return False
        if raw in {"1", "y", "Y", "yes", "YES"}:
            return True
        if raw in {"0", "n", "N", "no", "NO"}:
            return False
        print("输入无效，请输入 1 或 0。")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OpenAI-native staged production flow.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--collect-metrics", choices=["true", "false"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config: dict[str, Any] = {}
    series_name = ""
    collect_metrics = False
    recorder: TelemetryRecorder | None = None
    metrics_json_path: Path | None = None
    metrics_md_path: Path | None = None
    try:
        config = load_runtime_config(args.config)
        collect_metrics_override = parse_bool_flag(args.collect_metrics)
        if collect_metrics_override is not None:
            config.setdefault("runtime", {})
            config["runtime"]["collect_metrics"] = collect_metrics_override
        series_name = resolve_series_name(config)
        print_status(f"加载配置：{args.config}")
        print_status(f"剧名：{series_name}")
        print_status(f"剧本目录：{config['series']['script_series_dir']}")
        print_status(f"运行模式：{'dry_run 预演' if config.get('runtime', {}).get('dry_run', True) else '正式运行'}")
        print_status(f"统计报告：{'开启' if config.get('runtime', {}).get('collect_metrics', False) else '关闭'}")

        genre_bundle_json_path, genre_bundle_md_path = bundle_paths(PROJECT_ROOT, config, series_name)

        if config.get("runtime", {}).get("dry_run", True):
            print_status("当前为 dry_run 模式，会逐阶段预演配置与输入输出路径。")
            print(json.dumps(dry_run_summary(config, series_name=series_name), ensure_ascii=False, indent=2))
            return

        genre_bundle, genre_bundle_json_path, genre_bundle_md_path = load_or_build_genre_reference_bundle(
            project_root=PROJECT_ROOT,
            config=config,
            series_name=series_name,
        )
        config.setdefault("genre_reference", {})
        config["genre_reference"]["bundle_path"] = str(genre_bundle_json_path)
        print_status(f"题材参考包：{genre_bundle_json_path}")
        selected_genres = list(genre_bundle.get("selected_genres", []))
        if selected_genres:
            print_status(f"命中题材：{'、'.join(selected_genres)}")

        # 优化方案1+4：全局预计算compact bundle版本，避免每个阶段重复加载和压缩
        from genre_reference_bundle import filter_bundle_for_stage
        print_status("预计算各阶段的题材包compact版本（优化token消耗）...")
        bundle_compact_cache = {
            "director": filter_bundle_for_stage(genre_bundle, "director"),
            "art": filter_bundle_for_stage(genre_bundle, "art"),
            "storyboard": filter_bundle_for_stage(genre_bundle, "storyboard"),
        }
        original_bundle_size = len(json.dumps(genre_bundle, ensure_ascii=False))
        compact_director_size = len(json.dumps(bundle_compact_cache["director"], ensure_ascii=False))
        compact_art_size = len(json.dumps(bundle_compact_cache["art"], ensure_ascii=False))
        compact_storyboard_size = len(json.dumps(bundle_compact_cache["storyboard"], ensure_ascii=False))
        print_status(
            f"Bundle尺寸优化：原始 {original_bundle_size/1024:.1f}KB → "
            f"导演 {compact_director_size/1024:.1f}KB / 服化 {compact_art_size/1024:.1f}KB / 分镜 {compact_storyboard_size/1024:.1f}KB "
            f"（节省约 {(1-sum([compact_director_size,compact_art_size,compact_storyboard_size])/(original_bundle_size*3))*100:.0f}%）"
        )
        # 将预计算的compact bundle注入到config中，下游pipeline直接使用，无需再加载和压缩
        config["_precomputed_bundle_cache"] = bundle_compact_cache

        stages = config.get("stages", {})
        collect_metrics = bool(config.get("runtime", {}).get("collect_metrics", False))
        if collect_metrics:
            recorder = TelemetryRecorder(
                run_name="openai-agent-flow",
                context={
                    "series_name": series_name,
                    "episode_range": f"{episode_ids(config)[0]}-{episode_ids(config)[-1]}",
                    "script_series_dir": str(Path(config["series"]["script_series_dir"]).expanduser().resolve()),
                    "model": str(config.get("provider", {}).get("model", "gpt-5.4")),
                },
            )
            metrics_json_path, metrics_md_path = metrics_summary_paths(config, series_name)

        stage_results: dict[str, Any] = {}
        stage_results["explosive_rewrite"] = run_stage_for_missing_ranges(
            config=config,
            series_name=series_name,
            stage_enabled=stages.get("run_explosive_rewrite", False),
            stage_label="爆款改稿",
            stage_name="explosive_rewrite",
            missing_episode_ids=find_missing_explosive_episodes(config, series_name),
            config_builder=build_explosive_config,
            runner=run_explosive_pipeline,
            telemetry=recorder,
            force_rerun=False,
        )

        stage_results["director_analysis"] = run_stage_for_missing_ranges(
            config=config,
            series_name=series_name,
            stage_enabled=stages.get("run_director_analysis", True),
            stage_label="导演分析",
            stage_name="director_analysis",
            missing_episode_ids=find_missing_director_episodes(config, series_name),
            config_builder=build_director_config,
            runner=run_director_pipeline,
            telemetry=recorder,
            force_rerun=False,
        )

        # 质量门控：检查导演分析各集 quality_warning_count 是否超出阈值
        quality_gate_cfg = config.get("quality_gate") or {}
        director_max_warnings = quality_gate_cfg.get("director_max_warnings_per_episode")
        director_fail_on_gate = bool(quality_gate_cfg.get("director_fail_on_gate", False))
        if director_max_warnings is not None:
            director_episode_results = stage_results["director_analysis"].get("episode_results") or []
            gate_breaches = [
                r for r in director_episode_results
                if isinstance(r, dict) and r.get("quality_warning_count", 0) > director_max_warnings
            ]
            if gate_breaches:
                breach_summary = ", ".join(
                    f"{r.get('episode_id', '?')}({r.get('quality_warning_count', 0)}条告警)"
                    for r in gate_breaches
                )
                gate_msg = (
                    f"质量门控触发：以下集数导演分析告警数超过阈值({director_max_warnings})：{breach_summary}"
                )
                print_status(gate_msg)
                stage_results["director_analysis"]["quality_gate_triggered"] = True
                stage_results["director_analysis"]["quality_gate_breaches"] = [
                    {"episode_id": r.get("episode_id"), "warning_count": r.get("quality_warning_count")}
                    for r in gate_breaches
                ]
                if director_fail_on_gate:
                    stage_results["director_analysis"]["status"] = "failed"
                    stage_results["director_analysis"]["error_message"] = gate_msg

        if stage_results["director_analysis"].get("status") == "failed":
            stage_results["art_design"] = skipped_stage_summary(
                config=config,
                stage_enabled=stages.get("run_art_design", True),
                stage_label="服化道设计",
                stage_name="art_design",
                reason="skipped_upstream_failed",
                message="上游导演分析失败，服化道设计已跳过。",
                telemetry=recorder,
            )
            stage_results["storyboard"] = skipped_stage_summary(
                config=config,
                stage_enabled=stages.get("run_storyboard", True),
                stage_label="Seedance 分镜",
                stage_name="storyboard",
                reason="skipped_upstream_failed",
                message="上游导演分析失败，Seedance 分镜已跳过。",
                telemetry=recorder,
            )
        else:
            stage_results["art_design"] = run_stage_for_missing_ranges(
                config=config,
                series_name=series_name,
                stage_enabled=stages.get("run_art_design", True),
                stage_label="服化道设计",
                stage_name="art_design",
                missing_episode_ids=find_missing_art_episodes(config, series_name),
                config_builder=build_art_config,
                runner=run_art_pipeline,
                telemetry=recorder,
                force_rerun=False,
            )

            if stage_results["art_design"].get("status") == "failed":
                stage_results["storyboard"] = skipped_stage_summary(
                    config=config,
                    stage_enabled=stages.get("run_storyboard", True),
                    stage_label="Seedance 分镜",
                    stage_name="storyboard",
                    reason="skipped_upstream_failed",
                    message="上游服化道设计失败，Seedance 分镜已跳过。",
                    telemetry=recorder,
                )
            else:
                stage_results["storyboard"] = run_stage_for_missing_ranges(
                    config=config,
                    series_name=series_name,
                    stage_enabled=stages.get("run_storyboard", True),
                    stage_label="Seedance 分镜",
                    stage_name="storyboard",
                    missing_episode_ids=find_missing_storyboard_episodes(config, series_name),
                    config_builder=build_storyboard_config,
                    runner=run_storyboard_pipeline,
                    telemetry=recorder,
                    force_rerun=False,
                )

        refine_target_episode_ids = ready_storyboard_episodes(config, series_name)
        refine_prompt_enabled = bool(config.get("runtime", {}).get("prompt_for_seedance_prompt_refine", False))
        refine_stage_config_enabled = bool(stages.get("run_seedance_prompt_refine", False))
        refine_prompt_asked = False
        refine_prompt_selected = False

        if not refine_stage_config_enabled and refine_prompt_enabled:
            if refine_target_episode_ids and sys.stdin.isatty():
                refine_prompt_asked = True
                refine_prompt_selected = prompt_seedance_prompt_refine_after_pipeline(
                    series_name=series_name,
                    ready_episode_ids=refine_target_episode_ids,
                )

        refine_stage_enabled = refine_stage_config_enabled or refine_prompt_selected
        refine_missing_episode_ids = [
            episode_id
            for episode_id in find_missing_seedance_prompt_refine_episodes(config, series_name)
            if episode_id in refine_target_episode_ids
        ]
        if not refine_target_episode_ids:
            stage_results["seedance_prompt_refine"] = skipped_stage_summary(
                config=config,
                stage_enabled=refine_stage_config_enabled or refine_prompt_enabled,
                stage_label="Seedance Prompt Refine",
                stage_name="seedance_prompt_refine",
                reason="no_ready_storyboard",
                message="当前没有已完成且可 refine 的 02-seedance-prompts.md。",
                telemetry=recorder,
            )
        elif not refine_stage_enabled:
            skip_reason = "user_declined" if refine_prompt_asked else "disabled"
            skip_message = (
                "用户已跳过 Seedance Prompt refine。"
                if refine_prompt_asked
                else "配置中未启用 Seedance Prompt refine。"
            )
            stage_results["seedance_prompt_refine"] = skipped_stage_summary(
                config=config,
                stage_enabled=refine_stage_config_enabled or refine_prompt_enabled,
                stage_label="Seedance Prompt Refine",
                stage_name="seedance_prompt_refine",
                reason=skip_reason,
                message=skip_message,
                telemetry=recorder,
            )
        else:
            stage_results["seedance_prompt_refine"] = run_stage_for_missing_ranges(
                config=config,
                series_name=series_name,
                stage_enabled=True,
                stage_label="Seedance Prompt Refine",
                stage_name="seedance_prompt_refine",
                missing_episode_ids=refine_missing_episode_ids,
                config_builder=build_seedance_prompt_refine_config,
                runner=run_seedance_prompt_refine_pipeline,
                telemetry=recorder,
                force_rerun=refine_prompt_selected,
                target_episode_ids_override=refine_target_episode_ids,
            )
        stage_results["seedance_prompt_refine"]["requested_via_prompt"] = refine_prompt_asked
        stage_results["seedance_prompt_refine"]["selected_by_user"] = refine_prompt_selected
        stage_results["seedance_prompt_refine"]["ready_episode_ids"] = refine_target_episode_ids

        summary = {
            "series_name": series_name,
            "script_series_dir": str(Path(config["series"]["script_series_dir"]).expanduser().resolve()),
            "episode_ids": episode_ids(config),
            "stages": stages,
            "stage_results": stage_results,
            "collect_metrics": collect_metrics,
            "dry_run": bool(config.get("runtime", {}).get("dry_run", True)),
            "completed_at": utc_timestamp(),
        }
        overall_failed = any(result.get("status") == "failed" for result in stage_results.values())
        summary["final_status"] = "failed" if overall_failed else "completed"
        if collect_metrics:
            stage_metrics = {name: result.get("metrics", {}) for name, result in stage_results.items() if result.get("metrics")}
            if stage_metrics:
                summary["stage_metrics"] = stage_metrics
        if recorder and metrics_json_path and metrics_md_path:
            recorder.context["final_status"] = summary["final_status"]
            if overall_failed:
                failed_messages = [
                    f"{stage_name}: {result.get('error_message')}"
                    for stage_name, result in stage_results.items()
                    if result.get("status") == "failed" and str(result.get("error_message") or "").strip()
                ]
                if failed_messages:
                    recorder.context["failure_message"] = " | ".join(failed_messages)
            report = save_metrics(recorder, metrics_json_path, metrics_md_path)
            summary["metrics_summary_json_path"] = str(metrics_json_path)
            summary["metrics_summary_markdown_path"] = str(metrics_md_path)
            summary["metrics_totals"] = report["totals"]
            print_status(
                f"统计：总耗时 {report['totals']['duration_seconds']}s | "
                f"tokens in/out/total = {report['totals']['input_tokens']}/{report['totals']['output_tokens']}/{report['totals']['total_tokens']} | "
                f"步骤累计耗时 {report['totals'].get('summed_step_duration_seconds', 0)}s"
            )
            print_status(f"统计报告已写入：{metrics_json_path}")

        if overall_failed:
            print_status("OpenAI-native 生产链执行完成，但存在阶段失败；已输出 summary 与 metrics，不再强制中断。")
        else:
            print_status("OpenAI-native 生产链执行完成。")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        if recorder and metrics_json_path and metrics_md_path:
            recorder.context["final_status"] = "failed"
            recorder.context["failure_message"] = str(exc)
            failure_hints = derive_failure_hints(exc, recorder, config)
            if failure_hints:
                recorder.context["failure_hints"] = failure_hints
            report = save_metrics(recorder, metrics_json_path, metrics_md_path)
            print_status(
                f"统计：总耗时 {report['totals']['duration_seconds']}s | "
                f"tokens in/out/total = {report['totals']['input_tokens']}/{report['totals']['output_tokens']}/{report['totals']['total_tokens']} | "
                f"步骤累计耗时 {report['totals'].get('summed_step_duration_seconds', 0)}s"
            )
            print_status(f"失败时的统计报告已写入：{metrics_json_path}")
            for hint in failure_hints:
                print_status(f"建议：{hint}")
        print_status(f"流程出现未捕获异常，但已转为软失败输出：{exc}")
        print(
            json.dumps(
                {
                    "series_name": series_name,
                    "episode_ids": episode_ids(config),
                    "final_status": "failed",
                    "error_message": str(exc),
                    "metrics_summary_json_path": str(metrics_json_path) if metrics_json_path else "",
                    "metrics_summary_markdown_path": str(metrics_md_path) if metrics_md_path else "",
                    "completed_at": utc_timestamp(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return


if __name__ == "__main__":
    main()
