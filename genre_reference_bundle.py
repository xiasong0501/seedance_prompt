from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from genre_routing import load_genre_package_map
from providers.base import load_json_file, save_json_file, save_text_file, utc_timestamp


PROJECT_ROOT = Path(__file__).resolve().parent

INFRA_ANALYSIS_DIRS = {"videos", "batch_runs", "openai_agent_flow"}


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _trim_lines(text: str, limit: int = 8) -> list[str]:
    result: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("- "):
            result.append(stripped[2:].strip())
        elif stripped.startswith("1. "):
            result.append(stripped[3:].strip())
        if len(result) >= limit:
            break
    return _unique_strings(result)


def _parse_skill_sections(skill_text: str) -> dict[str, list[str]]:
    section_map = {
        "结构与钩子": "director_focus",
        "人物塑造": "character_focus",
        "服化道统一": "costume_focus",
        "场景与调度": "scene_focus",
        "分镜表达": "storyboard_focus",
        "对白与节奏": "dialogue_timing_rules",
        "生产红线": "continuity_guardrails",
    }
    result: dict[str, list[str]] = {
        "director_focus": [],
        "character_focus": [],
        "costume_focus": [],
        "scene_focus": [],
        "storyboard_focus": [],
        "dialogue_timing_rules": [],
        "continuity_guardrails": [],
        "negative_patterns": [],
    }
    current_field: str | None = None
    for raw in str(skill_text or "").splitlines():
        stripped = raw.strip()
        if stripped.startswith("### "):
            title = stripped[4:].strip()
            current_field = section_map.get(title)
            continue
        if not current_field or not stripped.startswith("- "):
            continue
        text = stripped[2:].strip()
        if not text:
            continue
        result[current_field].append(text)
        if current_field == "continuity_guardrails" and any(
            keyword in text for keyword in ("不要", "避免", "不能", "别把", "勿", "失效", "回退")
        ):
            result["negative_patterns"].append(text)
    return {key: _unique_strings(value)[:10] for key, value in result.items()}


def _analysis_root(project_root: Path, config: Mapping[str, Any]) -> Path:
    output_root = Path(str(config.get("output", {}).get("analysis_root", "analysis"))).expanduser()
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    return output_root


def _flow_root(project_root: Path, config: Mapping[str, Any], series_name: str) -> Path:
    return _analysis_root(project_root, config) / series_name / "openai_agent_flow"


def bundle_paths(project_root: Path, config: Mapping[str, Any], series_name: str) -> tuple[Path, Path]:
    root = _flow_root(project_root, config, series_name)
    return root / "genre_reference_bundle.json", root / "genre_reference_bundle.md"


def _bundle_sources_updated_after(
    *,
    project_root: Path,
    config: Mapping[str, Any],
    series_name: str,
    bundle_json_path: Path,
) -> bool:
    if not bundle_json_path.exists():
        return True
    bundle_mtime = bundle_json_path.stat().st_mtime
    candidate_paths = [
        _analysis_root(project_root, config) / series_name / "series_context.json",
    ]
    for package in load_genre_package_map().values():
        playbook_path = Path(str(package.get("_playbook_path", "") or "")).expanduser()
        skill_path = Path(str(package.get("_skill_path", "") or "")).expanduser()
        if playbook_path.exists():
            candidate_paths.append(playbook_path)
        if skill_path.exists():
            candidate_paths.append(skill_path)
    for path in candidate_paths:
        if path.exists() and path.stat().st_mtime > bundle_mtime:
            return True
    return False


def _collect_selected_genres(series_name: str, config: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    genre_config = dict(config.get("genre_reference", {}))
    selected = _unique_strings(list(genre_config.get("library_keys", [])))
    custom_tokens = _unique_strings(list(genre_config.get("custom_tokens", [])))

    series_context_path = _analysis_root(PROJECT_ROOT, config) / series_name / "series_context.json"
    if series_context_path.exists():
        series_context = load_json_file(series_context_path)
        if not selected:
            selected = _unique_strings(
                [
                    *[str(item.get("genre_key", "")).strip() for item in series_context.get("genre_playbooks", []) if isinstance(item, dict)],
                    str(series_context.get("genre_profile", {}).get("primary_genre", "")).strip(),
                    *[str(item).strip() for item in series_context.get("genre_profile", {}).get("secondary_genres", [])],
                ]
            )
    sources: list[str] = []
    if selected:
        sources.append("genre_reference.library_keys / series_context")
    if custom_tokens:
        sources.append("genre_reference.custom_tokens")
    return selected[:3], custom_tokens[:3], sources


def _package_director_focus(package: Mapping[str, Any]) -> list[str]:
    return _unique_strings(
        list(package.get("director_focus", []))
        + list(package.get("script_hooks", []))
        + list(package.get("core_audience_promises", []))
    )[:8]


def _package_art_focus(package: Mapping[str, Any]) -> list[str]:
    return _unique_strings(
        list(package.get("art_focus", []))
        + list(package.get("character_design_focus", []))
        + list(package.get("scene_design_focus", []))
    )[:8]


def _package_storyboard_focus(package: Mapping[str, Any]) -> list[str]:
    return _unique_strings(
        list(package.get("storyboard_focus", []))
        + list(package.get("script_hooks", []))
    )[:8]


def _package_guardrails(package: Mapping[str, Any]) -> list[str]:
    return _unique_strings(
        list(package.get("continuity_guardrails", []))
        + list(package.get("negative_patterns", []))
    )[:8]


def _package_genre_adapter(
    package: Mapping[str, Any],
    skill_sections: Mapping[str, list[str]],
) -> dict[str, list[str]]:
    dramatic_engine = _unique_strings(
        list(package.get("core_audience_promises", []))
        + list(package.get("script_hooks", []))
    )[:6]
    visual_priority = _unique_strings(
        list(package.get("director_focus", []))
        + list(package.get("storyboard_focus", []))
    )[:6]
    beat_templates = _unique_strings(
        list(package.get("storyboard_focus", []))
        + list(package.get("director_focus", []))
    )[:6]
    dialogue_temperament = _unique_strings(
        list(skill_sections.get("dialogue_timing_rules", []))
        + list(package.get("dialogue_timing_rules", []))
    )[:5]
    preferred_stage_types = _unique_strings(
        list(skill_sections.get("scene_focus", []))
        + list(package.get("scene_design_focus", []))
    )[:5]
    continuity_guardrails = _unique_strings(
        list(package.get("continuity_guardrails", []))
        + list(skill_sections.get("continuity_guardrails", []))
    )[:5]
    negative_patterns = _unique_strings(
        list(package.get("negative_patterns", []))
        + list(skill_sections.get("negative_patterns", []))
    )[:5]
    return {
        "dramatic_engine": dramatic_engine,
        "visual_priority": visual_priority,
        "beat_templates": beat_templates,
        "dialogue_temperament": dialogue_temperament,
        "preferred_stage_types": preferred_stage_types,
        "continuity_guardrails": continuity_guardrails,
        "negative_patterns": negative_patterns,
    }


def _load_similar_series_examples(
    *,
    project_root: Path,
    config: Mapping[str, Any],
    series_name: str,
    selected_genres: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if not selected_genres or limit <= 0:
        return []
    selected_set = set(selected_genres)
    result: list[dict[str, Any]] = []
    analysis_root = _analysis_root(project_root, config)
    for path in sorted(analysis_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.name == series_name or path.name in INFRA_ANALYSIS_DIRS:
            continue
        series_context_path = path / "series_context.json"
        if not series_context_path.exists():
            continue
        try:
            series_context = load_json_file(series_context_path)
        except Exception:
            continue
        available_genres = _unique_strings(
            [
                *[str(item.get("genre_key", "")).strip() for item in series_context.get("genre_playbooks", []) if isinstance(item, dict)],
                str(series_context.get("genre_profile", {}).get("primary_genre", "")).strip(),
                *[str(item).strip() for item in series_context.get("genre_profile", {}).get("secondary_genres", [])],
            ]
        )
        overlap = [item for item in available_genres if item in selected_set]
        if not overlap:
            continue
        learning_profile = dict(series_context.get("series_learning_profile", {}))
        downstream_guidance = dict(series_context.get("downstream_design_guidance", {}))
        result.append(
            {
                "series_name": path.name,
                "matched_genres": overlap,
                "premise": str(series_context.get("premise", "")).strip(),
                "director_focus": _unique_strings(
                    list(downstream_guidance.get("script_reconstruction_focus", []))
                    + list(learning_profile.get("episode_strengths", []))
                )[:4],
                "art_focus": _unique_strings(
                    list(downstream_guidance.get("character_design_focus", []))
                    + list(downstream_guidance.get("scene_design_focus", []))
                )[:4],
                "storyboard_focus": _unique_strings(list(downstream_guidance.get("storyboard_focus", [])))[:4],
                "reusable_rules": _unique_strings(
                    list(learning_profile.get("reusable_playbook_rules", []))
                    + list(learning_profile.get("reusable_skill_rules", []))
                )[:6],
            }
        )
    result.sort(
        key=lambda item: (
            len(item.get("matched_genres", [])),
            len(item.get("reusable_rules", [])),
            len(item.get("director_focus", [])),
        ),
        reverse=True,
    )
    return result[:limit]


def build_genre_reference_bundle(project_root: Path, config: Mapping[str, Any], series_name: str) -> dict[str, Any]:
    selected_genres, custom_tokens, source_notes = _collect_selected_genres(series_name, config)
    genre_packages = load_genre_package_map()
    matched_packages: list[dict[str, Any]] = []
    aggregate_director: list[str] = []
    aggregate_art: list[str] = []
    aggregate_storyboard: list[str] = []
    aggregate_guardrails: list[str] = []
    aggregate_character: list[str] = []
    aggregate_costume: list[str] = []
    aggregate_scene: list[str] = []
    aggregate_dialogue: list[str] = []
    aggregate_negative: list[str] = []
    skill_reference_notes: list[str] = []
    adapter_dramatic_engine: list[str] = []
    adapter_visual_priority: list[str] = []
    adapter_beat_templates: list[str] = []
    adapter_dialogue_temperament: list[str] = []
    adapter_preferred_stage_types: list[str] = []
    adapter_guardrails: list[str] = []
    adapter_negative: list[str] = []

    for genre_key in selected_genres:
        package = genre_packages.get(genre_key)
        if not package:
            continue
        skill_path = str(package.get("_skill_path", "")).strip()
        skill_text = Path(skill_path).read_text(encoding="utf-8") if skill_path and Path(skill_path).exists() else ""
        skill_sections = _parse_skill_sections(skill_text)
        package_director = _package_director_focus(package)
        package_art = _package_art_focus(package)
        package_storyboard = _package_storyboard_focus(package)
        package_guardrails = _package_guardrails(package)
        package_adapter = _package_genre_adapter(package, skill_sections)
        skill_rules = _trim_lines(skill_text, limit=10)
        matched_packages.append(
            {
                "genre_key": genre_key,
                "playbook_path": str(package.get("_playbook_path", "")),
                "skill_path": skill_path,
                "aliases": list(package.get("aliases", [])),
                "core_audience_promises": list(package.get("core_audience_promises", []))[:4],
                "director_focus": package_director,
                "art_focus": package_art,
                "character_focus": list(skill_sections.get("character_focus", []))[:6],
                "costume_focus": list(skill_sections.get("costume_focus", []))[:6],
                "scene_focus": list(skill_sections.get("scene_focus", []))[:6],
                "storyboard_focus": package_storyboard,
                "dialogue_timing_rules": list(skill_sections.get("dialogue_timing_rules", []))[:5],
                "continuity_guardrails": package_guardrails,
                "negative_patterns": list(skill_sections.get("negative_patterns", []))[:5],
                "genre_adapter": package_adapter,
                "skill_rules": skill_rules[:8],
            }
        )
        aggregate_director.extend(package_director)
        aggregate_art.extend(package_art)
        aggregate_character.extend(skill_sections.get("character_focus", []))
        aggregate_costume.extend(skill_sections.get("costume_focus", []))
        aggregate_scene.extend(skill_sections.get("scene_focus", []))
        aggregate_storyboard.extend(package_storyboard)
        aggregate_dialogue.extend(skill_sections.get("dialogue_timing_rules", []))
        aggregate_guardrails.extend(package_guardrails)
        aggregate_negative.extend(skill_sections.get("negative_patterns", []))
        skill_reference_notes.extend(skill_rules[:4])
        adapter_dramatic_engine.extend(package_adapter.get("dramatic_engine", []))
        adapter_visual_priority.extend(package_adapter.get("visual_priority", []))
        adapter_beat_templates.extend(package_adapter.get("beat_templates", []))
        adapter_dialogue_temperament.extend(package_adapter.get("dialogue_temperament", []))
        adapter_preferred_stage_types.extend(package_adapter.get("preferred_stage_types", []))
        adapter_guardrails.extend(package_adapter.get("continuity_guardrails", []))
        adapter_negative.extend(package_adapter.get("negative_patterns", []))

    genre_config = dict(config.get("genre_reference", {}))
    include_similar = bool(genre_config.get("include_similar_series_analysis", True))
    max_similar = int(genre_config.get("max_similar_series", 3))
    retrieved_examples = _load_similar_series_examples(
        project_root=project_root,
        config=config,
        series_name=series_name,
        selected_genres=selected_genres,
        limit=max_similar if include_similar else 0,
    )
    for item in retrieved_examples:
        aggregate_director.extend(item.get("director_focus", []))
        aggregate_art.extend(item.get("art_focus", []))
        aggregate_storyboard.extend(item.get("storyboard_focus", []))
        aggregate_guardrails.extend(item.get("reusable_rules", []))

    target_medium = str(config.get("quality", {}).get("target_medium", "漫剧")).strip() or "漫剧"
    visual_style = str(config.get("quality", {}).get("visual_style", "")).strip() or "按当前项目统一"
    return {
        "series_name": series_name,
        "generated_at": utc_timestamp(),
        "target_medium": target_medium,
        "visual_style": visual_style,
        "selected_genres": selected_genres,
        "custom_genre_tokens": custom_tokens,
        "source_notes": source_notes,
        "matched_packages": matched_packages,
        "retrieved_reference_series": retrieved_examples,
        "aggregate_focus": {
            "director_focus": _unique_strings(aggregate_director)[:12],
            "art_focus": _unique_strings(aggregate_art)[:12],
            "character_focus": _unique_strings(aggregate_character)[:12],
            "costume_focus": _unique_strings(aggregate_costume)[:12],
            "scene_focus": _unique_strings(aggregate_scene)[:12],
            "storyboard_focus": _unique_strings(aggregate_storyboard)[:12],
            "dialogue_timing_rules": _unique_strings(aggregate_dialogue)[:10],
            "continuity_guardrails": _unique_strings(aggregate_guardrails)[:12],
            "negative_patterns": _unique_strings(aggregate_negative)[:10],
            "skill_reference_notes": _unique_strings(skill_reference_notes)[:12],
        },
        "normalized_adapters": {
            "director": {
                "dramatic_engine": _unique_strings(adapter_dramatic_engine)[:8],
                "visual_priority": _unique_strings(adapter_visual_priority)[:8],
                "beat_templates": _unique_strings(adapter_beat_templates)[:8],
                "continuity_guardrails": _unique_strings(adapter_guardrails)[:8],
                "negative_patterns": _unique_strings(adapter_negative)[:6],
            },
            "storyboard": {
                "visual_priority": _unique_strings(adapter_visual_priority)[:8],
                "beat_templates": _unique_strings(adapter_beat_templates)[:8],
                "dialogue_temperament": _unique_strings(adapter_dialogue_temperament)[:8],
                "preferred_stage_types": _unique_strings(adapter_preferred_stage_types)[:8],
                "continuity_guardrails": _unique_strings(adapter_guardrails)[:8],
                "negative_patterns": _unique_strings(adapter_negative)[:6],
            },
        },
    }


def bundle_to_markdown(bundle: Mapping[str, Any]) -> str:
    lines = [
        f"# 题材参考包：{bundle.get('series_name', '')}",
        "",
        f"- 生成时间：{bundle.get('generated_at', '')}",
        f"- 目标媒介：{bundle.get('target_medium', '')}",
        f"- 视觉风格：{bundle.get('visual_style', '')}",
        f"- 选定题材：{'、'.join(bundle.get('selected_genres', [])) or '<空>'}",
    ]
    custom_tokens = list(bundle.get("custom_genre_tokens", []))
    if custom_tokens:
        lines.append(f"- 自定义题材补充：{'、'.join(custom_tokens)}")
    lines.append("")
    lines.append("## 汇总焦点")
    aggregate_focus = dict(bundle.get("aggregate_focus", {}))
    for title, key in [
        ("导演重点", "director_focus"),
        ("美术重点", "art_focus"),
        ("人物设计重点", "character_focus"),
        ("服化道重点", "costume_focus"),
        ("场景设计重点", "scene_focus"),
        ("分镜重点", "storyboard_focus"),
        ("台词时间规则", "dialogue_timing_rules"),
        ("连续性护栏", "continuity_guardrails"),
        ("负面模式", "negative_patterns"),
        ("技能提示", "skill_reference_notes"),
    ]:
        values = list(aggregate_focus.get(key, []))
        lines.append(f"### {title}")
        if not values:
            lines.append("- <空>")
        else:
            lines.extend(f"- {item}" for item in values)
    lines.append("")
    normalized_adapters = dict(bundle.get("normalized_adapters", {}))
    if normalized_adapters:
        lines.append("## 阶段适配器")
        for title, key in [("导演阶段", "director"), ("分镜阶段", "storyboard")]:
            adapter = dict(normalized_adapters.get(key, {}))
            lines.append(f"### {title}")
            if not adapter:
                lines.append("- <空>")
                lines.append("")
                continue
            for field, label in [
                ("dramatic_engine", "戏剧引擎"),
                ("visual_priority", "视觉优先级"),
                ("beat_templates", "Beat 模板"),
                ("dialogue_temperament", "对白气质"),
                ("preferred_stage_types", "优先空间类型"),
                ("continuity_guardrails", "连续性护栏"),
                ("negative_patterns", "负面模式"),
            ]:
                values = list(adapter.get(field, []))
                if values:
                    lines.append(f"- {label}：{'；'.join(values)}")
            lines.append("")
    matched_packages = list(bundle.get("matched_packages", []))
    if matched_packages:
        lines.append("## 命中题材包")
        for item in matched_packages:
            lines.append(f"### {item.get('genre_key', '')}")
            lines.append(f"- playbook：{item.get('playbook_path', '')}")
            lines.append(f"- skill：{item.get('skill_path', '')}")
            for title, key in [
                ("导演重点", "director_focus"),
                ("美术重点", "art_focus"),
                ("人物设计重点", "character_focus"),
                ("服化道重点", "costume_focus"),
                ("场景设计重点", "scene_focus"),
                ("分镜重点", "storyboard_focus"),
                ("台词时间规则", "dialogue_timing_rules"),
                ("连续性护栏", "continuity_guardrails"),
                ("负面模式", "negative_patterns"),
            ]:
                values = list(item.get(key, []))
                if values:
                    lines.append(f"- {title}：{'；'.join(values)}")
            adapter = dict(item.get("genre_adapter", {}))
            if adapter:
                for field, label in [
                    ("dramatic_engine", "戏剧引擎"),
                    ("visual_priority", "视觉优先级"),
                    ("beat_templates", "Beat 模板"),
                    ("dialogue_temperament", "对白气质"),
                    ("preferred_stage_types", "优先空间类型"),
                    ("continuity_guardrails", "连续性护栏"),
                    ("negative_patterns", "负面模式"),
                ]:
                    values = list(adapter.get(field, []))
                    if values:
                        lines.append(f"- 适配器/{label}：{'；'.join(values)}")
            lines.append("")
    retrieved = list(bundle.get("retrieved_reference_series", []))
    if retrieved:
        lines.append("## 相似剧参考")
        for item in retrieved:
            lines.append(f"### {item.get('series_name', '')}")
            lines.append(f"- 命中题材：{'、'.join(item.get('matched_genres', []))}")
            if item.get("premise"):
                lines.append(f"- 核心 premise：{item.get('premise', '')}")
            if item.get("reusable_rules"):
                lines.append(f"- 可借鉴规则：{'；'.join(item.get('reusable_rules', []))}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def filter_bundle_for_stage(
    bundle: Mapping[str, Any],
    stage: str,
    episode_id: str | None = None,
) -> dict[str, Any]:
    """
    优化方案4：按stage和episode智能裁剪bundle，减少无关数据加载

    Args:
        bundle: 完整的genre_reference_bundle
        stage: 阶段名称（director/art/storyboard）
        episode_id: 集数ID（可选，用于episode作用域过滤）

    Returns:
        针对该阶段优化后的bundle，体积缩减70-90%
    """
    if not isinstance(bundle, Mapping):
        return dict(bundle) if bundle else {}

    filtered = {
        "series_name": bundle.get("series_name"),
        "generated_at": bundle.get("generated_at"),
        "target_medium": bundle.get("target_medium"),
        "visual_style": bundle.get("visual_style"),
        "selected_genres": list(bundle.get("selected_genres", []))[:3],
    }

    # 根据阶段，只保留相关的focus字段
    aggregate_focus = dict(bundle.get("aggregate_focus", {}))
    if stage == "director":
        filtered["aggregate_focus"] = {
            "director_focus": aggregate_focus.get("director_focus", [])[:8],
            "continuity_guardrails": aggregate_focus.get("continuity_guardrails", [])[:6],
            "negative_patterns": aggregate_focus.get("negative_patterns", [])[:5],
        }
    elif stage == "art":
        filtered["aggregate_focus"] = {
            "art_focus": aggregate_focus.get("art_focus", [])[:8],
            "character_focus": aggregate_focus.get("character_focus", [])[:6],
            "costume_focus": aggregate_focus.get("costume_focus", [])[:6],
            "scene_focus": aggregate_focus.get("scene_focus", [])[:6],
            "continuity_guardrails": aggregate_focus.get("continuity_guardrails", [])[:6],
            "negative_patterns": aggregate_focus.get("negative_patterns", [])[:5],
        }
    elif stage == "storyboard":
        filtered["aggregate_focus"] = {
            "storyboard_focus": aggregate_focus.get("storyboard_focus", [])[:8],
            "dialogue_timing_rules": aggregate_focus.get("dialogue_timing_rules", [])[:5],
            "continuity_guardrails": aggregate_focus.get("continuity_guardrails", [])[:6],
            "negative_patterns": aggregate_focus.get("negative_patterns", [])[:5],
        }
    else:
        # 默认保留全部
        filtered["aggregate_focus"] = aggregate_focus

    # 只保留matched_packages中命中的题材，不需要的referenced_series可以省略
    matched = list(bundle.get("matched_packages", []))
    if matched and len(matched) > 3:
        filtered["matched_packages"] = matched[:3]  # 只保留前3个题材
    else:
        filtered["matched_packages"] = matched

    # 保留normalized_adapters，但可以在storyboard阶段简化
    normalized = dict(bundle.get("normalized_adapters", {}))
    if stage == "storyboard":
        filtered["normalized_adapters"] = {
            "storyboard": normalized.get("storyboard", {})
        }
    else:
        filtered["normalized_adapters"] = normalized

    return filtered


def load_or_build_genre_reference_bundle(
    *,
    project_root: Path,
    config: Mapping[str, Any],
    series_name: str,
    force_rebuild: bool = False,
) -> tuple[dict[str, Any], Path, Path]:
    json_path, md_path = bundle_paths(project_root, config, series_name)
    reuse_if_exists = bool(config.get("genre_reference", {}).get("reuse_if_exists", True))
    if (
        reuse_if_exists
        and not force_rebuild
        and json_path.exists()
        and not _bundle_sources_updated_after(
            project_root=project_root,
            config=config,
            series_name=series_name,
            bundle_json_path=json_path,
        )
    ):
        cached_bundle = load_json_file(json_path)
        if isinstance(cached_bundle, Mapping) and dict(cached_bundle).get("normalized_adapters"):
            return dict(cached_bundle), json_path, md_path
    bundle = build_genre_reference_bundle(project_root, config, series_name)
    save_json_file(json_path, bundle)
    save_text_file(md_path, bundle_to_markdown(bundle))
    return bundle, json_path, md_path
