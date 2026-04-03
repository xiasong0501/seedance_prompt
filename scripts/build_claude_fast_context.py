#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflow_context_compaction import (
    compact_director_analysis_text,
    compact_episode_scoped_prompt_library,
    normalize_episode_id,
    shorten_text,
)
GENRES_ROOT = PROJECT_ROOT / "skills" / "production" / "video-script-reconstruction-skill" / "genres"

SKILL_SECTION_BUCKETS: dict[str, tuple[str, ...]] = {
    "director": ("分镜表达", "结构与钩子", "对白与节奏"),
    "art": ("人物塑造", "场景与调度", "服化道统一"),
    "storyboard": ("分镜表达", "对白与节奏", "结构与钩子", "场景与调度"),
    "guardrails": ("生产红线",),
}


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def dedupe_lines(values: list[str], *, item_limit: int = 120, total_limit: int = 12) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = shorten_text(str(raw or "").strip(), item_limit)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= total_limit:
            break
    return result


def extend_values(target: list[str], source: Any) -> None:
    if isinstance(source, list):
        for item in source:
            target.append(str(item or "").strip())
    elif isinstance(source, str):
        target.append(source.strip())


def parse_skill_sections(text: str) -> dict[str, list[str]]:
    current_section: str | None = None
    sections: dict[str, list[str]] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("### "):
            current_section = line[4:].strip()
            sections.setdefault(current_section, [])
            continue
        if line.startswith("## ") and "题材补充" not in line:
            current_section = line[3:].strip()
            sections.setdefault(current_section, [])
            continue
        if line.startswith("- ") and current_section:
            sections.setdefault(current_section, []).append(line[2:].strip())
    return sections


def collect_skill_bucket(section_map: Mapping[str, list[str]], bucket: str) -> list[str]:
    collected: list[str] = []
    for section_name in SKILL_SECTION_BUCKETS.get(bucket, ()):
        extend_values(collected, section_map.get(section_name, []))
    return collected


def official_genre_dirs() -> list[str]:
    result: list[str] = []
    for path in sorted(GENRES_ROOT.iterdir()):
        if not path.is_dir() or path.name == "__drafts__":
            continue
        result.append(path.name)
    return result


def resolve_genre_dir(genre_name: str) -> Path | None:
    direct_path = GENRES_ROOT / genre_name
    if direct_path.exists():
        return direct_path
    for genre_dir in GENRES_ROOT.iterdir():
        if not genre_dir.is_dir() or genre_dir.name == "__drafts__":
            continue
        playbook_path = genre_dir / "playbook.json"
        if not playbook_path.exists():
            continue
        playbook = load_json(playbook_path)
        aliases = [str(item).strip() for item in list(playbook.get("aliases") or []) if str(item).strip()]
        if genre_name == str(playbook.get("genre_key") or "").strip() or genre_name in aliases:
            return genre_dir
    return None


def normalize_episode_record(data: Mapping[str, Any], episode_id: str) -> dict[str, Any]:
    if not data:
        return {}
    normalized_target = normalize_episode_id(episode_id) or episode_id
    if normalize_episode_id(str(data.get("episode") or "")) == normalized_target:
        return dict(data)
    for key in ("episodes", "items", "records"):
        records = data.get(key)
        if not isinstance(records, list):
            continue
        for item in records:
            if not isinstance(item, Mapping):
                continue
            if normalize_episode_id(str(item.get("episode") or item.get("episode_id") or "")) == normalized_target:
                return dict(item)
    return {}


def choose_assets_dir(series_name: str) -> Path:
    preferred = PROJECT_ROOT / "assets" / f"{series_name}-claude"
    fallback = PROJECT_ROOT / "assets" / series_name
    if preferred.exists():
        return preferred
    return fallback


def choose_outputs_dir(series_name: str) -> Path:
    preferred = PROJECT_ROOT / "outputs" / f"{series_name}-claude"
    fallback = PROJECT_ROOT / "outputs" / series_name
    if preferred.exists():
        return preferred
    return fallback


def normalize_bundle_package(item: Mapping[str, Any]) -> dict[str, Any]:
    director_focus: list[str] = []
    art_focus: list[str] = []
    storyboard_focus: list[str] = []
    guardrails: list[str] = []
    negatives: list[str] = []

    for key in ("director_focus", "script_hooks", "core_audience_promises"):
        extend_values(director_focus, item.get(key))
    extend_values(director_focus, list(item.get("skill_rules") or [])[:4])

    for key in ("art_focus", "character_focus", "costume_focus", "scene_focus", "character_design_focus", "scene_design_focus"):
        extend_values(art_focus, item.get(key))
    extend_values(art_focus, list(item.get("skill_rules") or [])[4:8])

    for key in ("storyboard_focus", "script_hooks", "dialogue_timing_rules", "scene_focus", "director_focus"):
        extend_values(storyboard_focus, item.get(key))
    extend_values(storyboard_focus, list(item.get("skill_rules") or [])[:6])

    extend_values(guardrails, item.get("continuity_guardrails"))
    extend_values(guardrails, list(item.get("skill_rules") or [])[4:8])

    extend_values(negatives, item.get("negative_patterns"))

    return {
        "genre_key": str(item.get("genre_key") or "").strip(),
        "director_focus": dedupe_lines(director_focus, total_limit=12),
        "art_focus": dedupe_lines(art_focus, total_limit=12),
        "storyboard_focus": dedupe_lines(storyboard_focus, total_limit=12),
        "continuity_guardrails": dedupe_lines(guardrails, total_limit=10),
        "negative_patterns": dedupe_lines(negatives, total_limit=8),
        "source_mode": "genre_reference_bundle",
    }


def normalize_direct_package(genre_dir: Path) -> dict[str, Any]:
    playbook = load_json(genre_dir / "playbook.json")
    skill_text = read_text(genre_dir / "skill.md")
    skill_sections = parse_skill_sections(skill_text)

    director_focus: list[str] = []
    art_focus: list[str] = []
    storyboard_focus: list[str] = []
    guardrails: list[str] = []
    negatives: list[str] = []

    for key in ("director_focus", "script_hooks", "core_audience_promises", "storyboard_focus"):
        extend_values(director_focus, playbook.get(key))
    extend_values(director_focus, collect_skill_bucket(skill_sections, "director"))

    for key in ("art_focus", "character_focus", "costume_focus", "scene_focus", "character_design_focus", "scene_design_focus"):
        extend_values(art_focus, playbook.get(key))
    extend_values(art_focus, collect_skill_bucket(skill_sections, "art"))

    for key in ("storyboard_focus", "script_hooks", "dialogue_timing_rules", "scene_focus", "director_focus"):
        extend_values(storyboard_focus, playbook.get(key))
    extend_values(storyboard_focus, collect_skill_bucket(skill_sections, "storyboard"))

    extend_values(guardrails, playbook.get("continuity_guardrails"))
    extend_values(guardrails, collect_skill_bucket(skill_sections, "guardrails"))

    extend_values(negatives, playbook.get("negative_patterns"))

    return {
        "genre_key": str(playbook.get("genre_key") or genre_dir.name).strip(),
        "director_focus": dedupe_lines(director_focus, total_limit=12),
        "art_focus": dedupe_lines(art_focus, total_limit=12),
        "storyboard_focus": dedupe_lines(storyboard_focus, total_limit=12),
        "continuity_guardrails": dedupe_lines(guardrails, total_limit=10),
        "negative_patterns": dedupe_lines(negatives, total_limit=8),
        "source_mode": "raw_playbook_and_skill",
    }


def merge_phase_focus(packages: list[Mapping[str, Any]], key: str, *, total_limit: int = 12) -> list[str]:
    merged: list[str] = []
    for item in packages:
        extend_values(merged, item.get(key))
    return dedupe_lines(merged, total_limit=total_limit)


def build_markdown(pack: Mapping[str, Any]) -> str:
    lines = [
        "# Claude Fast Context Pack",
        "",
        f"- 剧名：{pack['series_name']}",
        f"- 集数：{pack['episode_id']}",
        f"- 生成时间：{pack['generated_at']}",
        f"- 视觉风格：{pack['visual_style'] or '未锁定'}",
        f"- 目标媒介：{pack['target_medium'] or '未锁定'}",
        f"- 剧本版本：{pack['selected_script_label'] or '未锁定'}",
        f"- 剧本文件：{pack['selected_script_file'] or '未锁定'}",
        "",
        "## 使用规则",
        "",
        "- 默认只读取本文件和当前阶段必需的上游产物。",
        "- 仅当本文件出现 warning、phase_focus 明显不足、或用户明确要求深挖时，才回退读取 raw genre playbook / skill。",
        "- 历史素材窗口只保留当前集与最近少量 episode block；不要默认加载整份系列库。",
        "",
        "## Warnings",
        "",
    ]
    warnings = list(pack.get("warnings") or [])
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- 无")

    def append_focus_block(title: str, payload: Mapping[str, Any]) -> None:
        lines.extend(["", f"## {title}", ""])
        lines.extend([f"- {item}" for item in list(payload.get("phase_focus") or [])] or ["- 无"])
        if payload.get("continuity_guardrails"):
            lines.extend(["", "连续性红线："])
            lines.extend([f"- {item}" for item in list(payload.get("continuity_guardrails") or [])])
        if payload.get("negative_patterns"):
            lines.extend(["", "高风险负面模式："])
            lines.extend([f"- {item}" for item in list(payload.get("negative_patterns") or [])])

    append_focus_block("导演速读卡", pack.get("director_context") or {})
    append_focus_block("服化道速读卡", pack.get("art_context") or {})
    append_focus_block("分镜速读卡", pack.get("storyboard_context") or {})

    compact_inputs = pack.get("compact_inputs") or {}
    lines.extend(
        [
            "",
            "## 历史素材窗口",
            "",
            "### Character Prompts",
            "",
            str(compact_inputs.get("character_prompts") or "<空>"),
            "",
            "### Scene Prompts",
            "",
            str(compact_inputs.get("scene_prompts") or "<空>"),
            "",
            "## 上游导演分析窗口",
            "",
            str(compact_inputs.get("director_analysis") or "<空>"),
            "",
            "## 题材来源",
            "",
        ]
    )

    for item in list(pack.get("genre_packages") or []):
        lines.append(
            f"- {item.get('genre_key')}：{item.get('source_mode')}，导演 {len(item.get('director_focus') or [])} 条，"
            f"服化道 {len(item.get('art_focus') or [])} 条，分镜 {len(item.get('storyboard_focus') or [])} 条"
        )

    return "\n".join(lines).rstrip() + "\n"


def build_context_pack(series_name: str, episode_id: str, *, max_recent_blocks: int) -> dict[str, Any]:
    analysis_dir = PROJECT_ROOT / "analysis" / series_name
    genre_selection = load_json(analysis_dir / "genre-selection.json")
    script_selection = normalize_episode_record(load_json(analysis_dir / "script-version-selection.json"), episode_id)
    bundle = load_json(analysis_dir / "openai_agent_flow" / "genre_reference_bundle.json")
    series_context = load_json(analysis_dir / "series_context.json")

    assets_dir = choose_assets_dir(series_name)
    outputs_dir = choose_outputs_dir(series_name)

    official_genres = official_genre_dirs()
    warnings: list[str] = []

    confirmed_genres = [str(item).strip() for item in list(genre_selection.get("confirmed_genres") or []) if str(item).strip()]
    provisional_genres = [str(item).strip() for item in list(genre_selection.get("provisional_genres") or []) if str(item).strip()]
    selected_genres = confirmed_genres or provisional_genres or [str(item).strip() for item in list(bundle.get("selected_genres") or []) if str(item).strip()]

    if confirmed_genres:
        for genre_name in confirmed_genres:
            if genre_name not in official_genres:
                warnings.append(f"已确认题材 `{genre_name}` 不在当前官方题材目录中，请核对 genre-selection.json。")

    bundle_matched = [item for item in list(bundle.get("matched_packages") or []) if isinstance(item, Mapping)]
    bundle_genre_keys = [str(item.get("genre_key") or "").strip() for item in bundle_matched if str(item.get("genre_key") or "").strip()]
    matched_bundle_packages = [item for item in bundle_matched if str(item.get("genre_key") or "").strip() in selected_genres] if selected_genres else bundle_matched[:3]

    if confirmed_genres and bundle_matched and not matched_bundle_packages:
        warnings.append(
            "confirmed_genres 与 genre_reference_bundle.matched_packages 无交集；fast pack 已回退到已确认题材的原始题材包。"
        )
    elif confirmed_genres and bundle_matched:
        bundle_only = [item for item in bundle_genre_keys if item not in confirmed_genres]
        if bundle_only:
            warnings.append(
                f"bundle 命中了额外题材 {bundle_only}；fast pack 仅保留与 confirmed_genres 交集，未直接注入这些额外题材。"
            )
        confirmed_missing_from_bundle = [item for item in confirmed_genres if item not in bundle_genre_keys]
        if confirmed_missing_from_bundle:
            warnings.append(
                f"confirmed_genres 中的 {confirmed_missing_from_bundle} 未命中 bundle；fast pack 已补读这些题材的原始 playbook / skill。"
            )

    genre_packages: list[dict[str, Any]] = [normalize_bundle_package(item) for item in matched_bundle_packages]

    fallback_direct_packages: list[dict[str, Any]] = []
    for genre_name in selected_genres:
        genre_dir = resolve_genre_dir(genre_name)
        if not genre_dir:
            warnings.append(f"无法解析题材目录：`{genre_name}`。")
            continue
        normalized = normalize_direct_package(genre_dir)
        fallback_direct_packages.append(normalized)

    if not genre_packages and fallback_direct_packages:
        genre_packages = fallback_direct_packages
    elif genre_packages and fallback_direct_packages:
        bundle_keys = {str(item.get("genre_key") or "").strip() for item in genre_packages}
        for item in fallback_direct_packages:
            if str(item.get("genre_key") or "").strip() not in bundle_keys:
                genre_packages.append(item)

    if not genre_packages:
        warnings.append("没有可用的题材经验包；fast 模式建议回退默认 `.claude` 工作流。")

    character_prompts_text = read_text(assets_dir / "character-prompts.md")
    scene_prompts_text = read_text(assets_dir / "scene-prompts.md")
    director_analysis_path = outputs_dir / episode_id / "01-director-analysis.md"
    director_analysis_text = read_text(director_analysis_path)

    compact_character_prompts = compact_episode_scoped_prompt_library(
        character_prompts_text,
        episode_id,
        limit=1400,
        max_recent_blocks=max_recent_blocks,
    ) or "<空>"
    compact_scene_prompts = compact_episode_scoped_prompt_library(
        scene_prompts_text,
        episode_id,
        limit=1600,
        max_recent_blocks=max_recent_blocks,
    ) or "<空>"
    compact_director_text = compact_director_analysis_text(director_analysis_text) or "<空>"

    selected_script_file = str(script_selection.get("selected_script_file") or "").strip()
    if selected_script_file:
        script_path = PROJECT_ROOT / selected_script_file
        if not script_path.exists():
            warnings.append(f"选中的剧本文件不存在：{selected_script_file}")
    else:
        script_dir = PROJECT_ROOT / "script" / series_name
        candidates = sorted(script_dir.glob(f"{episode_id}*.md")) if script_dir.exists() else []
        if len(candidates) == 1:
            selected_script_file = str(candidates[0].relative_to(PROJECT_ROOT))
            script_selection = {
                "episode": episode_id,
                "selected_script_file": selected_script_file,
                "selected_script_label": "auto_single_fallback",
                "selection_mode": "auto_single_fallback",
            }
        elif len(candidates) > 1:
            warnings.append("script-version-selection.json 缺失，且当前集存在多个剧本版本；fast 模式建议先完成剧本版本锁定。")
        else:
            warnings.append("script-version-selection.json 未锁定当前集剧本版本；fast 模式质量会下降。")

    if not genre_selection:
        warnings.append("genre-selection.json 缺失；fast pack 只能依赖 bundle 或原始题材目录。")

    if series_context and not bundle:
        warnings.append("当前剧缺少 genre_reference_bundle.json；fast pack 只能直接读原始题材包，速度会稍慢。")

    director_context = {
        "phase_focus": merge_phase_focus(genre_packages, "director_focus", total_limit=14),
        "continuity_guardrails": merge_phase_focus(genre_packages, "continuity_guardrails", total_limit=8),
        "negative_patterns": merge_phase_focus(genre_packages, "negative_patterns", total_limit=6),
    }
    art_context = {
        "phase_focus": merge_phase_focus(genre_packages, "art_focus", total_limit=14),
        "continuity_guardrails": merge_phase_focus(genre_packages, "continuity_guardrails", total_limit=8),
        "negative_patterns": merge_phase_focus(genre_packages, "negative_patterns", total_limit=6),
    }
    storyboard_context = {
        "phase_focus": merge_phase_focus(genre_packages, "storyboard_focus", total_limit=14),
        "continuity_guardrails": merge_phase_focus(genre_packages, "continuity_guardrails", total_limit=8),
        "negative_patterns": merge_phase_focus(genre_packages, "negative_patterns", total_limit=6),
    }

    for label, payload in (("导演", director_context), ("服化道", art_context), ("分镜", storyboard_context)):
        if not payload["phase_focus"]:
            warnings.append(f"{label}阶段没有抽取到有效 phase_focus；fast 模式建议补读原始题材包。")

    hard_review_required = bool(
        warnings
        or len(selected_genres) > 2
        or not selected_script_file
        or not genre_packages
    )

    return {
        "series_name": series_name,
        "episode_id": normalize_episode_id(episode_id) or episode_id,
        "generated_at": utc_timestamp(),
        "visual_style": str(genre_selection.get("visual_style") or bundle.get("visual_style") or "").strip(),
        "target_medium": str(genre_selection.get("target_medium") or bundle.get("target_medium") or "").strip(),
        "selection_mode": str(genre_selection.get("selection_mode") or "").strip(),
        "selected_genres": selected_genres,
        "official_genres": official_genres,
        "selected_script_file": selected_script_file,
        "selected_script_label": str(script_selection.get("selected_script_label") or "").strip(),
        "hard_review_required": hard_review_required,
        "warnings": dedupe_lines(warnings, item_limit=180, total_limit=12),
        "genre_packages": genre_packages,
        "director_context": director_context,
        "art_context": art_context,
        "storyboard_context": storyboard_context,
        "compact_inputs": {
            "character_prompts": compact_character_prompts,
            "scene_prompts": compact_scene_prompts,
            "director_analysis": compact_director_text,
        },
        "source_paths": {
            "genre_selection_json": str(analysis_dir / "genre-selection.json"),
            "script_version_selection_json": str(analysis_dir / "script-version-selection.json"),
            "genre_reference_bundle_json": str(analysis_dir / "openai_agent_flow" / "genre_reference_bundle.json"),
            "series_context_json": str(analysis_dir / "series_context.json"),
            "character_prompts_md": str(assets_dir / "character-prompts.md"),
            "scene_prompts_md": str(assets_dir / "scene-prompts.md"),
            "director_analysis_md": str(director_analysis_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact episode-scoped context for .claude fast workflow.")
    parser.add_argument("--series-name", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--max-recent-blocks", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    series_name = str(args.series_name).strip()
    episode_id = normalize_episode_id(args.episode_id) or str(args.episode_id).strip()
    output_dir = PROJECT_ROOT / "analysis" / series_name / "claude_fast" / episode_id
    output_dir.mkdir(parents=True, exist_ok=True)

    pack = build_context_pack(series_name, episode_id, max_recent_blocks=max(1, int(args.max_recent_blocks)))
    markdown = build_markdown(pack)

    json_path = output_dir / "context_pack.json"
    md_path = output_dir / "context_pack.md"
    json_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    print(
        json.dumps(
            {
                "series_name": series_name,
                "episode_id": episode_id,
                "context_pack_json": str(json_path),
                "context_pack_markdown": str(md_path),
                "hard_review_required": pack["hard_review_required"],
                "warnings": pack["warnings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
