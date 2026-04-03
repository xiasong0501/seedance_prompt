from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_director_analysis import choose_script_path
from generate_seedance_prompt_review import load_storyboard_package
from generate_seedance_prompts import (
    SEEDANCE_PROMPTS_SCHEMA,
    build_asset_catalog,
    merge_telemetry_recorders,
    normalize_storyboard_result,
    render_seedance_markdown,
    repair_storyboard_density,
    resolve_assets_dir,
    resolve_episode_output_dir,
    resolve_series_name,
    resolve_storyboard_profile,
    storyboard_profile_settings,
    validate_scene_reference_presence,
    validate_storyboard_density,
)
from openai_agents.runtime_utils import (
    build_episode_ids,
    configure_openai_api,
    load_runtime_config,
    openai_json_completion,
    read_text,
)
from pipeline_telemetry import TelemetryRecorder, telemetry_span
from prompt_utils import render_prompt
from providers.base import build_provider_model_tag, save_json_file, utc_timestamp

DEFAULT_CONFIG_PATH = Path("config/openai_agent_flow.local.json")
DEFAULT_TECHNIQUES_PATH = PROJECT_ROOT / "prompts" / "seedance_prompt_refine" / "techniques.md"
MAX_SCRIPT_CHARS_FOR_REFINE = 18000

SEEDANCE_PROMPT_REFINE_DELTA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["changed_points"],
    "properties": {
        "changed_points": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "point_id",
                    "changed_fields",
                    "title",
                    "duration_hint",
                    "continuity_bridge",
                    "master_timeline",
                    "audio_design",
                    "prompt_text",
                    "risk_notes",
                ],
                "properties": {
                    "point_id": {"type": "string"},
                    "changed_fields": {"type": "array", "items": {"type": "string"}},
                    "title": {"type": ["string", "null"]},
                    "duration_hint": {"type": ["string", "null"]},
                    "continuity_bridge": {"type": ["string", "null"]},
                    "master_timeline": {
                        "anyOf": [
                            SEEDANCE_PROMPTS_SCHEMA["properties"]["prompt_entries"]["items"]["properties"]["master_timeline"],  # type: ignore[index]
                            {"type": "null"},
                        ]
                    },
                    "audio_design": {"type": ["string", "null"]},
                    "prompt_text": {"type": ["string", "null"]},
                    "risk_notes": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ]
                    },
                },
            },
        }
    },
}


def print_status(message: str) -> None:
    print(f"[seedance-refine] {message}", flush=True)


def metrics_paths(episode_output_dir: Path, provider_tag: str) -> tuple[Path, Path]:
    base = episode_output_dir / f"02-seedance-prompts.refine.metrics__{provider_tag}"
    return Path(f"{base}.json"), Path(f"{base}.md")


def stamp_path(episode_output_dir: Path, provider_tag: str) -> Path:
    return episode_output_dir / f"02-seedance-prompts.refine__{provider_tag}.json"


def report_markdown_path(episode_output_dir: Path, provider_tag: str) -> Path:
    return episode_output_dir / f"02-seedance-prompts.refine__{provider_tag}.md"


def render_metrics_markdown(report: Mapping[str, Any]) -> str:
    context = dict(report.get("context", {}) or {})
    totals = dict(report.get("totals", {}) or {})
    lines = [
        "# Seedance Prompt Refine 统计报告",
        "",
        f"- series_name：{context.get('series_name', '')}",
        f"- episode_id：{context.get('episode_id', '')}",
        f"- model：{context.get('model', '')}",
        f"- final_status：{context.get('final_status', '')}",
        f"- changed_point_count：{context.get('changed_point_count', 0)}",
        "",
        "## 总计",
        "",
        f"- steps：{totals.get('step_count', 0)}",
        f"- duration_seconds：{totals.get('duration_seconds', 0)}",
        f"- input_tokens：{totals.get('input_tokens', 0)}",
        f"- output_tokens：{totals.get('output_tokens', 0)}",
        f"- total_tokens：{totals.get('total_tokens', 0)}",
        "",
        "## 步骤",
        "",
        "| Step ID | 名称 | 状态 | 耗时(秒) | 输入tokens | 输出tokens | 总tokens | 备注 |",
        "|---------|------|------|---------:|-----------:|-----------:|---------:|------|",
    ]
    for step in report.get("steps", []):
        metadata = dict(step.get("metadata", {}) or {})
        note_parts: list[str] = []
        if "prompt_chars" in metadata:
            note_parts.append(f"prompt_chars={metadata['prompt_chars']}")
        if "source_script_path" in metadata:
            note_parts.append(f"script={Path(str(metadata['source_script_path'])).name}")
        if "storyboard_markdown_path" in metadata:
            note_parts.append(f"storyboard={Path(str(metadata['storyboard_markdown_path'])).name}")
        lines.append(
            f"| {step.get('step_id', '')} | {step.get('name', '')} | {step.get('status', '')} | "
            f"{step.get('duration_seconds', 0)} | {step.get('input_tokens', 0)} | {step.get('output_tokens', 0)} | {step.get('total_tokens', 0)} | {'；'.join(note_parts)} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def save_metrics(recorder: TelemetryRecorder, json_path: Path, md_path: Path) -> dict[str, Any]:
    report = recorder.to_dict()
    save_json_file(json_path, report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_metrics_markdown(report), encoding="utf-8")
    return report


def resolve_refine_techniques_path(config: Mapping[str, Any]) -> Path:
    explicit = str(config.get("sources", {}).get("seedance_prompt_refine_techniques_path") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return DEFAULT_TECHNIQUES_PATH


def resolve_source_script_path(config: Mapping[str, Any], episode_id: str) -> Path:
    selection_config = {
        "series": {"series_name": config.get("series", {}).get("series_name", "")},
        "script": {
            "series_dir": config.get("script", {}).get("series_dir")
            or config.get("series", {}).get("script_series_dir", ""),
            "script_path": config.get("script", {}).get("script_path")
            or config.get("source", {}).get("script_path_override", ""),
            "episode_id": "",
            "preferred_filename_suffixes": config.get("script", {}).get("preferred_filename_suffixes")
            or config.get("source", {}).get("preferred_filename_suffixes", []),
        },
    }
    return Path(choose_script_path(selection_config, episode_id)).expanduser().resolve()


def compress_text_middle(text: str, max_chars: int) -> str:
    content = str(text or "").strip()
    if len(content) <= max_chars:
        return content
    head = int(max_chars * 0.6)
    tail = max_chars - head - 32
    return f"{content[:head].rstrip()}\n\n...[中间已省略以控制输入体积]...\n\n{content[-tail:].lstrip()}"


def summarize_dialogue_blocks(raw_blocks: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in list(raw_blocks or []):
        if not isinstance(block, Mapping):
            continue
        blocks.append(
            {
                "speaker": str(block.get("speaker") or "").strip(),
                "line": str(block.get("line") or "").strip(),
                "start_second": block.get("start_second"),
                "end_second": block.get("end_second"),
                "delivery_note": str(block.get("delivery_note") or "").strip(),
            }
        )
    return blocks[:3]


def compact_master_timeline(raw_entries: Any) -> list[str]:
    compact_entries: list[str] = []
    for entry in list(raw_entries or []):
        if not isinstance(entry, Mapping):
            continue
        start_second = entry.get("start_second")
        end_second = entry.get("end_second")
        visual_beat = compress_text_middle(str(entry.get("visual_beat") or "").strip(), 240)
        speaker = str(entry.get("speaker") or "").strip()
        spoken_line = compress_text_middle(str(entry.get("spoken_line") or "").strip(), 80)
        audio_cues = compress_text_middle(str(entry.get("audio_cues") or "").strip(), 90)
        transition_hook = compress_text_middle(str(entry.get("transition_hook") or "").strip(), 90)
        dialogue_summary = ""
        dialogue_blocks = summarize_dialogue_blocks(entry.get("dialogue_blocks"))
        if dialogue_blocks:
            parts: list[str] = []
            for block in dialogue_blocks[:2]:
                speaker_label = str(block.get("speaker") or "").strip()
                line = compress_text_middle(str(block.get("line") or "").strip(), 56)
                if speaker_label or line:
                    parts.append(f"{speaker_label}:{line}".strip(":"))
            dialogue_summary = " / ".join(parts)
        compact_entries.append(
            f"{start_second}-{end_second}s | 画面:{visual_beat}"
            + (f" | 对白:{dialogue_summary}" if dialogue_summary else "")
            + (f" | 声音:{audio_cues}" if audio_cues else "")
            + (f" | 交棒:{transition_hook}" if transition_hook else "")
            + (f" | 主说话人:{speaker}:{spoken_line}" if speaker and spoken_line else "")
        )
    return compact_entries[:6]


def compact_storyboard_entry(item: Mapping[str, Any]) -> dict[str, Any]:
    prompt_text = compress_text_middle(str(item.get("prompt_text") or "").strip(), 900)
    return {
        "point_id": str(item.get("point_id") or "").strip(),
        "title": str(item.get("title") or "").strip(),
        "duration_hint": str(item.get("duration_hint") or "").strip(),
        "continuity_bridge": str(item.get("continuity_bridge") or "").strip(),
        "primary_refs": normalize_ref_list(item.get("primary_refs")),
        "secondary_refs": normalize_ref_list(item.get("secondary_refs")),
        "audio_design": str(item.get("audio_design") or "").strip(),
        "prompt_text": prompt_text,
        "risk_notes": [str(x).strip() for x in list(item.get("risk_notes") or []) if str(x).strip()][:4],
        "master_timeline": compact_master_timeline(item.get("master_timeline")),
    }


def build_compact_storyboard_package(storyboard_package: Mapping[str, Any]) -> dict[str, Any]:
    prompt_entries = [dict(item) for item in list(storyboard_package.get("prompt_entries") or []) if isinstance(item, Mapping)]
    compact_entries: list[dict[str, Any]] = []
    for index, item in enumerate(prompt_entries):
        current = compact_storyboard_entry(item)
        compact_entry = dict(current)
        compact_entry["prev_context"] = {}
        compact_entry["next_context"] = {}
        if index > 0:
            previous = prompt_entries[index - 1]
            compact_entry["prev_context"] = {
                "point_id": str(previous.get("point_id") or "").strip(),
                "title": str(previous.get("title") or "").strip(),
                "continuity_bridge": str(previous.get("continuity_bridge") or "").strip(),
                "duration_hint": str(previous.get("duration_hint") or "").strip(),
                "prompt_text": compress_text_middle(str(previous.get("prompt_text") or "").strip(), 420),
                "tail_transition_hook": (
                    str((list(previous.get("master_timeline") or [])[-1] or {}).get("transition_hook") or "").strip()
                    if list(previous.get("master_timeline") or [])
                    else ""
                ),
            }
        if index + 1 < len(prompt_entries):
            nxt = prompt_entries[index + 1]
            compact_entry["next_context"] = {
                "point_id": str(nxt.get("point_id") or "").strip(),
                "title": str(nxt.get("title") or "").strip(),
                "continuity_bridge": str(nxt.get("continuity_bridge") or "").strip(),
                "duration_hint": str(nxt.get("duration_hint") or "").strip(),
                "prompt_text": compress_text_middle(str(nxt.get("prompt_text") or "").strip(), 420),
                "opening_visual_beat": (
                    str((list(nxt.get("master_timeline") or [])[0] or {}).get("visual_beat") or "").strip()
                    if list(nxt.get("master_timeline") or [])
                    else ""
                ),
            }
        compact_entries.append(compact_entry)
    return {
        "episode_id": str(storyboard_package.get("episode_id") or "").strip(),
        "episode_title": str(storyboard_package.get("episode_title") or "").strip(),
        "materials_overview": compress_text_middle(str(storyboard_package.get("materials_overview") or "").strip(), 1600),
        "global_notes": [str(x).strip() for x in list(storyboard_package.get("global_notes") or []) if str(x).strip()][:16],
        "prompt_entries": compact_entries,
    }


def build_refine_prompt(
    *,
    compact_storyboard_package_json: str,
    seedance_prompt_refine_techniques: str,
    source_script_text: str,
) -> str:
    return render_prompt(
        "seedance_prompt_refine/user.md",
        {
            "compact_storyboard_package_json": compact_storyboard_package_json,
            "seedance_prompt_refine_techniques": seedance_prompt_refine_techniques,
            "source_script_text": source_script_text,
        },
    )


def point_payload_map(data: Mapping[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in list(data.get("prompt_entries") or []):
        point_id = str(item.get("point_id") or "").strip()
        if not point_id:
            continue
        mapping[point_id] = json.dumps(item, ensure_ascii=False, sort_keys=True)
    return mapping


def point_item_map(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for item in list(data.get("prompt_entries") or []):
        point_id = str(item.get("point_id") or "").strip()
        if point_id:
            mapping[point_id] = dict(item)
    return mapping


def render_text_diff(before: str, after: str, *, context: int = 1) -> list[str]:
    before_lines = [line.rstrip() for line in str(before or "").splitlines()]
    after_lines = [line.rstrip() for line in str(after or "").splitlines()]
    diff_lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile="before",
            tofile="after",
            lineterm="",
            n=context,
        )
    )
    return diff_lines[:80]


def summarize_changed_fields(before_item: Mapping[str, Any], after_item: Mapping[str, Any]) -> list[str]:
    fields = {
        "prompt_text": "正文提示词",
        "master_timeline": "主时间线",
        "audio_design": "声音设计",
        "continuity_bridge": "承接关系",
        "duration_hint": "建议时长",
        "title": "标题",
    }
    changed: list[str] = []
    for field, label in fields.items():
        before_value = before_item.get(field)
        after_value = after_item.get(field)
        if json.dumps(before_value, ensure_ascii=False, sort_keys=True) != json.dumps(after_value, ensure_ascii=False, sort_keys=True):
            changed.append(label)
    return changed


def backup_file(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.{suffix}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def normalize_ref_list(raw: Any) -> list[str]:
    return [str(item).strip() for item in list(raw or []) if str(item).strip()]


def normalize_changed_fields(raw: Any) -> list[str]:
    allowed = {
        "title",
        "duration_hint",
        "continuity_bridge",
        "master_timeline",
        "audio_design",
        "prompt_text",
        "risk_notes",
    }
    normalized: list[str] = []
    for item in list(raw or []):
        field = str(item).strip()
        if field and field in allowed and field not in normalized:
            normalized.append(field)
    return normalized


def merge_global_notes(original_package: Mapping[str, Any], revised_package: Mapping[str, Any]) -> list[str]:
    notes: list[str] = []
    for note in list(original_package.get("global_notes") or []):
        clean = str(note).strip()
        if clean and clean not in notes:
            notes.append(clean)
    for note in list(revised_package.get("global_notes") or []):
        clean = str(note).strip()
        if clean and clean not in notes:
            notes.append(clean)
    return notes[:20]


def merge_refine_result_with_original(
    *,
    original_package: Mapping[str, Any],
    refine_delta: Mapping[str, Any],
) -> dict[str, Any]:
    original_entries = [dict(item) for item in list(original_package.get("prompt_entries") or []) if isinstance(item, Mapping)]
    revised_entry_map = {
        str(item.get("point_id") or "").strip(): dict(item)
        for item in list(refine_delta.get("changed_points") or [])
        if isinstance(item, Mapping) and str(item.get("point_id") or "").strip()
    }

    merged_entries: list[dict[str, Any]] = []
    for original_item in original_entries:
        point_id = str(original_item.get("point_id") or "").strip()
        revised_item = revised_entry_map.get(point_id)
        if not revised_item:
            merged_entries.append(original_item)
            continue

        merged = dict(original_item)
        changed_fields = normalize_changed_fields(revised_item.get("changed_fields"))
        if "title" in changed_fields:
            merged["title"] = str(revised_item.get("title") or "").strip() or merged.get("title", "")
        if "duration_hint" in changed_fields:
            merged["duration_hint"] = str(revised_item.get("duration_hint") or "").strip()
        if "continuity_bridge" in changed_fields:
            merged["continuity_bridge"] = str(revised_item.get("continuity_bridge") or "").strip()
        if "audio_design" in changed_fields:
            merged["audio_design"] = str(revised_item.get("audio_design") or "").strip()
        if "prompt_text" in changed_fields:
            merged["prompt_text"] = str(revised_item.get("prompt_text") or "").strip()
        if "master_timeline" in changed_fields:
            merged["master_timeline"] = [
                dict(item) for item in list(revised_item.get("master_timeline") or []) if isinstance(item, Mapping)
            ]
        if "risk_notes" in changed_fields:
            merged["risk_notes"] = [str(x).strip() for x in list(revised_item.get("risk_notes") or []) if str(x).strip()]

        merged_entries.append(merged)

    merged_package = dict(original_package)
    merged_package["episode_id"] = str(original_package.get("episode_id") or "").strip()
    merged_package["episode_title"] = str(original_package.get("episode_title") or "").strip()
    merged_package["materials_overview"] = str(original_package.get("materials_overview") or "").strip()
    merged_package["global_notes"] = merge_global_notes(original_package, {})
    merged_package["prompt_entries"] = merged_entries
    return merged_package


def extract_ref_ids_from_text(text: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for ref_id in re.findall(r"@图片\d+", str(text or "")):
        if ref_id in seen:
            continue
        seen.add(ref_id)
        ordered.append(ref_id)
    return ordered


def extract_ref_ids_from_entry_texts(item: Mapping[str, Any]) -> list[str]:
    chunks: list[str] = [
        str(item.get("title") or ""),
        str(item.get("duration_hint") or ""),
        str(item.get("continuity_bridge") or ""),
        str(item.get("audio_design") or ""),
        str(item.get("prompt_text") or ""),
    ]
    for entry in list(item.get("master_timeline") or []):
        if not isinstance(entry, Mapping):
            continue
        for value in entry.values():
            if isinstance(value, str):
                chunks.append(value)
            elif isinstance(value, list):
                for sub_value in value:
                    if isinstance(sub_value, Mapping):
                        for leaf in sub_value.values():
                            if isinstance(leaf, str):
                                chunks.append(leaf)
                    elif isinstance(sub_value, str):
                        chunks.append(sub_value)
    return extract_ref_ids_from_text("\n".join(chunks))


def freeze_ref_integrity(
    *,
    original_package: Mapping[str, Any],
    refined_package: Mapping[str, Any],
    episode_id: str,
) -> tuple[dict[str, Any], list[str]]:
    original_items = point_item_map(original_package)
    frozen_entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    for raw_item in list(refined_package.get("prompt_entries") or []):
        if not isinstance(raw_item, Mapping):
            continue
        item = dict(raw_item)
        point_id = str(item.get("point_id") or "").strip()
        original_item = original_items.get(point_id)
        if not original_item:
            frozen_entries.append(item)
            continue

        original_primary_refs = normalize_ref_list(original_item.get("primary_refs"))
        original_secondary_refs = normalize_ref_list(original_item.get("secondary_refs"))
        item["primary_refs"] = original_primary_refs
        item["secondary_refs"] = original_secondary_refs

        allowed_refs = set(original_primary_refs + original_secondary_refs)
        mentioned_refs = extract_ref_ids_from_entry_texts(item)
        illegal_refs = [ref_id for ref_id in mentioned_refs if ref_id not in allowed_refs]
        if illegal_refs:
            for field in ("continuity_bridge", "audio_design", "prompt_text", "master_timeline", "risk_notes"):
                item[field] = original_item.get(field)
            warnings.append(
                f"{episode_id} 的 {point_id} 在 refine 文本中引入了未声明引用 {illegal_refs}，已回退该分镜的文本类改动并锁定原始引用映射。"
            )
        frozen_entries.append(item)

    frozen_package = dict(refined_package)
    frozen_package["materials_overview"] = str(original_package.get("materials_overview") or "").strip()
    frozen_package["prompt_entries"] = frozen_entries
    return frozen_package, warnings


def render_refine_report(
    *,
    series_name: str,
    episode_id: str,
    model: str,
    original_package: Mapping[str, Any],
    refined_package: Mapping[str, Any],
    changed_points: list[str],
    storyboard_markdown_path: Path,
    source_script_path: Path,
    refine_techniques_path: Path,
    backup_markdown_path: Path | None,
    backup_json_path: Path | None,
) -> str:
    original_items = point_item_map(original_package)
    refined_items = point_item_map(refined_package)
    lines = [
        f"# Seedance Prompt Refine 报告 -- {series_name} {episode_id}",
        "",
        f"- model：{model}",
        f"- 当前 Seedance 提示词：{storyboard_markdown_path}",
        f"- 参考剧本：{source_script_path}",
        f"- refine 技巧来源：{refine_techniques_path}",
        f"- 备份 markdown：{backup_markdown_path or ''}",
        f"- 备份 json：{backup_json_path or ''}",
        f"- 修改分镜数：{len(changed_points)}",
        "",
        "## 本轮 refine 关注点",
        "",
        "- 人物引用、空间关系、入画退场、站位与视线是否清晰。",
        "- 相邻分镜是否重复、承接是否紧凑、场景变化是否合理。",
        "- 节奏、动作、特效、表情、材质、光影和声音细节是否更有吸引力。",
        "",
    ]
    if not changed_points:
        lines.extend(["本轮 refine 未发现需要落盘的改动。", ""])
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(["## 修改摘要", ""])
    for point_id in changed_points:
        before_item = original_items.get(point_id, {})
        after_item = refined_items.get(point_id, {})
        title = str(after_item.get("title") or before_item.get("title") or "").strip()
        changed_fields = summarize_changed_fields(before_item, after_item)
        lines.append(f"- {point_id} {title}：{('、'.join(changed_fields) if changed_fields else '内容有调整')}")

    for point_id in changed_points:
        before_item = original_items.get(point_id, {})
        after_item = refined_items.get(point_id, {})
        title = str(after_item.get("title") or before_item.get("title") or "").strip()
        changed_fields = summarize_changed_fields(before_item, after_item)
        lines.extend(["", f"## {point_id} {title}", ""])
        lines.append(f"- 变更字段：{('、'.join(changed_fields) if changed_fields else '未细分')}")
        prompt_diff = render_text_diff(str(before_item.get("prompt_text") or ""), str(after_item.get("prompt_text") or ""))
        if prompt_diff:
            lines.extend(["", "**正文差异**", "", "```diff"])
            lines.extend(prompt_diff)
            lines.append("```")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_pipeline(config: Mapping[str, Any], telemetry: TelemetryRecorder | None = None) -> dict[str, Any]:
    model, api_key = configure_openai_api(config)
    series_name = resolve_series_name(config)
    episode_ids = build_episode_ids(config.get("series", {}))
    assets_dir = resolve_assets_dir(config, series_name)
    character_prompts_text = read_text(assets_dir / "character-prompts.md")
    scene_prompts_text = read_text(assets_dir / "scene-prompts.md")
    refine_techniques_path = resolve_refine_techniques_path(config)
    refine_techniques_text = read_text(refine_techniques_path)
    timeout_seconds = int(config.get("run", {}).get("timeout_seconds", config.get("runtime", {}).get("timeout_seconds", 600)))
    temperature = float(config.get("run", {}).get("temperature", config.get("runtime", {}).get("temperature", 0.15)))
    dry_run = bool(config.get("run", {}).get("dry_run", config.get("runtime", {}).get("dry_run", False)))
    write_refine_metrics = bool(
        config.get("run", {}).get(
            "write_refine_metrics",
            config.get("runtime", {}).get("write_seedance_prompt_refine_metrics", True),
        )
    )
    storyboard_profile = resolve_storyboard_profile(config)
    profile_settings = storyboard_profile_settings(storyboard_profile)
    provider_tag = build_provider_model_tag("openai", model)

    print_status(f"剧名：{series_name}")
    print_status(f"素材目录：{assets_dir}")
    print_status(f"refine 技巧：{refine_techniques_path}")

    previews: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        asset_catalog = build_asset_catalog(
            character_prompts_text,
            scene_prompts_text,
            episode_id=episode_id,
            assets_dir=assets_dir,
        )
        episode_output_dir = resolve_episode_output_dir(config, series_name, episode_id)
        storyboard_md_path = episode_output_dir / "02-seedance-prompts.md"
        target_json_path = episode_output_dir / f"02-seedance-prompts__{provider_tag}.json"
        stamp_json_path = stamp_path(episode_output_dir, provider_tag)
        report_md_path = report_markdown_path(episode_output_dir, provider_tag)
        metrics_json_path, metrics_md_path = metrics_paths(episode_output_dir, provider_tag)

        preview: dict[str, Any] = {
            "episode_id": episode_id,
            "storyboard_markdown_path": str(storyboard_md_path),
            "storyboard_json_path": str(target_json_path),
            "refine_stamp_path": str(stamp_json_path),
            "refine_report_markdown_path": str(report_md_path),
        }
        if not storyboard_md_path.exists():
            preview["status"] = "skipped_missing_storyboard"
            preview["message"] = f"{episode_id} 未找到 02-seedance-prompts.md，无法 refine。"
            previews.append(preview)
            if not dry_run:
                print_status(preview["message"])
                results.append(preview)
            continue

        storyboard_package, storyboard_source_path, existing_json_path = load_storyboard_package(
            episode_output_dir,
            provider_tag,
            episode_id=episode_id,
        )
        compact_storyboard_package = build_compact_storyboard_package(storyboard_package)
        source_script_path = resolve_source_script_path(config, episode_id)
        source_script_text = compress_text_middle(read_text(source_script_path), MAX_SCRIPT_CHARS_FOR_REFINE)
        preview["storyboard_source_path"] = str(storyboard_source_path)
        preview["source_script_path"] = str(source_script_path)
        preview["refine_techniques_path"] = str(refine_techniques_path)
        preview["compact_prompt_entry_count"] = len(list(compact_storyboard_package.get("prompt_entries") or []))
        previews.append(preview)
        if dry_run:
            continue

        episode_telemetry = TelemetryRecorder(
            run_name="seedance-prompt-refine",
            context={
                "series_name": series_name,
                "episode_id": episode_id,
                "model": model,
                "storyboard_profile": storyboard_profile,
            },
        )
        try:
            with telemetry_span(
                episode_telemetry,
                stage="seedance_refine",
                name="load_seedance_refine_inputs",
                metadata={
                    "episode_id": episode_id,
                    "storyboard_markdown_path": str(storyboard_md_path),
                    "source_script_path": str(source_script_path),
                    "refine_techniques_path": str(refine_techniques_path),
                },
            ) as step:
                step["metadata"]["prompt_entry_count"] = len(list(storyboard_package.get("prompt_entries") or []))

            print_status(f"开始 refine {episode_id} 的 Seedance 提示词。")
            with telemetry_span(
                episode_telemetry,
                stage="seedance_refine",
                name="build_seedance_refine_prompt",
                metadata={"episode_id": episode_id},
            ) as step:
                compact_storyboard_package_json = json.dumps(
                    compact_storyboard_package,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                refine_prompt = build_refine_prompt(
                    compact_storyboard_package_json=compact_storyboard_package_json,
                    seedance_prompt_refine_techniques=refine_techniques_text,
                    source_script_text=source_script_text,
                )
                step["metadata"]["prompt_chars"] = len(refine_prompt)
                step["metadata"]["compact_storyboard_chars"] = len(compact_storyboard_package_json)
                step["metadata"]["source_script_chars"] = len(source_script_text)

            refine_delta = openai_json_completion(
                model=model,
                api_key=api_key,
                system_prompt=render_prompt("seedance_prompt_refine/system.md", {}),
                user_prompt=refine_prompt,
                schema_name="seedance_prompt_package_refine_delta",
                schema=SEEDANCE_PROMPT_REFINE_DELTA_SCHEMA,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                telemetry=episode_telemetry,
                stage="seedance_refine",
                step_name="seedance_prompt_refine_model_call",
                metadata={"episode_id": episode_id},
            )

            with telemetry_span(
                episode_telemetry,
                stage="seedance_refine",
                name="materialize_seedance_refine_outputs",
                metadata={"episode_id": episode_id},
            ) as step:
                merged_package = merge_refine_result_with_original(
                    original_package=storyboard_package,
                    refine_delta=refine_delta,
                )
                refined_package = normalize_storyboard_result(
                    merged_package,
                    frame_orientation=str(config.get("quality", {}).get("frame_orientation") or "9:16竖屏"),
                    storyboard_profile=storyboard_profile,
                    asset_catalog=asset_catalog,
                )
                refined_package = repair_storyboard_density(
                    refined_package,
                    max_shot_beats=profile_settings["max_shot_beats"],
                )
                refined_package, ref_integrity_warnings = freeze_ref_integrity(
                    original_package=storyboard_package,
                    refined_package=refined_package,
                    episode_id=episode_id,
                )
                grounding_warnings: list[str] = []
                density_warnings = validate_storyboard_density(refined_package, episode_id=episode_id)
                scene_ref_warnings = validate_scene_reference_presence(
                    refined_package,
                    asset_catalog,
                    episode_id=episode_id,
                )
                quality_warnings = density_warnings + scene_ref_warnings
                if quality_warnings:
                    step["metadata"]["quality_warning_count"] = len(quality_warnings)
                    step["metadata"]["quality_warnings"] = quality_warnings[:20]
                    episode_telemetry.context["quality_warning_count"] = len(quality_warnings)
                    episode_telemetry.context["quality_warnings"] = quality_warnings[:20]
                    print_status(
                        f"{episode_id} refine 后存在 {len(quality_warnings)} 条质量提示：结果照常落盘。"
                    )

                original_map = point_payload_map(storyboard_package)
                refined_map = point_payload_map(refined_package)
                changed_points = [
                    point_id
                    for point_id, payload in refined_map.items()
                    if original_map.get(point_id) != payload
                ]
                step["metadata"]["delta_changed_point_count"] = len(list(refine_delta.get("changed_points") or []))
                timestamp = utc_timestamp().replace(":", "").replace("-", "")
                md_backup = backup_file(storyboard_md_path, f"before-refine__{timestamp}")
                json_backup = backup_file(
                    target_json_path if target_json_path.exists() else existing_json_path or target_json_path,
                    f"before-refine__{timestamp}",
                )

                markdown = render_seedance_markdown(
                    series_name=series_name,
                    data=refined_package,
                    asset_catalog=asset_catalog,
                )
                storyboard_md_path.write_text(markdown, encoding="utf-8")
                save_json_file(target_json_path, refined_package)
                report_md_path.write_text(
                    render_refine_report(
                        series_name=series_name,
                        episode_id=episode_id,
                        model=model,
                        original_package=storyboard_package,
                        refined_package=refined_package,
                        changed_points=changed_points,
                        storyboard_markdown_path=storyboard_md_path,
                        source_script_path=source_script_path,
                        refine_techniques_path=refine_techniques_path,
                        backup_markdown_path=md_backup,
                        backup_json_path=json_backup,
                    ),
                    encoding="utf-8",
                )
                stamp = {
                    "episode_id": episode_id,
                    "series_name": series_name,
                    "model": model,
                    "storyboard_source_path": str(storyboard_source_path),
                    "storyboard_markdown_path": str(storyboard_md_path),
                    "source_script_path": str(source_script_path),
                    "refine_techniques_path": str(refine_techniques_path),
                    "backup_markdown_path": str(md_backup) if md_backup else "",
                    "backup_json_path": str(json_backup) if json_backup else "",
                    "changed_point_ids": changed_points,
                    "changed_point_count": len(changed_points),
                    "grounding_warning_count": len(grounding_warnings),
                    "ref_integrity_warning_count": len(ref_integrity_warnings),
                    "report_markdown_path": str(report_md_path),
                    "applied_at": utc_timestamp(),
                }
                save_json_file(stamp_json_path, stamp)
                episode_telemetry.context["changed_point_count"] = len(changed_points)
                step["metadata"]["changed_point_count"] = len(changed_points)
                step["metadata"]["backup_markdown_path"] = str(md_backup) if md_backup else ""
                step["metadata"]["backup_json_path"] = str(json_backup) if json_backup else ""
                if grounding_warnings:
                    step["metadata"]["grounding_warnings"] = grounding_warnings[:10]
                if ref_integrity_warnings:
                    step["metadata"]["ref_integrity_warnings"] = ref_integrity_warnings[:10]

            results.append(
                {
                    "episode_id": episode_id,
                    "status": "completed",
                    "storyboard_markdown_path": str(storyboard_md_path),
                    "storyboard_json_path": str(target_json_path),
                    "refine_stamp_path": str(stamp_json_path),
                    "refine_report_markdown_path": str(report_md_path),
                    "source_script_path": str(source_script_path),
                    "changed_point_count": len(changed_points),
                    "generated_at": utc_timestamp(),
                }
            )
            print_status(
                f"{episode_id} refine 完成：修改分镜 {results[-1]['changed_point_count']} 条，原文件已原地更新为 02-seedance-prompts。"
            )
            episode_telemetry.context["final_status"] = "completed"
        except Exception as exc:
            episode_telemetry.context["final_status"] = "failed"
            episode_telemetry.context["error"] = str(exc).strip()
            raise
        finally:
            if write_refine_metrics:
                metrics_report = save_metrics(episode_telemetry, metrics_json_path, metrics_md_path)
                print_status(
                    f"{episode_id} refine 统计：耗时 {metrics_report['totals']['duration_seconds']}s | "
                    f"tokens in/out/total = {metrics_report['totals']['input_tokens']}/{metrics_report['totals']['output_tokens']}/{metrics_report['totals']['total_tokens']}"
                )
                print_status(f"{episode_id} refine 统计报告：{metrics_json_path}")
            merge_telemetry_recorders(telemetry, episode_telemetry)

    if dry_run:
        preview = {
            "series_name": series_name,
            "model": model,
            "assets_dir": str(assets_dir),
            "refine_techniques_path": str(refine_techniques_path),
            "episodes": previews,
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return preview

    if episode_ids:
        series_output_dir = resolve_episode_output_dir(config, series_name, episode_ids[0]).parent
    else:
        series_output_dir = PROJECT_ROOT / "outputs"
    summary = {
        "series_name": series_name,
        "model": model,
        "assets_dir": str(assets_dir),
        "refine_techniques_path": str(refine_techniques_path),
        "results": results,
        "generated_at": utc_timestamp(),
    }
    summary_path = series_output_dir / "seedance-prompt-refine-summary.json"
    save_json_file(summary_path, summary)
    print_status(f"Seedance Prompt refine 链路完成：{summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine 02-seedance-prompts after the main OpenAI flow.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_runtime_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
