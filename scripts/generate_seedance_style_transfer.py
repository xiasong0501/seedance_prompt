from __future__ import annotations

import argparse
import copy
import difflib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_director_analysis import choose_script_path
from generate_seedance_api_script import (
    choose_from_list,
    episode_sort_key,
    list_series_dirs,
)
from generate_seedance_prompt_refine import (
    MAX_SCRIPT_CHARS_FOR_REFINE,
    SEEDANCE_PROMPT_REFINE_DELTA_SCHEMA,
    build_compact_storyboard_package,
    compress_text_middle,
    freeze_ref_integrity,
    merge_refine_result_with_original,
    normalize_changed_fields,
    point_item_map,
    point_payload_map,
    render_text_diff,
    summarize_changed_fields,
)
from generate_seedance_prompt_review import load_storyboard_package
from generate_seedance_prompts import (
    build_asset_catalog,
    materialize_storyboard_item_from_master_timeline,
    normalize_storyboard_result,
    render_seedance_markdown,
    repair_storyboard_density,
    resolve_storyboard_profile,
    storyboard_profile_settings,
    validate_scene_reference_presence,
    validate_storyboard_density,
)
from openai_agents.runtime_utils import (
    configure_openai_api,
    load_runtime_config,
    openai_json_completion,
    read_text,
)
from pipeline_telemetry import TelemetryRecorder, telemetry_span
from prompt_utils import render_prompt
from providers.base import build_provider_model_tag, load_json_file, save_json_file, utc_timestamp

DEFAULT_CONFIG_PATH = Path("config/openai_agent_flow.local.json")
DEFAULT_BATCH_SIZE = 4
MIN_MEANINGFUL_PROMPT_RATIO = 0.9
MIN_MEANINGFUL_TIMELINE_RATIO = 0.82
DEFAULT_SEARCH_INDEX_PATH = PROJECT_ROOT / "prompt_library" / "SEARCH_INDEX.json"
DEFAULT_PROMPT_LIBRARY_ROOT = PROJECT_ROOT / "prompt_library"
DEFAULT_TEMPLATE_MATCH_CANDIDATE_COUNT = 6

PURPOSE_HINT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "揭示": ("揭", "真相", "记忆", "入口", "身份", "命运", "灵纹", "绑定", "回忆"),
    "羞辱": ("羞辱", "妖女", "毁容", "伤痕", "众目", "围观", "补刀", "污名", "丑"),
    "权力": ("宣判", "裁决", "高位", "门楣", "妾室", "规则", "压场", "神尊", "降妻"),
    "反击": ("反击", "退婚", "打断", "定义权", "回击", "冷下", "站回", "夺回"),
    "爱情": ("秘境", "相拥", "亲密", "温泉", "红线", "旖旎"),
    "守护": ("守", "护", "挡", "带走", "接住"),
    "痛苦": ("痛", "刺", "压住", "受辱", "失态"),
    "觉醒": ("觉醒", "骤变", "冷意", "命运", "灵体"),
    "尾钩": ("尾钩", "悬停", "卡点", "余震"),
}

TEMPLATE_MATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scored_templates"],
    "properties": {
        "scored_templates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["template_id", "match_score", "rationale", "learning_focus"],
                "properties": {
                    "template_id": {"type": "string"},
                    "match_score": {"type": "number"},
                    "rationale": {"type": "string"},
                    "learning_focus": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}


def print_status(message: str) -> None:
    print(f"[seedance-style] {message}", flush=True)


def render_metrics_markdown(report: Mapping[str, Any]) -> str:
    totals = dict(report.get("totals", {}) or {})
    lines = [
        f"# Seedance Style Transfer 统计 -- {report.get('context', {}).get('series_name', '')} {report.get('context', {}).get('episode_id', '')}",
        "",
        f"- run_name：{report.get('run_name', '')}",
        f"- started_at：{report.get('started_at', '')}",
        f"- finished_at：{report.get('finished_at', '')}",
        f"- steps：{totals.get('step_count', 0)}",
        f"- duration_seconds：{totals.get('duration_seconds', 0)}",
        f"- input_tokens：{totals.get('input_tokens', 0)}",
        f"- output_tokens：{totals.get('output_tokens', 0)}",
        f"- total_tokens：{totals.get('total_tokens', 0)}",
        "",
        "## 步骤",
        "",
        "| Step ID | 阶段 | 名称 | 状态 | 耗时(秒) | 输入tokens | 输出tokens | 总tokens | 备注 |",
        "|---------|------|------|------|---------:|-----------:|-----------:|---------:|------|",
    ]
    for step in list(report.get("steps") or []):
        metadata = dict(step.get("metadata", {}) or {})
        note_parts: list[str] = []
        for key in ["episode_id", "point_batch", "stage", "retry", "schema_name", "temperature"]:
            value = metadata.get(key)
            if value not in (None, "", [], {}):
                note_parts.append(f"{key}={value}")
        lines.append(
            f"| {step.get('step_id', '')} | {step.get('stage', '')} | {step.get('name', '')} | {step.get('status', '')} | "
            f"{step.get('duration_seconds', 0)} | {step.get('input_tokens', 0)} | {step.get('output_tokens', 0)} | {step.get('total_tokens', 0)} | {'；'.join(note_parts)} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def save_metrics(recorder: TelemetryRecorder, json_path: Path, md_path: Path) -> dict[str, Any]:
    report = recorder.to_dict()
    save_json_file(json_path, report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_metrics_markdown(report), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transfer style from prompt templates onto frozen Seedance prompts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--storyboard-json", default="", help="现有 02-seedance-prompts__*.json 路径")
    parser.add_argument("--storyboard-md", default="", help="现有 02-seedance-prompts.md 路径")
    parser.add_argument("--beat-catalog-json", default="", help="兼容旧流程的 seedance_beat_catalog.json 路径，可留空")
    parser.add_argument("--search-index-json", default="", help="prompt_library/SEARCH_INDEX.json 路径")
    parser.add_argument("--source-script", default="", help="可选。参考剧本路径")
    parser.add_argument("--point-ids", default="", help="逗号分隔的 point_id；留空或 all 表示全部")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-mode", choices=["sidecar", "overwrite"], default="")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--temperature", type=float, default=-1.0)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--auto-retry-weak-changes", action="store_true", help="检测到改动过小时自动补跑一轮")
    return parser


def prompt_input(message: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{message}{suffix}: ").strip()
    return raw or default


def prompt_yes_no(message: str, *, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"{message} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true"}


def prompt_choice(message: str, options: Sequence[str], default: str) -> str:
    display = "/".join(options)
    while True:
        raw = input(f"{message} [{display}] (default: {default}): ").strip().lower()
        if not raw:
            return default
        if raw in options:
            return raw
        print_status(f"无效选项：{raw}")


def ensure_file(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} 不存在：{path}")
    return path


def infer_series_name_from_output_dir(series_dir_name: str) -> str:
    for suffix in ("-gpt", "-claude", "-gpt-v0", "-gpt-v1", "-gpt-v2", "-gpt-v3", "-gpt-0", "-gpt-1"):
        if series_dir_name.endswith(suffix):
            return series_dir_name[: -len(suffix)]
    return re.sub(r"-(gpt|claude)(?:[-_].+)?$", "", series_dir_name, flags=re.IGNORECASE) or series_dir_name


def infer_context_from_storyboard_path(storyboard_json_path: Path) -> dict[str, str]:
    episode_id = storyboard_json_path.parent.name
    outputs_series_name = storyboard_json_path.parent.parent.name
    series_name = infer_series_name_from_output_dir(outputs_series_name)
    return {
        "episode_id": episode_id,
        "series_name": series_name,
        "outputs_series_name": outputs_series_name,
    }


def normalize_episode_id(episode_id: str) -> str:
    clean = str(episode_id or "").strip()
    if re.fullmatch(r"ep\d{2,}", clean):
        return clean
    if re.fullmatch(r"ep\d", clean):
        return f"ep{int(clean[2:]):02d}"
    return clean


def default_storyboard_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    storyboard_json = args.storyboard_json.strip()
    storyboard_md = args.storyboard_md.strip()
    if storyboard_json:
        json_path = ensure_file(storyboard_json, "storyboard json")
        md_default = find_storyboard_markdown_path(json_path.parent)
        md_path = ensure_file(storyboard_md or str(md_default), "storyboard markdown")
        return json_path, md_path
    if storyboard_md:
        md_path = ensure_file(storyboard_md, "storyboard markdown")
        json_default = find_storyboard_json_path(md_path.parent)
        json_path = ensure_file(str(json_default), "storyboard json")
        return json_path, md_path
    raise FileNotFoundError("请至少提供 storyboard json 或 markdown 路径。")


def choose_storyboard_paths_interactively() -> tuple[Path, Path]:
    series_dirs = list_series_dirs()
    if not series_dirs:
        raise RuntimeError("outputs/ 下没有找到可用的 Seedance 分镜目录。")
    series_idx = choose_from_list(
        "请选择要做 Seedance Style Transfer 的剧：",
        [path.name for path in series_dirs],
        default_index=0,
    )
    series_dir = series_dirs[series_idx]
    episode_dirs = sorted(
        [
            path
            for path in series_dir.iterdir()
            if path.is_dir() and (find_storyboard_markdown_path(path).exists() or find_storyboard_json_path(path).exists())
        ],
        key=lambda item: episode_sort_key(item.name),
    )
    if not episode_dirs:
        raise RuntimeError(f"{series_dir.name} 下没有找到可用集数。")
    episode_idx = choose_from_list(
        f"请选择 {series_dir.name} 的集数：",
        [path.name for path in episode_dirs],
        default_index=0,
    )
    episode_dir = episode_dirs[episode_idx]
    storyboard_json_path = ensure_file(str(find_storyboard_json_path(episode_dir)), "storyboard json")
    storyboard_md_path = ensure_file(str(find_storyboard_markdown_path(episode_dir)), "storyboard markdown")
    return storyboard_json_path, storyboard_md_path


def build_default_paths(storyboard_json_path: Path) -> dict[str, Path]:
    context = infer_context_from_storyboard_path(storyboard_json_path)
    assets_dir = PROJECT_ROOT / "assets" / context["outputs_series_name"]
    return {
        "search_index_json": DEFAULT_SEARCH_INDEX_PATH,
        "prompt_library_root": DEFAULT_PROMPT_LIBRARY_ROOT,
        "source_script_dir": PROJECT_ROOT / "script" / context["series_name"],
        "assets_dir": assets_dir,
    }


def find_storyboard_markdown_path(episode_dir: Path) -> Path:
    exact = episode_dir / "02-seedance-prompts.md"
    if exact.exists():
        return exact
    candidates = sorted(
        [
            path for path in episode_dir.glob("02-seedance-prompts*.md")
            if path.is_file() and not path.name.endswith(".report.md")
        ],
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not candidates:
        return exact
    return candidates[-1]


def find_storyboard_json_path(episode_dir: Path) -> Path:
    exact = episode_dir / "02-seedance-prompts__openai__gpt-5.4.json"
    if exact.exists():
        return exact
    candidates = sorted(
        [path for path in episode_dir.glob("02-seedance-prompts__*.json") if path.is_file()],
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not candidates:
        return exact
    return candidates[-1]


def resolve_source_script_path(
    config: Mapping[str, Any],
    *,
    series_name: str,
    episode_id: str,
    explicit_path: str,
) -> Path | None:
    if explicit_path.strip():
        path = Path(explicit_path).expanduser().resolve()
        return path if path.exists() else None
    selection_config = {
        "series": {"series_name": series_name},
        "script": {
            "series_dir": str(PROJECT_ROOT / "script" / series_name),
            "script_path": "",
            "episode_id": "",
            "preferred_filename_suffixes": config.get("source", {}).get("preferred_filename_suffixes", []),
        },
    }
    try:
        path = Path(choose_script_path(selection_config, episode_id)).expanduser().resolve()
    except Exception:
        return None
    return path if path.exists() else None


def read_json_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def chinese_ngrams(text: str, n: int) -> set[str]:
    compact = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", text or "")
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(0, len(compact) - n + 1)}


def text_similarity_score(left: str, right: str) -> float:
    score = 0.0
    for n, weight in ((2, 1.0), (3, 1.3), (4, 1.6)):
        left_set = chinese_ngrams(left, n)
        right_set = chinese_ngrams(right, n)
        if not left_set or not right_set:
            continue
        overlap = len(left_set & right_set)
        if overlap:
            score += weight * overlap / max(min(len(left_set), len(right_set)), 1)
    return score


def purpose_hint_score(point_text: str, beat: Mapping[str, Any]) -> float:
    purpose = str(beat.get("primary_purpose") or "").strip()
    if not purpose:
        return 0.0
    keywords = PURPOSE_HINT_KEYWORDS.get(purpose, ())
    hits = sum(1 for word in keywords if word and word in point_text)
    return 0.18 * hits


def duration_score(point: Mapping[str, Any], beat: Mapping[str, Any]) -> float:
    duration_hint = str(point.get("duration_hint") or "").strip()
    beat_duration = float(beat.get("restored_duration_seconds") or beat.get("duration_seconds") or 0.0)
    match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", duration_hint)
    if not match or not beat_duration:
        return 0.0
    low = float(match.group(1))
    high = float(match.group(2))
    center = (low + high) / 2.0
    diff = abs(center - beat_duration)
    return max(0.0, 0.18 - 0.03 * diff)


STYLE_TRANSFER_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["planned_points"],
    "properties": {
        "planned_points": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "point_id",
                    "template_ids",
                    "rewrite_goals",
                    "skeleton_steps",
                    "dialogue_freeze_notes",
                ],
                "properties": {
                    "point_id": {"type": "string"},
                    "template_ids": {"type": "array", "items": {"type": "string"}},
                    "rewrite_goals": {"type": "array", "items": {"type": "string"}},
                    "skeleton_steps": {"type": "array", "items": {"type": "string"}},
                    "dialogue_freeze_notes": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}


def extract_markdown_section(markdown: str, heading: str) -> str:
    pattern = re.compile(rf"^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s+|\Z)", flags=re.MULTILINE | re.DOTALL)
    match = pattern.search(str(markdown or ""))
    return match.group(1).strip() if match else ""


def extract_markdown_bullet_value(markdown: str, label: str) -> str:
    pattern = re.compile(rf"^- {re.escape(label)}：(.+?)$", flags=re.MULTILINE)
    match = pattern.search(str(markdown or ""))
    return match.group(1).strip() if match else ""


def parse_prompt_template_markdown(path: Path) -> dict[str, Any]:
    markdown = read_text(path)
    slots_text = extract_markdown_bullet_value(markdown, "必填槽位")
    slots = [slot.strip() for slot in re.split(r"[、，,]", slots_text) if slot.strip()]
    return {
        "prompt_library_path": str(path),
        "required_slots": slots,
        "restored_prompt": extract_markdown_section(markdown, "还原版 Prompt"),
        "general_template_prompt": extract_markdown_section(markdown, "通用模板 Prompt"),
        "retrieval_hint": extract_markdown_bullet_value(markdown, "检索建议"),
    }


def load_prompt_search_templates(search_index_path: Path) -> list[dict[str, Any]]:
    payload = load_json_file(search_index_path)
    templates: list[dict[str, Any]] = []
    for purpose_item in list(payload.get("purposes") or []):
        if not isinstance(purpose_item, Mapping):
            continue
        for raw_template in list(purpose_item.get("templates") or []):
            if not isinstance(raw_template, Mapping):
                continue
            template = dict(raw_template)
            template_path = PROJECT_ROOT / str(template.get("prompt_library_path") or "").strip()
            if not template_path.exists():
                continue
            template.update(parse_prompt_template_markdown(template_path))
            templates.append(template)
    return templates


def infer_point_purpose_candidates(point: Mapping[str, Any], available_purposes: Sequence[str]) -> list[str]:
    point_text = " ".join(
        [
            str(point.get("title") or ""),
            str(point.get("continuity_bridge") or ""),
            str(point.get("audio_design") or ""),
            str(point.get("prompt_text") or ""),
        ]
    )
    scored: list[tuple[float, str]] = []
    for purpose in available_purposes:
        keywords = PURPOSE_HINT_KEYWORDS.get(str(purpose).strip(), ())
        if not keywords:
            continue
        hits = sum(1 for word in keywords if word and word in point_text)
        if hits:
            scored.append((hits, purpose))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [purpose for _, purpose in scored[:3]]
    return selected or list(available_purposes)


def template_keyword_score(point_text: str, template: Mapping[str, Any]) -> float:
    score = 0.0
    for keyword in list(template.get("search_keywords") or []):
        keyword_text = str(keyword or "").strip()
        if keyword_text and keyword_text in point_text:
            score += 0.06 if len(keyword_text) <= 4 else 0.1
    return min(score, 0.6)


def template_structured_tag_score(point_text: str, template: Mapping[str, Any]) -> float:
    score = 0.0
    weighted_fields = [
        ("secondary_purposes", 0.1),
        ("scene_tags", 0.08),
        ("relation_tags", 0.08),
        ("staging_tags", 0.06),
        ("camera_tags", 0.06),
        ("emotion_tags", 0.07),
        ("narrative_tags", 0.07),
    ]
    for field, weight in weighted_fields:
        for raw_tag in list(template.get(field) or []):
            tag = str(raw_tag or "").strip()
            if tag and tag in point_text:
                score += weight
    return min(score, 0.45)


def heuristic_prompt_template_candidates(
    point: Mapping[str, Any],
    templates: Sequence[Mapping[str, Any]],
    *,
    top_k: int = DEFAULT_TEMPLATE_MATCH_CANDIDATE_COUNT,
) -> list[dict[str, Any]]:
    point_text = " ".join(
        [
            str(point.get("title") or ""),
            str(point.get("continuity_bridge") or ""),
            str(point.get("audio_design") or ""),
            str(point.get("prompt_text") or ""),
        ]
    )
    candidate_purposes = set(
        infer_point_purpose_candidates(
            point,
            sorted(
                {
                    str(item.get("primary_purpose") or item.get("purpose") or "").strip()
                    for item in templates
                    if str(item.get("primary_purpose") or item.get("purpose") or "").strip()
                }
            ),
        )
    )
    scored: list[dict[str, Any]] = []
    for template in templates:
        template_text = " ".join(
            [
                str(template.get("primary_purpose") or template.get("purpose") or ""),
                " ".join(str(item or "") for item in list(template.get("secondary_purposes") or [])),
                " ".join(str(item or "") for item in list(template.get("scene_tags") or [])),
                " ".join(str(item or "") for item in list(template.get("relation_tags") or [])),
                " ".join(str(item or "") for item in list(template.get("staging_tags") or [])),
                " ".join(str(item or "") for item in list(template.get("camera_tags") or [])),
                " ".join(str(item or "") for item in list(template.get("emotion_tags") or [])),
                " ".join(str(item or "") for item in list(template.get("narrative_tags") or [])),
                str(template.get("retrieval_title") or ""),
                str(template.get("search_hint") or ""),
                str(template.get("search_text") or ""),
                str(template.get("restored_prompt") or "")[:1200],
                str(template.get("general_template_prompt") or "")[:900],
            ]
        )
        score = text_similarity_score(point_text, template_text)
        score += template_keyword_score(point_text, template)
        score += template_structured_tag_score(point_text, template)
        score += duration_score(
            point,
            {
                "duration_seconds": template.get("duration_seconds"),
                "restored_duration_seconds": template.get("duration_seconds"),
            },
        )
        template_primary = str(template.get("primary_purpose") or template.get("purpose") or "").strip()
        if template_primary in candidate_purposes:
            score += 0.55
        elif set(str(item).strip() for item in list(template.get("secondary_purposes") or []) if str(item).strip()) & candidate_purposes:
            score += 0.18
        score += float(template.get("quality_score") or 0.0) * 0.15
        heuristic_score = max(0.0, min(round(score, 4), 1.0))
        scored.append({"score": heuristic_score, "template": dict(template)})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def build_template_match_prompt(
    *,
    point: Mapping[str, Any],
    candidate_templates: Sequence[Mapping[str, Any]],
) -> str:
    point_payload = {
        "point_id": str(point.get("point_id") or "").strip(),
        "title": str(point.get("title") or "").strip(),
        "duration_hint": str(point.get("duration_hint") or "").strip(),
        "continuity_bridge": compress_text_middle(str(point.get("continuity_bridge") or "").strip(), 220),
        "audio_design": compress_text_middle(str(point.get("audio_design") or "").strip(), 220),
        "prompt_text": compress_text_middle(str(point.get("prompt_text") or "").strip(), 2600),
        "master_timeline": [
            {
                "start_second": entry.get("start_second"),
                "end_second": entry.get("end_second"),
                "visual_beat": compress_text_middle(str(entry.get("visual_beat") or "").strip(), 280),
                "speaker": str(entry.get("speaker") or "").strip(),
                "spoken_line": compress_text_middle(str(entry.get("spoken_line") or "").strip(), 120),
            }
            for entry in list(point.get("master_timeline") or [])
            if isinstance(entry, Mapping)
        ],
    }
    template_payload = []
    for item in candidate_templates:
        template_payload.append(
            {
                "template_id": str(item.get("template_id") or "").strip(),
                "purpose": str(item.get("primary_purpose") or item.get("purpose") or "").strip(),
                "secondary_purposes": list(item.get("secondary_purposes") or []),
                "retrieval_title": str(item.get("retrieval_title") or "").strip(),
                "search_hint": str(item.get("search_hint") or "").strip(),
                "scene_tags": list(item.get("scene_tags") or []),
                "relation_tags": list(item.get("relation_tags") or []),
                "staging_tags": list(item.get("staging_tags") or []),
                "camera_tags": list(item.get("camera_tags") or []),
                "emotion_tags": list(item.get("emotion_tags") or []),
                "narrative_tags": list(item.get("narrative_tags") or []),
                "required_slots": list(item.get("required_slots") or []),
                "heuristic_score": float(item.get("heuristic_score") or item.get("score") or 0.0),
                "general_template_prompt": compress_text_middle(str(item.get("general_template_prompt") or "").strip(), 1600),
                "restored_prompt": compress_text_middle(str(item.get("restored_prompt") or "").strip(), 2200),
            }
        )
    return render_prompt(
        "seedance_style_template_match/user.md",
        {
            "point_json": json.dumps(point_payload, ensure_ascii=False, indent=2),
            "candidate_templates_json": json.dumps(template_payload, ensure_ascii=False, indent=2),
        },
    )


def rerank_prompt_templates_with_model(
    *,
    point: Mapping[str, Any],
    candidate_items: Sequence[Mapping[str, Any]],
    model: str,
    api_key: str,
    timeout_seconds: int,
    telemetry: TelemetryRecorder | None = None,
) -> list[dict[str, Any]]:
    if not candidate_items:
        return []
    candidate_templates = [dict(item.get("template") or item) for item in candidate_items]
    prompt = build_template_match_prompt(
        point=point,
        candidate_templates=candidate_templates,
    )
    scored_result = openai_json_completion(
        model=model,
        api_key=api_key,
        system_prompt=render_prompt("seedance_style_template_match/system.md", {}),
        user_prompt=prompt,
        schema_name="seedance_style_template_match",
        schema=TEMPLATE_MATCH_SCHEMA,
        temperature=0.1,
        timeout_seconds=timeout_seconds,
        telemetry=telemetry,
        stage="seedance_style_transfer",
        step_name="seedance_style_template_match_model_call",
        metadata={
            "point_id": str(point.get("point_id") or "").strip(),
            "candidate_count": len(candidate_items),
            "stage": "template-match",
        },
    )
    by_id = {
        str(item.get("template_id") or "").strip(): item
        for item in list(scored_result.get("scored_templates") or [])
        if isinstance(item, Mapping) and str(item.get("template_id") or "").strip()
    }
    reranked: list[dict[str, Any]] = []
    for item in candidate_items:
        template = dict(item.get("template") or item)
        template_id = str(template.get("template_id") or "").strip()
        model_item = dict(by_id.get(template_id) or {})
        heuristic_score = float(item.get("score") or item.get("heuristic_score") or 0.0)
        model_score = max(0.0, min(float(model_item.get("match_score") or 0.0), 1.0))
        final_score = round(model_score * 0.85 + heuristic_score * 0.15, 4)
        reranked.append(
            {
                "score": final_score,
                "model_score": round(model_score, 4),
                "heuristic_score": round(heuristic_score, 4),
                "rationale": str(model_item.get("rationale") or "").strip(),
                "learning_focus": [str(x).strip() for x in list(model_item.get("learning_focus") or []) if str(x).strip()],
                "template": template,
            }
        )
    reranked.sort(key=lambda item: (item["model_score"], item["score"], item["heuristic_score"]), reverse=True)
    return reranked


def recommend_prompt_templates(
    point: Mapping[str, Any],
    templates: Sequence[Mapping[str, Any]],
    *,
    model: str,
    api_key: str,
    timeout_seconds: int,
    telemetry: TelemetryRecorder | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    heuristic_candidates = heuristic_prompt_template_candidates(
        point,
        templates,
        top_k=max(top_k, DEFAULT_TEMPLATE_MATCH_CANDIDATE_COUNT),
    )
    reranked = rerank_prompt_templates_with_model(
        point=point,
        candidate_items=heuristic_candidates,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        telemetry=telemetry,
    )
    return reranked[:top_k]


def build_style_beat_summary(beat: Mapping[str, Any]) -> dict[str, Any]:
    shot_lines: list[str] = []
    for shot in list(beat.get("shot_chain") or [])[:4]:
        if not isinstance(shot, Mapping):
            continue
        shot_lines.append(
            " | ".join(
                part
                for part in [
                    f"{shot.get('time_range', '')}",
                    f"镜头入口:{compress_text_middle(str(shot.get('camera_entry') or '').strip(), 80)}",
                    f"镜头语言:{compress_text_middle(str(shot.get('camera_language') or '').strip(), 90)}",
                    f"动作:{compress_text_middle(str(shot.get('action_timeline') or '').strip(), 110)}",
                    f"光感:{compress_text_middle(str(shot.get('lighting_and_texture') or '').strip(), 110)}",
                    f"声音:{compress_text_middle(str(shot.get('sound_bed') or '').strip(), 90)}",
                    f"切镜钩子:{compress_text_middle(str(shot.get('transition_trigger') or '').strip(), 80)}",
                ]
                if part and not part.endswith(":")
            )
        )
    return {
        "beat_id": str(beat.get("beat_id") or "").strip(),
        "primary_purpose": str(beat.get("primary_purpose") or "").strip(),
        "display_title": str(beat.get("display_title") or beat.get("beat_id") or "").strip(),
        "display_summary": compress_text_middle(str(beat.get("display_summary") or "").strip(), 240),
        "dramatic_goal": compress_text_middle(str(beat.get("dramatic_goal") or "").strip(), 240),
        "visual_second_pass_summary": compress_text_middle(str(beat.get("visual_second_pass_summary") or "").strip(), 240),
        "restored_duration_seconds": beat.get("restored_duration_seconds"),
        "shot_style_lines": shot_lines,
    }


def recommend_style_beats(
    point: Mapping[str, Any],
    beats: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    point_text = " ".join(
        [
            str(point.get("title") or ""),
            str(point.get("continuity_bridge") or ""),
            str(point.get("audio_design") or ""),
            str(point.get("prompt_text") or ""),
        ]
    )
    scored: list[dict[str, Any]] = []
    for beat in beats:
        beat_text = " ".join(
            [
                str(beat.get("display_title") or ""),
                str(beat.get("display_summary") or ""),
                str(beat.get("dramatic_goal") or ""),
                str(beat.get("visual_second_pass_summary") or ""),
                str(beat.get("primary_purpose") or ""),
            ]
        )
        score = text_similarity_score(point_text, beat_text)
        score += purpose_hint_score(point_text, beat)
        score += duration_score(point, beat)
        scored.append(
            {
                "score": round(score, 4),
                "beat": beat,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def parse_point_selection(raw: str, available_ids: Sequence[str]) -> list[str]:
    clean = raw.strip().lower()
    if not clean or clean == "all":
        return list(available_ids)
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    valid = set(available_ids)
    unknown = [item for item in requested if item not in valid]
    if unknown:
        raise ValueError(f"未知 point_id：{unknown}")
    return requested


def choose_style_mapping(
    selected_points: Sequence[Mapping[str, Any]],
    templates: Sequence[Mapping[str, Any]],
    *,
    model: str,
    api_key: str,
    timeout_seconds: int,
    interactive: bool,
    telemetry: TelemetryRecorder | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    mapping: dict[str, list[str]] = {}
    recommendations_by_point: dict[str, list[dict[str, Any]]] = {}
    for point in selected_points:
        point_id = str(point.get("point_id") or "").strip()
        recommendations = recommend_prompt_templates(
            point,
            templates,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            telemetry=telemetry,
        )
        recommendations_by_point[point_id] = recommendations

    print_status("推荐 prompt 模板预览：")
    for point in selected_points:
        point_id = str(point.get("point_id") or "").strip()
        title = str(point.get("title") or "").strip()
        recs = recommendations_by_point.get(point_id, [])
        preview = "；".join(
            f"{index + 1}.{str(item['template'].get('template_id') or '')} {str(item['template'].get('retrieval_title') or '')} model={item.get('model_score', item['score'])} final={item['score']}"
            for index, item in enumerate(recs)
        )
        print_status(f"{point_id} {title} -> {preview}")

    if not interactive:
        for point in selected_points:
            point_id = str(point.get("point_id") or "").strip()
            mapping[point_id] = [
                str(item["template"].get("template_id") or "").strip()
                for item in recommendations_by_point.get(point_id, [])[:2]
                if str(item["template"].get("template_id") or "").strip()
            ]
        return mapping, recommendations_by_point

    if prompt_yes_no("是否直接采用全部推荐映射（每条取前2个 prompt 模板）", default=True):
        for point in selected_points:
            point_id = str(point.get("point_id") or "").strip()
            mapping[point_id] = [
                str(item["template"].get("template_id") or "").strip()
                for item in recommendations_by_point.get(point_id, [])[:2]
                if str(item["template"].get("template_id") or "").strip()
            ]
        return mapping, recommendations_by_point

    template_index = {
        str(template.get("template_id") or "").strip(): template
        for template in templates
        if str(template.get("template_id") or "").strip()
    }
    for point in selected_points:
        point_id = str(point.get("point_id") or "").strip()
        title = str(point.get("title") or "").strip()
        recs = recommendations_by_point.get(point_id, [])
        print()
        print_status(f"{point_id} {title}")
        for index, item in enumerate(recs, start=1):
            template = item["template"]
            print_status(
                f"  {index}. {template.get('template_id', '')} | {template.get('purpose', '')} | {template.get('retrieval_title', '')} | model={item.get('model_score', item['score'])} | final={item['score']}"
            )
        raw = input("输入 1 / 1,2 / 权力__ep01__SB04,羞辱__ep01__SB03 / skip，回车=默认前2个: ").strip()
        if not raw:
            mapping[point_id] = [
                str(item["template"].get("template_id") or "").strip()
                for item in recs[:2]
                if str(item["template"].get("template_id") or "").strip()
            ]
            continue
        if raw.lower() == "skip":
            mapping[point_id] = []
            continue
        chosen_ids: list[str] = []
        for token in [item.strip() for item in raw.split(",") if item.strip()]:
            if token.isdigit():
                index = int(token) - 1
                if 0 <= index < len(recs):
                    template_id = str(recs[index]["template"].get("template_id") or "").strip()
                    if template_id and template_id not in chosen_ids:
                        chosen_ids.append(template_id)
                continue
            normalized = token
            if normalized in template_index and normalized not in chosen_ids:
                chosen_ids.append(normalized)
        mapping[point_id] = chosen_ids
    return mapping, recommendations_by_point


def extract_point_dialogue_signature(item: Mapping[str, Any]) -> list[dict[str, str]]:
    signature: list[dict[str, str]] = []
    for entry in list(item.get("master_timeline") or []):
        if not isinstance(entry, Mapping):
            continue
        speaker = str(entry.get("speaker") or "").strip()
        spoken_line = str(entry.get("spoken_line") or "").strip()
        if speaker or spoken_line:
            signature.append({"speaker": speaker, "line": spoken_line})
        for block in list(entry.get("dialogue_blocks") or []):
            if not isinstance(block, Mapping):
                continue
            signature.append(
                {
                    "speaker": str(block.get("speaker") or "").strip(),
                    "line": str(block.get("line") or "").strip(),
                }
            )
    return signature


def _clone_dialogue_blocks(raw_blocks: Any) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for block in list(raw_blocks or []):
        if isinstance(block, Mapping):
            cloned.append(dict(block))
    return cloned


def _freeze_dialogue_payload_on_item(
    *,
    original_item: Mapping[str, Any],
    revised_item: Mapping[str, Any],
) -> dict[str, Any]:
    original_entries = [dict(entry) for entry in list(original_item.get("master_timeline") or []) if isinstance(entry, Mapping)]
    revised_entries = [dict(entry) for entry in list(revised_item.get("master_timeline") or []) if isinstance(entry, Mapping)]
    if not original_entries or not revised_entries:
        fallback = dict(revised_item)
        fallback["master_timeline"] = copy.deepcopy(original_item.get("master_timeline") or [])
        fallback["prompt_text"] = str(original_item.get("prompt_text") or "").strip()
        return fallback

    original_dialogue_payloads = [
        _clone_dialogue_blocks(entry.get("dialogue_blocks") or [])
        for entry in original_entries
    ]
    occupied_targets: set[int] = set()
    for entry in revised_entries:
        entry["dialogue_blocks"] = []
        entry["speaker"] = ""
        entry["spoken_line"] = ""
        entry["delivery_note"] = ""

    for original_index, dialogue_blocks in enumerate(original_dialogue_payloads):
        if not dialogue_blocks:
            continue
        if len(revised_entries) == len(original_entries):
            target_index = original_index
        elif len(original_entries) == 1:
            target_index = 0
        else:
            target_index = round(original_index * (len(revised_entries) - 1) / max(len(original_entries) - 1, 1))
        target_index = max(0, min(target_index, len(revised_entries) - 1))
        while target_index in occupied_targets and target_index + 1 < len(revised_entries):
            target_index += 1
        occupied_targets.add(target_index)
        revised_entries[target_index]["dialogue_blocks"] = dialogue_blocks

    patched_item = dict(revised_item)
    patched_item["master_timeline"] = revised_entries
    patched_item["prompt_text"] = ""
    return materialize_storyboard_item_from_master_timeline(patched_item)


def freeze_dialogue_integrity(
    *,
    original_package: Mapping[str, Any],
    refined_package: Mapping[str, Any],
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
        original_dialogue = extract_point_dialogue_signature(original_item)
        revised_dialogue = extract_point_dialogue_signature(item)
        if original_dialogue != revised_dialogue:
            item = _freeze_dialogue_payload_on_item(
                original_item=original_item,
                revised_item=item,
            )
            warnings.append(
                f"{point_id} 的对白签名发生变化，已冻结对白内容并按修订后的镜头结构回写 prompt_text。"
            )
        frozen_entries.append(item)
    frozen_package = dict(refined_package)
    frozen_package["prompt_entries"] = frozen_entries
    return frozen_package, warnings


def find_prompt_templates_by_ids(
    templates: Sequence[Mapping[str, Any]],
    template_ids: Sequence[str],
) -> list[dict[str, Any]]:
    by_id = {
        str(item.get("template_id") or "").strip(): dict(item)
        for item in templates
        if str(item.get("template_id") or "").strip()
    }
    selected: list[dict[str, Any]] = []
    for template_id in template_ids:
        item = by_id.get(template_id)
        if item:
            selected.append(item)
    return selected


def build_base_fact_locked_entry(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "point_id": str(item.get("point_id") or "").strip(),
        "title": str(item.get("title") or "").strip(),
        "duration_hint": str(item.get("duration_hint") or "").strip(),
        "continuity_bridge": compress_text_middle(str(item.get("continuity_bridge") or "").strip(), 260),
        "audio_design": compress_text_middle(str(item.get("audio_design") or "").strip(), 220),
        "primary_refs": copy.deepcopy(list(item.get("primary_refs") or [])),
        "secondary_refs": copy.deepcopy(list(item.get("secondary_refs") or [])),
        "master_timeline_facts": [
            {
                "start_second": entry.get("start_second"),
                "end_second": entry.get("end_second"),
                "visual_beat": compress_text_middle(str(entry.get("visual_beat") or "").strip(), 220),
                "speaker": str(entry.get("speaker") or "").strip(),
                "spoken_line": compress_text_middle(str(entry.get("spoken_line") or "").strip(), 120),
            }
            for entry in list(item.get("master_timeline") or [])
            if isinstance(entry, Mapping)
        ],
        "dialogue_signature": extract_point_dialogue_signature(item),
        "risk_notes": [str(note).strip() for note in list(item.get("risk_notes") or []) if str(note).strip()],
        "original_wording_is_disposable": True,
    }


def build_selected_points_package(
    *,
    storyboard_package: Mapping[str, Any],
    selected_point_ids: Sequence[str],
    style_mapping: Mapping[str, Sequence[str]],
    template_recommendations: Mapping[str, Sequence[Mapping[str, Any]]],
    prompt_templates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    compact_package = build_compact_storyboard_package(storyboard_package)
    compact_lookup = {
        str(item.get("point_id") or "").strip(): dict(item)
        for item in list(compact_package.get("prompt_entries") or [])
        if isinstance(item, Mapping) and str(item.get("point_id") or "").strip()
    }
    raw_lookup = {
        str(item.get("point_id") or "").strip(): copy.deepcopy(dict(item))
        for item in list(storyboard_package.get("prompt_entries") or [])
        if isinstance(item, Mapping) and str(item.get("point_id") or "").strip()
    }
    selected_entries: list[dict[str, Any]] = []
    template_pool: dict[str, dict[str, Any]] = {}
    for point_id in selected_point_ids:
        raw_item = raw_lookup[point_id]
        compact_item = compact_lookup[point_id]
        chosen_template_ids = [template_id for template_id in style_mapping.get(point_id, []) if template_id]
        chosen_templates = find_prompt_templates_by_ids(prompt_templates, chosen_template_ids)
        selected_recommendations = {
            str(item.get("template", {}).get("template_id") or ""): item
            for item in list(template_recommendations.get(point_id) or [])
            if isinstance(item, Mapping)
        }
        selected_entries.append(
            {
                "base_entry": build_base_fact_locked_entry(raw_item),
                "compact_preview": {
                    "master_timeline": compact_item.get("master_timeline"),
                    "prev_context": compact_item.get("prev_context"),
                    "next_context": compact_item.get("next_context"),
                    "original_prompt_excerpt": compress_text_middle(str(compact_item.get("prompt_text") or "").strip(), 420),
                    "discard_original_wording": True,
                },
                "rewrite_mandate": {
                    "rewrite_from_scratch": True,
                    "preserve_facts_only": True,
                    "minimum_template_features_to_adopt": 5,
                    "discard_original_wording": True,
                },
                "selected_prompt_templates": [
                    {
                        "template_id": str(template.get("template_id") or "").strip(),
                        "purpose": str(template.get("primary_purpose") or template.get("purpose") or "").strip(),
                        "secondary_purposes": list(template.get("secondary_purposes") or []),
                        "retrieval_title": str(template.get("retrieval_title") or "").strip(),
                        "match_score": float(
                            selected_recommendations.get(str(template.get("template_id") or "").strip(), {}).get("model_score")
                            or selected_recommendations.get(str(template.get("template_id") or "").strip(), {}).get("score")
                            or 0.0
                        ),
                        "match_rationale": str(
                            selected_recommendations.get(str(template.get("template_id") or "").strip(), {}).get("rationale") or ""
                        ).strip(),
                        "learning_focus": [
                            str(x).strip()
                            for x in list(
                                selected_recommendations.get(str(template.get("template_id") or "").strip(), {}).get("learning_focus")
                                or []
                            )
                            if str(x).strip()
                        ],
                        "search_hint": str(template.get("search_hint") or "").strip(),
                        "scene_tags": list(template.get("scene_tags") or []),
                        "relation_tags": list(template.get("relation_tags") or []),
                        "staging_tags": list(template.get("staging_tags") or []),
                        "camera_tags": list(template.get("camera_tags") or []),
                        "emotion_tags": list(template.get("emotion_tags") or []),
                        "narrative_tags": list(template.get("narrative_tags") or []),
                        "required_slots": list(template.get("required_slots") or []),
                        "general_template_prompt": compress_text_middle(str(template.get("general_template_prompt") or "").strip(), 2200),
                        "restored_prompt": compress_text_middle(str(template.get("restored_prompt") or "").strip(), 3200),
                        "prompt_library_path": str(template.get("prompt_library_path") or "").strip(),
                    }
                    for template in chosen_templates
                ],
                "frozen_dialogue_signature": extract_point_dialogue_signature(raw_item),
            }
        )
        for template in chosen_templates:
            template_id = str(template.get("template_id") or "").strip()
            if template_id and template_id not in template_pool:
                template_pool[template_id] = {
                    "template_id": template_id,
                    "purpose": str(template.get("primary_purpose") or template.get("purpose") or "").strip(),
                    "secondary_purposes": list(template.get("secondary_purposes") or []),
                    "retrieval_title": str(template.get("retrieval_title") or "").strip(),
                    "match_score": float(
                        selected_recommendations.get(template_id, {}).get("model_score")
                        or selected_recommendations.get(template_id, {}).get("score")
                        or 0.0
                    ),
                    "match_rationale": str(selected_recommendations.get(template_id, {}).get("rationale") or "").strip(),
                    "learning_focus": [
                        str(x).strip()
                        for x in list(selected_recommendations.get(template_id, {}).get("learning_focus") or [])
                        if str(x).strip()
                    ],
                    "search_hint": str(template.get("search_hint") or "").strip(),
                    "scene_tags": list(template.get("scene_tags") or []),
                    "relation_tags": list(template.get("relation_tags") or []),
                    "staging_tags": list(template.get("staging_tags") or []),
                    "camera_tags": list(template.get("camera_tags") or []),
                    "emotion_tags": list(template.get("emotion_tags") or []),
                    "narrative_tags": list(template.get("narrative_tags") or []),
                    "required_slots": list(template.get("required_slots") or []),
                    "general_template_prompt": compress_text_middle(str(template.get("general_template_prompt") or "").strip(), 2200),
                    "restored_prompt": compress_text_middle(str(template.get("restored_prompt") or "").strip(), 3200),
                    "prompt_library_path": str(template.get("prompt_library_path") or "").strip(),
                }
    return {
        "episode_id": str(compact_package.get("episode_id") or "").strip(),
        "episode_title": str(compact_package.get("episode_title") or "").strip(),
        "materials_overview": str(compact_package.get("materials_overview") or "").strip(),
        "global_notes": list(compact_package.get("global_notes") or []),
        "selected_points": selected_entries,
        "template_pool": list(template_pool.values()),
    }


def build_transfer_prompt(
    *,
    job_context: Mapping[str, Any],
    selected_points_package: Mapping[str, Any],
    style_plan: Mapping[str, Any],
    source_script_text: str,
) -> str:
    return render_prompt(
        "seedance_style_transfer/user.md",
        {
            "job_context_json": json.dumps(job_context, ensure_ascii=False, indent=2),
            "selected_points_package_json": json.dumps(selected_points_package, ensure_ascii=False, indent=2),
            "selected_prompt_templates_json": json.dumps(selected_points_package.get("template_pool") or [], ensure_ascii=False, indent=2),
            "style_plan_json": json.dumps(style_plan, ensure_ascii=False, indent=2),
            "source_script_text": source_script_text,
        },
    )


def build_style_transfer_plan_prompt(
    *,
    job_context: Mapping[str, Any],
    selected_points_package: Mapping[str, Any],
    source_script_text: str,
) -> str:
    return render_prompt(
        "seedance_style_transfer_plan/user.md",
        {
            "job_context_json": json.dumps(job_context, ensure_ascii=False, indent=2),
            "selected_points_package_json": json.dumps(selected_points_package, ensure_ascii=False, indent=2),
            "selected_prompt_templates_json": json.dumps(selected_points_package.get("template_pool") or [], ensure_ascii=False, indent=2),
            "source_script_text": source_script_text,
        },
    )


def extract_materials_section(markdown_text: str) -> str:
    text = str(markdown_text or "")
    match = re.search(r"## 素材对应表\s*\n.*?(?=\n---\n)", text, flags=re.DOTALL)
    return match.group(0).strip() if match else ""


def splice_materials_section(rendered_markdown: str, original_markdown: str) -> str:
    original_section = extract_materials_section(original_markdown)
    if not original_section:
        return rendered_markdown
    pattern = re.compile(r"## 素材对应表\s*\n.*?(?=\n---\n)", flags=re.DOTALL)
    if not pattern.search(rendered_markdown):
        return rendered_markdown
    return pattern.sub(original_section, rendered_markdown, count=1)


def backup_file(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.{suffix}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def changed_fields_set(delta_entry: Mapping[str, Any]) -> set[str]:
    return set(normalize_changed_fields(delta_entry.get("changed_fields")))


def prompt_similarity_ratio(before_text: str, after_text: str) -> float:
    return difflib.SequenceMatcher(a=str(before_text or ""), b=str(after_text or "")).ratio()


def timeline_similarity_ratio(before_timeline: Any, after_timeline: Any) -> float:
    before_text = json.dumps(before_timeline or [], ensure_ascii=False, sort_keys=True)
    after_text = json.dumps(after_timeline or [], ensure_ascii=False, sort_keys=True)
    return difflib.SequenceMatcher(a=before_text, b=after_text).ratio()


def identify_weak_changed_points(
    *,
    original_package: Mapping[str, Any],
    refine_delta: Mapping[str, Any],
    selected_point_ids: Sequence[str],
) -> list[str]:
    original_items = point_item_map(original_package)
    delta_items = {
        str(item.get("point_id") or "").strip(): dict(item)
        for item in list(refine_delta.get("changed_points") or [])
        if isinstance(item, Mapping) and str(item.get("point_id") or "").strip()
    }
    weak_ids: list[str] = []
    for point_id in selected_point_ids:
        original_item = original_items.get(point_id, {})
        delta_item = delta_items.get(point_id)
        if not delta_item:
            weak_ids.append(point_id)
            continue
        fields = changed_fields_set(delta_item)
        if "prompt_text" not in fields or "master_timeline" not in fields:
            weak_ids.append(point_id)
            continue
        new_prompt_text = str(delta_item.get("prompt_text") or "").strip()
        before_prompt_text = str(original_item.get("prompt_text") or "").strip()
        before_timeline = list(original_item.get("master_timeline") or [])
        after_timeline = list(delta_item.get("master_timeline") or [])
        if not new_prompt_text:
            weak_ids.append(point_id)
            continue
        prompt_ratio = prompt_similarity_ratio(before_prompt_text, new_prompt_text)
        timeline_ratio = timeline_similarity_ratio(before_timeline, after_timeline)
        if prompt_ratio >= 0.97:
            weak_ids.append(point_id)
            continue
        if prompt_ratio >= MIN_MEANINGFUL_PROMPT_RATIO and timeline_ratio >= MIN_MEANINGFUL_TIMELINE_RATIO:
            weak_ids.append(point_id)
    return weak_ids


def merge_delta_maps(*deltas: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    for delta in deltas:
        for item in list(delta.get("changed_points") or []):
            if not isinstance(item, Mapping):
                continue
            point_id = str(item.get("point_id") or "").strip()
            if point_id:
                merged[point_id] = dict(item)
    return {"changed_points": list(merged.values())}


def render_style_transfer_report(
    *,
    series_name: str,
    episode_id: str,
    model: str,
    original_package: Mapping[str, Any],
    refined_package: Mapping[str, Any],
    changed_points: Sequence[str],
    style_mapping: Mapping[str, Sequence[str]],
    storyboard_json_path: Path,
    search_index_json_path: Path,
    source_script_path: Path | None,
    output_mode: str,
    output_json_path: Path,
    output_md_path: Path,
    warnings: Sequence[str],
    metrics_report: Mapping[str, Any] | None = None,
) -> str:
    original_items = point_item_map(original_package)
    refined_items = point_item_map(refined_package)
    lines = [
        f"# Seedance Style Transfer 报告 -- {series_name} {episode_id}",
        "",
        f"- model：{model}",
        f"- storyboard_json：{storyboard_json_path}",
        f"- search_index_json：{search_index_json_path}",
        f"- source_script：{source_script_path or ''}",
        f"- output_mode：{output_mode}",
        f"- output_json：{output_json_path}",
        f"- output_md：{output_md_path}",
        f"- changed_point_count：{len(changed_points)}",
        "",
    ]
    if metrics_report:
        totals = dict(metrics_report.get("totals", {}) or {})
        lines.extend(
            [
                "## Token 统计",
                "",
                f"- steps：{totals.get('step_count', 0)}",
                f"- duration_seconds：{totals.get('duration_seconds', 0)}",
                f"- input_tokens：{totals.get('input_tokens', 0)}",
                f"- output_tokens：{totals.get('output_tokens', 0)}",
                f"- total_tokens：{totals.get('total_tokens', 0)}",
                "",
            ]
        )
    lines.extend(["## Style 映射", ""])
    for point_id, beat_ids in style_mapping.items():
        lines.append(f"- {point_id} -> {', '.join(beat_ids) if beat_ids else '(无模板)'}")
    if warnings:
        lines.extend(["", "## 保护性回退告警", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    if not changed_points:
        lines.extend(["", "本轮未落盘任何改动。", ""])
        return "\n".join(lines).rstrip() + "\n"
    lines.extend(["", "## 修改摘要", ""])
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
        lines.extend(["", f"## {point_id} {title}", ""])
        prompt_diff = render_text_diff(str(before_item.get("prompt_text") or ""), str(after_item.get("prompt_text") or ""))
        if prompt_diff:
            lines.extend(["```diff"])
            lines.extend(prompt_diff)
            lines.append("```")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def chunked(items: Sequence[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [list(items)]
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def collect_changed_points(delta_batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    changed: dict[str, dict[str, Any]] = {}
    for batch in delta_batches:
        for item in list(batch.get("changed_points") or []):
            if not isinstance(item, Mapping):
                continue
            point_id = str(item.get("point_id") or "").strip()
            if point_id:
                changed[point_id] = dict(item)
    return {"changed_points": list(changed.values())}


def build_assets_dir_from_storyboard(storyboard_json_path: Path) -> Path:
    outputs_series_name = storyboard_json_path.parent.parent.name
    return (PROJECT_ROOT / "assets" / outputs_series_name).resolve()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_runtime_config(args.config)
    model, api_key = configure_openai_api(config)
    template_match_model = str(
        config.get("runtime", {}).get("seedance_style_transfer_template_match_model")
        or config.get("provider", {}).get("template_match_model")
        or model
    ).strip() or model
    if args.temperature >= 0:
        temperature = float(args.temperature)
    else:
        temperature = float(
            config.get("runtime", {}).get("seedance_style_transfer_temperature")
            or config.get("runtime", {}).get("seedance_prompt_refine_temperature")
            or config.get("runtime", {}).get("temperature")
            or 0.45
        )
    timeout_seconds = args.timeout_seconds or int(
        config.get("runtime", {}).get("timeout_seconds")
        or config.get("run", {}).get("timeout_seconds")
        or 600
    )
    provider_tag = build_provider_model_tag("openai", model)
    write_style_transfer_metrics = bool(
        config.get("run", {}).get(
            "write_seedance_style_transfer_metrics",
            config.get("runtime", {}).get("write_seedance_style_transfer_metrics", True),
        )
    )
    auto_retry_weak_changes = bool(
        args.auto_retry_weak_changes
        or config.get("run", {}).get(
            "auto_retry_weak_style_transfer_changes",
            config.get("runtime", {}).get("auto_retry_weak_style_transfer_changes", False),
        )
    )

    if args.non_interactive:
        storyboard_json_path, storyboard_md_path = default_storyboard_paths(args)
    else:
        if args.storyboard_json.strip() or args.storyboard_md.strip():
            storyboard_json_path, storyboard_md_path = default_storyboard_paths(args)
        else:
            storyboard_json_path, storyboard_md_path = choose_storyboard_paths_interactively()

    inferred = infer_context_from_storyboard_path(storyboard_json_path)
    storyboard_package = load_json_file(storyboard_json_path)
    package_episode_id = normalize_episode_id(str(storyboard_package.get("episode_id") or "").strip())
    if package_episode_id:
        inferred["episode_id"] = package_episode_id
    defaults = build_default_paths(storyboard_json_path)
    search_index_input = args.search_index_json.strip() or str(defaults["search_index_json"])
    search_index_json_path = ensure_file(search_index_input, "prompt library search index")
    source_script_path = resolve_source_script_path(
        config,
        series_name=inferred["series_name"],
        episode_id=inferred["episode_id"],
        explicit_path=args.source_script.strip(),
    )

    if args.output_mode:
        output_mode = args.output_mode
    else:
        output_mode = "sidecar" if args.non_interactive else prompt_choice(
            "输出模式", ["sidecar", "overwrite"], "sidecar"
        )

    prompt_templates = load_prompt_search_templates(search_index_json_path)
    if not prompt_templates:
        raise RuntimeError(f"未从 SEARCH_INDEX / prompt_library 解析到任何可用模板：{search_index_json_path}")
    episode_telemetry = TelemetryRecorder(
        run_name="seedance-style-transfer",
        context={
            "series_name": inferred["series_name"],
            "episode_id": inferred["episode_id"],
            "model": model,
            "template_match_model": template_match_model,
        },
    )
    with telemetry_span(
        episode_telemetry,
        stage="seedance_style_transfer",
        name="load_seedance_style_transfer_inputs",
        metadata={
            "episode_id": inferred["episode_id"],
            "storyboard_json_path": str(storyboard_json_path),
            "storyboard_md_path": str(storyboard_md_path),
            "search_index_json_path": str(search_index_json_path),
            "source_script_path": str(source_script_path) if source_script_path else "",
        },
    ) as step:
        step["metadata"]["prompt_template_count"] = len(prompt_templates)
        step["metadata"]["prompt_entry_count"] = len(list(storyboard_package.get("prompt_entries") or []))
    points = [dict(item) for item in list(storyboard_package.get("prompt_entries") or []) if isinstance(item, Mapping)]
    available_point_ids = [str(item.get("point_id") or "").strip() for item in points if str(item.get("point_id") or "").strip()]
    raw_point_ids = args.point_ids.strip()
    if args.non_interactive:
        selected_point_ids = parse_point_selection(raw_point_ids or "all", available_point_ids)
        batch_size = max(1, int(args.batch_size or DEFAULT_BATCH_SIZE))
    else:
        print_status(f"可选 point：{', '.join(available_point_ids)}")
        selected_point_ids = parse_point_selection(
            prompt_input("请输入 point_id 选择（all 或逗号分隔）", raw_point_ids or "all"),
            available_point_ids,
        )
        batch_size = max(1, int(prompt_input("每批处理多少条 point", str(args.batch_size or DEFAULT_BATCH_SIZE))))

    selected_points = [item for item in points if str(item.get("point_id") or "").strip() in set(selected_point_ids)]
    style_mapping, template_recommendations = choose_style_mapping(
        selected_points,
        prompt_templates,
        model=template_match_model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        interactive=not args.non_interactive,
        telemetry=episode_telemetry,
    )

    source_script_text = ""
    if source_script_path and source_script_path.exists():
        source_script_text = compress_text_middle(read_text(source_script_path), MAX_SCRIPT_CHARS_FOR_REFINE)

    delta_batches: list[dict[str, Any]] = []
    selected_batches = chunked(selected_point_ids, batch_size)
    for batch_index, point_batch in enumerate(selected_batches, start=1):
        selected_points_package = build_selected_points_package(
            storyboard_package=storyboard_package,
            selected_point_ids=point_batch,
            style_mapping=style_mapping,
            template_recommendations=template_recommendations,
            prompt_templates=prompt_templates,
        )
        job_context = {
            "series_name": inferred["series_name"],
            "episode_id": inferred["episode_id"],
            "template_match_model": template_match_model,
            "batch_index": batch_index,
            "batch_count": len(selected_batches),
            "selected_point_ids": point_batch,
            "hard_rules": {
                "freeze_refs": True,
                "freeze_dialogue": True,
                "freeze_plot_order": False,
                "freeze_plot_fact_nodes": True,
                "style_is_not_fact": True,
            },
            "rewrite_strength": "template-led-full-rewrite",
            "template_learning_priority": "very_high",
            "performance_layer_policy": "rewrite_aggressively_keep_facts",
        }
        style_plan_prompt = build_style_transfer_plan_prompt(
            job_context=job_context,
            selected_points_package=selected_points_package,
            source_script_text=source_script_text,
        )
        style_plan = openai_json_completion(
            model=model,
            api_key=api_key,
            system_prompt=render_prompt("seedance_style_transfer_plan/system.md", {}),
            user_prompt=style_plan_prompt,
            schema_name="seedance_style_transfer_plan",
            schema=STYLE_TRANSFER_PLAN_SCHEMA,
            temperature=min(temperature, 0.35),
            timeout_seconds=timeout_seconds,
            telemetry=episode_telemetry,
            stage="seedance_style_transfer",
            step_name="seedance_style_transfer_plan_model_call",
            metadata={"episode_id": inferred["episode_id"], "point_batch": point_batch, "stage": "plan"},
        )
        prompt = build_transfer_prompt(
            job_context=job_context,
            selected_points_package=selected_points_package,
            style_plan=style_plan,
            source_script_text=source_script_text,
        )
        print_status(
            f"开始处理 batch {batch_index}/{len(selected_batches)}：{', '.join(point_batch)}"
        )
        delta_batches.append(
            openai_json_completion(
                model=model,
                api_key=api_key,
                system_prompt=render_prompt("seedance_style_transfer/system.md", {}),
                user_prompt=prompt,
                schema_name="seedance_style_transfer_delta",
                schema=SEEDANCE_PROMPT_REFINE_DELTA_SCHEMA,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                telemetry=episode_telemetry,
                stage="seedance_style_transfer",
                step_name="seedance_style_transfer_model_call",
                metadata={"episode_id": inferred["episode_id"], "point_batch": point_batch},
            )
        )

    merged_delta = collect_changed_points(delta_batches)
    weak_point_ids = identify_weak_changed_points(
        original_package=storyboard_package,
        refine_delta=merged_delta,
        selected_point_ids=selected_point_ids,
    )
    if weak_point_ids and auto_retry_weak_changes:
        print_status(f"检测到 {len(weak_point_ids)} 条改动过小或未实质改写，开始强制重写补跑：{', '.join(weak_point_ids)}")
        selected_points_package = build_selected_points_package(
            storyboard_package=storyboard_package,
            selected_point_ids=weak_point_ids,
            style_mapping=style_mapping,
            template_recommendations=template_recommendations,
            prompt_templates=prompt_templates,
        )
        retry_job_context = {
            "series_name": inferred["series_name"],
            "episode_id": inferred["episode_id"],
            "batch_index": "retry",
            "batch_count": len(selected_batches),
            "selected_point_ids": weak_point_ids,
            "hard_rules": {
                "freeze_refs": True,
                "freeze_dialogue": True,
                "freeze_plot_order": False,
                "freeze_plot_fact_nodes": True,
                "style_is_not_fact": True,
            },
            "rewrite_strength": "template-led-maximum-rewrite",
            "template_learning_priority": "maximum",
            "performance_layer_policy": "rewrite_aggressively_keep_facts",
            "extra_instruction": "这些 point 上一轮改动过小。本轮不要再做润色式修订，而要在锁定人物、剧情事实、对白语义、引用映射和剧情结果的前提下，彻底重写表现层。必须明显改写镜头入口、段落骨架、动作链、景别变化、空间调度、受光材质、声音床、尾帧交棒和场景描述方式，让成稿在风格上真正像所选模板。",
        }
        retry_style_plan_prompt = build_style_transfer_plan_prompt(
            job_context=retry_job_context,
            selected_points_package=selected_points_package,
            source_script_text=source_script_text,
        )
        retry_style_plan = openai_json_completion(
            model=model,
            api_key=api_key,
            system_prompt=render_prompt("seedance_style_transfer_plan/system.md", {}),
            user_prompt=retry_style_plan_prompt,
            schema_name="seedance_style_transfer_plan_retry",
            schema=STYLE_TRANSFER_PLAN_SCHEMA,
            temperature=min(max(temperature, 0.3), 0.45),
            timeout_seconds=timeout_seconds,
            telemetry=episode_telemetry,
            stage="seedance_style_transfer",
            step_name="seedance_style_transfer_plan_retry_model_call",
            metadata={"episode_id": inferred["episode_id"], "point_batch": weak_point_ids, "stage": "plan-retry"},
        )
        retry_prompt = build_transfer_prompt(
            job_context=retry_job_context,
            selected_points_package=selected_points_package,
            style_plan=retry_style_plan,
            source_script_text=source_script_text,
        )
        retry_delta = openai_json_completion(
            model=model,
            api_key=api_key,
            system_prompt=render_prompt("seedance_style_transfer/system.md", {}),
            user_prompt=retry_prompt,
            schema_name="seedance_style_transfer_delta_retry",
            schema=SEEDANCE_PROMPT_REFINE_DELTA_SCHEMA,
            temperature=max(temperature, 0.4),
            timeout_seconds=timeout_seconds,
            telemetry=episode_telemetry,
            stage="seedance_style_transfer",
            step_name="seedance_style_transfer_retry_model_call",
            metadata={"episode_id": inferred["episode_id"], "point_batch": weak_point_ids, "retry": True},
        )
        merged_delta = merge_delta_maps(merged_delta, retry_delta)
    elif weak_point_ids:
        print_status(
            f"检测到 {len(weak_point_ids)} 条改动过小或未实质改写：{', '.join(weak_point_ids)}；默认不自动补跑，避免额外消耗 tokens。"
        )
    merged_package = merge_refine_result_with_original(
        original_package=storyboard_package,
        refine_delta=merged_delta,
    )

    assets_dir = build_assets_dir_from_storyboard(storyboard_json_path)
    asset_catalog = build_asset_catalog(
        read_text(assets_dir / "character-prompts.md"),
        read_text(assets_dir / "scene-prompts.md"),
        episode_id=inferred["episode_id"],
        assets_dir=assets_dir,
    )
    storyboard_profile = resolve_storyboard_profile(config)
    profile_settings = storyboard_profile_settings(storyboard_profile)
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
    refined_package, ref_warnings = freeze_ref_integrity(
        original_package=storyboard_package,
        refined_package=refined_package,
        episode_id=inferred["episode_id"],
    )
    refined_package, dialogue_warnings = freeze_dialogue_integrity(
        original_package=storyboard_package,
        refined_package=refined_package,
    )
    quality_warnings = validate_storyboard_density(refined_package, episode_id=inferred["episode_id"])
    quality_warnings += validate_scene_reference_presence(refined_package, asset_catalog, episode_id=inferred["episode_id"])
    if quality_warnings:
        print_status(f"检测到 {len(quality_warnings)} 条质量提示，结果仍会落盘。")

    original_map = point_payload_map(storyboard_package)
    refined_map = point_payload_map(refined_package)
    changed_points = [
        point_id for point_id, payload in refined_map.items() if original_map.get(point_id) != payload
    ]

    base_name = "02-seedance-prompts.style-transfer"
    output_dir = storyboard_json_path.parent
    if output_mode == "overwrite":
        timestamp = utc_timestamp().replace(":", "").replace("-", "")
        backup_file(storyboard_json_path, f"before-style-transfer__{timestamp}")
        backup_file(storyboard_md_path, f"before-style-transfer__{timestamp}")
        output_json_path = storyboard_json_path
        output_md_path = storyboard_md_path
    else:
        output_json_path = output_dir / f"{base_name}__{provider_tag}.json"
        output_md_path = output_dir / f"{base_name}__{provider_tag}.md"

    report_md_path = output_dir / f"{base_name}__{provider_tag}.report.md"
    stamp_json_path = output_dir / f"{base_name}__{provider_tag}.plan.json"
    metrics_json_path = output_dir / f"{base_name}__{provider_tag}.metrics.json"
    metrics_md_path = output_dir / f"{base_name}__{provider_tag}.metrics.md"

    with telemetry_span(
        episode_telemetry,
        stage="seedance_style_transfer",
        name="materialize_seedance_style_transfer_outputs",
        metadata={"episode_id": inferred["episode_id"], "output_mode": output_mode},
    ) as step:
        markdown = render_seedance_markdown(
            series_name=inferred["series_name"],
            data=refined_package,
            asset_catalog=asset_catalog,
        )
        markdown = splice_materials_section(markdown, storyboard_md_path.read_text(encoding="utf-8"))
        save_json_file(output_json_path, refined_package)
        output_md_path.write_text(markdown, encoding="utf-8")
        step["metadata"]["changed_point_count"] = len(changed_points)
        step["metadata"]["warning_count"] = len(ref_warnings) + len(dialogue_warnings) + len(quality_warnings)

    warnings = list(ref_warnings) + list(dialogue_warnings) + list(quality_warnings)
    episode_telemetry.context["changed_point_count"] = len(changed_points)
    episode_telemetry.context["warning_count"] = len(warnings)
    metrics_report = save_metrics(episode_telemetry, metrics_json_path, metrics_md_path) if write_style_transfer_metrics else None
    report_md_path.write_text(
        render_style_transfer_report(
            series_name=inferred["series_name"],
            episode_id=inferred["episode_id"],
            model=model,
            original_package=storyboard_package,
            refined_package=refined_package,
            changed_points=changed_points,
            style_mapping=style_mapping,
            storyboard_json_path=storyboard_json_path,
            search_index_json_path=search_index_json_path,
            source_script_path=source_script_path,
            output_mode=output_mode,
            output_json_path=output_json_path,
            output_md_path=output_md_path,
            warnings=warnings,
            metrics_report=metrics_report,
        ),
        encoding="utf-8",
    )

    stamp = {
        "series_name": inferred["series_name"],
        "episode_id": inferred["episode_id"],
        "model": model,
        "storyboard_json_path": str(storyboard_json_path),
        "storyboard_md_path": str(storyboard_md_path),
        "search_index_json_path": str(search_index_json_path),
        "source_script_path": str(source_script_path) if source_script_path else "",
        "selected_point_ids": selected_point_ids,
        "style_mapping": {key: list(value) for key, value in style_mapping.items()},
        "batch_size": batch_size,
        "auto_retry_weak_changes": auto_retry_weak_changes,
        "weak_point_ids": weak_point_ids,
        "changed_point_ids": changed_points,
        "changed_point_count": len(changed_points),
        "output_mode": output_mode,
        "output_json_path": str(output_json_path),
        "output_md_path": str(output_md_path),
        "report_md_path": str(report_md_path),
        "metrics_json_path": str(metrics_json_path) if write_style_transfer_metrics else "",
        "metrics_md_path": str(metrics_md_path) if write_style_transfer_metrics else "",
        "warnings": warnings,
        "generated_at": utc_timestamp(),
    }
    save_json_file(stamp_json_path, stamp)

    summary = {
        "series_name": inferred["series_name"],
        "episode_id": inferred["episode_id"],
        "model": model,
        "search_index_json_path": str(search_index_json_path),
        "selected_point_ids": selected_point_ids,
        "auto_retry_weak_changes": auto_retry_weak_changes,
        "weak_point_ids": weak_point_ids,
        "changed_point_ids": changed_points,
        "changed_point_count": len(changed_points),
        "output_mode": output_mode,
        "output_json_path": str(output_json_path),
        "output_md_path": str(output_md_path),
        "report_md_path": str(report_md_path),
        "stamp_json_path": str(stamp_json_path),
        "metrics_json_path": str(metrics_json_path) if write_style_transfer_metrics else "",
        "metrics_md_path": str(metrics_md_path) if write_style_transfer_metrics else "",
        "warnings": warnings,
    }
    if metrics_report:
        summary["token_usage"] = dict(metrics_report.get("totals", {}) or {})
        print_status(
            f"{inferred['episode_id']} style transfer 统计：耗时 {metrics_report['totals']['duration_seconds']}s | "
            f"tokens in/out/total = {metrics_report['totals']['input_tokens']}/{metrics_report['totals']['output_tokens']}/{metrics_report['totals']['total_tokens']}"
        )
        print_status(f"{inferred['episode_id']} style transfer 统计报告：{metrics_json_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
