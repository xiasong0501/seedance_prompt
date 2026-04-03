from __future__ import annotations

import copy
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_telemetry import TelemetryRecorder, telemetry_span
from openai_agents.runtime_utils import (
    build_episode_ids,
    configure_openai_api,
    load_runtime_config,
    openai_json_completion,
    read_text,
)
from prompt_utils import (
    build_frame_composition_guidance,
    load_prompt,
    normalize_frame_orientation,
    render_bullets,
    render_prompt,
)
from skill_utils import load_skill
import workflow_context_compaction as ctx_compact
from providers.base import (
    build_provider_model_tag,
    load_json_file,
    save_json_file,
    save_text_file,
    utc_timestamp,
)
from seedance_learning import PURPOSE_ORDER, infer_primary_purpose_from_parts
from series_paths import build_series_paths, infer_episode_id_from_name


DEFAULT_CONFIG_PATH = Path("config/director_analysis_pipeline.local.json")

# 字段截断上限：同步修改时需同步更新 normalize_director_result 和 repair_director_density
MICRO_BEATS_CAP = 8
DETAIL_ANCHOR_LINES_CAP = 6
KEY_DIALOGUE_BEATS_CAP = 6

DIRECTOR_STORY_POINT_REQUIRED = [
    "point_id",
    "title",
    "primary_purpose",
    "characters",
    "scenes",
    "shot_group",
    "pace_label",
    "duration_suggestion",
    "narrative_function",
    "entry_state",
    "micro_beats",
    "detail_anchor_lines",
    "key_dialogue_beats",
    "sound_design_notes",
    "director_statement",
    "exit_state",
    "timeline_adjustment_note",
]

DIRECTOR_STORY_POINT_PROPERTIES: dict[str, Any] = {
    "point_id": {"type": "string"},
    "title": {"type": "string"},
    "primary_purpose": {"type": "string", "enum": PURPOSE_ORDER},
    "characters": {"type": "array", "items": {"type": "string"}},
    "scenes": {"type": "array", "items": {"type": "string"}},
    "shot_group": {"type": "string"},
    "pace_label": {"type": "string", "enum": ["快压推进", "中速推进", "舒缓铺陈"]},
    "duration_suggestion": {"type": "string"},
    "narrative_function": {"type": "string"},
    "entry_state": {"type": "string"},
    "micro_beats": {"type": "array", "items": {"type": "string"}},
    "detail_anchor_lines": {"type": "array", "items": {"type": "string"}},
    "key_dialogue_beats": {"type": "array", "items": {"type": "string"}},
    "sound_design_notes": {"type": "string"},
    "director_statement": {"type": "string"},
    "exit_state": {"type": "string"},
    "timeline_adjustment_note": {"type": "string"},
}

DIRECTOR_STORY_POINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": DIRECTOR_STORY_POINT_REQUIRED,
    "properties": DIRECTOR_STORY_POINT_PROPERTIES,
}

DIRECTOR_CHARACTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "age", "appearance_keywords", "asset_status", "reuse_note"],
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "string"},
        "appearance_keywords": {"type": "string"},
        "asset_status": {"type": "string", "enum": ["新增", "复用", "变体"]},
        "reuse_note": {"type": "string"},
    },
}

DIRECTOR_SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "time_of_day", "lighting_palette", "mood_keywords", "asset_status", "reuse_note"],
    "properties": {
        "name": {"type": "string"},
        "time_of_day": {"type": "string"},
        "lighting_palette": {"type": "string"},
        "mood_keywords": {"type": "string"},
        "asset_status": {"type": "string", "enum": ["新增", "复用", "变体"]},
        "reuse_note": {"type": "string"},
    },
}


def nullable_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "anyOf": [
            copy.deepcopy(dict(schema)),
            {"type": "null"},
        ]
    }

DIRECTOR_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "episode_id",
        "episode_title",
        "visual_style",
        "target_medium",
        "structure_overview",
        "emotional_curve",
        "story_points",
        "characters",
        "scenes",
        "director_notes",
    ],
    "properties": {
        "episode_id": {"type": "string"},
        "episode_title": {"type": "string"},
        "visual_style": {"type": "string"},
        "target_medium": {"type": "string"},
        "structure_overview": {"type": "string"},
        "emotional_curve": {"type": "string"},
        "story_points": {
            "type": "array",
            "items": DIRECTOR_STORY_POINT_SCHEMA,
        },
        "characters": {
            "type": "array",
            "items": DIRECTOR_CHARACTER_SCHEMA,
        },
        "scenes": {
            "type": "array",
            "items": DIRECTOR_SCENE_SCHEMA,
        },
        "director_notes": {"type": "array", "items": {"type": "string"}},
    },
}

DIRECTOR_REVIEW_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "structure_overview",
        "emotional_curve",
        "characters",
        "scenes",
        "director_notes",
        "story_point_patches",
        "delete_story_point_ids",
        "story_point_insertions",
    ],
    "properties": {
        "structure_overview": nullable_schema({"type": "string"}),
        "emotional_curve": nullable_schema({"type": "string"}),
        "characters": nullable_schema({"type": "array", "items": DIRECTOR_CHARACTER_SCHEMA}),
        "scenes": nullable_schema({"type": "array", "items": DIRECTOR_SCENE_SCHEMA}),
        "director_notes": nullable_schema({"type": "array", "items": {"type": "string"}}),
        "story_point_patches": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": list(DIRECTOR_STORY_POINT_PROPERTIES.keys()),
                "properties": {
                    "point_id": {"type": "string"},
                    **{
                        key: nullable_schema(value)
                        for key, value in DIRECTOR_STORY_POINT_PROPERTIES.items()
                        if key != "point_id"
                    },
                },
            },
        },
        "delete_story_point_ids": {"type": "array", "items": {"type": "string"}},
        "story_point_insertions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["insert_after_point_id", "story_points"],
                "properties": {
                    "insert_after_point_id": nullable_schema({"type": "string"}),
                    "story_points": {"type": "array", "items": DIRECTOR_STORY_POINT_SCHEMA},
                },
            },
        },
    },
}

SCRIPT_FILENAME_PATTERN = re.compile(r"^(?P<episode>ep\d+).+?\.md$", re.IGNORECASE)
SCENE_MARKER_PATTERN = re.compile(r"^(?:第.{0,4}场|场景[:：]|【.+?】|INT\\.|EXT\\.|SCENE\\b)", re.IGNORECASE)
# 覆盖三种常见对白格式：
#   1. 人名：对白（原始格式）
#   2. 人名+说/道/问/答等动词后接对白（古风/现代叙事体）
#   3. OS/旁白 行
# \u300c=「 \u201c=left-quote \u2018=left-single-quote \u300e=【
_SPEAKER_PATTERN = (
    r"^[^\s:：]{1,20}[：:].+"
    + r"|^.{1,12}(?:说道?|答道?|问道?|低声道?|笑道?|冷声道?|轻声道?)[：:\u300c\u201c\u2018\u300e]"
    + r"|^(?:OS|旁白|画外音)[：:]"
)
SPEAKER_LINE_PATTERN = re.compile(_SPEAKER_PATTERN)


def print_status(message: str) -> None:
    print(f"[director-analysis] {message}", flush=True)


def resolve_series_name(config: Mapping[str, Any]) -> str:
    explicit = str(config.get("series", {}).get("series_name") or "").strip()
    if explicit:
        return explicit
    script_path = str(config.get("script", {}).get("script_path") or "").strip()
    series_dir = str(config.get("script", {}).get("series_dir") or "").strip()
    if script_path:
        return build_series_paths(project_root=PROJECT_ROOT, script_path=script_path).series_name
    if series_dir:
        return Path(series_dir).expanduser().resolve().name
    raise RuntimeError("无法推导剧名，请提供 series.series_name 或 script.series_dir。")


def explicit_script_map(config: Mapping[str, Any]) -> dict[str, Path]:
    script_path = str(config.get("script", {}).get("script_path") or "").strip()
    if not script_path:
        return {}
    resolved = Path(script_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"剧本文件不存在：{resolved}")
    episode_id = str(config.get("script", {}).get("episode_id") or "").strip().lower()
    if not episode_id:
        episode_id = infer_episode_id_from_name(resolved.name) or ""
    if not episode_id:
        raise RuntimeError(f"无法从剧本文件名推导集数：{resolved.name}")
    return {episode_id: resolved}


def collect_scripts_by_episode(series_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(series_dir.glob("*.md")):
        episode_id = infer_episode_id_from_name(path.name)
        if not episode_id:
            match = SCRIPT_FILENAME_PATTERN.match(path.name)
            episode_id = match.group("episode").lower() if match else None
        if not episode_id:
            continue
        grouped.setdefault(episode_id, []).append(path.resolve())
    return grouped


def choose_script_path(config: Mapping[str, Any], episode_id: str) -> Path:
    explicit = explicit_script_map(config)
    if explicit:
        if episode_id not in explicit:
            raise RuntimeError(f"显式 script_path 只对应 {', '.join(sorted(explicit))}，未覆盖 {episode_id}")
        return explicit[episode_id]

    series_dir_value = str(config.get("script", {}).get("series_dir") or "").strip()
    if not series_dir_value:
        raise RuntimeError("缺少 script.series_dir。")
    series_dir = Path(series_dir_value).expanduser().resolve()
    if not series_dir.exists():
        raise FileNotFoundError(f"剧本目录不存在：{series_dir}")
    grouped = collect_scripts_by_episode(series_dir)
    candidates = grouped.get(episode_id, [])
    if not candidates:
        raise FileNotFoundError(f"未找到 {episode_id} 对应剧本：{series_dir}")

    preferred_suffixes = list(config.get("script", {}).get("preferred_filename_suffixes", []))
    for suffix in preferred_suffixes:
        normalized_suffix = str(suffix).strip()
        if not normalized_suffix:
            continue
        for candidate in candidates:
            if candidate.name.endswith(normalized_suffix):
                return candidate
    return candidates[-1]


def resolve_assets_dir(config: Mapping[str, Any], series_name: str) -> Path:
    output_config = config.get("output", {})
    explicit_assets_series_name = str(output_config.get("assets_series_name") or "").strip()
    assets_series_suffix = str(output_config.get("assets_series_suffix") or "-gpt").strip()
    target_series_name = explicit_assets_series_name or f"{series_name}{assets_series_suffix}"
    return (PROJECT_ROOT / "assets" / target_series_name).resolve()


def resolve_outputs_root(config: Mapping[str, Any]) -> Path:
    outputs_root = Path(config.get("output", {}).get("outputs_root") or "outputs").expanduser()
    if not outputs_root.is_absolute():
        outputs_root = (PROJECT_ROOT / outputs_root).resolve()
    return outputs_root


def resolve_output_series_name(config: Mapping[str, Any], series_name: str) -> str:
    output_config = config.get("output", {})
    explicit = str(output_config.get("outputs_series_name") or "").strip()
    suffix = str(output_config.get("outputs_series_suffix") or "-gpt").strip()
    return explicit or f"{series_name}{suffix}"


def resolve_episode_output_dir(config: Mapping[str, Any], series_name: str, episode_id: str) -> Path:
    return resolve_outputs_root(config) / resolve_output_series_name(config, series_name) / episode_id


def load_genre_reference_bundle(config: Mapping[str, Any], series_name: str) -> dict[str, Any]:
    explicit = str(config.get("sources", {}).get("genre_reference_bundle_path") or "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.exists():
            return load_json_file(path)
    candidate = PROJECT_ROOT / "analysis" / series_name / "openai_agent_flow" / "genre_reference_bundle.json"
    if candidate.exists():
        return load_json_file(candidate)
    return {}


def resolve_analysis_root(config: Mapping[str, Any]) -> Path:
    analysis_root = (
        str(config.get("run", {}).get("analysis_root") or "").strip()
        or str(config.get("output", {}).get("analysis_root") or "").strip()
        or "analysis"
    )
    path = Path(analysis_root).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def format_asset_status(item: Mapping[str, Any]) -> str:
    status = str(item.get("asset_status") or "").strip()
    note = str(item.get("reuse_note") or "").strip()
    if note:
        return f"{status} {note}".strip()
    return status


def shorten_text(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def merge_transition_state(*parts: Any) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = str(part or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        values.append(text)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return "；".join(values)


def parse_duration_window(raw: str) -> tuple[float, float]:
    values = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(raw or ""))]
    if not values:
        return (8.0, 10.0)
    if len(values) == 1:
        return (values[0], values[0])
    return (values[0], values[1])


def normalize_duration_suggestion(raw: str) -> str:
    start, end = parse_duration_window(raw)
    if start > end:
        start, end = end, start
    start = max(6.0, min(start, 11.0))
    end = max(start, min(12.5, end))
    if end < 7.0:
        start, end = 8.0, 10.0
    if start > 12.5:
        start, end = 9.0, 11.0
    if end - start < 1.0:
        end = min(12.5, start + 1.8)
        start = max(6.0, min(start, end - 1.0))
    if abs(start - round(start)) < 1e-6 and abs(end - round(end)) < 1e-6:
        return f"{int(round(start))}-{int(round(end))}秒"
    return f"{start:.1f}-{end:.1f}秒"


def normalize_pace_label(raw: str) -> str:
    value = str(raw or "").strip()
    if value in {"快压推进", "中速推进", "舒缓铺陈"}:
        return value
    if any(token in value for token in ("快", "急", "压", "高压")):
        return "快压推进"
    if any(token in value for token in ("慢", "缓", "铺陈", "静")):
        return "舒缓铺陈"
    return "中速推进"


def normalize_primary_purpose(raw: Any, *, fallback_parts: Sequence[Any] | None = None) -> str:
    value = str(raw or "").strip()
    if value in PURPOSE_ORDER:
        return value
    return infer_primary_purpose_from_parts(fallback_parts or [], fallback="对峙")


def extract_script_density_metrics(script_text: str) -> dict[str, int]:
    lines = [line.strip() for line in str(script_text or "").splitlines() if line.strip()]
    scene_markers = sum(1 for line in lines if SCENE_MARKER_PATTERN.match(line))
    dialogue_turns = sum(1 for line in lines if SPEAKER_LINE_PATTERN.match(line))
    quoted_dialogues = sum(len(re.findall('[\u201c\u201d\u300c\u300d\u2018\u2019\u300e\u300f].+?[\u201c\u201d\u300c\u300d\u2018\u2019\u300e\u300f]', line)) for line in lines)
    action_lines = max(0, len(lines) - dialogue_turns - scene_markers)
    # 字符数补权重：长对白/长动作段比短行信息量更大，用字符数微调权重
    # 每 40 字对白 ≈ 1 单元，每 60 字动作 ≈ 1 单元（权重低于行数以避免过度膨胀）
    dialogue_chars = sum(len(line) for line in lines if SPEAKER_LINE_PATTERN.match(line))
    action_chars = sum(
        len(line) for line in lines
        if not SPEAKER_LINE_PATTERN.match(line) and not SCENE_MARKER_PATTERN.match(line)
    )
    weighted_units = (
        dialogue_turns * 12
        + quoted_dialogues * 6
        + action_lines * 7
        + scene_markers * 10
        + dialogue_chars // 40
        + action_chars // 60
    )
    # 相比旧版进一步收紧拆点密度：目标把同一戏剧目的下的弱过场与准备/兑现链合并，
    # 常规集数的剧情点数量较旧版下降约 35%-45%，避免出现 20+ 个碎点。
    recommended_story_points = max(5, min(16, math.ceil(weighted_units / 70)))
    return {
        "nonempty_lines": len(lines),
        "scene_markers": scene_markers,
        "dialogue_turns": dialogue_turns,
        "quoted_dialogues": quoted_dialogues,
        "action_lines": action_lines,
        "dialogue_chars": dialogue_chars,
        "action_chars": action_chars,
        "recommended_story_points": recommended_story_points,
    }


def build_script_density_guidance(metrics: Mapping[str, int]) -> str:
    return (
        f"剧本密度指标：非空行 {metrics.get('nonempty_lines', 0)}，场景线索 {metrics.get('scene_markers', 0)}，"
        f"对白轮次 {metrics.get('dialogue_turns', 0)}（约 {metrics.get('dialogue_chars', 0)} 字），"
        f"引号对白 {metrics.get('quoted_dialogues', 0)}，"
        f"动作/叙述段 {metrics.get('action_lines', 0)}（约 {metrics.get('action_chars', 0)} 字）。"
        f"建议拆成约 {metrics.get('recommended_story_points', 5)} 个高效剧情点；"
        "整体节奏相较旧版导演流程需明显提快，目标约快 35%-45%，但不能牺牲因果连续、人物状态连续和切点自然。"
        "每个剧情点默认以 8-11 秒为常用区间，高价值爆点/情绪兑现可放宽到 9-12 秒；纯过场、重复反应和说明性段落应主动压到 6-8 秒并并入前后强点。"
        "静态说明、重复反应和弱过场要快速压缩，把篇幅让给动作推进、关系碰撞、镜头调度、奇观兑现与大场面逻辑。"
        "同一戏剧目的下的准备动作、兑现反应和短过桥优先并入同一点，不要拆成大量子点；燃点、羞辱反击、揭示翻盘、权力转向和情绪兑现优先保留并放大。"
        "过场、重复反应、解释性铺垫和不改变局势的细碎动作要尽量压进相邻燃点，但不能压掉必要的物理桥、视线桥、入画桥和空间承接。"
    )


def resolve_scene_budget(config: Mapping[str, Any], metrics: Mapping[str, int] | None = None) -> dict[str, int]:
    budget_config = config.get("quality", {}).get("director_scene_budget", {})
    recommended_story_points = int((metrics or {}).get("recommended_story_points", 0) or 0)
    scene_markers = int((metrics or {}).get("scene_markers", 0) or 0)
    default_min = 4
    default_max = 6
    if 0 < recommended_story_points <= 6:
        default_min = 3
        default_max = 5

    baseline_min = int(
        budget_config.get("baseline_preferred_min", budget_config.get("preferred_min", default_min)) or default_min
    )
    baseline_max = int(
        budget_config.get("baseline_preferred_max", budget_config.get("preferred_max", default_max)) or default_max
    )
    max_adaptive_expansion = int(budget_config.get("max_adaptive_expansion", 2) or 2)

    adaptive_expansion = 0
    if recommended_story_points >= 11:
        adaptive_expansion += 1
    if recommended_story_points >= 15:
        adaptive_expansion += 1
    if scene_markers >= 8:
        adaptive_expansion = max(adaptive_expansion, 1)
    if scene_markers >= 12:
        adaptive_expansion = max(adaptive_expansion, 2)
    adaptive_expansion = max(0, min(adaptive_expansion, max_adaptive_expansion))

    preferred_min = max(1, baseline_min)
    preferred_max = max(preferred_min, baseline_max + adaptive_expansion)
    return {
        "baseline_preferred_min": max(1, baseline_min),
        "baseline_preferred_max": max(max(1, baseline_min), baseline_max),
        "preferred_min": preferred_min,
        "preferred_max": preferred_max,
        "adaptive_expansion": adaptive_expansion,
    }


def build_scene_budget_guidance(config: Mapping[str, Any], metrics: Mapping[str, int]) -> str:
    budget = resolve_scene_budget(config, metrics)
    baseline_min = budget["baseline_preferred_min"]
    baseline_max = budget["baseline_preferred_max"]
    preferred_min = budget["preferred_min"]
    preferred_max = budget["preferred_max"]
    adaptive_expansion = budget["adaptive_expansion"]
    adaptive_line = ""
    if adaptive_expansion > 0:
        adaptive_line = (
            f"当前剧本密度和场景线索较高，建议区间可自适应放宽到 {preferred_min}-{preferred_max} 个，"
            "但前提必须是这些场景确实属于不可互相复用的大空间或大体系切换。"
        )
    return (
        f"场景母体策略：默认建议控制在 {baseline_min}-{baseline_max} 个代表性母体场景。"
        f"{adaptive_line}"
        "数量不是硬卡出来的，必须根据剧本真实空间结构决定；总原则是优先少而准、优先母体复用，但不能少到让后续分镜失去稳定空间锚点。"
        "同一地点的门外、门内、走廊、前院、后院、高台、边缘、上空、废墟态、对峙区、入口区和机位差异，"
        "原则上优先并入同一个代表性母体场景，并把分区、战损、秩序变化和镜头锚点写进 reuse_note；"
        "但若某个分区会被后续剧情反复调用、承担独立 establishing/桥接/高价值状态切换功能，则允许把它单独保留为稳定子场景。"
        "剧情点可以比场景更多；优先通过剧情点拆分动作和关系推进，而不是为每个剧情点单独发明一个新场景。"
        f"如果最终总场景数低于建议下限 {preferred_min} 个，要主动检查是否漏了真正有叙事职能的辅助/子场景，"
        "例如外围火力源、高台裁判位、龙辇入场长阶轴线、观战外圈/四门环廊、镜门门位等。"
        "最终保留下来的每个场景都必须一眼能区分其世界位置、权力结构或叙事功能，避免多个只在局部机位上有区别的近义场景并存。"
    )


def normalize_director_result(
    data: Mapping[str, Any],
    *,
    scene_budget: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    normalized = dict(data)
    truncation_events: list[dict[str, Any]] = []
    story_points: list[dict[str, Any]] = []
    for raw_item in list(data.get("story_points") or []):
        item = dict(raw_item)
        item["pace_label"] = normalize_pace_label(item.get("pace_label"))
        item["duration_suggestion"] = normalize_duration_suggestion(str(item.get("duration_suggestion") or "8-11秒"))
        item["entry_state"] = merge_transition_state(
            item.get("entry_state"),
            item.get("opening_visual_state"),
            item.get("continuity_hook_in"),
        )
        # 截断前记录超限事件，保留质量信号供后续分析
        raw_micro = [str(x).strip() for x in list(item.get("micro_beats") or []) if str(x).strip()]
        raw_anchors = [str(x).strip() for x in list(item.get("detail_anchor_lines") or []) if str(x).strip()]
        raw_dialogue = [str(x).strip() for x in list(item.get("key_dialogue_beats") or []) if str(x).strip()]
        for field, raw_list, cap in (("micro_beats", raw_micro, MICRO_BEATS_CAP), ("detail_anchor_lines", raw_anchors, DETAIL_ANCHOR_LINES_CAP), ("key_dialogue_beats", raw_dialogue, KEY_DIALOGUE_BEATS_CAP)):
            if len(raw_list) > cap:
                truncation_events.append({"point_id": item.get("point_id"), "field": field, "original_count": len(raw_list), "truncated_to": cap})
        item["micro_beats"] = raw_micro[:MICRO_BEATS_CAP]
        item["detail_anchor_lines"] = raw_anchors[:DETAIL_ANCHOR_LINES_CAP]
        item["key_dialogue_beats"] = raw_dialogue[:KEY_DIALOGUE_BEATS_CAP]
        item["sound_design_notes"] = str(item.get("sound_design_notes") or "").strip()
        item["director_statement"] = str(item.get("director_statement") or "").strip()
        item["exit_state"] = merge_transition_state(
            item.get("exit_state"),
            item.get("closing_visual_state"),
            item.get("continuity_hook_out"),
        )
        item["timeline_adjustment_note"] = str(item.get("timeline_adjustment_note") or "").strip()
        item["primary_purpose"] = normalize_primary_purpose(
            item.get("primary_purpose"),
            fallback_parts=[
                item.get("title"),
                item.get("narrative_function"),
                item.get("entry_state"),
                *raw_micro,
                *raw_anchors,
                *raw_dialogue,
                item.get("director_statement"),
                item.get("exit_state"),
            ],
        )
        for legacy_field in ("continuity_hook_in", "opening_visual_state", "closing_visual_state", "continuity_hook_out"):
            item.pop(legacy_field, None)
        story_points.append(item)
    normalized["story_points"] = story_points
    if truncation_events:
        normalized["_truncation_events"] = truncation_events
        print_status(f"[质量信号] 发现 {len(truncation_events)} 次字段截断，说明模型原始输出超出上限——这是正常高信息密度的信号，已记录到 _truncation_events。")
    filtered_characters: list[dict[str, Any]] = []
    filtered_mixed_crowd_names: list[str] = []
    for raw_item in list(data.get("characters") or []):
        if not isinstance(raw_item, Mapping):
            continue
        name = str(raw_item.get("name") or "").strip()
        if ctx_compact.is_mixed_crowd_character_asset(
            name,
            appearance_keywords=str(raw_item.get("appearance_keywords") or ""),
            reuse_note=str(raw_item.get("reuse_note") or ""),
        ):
            if name and name not in filtered_mixed_crowd_names:
                filtered_mixed_crowd_names.append(name)
            continue
        filtered_characters.append(dict(raw_item))
    normalized["characters"] = filtered_characters
    if filtered_mixed_crowd_names:
        normalized["_filtered_mixed_crowd_characters"] = filtered_mixed_crowd_names
    normalized["scenes"] = normalize_scene_catalog(
        list(data.get("scenes") or []),
        target_max=int((scene_budget or {}).get("preferred_max", 0) or 0),
    )
    normalized = ensure_scene_catalog_covers_story_points(
        normalized,
        target_max=int((scene_budget or {}).get("preferred_max", 0) or 0),
    )
    normalized["director_notes"] = [str(x).strip() for x in list(data.get("director_notes") or []) if str(x).strip()][:12]
    return normalized


def _normalize_scene_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip())


def _scene_parent_name(name: str) -> str:
    normalized = _normalize_scene_name(name)
    for delimiter in ("·", "｜", "|", "：", ":"):
        if delimiter in normalized:
            head = normalized.split(delimiter, 1)[0].strip()
            if len(head) >= 3:
                return head
    return normalized


def _pick_scene_asset_status(items: list[Mapping[str, Any]]) -> str:
    statuses = [str(item.get("asset_status") or "").strip() for item in items]
    if "新增" in statuses:
        return "新增"
    if "变体" in statuses:
        return "变体"
    return "复用"


def _merge_scene_text_values(values: list[str], *, sep: str = "；", limit: int = 72) -> str:
    merged = _unique_ordered_text(values)
    text = sep.join(item for item in merged if item)
    return shorten_text(text, limit) if text else ""


SCENE_MACRO_SUFFIXES = (
    "飞舟舰队",
    "舰队",
    "王府",
    "府",
    "宫门",
    "皇宫",
    "宫",
    "寝殿",
    "殿",
    "城门",
    "城",
    "死牢",
    "天牢",
    "牢",
    "狱",
    "主台",
    "外场",
    "高台",
    "祭台",
    "台",
    "大门",
    "府门",
    "山头",
    "高地",
    "山门",
    "山",
    "谷",
    "河",
    "湖",
    "海",
    "林",
    "庭院",
    "前院",
    "后院",
    "院",
    "回廊",
    "外廊",
    "长廊",
    "廊",
    "居室",
    "寝宫",
    "阁楼",
    "楼阁",
    "高阁",
    "楼",
    "阁",
    "街巷",
    "街区",
    "集结区",
    "广场",
    "天空",
    "天际",
)
GENERIC_SCENE_ANCHORS = {"天空", "天际", "广场", "集结区"}
PRESERVABLE_SCENE_VARIANT_KEYWORDS = (
    "高台",
    "裁判位",
    "镜门",
    "门位",
    "长阶",
    "石阶",
    "中轴",
    "入场",
    "轴线",
    "龙辇",
    "观战",
    "观礼",
    "外圈",
    "外围",
    "环廊",
    "四门",
    "出口",
    "阵地",
    "防线",
    "山头",
    "高地",
    "炮位",
    "比武区",
)
AUXILIARY_SCENE_SIGNAL_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("高台", "裁判"), "高台裁判位"),
    (("龙辇", "石阶", "入场", "中轴"), "龙辇入场长阶轴线"),
    (("观战", "观礼", "外圈", "四门", "环廊"), "观战外圈/四门环廊"),
    (("镜门", "传送镜"), "镜门门位"),
    (("山头", "防线", "炮位", "坦克", "无人机"), "外围山头防线"),
)


def _extract_macro_scene_anchor(name: str) -> str:
    normalized = re.sub(r"[【】（）()]", "", _scene_parent_name(name))
    for suffix in SCENE_MACRO_SUFFIXES:
        index = normalized.find(suffix)
        if index <= 0:
            continue
        anchor = normalized[: index + len(suffix)].strip()
        if len(anchor) >= 2 and anchor not in GENERIC_SCENE_ANCHORS:
            return anchor
    return normalized.strip()


def _scene_child_suffix(name: str, parent: str) -> str:
    normalized_name = _normalize_scene_name(name)
    normalized_parent = _normalize_scene_name(parent)
    if not normalized_name or not normalized_parent or normalized_name == normalized_parent:
        return ""
    if normalized_name.startswith(normalized_parent):
        suffix = normalized_name[len(normalized_parent):].strip("·｜|：: ")
        return suffix
    return normalized_name


def _should_preserve_scene_variant(item: Mapping[str, Any], parent: str) -> bool:
    name = _normalize_scene_name(str(item.get("name") or ""))
    suffix = _scene_child_suffix(name, parent)
    if not suffix:
        return False
    if any(keyword in suffix for keyword in PRESERVABLE_SCENE_VARIANT_KEYWORDS):
        return True
    reuse_note = str(item.get("reuse_note") or "")
    return any(token in reuse_note for token in ("辅助", "establish", "状态变体", "空间桥接", "重复引用", "稳定锚点"))


def _merge_scene_group(scene_name: str, items: list[Mapping[str, Any]]) -> dict[str, Any]:
    time_of_day = _merge_scene_text_values([str(item.get("time_of_day") or "") for item in items], sep=" / ", limit=24)
    lighting_palette = _merge_scene_text_values(
        [str(item.get("lighting_palette") or "") for item in items],
        limit=96,
    )
    mood_keywords = _merge_scene_text_values(
        [str(item.get("mood_keywords") or "") for item in items],
        sep="、",
        limit=72,
    )
    child_names = [str(item.get("name") or "").strip() for item in items if str(item.get("name") or "").strip() != scene_name]
    reuse_notes = [str(item.get("reuse_note") or "").strip() for item in items if str(item.get("reuse_note") or "").strip()]
    merged_note_parts: list[str] = []
    if child_names:
        merged_note_parts.append(f"母体场景覆盖：{'、'.join(_unique_ordered_text(child_names))}")
    if reuse_notes:
        merged_note_parts.append(_merge_scene_text_values(reuse_notes, limit=120))
    return {
        "name": scene_name,
        "time_of_day": time_of_day or (str(items[0].get("time_of_day") or "").strip() if items else ""),
        "lighting_palette": lighting_palette or (str(items[0].get("lighting_palette") or "").strip() if items else ""),
        "mood_keywords": mood_keywords or (str(items[0].get("mood_keywords") or "").strip() if items else ""),
        "asset_status": _pick_scene_asset_status(list(items)),
        "reuse_note": _merge_scene_text_values(merged_note_parts, limit=160),
    }


def _collapse_scene_catalog_by_macro_anchor(
    scenes: list[dict[str, Any]],
    *,
    target_max: int,
) -> list[dict[str, Any]]:
    if target_max <= 0 or len(scenes) <= target_max:
        return scenes

    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for item in scenes:
        anchor = _extract_macro_scene_anchor(str(item.get("name") or ""))
        if anchor not in grouped:
            grouped[anchor] = []
            order.append(anchor)
        grouped[anchor].append(item)

    collapsed: list[dict[str, Any]] = []
    for anchor in order:
        items = grouped[anchor]
        if len(items) == 1:
            collapsed.append(items[0])
            continue
        collapsed.append(_merge_scene_group(anchor, items))

    if len(collapsed) < len(scenes):
        return collapsed
    return scenes


def normalize_scene_catalog(raw_scenes: list[Mapping[str, Any]], *, target_max: int = 0) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for raw_item in raw_scenes:
        item = {
            "name": _normalize_scene_name(str(raw_item.get("name") or "")),
            "time_of_day": str(raw_item.get("time_of_day") or "").strip(),
            "lighting_palette": str(raw_item.get("lighting_palette") or "").strip(),
            "mood_keywords": str(raw_item.get("mood_keywords") or "").strip(),
            "asset_status": str(raw_item.get("asset_status") or "").strip() or "新增",
            "reuse_note": str(raw_item.get("reuse_note") or "").strip(),
        }
        if item["name"]:
            prepared.append(item)

    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for item in prepared:
        parent = _scene_parent_name(item["name"])
        group_key = item["name"] if _should_preserve_scene_variant(item, parent) else parent
        if group_key not in grouped:
            grouped[group_key] = []
            order.append(group_key)
        grouped[group_key].append(item)

    normalized: list[dict[str, Any]] = []
    for parent in order:
        items = grouped[parent]
        if len(items) == 1 and items[0]["name"] == parent:
            normalized.append(items[0])
            continue
        merged_item = _merge_scene_group(parent, items)
        child_names = [item["name"] for item in items if item["name"] != parent]
        child_suffixes: list[str] = []
        for child_name in child_names:
            suffix = child_name
            if child_name.startswith(parent):
                suffix = child_name[len(parent):].strip("·｜|：: ")
            if suffix:
                child_suffixes.append(suffix)
        if child_suffixes:
            merged_item["reuse_note"] = _merge_scene_text_values(
                [f"母体场景覆盖：{'、'.join(_unique_ordered_text(child_suffixes))}", merged_item["reuse_note"]],
                limit=160,
            )
        normalized.append(merged_item)
    return _collapse_scene_catalog_by_macro_anchor(normalized, target_max=target_max)


def infer_missing_scene_coverage_hints(data: Mapping[str, Any]) -> list[str]:
    scene_name_text = " ".join(str(item.get("name") or "") for item in list(data.get("scenes") or []))
    point_texts: list[str] = []
    for point in list(data.get("story_points") or []):
        point_texts.append(
            " ".join(
                [
                    str(point.get("title") or ""),
                    " ".join(str(item or "") for item in list(point.get("scenes") or [])),
                    str(point.get("entry_state") or ""),
                    str(point.get("exit_state") or ""),
                    str(point.get("director_statement") or ""),
                ]
            )
        )
    hints: list[str] = []
    for tokens, label in AUXILIARY_SCENE_SIGNAL_HINTS:
        if any(token in scene_name_text for token in tokens):
            continue
        token_hits = sum(sum(text.count(token) for token in tokens) for text in point_texts)
        if token_hits >= 2:
            hints.append(label)
    return _unique_ordered_text(hints)


def _unique_ordered_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def scene_names_used_in_story_points(data: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for point in list(data.get("story_points") or []):
        for raw_name in list(point.get("scenes") or []):
            name = _normalize_scene_name(str(raw_name or ""))
            if name and name not in names:
                names.append(name)
    return names


def _infer_scene_catalog_entry_from_story_points(scene_name: str, story_points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    supporting_points = [
        point for point in story_points
        if scene_name in [ _normalize_scene_name(str(item or "")) for item in list(point.get("scenes") or []) ]
    ]
    joined_text = " ".join(
        " ".join(
            [
                str(point.get("title") or ""),
                str(point.get("entry_state") or ""),
                str(point.get("exit_state") or ""),
                str(point.get("director_statement") or ""),
            ]
        )
        for point in supporting_points
    )
    time_of_day = "幻境时空" if any(token in scene_name for token in ("秘境", "幻境")) else "白日"
    lighting_palette = "暖雾、逆光、水光虚化" if "秘境" in scene_name else "冷白日光、石阶高差、硬边缘光"
    mood_keywords = "命运牵引、危险吸引、空间失衡" if "秘境" in scene_name else "审判、压迫、围观、等级秩序"
    reuse_note_parts = ["由剧情点场景引用自动补齐到总场景表。"]
    if "长阶" in scene_name or "主位" in scene_name or "对视区" in scene_name or "近水岩石" in scene_name:
        reuse_note_parts.append("稳定子场景；在剧情点中承担明确空间桥接或状态切换功能。")
    if any(token in joined_text for token in ("桥接", "承接", "切到", "切向", "入场", "离场", "高差")):
        reuse_note_parts.append("可复用空间锚点；后续分镜需保持命名连续。")
    return {
        "name": scene_name,
        "time_of_day": time_of_day,
        "lighting_palette": lighting_palette,
        "mood_keywords": mood_keywords,
        "asset_status": "新增",
        "reuse_note": shorten_text(" ".join(reuse_note_parts), 160),
    }


def ensure_scene_catalog_covers_story_points(
    data: Mapping[str, Any],
    *,
    target_max: int = 0,
) -> dict[str, Any]:
    normalized = dict(data)
    story_points = [dict(item) for item in list(data.get("story_points") or [])]
    catalog = [
        {
            "name": _normalize_scene_name(str(item.get("name") or "")),
            "time_of_day": str(item.get("time_of_day") or "").strip(),
            "lighting_palette": str(item.get("lighting_palette") or "").strip(),
            "mood_keywords": str(item.get("mood_keywords") or "").strip(),
            "asset_status": str(item.get("asset_status") or "").strip() or "新增",
            "reuse_note": str(item.get("reuse_note") or "").strip(),
        }
        for item in list(data.get("scenes") or [])
        if _normalize_scene_name(str(item.get("name") or ""))
    ]
    exact_names = {_normalize_scene_name(str(item.get("name") or "")) for item in catalog}
    used_names = scene_names_used_in_story_points(data)
    missing_names = [name for name in used_names if name not in exact_names]
    if missing_names:
        for scene_name in missing_names:
            catalog.append(_infer_scene_catalog_entry_from_story_points(scene_name, story_points))
        normalized["_scene_catalog_auto_filled"] = missing_names
    normalized["scenes"] = normalize_scene_catalog(catalog, target_max=target_max)
    return normalized


HIGH_VALUE_PURPOSES = {"羞辱", "反击", "报仇", "揭示", "权力", "觉醒", "危险", "尾钩", "特效"}
WEAK_TRANSITION_HINTS = ("承接", "铺垫", "过场", "转场", "反应", "静观", "抬眼", "回望", "环视")


def _point_title_text(point: Mapping[str, Any]) -> str:
    return str(point.get("title") or "").strip()


def _is_transition_like_point(point: Mapping[str, Any]) -> bool:
    title = _point_title_text(point)
    if any(token in title for token in WEAK_TRANSITION_HINTS):
        return True
    pace_label = normalize_pace_label(point.get("pace_label"))
    _, end = parse_duration_window(str(point.get("duration_suggestion") or ""))
    detail_count = len([str(x).strip() for x in list(point.get("detail_anchor_lines") or []) if str(x).strip()])
    dialogue_count = len([str(x).strip() for x in list(point.get("key_dialogue_beats") or []) if str(x).strip()])
    return pace_label == "舒缓铺陈" and end <= 8.5 and detail_count <= 3 and dialogue_count <= 2


def _is_high_value_point(point: Mapping[str, Any]) -> bool:
    purpose = normalize_primary_purpose(
        point.get("primary_purpose"),
        fallback_parts=[
            point.get("title"),
            point.get("narrative_function"),
            point.get("director_statement"),
        ],
    )
    if purpose in HIGH_VALUE_PURPOSES:
        return True
    title = _point_title_text(point)
    return any(token in title for token in ("翻盘", "揭示", "反击", "羞辱", "爆发", "觉醒", "对轰", "镇压"))


def _merge_duration_suggestion(left: Mapping[str, Any], right: Mapping[str, Any], pace_label: str) -> str:
    if pace_label == "快压推进":
        return "9-12秒"
    if pace_label == "中速推进":
        return "8-11秒"
    return "7-9秒"


def _merge_shot_group(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    left_group = str(left.get("shot_group") or "").strip()
    right_group = str(right.get("shot_group") or "").strip()
    if left_group and left_group == right_group:
        return left_group
    if not left_group:
        return right_group or "双段切换"
    if not right_group:
        return left_group
    return "双段切换"


def _merge_titles(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    titles = _unique_ordered_text([_point_title_text(left), _point_title_text(right)])
    if not titles:
        return "剧情推进"
    if len(titles) == 1:
        return titles[0]
    if _is_transition_like_point(left) and not _is_transition_like_point(right):
        return titles[-1]
    if _is_transition_like_point(right) and not _is_transition_like_point(left):
        return titles[0]
    return shorten_text(" / ".join(titles), 26)


def _merge_text_fields(left_text: str, right_text: str, *, limit: int = 220) -> str:
    pieces = _unique_ordered_text(
        [piece.strip() for piece in (str(left_text or ""), str(right_text or "")) if str(piece or "").strip()]
    )
    if not pieces:
        return ""
    return shorten_text("；".join(pieces), limit)


def merge_story_points(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_chars = [str(item).strip() for item in list(left.get("characters") or []) if str(item).strip()]
    right_chars = [str(item).strip() for item in list(right.get("characters") or []) if str(item).strip()]
    left_scenes = [str(item).strip() for item in list(left.get("scenes") or []) if str(item).strip()]
    right_scenes = [str(item).strip() for item in list(right.get("scenes") or []) if str(item).strip()]
    micro_beats = _unique_ordered_text(
        [str(item).strip() for item in list(left.get("micro_beats") or []) + list(right.get("micro_beats") or []) if str(item).strip()]
    )[:MICRO_BEATS_CAP]
    detail_anchor_lines = _unique_ordered_text(
        [
            str(item).strip()
            for item in list(left.get("detail_anchor_lines") or []) + list(right.get("detail_anchor_lines") or [])
            if str(item).strip()
        ]
    )[:DETAIL_ANCHOR_LINES_CAP]
    key_dialogue_beats = _unique_ordered_text(
        [
            str(item).strip()
            for item in list(left.get("key_dialogue_beats") or []) + list(right.get("key_dialogue_beats") or [])
            if str(item).strip()
        ]
    )[:KEY_DIALOGUE_BEATS_CAP]
    left_high = _is_high_value_point(left)
    right_high = _is_high_value_point(right)
    if left_high and right_high and (
        normalize_pace_label(left.get("pace_label")) == "快压推进"
        or normalize_pace_label(right.get("pace_label")) == "快压推进"
    ):
        pace_label = "快压推进"
    elif left_high or right_high or any(
        normalize_pace_label(item.get("pace_label")) == "中速推进" for item in (left, right)
    ):
        pace_label = "中速推进"
    else:
        pace_label = "舒缓铺陈"
    merged = {
        "point_id": str(left.get("point_id") or right.get("point_id") or "").strip(),
        "title": _merge_titles(left, right),
        "primary_purpose": normalize_primary_purpose(
            right.get("primary_purpose") if right_high and not left_high else left.get("primary_purpose"),
            fallback_parts=[
                left.get("title"),
                right.get("title"),
                left.get("narrative_function"),
                right.get("narrative_function"),
                *micro_beats,
                *detail_anchor_lines,
                *key_dialogue_beats,
            ],
        ),
        "characters": _unique_ordered_text(left_chars + right_chars),
        "scenes": _unique_ordered_text(left_scenes + right_scenes),
        "shot_group": _merge_shot_group(left, right),
        "pace_label": pace_label,
        "duration_suggestion": _merge_duration_suggestion(left, right, pace_label),
        "narrative_function": _merge_text_fields(
            str(left.get("narrative_function") or ""),
            str(right.get("narrative_function") or ""),
            limit=180,
        ),
        "entry_state": str(left.get("entry_state") or "").strip() or str(right.get("entry_state") or "").strip(),
        "micro_beats": micro_beats,
        "detail_anchor_lines": detail_anchor_lines,
        "key_dialogue_beats": key_dialogue_beats,
        "sound_design_notes": _merge_text_fields(
            str(left.get("sound_design_notes") or ""),
            str(right.get("sound_design_notes") or ""),
            limit=160,
        ),
        "director_statement": _merge_text_fields(
            str(left.get("director_statement") or ""),
            str(right.get("director_statement") or ""),
            limit=320,
        ),
        "exit_state": str(right.get("exit_state") or "").strip() or str(left.get("exit_state") or "").strip(),
        "timeline_adjustment_note": _merge_text_fields(
            str(left.get("timeline_adjustment_note") or ""),
            str(right.get("timeline_adjustment_note") or ""),
            limit=120,
        ) or "无，按原剧本顺序推进。",
    }
    return merged


def _pair_merge_priority(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[int, int]:
    score = 0
    if _is_transition_like_point(left):
        score -= 3
    if _is_transition_like_point(right):
        score -= 3
    if not _is_high_value_point(left):
        score -= 1
    if not _is_high_value_point(right):
        score -= 1
    if normalize_pace_label(left.get("pace_label")) == "舒缓铺陈":
        score -= 1
    if normalize_pace_label(right.get("pace_label")) == "舒缓铺陈":
        score -= 1
    left_dialogue = len([str(x).strip() for x in list(left.get("key_dialogue_beats") or []) if str(x).strip()])
    right_dialogue = len([str(x).strip() for x in list(right.get("key_dialogue_beats") or []) if str(x).strip()])
    if left_dialogue + right_dialogue <= 3:
        score -= 1
    if _is_high_value_point(left) and _is_high_value_point(right):
        score += 4
    return (score, left_dialogue + right_dialogue)


def compact_story_points_to_target(
    story_points: Sequence[Mapping[str, Any]],
    *,
    target_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    points = [dict(item) for item in story_points]
    merge_log: list[dict[str, str]] = []
    safe_target = max(5, target_count)
    while len(points) > safe_target and len(points) >= 2:
        best_index = min(
            range(len(points) - 1),
            key=lambda idx: _pair_merge_priority(points[idx], points[idx + 1]),
        )
        left = points[best_index]
        right = points[best_index + 1]
        merged = merge_story_points(left, right)
        merge_log.append(
            {
                "left_point_id": str(left.get("point_id") or ""),
                "right_point_id": str(right.get("point_id") or ""),
                "merged_title": str(merged.get("title") or ""),
            }
        )
        points[best_index : best_index + 2] = [merged]
    return points, merge_log


def renumber_story_points(story_points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    width = max(2, len(str(len(story_points) or 1)))
    renumbered: list[dict[str, Any]] = []
    for index, raw_point in enumerate(story_points, start=1):
        point = dict(raw_point)
        point["point_id"] = str(index).zfill(width)
        renumbered.append(point)
    return renumbered


def finalize_story_point_sequence(
    data: Mapping[str, Any],
    *,
    metrics: Mapping[str, int],
) -> dict[str, Any]:
    finalized = dict(data)
    original_story_points = [dict(item) for item in list(data.get("story_points") or [])]
    target_count = int(metrics.get("recommended_story_points", len(original_story_points) or 5) or 5)
    compacted_story_points, merge_log = compact_story_points_to_target(
        original_story_points,
        target_count=target_count,
    )
    finalized["story_points"] = renumber_story_points(compacted_story_points)
    if merge_log:
        finalized["_story_point_compaction"] = {
            "original_count": len(original_story_points),
            "target_count": target_count,
            "final_count": len(finalized["story_points"]),
            "merged_pairs": merge_log,
        }
    return finalized


def _extract_backup_detail_lines(point: Mapping[str, Any]) -> list[str]:
    candidates: list[str] = []
    for field in ("key_dialogue_beats", "micro_beats"):
        for item in list(point.get(field) or []):
            text = str(item or "").strip()
            if text:
                candidates.append(text)
    for field in (
        "narrative_function",
        "director_statement",
        "entry_state",
        "exit_state",
    ):
        raw = str(point.get(field) or "").strip()
        if not raw:
            continue
        for piece in re.split(r"[；。！？\n]", raw):
            text = piece.strip()
            if text:
                candidates.append(text)
    return _unique_ordered_text(candidates)


def repair_director_density(data: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    repaired_story_points: list[dict[str, Any]] = []
    for raw_point in list(data.get("story_points") or []):
        point = dict(raw_point)
        pace_label = normalize_pace_label(point.get("pace_label"))
        minimum_beats = 5 if pace_label == "快压推进" else 4 if pace_label == "中速推进" else 3
        micro_beats = _unique_ordered_text(list(point.get("micro_beats") or []))[:MICRO_BEATS_CAP]
        detail_anchor_lines = _unique_ordered_text(list(point.get("detail_anchor_lines") or []))[:DETAIL_ANCHOR_LINES_CAP]
        key_dialogue_beats = _unique_ordered_text(list(point.get("key_dialogue_beats") or []))[:KEY_DIALOGUE_BEATS_CAP]
        backup_lines = _extract_backup_detail_lines(point)

        for candidate in backup_lines:
            if len(detail_anchor_lines) >= 2:
                break
            if candidate not in detail_anchor_lines:
                detail_anchor_lines.append(candidate)

        supplement_pool = _unique_ordered_text(
            [
                *detail_anchor_lines,
                *key_dialogue_beats,
                str(point.get("entry_state") or "").strip(),
                str(point.get("exit_state") or "").strip(),
                *backup_lines,
            ]
        )
        for candidate in supplement_pool:
            if len(micro_beats) >= minimum_beats:
                break
            if candidate and candidate not in micro_beats:
                micro_beats.append(candidate)

        if len(micro_beats) < minimum_beats:
            if pace_label == "快压推进" and len(micro_beats) >= 4:
                pace_label = "中速推进"
            elif pace_label == "中速推进" and len(micro_beats) >= 3:
                pace_label = "舒缓铺陈"

        point["pace_label"] = pace_label
        if not str(point.get("entry_state") or "").strip():
            point["entry_state"] = micro_beats[0] if micro_beats else "延续上一点的在镜状态进入当前动作。"
        if not str(point.get("exit_state") or "").strip():
            point["exit_state"] = micro_beats[-1] if micro_beats else "停在可直接切向下一点的明确状态。"
        if not str(point.get("timeline_adjustment_note") or "").strip():
            point["timeline_adjustment_note"] = "无，按原剧本顺序推进。"
        point["primary_purpose"] = normalize_primary_purpose(
            point.get("primary_purpose"),
            fallback_parts=[
                point.get("title"),
                point.get("narrative_function"),
                *micro_beats,
                *detail_anchor_lines,
                *key_dialogue_beats,
                point.get("entry_state"),
                point.get("exit_state"),
            ],
        )
        point["micro_beats"] = micro_beats[:MICRO_BEATS_CAP]
        point["detail_anchor_lines"] = detail_anchor_lines[:DETAIL_ANCHOR_LINES_CAP]
        point["key_dialogue_beats"] = key_dialogue_beats[:KEY_DIALOGUE_BEATS_CAP]
        repaired_story_points.append(point)

    normalized["story_points"] = repaired_story_points
    return normalized


def validate_director_density(data: Mapping[str, Any], metrics: Mapping[str, int], *, episode_id: str) -> list[str]:
    warnings: list[str] = []
    story_points = list(data.get("story_points") or [])
    minimum = int(metrics.get("recommended_story_points", 5))
    if len(story_points) < minimum:
        warnings.append(
            f"{episode_id} 的导演分析拆分过粗：当前剧情点 {len(story_points)} 个，"
            f"低于脚本密度建议下限 {minimum} 个。请增加剧情点拆分，避免吞掉对白与动作细节。"
        )
    maximum = max(minimum, math.ceil(minimum * 1.15))
    if len(story_points) > maximum:
        warnings.append(
            f"{episode_id} 的导演分析拆分过碎：当前剧情点 {len(story_points)} 个，"
            f"高于建议上限 {maximum} 个。请优先合并同一戏剧目的下的准备动作、短过桥与重复反应。"
        )
    for point in story_points:
        detail_anchor_lines = [str(x).strip() for x in list(point.get("detail_anchor_lines") or []) if str(x).strip()]
        pace_label = normalize_pace_label(point.get("pace_label"))
        micro_beats = [str(x).strip() for x in list(point.get("micro_beats") or []) if str(x).strip()]
        minimum_beats = 5 if pace_label == "快压推进" else 4 if pace_label == "中速推进" else 3
        if str(point.get("primary_purpose") or "").strip() not in PURPOSE_ORDER:
            warnings.append(
                f"{episode_id} 的 {point.get('point_id') or '未知剧情点'} 缺少合法 primary_purpose，"
                "必须使用固定分镜目的分类。"
            )
        if len(detail_anchor_lines) < 2:
            warnings.append(
                f"{episode_id} 的 {point.get('point_id') or '未知剧情点'} 细节锚点不足，"
                "每个剧情点至少需要 2 条 detail_anchor_lines 来绑定原剧本细节。"
            )
        if len(micro_beats) < minimum_beats:
            warnings.append(
                f"{episode_id} 的 {point.get('point_id') or '未知剧情点'} 节奏拆解不足："
                f"{pace_label} 至少需要 {minimum_beats} 个 micro_beats，当前只有 {len(micro_beats)} 个。"
            )
    if story_points:
        slow_points = 0
        overly_long_points = 0
        for point in story_points:
            pace_label = normalize_pace_label(point.get("pace_label"))
            _, end = parse_duration_window(str(point.get("duration_suggestion") or ""))
            if pace_label == "舒缓铺陈":
                slow_points += 1
            if end >= 11.5:
                overly_long_points += 1
        max_slow_points = max(1, math.ceil(len(story_points) * 0.3))
        if slow_points > max_slow_points:
            warnings.append(
                f"{episode_id} 的导演节奏仍偏慢：舒缓铺陈点位有 {slow_points} 个，超过建议上限 {max_slow_points} 个。"
                "请压缩弱过场与重复反应，把更多点位改成快压推进或中速推进。"
            )
        if len(story_points) >= 3 and sum(1 for point in story_points[:2] if normalize_pace_label(point.get("pace_label")) == "舒缓铺陈") >= 1:
            warnings.append(
                f"{episode_id} 的开场推进偏慢：前两条剧情点中已有舒缓铺陈。除非剧本明确要求静压开场，否则前两点应优先快压或中速推进。"
            )
        max_long_points = max(1, math.ceil(len(story_points) * 0.35))
        if overly_long_points > max_long_points:
            warnings.append(
                f"{episode_id} 的时长分配偏松：上限达到 11.5 秒以上的剧情点有 {overly_long_points} 个，"
                f"超过建议上限 {max_long_points} 个。请收短说明性段落，把时长集中给真正的燃点和兑现点。"
            )
    return warnings


def _looks_generic_transition_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    generic_exact = {
        "无",
        "承接上一段",
        "承接上一点",
        "延续上一段",
        "延续上一点",
        "情绪延续",
        "冲突升级",
        "进入下一段",
        "进入下一点",
        "为下一段铺垫",
        "自然推进到下一段",
        "自然切到下一段",
    }
    if value in generic_exact:
        return True
    if len(value) < 8:
        return True
    weak_patterns = (
        "情绪延续",
        "冲突升级",
        "自然进入下一段",
        "自然切到下一段",
        "继续推进剧情",
        "继续发展",
    )
    return any(pattern in value and len(value) <= 18 for pattern in weak_patterns)


def validate_director_continuity(data: Mapping[str, Any], *, episode_id: str) -> list[str]:
    warnings: list[str] = []
    for point in list(data.get("story_points") or []):
        for field_name, label in (
            ("entry_state", "入场状态"),
            ("exit_state", "出场状态"),
        ):
            if _looks_generic_transition_text(str(point.get(field_name) or "").strip()):
                warnings.append(
                    f"{episode_id} 的 {point.get('point_id') or '未知剧情点'} {label} 过空或过泛："
                    "请写出具体的角色、空间、动作、视线或道具连续状态。"
                )
        if not str(point.get("timeline_adjustment_note") or "").strip():
            warnings.append(
                f"{episode_id} 的 {point.get('point_id') or '未知剧情点'} 缺少时序整理说明："
                "即便无调整，也要明确写\u300c无，按原剧本顺序推进\u300d。"
            )
    return warnings


def validate_scene_budget(
    data: Mapping[str, Any],
    scene_budget: Mapping[str, int],
    *,
    episode_id: str,
) -> list[str]:
    warnings: list[str] = []
    scenes = list(data.get("scenes") or [])
    preferred_min = int(scene_budget.get("preferred_min", 1) or 1)
    preferred_max = int(scene_budget.get("preferred_max", preferred_min) or preferred_min)
    coverage_hints = infer_missing_scene_coverage_hints(data)
    used_scene_names = scene_names_used_in_story_points(data)
    catalog_scene_names = {_normalize_scene_name(str(item.get("name") or "")) for item in scenes}
    missing_catalog_names = [name for name in used_scene_names if name not in catalog_scene_names]

    if len(scenes) < preferred_min:
        hint_text = ""
        if coverage_hints:
            hint_text = f" 当前剧本里反复出现但尚未升格为稳定场景的空间锚点包括：{'、'.join(coverage_hints[:3])}。"
        warnings.append(
            f"{episode_id} 的场景母体偏少：当前 {len(scenes)} 个，低于建议区间 {preferred_min}-{preferred_max} 个。"
            "请优先补足 1-3 个真正可复用的辅助 establishing / 稳定子场景，"
            "不要把所有高价值分区都埋进单一主场景的 reuse_note 里。"
            + hint_text
        )

    if len(scenes) > preferred_max:
        warnings.append(
            f"{episode_id} 的场景母体偏多：当前 {len(scenes)} 个，高于建议区间 {preferred_min}-{preferred_max} 个。"
            "若这些场景确属多个不可互相复用的大空间可以保留，否则请继续优先并回可复用的大场景母体，避免为了局部机位差异单独起场景名。"
        )
    elif coverage_hints and len(scenes) <= preferred_min:
        warnings.append(
            f"{episode_id} 可能漏掉了稳定可复用的空间锚点：{'、'.join(coverage_hints[:3])}。"
            "若这些空间在多个剧情点里反复承担 establishing、桥接或高价值状态切换，请把它们提升为正式场景，并在对应 story_point.scenes 中点名。"
        )
    if missing_catalog_names:
        warnings.append(
            f"{episode_id} 的场景清单与剧情点引用失联：以下场景已在 story_points.scenes 中被点名，"
            f"但未出现在总场景表中：{'、'.join(missing_catalog_names[:6])}。请补齐总场景表，至少保证显式引用场景全部可追踪。"
        )
    return warnings


def compact_series_context_for_director(series_context: Mapping[str, Any]) -> dict[str, Any]:
    active_characters = []
    for item in list(series_context.get("active_characters") or [])[:6]:
        active_characters.append(
            {
                "name": item.get("name", ""),
                "role": item.get("role", ""),
                "relationship": item.get("relationship_to_protagonist", ""),
                "state": shorten_text(item.get("latest_state", ""), 120),
            }
        )

    active_locations = []
    for item in list(series_context.get("active_locations") or [])[:6]:
        if isinstance(item, Mapping):
            active_locations.append(
                {
                    "name": item.get("name", ""),
                    "state": shorten_text(item.get("latest_state", ""), 100),
                }
            )
        else:
            active_locations.append(shorten_text(str(item), 100))

    unresolved_threads = [shorten_text(item, 120) for item in list(series_context.get("unresolved_threads") or [])[:6]]

    recent_timeline = []
    for item in list(series_context.get("recent_timeline") or [])[-3:]:
        recent_timeline.append(
            {
                "episode_id": item.get("episode_id", ""),
                "title": item.get("title", ""),
                "synopsis": shorten_text(item.get("synopsis", ""), 180),
                "key_events": [shorten_text(event, 90) for event in list(item.get("key_events") or [])[:4]],
                "continuity_hooks": [shorten_text(event, 80) for event in list(item.get("continuity_hooks") or [])[:3]],
            }
        )

    return {
        "series_name": series_context.get("series_name", ""),
        "premise": shorten_text(series_context.get("premise", ""), 220),
        "latest_episode_id": series_context.get("latest_episode_id", ""),
        "continuity_rules": [shorten_text(item, 100) for item in list(series_context.get("continuity_rules") or [])[:6]],
        "genre_profile": series_context.get("genre_profile", {}),
        "downstream_design_guidance": series_context.get("downstream_design_guidance", {}),
        "active_characters": active_characters,
        "active_locations": active_locations,
        "unresolved_threads": unresolved_threads,
        "recent_timeline": recent_timeline,
    }


def compact_genre_reference_bundle_for_director(bundle: Mapping[str, Any]) -> dict[str, Any]:
    aggregate_focus = bundle.get("aggregate_focus") or {}
    matched_packages = []
    for item in list(bundle.get("matched_packages") or [])[:3]:
        matched_packages.append(
            {
                "genre_key": item.get("genre_key", ""),
                "director_focus": [shorten_text(x, 100) for x in list(item.get("director_focus") or [])[:4]],
                "art_focus": [shorten_text(x, 90) for x in list(item.get("art_focus") or [])[:3]],
                "storyboard_focus": [shorten_text(x, 90) for x in list(item.get("storyboard_focus") or [])[:3]],
                "continuity_guardrails": [shorten_text(x, 90) for x in list(item.get("continuity_guardrails") or [])[:3]],
            }
        )

    retrieved_reference_series = []
    for item in list(bundle.get("retrieved_reference_series") or [])[:3]:
        if isinstance(item, Mapping):
            retrieved_reference_series.append(
                {
                    "series_name": item.get("series_name", ""),
                    "genres": item.get("genres", []),
                    "why_relevant": shorten_text(item.get("why_relevant", ""), 120),
                    "takeaways": [shorten_text(x, 100) for x in list(item.get("takeaways") or [])[:3]],
                }
            )
        else:
            retrieved_reference_series.append(shorten_text(str(item), 120))

    return {
        "series_name": bundle.get("series_name", ""),
        "target_medium": bundle.get("target_medium", ""),
        "visual_style": shorten_text(bundle.get("visual_style", ""), 160),
        "selected_genres": bundle.get("selected_genres", []),
        "source_notes": [shorten_text(item, 120) for item in list(bundle.get("source_notes") or [])[:4]],
        "aggregate_focus": {
            "director_focus": [shorten_text(x, 100) for x in list(aggregate_focus.get("director_focus") or [])[:8]],
            "art_focus": [shorten_text(x, 90) for x in list(aggregate_focus.get("art_focus") or [])[:6]],
            "storyboard_focus": [shorten_text(x, 90) for x in list(aggregate_focus.get("storyboard_focus") or [])[:6]],
            "continuity_guardrails": [shorten_text(x, 90) for x in list(aggregate_focus.get("continuity_guardrails") or [])[:6]],
            "skill_reference_notes": [shorten_text(x, 90) for x in list(aggregate_focus.get("skill_reference_notes") or [])[:6]],
        },
        "matched_packages": matched_packages,
        "retrieved_reference_series": retrieved_reference_series,
    }


def build_script_anchor_text(draft_package: Mapping[str, Any]) -> str:
    """从初稿 JSON 提取关键对白与细节锚点，作为 Review 时的剧本参考（替代完整 script_text）。"""
    story_points = list((draft_package or {}).get("story_points") or [])
    if not story_points:
        return "<初稿无剧情点>"
    lines: list[str] = []
    for sp in story_points:
        pid = sp.get("point_id") or ""
        title = sp.get("title") or ""
        anchors = list(sp.get("detail_anchor_lines") or [])
        dialogues = list(sp.get("key_dialogue_beats") or [])
        if not anchors and not dialogues:
            continue
        lines.append(f"[{pid}] {title}")
        for a in anchors:
            lines.append(f"  锚点: {a}")
        for d in dialogues:
            lines.append(f"  对白: {d}")
        lines.append("")
    return "\n".join(lines).strip() or "<无关键对白锚点>"


def build_episode_prompt(
    *,
    config: Mapping[str, Any],
    series_name: str,
    episode_id: str,
    script_path: Path,
    script_text: str,
    script_density_guidance: str,
    scene_budget_guidance: str,
    series_context: Mapping[str, Any],
    genre_reference_bundle: Mapping[str, Any],
    existing_character_prompts: str,
    existing_scene_prompts: str,
) -> str:
    director_agent = read_text(PROJECT_ROOT / ".claude/agents/director.md")
    compact_series_context = ctx_compact.compact_series_context_for_director(series_context)
    compact_genre_reference_bundle = ctx_compact.compact_genre_reference_bundle_for_director(genre_reference_bundle)
    style = str(config.get("quality", {}).get("visual_style") or "").strip()
    medium = str(config.get("quality", {}).get("target_medium") or "").strip()
    frame_orientation = normalize_frame_orientation(config.get("quality", {}).get("frame_orientation"))
    frame_composition_guidance = build_frame_composition_guidance(frame_orientation)
    extra_rules = list(config.get("quality", {}).get("extra_rules", []))
    extra_rules_block = ""
    if extra_rules:
        extra_rules_block = "补充要求：\n" + render_bullets(extra_rules)
    return render_prompt(
        "director_analysis/draft_user.md",
        {
            "series_name": series_name,
            "episode_id": episode_id,
            "script_path": str(script_path),
            "visual_style": style or "未指定，请根据题材与剧本气质统一风格",
            "target_medium": medium or "漫剧",
            "frame_orientation": frame_orientation,
            "frame_composition_guidance": frame_composition_guidance,
            "extra_rules_block": extra_rules_block,
            "script_density_guidance": script_density_guidance,
            "scene_budget_guidance": scene_budget_guidance,
            "director_agent": ctx_compact.compact_reference_text(director_agent, 900),
            "series_context_json": json.dumps(compact_series_context, ensure_ascii=False, indent=2),
            "genre_reference_bundle_json": json.dumps(compact_genre_reference_bundle, ensure_ascii=False, indent=2),
            "existing_character_assets_summary": ctx_compact.compact_existing_character_assets_for_director(
                existing_character_prompts,
                episode_id,
                limit=220,
            ) or "<空>",
            "existing_scene_assets_summary": ctx_compact.compact_existing_scene_assets_for_director(
                existing_scene_prompts,
                episode_id,
                limit=260,
            ) or "<空>",
            "script_text": script_text,
        },
    )


def build_review_prompt(
    *,
    config: Mapping[str, Any],
    series_name: str,
    episode_id: str,
    script_path: Path,
    script_density_guidance: str,
    scene_budget_guidance: str,
    genre_reference_bundle: Mapping[str, Any],
    draft_package: Mapping[str, Any],
    draft_defects: list[str] | None = None,
) -> str:
    review_skill = load_skill("production/script-analysis-review-skill/SKILL.md")
    compact_genre_reference_bundle = ctx_compact.compact_genre_reference_bundle_for_director(genre_reference_bundle)
    compliance_skill = load_skill("production/compliance-review-skill/SKILL.md")
    style = str(config.get("quality", {}).get("visual_style") or "").strip()
    medium = str(config.get("quality", {}).get("target_medium") or "").strip()
    frame_orientation = normalize_frame_orientation(config.get("quality", {}).get("frame_orientation"))
    frame_composition_guidance = build_frame_composition_guidance(frame_orientation)
    if draft_defects:
        draft_defect_report = (
            "【初稿已检测到以下问题，必须优先修复，不得遗漏】\n"
            + "\n".join(f"- {defect}" for defect in draft_defects)
        )
    else:
        draft_defect_report = ""
    return render_prompt(
        "director_analysis/review_user.md",
        {
            "series_name": series_name,
            "episode_id": episode_id,
            "script_path": str(script_path),
            "visual_style": style or "按当前项目统一",
            "target_medium": medium or "漫剧",
            "frame_orientation": frame_orientation,
            "frame_composition_guidance": frame_composition_guidance,
            "script_density_guidance": script_density_guidance,
            "scene_budget_guidance": scene_budget_guidance,
            "genre_reference_bundle_json": json.dumps(compact_genre_reference_bundle, ensure_ascii=False, indent=2),
            "draft_package_json": json.dumps(draft_package, ensure_ascii=False, indent=2),
            "review_skill": ctx_compact.compact_reference_text(review_skill, 2200),
            "compliance_skill": ctx_compact.compact_reference_text(compliance_skill, 700),
            "draft_defect_report": draft_defect_report,
        },
    )


def merge_director_review_patch(
    draft_package: Mapping[str, Any],
    review_patch: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(draft_package)
    for field in ("structure_overview", "emotional_curve", "characters", "scenes", "director_notes"):
        if field in review_patch and review_patch.get(field) is not None:
            merged[field] = review_patch.get(field)

    story_points = [dict(item) for item in list(draft_package.get("story_points") or [])]
    delete_ids = {str(item).strip() for item in list(review_patch.get("delete_story_point_ids") or []) if str(item).strip()}
    if delete_ids:
        story_points = [item for item in story_points if str(item.get("point_id") or "").strip() not in delete_ids]

    patch_by_id = {
        str(item.get("point_id") or "").strip(): dict(item)
        for item in list(review_patch.get("story_point_patches") or [])
        if str(item.get("point_id") or "").strip()
    }
    if patch_by_id:
        updated: list[dict[str, Any]] = []
        for item in story_points:
            point_id = str(item.get("point_id") or "").strip()
            patch = patch_by_id.get(point_id)
            if not patch:
                updated.append(item)
                continue
            next_item = dict(item)
            for key, value in patch.items():
                if key == "point_id" or value is None:
                    continue
                next_item[key] = value
            updated.append(next_item)
        story_points = updated

    for insertion in list(review_patch.get("story_point_insertions") or []):
        anchor_id = str(insertion.get("insert_after_point_id") or "").strip()
        new_points = [dict(item) for item in list(insertion.get("story_points") or [])]
        if not new_points:
            continue
        existing_ids = {str(item.get("point_id") or "").strip() for item in story_points}
        deduped_points: list[dict[str, Any]] = []
        for item in new_points:
            point_id = str(item.get("point_id") or "").strip()
            if point_id and point_id in existing_ids:
                story_points = [sp for sp in story_points if str(sp.get("point_id") or "").strip() != point_id]
                existing_ids.discard(point_id)
            deduped_points.append(item)
        if not anchor_id:
            story_points.extend(deduped_points)
            continue
        insert_at = next((idx + 1 for idx, item in enumerate(story_points) if str(item.get("point_id") or "").strip() == anchor_id), len(story_points))
        story_points[insert_at:insert_at] = deduped_points

    merged["story_points"] = story_points
    return merged


def render_director_markdown(
    *,
    series_name: str,
    script_path: Path,
    data: Mapping[str, Any],
) -> str:
    lines = [
        "# 导演讲戏本",
        "",
        f"**项目**：{series_name}",
        f"**集数**：{data['episode_id']}",
        f"**源剧本**：{script_path.name}",
        f"**视觉风格**：{data['visual_style']}",
        f"**目标媒介**：{data['target_medium']}",
        "",
        "---",
        "",
        "## 剧情结构分析",
        "",
        data["structure_overview"].strip(),
        "",
        f"**情绪曲线**：{data['emotional_curve']}",
        "",
        "---",
        "",
    ]

    for point in data.get("story_points", []):
        micro_beats = [str(item).strip() for item in point.get("micro_beats", []) if str(item).strip()]
        detail_anchor_lines = [str(item).strip() for item in point.get("detail_anchor_lines", []) if str(item).strip()]
        dialogue_beats = [str(item).strip() for item in point.get("key_dialogue_beats", []) if str(item).strip()]
        lines.extend(
            [
                f"## {point['point_id']} - {point['title']}",
                "",
                f"- 分镜目的：{point.get('primary_purpose') or '对峙'}",
                f"- 人物：{'、'.join(point.get('characters', [])) or '无'}",
                f"- 场景：{'、'.join(point.get('scenes', [])) or '无'}",
                f"- 镜头组：{point['shot_group']}",
                f"- 节奏档位：{point.get('pace_label') or '中速推进'}",
                f"- 时长建议：{point['duration_suggestion']}",
                f"- 叙事功能：{point['narrative_function']}",
                f"- 入场状态：{point.get('entry_state') or '无'}",
                f"- 出场状态：{point.get('exit_state') or '无'}",
                f"- 时序整理说明：{point.get('timeline_adjustment_note') or '无'}",
                f"- 声音设计：{point.get('sound_design_notes') or '无'}",
                "",
                "**关键对白顺序**：",
                "",
            ]
        )
        if dialogue_beats:
            lines.extend([f"- {item}" for item in dialogue_beats])
        else:
            lines.append("- 无")

        lines.extend(["", "**微节拍拆解**：", ""])
        if micro_beats:
            lines.extend([f"- {item}" for item in micro_beats])
        else:
            lines.append("- 无")

        lines.extend(["", "**必须保留的剧本细节锚点**：", ""])
        if detail_anchor_lines:
            lines.extend([f"- {item}" for item in detail_anchor_lines])
        else:
            lines.append("- 无")

        lines.extend(
            [
                "",
                "**导演阐述**：",
                "",
                point["director_statement"].strip(),
                "",
                "---",
                "",
            ]
        )

    lines.extend(["## 人物清单", "", "| 人物 | 年龄 | 外观关键词 | 素材状态 |", "|------|------|----------|---------|"])
    for item in data.get("characters", []):
        lines.append(
            f"| {item['name']} | {item['age']} | {item['appearance_keywords']} | {format_asset_status(item)} |"
        )
    if not data.get("characters"):
        lines.append("| 无 | - | - | - |")

    lines.extend(["", "## 场景清单", "", "| 场景母体 | 时间 | 光线/色调 | 氛围关键词 | 素材状态 |", "|------|------|----------|----------|---------|"])
    for item in data.get("scenes", []):
        lines.append(
            f"| {item['name']} | {item['time_of_day']} | {item['lighting_palette']} | {item['mood_keywords']} | {format_asset_status(item)} |"
        )
    if not data.get("scenes"):
        lines.append("| 无 | - | - | - | - |")

    if data.get("director_notes"):
        lines.extend(["", "## 导演补充备注", ""])
        lines.extend([f"- {item}" for item in data["director_notes"]])

    return "\n".join(lines).rstrip() + "\n"

def run_pipeline(config: Mapping[str, Any], telemetry: TelemetryRecorder | None = None) -> dict[str, Any]:
    model, api_key = configure_openai_api(config)
    series_name = resolve_series_name(config)
    episode_ids = build_episode_ids(config.get("series", {}))
    assets_dir = resolve_assets_dir(config, series_name)
    outputs_root = resolve_outputs_root(config)
    analysis_root = resolve_analysis_root(config)
    series_context_path = analysis_root / series_name / "series_context.json"
    series_context = load_json_file(series_context_path) if series_context_path.exists() else {}
    # 优化方案1：优先使用预计算的compact bundle，避免重复加载和压缩
    genre_reference_bundle = config.get("_precomputed_bundle_cache", {}).get("director") or load_genre_reference_bundle(config, series_name)
    timeout_seconds = int(config.get("run", {}).get("timeout_seconds", 300))
    temperature = float(config.get("run", {}).get("temperature", 0.45))
    enable_review_pass = bool(config.get("run", {}).get("enable_review_pass", True))
    skip_existing_output = bool(config.get("run", {}).get("skip_existing_output", True))
    dry_run = bool(config.get("run", {}).get("dry_run", False))

    print_status(f"剧名：{series_name}")
    print_status(f"输出根目录：{outputs_root / resolve_output_series_name(config, series_name)}")

    preview_items: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    provider_tag = build_provider_model_tag("openai", model)

    for episode_id in episode_ids:
        script_path = choose_script_path(config, episode_id)
        episode_output_dir = resolve_episode_output_dir(config, series_name, episode_id)
        director_md_path = episode_output_dir / "01-director-analysis.md"
        director_json_path = episode_output_dir / f"01-director-analysis__{provider_tag}.json"

        preview_items.append(
            {
                "episode_id": episode_id,
                "script_path": str(script_path),
                "director_markdown_path": str(director_md_path),
                "director_json_path": str(director_json_path),
            }
        )

        if dry_run:
            continue

        if skip_existing_output and director_md_path.exists():
            print_status(f"跳过 {episode_id}：已存在导演讲戏本 {director_md_path}")
            results.append(
                {
                    "episode_id": episode_id,
                    "script_path": str(script_path),
                    "director_markdown_path": str(director_md_path),
                    "director_json_path": str(director_json_path),
                    "story_points": 0,
                    "characters": 0,
                    "scenes": 0,
                    "review_pass_enabled": enable_review_pass,
                    "skipped_existing": True,
                    "generated_at": utc_timestamp(),
                }
            )
            continue

        with telemetry_span(
            telemetry,
            stage="director_analysis",
            name="load_director_stage_inputs",
            metadata={"episode_id": episode_id, "script_path": str(script_path)},
        ) as step:
            script_text = script_path.read_text(encoding="utf-8")
            script_density_metrics = extract_script_density_metrics(script_text)
            script_density_guidance = build_script_density_guidance(script_density_metrics)
            scene_budget = resolve_scene_budget(config, script_density_metrics)
            scene_budget_guidance = build_scene_budget_guidance(config, script_density_metrics)
            existing_character_prompts = read_text(assets_dir / "character-prompts.md")
            existing_scene_prompts = read_text(assets_dir / "scene-prompts.md")
            compact_series_context = ctx_compact.compact_series_context_for_director(series_context)
            compact_genre_reference_bundle = ctx_compact.compact_genre_reference_bundle_for_director(genre_reference_bundle)
            step["metadata"]["script_chars"] = len(script_text)
            step["metadata"]["script_density_guidance"] = script_density_guidance
            step["metadata"]["scene_budget"] = scene_budget
            step["metadata"]["scene_budget_guidance"] = scene_budget_guidance
            step["metadata"]["recommended_story_points"] = script_density_metrics["recommended_story_points"]
            step["metadata"]["existing_character_prompt_chars"] = len(existing_character_prompts)
            step["metadata"]["existing_scene_prompt_chars"] = len(existing_scene_prompts)
            step["metadata"]["series_context_chars"] = len(json.dumps(series_context, ensure_ascii=False))
            step["metadata"]["series_context_compact_chars"] = len(json.dumps(compact_series_context, ensure_ascii=False))
            step["metadata"]["genre_reference_bundle_chars"] = len(json.dumps(genre_reference_bundle, ensure_ascii=False))
            step["metadata"]["genre_reference_bundle_compact_chars"] = len(json.dumps(compact_genre_reference_bundle, ensure_ascii=False))

        print_status(f"开始生成 {episode_id} 的导演分析：{script_path.name}")
        with telemetry_span(
            telemetry,
            stage="director_analysis",
            name="build_director_draft_prompt",
            metadata={"episode_id": episode_id},
        ) as step:
            draft_prompt = build_episode_prompt(
                config=config,
                series_name=series_name,
                episode_id=episode_id,
                script_path=script_path,
                script_text=script_text,
                script_density_guidance=script_density_guidance,
                scene_budget_guidance=scene_budget_guidance,
                series_context=series_context,
                genre_reference_bundle=genre_reference_bundle,
                existing_character_prompts=existing_character_prompts,
                existing_scene_prompts=existing_scene_prompts,
            )
            step["metadata"]["prompt_chars"] = len(draft_prompt)
        draft_result = openai_json_completion(
            model=model,
            api_key=api_key,
            system_prompt=load_prompt("director_analysis/draft_system.md"),
            user_prompt=draft_prompt,
            schema_name="director_analysis_package",
            schema=DIRECTOR_ANALYSIS_SCHEMA,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            telemetry=telemetry,
            stage="director_analysis",
            step_name="director_draft_model_call",
            metadata={"episode_id": episode_id},
        )
        final_result = draft_result
        review_completed = not enable_review_pass
        review_failed = False

        if enable_review_pass:
            print_status(f"开始复审并修订 {episode_id} 的导演分析。")
            # 在 review 之前先做一轮快速验证，把初稿缺陷注入 review prompt，让 review 精准修复
            pre_review_normalized = normalize_director_result(draft_result, scene_budget=scene_budget)
            pre_review_density_defects = validate_director_density(pre_review_normalized, script_density_metrics, episode_id=episode_id)
            pre_review_continuity_defects = validate_director_continuity(pre_review_normalized, episode_id=episode_id)
            pre_review_scene_defects = validate_scene_budget(pre_review_normalized, scene_budget, episode_id=episode_id)
            draft_defects = pre_review_density_defects + pre_review_continuity_defects + pre_review_scene_defects
            if draft_defects:
                print_status(f"{episode_id} 初稿发现 {len(draft_defects)} 条问题，已注入 review prompt 优先修复。")
            try:
                with telemetry_span(
                    telemetry,
                    stage="director_analysis",
                    name="build_director_review_prompt",
                    metadata={"episode_id": episode_id},
                ) as step:
                    review_prompt = build_review_prompt(
                        config=config,
                        series_name=series_name,
                        episode_id=episode_id,
                        script_path=script_path,
                        script_density_guidance=script_density_guidance,
                        scene_budget_guidance=scene_budget_guidance,
                        genre_reference_bundle=genre_reference_bundle,
                        draft_package=draft_result,
                        draft_defects=draft_defects if draft_defects else None,
                    )
                    step["metadata"]["prompt_chars"] = len(review_prompt)
                review_patch = openai_json_completion(
                    model=model,
                    api_key=api_key,
                    system_prompt=load_prompt("director_analysis/review_system.md"),
                    user_prompt=review_prompt,
                    schema_name="director_analysis_review_patch",
                    schema=DIRECTOR_REVIEW_PATCH_SCHEMA,
                    temperature=max(0.15, min(temperature, 0.25)),
                    timeout_seconds=timeout_seconds,
                    telemetry=telemetry,
                    stage="director_analysis",
                    step_name="director_review_model_call",
                    metadata={"episode_id": episode_id},
                )
                final_result = merge_director_review_patch(draft_result, review_patch)
                review_completed = True
            except Exception as exc:
                review_failed = True
                final_result = draft_result
                print_status(
                    f"{episode_id} 的导演 review 失败，将先落盘 draft 结果继续后续流程：{exc}"
                )

        with telemetry_span(
            telemetry,
            stage="director_analysis",
            name="render_and_save_director_outputs",
            metadata={
                "episode_id": episode_id,
                "director_markdown_path": str(director_md_path),
                "director_json_path": str(director_json_path),
                "review_completed": review_completed,
                "review_failed": review_failed,
            },
        ) as step:
            final_result = normalize_director_result(final_result, scene_budget=scene_budget)
            final_result = finalize_story_point_sequence(final_result, metrics=script_density_metrics)
            compaction_meta = dict(final_result.get("_story_point_compaction") or {})
            if compaction_meta:
                step["metadata"]["story_point_compaction"] = compaction_meta
                print_status(
                    f"{episode_id} 已将剧情点从 {compaction_meta.get('original_count')} 个压缩到 {compaction_meta.get('final_count')} 个，并统一重排为纯数字序号。"
                )
            # repair 降级为最终保底机制：先做一次验证，只有仍存在结构缺陷时才触发修补
            post_review_density = validate_director_density(final_result, script_density_metrics, episode_id=episode_id)
            post_review_continuity = validate_director_continuity(final_result, episode_id=episode_id)
            remaining_defects = post_review_density + post_review_continuity
            if remaining_defects:
                repaired = repair_director_density(final_result)
                repaired = finalize_story_point_sequence(repaired, metrics=script_density_metrics)
                # 标记哪些 story_point 被自动修补，方便追踪质量
                repaired_ids = {sp.get("point_id") for sp in repaired.get("story_points", []) if sp.get("point_id")}
                original_ids = {sp.get("point_id") for sp in final_result.get("story_points", []) if sp.get("point_id")}
                auto_repaired_points = list(repaired_ids & original_ids)
                repaired["_auto_repaired"] = True
                repaired["_auto_repaired_points"] = auto_repaired_points
                final_result = repaired
                print_status(
                    f"{episode_id} review 后仍有 {len(remaining_defects)} 条结构缺陷，已触发自动保底修补（_auto_repaired=True）。"
                )
                # repair 改变了 final_result，需重新验证
                density_warnings = validate_director_density(final_result, script_density_metrics, episode_id=episode_id)
                continuity_warnings = validate_director_continuity(final_result, episode_id=episode_id)
            else:
                # 无需 repair，直接复用已有结果，避免重复遍历
                density_warnings = post_review_density
                continuity_warnings = post_review_continuity
            scene_warnings = validate_scene_budget(final_result, scene_budget, episode_id=episode_id)
            quality_warnings = density_warnings + continuity_warnings + scene_warnings
            if quality_warnings:
                step["metadata"]["quality_warning_count"] = len(quality_warnings)
                step["metadata"]["quality_warnings"] = quality_warnings[:20]
                if telemetry is not None:
                    telemetry.context["director_quality_warning_count"] = len(quality_warnings)
                    telemetry.context["director_quality_warnings"] = quality_warnings[:20]
                print_status(
                    f"{episode_id} 的导演分析有 {len(quality_warnings)} 条质量告警：结果将继续保存，不再因密度/承接阈值中断。"
                )
            markdown = render_director_markdown(series_name=series_name, script_path=script_path, data=final_result)
            save_text_file(director_md_path, markdown)
            save_json_file(director_json_path, final_result)
            step["metadata"]["review_completed"] = review_completed
            step["metadata"]["review_failed"] = review_failed

        results.append(
            {
                "episode_id": episode_id,
                "script_path": str(script_path),
                "director_markdown_path": str(director_md_path),
                "director_json_path": str(director_json_path),
                "story_points": len(final_result.get("story_points", [])),
                "characters": len(final_result.get("characters", [])),
                "scenes": len(final_result.get("scenes", [])),
                "quality_warning_count": len(quality_warnings),
                "auto_repaired": bool(final_result.get("_auto_repaired", False)),
                "truncation_event_count": len(final_result.get("_truncation_events") or []),
                "review_pass_enabled": enable_review_pass,
                "review_completed": review_completed,
                "review_failed": review_failed,
                "generated_at": utc_timestamp(),
            }
        )
        print_status(
            f"{episode_id} 完成：剧情点 {results[-1]['story_points']}，人物 {results[-1]['characters']}，场景 {results[-1]['scenes']}。"
        )

    if dry_run:
        preview = {
            "series_name": series_name,
            "model": model,
            "assets_dir": str(assets_dir),
            "outputs_root": str(outputs_root / resolve_output_series_name(config, series_name)),
            "episodes": preview_items,
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return preview

    summary = {
        "series_name": series_name,
        "model": model,
        "outputs_root": str(outputs_root / resolve_output_series_name(config, series_name)),
        "assets_dir": str(assets_dir),
        "results": results,
        "generated_at": utc_timestamp(),
    }
    summary_path = outputs_root / resolve_output_series_name(config, series_name) / "director-analysis-summary.json"
    save_json_file(summary_path, summary)
    print_status(f"导演分析链路完成：{summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 01-director-analysis.md from script/<series>/ episodes.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print_status(f"加载配置：{args.config}")
    config = load_runtime_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
