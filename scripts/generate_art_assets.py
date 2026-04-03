from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_telemetry import TelemetryRecorder, apply_provider_usage, telemetry_span
from providers.base import (
    build_provider_model_tag,
    extract_json_from_text,
    load_json_file,
    save_json_file,
    utc_timestamp,
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
from series_paths import build_series_paths


DEFAULT_CONFIG_PATH = Path("config/art_assets_pipeline.local.json")
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 520}

ART_CHARACTER_ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "title", "image_requirement", "prompt", "change_type"],
    "properties": {
        "name": {"type": "string"},
        "title": {"type": "string"},
        "image_requirement": {"type": "string"},
        "prompt": {"type": "string"},
        "change_type": {"type": "string", "enum": ["新增", "变体"]},
    },
}

ART_SCENE_PANEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["slot", "scene_name", "view_prompt", "focus"],
    "properties": {
        "slot": {"type": "string"},
        "scene_name": {"type": "string"},
        "view_prompt": {"type": "string"},
        "focus": {"type": "string"},
    },
}

ART_SCENE_GRID_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "title",
        "grid_layout",
        "visual_style",
        "color_palette",
        "material_texture",
        "layout_notes",
        "panels",
        "change_type",
    ],
    "properties": {
        "title": {"type": "string"},
        "grid_layout": {"type": "string"},
        "visual_style": {"type": "string"},
        "color_palette": {"type": "string"},
        "material_texture": {"type": "string"},
        "layout_notes": {"type": "string"},
        "change_type": {"type": "string", "enum": ["新增", "变体"]},
        "panels": {"type": "array", "items": ART_SCENE_PANEL_SCHEMA},
    },
}

ART_PROMPTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["episode_id", "character_entries", "scene_grids"],
    "properties": {
        "episode_id": {"type": "string"},
        "character_entries": {"type": "array", "items": ART_CHARACTER_ENTRY_SCHEMA},
        "scene_grids": {"type": "array", "items": ART_SCENE_GRID_SCHEMA},
    },
}

ART_REVIEW_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "replace_character_entries",
        "delete_character_entry_titles",
        "replace_scene_grids",
        "delete_scene_grid_titles",
    ],
    "properties": {
        "replace_character_entries": {"type": "array", "items": ART_CHARACTER_ENTRY_SCHEMA},
        "delete_character_entry_titles": {"type": "array", "items": {"type": "string"}},
        "replace_scene_grids": {"type": "array", "items": ART_SCENE_GRID_SCHEMA},
        "delete_scene_grid_titles": {"type": "array", "items": {"type": "string"}},
    },
}


def print_status(message: str) -> None:
    print(f"[art-assets] {message}", flush=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_runtime_config(path: str | Path) -> dict[str, Any]:
    config = load_json(path)
    base_path = config.get("base_config")
    if not base_path:
        return config
    return deep_merge(load_json(base_path), config)


def request_json(
    *,
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    total_attempts = 3
    last_error: Exception | None = None
    for attempt in range(1, total_attempts + 1):
        request = urllib.request.Request(url=url, data=data, method="POST")
        for key, value in headers.items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"模型请求失败，状态码 {exc.code}，响应：{body}")
            if exc.code in RETRYABLE_HTTP_STATUS_CODES and attempt < total_attempts:
                time.sleep(min(6.0, 1.5 * attempt))
                continue
            raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"模型网络请求失败：{exc}")
            if attempt < total_attempts:
                time.sleep(min(6.0, 1.5 * attempt))
                continue
            raise last_error from exc
    if last_error:
        raise last_error
    raise RuntimeError("模型请求失败：未知错误。")


def extract_openai_text(response: Mapping[str, Any]) -> str:
    texts: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                texts.append(content["text"])
    if texts:
        return "\n".join(texts).strip()
    raise RuntimeError(f"OpenAI 响应中没有 output_text：{response}")


def extract_gemini_text(response: Mapping[str, Any]) -> str:
    texts: list[str] = []
    for candidate in response.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                texts.append(text)
    if texts:
        return "\n".join(texts).strip()
    raise RuntimeError(f"Gemini 响应中没有文本：{response}")


def model_json_completion(
    *,
    selected_provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    prompt: str,
    schema: Mapping[str, Any],
    temperature: float,
    timeout_seconds: int,
    telemetry: TelemetryRecorder | None = None,
    stage: str = "art_assets",
    step_name: str = "art_assets_model_call",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if selected_provider == "openai":
        payload = {
            "model": model,
            "temperature": temperature,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "art_asset_package",
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        with telemetry_span(
            telemetry,
            stage=stage,
            name=step_name,
            provider="openai",
            model=model,
            metadata=metadata,
        ) as step:
            response = request_json(
                url="https://api.openai.com/v1/responses",
                payload=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout_seconds=timeout_seconds,
            )
            apply_provider_usage(step, "openai", response)
        return extract_json_from_text(extract_openai_text(response))

    if selected_provider == "gemini":
        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": system_prompt,
                    }
                ]
            },
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            },
        }
        with telemetry_span(
            telemetry,
            stage=stage,
            name=step_name,
            provider="gemini",
            model=model,
            metadata=metadata,
        ) as step:
            response = request_json(
                url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                payload=payload,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                timeout_seconds=timeout_seconds,
            )
            apply_provider_usage(step, "gemini", response)
        return extract_json_from_text(extract_gemini_text(response))

    raise ValueError(f"不支持的 provider: {selected_provider}")


def read_text(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def resolve_output_series_names(config: dict[str, Any], series_name: str) -> list[str]:
    names: list[str] = []
    output_config = config.get("output", {})
    sources = config.get("sources", {})
    explicit = (output_config.get("outputs_series_name") or sources.get("outputs_series_name") or "").strip()
    suffix = (output_config.get("outputs_series_suffix") or sources.get("outputs_series_suffix") or "-gpt").strip()
    if explicit:
        names.append(explicit)
    else:
        names.append(f"{series_name}{suffix}")
    names.append(series_name)
    deduped: list[str] = []
    for name in names:
        if name and name not in deduped:
            deduped.append(name)
    return deduped


def find_analysis_path(config: dict[str, Any], series_name: str, episode_id: str) -> Path | None:
    sources = config["sources"]
    explicit = (sources.get("analysis_path") or "").strip()
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        return candidate if candidate.exists() else None

    preferred_provider = (sources.get("analysis_provider") or "").strip()
    preferred_model = (sources.get("analysis_model") or "").strip()
    analysis_dir = Path(config["run"].get("analysis_root", "analysis")) / series_name / episode_id
    analysis_dir = analysis_dir.expanduser().resolve()
    if preferred_provider and preferred_model:
        candidate = analysis_dir / f"episode_analysis__{build_provider_model_tag(preferred_provider, preferred_model)}.json"
        if candidate.exists():
            return candidate

    candidates = sorted(analysis_dir.glob("episode_analysis__*.json"))
    if not candidates:
        legacy = analysis_dir / "episode_analysis.json"
        if legacy.exists():
            return legacy
        return None
    return candidates[-1]


def find_director_analysis_path(config: dict[str, Any], series_name: str, episode_id: str) -> Path | None:
    sources = config.get("sources", {})
    explicit = (sources.get("director_analysis_path") or "").strip()
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        return candidate if candidate.exists() else None

    outputs_root = Path(sources.get("director_outputs_root") or "outputs").expanduser()
    if not outputs_root.is_absolute():
        outputs_root = (PROJECT_ROOT / outputs_root).resolve()

    candidates = []
    for candidate_series_name in resolve_output_series_names(config, series_name):
        candidates.append(outputs_root / candidate_series_name / episode_id / "01-director-analysis.md")
    candidates.append(outputs_root / episode_id / "01-director-analysis.md")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_director_analysis_json_path(config: dict[str, Any], series_name: str, episode_id: str) -> Path | None:
    outputs_root = Path(config.get("sources", {}).get("director_outputs_root") or "outputs").expanduser()
    if not outputs_root.is_absolute():
        outputs_root = (PROJECT_ROOT / outputs_root).resolve()
    for candidate_series_name in resolve_output_series_names(config, series_name):
        episode_dir = outputs_root / candidate_series_name / episode_id
        candidates = sorted(episode_dir.glob("01-director-analysis__*.json"))
        if candidates:
            return candidates[-1]
    return None


def build_bootstrap_analysis(
    *,
    episode_id: str,
    director_json: Mapping[str, Any] | None,
    director_analysis_text: str,
    series_context: Mapping[str, Any],
) -> dict[str, Any]:
    director_json = dict(director_json or {})
    story_points = list(director_json.get("story_points", []))
    characters = list(director_json.get("characters", []))
    scenes = list(director_json.get("scenes", []))
    director_notes = list(director_json.get("director_notes", []))
    story_beats: list[dict[str, Any]] = []
    for index, item in enumerate(story_points, start=1):
        story_beats.append(
            {
                "beat_id": str(item.get("point_id") or f"B{index:02d}"),
                "title": str(item.get("title", "")).strip(),
                "summary": str(item.get("narrative_function", "")).strip() or str(item.get("director_statement", ""))[:180].strip(),
                "characters": list(item.get("characters", [])),
                "locations": list(item.get("scenes", [])),
                "camera_language": str(item.get("shot_group", "")).strip(),
                "art_direction_cues": [str(item.get("director_statement", "")).strip()[:200]] if item.get("director_statement") else [],
                "storyboard_value": str(item.get("duration_suggestion", "")).strip(),
            }
        )

    downstream_guidance = dict(series_context.get("downstream_design_guidance", {}))
    if director_notes:
        downstream_guidance = {
            "script_reconstruction_focus": list(downstream_guidance.get("script_reconstruction_focus", [])),
            "character_design_focus": list(downstream_guidance.get("character_design_focus", [])),
            "scene_design_focus": list(downstream_guidance.get("scene_design_focus", [])),
            "storyboard_focus": list(downstream_guidance.get("storyboard_focus", [])) + director_notes[:6],
        }

    return {
        "episode_id": episode_id,
        "episode_title": str(director_json.get("episode_title", "")).strip(),
        "structure_overview": str(director_json.get("structure_overview", "")).strip(),
        "emotional_curve": str(director_json.get("emotional_curve", "")).strip(),
        "characters": characters,
        "locations": scenes,
        "story_beats": story_beats,
        "director_notes": director_notes,
        "downstream_design_guidance": downstream_guidance,
        "bootstrap_source": "director_analysis_json" if director_json else "director_analysis_markdown",
        "director_analysis_excerpt": director_analysis_text[:2500],
    }


def resolve_assets_dir(config: dict[str, Any], series_name: str) -> Path:
    output_config = config.get("output", {})
    explicit_assets_series_name = (output_config.get("assets_series_name") or "").strip()
    assets_series_suffix = (output_config.get("assets_series_suffix") or "").strip()
    target_series_name = explicit_assets_series_name or f"{series_name}{assets_series_suffix}"
    return (PROJECT_ROOT / "assets" / target_series_name).resolve()


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


def shorten_text(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def normalize_text_token(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip())


def extract_title_descriptor(name: str, title: str) -> str:
    current = str(title or "").strip()
    character_name = str(name or "").strip()
    if character_name and current.startswith(character_name):
        current = current[len(character_name):].strip()
    current = current.lstrip("（(|｜|:：- ").rstrip("）)")
    return current.strip()


def is_default_debut_descriptor(descriptor: str, episode_id: str) -> bool:
    compact = normalize_text_token(descriptor)
    if not compact:
        return False
    episode_compact = normalize_text_token(episode_id)
    if episode_compact and compact.startswith(episode_compact):
        compact = compact[len(episode_compact):]
    default_markers = {
        normalize_text_token("首集主形象设定"),
        normalize_text_token("首集正式出场设定"),
        normalize_text_token("首次出场设定"),
        normalize_text_token("首集初登场造型"),
    }
    return compact in default_markers


def canonical_default_debut_title(name: str, episode_id: str) -> str:
    return f"{name}｜{episode_id}首集正式出场设定"


def normalize_character_title_key(name: str, title: str) -> str:
    character_name = str(name or "").strip()
    normalized = str(title or "").strip()
    descriptor = extract_title_descriptor(character_name, normalized)
    compact = normalize_text_token(descriptor)
    compact = re.sub(r"^ep\d+", "", compact, flags=re.IGNORECASE)
    if compact in {
        normalize_text_token("首集主形象设定"),
        normalize_text_token("首集正式出场设定"),
        normalize_text_token("首次出场设定"),
        normalize_text_token("首集初登场造型"),
    }:
        return f"{character_name}|default_debut"
    normalized = re.sub(r"（\s*ep\d+\s*(新增|变体)\s*[:：-]?\s*", "（", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def normalize_title_for_match(title: str) -> str:
    normalized = str(title or "").strip()
    normalized = re.sub(r"（\s*ep\d+\s*(新增|变体)\s*[:：-]?\s*", "（", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def split_director_variant_text(raw_text: str) -> tuple[str, str, str]:
    text = str(raw_text or "").strip()
    day_match = re.search(r"白天为([^；。]+)", text)
    night_match = re.search(r"夜晚为([^；。]+)", text)
    memory_match = re.search(r"(少女[^；。]*|回忆态[^；。]*|少年态[^；。]*|童年态[^；。]*)", text)
    return (
        day_match.group(1).strip() if day_match else "",
        night_match.group(1).strip() if night_match else "",
        memory_match.group(1).strip() if memory_match else "",
    )


def build_character_presence_map(director_json: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for story_point in list(director_json.get("story_points") or []):
        if not isinstance(story_point, Mapping):
            continue
        seen_in_point: set[str] = set()
        for raw_name in list(story_point.get("characters") or []):
            name = str(raw_name or "").strip()
            if not name or name in seen_in_point:
                continue
            seen_in_point.add(name)
            counts[name] = counts.get(name, 0) + 1
    return counts


def classify_character_visual_priority(
    *,
    name: str,
    appearance_keywords: str,
    reuse_note: str,
    story_presence: int,
    total_story_points: int,
) -> tuple[str, str]:
    normalized = f"{appearance_keywords} {reuse_note}"
    low_status_tokens = ("丫鬟", "旧", "素", "粗布", "布衣", "囚", "病", "破", "灰", "洗旧")
    luxury_tokens = ("礼服", "华服", "华贵", "世家", "贵气", "金", "玉", "绣", "冠", "凤", "宫", "大夫人", "家主")
    high_presence_threshold = max(3, math.ceil(max(1, total_story_points) * 0.35))
    if story_presence >= high_presence_threshold:
        priority = "核心主角级"
    elif story_presence >= 2:
        priority = "关键角色级"
    else:
        priority = "功能配角级"

    has_low_status = any(token in normalized for token in low_status_tokens)
    has_luxury_signal = any(token in normalized for token in luxury_tokens)
    if priority == "核心主角级" and has_low_status:
        finish = (
            "即便身份与服装设定偏朴素，也必须做出主角级完成度：主材质/副材质分层明确，领口、肩颈、胸前、腰线、袖口与手持物要有稳定识别锚点，"
            "布料纹理、针脚、压褶、旧化层次和微金属/木玉点缀要精细，不得做成廉价粗糙的基础款。"
        )
    elif priority == "核心主角级" or has_luxury_signal:
        finish = (
            "这是镜头主视线人物，服化道必须更精美：层叠面料、刺绣/织纹、金属与玉石细节、腰封结构、发饰工艺和主辅材质反差都要清楚，"
            "让人物在9:16中近景里一眼看出贵重度、身份感和电影级完成度。"
        )
    elif priority == "关键角色级":
        finish = (
            "需要明显高于普通配角的工艺完成度：至少给出清楚的主副材质、发饰/配件、胸前结构、肩颈轮廓和一两个镜头级识别锚点。"
        )
    else:
        finish = "保持项目统一质感即可，避免过度简化成低成本模板脸或单层服装。"
    return priority, finish


def infer_director_character_requirements(episode_id: str, director_json: Mapping[str, Any]) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    characters = list(director_json.get("characters") or [])
    story_presence_map = build_character_presence_map(director_json)
    total_story_points = len(list(director_json.get("story_points") or []))
    for item in characters:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        if ctx_compact.is_mixed_crowd_character_asset(
            name,
            appearance_keywords=str(item.get("appearance_keywords", "")).strip(),
            reuse_note=str(item.get("reuse_note", "")).strip(),
        ):
            continue
        asset_status = str(item.get("asset_status", "")).strip()
        # 允许"复用"人物被生成：即使在前集出现过，本集若在导演分析中列出，仍需要提示词参考
        # 只过滤掉明确标记为"无关"或空值的人物
        if asset_status == "无关":
            continue

        appearance_keywords = str(item.get("appearance_keywords", "")).strip()
        reuse_note = str(item.get("reuse_note", "")).strip()
        story_presence = int(story_presence_map.get(name, 0) or 0)
        visual_priority, finish_guidance = classify_character_visual_priority(
            name=name,
            appearance_keywords=appearance_keywords,
            reuse_note=reuse_note,
            story_presence=story_presence,
            total_story_points=total_story_points,
        )
        day_desc, night_desc, memory_desc = split_director_variant_text(appearance_keywords)

        def add_requirement(title: str, prompt_focus: str, match_keywords: list[str]) -> None:
            requirements.append(
                {
                    "name": name,
                    "title": title,
                    "change_type": "新增" if asset_status.startswith("新增") else "变体",
                    "image_requirement": "一张图，左半边面部特写，右半边全身正面、侧面、背面三视图设定图，白色背景",
                    "prompt_focus": prompt_focus.strip(),
                    "appearance_keywords": appearance_keywords,
                    "reuse_note": reuse_note,
                    "story_presence": story_presence,
                    "visual_priority": visual_priority,
                    "finish_guidance": finish_guidance,
                    "match_keywords": [keyword for keyword in match_keywords if keyword],
                }
            )

        if day_desc:
            add_requirement(
                f"{name}（现实职场造型）",
                f"当前版本只聚焦现实/白天状态，重点使用：{day_desc}。不要混入夜宴礼服或回忆时期元素。",
                [name, "现实", "职场"],
            )
        if night_desc:
            add_requirement(
                f"{name}（夜宴礼服造型）",
                f"当前版本只聚焦夜晚宴会状态，重点使用：{night_desc}。不要混入白天职场工牌或回忆时期元素。",
                [name, "夜", "礼服"],
            )
        if memory_desc:
            add_requirement(
                f"{name}（少女舞者回忆造型）" if "舞" in memory_desc else f"{name}（回忆时期造型）",
                f"当前版本只聚焦回忆/旧日状态，重点使用：{memory_desc}。画面应与成年现实态拉开区分。",
                [name, "回忆", "少女" if "少" in memory_desc else ""],
            )

        if not day_desc and not night_desc and not memory_desc:
            default_title = canonical_default_debut_title(name, episode_id)
            if "宴会" in appearance_keywords or "西装" in appearance_keywords:
                default_title = f"{name}（商务宴会初登场造型）"
            elif "职场" in appearance_keywords or "商务" in appearance_keywords:
                default_title = f"{name}（现实职场造型）"
            add_requirement(
                default_title,
                "当前版本聚焦本集正式出场状态，必须忠于导演讲戏本中的首次亮相气质与服装信息。",
                [name, "首", "形象"] if "首集" in default_title else [name, "宴会" if "宴会" in default_title else "职场"],
            )

    deduped: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in requirements:
        key = (item["name"], normalize_title_for_match(item["title"]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(item)
    return deduped


def render_required_character_coverage(requirements: list[dict[str, Any]]) -> str:
    if not requirements:
        return "<空>"
    lines: list[str] = []
    for item in requirements:
        lines.append(
            f"- {item['title']}：主名={item['name']}；角色层级={item.get('visual_priority') or '未标注'}；"
            f"覆盖重点={item['prompt_focus']}；审美完成度要求={shorten_text(item.get('finish_guidance', ''), 120)}；"
            f"导演依据={shorten_text(item.get('appearance_keywords', ''), 120)}"
        )
    return "\n".join(lines)


def build_scene_presence_map(director_json: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for story_point in list(director_json.get("story_points") or []):
        if not isinstance(story_point, Mapping):
            continue
        seen_in_point: set[str] = set()
        for raw_name in list(story_point.get("scenes") or []):
            name = str(raw_name or "").strip()
            if not name or name in seen_in_point:
                continue
            seen_in_point.add(name)
            counts[name] = counts.get(name, 0) + 1
    return counts


def classify_scene_visual_priority(name: str, coverage_role: str, story_presence: int) -> tuple[str, str]:
    macro_tokens = ("宫", "殿", "门", "前院", "高台", "法阵", "阵地", "外圈", "长阶", "山门", "府")
    private_tokens = ("居室", "卧房", "病室", "后院", "内室", "偏厅")
    if coverage_role == "辅助场景":
        return "桥接 establishing 级", "重点保证空间体量、远端轴线、出入口与主光方向，不要做成零散小景。"
    if story_presence >= 2 or any(token in name for token in macro_tokens):
        return "宏大母体级", "优先强化建筑体量、门楼/石阶/立柱/廊道纵深、主光方向和可重复取景的稳定骨架。"
    if any(token in name for token in private_tokens):
        return "情绪私景级", "重点强化室内层级、帘幕/床榻/屏风/窗棂/火盆等精细材质与局部暖冷光反差。"
    return "标准场景级", "在统一画风下保证清晰的空间结构、材质分层和9:16取景友好度。"


def infer_director_scene_requirements(episode_id: str, director_json: Mapping[str, Any]) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    supporting_scene_limit = 2
    supporting_scene_count = 0
    scene_presence_map = build_scene_presence_map(director_json)
    for item in list(director_json.get("scenes") or []):
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        asset_status = str(item.get("asset_status", "")).strip()
        # 允许"复用"场景被生成：即使在前集出现过，本集若在导演分析中列出，仍需要提示词参考
        # 只过滤掉明确标记为"无关"的场景
        if asset_status == "无关":
            continue
        normalized_name = normalize_text_token(name)
        if normalized_name in seen_names:
            continue
        seen_names.add(normalized_name)
        story_presence = int(scene_presence_map.get(name, 0) or 0)
        visual_priority, design_focus = classify_scene_visual_priority(name, "主场景", story_presence)
        requirements.append(
            {
                "name": name,
                "title": f"{episode_id} 场景宫格：{name}",
                "change_type": "新增" if asset_status.startswith("新增") else "变体",
                "time_of_day": str(item.get("time_of_day", "")).strip(),
                "lighting_palette": str(item.get("lighting_palette", "")).strip(),
                "mood_keywords": str(item.get("mood_keywords", "")).strip(),
                "reuse_note": str(item.get("reuse_note", "")).strip(),
                "coverage_role": "主场景",
                "story_presence": story_presence,
                "visual_priority": visual_priority,
                "design_focus": design_focus,
            }
        )

    supporting_keywords = (
        "外圈",
        "外围",
        "阵地",
        "高台",
        "长阶",
        "山门",
        "门外",
        "通道",
        "镜门",
        "法阵外圈",
        "远端",
        "轴线",
        "防线",
        "观礼台",
        "看台",
        "廊桥",
    )
    for story_point in list(director_json.get("story_points") or []):
        if supporting_scene_count >= supporting_scene_limit:
            break
        if not isinstance(story_point, Mapping):
            continue
        point_title = str(story_point.get("title", "")).strip()
        point_statement = " ".join(
            [
                point_title,
                str(story_point.get("opening_visual_state", "")).strip(),
                str(story_point.get("closing_visual_state", "")).strip(),
                str(story_point.get("director_statement", "")).strip(),
            ]
        )
        for scene_name in list(story_point.get("scenes") or []):
            name = str(scene_name or "").strip()
            if not name:
                continue
            normalized_name = normalize_text_token(name)
            if normalized_name in seen_names:
                continue
            if not any(keyword in name or keyword in point_statement for keyword in supporting_keywords):
                continue
            seen_names.add(normalized_name)
            supporting_scene_count += 1
            visual_priority, design_focus = classify_scene_visual_priority(name, "辅助场景", 1)
            requirements.append(
                {
                    "name": name,
                    "title": f"{episode_id} 场景宫格：{name}",
                    "change_type": "新增",
                    "time_of_day": "",
                    "lighting_palette": "",
                    "mood_keywords": shorten_text(point_statement, 120),
                    "reuse_note": f"辅助 establishing 场景；服务 {point_title or '对应剧情点'} 的体量建立、空间桥接或远端轴线交代。",
                    "coverage_role": "辅助场景",
                    "story_presence": 1,
                    "visual_priority": visual_priority,
                    "design_focus": design_focus,
                }
            )
            if supporting_scene_count >= supporting_scene_limit:
                break
    return requirements


def render_required_scene_coverage(requirements: list[dict[str, Any]]) -> str:
    if not requirements:
        return "<空>"
    lines: list[str] = []
    for item in requirements:
        lines.append(
            f"- {item['name']}：类型={item.get('coverage_role') or '主场景'}；视觉层级={item.get('visual_priority') or '未标注'}；"
            f"时间={item.get('time_of_day') or '未写'}；光线/色调={item.get('lighting_palette') or '未写'}；"
            f"氛围={item.get('mood_keywords') or '未写'}；设计重点={item.get('design_focus') or '未写'}；"
            f"状态={item.get('change_type')}；复用说明={item.get('reuse_note') or '未写'}"
        )
    return "\n".join(lines)


def compact_analysis_context_for_art(
    analysis: Mapping[str, Any],
    *,
    director_available: bool,
) -> dict[str, Any]:
    if not director_available:
        return ctx_compact.compact_analysis_for_art(analysis)

    story_beats: list[dict[str, Any]] = []
    for item in list(analysis.get("story_beats") or [])[:2]:
        story_beats.append(
            {
                "beat_id": item.get("beat_id", ""),
                "title": shorten_text(item.get("title", ""), 36),
                "summary": shorten_text(item.get("summary", ""), 80),
                "art_direction_cues": [shorten_text(x, 70) for x in list(item.get("art_direction_cues") or [])[:2]],
            }
        )

    characters = []
    for item in list(analysis.get("characters") or [])[:4]:
        if isinstance(item, Mapping):
            characters.append({"name": item.get("name", ""), "role": item.get("role", "")})
        else:
            characters.append(shorten_text(str(item), 50))

    locations = []
    for item in list(analysis.get("locations") or [])[:4]:
        if isinstance(item, Mapping):
            locations.append({"name": item.get("name", "")})
        else:
            locations.append(shorten_text(str(item), 50))

    return {
        "episode_id": analysis.get("episode_id", ""),
        "episode_title": analysis.get("episode_title", ""),
        "structure_overview": shorten_text(analysis.get("structure_overview", ""), 110),
        "characters": characters,
        "locations": locations,
        "story_beats": story_beats,
        "bootstrap_source": analysis.get("bootstrap_source", ""),
    }


def render_director_art_bundle(
    *,
    director_brief: str,
    character_requirements: list[dict[str, Any]],
    scene_requirements: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    brief = str(director_brief or "").strip()
    if brief:
        lines.append("导演执行摘要：")
        lines.extend(brief.splitlines())
    if character_requirements:
        lines.append("人物硬覆盖：")
        for item in character_requirements:
            lines.append(
                "- "
                + shorten_text(
                    f"{item['title']}｜层级={item.get('visual_priority') or '未标注'}｜"
                    f"重点={item.get('prompt_focus') or '未写'}｜"
                    f"依据={item.get('appearance_keywords') or '未写'}",
                    170,
                )
            )
    if scene_requirements:
        lines.append("场景硬覆盖：")
        for item in scene_requirements:
            lines.append(
                "- "
                + shorten_text(
                    f"{item['name']}｜类型={item.get('coverage_role') or '主场景'}｜"
                    f"视觉层级={item.get('visual_priority') or '未标注'}｜"
                    f"重点={item.get('design_focus') or '未写'}｜"
                    f"状态={item.get('change_type') or '未写'}",
                    170,
                )
            )
    return "\n".join(lines).strip() or "<空>"


def merge_art_review_patch(
    draft_package: Mapping[str, Any],
    review_patch: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(draft_package)

    delete_character_titles = {
        normalize_title_for_match(str(item))
        for item in list(review_patch.get("delete_character_entry_titles") or [])
        if str(item).strip()
    }
    character_entries = [
        dict(item)
        for item in list(draft_package.get("character_entries") or [])
        if normalize_title_for_match(str(item.get("title") or "")) not in delete_character_titles
    ]
    replace_character_map = {
        normalize_title_for_match(str(item.get("title") or "")): dict(item)
        for item in list(review_patch.get("replace_character_entries") or [])
        if normalize_title_for_match(str(item.get("title") or ""))
    }
    if replace_character_map:
        next_entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in character_entries:
            key = normalize_title_for_match(str(item.get("title") or ""))
            patch_item = replace_character_map.get(key)
            next_entries.append(patch_item if patch_item is not None else item)
            seen.add(key)
        for key, item in replace_character_map.items():
            if key not in seen:
                next_entries.append(item)
        character_entries = next_entries
    merged["character_entries"] = character_entries

    delete_scene_grid_titles = {
        normalize_text_token(str(item))
        for item in list(review_patch.get("delete_scene_grid_titles") or [])
        if str(item).strip()
    }
    scene_grids = [
        dict(item)
        for item in list(draft_package.get("scene_grids") or [])
        if normalize_text_token(str(item.get("title") or "")) not in delete_scene_grid_titles
    ]
    replace_scene_map = {
        normalize_text_token(str(item.get("title") or "")): dict(item)
        for item in list(review_patch.get("replace_scene_grids") or [])
        if normalize_text_token(str(item.get("title") or ""))
    }
    if replace_scene_map:
        next_grids: list[dict[str, Any]] = []
        seen_grid_keys: set[str] = set()
        for item in scene_grids:
            key = normalize_text_token(str(item.get("title") or ""))
            patch_item = replace_scene_map.get(key)
            next_grids.append(patch_item if patch_item is not None else item)
            seen_grid_keys.add(key)
        for key, item in replace_scene_map.items():
            if key not in seen_grid_keys:
                next_grids.append(item)
        scene_grids = next_grids
    merged["scene_grids"] = scene_grids
    return merged


def recommended_scene_grid_layout(scene_count: int) -> str:
    count = max(0, int(scene_count))
    if count <= 4:
        return "2×2 宫格"
    if count <= 6:
        return "2×3 宫格"
    if count <= 9:
        return "3×3 九宫格"
    if count <= 12:
        return "3×4 宫格"
    return "4×4 宫格"


def build_character_style_preamble(target_medium: str, visual_style: str) -> str:
    medium = str(target_medium or "").strip()
    style = str(visual_style or "").strip()
    if medium == "漫剧":
        return (
            "角色设定图，统一采用高完成度半写实国漫电影角色概念设定体系，不做实拍照片感，"
            "但强调真实骨相、成熟受光、细腻皮肤与织物/金属材质、边缘光、空气透视和电影级体积感；"
            "避免低幼、Q版、扁平赛璐璐、粗重外轮廓和廉价平涂感，适合高水准动态视频参考。"
        )
    if medium in {"电影", "电视剧"}:
        return "角色设定图，统一采用影视设定图体系，保持人物风格统一、可执行、便于后续服化道与动态镜头开发。"
    if style:
        return f"角色设定图，统一采用与当前项目匹配的{style}设定图体系。"
    return "角色设定图，统一采用与当前项目匹配的稳定角色设定图体系。"


def build_fallback_character_prompt(*, target_medium: str, visual_style: str, item: Mapping[str, Any]) -> str:
    preamble = build_character_style_preamble(target_medium, visual_style)
    name = str(item.get("name", "")).strip()
    title = str(item.get("title", "")).strip()
    appearance_keywords = str(item.get("appearance_keywords", "")).strip()
    reuse_note = str(item.get("reuse_note", "")).strip()
    prompt_focus = str(item.get("prompt_focus", "")).strip()
    visual_priority = str(item.get("visual_priority", "")).strip()
    finish_guidance = str(item.get("finish_guidance", "")).strip()
    focus_note = f"{prompt_focus} " if prompt_focus else ""
    reuse_line = f"连续性要求：{reuse_note} " if reuse_note else ""
    priority_line = f"角色层级：{visual_priority}。{finish_guidance} " if visual_priority or finish_guidance else ""
    return (
        f"{preamble} 纯白背景，左侧为面部特写，右侧为全身正面、侧面、背面三视图，三视图等比例水平排列，"
        f"要求发型、服装轮廓、鞋履、配饰与体态线条清楚可辨。角色为{name}，当前造型标题为“{title}”。"
        f"{priority_line}{focus_note}导演讲戏本提供的外观与气质关键词为：{appearance_keywords}。"
        f"{reuse_line}提示词必须忠于导演讲戏本，不擅自发明无依据的复杂道具、品牌信息或性感化设计；"
        "重点把人物的年龄层、脸型五官、发型、体态、服装颜色/材质/层次和整体气质描述完整，"
        "并明确这是同剧统一风格体系中的稳定角色设定图。额外强调成熟电影感受光、皮肤与布料微质感、金属与玉石反射、"
        "体积雾和边缘光，不要写成低幼卡通、平涂插画或简单线条风。"
    ).strip()


def entry_matches_requirement(entry: Mapping[str, Any], requirement: Mapping[str, Any]) -> bool:
    entry_name = str(entry.get("name", "")).strip()
    requirement_name = str(requirement.get("name", "")).strip()
    if not entry_name or entry_name != requirement_name:
        return False

    entry_title = normalize_character_title_key(entry_name, str(entry.get("title", "")))
    required_title = normalize_character_title_key(requirement_name, str(requirement.get("title", "")))
    if entry_title == required_title or required_title in entry_title or entry_title in required_title:
        return True

    haystack = normalize_text_token(" ".join([str(entry.get("title", "")), str(entry.get("prompt", ""))]))
    keywords = [normalize_text_token(x) for x in list(requirement.get("match_keywords") or []) if normalize_text_token(x)]
    return bool(keywords) and all(keyword in haystack for keyword in keywords)


def ensure_character_entry_coverage(
    *,
    episode_id: str,
    target_medium: str,
    visual_style: str,
    package_entries: list[dict[str, Any]],
    director_json: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    required_items = infer_director_character_requirements(episode_id, director_json)
    if not required_items:
        return package_entries, []

    merged_entries = [dict(item) for item in package_entries]
    missing_titles: list[str] = []
    for requirement in required_items:
        if any(entry_matches_requirement(entry, requirement) for entry in merged_entries):
            continue
        missing_titles.append(requirement["title"])
        merged_entries.append(
            {
                "name": requirement["name"],
                "title": requirement["title"],
                "image_requirement": requirement["image_requirement"],
                "prompt": build_fallback_character_prompt(
                    target_medium=target_medium,
                    visual_style=visual_style,
                    item=requirement,
                ),
                "change_type": requirement["change_type"],
            }
        )

    order_map = {
        (item["name"], normalize_title_for_match(item["title"])): index
        for index, item in enumerate(required_items)
    }
    merged_entries.sort(
        key=lambda item: (
            order_map.get((str(item.get("name", "")).strip(), normalize_title_for_match(str(item.get("title", "")))), 10_000),
            str(item.get("name", "")),
            str(item.get("title", "")),
        )
    )
    return merged_entries, missing_titles


def _scene_name_match_key(text: str) -> str:
    return normalize_text_token(re.sub(r"[【】（）()]", "", str(text or "").strip()))


def _match_scene_requirement(panel_name: str, required_names: list[str], used: set[str]) -> str | None:
    panel_key = _scene_name_match_key(panel_name)
    exact = [name for name in required_names if name not in used and _scene_name_match_key(name) == panel_key]
    if exact:
        return exact[0]
    fuzzy = [
        name
        for name in required_names
        if name not in used and (
            _scene_name_match_key(name) in panel_key or panel_key in _scene_name_match_key(name)
        )
    ]
    if fuzzy:
        fuzzy.sort(key=lambda name: len(_scene_name_match_key(name)))
        return fuzzy[0]
    return None


def validate_scene_grid_coverage(
    grids: list[Mapping[str, Any]],
    scene_requirements: list[Mapping[str, Any]],
    *,
    episode_id: str,
) -> list[str]:
    warnings: list[str] = []
    required_names = [str(item.get("name", "")).strip() for item in scene_requirements if str(item.get("name", "")).strip()]
    panel_names = [
        str(panel.get("scene_name", "")).strip()
        for grid in grids
        for panel in list(grid.get("panels") or [])
        if str(panel.get("scene_name", "")).strip()
    ]

    if not required_names:
        if panel_names:
            warnings.append(f"{episode_id} 没有新增/变体场景，但 scene_grids 仍输出了 {len(panel_names)} 个 panel。")
        return warnings

    if len(panel_names) != len(required_names):
        warnings.append(
            f"{episode_id} 的场景宫格数量不匹配：导演要求 {len(required_names)} 个真实场景，"
            f"当前输出了 {len(panel_names)} 个 panel。请不要为了凑满宫格虚构额外场景。"
        )

    used_required: set[str] = set()
    unmatched_panels: list[str] = []
    for panel_name in panel_names:
        matched_name = _match_scene_requirement(panel_name, required_names, used_required)
        if matched_name is None:
            unmatched_panels.append(panel_name)
            continue
        used_required.add(matched_name)

    missing_required = [name for name in required_names if name not in used_required]
    if unmatched_panels or missing_required:
        warnings.append(
            f"{episode_id} 的场景宫格与导演场景清单不一致：未对齐 panel={unmatched_panels or '无'}；"
            f"缺失场景={missing_required or '无'}。请只输出真实需要的场景，并让 panel 与导演清单一一对应。"
        )
    return warnings


def compact_analysis_for_art(analysis: Mapping[str, Any]) -> dict[str, Any]:
    story_beats = []
    for item in list(analysis.get("story_beats") or [])[:6]:
        story_beats.append(
            {
                "beat_id": item.get("beat_id", ""),
                "title": shorten_text(item.get("title", ""), 50),
                "summary": shorten_text(item.get("summary", ""), 120),
                "characters": list(item.get("characters") or [])[:4],
                "locations": list(item.get("locations") or [])[:4],
                "camera_language": shorten_text(item.get("camera_language", ""), 80),
                "art_direction_cues": [shorten_text(x, 90) for x in list(item.get("art_direction_cues") or [])[:3]],
                "storyboard_value": shorten_text(item.get("storyboard_value", ""), 70),
            }
        )

    characters = []
    for item in list(analysis.get("characters") or [])[:8]:
        if isinstance(item, Mapping):
            characters.append({
                "name": item.get("name", ""),
                "role": item.get("role", ""),
                "core_state": shorten_text(item.get("core_state", item.get("relationship_to_protagonist", "")), 100),
            })
        else:
            characters.append(shorten_text(str(item), 100))

    locations = []
    for item in list(analysis.get("locations") or [])[:8]:
        if isinstance(item, Mapping):
            locations.append({
                "name": item.get("name", ""),
                "mood": shorten_text(item.get("mood", item.get("time_of_day", "")), 90),
            })
        else:
            locations.append(shorten_text(str(item), 90))

    return {
        "episode_id": analysis.get("episode_id", ""),
        "episode_title": analysis.get("episode_title", ""),
        "structure_overview": shorten_text(analysis.get("structure_overview", ""), 180),
        "emotional_curve": shorten_text(analysis.get("emotional_curve", ""), 120),
        "characters": characters,
        "locations": locations,
        "story_beats": story_beats,
        "director_notes": [shorten_text(x, 100) for x in list(analysis.get("director_notes") or [])[:6]],
        "downstream_design_guidance": analysis.get("downstream_design_guidance", {}),
        "bootstrap_source": analysis.get("bootstrap_source", ""),
    }


def compact_series_context_for_art(series_context: Mapping[str, Any]) -> dict[str, Any]:
    active_characters = []
    for item in list(series_context.get("active_characters") or [])[:8]:
        active_characters.append({
            "name": item.get("name", ""),
            "role": item.get("role", ""),
            "state": shorten_text(item.get("latest_state", ""), 100),
        })

    active_locations = []
    for item in list(series_context.get("active_locations") or [])[:8]:
        if isinstance(item, Mapping):
            active_locations.append({
                "name": item.get("name", ""),
                "state": shorten_text(item.get("latest_state", ""), 90),
            })
        else:
            active_locations.append(shorten_text(str(item), 90))

    return {
        "series_name": series_context.get("series_name", ""),
        "premise": shorten_text(series_context.get("premise", ""), 180),
        "genre_profile": series_context.get("genre_profile", {}),
        "downstream_design_guidance": series_context.get("downstream_design_guidance", {}),
        "continuity_rules": [shorten_text(x, 90) for x in list(series_context.get("continuity_rules") or [])[:6]],
        "active_characters": active_characters,
        "active_locations": active_locations,
        "unresolved_threads": [shorten_text(x, 100) for x in list(series_context.get("unresolved_threads") or [])[:6]],
        "recent_timeline": [
            {
                "episode_id": item.get("episode_id", ""),
                "title": item.get("title", ""),
                "synopsis": shorten_text(item.get("synopsis", ""), 140),
            }
            for item in list(series_context.get("recent_timeline") or [])[-3:]
        ],
    }


def compact_genre_reference_bundle_for_art(bundle: Mapping[str, Any]) -> dict[str, Any]:
    aggregate_focus = bundle.get("aggregate_focus") or {}
    matched_packages = []
    for item in list(bundle.get("matched_packages") or [])[:3]:
        matched_packages.append({
            "genre_key": item.get("genre_key", ""),
            "art_focus": [shorten_text(x, 90) for x in list(item.get("art_focus") or [])[:5]],
            "director_focus": [shorten_text(x, 90) for x in list(item.get("director_focus") or [])[:3]],
            "storyboard_focus": [shorten_text(x, 90) for x in list(item.get("storyboard_focus") or [])[:3]],
            "continuity_guardrails": [shorten_text(x, 90) for x in list(item.get("continuity_guardrails") or [])[:4]],
        })

    return {
        "series_name": bundle.get("series_name", ""),
        "target_medium": bundle.get("target_medium", ""),
        "visual_style": shorten_text(bundle.get("visual_style", ""), 140),
        "selected_genres": bundle.get("selected_genres", []),
        "source_notes": [shorten_text(x, 100) for x in list(bundle.get("source_notes") or [])[:4]],
        "aggregate_focus": {
            "art_focus": [shorten_text(x, 90) for x in list(aggregate_focus.get("art_focus") or [])[:8]],
            "director_focus": [shorten_text(x, 90) for x in list(aggregate_focus.get("director_focus") or [])[:4]],
            "storyboard_focus": [shorten_text(x, 90) for x in list(aggregate_focus.get("storyboard_focus") or [])[:4]],
            "continuity_guardrails": [shorten_text(x, 90) for x in list(aggregate_focus.get("continuity_guardrails") or [])[:6]],
            "skill_reference_notes": [shorten_text(x, 90) for x in list(aggregate_focus.get("skill_reference_notes") or [])[:6]],
        },
        "matched_packages": matched_packages,
    }


def build_episode_prompt(
    *,
    config: dict[str, Any],
    series_name: str,
    episode_id: str,
    analysis: Mapping[str, Any],
    series_context: Mapping[str, Any],
    genre_reference_bundle: Mapping[str, Any],
    director_analysis_text: str,
    director_json: Mapping[str, Any],
    existing_character_prompts: str,
    existing_scene_prompts: str,
) -> str:
    director_available = bool(director_json or str(director_analysis_text or "").strip())
    compact_analysis = compact_analysis_context_for_art(analysis, director_available=director_available)
    compact_series_context = ctx_compact.compact_series_context_for_art(series_context)
    compact_genre_reference_bundle = ctx_compact.compact_genre_reference_bundle_for_art(genre_reference_bundle)
    compact_director_text = ctx_compact.compact_director_brief_for_art(director_json, director_analysis_text)
    character_requirements = infer_director_character_requirements(episode_id, director_json)
    scene_requirements = infer_director_scene_requirements(episode_id, director_json)
    director_art_bundle = render_director_art_bundle(
        director_brief=compact_director_text,
        character_requirements=character_requirements,
        scene_requirements=scene_requirements,
    )
    recommended_grid_layout = recommended_scene_grid_layout(len(scene_requirements))
    style = config["quality"].get("visual_style", "").strip()
    medium = config["quality"].get("target_medium", "").strip()
    frame_orientation = normalize_frame_orientation(config.get("quality", {}).get("frame_orientation"))
    frame_composition_guidance = build_frame_composition_guidance(frame_orientation)
    extra_rules = config["quality"].get("extra_rules", [])
    extra_rules_block = ""
    if extra_rules:
        extra_rules_block = "补充要求：\n" + render_bullets(extra_rules)
    return render_prompt(
        "art_assets/draft_user.md",
        {
            "series_name": series_name,
            "episode_id": episode_id,
            "visual_style": style or "未指定，请根据题材与现有分析自行统一风格",
            "target_medium": medium or "高完成度半写实国漫电影动态视频参考图",
            "frame_orientation": frame_orientation,
            "frame_composition_guidance": frame_composition_guidance,
            "extra_rules_block": extra_rules_block,
            "director_art_bundle": director_art_bundle or "<空，未找到时退回 episode_analysis>",
            "analysis_json": json.dumps(compact_analysis, ensure_ascii=False, indent=2),
            "series_context_json": json.dumps(compact_series_context, ensure_ascii=False, indent=2),
            "genre_reference_bundle_json": json.dumps(compact_genre_reference_bundle, ensure_ascii=False, indent=2),
            "recommended_scene_grid_layout": recommended_grid_layout,
            "existing_character_prompts": ctx_compact.compact_episode_scoped_prompt_library(existing_character_prompts, episode_id, limit=900, max_recent_blocks=2) or "<空>",
            "existing_scene_prompts": ctx_compact.compact_episode_scoped_prompt_library(existing_scene_prompts, episode_id, limit=1000, max_recent_blocks=2) or "<空>",
        },
    )


def build_review_prompt(
    *,
    config: dict[str, Any],
    series_name: str,
    episode_id: str,
    analysis: Mapping[str, Any],
    director_json: Mapping[str, Any],
    draft_package: Mapping[str, Any],
) -> str:
    review_skill = load_skill("production/art-direction-review-skill/SKILL.md")
    compliance_skill = load_skill("production/compliance-review-skill/SKILL.md")
    compact_analysis = compact_analysis_context_for_art(analysis, director_available=bool(director_json))
    character_requirements = infer_director_character_requirements(episode_id, director_json)
    scene_requirements = infer_director_scene_requirements(episode_id, director_json)
    compact_director_text = ctx_compact.compact_director_brief_for_art(director_json, "")
    director_art_bundle = render_director_art_bundle(
        director_brief=compact_director_text,
        character_requirements=character_requirements,
        scene_requirements=scene_requirements,
    )
    recommended_grid_layout = recommended_scene_grid_layout(len(scene_requirements))
    style = config["quality"].get("visual_style", "").strip()
    medium = config["quality"].get("target_medium", "").strip()
    frame_orientation = normalize_frame_orientation(config.get("quality", {}).get("frame_orientation"))
    frame_composition_guidance = build_frame_composition_guidance(frame_orientation)
    return render_prompt(
        "art_assets/review_user.md",
        {
            "series_name": series_name,
            "episode_id": episode_id,
            "visual_style": style or "按导演分析统一",
            "target_medium": medium or "高完成度半写实国漫电影动态视频参考图",
            "frame_orientation": frame_orientation,
            "frame_composition_guidance": frame_composition_guidance,
            "analysis_json": json.dumps(compact_analysis, ensure_ascii=False, indent=2),
            "director_art_bundle": director_art_bundle or "<空，未找到导演资产清单>",
            "recommended_scene_grid_layout": recommended_grid_layout,
            "draft_package_json": json.dumps(draft_package, ensure_ascii=False, indent=2),
            "review_skill": ctx_compact.compact_reference_text(review_skill, 1200),
            "compliance_skill": ctx_compact.compact_reference_text(compliance_skill, 520),
        },
    )


def render_character_block(episode_id: str, entries: list[dict[str, Any]]) -> str:
    lines = [f"<!-- episode: {episode_id} start -->", "", f"<!-- {episode_id} -->", ""]
    for item in entries:
        lines.extend(
            [
                f"## {item['title']}",
                "",
                f"**出图要求**：{item['image_requirement']}",
                "",
                "**提示词**：",
                "",
                item["prompt"].strip(),
                "",
                "---",
                "",
            ]
        )
    if not entries:
        lines.extend([f"<!-- {episode_id} 无新增人物或变体 -->", "", "---", "",])
    lines.append(f"<!-- episode: {episode_id} end -->")
    lines.append("")
    return "\n".join(lines)


def render_scene_block(
    episode_id: str,
    grids: list[dict[str, Any]],
    *,
    frame_orientation: str = "9:16竖屏",
) -> str:
    normalized_orientation = normalize_frame_orientation(frame_orientation)
    portrait_canvas = "竖" in normalized_orientation or normalized_orientation.startswith("9:16")
    canvas_requirement = (
        "总画布必须是竖版（高大于宽），禁止横向长条画布；即使是宫格也要保证后续可稳定切出9:16竖向场景参考。"
        if portrait_canvas
        else "总画布必须是横版（宽大于高），不要生成竖向长图。"
    )
    lines = [f"<!-- episode: {episode_id} start -->", "", f"<!-- {episode_id} -->", ""]
    for grid in grids:
        lines.extend(
            [
                f"## {grid['title']}",
                "",
                f"请生成一张 {grid['grid_layout']} 布局的电影场景环境图像。每个格子代表一个独立场景，所有格子必须保持视觉风格统一。请按照以下规范生成。",
                f"总画布约束：{canvas_requirement}",
                "",
                "### 视觉规范",
                f"整体风格：{grid['visual_style']}",
                f"色彩基调：{grid['color_palette']}",
                f"材质质感：{grid['material_texture']}",
                "",
                "### 宫格布局",
                grid["layout_notes"],
                "",
                "### Panel Breakdown（场景拆解）",
                "",
            ]
        )
        for panel in grid.get("panels", []):
            lines.extend(
                [
                    f"{panel['slot']}——【{panel['scene_name']}】",
                    f"视角：{panel['view_prompt']}",
                    f"重点：{panel['focus']}",
                    "",
                ]
            )
        lines.extend(["---", ""])
    if not grids:
        lines.extend([f"<!-- {episode_id} 无新增场景或变体 -->", "", "---", "",])
    lines.append(f"<!-- episode: {episode_id} end -->")
    lines.append("")
    return "\n".join(lines)


def normalize_character_entries(episode_id: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, str]] = []
    for item in entries:
        current = dict(item)
        name = str(current.get("name", "")).strip()
        if ctx_compact.is_mixed_crowd_character_asset(
            name,
            appearance_keywords=str(current.get("prompt", "")).strip(),
            reuse_note=str(current.get("title", "")).strip(),
        ):
            continue
        title = str(current.get("title", "")).strip()
        change_type = str(current.get("change_type", "新增")).strip() or "新增"

        if not title:
            title = name

        if name and name not in title:
            descriptor = re.sub(rf"^{re.escape(episode_id)}\s*", "", title, flags=re.IGNORECASE).strip("：:- ")
            descriptor = re.sub(r"^(新增|变体)\s*[:：-]?\s*", "", descriptor).strip()
            if descriptor and descriptor != name:
                title = f"{name}（{episode_id} {change_type}：{descriptor}）"
            else:
                title = f"{name}（{episode_id} {change_type}）"
        elif name and is_default_debut_descriptor(extract_title_descriptor(name, title), episode_id):
            title = canonical_default_debut_title(name, episode_id)

        current["name"] = name
        current["title"] = title
        current["image_requirement"] = (
            str(current.get("image_requirement", "")).strip()
            or "一张图，左半边面部特写，右半边全身正面、侧面、背面三视图设定图，白色背景"
        )
        key = (name, normalize_character_title_key(name, title))
        existing = normalized_by_key.get(key)
        if existing is None:
            normalized_by_key[key] = current
            ordered_keys.append(key)
            continue

        current_prompt_len = len(str(current.get("prompt", "")).strip())
        existing_prompt_len = len(str(existing.get("prompt", "")).strip())
        current_title_len = len(str(current.get("title", "")).strip())
        existing_title_len = len(str(existing.get("title", "")).strip())
        if (current_prompt_len, current_title_len) > (existing_prompt_len, existing_title_len):
            normalized_by_key[key] = current
    return [normalized_by_key[key] for key in ordered_keys]


def normalize_scene_grids(episode_id: str, grids: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in grids:
        current = dict(item)
        title = str(current.get("title", "")).strip()
        if not title:
            title = f"{episode_id} 场景宫格"
        elif episode_id.lower() not in title.lower():
            title = f"{episode_id} 场景宫格：{title}"
        current["title"] = title
        normalized.append(current)
    return normalized


def write_episode_block(path: Path, header: str, episode_id: str, block: str) -> Path:
    start_marker = f"<!-- episode: {episode_id} start -->"
    end_marker = f"<!-- episode: {episode_id} end -->"
    content = read_text(path)
    if not content.strip():
        new_content = f"{header}\n\n{block}".rstrip() + "\n"
    else:
        pattern = re.compile(
            re.escape(start_marker) + r".*?" + re.escape(end_marker) + r"\n?",
            flags=re.DOTALL,
        )
        if pattern.search(content):
            new_content = pattern.sub(block, content)
        else:
            new_content = content.rstrip() + "\n\n" + block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_content, encoding="utf-8")
    return path


def build_episode_ids(config: dict[str, Any]) -> list[str]:
    series = config["series"]
    prefix = series.get("episode_id_prefix", "ep")
    padding = int(series.get("episode_id_padding", 2))
    start_episode = int(series["start_episode"])
    end_episode = int(series["end_episode"])
    return [f"{prefix}{index:0{padding}d}" for index in range(start_episode, end_episode + 1)]


def art_review_pending_marker_path(assets_dir: Path, episode_id: str) -> Path:
    return assets_dir / f"{episode_id}__art-prompts.review-pending"


def art_review_draft_json_path(assets_dir: Path, episode_id: str, provider_tag: str) -> Path:
    return assets_dir / f"{episode_id}__art-prompts.draft__{provider_tag}.json"


def configure_api(config: dict[str, Any]) -> tuple[str, str, str]:
    selected_provider = config["provider"]["selected_provider"]
    provider_config = config["provider"][selected_provider]
    api_key = (provider_config.get("api_key") or "").strip()
    if not api_key:
        env_key = "OPENAI_API_KEY" if selected_provider == "openai" else "GEMINI_API_KEY"
        api_key = os.getenv(env_key, "").strip()
    if not api_key:
        fallback = load_json(PROJECT_ROOT / "config/video_pipeline.local.json")
        api_key = fallback.get("providers", {}).get(selected_provider, {}).get("api_key", "").strip()
    if not api_key:
        raise RuntimeError(f"缺少 {selected_provider} API key。")
    return selected_provider, provider_config["model"], api_key


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate character and scene prompts into assets/<series>/ from episode analysis.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser


def run_pipeline(config: dict[str, Any], telemetry: TelemetryRecorder | None = None) -> dict[str, Any]:
    selected_provider, model, api_key = configure_api(config)

    script_path = config["script"].get("script_path")
    explicit_series_name = config["series"].get("series_name")
    if not script_path and not explicit_series_name:
        raise RuntimeError("script.script_path 与 series.series_name 至少需要提供一个。")

    paths = build_series_paths(
        project_root=PROJECT_ROOT,
        script_path=script_path,
        series_name=explicit_series_name,
        episode_id=config["script"].get("episode_id"),
    )
    episode_ids = build_episode_ids(config)

    assets_dir = resolve_assets_dir(config, paths.series_name)
    character_path = assets_dir / "character-prompts.md"
    scene_path = assets_dir / "scene-prompts.md"
    series_context_path = paths.analysis_dir / "series_context.json"
    series_context = load_json_file(series_context_path) if series_context_path.exists() else {}
    # 优化方案1：优先使用预计算的compact bundle，避免重复加载和压缩
    genre_reference_bundle = config.get("_precomputed_bundle_cache", {}).get("art") or load_genre_reference_bundle(config, paths.series_name)

    results: list[dict[str, Any]] = []
    timeout_seconds = int(config["run"].get("timeout_seconds", 300))
    temperature = float(config["run"].get("temperature", 0.3))
    enable_review_pass = bool(config["run"].get("enable_review_pass", True))
    dry_run = bool(config["run"].get("dry_run", False))
    provider_tag = build_provider_model_tag(selected_provider, model)

    print_status(f"剧名：{paths.series_name}")
    print_status(f"art assets 输出目录：{assets_dir}")

    if dry_run:
        preview: list[dict[str, Any]] = []
        for episode_id in episode_ids:
            analysis_path = find_analysis_path(config, paths.series_name, episode_id)
            director_analysis_path = find_director_analysis_path(config, paths.series_name, episode_id)
            director_analysis_json_path = find_director_analysis_json_path(config, paths.series_name, episode_id)
            preview.append(
                {
                    "episode_id": episode_id,
                    "analysis_path": str(analysis_path) if analysis_path else None,
                    "director_analysis_path": str(director_analysis_path) if director_analysis_path else None,
                    "director_analysis_json_path": str(director_analysis_json_path) if director_analysis_json_path else None,
                    "analysis_source_mode": "episode_analysis" if analysis_path else "bootstrap_from_director",
                    "character_prompts_path": str(character_path),
                    "scene_prompts_path": str(scene_path),
                }
            )
        payload = {
            "series_name": paths.series_name,
            "provider": selected_provider,
            "model": model,
            "assets_dir": str(assets_dir),
            "episodes": preview,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    for episode_id in episode_ids:
        with telemetry_span(
            telemetry,
            stage="art_assets",
            name="load_art_stage_inputs",
            metadata={"episode_id": episode_id, "series_name": paths.series_name},
        ) as step:
            analysis_path = find_analysis_path(config, paths.series_name, episode_id)
            director_analysis_path = find_director_analysis_path(config, paths.series_name, episode_id)
            director_analysis_json_path = find_director_analysis_json_path(config, paths.series_name, episode_id)
            director_analysis_text = read_text(director_analysis_path) if director_analysis_path else ""
            director_analysis_json = (
                load_json_file(director_analysis_json_path) if director_analysis_json_path and director_analysis_json_path.exists() else {}
            )
            if analysis_path and analysis_path.exists():
                analysis = load_json_file(analysis_path)
                analysis_source_mode = "episode_analysis"
            else:
                analysis = build_bootstrap_analysis(
                    episode_id=episode_id,
                    director_json=director_analysis_json,
                    director_analysis_text=director_analysis_text,
                    series_context=series_context,
                )
                analysis_source_mode = "bootstrap_from_director"
            existing_character_prompts = read_text(character_path)
            existing_scene_prompts = read_text(scene_path)
            compact_analysis = compact_analysis_context_for_art(
                analysis,
                director_available=bool(director_analysis_json or director_analysis_text.strip()),
            )
            compact_series_context = ctx_compact.compact_series_context_for_art(series_context)
            compact_genre_reference_bundle = ctx_compact.compact_genre_reference_bundle_for_art(genre_reference_bundle)
            compact_director_text = ctx_compact.compact_director_brief_for_art(director_analysis_json, director_analysis_text)
            step["metadata"]["analysis_path"] = str(analysis_path) if analysis_path else ""
            step["metadata"]["director_analysis_path"] = str(director_analysis_path) if director_analysis_path else ""
            step["metadata"]["director_analysis_json_path"] = str(director_analysis_json_path) if director_analysis_json_path else ""
            step["metadata"]["analysis_source_mode"] = analysis_source_mode
            step["metadata"]["analysis_story_beats"] = len(analysis.get("story_beats", []))
            step["metadata"]["director_analysis_chars"] = len(director_analysis_text)
            step["metadata"]["director_brief_chars"] = len(compact_director_text)
            step["metadata"]["analysis_chars"] = len(json.dumps(analysis, ensure_ascii=False))
            step["metadata"]["analysis_compact_chars"] = len(json.dumps(compact_analysis, ensure_ascii=False))
            step["metadata"]["series_context_chars"] = len(json.dumps(series_context, ensure_ascii=False))
            step["metadata"]["series_context_compact_chars"] = len(json.dumps(compact_series_context, ensure_ascii=False))
            step["metadata"]["genre_reference_bundle_chars"] = len(json.dumps(genre_reference_bundle, ensure_ascii=False))
            step["metadata"]["genre_reference_bundle_compact_chars"] = len(json.dumps(compact_genre_reference_bundle, ensure_ascii=False))
            step["metadata"]["existing_character_prompt_chars"] = len(existing_character_prompts)
            step["metadata"]["existing_scene_prompt_chars"] = len(existing_scene_prompts)
            draft_cache_path = art_review_draft_json_path(assets_dir, episode_id, provider_tag)
            review_pending_path = art_review_pending_marker_path(assets_dir, episode_id)
            resumable_review = enable_review_pass and draft_cache_path.exists() and review_pending_path.exists()
            step["metadata"]["art_review_draft_json_path"] = str(draft_cache_path)
            step["metadata"]["art_review_pending_path"] = str(review_pending_path)
            step["metadata"]["resumable_review"] = resumable_review
        if analysis_source_mode == "bootstrap_from_director":
            print_status(f"{episode_id} 未找到 episode_analysis，改为基于导演讲戏本进行冷启动服化道生成。")
        if resumable_review:
            print_status(f"检测到 {episode_id} 已有待复审初稿，跳过 draft，直接继续 review 修订。")
            draft_package = load_json_file(draft_cache_path)
        else:
            with telemetry_span(
                telemetry,
                stage="art_assets",
                name="build_art_draft_prompt",
                metadata={"episode_id": episode_id, "provider": selected_provider},
            ) as step:
                prompt = build_episode_prompt(
                    config=config,
                    series_name=paths.series_name,
                    episode_id=episode_id,
                    analysis=analysis,
                    series_context=series_context,
                    genre_reference_bundle=genre_reference_bundle,
                    director_analysis_text=director_analysis_text,
                    director_json=director_analysis_json,
                    existing_character_prompts=existing_character_prompts,
                    existing_scene_prompts=existing_scene_prompts,
                )
                step["metadata"]["prompt_chars"] = len(prompt)

            print_status(f"开始生成 {paths.series_name} {episode_id} 的人物/场景提示词初稿。")
            if director_analysis_path:
                print_status(f"{episode_id} 导演讲戏本：{director_analysis_path}")
            draft_package = model_json_completion(
                selected_provider=selected_provider,
                model=model,
                api_key=api_key,
                system_prompt=load_prompt("art_assets/system.md"),
                prompt=prompt,
                schema=ART_PROMPTS_SCHEMA,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                telemetry=telemetry,
                stage="art_assets",
                step_name="art_draft_model_call",
                metadata={"episode_id": episode_id, "selected_provider": selected_provider},
            )
        package = draft_package
        review_completed = not enable_review_pass
        review_failed = False
        if enable_review_pass:
            save_json_file(draft_cache_path, draft_package)
            review_pending_path.parent.mkdir(parents=True, exist_ok=True)
            review_pending_path.write_text("pending\n", encoding="utf-8")
            print_status(f"开始复审并修订 {paths.series_name} {episode_id} 的人物/场景提示词。")
            try:
                with telemetry_span(
                    telemetry,
                    stage="art_assets",
                    name="build_art_review_prompt",
                    metadata={"episode_id": episode_id, "provider": selected_provider},
                ) as step:
                    review_prompt = build_review_prompt(
                        config=config,
                        series_name=paths.series_name,
                        episode_id=episode_id,
                        analysis=analysis,
                        director_json=director_analysis_json,
                        draft_package=draft_package,
                    )
                    step["metadata"]["prompt_chars"] = len(review_prompt)
                review_patch = model_json_completion(
                    selected_provider=selected_provider,
                    model=model,
                    api_key=api_key,
                    system_prompt=load_prompt("art_assets/review_system.md"),
                    prompt=review_prompt,
                    schema=ART_REVIEW_PATCH_SCHEMA,
                    temperature=max(0.1, min(temperature, 0.2)),
                    timeout_seconds=timeout_seconds,
                    telemetry=telemetry,
                    stage="art_assets",
                    step_name="art_review_model_call",
                    metadata={"episode_id": episode_id, "selected_provider": selected_provider},
                )
                package = merge_art_review_patch(draft_package, review_patch)
                review_completed = True
            except Exception as exc:
                review_failed = True
                package = draft_package
                print_status(
                    f"{episode_id} 的服化道 review 失败，将先落盘 draft 结果继续后续流程：{exc}"
                )

        with telemetry_span(
            telemetry,
            stage="art_assets",
            name="normalize_and_save_art_outputs",
            metadata={
                "episode_id": episode_id,
                "character_prompts_path": str(character_path),
                "scene_prompts_path": str(scene_path),
            },
        ) as step:
            scene_requirements = infer_director_scene_requirements(episode_id, director_analysis_json)
            package["character_entries"], missing_character_titles = ensure_character_entry_coverage(
                episode_id=episode_id,
                target_medium=config["quality"].get("target_medium", ""),
                visual_style=config["quality"].get("visual_style", ""),
                package_entries=package.get("character_entries", []),
                director_json=director_analysis_json,
            )
            package["character_entries"] = normalize_character_entries(episode_id, package.get("character_entries", []))
            package["scene_grids"] = normalize_scene_grids(episode_id, package.get("scene_grids", []))
            scene_grid_warnings = validate_scene_grid_coverage(
                package.get("scene_grids", []), scene_requirements, episode_id=episode_id
            )
            if scene_grid_warnings:
                step["metadata"]["quality_warning_count"] = len(scene_grid_warnings)
                step["metadata"]["quality_warnings"] = scene_grid_warnings[:20]
                if telemetry is not None:
                    telemetry.context["art_quality_warning_count"] = len(scene_grid_warnings)
                    telemetry.context["art_quality_warnings"] = scene_grid_warnings[:20]
                print_status(
                    f"{episode_id} 的场景宫格检查提示 {len(scene_grid_warnings)} 条：结果将继续保存，不再因宫格覆盖阈值中断。"
                )

            character_block = render_character_block(episode_id, package.get("character_entries", []))
            scene_block = render_scene_block(
                episode_id,
                package.get("scene_grids", []),
                frame_orientation=config.get("quality", {}).get("frame_orientation", "9:16竖屏"),
            )
            write_episode_block(character_path, "# 人物提示词", episode_id, character_block)
            write_episode_block(scene_path, "# 场景道具提示词", episode_id, scene_block)
            if not review_failed:
                draft_cache_path.unlink(missing_ok=True)
                review_pending_path.unlink(missing_ok=True)
            step["metadata"]["character_entries"] = len(package.get("character_entries", []))
            step["metadata"]["scene_grids"] = len(package.get("scene_grids", []))
            step["metadata"]["required_scene_count"] = len(scene_requirements)
            step["metadata"]["director_coverage_backfill_count"] = len(missing_character_titles)
            step["metadata"]["review_completed"] = review_completed
            step["metadata"]["review_failed"] = review_failed
            if missing_character_titles:
                step["metadata"]["director_coverage_backfill_titles"] = "、".join(missing_character_titles)

        result_item = {
            "episode_id": episode_id,
            "analysis_path": str(analysis_path) if analysis_path else None,
            "director_analysis_path": str(director_analysis_path) if director_analysis_path else None,
            "analysis_source_mode": analysis_source_mode,
            "character_entries": len(package.get("character_entries", [])),
            "scene_grids": len(package.get("scene_grids", [])),
            "director_coverage_backfill_titles": missing_character_titles,
            "review_pass_enabled": enable_review_pass,
            "review_completed": review_completed,
            "review_failed": review_failed,
            "generated_at": utc_timestamp(),
        }
        results.append(result_item)
        if missing_character_titles:
            print_status(f"{episode_id} 已按导演讲戏本自动补齐人物/造型：{'、'.join(missing_character_titles)}")
        print_status(f"{episode_id} 完成：人物 {result_item['character_entries']} 条，场景宫格 {result_item['scene_grids']} 条。")

    summary = {
        "series_name": paths.series_name,
        "provider": selected_provider,
        "model": model,
        "character_prompts_path": str(character_path.resolve()),
        "scene_prompts_path": str(scene_path.resolve()),
        "results": results,
    }
    summary_path = assets_dir / "art-assets-summary.json"
    save_json_file(summary_path, summary)
    print_status(f"人物/场景提示词已写入：{summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    args = build_arg_parser().parse_args()
    print_status(f"加载配置：{args.config}")
    config = load_runtime_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
