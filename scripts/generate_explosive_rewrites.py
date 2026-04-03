from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from genre_routing import load_genre_package_map
from pipeline_telemetry import TelemetryRecorder, apply_provider_usage, telemetry_span
from providers.base import (
    build_provider_model_tag,
    extract_json_from_text,
    load_json_file,
    save_json_file,
    save_text_file,
    utc_timestamp,
)
from prompt_utils import load_prompt, render_bullets, render_prompt
from skill_utils import load_skill
from series_paths import infer_episode_id_from_name


DEFAULT_CONFIG_PATH = Path("config/explosive_rewrite_pipeline.local.json")
SCRIPT_FILENAME_PATTERN = re.compile(
    r"^(?P<episode>ep\d+)(?:__(?P<provider>[^_]+)__(?P<model>.+?))?\.md$",
    re.IGNORECASE,
)

EPISODE_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "episode_id",
        "episode_title",
        "style_target_label",
        "overall_winner",
        "variants",
        "genre_reference_notes",
        "style_shift_strategy",
        "comparative_takeaways",
        "rewrite_blueprint",
    ],
    "properties": {
        "episode_id": {"type": "string"},
        "episode_title": {"type": "string"},
        "style_target_label": {"type": "string"},
        "overall_winner": {"type": "string"},
        "variants": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "source_path",
                    "label",
                    "provider",
                    "model",
                    "explosive_score",
                    "hook_score",
                    "conflict_score",
                    "emotion_score",
                    "pace_score",
                    "character_pull_score",
                    "cliffhanger_score",
                    "target_style_fit_score",
                    "strengths",
                    "weaknesses",
                    "viral_points",
                    "target_style_fit_notes",
                    "rewrite_priority",
                ],
                "properties": {
                    "source_path": {"type": "string"},
                    "label": {"type": "string"},
                    "provider": {"type": "string"},
                    "model": {"type": "string"},
                    "explosive_score": {"type": "integer"},
                    "hook_score": {"type": "integer"},
                    "conflict_score": {"type": "integer"},
                    "emotion_score": {"type": "integer"},
                    "pace_score": {"type": "integer"},
                    "character_pull_score": {"type": "integer"},
                    "cliffhanger_score": {"type": "integer"},
                    "target_style_fit_score": {"type": "integer"},
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "weaknesses": {"type": "array", "items": {"type": "string"}},
                    "viral_points": {"type": "array", "items": {"type": "string"}},
                    "target_style_fit_notes": {"type": "array", "items": {"type": "string"}},
                    "rewrite_priority": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "genre_reference_notes": {"type": "array", "items": {"type": "string"}},
        "style_shift_strategy": {"type": "array", "items": {"type": "string"}},
        "comparative_takeaways": {"type": "array", "items": {"type": "string"}},
        "rewrite_blueprint": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "must_keep",
                "must_upgrade",
                "opening_hook_upgrades",
                "line_polish_rules",
                "cliffhanger_target",
                "forbidden_moves",
            ],
            "properties": {
                "must_keep": {"type": "array", "items": {"type": "string"}},
                "must_upgrade": {"type": "array", "items": {"type": "string"}},
                "opening_hook_upgrades": {"type": "array", "items": {"type": "string"}},
                "line_polish_rules": {"type": "array", "items": {"type": "string"}},
                "cliffhanger_target": {"type": "string"},
                "forbidden_moves": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}

EPISODE_REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "episode_id",
        "revised_title",
        "style_target_label",
        "predicted_explosive_score",
        "score_delta",
        "top_improvements",
        "style_adaptation_notes",
        "applied_plan",
        "explosive_insertions",
        "change_log",
        "revised_script_markdown",
    ],
    "properties": {
        "episode_id": {"type": "string"},
        "revised_title": {"type": "string"},
        "style_target_label": {"type": "string"},
        "predicted_explosive_score": {"type": "integer"},
        "score_delta": {"type": "integer"},
        "top_improvements": {"type": "array", "items": {"type": "string"}},
        "style_adaptation_notes": {"type": "array", "items": {"type": "string"}},
        "applied_plan": {"type": "array", "items": {"type": "string"}},
        "explosive_insertions": {"type": "array", "items": {"type": "string"}},
        "change_log": {"type": "array", "items": {"type": "string"}},
        "revised_script_markdown": {"type": "string"},
    },
}

SERIES_PLAYBOOK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "series_name",
        "audience_positioning",
        "style_target_label",
        "core_hook_formula",
        "recurring_strengths",
        "recurring_weaknesses",
        "viral_must_haves",
        "explosive_rules",
        "opening_patterns",
        "cliffhanger_patterns",
        "dialogue_upgrade_rules",
    ],
    "properties": {
        "series_name": {"type": "string"},
        "audience_positioning": {"type": "string"},
        "style_target_label": {"type": "string"},
        "core_hook_formula": {"type": "array", "items": {"type": "string"}},
        "recurring_strengths": {"type": "array", "items": {"type": "string"}},
        "recurring_weaknesses": {"type": "array", "items": {"type": "string"}},
        "viral_must_haves": {"type": "array", "items": {"type": "string"}},
        "explosive_rules": {"type": "array", "items": {"type": "string"}},
        "opening_patterns": {"type": "array", "items": {"type": "string"}},
        "cliffhanger_patterns": {"type": "array", "items": {"type": "string"}},
        "dialogue_upgrade_rules": {"type": "array", "items": {"type": "string"}},
    },
}

SERIES_NARRATIVE_CONTEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "series_name",
        "overall_story_tone",
        "global_story_summary",
        "protagonist_core_drive",
        "core_relationship_axes",
        "continuity_guardrails",
        "narrative_do_not_break",
        "episode_cards",
    ],
    "properties": {
        "series_name": {"type": "string"},
        "overall_story_tone": {"type": "array", "items": {"type": "string"}},
        "global_story_summary": {"type": "string"},
        "protagonist_core_drive": {"type": "array", "items": {"type": "string"}},
        "core_relationship_axes": {"type": "array", "items": {"type": "string"}},
        "continuity_guardrails": {"type": "array", "items": {"type": "string"}},
        "narrative_do_not_break": {"type": "array", "items": {"type": "string"}},
        "episode_cards": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "episode_id",
                    "title",
                    "role_in_arc",
                    "plot_summary",
                    "opening_state",
                    "closing_state",
                    "bridge_from_previous",
                    "bridge_to_next",
                    "must_preserve",
                    "continuity_risks",
                ],
                "properties": {
                    "episode_id": {"type": "string"},
                    "title": {"type": "string"},
                    "role_in_arc": {"type": "string"},
                    "plot_summary": {"type": "string"},
                    "opening_state": {"type": "array", "items": {"type": "string"}},
                    "closing_state": {"type": "array", "items": {"type": "string"}},
                    "bridge_from_previous": {"type": "array", "items": {"type": "string"}},
                    "bridge_to_next": {"type": "array", "items": {"type": "string"}},
                    "must_preserve": {"type": "array", "items": {"type": "string"}},
                    "continuity_risks": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


def print_status(message: str) -> None:
    print(f"[explosive-rewrite] {message}", flush=True)


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        item = str(raw or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def sanitize_filename_component(value: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "-", str(value or "").strip())
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    cleaned = cleaned.strip("-_.")
    return cleaned or "default"


def trim_text(text: str, limit: int) -> str:
    content = str(text or "").strip()
    if limit <= 0 or len(content) <= limit:
        return content
    return content[: max(0, limit - 1)].rstrip() + "…"


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json_file(path)


def resolve_style_target(config: dict[str, Any]) -> dict[str, Any]:
    style_config = dict(config.get("style_target", {}))
    genre_keys = unique_strings(list(style_config.get("genre_keys", [])))
    custom_tokens = unique_strings(list(style_config.get("custom_style_tokens", [])))
    style_label = str(style_config.get("style_label") or "").strip()
    if not style_label:
        style_label = " / ".join(genre_keys or custom_tokens)
    style_slug = sanitize_filename_component("-".join(genre_keys or custom_tokens or [style_label]))
    return {
        "genre_keys": genre_keys,
        "custom_style_tokens": custom_tokens,
        "style_label": style_label,
        "style_slug": style_slug if (genre_keys or custom_tokens or style_label) else "",
        "include_current_series_genres": bool(style_config.get("include_current_series_genres", True)),
        "reference_limit": int(style_config.get("prompt_genre_reference_limit", 3)),
    }


def resolve_genre_packages(genre_tokens: list[str]) -> list[dict[str, Any]]:
    package_map = load_genre_package_map()
    if not genre_tokens:
        return []
    lowered_tokens = {token.lower(): token for token in genre_tokens if token}
    matched: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for package in package_map.values():
        genre_key = str(package.get("genre_key", "")).strip()
        aliases = [str(item).strip() for item in package.get("aliases", [])]
        normalized = {genre_key.lower(), *(item.lower() for item in aliases if item)}
        if normalized & set(lowered_tokens.keys()):
            if genre_key and genre_key not in seen_keys:
                matched.append(package)
                seen_keys.add(genre_key)
    return matched


def load_current_series_genre_context(series_name: str) -> dict[str, Any]:
    analysis_root = (PROJECT_ROOT / "analysis" / series_name).resolve()
    series_context = read_json_if_exists(analysis_root / "series_context.json")
    genre_profile = dict(series_context.get("genre_profile", {}))
    route_tokens = unique_strings(
        [
            str(genre_profile.get("primary_genre", "")).strip(),
            *[str(item).strip() for item in genre_profile.get("secondary_genres", [])],
            *[
                str(item.get("genre_key", "")).strip()
                for item in series_context.get("genre_playbooks", [])
                if isinstance(item, dict)
            ],
        ]
    )
    matched_packages = resolve_genre_packages(route_tokens)
    return {
        "series_context_path": str((analysis_root / "series_context.json").resolve()),
        "genre_profile": genre_profile,
        "matched_packages": matched_packages,
        "route_tokens": route_tokens,
    }


def summarize_genre_packages(packages: list[dict[str, Any]], *, limit: int) -> str:
    if not packages:
        return "<空>"
    summary: list[dict[str, Any]] = []
    for package in packages[:limit]:
        summary.append(
            {
                "genre_key": package.get("genre_key", ""),
                "core_audience_promises": package.get("core_audience_promises", [])[:4],
                "script_hooks": package.get("script_hooks", [])[:5],
                "character_design_focus": package.get("character_design_focus", [])[:4],
                "scene_design_focus": package.get("scene_design_focus", [])[:4],
                "storyboard_focus": package.get("storyboard_focus", [])[:4],
            }
        )
    return json.dumps(summary, ensure_ascii=False, indent=2)


def summarize_genre_skills(packages: list[dict[str, Any]], *, limit_chars: int = 9000) -> str:
    parts: list[str] = []
    for package in packages:
        skill_path = str(package.get("_skill_path", "")).strip()
        if not skill_path:
            continue
        text = read_text_if_exists(Path(skill_path))
        if not text:
            continue
        parts.append(text.strip())
    combined = "\n\n".join(part for part in parts if part).strip()
    return trim_text(combined, limit_chars) if combined else "<空>"


def _normalized_text_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def build_genre_reference_bundle_text(
    *,
    current_playbook_text: str,
    current_skill_text: str,
    target_playbook_text: str,
    target_skill_text: str,
    playbook_text: str,
    limit_chars: int = 12000,
) -> str:
    sections: list[str] = []
    seen: set[str] = set()

    def append_section(title: str, text: str) -> None:
        content = str(text or "").strip()
        if not content or content == "<空>":
            return
        normalized = _normalized_text_key(content)
        if normalized in seen:
            return
        seen.add(normalized)
        sections.extend([f"## {title}", "", content, ""])

    if _normalized_text_key(current_playbook_text) and _normalized_text_key(current_playbook_text) == _normalized_text_key(target_playbook_text):
        append_section("当前剧 / 目标风格共享题材经验参考", current_playbook_text)
    else:
        append_section("当前剧题材经验参考", current_playbook_text)
        append_section("目标风格题材经验参考", target_playbook_text)

    if _normalized_text_key(current_skill_text) and _normalized_text_key(current_skill_text) == _normalized_text_key(target_skill_text):
        append_section("当前剧 / 目标风格共享题材 Skill 参考", current_skill_text)
    else:
        append_section("当前剧题材 Skill 参考", current_skill_text)
        append_section("目标风格题材 Skill 参考", target_skill_text)

    append_section("本剧专属爆款经验", playbook_text)
    rendered = "\n".join(sections).strip()
    return trim_text(rendered, limit_chars) if rendered else "<空>"


def build_target_style_reference(config: dict[str, Any], series_name: str) -> dict[str, Any]:
    style_target = resolve_style_target(config)
    target_packages = resolve_genre_packages(style_target["genre_keys"])
    current_series_context = load_current_series_genre_context(series_name) if style_target["include_current_series_genres"] else {
        "series_context_path": "",
        "genre_profile": {},
        "matched_packages": [],
        "route_tokens": [],
    }
    style_goal_lines: list[str] = []
    if style_target["genre_keys"]:
        style_goal_lines.append("本次目标爆改题材：{}".format("、".join(style_target["genre_keys"])))
    if style_target["custom_style_tokens"]:
        style_goal_lines.append("本次附加自定义风格词：{}".format("、".join(style_target["custom_style_tokens"])))
    if current_series_context["route_tokens"]:
        style_goal_lines.append("当前剧现有题材参考：{}".format("、".join(current_series_context["route_tokens"])))
    if not style_goal_lines:
        style_goal_lines.append("本次未指定额外题材改稿方向，以通用爆款优化为主。")
    target_playbook_text = summarize_genre_packages(target_packages, limit=style_target["reference_limit"])
    target_skill_text = summarize_genre_skills(target_packages)
    current_playbook_text = summarize_genre_packages(
        current_series_context.get("matched_packages", []),
        limit=style_target["reference_limit"],
    )
    current_skill_text = summarize_genre_skills(current_series_context.get("matched_packages", []))
    return {
        "style_target": style_target,
        "target_packages": target_packages,
        "current_series_context": current_series_context,
        "target_playbook_text": target_playbook_text,
        "target_skill_text": target_skill_text,
        "current_playbook_text": current_playbook_text,
        "current_skill_text": current_skill_text,
        "style_goal_text": "\n".join(f"- {item}" for item in style_goal_lines),
    }


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


def parse_bool_flag(raw: str | None) -> bool | None:
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"无法解析布尔值：{raw}")


def request_json(
    *,
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, method="POST")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"模型请求失败，状态码 {exc.code}，响应：{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"模型网络请求失败：{exc}") from exc


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


def openai_json_completion(
    *,
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: Mapping[str, Any],
    temperature: float,
    timeout_seconds: int,
    telemetry: TelemetryRecorder | None = None,
    stage: str = "explosive_rewrite",
    step_name: str = "explosive_model_call",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
                        "text": user_prompt,
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
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
        step["metadata"]["temperature"] = temperature
        step["metadata"]["schema_name"] = schema_name
    return extract_json_from_text(extract_openai_text(response))


def configure_openai(config: dict[str, Any]) -> tuple[str, str]:
    model = config["provider"]["openai"]["model"]
    api_key = (config["provider"]["openai"].get("api_key") or "").strip()
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        fallback = load_json(PROJECT_ROOT / "config/video_pipeline.local.json")
        api_key = fallback.get("providers", {}).get("openai", {}).get("api_key", "").strip()
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY。")
    return model, api_key


def episode_id_sequence(config: dict[str, Any]) -> list[str]:
    series = config["series"]
    prefix = series.get("episode_id_prefix", "ep")
    padding = int(series.get("episode_id_padding", 2))
    start_episode = int(series["start_episode"])
    end_episode = int(series["end_episode"])
    return [f"{prefix}{index:0{padding}d}" for index in range(start_episode, end_episode + 1)]


def resolve_series_name(config: dict[str, Any]) -> str:
    explicit = (config["series"].get("series_name") or "").strip()
    if explicit:
        return explicit
    script_dir = Path(config["script"]["series_dir"]).expanduser().resolve()
    return script_dir.name


def parse_script_filename(path: Path) -> tuple[str | None, str, str]:
    match = SCRIPT_FILENAME_PATTERN.match(path.name)
    if not match:
        episode_id = infer_episode_id_from_name(path.name)
        return episode_id, "unknown", "unknown"
    return (
        match.group("episode").lower(),
        (match.group("provider") or "unknown").strip(),
        (match.group("model") or "unknown").strip(),
    )


def is_explosive_variant(path: Path, config: Mapping[str, Any]) -> bool:
    suffix = str(config.get("output", {}).get("rewrite_filename_suffix", "__explosive")).strip() or "__explosive"
    return suffix in path.stem


def base_variant_rank(item: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[int, int, int, str]:
    source = config.get("source", {})
    preferred_patterns = [str(value).strip() for value in source.get("preferred_base_variant_patterns", []) if str(value).strip()]
    filename = str(item.get("filename", ""))
    provider = str(item.get("provider", "")).strip().lower()
    model = str(item.get("model", "")).strip().lower()
    provider_model = f"{provider}/{model}"
    matched_index = len(preferred_patterns) + 1
    for index, pattern in enumerate(preferred_patterns):
        if pattern in filename or pattern.lower() == provider_model:
            matched_index = index
            break
    segment_count = len(Path(filename).stem.split("__"))
    is_openai = 0 if provider == "openai" else 1
    return (matched_index, segment_count, is_openai, filename)


def select_base_variants(items: list[dict[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = config.get("source", {})
    selection_mode = str(source.get("base_variant_selection_mode", "single")).strip().lower() or "single"
    if selection_mode in {"all", "multi", "multiple"}:
        return items
    if not items:
        return []
    ranked = sorted(items, key=lambda item: base_variant_rank(item, config))
    return [ranked[0]]


def collect_episode_variants(config: dict[str, Any], target_episode_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    series_dir = Path(config["script"]["series_dir"]).expanduser().resolve()
    if not series_dir.exists():
        raise FileNotFoundError(f"剧本目录不存在：{series_dir}")

    grouped_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(series_dir.glob("*.md")):
        episode_id, provider, model = parse_script_filename(path)
        if not episode_id or episode_id not in target_episode_ids:
            continue
        grouped_all[episode_id].append(
            {
                "path": str(path.resolve()),
                "filename": path.name,
                "provider": provider,
                "model": model,
                "label": f"{provider}/{model}",
                "is_explosive_variant": is_explosive_variant(path, config),
                "text": path.read_text(encoding="utf-8"),
            }
        )
    include_existing_explosive = bool(config.get("source", {}).get("include_existing_explosive_variants", False))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode_id, items in grouped_all.items():
        if include_existing_explosive:
            grouped[episode_id] = items
            continue
        base_items = [item for item in items if not bool(item.get("is_explosive_variant", False))]
        grouped[episode_id] = select_base_variants(base_items or items, config)
    return grouped


def current_analysis_root(config: Mapping[str, Any], series_name: str) -> Path:
    return (PROJECT_ROOT / "analysis" / series_name / str(config.get("output", {}).get("analysis_dir_name", "explosive-actor-gpt"))).resolve()


def series_narrative_context_paths(config: Mapping[str, Any], series_name: str, model: str) -> tuple[Path, Path]:
    root = current_analysis_root(config, series_name)
    json_path = root / f"series_narrative_context__openai__{model}.json"
    md_path = root / f"series_narrative_context__openai__{model}.md"
    return json_path, md_path


def collect_series_context_sources(
    series_name: str,
    episode_ids: list[str],
    grouped: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    analysis_series_root = (PROJECT_ROOT / "analysis" / series_name).resolve()
    episode_summary_root = analysis_series_root / "episode_summaries"
    episode_summaries: list[dict[str, Any]] = []
    missing_episode_summaries: list[str] = []
    for episode_id in episode_ids:
        summary_path = episode_summary_root / f"{episode_id}.json"
        if summary_path.exists():
            episode_summaries.append(load_json_file(summary_path))
        else:
            missing_episode_summaries.append(episode_id)
    fallback_script_glimpses: list[dict[str, Any]] = []
    for episode_id in missing_episode_summaries:
        variants = grouped.get(episode_id, [])
        if not variants:
            continue
        primary = variants[-1]
        fallback_script_glimpses.append(
            {
                "episode_id": episode_id,
                "source_path": primary["path"],
                "script_excerpt": trim_text(primary["text"], 2400),
            }
        )
    return {
        "series_context": read_json_if_exists(analysis_series_root / "series_context.json"),
        "series_bible": read_json_if_exists(analysis_series_root / "series_bible.json"),
        "episode_summaries": episode_summaries,
        "fallback_script_glimpses": fallback_script_glimpses,
    }


def read_supporting_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def build_episode_analysis_prompt(
    *,
    config: dict[str, Any],
    series_name: str,
    episode_id: str,
    variants: list[dict[str, Any]],
    playbook_text: str,
    style_reference: Mapping[str, Any],
    episode_context_bundle: Mapping[str, str],
) -> str:
    skill_text = load_skill("production/explosive-screenwriter-skill/SKILL.md")
    hook_playbook = read_supporting_text(
        PROJECT_ROOT / "skills/production/explosive-screenwriter-skill/playbooks/hit-hook-playbook.md"
    )
    analysis_template = read_supporting_text(
        PROJECT_ROOT / "skills/production/explosive-screenwriter-skill/templates/explosive-analysis-template.md"
    )
    audience = config["quality"].get("target_audience", "").strip()
    extra_rules = config["quality"].get("extra_rules", [])
    extra_rules_block = ""
    if extra_rules:
        extra_rules_block = "补充要求：\n" + render_bullets(extra_rules)
    variant_blocks: list[str] = []
    for item in variants:
        variant_blocks.extend(
            [
                f"### 版本：{item['label']}",
                f"路径：{item['path']}",
                item["text"],
                "",
            ]
        )
    genre_reference_bundle_text = build_genre_reference_bundle_text(
        current_playbook_text=str(style_reference.get("current_playbook_text") or ""),
        current_skill_text=str(style_reference.get("current_skill_text") or ""),
        target_playbook_text=str(style_reference.get("target_playbook_text") or ""),
        target_skill_text=str(style_reference.get("target_skill_text") or ""),
        playbook_text=playbook_text,
    )
    return render_prompt(
        "explosive_rewrite/episode_analysis.md",
        {
            "series_name": series_name,
            "episode_id": episode_id,
            "target_audience": audience or "女频古言短剧/漫剧用户",
            "extra_rules_block": extra_rules_block,
            "style_target_label": str(style_reference["style_target"].get("style_label") or "未指定"),
            "style_goal_text": style_reference.get("style_goal_text", "<空>"),
            "genre_reference_bundle_text": genre_reference_bundle_text,
            "series_narrative_context_text": episode_context_bundle.get("series_narrative_context_text", "<空>"),
            "previous_episode_context_text": episode_context_bundle.get("previous_episode_context_text", "<空>"),
            "current_episode_context_text": episode_context_bundle.get("current_episode_context_text", "<空>"),
            "next_episode_context_text": episode_context_bundle.get("next_episode_context_text", "<空>"),
            "skill_text": skill_text[:7000],
            "hook_playbook": hook_playbook[:4000],
            "analysis_template": analysis_template[:3000],
            "variants_block": "\n".join(variant_blocks).strip(),
        },
    )


def build_episode_rewrite_prompt(
    *,
    config: dict[str, Any],
    series_name: str,
    episode_id: str,
    variants: list[dict[str, Any]],
    analysis_result: Mapping[str, Any],
    playbook_text: str,
    style_reference: Mapping[str, Any],
    episode_context_bundle: Mapping[str, str],
) -> str:
    skill_text = load_skill("production/explosive-screenwriter-skill/SKILL.md")
    rewrite_template = read_supporting_text(
        PROJECT_ROOT / "skills/production/explosive-screenwriter-skill/templates/explosive-rewrite-template.md"
    )
    audience = config["quality"].get("target_audience", "").strip()
    extra_rules = config["quality"].get("extra_rules", [])
    extra_rules_block = ""
    if extra_rules:
        extra_rules_block = "补充要求：\n" + render_bullets(extra_rules)
    variant_blocks: list[str] = []
    for item in variants:
        variant_blocks.extend(
            [
                f"### 版本：{item['label']}",
                item["text"],
                "",
            ]
        )
    genre_reference_bundle_text = build_genre_reference_bundle_text(
        current_playbook_text=str(style_reference.get("current_playbook_text") or ""),
        current_skill_text=str(style_reference.get("current_skill_text") or ""),
        target_playbook_text=str(style_reference.get("target_playbook_text") or ""),
        target_skill_text=str(style_reference.get("target_skill_text") or ""),
        playbook_text=playbook_text,
    )
    return render_prompt(
        "explosive_rewrite/episode_rewrite.md",
        {
            "series_name": series_name,
            "episode_id": episode_id,
            "target_audience": audience or "女频古言短剧/漫剧用户",
            "extra_rules_block": extra_rules_block,
            "style_target_label": str(style_reference["style_target"].get("style_label") or "未指定"),
            "style_goal_text": style_reference.get("style_goal_text", "<空>"),
            "genre_reference_bundle_text": genre_reference_bundle_text,
            "series_narrative_context_text": episode_context_bundle.get("series_narrative_context_text", "<空>"),
            "previous_episode_context_text": episode_context_bundle.get("previous_episode_context_text", "<空>"),
            "current_episode_context_text": episode_context_bundle.get("current_episode_context_text", "<空>"),
            "next_episode_context_text": episode_context_bundle.get("next_episode_context_text", "<空>"),
            "skill_text": skill_text[:7000],
            "rewrite_template": rewrite_template[:3000],
            "analysis_result_json": json.dumps(analysis_result, ensure_ascii=False, indent=2),
            "variants_block": "\n".join(variant_blocks).strip(),
        },
    )


def build_series_playbook_prompt(
    *,
    config: dict[str, Any],
    series_name: str,
    episode_reports: list[Mapping[str, Any]],
    style_reference: Mapping[str, Any],
) -> str:
    skill_text = load_skill("production/explosive-screenwriter-skill/SKILL.md")
    audience = config["quality"].get("target_audience", "").strip()
    return render_prompt(
        "explosive_rewrite/series_playbook.md",
        {
            "series_name": series_name,
            "target_audience": audience or "女频古言短剧/漫剧用户",
            "style_target_label": str(style_reference["style_target"].get("style_label") or "未指定"),
            "style_goal_text": style_reference.get("style_goal_text", "<空>"),
            "target_genre_playbook_text": style_reference.get("target_playbook_text", "<空>"),
            "target_genre_skill_text": style_reference.get("target_skill_text", "<空>"),
            "skill_text": skill_text[:6000],
            "episode_reports_json": json.dumps(episode_reports, ensure_ascii=False, indent=2),
        },
    )


def build_series_narrative_context_prompt(
    *,
    config: dict[str, Any],
    series_name: str,
    episode_ids: list[str],
    context_sources: Mapping[str, Any],
) -> str:
    skill_text = load_skill("production/explosive-screenwriter-skill/SKILL.md")
    audience = config["quality"].get("target_audience", "").strip()
    return render_prompt(
        "explosive_rewrite/series_narrative_context.md",
        {
            "series_name": series_name,
            "target_audience": audience or "女频古言短剧/漫剧用户",
            "episode_ids_text": "、".join(episode_ids),
            "skill_text": trim_text(skill_text, 6000),
            "series_context_json": json.dumps(context_sources.get("series_context", {}), ensure_ascii=False, indent=2),
            "series_bible_json": json.dumps(context_sources.get("series_bible", {}), ensure_ascii=False, indent=2),
            "episode_summaries_json": json.dumps(context_sources.get("episode_summaries", []), ensure_ascii=False, indent=2),
            "fallback_script_glimpses_json": json.dumps(context_sources.get("fallback_script_glimpses", []), ensure_ascii=False, indent=2),
        },
    )


def render_episode_report_markdown(
    analysis_data: Mapping[str, Any],
    rewrite_data: Mapping[str, Any],
    rewritten_script_path: Path,
) -> str:
    lines = [
        "# 单集爆款评分与改稿报告",
        "",
        f"- 集数：{analysis_data['episode_id']}",
        f"- 标题：{analysis_data['episode_title']}",
        f"- 目标风格：{analysis_data.get('style_target_label', '未指定')}",
        f"- 当前最优版本：{analysis_data['overall_winner']}",
        f"- 强化版预测爆款分：{rewrite_data['predicted_explosive_score']}",
        f"- 提升幅度：+{rewrite_data['score_delta']}",
        f"- 强化版剧本：{rewritten_script_path}",
        "",
        "## 当前剧本评分",
        "",
    ]
    for item in analysis_data.get("variants", []):
        lines.extend(
            [
                f"### {item['label']}",
                f"- 来源：{item['source_path']}",
                f"- 爆款总分：{item['explosive_score']}",
                f"- 开篇钩子：{item['hook_score']}",
                f"- 冲突强度：{item['conflict_score']}",
                f"- 情绪拉力：{item['emotion_score']}",
                f"- 节奏效率：{item['pace_score']}",
                f"- 人物拉力：{item['character_pull_score']}",
                f"- 结尾卡点：{item['cliffhanger_score']}",
                f"- 目标风格匹配：{item.get('target_style_fit_score', 0)}",
                "- 优点：",
                *[f"  - {point}" for point in item.get("strengths", [])],
                "- 问题：",
                *[f"  - {point}" for point in item.get("weaknesses", [])],
                "- 可放大的爆点：",
                *[f"  - {point}" for point in item.get("viral_points", [])],
                "- 目标风格改造机会：",
                *[f"  - {point}" for point in item.get("target_style_fit_notes", [])],
                "",
            ]
        )
    lines.extend(["## 题材与知识库参考", ""])
    lines.extend([f"- {item}" for item in analysis_data.get("genre_reference_notes", [])])
    lines.extend(["", "## 目标风格改造策略", ""])
    lines.extend([f"- {item}" for item in analysis_data.get("style_shift_strategy", [])])
    lines.extend(["## 当前版本哪里好", ""])
    for item in analysis_data.get("variants", []):
        lines.append(f"### {item['label']}")
        lines.extend([f"- {point}" for point in item.get("strengths", [])[:5]])
        lines.append("")

    lines.extend(["## 当前版本哪里还不够好", ""])
    for item in analysis_data.get("variants", []):
        lines.append(f"### {item['label']}")
        lines.extend([f"- {point}" for point in item.get("weaknesses", [])[:5]])
        lines.append("")

    lines.extend(["## 综合判断", ""])
    lines.extend([f"- {item}" for item in analysis_data.get("comparative_takeaways", [])])
    lines.extend(
        [
            "",
            "## 我准备怎么改",
            "",
            "- 必须保留：",
            *[f"  - {item}" for item in analysis_data["rewrite_blueprint"].get("must_keep", [])],
            "- 必须升级：",
            *[f"  - {item}" for item in analysis_data["rewrite_blueprint"].get("must_upgrade", [])],
            "- 开头升级：",
            *[f"  - {item}" for item in analysis_data["rewrite_blueprint"].get("opening_hook_upgrades", [])],
            f"- 结尾目标：{analysis_data['rewrite_blueprint'].get('cliffhanger_target', '')}",
            "- 台词规则：",
            *[f"  - {item}" for item in analysis_data["rewrite_blueprint"].get("line_polish_rules", [])],
            "- 禁止项：",
            *[f"  - {item}" for item in analysis_data["rewrite_blueprint"].get("forbidden_moves", [])],
            "",
            "## 实际改动方案",
            "",
            *[f"- {item}" for item in rewrite_data.get("applied_plan", [])],
            "",
            "## 风格化改稿思路",
            "",
            *[f"- {item}" for item in rewrite_data.get("style_adaptation_notes", [])],
            "",
            "## 爆款点具体加在了哪里",
            "",
            *[f"- {item}" for item in rewrite_data.get("explosive_insertions", [])],
            "",
            "## 具体改了哪些地方",
            "",
            *[f"- {item}" for item in rewrite_data.get("change_log", [])],
            "",
            "## 这版为什么会更抓人",
            "",
            *[f"- {item}" for item in rewrite_data.get("top_improvements", [])],
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_episode_change_log_markdown(
    analysis_data: Mapping[str, Any],
    rewrite_data: Mapping[str, Any],
    rewritten_script_path: Path,
) -> str:
    lines = [
        "# 单集爆款改稿变更说明",
        "",
        f"- 集数：{analysis_data['episode_id']}",
        f"- 目标风格：{rewrite_data.get('style_target_label', analysis_data.get('style_target_label', '未指定'))}",
        f"- 最终剧本：{rewritten_script_path}",
        "",
        "## 修改思路",
        "",
        *[f"- {item}" for item in rewrite_data.get("style_adaptation_notes", [])],
        "",
        "## 具体改动",
        "",
        *[f"- {item}" for item in rewrite_data.get("change_log", [])],
        "",
        "## 爆点加点位",
        "",
        *[f"- {item}" for item in rewrite_data.get("explosive_insertions", [])],
        "",
        "## 保留原则",
        "",
        *[f"- {item}" for item in analysis_data.get("rewrite_blueprint", {}).get("must_keep", [])],
        "",
        "## 强化后预期收益",
        "",
        *[f"- {item}" for item in rewrite_data.get("top_improvements", [])],
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_series_playbook_markdown(data: Mapping[str, Any]) -> str:
    lines = [
        "# 本剧爆款玩法手册",
        "",
        f"- 剧名：{data['series_name']}",
        f"- 受众定位：{data['audience_positioning']}",
        f"- 目标风格：{data.get('style_target_label', '未指定')}",
        "",
        "## 核心钩子公式",
        "",
        *[f"- {item}" for item in data.get("core_hook_formula", [])],
        "",
        "## 当前已经有效的优点",
        "",
        *[f"- {item}" for item in data.get("recurring_strengths", [])],
        "",
        "## 当前最需要避免的问题",
        "",
        *[f"- {item}" for item in data.get("recurring_weaknesses", [])],
        "",
        "## 必备爆点",
        "",
        *[f"- {item}" for item in data.get("viral_must_haves", [])],
        "",
        "## 改稿总规则",
        "",
        *[f"- {item}" for item in data.get("explosive_rules", [])],
        "",
        "## 开头常用打法",
        "",
        *[f"- {item}" for item in data.get("opening_patterns", [])],
        "",
        "## 结尾卡点打法",
        "",
        *[f"- {item}" for item in data.get("cliffhanger_patterns", [])],
        "",
        "## 台词升级规则",
        "",
        *[f"- {item}" for item in data.get("dialogue_upgrade_rules", [])],
        "",
    ]
    return "\n".join(lines)


def render_series_narrative_context_markdown(data: Mapping[str, Any]) -> str:
    lines = [
        "# 爆款改稿叙事上下文卡",
        "",
        f"- 剧名：{data.get('series_name', '')}",
        "",
        "## 整体故事基调",
        "",
        *[f"- {item}" for item in data.get("overall_story_tone", [])],
        "",
        "## 整体故事概述",
        "",
        str(data.get("global_story_summary", "")).strip(),
        "",
        "## 主角核心驱动力",
        "",
        *[f"- {item}" for item in data.get("protagonist_core_drive", [])],
        "",
        "## 核心关系轴",
        "",
        *[f"- {item}" for item in data.get("core_relationship_axes", [])],
        "",
        "## 连续性护栏",
        "",
        *[f"- {item}" for item in data.get("continuity_guardrails", [])],
        "",
        "## 绝对不能改坏的地方",
        "",
        *[f"- {item}" for item in data.get("narrative_do_not_break", [])],
        "",
        "## 分集上下文卡",
        "",
    ]
    for card in data.get("episode_cards", []):
        lines.extend(
            [
                f"### {card.get('episode_id', '')} {card.get('title', '')}",
                f"- 本集在整季中的作用：{card.get('role_in_arc', '')}",
                f"- 剧情摘要：{card.get('plot_summary', '')}",
                "- 开局状态：",
                *[f"  - {item}" for item in card.get("opening_state", [])],
                "- 收束状态：",
                *[f"  - {item}" for item in card.get("closing_state", [])],
                "- 与上一集的承接：",
                *[f"  - {item}" for item in card.get("bridge_from_previous", [])],
                "- 给下一集的钩子：",
                *[f"  - {item}" for item in card.get("bridge_to_next", [])],
                "- 必须保留：",
                *[f"  - {item}" for item in card.get("must_preserve", [])],
                "- 改稿时要警惕：",
                *[f"  - {item}" for item in card.get("continuity_risks", [])],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_series_narrative_context_brief(data: Mapping[str, Any]) -> str:
    lines = [
        "# 爆款改稿叙事总纲",
        "",
        f"- 剧名：{data.get('series_name', '')}",
        "",
        "## 整体故事基调",
        "",
        *[f"- {item}" for item in data.get("overall_story_tone", [])],
        "",
        "## 整体故事概述",
        "",
        str(data.get("global_story_summary", "")).strip(),
        "",
        "## 主角核心驱动力",
        "",
        *[f"- {item}" for item in data.get("protagonist_core_drive", [])],
        "",
        "## 核心关系轴",
        "",
        *[f"- {item}" for item in data.get("core_relationship_axes", [])],
        "",
        "## 连续性护栏",
        "",
        *[f"- {item}" for item in data.get("continuity_guardrails", [])],
        "",
        "## 绝对不能改坏的地方",
        "",
        *[f"- {item}" for item in data.get("narrative_do_not_break", [])],
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_episode_context_bundle(
    narrative_context: Mapping[str, Any],
    episode_id: str,
) -> dict[str, str]:
    cards = [item for item in narrative_context.get("episode_cards", []) if isinstance(item, dict)]
    indexed = {str(item.get("episode_id", "")).strip(): item for item in cards}
    order = [str(item.get("episode_id", "")).strip() for item in cards if str(item.get("episode_id", "")).strip()]
    current_index = order.index(episode_id) if episode_id in order else -1

    def render_card(card: Mapping[str, Any] | None) -> str:
        if not card:
            return "<空>"
        return json.dumps(card, ensure_ascii=False, indent=2)

    previous_card = indexed.get(order[current_index - 1]) if current_index > 0 else None
    current_card = indexed.get(episode_id)
    next_card = indexed.get(order[current_index + 1]) if 0 <= current_index < len(order) - 1 else None
    return {
        "series_narrative_context_text": render_series_narrative_context_brief(narrative_context),
        "previous_episode_context_text": render_card(previous_card),
        "current_episode_context_text": render_card(current_card),
        "next_episode_context_text": render_card(next_card),
    }


def style_filename_suffix(style_reference: Mapping[str, Any]) -> str:
    style_slug = str(style_reference.get("style_target", {}).get("style_slug", "")).strip()
    return f"__{style_slug}" if style_slug else ""


def rewrite_output_filename(config: dict[str, Any], episode_id: str, model: str, style_reference: Mapping[str, Any]) -> str:
    suffix = config["output"].get("rewrite_filename_suffix", "__explosive")
    return f"{episode_id}__openai__{model}{style_filename_suffix(style_reference)}{suffix}.md"


def choose_rewrite_output_dir(config: dict[str, Any], script_dir: Path, series_name: str) -> Path:
    suffix = (config["output"].get("script_series_suffix") or "").strip()
    if suffix:
        return (PROJECT_ROOT / "script" / f"{series_name}{suffix}").resolve()
    return script_dir


def print_episode_feedback(
    episode_id: str,
    analysis_result: Mapping[str, Any],
    rewrite_result: Mapping[str, Any],
) -> None:
    print_status(f"{episode_id} 目标风格：{analysis_result.get('style_target_label', '未指定')}")
    print_status(f"{episode_id} 当前爆款评分：")
    for item in analysis_result.get("variants", []):
        print_status(
            f"- {item['label']}: {item['explosive_score']}/100 "
            f"(钩子{item['hook_score']} 冲突{item['conflict_score']} 情绪{item['emotion_score']} "
            f"节奏{item['pace_score']} 人物{item['character_pull_score']} 结尾{item['cliffhanger_score']} "
            f"风格匹配{item.get('target_style_fit_score', 0)})"
        )
        if item.get("strengths"):
            print_status(f"  优点：{item['strengths'][0]}")
        if item.get("weaknesses"):
            print_status(f"  短板：{item['weaknesses'][0]}")
    print_status(f"{episode_id} 改稿方向：")
    for point in rewrite_result.get("applied_plan", [])[:3]:
        print_status(f"- {point}")
    print_status(f"{episode_id} 爆点加点位：")
    for point in rewrite_result.get("explosive_insertions", [])[:3]:
        print_status(f"- {point}")
    print_status(
        f"{episode_id} 强化后预测爆款分：{rewrite_result['predicted_explosive_score']}/100 "
        f"(预计提升 +{rewrite_result['score_delta']})"
    )


def render_metrics_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# 爆款改稿统计报告",
        "",
        f"- run_name：{report.get('run_name', '')}",
        f"- final_status：{report.get('context', {}).get('final_status', '')}",
        f"- series_name：{report.get('context', {}).get('series_name', '')}",
        f"- episode_range：{report.get('context', {}).get('episode_range', '')}",
        f"- model：{report.get('context', {}).get('model', '')}",
        f"- style_target：{report.get('context', {}).get('style_target', '')}",
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


def load_or_build_series_narrative_context(
    *,
    config: dict[str, Any],
    series_name: str,
    model: str,
    api_key: str,
    episode_ids: list[str],
    grouped: Mapping[str, list[dict[str, Any]]],
    timeout_seconds: int,
    telemetry: TelemetryRecorder | None = None,
) -> dict[str, Any]:
    json_path, md_path = series_narrative_context_paths(config, series_name, model)
    reuse_existing = bool(config.get("context_summary", {}).get("reuse_if_exists", True))
    if reuse_existing and json_path.exists():
        cached = load_json_file(json_path)
        covered_ids = {
            str(item.get("episode_id", "")).strip()
            for item in cached.get("episode_cards", [])
            if isinstance(item, dict)
        }
        if set(episode_ids).issubset(covered_ids):
            print_status(f"复用整剧叙事上下文：{json_path}")
            return cached
        print_status("已有整剧叙事上下文未覆盖当前集数，将重新生成。")

    context_sources = collect_series_context_sources(series_name, episode_ids, grouped)
    with telemetry_span(
        telemetry,
        stage="explosive_rewrite",
        name="build_series_narrative_context_prompt",
        metadata={"episode_count": len(episode_ids), "series_name": series_name},
    ) as step:
        prompt = build_series_narrative_context_prompt(
            config=config,
            series_name=series_name,
            episode_ids=episode_ids,
            context_sources=context_sources,
        )
        step["metadata"]["prompt_chars"] = len(prompt)
    narrative_context = openai_json_completion(
        model=model,
        api_key=api_key,
        system_prompt=load_prompt("explosive_rewrite/series_narrative_context_system.md"),
        user_prompt=prompt,
        schema_name="series_narrative_context",
        schema=SERIES_NARRATIVE_CONTEXT_SCHEMA,
        temperature=min(float(config["run"].get("temperature", 0.3)), 0.2),
        timeout_seconds=timeout_seconds,
        telemetry=telemetry,
        stage="explosive_rewrite",
        step_name="series_narrative_context_model_call",
        metadata={"episode_count": len(episode_ids), "series_name": series_name},
    )
    save_json_file(json_path, narrative_context)
    save_text_file(md_path, render_series_narrative_context_markdown(narrative_context))
    print_status(f"整剧叙事上下文已写入：{json_path}")
    return narrative_context


def metrics_summary_paths(config: Mapping[str, Any], series_name: str, model: str, style_reference: Mapping[str, Any]) -> tuple[Path, Path]:
    ids = episode_id_sequence(dict(config))
    first_episode = ids[0]
    last_episode = ids[-1]
    output_tag = build_provider_model_tag("openai", model)
    style_suffix = style_filename_suffix(style_reference)
    base = (
        PROJECT_ROOT
        / "analysis"
        / series_name
        / str(config.get("output", {}).get("analysis_dir_name", "explosive-actor-gpt"))
        / f"metrics_summary__{output_tag}__{first_episode}-{last_episode}{style_suffix}"
    )
    return Path(f"{base}.json"), Path(f"{base}.md")


def save_metrics(recorder: TelemetryRecorder, json_path: Path, md_path: Path) -> dict[str, Any]:
    report = recorder.to_dict()
    save_json_file(json_path, report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_metrics_markdown(report), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze multiple script variants and generate more explosive rewritten scripts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--collect-metrics", choices=["true", "false"])
    return parser


def run_pipeline(config: dict[str, Any], telemetry: TelemetryRecorder | None = None) -> dict[str, Any]:
    series_name = resolve_series_name(config)
    model, api_key = configure_openai(config)
    episode_ids = episode_id_sequence(config)
    style_reference = build_target_style_reference(config, series_name)
    style_slug_suffix = style_filename_suffix(style_reference)

    script_dir = Path(config["script"]["series_dir"]).expanduser().resolve()
    analysis_dir_name = config["output"].get("analysis_dir_name", "explosive-actor-gpt")
    analysis_root = (PROJECT_ROOT / "analysis" / series_name / analysis_dir_name).resolve()
    rewrite_series_dir = choose_rewrite_output_dir(config, script_dir, series_name)
    playbook_path = analysis_root / f"series_hit_playbook{style_slug_suffix or ''}.md"
    playbook_text = read_supporting_text(playbook_path) or read_supporting_text(analysis_root / "series_hit_playbook.md")
    narrative_context_json_path, narrative_context_md_path = series_narrative_context_paths(config, series_name, model)

    grouped = collect_episode_variants(config, episode_ids)
    missing = [episode_id for episode_id in episode_ids if not grouped.get(episode_id)]
    if missing:
        raise RuntimeError(f"以下集数未找到可用剧本版本：{', '.join(missing)}")

    print_status(f"输入剧本目录：{script_dir}")
    print_status(f"爆款分析目录：{analysis_root}")
    print_status(f"强化版剧本目录：{rewrite_series_dir}")
    print_status(
        "目标风格："
        + (
            style_reference["style_target"].get("style_label")
            or "未指定，按通用爆款改稿处理"
        )
    )
    if style_reference["target_packages"]:
        print_status(
            "将重点参考题材库："
            + "、".join(str(item.get("genre_key", "")).strip() for item in style_reference["target_packages"])
        )
    elif style_reference["style_target"].get("custom_style_tokens"):
        print_status(
            "目标风格为自定义词："
            + "、".join(style_reference["style_target"]["custom_style_tokens"])
        )
    if style_reference["current_series_context"].get("matched_packages"):
        print_status(
            "原剧当前题材参考："
            + "、".join(
                str(item.get("genre_key", "")).strip()
                for item in style_reference["current_series_context"]["matched_packages"]
            )
        )

    if config["run"].get("dry_run", False):
        preview = {
            "series_name": series_name,
            "episode_ids": episode_ids,
            "script_dir": str(script_dir),
            "analysis_root": str(analysis_root),
            "rewrite_series_dir": str(rewrite_series_dir),
            "series_narrative_context_json_path": str(narrative_context_json_path),
            "series_narrative_context_markdown_path": str(narrative_context_md_path),
            "style_target": style_reference["style_target"],
            "target_genre_keys": [item.get("genre_key", "") for item in style_reference["target_packages"]],
            "episodes": {
                episode_id: [item["filename"] for item in grouped.get(episode_id, [])]
                for episode_id in episode_ids
            },
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    timeout_seconds = int(config["run"].get("timeout_seconds", 300))
    temperature = float(config["run"].get("temperature", 0.3))
    narrative_context = load_or_build_series_narrative_context(
        config=config,
        series_name=series_name,
        model=model,
        api_key=api_key,
        episode_ids=episode_ids,
        grouped=grouped,
        timeout_seconds=timeout_seconds,
        telemetry=telemetry,
    )

    episode_reports: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        variants = grouped[episode_id]
        print_status(f"开始分析 {episode_id}，共 {len(variants)} 个版本。")
        for item in variants:
            print_status(f"{episode_id} 版本路径：{item['path']}")
        episode_context_bundle = build_episode_context_bundle(narrative_context, episode_id)
        with telemetry_span(
            telemetry,
            stage="explosive_rewrite",
            name="build_episode_analysis_prompt",
            metadata={"episode_id": episode_id, "variant_count": len(variants)},
        ) as step:
            analysis_prompt = build_episode_analysis_prompt(
                config=config,
                series_name=series_name,
                episode_id=episode_id,
                variants=variants,
                playbook_text=playbook_text,
                style_reference=style_reference,
                episode_context_bundle=episode_context_bundle,
            )
            step["metadata"]["prompt_chars"] = len(analysis_prompt)
        analysis_result = openai_json_completion(
            model=model,
            api_key=api_key,
            system_prompt=load_prompt("explosive_rewrite/analysis_system.md"),
            user_prompt=analysis_prompt,
            schema_name="episode_explosive_analysis",
            schema=EPISODE_ANALYSIS_SCHEMA,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            telemetry=telemetry,
            stage="explosive_rewrite",
            step_name="episode_analysis_model_call",
            metadata={"episode_id": episode_id, "variant_count": len(variants)},
        )

        with telemetry_span(
            telemetry,
            stage="explosive_rewrite",
            name="build_episode_rewrite_prompt",
            metadata={"episode_id": episode_id},
        ) as step:
            rewrite_prompt = build_episode_rewrite_prompt(
                config=config,
                series_name=series_name,
                episode_id=episode_id,
                variants=variants,
                analysis_result=analysis_result,
                playbook_text=playbook_text,
                style_reference=style_reference,
                episode_context_bundle=episode_context_bundle,
            )
            step["metadata"]["prompt_chars"] = len(rewrite_prompt)
        rewrite_result = openai_json_completion(
            model=model,
            api_key=api_key,
            system_prompt=load_prompt("explosive_rewrite/rewrite_system.md"),
            user_prompt=rewrite_prompt,
            schema_name="episode_explosive_rewrite",
            schema=EPISODE_REWRITE_SCHEMA,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            telemetry=telemetry,
            stage="explosive_rewrite",
            step_name="episode_rewrite_model_call",
            metadata={"episode_id": episode_id},
        )

        episode_dir = analysis_root / episode_id
        report_json_path = episode_dir / f"episode_explosive_report__openai__{model}{style_slug_suffix}.json"
        report_md_path = episode_dir / f"episode_explosive_report__openai__{model}{style_slug_suffix}.md"
        rewrite_json_path = episode_dir / f"episode_explosive_rewrite__openai__{model}{style_slug_suffix}.json"
        change_log_json_path = episode_dir / f"episode_explosive_change_log__openai__{model}{style_slug_suffix}.json"
        change_log_md_path = episode_dir / f"episode_explosive_change_log__openai__{model}{style_slug_suffix}.md"
        rewritten_script_path = rewrite_series_dir / rewrite_output_filename(config, episode_id, model, style_reference)

        with telemetry_span(
            telemetry,
            stage="explosive_rewrite",
            name="save_episode_rewrite_outputs",
            metadata={
                "episode_id": episode_id,
                "report_json_path": str(report_json_path),
                "report_md_path": str(report_md_path),
                "rewrite_json_path": str(rewrite_json_path),
                "change_log_json_path": str(change_log_json_path),
                "change_log_md_path": str(change_log_md_path),
                "rewritten_script_path": str(rewritten_script_path),
            },
        ):
            save_json_file(report_json_path, analysis_result)
            save_json_file(rewrite_json_path, rewrite_result)
            save_json_file(
                change_log_json_path,
                {
                    "episode_id": episode_id,
                    "style_target_label": rewrite_result.get("style_target_label", analysis_result.get("style_target_label", "")),
                    "applied_plan": rewrite_result.get("applied_plan", []),
                    "style_adaptation_notes": rewrite_result.get("style_adaptation_notes", []),
                    "change_log": rewrite_result.get("change_log", []),
                    "explosive_insertions": rewrite_result.get("explosive_insertions", []),
                    "top_improvements": rewrite_result.get("top_improvements", []),
                    "rewrite_path": str(rewritten_script_path),
                },
            )
            save_text_file(rewritten_script_path, rewrite_result["revised_script_markdown"])
            save_text_file(report_md_path, render_episode_report_markdown(analysis_result, rewrite_result, rewritten_script_path))
            save_text_file(change_log_md_path, render_episode_change_log_markdown(analysis_result, rewrite_result, rewritten_script_path))

        episode_reports.append(
            {
                "episode_id": episode_id,
                "report_path": str(report_json_path),
                "change_log_path": str(change_log_md_path),
                "rewrite_path": str(rewritten_script_path),
                "style_target_label": rewrite_result.get("style_target_label", analysis_result.get("style_target_label", "")),
                "overall_winner": analysis_result["overall_winner"],
                "predicted_explosive_score": rewrite_result["predicted_explosive_score"],
                "score_delta": rewrite_result["score_delta"],
                "top_improvements": rewrite_result["top_improvements"],
            }
        )
        print_episode_feedback(episode_id, analysis_result, rewrite_result)
        print_status(f"{episode_id} 完成：强化版剧本已写入 {rewritten_script_path}")

    with telemetry_span(
        telemetry,
        stage="explosive_rewrite",
        name="build_series_playbook_prompt",
        metadata={"episode_count": len(episode_reports)},
    ) as step:
        series_playbook_prompt = build_series_playbook_prompt(
            config=config,
            series_name=series_name,
            episode_reports=episode_reports,
            style_reference=style_reference,
        )
        step["metadata"]["prompt_chars"] = len(series_playbook_prompt)
    series_playbook = openai_json_completion(
        model=model,
        api_key=api_key,
        system_prompt=load_prompt("explosive_rewrite/series_playbook_system.md"),
        user_prompt=series_playbook_prompt,
        schema_name="series_hit_playbook",
        schema=SERIES_PLAYBOOK_SCHEMA,
        temperature=max(0.1, min(temperature, 0.2)),
        timeout_seconds=timeout_seconds,
        telemetry=telemetry,
        stage="explosive_rewrite",
        step_name="series_playbook_model_call",
        metadata={"episode_count": len(episode_reports)},
    )

    with telemetry_span(
        telemetry,
        stage="explosive_rewrite",
        name="save_series_playbook_outputs",
        metadata={"playbook_path": str(playbook_path)},
    ):
        save_json_file(analysis_root / f"series_hit_playbook__openai__{model}{style_slug_suffix}.json", series_playbook)
        save_text_file(playbook_path, render_series_playbook_markdown(series_playbook))

    summary = {
        "series_name": series_name,
        "model": model,
        "style_target": style_reference["style_target"],
        "source_script_dir": str(script_dir),
        "rewrite_script_dir": str(rewrite_series_dir),
        "analysis_root": str(analysis_root),
        "series_narrative_context_json_path": str(narrative_context_json_path),
        "series_narrative_context_markdown_path": str(narrative_context_md_path),
        "episodes": episode_reports,
        "generated_at": utc_timestamp(),
    }
    with telemetry_span(
        telemetry,
        stage="explosive_rewrite",
        name="save_explosive_summary",
        metadata={"analysis_root": str(analysis_root)},
    ):
        save_json_file(analysis_root / f"explosive_rewrite_summary{style_slug_suffix}.json", summary)
    print_status("爆款改稿链路完成。")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_runtime_config(args.config)
    collect_metrics_override = parse_bool_flag(args.collect_metrics)
    if collect_metrics_override is not None:
        config.setdefault("run", {})
        config["run"]["collect_metrics"] = collect_metrics_override

    print_status(f"加载配置：{args.config}")
    series_name = resolve_series_name(config)
    model, _ = configure_openai(config)
    style_reference = build_target_style_reference(config, series_name)
    collect_metrics = bool(config.get("run", {}).get("collect_metrics", False))
    print_status(f"统计报告：{'开启' if collect_metrics else '关闭'}")

    recorder: TelemetryRecorder | None = None
    metrics_json_path: Path | None = None
    metrics_md_path: Path | None = None
    if collect_metrics:
        recorder = TelemetryRecorder(
            run_name="explosive-rewrite",
            context={
                "series_name": series_name,
                "episode_range": f"{episode_id_sequence(config)[0]}-{episode_id_sequence(config)[-1]}",
                "script_series_dir": str(Path(config["script"]["series_dir"]).expanduser().resolve()),
                "model": model,
                "style_target": str(style_reference["style_target"].get("style_label") or "未指定"),
            },
        )
        metrics_json_path, metrics_md_path = metrics_summary_paths(config, series_name, model, style_reference)

    try:
        summary = run_pipeline(config, telemetry=recorder)
        if recorder and metrics_json_path and metrics_md_path:
            recorder.context["final_status"] = "completed"
            report = save_metrics(recorder, metrics_json_path, metrics_md_path)
            if isinstance(summary, dict):
                summary["metrics_summary_json_path"] = str(metrics_json_path)
                summary["metrics_summary_markdown_path"] = str(metrics_md_path)
                summary["metrics_totals"] = report["totals"]
            print_status(
                f"统计：总耗时 {report['totals']['duration_seconds']}s | "
                f"tokens in/out/total = {report['totals']['input_tokens']}/{report['totals']['output_tokens']}/{report['totals']['total_tokens']}"
            )
            print_status(f"统计报告已写入：{metrics_json_path}")
    except Exception as exc:
        if recorder and metrics_json_path and metrics_md_path:
            recorder.context["final_status"] = "failed"
            recorder.context["failure_message"] = str(exc)
            report = save_metrics(recorder, metrics_json_path, metrics_md_path)
            print_status(
                f"统计：总耗时 {report['totals']['duration_seconds']}s | "
                f"tokens in/out/total = {report['totals']['input_tokens']}/{report['totals']['output_tokens']}/{report['totals']['total_tokens']}"
            )
            print_status(f"失败时的统计报告已写入：{metrics_json_path}")
        raise


if __name__ == "__main__":
    main()
