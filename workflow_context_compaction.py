from __future__ import annotations

import json
import re
from typing import Any, Mapping


MIXED_CROWD_ROLE_TOKENS = (
    "宾客",
    "宾客群像",
    "侍从",
    "婢女",
    "侍卫",
    "护卫",
    "守卫",
    "群像",
    "围观者",
    "看客",
)


def shorten_text(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def normalize_episode_id(raw: str | None) -> str | None:
    if not raw:
        return None
    match = re.search(r"ep(\d+)", str(raw), flags=re.IGNORECASE)
    if not match:
        return None
    return f"ep{int(match.group(1)):02d}"


def _episode_sort_key(raw: str | None) -> tuple[int, str]:
    episode_id = normalize_episode_id(raw)
    if not episode_id:
        return (10**9, str(raw or ""))
    return (int(re.search(r"(\d+)", episode_id).group(1)), episode_id)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def is_mixed_crowd_character_asset(
    name: str,
    *,
    appearance_keywords: str = "",
    reuse_note: str = "",
) -> bool:
    joined = " ".join(normalize_spaces(part) for part in (name, appearance_keywords, reuse_note) if normalize_spaces(part))
    if not joined:
        return False
    token_hits = {token for token in MIXED_CROWD_ROLE_TOKENS if token in joined}
    separator_like = any(mark in str(name or "") for mark in ("/", "／", "、", "，", ","))
    if "群像" in token_hits and len(token_hits) >= 2:
        return True
    if len(token_hits) >= 3:
        return True
    if separator_like and len(token_hits) >= 2:
        return True
    return False


def _extract_episode_blocks(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"<!--\s*episode:\s*(ep\d+)\s*start\s*-->(.*?)<!--\s*episode:\s*\1\s*end\s*-->",
        flags=re.IGNORECASE | re.DOTALL,
    )
    blocks: list[tuple[str, str]] = []
    for match in pattern.finditer(text or ""):
        episode_id = normalize_episode_id(match.group(1)) or match.group(1)
        block_text = match.group(2).strip()
        if block_text:
            blocks.append((episode_id, block_text))
    return blocks


def compact_episode_scoped_prompt_library(
    text: str,
    episode_id: str | None,
    *,
    limit: int = 1800,
    max_recent_blocks: int = 2,
) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    blocks = _extract_episode_blocks(raw)
    if not blocks:
        return shorten_text(raw, limit)

    current = normalize_episode_id(episode_id)
    if current:
        eligible = [item for item in blocks if _episode_sort_key(item[0]) <= _episode_sort_key(current)]
    else:
        eligible = blocks
    if not eligible:
        eligible = blocks

    selected = eligible[-max_recent_blocks:]
    rendered = []
    for block_episode_id, block_text in selected:
        rendered.append(f"[{block_episode_id}]\n{block_text}")
    return shorten_text("\n\n".join(rendered), limit)


def _collect_markdown_section_titles(text: str, *, limit: int = 8) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"^##\s+(.+)$", str(text or ""), flags=re.MULTILINE):
        title = str(match.group(1) or "").strip()
        normalized = re.sub(r"\s+", " ", title)
        if not normalized or normalized.lower().startswith("ep"):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        titles.append(normalized)
        if len(titles) >= limit:
            break
    return titles


def compact_existing_character_assets_for_director(
    text: str,
    episode_id: str | None,
    *,
    limit: int = 220,
) -> str:
    scoped = compact_episode_scoped_prompt_library(text, episode_id, limit=4000, max_recent_blocks=1)
    if not scoped:
        return "<空>"
    titles = _collect_markdown_section_titles(scoped, limit=8)
    if not titles:
        return "<空>"
    summary = "已有角色资产摘要：" + "；".join(titles) + "。仅用于判断复用、变体与命名连续性，不回抄完整造型描述。"
    return shorten_text(summary, limit)


def compact_existing_scene_assets_for_director(
    text: str,
    episode_id: str | None,
    *,
    limit: int = 260,
) -> str:
    scoped = compact_episode_scoped_prompt_library(text, episode_id, limit=5000, max_recent_blocks=1)
    if not scoped:
        return "<空>"
    names: list[str] = []
    seen: set[str] = set()
    patterns = [
        r"(?:^|\n)\s*(?:-+\s*)?格\d+\s*[：:]\s*([^\n（(]+)",
        r"(?:^|\n)\s*(?:-+\s*)?场景\d+\s*[：:]\s*([^\n（(]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, scoped):
            name = re.sub(r"\s+", " ", str(match.group(1) or "").strip())
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
            if len(names) >= 10:
                break
        if len(names) >= 10:
            break
    if not names:
        names = _collect_markdown_section_titles(scoped, limit=8)
    if not names:
        return "<空>"
    summary = "已有场景资产摘要：" + "；".join(names) + "。仅用于判断母体场景复用、子场景拆分与命名连续性，不回抄完整环境提示词。"
    return shorten_text(summary, limit)


def compact_reference_text(text: str, limit: int) -> str:
    cleaned = re.sub(r"<!--.*?-->", "", str(text or ""), flags=re.DOTALL)
    return shorten_text(cleaned, limit)


def compact_series_context_for_director(series_context: Mapping[str, Any]) -> dict[str, Any]:
    active_characters = []
    for item in list(series_context.get("active_characters") or [])[:4]:
        active_characters.append(
            {
                "name": item.get("name", ""),
                "role": item.get("role", ""),
                "relationship": item.get("relationship_to_protagonist", ""),
                "state": shorten_text(item.get("latest_state", ""), 120),
            }
        )

    active_locations = []
    for item in list(series_context.get("active_locations") or [])[:4]:
        if isinstance(item, Mapping):
            active_locations.append(
                {
                    "name": item.get("name", ""),
                    "state": shorten_text(item.get("latest_state", ""), 100),
                }
            )
        else:
            active_locations.append(shorten_text(str(item), 100))

    unresolved_threads = [shorten_text(item, 110) for item in list(series_context.get("unresolved_threads") or [])[:4]]

    recent_timeline = []
    for item in list(series_context.get("recent_timeline") or [])[-2:]:
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
        "continuity_rules": [shorten_text(item, 90) for item in list(series_context.get("continuity_rules") or [])[:4]],
        "genre_profile": series_context.get("genre_profile", {}),
        "downstream_design_guidance": series_context.get("downstream_design_guidance", {}),
        "active_characters": active_characters,
        "active_locations": active_locations,
        "unresolved_threads": unresolved_threads,
        "recent_timeline": recent_timeline,
    }


def compact_genre_reference_bundle_for_director(bundle: Mapping[str, Any]) -> dict[str, Any]:
    aggregate_focus = bundle.get("aggregate_focus") or {}
    normalized_adapters = bundle.get("normalized_adapters") or {}
    director_adapter = normalized_adapters.get("director") or {}
    return {
        "selected_genres": list(bundle.get("selected_genres") or [])[:3],
        "stage_adapter_for_director": {
            "dramatic_engine": [shorten_text(x, 80) for x in list(director_adapter.get("dramatic_engine") or [])[:3]],
            "visual_priority": [shorten_text(x, 80) for x in list(director_adapter.get("visual_priority") or [])[:3]],
            "beat_templates": [shorten_text(x, 80) for x in list(director_adapter.get("beat_templates") or [])[:3]],
            "continuity_guardrails": [shorten_text(x, 80) for x in list(director_adapter.get("continuity_guardrails") or [])[:3]],
            "negative_patterns": [shorten_text(x, 80) for x in list(director_adapter.get("negative_patterns") or [])[:3]],
        },
        "aggregate_focus": {
            "director_focus": [shorten_text(x, 80) for x in list(aggregate_focus.get("director_focus") or [])[:4]],
            "dialogue_timing_rules": [shorten_text(x, 75) for x in list(aggregate_focus.get("dialogue_timing_rules") or [])[:3]],
            "continuity_guardrails": [shorten_text(x, 75) for x in list(aggregate_focus.get("continuity_guardrails") or [])[:3]],
            "negative_patterns": [shorten_text(x, 75) for x in list(aggregate_focus.get("negative_patterns") or [])[:3]],
        },
    }


def compact_analysis_for_art(analysis: Mapping[str, Any]) -> dict[str, Any]:
    story_beats = []
    for item in list(analysis.get("story_beats") or [])[:4]:
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
    for item in list(analysis.get("characters") or [])[:6]:
        if isinstance(item, Mapping):
            characters.append({
                "name": item.get("name", ""),
                "role": item.get("role", ""),
                "core_state": shorten_text(item.get("core_state", item.get("relationship_to_protagonist", "")), 100),
            })
        else:
            characters.append(shorten_text(str(item), 100))

    locations = []
    for item in list(analysis.get("locations") or [])[:6]:
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
            for item in list(series_context.get("recent_timeline") or [])[-2:]
        ],
    }


def compact_genre_reference_bundle_for_art(bundle: Mapping[str, Any]) -> dict[str, Any]:
    aggregate_focus = bundle.get("aggregate_focus") or {}
    return {
        "selected_genres": list(bundle.get("selected_genres") or [])[:3],
        "aggregate_focus": {
            "character_focus": [shorten_text(x, 80) for x in list(aggregate_focus.get("character_focus") or [])[:4]],
            "costume_focus": [shorten_text(x, 80) for x in list(aggregate_focus.get("costume_focus") or [])[:4]],
            "scene_focus": [shorten_text(x, 80) for x in list(aggregate_focus.get("scene_focus") or [])[:4]],
            "continuity_guardrails": [shorten_text(x, 75) for x in list(aggregate_focus.get("continuity_guardrails") or [])[:3]],
            "negative_patterns": [shorten_text(x, 75) for x in list(aggregate_focus.get("negative_patterns") or [])[:3]],
        },
    }


def compact_director_analysis_text(text: str) -> str:
    return compact_reference_text(text, 3600)


def compact_director_analysis_text_for_storyboard_review(text: str) -> str:
    return compact_reference_text(text, 1200)


def _render_compact_director_brief_lines(
    data: Mapping[str, Any] | None,
    *,
    point_limit: int,
    point_text_limit: int,
    note_limit: int,
) -> list[str]:
    payload = dict(data or {})
    lines: list[str] = []
    episode_title = str(payload.get("episode_title") or "").strip()
    structure_overview = str(payload.get("structure_overview") or "").strip()
    emotional_curve = str(payload.get("emotional_curve") or "").strip()
    if episode_title:
        lines.append(f"标题：{episode_title}")
    if structure_overview:
        lines.append(f"结构：{shorten_text(structure_overview, 220)}")
    if emotional_curve:
        lines.append(f"情绪曲线：{shorten_text(emotional_curve, 150)}")
    director_notes = [shorten_text(str(item or ""), 110) for item in list(payload.get("director_notes") or [])[:note_limit]]
    if director_notes:
        lines.append("导演重点：")
        lines.extend(f"- {item}" for item in director_notes if item)
    story_points = list(payload.get("story_points") or [])
    if story_points:
        lines.append("剧情点执行摘要：")
    for item in story_points[:point_limit]:
        point_id = str(item.get("point_id") or "").strip()
        title = str(item.get("title") or "").strip()
        statement = str(item.get("director_statement") or item.get("narrative_function") or "").strip()
        summary = shorten_text(statement, point_text_limit)
        if not summary:
            continue
        header = " ".join(part for part in [point_id, title] if part).strip() or "剧情点"
        lines.append(f"- {header}：{summary}")
    return lines


def compact_director_brief_for_storyboard(
    data: Mapping[str, Any] | None,
    text: str,
    *,
    profile: str = "normal",
) -> str:
    profile = normalize_storyboard_profile(profile)
    is_fast = profile == "fast"
    lines = _render_compact_director_brief_lines(
        data,
        point_limit=10 if is_fast else 18,
        point_text_limit=74 if is_fast else 112,
        note_limit=3 if is_fast else 5,
    )
    if not lines:
        return compact_reference_text(text, 900 if is_fast else 1800)
    return shorten_text("\n".join(lines), 900 if is_fast else 1800)


def compact_director_brief_for_art(data: Mapping[str, Any] | None, text: str) -> str:
    payload = dict(data or {})
    lines = _render_compact_director_brief_lines(
        payload,
        point_limit=8,
        point_text_limit=92,
        note_limit=4,
    )
    characters = []
    for item in list(payload.get("characters") or [])[:8]:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        appearance = shorten_text(str(item.get("appearance_keywords") or ""), 56)
        status = str(item.get("asset_status") or "").strip()
        if not name:
            continue
        suffix = " / ".join(part for part in [appearance, status] if part)
        characters.append(f"{name}（{suffix}）" if suffix else name)
    if characters:
        lines.append("人物资产重点：")
        lines.extend(f"- {item}" for item in characters)
    scenes = []
    for item in list(payload.get("scenes") or [])[:8]:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        lighting = shorten_text(str(item.get("lighting_palette") or ""), 48)
        mood = shorten_text(str(item.get("mood_keywords") or ""), 48)
        if not name:
            continue
        suffix = " / ".join(part for part in [lighting, mood] if part)
        scenes.append(f"{name}（{suffix}）" if suffix else name)
    if scenes:
        lines.append("场景资产重点：")
        lines.extend(f"- {item}" for item in scenes)
    if not lines:
        return compact_reference_text(text, 1800)
    return shorten_text("\n".join(lines), 1800)


def normalize_storyboard_profile(raw: str | None) -> str:
    normalized = str(raw or "").strip().lower()
    return "fast" if normalized in {"fast", "极速", "极速版", "quick"} else "normal"


def compact_director_json_for_storyboard(data: Mapping[str, Any] | None, *, profile: str = "normal") -> dict[str, Any]:
    profile = normalize_storyboard_profile(profile)
    data = dict(data or {})
    is_fast = profile == "fast"
    story_points = []
    for item in list(data.get("story_points") or []):
        story_points.append(
            {
                "point_id": item.get("point_id", ""),
                "title": shorten_text(item.get("title", ""), 72 if is_fast else 96),
                "characters": list(item.get("characters") or [])[:4 if is_fast else 7],
                "scenes": list(item.get("scenes") or [])[:4 if is_fast else 7],
                "pace_label": item.get("pace_label", ""),
                "duration_suggestion": shorten_text(item.get("duration_suggestion", ""), 48 if is_fast else 72),
                "entry_state": shorten_text(
                    item.get("entry_state", "") or item.get("opening_visual_state", "") or item.get("continuity_hook_in", ""),
                    72 if is_fast else 104,
                ),
                "micro_beats": [
                    shorten_text(x, 78 if is_fast else 106)
                    for x in list(item.get("micro_beats") or [])[:2 if is_fast else 4]
                ],
                "detail_anchor_lines": [
                    shorten_text(x, 82 if is_fast else 118)
                    for x in list(item.get("detail_anchor_lines") or [])[:2 if is_fast else 3]
                ],
                "key_dialogue_beats": [
                    shorten_text(x, 76 if is_fast else 104)
                    for x in list(item.get("key_dialogue_beats") or [])[:1 if is_fast else 2]
                ],
                "exit_state": shorten_text(
                    item.get("exit_state", "") or item.get("closing_visual_state", "") or item.get("continuity_hook_out", ""),
                    72 if is_fast else 104,
                ),
            }
        )
    return {
        "episode_id": data.get("episode_id", ""),
        "episode_title": data.get("episode_title", ""),
        "story_points": story_points,
        "characters": list(data.get("characters") or [])[:5 if is_fast else 8],
        "scenes": list(data.get("scenes") or [])[:5 if is_fast else 8],
    }


def compact_director_json_for_storyboard_review(data: Mapping[str, Any] | None, *, profile: str = "normal") -> dict[str, Any]:
    profile = normalize_storyboard_profile(profile)
    data = dict(data or {})
    is_fast = profile == "fast"
    story_points = []
    for item in list(data.get("story_points") or []):
        story_points.append(
            {
                "point_id": item.get("point_id", ""),
                "title": shorten_text(item.get("title", ""), 60 if is_fast else 76),
                "pace_label": item.get("pace_label", ""),
                "duration_suggestion": shorten_text(item.get("duration_suggestion", ""), 40 if is_fast else 54),
                "continuity_hook_in": shorten_text(item.get("continuity_hook_in", ""), 60 if is_fast else 84),
                "opening_visual_state": shorten_text(item.get("opening_visual_state", ""), 68 if is_fast else 96),
                "micro_beats": [shorten_text(x, 68 if is_fast else 92) for x in list(item.get("micro_beats") or [])[:2 if is_fast else 4]],
                "detail_anchor_lines": [shorten_text(x, 76 if is_fast else 102) for x in list(item.get("detail_anchor_lines") or [])[:2 if is_fast else 3]],
                "key_dialogue_beats": [shorten_text(x, 68 if is_fast else 92) for x in list(item.get("key_dialogue_beats") or [])[:2 if is_fast else 3]],
                "closing_visual_state": shorten_text(item.get("closing_visual_state", ""), 68 if is_fast else 96),
                "timeline_adjustment_note": shorten_text(item.get("timeline_adjustment_note", ""), 56 if is_fast else 84),
                "continuity_hook_out": shorten_text(item.get("continuity_hook_out", ""), 60 if is_fast else 84),
            }
        )
    return {
        "episode_id": data.get("episode_id", ""),
        "episode_title": data.get("episode_title", ""),
        "structure_overview": shorten_text(data.get("structure_overview", ""), 120 if is_fast else 180),
        "emotional_curve": shorten_text(data.get("emotional_curve", ""), 80 if is_fast else 108),
        "story_points": story_points,
        "director_notes": [shorten_text(x, 64 if is_fast else 84) for x in list(data.get("director_notes") or [])[:2 if is_fast else 3]],
    }


def compact_director_checklist_for_storyboard_review(
    data: Mapping[str, Any] | None,
    *,
    profile: str = "normal",
    focus_point_ids: Sequence[str] | None = None,
) -> str:
    profile = normalize_storyboard_profile(profile)
    is_fast = profile == "fast"
    payload = dict(data or {})
    lines: list[str] = []
    episode_title = str(payload.get("episode_title") or "").strip()
    structure_overview = str(payload.get("structure_overview") or "").strip()
    emotional_curve = str(payload.get("emotional_curve") or "").strip()
    story_points = list(payload.get("story_points") or [])

    if episode_title:
        lines.append(f"标题：{episode_title}")
    if structure_overview:
        lines.append(f"结构：{shorten_text(structure_overview, 120 if is_fast else 180)}")
    if emotional_curve:
        lines.append(f"情绪曲线：{shorten_text(emotional_curve, 72 if is_fast else 108)}")
    if story_points:
        lines.append(f"剧情点数量：{len(story_points)}")
        lines.append("逐点校对：")

    focus_ids = {str(item).strip() for item in list(focus_point_ids or []) if str(item).strip()}
    for item in story_points:
        point_id = str(item.get("point_id") or "").strip()
        if focus_ids and point_id not in focus_ids:
            continue
        title = shorten_text(str(item.get("title") or "").strip(), 44 if is_fast else 64)
        primary_purpose = str(item.get("primary_purpose") or "").strip()
        opening = shorten_text(
            str(item.get("entry_state") or item.get("opening_visual_state") or "").strip(),
            36 if is_fast else 54,
        )
        tail = shorten_text(
            str(item.get("exit_state") or item.get("continuity_hook_out") or item.get("closing_visual_state") or "").strip(),
            36 if is_fast else 54,
        )
        dialogue = " / ".join(
            shorten_text(str(entry or ""), 28 if is_fast else 40)
            for entry in list(item.get("key_dialogue_beats") or [])[:1 if is_fast else 2]
            if str(entry or "").strip()
        )
        anchors = " / ".join(
            shorten_text(str(entry or ""), 28 if is_fast else 42)
            for entry in list(item.get("detail_anchor_lines") or [])[:1 if is_fast else 2]
            if str(entry or "").strip()
        )
        parts = [f"{point_id} {title}".strip()]
        if primary_purpose:
            parts.append(f"目的:{primary_purpose}")
        if opening:
            parts.append(f"开场:{opening}")
        if dialogue:
            parts.append(f"对白:{dialogue}")
        if anchors:
            parts.append(f"锚点:{anchors}")
        if tail:
            parts.append(f"收尾:{tail}")
        lines.append("- " + "｜".join(part for part in parts if part))

    return "\n".join(lines).strip()


def compact_genre_reference_bundle_for_storyboard(bundle: Mapping[str, Any], *, profile: str = "normal") -> dict[str, Any]:
    profile = normalize_storyboard_profile(profile)
    is_fast = profile == "fast"
    aggregate_focus = bundle.get("aggregate_focus") or {}
    normalized_adapters = bundle.get("normalized_adapters") or {}
    storyboard_adapter = normalized_adapters.get("storyboard") or {}
    return {
        "selected_genres": list(bundle.get("selected_genres") or [])[:2 if is_fast else 3],
        "stage_adapter_for_storyboard": {
            "visual_priority": [shorten_text(x, 72 if is_fast else 88) for x in list(storyboard_adapter.get("visual_priority") or [])[:3 if is_fast else 4]],
            "beat_templates": [shorten_text(x, 72 if is_fast else 88) for x in list(storyboard_adapter.get("beat_templates") or [])[:3 if is_fast else 4]],
            "dialogue_temperament": [shorten_text(x, 72 if is_fast else 88) for x in list(storyboard_adapter.get("dialogue_temperament") or [])[:2 if is_fast else 3]],
            "continuity_guardrails": [shorten_text(x, 70 if is_fast else 84) for x in list(storyboard_adapter.get("continuity_guardrails") or [])[:3 if is_fast else 4]],
            "negative_patterns": [shorten_text(x, 70 if is_fast else 84) for x in list(storyboard_adapter.get("negative_patterns") or [])[:3 if is_fast else 4]],
        },
        "aggregate_focus": {
            "storyboard_focus": [shorten_text(x, 72 if is_fast else 88) for x in list(aggregate_focus.get("storyboard_focus") or [])[:3 if is_fast else 4]],
            "dialogue_timing_rules": [shorten_text(x, 72 if is_fast else 88) for x in list(aggregate_focus.get("dialogue_timing_rules") or [])[:3 if is_fast else 4]],
            "continuity_guardrails": [shorten_text(x, 70 if is_fast else 84) for x in list(aggregate_focus.get("continuity_guardrails") or [])[:3 if is_fast else 4]],
            "negative_patterns": [shorten_text(x, 70 if is_fast else 84) for x in list(aggregate_focus.get("negative_patterns") or [])[:3 if is_fast else 4]],
        },
    }


def compact_genre_reference_bundle_for_storyboard_review(bundle: Mapping[str, Any], *, profile: str = "normal") -> dict[str, Any]:
    profile = normalize_storyboard_profile(profile)
    is_fast = profile == "fast"
    aggregate_focus = bundle.get("aggregate_focus") or {}
    normalized_adapters = bundle.get("normalized_adapters") or {}
    storyboard_adapter = normalized_adapters.get("storyboard") or {}
    matched_packages = []
    for item in list(bundle.get("matched_packages") or [])[:1]:
        item_adapter = dict(item.get("genre_adapter") or {})
        matched_packages.append(
            {
                "genre_key": item.get("genre_key", ""),
                "storyboard_focus": [shorten_text(x, 68 if is_fast else 82) for x in list(item.get("storyboard_focus") or [])[:2 if is_fast else 3]],
                "dialogue_timing_rules": [shorten_text(x, 68 if is_fast else 82) for x in list(item.get("dialogue_timing_rules") or [])[:2 if is_fast else 3]],
                "continuity_guardrails": [shorten_text(x, 68 if is_fast else 82) for x in list(item.get("continuity_guardrails") or [])[:2 if is_fast else 3]],
                "negative_patterns": [shorten_text(x, 68 if is_fast else 82) for x in list(item.get("negative_patterns") or [])[:2 if is_fast else 3]],
                "storyboard_adapter": {
                    "visual_priority": [shorten_text(x, 68 if is_fast else 82) for x in list(item_adapter.get("visual_priority") or [])[:2 if is_fast else 3]],
                    "beat_templates": [shorten_text(x, 68 if is_fast else 82) for x in list(item_adapter.get("beat_templates") or [])[:2 if is_fast else 3]],
                    "dialogue_temperament": [shorten_text(x, 68 if is_fast else 82) for x in list(item_adapter.get("dialogue_temperament") or [])[:2 if is_fast else 3]],
                    "preferred_stage_types": [shorten_text(x, 68 if is_fast else 82) for x in list(item_adapter.get("preferred_stage_types") or [])[:2 if is_fast else 3]],
                },
            }
        )
    return {
        "series_name": bundle.get("series_name", ""),
        "target_medium": bundle.get("target_medium", ""),
        "stage_adapter_for_storyboard": {
            "visual_priority": [shorten_text(x, 68 if is_fast else 82) for x in list(storyboard_adapter.get("visual_priority") or [])[:3 if is_fast else 4]],
            "beat_templates": [shorten_text(x, 68 if is_fast else 82) for x in list(storyboard_adapter.get("beat_templates") or [])[:3 if is_fast else 4]],
            "dialogue_temperament": [shorten_text(x, 68 if is_fast else 82) for x in list(storyboard_adapter.get("dialogue_temperament") or [])[:3 if is_fast else 4]],
            "preferred_stage_types": [shorten_text(x, 68 if is_fast else 82) for x in list(storyboard_adapter.get("preferred_stage_types") or [])[:2 if is_fast else 3]],
            "continuity_guardrails": [shorten_text(x, 68 if is_fast else 82) for x in list(storyboard_adapter.get("continuity_guardrails") or [])[:3 if is_fast else 4]],
            "negative_patterns": [shorten_text(x, 68 if is_fast else 82) for x in list(storyboard_adapter.get("negative_patterns") or [])[:3 if is_fast else 4]],
        },
        "aggregate_focus": {
            "storyboard_focus": [shorten_text(x, 68 if is_fast else 82) for x in list(aggregate_focus.get("storyboard_focus") or [])[:3 if is_fast else 5]],
            "dialogue_timing_rules": [shorten_text(x, 68 if is_fast else 82) for x in list(aggregate_focus.get("dialogue_timing_rules") or [])[:3 if is_fast else 4]],
            "continuity_guardrails": [shorten_text(x, 68 if is_fast else 82) for x in list(aggregate_focus.get("continuity_guardrails") or [])[:3 if is_fast else 4]],
            "negative_patterns": [shorten_text(x, 68 if is_fast else 82) for x in list(aggregate_focus.get("negative_patterns") or [])[:3 if is_fast else 4]],
        },
        "matched_packages": matched_packages,
    }


def compact_asset_catalog_for_storyboard(catalog: list[Mapping[str, str]], *, profile: str = "normal") -> list[dict[str, str]]:
    profile = normalize_storyboard_profile(profile)
    is_fast = profile == "fast"
    compact = []
    characters = [item for item in list(catalog) if str(item.get("asset_type") or "") == "人物参考"]
    scenes = [item for item in list(catalog) if str(item.get("asset_type") or "") == "场景参考"]
    selected = characters[:12 if is_fast else 16] + scenes[:8 if is_fast else 12]
    for item in selected:
        compact.append({
            "ref_id": item.get("ref_id", ""),
            "asset_type": item.get("asset_type", ""),
            "display_name": shorten_text(item.get("display_name", ""), 72 if is_fast else 90),
            "lookup_name": shorten_text(item.get("lookup_name", ""), 44 if is_fast else 60),
        })
    return compact


def compact_asset_catalog_for_storyboard_review(catalog: list[Mapping[str, str]], *, profile: str = "normal") -> list[dict[str, str]]:
    profile = normalize_storyboard_profile(profile)
    is_fast = profile == "fast"
    compact = []
    characters = [item for item in list(catalog) if str(item.get("asset_type") or "") == "人物参考"]
    scenes = [item for item in list(catalog) if str(item.get("asset_type") or "") == "场景参考"]
    selected = characters[:8 if is_fast else 12] + scenes[:6 if is_fast else 9]
    for item in selected:
        compact.append(
            {
                "ref_id": item.get("ref_id", ""),
                "asset_type": item.get("asset_type", ""),
                "display_name": shorten_text(item.get("display_name", ""), 60 if is_fast else 72),
                "lookup_name": shorten_text(item.get("lookup_name", ""), 36 if is_fast else 48),
            }
        )
    return compact


def compact_prompt_markdown_for_storyboard(text: str, limit: int = 2800) -> str:
    return compact_reference_text(text, limit)


def compact_storyboard_draft_package_for_review(
    data: Mapping[str, Any] | None,
    *,
    profile: str = "normal",
    focus_point_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    profile = normalize_storyboard_profile(profile)
    is_fast = profile == "fast"
    data = dict(data or {})
    focus_ids = {str(item).strip() for item in list(focus_point_ids or []) if str(item).strip()}
    prompt_entries = []
    for item in list(data.get("prompt_entries") or []):
        point_id = str(item.get("point_id") or "").strip()
        if focus_ids and point_id not in focus_ids:
            continue
        prompt_entries.append(
            {
                "point_id": point_id,
                "title": shorten_text(item.get("title", ""), 64 if is_fast else 108),
                "pace_label": item.get("pace_label", ""),
                "density_strategy": shorten_text(item.get("density_strategy", ""), 40 if is_fast else 64),
                "duration_hint": shorten_text(item.get("duration_hint", ""), 40),
                "continuity_bridge": shorten_text(item.get("continuity_bridge", ""), 40 if is_fast else 72),
                "primary_refs": list(item.get("primary_refs") or [])[:2 if is_fast else 5],
                "secondary_refs": list(item.get("secondary_refs") or [])[:1 if is_fast else 4],
                "master_timeline": [
                    {
                        "start_second": entry.get("start_second", 0),
                        "end_second": entry.get("end_second", 0),
                        "visual_beat": shorten_text(entry.get("visual_beat", ""), 48 if is_fast else 104),
                        "speaker": shorten_text(entry.get("speaker", ""), 16),
                        "spoken_line": shorten_text(entry.get("spoken_line", ""), 24 if is_fast else 56),
                        "dialogue_blocks": [
                            {
                                "speaker": shorten_text(dialogue.get("speaker", ""), 16),
                                "line": shorten_text(dialogue.get("line", ""), 24 if is_fast else 56),
                                "start_second": dialogue.get("start_second", 0),
                                "end_second": dialogue.get("end_second", 0),
                            }
                            for dialogue in list(entry.get("dialogue_blocks") or [])[:1 if is_fast else 3]
                            if isinstance(dialogue, Mapping)
                        ],
                    }
                    for entry in list(item.get("master_timeline") or [])[:2 if is_fast else 4]
                    if isinstance(entry, Mapping)
                ],
                "prompt_text": shorten_text(item.get("prompt_text", ""), 110 if is_fast else 260),
            }
        )
    return {
        "episode_id": data.get("episode_id", ""),
        "episode_title": data.get("episode_title", ""),
        "materials_overview": shorten_text(data.get("materials_overview", ""), 100 if is_fast else 220),
        "prompt_entries": prompt_entries,
        "global_notes": [shorten_text(x, 44 if is_fast else 88) for x in list(data.get("global_notes") or [])[:2 if is_fast else 4]],
    }
