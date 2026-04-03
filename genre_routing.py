from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from providers.base import EpisodeInputBundle, coerce_mapping, ensure_object_field
from skill_utils import load_skill


PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_SKILL_ROOT = PROJECT_ROOT / "skills" / "production" / "video-script-reconstruction-skill"
CORE_SKILL_PATH = VIDEO_SKILL_ROOT / "SKILL.md"
GENRE_PACKAGE_ROOT = VIDEO_SKILL_ROOT / "genres"


@dataclass
class GenreRoutingResolution:
    core_skill_path: Path
    core_skill_text: str
    genre_skill_paths: list[Path]
    genre_skill_texts: list[str]
    playbook_library_path: Path
    matched_playbooks: list[dict[str, Any]]
    route_tokens: list[str]
    route_sources: list[str]
    route_mode: str

    def combined_skill_text(self) -> str:
        parts = [self.core_skill_text.strip()]
        for item in self.genre_skill_texts:
            text = item.strip()
            if text:
                parts.append(text)
        return "\n\n".join(item for item in parts if item).strip()

    def playbook_reference_text(self) -> str:
        if self.matched_playbooks:
            return json.dumps(self.matched_playbooks, ensure_ascii=False, indent=2)
        library = load_genre_playbook_library()
        return json.dumps(library, ensure_ascii=False, indent=2)

    def routing_note_text(self) -> str:
        source_text = "；".join(self.route_sources) if self.route_sources else "无明确题材先验，仅使用基础方法论"
        token_text = "、".join(self.route_tokens) if self.route_tokens else "未命中"
        if self.matched_playbooks:
            playbook_text = "、".join(str(item.get("genre_key", "")).strip() for item in self.matched_playbooks if str(item.get("genre_key", "")).strip())
            return (
                f"预路由模式：{self.route_mode}。"
                f"预判题材线索：{token_text}。"
                f"来源：{source_text}。"
                f"将优先参考这些题材经验：{playbook_text}。"
                "这些题材提示只作为先验，若与当前视频证据冲突，必须以当前视频为准。"
            )
        return (
            f"预路由模式：{self.route_mode}。"
            f"预判题材线索：{token_text}。"
            f"来源：{source_text}。"
            "未提前命中特定题材，本次使用基础方法论 + 全量题材经验库作为参考；若分析后识别到明确题材，将在连续性层固化。"
        )


def _normalize_identity(raw: str) -> str:
    clean = re.sub(r"\s+", "", (raw or "").strip()).lower()
    clean = re.sub(r"[^\w\u4e00-\u9fff]+", "", clean)
    return clean


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        text = (item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def extract_confirmed_user_genres(bundle: EpisodeInputBundle) -> list[str]:
    values: list[str] = []
    values.extend(str(item).strip() for item in bundle.metadata.get("user_genre_hints", []))
    values.extend(str(item).strip() for item in bundle.metadata.get("user_custom_genre_hints", []))
    return _unique_strings(values)


def build_confirmed_genre_block(bundle: EpisodeInputBundle) -> str:
    confirmed = extract_confirmed_user_genres(bundle)
    if not confirmed:
        return "本次没有用户强锁定题材，可自由判断，但仍应优先使用题材库中的标准标签。"
    return (
        "用户已确认本剧题材为："
        + "、".join(confirmed)
        + "。请严格遵守这些标签：`primary_genre` 必须从这里面精确选择一个，"
        "`secondary_genres` 也只能从这里面精确选择。若你发现更强的新题材，只能写到 `genre_override_request`。"
    )


def load_genre_playbook_library() -> list[dict[str, Any]]:
    if not GENRE_PACKAGE_ROOT.exists():
        return []
    result: list[dict[str, Any]] = []
    for path in sorted(GENRE_PACKAGE_ROOT.iterdir(), key=lambda item: item.name):
        playbook_path = path / "playbook.json"
        if not path.is_dir() or not playbook_path.exists():
            continue
        with playbook_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            continue
        result.append(data)
    return result


def load_genre_package_map() -> dict[str, dict[str, Any]]:
    if not GENRE_PACKAGE_ROOT.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(GENRE_PACKAGE_ROOT.iterdir(), key=lambda item: item.name):
        playbook_path = path / "playbook.json"
        skill_path = path / "skill.md"
        if not path.is_dir() or not playbook_path.exists():
            continue
        with playbook_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            continue
        genre_key = str(data.get("genre_key", "")).strip()
        if not genre_key:
            continue
        package_data = dict(data)
        package_data["_package_dir"] = str(path.resolve())
        package_data["_playbook_path"] = str(playbook_path.resolve())
        package_data["_skill_path"] = str(skill_path.resolve()) if skill_path.exists() else ""
        result[genre_key] = package_data
    return result


def _collect_pre_route_tokens(bundle: EpisodeInputBundle) -> tuple[list[str], list[str]]:
    route_tokens: list[str] = []
    route_sources: list[str] = []

    user_genre_hints = list(bundle.metadata.get("user_genre_hints", []))
    if user_genre_hints:
        route_tokens.extend(str(item).strip() for item in user_genre_hints)
        route_sources.append("user_genre_hints")

    user_custom_genre_hints = list(bundle.metadata.get("user_custom_genre_hints", []))
    if user_custom_genre_hints:
        route_tokens.extend(str(item).strip() for item in user_custom_genre_hints)
        route_sources.append("user_custom_genre_hints")

    continuity_context = dict(bundle.metadata.get("continuity_context", {}))
    genre_profile = dict(continuity_context.get("genre_profile", {}))
    if genre_profile:
        route_tokens.extend(
            [
                str(genre_profile.get("primary_genre", "")).strip(),
                str(genre_profile.get("narrative_device", "")).strip(),
                *[str(item).strip() for item in genre_profile.get("secondary_genres", [])],
            ]
        )
        route_sources.append("series_context.genre_profile")

    title = str(bundle.title or "").strip()
    if title:
        route_tokens.append(title)
        route_sources.append("episode_title")

    if bundle.video_path:
        resolved = bundle.resolved_video_path()
        if resolved:
            route_tokens.extend([resolved.stem, resolved.parent.name])
            route_sources.append("video_path")

    if bundle.synopsis_text:
        route_tokens.append(str(bundle.synopsis_text).strip())
        route_sources.append("synopsis_hint")

    return _unique_strings(route_tokens), _unique_strings(route_sources)


def suggest_library_genres_for_series(
    *,
    series_label: str,
    video_dir: str | Path | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    tokens = [str(series_label or "").strip()]
    if video_dir:
        path = Path(video_dir).expanduser().resolve()
        tokens.append(path.name)
        try:
            videos_root = PROJECT_ROOT / "videos"
            tokens.append(str(path.relative_to(videos_root)))
        except Exception:
            pass
    tokens = _unique_strings(tokens)
    matches = _match_playbooks(tokens)
    return matches[:limit]


def _collect_post_route_tokens(
    analysis: Mapping[str, Any],
    series_context: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    route_tokens: list[str] = []
    route_sources: list[str] = []

    genre_profile = dict(analysis.get("genre_classification", {}))
    if genre_profile:
        route_tokens.extend(
            [
                str(genre_profile.get("primary_genre", "")).strip(),
                str(genre_profile.get("narrative_device", "")).strip(),
                *[str(item).strip() for item in genre_profile.get("secondary_genres", [])],
            ]
        )
        route_sources.append("episode_analysis.genre_classification")

    if series_context:
        context_playbooks = series_context.get("genre_playbooks", [])
        if isinstance(context_playbooks, list) and context_playbooks:
            route_tokens.extend(str(item.get("genre_key", "")).strip() for item in context_playbooks if isinstance(item, dict))
            route_sources.append("series_context.genre_playbooks")

    return _unique_strings(route_tokens), _unique_strings(route_sources)


def _match_playbooks(route_tokens: list[str]) -> list[dict[str, Any]]:
    library = list(load_genre_package_map().values())
    if not route_tokens:
        return []
    normalized_tokens = {_normalize_identity(item) for item in route_tokens if _normalize_identity(item)}
    short_tokens = [item.strip() for item in route_tokens if item and len(item.strip()) <= 24]
    scored: list[tuple[int, dict[str, Any]]] = []
    for item in library:
        genre_key = str(item.get("genre_key", "")).strip()
        aliases = [str(alias).strip() for alias in item.get("aliases", [])]
        normalized_keys = {_normalize_identity(genre_key)}
        normalized_keys.update(_normalize_identity(alias) for alias in aliases)
        score = 0
        if normalized_tokens & normalized_keys:
            score += 10
        if any(genre_key and genre_key in token for token in short_tokens):
            score += 3
        if any(alias and alias in token for alias in aliases for token in short_tokens):
            score += 2
        if score > 0:
            scored.append((score, item))
    unique_keys: set[str] = set()
    result: list[dict[str, Any]] = []
    for _, item in sorted(
        scored,
        key=lambda pair: (-pair[0], str(pair[1].get("genre_key", "")).strip()),
    ):
        key = str(item.get("genre_key", "")).strip()
        if not key or key in unique_keys:
            continue
        unique_keys.add(key)
        result.append({k: v for k, v in item.items() if not str(k).startswith("_")})
    return result[:3]


def _load_genre_skill_texts(playbooks: list[dict[str, Any]]) -> tuple[list[Path], list[str]]:
    package_map = load_genre_package_map()
    paths: list[Path] = []
    texts: list[str] = []
    for item in playbooks:
        genre_key = str(item.get("genre_key", "")).strip()
        package = package_map.get(genre_key)
        if not package:
            continue
        raw_path = str(package.get("_skill_path", "")).strip()
        if not raw_path:
            continue
        target = Path(raw_path)
        paths.append(target)
        texts.append(target.read_text(encoding="utf-8"))
    return paths, texts


def _build_allowed_genre_alias_map(allowed_genres: list[str]) -> dict[str, set[str]]:
    package_map = load_genre_package_map()
    result: dict[str, set[str]] = {}
    for genre in allowed_genres:
        aliases = {_normalize_identity(genre)}
        package = package_map.get(genre)
        if package:
            aliases.update(
                _normalize_identity(str(item))
                for item in package.get("aliases", [])
                if str(item).strip()
            )
        result[genre] = {item for item in aliases if item}
    return result


def _match_candidate_to_allowed(candidate: str, alias_map: Mapping[str, set[str]]) -> str | None:
    normalized_candidate = _normalize_identity(candidate)
    if not normalized_candidate:
        return None
    for allowed, aliases in alias_map.items():
        if normalized_candidate in aliases:
            return allowed
    for allowed, aliases in alias_map.items():
        if any(alias and (alias in normalized_candidate or normalized_candidate in alias) for alias in aliases):
            return allowed
    return None


def enforce_user_genre_alignment(bundle: EpisodeInputBundle, analysis: dict[str, Any]) -> dict[str, Any]:
    confirmed = extract_confirmed_user_genres(bundle)
    genre_classification = ensure_object_field(analysis, "genre_classification")
    override_request = ensure_object_field(
        analysis,
        "genre_override_request",
        {
            "needs_user_confirmation": False,
            "proposed_primary_genre": "",
            "proposed_secondary_genres": [],
            "proposed_new_genres": [],
            "reason": "",
        },
    )
    if not confirmed:
        genre_classification.setdefault("confirmed_user_genres", [])
        genre_classification.setdefault("genre_resolution_mode", "freeform")
        override_request.setdefault("needs_user_confirmation", False)
        override_request.setdefault("proposed_primary_genre", "")
        override_request.setdefault("proposed_secondary_genres", [])
        override_request.setdefault("proposed_new_genres", [])
        override_request.setdefault("reason", "")
        return analysis

    original_primary = str(genre_classification.get("primary_genre", "")).strip()
    original_secondary = [
        str(item).strip() for item in genre_classification.get("secondary_genres", []) if str(item).strip()
    ]
    alias_map = _build_allowed_genre_alias_map(confirmed)

    matched_allowed: list[str] = []
    unmatched_original: list[str] = []
    for candidate in [original_primary, *original_secondary]:
        if not candidate:
            continue
        matched = _match_candidate_to_allowed(candidate, alias_map)
        if matched:
            if matched not in matched_allowed:
                matched_allowed.append(matched)
        elif candidate not in unmatched_original:
            unmatched_original.append(candidate)

    final_primary = matched_allowed[0] if matched_allowed else confirmed[0]
    final_secondary = [item for item in matched_allowed[1:] if item != final_primary]
    if not final_secondary:
        final_secondary = [item for item in confirmed if item != final_primary][: max(0, min(2, len(confirmed) - 1))]

    genre_classification["primary_genre"] = final_primary
    genre_classification["secondary_genres"] = final_secondary
    genre_classification["confirmed_user_genres"] = confirmed
    genre_classification["genre_resolution_mode"] = "user_confirmed_locked"

    proposed_new_genres = [
        item
        for item in unmatched_original
        if item and item not in confirmed and item not in final_secondary and item != final_primary
    ][:3]
    override_request["needs_user_confirmation"] = bool(proposed_new_genres)
    override_request["proposed_primary_genre"] = original_primary if original_primary and original_primary != final_primary else ""
    override_request["proposed_secondary_genres"] = [
        item for item in original_secondary if item and item not in final_secondary
    ][:3]
    override_request["proposed_new_genres"] = proposed_new_genres
    override_request["reason"] = (
        "已按用户确认题材锁定 genre_classification；模型提出的额外题材建议已转存到 genre_override_request，需用户确认后才应修改后续题材路由。"
        if proposed_new_genres
        else "已按用户确认题材锁定 genre_classification。"
    )
    return analysis


def resolve_pre_analysis_genre_routing(bundle: EpisodeInputBundle) -> GenreRoutingResolution:
    route_tokens, route_sources = _collect_pre_route_tokens(bundle)
    confirmed = extract_confirmed_user_genres(bundle)
    if confirmed:
        package_map = load_genre_package_map()
        matched_playbooks = [
            {k: v for k, v in package_map[genre].items() if not str(k).startswith("_")}
            for genre in confirmed
            if genre in package_map
        ]
        route_mode = "user_confirmed_locked" if matched_playbooks else "user_confirmed_locked_missing_library"
    else:
        matched_playbooks = _match_playbooks(route_tokens)
        route_mode = "matched_playbooks" if matched_playbooks else "fallback_full_library"
    genre_skill_paths, genre_skill_texts = _load_genre_skill_texts(matched_playbooks)
    return GenreRoutingResolution(
        core_skill_path=CORE_SKILL_PATH.resolve(),
        core_skill_text=load_skill("production/video-script-reconstruction-skill/SKILL.md"),
        genre_skill_paths=genre_skill_paths,
        genre_skill_texts=genre_skill_texts,
        playbook_library_path=GENRE_PACKAGE_ROOT.resolve(),
        matched_playbooks=matched_playbooks,
        route_tokens=route_tokens,
        route_sources=route_sources,
        route_mode=route_mode,
    )


def resolve_post_analysis_genre_routing(
    analysis: Mapping[str, Any],
    *,
    series_context: Mapping[str, Any] | None = None,
) -> GenreRoutingResolution:
    route_tokens, route_sources = _collect_post_route_tokens(analysis, series_context=series_context)
    matched_playbooks = _match_playbooks(route_tokens)
    genre_skill_paths, genre_skill_texts = _load_genre_skill_texts(matched_playbooks)
    return GenreRoutingResolution(
        core_skill_path=CORE_SKILL_PATH.resolve(),
        core_skill_text=load_skill("production/video-script-reconstruction-skill/SKILL.md"),
        genre_skill_paths=genre_skill_paths,
        genre_skill_texts=genre_skill_texts,
        playbook_library_path=GENRE_PACKAGE_ROOT.resolve(),
        matched_playbooks=matched_playbooks,
        route_tokens=route_tokens,
        route_sources=route_sources,
        route_mode="matched_playbooks" if matched_playbooks else "fallback_full_library",
    )


def build_genre_debug_report(
    *,
    bundle: EpisodeInputBundle,
    analysis: Mapping[str, Any],
    series_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pre = resolve_pre_analysis_genre_routing(bundle)
    post = resolve_post_analysis_genre_routing(analysis, series_context=series_context)
    return {
        "episode_id": bundle.episode_id,
        "title": bundle.title or bundle.episode_id,
        "video_path": str(bundle.video_path or ""),
        "core_skill_path": str(pre.core_skill_path),
        "playbook_library_path": str(pre.playbook_library_path),
        "pre_analysis_routing": {
            "route_mode": pre.route_mode,
            "route_tokens": pre.route_tokens,
            "route_sources": pre.route_sources,
            "genre_skill_paths": [str(path) for path in pre.genre_skill_paths],
            "matched_playbook_keys": [str(item.get("genre_key", "")).strip() for item in pre.matched_playbooks],
            "routing_note": pre.routing_note_text(),
        },
        "analysis_genre_classification": coerce_mapping(analysis.get("genre_classification")),
        "analysis_hook_profile": coerce_mapping(analysis.get("hook_profile")),
        "genre_override_request": coerce_mapping(analysis.get("genre_override_request")),
        "post_analysis_routing": {
            "route_mode": post.route_mode,
            "route_tokens": post.route_tokens,
            "route_sources": post.route_sources,
            "genre_skill_paths": [str(path) for path in post.genre_skill_paths],
            "matched_playbook_keys": [str(item.get("genre_key", "")).strip() for item in post.matched_playbooks],
            "routing_note": post.routing_note_text(),
        },
        "downstream_design_guidance": coerce_mapping(analysis.get("downstream_design_guidance")),
        "series_context_snapshot": dict(series_context or {}),
    }


def render_genre_debug_markdown(report: Mapping[str, Any]) -> str:
    pre = dict(report.get("pre_analysis_routing", {}))
    post = dict(report.get("post_analysis_routing", {}))
    lines = [
        "# 题材路由与经验注入调试报告",
        "",
        f"- 集数：{report.get('episode_id', '')}",
        f"- 标题：{report.get('title', '')}",
        f"- 视频：{report.get('video_path', '')}",
        f"- 基础 Skill：{report.get('core_skill_path', '')}",
        f"- Playbook 源目录：{report.get('playbook_library_path', '')}",
        "",
        "## 预分析路由",
        "",
        f"- 模式：{pre.get('route_mode', '')}",
        f"- 题材线索：{'、'.join(pre.get('route_tokens', [])) or '无'}",
        f"- 来源：{'；'.join(pre.get('route_sources', [])) or '无'}",
        f"- 题材 Skill：{'；'.join(pre.get('genre_skill_paths', [])) or '未加载'}",
        f"- 预命中 Playbook：{'、'.join(pre.get('matched_playbook_keys', [])) or '未命中'}",
        f"- 说明：{pre.get('routing_note', '')}",
        "",
        "## 模型分析后的题材判断",
        "",
        f"- 主题材：{report.get('analysis_genre_classification', {}).get('primary_genre', '')}",
        f"- 副题材：{'、'.join(report.get('analysis_genre_classification', {}).get('secondary_genres', []))}",
        f"- 用户确认题材：{'、'.join(report.get('analysis_genre_classification', {}).get('confirmed_user_genres', []))}",
        f"- 题材决议模式：{report.get('analysis_genre_classification', {}).get('genre_resolution_mode', '')}",
        f"- 叙事装置：{report.get('analysis_genre_classification', {}).get('narrative_device', '')}",
        f"- 观众期待：{report.get('analysis_genre_classification', {}).get('audience_expectation', '')}",
        "",
        "## AI 题材修正建议",
        "",
        f"- 是否需要用户确认：{report.get('genre_override_request', {}).get('needs_user_confirmation', False)}",
        f"- 建议主题材：{report.get('genre_override_request', {}).get('proposed_primary_genre', '')}",
        f"- 建议副题材：{'、'.join(report.get('genre_override_request', {}).get('proposed_secondary_genres', [])) or '无'}",
        f"- 建议新增题材：{'、'.join(report.get('genre_override_request', {}).get('proposed_new_genres', [])) or '无'}",
        f"- 说明：{report.get('genre_override_request', {}).get('reason', '')}",
        "",
        "## 分析后路由",
        "",
        f"- 模式：{post.get('route_mode', '')}",
        f"- 题材线索：{'、'.join(post.get('route_tokens', [])) or '无'}",
        f"- 来源：{'；'.join(post.get('route_sources', [])) or '无'}",
        f"- 题材 Skill：{'；'.join(post.get('genre_skill_paths', [])) or '未加载'}",
        f"- 最终命中 Playbook：{'、'.join(post.get('matched_playbook_keys', [])) or '未命中'}",
        f"- 说明：{post.get('routing_note', '')}",
        "",
        "## 下游设计指导",
        "",
        f"- 剧本重建重点：{'；'.join(report.get('downstream_design_guidance', {}).get('script_reconstruction_focus', [])) or '无'}",
        f"- 人物设计重点：{'；'.join(report.get('downstream_design_guidance', {}).get('character_design_focus', [])) or '无'}",
        f"- 场景设计重点：{'；'.join(report.get('downstream_design_guidance', {}).get('scene_design_focus', [])) or '无'}",
        f"- 分镜重点：{'；'.join(report.get('downstream_design_guidance', {}).get('storyboard_focus', [])) or '无'}",
        "",
    ]
    return "\n".join(lines)
