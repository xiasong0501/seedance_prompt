from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = PROJECT_ROOT / "skills" / "production" / "video-script-reconstruction-skill"
GENRES_ROOT = SKILL_ROOT / "genres"
GENRE_DRAFT_ROOT = GENRES_ROOT / "__drafts__"
LEGACY_PLAYBOOK_DRAFT_ROOT = SKILL_ROOT / "playbooks" / "__drafts__"
STOPWORDS = {
    "优先",
    "尽快",
    "适合",
    "强调",
    "体现",
    "处理",
    "设计",
    "分析",
    "剧本",
    "角色",
    "人物",
    "场景",
    "镜头",
    "画面",
    "空间",
    "关系",
    "情绪",
    "台词",
    "动作",
    "最好",
    "应当",
    "可以",
    "需要",
}
SOURCE_BASE_CONFIDENCE = {
    "character_design_rules": 0.82,
    "costume_makeup_rules": 0.80,
    "scene_design_rules": 0.80,
    "camera_language_rules": 0.81,
    "storyboard_execution_rules": 0.83,
    "dialogue_timing_rules": 0.79,
    "continuity_guardrails": 0.84,
    "negative_patterns": 0.78,
    "reusable_playbook_rules": 0.76,
    "reusable_skill_rules": 0.74,
    "character_appeal_patterns": 0.71,
    "scene_staging_patterns": 0.73,
    "dialogue_patterns": 0.69,
}
SOURCE_DISPLAY = {
    "character_design_rules": "人物设计规则",
    "costume_makeup_rules": "服化道统一规则",
    "scene_design_rules": "场景设计规则",
    "camera_language_rules": "镜头语言规则",
    "storyboard_execution_rules": "分镜执行规则",
    "dialogue_timing_rules": "台词时间规则",
    "continuity_guardrails": "连续性红线",
    "negative_patterns": "高风险负面模式",
    "reusable_playbook_rules": "整剧玩法规则",
    "reusable_skill_rules": "整剧方法规则",
    "character_appeal_patterns": "人物吸引力模式",
    "scene_staging_patterns": "场景调度模式",
    "dialogue_patterns": "对白模式",
}
FIELD_DISPLAY = {
    "director_focus": "导演重点",
    "script_hooks": "剧情抓点",
    "character_focus": "人物设计重点",
    "costume_focus": "服化道重点",
    "scene_focus": "场景设计重点",
    "storyboard_focus": "分镜重点",
    "dialogue_timing_rules": "台词时间规则",
    "continuity_guardrails": "连续性红线",
    "negative_patterns": "负面模式",
    "character_design_focus": "人物设计重点（旧）",
    "scene_design_focus": "场景设计重点（旧）",
}
SECTION_DISPLAY = {
    "结构与钩子": "结构与钩子",
    "人物塑造": "人物塑造",
    "服化道统一": "服化道统一",
    "场景与调度": "场景与调度",
    "对白与节奏": "对白与节奏",
    "分镜表达": "分镜表达",
    "生产红线": "生产红线",
}


def print_status(message: str) -> None:
    print(f"[series-learning-sync] {message}", flush=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, data: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return target


def save_text(path: str | Path, text: str) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
    return target


def slugify(value: str) -> str:
    clean = re.sub(r"\s+", "_", str(value).strip())
    clean = re.sub(r"[\\\\/:*?\"<>|（）()\[\]{}，,。；;！!？?]+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean[:48] or "genre"


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result

def load_optional_json(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return {}
    return load_json(target)


def collect_series_specific_tokens(
    *,
    series_name: str,
    series_bible: dict[str, Any] | None,
    character_registry: dict[str, Any] | None,
    location_registry: dict[str, Any] | None,
) -> list[str]:
    tokens: list[str] = []
    if series_name:
        tokens.append(series_name)
    if series_bible:
        tokens.extend([str(item).strip() for item in series_bible.get("active_characters", [])])
        tokens.extend([str(item).strip() for item in series_bible.get("recurring_locations", [])])
    if character_registry:
        for item in character_registry.get("characters", []):
            tokens.append(str(item.get("canonical_name", "")).strip())
            tokens.extend([str(alias).strip() for alias in item.get("aliases", [])])
    if location_registry:
        for item in location_registry.get("locations", []):
            tokens.append(str(item.get("canonical_name", "")).strip())
    cleaned = []
    for token in tokens:
        token = token.strip()
        if len(token) < 2:
            continue
        if token.startswith('ep') and token[2:].isdigit():
            continue
        cleaned.append(token)
    return unique(cleaned)


def find_series_specific_matches(text: str, tokens: list[str], limit: int = 4) -> list[str]:
    raw = str(text)
    matches: list[str] = []
    for token in tokens:
        if token and token in raw and token not in matches:
            matches.append(token)
        if len(matches) >= limit:
            break
    return matches


def generic_role_hint_present(text: str) -> bool:
    raw = str(text)
    generic_terms = (
        '女主', '男主', '反派', '女配', '男配', '长辈', '掌权者', '保镖', '高手', '主角', '配角',
        '公开空间', '封闭空间', '车厢', '厅堂', '府邸', '宅院', '高位者', '上位者', '羞辱场',
        '认亲场', '重逢场', '冲突场', '群像场', '公开场', '仪式场'
    )
    return any(term in raw for term in generic_terms)


def apply_series_specific_penalty(
    *,
    text: str,
    confidence: float,
    reason: str,
    series_specific_tokens: list[str],
) -> tuple[float, str, list[str]]:
    matches = find_series_specific_matches(text, series_specific_tokens)
    if not matches:
        return confidence, reason, []
    penalty = 0.28 + max(0, len(matches) - 1) * 0.04
    adjusted = max(0.0, round(confidence - penalty, 2))
    if not generic_role_hint_present(text):
        adjusted = min(adjusted, 0.45)
    extra = f" 含本剧专属名字/场景名“{'、'.join(matches)}”，泛化能力较弱，已自动降权。"
    return adjusted, reason + extra, matches


def shorten_text(text: str, max_len: int = 30) -> str:
    clean = re.sub(r"\s+", "", str(text).strip())
    if not clean:
        return ""
    sentence_parts = [item.strip() for item in re.split(r"[。；;]", clean) if item.strip()]
    primary = sentence_parts[0] if sentence_parts else clean
    clause_parts = [item.strip() for item in re.split(r"[，,]", primary) if item.strip()]
    if len(primary) > max_len and clause_parts:
        primary = "，".join(clause_parts[:2])
    if len(primary) > max_len:
        primary = primary[: max_len - 1].rstrip("，,、；;：:") + "…"
    return primary


def extract_keywords(texts: list[str]) -> list[str]:
    keywords: list[str] = []
    for text in texts:
        raw = str(text).strip()
        if not raw:
            continue
        pieces = [item.strip() for item in re.split(r"[\n，,。；;：:、/|\s]+", raw) if item.strip()]
        pieces.append(raw)
        for piece in pieces:
            token = piece.strip(" -")
            if not token or token in STOPWORDS:
                continue
            if len(token) < 2 or len(token) > 16:
                continue
            if token not in keywords:
                keywords.append(token)
    keywords.sort(key=len, reverse=True)
    return keywords


def build_package_profile(package: dict[str, Any] | None) -> dict[str, Any]:
    if not package:
        return {
            "aliases": [],
            "all_keywords": [],
            "lens_keywords": [],
            "field_keywords": {},
        }
    playbook = dict(package.get("playbook", {}))
    skill_text = str(package.get("skill_text", ""))
    aliases = unique([str(playbook.get("genre_key", "")).strip(), *playbook.get("aliases", [])])

    def field_values(*names: str) -> list[str]:
        values: list[str] = []
        for name in names:
            values.extend(list(playbook.get(name, [])))
        return values

    field_keywords = {
        "director_focus": extract_keywords(
            aliases + field_values("director_focus", "script_hooks", "storyboard_focus") + skill_text.splitlines()
        ),
        "script_hooks": extract_keywords(
            aliases + field_values("core_audience_promises", "script_hooks") + skill_text.splitlines()
        ),
        "character_focus": extract_keywords(
            aliases + field_values("character_focus", "character_design_focus") + skill_text.splitlines()
        ),
        "costume_focus": extract_keywords(
            aliases + field_values("costume_focus", "character_design_focus", "scene_design_focus") + skill_text.splitlines()
        ),
        "scene_focus": extract_keywords(
            aliases + field_values("scene_focus", "scene_design_focus") + skill_text.splitlines()
        ),
        "storyboard_focus": extract_keywords(
            aliases + field_values("storyboard_focus") + skill_text.splitlines()
        ),
        "dialogue_timing_rules": extract_keywords(
            aliases + field_values("dialogue_timing_rules", "script_hooks") + skill_text.splitlines()
        ),
        "continuity_guardrails": extract_keywords(
            aliases + field_values("continuity_guardrails") + skill_text.splitlines()
        ),
        "negative_patterns": extract_keywords(
            aliases + field_values("negative_patterns") + skill_text.splitlines()
        ),
        "character_design_focus": extract_keywords(
            aliases + field_values("character_design_focus") + skill_text.splitlines()
        ),
        "scene_design_focus": extract_keywords(
            aliases + field_values("scene_design_focus") + skill_text.splitlines()
        ),
    }
    all_keywords = extract_keywords(
        aliases
        + field_values(
            "core_audience_promises",
            "script_hooks",
            "director_focus",
            "character_focus",
            "costume_focus",
            "scene_focus",
            "storyboard_focus",
            "dialogue_timing_rules",
            "continuity_guardrails",
            "negative_patterns",
            "character_design_focus",
            "scene_design_focus",
        )
        + skill_text.splitlines()
    )
    lens_keywords = extract_keywords(
        aliases
        + field_values("core_audience_promises", "script_hooks", "director_focus", "storyboard_focus")
        + skill_text.splitlines()[:20]
    )
    return {
        "aliases": aliases,
        "all_keywords": all_keywords,
        "lens_keywords": lens_keywords,
        "field_keywords": field_keywords,
    }


def classify_skill_section(text: str, source_type: str) -> str:
    raw = str(text)
    if source_type in ("character_design_rules", "character_appeal_patterns"):
        return "人物塑造"
    if source_type == "costume_makeup_rules":
        return "服化道统一"
    if source_type in ("scene_design_rules", "scene_staging_patterns"):
        return "场景与调度"
    if source_type in ("dialogue_timing_rules", "dialogue_patterns"):
        return "对白与节奏"
    if source_type in ("camera_language_rules", "storyboard_execution_rules"):
        return "分镜表达"
    if source_type in ("continuity_guardrails", "negative_patterns"):
        return "生产红线"
    if any(keyword in raw for keyword in ("妆", "服", "发", "配饰", "材质", "色彩")):
        return "服化道统一"
    if any(keyword in raw for keyword in ("镜头", "画面", "调度", "特写", "蒙太奇", "卡点", "转场")):
        return "分镜表达"
    if any(keyword in raw for keyword in ("台词", "对白", "旁白", "切条", "停顿", "留白", "音效")):
        return "对白与节奏"
    if any(keyword in raw for keyword in ("女主", "男主", "长辈", "人物", "吸引力", "魅力", "造型")):
        return "人物塑造"
    if any(keyword in raw for keyword in ("厅堂", "卧房", "院内", "空间", "道具", "站位", "布景", "背景", "环境")):
        return "场景与调度"
    if any(keyword in raw for keyword in ("不要", "避免", "不能", "红线", "一致性")):
        return "生产红线"
    return "结构与钩子"


def keywords_in_text(text: str, keywords: list[str], limit: int = 4) -> list[str]:
    matches: list[str] = []
    raw = str(text)
    for keyword in keywords:
        if keyword and keyword in raw and keyword not in matches:
            matches.append(keyword)
        if len(matches) >= limit:
            break
    return matches


def score_candidate(
    *,
    text: str,
    source_type: str,
    target_field: str | None,
    section: str | None,
    profile: dict[str, Any],
) -> tuple[float, list[str]]:
    base = SOURCE_BASE_CONFIDENCE.get(source_type, 0.68)
    if target_field:
        field_keywords = list(profile.get("field_keywords", {}).get(target_field, []))
    elif section == "人物塑造":
        field_keywords = list(profile.get("field_keywords", {}).get("character_focus", []))
    elif section == "服化道统一":
        field_keywords = list(profile.get("field_keywords", {}).get("costume_focus", []))
    elif section == "场景与调度":
        field_keywords = list(profile.get("field_keywords", {}).get("scene_focus", []))
    elif section == "分镜表达":
        field_keywords = list(profile.get("field_keywords", {}).get("storyboard_focus", []))
    elif section == "对白与节奏":
        field_keywords = list(profile.get("field_keywords", {}).get("dialogue_timing_rules", []))
    elif section == "生产红线":
        field_keywords = list(profile.get("field_keywords", {}).get("continuity_guardrails", []))
        field_keywords += list(profile.get("field_keywords", {}).get("negative_patterns", []))
    else:
        field_keywords = list(profile.get("field_keywords", {}).get("script_hooks", []))
    matched_field = keywords_in_text(text, field_keywords)
    matched_all = keywords_in_text(text, list(profile.get("all_keywords", [])), limit=6)
    matched_alias = keywords_in_text(text, list(profile.get("aliases", [])), limit=2)
    score = base
    score += min(0.16, 0.05 * len(matched_field))
    score += min(0.08, 0.02 * max(0, len(matched_all) - len(matched_field)))
    if matched_alias:
        score += 0.05
    if profile.get("all_keywords") and not matched_all:
        score -= 0.12
    if section == "场景与调度" and source_type in ("scene_staging_patterns", "scene_design_rules"):
        score += 0.03
    if section == "人物塑造" and source_type in ("character_appeal_patterns", "character_design_rules"):
        score += 0.03
    if section == "服化道统一" and source_type == "costume_makeup_rules":
        score += 0.03
    if section == "生产红线" and source_type in ("continuity_guardrails", "negative_patterns"):
        score += 0.04
    if target_field == "script_hooks" and source_type == "reusable_playbook_rules":
        score += 0.03
    return max(0.0, min(1.0, round(score, 2))), unique([*matched_field, *matched_all, *matched_alias])[:4]


def build_reason(source_type: str, matched_keywords: list[str]) -> str:
    source_label = SOURCE_DISPLAY.get(source_type, source_type)
    if matched_keywords:
        return f"来自{source_label}，且与当前题材关键词“{'、'.join(matched_keywords)}”高度相关。"
    return f"来自{source_label}，但与当前题材的显式关键词匹配较弱，需人工确认。"


def apply_genre_specificity_adjustment(
    *,
    text: str,
    source_type: str,
    target_field: str | None,
    section: str | None,
    genre_key: str,
    profile: dict[str, Any],
    genre_profiles: dict[str, dict[str, Any]],
    confidence: float,
    matched_keywords: list[str],
    reason: str,
) -> tuple[float, str, str, float, float]:
    if len(genre_profiles) <= 1:
        return round(confidence, 2), reason, "", 0.0, 1.0

    competitor_snapshots: list[dict[str, Any]] = []
    for other_genre, other_profile in genre_profiles.items():
        if other_genre == genre_key:
            continue
        other_confidence, other_keywords = score_candidate(
            text=text,
            source_type=source_type,
            target_field=target_field,
            section=section,
            profile=other_profile,
        )
        competitor_snapshots.append(
            {
                "genre_key": other_genre,
                "confidence": other_confidence,
                "matched_keywords": other_keywords,
            }
        )
    if not competitor_snapshots:
        return round(confidence, 2), reason, "", 0.0, 1.0

    best_competitor = max(
        competitor_snapshots,
        key=lambda item: (float(item.get("confidence", 0.0)), len(item.get("matched_keywords", []))),
    )
    best_genre = str(best_competitor.get("genre_key", "")).strip()
    best_confidence = float(best_competitor.get("confidence", 0.0))
    gap = round(confidence - best_confidence, 2)
    adjusted = confidence

    current_alias_hits = keywords_in_text(text, list(profile.get("aliases", [])), limit=2)
    lens_matches = keywords_in_text(text, list(profile.get("lens_keywords", [])), limit=3)
    competitor_alias_hits = keywords_in_text(
        text,
        list(genre_profiles.get(best_genre, {}).get("aliases", [])),
        limit=2,
    )
    if lens_matches:
        adjusted += 0.03
        reason += f" 命中了当前题材叙事核心词“{'、'.join(lens_matches)}”。"
    else:
        adjusted -= 0.10
        reason += " 缺少当前题材叙事核心词命中，题材归属偏泛。"

    if current_alias_hits and not competitor_alias_hits:
        adjusted += 0.04
        reason += f" 当前题材别名命中“{'、'.join(current_alias_hits)}”，归属更明确。"
    if len(genre_profiles) > 1 and not lens_matches and not current_alias_hits:
        adjusted = min(adjusted, 0.64)
        reason += " 未命中当前题材核心叙事词，也未命中题材别名，按跨题材泛化规则降权。"

    if best_confidence > confidence + 0.02:
        adjusted -= 0.22
        adjusted = min(adjusted, 0.64)
        reason += f" 但它与“{best_genre}”更贴合（对比得分 {best_confidence:.2f}）。"
    elif gap < 0.04:
        adjusted -= 0.14
        reason += f" 它与“{best_genre}”的区分度很弱（差值 {gap:.2f}）。"
    elif gap < 0.08:
        adjusted -= 0.08
        reason += f" 它与“{best_genre}”仍有明显重叠（差值 {gap:.2f}）。"

    if not matched_keywords and best_confidence >= confidence:
        adjusted -= 0.04

    adjusted = max(0.0, min(1.0, round(adjusted, 2)))
    return adjusted, reason, best_genre, round(best_confidence, 2), gap


def rank_and_trim_candidates(
    candidates: list[dict[str, Any]],
    *,
    group_key: str,
    min_items: int,
    max_items: int,
    min_confidence: float,
    medium_confidence_threshold: float,
    high_confidence_threshold: float,
    default_group_limit: int,
    high_confidence_group_limit: int,
    fallback_count: int = 2,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            float(item.get("confidence", 0.0)),
            float(item.get("genre_confidence_gap", 0.0)),
            len(str(item.get("matched_keywords", []))),
            len(str(item.get("full_text", ""))),
        ),
        reverse=True,
    )
    eligible_scores = [
        float(item.get("confidence", 0.0))
        for item in sorted_candidates
        if float(item.get("confidence", 0.0)) >= min_confidence
    ]
    dynamic_budget = float(min_items)
    for score in eligible_scores:
        if score >= high_confidence_threshold:
            dynamic_budget += 1.0
        elif score >= medium_confidence_threshold:
            dynamic_budget += 0.5
        elif score >= min_confidence + 0.05:
            dynamic_budget += 0.25
    target_budget = max(min_items, min(max_items, int(math.ceil(dynamic_budget))))
    groups_in_budget = {
        str(item.get(group_key, "")).strip() or "_default"
        for item in sorted_candidates
        if float(item.get("confidence", 0.0)) >= min_confidence
    }
    dynamic_group_floor = max(1, int(math.ceil(target_budget / max(1, len(groups_in_budget)))))
    dynamic_group_high_floor = max(dynamic_group_floor, dynamic_group_floor + 1)

    grouped_limits: dict[str, int] = {}
    result: list[dict[str, Any]] = []
    used_texts: set[str] = set()
    for item in sorted_candidates:
        short_text = str(item.get("text", "")).strip()
        if not short_text or short_text in used_texts:
            continue
        if float(item.get("confidence", 0.0)) < min_confidence:
            continue
        group = str(item.get(group_key, "")).strip() or "_default"
        group_limit = (
            high_confidence_group_limit
            if float(item.get("confidence", 0.0)) >= high_confidence_threshold
            else default_group_limit
        )
        if float(item.get("confidence", 0.0)) >= high_confidence_threshold:
            group_limit = max(group_limit, dynamic_group_high_floor)
        else:
            group_limit = max(group_limit, dynamic_group_floor)
        if grouped_limits.get(group, 0) >= group_limit:
            continue
        used_texts.add(short_text)
        grouped_limits[group] = grouped_limits.get(group, 0) + 1
        result.append(item)
        if len(result) >= target_budget:
            break
    if result:
        return result
    for item in sorted_candidates[:fallback_count]:
        short_text = str(item.get("text", "")).strip()
        if not short_text or short_text in used_texts:
            continue
        used_texts.add(short_text)
        result.append(item)
    return result


def prune_cross_genre_candidate_overlap(
    items: list[dict[str, Any]],
    *,
    candidate_field: str,
    bucket_key: str,
) -> None:
    owners: dict[str, tuple[str, tuple[float, float, int]]] = {}
    for item in items:
        genre_key = str(item.get("genre_key", "")).strip()
        for candidate in item.get(candidate_field, []):
            text = str(candidate.get("full_text") or candidate.get("text", "")).strip()
            bucket = str(candidate.get(bucket_key, "")).strip()
            if not text:
                continue
            identity = f"{bucket}::{text}"
            rank = (
                float(candidate.get("confidence", 0.0)),
                float(candidate.get("genre_confidence_gap", 0.0)),
                len(candidate.get("matched_keywords", [])),
            )
            current_owner = owners.get(identity)
            if current_owner is None or rank > current_owner[1]:
                owners[identity] = (genre_key, rank)

    for item in items:
        genre_key = str(item.get("genre_key", "")).strip()
        filtered: list[dict[str, Any]] = []
        for candidate in item.get(candidate_field, []):
            text = str(candidate.get("full_text") or candidate.get("text", "")).strip()
            bucket = str(candidate.get(bucket_key, "")).strip()
            if not text:
                continue
            identity = f"{bucket}::{text}"
            owner = owners.get(identity)
            if owner and owner[0] != genre_key:
                continue
            filtered.append(candidate)
        item[candidate_field] = filtered


def load_genre_packages() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(GENRES_ROOT.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        playbook_path = path / "playbook.json"
        skill_path = path / "skill.md"
        if not playbook_path.exists() or not skill_path.exists():
            continue
        playbook = load_json(playbook_path)
        genre_key = str(playbook.get("genre_key", "")).strip()
        if not genre_key:
            continue
        result[genre_key] = {
            "dir_path": path.resolve(),
            "playbook_path": playbook_path.resolve(),
            "skill_path": skill_path.resolve(),
            "playbook": playbook,
            "skill_text": skill_path.read_text(encoding="utf-8"),
        }
    return result


def select_target_genres(override_report: dict[str, Any], series_bible: dict[str, Any] | None = None) -> list[str]:
    confirmed = unique(list(override_report.get("confirmed_genres", [])))
    if confirmed:
        return confirmed
    if series_bible:
        genre_profile = dict(series_bible.get("genre_profile", {}))
        values = [
            str(genre_profile.get("primary_genre", "")).strip(),
            *[str(item).strip() for item in genre_profile.get("secondary_genres", [])],
        ]
        values = unique(values)
        if values:
            return values
    return []


def build_playbook_candidates(
    *,
    strength: dict[str, Any],
    genre_key: str,
    profile: dict[str, Any],
    genre_profiles: dict[str, dict[str, Any]],
    series_specific_tokens: list[str],
    min_items: int,
    max_items: int,
    min_confidence: float,
    medium_confidence_threshold: float,
    high_confidence_threshold: float,
    default_group_limit: int,
    high_confidence_group_limit: int,
) -> list[dict[str, Any]]:
    raw_candidates: list[dict[str, Any]] = []
    mapping = [
        ("camera_language_rules", "director_focus"),
        ("character_design_rules", "character_focus"),
        ("costume_makeup_rules", "costume_focus"),
        ("scene_design_rules", "scene_focus"),
        ("storyboard_execution_rules", "storyboard_focus"),
        ("dialogue_timing_rules", "dialogue_timing_rules"),
        ("continuity_guardrails", "continuity_guardrails"),
        ("negative_patterns", "negative_patterns"),
        ("reusable_playbook_rules", "script_hooks"),
        ("character_appeal_patterns", "character_focus"),
        ("scene_staging_patterns", "scene_focus"),
        ("camera_language_patterns", "director_focus"),
        ("storyboard_execution_patterns", "storyboard_focus"),
        ("dialogue_patterns", "dialogue_timing_rules"),
    ]
    for source_type, target_field in mapping:
        for item in strength.get(source_type, []):
            full_text = str(item).strip()
            if not full_text:
                continue
            preview_text = shorten_text(full_text)
            confidence, matched_keywords = score_candidate(
                text=full_text,
                source_type=source_type,
                target_field=target_field,
                section=None,
                profile=profile,
            )
            reason = build_reason(source_type, matched_keywords)
            confidence, reason, best_competing_genre, best_competing_confidence, gap = apply_genre_specificity_adjustment(
                text=full_text,
                source_type=source_type,
                target_field=target_field,
                section=None,
                genre_key=genre_key,
                profile=profile,
                genre_profiles=genre_profiles,
                confidence=confidence,
                matched_keywords=matched_keywords,
                reason=reason,
            )
            confidence, reason, series_specific_matches = apply_series_specific_penalty(
                text=full_text,
                confidence=confidence,
                reason=reason,
                series_specific_tokens=series_specific_tokens,
            )
            raw_candidates.append(
                {
                    "genre_key": genre_key,
                    "target_field": target_field,
                    "target_field_label": FIELD_DISPLAY.get(target_field, target_field),
                    "text": full_text,
                    "full_text": full_text,
                    "preview_text": preview_text,
                    "confidence": confidence,
                    "matched_keywords": matched_keywords,
                    "reason": reason,
                    "series_specific_matches": series_specific_matches,
                    "best_competing_genre": best_competing_genre,
                    "best_competing_confidence": best_competing_confidence,
                    "genre_confidence_gap": gap,
                    "source_type": source_type,
                    "source_label": SOURCE_DISPLAY.get(source_type, source_type),
                }
            )
    for source_type in (
        "dialogue_patterns",
        "reusable_skill_rules",
        "camera_language_rules",
        "storyboard_execution_rules",
        "dialogue_timing_rules",
    ):
        for item in strength.get(source_type, []):
            full_text = str(item).strip()
            if not full_text:
                continue
            preview_text = shorten_text(full_text)
            if source_type == "dialogue_timing_rules" or any(
                keyword in full_text for keyword in ("台词", "对白", "停顿", "留白", "重叠", "音效")
            ):
                target_field = "dialogue_timing_rules"
            elif source_type in ("camera_language_rules", "storyboard_execution_rules") or any(
                keyword in full_text for keyword in ("镜头", "画面", "动作", "特写", "调度", "节奏", "切条", "结尾", "卡点", "转场")
            ):
                target_field = "storyboard_focus"
            else:
                target_field = "script_hooks"
            confidence, matched_keywords = score_candidate(
                text=full_text,
                source_type=source_type,
                target_field=target_field,
                section=None,
                profile=profile,
            )
            reason = build_reason(source_type, matched_keywords)
            confidence, reason, best_competing_genre, best_competing_confidence, gap = apply_genre_specificity_adjustment(
                text=full_text,
                source_type=source_type,
                target_field=target_field,
                section=None,
                genre_key=genre_key,
                profile=profile,
                genre_profiles=genre_profiles,
                confidence=confidence,
                matched_keywords=matched_keywords,
                reason=reason,
            )
            confidence, reason, series_specific_matches = apply_series_specific_penalty(
                text=full_text,
                confidence=confidence,
                reason=reason,
                series_specific_tokens=series_specific_tokens,
            )
            raw_candidates.append(
                {
                    "genre_key": genre_key,
                    "target_field": target_field,
                    "target_field_label": FIELD_DISPLAY.get(target_field, target_field),
                    "text": full_text,
                    "full_text": full_text,
                    "preview_text": preview_text,
                    "confidence": confidence,
                    "matched_keywords": matched_keywords,
                    "reason": reason,
                    "series_specific_matches": series_specific_matches,
                    "best_competing_genre": best_competing_genre,
                    "best_competing_confidence": best_competing_confidence,
                    "genre_confidence_gap": gap,
                    "source_type": source_type,
                    "source_label": SOURCE_DISPLAY.get(source_type, source_type),
                }
            )
    return rank_and_trim_candidates(
        raw_candidates,
        group_key="target_field",
        min_items=min_items,
        max_items=max_items,
        min_confidence=min_confidence,
        medium_confidence_threshold=medium_confidence_threshold,
        high_confidence_threshold=high_confidence_threshold,
        default_group_limit=default_group_limit,
        high_confidence_group_limit=high_confidence_group_limit,
    )


def build_skill_candidates(
    *,
    strength: dict[str, Any],
    genre_key: str,
    profile: dict[str, Any],
    genre_profiles: dict[str, dict[str, Any]],
    series_specific_tokens: list[str],
    min_items: int,
    max_items: int,
    min_confidence: float,
    medium_confidence_threshold: float,
    high_confidence_threshold: float,
    default_group_limit: int,
    high_confidence_group_limit: int,
) -> list[dict[str, Any]]:
    raw_candidates: list[dict[str, Any]] = []
    for source_type in (
        "character_design_rules",
        "costume_makeup_rules",
        "scene_design_rules",
        "camera_language_rules",
        "storyboard_execution_rules",
        "dialogue_timing_rules",
        "continuity_guardrails",
        "negative_patterns",
        "reusable_skill_rules",
        "reusable_playbook_rules",
        "character_appeal_patterns",
        "scene_staging_patterns",
        "dialogue_patterns",
    ):
        for item in strength.get(source_type, []):
            full_text = str(item).strip()
            if not full_text:
                continue
            preview_text = shorten_text(full_text)
            section = classify_skill_section(full_text, source_type)
            confidence, matched_keywords = score_candidate(
                text=full_text,
                source_type=source_type,
                target_field=None,
                section=section,
                profile=profile,
            )
            reason = build_reason(source_type, matched_keywords)
            confidence, reason, best_competing_genre, best_competing_confidence, gap = apply_genre_specificity_adjustment(
                text=full_text,
                source_type=source_type,
                target_field=None,
                section=section,
                genre_key=genre_key,
                profile=profile,
                genre_profiles=genre_profiles,
                confidence=confidence,
                matched_keywords=matched_keywords,
                reason=reason,
            )
            confidence, reason, series_specific_matches = apply_series_specific_penalty(
                text=full_text,
                confidence=confidence,
                reason=reason,
                series_specific_tokens=series_specific_tokens,
            )
            raw_candidates.append(
                {
                    "genre_key": genre_key,
                    "section": section,
                    "section_label": SECTION_DISPLAY.get(section, section),
                    "text": full_text,
                    "full_text": full_text,
                    "preview_text": preview_text,
                    "confidence": confidence,
                    "matched_keywords": matched_keywords,
                    "reason": reason,
                    "series_specific_matches": series_specific_matches,
                    "best_competing_genre": best_competing_genre,
                    "best_competing_confidence": best_competing_confidence,
                    "genre_confidence_gap": gap,
                    "source_type": source_type,
                    "source_label": SOURCE_DISPLAY.get(source_type, source_type),
                }
            )
    return rank_and_trim_candidates(
        raw_candidates,
        group_key="section",
        min_items=min_items,
        max_items=max_items,
        min_confidence=min_confidence,
        medium_confidence_threshold=medium_confidence_threshold,
        high_confidence_threshold=high_confidence_threshold,
        default_group_limit=default_group_limit,
        high_confidence_group_limit=high_confidence_group_limit,
    )


def apply_playbook_candidates(
    base_playbook: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    min_confidence_to_apply: float,
    per_field_limit: int = 2,
) -> dict[str, Any]:
    merged = copy.deepcopy(base_playbook)
    field_counts: dict[str, int] = {}
    for candidate in candidates:
        if float(candidate.get("confidence", 0.0)) < min_confidence_to_apply:
            continue
        target_field = str(candidate.get("target_field", "")).strip()
        text = str(candidate.get("full_text") or candidate.get("text", "")).strip()
        if not target_field or not text:
            continue
        if field_counts.get(target_field, 0) >= per_field_limit:
            continue
        existing = [str(item).strip() for item in merged.get(target_field, [])]
        if text in existing:
            continue
        merged[target_field] = existing + [text]
        field_counts[target_field] = field_counts.get(target_field, 0) + 1
    merged["_update_candidate"] = {
        "source": "series_strength_playbook_draft",
        "mode": "top_ranked_candidates_only",
        "applied_min_confidence": min_confidence_to_apply,
    }
    return merged


def build_skill_draft_markdown(series_name: str, genre_key: str, candidates: list[dict[str, Any]]) -> str:
    lines = [
        f"# 题材补充 Skill 更新建议：{genre_key}",
        "",
        f"> 来源剧目：{series_name}",
        "> 说明：以下为按置信度排序的精简候选规则，建议人工审核后再并入正式 skill。",
        "",
    ]
    if not candidates:
        lines.extend(["暂无高质量候选建议。", ""])
        return "\n".join(lines).rstrip() + "\n"
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        grouped.setdefault(str(item.get("section_label", "")), []).append(item)
    for section, items in grouped.items():
        lines.append(f"## {section}")
        lines.append("")
        for item in items:
            text = str(item.get("full_text") or item.get("text", "")).strip()
            lines.append(
                f"- [{float(item.get('confidence', 0.0)):.2f}] {text}"
            )
            lines.append(f"  说明：{item.get('reason', '')}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_skill_appendix_from_candidates(
    series_name: str,
    candidates: list[dict[str, Any]],
    *,
    min_confidence_to_apply: float,
) -> str:
    lines = [
        "",
        f"## 基于《{series_name}》提炼的高置信经验",
        "",
        f"以下规则仅并入置信度不低于 {min_confidence_to_apply:.2f} 的候选。",
        "",
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        if float(item.get("confidence", 0.0)) < min_confidence_to_apply:
            continue
        grouped.setdefault(str(item.get("section_label", "")), []).append(item)
    if not grouped:
        lines.append("- 暂无达到写回阈值的候选规则。")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n"
    for section, items in grouped.items():
        lines.append(f"### {section}")
        lines.append("")
        for item in items:
            lines.append(f"- {str(item.get('full_text') or item.get('text', '')).strip()}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def resolve_input_paths(config: dict[str, Any]) -> tuple[Path, Path | None, Path | None]:
    analysis_root = Path(config.get("analysis_root", "analysis")).expanduser().resolve()
    series_name = str(config["series_name"]).strip()
    if not series_name:
        raise ValueError("config.series_name 不能为空。")
    strength_path = analysis_root / series_name / "series_strength_playbook_draft.json"
    override_path = config.get("genre_override_report_path")
    if override_path:
        override_report_path = Path(str(override_path)).expanduser().resolve()
    else:
        provider = str(config.get("provider", "")).strip()
        model = str(config.get("model", "")).strip()
        if provider and model:
            tag = f"{provider}__{model}"
            candidate = analysis_root / series_name / f"genre_override_request__{tag}.json"
            override_report_path = candidate if candidate.exists() else None
        else:
            matches = sorted((analysis_root / series_name).glob("genre_override_request__*.json"))
            override_report_path = matches[-1] if matches else None
    series_bible_path = analysis_root / series_name / "series_bible.json"
    return strength_path, override_report_path, series_bible_path if series_bible_path.exists() else None


def build_update_plan(config: dict[str, Any]) -> dict[str, Any]:
    strength_path, override_report_path, series_bible_path = resolve_input_paths(config)
    strength = load_json(strength_path)
    override_report = load_json(override_report_path) if override_report_path else {}
    series_bible = load_json(series_bible_path) if series_bible_path else {}
    series_dir = strength_path.parent
    character_registry = load_optional_json(series_dir / "character_registry.json")
    location_registry = load_optional_json(series_dir / "location_registry.json")
    series_specific_tokens = collect_series_specific_tokens(
        series_name=str(config["series_name"]).strip(),
        series_bible=series_bible,
        character_registry=character_registry,
        location_registry=location_registry,
    )
    packages = load_genre_packages()
    target_genres = select_target_genres(override_report, series_bible)
    if not target_genres:
        raise FileNotFoundError(
            "未找到可用题材来源：既没有整剧题材修正报告 genre_override_request__*.json，"
            "series_bible.json 中也没有可用的 genre_profile。"
        )
    genre_profiles = {
        genre_key: build_package_profile(packages.get(genre_key))
        for genre_key in target_genres
    }
    min_items = int(config.get("min_candidates_per_genre", 6))
    max_items = int(config.get("max_candidates_per_genre", config.get("max_items_per_section", 14)))
    min_confidence = float(config.get("min_candidate_confidence", 0.68))
    min_confidence_to_apply = float(config.get("min_confidence_to_apply", 0.82))
    medium_confidence_threshold = float(config.get("dynamic_medium_confidence_threshold", 0.82))
    high_confidence_threshold = float(config.get("dynamic_high_confidence_threshold", 0.90))
    default_group_limit = int(config.get("dynamic_default_group_limit", 2))
    high_confidence_group_limit = int(config.get("dynamic_high_confidence_group_limit", 3))
    series_name = str(config["series_name"]).strip()
    apply_updates = bool(config.get("apply_updates", False))

    pending_items: list[dict[str, Any]] = []
    for genre_key in target_genres:
        package = packages.get(genre_key)
        profile = genre_profiles.get(genre_key) or build_package_profile(package)
        base_playbook = copy.deepcopy(package["playbook"]) if package else {
            "genre_key": genre_key,
            "aliases": [genre_key],
            "core_audience_promises": [],
            "script_hooks": [],
            "character_design_focus": [],
            "scene_design_focus": [],
            "storyboard_focus": [],
        }
        base_skill_text = package["skill_text"] if package else f"# 题材补充 Skill：{genre_key}\n"
        target_slug = package["dir_path"].name if package else slugify(genre_key)
        draft_dir = GENRE_DRAFT_ROOT / f"sync__{slugify(series_name)}__{target_slug}"
        draft_playbook_path = draft_dir / "playbook.json"
        draft_skill_path = draft_dir / "skill.md"
        legacy_playbook_path = LEGACY_PLAYBOOK_DRAFT_ROOT / f"sync__{slugify(series_name)}__{target_slug}.json"
        playbook_candidates = build_playbook_candidates(
            strength=strength,
            genre_key=genre_key,
            profile=profile,
            genre_profiles=genre_profiles,
            series_specific_tokens=series_specific_tokens,
            min_items=min_items,
            max_items=max_items,
            min_confidence=min_confidence,
            medium_confidence_threshold=medium_confidence_threshold,
            high_confidence_threshold=high_confidence_threshold,
            default_group_limit=default_group_limit,
            high_confidence_group_limit=high_confidence_group_limit,
        )
        skill_candidates = build_skill_candidates(
            strength=strength,
            genre_key=genre_key,
            profile=profile,
            genre_profiles=genre_profiles,
            series_specific_tokens=series_specific_tokens,
            min_items=min_items,
            max_items=max_items,
            min_confidence=min_confidence,
            medium_confidence_threshold=medium_confidence_threshold,
            high_confidence_threshold=high_confidence_threshold,
            default_group_limit=default_group_limit,
            high_confidence_group_limit=high_confidence_group_limit,
        )
        pending_items.append(
            {
                "genre_key": genre_key,
                "package": package,
                "base_playbook": base_playbook,
                "base_skill_text": base_skill_text,
                "draft_playbook_path": draft_playbook_path,
                "draft_skill_path": draft_skill_path,
                "legacy_draft_playbook_path": legacy_playbook_path,
                "playbook_update_candidates": playbook_candidates,
                "skill_update_candidates": skill_candidates,
            }
        )

    prune_cross_genre_candidate_overlap(
        pending_items,
        candidate_field="playbook_update_candidates",
        bucket_key="target_field",
    )
    prune_cross_genre_candidate_overlap(
        pending_items,
        candidate_field="skill_update_candidates",
        bucket_key="section",
    )

    items: list[dict[str, Any]] = []
    for pending in pending_items:
        genre_key = str(pending.get("genre_key", "")).strip()
        package = pending.get("package")
        base_playbook = copy.deepcopy(pending.get("base_playbook", {}))
        base_skill_text = str(pending.get("base_skill_text", ""))
        draft_playbook_path = Path(pending["draft_playbook_path"])
        draft_skill_path = Path(pending["draft_skill_path"])
        legacy_playbook_path = Path(pending["legacy_draft_playbook_path"])
        playbook_candidates = list(pending.get("playbook_update_candidates", []))
        skill_candidates = list(pending.get("skill_update_candidates", []))

        merged_playbook = apply_playbook_candidates(
            base_playbook,
            playbook_candidates,
            min_confidence_to_apply=min_confidence_to_apply,
        )
        merged_skill_text = (
            base_skill_text.rstrip()
            + "\n\n"
            + build_skill_appendix_from_candidates(
                series_name,
                skill_candidates,
                min_confidence_to_apply=min_confidence_to_apply,
            )
        )

        save_json(
            draft_playbook_path,
            {
                "genre_key": genre_key,
                "source_series": series_name,
                "mode": "ranked_update_candidates",
                "playbook_update_candidates": playbook_candidates,
                "skill_update_candidates": skill_candidates,
            },
        )
        save_text(draft_skill_path, build_skill_draft_markdown(series_name, genre_key, skill_candidates))
        save_json(
            legacy_playbook_path,
            {
                "genre_key": genre_key,
                "source_series": series_name,
                "playbook_update_candidates": playbook_candidates,
                "skill_update_candidates": skill_candidates,
                "_source_series": series_name,
            },
        )

        applied_playbook_path = ""
        applied_skill_path = ""
        if apply_updates and package:
            save_json(package["playbook_path"], merged_playbook)
            save_text(package["skill_path"], merged_skill_text)
            applied_playbook_path = str(package["playbook_path"])
            applied_skill_path = str(package["skill_path"])

        items.append(
            {
                "genre_key": genre_key,
                "mode": "update_existing" if package else "create_new_draft_only",
                "source_playbook_path": str(package["playbook_path"]) if package else "",
                "source_skill_path": str(package["skill_path"]) if package else "",
                "draft_playbook_path": str(draft_playbook_path.resolve()),
                "draft_skill_path": str(draft_skill_path.resolve()),
                "legacy_draft_playbook_path": str(legacy_playbook_path.resolve()),
                "applied_playbook_path": applied_playbook_path,
                "applied_skill_path": applied_skill_path,
                "playbook_update_candidates": playbook_candidates,
                "skill_update_candidates": skill_candidates,
                "min_confidence_to_apply": min_confidence_to_apply,
            }
        )

    return {
        "series_name": series_name,
        "apply_updates": apply_updates,
        "source_strength_path": str(strength_path),
        "source_genre_override_report_path": str(override_report_path) if override_report_path else "",
        "target_genres": target_genres,
        "min_candidates_per_genre": min_items,
        "max_candidates_per_genre": max_items,
        "min_candidate_confidence": min_confidence,
        "min_confidence_to_apply": min_confidence_to_apply,
        "dynamic_medium_confidence_threshold": medium_confidence_threshold,
        "dynamic_high_confidence_threshold": high_confidence_threshold,
        "dynamic_default_group_limit": default_group_limit,
        "dynamic_high_confidence_group_limit": high_confidence_group_limit,
        "items": items,
    }


def render_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# 整剧经验同步到题材库更新计划",
        "",
        f"- 剧名：{plan.get('series_name', '')}",
        f"- 是否已正式写回题材库：{plan.get('apply_updates', False)}",
        f"- 经验来源：{plan.get('source_strength_path', '')}",
        f"- 题材报告来源：{plan.get('source_genre_override_report_path', '')}",
        f"- 目标题材：{'、'.join(plan.get('target_genres', [])) or '无'}",
        f"- 基础候选条数：{int(plan.get('min_candidates_per_genre', 0))}",
        f"- 动态候选上限：{int(plan.get('max_candidates_per_genre', 0))}",
        f"- 候选保留阈值：{float(plan.get('min_candidate_confidence', 0.0)):.2f}",
        f"- 正式写回阈值：{float(plan.get('min_confidence_to_apply', 0.0)):.2f}",
        f"- 中高分阈值：{float(plan.get('dynamic_medium_confidence_threshold', 0.0)):.2f}",
        f"- 高分阈值：{float(plan.get('dynamic_high_confidence_threshold', 0.0)):.2f}",
        "",
    ]
    for item in plan.get("items", []):
        lines.extend(
            [
                f"## {item.get('genre_key', '')}",
                "",
                f"- 模式：{item.get('mode', '')}",
                f"- 原始 playbook：{item.get('source_playbook_path', '') or '无'}",
                f"- 原始 skill：{item.get('source_skill_path', '') or '无'}",
                f"- Draft playbook：{item.get('draft_playbook_path', '')}",
                f"- Draft skill：{item.get('draft_skill_path', '')}",
                f"- Legacy draft：{item.get('legacy_draft_playbook_path', '')}",
                f"- 正式写回 playbook：{item.get('applied_playbook_path', '') or '未写回'}",
                f"- 正式写回 skill：{item.get('applied_skill_path', '') or '未写回'}",
                "",
                "### Playbook 候选更新",
                "",
            ]
        )
        playbook_candidates = item.get("playbook_update_candidates", [])
        if playbook_candidates:
            for candidate in playbook_candidates:
                lines.append(
                    f"- [{float(candidate.get('confidence', 0.0)):.2f}] {candidate.get('target_field_label', '')}：{candidate.get('text', '')}"
                )
                lines.append(f"  说明：{candidate.get('reason', '')}")
        else:
            lines.append("- 暂无高质量 playbook 候选。")
        lines.extend(["", "### Skill 候选更新", ""])
        skill_candidates = item.get("skill_update_candidates", [])
        if skill_candidates:
            for candidate in skill_candidates:
                lines.append(
                    f"- [{float(candidate.get('confidence', 0.0)):.2f}] {candidate.get('section_label', '')}：{candidate.get('text', '')}"
                )
                lines.append(f"  说明：{candidate.get('reason', '')}")
        else:
            lines.append("- 暂无高质量 skill 候选。")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def output_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    analysis_root = Path(config.get("analysis_root", "analysis")).expanduser().resolve()
    series_name = str(config["series_name"]).strip()
    provider = str(config.get("provider", "")).strip() or "unknown"
    model = str(config.get("model", "")).strip() or "unknown"
    tag = f"{provider}__{model}"
    base = analysis_root / series_name / f"genre_library_update_plan__{tag}"
    return Path(f"{base}.json"), Path(f"{base}.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync series learning draft into genre package update drafts.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    plan = build_update_plan(config)
    json_path, md_path = output_paths(config)
    save_json(json_path, plan)
    save_text(md_path, render_plan_markdown(plan))
    print_status(f"更新计划已写入：{md_path}")
    for item in plan.get("items", []):
        print_status(
            f"{item.get('genre_key', '')} | 模式={item.get('mode', '')} | draft={item.get('draft_playbook_path', '')}"
        )


if __name__ == "__main__":
    main()
