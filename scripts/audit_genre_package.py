from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai_agents.runtime_utils import configure_openai_api, load_runtime_config, openai_json_completion
from prompt_utils import render_prompt


DEFAULT_CONFIG_PATH = Path("config/genre_package_audit.local.json")
SKILL_ROOT = PROJECT_ROOT / "skills" / "production" / "video-script-reconstruction-skill"
GENRES_ROOT = SKILL_ROOT / "genres"
AUDIT_ROOT = GENRES_ROOT / "__audits__"
GENRE_DRAFT_ROOT = GENRES_ROOT / "__drafts__"
PLAYBOOK_SKIP_FIELDS = {"genre_key", "aliases", "_update_candidate"}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}

AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "genre_key",
        "health_score",
        "overall_assessment",
        "key_strengths",
        "findings",
        "playbook_edit_suggestions",
        "skill_edit_suggestions",
        "questions_for_human",
    ],
    "properties": {
        "genre_key": {"type": "string"},
        "health_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "overall_assessment": {"type": "string"},
        "key_strengths": {"type": "array", "items": {"type": "string"}},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "finding_id",
                    "severity",
                    "category",
                    "location",
                    "problem",
                    "why_it_matters",
                    "recommendation",
                    "confidence",
                ],
                "properties": {
                    "finding_id": {"type": "string"},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "category": {
                        "type": "string",
                        "enum": [
                            "duplicate",
                            "reasonableness",
                            "scope",
                            "missing",
                            "structure",
                            "clarity",
                            "cross_genre_overlap",
                        ],
                    },
                    "location": {"type": "string"},
                    "problem": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "recommendation": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "playbook_edit_suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "edit_id",
                    "priority",
                    "action",
                    "source_field",
                    "target_field",
                    "old_text",
                    "new_text",
                    "reason",
                ],
                "properties": {
                    "edit_id": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "action": {"type": "string", "enum": ["add", "rewrite", "remove", "move"]},
                    "source_field": {"type": "string"},
                    "target_field": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "skill_edit_suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "edit_id",
                    "priority",
                    "action",
                    "source_section",
                    "target_section",
                    "old_text",
                    "new_text",
                    "reason",
                ],
                "properties": {
                    "edit_id": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "action": {"type": "string", "enum": ["add", "rewrite", "remove", "move"]},
                    "source_section": {"type": "string"},
                    "target_section": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "questions_for_human": {"type": "array", "items": {"type": "string"}},
    },
}

REFINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["genre_key", "summary", "playbook_json", "skill_markdown"],
    "properties": {
        "genre_key": {"type": "string"},
        "summary": {"type": "string"},
        "playbook_json": {"type": "string"},
        "skill_markdown": {"type": "string"},
    },
}


def print_status(message: str) -> None:
    print(f"[genre-package-audit] {message}", flush=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, data: Mapping[str, Any]) -> Path:
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


def timestamp_tag() -> str:
    return time.strftime("%Y%m%d%H%M%S", time.gmtime())


def slugify(value: str) -> str:
    clean = re.sub(r"\s+", "_", str(value).strip())
    clean = re.sub(r"[\\\\/:*?\"<>|（）()\[\]{}，,。；;！!？?]+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean[:48] or "genre"


def normalize_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(text).strip().lower())


def similarity(a: str, b: str) -> float:
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        shorter = min(len(na), len(nb))
        longer = max(len(na), len(nb))
        return round(shorter / longer, 3)
    return round(SequenceMatcher(None, na, nb).ratio(), 3)


def unique_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def prompt_named_choice(prompt: str, options: list[tuple[str, str]], default_index: int = 0) -> tuple[str, str]:
    print(prompt)
    for index, (label, _) in enumerate(options, start=1):
        suffix = "  [默认]" if index - 1 == default_index else ""
        print(f"  {index}. {label}{suffix}")
    while True:
        raw = input(f"请输入序号（默认 {default_index + 1}）：").strip()
        if not raw:
            return options[default_index]
        if raw.isdigit():
            value = int(raw)
            if 1 <= value <= len(options):
                return options[value - 1]
        print("输入无效，请重新输入。")


def prompt_zero_one_two(prompt: str, default_value: int = 2) -> int:
    default_text = str(default_value)
    while True:
        raw = input(f"{prompt}（1采纳 / 0跳过 / 2稍后，默认 {default_text}）：").strip()
        if not raw:
            return default_value
        if raw in {"0", "1", "2"}:
            return int(raw)
        print("输入无效，请输入 1、0 或 2。")


def prompt_yes_no(prompt: str, default_value: bool = False) -> bool:
    default_text = "1" if default_value else "0"
    while True:
        raw = input(f"{prompt}（1/0，默认 {default_text}）：").strip()
        if not raw:
            return default_value
        if raw in {"0", "1"}:
            return raw == "1"
        print("输入无效，请输入 1 或 0。")


def parse_skill_sections(skill_text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_title = "未分组"
    current_lines: list[str] = []
    for raw in skill_text.splitlines():
        if raw.startswith("### "):
            if current_lines:
                sections.append({"section": current_title, "items": current_lines[:]})
            current_title = raw[4:].strip() or "未分组"
            current_lines = []
            continue
        stripped = raw.strip()
        if stripped.startswith("- "):
            current_lines.append(stripped[2:].strip())
    if current_lines:
        sections.append({"section": current_title, "items": current_lines[:]})
    return sections


def flatten_playbook_entries(playbook: Mapping[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for field, values in playbook.items():
        if field in PLAYBOOK_SKIP_FIELDS or str(field).startswith("_"):
            continue
        if not isinstance(values, list):
            continue
        for item in values:
            text = str(item).strip()
            if not text:
                continue
            result.append(
                {
                    "source_type": "playbook",
                    "location": f"playbook.json::{field}",
                    "field": str(field),
                    "section": str(field),
                    "text": text,
                }
            )
    return result


def flatten_skill_entries(skill_text: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for section in parse_skill_sections(skill_text):
        section_name = str(section.get("section") or "未分组")
        for item in list(section.get("items") or []):
            text = str(item).strip()
            if not text:
                continue
            result.append(
                {
                    "source_type": "skill",
                    "location": f"skill.md::{section_name}",
                    "field": section_name,
                    "section": section_name,
                    "text": text,
                }
            )
    return result


def build_package_stats(playbook: Mapping[str, Any], skill_text: str) -> dict[str, Any]:
    playbook_entries = flatten_playbook_entries(playbook)
    skill_sections = parse_skill_sections(skill_text)
    field_counts = {
        str(field): len([str(item).strip() for item in values if str(item).strip()])
        for field, values in playbook.items()
        if isinstance(values, list) and field not in PLAYBOOK_SKIP_FIELDS and not str(field).startswith("_")
    }
    section_counts = {
        str(section.get("section") or "未分组"): len([str(item).strip() for item in list(section.get("items") or []) if str(item).strip()])
        for section in skill_sections
    }
    return {
        "playbook_entry_count": len(playbook_entries),
        "playbook_field_counts": field_counts,
        "skill_section_count": len(skill_sections),
        "skill_rule_count": sum(section_counts.values()),
        "skill_section_counts": section_counts,
        "empty_playbook_fields": sorted([field for field, count in field_counts.items() if count == 0]),
        "empty_skill_sections": sorted([section for section, count in section_counts.items() if count == 0]),
    }


def load_official_packages() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(GENRES_ROOT.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.name.startswith("__"):
            continue
        playbook_path = path / "playbook.json"
        skill_path = path / "skill.md"
        if not playbook_path.exists() or not skill_path.exists():
            continue
        playbook = load_json(playbook_path)
        genre_key = str(playbook.get("genre_key") or path.name).strip()
        if not genre_key:
            continue
        result[genre_key] = {
            "genre_key": genre_key,
            "dir_path": path.resolve(),
            "playbook_path": playbook_path.resolve(),
            "skill_path": skill_path.resolve(),
            "playbook": playbook,
            "skill_text": skill_path.read_text(encoding="utf-8"),
        }
    return result


def load_existing_audit_drafts(genre_key: str) -> list[dict[str, Any]]:
    if not GENRE_DRAFT_ROOT.exists():
        return []
    slug = slugify(genre_key)
    prefix = f"audit__{slug}__"
    result: list[dict[str, Any]] = []
    for path in sorted(GENRE_DRAFT_ROOT.iterdir(), key=lambda item: item.name, reverse=True):
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        playbook_path = path / "playbook.json"
        skill_path = path / "skill.md"
        judgement_path = path / "judgement.json"
        if not playbook_path.exists() or not skill_path.exists():
            continue
        playbook = load_json(playbook_path)
        judgement = load_json(judgement_path) if judgement_path.exists() else {}
        draft_genre_key = str(playbook.get("genre_key") or "").strip() or genre_key
        if draft_genre_key != genre_key:
            continue
        result.append(
            {
                "genre_key": genre_key,
                "draft_dir": path.resolve(),
                "playbook_path": playbook_path.resolve(),
                "skill_path": skill_path.resolve(),
                "judgement_path": judgement_path.resolve() if judgement_path.exists() else None,
                "playbook": playbook,
                "skill_text": skill_path.read_text(encoding="utf-8"),
                "judgement": judgement,
                "accepted_playbook_count": len(list(judgement.get("accepted_playbook_suggestions") or [])),
                "accepted_skill_count": len(list(judgement.get("accepted_skill_suggestions") or [])),
                "refine_summary": str(judgement.get("refine_summary") or "").strip(),
            }
        )
    return result


def resolve_existing_audit_draft(
    *,
    genre_key: str,
    selected_draft_dir: str = "",
) -> dict[str, Any]:
    drafts = load_existing_audit_drafts(genre_key)
    if not drafts:
        raise RuntimeError(f"未找到题材 `{genre_key}` 的审计草稿。")

    if selected_draft_dir:
        selected = Path(selected_draft_dir).expanduser().resolve()
        for item in drafts:
            if Path(item["draft_dir"]).resolve() == selected:
                return item
        raise RuntimeError(f"指定的 draft 不属于题材 `{genre_key}`：{selected}")
    return drafts[0]


def render_apply_record_markdown(record: Mapping[str, Any]) -> str:
    lines = [
        f"# 题材包应用记录：{record.get('genre_key', '')}",
        "",
        f"- 应用模式：{record.get('apply_mode', '')}",
        f"- 正式 Playbook：{record.get('official_playbook_path', '')}",
        f"- 正式 Skill：{record.get('official_skill_path', '')}",
        f"- 备份目录：{record.get('backup_dir', '')}",
        f"- 来源 Draft：{record.get('source_draft_dir', '') or '无'}",
        f"- 决策文件：{record.get('judgement_path', '') or '无'}",
        f"- draft 清理状态：{record.get('draft_cleanup_status', '')}",
    ]
    if str(record.get("draft_cleanup_error") or "").strip():
        lines.append(f"- draft 清理错误：{record.get('draft_cleanup_error', '')}")
    if str(record.get("refine_summary") or "").strip():
        lines.extend(["", "## 修订摘要", "", str(record.get("refine_summary", "")).strip()])
    return "\n".join(lines).rstrip() + "\n"


def apply_official_package(
    *,
    package: Mapping[str, Any],
    refined_draft: Mapping[str, Any],
    apply_mode: str,
    source_draft_dir: Path | None = None,
    judgement_path: Path | None = None,
    cleanup_applied_draft: bool = True,
) -> dict[str, Any]:
    ts = timestamp_tag()
    genre_slug = slugify(str(package["genre_key"]))
    backup_dir = AUDIT_ROOT / "backups" / f"{genre_slug}__{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(package["playbook_path"]), str(backup_dir / "playbook.json"))
    shutil.copy2(str(package["skill_path"]), str(backup_dir / "skill.md"))

    save_json(package["playbook_path"], refined_draft["playbook"])
    save_text(package["skill_path"], refined_draft["skill_markdown"])
    print_status(f"正式题材包已更新，原文件备份在：{backup_dir}")

    cleanup_status = "not_requested"
    cleanup_error = ""
    deleted_draft_dir = ""
    if cleanup_applied_draft and source_draft_dir is not None:
        try:
            shutil.rmtree(source_draft_dir)
            cleanup_status = "deleted"
            deleted_draft_dir = str(source_draft_dir.resolve())
            print_status(f"已删除已应用的 draft：{source_draft_dir}")
        except FileNotFoundError:
            cleanup_status = "already_missing"
        except Exception as exc:
            cleanup_status = "failed"
            cleanup_error = str(exc)
            print_status(f"删除 draft 失败，请手动检查：{source_draft_dir} | {exc}")

    record = {
        "genre_key": str(package["genre_key"]),
        "apply_mode": apply_mode,
        "official_playbook_path": str(Path(package["playbook_path"]).resolve()),
        "official_skill_path": str(Path(package["skill_path"]).resolve()),
        "backup_dir": str(backup_dir.resolve()),
        "source_draft_dir": str(source_draft_dir.resolve()) if source_draft_dir is not None else "",
        "judgement_path": str(judgement_path.resolve()) if judgement_path is not None else "",
        "refine_summary": str(refined_draft.get("summary") or "").strip(),
        "draft_cleanup_requested": bool(cleanup_applied_draft and source_draft_dir is not None),
        "draft_cleanup_status": cleanup_status,
        "draft_cleanup_error": cleanup_error,
        "deleted_draft_dir": deleted_draft_dir,
    }
    audit_dir = AUDIT_ROOT / genre_slug
    apply_json_path = save_json(audit_dir / f"apply__{ts}.json", record)
    apply_md_path = save_text(audit_dir / f"apply__{ts}.md", render_apply_record_markdown(record))
    record["apply_json_path"] = str(apply_json_path)
    record["apply_md_path"] = str(apply_md_path)
    return record


def find_within_package_duplicates(
    playbook: Mapping[str, Any],
    skill_text: str,
    *,
    max_clusters: int,
    threshold: float,
) -> list[dict[str, Any]]:
    entries = flatten_playbook_entries(playbook) + flatten_skill_entries(skill_text)
    clusters: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for index, left in enumerate(entries):
        for other_index in range(index + 1, len(entries)):
            if (index, other_index) in seen_pairs:
                continue
            right = entries[other_index]
            score = similarity(left["text"], right["text"])
            if score < threshold:
                continue
            seen_pairs.add((index, other_index))
            clusters.append(
                {
                    "similarity": score,
                    "left": {
                        "location": left["location"],
                        "text": left["text"],
                    },
                    "right": {
                        "location": right["location"],
                        "text": right["text"],
                    },
                }
            )
    clusters.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
    return clusters[:max_clusters]


def find_cross_genre_overlaps(
    *,
    current_genre_key: str,
    current_playbook: Mapping[str, Any],
    current_skill_text: str,
    packages: Mapping[str, Mapping[str, Any]],
    max_related_genres: int,
    max_examples_per_genre: int,
    threshold: float,
) -> list[dict[str, Any]]:
    current_entries = flatten_playbook_entries(current_playbook) + flatten_skill_entries(current_skill_text)
    related: list[dict[str, Any]] = []
    for other_genre_key, other_package in packages.items():
        if other_genre_key == current_genre_key:
            continue
        other_entries = flatten_playbook_entries(other_package["playbook"]) + flatten_skill_entries(other_package["skill_text"])
        matches: list[dict[str, Any]] = []
        for left in current_entries:
            for right in other_entries:
                score = similarity(left["text"], right["text"])
                if score < threshold:
                    continue
                matches.append(
                    {
                        "similarity": score,
                        "current_location": left["location"],
                        "current_text": left["text"],
                        "other_location": right["location"],
                        "other_text": right["text"],
                    }
                )
        if not matches:
            continue
        matches.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
        related.append(
            {
                "genre_key": other_genre_key,
                "match_count": len(matches),
                "examples": matches[:max_examples_per_genre],
            }
        )
    related.sort(key=lambda item: (item.get("match_count", 0), item.get("genre_key", "")), reverse=True)
    return related[:max_related_genres]


def find_suspicious_entries(
    playbook: Mapping[str, Any],
    skill_text: str,
    *,
    max_items: int,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for entry in flatten_playbook_entries(playbook) + flatten_skill_entries(skill_text):
        text = entry["text"]
        normalized = text.strip()
        if normalized.endswith("…") or normalized.endswith("..."):
            issues.append(
                {
                    "type": "truncated_text",
                    "location": entry["location"],
                    "text": text,
                }
            )
        if len(normalized) <= 8:
            issues.append(
                {
                    "type": "too_short",
                    "location": entry["location"],
                    "text": text,
                }
            )
    return issues[:max_items]


def build_precheck_report(
    *,
    genre_key: str,
    package: Mapping[str, Any],
    packages: Mapping[str, Mapping[str, Any]],
    max_related_genres: int,
    max_duplicate_clusters: int,
    max_examples_per_genre: int,
    max_suspicious_items: int,
) -> dict[str, Any]:
    playbook = dict(package["playbook"])
    skill_text = str(package["skill_text"])
    stats = build_package_stats(playbook, skill_text)
    return {
        "genre_key": genre_key,
        "stats": stats,
        "within_package_duplicates": find_within_package_duplicates(
            playbook,
            skill_text,
            max_clusters=max_duplicate_clusters,
            threshold=0.9,
        ),
        "cross_genre_overlaps": find_cross_genre_overlaps(
            current_genre_key=genre_key,
            current_playbook=playbook,
            current_skill_text=skill_text,
            packages=packages,
            max_related_genres=max_related_genres,
            max_examples_per_genre=max_examples_per_genre,
            threshold=0.92,
        ),
        "suspicious_entries": find_suspicious_entries(
            playbook,
            skill_text,
            max_items=max_suspicious_items,
        ),
    }


def build_audit_prompt(
    *,
    genre_key: str,
    package: Mapping[str, Any],
    precheck_report: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    max_findings = int(config.get("run", {}).get("max_findings", 12))
    max_suggestions_per_type = int(config.get("run", {}).get("max_suggestions_per_type", 8))
    return render_prompt(
        "genre_package_audit/audit_user.md",
        {
            "genre_key": genre_key,
            "playbook_json": json.dumps(package["playbook"], ensure_ascii=False, indent=2),
            "skill_text": str(package["skill_text"]).strip(),
            "precheck_report_json": json.dumps(precheck_report, ensure_ascii=False, indent=2),
            "max_findings": max_findings,
            "max_suggestions_per_type": max_suggestions_per_type,
        },
    )


def build_refine_prompt(
    *,
    genre_key: str,
    package: Mapping[str, Any],
    accepted_playbook_suggestions: list[dict[str, Any]],
    accepted_skill_suggestions: list[dict[str, Any]],
    audit_report: Mapping[str, Any],
) -> str:
    return render_prompt(
        "genre_package_audit/refine_user.md",
        {
            "genre_key": genre_key,
            "playbook_json": json.dumps(package["playbook"], ensure_ascii=False, indent=2),
            "skill_text": str(package["skill_text"]).strip(),
            "accepted_playbook_suggestions_json": json.dumps(accepted_playbook_suggestions, ensure_ascii=False, indent=2),
            "accepted_skill_suggestions_json": json.dumps(accepted_skill_suggestions, ensure_ascii=False, indent=2),
            "audit_summary": str(audit_report.get("overall_assessment") or "").strip(),
        },
    )


def render_precheck_markdown(precheck_report: Mapping[str, Any]) -> str:
    lines = [
        "## 预检摘要",
        "",
        f"- playbook 条目数：{precheck_report.get('stats', {}).get('playbook_entry_count', 0)}",
        f"- skill 规则数：{precheck_report.get('stats', {}).get('skill_rule_count', 0)}",
        f"- 包内疑似重复：{len(list(precheck_report.get('within_package_duplicates') or []))}",
        f"- 跨题材高重合：{len(list(precheck_report.get('cross_genre_overlaps') or []))}",
        f"- 可疑条目：{len(list(precheck_report.get('suspicious_entries') or []))}",
        "",
    ]
    return "\n".join(lines)


def render_audit_markdown(report: Mapping[str, Any], precheck_report: Mapping[str, Any]) -> str:
    lines = [
        f"# 题材包体检报告：{report.get('genre_key', '')}",
        "",
        f"- 健康分：{report.get('health_score', 0)}",
        f"- 综合结论：{report.get('overall_assessment', '')}",
        "",
        render_precheck_markdown(precheck_report).rstrip(),
        "",
        "## 亮点",
        "",
    ]
    strengths = [str(item).strip() for item in list(report.get("key_strengths") or []) if str(item).strip()]
    if strengths:
        lines.extend([f"- {item}" for item in strengths])
    else:
        lines.append("- 暂无明确亮点。")
    lines.extend(["", "## 发现的问题", ""])
    findings = sorted(
        list(report.get("findings") or []),
        key=lambda item: (
            SEVERITY_ORDER.get(str(item.get("severity") or "low"), 3),
            str(item.get("finding_id") or ""),
        ),
    )
    if findings:
        for item in findings:
            lines.extend(
                [
                    f"### {item.get('finding_id', '')} [{item.get('severity', '')}] {item.get('category', '')}",
                    "",
                    f"- 位置：{item.get('location', '')}",
                    f"- 问题：{item.get('problem', '')}",
                    f"- 影响：{item.get('why_it_matters', '')}",
                    f"- 建议：{item.get('recommendation', '')}",
                    f"- 置信度：{item.get('confidence', 0)}",
                    "",
                ]
            )
    else:
        lines.append("- 未发现显著问题。")
        lines.append("")

    lines.extend(["## Playbook 修改建议", ""])
    playbook_suggestions = list(report.get("playbook_edit_suggestions") or [])
    if playbook_suggestions:
        for item in playbook_suggestions:
            lines.extend(
                [
                    f"- {item.get('edit_id', '')} [{item.get('priority', '')}] {item.get('action', '')} {item.get('source_field', '')} -> {item.get('target_field', '')}",
                    f"  原文：{item.get('old_text', '') or '<空>'}",
                    f"  建议：{item.get('new_text', '') or '<删除>'}",
                    f"  原因：{item.get('reason', '')}",
                ]
            )
    else:
        lines.append("- 无 playbook 级修改建议。")

    lines.extend(["", "## Skill 修改建议", ""])
    skill_suggestions = list(report.get("skill_edit_suggestions") or [])
    if skill_suggestions:
        for item in skill_suggestions:
            lines.extend(
                [
                    f"- {item.get('edit_id', '')} [{item.get('priority', '')}] {item.get('action', '')} {item.get('source_section', '')} -> {item.get('target_section', '')}",
                    f"  原文：{item.get('old_text', '') or '<空>'}",
                    f"  建议：{item.get('new_text', '') or '<删除>'}",
                    f"  原因：{item.get('reason', '')}",
                ]
            )
    else:
        lines.append("- 无 skill 级修改建议。")

    lines.extend(["", "## 需要人工判断的问题", ""])
    questions = [str(item).strip() for item in list(report.get("questions_for_human") or []) if str(item).strip()]
    if questions:
        lines.extend([f"- {item}" for item in questions])
    else:
        lines.append("- 暂无额外问题。")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def sort_suggestions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            PRIORITY_ORDER.get(str(item.get("priority") or "low"), 3),
            str(item.get("edit_id") or ""),
        ),
    )


def summarize_suggestion(item: Mapping[str, Any], *, suggestion_type: str) -> list[str]:
    if suggestion_type == "playbook":
        header = (
            f"{item.get('edit_id', '')} [{item.get('priority', '')}] "
            f"{item.get('action', '')} {item.get('source_field', '')} -> {item.get('target_field', '')}"
        )
    else:
        header = (
            f"{item.get('edit_id', '')} [{item.get('priority', '')}] "
            f"{item.get('action', '')} {item.get('source_section', '')} -> {item.get('target_section', '')}"
        )
    return [
        header,
        f"原文：{item.get('old_text', '') or '<空>'}",
        f"建议：{item.get('new_text', '') or '<删除>'}",
        f"原因：{item.get('reason', '')}",
    ]


def interactive_review(
    report: Mapping[str, Any],
    *,
    interactive_enabled: bool,
    default_review_mode: str,
) -> dict[str, Any]:
    findings = sorted(
        list(report.get("findings") or []),
        key=lambda item: (
            SEVERITY_ORDER.get(str(item.get("severity") or "low"), 3),
            str(item.get("finding_id") or ""),
        ),
    )
    print_status(f"题材 {report.get('genre_key', '')} 体检完成，健康分 {report.get('health_score', 0)}。")
    print(f"综合结论：{report.get('overall_assessment', '')}")
    if findings:
        print("\n主要问题：")
        for item in findings[:6]:
            print(f"- {item.get('finding_id', '')} [{item.get('severity', '')}] {item.get('problem', '')}")

    decisions: dict[str, Any] = {
        "review_mode": "report_only",
        "accepted_playbook_suggestions": [],
        "accepted_skill_suggestions": [],
        "rejected_playbook_suggestions": [],
        "rejected_skill_suggestions": [],
        "deferred_playbook_suggestions": [],
        "deferred_skill_suggestions": [],
    }
    if not interactive_enabled:
        return decisions

    review_options = [
        ("仅审高优先级（推荐）", "high_priority"),
        ("审全部建议", "all"),
        ("只生成报告，不逐条判断", "report_only"),
    ]
    default_index = next((i for i, (_, value) in enumerate(review_options) if value == default_review_mode), 0)
    _, review_mode = prompt_named_choice("请选择这次体检的人工复核强度：", review_options, default_index=default_index)
    decisions["review_mode"] = review_mode
    if review_mode == "report_only":
        return decisions

    playbook_suggestions = sort_suggestions(list(report.get("playbook_edit_suggestions") or []))
    skill_suggestions = sort_suggestions(list(report.get("skill_edit_suggestions") or []))
    if review_mode == "high_priority":
        playbook_suggestions = [item for item in playbook_suggestions if str(item.get("priority") or "") == "high"]
        skill_suggestions = [item for item in skill_suggestions if str(item.get("priority") or "") == "high"]

    for suggestion_type, items in [("playbook", playbook_suggestions), ("skill", skill_suggestions)]:
        if not items:
            continue
        print(f"\n开始复核 {suggestion_type} 建议，共 {len(items)} 条。")
        for item in items:
            print("\n" + "=" * 72)
            for line in summarize_suggestion(item, suggestion_type=suggestion_type):
                print(line)
            decision = prompt_zero_one_two("这条建议怎么处理", default_value=2)
            if decision == 1:
                target_key = f"accepted_{suggestion_type}_suggestions"
            elif decision == 0:
                target_key = f"rejected_{suggestion_type}_suggestions"
            else:
                target_key = f"deferred_{suggestion_type}_suggestions"
            decisions[target_key].append(copy.deepcopy(item))
    return decisions


def render_decision_markdown(
    *,
    genre_key: str,
    report_path: Path,
    decisions: Mapping[str, Any],
) -> str:
    lines = [
        f"# 题材包人工复核结果：{genre_key}",
        "",
        f"- 对应报告：{report_path}",
        f"- 复核模式：{decisions.get('review_mode', 'report_only')}",
        "",
    ]
    for label, key in [
        ("已采纳 Playbook 建议", "accepted_playbook_suggestions"),
        ("已拒绝 Playbook 建议", "rejected_playbook_suggestions"),
        ("暂缓 Playbook 建议", "deferred_playbook_suggestions"),
        ("已采纳 Skill 建议", "accepted_skill_suggestions"),
        ("已拒绝 Skill 建议", "rejected_skill_suggestions"),
        ("暂缓 Skill 建议", "deferred_skill_suggestions"),
    ]:
        lines.extend([f"## {label}", ""])
        items = list(decisions.get(key) or [])
        if not items:
            lines.append("- 无")
            lines.append("")
            continue
        for item in items:
            if "source_field" in item:
                lines.append(
                    f"- {item.get('edit_id', '')} [{item.get('priority', '')}] {item.get('action', '')} {item.get('source_field', '')} -> {item.get('target_field', '')}"
                )
            else:
                lines.append(
                    f"- {item.get('edit_id', '')} [{item.get('priority', '')}] {item.get('action', '')} {item.get('source_section', '')} -> {item.get('target_section', '')}"
                )
            lines.append(f"  原文：{item.get('old_text', '') or '<空>'}")
            lines.append(f"  建议：{item.get('new_text', '') or '<删除>'}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def generate_refined_draft(
    *,
    genre_key: str,
    package: Mapping[str, Any],
    accepted_playbook_suggestions: list[dict[str, Any]],
    accepted_skill_suggestions: list[dict[str, Any]],
    audit_report: Mapping[str, Any],
    model: str,
    api_key: str,
    temperature: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    prompt = build_refine_prompt(
        genre_key=genre_key,
        package=package,
        accepted_playbook_suggestions=accepted_playbook_suggestions,
        accepted_skill_suggestions=accepted_skill_suggestions,
        audit_report=audit_report,
    )
    refined = openai_json_completion(
        model=model,
        api_key=api_key,
        system_prompt=render_prompt("genre_package_audit/refine_system.md", {}),
        user_prompt=prompt,
        schema_name="genre_package_refine_result",
        schema=REFINE_SCHEMA,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        stage="genre_package_audit",
        step_name="refine_genre_package",
    )
    playbook = json.loads(str(refined.get("playbook_json") or "{}"))
    if not isinstance(playbook, dict):
        raise RuntimeError("LLM 返回的 playbook_json 不是合法对象。")
    if str(refined.get("genre_key") or "").strip() and str(refined.get("genre_key") or "").strip() != genre_key:
        raise RuntimeError(f"LLM 返回的 genre_key 不匹配：{refined.get('genre_key')} != {genre_key}")
    playbook["genre_key"] = genre_key
    return {
        "genre_key": genre_key,
        "summary": str(refined.get("summary") or "").strip(),
        "playbook": playbook,
        "skill_markdown": str(refined.get("skill_markdown") or "").strip() + "\n",
    }


def maybe_apply_official_package(
    *,
    package: Mapping[str, Any],
    refined_draft: Mapping[str, Any],
    interactive_enabled: bool,
    source_draft_dir: Path | None = None,
    judgement_path: Path | None = None,
    cleanup_applied_draft: bool = True,
    apply_mode: str = "audit_only",
) -> dict[str, Any] | None:
    if not interactive_enabled:
        return None
    if not prompt_yes_no("是否把这份修订草稿直接写回正式题材包", default_value=False):
        return None
    return apply_official_package(
        package=package,
        refined_draft=refined_draft,
        apply_mode=apply_mode,
        source_draft_dir=source_draft_dir,
        judgement_path=judgement_path,
        cleanup_applied_draft=cleanup_applied_draft,
    )


def run_pipeline(config: Mapping[str, Any]) -> dict[str, Any]:
    packages = load_official_packages()
    genre_key = str(config.get("genre_key") or "").strip()
    if not genre_key:
        raise RuntimeError("缺少 genre_key。请在 config 中指定，或走交互模式。")
    if genre_key not in packages:
        available = "、".join(sorted(packages.keys()))
        raise RuntimeError(f"未找到题材 `{genre_key}`。当前可选：{available}")

    package = packages[genre_key]
    run_config = dict(config.get("run", {}))
    dry_run = bool(run_config.get("dry_run", False))
    apply_mode = str(run_config.get("apply_mode") or "audit_only").strip() or "audit_only"
    cleanup_applied_draft = bool(run_config.get("cleanup_applied_draft", True))
    selected_draft_dir = str(run_config.get("selected_draft_dir") or "").strip()
    if apply_mode not in {"audit_only", "audit_then_apply", "apply_existing_draft"}:
        raise RuntimeError(f"不支持的 apply_mode：{apply_mode}")

    if apply_mode == "apply_existing_draft":
        draft_package = resolve_existing_audit_draft(
            genre_key=genre_key,
            selected_draft_dir=selected_draft_dir,
        )
        judgement = dict(draft_package.get("judgement") or {})
        if dry_run:
            preview = {
                "genre_key": genre_key,
                "mode": "apply_existing_draft_dry_run",
                "selected_draft_dir": str(draft_package["draft_dir"]),
                "judgement_path": str(draft_package.get("judgement_path") or ""),
                "accepted_playbook_count": int(draft_package.get("accepted_playbook_count", 0)),
                "accepted_skill_count": int(draft_package.get("accepted_skill_count", 0)),
                "refine_summary": str(draft_package.get("refine_summary") or "").strip(),
                "official_playbook_path": str(package["playbook_path"]),
                "official_skill_path": str(package["skill_path"]),
            }
            audit_dir = AUDIT_ROOT / slugify(genre_key)
            ts = timestamp_tag()
            preview_json_path = save_json(audit_dir / f"apply_preview__{ts}.json", preview)
            preview_md_path = save_text(
                audit_dir / f"apply_preview__{ts}.md",
                render_apply_record_markdown(
                    {
                        "genre_key": genre_key,
                        "apply_mode": "apply_existing_draft_dry_run",
                        "official_playbook_path": str(package["playbook_path"]),
                        "official_skill_path": str(package["skill_path"]),
                        "backup_dir": "",
                        "source_draft_dir": str(draft_package["draft_dir"]),
                        "judgement_path": str(draft_package.get("judgement_path") or ""),
                        "draft_cleanup_status": "not_requested",
                        "draft_cleanup_error": "",
                        "refine_summary": str(draft_package.get("refine_summary") or "").strip(),
                    }
                ),
            )
            print_status(f"apply_existing_draft 预演已写入：{preview_md_path}")
            return {
                "mode": "apply_existing_draft_dry_run",
                "preview_json_path": str(preview_json_path),
                "preview_md_path": str(preview_md_path),
                "selected_draft_dir": str(draft_package["draft_dir"]),
                "judgement_path": str(draft_package.get("judgement_path") or ""),
            }

        refined_draft = {
            "genre_key": genre_key,
            "summary": str(judgement.get("refine_summary") or draft_package.get("refine_summary") or "").strip(),
            "playbook": dict(draft_package["playbook"]),
            "skill_markdown": str(draft_package["skill_text"]).rstrip() + "\n",
        }
        apply_result = apply_official_package(
            package=package,
            refined_draft=refined_draft,
            apply_mode=apply_mode,
            source_draft_dir=Path(draft_package["draft_dir"]),
            judgement_path=Path(draft_package["judgement_path"]) if draft_package.get("judgement_path") else None,
            cleanup_applied_draft=cleanup_applied_draft,
        )
        return {
            "mode": apply_mode,
            "selected_draft_dir": str(draft_package["draft_dir"]),
            "judgement_path": str(draft_package.get("judgement_path") or ""),
            "accepted_playbook_count": int(draft_package.get("accepted_playbook_count", 0)),
            "accepted_skill_count": int(draft_package.get("accepted_skill_count", 0)),
            "applied_official_changes": True,
            **apply_result,
        }

    precheck_report = build_precheck_report(
        genre_key=genre_key,
        package=package,
        packages=packages,
        max_related_genres=int(run_config.get("max_related_genres", 4)),
        max_duplicate_clusters=int(run_config.get("max_duplicate_clusters", 12)),
        max_examples_per_genre=int(run_config.get("max_examples_per_genre", 4)),
        max_suspicious_items=int(run_config.get("max_suspicious_items", 10)),
    )

    ts = timestamp_tag()
    audit_dir = AUDIT_ROOT / slugify(genre_key)
    audit_json_path = audit_dir / f"audit__{ts}.json"
    audit_md_path = audit_dir / f"audit__{ts}.md"

    if dry_run:
        preview = {
            "genre_key": genre_key,
            "mode": "dry_run",
            "apply_mode": apply_mode,
            "precheck_report": precheck_report,
            "official_playbook_path": str(package["playbook_path"]),
            "official_skill_path": str(package["skill_path"]),
        }
        save_json(audit_json_path, preview)
        save_text(
            audit_md_path,
            f"# 题材包体检预演：{genre_key}\n\n{render_precheck_markdown(precheck_report)}\n",
        )
        print_status(f"dry_run 预演已写入：{audit_md_path}")
        return {
            "audit_json_path": str(audit_json_path),
            "audit_md_path": str(audit_md_path),
            "mode": "dry_run",
        }

    model, api_key = configure_openai_api(config)
    audit_prompt = build_audit_prompt(
        genre_key=genre_key,
        package=package,
        precheck_report=precheck_report,
        config=config,
    )
    audit_report = openai_json_completion(
        model=model,
        api_key=api_key,
        system_prompt=render_prompt("genre_package_audit/audit_system.md", {}),
        user_prompt=audit_prompt,
        schema_name="genre_package_audit_report",
        schema=AUDIT_SCHEMA,
        temperature=float(run_config.get("temperature", 0.15)),
        timeout_seconds=int(run_config.get("timeout_seconds", 300)),
        stage="genre_package_audit",
        step_name="audit_genre_package",
    )
    save_json(audit_json_path, audit_report)
    save_text(audit_md_path, render_audit_markdown(audit_report, precheck_report))
    print_status(f"体检报告已写入：{audit_md_path}")

    interactive_enabled = sys.stdin.isatty() and bool(run_config.get("interactive_review", True))
    decisions = interactive_review(
        audit_report,
        interactive_enabled=interactive_enabled,
        default_review_mode=str(run_config.get("review_mode") or "high_priority"),
    )
    decision_json_path = audit_dir / f"decision__{ts}.json"
    decision_md_path = audit_dir / f"decision__{ts}.md"
    save_json(decision_json_path, decisions)
    save_text(
        decision_md_path,
        render_decision_markdown(
            genre_key=genre_key,
            report_path=audit_md_path,
            decisions=decisions,
        ),
    )

    accepted_playbook = list(decisions.get("accepted_playbook_suggestions") or [])
    accepted_skill = list(decisions.get("accepted_skill_suggestions") or [])
    if not accepted_playbook and not accepted_skill:
        print_status("当前没有被人工采纳的修改建议，本轮不会生成修订草稿。")
        return {
            "audit_json_path": str(audit_json_path),
            "audit_md_path": str(audit_md_path),
            "decision_json_path": str(decision_json_path),
            "decision_md_path": str(decision_md_path),
        }

    if interactive_enabled and not prompt_yes_no("是否基于已采纳建议生成修订草稿", default_value=True):
        return {
            "audit_json_path": str(audit_json_path),
            "audit_md_path": str(audit_md_path),
            "decision_json_path": str(decision_json_path),
            "decision_md_path": str(decision_md_path),
        }

    refined_draft = generate_refined_draft(
        genre_key=genre_key,
        package=package,
        accepted_playbook_suggestions=accepted_playbook,
        accepted_skill_suggestions=accepted_skill,
        audit_report=audit_report,
        model=model,
        api_key=api_key,
        temperature=min(float(run_config.get("temperature", 0.15)), 0.15),
        timeout_seconds=int(run_config.get("timeout_seconds", 300)),
    )

    draft_dir = GENRE_DRAFT_ROOT / f"audit__{slugify(genre_key)}__{ts}"
    draft_playbook_path = save_json(draft_dir / "playbook.json", refined_draft["playbook"])
    draft_skill_path = save_text(draft_dir / "skill.md", refined_draft["skill_markdown"])
    save_json(
        draft_dir / "judgement.json",
        {
            "genre_key": genre_key,
            "audit_report_path": str(audit_md_path),
            "decision_record_path": str(decision_md_path),
            "accepted_playbook_suggestions": accepted_playbook,
            "accepted_skill_suggestions": accepted_skill,
            "refine_summary": refined_draft["summary"],
        },
    )
    print_status(f"修订草稿已写入：{draft_dir}")
    apply_result: dict[str, Any] | None = None
    if apply_mode == "audit_then_apply":
        print_status("已选择“先体检再直接并入正式题材包”，开始写回正式文件。")
        apply_result = apply_official_package(
            package=package,
            refined_draft=refined_draft,
            apply_mode=apply_mode,
            source_draft_dir=draft_dir,
            judgement_path=decision_json_path,
            cleanup_applied_draft=cleanup_applied_draft,
        )
    else:
        apply_result = maybe_apply_official_package(
            package=package,
            refined_draft=refined_draft,
            interactive_enabled=interactive_enabled,
            source_draft_dir=draft_dir,
            judgement_path=decision_json_path,
            cleanup_applied_draft=cleanup_applied_draft,
            apply_mode=apply_mode,
        )
    return {
        "mode": apply_mode,
        "audit_json_path": str(audit_json_path),
        "audit_md_path": str(audit_md_path),
        "decision_json_path": str(decision_json_path),
        "decision_md_path": str(decision_md_path),
        "draft_dir": str(draft_dir),
        "draft_playbook_path": str(draft_playbook_path),
        "draft_skill_path": str(draft_skill_path),
        "applied_official_changes": bool(apply_result),
        "apply_result": apply_result or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a genre playbook/skill package with LLM + human review.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    config = load_runtime_config(args.config)
    result = run_pipeline(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
