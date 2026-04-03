from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = PROJECT_ROOT / "skills" / "production" / "video-script-reconstruction-skill"
GENRES_ROOT = SKILL_ROOT / "genres"
GENRE_DRAFT_ROOT = GENRES_ROOT / "__drafts__"
LEGACY_PLAYBOOK_DRAFT_ROOT = SKILL_ROOT / "playbooks" / "__drafts__"
AUTO_DROP_SIMILARITY_THRESHOLD = 0.5


def print_status(message: str) -> None:
    print(f"[genre-draft-review] {message}", flush=True)


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


def normalize_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(text).strip().lower())


def slugify(text: str) -> str:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "_", str(text).strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "untitled"


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


@dataclass
class OfficialPackage:
    genre_key: str
    dir_path: Path
    playbook_path: Path
    skill_path: Path
    playbook: dict[str, Any]
    skill_text: str
    changed: bool = False
    exists_on_disk: bool = True


@dataclass
class ReviewStats:
    interactive_reviewed: int = 0
    auto_dropped: int = 0

    @property
    def total_handled(self) -> int:
        return self.interactive_reviewed + self.auto_dropped


def load_official_packages() -> dict[str, OfficialPackage]:
    result: dict[str, OfficialPackage] = {}
    for path in sorted(GENRES_ROOT.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.name == "__drafts__":
            continue
        playbook_path = path / "playbook.json"
        skill_path = path / "skill.md"
        if not playbook_path.exists() or not skill_path.exists():
            continue
        playbook = load_json(playbook_path)
        genre_key = str(playbook.get("genre_key", "")).strip()
        if not genre_key:
            continue
        result[genre_key] = OfficialPackage(
            genre_key=genre_key,
            dir_path=path.resolve(),
            playbook_path=playbook_path.resolve(),
            skill_path=skill_path.resolve(),
            playbook=playbook,
            skill_text=skill_path.read_text(encoding="utf-8"),
            exists_on_disk=True,
        )
    return result


def ensure_official_package(packages: dict[str, OfficialPackage], genre_key: str) -> OfficialPackage:
    existing = packages.get(genre_key)
    if existing:
        return existing
    dir_path = GENRES_ROOT / genre_key
    playbook = {
        "genre_key": genre_key,
        "aliases": [genre_key],
        "core_audience_promises": [],
        "script_hooks": [],
        "character_design_focus": [],
        "scene_design_focus": [],
        "storyboard_focus": [],
    }
    skill_text = f"# 题材补充 Skill：{genre_key}\n"
    package = OfficialPackage(
        genre_key=genre_key,
        dir_path=dir_path.resolve(),
        playbook_path=(dir_path / "playbook.json").resolve(),
        skill_path=(dir_path / "skill.md").resolve(),
        playbook=playbook,
        skill_text=skill_text,
        changed=False,
        exists_on_disk=False,
    )
    packages[genre_key] = package
    return package


def parse_skill_candidates_from_markdown(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    section = ""
    result: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in lines:
        if raw.startswith("## "):
            section = raw[3:].strip()
            continue
        match = re.match(r"- \[(\d+(?:\.\d+)?)\]\s*(.+)", raw.strip())
        if match:
            current = {
                "section_label": section,
                "text": match.group(2).strip(),
                "confidence": float(match.group(1)),
                "reason": "",
                "candidate_type": "skill",
            }
            result.append(current)
            continue
        if current and raw.strip().startswith("说明："):
            current["reason"] = raw.strip().replace("说明：", "", 1).strip()
            current = None
    return result


def parse_official_skill_rules(skill_text: str) -> list[dict[str, str]]:
    section = "基础规则"
    result: list[dict[str, str]] = []
    for raw in skill_text.splitlines():
        if raw.startswith("## "):
            section = raw[3:].strip()
            continue
        if raw.startswith("### "):
            section = raw[4:].strip()
            continue
        stripped = raw.strip()
        if stripped.startswith("- "):
            result.append({"section": section, "text": stripped[2:].strip()})
    return result


def find_similar_playbook_rule(candidate_text: str, existing_values: list[str]) -> tuple[str, float]:
    best_text = ""
    best_score = 0.0
    for item in existing_values:
        score = similarity(candidate_text, item)
        if score > best_score:
            best_text = item
            best_score = score
    return best_text, best_score


def find_similar_skill_rule(candidate_text: str, existing_rules: list[dict[str, str]]) -> tuple[str, float]:
    best_text = ""
    best_score = 0.0
    for item in existing_rules:
        score = similarity(candidate_text, item.get("text", ""))
        if score > best_score:
            best_text = item.get("text", "")
            best_score = score
    return best_text, best_score


def prompt_zero_one(prompt: str, default_value: int) -> int:
    default_text = "1" if default_value == 1 else "0"
    while True:
        try:
            raw = input(f"{prompt}（1/0，默认 {default_text}）：").strip()
        except EOFError:
            print_status(f"{prompt} 未读取到交互输入，按默认 {default_text} 处理。")
            return default_value
        if not raw:
            return default_value
        if raw in {"0", "1"}:
            return int(raw)
        print("输入无效，请输入 1 或 0。")


def add_playbook_rule(package: OfficialPackage, target_field: str, text: str) -> bool:
    existing = [str(item).strip() for item in package.playbook.get(target_field, [])]
    if any(similarity(text, item) >= 0.93 for item in existing):
        return False
    package.playbook[target_field] = existing + [text]
    package.changed = True
    return True


def add_skill_rule(package: OfficialPackage, section_label: str, text: str) -> bool:
    existing_rules = parse_official_skill_rules(package.skill_text)
    if any(similarity(text, item.get("text", "")) >= 0.93 for item in existing_rules):
        return False
    managed_header = "## 人工审核并入经验"
    section_header = f"### {section_label}"
    content = package.skill_text.rstrip()
    if managed_header not in content:
        content += f"\n\n{managed_header}\n\n{section_header}\n- {text}\n"
        package.skill_text = content
        package.changed = True
        return True
    before, after = content.split(managed_header, 1)
    managed = managed_header + after
    if section_header not in managed:
        managed = managed.rstrip() + f"\n\n{section_header}\n- {text}\n"
        package.skill_text = before.rstrip() + "\n\n" + managed.strip() + "\n"
        package.changed = True
        return True
    parts = managed.split(section_header, 1)
    prefix = parts[0] + section_header
    suffix = parts[1]
    next_section = re.search(r"\n###\s+", suffix)
    if next_section:
        insert_at = next_section.start()
        section_body = suffix[:insert_at].rstrip()
        tail = suffix[insert_at:]
    else:
        section_body = suffix.rstrip()
        tail = ""
    section_body += f"\n- {text}"
    managed_new = prefix + section_body + tail
    package.skill_text = before.rstrip() + "\n\n" + managed_new.strip() + "\n"
    package.changed = True
    return True


def save_package(package: OfficialPackage) -> None:
    save_json(package.playbook_path, package.playbook)
    save_text(package.skill_path, package.skill_text)


def collect_draft_entries(series_name: str | None = None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    handled_legacy: set[str] = set()
    series_slug = slugify(series_name) if series_name else ""
    if GENRE_DRAFT_ROOT.exists():
        for path in sorted(GENRE_DRAFT_ROOT.iterdir(), key=lambda item: item.name):
            if not path.is_dir():
                continue
            playbook_path = path / "playbook.json"
            skill_path = path / "skill.md"
            if not playbook_path.exists() and not skill_path.exists():
                continue
            payload = load_json(playbook_path) if playbook_path.exists() else {}
            genre_key = str(payload.get("genre_key", "")).strip()
            legacy_path = LEGACY_PLAYBOOK_DRAFT_ROOT / f"{path.name}.json"
            playbook_candidates = list(payload.get("playbook_update_candidates", []))
            skill_candidates = list(payload.get("skill_update_candidates", []))
            if not skill_candidates:
                skill_candidates = parse_skill_candidates_from_markdown(skill_path)
            entries.append(
                {
                    "entry_name": path.name,
                    "genre_key": genre_key,
                    "source_series": str(payload.get("source_series", "")).strip(),
                    "draft_dir": path.resolve(),
                    "legacy_path": legacy_path.resolve() if legacy_path.exists() else None,
                    "playbook_candidates": playbook_candidates,
                    "skill_candidates": skill_candidates,
                }
            )
            handled_legacy.add(path.name)
    if LEGACY_PLAYBOOK_DRAFT_ROOT.exists():
        for path in sorted(LEGACY_PLAYBOOK_DRAFT_ROOT.glob("*.json"), key=lambda item: item.name):
            if path.stem in handled_legacy:
                continue
            payload = load_json(path)
            entries.append(
                {
                    "entry_name": path.stem,
                    "genre_key": str(payload.get("genre_key", "")).strip(),
                    "source_series": str(payload.get("source_series", payload.get("_source_series", ""))).strip(),
                    "draft_dir": None,
                    "legacy_path": path.resolve(),
                    "playbook_candidates": list(payload.get("playbook_update_candidates", [])),
                    "skill_candidates": list(payload.get("skill_update_candidates", [])),
                }
            )
    if not series_name:
        return entries
    filtered: list[dict[str, Any]] = []
    expected_prefix = f"sync__{series_slug}__"
    for entry in entries:
        source_series = str(entry.get("source_series", "")).strip()
        entry_name = str(entry.get("entry_name", "")).strip()
        if source_series == series_name or entry_name.startswith(expected_prefix):
            filtered.append(entry)
    return filtered


def review_playbook_candidates(package: OfficialPackage, candidates: list[dict[str, Any]]) -> ReviewStats:
    sorted_candidates = sorted(candidates, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    stats = ReviewStats()
    for index, candidate in enumerate(sorted_candidates, start=1):
        target_field = str(candidate.get("target_field", "")).strip()
        text = str(candidate.get("full_text") or candidate.get("text", "")).strip()
        if not target_field or not text:
            continue
        existing_values = [str(item).strip() for item in package.playbook.get(target_field, [])]
        similar_text, similar_score = find_similar_playbook_rule(text, existing_values)
        if similar_score > AUTO_DROP_SIMILARITY_THRESHOLD:
            stats.auto_dropped += 1
            print_status(
                f"[Playbook {index}/{len(sorted_candidates)}] 题材={package.genre_key} | "
                f"{candidate.get('target_field_label', target_field)} | 置信度={float(candidate.get('confidence', 0.0)):.2f} | "
                f"与现有规则相似度 {similar_score:.2f} > {AUTO_DROP_SIMILARITY_THRESHOLD:.2f}，已自动删除候选。"
            )
            print(f"候选：{text}")
            if similar_text:
                print(f"命中现有规则：{similar_text}")
            continue
        stats.interactive_reviewed += 1
        recommended = 0 if similar_score >= 0.84 else 1
        print_status(
            f"[Playbook {index}/{len(sorted_candidates)}] 题材={package.genre_key} | "
            f"{candidate.get('target_field_label', target_field)} | 置信度={float(candidate.get('confidence', 0.0)):.2f}"
        )
        print(f"候选：{text}")
        if candidate.get("reason"):
            print(f"说明：{candidate.get('reason', '')}")
        if similar_text:
            print(f"相似现有规则（{similar_score:.2f}）：{similar_text}")
        else:
            print("相似现有规则：无")
        decision = prompt_zero_one("是否并入正式 playbook", recommended)
        if decision != 1:
            print_status("已跳过。")
            continue
        if add_playbook_rule(package, target_field, text):
            print_status("已并入正式 playbook。")
        else:
            print_status("检测到已存在高度相似规则，未重复写入。")
    return stats


def review_skill_candidates(package: OfficialPackage, candidates: list[dict[str, Any]]) -> ReviewStats:
    sorted_candidates = sorted(candidates, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    stats = ReviewStats()
    for index, candidate in enumerate(sorted_candidates, start=1):
        text = str(candidate.get("full_text") or candidate.get("text", "")).strip()
        section_label = str(candidate.get("section_label", "") or candidate.get("section", "")).strip() or "补充经验"
        if not text:
            continue
        similar_text, similar_score = find_similar_skill_rule(text, parse_official_skill_rules(package.skill_text))
        if similar_score > AUTO_DROP_SIMILARITY_THRESHOLD:
            stats.auto_dropped += 1
            print_status(
                f"[Skill {index}/{len(sorted_candidates)}] 题材={package.genre_key} | "
                f"{section_label} | 置信度={float(candidate.get('confidence', 0.0)):.2f} | "
                f"与现有规则相似度 {similar_score:.2f} > {AUTO_DROP_SIMILARITY_THRESHOLD:.2f}，已自动删除候选。"
            )
            print(f"候选：{text}")
            if similar_text:
                print(f"命中现有规则：{similar_text}")
            continue
        stats.interactive_reviewed += 1
        recommended = 0 if similar_score >= 0.84 else 1
        print_status(
            f"[Skill {index}/{len(sorted_candidates)}] 题材={package.genre_key} | "
            f"{section_label} | 置信度={float(candidate.get('confidence', 0.0)):.2f}"
        )
        print(f"候选：{text}")
        if candidate.get("reason"):
            print(f"说明：{candidate.get('reason', '')}")
        if similar_text:
            print(f"相似现有规则（{similar_score:.2f}）：{similar_text}")
        else:
            print("相似现有规则：无")
        decision = prompt_zero_one("是否并入正式 skill", recommended)
        if decision != 1:
            print_status("已跳过。")
            continue
        if add_skill_rule(package, section_label, text):
            print_status("已并入正式 skill。")
        else:
            print_status("检测到已存在高度相似规则，未重复写入。")
    return stats


def cleanup_entry(entry: dict[str, Any]) -> None:
    draft_dir = entry.get("draft_dir")
    if draft_dir:
        path = Path(draft_dir)
        for child in path.iterdir():
            child.unlink(missing_ok=True)
        path.rmdir()
        print_status(f"已删除 draft 目录：{path}")
    legacy_path = entry.get("legacy_path")
    if legacy_path:
        Path(legacy_path).unlink(missing_ok=True)
        print_status(f"已删除 legacy draft：{legacy_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review genre draft candidates and merge them into official genre packages.")
    parser.add_argument("--series-name", default="", help="只审核指定剧名本次同步生成的 draft。")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    series_name = str(args.series_name or "").strip()
    if not GENRE_DRAFT_ROOT.exists() and not LEGACY_PLAYBOOK_DRAFT_ROOT.exists():
        print_status("未找到任何 draft 目录。")
        return
    entries = collect_draft_entries(series_name=series_name or None)
    if not entries:
        if series_name:
            print_status(f"当前没有待审核的题材更新 draft：series={series_name}")
        else:
            print_status("当前没有待审核的题材更新 draft。")
        return

    packages = load_official_packages()
    if series_name:
        print_status(f"发现 {len(entries)} 组待审核 draft（已限定剧名：{series_name}）。")
    else:
        print_status(f"发现 {len(entries)} 组待审核 draft。")
    for entry_index, entry in enumerate(entries, start=1):
        genre_key = str(entry.get("genre_key", "")).strip()
        if not genre_key:
            print_status(f"第 {entry_index} 组 draft 缺少 genre_key，已跳过并保留，等待人工处理。")
            continue
        print_status(
            f"开始审核第 {entry_index}/{len(entries)} 组：题材={genre_key} | 来源剧目={entry.get('source_series', '') or '未知'}"
        )
        package = ensure_official_package(packages, genre_key)
        reviewed_playbook = review_playbook_candidates(package, list(entry.get("playbook_candidates", [])))
        reviewed_skill = review_skill_candidates(package, list(entry.get("skill_candidates", [])))
        reviewed_total = reviewed_playbook.total_handled + reviewed_skill.total_handled
        interactive_total = reviewed_playbook.interactive_reviewed + reviewed_skill.interactive_reviewed
        auto_dropped_total = reviewed_playbook.auto_dropped + reviewed_skill.auto_dropped
        if package.changed:
            save_package(package)
            package.changed = False
            print_status(f"正式题材包已保存：{package.dir_path}")
        if reviewed_total == 0:
            print_status("当前 draft 没有任何可审核候选，默认保留，不自动删除。")
            continue
        if auto_dropped_total:
            print_status(
                f"本组候选已处理：人工审核 {interactive_total} 条，自动删除重复候选 {auto_dropped_total} 条。"
            )
        if interactive_total == 0 and auto_dropped_total > 0:
            print_status("本组 draft 仅包含高相似重复候选，已直接清理，无需额外交互。")
            cleanup_entry(entry)
            continue
        cleanup_decision = prompt_zero_one("是否删除本组 draft 文件", 1)
        if cleanup_decision == 1:
            cleanup_entry(entry)
        else:
            print_status("已保留本组 draft 文件。")
    print_status("全部 draft 审核完成。")


if __name__ == "__main__":
    main()
