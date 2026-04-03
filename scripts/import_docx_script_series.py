from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from providers.base import utc_timestamp

DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
EPISODE_HEADING_RE = re.compile(r"^第([一二三四五六七八九十百零〇两\d]+)集$")
CHARACTER_LINE_RE = re.compile(r"^([^：:]{1,20})[：:]\s*(.+)$")
SCENE_HEADING_RE = re.compile(r"^(?:\d+-\d+、)?([^，。,]+)[，,](日|夜|白天|夜晚|傍晚|清晨|凌晨|午后)[，,](内|外|内景|外景)[。.]?$")


@dataclass
class EpisodeBlock:
    episode_id: str
    episode_number: int
    heading: str
    lines: list[str]


def read_docx_paragraphs(docx_path: Path) -> list[str]:
    with ZipFile(docx_path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []
    for para in root.findall(".//w:p", DOCX_NS):
        texts: list[str] = []
        for node in para.findall(".//w:t", DOCX_NS):
            if node.text:
                texts.append(node.text)
        text = "".join(texts).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def chinese_number_to_int(value: str) -> int:
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100}
    total = 0
    current = 0
    for char in stripped:
        if char in digits:
            current = digits[char]
        elif char in units:
            unit = units[char]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
    return total + current


def normalize_series_name(title: str) -> str:
    base = re.split(r"[，,：:《》\s]+", title.strip())[0].strip()
    if 2 <= len(base) <= 20:
        return base
    cleaned = re.sub(r"[《》“”\"'，,：:·\s]+", "", title.strip())
    return cleaned[:20] or title.strip()[:20]


def safe_json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(safe_json_dump(data), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def split_sections(paragraphs: list[str]) -> tuple[str, str, str, str, list[str], list[str], list[EpisodeBlock]]:
    if not paragraphs:
        raise ValueError("docx 解析后没有正文段落。")

    title = paragraphs[0]
    type_line = ""
    volume_line = ""
    intro_line = ""
    character_start = None
    outline_start = None
    episode_indexes: list[tuple[int, int, str]] = []

    for index, text in enumerate(paragraphs):
        if text.startswith("类型：") or text.startswith("类型:"):
            type_line = text
        elif text.startswith("体量：") or text.startswith("体量:"):
            volume_line = text
        elif text.startswith("简介：") or text.startswith("简介:"):
            intro_line = text
        elif text == "人物设定":
            character_start = index
        elif text == "剧本大纲":
            outline_start = index
        else:
            match = EPISODE_HEADING_RE.match(text)
            if match:
                episode_indexes.append((index, chinese_number_to_int(match.group(1)), text))

    if character_start is None or outline_start is None or not episode_indexes:
        raise ValueError("未能识别到人物设定、剧本大纲或集数标题，暂时无法自动切分。")

    character_lines = paragraphs[character_start + 1 : outline_start]
    first_episode_index = episode_indexes[0][0]
    outline_lines = paragraphs[outline_start + 1 : first_episode_index]

    episode_blocks: list[EpisodeBlock] = []
    for idx, (start_index, episode_number, heading) in enumerate(episode_indexes):
        end_index = episode_indexes[idx + 1][0] if idx + 1 < len(episode_indexes) else len(paragraphs)
        lines = paragraphs[start_index + 1 : end_index]
        episode_blocks.append(
            EpisodeBlock(
                episode_id=f"ep{episode_number:02d}",
                episode_number=episode_number,
                heading=heading,
                lines=lines,
            )
        )

    return title, type_line, volume_line, intro_line, character_lines, outline_lines, episode_blocks


def extract_type_tokens(type_line: str, file_name: str) -> list[str]:
    raw = type_line.split("：", 1)[-1].split(":", 1)[-1].strip()
    tokens = [token.strip() for token in re.split(r"[\s/、，,]+", raw) if token.strip()]
    for extra in ["甜宠", "女主", "都市"]:
        if extra in file_name and extra not in tokens:
            tokens.append(extra)
    return tokens


def canonical_id_from_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "_", name).strip("_")
    return f"char_{cleaned or 'unknown'}"


def parse_character_profiles(character_lines: list[str], episode_blocks: list[EpisodeBlock]) -> list[dict[str, Any]]:
    joined_episode_text = "\n".join("\n".join(block.lines) for block in episode_blocks)
    profiles: list[dict[str, Any]] = []
    for line in character_lines:
        match = CHARACTER_LINE_RE.match(line)
        if not match:
            continue
        name = match.group(1).strip()
        description = match.group(2).strip()
        aliases: list[str] = [name]
        role = "配角"
        if "女主" in description:
            role = "主角"
        elif "男主" in description:
            role = "男主"
        elif "反派" in description:
            role = "反派"
        relationship = "自身" if role == "主角" else "关键关系角色"
        seen = [
            block.episode_id
            for block in episode_blocks
            if name in "\n".join(block.lines)
        ]
        latest_episode = seen[-1] if seen else episode_blocks[0].episode_id
        first_episode = seen[0] if seen else episode_blocks[0].episode_id
        visual_profile = description.split("性格", 1)[0].strip("，。；; ")
        if not visual_profile:
            visual_profile = description[:80]
        profiles.append(
            {
                "canonical_id": canonical_id_from_name(name),
                "canonical_name": name,
                "aliases": aliases,
                "role": role,
                "relationship_to_protagonist": relationship,
                "visual_profile": visual_profile,
                "latest_state": description,
                "first_episode": first_episode,
                "latest_episode": latest_episode,
                "seen_episodes": seen,
                "state_history": [
                    {
                        "episode_id": latest_episode,
                        "state": description,
                    }
                ],
            }
        )
    if not profiles and joined_episode_text:
        profiles.append(
            {
                "canonical_id": "char_女主",
                "canonical_name": "女主",
                "aliases": ["女主"],
                "role": "主角",
                "relationship_to_protagonist": "自身",
                "visual_profile": "",
                "latest_state": "原始 docx 中未解析出结构化人物卡，需后续补录。",
                "first_episode": episode_blocks[0].episode_id,
                "latest_episode": episode_blocks[-1].episode_id,
                "seen_episodes": [block.episode_id for block in episode_blocks],
                "state_history": [],
            }
        )
    return profiles


def compact_line(text: str) -> str:
    return re.sub(r"^[△【】\[\]（）()]+", "", text).strip()


def infer_episode_summary(block: EpisodeBlock) -> tuple[str, list[str], list[str], list[str]]:
    involved_characters: list[str] = []
    locations: list[str] = []
    summary_candidates: list[str] = []
    key_events: list[str] = []

    for line in block.lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("【出场人物：") and "】" in stripped:
            raw_names = stripped.split("：", 1)[1].rsplit("】", 1)[0]
            involved_characters.extend([item.strip() for item in re.split(r"[，,、]", raw_names) if item.strip()])
            continue
        scene_match = SCENE_HEADING_RE.match(stripped)
        if scene_match:
            locations.append(scene_match.group(1).strip())
            continue
        cleaned = compact_line(stripped)
        if not cleaned:
            continue
        if len(summary_candidates) < 6:
            summary_candidates.append(cleaned)
        if len(key_events) < 8 and not cleaned.startswith(("OS", "VO")):
            key_events.append(cleaned)

    synopsis = "；".join(summary_candidates[:4]).strip("；")
    synopsis = synopsis[:220] if synopsis else f"{block.heading} 已从原始 docx 导入，待后续 agent 深化整理。"
    return synopsis, list(dict.fromkeys(key_events)), list(dict.fromkeys(involved_characters)), list(dict.fromkeys(locations))


def build_source_outline(
    *,
    docx_path: Path,
    series_name: str,
    title: str,
    type_line: str,
    volume_line: str,
    intro_line: str,
    character_lines: list[str],
    outline_lines: list[str],
    episode_blocks: list[EpisodeBlock],
) -> dict[str, Any]:
    type_tokens = extract_type_tokens(type_line, docx_path.name)
    return {
        "series_name": series_name,
        "source_docx_path": str(docx_path.resolve()),
        "source_title": title,
        "source_type_line": type_line,
        "source_volume_line": volume_line,
        "source_intro_line": intro_line,
        "type_tokens": type_tokens,
        "character_lines": character_lines,
        "outline_lines": outline_lines,
        "episode_count_detected": len(episode_blocks),
        "episode_headings": [
            {
                "episode_id": block.episode_id,
                "episode_number": block.episode_number,
                "heading": block.heading,
            }
            for block in episode_blocks
        ],
    }


def build_series_bible(
    *,
    series_name: str,
    title: str,
    intro_line: str,
    outline_lines: list[str],
    type_tokens: list[str],
    character_profiles: list[dict[str, Any]],
    episode_blocks: list[EpisodeBlock],
) -> dict[str, Any]:
    intro = intro_line.split("：", 1)[-1].split(":", 1)[-1].strip()
    main_arc = outline_lines[:8]
    themes = [token for token in type_tokens if token not in {"类型"}]
    unresolved_threads = outline_lines[8:16]
    return {
        "series_name": series_name,
        "updated_at": utc_timestamp(),
        "title": title,
        "premise": intro,
        "genre_profile": {
            "primary_genre": type_tokens[0] if type_tokens else "未知",
            "secondary_genres": type_tokens[1:4],
            "narrative_device": "寡妇系统 / 穿越者契约婚姻 / 继承遗产反向致富",
            "audience_expectation": "高频反转、契约婚姻爽点、女主持续升级做大做强。",
        },
        "themes": themes[:6],
        "core_characters": [
            {
                "name": item["canonical_name"],
                "role": item["role"],
                "summary": item["latest_state"],
            }
            for item in character_profiles[:12]
        ],
        "main_arc_outline": main_arc,
        "unresolved_threads": unresolved_threads,
        "episode_count_detected": len(episode_blocks),
    }


def build_series_context(
    *,
    series_name: str,
    intro_line: str,
    character_profiles: list[dict[str, Any]],
    episode_summaries: list[dict[str, Any]],
    type_tokens: list[str],
    outline_lines: list[str],
) -> dict[str, Any]:
    premise = intro_line.split("：", 1)[-1].split(":", 1)[-1].strip()
    active_locations: list[dict[str, Any]] = []
    location_seen: set[str] = set()
    for summary in episode_summaries:
        for location in summary.get("locations", []):
            if location in location_seen:
                continue
            location_seen.add(location)
            active_locations.append(
                {
                    "name": location,
                    "time_of_day": "未标定",
                    "visual_profile": "由原始导入剧本识别出的常用场景，后续可在导演分析阶段继续细化。",
                }
            )

    return {
        "series_name": series_name,
        "updated_at": utc_timestamp(),
        "premise": premise,
        "latest_episode_id": episode_summaries[-1]["episode_id"] if episode_summaries else "ep01",
        "continuity_rules": [
            "当前系列为 docx 冷启动导入，后续新增分析时应优先保留原始剧情主线与人物关系。",
            "如后续改稿或分镜对人物动机、关系走向作强化，需保持前后集承接一致，不得破坏原始设定。",
            "未在当前剧本中直接写明的内容，只能作为题材经验参考，不能覆盖原文硬事实。",
        ],
        "genre_profile": {
            "primary_genre": type_tokens[0] if type_tokens else "未知",
            "secondary_genres": type_tokens[1:4],
            "narrative_device": "寡妇系统 / 穿越者联姻 / 守寡继承 / 女主逆袭致富",
            "audience_expectation": "都市幻想轻喜节奏下的高密度结婚-守寡-继承-升级爽点。",
        },
        "genre_playbooks": [],
        "downstream_design_guidance": {
            "script_reconstruction_focus": [
                "保留女主明艳爽感与系统流快节奏信息释放。",
                "强化每集的结婚、守寡、继承、翻车与反转节点。",
                "兼顾轻喜荒诞感与都市豪门爽感。 ",
            ],
            "character_design_focus": [
                "陆昭昭应具备明艳活泼、能扛事又带喜感的核心辨识度。",
                "傅砚舟需要有高岭之花、冷感权势与刑侦观察者气质。",
                "不同穿越者丈夫与豪门配角应做明显身份差异化。 ",
            ],
            "scene_design_focus": [
                "海边分手、民政局闪婚、豪门办公室、医院急救室等高识别都市爽剧空间应优先强化。",
                "系统触发与财富逆袭节点适合加入夸张视觉提示。 ",
            ],
            "storyboard_focus": [
                "每集前半段尽快抛出钩子，后半段落在反转或继承结果上。",
                "多用人物近景、反应镜头和信息揭示镜头强化爽感。 ",
            ],
        },
        "active_characters": [
            {
                "name": item["canonical_name"],
                "role": item["role"],
                "relationship_to_protagonist": item["relationship_to_protagonist"],
                "latest_state": item["latest_state"],
            }
            for item in character_profiles[:12]
        ],
        "active_locations": active_locations,
        "unresolved_threads": outline_lines[4:14],
        "recent_timeline": [
            {
                "episode_id": item["episode_id"],
                "title": item["title"],
                "synopsis": item["synopsis"],
                "key_events": item["key_events"],
                "unresolved_threads": [],
                "continuity_hooks": [],
                "involved_characters": item["involved_characters"],
            }
            for item in episode_summaries
        ],
    }


def render_source_outline_markdown(
    *,
    title: str,
    series_name: str,
    type_line: str,
    volume_line: str,
    intro_line: str,
    character_lines: list[str],
    outline_lines: list[str],
    episode_blocks: list[EpisodeBlock],
) -> str:
    character_block = "\n".join(f"- {line}" for line in character_lines)
    outline_block = "\n".join(f"{idx}. {line}" for idx, line in enumerate(outline_lines, 1))
    episode_block = "\n".join(f"- {block.episode_id}: {block.heading}" for block in episode_blocks)
    resolved_type = type_line.split("：", 1)[-1].split(":", 1)[-1].strip()
    resolved_volume = volume_line.split("：", 1)[-1].split(":", 1)[-1].strip()
    resolved_intro = intro_line.split("：", 1)[-1].split(":", 1)[-1].strip()
    lines = [
        "# 原始剧本导入总览",
        "",
        f"- 剧名：{title}",
        f"- 系列目录名：{series_name}",
        f"- 类型：{resolved_type}",
        f"- 体量：{resolved_volume}",
        f"- 已识别剧集数：{len(episode_blocks)}",
        "",
        "## 简介",
        "",
        resolved_intro,
        "",
        "## 人物设定",
        "",
        character_block,
        "",
        "## 剧本大纲",
        "",
        outline_block,
        "",
        "## 已切分剧集",
        "",
        episode_block,
        "",
    ]
    return "\n".join(lines)


def render_episode_markdown(title: str, block: EpisodeBlock) -> str:
    body = "\n\n".join(line.strip() for line in block.lines if line.strip())
    return f"第{block.episode_number}集：{title}\n\n{body}\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将整部 docx 剧本切分为标准剧集目录，并生成冷启动 analysis 资料。")
    parser.add_argument("--docx-path", required=True, help="原始 docx 路径。")
    parser.add_argument("--script-root", default="script", help="剧本输出根目录。")
    parser.add_argument("--analysis-root", default="analysis", help="分析输出根目录。")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的同名文件。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docx_path = Path(args.docx_path).expanduser().resolve()
    if not docx_path.exists():
        raise FileNotFoundError(f"未找到 docx：{docx_path}")

    script_root = Path(args.script_root)
    if not script_root.is_absolute():
        script_root = (PROJECT_ROOT / script_root).resolve()
    analysis_root = Path(args.analysis_root)
    if not analysis_root.is_absolute():
        analysis_root = (PROJECT_ROOT / analysis_root).resolve()

    paragraphs = read_docx_paragraphs(docx_path)
    title, type_line, volume_line, intro_line, character_lines, outline_lines, episode_blocks = split_sections(paragraphs)
    series_name = normalize_series_name(title)
    type_tokens = extract_type_tokens(type_line, docx_path.name)

    series_script_dir = script_root / series_name
    series_analysis_dir = analysis_root / series_name
    if series_script_dir.exists() and any(series_script_dir.iterdir()) and not args.force:
        raise FileExistsError(f"剧本目录已存在且非空：{series_script_dir}；如需覆盖请加 --force")

    character_profiles = parse_character_profiles(character_lines, episode_blocks)
    episode_summaries: list[dict[str, Any]] = []
    for block in episode_blocks:
        synopsis, key_events, involved_characters, locations = infer_episode_summary(block)
        episode_summaries.append(
            {
                "episode_id": block.episode_id,
                "title": block.heading,
                "synopsis": synopsis,
                "key_events": key_events,
                "involved_characters": involved_characters,
                "locations": locations,
            }
        )

    source_outline = build_source_outline(
        docx_path=docx_path,
        series_name=series_name,
        title=title,
        type_line=type_line,
        volume_line=volume_line,
        intro_line=intro_line,
        character_lines=character_lines,
        outline_lines=outline_lines,
        episode_blocks=episode_blocks,
    )
    series_bible = build_series_bible(
        series_name=series_name,
        title=title,
        intro_line=intro_line,
        outline_lines=outline_lines,
        type_tokens=type_tokens,
        character_profiles=character_profiles,
        episode_blocks=episode_blocks,
    )
    series_context = build_series_context(
        series_name=series_name,
        intro_line=intro_line,
        character_profiles=character_profiles,
        episode_summaries=episode_summaries,
        type_tokens=type_tokens,
        outline_lines=outline_lines,
    )

    for block in episode_blocks:
        write_text(series_script_dir / f"{block.episode_id}__openai__gpt-5.4.md", render_episode_markdown(title, block))

    write_json(series_analysis_dir / "source_outline.json", source_outline)
    write_text(
        series_analysis_dir / "source_outline.md",
        render_source_outline_markdown(
            title=title,
            series_name=series_name,
            type_line=type_line,
            volume_line=volume_line,
            intro_line=intro_line,
            character_lines=character_lines,
            outline_lines=outline_lines,
            episode_blocks=episode_blocks,
        ),
    )
    write_json(series_analysis_dir / "series_bible.json", series_bible)
    write_json(series_analysis_dir / "series_context.json", series_context)
    write_json(
        series_analysis_dir / "character_registry.json",
        {
            "series_name": series_name,
            "updated_at": utc_timestamp(),
            "characters": character_profiles,
        },
    )
    write_json(
        series_analysis_dir / "import_manifest.json",
        {
            "series_name": series_name,
            "source_docx_path": str(docx_path),
            "imported_at": utc_timestamp(),
            "episode_count_detected": len(episode_blocks),
            "script_output_dir": str(series_script_dir),
            "analysis_output_dir": str(series_analysis_dir),
        },
    )
    for item in episode_summaries:
        write_json(series_analysis_dir / "episode_summaries" / f"{item['episode_id']}.json", item)

    print(safe_json_dump(
        {
            "series_name": series_name,
            "title": title,
            "episode_count_detected": len(episode_blocks),
            "script_output_dir": str(series_script_dir),
            "analysis_output_dir": str(series_analysis_dir),
        }
    ))


if __name__ == "__main__":
    main()
