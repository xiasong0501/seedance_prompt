from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_seedance_api_script import choose_from_list, episode_sort_key, list_series_dirs
from generate_seedance_prompts import (
    SEEDANCE_PROMPTS_SCHEMA,
    normalize_storyboard_result,
    render_seedance_markdown,
    repair_storyboard_density,
)
from openai_agents.runtime_utils import configure_openai_api, load_runtime_config, openai_json_completion

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "openai_agent_flow.local.json"

SCENE_REF_IDS = {f"@图片{i}" for i in range(10, 16)}
SERIES_SUFFIX_RE = re.compile(r"-(gpt|claude)(?:[-_].+)?$", re.IGNORECASE)
ASSET_ROW_RE = re.compile(r"^\|\s*(@图片\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$")
CHAR_DECL_RE = re.compile(r"画面人物明确为参考(@图片\d+)(?:的)?([^。；，]+)")
SCENE_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"参考@图片10(?:的)?龙辇入场长阶轴线"), "龙辇所在长阶"),
    (re.compile(r"参考@图片11(?:的)?镜门门位"), "镜门"),
    (re.compile(r"参考@图片12(?:的)?斩龙台主赛场"), "主赛场"),
    (re.compile(r"参考@图片12(?:的)?斩龙台"), "赛场"),
    (re.compile(r"参考@图片13(?:的)?斩龙台高台"), "裁决高位"),
    (re.compile(r"参考@图片13(?:的)?高台"), "裁决高位"),
    (re.compile(r"参考@图片14(?:的)?斩龙台外围山头"), "外围火力源"),
    (re.compile(r"参考@图片14(?:的)?外围山头"), "外围火力源"),
    (re.compile(r"参考@图片15(?:的)?斩龙台四门封锁线"), "封锁门线"),
    (re.compile(r"参考@图片15(?:的)?四门封锁线"), "封锁门线"),
]
REMOVE_PATTERNS = [
    re.compile(r"空间沿用[^。；]*[。；]?"),
    re.compile(r"保持同一场位[^。；]*[。；]?"),
    re.compile(r"确认同一场位[^。；]*[。；]?"),
    re.compile(r"确认还是同一空间[^。；]*[。；]?"),
    re.compile(r"结构与主光关系"),
    re.compile(r"同一场位和材质连续性"),
    re.compile(r"后景仍(?:能)?隐约挂住[^。；]*[。；]?"),
]
DROP_NOTE_PATTERNS = [
    re.compile(r"^自动修正："),
    re.compile(r"^严格保持长阶"),
    re.compile(r"^全集主光统一"),
    re.compile(r"^9:16竖屏优先"),
    re.compile(r"^所有关键动作都写出"),
    re.compile(r"^切镜均由"),
    re.compile(r"^全局执行要求：同一集内保持统一主光方向"),
]


def print_status(message: str) -> None:
    print(f"[seedance-simplify] {message}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="简化 Seedance 提示词，降低场景/人物硬约束。")
    parser.add_argument("--storyboard-json", default="", help="输入的 02-seedance-prompts__*.json")
    parser.add_argument("--storyboard-md", default="", help="输入的 02-seedance-prompts.md")
    parser.add_argument("--output-mode", choices=["sidecar", "overwrite"], default="sidecar")
    parser.add_argument("--engine", choices=["llm", "rules"], default="llm", help="简化引擎：默认走大模型。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="OpenAI runtime config 路径。")
    parser.add_argument("--non-interactive", action="store_true")
    return parser


def find_storyboard_markdown_path(episode_dir: Path) -> Path:
    exact = episode_dir / "02-seedance-prompts.md"
    if exact.exists():
        return exact
    candidates = sorted(
        [path for path in episode_dir.glob("02-seedance-prompts*.md") if path.is_file()],
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    return candidates[-1] if candidates else exact


def find_storyboard_json_path(episode_dir: Path) -> Path:
    exact = episode_dir / "02-seedance-prompts__openai__gpt-5.4.json"
    if exact.exists():
        return exact
    candidates = sorted(
        [path for path in episode_dir.glob("02-seedance-prompts__*.json") if path.is_file()],
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    return candidates[-1] if candidates else exact


def ensure_file(path_text: str, label: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} 不存在：{path}")
    return path


def choose_storyboard_paths_interactively() -> tuple[Path, Path]:
    series_dirs = list_series_dirs()
    usable_series: list[Path] = []
    for series_dir in series_dirs:
        if any(find_storyboard_json_path(path).exists() or find_storyboard_markdown_path(path).exists() for path in series_dir.iterdir() if path.is_dir()):
            usable_series.append(series_dir)
    if not usable_series:
        raise RuntimeError("outputs/ 下没有找到可用的 Seedance 分镜目录。")
    series_idx = choose_from_list("请选择要做提示词简化的剧：", [path.name for path in usable_series], default_index=0)
    series_dir = usable_series[series_idx]
    episode_dirs = sorted(
        [path for path in series_dir.iterdir() if path.is_dir() and (find_storyboard_json_path(path).exists() or find_storyboard_markdown_path(path).exists())],
        key=lambda item: episode_sort_key(item.name),
    )
    episode_idx = choose_from_list(f"请选择 {series_dir.name} 的集数：", [path.name for path in episode_dirs], default_index=0)
    episode_dir = episode_dirs[episode_idx]
    return ensure_file(str(find_storyboard_json_path(episode_dir)), "storyboard json"), ensure_file(
        str(find_storyboard_markdown_path(episode_dir)), "storyboard markdown"
    )


def resolve_storyboard_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    storyboard_json = args.storyboard_json.strip()
    storyboard_md = args.storyboard_md.strip()
    if storyboard_json or storyboard_md:
        json_path = ensure_file(storyboard_json or str(find_storyboard_json_path(Path(storyboard_md).expanduser().resolve().parent)), "storyboard json")
        md_path = ensure_file(storyboard_md or str(find_storyboard_markdown_path(json_path.parent)), "storyboard markdown")
        return json_path, md_path
    if args.non_interactive:
        raise FileNotFoundError("非交互模式下必须提供 --storyboard-json 或 --storyboard-md。")
    return choose_storyboard_paths_interactively()


def parse_asset_catalog(markdown_path: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for raw_line in markdown_path.read_text(encoding="utf-8").splitlines():
        match = ASSET_ROW_RE.match(raw_line.strip())
        if not match:
            continue
        ref_id, asset_type, display_name = match.groups()
        if ref_id == "引用编号":
            continue
        items.append({"ref_id": ref_id.strip(), "asset_type": asset_type.strip(), "display_name": display_name.strip()})
    return items


def infer_series_name(outputs_series_name: str) -> str:
    return SERIES_SUFFIX_RE.sub("", outputs_series_name) or outputs_series_name


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def cleanup_text(text: str) -> str:
    value = normalize_spaces(text)
    value = re.sub(r"全局执行要求：", "", value)
    value = re.sub(r"\s*([，。；：！？])\s*", r"\1", value)
    value = value.replace("赛场赛场", "赛场")
    value = value.replace("长阶长阶", "长阶")
    value = value.replace("镜门镜门", "镜门")
    value = value.replace("高位高位", "高位")
    value = value.replace("高台高台", "高台")
    value = re.sub(r"[；]{2,}", "；", value)
    value = re.sub(r"[。]{2,}", "。", value)
    value = re.sub(r"^[，。；：]+", "", value)
    value = re.sub(r"[，；：]+$", "", value)
    return value.strip()


def compress_text_middle(text: str, max_chars: int) -> str:
    content = cleanup_text(text)
    if len(content) <= max_chars:
        return content
    head = int(max_chars * 0.6)
    tail = max(24, max_chars - head - 18)
    return f"{content[:head].rstrip()}...[省略]...{content[-tail:].lstrip()}"


def compact_clauses(text: str, *, max_clauses: int, max_chars: int) -> str:
    clauses = [cleanup_text(chunk) for chunk in re.split(r"[。；]", str(text or "")) if cleanup_text(chunk)]
    if not clauses:
        return ""
    compact = "；".join(clauses[:max_clauses])
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip("，；： ")
    return compact


def simplify_character_declarations(text: str) -> str:
    found: list[tuple[str, str]] = []

    def repl(match: re.Match[str]) -> str:
        ref_id = match.group(1).strip()
        name = cleanup_text(match.group(2))
        if ref_id in SCENE_REF_IDS or not name:
            return ""
        pair = (ref_id, name)
        if pair not in found:
            found.append(pair)
        return ""

    simplified = CHAR_DECL_RE.sub(repl, str(text or ""))
    if not found:
        return cleanup_text(simplified)
    prefix = "人物：" + "、".join(f"参考{ref_id}{name}" for ref_id, name in found)
    body = cleanup_text(simplified)
    if not body:
        return prefix
    return f"{prefix}。{body}"


def simplify_text(text: str) -> str:
    simplified = str(text or "")
    for pattern, replacement in SCENE_REPLACEMENTS:
        simplified = pattern.sub(replacement, simplified)
    for pattern in REMOVE_PATTERNS:
        simplified = pattern.sub("", simplified)
    simplified = re.sub(r"参考@图片1[0-5](?:的)?", "", simplified)
    simplified = re.sub(r"@图片1[0-5](?!\d)", "", simplified)
    simplified = re.sub(r"参考(@图片[1-9])的", r"参考\1", simplified)
    simplified = simplify_character_declarations(simplified)
    return cleanup_text(simplified)


def filter_refs(refs: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in refs:
        ref_id = str(raw or "").strip()
        if not ref_id or ref_id in SCENE_REF_IDS or ref_id in seen:
            continue
        seen.add(ref_id)
        result.append(ref_id)
    return result


def simplify_dialogue_timeline(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in items:
        speaker = str(item.get("speaker") or "").strip()
        line = str(item.get("line") or "").strip()
        if not speaker or not line:
            continue
        result.append(
            {
                "speaker": speaker,
                "line": line,
                "start_second": item.get("start_second"),
                "end_second": item.get("end_second"),
                "delivery_note": compact_clauses(str(item.get("delivery_note") or ""), max_clauses=1, max_chars=24),
            }
        )
    return result


def simplify_master_timeline(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in entries:
        dialogue_blocks = simplify_dialogue_timeline(list(entry.get("dialogue_blocks") or []))
        simplified_entry = copy.deepcopy(dict(entry))
        simplified_entry["visual_beat"] = compact_clauses(simplify_text(str(entry.get("visual_beat") or "")), max_clauses=4, max_chars=260)
        simplified_entry["audio_cues"] = compact_clauses(simplify_text(str(entry.get("audio_cues") or "")), max_clauses=2, max_chars=120)
        simplified_entry["transition_hook"] = compact_clauses(simplify_text(str(entry.get("transition_hook") or "")), max_clauses=1, max_chars=90)
        simplified_entry["delivery_note"] = compact_clauses(str(entry.get("delivery_note") or ""), max_clauses=1, max_chars=24)
        simplified_entry["dialogue_blocks"] = dialogue_blocks
        result.append(simplified_entry)
    return result


def render_prompt_text(entries: Sequence[Mapping[str, Any]]) -> str:
    blocks: list[str] = []
    for entry in entries:
        start_second = entry.get("start_second")
        end_second = entry.get("end_second")
        visual = cleanup_text(str(entry.get("visual_beat") or ""))
        if start_second is None or end_second is None or not visual:
            continue
        parts = [f"{start_second:.1f}-{end_second:.1f}秒，{visual}"]
        dialogue_blocks = list(entry.get("dialogue_blocks") or [])
        if dialogue_blocks:
            dialogue_parts = []
            for item in dialogue_blocks:
                speaker = str(item.get("speaker") or "").strip()
                line = str(item.get("line") or "").strip()
                note = str(item.get("delivery_note") or "").strip()
                if not speaker or not line:
                    continue
                if note:
                    dialogue_parts.append(f"{speaker}（{note}）开口说“{line}”")
                else:
                    dialogue_parts.append(f"{speaker}开口说“{line}”")
            if dialogue_parts:
                parts.append("对白：" + "；".join(dialogue_parts))
        audio_cues = cleanup_text(str(entry.get("audio_cues") or ""))
        if audio_cues:
            parts.append("声音：" + audio_cues)
        transition_hook = cleanup_text(str(entry.get("transition_hook") or ""))
        if transition_hook:
            parts.append("收束：" + transition_hook)
        blocks.append("；".join(parts) + "。")
    return "\n".join(blocks).strip() + ("\n" if blocks else "")


def simplify_point(point: Mapping[str, Any]) -> dict[str, Any]:
    simplified = copy.deepcopy(dict(point))
    simplified["primary_refs"] = filter_refs(point.get("primary_refs") or [])
    simplified["secondary_refs"] = filter_refs(point.get("secondary_refs") or [])
    simplified["density_strategy"] = compact_clauses(simplify_text(str(point.get("density_strategy") or "")), max_clauses=2, max_chars=110)
    simplified["continuity_bridge"] = compact_clauses(simplify_text(str(point.get("continuity_bridge") or "")), max_clauses=2, max_chars=130)
    simplified["audio_design"] = compact_clauses(simplify_text(str(point.get("audio_design") or "")), max_clauses=2, max_chars=130)
    simplified["risk_notes"] = [
        note
        for note in [
            compact_clauses(simplify_text(str(item or "")), max_clauses=1, max_chars=80)
            for item in list(point.get("risk_notes") or [])
        ]
        if note
    ]
    simplified["master_timeline"] = simplify_master_timeline(list(point.get("master_timeline") or []))
    simplified["prompt_text"] = render_prompt_text(list(simplified.get("master_timeline") or []))
    simplified["shot_beat_plan"] = [
        f"{entry.get('start_second'):.1f}-{entry.get('end_second'):.1f}秒：{cleanup_text(str(entry.get('visual_beat') or ''))}"
        for entry in list(simplified.get("master_timeline") or [])
        if entry.get("start_second") is not None and entry.get("end_second") is not None and str(entry.get("visual_beat") or "").strip()
    ]
    simplified["dialogue_timeline"] = simplify_dialogue_timeline(list(point.get("dialogue_timeline") or []))
    return simplified


def build_materials_overview(asset_catalog: Sequence[Mapping[str, str]], used_refs: Sequence[str]) -> str:
    ref_map = {str(item.get("ref_id") or "").strip(): dict(item) for item in asset_catalog}
    person_bits = []
    for ref_id in used_refs:
        item = ref_map.get(ref_id)
        if not item:
            continue
        person_bits.append(f"{ref_id}{item.get('display_name', '').split('｜', 1)[0]}")
    people = "，".join(person_bits) if person_bits else "无"
    return f"人物参考：{people}。简化原则：保留剧情、动作、对白和人物指向；不强绑固定场景参考图，不过度锁死机位、主光和场位。"


def simplify_global_notes(notes: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in notes:
        note = cleanup_text(str(item or ""))
        if not note:
            continue
        if any(pattern.search(note) for pattern in DROP_NOTE_PATTERNS):
            continue
        note = simplify_text(note)
        if not note:
            continue
        if note in seen:
            continue
        seen.add(note)
        result.append(note)
    if not result:
        result = [
            "保留剧情顺序、动作链、对白和人物指向，场景空间不做硬绑定。",
            "默认竖屏叙事，运镜从简，转场由动作、视线和声响驱动。",
        ]
    return result


def collect_used_refs(prompt_entries: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for point in prompt_entries:
        for ref_id in list(point.get("primary_refs") or []) + list(point.get("secondary_refs") or []):
            ref_text = str(ref_id or "").strip()
            if not ref_text or ref_text in seen:
                continue
            seen.add(ref_text)
            result.append(ref_text)
    return result


def simplify_storyboard_data(data: Mapping[str, Any], asset_catalog: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    simplified = copy.deepcopy(dict(data))
    prompt_entries = [simplify_point(dict(item)) for item in list(data.get("prompt_entries") or []) if isinstance(item, Mapping)]
    used_refs = collect_used_refs(prompt_entries)
    simplified["prompt_entries"] = prompt_entries
    simplified["materials_overview"] = build_materials_overview(asset_catalog, used_refs)
    simplified["global_notes"] = simplify_global_notes(list(data.get("global_notes") or []))
    simplified["workflow_name"] = "提示词简化工作流"
    simplified["workflow_goal"] = "减少 Seedance 输入中的冗余场景/人物硬约束，保留剧情与人物引用正确性。"
    return simplified


def normalize_refs(raw_refs: Any) -> list[str]:
    refs: list[str] = []
    for raw in list(raw_refs or []):
        ref = str(raw or "").strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def compact_storyboard_for_llm(data: Mapping[str, Any], asset_catalog: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    catalog_by_ref = {str(item.get("ref_id") or "").strip(): dict(item) for item in asset_catalog}
    prompt_entries: list[dict[str, Any]] = []
    for item in list(data.get("prompt_entries") or []):
        if not isinstance(item, Mapping):
            continue
        primary_refs = normalize_refs(item.get("primary_refs"))
        secondary_refs = normalize_refs(item.get("secondary_refs"))
        visible_refs = primary_refs + [ref for ref in secondary_refs if ref not in primary_refs]
        prompt_entries.append(
            {
                "point_id": str(item.get("point_id") or "").strip(),
                "title": str(item.get("title") or "").strip(),
                "pace_label": str(item.get("pace_label") or "").strip(),
                "duration_hint": str(item.get("duration_hint") or "").strip(),
                "primary_refs": primary_refs,
                "secondary_refs": secondary_refs,
                "ref_labels": [
                    {
                        "ref_id": ref,
                        "asset_type": str(catalog_by_ref.get(ref, {}).get("asset_type") or "").strip(),
                        "display_name": str(catalog_by_ref.get(ref, {}).get("display_name") or "").strip(),
                    }
                    for ref in visible_refs
                ],
                "density_strategy": compress_text_middle(str(item.get("density_strategy") or ""), 160),
                "continuity_bridge": compress_text_middle(str(item.get("continuity_bridge") or ""), 180),
                "audio_design": compress_text_middle(str(item.get("audio_design") or ""), 160),
                "risk_notes": [compress_text_middle(str(x), 80) for x in list(item.get("risk_notes") or []) if str(x).strip()][:4],
                "master_timeline": [
                    {
                        "start_second": entry.get("start_second"),
                        "end_second": entry.get("end_second"),
                        "visual_beat": compress_text_middle(str(entry.get("visual_beat") or ""), 220),
                        "dialogue_blocks": [
                            {
                                "speaker": str(block.get("speaker") or "").strip(),
                                "line": compress_text_middle(str(block.get("line") or ""), 60),
                            }
                            for block in list(entry.get("dialogue_blocks") or [])
                            if isinstance(block, Mapping) and str(block.get("line") or "").strip()
                        ][:3],
                        "audio_cues": compress_text_middle(str(entry.get("audio_cues") or ""), 100),
                        "transition_hook": compress_text_middle(str(entry.get("transition_hook") or ""), 100),
                    }
                    for entry in list(item.get("master_timeline") or [])
                    if isinstance(entry, Mapping)
                ][:6],
                "prompt_text": compress_text_middle(str(item.get("prompt_text") or ""), 1200),
            }
        )
    return {
        "episode_id": str(data.get("episode_id") or "").strip(),
        "episode_title": str(data.get("episode_title") or "").strip(),
        "materials_overview": compress_text_middle(str(data.get("materials_overview") or ""), 1200),
        "global_notes": [compress_text_middle(str(x), 140) for x in list(data.get("global_notes") or []) if str(x).strip()][:12],
        "asset_catalog": list(asset_catalog),
        "prompt_entries": prompt_entries,
    }


def build_llm_simplify_prompts(compact_package: Mapping[str, Any]) -> tuple[str, str]:
    system_prompt = (
        "你是一名 Seedance 提示词简化导演。你的任务不是润色，而是系统性减负。"
        "你必须在不改变剧情顺序、人物关系、关键动作因果和对白归属的前提下，重写成更利于 Seedance 发挥的简化版提示词。"
        "请严格按 schema 输出完整 JSON。"
    )
    user_prompt = (
        "请把这份 Seedance 分镜提示词重写为“提示词简化工作流”版本。\n\n"
        "目标：\n"
        "1. 明显减少约束密度，重点保留场景、动作、剧情推进。\n"
        "2. 人物引用关系不能错，但人物引用次数可以减少，只在开场、换人、关键动作或易混淆处保留。\n"
        "3. 尽量减少固定机位、固定主光、固定场位、连续性口号、过细声音修饰、重复心理修饰。\n"
        "4. 场景参考不要被写成硬绑定说明；允许保留必要空间锚点，但不要反复要求“同一场位/同一主光/同一空间”。\n"
        "5. `prompt_text` 必须比原稿短很多，读起来更直接，更像给 Seedance 的执行稿。\n"
        "6. `master_timeline` 每拍以“谁在做什么、发生什么、剧情如何推进”为主，不要堆砌修辞。\n"
        "7. `audio_design`、`density_strategy`、`continuity_bridge` 都要压短，只保留真正有用的信息。\n"
        "8. 保持 `point_id` 数量与顺序不变。\n\n"
        "硬性要求：\n"
        "- 可以减少人物 ref 的出现次数，但不能把角色认错、串错、漏掉关键出镜者。\n"
        "- 可以减少场景 ref，但不能让剧情空间失真。\n"
        "- 不要新增原稿没有的人物或剧情。\n"
        "- `global_notes` 只保留 2-4 条真正必要的执行原则。\n\n"
        "下面是待简化输入 JSON：\n"
        f"{json.dumps(compact_package, ensure_ascii=False, indent=2)}"
    )
    return system_prompt, user_prompt


def simplify_storyboard_data_with_llm(
    *,
    data: Mapping[str, Any],
    asset_catalog: Sequence[Mapping[str, str]],
    config_path: Path,
) -> dict[str, Any]:
    config = load_runtime_config(config_path)
    model, api_key = configure_openai_api(config, provider_key="openai")
    compact_package = compact_storyboard_for_llm(data, asset_catalog)
    system_prompt, user_prompt = build_llm_simplify_prompts(compact_package)
    print_status(f"开始调用大模型做系统性简化：model={model}")
    simplified = openai_json_completion(
        model=model,
        api_key=api_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="seedance_prompt_package_simplified",
        schema=SEEDANCE_PROMPTS_SCHEMA,
        temperature=0.35,
        timeout_seconds=1800,
        stage="seedance_simplify",
        step_name="seedance_prompt_simplify_model_call",
        metadata={
            "episode_id": str(data.get("episode_id") or ""),
            "prompt_entry_count": len(list(data.get("prompt_entries") or [])),
        },
    )
    normalized = normalize_storyboard_result(
        simplified,
        frame_orientation="9:16竖屏",
        storyboard_profile="normal",
        asset_catalog=list(asset_catalog),
    )
    normalized = repair_storyboard_density(normalized)
    normalized["workflow_name"] = "提示词简化工作流"
    normalized["workflow_goal"] = "通过大模型系统性减负 Seedance 提示词，突出场景、动作、剧情，减少人物与场景硬约束。"
    normalized["materials_overview"] = build_materials_overview(asset_catalog, collect_used_refs(list(normalized.get("prompt_entries") or [])))
    normalized["global_notes"] = [
        "保留剧情顺序、关键动作、对白归属与必要人物锚点。",
        "场景只保留必要空间信息，不强绑固定机位、主光和场位。",
        "人物引用减到必要次数，重点保证不串人、不丢关键出镜。",
        "提示词以场景、动作、剧情推进为主，避免修辞堆叠。",
    ]
    return normalized


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def backup_file(path: Path) -> Path:
    backup_path = path.with_name(path.name + ".bak.simplify")
    shutil.copy2(path, backup_path)
    return backup_path


def count_scene_refs(text: str) -> int:
    return len(re.findall(r"@图片1[0-5](?!\d)", str(text or "")))


def render_report(
    *,
    original_data: Mapping[str, Any],
    simplified_data: Mapping[str, Any],
    input_json_path: Path,
    output_json_path: Path,
    output_md_path: Path,
    output_mode: str,
) -> str:
    original_json = json.dumps(original_data, ensure_ascii=False, indent=2)
    simplified_json = json.dumps(simplified_data, ensure_ascii=False, indent=2)
    original_prompt_chars = sum(len(str(item.get("prompt_text") or "")) for item in list(original_data.get("prompt_entries") or []))
    simplified_prompt_chars = sum(len(str(item.get("prompt_text") or "")) for item in list(simplified_data.get("prompt_entries") or []))
    lines = [
        "# 提示词简化工作流报告",
        "",
        f"- 输入 JSON：{input_json_path}",
        f"- 输出 JSON：{output_json_path}",
        f"- 输出 MD：{output_md_path}",
        f"- 输出模式：{output_mode}",
        f"- prompt_entries：{len(list(simplified_data.get('prompt_entries') or []))}",
        f"- 原始 JSON 字符数：{len(original_json)}",
        f"- 简化后 JSON 字符数：{len(simplified_json)}",
        f"- 原始 prompt_text 总字符数：{original_prompt_chars}",
        f"- 简化后 prompt_text 总字符数：{simplified_prompt_chars}",
        f"- 原始场景引用次数：{count_scene_refs(original_json)}",
        f"- 简化后场景引用次数：{count_scene_refs(simplified_json)}",
        "",
        "## 简化原则",
        "",
        "- 保留剧情顺序、对白、时间轴、人物引用。",
        "- 去掉 `@图片10-15` 这类场景参考图绑定和固定场位锁定。",
        "- 压缩重复的 `画面人物明确为...`、空间连续性、主光关系等冗余约束。",
        "- 重新生成更短的 `prompt_text`，让 Seedance 获得更大表现自由度。",
        "",
    ]
    return "\n".join(lines)


def output_paths(input_json_path: Path, output_mode: str) -> tuple[Path, Path, Path]:
    if output_mode == "overwrite":
        md_path = input_json_path.parent / "02-seedance-prompts.md"
        return input_json_path, md_path, input_json_path.parent / "02-seedance-prompts.simplified.report.md"
    json_path = input_json_path.parent / "02-seedance-prompts.simplified.json"
    md_path = input_json_path.parent / "02-seedance-prompts.simplified.md"
    report_path = input_json_path.parent / "02-seedance-prompts.simplified.report.md"
    return json_path, md_path, report_path


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    storyboard_json_path, storyboard_md_path = resolve_storyboard_paths(args)
    asset_catalog = parse_asset_catalog(storyboard_md_path)
    original_data = json.loads(storyboard_json_path.read_text(encoding="utf-8"))
    if args.engine == "llm":
        simplified_data = simplify_storyboard_data_with_llm(
            data=original_data,
            asset_catalog=asset_catalog,
            config_path=Path(args.config).expanduser().resolve(),
        )
    else:
        print_status("当前使用规则版简化引擎。")
        simplified_data = simplify_storyboard_data(original_data, asset_catalog)
    outputs_series_name = storyboard_json_path.parent.parent.name
    series_name = infer_series_name(outputs_series_name)
    rendered_md = render_seedance_markdown(series_name=series_name, data=simplified_data, asset_catalog=list(asset_catalog))
    output_json_path, output_md_path, report_path = output_paths(storyboard_json_path, args.output_mode)

    if args.output_mode == "overwrite":
        backup_file(storyboard_json_path)
        backup_file(storyboard_md_path)

    write_json(output_json_path, simplified_data)
    output_md_path.write_text(rendered_md, encoding="utf-8")
    report_path.write_text(
        render_report(
            original_data=original_data,
            simplified_data=simplified_data,
            input_json_path=storyboard_json_path,
            output_json_path=output_json_path,
            output_md_path=output_md_path,
            output_mode=args.output_mode,
        ),
        encoding="utf-8",
    )

    print_status(f"已生成简化版 JSON：{output_json_path}")
    print_status(f"已生成简化版 Markdown：{output_md_path}")
    print_status(f"已生成简化报告：{report_path}")


if __name__ == "__main__":
    main()
