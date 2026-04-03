from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from genre_routing import load_genre_playbook_library, suggest_library_genres_for_series
from run_series_pipeline import extract_episode_number
from series_paths import infer_episode_id_from_name
from generate_nano_banana_assets import (
    extract_scene_grid_layout,
    find_storyboard_prompt_path,
    parse_scene_prompts as parse_nano_scene_prompts,
)


DEFAULT_SCRIPT_SUFFIXES = [
    "__explosive.md",
    "__openai__gpt-5.4.md",
    "__qwen__qwen3-vl-plus.md",
    "__gemini__gemini-3-pro-preview.md",
]
BASE_SCRIPT_SUFFIXES = [
    "__openai__gpt-5.4.md",
    "__qwen__qwen3-vl-plus.md",
    "__gemini__gemini-3-pro-preview.md",
]
EXPLOSIVE_ONLY_SUFFIXES = [
    "__explosive.md",
]
OPENAI_FLOW_SCRIPT_SOURCE_OPTIONS = [
    ("爆改版优先", "explosive_first"),
    ("原始基础版优先", "base_first"),
    ("仅爆改版", "explosive_only"),
]
OPENAI_FLOW_MEDIUM_STYLE_OPTIONS = [
    ("漫剧", "漫剧", "高识别度漫剧视觉，默认按9:16竖屏移动端构图，角色统一、情绪直给、镜头表达清晰，适合 Seedance 2.0 动态镜头开发"),
    ("电影", "电影", "电影级写实质感，默认按9:16竖屏电影感构图，光影层次丰富，强调中轴组织、纵深调度与情绪沉浸"),
    ("短剧", "短剧", "短剧高钩子风格，默认按9:16竖屏移动端构图，冲突更密、节奏更快、情绪爆点更直接，适合强留存与强卡点表达"),
    ("电视剧", "电视剧", "电视剧叙事风格，默认按9:16竖屏构图，人物关系推进更稳定，场景与表演更生活化，兼顾连续性和情绪铺垫"),
]
STORYBOARD_PROFILE_OPTIONS = [
    ("Normal 细节优先", "normal"),
    ("极速版 速度优先", "fast"),
]
GENRE_BUNDLE_REUSE_OPTIONS = [
    ("复用已有 genre bundle", "reuse"),
    ("强制重建 genre bundle", "rebuild"),
]
DEFAULT_SYNC_CONFIG = PROJECT_ROOT / "config" / "sync_series_learning.local.json"
SYNC_RUNNER = PROJECT_ROOT / "scripts" / "sync_series_learning_to_genres.py"
GENRE_AUDIT_ROOT = PROJECT_ROOT / "skills" / "production" / "video-script-reconstruction-skill" / "genres"
GENRE_AUDIT_DRAFT_ROOT = GENRE_AUDIT_ROOT / "__drafts__"


def print_status(tag: str, message: str) -> None:
    print(f"[{tag}] {message}", flush=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def slugify_label(value: str) -> str:
    clean = re.sub(r"\s+", "_", str(value).strip())
    clean = re.sub(r"[\\\\/:*?\"<>|（）()\[\]{}，,。；;！!？?]+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean[:48] or "genre"


def write_temp_config(prefix: str, data: dict[str, Any]) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=prefix,
        encoding="utf-8",
        delete=False,
        dir="/tmp",
    ) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        return Path(handle.name)


def python_bin() -> Path:
    preferred = PROJECT_ROOT / ".venv" / "bin" / "python"
    return preferred if preferred.exists() else Path(sys.executable).resolve()


def prompt_choice(prompt: str, options: list[tuple[str, Path]], default_index: int = 0) -> tuple[str, Path]:
    if not options:
        raise RuntimeError("没有可选项。")
    print(prompt)
    for index, (label, _) in enumerate(options, start=1):
        default_text = "  [默认]" if index - 1 == default_index else ""
        print(f"  {index}. {label}{default_text}")
    while True:
        raw = input(f"请输入序号（默认 {default_index + 1}）：").strip()
        if not raw:
            return options[default_index]
        if raw.isdigit():
            value = int(raw)
            if 1 <= value <= len(options):
                return options[value - 1]
        print("输入无效，请重新输入。")


def prompt_episode_range(default_start: int, default_end: int) -> tuple[int, int]:
    while True:
        start_raw = input(f"起始集数（默认 {default_start}）：").strip()
        end_raw = input(f"结束集数（默认 {default_end}）：").strip()
        start = int(start_raw) if start_raw else default_start
        end = int(end_raw) if end_raw else default_end
        if start <= 0 or end <= 0:
            print("集数必须大于 0，请重新输入。")
            continue
        if start > end:
            print("起始集数不能大于结束集数，请重新输入。")
            continue
        return start, end


def prompt_named_choice(prompt: str, options: list[tuple[str, str]], default_index: int = 0) -> tuple[str, str]:
    if not options:
        raise RuntimeError("没有可选项。")
    print(prompt)
    for index, (label, _) in enumerate(options, start=1):
        default_text = "  [默认]" if index - 1 == default_index else ""
        print(f"  {index}. {label}{default_text}")
    while True:
        raw = input(f"请输入序号（默认 {default_index + 1}）：").strip()
        if not raw:
            return options[default_index]
        if raw.isdigit():
            value = int(raw)
            if 1 <= value <= len(options):
                return options[value - 1]
        print("输入无效，请重新输入。")


def resolve_openai_flow_suffixes(selection_key: str) -> list[str]:
    if selection_key == "base_first":
        return BASE_SCRIPT_SUFFIXES + DEFAULT_SCRIPT_SUFFIXES
    if selection_key == "explosive_only":
        return EXPLOSIVE_ONLY_SUFFIXES
    return DEFAULT_SCRIPT_SUFFIXES


def prompt_openai_flow_style_selection(
    *,
    default_target_medium: str,
    default_visual_style: str,
) -> dict[str, str]:
    default_index = next(
        (
            index
            for index, (_, medium, _) in enumerate(OPENAI_FLOW_MEDIUM_STYLE_OPTIONS)
            if medium == default_target_medium
        ),
        0,
    )
    label, medium_key = prompt_named_choice(
        "请选择要生成的影视风格：",
        [(label, medium) for label, medium, _ in OPENAI_FLOW_MEDIUM_STYLE_OPTIONS],
        default_index=default_index,
    )
    preset = next(
        (
            {"label": option_label, "target_medium": option_medium, "visual_style": option_style}
            for option_label, option_medium, option_style in OPENAI_FLOW_MEDIUM_STYLE_OPTIONS
            if option_medium == medium_key
        ),
        {
            "label": default_target_medium or "漫剧",
            "target_medium": default_target_medium or "漫剧",
            "visual_style": default_visual_style or "按当前项目统一",
        },
    )
    custom_style = input("可选：输入更具体的视觉补充说明（直接回车跳过）：").strip()
    if custom_style:
        preset["visual_style"] = f"{preset['visual_style']}；补充要求：{custom_style}"
    return preset


def prompt_bool(prompt: str, default_value: bool) -> bool:
    default_text = "true" if default_value else "false"
    while True:
        raw = input(f"{prompt}（true/false，默认 {default_text}）：").strip().lower()
        if not raw:
            return default_value
        if raw in {"true", "1", "yes", "y", "on"}:
            return True
        if raw in {"false", "0", "no", "n", "off"}:
            return False
        print("输入无效，请输入 true 或 false。")


def prompt_zero_one(prompt: str, default_value: int = 0) -> int:
    default_text = "1" if default_value == 1 else "0"
    while True:
        raw = input(f"{prompt}（1/0，默认 {default_text}）：").strip()
        if not raw:
            return default_value
        if raw in {"1", "0"}:
            return int(raw)
        print("输入无效，请输入 1 或 0。")


def parse_layout_text(raw: str) -> tuple[int, int] | None:
    normalized = raw.strip().lower().replace('*', 'x').replace('×', 'x').replace(',', ' ')
    match = re.fullmatch(r'(\d+)x(\d+)', normalized)
    if match:
        rows, cols = int(match.group(1)), int(match.group(2))
        return (rows, cols) if rows > 0 and cols > 0 else None
    space_match = re.fullmatch(r'(\d+)\s+(\d+)', normalized)
    if space_match:
        rows, cols = int(space_match.group(1)), int(space_match.group(2))
        return (rows, cols) if rows > 0 and cols > 0 else None
    return None


def format_layout(layout: tuple[int, int] | None) -> str:
    if not layout:
        return '未识别'
    return f'{layout[0]}x{layout[1]}'


def prompt_scene_layout_overrides(*, episode_id: str, scene_prompt_path: Path) -> dict[str, str]:
    if not scene_prompt_path.exists():
        return {}
    try:
        items = parse_nano_scene_prompts(scene_prompt_path)
    except Exception:
        return {}
    scene_items = [item for item in items if item.prompt_type == 'scene']
    if not scene_items:
        return {}

    overrides: dict[str, str] = {}
    print_status('nano-banana', f'{episode_id} 检测到以下场景宫格布局，切分前可人工确认：')
    for item in scene_items:
        detected = extract_scene_grid_layout(item.prompt)
        current_text = format_layout(detected)
        while True:
            raw = input(f'  - {item.source_label}（当前 {current_text}，直接回车保留，或输入 2 3 / 2x3 / 3x3 / 1x3）：').strip()
            if not raw:
                break
            parsed = parse_layout_text(raw)
            if parsed is None:
                print('输入无效，请输入类似 2 3、2x3、3*3、1x3 的格式。')
                continue
            overrides[item.source_label] = f'{parsed[0]}x{parsed[1]}'
            break
    return overrides


def prompt_genre_selection(
    *,
    tag: str = "series-pipeline",
    series_label: str,
    video_dir: Path,
    default_library_keys: list[str] | None = None,
    default_custom_tokens: list[str] | None = None,
) -> dict[str, Any]:
    library = load_genre_playbook_library()
    suggestions = suggest_library_genres_for_series(series_label=series_label, video_dir=video_dir, limit=3)
    suggestion_keys = [str(item.get("genre_key", "")).strip() for item in suggestions if str(item.get("genre_key", "")).strip()]

    print(f"[{tag}] AI 预判题材（最多 3 类）：")
    if suggestion_keys:
        for index, key in enumerate(suggestion_keys, start=1):
            print(f"  - {index}. {key}")
    else:
        print("  - 暂未命中现有题材库，将以基础方法论分析。")

    print(f"[{tag}] 现有题材库：")
    for index, item in enumerate(library, start=1):
        print(f"  {index}. {item.get('genre_key', '')}")

    default_library_keys = [item for item in (default_library_keys or []) if item]
    default_custom_tokens = [item for item in (default_custom_tokens or []) if item]
    default_text = " ".join(default_library_keys) if default_library_keys else ("0 " + " ".join(default_custom_tokens)).strip() if default_custom_tokens else ""

    while True:
        prompt = "请输入题材序号（最多 3 个，用空格隔开；输入 0 表示手动输入题材）"
        raw = input(f"{prompt}{f'，默认 {default_text}' if default_text else ''}：").strip()
        if not raw and suggestion_keys:
            return {
                "library_keys": suggestion_keys[:3],
                "custom_tokens": [],
                "ai_suggested_keys": suggestion_keys[:3],
                "source": "interactive_default_ai_suggestions",
            }
        if not raw and default_library_keys:
            return {
                "library_keys": default_library_keys[:3],
                "custom_tokens": default_custom_tokens[:3],
                "ai_suggested_keys": suggestion_keys[:3],
                "source": "interactive_default_config",
            }
        if not raw and default_custom_tokens:
            return {
                "library_keys": [],
                "custom_tokens": default_custom_tokens[:3],
                "ai_suggested_keys": suggestion_keys[:3],
                "source": "interactive_default_custom",
            }

        parts = [item.strip() for item in raw.split() if item.strip()]
        if not parts:
            print("输入不能为空，请重新输入。")
            continue
        if parts[0] == "0":
            manual_raw = input("请输入自定义题材（空格隔开，例如：重生 女帝 爱情）：").strip()
            manual_tokens = [item.strip() for item in manual_raw.split() if item.strip()]
            if not manual_tokens:
                print("自定义题材不能为空，请重新输入。")
                continue
            return {
                "library_keys": [],
                "custom_tokens": manual_tokens[:3],
                "ai_suggested_keys": suggestion_keys[:3],
                "source": "interactive_manual_custom",
            }
        if len(parts) > 3:
            print("最多选择 3 个题材，请重新输入。")
            continue
        if not all(item.isdigit() for item in parts):
            print("请输入题材序号，或输入 0 后再手动输入题材。")
            continue
        indices = [int(item) for item in parts]
        if any(index < 1 or index > len(library) for index in indices):
            print("题材序号超出范围，请重新输入。")
            continue
        selected_keys = []
        for index in indices:
            key = str(library[index - 1].get("genre_key", "")).strip()
            if key and key not in selected_keys:
                selected_keys.append(key)
        if not selected_keys:
            print("没有选到有效题材，请重新输入。")
            continue
        return {
            "library_keys": selected_keys[:3],
            "custom_tokens": [],
            "ai_suggested_keys": suggestion_keys[:3],
            "source": "interactive_library_selection",
        }


def prompt_explosive_style_selection(
    *,
    series_label: str,
    script_dir: Path,
    default_library_keys: list[str] | None = None,
    default_custom_tokens: list[str] | None = None,
) -> dict[str, Any]:
    library = load_genre_playbook_library()
    suggestions = suggest_library_genres_for_series(series_label=series_label, video_dir=script_dir, limit=3)
    suggestion_keys = [str(item.get("genre_key", "")).strip() for item in suggestions if str(item.get("genre_key", "")).strip()]

    print("[explosive-rewrite] AI 参考的题材方向（最多 3 类，可直接用作目标风格或仅作参考）：")
    if suggestion_keys:
        for index, key in enumerate(suggestion_keys, start=1):
            print(f"  - {index}. {key}")
    else:
        print("  - 暂未命中现有题材库，将按通用爆款方法论处理。")

    print("[explosive-rewrite] 现有题材库：")
    for index, item in enumerate(library, start=1):
        print(f"  {index}. {item.get('genre_key', '')}")

    default_library_keys = [item for item in (default_library_keys or []) if item]
    default_custom_tokens = [item for item in (default_custom_tokens or []) if item]
    default_text = " ".join(default_library_keys) if default_library_keys else ("0 " + " ".join(default_custom_tokens)).strip() if default_custom_tokens else ""

    while True:
        prompt = "请输入希望改向的爆款风格题材序号（最多 3 个，用空格隔开；输入 0 表示手动输入题材）"
        raw = input(f"{prompt}{f'，默认 {default_text}' if default_text else ''}：").strip()
        if not raw and default_library_keys:
            return {
                "genre_keys": default_library_keys[:3],
                "custom_style_tokens": default_custom_tokens[:3],
                "style_label": " / ".join(default_library_keys[:3] or default_custom_tokens[:3]),
                "source": "interactive_default_config",
                "ai_suggested_keys": suggestion_keys[:3],
            }
        if not raw and default_custom_tokens:
            return {
                "genre_keys": [],
                "custom_style_tokens": default_custom_tokens[:3],
                "style_label": " / ".join(default_custom_tokens[:3]),
                "source": "interactive_default_custom",
                "ai_suggested_keys": suggestion_keys[:3],
            }
        if not raw and suggestion_keys:
            return {
                "genre_keys": suggestion_keys[:3],
                "custom_style_tokens": [],
                "style_label": " / ".join(suggestion_keys[:3]),
                "source": "interactive_default_ai_suggestions",
                "ai_suggested_keys": suggestion_keys[:3],
            }

        parts = [item.strip() for item in raw.split() if item.strip()]
        if not parts:
            print("输入不能为空，请重新输入。")
            continue
        if parts[0] == "0":
            manual_raw = input("请输入目标风格题材（空格隔开，例如：萌宝 爱情 女帝）：").strip()
            manual_tokens = [item.strip() for item in manual_raw.split() if item.strip()]
            if not manual_tokens:
                print("自定义题材不能为空，请重新输入。")
                continue
            return {
                "genre_keys": [],
                "custom_style_tokens": manual_tokens[:3],
                "style_label": " / ".join(manual_tokens[:3]),
                "source": "interactive_manual_custom",
                "ai_suggested_keys": suggestion_keys[:3],
            }
        if len(parts) > 3:
            print("最多选择 3 个题材，请重新输入。")
            continue
        if not all(item.isdigit() for item in parts):
            print("请输入题材序号，或输入 0 后再手动输入题材。")
            continue
        indices = [int(item) for item in parts]
        if any(index < 1 or index > len(library) for index in indices):
            print("题材序号超出范围，请重新输入。")
            continue
        selected_keys: list[str] = []
        for index in indices:
            key = str(library[index - 1].get("genre_key", "")).strip()
            if key and key not in selected_keys:
                selected_keys.append(key)
        if not selected_keys:
            print("没有选到有效题材，请重新输入。")
            continue
        return {
            "genre_keys": selected_keys[:3],
            "custom_style_tokens": [],
            "style_label": " / ".join(selected_keys[:3]),
            "source": "interactive_library_selection",
            "ai_suggested_keys": suggestion_keys[:3],
        }


def canonical_series_base_name(name: str) -> str:
    normalized = str(name or "").strip()
    removable_patterns = [
        r"-(gpt|claude)(?:[-_][A-Za-z0-9]+)*$",
        r"-(real|old)$",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in removable_patterns:
            updated = re.sub(pattern, "", normalized)
            if updated != normalized:
                normalized = updated
                changed = True
                break
    return normalized


def list_script_series_dirs() -> list[tuple[str, Path]]:
    script_root = PROJECT_ROOT / "script"
    if not script_root.exists():
        return []
    result: list[tuple[str, Path]] = []
    for path in sorted(script_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        if any(path.glob("*.md")):
            result.append((path.name, path.resolve()))
    return result


def choose_preferred_series_dir(existing: Path | None, candidate: Path) -> Path:
    if existing is None:
        return candidate
    existing_is_gpt = existing.name.endswith("-gpt")
    candidate_is_gpt = candidate.name.endswith("-gpt")
    if candidate_is_gpt and not existing_is_gpt:
        return candidate
    return existing


def choose_preferred_script_series_dir(existing: Path | None, candidate: Path) -> Path:
    if existing is None:
        return candidate
    existing_is_gpt = existing.name.endswith("-gpt")
    candidate_is_gpt = candidate.name.endswith("-gpt")
    if existing_is_gpt and not candidate_is_gpt:
        return candidate
    return existing


def list_ready_nano_banana_series() -> list[tuple[str, Path, Path, Path | None]]:
    script_dirs_exact = {label: script_dir for label, script_dir in list_script_series_dirs()}
    script_dirs_by_base: dict[str, Path] = {}
    for label, script_dir in list_script_series_dirs():
        base_name = canonical_series_base_name(label)
        script_dirs_by_base[base_name] = choose_preferred_script_series_dir(script_dirs_by_base.get(base_name), script_dir)

    assets_root = PROJECT_ROOT / "assets"
    asset_dirs: dict[str, Path] = {}
    if assets_root.exists():
        for candidate in sorted(assets_root.iterdir(), key=lambda item: item.name):
            if not candidate.is_dir():
                continue
            has_top_level_prompts = (candidate / "character-prompts.md").exists() and (candidate / "scene-prompts.md").exists()
            has_episode_level_prompts = any(
                (episode_dir / "character-prompts.md").exists() and (episode_dir / "scene-prompts.md").exists()
                for episode_dir in candidate.glob("ep*")
            )
            if not has_top_level_prompts and not has_episode_level_prompts:
                continue
            asset_dirs[candidate.name] = candidate.resolve()

    outputs_root = PROJECT_ROOT / "outputs"
    output_dirs: dict[str, Path] = {}
    if outputs_root.exists():
        for candidate in sorted(outputs_root.iterdir(), key=lambda item: item.name):
            if not candidate.is_dir():
                continue
            output_dirs[candidate.name] = candidate.resolve()

    ready_names = sorted(set(asset_dirs))
    result: list[tuple[str, Path, Path, Path | None]] = []
    for series_dir_name in ready_names:
        script_dir = script_dirs_exact.get(series_dir_name)
        if script_dir is None:
            script_dir = script_dirs_by_base.get(canonical_series_base_name(series_dir_name))
        if script_dir is None:
            continue
        asset_dir = asset_dirs[series_dir_name]
        output_dir = output_dirs.get(series_dir_name)
        result.append((series_dir_name, script_dir, asset_dir, output_dir))
    return result


def detect_output_episode_numbers(series_output_dir: Path | None) -> list[int]:
    if series_output_dir is None:
        return []
    values: list[int] = []
    for prompt_path in sorted(series_output_dir.glob("ep*/02-seedance-prompts.md")):
        episode_id = infer_episode_id_from_name(prompt_path.parent.name)
        if not episode_id:
            continue
        try:
            values.append(int(episode_id[2:]))
        except ValueError:
            continue
    return sorted(set(values))


def list_video_series_dirs(file_extensions: list[str]) -> list[tuple[str, Path]]:
    videos_root = PROJECT_ROOT / "videos"
    if not videos_root.exists():
        return []
    matched_dirs: set[Path] = set()
    extension_set = {item.lower() for item in file_extensions}
    for path in videos_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in extension_set:
            matched_dirs.add(path.parent.resolve())
    result: list[tuple[str, Path]] = []
    for path in sorted(matched_dirs, key=lambda item: str(item.relative_to(videos_root))):
        result.append((str(path.relative_to(videos_root)), path))
    return result


def list_analysis_series_dirs(analysis_root: Path) -> list[tuple[str, Path]]:
    if not analysis_root.exists():
        return []
    result: list[tuple[str, Path]] = []
    for path in sorted(analysis_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        if path.name in {"videos", "batch_runs", "openai_agent_flow"}:
            continue
        if (path / "series_strength_playbook_draft.json").exists():
            result.append((path.name, path.resolve()))
    return result


def list_official_genre_dirs() -> list[tuple[str, Path]]:
    if not GENRE_AUDIT_ROOT.exists():
        return []
    result: list[tuple[str, Path]] = []
    for path in sorted(GENRE_AUDIT_ROOT.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.name.startswith("__"):
            continue
        if (path / "playbook.json").exists() and (path / "skill.md").exists():
            result.append((path.name, path.resolve()))
    return result


def list_genre_audit_draft_dirs(genre_key: str) -> list[tuple[str, Path]]:
    if not GENRE_AUDIT_DRAFT_ROOT.exists():
        return []
    prefix = f"audit__{slugify_label(genre_key)}__"
    result: list[tuple[str, Path]] = []
    for path in sorted(GENRE_AUDIT_DRAFT_ROOT.iterdir(), key=lambda item: item.name, reverse=True):
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        judgement_path = path / "judgement.json"
        judgement = load_json(judgement_path) if judgement_path.exists() else {}
        accepted_playbook = len(list(judgement.get("accepted_playbook_suggestions") or []))
        accepted_skill = len(list(judgement.get("accepted_skill_suggestions") or []))
        summary = str(judgement.get("refine_summary") or "").strip()
        label = f"{path.name} | playbook {accepted_playbook} | skill {accepted_skill}"
        if summary:
            label += f" | {summary[:28]}{'…' if len(summary) > 28 else ''}"
        result.append((label, path.resolve()))
    return result


def detect_script_episode_numbers(series_dir: Path) -> list[int]:
    values: list[int] = []
    for path in sorted(series_dir.glob("*.md")):
        episode_id = infer_episode_id_from_name(path.name)
        if not episode_id:
            continue
        try:
            values.append(int(episode_id[2:]))
        except ValueError:
            continue
    return sorted(set(values))


def list_nano_banana_script_series_dirs(series_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add_candidate(path: Path | None) -> None:
        if path is None:
            return
        resolved = path.resolve()
        if resolved in seen or not resolved.is_dir():
            return
        if not any(resolved.glob("*.md")):
            return
        seen.add(resolved)
        candidates.append(resolved)

    add_candidate(series_dir)
    base_name = canonical_series_base_name(series_dir.name)
    if not base_name:
        return candidates

    script_root = PROJECT_ROOT / "script"
    add_candidate(script_root / base_name)
    for _, candidate in list_script_series_dirs():
        if canonical_series_base_name(candidate.name) != base_name:
            continue
        add_candidate(candidate)
    return candidates


def detect_nano_banana_script_episode_numbers(series_dir: Path) -> list[int]:
    values: list[int] = []
    for candidate_dir in list_nano_banana_script_series_dirs(series_dir):
        values.extend(detect_script_episode_numbers(candidate_dir))
    return sorted(set(values))


def detect_video_episode_numbers(series_dir: Path, file_extensions: list[str], custom_regex: str | None) -> list[int]:
    values: list[int] = []
    extension_set = {item.lower() for item in file_extensions}
    for path in sorted(series_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extension_set:
            continue
        episode_number = extract_episode_number(path.stem, custom_regex)
        if episode_number is not None:
            values.append(episode_number)
    return sorted(set(values))


def select_script_for_episode(series_dir: Path, episode_id: str, preferred_suffixes: list[str]) -> Path:
    candidates = sorted(path.resolve() for path in series_dir.glob(f"{episode_id}*.md"))
    if not candidates:
        candidates = sorted(
            path.resolve() for path in series_dir.glob("*.md") if episode_id.lower() in path.name.lower()
        )
    if not candidates:
        raise FileNotFoundError(f"未找到 {episode_id} 对应剧本：{series_dir}")

    suffixes = preferred_suffixes or DEFAULT_SCRIPT_SUFFIXES
    for suffix in suffixes:
        for candidate in candidates:
            if candidate.name.endswith(suffix):
                return candidate
    return candidates[-1]


def run_python_runner(runner: Path, config_path: Path) -> None:
    command = [str(python_bin()), str(runner), "--config", str(config_path)]
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        print_status(
            "interactive-launcher",
            f"子流程返回非零退出码 {exc.returncode}，已保留上游输出，请查看上方 summary / metrics 继续排查。",
        )
        raise SystemExit(exc.returncode)


def load_sync_base_config() -> dict[str, Any]:
    if DEFAULT_SYNC_CONFIG.exists():
        return load_json(DEFAULT_SYNC_CONFIG)
    return {
        "series_name": "",
        "analysis_root": "analysis",
        "provider": "openai",
        "model": "gpt-5.4",
        "genre_override_report_path": "",
        "max_candidates_per_genre": 6,
        "min_candidate_confidence": 0.68,
        "min_confidence_to_apply": 0.82,
        "apply_updates": False,
    }


def infer_provider_model_from_analysis(series_dir: Path, fallback_provider: str, fallback_model: str) -> tuple[str, str]:
    patterns = [
        "genre_override_request__*.json",
        "run_summary__*.json",
        "genre_routing_debug__*.json",
        "episode_analysis__*.json",
    ]
    for pattern in patterns:
        matches = sorted(series_dir.rglob(pattern))
        if not matches:
            continue
        name = matches[-1].stem
        raw = name
        for prefix in (
            "genre_override_request__",
            "run_summary__",
            "genre_routing_debug__",
            "episode_analysis__",
        ):
            raw = raw.replace(prefix, "", 1)
        if "__" in raw:
            provider, model = raw.split("__", 1)
            provider = provider.strip() or fallback_provider
            model = model.strip() or fallback_model
            return provider, model
    return fallback_provider, fallback_model


def maybe_prompt_series_learning_sync(
    *,
    current_series_name: str,
    analysis_root: Path,
    fallback_provider: str,
    fallback_model: str,
) -> None:
    choice = prompt_zero_one("本次批量处理已完成，是否需要积累经验", default_value=0)
    if choice != 1:
        print_status("series-learning-sync", "已跳过经验积累。")
        return

    choices = list_analysis_series_dirs(analysis_root)
    if not choices:
        print_status("series-learning-sync", f"未在 {analysis_root} 下找到可积累经验的剧。")
        return

    default_index = next((i for i, (label, _) in enumerate(choices) if label == current_series_name), 0)
    label, series_dir = prompt_choice("请选择要积累经验的剧：", choices, default_index=default_index)
    provider, model = infer_provider_model_from_analysis(series_dir, fallback_provider, fallback_model)
    base_config = load_sync_base_config()
    updated = copy.deepcopy(base_config)
    updated["series_name"] = label
    updated["analysis_root"] = str(analysis_root)
    updated["provider"] = provider
    updated["model"] = model
    updated["genre_override_report_path"] = ""

    temp_config = write_temp_config("sync_series_learning.interactive.", updated)
    try:
        print_status("series-learning-sync", f"已选择：{label}")
        print_status("series-learning-sync", f"provider/model：{provider}/{model}")
        run_python_runner(SYNC_RUNNER, temp_config)
    finally:
        temp_config.unlink(missing_ok=True)

def interactive_sync_series_learning(config_path: Path, runner: Path, mode: str) -> None:
    config = load_json(config_path)
    analysis_root = Path(str(config.get("analysis_root", "analysis"))).expanduser().resolve()
    choices = list_analysis_series_dirs(analysis_root)
    if not choices:
        raise RuntimeError(f"未在 {analysis_root} 下找到可积累经验的剧。")

    current_series_name = str(config.get("series_name", "")).strip()
    default_index = next((i for i, (label, _) in enumerate(choices) if label == current_series_name), 0)
    label, series_dir = prompt_choice("请选择要积累经验的剧：", choices, default_index=default_index)

    fallback_provider = str(config.get("provider", "openai")).strip() or "openai"
    fallback_model = str(config.get("model", "gpt-5.4")).strip() or "gpt-5.4"
    provider, model = infer_provider_model_from_analysis(series_dir, fallback_provider, fallback_model)

    updated = copy.deepcopy(config)
    updated["series_name"] = label
    updated["analysis_root"] = str(analysis_root)
    updated["provider"] = provider
    updated["model"] = model
    updated["genre_override_report_path"] = ""

    temp_config = write_temp_config("sync_series_learning.interactive.", updated)
    try:
        print_status("series-learning-sync", f"已选择：{label}")
        print_status("series-learning-sync", f"analysis 目录：{series_dir}")
        print_status("series-learning-sync", f"provider/model：{provider}/{model}")
        run_python_runner(runner, temp_config)
    finally:
        temp_config.unlink(missing_ok=True)


def interactive_genre_package_audit(config_path: Path, runner: Path, mode: str) -> None:
    config = load_json(config_path)
    choices = list_official_genre_dirs()
    if not choices:
        raise RuntimeError(f"未在 {GENRE_AUDIT_ROOT} 下找到正式题材目录。")

    current_genre_key = str(config.get("genre_key", "")).strip()
    default_index = next((i for i, (label, _) in enumerate(choices) if label == current_genre_key), 0)
    label, genre_dir = prompt_choice("请选择要体检的题材包：", choices, default_index=default_index)

    updated = copy.deepcopy(config)
    updated["genre_key"] = label
    updated.setdefault("run", {})
    if mode == "preview":
        updated["run"]["apply_mode"] = "audit_only"
        updated["run"]["selected_draft_dir"] = ""
        updated["run"]["cleanup_applied_draft"] = True
    else:
        apply_options = [
            ("先体检并按采纳建议直接并入正式包", "audit_then_apply"),
            ("直接应用已有 Draft 决策", "apply_existing_draft"),
            ("只体检并生成草稿", "audit_only"),
        ]
        _, apply_mode = prompt_named_choice(
            "请选择这次题材包体检的处理方式：",
            apply_options,
            default_index=0,
        )
        updated["run"]["apply_mode"] = apply_mode
        updated["run"]["cleanup_applied_draft"] = True
        updated["run"]["selected_draft_dir"] = ""
        if apply_mode == "apply_existing_draft":
            draft_choices = list_genre_audit_draft_dirs(label)
            if not draft_choices:
                raise RuntimeError(f"未找到题材 `{label}` 的可用 audit draft。")
            draft_label, draft_dir = prompt_choice(
                "请选择要直接应用的已有 Draft：",
                draft_choices,
                default_index=0,
            )
            updated["run"]["selected_draft_dir"] = str(draft_dir)
            print_status("genre-package-audit", f"将直接应用 draft：{draft_label}")
    updated = apply_dry_run_override(updated, mode, key_path=("run", "dry_run"))

    temp_config = write_temp_config("genre_package_audit.interactive.", updated)
    try:
        print_status("genre-package-audit", f"已选择题材：{label}")
        print_status("genre-package-audit", f"题材目录：{genre_dir}")
        print_status("genre-package-audit", f"dry_run：{updated.get('run', {}).get('dry_run', False)}")
        print_status("genre-package-audit", f"apply_mode：{updated.get('run', {}).get('apply_mode', 'audit_only')}")
        run_python_runner(runner, temp_config)
    finally:
        temp_config.unlink(missing_ok=True)


def apply_dry_run_override(config: dict[str, Any], mode: str, *, key_path: tuple[str, ...]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    target = updated
    for key in key_path[:-1]:
        target = target.setdefault(key, {})
    last_key = key_path[-1]
    if mode == "preview":
        target[last_key] = True
    elif mode == "run":
        target[last_key] = False
    return updated


def interactive_openai_flow(
    config_path: Path,
    runner: Path,
    mode: str,
    collect_metrics_override: bool | None = None,
) -> None:
    config = load_json(config_path)
    choices = list_script_series_dirs()
    default_series_name = str(config.get("series", {}).get("series_name") or "").strip()
    default_index = next((i for i, (label, _) in enumerate(choices) if label == default_series_name), 0)
    label, series_dir = prompt_choice("请选择要处理的剧本：", choices, default_index=default_index)
    episode_numbers = detect_script_episode_numbers(series_dir)
    default_start = episode_numbers[0] if episode_numbers else int(config.get("series", {}).get("start_episode", 1))
    default_end = episode_numbers[-1] if episode_numbers else int(config.get("series", {}).get("end_episode", default_start))
    start_episode, end_episode = prompt_episode_range(default_start, default_end)
    existing_genre_reference = dict(config.get("genre_reference", {}))
    genre_selection = prompt_genre_selection(
        tag="openai-agent-flow",
        series_label=label,
        video_dir=series_dir,
        default_library_keys=list(existing_genre_reference.get("library_keys", [])),
        default_custom_tokens=list(existing_genre_reference.get("custom_tokens", [])),
    )
    default_source_index = 0
    existing_suffixes = list(config.get("source", {}).get("preferred_filename_suffixes", []))
    if existing_suffixes:
        if existing_suffixes[:1] == EXPLOSIVE_ONLY_SUFFIXES:
            default_source_index = 2
        elif existing_suffixes[:2] == BASE_SCRIPT_SUFFIXES[:2]:
            default_source_index = 1
    source_label, source_key = prompt_named_choice(
        "请选择后续阶段使用的剧本来源：",
        OPENAI_FLOW_SCRIPT_SOURCE_OPTIONS,
        default_index=default_source_index,
    )
    style_selection = prompt_openai_flow_style_selection(
        default_target_medium=str(config.get("quality", {}).get("target_medium") or "漫剧").strip(),
        default_visual_style=str(config.get("quality", {}).get("visual_style") or "").strip(),
    )
    default_storyboard_profile = str(
        config.get("runtime", {}).get("storyboard_profile")
        or config.get("quality", {}).get("storyboard_profile")
        or "normal"
    ).strip().lower()
    storyboard_profile_default_index = next(
        (index for index, (_, key) in enumerate(STORYBOARD_PROFILE_OPTIONS) if key == default_storyboard_profile),
        0,
    )
    storyboard_profile_label, storyboard_profile = prompt_named_choice(
        "请选择 storyboard 生成模式：",
        STORYBOARD_PROFILE_OPTIONS,
        default_index=storyboard_profile_default_index,
    )
    default_genre_bundle_mode = "reuse" if bool(existing_genre_reference.get("reuse_if_exists", True)) else "rebuild"
    genre_bundle_default_index = next(
        (index for index, (_, key) in enumerate(GENRE_BUNDLE_REUSE_OPTIONS) if key == default_genre_bundle_mode),
        0,
    )
    genre_bundle_mode_label, genre_bundle_mode = prompt_named_choice(
        "请选择题材参考 bundle 的使用方式：",
        GENRE_BUNDLE_REUSE_OPTIONS,
        default_index=genre_bundle_default_index,
    )
    default_run_explosive = bool(config.get("stages", {}).get("run_explosive_rewrite", False))
    run_explosive_rewrite = prompt_bool("是否先运行爆款改稿阶段", default_run_explosive)
    if collect_metrics_override is None:
        default_collect_metrics = bool(config.get("runtime", {}).get("collect_metrics", False))
        collect_metrics = prompt_bool("是否记录时间和 token 统计报告", default_collect_metrics)
    else:
        collect_metrics = collect_metrics_override

    updated = copy.deepcopy(config)
    updated.setdefault("series", {})
    updated["series"]["series_name"] = label
    updated["series"]["script_series_dir"] = str(series_dir)
    updated["series"]["start_episode"] = start_episode
    updated["series"]["end_episode"] = end_episode
    updated.setdefault("source", {})
    updated["source"]["preferred_filename_suffixes"] = resolve_openai_flow_suffixes(source_key)
    updated["source"]["interactive_source_mode"] = source_key
    updated.setdefault("genre_reference", {})
    updated["genre_reference"]["library_keys"] = genre_selection.get("library_keys", [])
    updated["genre_reference"]["custom_tokens"] = genre_selection.get("custom_tokens", [])
    updated["genre_reference"]["ai_suggested_keys"] = genre_selection.get("ai_suggested_keys", [])
    updated["genre_reference"]["source"] = genre_selection.get("source", "interactive")
    updated["genre_reference"]["reuse_if_exists"] = genre_bundle_mode == "reuse"
    updated.setdefault("quality", {})
    updated["quality"]["target_medium"] = style_selection.get("target_medium", "漫剧")
    updated["quality"]["visual_style"] = style_selection.get("visual_style", "按当前项目统一")
    updated["quality"]["frame_orientation"] = str(config.get("quality", {}).get("frame_orientation") or "9:16竖屏").strip() or "9:16竖屏"
    updated.setdefault("stages", {})
    updated["stages"]["run_explosive_rewrite"] = run_explosive_rewrite
    updated.setdefault("runtime", {})
    updated["runtime"]["collect_metrics"] = collect_metrics
    updated["runtime"]["storyboard_profile"] = storyboard_profile
    updated["runtime"]["write_storyboard_metrics"] = True
    updated["runtime"]["prompt_for_seedance_prompt_refine"] = mode != "preview"
    updated = apply_dry_run_override(updated, mode, key_path=("runtime", "dry_run"))

    temp_config = write_temp_config("openai_agent_flow.interactive.", updated)
    try:
        print_status("openai-agent-flow", f"已选择剧本：{label}")
        print_status("openai-agent-flow", f"集数范围：ep{start_episode:02d}-ep{end_episode:02d}")
        print_status(
            "openai-agent-flow",
            "题材确认："
            + (
                "、".join(genre_selection.get("library_keys", []))
                if genre_selection.get("library_keys")
                else "自定义 " + "、".join(genre_selection.get("custom_tokens", []))
            ),
        )
        print_status("openai-agent-flow", f"剧本来源：{source_label}")
        print_status(
            "openai-agent-flow",
            "目标媒介："
            f"{style_selection.get('target_medium', '漫剧')} / 画幅：{updated['quality'].get('frame_orientation', '9:16竖屏')} / "
            f"风格：{style_selection.get('visual_style', '按当前项目统一')}",
        )
        print_status("openai-agent-flow", f"storyboard 模式：{storyboard_profile_label}")
        print_status("openai-agent-flow", f"题材参考 bundle：{genre_bundle_mode_label}")
        print_status("openai-agent-flow", f"爆款改稿阶段：{'开启' if run_explosive_rewrite else '关闭'}")
        print_status("openai-agent-flow", f"统计报告：{'开启' if collect_metrics else '关闭'}")
        run_python_runner(runner, temp_config)
    finally:
        temp_config.unlink(missing_ok=True)


def interactive_explosive_rewrite(
    config_path: Path,
    runner: Path,
    mode: str,
    collect_metrics_override: bool | None = None,
) -> None:
    config = load_json(config_path)
    choices = list_script_series_dirs()
    default_series_name = str(config.get("series", {}).get("series_name") or "").strip()
    default_index = next((i for i, (label, _) in enumerate(choices) if label == default_series_name), 0)
    label, series_dir = prompt_choice("请选择要爆改的剧本：", choices, default_index=default_index)
    episode_numbers = detect_script_episode_numbers(series_dir)
    default_start = episode_numbers[0] if episode_numbers else int(config.get("series", {}).get("start_episode", 1))
    default_end = episode_numbers[-1] if episode_numbers else int(config.get("series", {}).get("end_episode", default_start))
    start_episode, end_episode = prompt_episode_range(default_start, default_end)

    style_target_config = dict(config.get("style_target", {}))
    style_selection = prompt_explosive_style_selection(
        series_label=label,
        script_dir=series_dir,
        default_library_keys=list(style_target_config.get("genre_keys", [])),
        default_custom_tokens=list(style_target_config.get("custom_style_tokens", [])),
    )
    if collect_metrics_override is None:
        default_collect_metrics = bool(config.get("run", {}).get("collect_metrics", False))
        collect_metrics = prompt_bool("是否记录时间和 token 统计报告", default_collect_metrics)
    else:
        collect_metrics = collect_metrics_override

    updated = copy.deepcopy(config)
    updated.setdefault("script", {})
    updated["script"]["series_dir"] = str(series_dir)
    updated["script"]["series_name"] = label
    updated.setdefault("series", {})
    updated["series"]["series_name"] = label
    updated["series"]["start_episode"] = start_episode
    updated["series"]["end_episode"] = end_episode
    updated.setdefault("style_target", {})
    updated["style_target"]["genre_keys"] = style_selection.get("genre_keys", [])
    updated["style_target"]["custom_style_tokens"] = style_selection.get("custom_style_tokens", [])
    updated["style_target"]["style_label"] = style_selection.get("style_label", "")
    updated.setdefault("run", {})
    updated["run"]["collect_metrics"] = collect_metrics
    updated = apply_dry_run_override(updated, mode, key_path=("run", "dry_run"))

    temp_config = write_temp_config("explosive_rewrite.interactive.", updated)
    try:
        print_status("explosive-rewrite", f"已选择剧本：{label}")
        print_status("explosive-rewrite", f"集数范围：ep{start_episode:02d}-ep{end_episode:02d}")
        print_status(
            "explosive-rewrite",
            "目标风格："
            + (
                "、".join(style_selection.get("genre_keys", []))
                if style_selection.get("genre_keys")
                else "自定义 " + "、".join(style_selection.get("custom_style_tokens", []))
            ),
        )
        print_status("explosive-rewrite", f"统计报告：{'开启' if collect_metrics else '关闭'}")
        run_python_runner(runner, temp_config)
    finally:
        temp_config.unlink(missing_ok=True)


def interactive_series_pipeline(
    config_path: Path,
    runner: Path,
    mode: str,
    collect_metrics_override: bool | None = None,
) -> None:
    config = load_json(config_path)
    file_extensions = list(config.get("series", {}).get("file_extensions", [".mp4"]))
    custom_regex = str(config.get("series", {}).get("episode_number_regex") or "").strip() or None
    choices = list_video_series_dirs(file_extensions)
    current_video_dir = Path(str(config.get("series", {}).get("video_dir", ""))).expanduser().resolve()
    default_index = next((i for i, (_, path) in enumerate(choices) if path == current_video_dir), 0)
    label, video_dir = prompt_choice("请选择要处理的视频目录：", choices, default_index=default_index)
    episode_numbers = detect_video_episode_numbers(video_dir, file_extensions, custom_regex)
    default_start = episode_numbers[0] if episode_numbers else int(config.get("series", {}).get("start_episode", 1))
    default_end = episode_numbers[-1] if episode_numbers else int(config.get("series", {}).get("end_episode", default_start))
    start_episode, end_episode = prompt_episode_range(default_start, default_end)
    existing_genre_hints = dict(config.get("series", {}).get("genre_hints", {}))
    genre_selection = prompt_genre_selection(
        series_label=label,
        video_dir=video_dir,
        default_library_keys=list(existing_genre_hints.get("library_keys", [])),
        default_custom_tokens=list(existing_genre_hints.get("custom_tokens", [])),
    )
    if collect_metrics_override is None:
        default_collect_metrics = bool(config.get("run", {}).get("collect_metrics", False))
        collect_metrics = prompt_bool("是否记录时间和 token 统计报告", default_collect_metrics)
    else:
        collect_metrics = collect_metrics_override

    updated = copy.deepcopy(config)
    updated.setdefault("series", {})
    updated["series"]["video_dir"] = str(video_dir)
    updated["series"]["series_name"] = video_dir.name
    updated["series"]["start_episode"] = start_episode
    updated["series"]["end_episode"] = end_episode
    updated["series"]["genre_hints"] = genre_selection
    updated.setdefault("run", {})
    updated["run"]["collect_metrics"] = collect_metrics
    updated["run"]["prompt_on_genre_override"] = True
    updated = apply_dry_run_override(updated, mode, key_path=("run", "dry_run"))

    temp_config = write_temp_config("series_pipeline.interactive.", updated)
    try:
        print_status("series-pipeline", f"已选择视频目录：{label}")
        print_status("series-pipeline", f"剧名将使用：{video_dir.name}")
        print_status("series-pipeline", f"集数范围：ep{start_episode:02d}-ep{end_episode:02d}")
        print_status(
            "series-pipeline",
            "题材确认："
            + (
                "、".join(genre_selection.get("library_keys", []))
                if genre_selection.get("library_keys")
                else "自定义 " + "、".join(genre_selection.get("custom_tokens", []))
            ),
        )
        print_status("series-pipeline", f"统计报告：{'开启' if collect_metrics else '关闭'}")
        run_python_runner(runner, temp_config)
        if not bool(updated.get("run", {}).get("dry_run", False)):
            maybe_prompt_series_learning_sync(
                current_series_name=video_dir.name,
                analysis_root=Path(str(updated.get("run", {}).get("analysis_root", "analysis"))).expanduser().resolve(),
                fallback_provider=str(updated.get("run", {}).get("selected_provider", "openai")),
                fallback_model=str(updated.get("providers", {}).get(updated.get("run", {}).get("selected_provider", ""), {}).get("model", "gpt-5.4")),
            )
    finally:
        temp_config.unlink(missing_ok=True)


def interactive_nano_banana(config_path: Path, runner: Path, mode: str) -> None:
    config = load_json(config_path)
    ready_series = list_ready_nano_banana_series()
    if not ready_series:
        raise RuntimeError("当前没有同时具备 script / assets 材料的剧，无法运行 Nano Banana。")

    current_script_path = str(config.get("script", {}).get("script_path") or "").strip()
    current_series_name = canonical_series_base_name(Path(current_script_path).expanduser().resolve().parent.name) if current_script_path else ""
    choices = [(label, script_dir) for label, script_dir, _, _ in ready_series]
    default_index = next((i for i, (_, script_dir, _, _) in enumerate(ready_series) if script_dir.name == current_series_name), 0)
    selected_label, selected_series_dir = prompt_choice("请选择要出图的已完成材料剧本：", choices, default_index=default_index)
    matched = next((item for item in ready_series if item[0] == selected_label and item[1] == selected_series_dir), None)
    if matched is None:
        raise RuntimeError("无法匹配已选择的 Nano Banana 剧本材料。")
    label, series_dir, chosen_assets_dir, chosen_outputs_dir = matched

    output_episode_numbers = detect_output_episode_numbers(chosen_outputs_dir)
    script_episode_numbers = detect_nano_banana_script_episode_numbers(series_dir)
    episode_numbers = output_episode_numbers or script_episode_numbers
    default_start = episode_numbers[0] if episode_numbers else 1
    default_end = episode_numbers[-1] if episode_numbers else default_start
    start_episode, end_episode = prompt_episode_range(default_start, default_end)

    preferred_suffixes = list(config.get("script", {}).get("preferred_filename_suffixes", [])) or DEFAULT_SCRIPT_SUFFIXES
    print_status("nano-banana", f"将优先读取 assets：{chosen_assets_dir}")
    if chosen_outputs_dir is not None:
        print_status("nano-banana", f"将优先读取 outputs：{chosen_outputs_dir}")
    else:
        print_status("nano-banana", "未检测到该剧 outputs 目录，先执行角色/场景出图，后续可补跑分镜切分。")

    failed: list[str] = []
    for episode_number in range(start_episode, end_episode + 1):
        episode_id = f"ep{episode_number:02d}"
        if chosen_outputs_dir is None:
            storyboard_episode_dir = PROJECT_ROOT / "outputs" / chosen_assets_dir.name / episode_id
        else:
            storyboard_episode_dir = chosen_outputs_dir / episode_id
        detected_storyboard_path = find_storyboard_prompt_path(storyboard_episode_dir)
        storyboard_prompt_path = detected_storyboard_path or (storyboard_episode_dir / "02-seedance-prompts.md")
        storyboard_exists = detected_storyboard_path is not None
        episode_assets_dir = chosen_assets_dir / episode_id
        character_prompt_file = episode_assets_dir / "character-prompts.md"
        scene_prompt_file = episode_assets_dir / "scene-prompts.md"
        if character_prompt_file.exists() and scene_prompt_file.exists():
            chosen_character_prompts = str(character_prompt_file.resolve())
            chosen_scene_prompts = str(scene_prompt_file.resolve())
        else:
            chosen_character_prompts = str((chosen_assets_dir / "character-prompts.md").resolve())
            chosen_scene_prompts = str((chosen_assets_dir / "scene-prompts.md").resolve())
        layout_overrides = prompt_scene_layout_overrides(
            episode_id=episode_id,
            scene_prompt_path=Path(chosen_scene_prompts),
        )
        script_path: Path | None = None
        for candidate_series_dir in list_nano_banana_script_series_dirs(series_dir):
            try:
                script_path = select_script_for_episode(candidate_series_dir, episode_id, preferred_suffixes)
                break
            except FileNotFoundError:
                continue
        if script_path is None:
            failed.append(episode_id)
            print_status("nano-banana", f"跳过 {episode_id}：未找到对应剧本文件。")
            continue

        updated = copy.deepcopy(config)
        updated.setdefault("script", {})
        updated["script"]["script_path"] = str(script_path)
        updated["script"]["series_name"] = series_dir.name
        updated["script"]["episode_id"] = episode_id
        updated.setdefault("sources", {})
        updated["sources"]["character_prompts_path"] = chosen_character_prompts
        updated["sources"]["scene_prompts_path"] = chosen_scene_prompts
        updated["sources"]["seedance_storyboard_path"] = str(storyboard_prompt_path.resolve()) if storyboard_exists else ""
        updated.setdefault("selection", {})
        current_layout_overrides = copy.deepcopy(updated["selection"].get("scene_layout_overrides") or {})
        if layout_overrides:
            current_layout_overrides[episode_id] = layout_overrides
        updated["selection"]["scene_layout_overrides"] = current_layout_overrides
        if not storyboard_exists:
            updated["selection"]["split_scenes_from_storyboard_table"] = False
            print_status(
                "nano-banana",
                f"{episode_id} 暂无 02-seedance-prompts.md，先只生成人物/场景图；待分镜产出后再运行一次即可自动切分场景素材。",
            )
        updated.setdefault("output", {})
        if not str(updated["output"].get("output_root") or "").strip():
            model_name = str(updated.get("provider", {}).get("gemini", {}).get("model", "gemini-3-pro-image-preview"))
            updated["output"]["output_root"] = str(
                chosen_assets_dir / "generated" / model_name / episode_id
            )
        updated = apply_dry_run_override(updated, mode, key_path=("run", "dry_run"))

        temp_config = write_temp_config("nano_banana.interactive.", updated)
        try:
            print_status("nano-banana", f"已选择剧本：{label} / {episode_id} -> {script_path.name}")
            run_python_runner(runner, temp_config)
        finally:
            temp_config.unlink(missing_ok=True)

    if failed:
        print_status("nano-banana", f"以下集数材料不完整，已跳过：{', '.join(failed)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive launcher for Seedance workflow shell wrappers.")
    parser.add_argument("--mode", choices=["openai_flow", "series_pipeline", "nano_banana", "explosive_rewrite", "sync_series_learning", "genre_package_audit"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--run-mode", choices=["interactive", "preview", "run"], default="interactive")
    parser.add_argument("--collect-metrics", choices=["true", "false"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not sys.stdin.isatty():
        raise RuntimeError("当前不是交互式终端，无法进行终端选择。请改用 config 模式。")

    config_path = Path(args.config).expanduser().resolve()
    runner = Path(args.runner).expanduser().resolve()
    if args.mode == "openai_flow":
        if args.collect_metrics is not None:
            interactive_openai_flow(
                config_path,
                runner,
                args.run_mode,
                collect_metrics_override=(args.collect_metrics == "true"),
            )
        else:
            interactive_openai_flow(config_path, runner, args.run_mode)
        return
    if args.mode == "series_pipeline":
        if args.collect_metrics is not None:
            interactive_series_pipeline(
                config_path,
                runner,
                args.run_mode,
                collect_metrics_override=(args.collect_metrics == "true"),
            )
        else:
            interactive_series_pipeline(config_path, runner, args.run_mode)
        return
    if args.mode == "explosive_rewrite":
        if args.collect_metrics is not None:
            interactive_explosive_rewrite(
                config_path,
                runner,
                args.run_mode,
                collect_metrics_override=(args.collect_metrics == "true"),
            )
        else:
            interactive_explosive_rewrite(config_path, runner, args.run_mode)
        return
    if args.mode == "sync_series_learning":
        interactive_sync_series_learning(config_path, runner, args.run_mode)
        return
    if args.mode == "genre_package_audit":
        interactive_genre_package_audit(config_path, runner, args.run_mode)
        return
    interactive_nano_banana(config_path, runner, args.run_mode)


if __name__ == "__main__":
    main()
