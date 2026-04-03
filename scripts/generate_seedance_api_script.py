from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
ASSETS_ROOT = PROJECT_ROOT / "assets"
TASK_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
DEFAULT_MODEL = "doubao-seedance-2-0-260128"
DEFAULT_RATIO = "9:16"
DEFAULT_RESOLUTION = "480p"
DEFAULT_DURATION = 8
DEFAULT_GENERATE_AUDIO = True
DEFAULT_WATERMARK = False
MAX_REFERENCE_IMAGES = 9
REFERENCE_MODE_OPTIONS = [
    ("tos", "TOS 公网/预签名 URL"),
    ("asset", "Asset 审核资产 ID（asset://）"),
]
REFERENCE_MODE_VALUES = {mode for mode, _ in REFERENCE_MODE_OPTIONS}


@dataclass
class MaterialReference:
    token: str
    token_number: int
    material_type: str
    label: str


@dataclass
class ScenePrompt:
    scene_id: str
    heading: str
    duration_text: str | None
    duration_value: int
    reference_tokens: list[str]
    prompt_text: str
    pace_label: str = ""
    density_strategy: str = ""
    continuity_bridge: str = ""
    master_timeline: list[dict[str, Any]] = field(default_factory=list)
    shot_beat_plan: list[str] = field(default_factory=list)
    dialogue_timeline: list[dict[str, Any]] = field(default_factory=list)
    audio_design: str = ""


def print_status(message: str) -> None:
    print(f"[seedance-api] {message}", flush=True)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def slugify(raw: str) -> str:
    clean = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", raw.strip())
    clean = re.sub(r"-{2,}", "-", clean).strip("-")
    return clean or "untitled"


def chinese_bigrams(raw: str) -> list[str]:
    chars = [char for char in str(raw or "") if "\u4e00" <= char <= "\u9fff"]
    if len(chars) < 2:
        return chars
    return ["".join(chars[index:index + 2]) for index in range(len(chars) - 1)]


def normalize_episode_key(raw: str | None) -> str | None:
    if not raw:
        return None
    match = re.search(r"ep(\d+)", raw, flags=re.IGNORECASE)
    if not match:
        return None
    return f"ep{int(match.group(1)):02d}"


def episode_sort_key(raw: str | None) -> tuple[int, str]:
    normalized = normalize_episode_key(raw)
    if not normalized:
        return (10**9, raw or "")
    match = re.search(r"(\d+)", normalized)
    return (int(match.group(1)), normalized)


def find_storyboard_path(episode_dir: Path) -> Path | None:
    exact_path = episode_dir / "02-seedance-prompts.md"
    if exact_path.exists():
        return exact_path

    candidates = [
        path for path in episode_dir.glob("02-seedance-prompts*.md")
        if path.is_file() and not path.name.endswith(".report.md")
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return candidates[-1]


def list_series_dirs() -> list[Path]:
    result: list[Path] = []
    if not OUTPUTS_ROOT.exists():
        return result
    for child in sorted(OUTPUTS_ROOT.iterdir()):
        if child.is_dir() and any(find_storyboard_path(episode_dir) for episode_dir in child.iterdir() if episode_dir.is_dir()):
            result.append(child)
    return result


def choose_from_list(title: str, options: list[str], default_index: int = 0) -> int:
    if not options:
        raise RuntimeError(f"没有可选项：{title}")
    print(title)
    for index, option in enumerate(options, start=1):
        suffix = "  [默认]" if index - 1 == default_index else ""
        print(f"  {index}. {option}{suffix}")
    raw = input(f"请输入序号（默认 {default_index + 1}）：").strip()
    if not raw:
        return default_index
    if not raw.isdigit():
        raise RuntimeError(f"输入无效：{raw}")
    selected = int(raw) - 1
    if selected < 0 or selected >= len(options):
        raise RuntimeError(f"输入超出范围：{raw}")
    return selected


def choose_range_from_list(title: str, options: list[str], default_start: int = 0, default_end: int | None = None) -> tuple[int, int]:
    if not options:
        raise RuntimeError(f"没有可选项：{title}")
    if default_end is None:
        default_end = default_start
    print(title)
    for index, option in enumerate(options, start=1):
        markers: list[str] = []
        if index - 1 == default_start:
            markers.append("默认起始")
        if index - 1 == default_end:
            markers.append("默认结束")
        suffix = f"  [{' / '.join(markers)}]" if markers else ""
        print(f"  {index}. {option}{suffix}")
    start_raw = input(f"请输入起始序号（默认 {default_start + 1}）：").strip()
    end_raw = input(f"请输入结束序号（默认 {default_end + 1}）：").strip()
    start_index = default_start if not start_raw else int(start_raw) - 1
    end_index = default_end if not end_raw else int(end_raw) - 1
    if start_index < 0 or end_index < 0 or start_index >= len(options) or end_index >= len(options):
        raise RuntimeError("输入超出范围。")
    if start_index > end_index:
        raise RuntimeError("起始序号不能大于结束序号。")
    return start_index, end_index


def workflow_config_path(episode_dir: Path) -> Path:
    return episode_dir / "seedance_references_workflow.json"


def load_reference_workflow_config(episode_dir: Path) -> dict[str, Any]:
    path = workflow_config_path(episode_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve_reference_mode_default(default_mode: str = "tos") -> str:
    env_default_mode = str(os.environ.get("SEEDANCE_REFERENCE_MODE_DEFAULT") or "").strip().lower()
    candidate = env_default_mode or str(default_mode or "").strip().lower() or "tos"
    return candidate if candidate in REFERENCE_MODE_VALUES else "tos"


def choose_reference_mode(default_mode: str = "tos") -> str:
    default_mode = resolve_reference_mode_default(default_mode)
    option_labels = [label for _, label in REFERENCE_MODE_OPTIONS]
    default_index = next(
        (index for index, (mode, _) in enumerate(REFERENCE_MODE_OPTIONS) if mode == default_mode),
        0,
    )
    selected_index = choose_from_list(
        "请选择 Seedance 引用素材模式：",
        option_labels,
        default_index=default_index,
    )
    return REFERENCE_MODE_OPTIONS[selected_index][0]


def save_reference_workflow_config(
    *,
    episode_dir: Path,
    series_name: str,
    episode_id: str,
    reference_mode: str,
    selected_scene_ids: list[str],
    asset_provider: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> Path:
    path = workflow_config_path(episode_dir)
    existing_payload = load_reference_workflow_config(episode_dir)
    payload = {
        **existing_payload,
        "series_name": series_name,
        "episode_id": episode_id,
        "reference_mode": reference_mode,
        "selected_scene_ids": selected_scene_ids,
        "updated_at": now_iso(),
    }
    if asset_provider is not None:
        payload["asset_provider"] = asset_provider
    if extra_fields:
        payload.update(extra_fields)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def parse_material_table(path: Path) -> dict[str, MaterialReference]:
    text = path.read_text(encoding="utf-8")
    table_match = re.search(r"^##\s+素材对应表\s*$", text, flags=re.MULTILINE)
    if not table_match:
        return {}
    remainder = text[table_match.end():]
    next_section = re.search(r"^##\s+", remainder, flags=re.MULTILINE)
    table_block = remainder[: next_section.start()] if next_section else remainder

    refs: dict[str, MaterialReference] = {}
    for raw_line in table_block.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        columns = [part.strip() for part in line.strip("|").split("|")]
        if len(columns) != 3:
            continue
        if columns[0] == "引用编号":
            continue
        token_match = re.search(r"@图片(\d+)", columns[0])
        if not token_match:
            continue
        token_number = int(token_match.group(1))
        token = f"@图片{token_number}"
        refs[token] = MaterialReference(
            token=token,
            token_number=token_number,
            material_type=columns[1],
            label=columns[2],
        )
    return refs


def derive_duration_value(duration_text: str | None) -> int:
    if not duration_text:
        return DEFAULT_DURATION
    numbers = [int(item) for item in re.findall(r"\d+", duration_text)]
    if not numbers:
        return DEFAULT_DURATION
    return max(numbers)


def normalize_scene_id(raw: str | None, *, fallback_index: int | None = None) -> str:
    text = str(raw or "").strip()
    numeric_match = re.fullmatch(r"[Pp]?0*(\d{1,3})", text)
    if numeric_match:
        return f"P{int(numeric_match.group(1)):02d}"
    if fallback_index is not None:
        return f"P{int(fallback_index):02d}"
    return text


def strip_scene_heading_prefix(title: str, scene_id: str) -> str:
    clean_title = str(title or "").strip()
    point_match = re.fullmatch(r"P(\d+)", scene_id, flags=re.IGNORECASE)
    if not clean_title or not point_match:
        return clean_title
    number = str(int(point_match.group(1)))
    patterns = [
        rf"^{re.escape(scene_id)}(?:[\s:：、.\-]+|$)",
        rf"^0*{re.escape(number)}(?:[\s:：、.\-]+|$)",
        rf"^[Pp]0*{re.escape(number)}(?:[\s:：、.\-]+|$)",
    ]
    for pattern in patterns:
        updated = re.sub(pattern, "", clean_title, count=1).strip()
        if updated and updated != clean_title:
            return updated
    return clean_title


def parse_scene_prompts(path: Path) -> list[ScenePrompt]:
    markdown_mtime_ns = path.stat().st_mtime_ns if path.exists() else -1
    json_candidates = sorted(
        path.parent.glob("02-seedance-prompts__*.json"),
        key=lambda candidate: candidate.stat().st_mtime_ns,
    )
    if json_candidates:
        latest_json = json_candidates[-1]
        latest_json_mtime_ns = latest_json.stat().st_mtime_ns
        if latest_json_mtime_ns >= markdown_mtime_ns:
            try:
                return parse_scene_prompts_from_json(latest_json)
            except Exception:
                pass
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^##\s+([^\s#]+)\s+(.+?)\s*$", flags=re.MULTILINE)
    matches = list(pattern.finditer(text))
    scenes: list[ScenePrompt] = []
    for index, match in enumerate(matches):
        scene_id = normalize_scene_id(match.group(1).strip(), fallback_index=index + 1)
        scene_title = strip_scene_heading_prefix(match.group(2).strip(), scene_id)
        heading = f"{scene_id} {scene_title}".strip()
        block_start = match.end()
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[block_start:block_end].strip()

        duration_match = re.search(r"建议时长：([^\n]+)", block)
        duration_text = duration_match.group(1).strip() if duration_match else None
        prompt_match = re.search(r"\*\*Seedance 2\.0 提示词\*\*：\s*\n\n(?P<prompt>.*)", block, flags=re.DOTALL)
        if not prompt_match:
            prompt_match = re.search(r"可直接投喂正文：\s*\n\n(?P<prompt>.*)", block, flags=re.DOTALL)
        if not prompt_match:
            continue
        prompt_text = prompt_match.group("prompt").strip()
        prompt_text = re.split(r"\n---+\n", prompt_text)[0].strip()

        refs_match = re.search(r"主要引用：([^\n]+)", block)
        ref_sources = []
        if refs_match:
            ref_sources.append(refs_match.group(1))
        ref_sources.append(prompt_text)
        reference_tokens: list[str] = []
        for source in ref_sources:
            for token in re.findall(r"@图片\d+", source):
                if token not in reference_tokens:
                    reference_tokens.append(token)

        scenes.append(
            ScenePrompt(
                scene_id=scene_id,
                heading=heading,
                duration_text=duration_text,
                duration_value=derive_duration_value(duration_text),
                reference_tokens=reference_tokens,
                prompt_text=prompt_text,
            )
        )
    return scenes


def parse_scene_prompts_from_json(path: Path) -> list[ScenePrompt]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenes: list[ScenePrompt] = []
    for index, item in enumerate(list(payload.get("prompt_entries") or []), start=1):
        scene_id = normalize_scene_id(str(item.get("point_id") or "").strip(), fallback_index=index)
        title = strip_scene_heading_prefix(str(item.get("title") or "").strip(), scene_id)
        if not scene_id:
            continue
        prompt_text = str(item.get("prompt_text") or "").strip()
        reference_tokens: list[str] = []
        for token in list(item.get("primary_refs") or []) + list(item.get("secondary_refs") or []):
            token_text = str(token).strip()
            if token_text and token_text not in reference_tokens:
                reference_tokens.append(token_text)
        for token in re.findall(r"@图片\d+", prompt_text):
            if token not in reference_tokens:
                reference_tokens.append(token)
        scenes.append(
            ScenePrompt(
                scene_id=scene_id,
                heading=f"{scene_id} {title}".strip(),
                duration_text=str(item.get("duration_hint") or "").strip() or None,
                duration_value=derive_duration_value(str(item.get("duration_hint") or "").strip()),
                reference_tokens=reference_tokens,
                prompt_text=prompt_text,
                pace_label=str(item.get("pace_label") or "").strip(),
                density_strategy=str(item.get("density_strategy") or "").strip(),
                continuity_bridge=str(item.get("continuity_bridge") or "").strip(),
                master_timeline=[dict(x) for x in list(item.get("master_timeline") or []) if isinstance(x, dict)],
                shot_beat_plan=[str(x).strip() for x in list(item.get("shot_beat_plan") or []) if str(x).strip()],
                dialogue_timeline=[dict(x) for x in list(item.get("dialogue_timeline") or []) if isinstance(x, dict)],
                audio_design=str(item.get("audio_design") or "").strip(),
            )
        )
    if not scenes:
        raise RuntimeError(f"未从 {path} 解析到任何 prompt_entries。")
    return scenes

def find_assets_dir(series_name: str) -> Path | None:
    for candidate in [ASSETS_ROOT / series_name, ASSETS_ROOT / f"{series_name}-gpt"]:
        if candidate.exists():
            return candidate
    return None


def find_latest_matching_asset(
    *,
    assets_dir: Path | None,
    episode_id: str,
    subdir: str,
    matcher: Callable[[Path], bool],
) -> Path | None:
    if assets_dir is None:
        return None
    generated_root = assets_dir / "generated"
    if not generated_root.exists():
        return None
    current_episode = normalize_episode_key(episode_id)
    candidates: list[tuple[tuple[int, str, str], Path]] = []
    for model_dir in generated_root.iterdir():
        if not model_dir.is_dir():
            continue
        for episode_dir in model_dir.iterdir():
            if not episode_dir.is_dir():
                continue
            episode_key = normalize_episode_key(episode_dir.name)
            if episode_key is None:
                continue
            if current_episode is not None and episode_sort_key(episode_key) > episode_sort_key(current_episode):
                continue
            target_dir = episode_dir / subdir
            if not target_dir.exists():
                continue
            for file_path in sorted(target_dir.iterdir()):
                if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue
                if matcher(file_path):
                    candidates.append(((episode_sort_key(episode_key)[0], model_dir.name, file_path.name), file_path.resolve()))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def load_scene_material_manifest_map(assets_dir: Path | None, episode_id: str) -> dict[int, Path]:
    if assets_dir is None:
        return {}
    generated_root = assets_dir / "generated"
    if not generated_root.exists():
        return {}
    current_episode = normalize_episode_key(episode_id)
    manifests: list[tuple[tuple[int, str], Path]] = []
    for model_dir in generated_root.iterdir():
        if not model_dir.is_dir():
            continue
        for episode_dir in model_dir.iterdir():
            if not episode_dir.is_dir():
                continue
            episode_key = normalize_episode_key(episode_dir.name)
            if episode_key is None:
                continue
            if current_episode is not None and episode_sort_key(episode_key) > episode_sort_key(current_episode):
                continue
            manifest_path = episode_dir / "scene_material_manifest.json"
            if manifest_path.exists():
                manifests.append(((episode_sort_key(episode_key)[0], model_dir.name), manifest_path))
    if not manifests:
        return {}
    manifests.sort(key=lambda item: item[0])
    manifest_path = manifests[-1][1]
    try:
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    result: dict[int, Path] = {}
    for item in payload.get('items', []):
        try:
            number = int(item.get('reference_number'))
        except Exception:
            continue
        output_path = item.get('output_path')
        if not output_path:
            continue
        path = Path(output_path)
        if path.exists():
            result[number] = path.resolve()
    return result


def is_global_cinematic_anchor_reference(ref: MaterialReference) -> bool:
    return ref.material_type == "人物参考" and "全局电影化锚点" in ref.label


def character_asset_matches_reference(path: Path, ref: MaterialReference) -> bool:
    if is_global_cinematic_anchor_reference(ref):
        return False
    stem_suffix = path.stem.partition("__")[2]
    if not stem_suffix:
        return False
    label_slug = slugify(ref.label)
    base_name = ref.label.split("（", 1)[0].split("｜", 1)[0].strip()
    base_slug = slugify(base_name)
    variant_qualified = bool(ref.label.strip() and ref.label.strip() != base_name)
    if stem_suffix == label_slug:
        return True
    if variant_qualified:
        return False
    if base_slug and stem_suffix == base_slug:
        return True
    if base_slug and stem_suffix.startswith(base_slug):
        return True
    return bool(base_name and base_name in stem_suffix)


def resolve_local_reference_path(series_name: str, episode_id: str, ref: MaterialReference) -> Path | None:
    assets_dir = find_assets_dir(series_name)
    prefix = f"{ref.token_number:03d}__"

    if ref.material_type == "人物参考":
        if is_global_cinematic_anchor_reference(ref):
            return None
        label_slug = slugify(ref.label)
        base_name = ref.label.split("（", 1)[0].split("｜", 1)[0].strip()
        base_slug = slugify(base_name)
        for matcher in (
            lambda path: path.stem.partition("__")[2] == label_slug,
            lambda path: path.stem.partition("__")[2] == base_slug,
            lambda path: path.stem.partition("__")[2].startswith(base_slug),
            lambda path: base_name in path.stem.partition("__")[2],
        ):
            preferred = find_latest_matching_asset(
                assets_dir=assets_dir,
                episode_id=episode_id,
                subdir="characters",
                matcher=matcher,
            )
            if preferred is not None:
                return preferred

        exact_by_prefix = find_latest_matching_asset(
            assets_dir=assets_dir,
            episode_id=episode_id,
            subdir="characters",
            matcher=lambda path: path.name.startswith(prefix),
        )
        if exact_by_prefix is not None and character_asset_matches_reference(exact_by_prefix, ref):
            return exact_by_prefix
        return None

    if ref.material_type == "场景参考":
        manifest_map = load_scene_material_manifest_map(assets_dir, episode_id)
        manifest_hit = manifest_map.get(ref.token_number)
        if manifest_hit is not None:
            return manifest_hit

        exact_by_prefix = find_latest_matching_asset(
            assets_dir=assets_dir,
            episode_id=episode_id,
            subdir="scene_materials",
            matcher=lambda path: path.name.startswith(prefix),
        )
        if exact_by_prefix is not None:
            return exact_by_prefix

        scene_name = ref.label.split("（", 1)[0].strip()
        scene_slug = slugify(scene_name)
        bigrams = chinese_bigrams(scene_name)
        semantic_matchers = [
            lambda path: path.stem.partition("__")[2] == scene_slug,
            lambda path: path.stem.partition("__")[2].startswith(scene_slug),
            lambda path: scene_name in path.stem.partition("__")[2],
            lambda path: sum(1 for token in bigrams if token and token in path.stem.partition("__")[2]) >= max(2, min(3, len(bigrams))),
        ]
        for matcher in semantic_matchers:
            preferred = find_latest_matching_asset(
                assets_dir=assets_dir,
                episode_id=episode_id,
                subdir="scene_materials",
                matcher=matcher,
            )
            if preferred is not None:
                return preferred

        exact_grid_by_prefix = find_latest_matching_asset(
            assets_dir=assets_dir,
            episode_id=episode_id,
            subdir="scenes",
            matcher=lambda path: path.name.startswith(prefix),
        )
        if exact_grid_by_prefix is not None:
            return exact_grid_by_prefix
        for matcher in semantic_matchers:
            preferred = find_latest_matching_asset(
                assets_dir=assets_dir,
                episode_id=episode_id,
                subdir="scenes",
                matcher=matcher,
            )
            if preferred is not None:
                return preferred
        return None

    if ref.material_type == "未知素材":
        for subdir in ("characters", "scene_materials", "scenes"):
            exact_by_prefix = find_latest_matching_asset(
                assets_dir=assets_dir,
                episode_id=episode_id,
                subdir=subdir,
                matcher=lambda path: path.name.startswith(prefix),
            )
            if exact_by_prefix is not None:
                return exact_by_prefix
        return None

    return None

def build_reference_payload(series_name: str, episode_id: str, refs: list[MaterialReference]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for ref in refs:
        if is_global_cinematic_anchor_reference(ref):
            continue
        items.append(
            {
                "token": ref.token,
                "token_number": ref.token_number,
                "material_type": ref.material_type,
                "label": ref.label,
                "env_var": f"REF_IMAGE_{ref.token_number}_URL",
                "local_path": str(resolve_local_reference_path(series_name, episode_id, ref) or ""),
            }
        )
    return items


def build_dialogue_timeline_text(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return ""
    parts: list[str] = []
    for item in entries:
        speaker = str(item.get("speaker") or "角色").strip() or "角色"
        line = str(item.get("line") or "").strip()
        if not line:
            continue
        delivery_note = str(item.get("delivery_note") or "").strip()
        delivery = f"（{delivery_note}）" if delivery_note else ""
        start_second = item.get("start_second")
        end_second = item.get("end_second")
        time_window = ""
        if start_second is not None and end_second is not None:
            try:
                time_window = f"{float(start_second):.1f}-{float(end_second):.1f}秒，"
            except (TypeError, ValueError):
                time_window = ""
        parts.append(f"{time_window}{speaker}{delivery}说\"{line}\"")
    if not parts:
        return ""
    return "对白按以下顺序串行出现：" + "；".join(parts) + "。前一句完全结束后下一句再进入，不允许双声叠台词。"


def compose_payload_prompt_text(scene: ScenePrompt) -> str:
    blocks: list[str] = []
    prompt_text = str(scene.prompt_text or "").strip()
    if prompt_text:
        blocks.append(prompt_text)
    if scene.continuity_bridge:
        blocks.append(f"首尾镜承接上，{scene.continuity_bridge}")
    if scene.shot_beat_plan and not scene.master_timeline:
        blocks.append("镜头依次推进为：" + "；".join(scene.shot_beat_plan))
    dialogue_timeline_text = build_dialogue_timeline_text(scene.dialogue_timeline)
    if dialogue_timeline_text and not scene.master_timeline:
        blocks.append(dialogue_timeline_text)
    if scene.audio_design and not scene.master_timeline:
        blocks.append(f"声场与触发物重点：{scene.audio_design}")
    return "\n\n".join(blocks).strip()


def limit_reference_payload(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(items) <= MAX_REFERENCE_IMAGES:
        return items, []
    scene_refs = [item for item in items if "场景" in str(item.get("material_type", ""))]
    other_refs = [item for item in items if item not in scene_refs]
    if len(scene_refs) >= MAX_REFERENCE_IMAGES:
        kept = scene_refs[:MAX_REFERENCE_IMAGES]
    else:
        remaining = MAX_REFERENCE_IMAGES - len(scene_refs)
        kept = scene_refs + other_refs[:remaining]
    dropped = [item for item in items if item not in kept]
    return kept, dropped


def render_payload_template(scene: ScenePrompt, references: list[dict[str, Any]]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": compose_payload_prompt_text(scene)}]
    for ref in references:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"__{ref['env_var']}__"},
                "role": "reference_image",
            }
        )
    return {
        "model": "__SEEDANCE_MODEL__",
        "content": content,
        "generate_audio": "__SEEDANCE_GENERATE_AUDIO__",
        "ratio": "__SEEDANCE_RATIO__",
        "resolution": "__SEEDANCE_RESOLUTION__",
        "duration": "__SEEDANCE_DURATION__",
        "watermark": "__SEEDANCE_WATERMARK__",
    }


def load_env_exports(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    exports: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            exports[key] = value
    return exports


def render_local_payload(
    *,
    scene: ScenePrompt,
    references: list[dict[str, Any]],
    payload_template: dict[str, Any],
    env_exports: Mapping[str, str],
) -> dict[str, Any] | None:
    required_reference_vars = [str(ref.get("env_var") or "").strip() for ref in references if str(ref.get("env_var") or "").strip()]
    if any(not env_exports.get(key) for key in required_reference_vars):
        return None

    replacement_values = {
        "SEEDANCE_MODEL": str(env_exports.get("SEEDANCE_MODEL") or DEFAULT_MODEL),
        "SEEDANCE_RATIO": str(env_exports.get("SEEDANCE_RATIO") or DEFAULT_RATIO),
        "SEEDANCE_RESOLUTION": str(env_exports.get("SEEDANCE_RESOLUTION") or DEFAULT_RESOLUTION),
        "SEEDANCE_DURATION": str(env_exports.get("SEEDANCE_DURATION") or scene.duration_value),
        "SEEDANCE_GENERATE_AUDIO": str(env_exports.get("SEEDANCE_GENERATE_AUDIO") or str(DEFAULT_GENERATE_AUDIO).lower()),
        "SEEDANCE_WATERMARK": str(env_exports.get("SEEDANCE_WATERMARK") or str(DEFAULT_WATERMARK).lower()),
    }
    for ref_var in required_reference_vars:
        replacement_values[ref_var] = str(env_exports.get(ref_var) or "")

    rendered_text = json.dumps(payload_template, ensure_ascii=False)
    for key, value in replacement_values.items():
        rendered_text = rendered_text.replace(f"__{key}__", value)

    payload = json.loads(rendered_text)
    for bool_key in ("generate_audio", "watermark"):
        if isinstance(payload.get(bool_key), str):
            payload[bool_key] = payload[bool_key].strip().lower() == "true"
    if isinstance(payload.get("duration"), str):
        payload["duration"] = int(payload["duration"])
    return payload


def build_shell_script(
    *,
    scene: ScenePrompt,
    payload_template_name: str,
    payload_rendered_name: str,
    references: list[dict[str, Any]],
) -> str:
    required_vars = json.dumps([ref["env_var"] for ref in references], ensure_ascii=False)
    local_ref_comments: list[str] = []
    ref_requirements: list[str] = []
    for ref in references:
        local_path = ref["local_path"] or "未找到本地素材，请手动补充"
        env_var = ref["env_var"]
        local_ref_comments.append(f"# {env_var} <- {local_path}")
        ref_requirements.append(f': "${{{env_var}:?请先设置 {env_var}，建议先上传本地素材后填入公网 URL：{local_path}}}"')

    local_ref_comments_block = "\n".join(local_ref_comments) if local_ref_comments else "# 当前场景未使用参考图片"
    ref_requirements_block = "\n".join(ref_requirements)

    script = f'''#!/usr/bin/env bash
set -euo pipefail

# 场景：{scene.heading}
# 建议时长：{scene.duration_text or scene.duration_value}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PAYLOAD_TEMPLATE="$SCRIPT_DIR/{payload_template_name}"
PAYLOAD_RENDERED="$SCRIPT_DIR/{payload_rendered_name}"
TASK_URL="{TASK_URL}"
SEEDANCE_URL_ENV="$SCRIPT_DIR/{scene.scene_id}__seedance_api_urls.env"
SUBMIT_RESPONSE_JSON="$SCRIPT_DIR/{scene.scene_id}__seedance_submit_response.json"
POLL_RESPONSE_JSON="$SCRIPT_DIR/{scene.scene_id}__seedance_poll_response.json"
OUTPUT_VIDEO="$SCRIPT_DIR/{scene.scene_id}__seedance_output.mp4"
LAST_FRAME_FILE="$SCRIPT_DIR/{scene.scene_id}__seedance_last_frame.jpg"
TASK_ID_FILE="$SCRIPT_DIR/{scene.scene_id}__seedance_task_id.txt"
CHECK_SCRIPT="$SCRIPT_DIR/{scene.scene_id}__seedance_check_last_task.sh"

if [[ -f "$SEEDANCE_URL_ENV" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$SEEDANCE_URL_ENV"
  set +a
fi

: "${{ARK_API_KEY:?请先设置 ARK_API_KEY，例如 export ARK_API_KEY=...}}"
export SEEDANCE_MODEL="${{SEEDANCE_MODEL:-{DEFAULT_MODEL}}}"
export SEEDANCE_RATIO="${{SEEDANCE_RATIO:-{DEFAULT_RATIO}}}"
export SEEDANCE_RESOLUTION="${{SEEDANCE_RESOLUTION:-{DEFAULT_RESOLUTION}}}"
export SEEDANCE_DURATION="${{SEEDANCE_DURATION:-{scene.duration_value}}}"
export SEEDANCE_GENERATE_AUDIO="${{SEEDANCE_GENERATE_AUDIO:-{str(DEFAULT_GENERATE_AUDIO).lower()}}}"
export SEEDANCE_WATERMARK="${{SEEDANCE_WATERMARK:-{str(DEFAULT_WATERMARK).lower()}}}"
export SEEDANCE_POLL_INTERVAL="${{SEEDANCE_POLL_INTERVAL:-10}}"
export SEEDANCE_MAX_POLLS="${{SEEDANCE_MAX_POLLS:-120}}"
export SEEDANCE_POLL_CONNECT_TIMEOUT="${{SEEDANCE_POLL_CONNECT_TIMEOUT:-10}}"
export SEEDANCE_POLL_REQUEST_TIMEOUT="${{SEEDANCE_POLL_REQUEST_TIMEOUT:-30}}"
export SEEDANCE_MAX_POLL_ERRORS="${{SEEDANCE_MAX_POLL_ERRORS:-20}}"

# 先把下面这些本地素材上传到可公网访问的 URL，再填入对应环境变量。
{local_ref_comments_block}

{ref_requirements_block}

python3 - "$PAYLOAD_TEMPLATE" "$PAYLOAD_RENDERED" <<'PY'
import json, os, sys
template_path, rendered_path = sys.argv[1], sys.argv[2]
text = open(template_path, 'r', encoding='utf-8').read()
required = {required_vars}
for key in required + ['SEEDANCE_MODEL', 'SEEDANCE_RATIO', 'SEEDANCE_RESOLUTION', 'SEEDANCE_DURATION', 'SEEDANCE_GENERATE_AUDIO', 'SEEDANCE_WATERMARK']:
    value = os.environ.get(key)
    if not value:
        raise SystemExit(f'缺少环境变量：{{key}}')
    text = text.replace(f'__{{key}}__', value)
payload = json.loads(text)
for bool_key in ('generate_audio', 'watermark'):
    if isinstance(payload.get(bool_key), str):
        payload[bool_key] = payload[bool_key].strip().lower() == 'true'
if isinstance(payload.get('duration'), str):
    payload['duration'] = int(payload['duration'])
with open(rendered_path, 'w', encoding='utf-8') as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
PY

curl -sS "$TASK_URL" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARK_API_KEY" \
  -d @"$PAYLOAD_RENDERED" \
  -o "$SUBMIT_RESPONSE_JSON"

TASK_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("id", ""))' "$SUBMIT_RESPONSE_JSON")"
if [[ -z "$TASK_ID" ]]; then
  echo "[seedance-api] 未能从提交响应中解析任务 ID" >&2
  cat "$SUBMIT_RESPONSE_JSON" >&2
  exit 1
fi

echo "[seedance-api] 任务已创建：$TASK_ID"
echo "$TASK_ID" > "$TASK_ID_FILE"
cat > "$CHECK_SCRIPT" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
: "${{ARK_API_KEY:?请先设置 ARK_API_KEY，例如 export ARK_API_KEY=...}}"
TASK_ID="$(cat "$SCRIPT_DIR/{scene.scene_id}__seedance_task_id.txt")"
TASK_QUERY_URL="{TASK_URL}/$TASK_ID"
curl -sS "$TASK_QUERY_URL" \
  -H "Authorization: Bearer $ARK_API_KEY" \
  -H "Content-Type: application/json"
echo
EOF
chmod +x "$CHECK_SCRIPT"
MANUAL_QUERY_CMD="bash \"$CHECK_SCRIPT\""
TASK_QUERY_URL="{TASK_URL}/$TASK_ID"

echo "[seedance-api] 如需手动查验任务状态，可执行："
echo "$MANUAL_QUERY_CMD"

POLL_ERRORS=0
for ((i=1; i<=SEEDANCE_MAX_POLLS; i++)); do
  if ! curl -sS --connect-timeout "$SEEDANCE_POLL_CONNECT_TIMEOUT" --max-time "$SEEDANCE_POLL_REQUEST_TIMEOUT" "$TASK_QUERY_URL" \
    -H "Authorization: Bearer $ARK_API_KEY" \
    -H "Content-Type: application/json" \
    -o "$POLL_RESPONSE_JSON"; then
    POLL_ERRORS=$((POLL_ERRORS + 1))
    echo "[seedance-api] 轮询请求超时或失败（连续第 ${{POLL_ERRORS}} 次），${{SEEDANCE_POLL_INTERVAL}}s 后自动重试。" >&2
    if (( POLL_ERRORS >= SEEDANCE_MAX_POLL_ERRORS )); then
      echo "[seedance-api] 连续轮询失败次数过多，请手动查验：" >&2
      echo "$MANUAL_QUERY_CMD" >&2
      exit 1
    fi
    sleep "$SEEDANCE_POLL_INTERVAL"
    continue
  fi
  POLL_ERRORS=0
  STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))' "$POLL_RESPONSE_JSON")"
  echo "[seedance-api] 轮询 $i/$SEEDANCE_MAX_POLLS：$STATUS"
  if [[ "$STATUS" == "succeeded" ]]; then
    VIDEO_URL="$(python3 -c 'import json,sys; payload=json.load(open(sys.argv[1], encoding="utf-8")); print((payload.get("content", {{}}) or {{}}).get("video_url", ""))' "$POLL_RESPONSE_JSON")"
    LAST_FRAME_URL="$(python3 -c 'import json,sys; payload=json.load(open(sys.argv[1], encoding="utf-8")); print((payload.get("content", {{}}) or {{}}).get("last_frame_url", ""))' "$POLL_RESPONSE_JSON")"
    if [[ -n "$VIDEO_URL" ]]; then
      echo "[seedance-api] 视频已生成：$VIDEO_URL"
      curl -L -sS "$VIDEO_URL" -o "$OUTPUT_VIDEO"
      echo "[seedance-api] 视频已下载到：$OUTPUT_VIDEO"
    else
      echo "[seedance-api] 任务已成功，但未找到 video_url，请查看：$POLL_RESPONSE_JSON" >&2
    fi
    if [[ -n "$LAST_FRAME_URL" ]]; then
      curl -L -sS "$LAST_FRAME_URL" -o "$LAST_FRAME_FILE" || true
      echo "[seedance-api] 尾帧已下载到：$LAST_FRAME_FILE"
    fi
    cat "$POLL_RESPONSE_JSON"
    exit 0
  fi
  if [[ "$STATUS" == "failed" || "$STATUS" == "expired" ]]; then
    echo "[seedance-api] 任务失败：$STATUS" >&2
    cat "$POLL_RESPONSE_JSON" >&2
    exit 1
  fi
  sleep "$SEEDANCE_POLL_INTERVAL"
done

echo "[seedance-api] 轮询超时，请手动查看：$POLL_RESPONSE_JSON" >&2
echo "[seedance-api] 也可直接手动查验任务状态：" >&2
echo "$MANUAL_QUERY_CMD" >&2
exit 1
'''
    return script

def build_scene_reference_payload(
    *,
    series_name: str,
    episode_id: str,
    scene: ScenePrompt,
    material_table: dict[str, MaterialReference],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    refs: list[MaterialReference] = []
    for token in scene.reference_tokens:
        ref = material_table.get(token)
        if ref is None:
            token_match = re.search(r"(\d+)", token)
            token_number = int(token_match.group(1)) if token_match else 0
            ref = MaterialReference(token=token, token_number=token_number, material_type="未知素材", label=token)
        refs.append(ref)
    payload = build_reference_payload(series_name, episode_id, refs)
    return limit_reference_payload(payload)


def save_outputs(
    *,
    episode_dir: Path,
    scene: ScenePrompt,
    references: list[dict[str, Any]],
    payload_template: dict[str, Any],
) -> dict[str, str]:
    script_path = episode_dir / f"{scene.scene_id}__seedance_api.sh"
    payload_template_path = episode_dir / f"{scene.scene_id}__seedance_api_payload.template.json"
    payload_rendered_path = episode_dir / f"{scene.scene_id}__seedance_api_payload.rendered.json"
    references_path = episode_dir / f"{scene.scene_id}__seedance_api_references.json"
    env_path = episode_dir / f"{scene.scene_id}__seedance_api_urls.env"

    payload_template_path.write_text(json.dumps(payload_template, ensure_ascii=False, indent=2), encoding="utf-8")
    references_path.write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8")
    script_text = build_shell_script(
        scene=scene,
        payload_template_name=payload_template_path.name,
        payload_rendered_name=payload_rendered_path.name,
        references=references,
    )
    script_path.write_text(script_text, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    rendered_payload = render_local_payload(
        scene=scene,
        references=references,
        payload_template=payload_template,
        env_exports=load_env_exports(env_path),
    )
    if rendered_payload is None:
        if payload_rendered_path.exists():
            payload_rendered_path.unlink()
    else:
        payload_rendered_path.write_text(json.dumps(rendered_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "script_path": str(script_path),
        "payload_template_path": str(payload_template_path),
        "payload_rendered_path": str(payload_rendered_path),
        "references_path": str(references_path),
    }


def main() -> None:
    series_dirs = list_series_dirs()
    if not series_dirs:
        raise RuntimeError("outputs/ 下没有找到可用的 02-seedance-prompts.md。")

    series_idx = choose_from_list(
        "请选择要生成 Seedance API 调用代码的剧本：",
        [path.name for path in series_dirs],
        default_index=0,
    )
    series_dir = series_dirs[series_idx]
    series_name = series_dir.name

    episode_dirs = sorted(
        [path for path in series_dir.iterdir() if path.is_dir() and find_storyboard_path(path) is not None],
        key=lambda item: episode_sort_key(item.name),
    )
    if not episode_dirs:
        raise RuntimeError(f"{series_name} 下没有找到可用集数。")

    episode_idx = choose_from_list(
        f"请选择 {series_name} 的集数：",
        [path.name for path in episode_dirs],
        default_index=0,
    )
    episode_dir = episode_dirs[episode_idx]
    episode_id = episode_dir.name
    storyboard_path = find_storyboard_path(episode_dir)
    if storyboard_path is None:
        raise RuntimeError(f"{episode_dir} 下没有找到可用的 02-seedance-prompts*.md。")
    existing_workflow = load_reference_workflow_config(episode_dir)
    default_reference_mode = str(existing_workflow.get("reference_mode") or "tos").strip().lower() or "tos"
    reference_mode = choose_reference_mode(default_mode=default_reference_mode)

    scenes = parse_scene_prompts(storyboard_path)
    if not scenes:
        raise RuntimeError(f"未从 {storyboard_path} 解析到任何 Pxx 场景。")

    start_idx, end_idx = choose_range_from_list(
        f"请选择 {series_name} {episode_id} 要生成的场景范围：",
        [scene.heading for scene in scenes],
        default_start=0,
        default_end=0,
    )
    selected_scenes = scenes[start_idx : end_idx + 1]
    workflow_path = save_reference_workflow_config(
        episode_dir=episode_dir,
        series_name=series_name,
        episode_id=episode_id,
        reference_mode=reference_mode,
        selected_scene_ids=[scene.scene_id for scene in selected_scenes],
    )
    print_status(f"引用模式已保存：{workflow_path} -> {reference_mode}")

    material_table = parse_material_table(storyboard_path)
    generated: list[dict[str, Any]] = []
    for scene in selected_scenes:
        reference_payload, dropped_references = build_scene_reference_payload(
            series_name=series_name,
            episode_id=episode_id,
            scene=scene,
            material_table=material_table,
        )
        payload_template = render_payload_template(scene, reference_payload)
        saved_paths = save_outputs(
            episode_dir=episode_dir,
            scene=scene,
            references=reference_payload,
            payload_template=payload_template,
        )
        summary = {
            "series_name": series_name,
            "episode_id": episode_id,
            "reference_mode": reference_mode,
            "scene_id": scene.scene_id,
            "scene_heading": scene.heading,
            "storyboard_path": str(storyboard_path),
            "references": reference_payload,
            "dropped_references": dropped_references,
            **saved_paths,
        }
        generated.append(summary)
        print_status(f"已生成场景调用脚本：{saved_paths['script_path']}")
        print_status(f"引用清单：{saved_paths['references_path']}")
        if dropped_references:
            print_status(
                f"{scene.scene_id} 超出 Seedance 参考图上限 {MAX_REFERENCE_IMAGES}，已裁剪 {len(dropped_references)} 张："
                + "、".join(str(item.get("label", "")).strip() for item in dropped_references)
            )

    print(json.dumps({
        "series_name": series_name,
        "episode_id": episode_id,
        "reference_mode": reference_mode,
        "reference_workflow_config_path": str(workflow_path),
        "scene_range": {
            "start": selected_scenes[0].scene_id,
            "end": selected_scenes[-1].scene_id,
            "count": len(selected_scenes),
        },
        "generated": generated,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
