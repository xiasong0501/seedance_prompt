from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_telemetry import TelemetryRecorder, apply_provider_usage, telemetry_span
from providers.base import save_json_file
from series_paths import SeriesPaths, build_series_paths


DEFAULT_CONFIG_PATH = Path("config/nano_banana_assets.local.json")
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 520}
MIN_REFERENCE_IMAGE_DIMENSION = 300


@dataclass
class PromptItem:
    prompt_type: str
    name: str
    prompt: str
    source_path: Path
    source_label: str
    episode_id: str | None = None
    entity_name: str | None = None
    variant_label: str | None = None


@dataclass
class GeneratedAssetRecord:
    prompt_type: str
    source_label: str
    name: str
    output_path: Path
    episode_id: str | None
    entity_name: str | None = None
    variant_label: str | None = None


@dataclass
class SceneMaterialReference:
    reference_number: int
    reference_token: str
    scene_name: str
    grid_title: str
    panel_number: int
    panel_label: str


class ImageSafetyBlockedError(RuntimeError):
    def __init__(self, *, finish_reason: str, finish_message: str, response: dict[str, Any]) -> None:
        super().__init__(finish_message or finish_reason or "图片生成被安全策略拦截。")
        self.finish_reason = finish_reason
        self.finish_message = finish_message
        self.response = response


def print_status(message: str) -> None:
    print(f"[nano-banana] {message}", flush=True)


def is_generic_invalid_argument_error(exc: BaseException) -> bool:
    message = str(exc)
    return (
        "状态码 400" in message
        and "INVALID_ARGUMENT" in message
        and "Request contains an invalid argument." in message
    )


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def request_json(
    *,
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: int = 300,
    max_attempts: int = 3,
    retry_base_delay_seconds: float = 1.5,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    attempts = max(1, int(max_attempts))
    retry_base = max(0.2, float(retry_base_delay_seconds))
    last_error: RuntimeError | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            url=url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(
                f"Gemini 图片生成失败，状态码 {exc.code}（第 {attempt}/{attempts} 次），响应：{body}"
            )
            if exc.code in RETRYABLE_HTTP_STATUS_CODES and attempt < attempts:
                wait_seconds = min(8.0, retry_base * attempt)
                print_status(
                    f"Gemini 请求瞬时失败（HTTP {exc.code}，第 {attempt}/{attempts} 次），"
                    f"{wait_seconds:.1f}s 后自动重试。"
                )
                time.sleep(wait_seconds)
                continue
            raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = RuntimeError(
                f"Gemini 图片生成网络失败（第 {attempt}/{attempts} 次）：{exc}"
            )
            if attempt < attempts:
                wait_seconds = min(8.0, retry_base * attempt)
                print_status(
                    f"Gemini 网络请求异常（第 {attempt}/{attempts} 次），{wait_seconds:.1f}s 后自动重试。"
                )
                time.sleep(wait_seconds)
                continue
            raise last_error from exc
    if last_error:
        raise last_error
    raise RuntimeError("Gemini 图片生成失败：未知错误。")


def slugify(raw: str) -> str:
    clean = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", raw.strip())
    clean = re.sub(r"-{2,}", "-", clean).strip("-")
    return clean or "untitled"


def normalize_spaces(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip()


NEGATED_STYLIZATION_PATTERNS = (
    r"非\s*插画",
    r"非\s*二次元",
    r"非\s*卡通",
    r"非\s*动漫",
    r"非\s*漫画",
    r"非\s*赛璐璐",
    r"非\s*cg(?:卡通渲染|渲染)?",
    r"避免\s*动漫",
    r"避免\s*二次元",
    r"避免\s*漫画",
    r"避免\s*插画",
    r"避免\s*卡通(?:化)?",
    r"避免\s*手绘",
    r"避免\s*赛璐璐",
    r"绝对不要生成\s*动漫",
    r"绝对不要\s*动漫",
    r"不要生成\s*动漫",
    r"不要\s*动漫",
)


def strip_negated_stylization_phrases(raw: str) -> str:
    text = normalize_spaces(raw or "").lower()
    for pattern in NEGATED_STYLIZATION_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return normalize_spaces(text)


FICTIONAL_REALISTIC_PATTERNS = (
    r"虚构人物",
    r"非\s*真人(?:照片|写真|肖像|演员|明星|面孔|脸)",
    r"不(?:联想|对应|参考)任何真实(?:演员|名人|人物)",
    r"非\s*照片感棚拍",
    r"非\s*棚拍(?:感)?",
)


def strip_fictional_realistic_phrases(raw: str) -> str:
    text = normalize_spaces(raw or "").lower()
    for pattern in FICTIONAL_REALISTIC_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return normalize_spaces(text)


def normalize_frame_orientation(raw: str | None) -> str:
    text = normalize_spaces(raw or "")
    if not text:
        return ""
    lowered = text.lower().replace("：", ":").replace("×", "x")
    if "竖" in text or "portrait" in lowered:
        return "portrait"
    if "横" in text or "landscape" in lowered:
        return "landscape"
    ratio_match = re.search(r"(\d+(?:\.\d+)?)\s*[:x/]\s*(\d+(?:\.\d+)?)", lowered)
    if ratio_match:
        left = float(ratio_match.group(1))
        right = float(ratio_match.group(2))
        if left > 0 and right > 0:
            return "portrait" if left < right else "landscape"
    return ""


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


def output_episode_dir_candidates(series_name: str, episode_id: str | None) -> list[Path]:
    episode_fragment = episode_id or ""
    names = [f"{series_name}-gpt", series_name]
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return [PROJECT_ROOT / "outputs" / name / episode_fragment for name in deduped]


def find_storyboard_prompt_path(episode_dir: Path) -> Path | None:
    exact_path = episode_dir / "02-seedance-prompts.md"
    if exact_path.exists():
        return exact_path.resolve()

    candidates = [
        path for path in episode_dir.glob("02-seedance-prompts*.md")
        if path.is_file() and not path.name.endswith(".report.md")
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return candidates[-1].resolve()


def resolve_default_storyboard_prompt_path(series_name: str, episode_id: str | None) -> Path:
    episode_dirs = output_episode_dir_candidates(series_name, episode_id)
    for episode_dir in episode_dirs:
        candidate = find_storyboard_prompt_path(episode_dir)
        if candidate is not None:
            return candidate
    return (episode_dirs[0] / "02-seedance-prompts.md").resolve()


def assets_root_from_source_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    try:
        resolved = path.expanduser().resolve()
    except Exception:
        return None
    current = resolved if resolved.is_dir() else resolved.parent
    while True:
        parent = current.parent
        if parent.name == "assets":
            return current.resolve()
        if parent == current:
            return None
        current = parent


def resolve_effective_assets_dir(paths: SeriesPaths, *source_paths: Path) -> Path:
    for source_path in source_paths:
        candidate = assets_root_from_source_path(source_path)
        if candidate is not None and candidate.exists():
            return candidate
    return paths.assets_dir


NON_REAL_STYLE_KEYWORDS = (
    "漫剧",
    "动画",
    "国漫",
    "动漫",
    "二次元",
    "漫画",
    "插画",
    "赛璐璐",
    "非真实",
    "3D动画",
    "绘本",
    "水墨",
    "卡通",
    "避免真人",
)

REALISTIC_STYLE_KEYWORDS = (
    "真人",
    "写实",
    "摄影",
    "棚拍",
    "电视剧",
)

# 当提示词中出现这些“反动漫/真人实拍”表达时，应优先尊重提示词，
# 不被上游 target_medium=漫剧 强制拉回非真人风格。
REALISTIC_PROMPT_OVERRIDE_KEYWORDS = (
    "真人电影级",
    "真人影视",
    "电影节",
    "实拍质感",
    "摄影级",
    "真实皮肤纹理",
    "非插画",
    "非二次元",
    "非cg",
    "非卡通",
    "避免动漫",
    "避免二次元",
    "避免插画",
    "绝对不要生成动漫",
)

NON_REAL_PROMPT_OVERRIDE_KEYWORDS = (
    "非摄影",
    "国漫",
    "漫剧",
    "动漫",
    "二次元",
    "漫画",
    "赛璐璐",
    "卡通",
    "q版",
    "插画风",
)


def split_character_title(title: str) -> tuple[str, str]:
    normalized = normalize_spaces(title)
    for delimiter in ("｜", "|", "·"):
        if delimiter in normalized:
            left, right = normalized.split(delimiter, 1)
            left = left.strip()
            right = right.strip()
            if left and right:
                return left, right
    parenthetical = re.match(r"^(?P<name>.+?)[（(](?P<variant>.+?)[）)]$", normalized)
    if parenthetical:
        return parenthetical.group("name").strip(), parenthetical.group("variant").strip()
    return normalized, "基础造型"


def load_runtime_style_hints(paths: SeriesPaths) -> dict[str, str | None]:
    hints: dict[str, str | None] = {
        "target_medium": None,
        "visual_style": None,
        "frame_orientation": None,
        "source": None,
    }
    for episode_dir in output_episode_dir_candidates(paths.series_name, paths.episode_id):
        for candidate in sorted(episode_dir.glob("01-director-analysis__*.json")):
            try:
                payload = load_json(candidate)
            except Exception:
                continue
            target_medium = normalize_spaces(str(payload.get("target_medium") or "")) or None
            visual_style = normalize_spaces(str(payload.get("visual_style") or "")) or None
            frame_orientation = normalize_spaces(str(payload.get("frame_orientation") or "")) or None
            if target_medium or visual_style or frame_orientation:
                hints["target_medium"] = target_medium
                hints["visual_style"] = visual_style
                hints["frame_orientation"] = frame_orientation
                hints["source"] = str(candidate)
                return hints
        director_md = episode_dir / "01-director-analysis.md"
        if director_md.exists():
            try:
                text = director_md.read_text(encoding="utf-8")
            except Exception:
                text = ""
            medium_match = re.search(r"\*\*目标媒介\*\*：(.+)", text)
            style_match = re.search(r"\*\*视觉风格\*\*：(.+)", text)
            orientation_match = re.search(r"\*\*目标画幅\*\*：(.+)", text)
            target_medium = normalize_spaces(medium_match.group(1)) if medium_match else None
            visual_style = normalize_spaces(style_match.group(1)) if style_match else None
            frame_orientation = normalize_spaces(orientation_match.group(1)) if orientation_match else None
            if target_medium or visual_style or frame_orientation:
                hints["target_medium"] = target_medium
                hints["visual_style"] = visual_style
                hints["frame_orientation"] = frame_orientation
                hints["source"] = str(director_md)
                return hints
    bundle_json = PROJECT_ROOT / "analysis" / paths.series_name / "openai_agent_flow" / "genre_reference_bundle.json"
    if bundle_json.exists():
        try:
            payload = load_json(bundle_json)
        except Exception:
            payload = {}
        target_medium = normalize_spaces(str(payload.get("target_medium") or "")) or None
        visual_style = normalize_spaces(str(payload.get("visual_style") or "")) or None
        frame_orientation = normalize_spaces(str(payload.get("frame_orientation") or "")) or None
        if target_medium or visual_style or frame_orientation:
            hints["target_medium"] = target_medium
            hints["visual_style"] = visual_style
            hints["frame_orientation"] = frame_orientation
            hints["source"] = str(bundle_json)
    return hints


def infer_character_style_mode(
    *,
    prompt: str,
    config: dict[str, Any],
    target_medium: str | None = None,
    visual_style: str | None = None,
) -> str:
    explicit = str(config.get("quality", {}).get("character_style_mode", "auto") or "auto").strip().lower()
    if explicit in {"non_real", "stylized", "animation", "illustration", "manga"}:
        return "non_real"
    if explicit in {"realistic", "photo", "film"}:
        return "realistic"
    prompt_text = normalize_spaces(prompt or "")
    prompt_text_lower = prompt_text.lower()
    has_realistic_override = any(keyword in prompt_text_lower for keyword in REALISTIC_PROMPT_OVERRIDE_KEYWORDS)
    sanitized_prompt_text = strip_fictional_realistic_phrases(strip_negated_stylization_phrases(prompt_text))
    has_non_real_override = any(keyword in sanitized_prompt_text for keyword in NON_REAL_PROMPT_OVERRIDE_KEYWORDS)
    if has_realistic_override and not has_non_real_override:
        return "realistic"
    if has_non_real_override and not has_realistic_override:
        return "non_real"
    target_medium_text = normalize_spaces(target_medium or "")
    if any(keyword in target_medium_text for keyword in ("漫剧", "动画", "国漫", "动漫", "插画", "绘本", "卡通", "3D动画")):
        return "non_real"
    if any(keyword in target_medium_text for keyword in ("电视剧", "真人秀", "纪实", "电影", "真人电影", "短片")):
        return "realistic"
    hint_text = " ".join(part for part in (target_medium or "", visual_style or "", sanitized_prompt_text) if part)
    sanitized_hint_text = strip_fictional_realistic_phrases(strip_negated_stylization_phrases(hint_text))
    if any(keyword in sanitized_hint_text for keyword in NON_REAL_STYLE_KEYWORDS):
        return "non_real"
    if any(keyword in hint_text for keyword in REALISTIC_STYLE_KEYWORDS):
        return "realistic"
    return "realistic"


def normalize_character_prompt_for_style(prompt: str, *, style_mode: str) -> str:
    text = prompt.strip()
    if style_mode != "non_real":
        return text
    replacements = [
        ("统一真人影视写实动漫风格设定图体系", "统一高完成度仿真人漫剧/国漫电影感角色设定图体系"),
        ("真人影视写实动漫风格设定图体系", "高完成度仿真人漫剧/国漫电影感角色设定图体系"),
        ("统一真人影视写实角色设定图体系", "统一高完成度仿真人漫剧/国漫电影感角色设定图体系"),
        ("真人影视写实角色设定图体系", "高完成度仿真人漫剧/国漫电影感角色设定图体系"),
        ("真人影视写实动漫风格", "高完成度仿真人漫剧/国漫电影感风格"),
        ("真人影视写实角色设定图", "高完成度仿真人漫剧/国漫电影感角色设定图"),
        ("真人影视写实", "非真人仿真人漫剧/国漫电影感"),
        ("真人影视", "非真人漫剧/国漫电影感"),
        ("真实中国演员气质，", ""),
        ("真实摄影级皮肤质感，", ""),
        ("统一柔和棚拍布光", "统一具有电影质感的风格化设定图布光"),
        ("统一棚拍设定图逻辑", "统一具有电影镜头感的风格化设定图逻辑"),
        ("统一镜头与布光逻辑", "统一具有电影镜头感的风格化设定图布光逻辑"),
        ("统一设定图布光", "统一具有电影质感的风格化设定图布光"),
        ("不联想任何真实演员或名人", "不对应任何真实人物"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    text = re.sub(r"真人(演员|明星)脸", "真实人物脸", text)
    text = re.sub(r"照片感", "真实质感", text)
    return text


def build_character_style_enforcement(*, style_mode: str) -> tuple[str, str]:
    if style_mode == "non_real":
        prompt_suffix = (
            "\n\n角色图统一风格要求：必须是同一套非真人、非摄影的高完成度仿真人漫剧/国漫电影感角色设定图体系，"
            "统一白底设定图版式、统一近似真人比例的人物结构、统一细腻五官刻画、统一布料与饰品质感、统一电影化布光与镜头审美。"
            "风格应接近高完成度国漫剧集或动画电影设定图：细节丰富、层次细腻、质感克制、画风成熟，但绝对不要生成真实演员脸、"
            "照片感皮肤、摄影棚拍人像或任何真人写实效果。禁止低幼卡通、Q版、廉价二次元平涂或与本剧不一致的插画风。"
        )
        negative_suffix = "避免真人、避免摄影、避免真实演员脸、避免照片感、避免皮肤毛孔摄影质感、避免超写实人像、避免低幼卡通、避免Q版、避免廉价二次元平涂、避免夸张漫画脸"
        return prompt_suffix, negative_suffix
    prompt_suffix = (
        "\n\n角色图统一风格要求：必须是同一套真人影视写实角色设定图体系，真实摄影级皮肤质感，真实中国演员气质，"
        "统一的白底定妆资料页版式与商业电影美术风格。即使是多视角拼版，也必须表现为同一位真人演员的多张真实定妆照片，"
        "而不是插画角色设定卡。绝对不要生成动漫、漫画、二次元、手绘、赛璐璐、游戏立绘或插画风。"
        "不要在画面中添加标题、中文属性文字、说明标签、信息面板、正面/侧面/背面字样或任何排版文字。"
        "同一剧所有角色都必须像出自同一个真人影视项目，而不是不同风格的素材拼盘。"
    )
    negative_suffix = "避免动漫、避免二次元、避免漫画、避免插画、避免手绘、避免赛璐璐、避免卡通化、避免游戏立绘感、避免角色设定卡文字、避免标题、避免中文标签、避免说明面板、避免正面侧面背面文字"
    return prompt_suffix, negative_suffix


SCENE_PERSON_LEAK_PATTERNS = (
    "人物",
    "主角",
    "人脸",
    "面部",
    "背影",
    "剪影",
    "身影",
    "轮廓",
    "下人",
    "群像",
    "人群",
    "孩童",
    "孩子",
    "少女",
    "妇人",
    "跪姿",
    "衣摆",
    "衣袖",
    "手腕",
    "双手",
    "小手",
    "一只手",
    "手势",
    "表情",
    "对峙",
)

SCENE_SENSITIVE_TEXT_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(迫击炮阵地)"), "远程重型器械阵地"),
    (re.compile(r"(迫击炮)"), "远程重型器械"),
    (re.compile(r"(弹药木箱|弹药箱)"), "补给木箱"),
    (re.compile(r"(火力覆盖后的状态)"), "高压余波后的状态"),
    (re.compile(r"(火力状态)"), "高压余波状态"),
    (re.compile(r"(火力三层环境反馈)"), "三层环境反馈"),
    (re.compile(r"(现代火力体系)"), "现代机械体系"),
    (re.compile(r"(现代火力异物感)"), "现代机械异物感"),
    (re.compile(r"(远程杀机)"), "远程压迫感"),
    (re.compile(r"(公开处刑感)"), "公开审判式压迫感"),
    (re.compile(r"(攻击过程)"), "冲击过程"),
    (re.compile(r"(炮位)"), "器械位"),
    (re.compile(r"(炮口)"), "器械前端"),
    (re.compile(r"(爆闪)"), "强光余辉"),
    (re.compile(r"(焦黑战损)"), "焦黑磨损"),
    (re.compile(r"(武器持握痕迹)"), "可识别人物活动痕迹"),
]


def normalize_scene_prompt_for_background(prompt: str) -> str:
    cleaned = prompt.strip()
    replacements = [
        ("人物动线痕迹", "空间层次与道具摆位"),
        ("人物表演", "环境静态信息"),
        ("如出现人物，仅允许极远景、背影、剪影、虚化或极小比例群众。", "绝对不要出现任何人物。"),
        ("如出现人物，仅允许极远景、背影、虚化、极小比例群众。", "绝对不要出现任何人物。"),
        ("如需出现人物，只能是远景、背影、虚化或极小比例群众，不能出现清晰人脸或可识别主角。", "绝对不要出现任何人物、人体局部或动作痕迹。"),
        ("不把人物作为视觉中心", "不出现人物"),
    ]
    for old, new in replacements:
        cleaned = cleaned.replace(old, new)
    for pattern, replacement in SCENE_SENSITIVE_TEXT_REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)

    sanitized_lines = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            sanitized_lines.append(raw_line)
            continue
        if line.startswith(("## ", "### ", "---", "<!--")):
            sanitized_lines.append(raw_line)
            continue
        if line.startswith("格") and "【" in line:
            sanitized_lines.append(raw_line)
            continue
        parts = re.split(r"(?<=[。！？；])", raw_line)
        kept_parts = []
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            if any(token in stripped for token in SCENE_PERSON_LEAK_PATTERNS):
                continue
            kept_parts.append(stripped)
        if kept_parts:
            sanitized_lines.append("".join(kept_parts))

    result = "\n".join(sanitized_lines)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    result += (
        "\n\n纯场景硬性要求：整张图只允许出现静态环境、静态陈设、静态道具、建筑结构、地面纹理、"
        "窗门帷帐、灯火、器物、植物、天气和光影层次。绝对不要出现任何人物、人体局部、肢体动作、"
        "衣摆掠过、手势关系、走位痕迹、对视关系、递交动作、跪姿、背影、剪影或可被理解为角色行为的叙事线索。"
    )
    return result


def stable_item_key(prompt_type: str, source_label: str) -> str:
    return f"{prompt_type}::{normalize_spaces(source_label)}"


def variant_priority(label: str | None) -> tuple[int, int]:
    raw = normalize_spaces(label or "")
    if not raw:
        return (9, 0)
    priorities = [
        ("首集基础", 0),
        ("基础主形象", 0),
        ("基础", 1),
        ("主形象", 1),
        ("主设", 1),
        ("标准", 2),
    ]
    for keyword, priority in priorities:
        if keyword in raw:
            return (priority, len(raw))
    return (5, len(raw))


def file_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def infer_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def parse_episode_blocks(text: str) -> list[tuple[str | None, str]]:
    start_pattern = re.compile(r"<!--\s*episode:\s*(ep\d+)\s+start\s*-->", flags=re.IGNORECASE)
    matches = list(start_pattern.finditer(text))
    if not matches:
        return [(None, text)]
    blocks: list[tuple[str | None, str]] = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        blocks.append((None, text[: matches[0].start()]))
    for index, match in enumerate(matches):
        episode_id = normalize_episode_key(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end]
        end_pattern = re.compile(
            rf"<!--\s*episode:\s*{re.escape(episode_id or '')}\s+end\s*-->",
            flags=re.IGNORECASE,
        )
        body = end_pattern.sub("", body)
        blocks.append((episode_id, body))
    return blocks


def parse_character_prompts(path: Path) -> list[PromptItem]:
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    items: list[PromptItem] = []
    for episode_id, block in parse_episode_blocks(text):
        sections = re.split(r"^##\s+", block, flags=re.MULTILINE)
        for section in sections[1:]:
            lines = section.splitlines()
            if not lines:
                continue
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            match = re.search(r"\*\*提示词\*\*：\s*\n(?P<prompt>.*)", body, flags=re.DOTALL)
            if not match:
                continue
            prompt = match.group("prompt").strip()
            prompt = re.split(r"\n---+\n", prompt)[0].strip()
            name, variant_label = split_character_title(title)
            items.append(
                PromptItem(
                    prompt_type="character",
                    name=name,
                    prompt=prompt,
                    source_path=path,
                    source_label=title,
                    episode_id=episode_id,
                    entity_name=name,
                    variant_label=variant_label,
                )
            )
    return items


def parse_scene_prompts(path: Path) -> list[PromptItem]:
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    items: list[PromptItem] = []
    for episode_id, block in parse_episode_blocks(text):
        sections = re.split(r"^##\s+", block, flags=re.MULTILINE)
        for section in sections[1:]:
            lines = section.splitlines()
            if not lines:
                continue
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            if not body:
                continue
            body = re.split(r"\n---+\n", body)[0].strip()
            items.append(
                PromptItem(
                    prompt_type="scene",
                    name=title,
                    prompt=body,
                    source_path=path,
                    source_label=title,
                    episode_id=episode_id,
                    entity_name=title,
                    variant_label=title,
                )
            )
    return items




def extract_scene_grid_layout(prompt: str) -> tuple[int, int] | None:
    match = re.search(r"请生成一张\s*(\d+)\s*[×xX*]\s*(\d+)", prompt)
    if not match:
        match = re.search(r"(\d+)\s*[×xX*]\s*(\d+)", prompt)
    if not match:
        return None
    rows = int(match.group(1))
    cols = int(match.group(2))
    if rows <= 0 or cols <= 0:
        return None
    return rows, cols


def parse_scene_layout_override(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            rows = int(value[0])
            cols = int(value[1])
        except (TypeError, ValueError):
            return None
        return (rows, cols) if rows > 0 and cols > 0 else None
    raw = str(value or '').strip().lower().replace('*', 'x').replace('×', 'x')
    match = re.fullmatch(r'(\d+)x(\d+)', raw)
    if not match:
        return None
    rows = int(match.group(1))
    cols = int(match.group(2))
    return (rows, cols) if rows > 0 and cols > 0 else None


def apply_scene_layout_overrides(
    *,
    scene_layouts: dict[str, tuple[int, int] | None],
    config: dict[str, Any],
    episode_id: str | None,
) -> dict[str, tuple[int, int] | None]:
    overrides_root = config.get('selection', {}).get('scene_layout_overrides') or {}
    if not isinstance(overrides_root, dict):
        return scene_layouts
    episode_key = normalize_episode_key(episode_id) or str(episode_id or '').strip()
    episode_overrides = overrides_root.get(episode_key) or overrides_root.get(str(episode_id or '').strip()) or {}
    if not isinstance(episode_overrides, dict):
        return scene_layouts
    updated = dict(scene_layouts)
    for source_label, override_value in episode_overrides.items():
        parsed = parse_scene_layout_override(override_value)
        if parsed is not None:
            updated[str(source_label)] = parsed
    return updated


def scene_expected_canvas_ratio(
    *,
    layout: tuple[int, int] | None,
    frame_mode: str,
) -> float | None:
    if layout is None:
        return None
    rows, cols = layout
    if rows <= 0 or cols <= 0:
        return None
    panel_ratio = 9.0 / 16.0 if frame_mode == "portrait" else 16.0 / 9.0
    return (cols / rows) * panel_ratio


def build_scene_canvas_guidance(
    *,
    layout: tuple[int, int] | None,
    frame_mode: str,
) -> str:
    expected_ratio = scene_expected_canvas_ratio(layout=layout, frame_mode=frame_mode)
    if frame_mode == "landscape":
        if expected_ratio is None:
            return "整张宫格图必须是横版总画布（宽大于高），不要生成竖版长图。"
        return (
            f"整张宫格图必须是横版总画布（宽大于高），总画布宽高比尽量靠近 {expected_ratio:.2f}，"
            "确保分格后仍是统一横屏空间。"
        )
    if expected_ratio is None:
        return "整张宫格图必须是竖版总画布（高大于宽），不要生成横版长条。"
    return (
        f"整张宫格图必须是竖版总画布（高大于宽），总画布宽高比尽量靠近 {expected_ratio:.2f}，"
        "确保分格后每格仍保留稳定的9:16竖向空间。"
    )


def resolve_scene_size_hint(
    *,
    configured_hint: str | None,
    frame_mode: str,
    layout: tuple[int, int] | None,
) -> str:
    hint = str(configured_hint or "").strip()
    if hint and hint.lower() not in {"auto", "自动"}:
        return hint
    expected_ratio = scene_expected_canvas_ratio(layout=layout, frame_mode=frame_mode)
    if frame_mode == "landscape":
        if expected_ratio is None:
            return "横版总画布（宽大于高），适合横屏宫格场景展示"
        return f"横版总画布（宽大于高），建议整体宽高比约 {expected_ratio:.2f}，适合宫格场景展示"
    if expected_ratio is None:
        return "竖版总画布（高大于宽），适合9:16竖屏宫格场景展示"
    return f"竖版总画布（高大于宽），建议整体宽高比约 {expected_ratio:.2f}，适合9:16竖屏宫格场景展示"


def inspect_image_geometry(image_bytes: bytes) -> tuple[int, int, float]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
    ratio = (width / height) if height else 0.0
    return width, height, ratio


def validate_scene_canvas(
    *,
    image_bytes: bytes,
    layout: tuple[int, int] | None,
    frame_mode: str,
) -> dict[str, Any]:
    width, height, ratio = inspect_image_geometry(image_bytes)
    expected_ratio = scene_expected_canvas_ratio(layout=layout, frame_mode=frame_mode)
    orientation_ok = width > height if frame_mode == "landscape" else height > width
    ratio_ok = True
    if expected_ratio is not None and expected_ratio > 0:
        ratio_ok = abs(ratio - expected_ratio) / expected_ratio <= 0.45
    return {
        "width": width,
        "height": height,
        "actual_ratio": round(ratio, 4),
        "expected_ratio": round(expected_ratio, 4) if expected_ratio is not None else None,
        "frame_mode": frame_mode,
        "orientation_ok": orientation_ok,
        "ratio_ok": ratio_ok,
        "ok": orientation_ok and ratio_ok,
    }


def validate_scene_canvas_from_file(
    *,
    image_path: Path,
    layout: tuple[int, int] | None,
    frame_mode: str,
) -> dict[str, Any] | None:
    try:
        image_bytes = image_path.read_bytes()
    except Exception:
        return None
    return validate_scene_canvas(
        image_bytes=image_bytes,
        layout=layout,
        frame_mode=frame_mode,
    )


def parse_seedance_scene_materials(path: Path) -> list[SceneMaterialReference]:
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    table_match = re.search(r"^##\s+素材对应表\s*$", text, flags=re.MULTILINE)
    if not table_match:
        return []
    remainder = text[table_match.end():]
    next_section = re.search(r"^##\s+", remainder, flags=re.MULTILINE)
    table_block = remainder[: next_section.start()] if next_section else remainder

    refs: list[SceneMaterialReference] = []
    for raw_line in table_block.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 3:
            continue
        if parts[0] == "引用编号" or set(parts[0].replace("-", "")) == {""}:
            continue
        ref_match = re.search(r"@图片(\d+)", parts[0])
        if not ref_match:
            continue
        material_type = parts[1]
        if material_type != "场景参考":
            continue
        material = parts[2]
        material_core = re.split(r"[，,]\s*参见\s+", material, maxsplit=1)[0].strip()
        material_match = None
        for pattern in (
            r"^(?P<scene_name>.+?)[（(](?:(?P<grid_title>.+?)\s*)?格(?P<panel_number>\d+)[）)]\s*$",
            r"^(?P<scene_name>.+?)\s*(?:——|--|—|-)\s*(?:(?P<grid_title>.+?)\s*)?格(?P<panel_number>\d+)(?:\s*[（(].*?[）)])?\s*$",
        ):
            material_match = re.match(pattern, material_core)
            if material_match:
                break
        if not material_match:
            continue
        grid_title = (material_match.group("grid_title") or "").strip()
        refs.append(
            SceneMaterialReference(
                reference_number=int(ref_match.group(1)),
                reference_token=f"@图片{int(ref_match.group(1))}",
                scene_name=material_match.group("scene_name").strip(),
                grid_title=grid_title,
                panel_number=int(material_match.group("panel_number")),
                panel_label=f"格{int(material_match.group('panel_number'))}",
            )
        )
    refs.sort(key=lambda item: item.reference_number)
    return refs


def resolve_scene_grid_target(
    *,
    grid_title: str,
    scene_output_paths: dict[str, Path],
    scene_layouts: dict[str, tuple[int, int] | None],
) -> tuple[Path | None, tuple[int, int] | None]:
    grid_path = scene_output_paths.get(grid_title)
    layout = scene_layouts.get(grid_title)
    if grid_path is not None and layout is not None:
        return grid_path, layout

    normalized = normalize_spaces(grid_title)
    for key, candidate_path in scene_output_paths.items():
        key_normalized = normalize_spaces(key)
        if normalized and (normalized in key_normalized or key_normalized in normalized):
            candidate_layout = scene_layouts.get(key)
            if candidate_layout is not None:
                return candidate_path, candidate_layout

    if len(scene_output_paths) == 1:
        only_key, only_path = next(iter(scene_output_paths.items()))
        return only_path, scene_layouts.get(only_key)

    return grid_path, layout


def crop_scene_panel(*, image_path: Path, rows: int, cols: int, panel_number: int, output_path: Path) -> dict[str, Any]:
    if panel_number <= 0 or panel_number > rows * cols:
        raise ValueError(f"场景宫格切分失败：panel_number={panel_number} 超出 {rows}x{cols} 布局范围。")

    with Image.open(image_path) as image:
        width, height = image.size
        row_index = (panel_number - 1) // cols
        col_index = (panel_number - 1) % cols
        left = round(col_index * width / cols)
        right = round((col_index + 1) * width / cols)
        top = round(row_index * height / rows)
        bottom = round((row_index + 1) * height / rows)
        cropped = image.crop((left, top, right, bottom))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image = cropped
        resized_for_min_dimension = False
        if min(cropped.size) < MIN_REFERENCE_IMAGE_DIMENSION:
            scale = max(
                MIN_REFERENCE_IMAGE_DIMENSION / max(1, cropped.size[0]),
                MIN_REFERENCE_IMAGE_DIMENSION / max(1, cropped.size[1]),
            )
            resized_width = max(MIN_REFERENCE_IMAGE_DIMENSION, round(cropped.size[0] * scale))
            resized_height = max(MIN_REFERENCE_IMAGE_DIMENSION, round(cropped.size[1] * scale))
            save_image = cropped.resize((resized_width, resized_height), Image.LANCZOS)
            resized_for_min_dimension = True
        if output_path.suffix.lower() in {".jpg", ".jpeg"} and cropped.mode not in {"RGB", "L"}:
            save_image = save_image.convert("RGB")
        save_image.save(output_path)
        crop_width, crop_height = save_image.size

    return {
        "source_size": [width, height],
        "crop_box": [left, top, right, bottom],
        "cropped_size": [cropped.size[0], cropped.size[1]],
        "output_size": [crop_width, crop_height],
        "min_reference_dimension": MIN_REFERENCE_IMAGE_DIMENSION,
        "resized_for_min_dimension": resized_for_min_dimension,
    }


def split_scene_materials_from_storyboard(
    *,
    storyboard_path: Path,
    scene_output_paths: dict[str, Path],
    scene_layouts: dict[str, tuple[int, int] | None],
    output_root: Path,
    telemetry: TelemetryRecorder | None,
    model: str,
    episode_id: str | None,
) -> dict[str, Any]:
    refs = parse_seedance_scene_materials(storyboard_path)
    materials_dir = output_root / "scene_materials"
    manifest_items: list[dict[str, Any]] = []
    generated_count = 0
    skipped_count = 0
    available_scene_grids = [
        {
            "grid_title": title,
            "grid_output_path": str(path),
            "layout": list(scene_layouts.get(title) or []),
        }
        for title, path in scene_output_paths.items()
    ]

    if not refs:
        manifest = {
            "status": "no_storyboard_scene_refs",
            "storyboard_path": str(storyboard_path),
            "output_root": str(materials_dir),
            "reason": "02-seedance-prompts.md 的素材对应表里没有任何“场景参考”行，因此当前无法按 @图片 编号自动切分场景宫格。",
            "generated_count": generated_count,
            "skipped_count": skipped_count,
            "available_scene_grids": available_scene_grids,
            "items": manifest_items,
        }
        manifest_path = output_root / "scene_material_manifest.json"
        save_json_file(manifest_path, manifest)
        return manifest

    materials_dir.mkdir(parents=True, exist_ok=True)

    for ref in refs:
        grid_path, layout = resolve_scene_grid_target(
            grid_title=ref.grid_title,
            scene_output_paths=scene_output_paths,
            scene_layouts=scene_layouts,
        )
        manifest_item = {
            "reference_number": ref.reference_number,
            "reference_token": ref.reference_token,
            "scene_name": ref.scene_name,
            "grid_title": ref.grid_title,
            "panel_number": ref.panel_number,
            "panel_label": ref.panel_label,
        }
        if grid_path is None or not grid_path.exists():
            manifest_item["status"] = "missing_grid_image"
            manifest_items.append(manifest_item)
            continue
        if layout is None:
            manifest_item["status"] = "missing_grid_layout"
            manifest_item["grid_output_path"] = str(grid_path)
            manifest_items.append(manifest_item)
            continue

        ext = grid_path.suffix.lower() if grid_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else ".jpg"
        output_path = materials_dir / f"{ref.reference_number:03d}__{slugify(ref.scene_name)}{ext}"
        try:
            with telemetry_span(
                telemetry,
                stage="nano_banana",
                name="split_scene_material_from_storyboard",
                provider="local",
                model=model,
                metadata={
                    "episode_id": normalize_episode_key(episode_id),
                    "reference_token": ref.reference_token,
                    "scene_name": ref.scene_name,
                    "grid_title": ref.grid_title,
                    "grid_output_path": str(grid_path),
                    "output_path": str(output_path),
                },
            ) as step:
                crop_meta = crop_scene_panel(
                    image_path=grid_path,
                    rows=layout[0],
                    cols=layout[1],
                    panel_number=ref.panel_number,
                    output_path=output_path,
                )
                step["status"] = "generated"
                step["metadata"].update(crop_meta)
            generated_count += 1
            manifest_item.update(
                {
                    "status": "generated",
                    "grid_output_path": str(grid_path),
                    "output_path": str(output_path),
                    "layout": [layout[0], layout[1]],
                    **crop_meta,
                }
            )
        except ValueError as exc:
            manifest_item.update(
                {
                    "status": "panel_out_of_layout",
                    "grid_output_path": str(grid_path),
                    "layout": [layout[0], layout[1]],
                    "error": str(exc),
                }
            )
        manifest_items.append(manifest_item)

    manifest = {
        "status": "ok",
        "storyboard_path": str(storyboard_path),
        "output_root": str(materials_dir),
        "generated_count": generated_count,
        "skipped_count": skipped_count,
        "available_scene_grids": available_scene_grids,
        "items": manifest_items,
    }
    manifest_path = output_root / "scene_material_manifest.json"
    save_json_file(manifest_path, manifest)
    return manifest


def filter_prompt_items(
    *,
    items: list[PromptItem],
    episode_id: str | None,
    include_all_history: bool,
) -> list[PromptItem]:
    normalized_episode = normalize_episode_key(episode_id)
    if not normalized_episode:
        return items
    if include_all_history:
        target_number = int(re.search(r"(\d+)", normalized_episode).group(1))
        result: list[PromptItem] = []
        for item in items:
            item_episode = normalize_episode_key(item.episode_id)
            if not item_episode:
                result.append(item)
                continue
            if int(re.search(r"(\d+)", item_episode).group(1)) <= target_number:
                result.append(item)
        return result

    return [item for item in items if normalize_episode_key(item.episode_id) == normalized_episode]


def generate_image(
    *,
    api_key: str,
    model: str,
    prompt: str,
    negative_prompt: str | None,
    image_size_hint: str | None,
    timeout_seconds: int,
    reference_images: list[Path] | None = None,
    request_retry_attempts: int = 3,
    request_retry_base_delay_seconds: float = 1.5,
) -> tuple[bytes, str, dict[str, Any]]:
    full_prompt = prompt.strip()
    if negative_prompt:
        full_prompt += f"\n\n补充约束：{negative_prompt.strip()}"
    if image_size_hint:
        full_prompt += f"\n\n画幅建议：{image_size_hint.strip()}"
    if reference_images:
        full_prompt += (
            "\n\n参考图要求：请严格保持参考图中同一角色的脸部结构、五官比例、年龄感、肤色、发型发色与整体辨识度一致，"
            "不要把角色生成成另一个人。参考图中的脸和人物身份优先级高于文字里任何会改变长相的描述；"
            "若文字与参考图冲突，以参考图的人脸与人物身份为准，文字只负责调整服装、妆发细节、姿态和情境。"
        )

    parts: list[dict[str, Any]] = []
    for reference_image in reference_images or []:
        parts.append(
            {
                "inline_data": {
                    "mime_type": infer_mime_type(reference_image),
                    "data": file_to_base64(reference_image),
                }
            }
        )
    parts.append({"text": full_prompt})

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    response = request_json(
        url=url,
        payload=payload,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_attempts=request_retry_attempts,
        retry_base_delay_seconds=request_retry_base_delay_seconds,
    )
    prompt_feedback = response.get("promptFeedback") or {}
    prompt_block_reason = str(prompt_feedback.get("blockReason") or "")
    prompt_block_message = str(prompt_feedback.get("blockReasonMessage") or "")

    if prompt_block_reason:
        raise ImageSafetyBlockedError(
            finish_reason=prompt_block_reason,
            finish_message=prompt_block_message or "图片生成请求被 Gemini 安全策略拦截。",
            response=response,
        )

    for candidate in response.get("candidates", []):
        finish_reason = str(candidate.get("finishReason") or "")
        finish_message = str(candidate.get("finishMessage") or "")
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            inline_data = part.get("inlineData") or part.get("inline_data")
            if not inline_data:
                continue
            data = inline_data.get("data")
            mime_type = inline_data.get("mimeType") or inline_data.get("mime_type") or "image/png"
            if not data:
                continue
            return base64.b64decode(data), mime_type, response
        if finish_reason == "IMAGE_SAFETY":
            raise ImageSafetyBlockedError(
                finish_reason=finish_reason,
                finish_message=finish_message,
                response=response,
            )

    raise RuntimeError(f"Gemini 返回中没有图片数据：{json.dumps(response, ensure_ascii=False)[:1200]}")


def build_output_stems(item: PromptItem, index: int) -> list[str]:
    stems: list[str] = []
    full_label_stem = f"{index:03d}__{slugify(item.source_label)}"
    legacy_name_stem = f"{index:03d}__{slugify(item.name)}"
    stems.append(full_label_stem)
    if legacy_name_stem != full_label_stem:
        stems.append(legacy_name_stem)
    return stems


def build_output_path(base_dir: Path, item: PromptItem, index: int, ext: str) -> Path:
    stem = build_output_stems(item, index)[0]
    return base_dir / f"{stem}{ext}"


def find_existing_output(base_dir: Path, item: PromptItem, index: int) -> Path | None:
    for stem in build_output_stems(item, index):
        for candidate in sorted(base_dir.glob(f"{stem}.*")):
            if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                return candidate
    return None


def load_generated_asset_records(model_root: Path) -> list[GeneratedAssetRecord]:
    records: list[GeneratedAssetRecord] = []
    if not model_root.exists():
        return records
    for manifest_path in sorted(model_root.glob("*/generation_manifest.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        episode_id = normalize_episode_key(payload.get("episode_id"))
        for item in payload.get("items", []):
            status = str(item.get("status") or "")
            if status not in {"generated", "generated_with_reference", "reused_from_previous_episode", "skipped_existing"}:
                continue
            output_path_raw = item.get("output_path")
            if not output_path_raw:
                continue
            output_path = Path(output_path_raw).expanduser().resolve()
            if not output_path.exists():
                continue
            prompt_type = str(item.get("prompt_type") or "")
            source_label = str(item.get("source_label") or "")
            name = str(item.get("name") or "")
            if prompt_type == "character":
                entity_name, variant_label = split_character_title(source_label or name)
            else:
                entity_name = source_label or name
                variant_label = source_label or name
            records.append(
                GeneratedAssetRecord(
                    prompt_type=prompt_type,
                    source_label=source_label,
                    name=name,
                    output_path=output_path,
                    episode_id=episode_id,
                    entity_name=entity_name,
                    variant_label=variant_label,
                )
            )
    return records


def find_prior_exact_record(
    *,
    records: list[GeneratedAssetRecord],
    item: PromptItem,
    current_episode_id: str | None,
) -> GeneratedAssetRecord | None:
    target_key = stable_item_key(item.prompt_type, item.source_label)
    candidates = [
        record
        for record in records
        if stable_item_key(record.prompt_type, record.source_label) == target_key
        and record.output_path.exists()
        and normalize_episode_key(record.episode_id) != normalize_episode_key(current_episode_id)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda record: episode_sort_key(record.episode_id))
    return candidates[-1]


def find_prior_exact_file(
    *,
    model_root: Path,
    item: PromptItem,
    current_episode_id: str | None,
) -> Path | None:
    if not model_root.exists():
        return None
    current_episode = normalize_episode_key(current_episode_id)
    suffixes = [slugify(item.source_label)]
    legacy_name = slugify(item.name)
    if legacy_name not in suffixes:
        suffixes.append(legacy_name)

    target_subdir = 'characters' if item.prompt_type == 'character' else 'scenes'
    candidates: list[tuple[tuple[int, int], Path]] = []
    for episode_dir in sorted(model_root.iterdir()):
        if not episode_dir.is_dir():
            continue
        episode_key = normalize_episode_key(episode_dir.name)
        if episode_key == current_episode:
            continue
        asset_dir = episode_dir / target_subdir
        if not asset_dir.exists():
            continue
        for file_path in sorted(asset_dir.iterdir()):
            if file_path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
                continue
            stem = file_path.stem
            if '__' not in stem:
                continue
            _, _, tail = stem.partition('__')
            if tail in suffixes:
                candidates.append((episode_sort_key(episode_key), file_path.resolve()))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def prompt_reuse_signature(prompt: str) -> str:
    return normalize_spaces(prompt or "").replace("：", ":").lower()


def find_episode_asset_file(
    *,
    model_root: Path,
    item: PromptItem,
    episode_id: str | None,
) -> Path | None:
    episode_key = normalize_episode_key(episode_id)
    if not episode_key:
        return None
    target_subdir = "characters" if item.prompt_type == "character" else "scenes"
    asset_dir = model_root / episode_key / target_subdir
    if not asset_dir.exists():
        return None
    suffixes = [slugify(item.source_label)]
    legacy_name = slugify(item.name)
    if legacy_name not in suffixes:
        suffixes.append(legacy_name)
    for file_path in sorted(asset_dir.iterdir()):
        if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        stem = file_path.stem
        if "__" not in stem:
            continue
        _, _, tail = stem.partition("__")
        if tail in suffixes:
            return file_path.resolve()
    return None


def find_history_episode_asset_file(
    *,
    model_root: Path,
    item: PromptItem,
    current_episode_id: str | None,
) -> Path | None:
    item_episode = normalize_episode_key(item.episode_id)
    current_episode = normalize_episode_key(current_episode_id)
    if not item_episode or not current_episode:
        return None
    if episode_sort_key(item_episode) >= episode_sort_key(current_episode):
        return None
    return find_episode_asset_file(
        model_root=model_root,
        item=item,
        episode_id=item_episode,
    )


def find_prior_prompt_equivalent_item(
    *,
    items: list[PromptItem],
    item: PromptItem,
    current_episode_id: str | None,
) -> PromptItem | None:
    current_episode = normalize_episode_key(current_episode_id)
    target_signature = prompt_reuse_signature(item.prompt)
    if not current_episode or not target_signature:
        return None
    target_entity = normalize_spaces(item.entity_name or item.name)
    candidates: list[PromptItem] = []
    for candidate in items:
        candidate_episode = normalize_episode_key(candidate.episode_id)
        if not candidate_episode or episode_sort_key(candidate_episode) >= episode_sort_key(current_episode):
            continue
        if candidate.prompt_type != item.prompt_type:
            continue
        if prompt_reuse_signature(candidate.prompt) != target_signature:
            continue
        if item.prompt_type == "character":
            candidate_entity = normalize_spaces(candidate.entity_name or candidate.name)
            if candidate_entity != target_entity:
                continue
        candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(key=lambda candidate: episode_sort_key(candidate.episode_id), reverse=True)
    return candidates[0]


def find_character_reference_images(
    *,
    records: list[GeneratedAssetRecord],
    item: PromptItem,
    max_count: int,
) -> list[Path]:
    if item.prompt_type != "character" or max_count <= 0:
        return []
    entity_name = normalize_spaces(item.entity_name or item.name)
    candidates = [
        record
        for record in records
        if record.prompt_type == "character"
        and normalize_spaces(record.entity_name or record.name) == entity_name
        and record.output_path.exists()
    ]
    if not candidates:
        return []

    by_base_priority = sorted(
        candidates,
        key=lambda record: (
            variant_priority(record.variant_label or record.source_label),
            episode_sort_key(record.episode_id),
        ),
    )
    by_recency = sorted(candidates, key=lambda record: episode_sort_key(record.episode_id), reverse=True)

    selected: list[Path] = []
    for record in [by_base_priority[0], by_recency[0]]:
        if record.output_path not in selected:
            selected.append(record.output_path)
        if len(selected) >= max_count:
            break
    return selected[:max_count]


def register_generated_record(records: list[GeneratedAssetRecord], item: PromptItem, output_path: Path) -> None:
    records.append(
        GeneratedAssetRecord(
            prompt_type=item.prompt_type,
            source_label=item.source_label,
            name=item.name,
            output_path=output_path,
            episode_id=item.episode_id,
            entity_name=item.entity_name or item.name,
            variant_label=item.variant_label or item.source_label,
        )
    )


def copy_asset(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate character and scene images from assets prompts via Gemini image models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser


def run_pipeline(config: dict[str, Any], telemetry: TelemetryRecorder | None = None) -> dict[str, Any]:
    script_path = config["script"]["script_path"]
    explicit_series_name = (config["script"].get("series_name") or "").strip()
    inferred_series_name = ""
    if script_path:
        inferred_series_name = Path(script_path).expanduser().resolve().parent.name
    safe_series_name = explicit_series_name
    if inferred_series_name and explicit_series_name and explicit_series_name != inferred_series_name:
        print_status(
            f"检测到 script.series_name 与 script_path 父目录不一致，将优先使用 script_path 推导的剧名：{inferred_series_name}"
        )
        safe_series_name = inferred_series_name
    if inferred_series_name and not safe_series_name:
        safe_series_name = inferred_series_name

    paths: SeriesPaths = build_series_paths(
        project_root=PROJECT_ROOT,
        script_path=script_path,
        series_name=safe_series_name,
        episode_id=config["script"].get("episode_id"),
    )

    gemini_config = config["provider"]["gemini"]
    api_key = gemini_config.get("api_key", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        fallback = load_json(PROJECT_ROOT / "config/video_pipeline.local.json")
        api_key = fallback.get("providers", {}).get("gemini", {}).get("api_key", "").strip()
    if not api_key and not config["run"].get("dry_run", False):
        raise RuntimeError("缺少 GEMINI_API_KEY。请在 config/nano_banana_assets.local.json 或环境变量中设置。")

    character_prompt_path = Path(
        config["sources"].get("character_prompts_path") or paths.assets_dir / "character-prompts.md"
    ).expanduser().resolve()
    scene_prompt_path = Path(
        config["sources"].get("scene_prompts_path") or paths.assets_dir / "scene-prompts.md"
    ).expanduser().resolve()
    storyboard_prompt_path = Path(
        config["sources"].get("seedance_storyboard_path")
        or resolve_default_storyboard_prompt_path(paths.series_name, paths.episode_id)
    ).expanduser().resolve()
    effective_assets_dir = resolve_effective_assets_dir(paths, character_prompt_path, scene_prompt_path)

    runtime_style_hints = load_runtime_style_hints(paths)

    print_status(f"剧名：{paths.series_name}")
    print_status(f"脚本：{paths.script_path}")
    print_status(f"角色提示词：{character_prompt_path}")
    print_status(f"场景提示词：{scene_prompt_path}")
    print_status(f"分镜提示词：{storyboard_prompt_path}")
    if effective_assets_dir != paths.assets_dir:
        print_status(
            f"检测到当前提示词来自不同素材目录，历史资产复用将优先使用：{effective_assets_dir}"
        )
    if (
        runtime_style_hints.get("target_medium")
        or runtime_style_hints.get("visual_style")
        or runtime_style_hints.get("frame_orientation")
    ):
        print_status(
            "风格提示：目标媒介={} | 视觉风格={} | 目标画幅={} | 来源={}".format(
                runtime_style_hints.get("target_medium") or "<空>",
                runtime_style_hints.get("visual_style") or "<空>",
                runtime_style_hints.get("frame_orientation") or "<空>",
                runtime_style_hints.get("source") or "<未知>",
            )
        )

    character_items = filter_prompt_items(
        items=parse_character_prompts(character_prompt_path),
        episode_id=paths.episode_id,
        include_all_history=bool(config["selection"].get("include_all_history_assets", True)),
    )
    scene_items = filter_prompt_items(
        items=parse_scene_prompts(scene_prompt_path),
        episode_id=paths.episode_id,
        include_all_history=bool(config["selection"].get("include_all_history_assets", True)),
    )
    scene_layouts = {item.source_label: extract_scene_grid_layout(item.prompt) for item in scene_items}
    scene_layouts = apply_scene_layout_overrides(
        scene_layouts=scene_layouts,
        config=config,
        episode_id=paths.episode_id,
    )

    if config["selection"].get("generate_characters", True) is False:
        character_items = []
    if config["selection"].get("generate_scenes", True) is False:
        scene_items = []

    all_items = character_items + scene_items
    if not all_items:
        raise RuntimeError(
            "没有找到可生成的角色或场景提示词。"
            f"\n剧名推导结果：{paths.series_name}"
            f"\n角色提示词路径：{character_prompt_path}"
            f"\n场景提示词路径：{scene_prompt_path}"
            "\n请先检查 assets/<剧名>/ 下的提示词文件，或确认 script_path / series_name / sources.* 配置是否正确。"
        )

    model = gemini_config.get("model", "gemini-3-pro-image-preview")
    model_root = (effective_assets_dir / "generated" / slugify(model)).expanduser().resolve()
    output_root = Path(
        config["output"].get("output_root")
        or model_root / (paths.episode_id or "all")
    ).expanduser().resolve()
    characters_dir = output_root / "characters"
    scenes_dir = output_root / "scenes"
    output_manifest_path = output_root / "generation_manifest.json"
    manifest_items: list[dict[str, Any]] = []
    generated_count = 0
    blocked_count = 0
    failed_count = 0
    skipped_existing_count = 0
    reused_count = 0
    generated_with_reference_count = 0

    print_status(f"待生成：角色 {len(character_items)} 项，场景 {len(scene_items)} 项")

    if config["run"].get("dry_run", False):
        payload = {
            "status": "dry_run",
            "series_name": paths.series_name,
            "episode_id": paths.episode_id,
            "model": model,
            "character_items": [item.source_label for item in character_items],
            "scene_items": [item.source_label for item in scene_items],
            "storyboard_path": str(storyboard_prompt_path),
            "split_scenes_from_storyboard_table": bool(config["selection"].get("split_scenes_from_storyboard_table", True)),
            "output_root": str(output_root),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    timeout_seconds = int(config["run"].get("timeout_seconds", 300))
    delay_seconds = float(config["run"].get("delay_seconds", 1.0))
    request_retry_attempts = max(1, int(config["run"].get("request_retry_attempts", 3)))
    request_retry_base_delay_seconds = max(
        0.2,
        float(config["run"].get("request_retry_base_delay_seconds", 1.5)),
    )
    continue_on_error = bool(config["run"].get("continue_on_error", False))
    skip_existing_images = bool(config["run"].get("skip_existing_images", True))
    negative_prompt = config["quality"].get("negative_prompt")
    character_size_hint = config["quality"].get("character_image_size_hint")
    scene_size_hint_config = config["quality"].get("scene_image_size_hint")
    scene_orientation_retry_times = max(1, int(config["run"].get("scene_orientation_retry_times", 2)))
    scene_invalid_argument_retry_times = max(1, int(config["run"].get("scene_invalid_argument_retry_times", 2)))
    scene_after_character_cooldown_seconds = max(
        0.0,
        float(config["run"].get("scene_after_character_cooldown_seconds", 6.0)),
    )
    scene_frame_mode = normalize_frame_orientation(runtime_style_hints.get("frame_orientation"))
    if not scene_frame_mode:
        scene_frame_mode = normalize_frame_orientation(config.get("quality", {}).get("frame_orientation"))
    if not scene_frame_mode:
        scene_frame_mode = "portrait"
    scene_hint_mode = normalize_frame_orientation(scene_size_hint_config)
    if scene_hint_mode and scene_hint_mode != scene_frame_mode:
        print_status(
            "检测到 quality.scene_image_size_hint 与目标画幅冲突，自动按目标画幅纠偏："
            f"hint={scene_size_hint_config} -> frame_mode={scene_frame_mode}"
        )
        scene_size_hint_config = "auto"
    reuse_previous_episode_assets = bool(config["selection"].get("reuse_previous_episode_assets", True))
    use_character_reference_images = bool(config["selection"].get("use_character_reference_images", True))
    max_reference_images_per_item = max(1, int(config["selection"].get("max_reference_images_per_item", 1)))
    split_scenes_from_storyboard_table = bool(config["selection"].get("split_scenes_from_storyboard_table", True))

    prior_records = load_generated_asset_records(model_root)
    current_run_records: list[GeneratedAssetRecord] = []
    scene_output_paths: dict[str, Path] = {}
    partial_output_without_manifest = output_root.exists() and not output_manifest_path.exists()
    partial_output_recovery_logged = False
    character_generation_requests_made = 0
    scene_transition_cooldown_applied = False

    for index, item in enumerate(all_items, start=1):
        is_character = item.prompt_type == "character"
        target_dir = characters_dir if is_character else scenes_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        scene_layout = scene_layouts.get(item.source_label) if not is_character else None
        size_hint = (
            character_size_hint
            if is_character
            else resolve_scene_size_hint(
                configured_hint=scene_size_hint_config,
                frame_mode=scene_frame_mode,
                layout=scene_layout,
            )
        )
        episode_id = normalize_episode_key(paths.episode_id)

        if reuse_previous_episode_assets:
            direct_history_path = find_history_episode_asset_file(
                model_root=model_root,
                item=item,
                current_episode_id=episode_id,
            )
            prompt_equivalent_prior_path: Path | None = None
            if direct_history_path is None:
                prompt_equivalent_item = find_prior_prompt_equivalent_item(
                    items=all_items,
                    item=item,
                    current_episode_id=episode_id,
                )
                if prompt_equivalent_item is not None:
                    prompt_equivalent_prior_path = find_episode_asset_file(
                        model_root=model_root,
                        item=prompt_equivalent_item,
                        episode_id=prompt_equivalent_item.episode_id,
                    )
            exact_prior_record = find_prior_exact_record(
                records=prior_records + current_run_records,
                item=item,
                current_episode_id=episode_id,
            )
            exact_prior_path = direct_history_path
            reuse_reason = ""
            if exact_prior_path is not None:
                reuse_reason = "history_episode"
            elif prompt_equivalent_prior_path is not None:
                exact_prior_path = prompt_equivalent_prior_path
                reuse_reason = "equivalent_prompt"
            elif exact_prior_record is not None:
                exact_prior_path = exact_prior_record.output_path
                reuse_reason = "previous_episode_record"
            if exact_prior_path is None:
                exact_prior_path = find_prior_exact_file(
                    model_root=model_root,
                    item=item,
                    current_episode_id=episode_id,
                )
                if exact_prior_path is not None:
                    reuse_reason = "previous_episode_file"
            if exact_prior_path is not None and exact_prior_path.exists():
                if item.prompt_type == "scene":
                    prior_canvas_check = validate_scene_canvas_from_file(
                        image_path=exact_prior_path,
                        layout=scene_layout,
                        frame_mode=scene_frame_mode,
                    )
                    if prior_canvas_check and not prior_canvas_check.get("ok"):
                        print_status(
                            "检测到历史场景画布与目标画幅不匹配，跳过复用并重新生成："
                            f"{item.source_label} <- {exact_prior_path.name} "
                            f"({prior_canvas_check.get('width')}x{prior_canvas_check.get('height')})"
                        )
                        exact_prior_path = None
                if exact_prior_path is not None:
                    output_path = build_output_path(target_dir, item, index, exact_prior_path.suffix)
                    reuse_mode = copy_asset(exact_prior_path, output_path)
                    reused_count += 1
                    print_status(
                        f"复用历史 {item.prompt_type}: {item.source_label} -> {exact_prior_path.name} ({reuse_mode})"
                        + (f" / reason={reuse_reason}" if reuse_reason else "")
                    )
                    with telemetry_span(
                        telemetry,
                        stage="nano_banana",
                        name="reuse_previous_episode_asset",
                        provider="gemini",
                        model=model,
                        metadata={
                            "episode_id": episode_id,
                            "prompt_type": item.prompt_type,
                            "source_label": item.source_label,
                            "reused_from": str(exact_prior_path),
                            "output_path": str(output_path),
                        },
                    ) as step:
                        step["status"] = "reused_from_previous_episode"
                        manifest_items.append(
                            {
                                "prompt_type": item.prompt_type,
                                "name": item.name,
                                "source_label": item.source_label,
                                "source_path": str(item.source_path),
                                "status": "reused_from_previous_episode",
                                "output_path": str(output_path),
                                "model": model,
                                "reused_from": str(exact_prior_path),
                                "reuse_mode": reuse_mode,
                                "reuse_reason": reuse_reason,
                                "reference_image_paths": [str(exact_prior_path)],
                            }
                        )
                    if item.prompt_type == "scene":
                        scene_output_paths[item.source_label] = output_path
                    register_generated_record(current_run_records, item, output_path)
                    continue

        existing_output = find_existing_output(target_dir, item, index) if skip_existing_images else None
        if existing_output is not None and partial_output_without_manifest:
            if not partial_output_recovery_logged:
                print_status(
                    "检测到输出目录缺少 generation_manifest；当前将把已存在的同名图片视为可恢复结果直接复用，"
                    "并在本次运行结束时补写新的 generation_manifest。"
                )
                partial_output_recovery_logged = True
        if existing_output is not None:
            if item.prompt_type == "scene":
                existing_canvas_check = validate_scene_canvas_from_file(
                    image_path=existing_output,
                    layout=scene_layout,
                    frame_mode=scene_frame_mode,
                )
                if existing_canvas_check and not existing_canvas_check.get("ok"):
                    print_status(
                        "检测到已有场景画布与当前布局/目标画幅不完全匹配，但当前文件已存在；"
                        "本轮不重新生成，继续使用现有宫格图执行切分："
                        f"{item.source_label} -> {existing_output.name} "
                        f"({existing_canvas_check.get('width')}x{existing_canvas_check.get('height')})"
                    )
            if existing_output is not None:
                with telemetry_span(
                    telemetry,
                    stage="nano_banana",
                    name="skip_existing_generated_image",
                    provider="gemini",
                    model=model,
                    metadata={
                        "episode_id": episode_id,
                        "prompt_type": item.prompt_type,
                        "source_label": item.source_label,
                        "output_path": str(existing_output),
                    },
                ) as step:
                    step["status"] = "skipped_existing"
                    skipped_existing_count += 1
                    print_status(f"跳过已有 {item.prompt_type}: {item.source_label} -> {existing_output.name}")
                    manifest_items.append(
                        {
                            "prompt_type": item.prompt_type,
                            "name": item.name,
                            "source_label": item.source_label,
                            "source_path": str(item.source_path),
                            "status": "skipped_existing",
                            "output_path": str(existing_output),
                            "model": model,
                        }
                    )
                if item.prompt_type == "scene":
                    scene_output_paths[item.source_label] = existing_output
                register_generated_record(current_run_records, item, existing_output)
                continue

        effective_prompt = item.prompt
        effective_negative_prompt = negative_prompt
        reference_images: list[Path] = []
        if is_character:
            character_style_mode = infer_character_style_mode(
                prompt=item.prompt,
                config=config,
                target_medium=runtime_style_hints.get("target_medium"),
                visual_style=runtime_style_hints.get("visual_style"),
            )
            print_status(f"人物风格模式：{item.source_label} -> {character_style_mode}")
            if use_character_reference_images:
                reference_pool = current_run_records if character_style_mode == "non_real" else prior_records + current_run_records
                reference_images = find_character_reference_images(
                    records=reference_pool,
                    item=item,
                    max_count=max_reference_images_per_item,
                )
                if reference_images:
                    print_status(
                        f"参考同角色形象生成 {item.prompt_type}: {item.source_label} <- "
                        + "、".join(path.name for path in reference_images)
                    )
            character_prompt_suffix, character_negative_suffix = build_character_style_enforcement(
                style_mode=character_style_mode
            )
            normalized_character_prompt = normalize_character_prompt_for_style(
                item.prompt,
                style_mode=character_style_mode,
            )
            effective_prompt = f"{normalized_character_prompt}{character_prompt_suffix}"
            effective_negative_prompt = (
                f"{negative_prompt}，{character_negative_suffix}" if negative_prompt else character_negative_suffix
            )
        else:
            normalized_scene_prompt = normalize_scene_prompt_for_background(item.prompt)
            target_medium_text = normalize_spaces(runtime_style_hints.get("target_medium") or "")
            visual_style_text = normalize_spaces(runtime_style_hints.get("visual_style") or "")
            scene_prompt_text = normalize_spaces(item.prompt or "").lower()
            scene_has_realistic_override = any(
                keyword in scene_prompt_text for keyword in REALISTIC_PROMPT_OVERRIDE_KEYWORDS
            )
            scene_canvas_guidance = build_scene_canvas_guidance(
                layout=scene_layout,
                frame_mode=scene_frame_mode,
            )
            if any(keyword in target_medium_text for keyword in ("漫剧", "动画", "国漫", "动漫")) and not scene_has_realistic_override:
                scene_style_suffix = (
                    "\n\n画风统一要求：场景图必须与本剧人物图处于同一套高完成度仿真人漫剧/国漫电影感视觉体系，"
                    "细节丰富、材质清晰、光影层次细腻、镜头审美成熟、空间透视稳定。不要低幼卡通背景，不要平涂舞台布景感，"
                    "不要偏真实摄影，也不要和人物图出现不同渲染体系。"
                )
                scene_negative_style = "避免低幼卡通背景、避免平涂感、避免儿童插画感、避免写实摄影感、避免与人物画风不一致"
            else:
                scene_style_suffix = (
                    "\n\n画风统一要求：场景图必须与本剧人物图处于同一套电影质感视觉体系，"
                    "保持统一的镜头审美、材质精度、布光逻辑与空间层次。"
                )
                scene_negative_style = "避免与人物画风不一致、避免廉价插画感、避免低完成度背景板效果"
            effective_prompt = (
                f"{normalized_scene_prompt}"
                f"{scene_style_suffix}"
                "\n\n场景图硬性要求：这是一张纯环境/背景参考图，只允许静态空间和静态道具。"
                "不允许出现任何主体人物、清晰人脸、半身角色、人物肢体、递交动作、跪姿、背影、剪影、"
                "衣摆掠过、双人对峙或任何可识别主角。画面不能带人物动作细节，不能暗示角色正在做什么。"
                "构图重点必须放在空间结构、景深层次、光影关系、建筑细节、器物摆位、材质纹理与环境氛围。"
                f"\n\n总画布硬约束：{scene_canvas_guidance}"
            )
            scene_negative_suffix = (
                "避免主体人物、避免清晰人脸、避免前景单人或双人角色、避免主角入镜、避免人物成为视觉中心、"
                "避免肢体动作、避免手势、避免背影、避免剪影、避免衣摆掠过、避免任何角色行为暗示、"
                "避免横版长条画布、避免宽银幕构图、"
                f"{scene_negative_style}"
            )
            effective_negative_prompt = (
                f"{negative_prompt}，{scene_negative_suffix}" if negative_prompt else scene_negative_suffix
            )

        if (
            item.prompt_type == "scene"
            and character_generation_requests_made > 0
            and not scene_transition_cooldown_applied
            and scene_after_character_cooldown_seconds > 0
        ):
            print_status(
                "检测到刚结束人物批量出图，首个场景请求前增加缓冲，降低 Gemini 阶段切换时的假 400 概率："
                f"{scene_after_character_cooldown_seconds:.1f}s"
            )
            time.sleep(scene_after_character_cooldown_seconds)
            scene_transition_cooldown_applied = True

        print_status(f"开始生成 {item.prompt_type}: {item.source_label}")
        with telemetry_span(
            telemetry,
            stage="nano_banana",
            name="generate_image",
            provider="gemini",
            model=model,
            metadata={
                "episode_id": episode_id,
                "prompt_type": item.prompt_type,
                "source_label": item.source_label,
                "source_path": str(item.source_path),
                "reference_image_paths": [str(path) for path in reference_images],
            },
        ) as step:
            scene_canvas_check: dict[str, Any] | None = None
            orientation_warning: dict[str, Any] | None = None
            try:
                max_attempts = scene_orientation_retry_times if not is_character else 1
                attempt_prompt = effective_prompt
                attempt_size_hint = size_hint
                image_bytes = b""
                mime_type = "image/png"
                raw_response: dict[str, Any] = {}
                if is_character:
                    character_generation_requests_made += 1

                for attempt in range(1, max_attempts + 1):
                    invalid_argument_retry_count = 0
                    while True:
                        try:
                            image_bytes, mime_type, raw_response = generate_image(
                                api_key=api_key,
                                model=model,
                                prompt=attempt_prompt,
                                negative_prompt=effective_negative_prompt,
                                image_size_hint=attempt_size_hint,
                                timeout_seconds=timeout_seconds,
                                reference_images=reference_images,
                                request_retry_attempts=request_retry_attempts,
                                request_retry_base_delay_seconds=request_retry_base_delay_seconds,
                            )
                            break
                        except RuntimeError as exc:
                            should_retry_invalid_argument = (
                                not is_character
                                and is_generic_invalid_argument_error(exc)
                                and invalid_argument_retry_count < scene_invalid_argument_retry_times - 1
                            )
                            if not should_retry_invalid_argument:
                                raise
                            invalid_argument_retry_count += 1
                            wait_seconds = max(
                                4.0,
                                scene_after_character_cooldown_seconds,
                                request_retry_base_delay_seconds * (invalid_argument_retry_count + 1),
                            )
                            print_status(
                                "Gemini 返回通用 INVALID_ARGUMENT，疑似人物阶段切到场景阶段时的瞬时异常；"
                                f"{wait_seconds:.1f}s 后重试 {item.source_label} "
                                f"({invalid_argument_retry_count + 1}/{scene_invalid_argument_retry_times})"
                            )
                            time.sleep(wait_seconds)
                    apply_provider_usage(step, "gemini", raw_response)

                    if is_character:
                        break

                    scene_canvas_check = validate_scene_canvas(
                        image_bytes=image_bytes,
                        layout=scene_layout,
                        frame_mode=scene_frame_mode,
                    )
                    step["metadata"]["scene_canvas_check"] = scene_canvas_check
                    step["metadata"]["scene_canvas_attempt"] = attempt

                    if scene_canvas_check.get("ok"):
                        break

                    if attempt < max_attempts:
                        print_status(
                            f"场景画布方向偏差，自动重试 {item.source_label} ({attempt + 1}/{max_attempts})："
                            f"{scene_canvas_check.get('width')}x{scene_canvas_check.get('height')}"
                        )
                        retry_orientation_line = (
                            "纠偏重试要求：上一轮画布方向错误。此次必须输出竖版总画布（高大于宽），"
                            "不要横版长条，不要宽银幕比例。"
                            if scene_frame_mode == "portrait"
                            else "纠偏重试要求：上一轮画布方向错误。此次必须输出横版总画布（宽大于高），不要竖版长图。"
                        )
                        attempt_prompt = f"{effective_prompt}\n\n{retry_orientation_line}"
                        attempt_size_hint = resolve_scene_size_hint(
                            configured_hint="auto",
                            frame_mode=scene_frame_mode,
                            layout=scene_layout,
                        )
                        continue

                    orientation_warning = scene_canvas_check
            except ImageSafetyBlockedError as exc:
                step["status"] = "blocked"
                step["metadata"]["block_reason"] = exc.finish_reason
                step["metadata"]["block_message"] = exc.finish_message
                step["metadata"]["raw_response_excerpt"] = json.dumps(exc.response, ensure_ascii=False)[:1200]
                blocked_count += 1
                print_status(f"已跳过 {item.prompt_type}: {item.source_label}，原因：{exc.finish_reason}")
                manifest_items.append(
                    {
                        "prompt_type": item.prompt_type,
                        "name": item.name,
                        "source_label": item.source_label,
                        "source_path": str(item.source_path),
                        "status": "blocked",
                        "block_reason": exc.finish_reason,
                        "block_message": exc.finish_message,
                        "model": model,
                        "raw_response_excerpt": json.dumps(exc.response, ensure_ascii=False)[:1200],
                        "reference_image_paths": [str(path) for path in reference_images],
                    }
                )
                time.sleep(delay_seconds)
                continue
            except Exception as exc:
                step["status"] = "failed"
                step["metadata"]["error_type"] = type(exc).__name__
                step["metadata"]["error_message"] = str(exc)
                failed_count += 1
                print_status(f"生成失败 {item.prompt_type}: {item.source_label} -> {exc}")
                manifest_items.append(
                    {
                        "prompt_type": item.prompt_type,
                        "name": item.name,
                        "source_label": item.source_label,
                        "source_path": str(item.source_path),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "model": model,
                        "reference_image_paths": [str(path) for path in reference_images],
                    }
                )
                if not continue_on_error:
                    raise
                time.sleep(delay_seconds)
                continue

            ext = ".png" if "png" in mime_type.lower() else ".jpg"
            output_path = build_output_path(target_dir, item, index, ext)
            output_path.write_bytes(image_bytes)
            generated_count += 1
            if reference_images:
                generated_with_reference_count += 1
            step["metadata"]["output_path"] = str(output_path)
            step["metadata"]["mime_type"] = mime_type
            if orientation_warning:
                step["metadata"]["scene_canvas_warning"] = True
                print_status(
                    f"警告：场景画布仍与目标画幅不匹配 {item.source_label} -> "
                    f"{orientation_warning.get('width')}x{orientation_warning.get('height')}"
                )
            status = "generated_with_reference" if reference_images else "generated"
            if orientation_warning:
                status = "generated_with_canvas_warning"
            manifest_items.append(
                {
                    "prompt_type": item.prompt_type,
                    "name": item.name,
                    "source_label": item.source_label,
                    "source_path": str(item.source_path),
                    "status": status,
                    "output_path": str(output_path),
                    "mime_type": mime_type,
                    "model": model,
                    "raw_response_excerpt": json.dumps(raw_response, ensure_ascii=False)[:1200],
                    "reference_image_paths": [str(path) for path in reference_images],
                    "scene_canvas_check": scene_canvas_check,
                    "scene_canvas_warning": bool(orientation_warning),
                }
            )
        if item.prompt_type == "scene":
            scene_output_paths[item.source_label] = output_path
        register_generated_record(current_run_records, item, output_path)
        time.sleep(delay_seconds)

    scene_material_manifest: dict[str, Any] | None = None
    if split_scenes_from_storyboard_table and storyboard_prompt_path.exists() and scene_output_paths:
        layout_preview = ", ".join(
            f"{label}={layout[0]}x{layout[1]}"
            for label, layout in scene_layouts.items()
            if layout is not None
        ) or "<未识别布局>"
        print_status(f"开始根据 02-seedance-prompts.md 的素材对应表切分场景图并命名。布局：{layout_preview}")
        scene_material_manifest = split_scene_materials_from_storyboard(
            storyboard_path=storyboard_prompt_path,
            scene_output_paths=scene_output_paths,
            scene_layouts=scene_layouts,
            output_root=output_root,
            telemetry=telemetry,
            model=model,
            episode_id=paths.episode_id,
        )
        print_status(
            f"场景切分完成：{output_root / 'scene_material_manifest.json'} | "
            f"已切分 {scene_material_manifest.get('generated_count', 0)} 张"
        )
        if scene_material_manifest.get("status") == "no_storyboard_scene_refs":
            print_status(
                "未从 02-seedance-prompts.md 的素材对应表中解析到任何“场景参考”行；"
                "这通常说明 storyboard 没把场景 ref 写进最终分镜。"
            )
    elif split_scenes_from_storyboard_table:
        if not storyboard_prompt_path.exists():
            print_status(f"未找到分镜提示词，跳过场景切分：{storyboard_prompt_path}")
        elif not scene_output_paths:
            print_status("当前没有可切分的场景宫格输出，跳过素材表切分。")

    manifest = {
        "series_name": paths.series_name,
        "episode_id": paths.episode_id,
        "script_path": str(paths.script_path) if paths.script_path else None,
        "model": model,
        "output_root": str(output_root),
        "generated_count": generated_count,
        "generated_with_reference_count": generated_with_reference_count,
        "reused_from_previous_episode_count": reused_count,
        "blocked_count": blocked_count,
        "failed_count": failed_count,
        "skipped_existing_count": skipped_existing_count,
        "continue_on_error": continue_on_error,
        "request_retry_attempts": request_retry_attempts,
        "request_retry_base_delay_seconds": request_retry_base_delay_seconds,
        "storyboard_path": str(storyboard_prompt_path),
        "scene_material_split_enabled": split_scenes_from_storyboard_table,
        "scene_material_manifest_path": str(output_root / 'scene_material_manifest.json') if scene_material_manifest else None,
        "scene_material_generated_count": int(scene_material_manifest.get('generated_count', 0)) if scene_material_manifest else 0,
        "items": manifest_items,
    }
    manifest_path = output_root / "generation_manifest.json"
    save_json_file(manifest_path, manifest)
    print_status(f"素材生成完成：{manifest_path}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_json(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
