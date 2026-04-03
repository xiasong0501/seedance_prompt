from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
PROMPTS_ROOT = PROJECT_ROOT / "prompts"
PLACEHOLDER_PATTERN = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")


def load_prompt(relative_path: str | Path) -> str:
    path = relative_path if isinstance(relative_path, Path) else PROMPTS_ROOT / relative_path
    return path.read_text(encoding="utf-8")


def render_prompt(relative_path: str | Path, context: Mapping[str, Any], *, strict: bool = True) -> str:
    template = load_prompt(relative_path)

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in context:
            value = context[key]
            return "" if value is None else str(value)
        if strict:
            raise KeyError(f"Prompt 模板缺少占位符变量：{key} ({relative_path})")
        return match.group(0)

    return PLACEHOLDER_PATTERN.sub(replace, template).strip()


def render_bullets(items: list[str] | tuple[str, ...], *, prefix: str = "- ") -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    return "\n".join(f"{prefix}{item}" for item in cleaned)


def normalize_frame_orientation(raw: Any, *, default: str = "9:16竖屏") -> str:
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    if not text:
        return default
    lower = text.lower()
    if "9:16" in lower or "竖" in text or "portrait" in lower or "vertical" in lower:
        if "9:16" in lower:
            return text if ("竖" in text or "portrait" in lower or "vertical" in lower) else f"{text}竖屏"
        return "9:16竖屏"
    if "16:9" in lower or "横" in text or "landscape" in lower or "horizontal" in lower:
        if "16:9" in lower:
            return text if ("横" in text or "landscape" in lower or "horizontal" in lower) else f"{text}横屏"
        return "16:9横屏"
    return text


def is_portrait_frame_orientation(raw: Any) -> bool:
    normalized = normalize_frame_orientation(raw, default="")
    if not normalized:
        return False
    lower = normalized.lower()
    return "竖" in normalized or "9:16" in lower or "portrait" in lower or "vertical" in lower


def build_frame_composition_guidance(raw: Any) -> str:
    normalized = normalize_frame_orientation(raw)
    if is_portrait_frame_orientation(normalized):
        return (
            "按9:16手机竖屏安全区构图，主体尽量沿中轴或中轴偏上/偏下组织；"
            "优先使用中景/中近景、纵深前后层次、上下高差、门框立柱台阶等竖向结构承载戏剧信息；"
            "减少超宽横向铺排与左右相距过远的对峙，把关键人物、关键道具和视线落点稳定放在竖屏安全区内。"
        )
    return (
        f"按{normalized}画幅安全区组织构图，优先保证主体清晰、动作路径明确、"
        "关键人物和关键道具不被边缘化，镜头调度始终服务叙事信息的集中表达。"
    )
