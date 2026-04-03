from __future__ import annotations

import argparse
import difflib
import json
import shutil
import re
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_seedance_prompts import (
    SEEDANCE_PROMPTS_SCHEMA,
    build_asset_catalog,
    merge_telemetry_recorders,
    normalize_storyboard_point_id,
    normalize_storyboard_result,
    render_seedance_markdown,
    repair_storyboard_density,
    repair_storyboard_ref_grounding,
    resolve_assets_dir,
    resolve_episode_output_dir,
    resolve_series_name,
    resolve_storyboard_profile,
    strip_storyboard_title_prefix,
    storyboard_profile_settings,
    validate_scene_reference_presence,
    validate_storyboard_density,
)
from openai_agents.runtime_utils import (
    build_episode_ids,
    configure_openai_api,
    load_runtime_config,
    openai_json_completion,
    read_text,
)
from pipeline_telemetry import TelemetryRecorder, telemetry_span
from prompt_utils import render_prompt
from providers.base import build_provider_model_tag, load_json_file, save_json_file, utc_timestamp
from seedance_logic_review_rules import (
    render_generalized_logic_rules_markdown,
    render_generalized_logic_rules_prompt,
)

DEFAULT_CONFIG_PATH = Path("config/openai_agent_flow.local.json")


def print_status(message: str) -> None:
    print(f"[seedance-review] {message}", flush=True)


def metrics_paths(episode_output_dir: Path, provider_tag: str) -> tuple[Path, Path]:
    base = episode_output_dir / f"02-seedance-prompts.logic-review.metrics__{provider_tag}"
    return Path(f"{base}.json"), Path(f"{base}.md")


def stamp_path(episode_output_dir: Path, provider_tag: str) -> Path:
    return episode_output_dir / f"02-seedance-prompts.logic-review__{provider_tag}.json"


def report_markdown_path(episode_output_dir: Path, provider_tag: str) -> Path:
    return episode_output_dir / f"02-seedance-prompts.logic-review__{provider_tag}.md"


def render_metrics_markdown(report: Mapping[str, Any]) -> str:
    context = dict(report.get("context", {}) or {})
    totals = dict(report.get("totals", {}) or {})
    lines = [
        "# Seedance Prompt 轻审核统计报告",
        "",
        f"- series_name：{context.get('series_name', '')}",
        f"- episode_id：{context.get('episode_id', '')}",
        f"- model：{context.get('model', '')}",
        f"- final_status：{context.get('final_status', '')}",
        f"- changed_point_count：{context.get('changed_point_count', 0)}",
        "",
        "## 总计",
        "",
        f"- steps：{totals.get('step_count', 0)}",
        f"- duration_seconds：{totals.get('duration_seconds', 0)}",
        f"- input_tokens：{totals.get('input_tokens', 0)}",
        f"- output_tokens：{totals.get('output_tokens', 0)}",
        f"- total_tokens：{totals.get('total_tokens', 0)}",
        "",
        "## 步骤",
        "",
        "| Step ID | 名称 | 状态 | 耗时(秒) | 输入tokens | 输出tokens | 总tokens | 备注 |",
        "|---------|------|------|---------:|-----------:|-----------:|---------:|------|",
    ]
    for step in report.get("steps", []):
        metadata = dict(step.get("metadata", {}) or {})
        note_parts: list[str] = []
        if "prompt_chars" in metadata:
            note_parts.append(f"prompt_chars={metadata['prompt_chars']}")
        if "storyboard_source_path" in metadata:
            note_parts.append(f"source={Path(str(metadata['storyboard_source_path'])).name}")
        lines.append(
            f"| {step.get('step_id', '')} | {step.get('name', '')} | {step.get('status', '')} | "
            f"{step.get('duration_seconds', 0)} | {step.get('input_tokens', 0)} | {step.get('output_tokens', 0)} | {step.get('total_tokens', 0)} | {'；'.join(note_parts)} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def save_metrics(recorder: TelemetryRecorder, json_path: Path, md_path: Path) -> dict[str, Any]:
    report = recorder.to_dict()
    save_json_file(json_path, report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_metrics_markdown(report), encoding="utf-8")
    return report


def find_storyboard_json_path(episode_output_dir: Path, provider_tag: str) -> Path | None:
    preferred = episode_output_dir / f"02-seedance-prompts__{provider_tag}.json"
    if preferred.exists():
        return preferred
    candidates = sorted(episode_output_dir.glob("02-seedance-prompts__*.json"))
    return candidates[-1] if candidates else None


def parse_storyboard_markdown(markdown: str, *, episode_id: str) -> dict[str, Any]:
    section_pattern = re.compile(r"^##\s+([^\s#]+)\s+(.+?)\n", flags=re.MULTILINE)
    matches = list(section_pattern.finditer(markdown))
    prompt_entries: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        raw_point_id = match.group(1).strip()
        point_id = normalize_storyboard_point_id(raw_point_id, fallback_index=index + 1)
        title = strip_storyboard_title_prefix(match.group(2).strip(), point_id)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        block = markdown[start:end].strip()
        prompt_marker = "**Seedance 2.0 提示词**："
        if prompt_marker in block:
            prompt_text = block.split(prompt_marker, 1)[1].strip()
        elif "可直接投喂正文：" in block:
            prompt_text = block.split("可直接投喂正文：", 1)[1].strip()
        else:
            prompt_text = block
        prompt_text = re.sub(r"\n---\s*$", "", prompt_text).strip()
        refs = re.findall(r"@图片\d+", block)
        dedup_refs: list[str] = []
        seen_refs: set[str] = set()
        for ref in refs:
            if ref not in seen_refs:
                dedup_refs.append(ref)
                seen_refs.add(ref)
        pace_label = ""
        density_strategy = ""
        duration_hint = ""
        continuity_bridge = ""
        audio_design = ""
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if line.startswith("- 节奏档位："):
                pace_label = line.split("：", 1)[1].strip()
            elif line.startswith("- 内容密度策略："):
                density_strategy = line.split("：", 1)[1].strip()
            elif line.startswith("- 建议时长："):
                duration_hint = line.split("：", 1)[1].strip()
            elif line.startswith("- 承接关系："):
                continuity_bridge = line.split("：", 1)[1].strip()
            elif line.startswith("- 声音设计："):
                audio_design = line.split("：", 1)[1].strip()
            elif not audio_design and line.startswith("音效："):
                audio_design = line
        master_timeline: list[dict[str, Any]] = []
        unified_match = re.search(
            r"\*\*统一复合提示词（主时间线）\*\*：\s*(.*?)(?:\n可直接投喂正文：|\n\*\*Seedance 2\.0 提示词\*\*：|\n---)",
            block,
            flags=re.DOTALL,
        )
        if unified_match:
            for raw_line in unified_match.group(1).splitlines():
                line = raw_line.strip()
                if not line.startswith("- "):
                    continue
                segments = [part.strip() for part in line[2:].split("｜") if part.strip()]
                if not segments:
                    continue
                time_match = re.match(
                    r"(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)秒",
                    segments[0],
                )
                if not time_match:
                    continue
                entry: dict[str, Any] = {
                    "start_second": float(time_match.group("start")),
                    "end_second": float(time_match.group("end")),
                    "visual_beat": "",
                    "speaker": "",
                    "spoken_line": "",
                    "delivery_note": "",
                    "dialogue_blocks": [],
                    "audio_cues": "",
                    "transition_hook": "",
                }
                for segment in segments[1:]:
                    if segment.startswith("画面："):
                        entry["visual_beat"] = segment.split("：", 1)[1].strip()
                    elif segment.startswith("对白窗："):
                        dialogue_window_text = segment.split("：", 1)[1].strip()
                        dialogue_chunks = [chunk.strip() for chunk in dialogue_window_text.split("；") if chunk.strip()]
                        for chunk in dialogue_chunks:
                            dialogue_match = re.match(
                                r"(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)秒\s+(?P<speaker>[^（“]+?)(?:（(?P<note>[^）]+)）)?[“\"](?P<line>.+?)[”\"]$",
                                chunk,
                            )
                            if not dialogue_match:
                                continue
                            entry["dialogue_blocks"].append(
                                {
                                    "speaker": dialogue_match.group("speaker").strip(),
                                    "line": dialogue_match.group("line").strip(),
                                    "start_second": float(dialogue_match.group("start")),
                                    "end_second": float(dialogue_match.group("end")),
                                    "delivery_note": str(dialogue_match.group("note") or "").strip(),
                                }
                            )
                    elif segment.startswith("对白："):
                        dialogue_text = segment.split("：", 1)[1].strip()
                        dialogue_match = re.match(
                            r"(?P<speaker>[^（“]+?)(?:（(?P<note>[^）]+)）)?[“\"](?P<line>.+?)[”\"]$",
                            dialogue_text,
                        )
                        if dialogue_match:
                            entry["speaker"] = dialogue_match.group("speaker").strip()
                            entry["delivery_note"] = str(dialogue_match.group("note") or "").strip()
                            entry["spoken_line"] = dialogue_match.group("line").strip()
                            entry["dialogue_blocks"].append(
                                {
                                    "speaker": entry["speaker"],
                                    "line": entry["spoken_line"],
                                    "start_second": entry["start_second"],
                                    "end_second": entry["end_second"],
                                    "delivery_note": entry["delivery_note"],
                                }
                            )
                        else:
                            entry["spoken_line"] = dialogue_text
                    elif segment.startswith("声音："):
                        entry["audio_cues"] = segment.split("：", 1)[1].strip()
                    elif segment.startswith("交棒："):
                        entry["transition_hook"] = segment.split("：", 1)[1].strip()
                if len(entry["dialogue_blocks"]) == 1:
                    only_block = entry["dialogue_blocks"][0]
                    entry["speaker"] = str(only_block.get("speaker") or "").strip()
                    entry["spoken_line"] = str(only_block.get("line") or "").strip()
                    entry["delivery_note"] = str(only_block.get("delivery_note") or "").strip()
                elif entry["dialogue_blocks"]:
                    entry["speaker"] = ""
                    entry["spoken_line"] = ""
                    entry["delivery_note"] = ""
                master_timeline.append(entry)
        shot_beats: list[str] = []
        beat_match = re.search(
            r"\*\*镜头节拍拆解\*\*：\s*(.*?)(?:\n\*\*对白时间线\*\*：|\n\*\*Seedance 2\.0 提示词\*\*：)",
            block,
            flags=re.DOTALL,
        )
        if beat_match:
            for raw_line in beat_match.group(1).splitlines():
                line = raw_line.strip()
                if line.startswith("- "):
                    shot_beats.append(line[2:].strip())
        prompt_entries.append(
            {
                "point_id": point_id,
                "title": title,
                "pace_label": pace_label,
                "density_strategy": density_strategy,
                "duration_hint": duration_hint,
                "continuity_bridge": continuity_bridge,
                "primary_refs": dedup_refs[:4],
                "secondary_refs": dedup_refs[4:],
                "master_timeline": master_timeline,
                "shot_beat_plan": shot_beats,
                "dialogue_timeline": [],
                "audio_design": audio_design,
                "prompt_text": prompt_text,
                "risk_notes": [],
            }
        )
    return {
        "episode_id": episode_id,
        "episode_title": "",
        "materials_overview": "",
        "prompt_entries": prompt_entries,
        "global_notes": [],
    }


def load_storyboard_package(episode_output_dir: Path, provider_tag: str, *, episode_id: str) -> tuple[dict[str, Any], Path, Path | None]:
    storyboard_json_path = find_storyboard_json_path(episode_output_dir, provider_tag)
    if storyboard_json_path and storyboard_json_path.exists():
        return load_json_file(storyboard_json_path), storyboard_json_path, storyboard_json_path
    storyboard_md_path = episode_output_dir / "02-seedance-prompts.md"
    if not storyboard_md_path.exists():
        raise FileNotFoundError(f"未找到可轻审核的 Seedance 提示词：{episode_output_dir}")
    return parse_storyboard_markdown(storyboard_md_path.read_text(encoding="utf-8"), episode_id=episode_id), storyboard_md_path, None


def build_review_prompt(storyboard_package: Mapping[str, Any]) -> str:
    return render_prompt(
        "seedance_prompt_review/user.md",
        {
            "generalized_review_rules": render_generalized_logic_rules_prompt(),
            "storyboard_package_json": json.dumps(storyboard_package, ensure_ascii=False, indent=2),
        },
    )


def point_payload_map(data: Mapping[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in list(data.get("prompt_entries") or []):
        point_id = str(item.get("point_id") or "").strip()
        if not point_id:
            continue
        mapping[point_id] = json.dumps(item, ensure_ascii=False, sort_keys=True)
    return mapping


def point_item_map(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for item in list(data.get("prompt_entries") or []):
        point_id = str(item.get("point_id") or "").strip()
        if point_id:
            mapping[point_id] = dict(item)
    return mapping


def render_text_diff(before: str, after: str, *, context: int = 1) -> list[str]:
    before_lines = [line.rstrip() for line in str(before or "").splitlines()]
    after_lines = [line.rstrip() for line in str(after or "").splitlines()]
    diff_lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile="before",
            tofile="after",
            lineterm="",
            n=context,
        )
    )
    return diff_lines[:80]


def summarize_changed_fields(before_item: Mapping[str, Any], after_item: Mapping[str, Any]) -> list[str]:
    fields = {
        "prompt_text": "正文提示词",
        "primary_refs": "主要引用",
        "secondary_refs": "次要引用",
        "shot_beat_plan": "镜头节拍",
        "audio_design": "声音设计",
        "continuity_bridge": "承接关系",
        "title": "标题",
    }
    changed: list[str] = []
    for field, label in fields.items():
        before_value = before_item.get(field)
        after_value = after_item.get(field)
        if json.dumps(before_value, ensure_ascii=False, sort_keys=True) != json.dumps(after_value, ensure_ascii=False, sort_keys=True):
            changed.append(label)
    return changed


def render_logic_review_report(
    *,
    series_name: str,
    episode_id: str,
    model: str,
    original_package: Mapping[str, Any],
    revised_package: Mapping[str, Any],
    changed_points: list[str],
    storyboard_source_path: Path,
    backup_markdown_path: Path | None,
    backup_json_path: Path | None,
) -> str:
    original_items = point_item_map(original_package)
    revised_items = point_item_map(revised_package)
    lines = [
        f"# Seedance Prompt 修稿报告 -- {series_name} {episode_id}",
        "",
        f"- model：{model}",
        f"- 原始来源：{storyboard_source_path}",
        f"- 备份 markdown：{backup_markdown_path or ''}",
        f"- 备份 json：{backup_json_path or ''}",
        f"- 修改分镜数：{len(changed_points)}",
        "",
    ]
    lines.append(render_generalized_logic_rules_markdown().rstrip())
    lines.append("")
    if not changed_points:
        lines.extend(["本轮逻辑检查未发现需要落地到文件的改动。", ""])
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(["## 修改摘要", ""])
    for point_id in changed_points:
        before_item = original_items.get(point_id, {})
        after_item = revised_items.get(point_id, {})
        title = str(after_item.get("title") or before_item.get("title") or "").strip()
        changed_fields = summarize_changed_fields(before_item, after_item)
        lines.append(f"- {point_id} {title}：{('、'.join(changed_fields) if changed_fields else '内容有调整')}")

    for point_id in changed_points:
        before_item = original_items.get(point_id, {})
        after_item = revised_items.get(point_id, {})
        title = str(after_item.get("title") or before_item.get("title") or "").strip()
        changed_fields = summarize_changed_fields(before_item, after_item)
        lines.extend(["", f"## {point_id} {title}", ""])
        lines.append(f"- 变更字段：{('、'.join(changed_fields) if changed_fields else '未细分')}")
        before_refs = list(before_item.get("primary_refs") or []) + list(before_item.get("secondary_refs") or [])
        after_refs = list(after_item.get("primary_refs") or []) + list(after_item.get("secondary_refs") or [])
        if before_refs != after_refs:
            lines.append(f"- 引用变化：{' '.join(before_refs) or '无'} -> {' '.join(after_refs) or '无'}")
        prompt_diff = render_text_diff(str(before_item.get("prompt_text") or ""), str(after_item.get("prompt_text") or ""))
        if prompt_diff:
            lines.extend(["", "**正文差异**", "", "```diff"])
            lines.extend(prompt_diff)
            lines.append("```")
        beat_diff = render_text_diff(
            "\n".join(str(x).strip() for x in list(before_item.get("shot_beat_plan") or []) if str(x).strip()),
            "\n".join(str(x).strip() for x in list(after_item.get("shot_beat_plan") or []) if str(x).strip()),
        )
        if beat_diff:
            lines.extend(["", "**镜头节拍差异**", "", "```diff"])
            lines.extend(beat_diff)
            lines.append("```")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def backup_file(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.{suffix}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def run_pipeline(config: Mapping[str, Any], telemetry: TelemetryRecorder | None = None) -> dict[str, Any]:
    model, api_key = configure_openai_api(config)
    series_name = resolve_series_name(config)
    episode_ids = build_episode_ids(config.get("series", {}))
    assets_dir = resolve_assets_dir(config, series_name)
    character_prompts_text = read_text(assets_dir / "character-prompts.md")
    scene_prompts_text = read_text(assets_dir / "scene-prompts.md")
    timeout_seconds = int(config.get("run", {}).get("timeout_seconds", config.get("runtime", {}).get("timeout_seconds", 600)))
    temperature = float(config.get("run", {}).get("temperature", config.get("runtime", {}).get("temperature", 0.1)))
    dry_run = bool(config.get("run", {}).get("dry_run", config.get("runtime", {}).get("dry_run", False)))
    write_review_metrics = bool(config.get("run", {}).get("write_review_metrics", config.get("runtime", {}).get("write_storyboard_review_metrics", True)))
    storyboard_profile = resolve_storyboard_profile(config)
    profile_settings = storyboard_profile_settings(storyboard_profile)
    provider_tag = build_provider_model_tag("openai", model)

    print_status(f"剧名：{series_name}")
    print_status(f"素材目录：{assets_dir}")

    previews: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        asset_catalog = build_asset_catalog(
            character_prompts_text,
            scene_prompts_text,
            episode_id=episode_id,
            assets_dir=assets_dir,
        )
        episode_output_dir = resolve_episode_output_dir(config, series_name, episode_id)
        storyboard_md_path = episode_output_dir / "02-seedance-prompts.md"
        storyboard_package, storyboard_source_path, existing_json_path = load_storyboard_package(
            episode_output_dir,
            provider_tag,
            episode_id=episode_id,
        )
        target_json_path = episode_output_dir / f"02-seedance-prompts__{provider_tag}.json"
        stamp_json_path = stamp_path(episode_output_dir, provider_tag)
        report_md_path = report_markdown_path(episode_output_dir, provider_tag)
        metrics_json_path, metrics_md_path = metrics_paths(episode_output_dir, provider_tag)
        previews.append(
            {
                "episode_id": episode_id,
                "storyboard_source_path": str(storyboard_source_path),
                "storyboard_markdown_path": str(storyboard_md_path),
                "storyboard_json_path": str(target_json_path),
                "logic_review_stamp_path": str(stamp_json_path),
                "logic_review_report_markdown_path": str(report_md_path),
            }
        )
        if dry_run:
            continue

        episode_telemetry = TelemetryRecorder(
            run_name="seedance-prompt-logic-review",
            context={
                "series_name": series_name,
                "episode_id": episode_id,
                "model": model,
                "storyboard_profile": storyboard_profile,
            },
        )
        try:
            with telemetry_span(
                episode_telemetry,
                stage="seedance_review",
                name="load_storyboard_for_logic_review",
                metadata={
                    "episode_id": episode_id,
                    "storyboard_source_path": str(storyboard_source_path),
                },
            ) as step:
                step["metadata"]["prompt_entry_count"] = len(list(storyboard_package.get("prompt_entries") or []))

            print_status(f"开始轻审核并修订 {episode_id} 的 Seedance 提示词。")
            with telemetry_span(
                episode_telemetry,
                stage="seedance_review",
                name="build_logic_review_prompt",
                metadata={"episode_id": episode_id},
            ) as step:
                review_prompt = build_review_prompt(storyboard_package)
                step["metadata"]["prompt_chars"] = len(review_prompt)
            revised_package = openai_json_completion(
                model=model,
                api_key=api_key,
                system_prompt=render_prompt("seedance_prompt_review/system.md", {}),
                user_prompt=review_prompt,
                schema_name="seedance_prompt_package_logic_reviewed",
                schema=SEEDANCE_PROMPTS_SCHEMA,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                telemetry=episode_telemetry,
                stage="seedance_review",
                step_name="seedance_prompt_logic_review_model_call",
                metadata={"episode_id": episode_id},
            )

            with telemetry_span(
                episode_telemetry,
                stage="seedance_review",
                name="materialize_logic_review_outputs",
                metadata={"episode_id": episode_id},
            ) as step:
                revised_package = normalize_storyboard_result(
                    revised_package,
                    frame_orientation=str(config.get("quality", {}).get("frame_orientation") or "9:16竖屏"),
                    storyboard_profile=storyboard_profile,
                    asset_catalog=asset_catalog,
                )
                revised_package = repair_storyboard_density(
                    revised_package,
                    max_shot_beats=profile_settings["max_shot_beats"],
                )
                revised_package, grounding_warnings = repair_storyboard_ref_grounding(
                    revised_package,
                    asset_catalog,
                    episode_id=episode_id,
                )
                density_warnings = validate_storyboard_density(revised_package, episode_id=episode_id)
                scene_ref_warnings = validate_scene_reference_presence(
                    revised_package, asset_catalog, episode_id=episode_id
                )
                quality_warnings = density_warnings + scene_ref_warnings
                if quality_warnings:
                    step["metadata"]["quality_warning_count"] = len(quality_warnings)
                    step["metadata"]["quality_warnings"] = quality_warnings[:20]
                    episode_telemetry.context["quality_warning_count"] = len(quality_warnings)
                    episode_telemetry.context["quality_warnings"] = quality_warnings[:20]
                    print_status(
                        f"{episode_id} 轻审核后存在 {len(quality_warnings)} 条质量提示：结果照常落盘，不再因密度/场景引用阈值中断。"
                    )

                original_map = point_payload_map(storyboard_package)
                revised_map = point_payload_map(revised_package)
                changed_points = [point_id for point_id, payload in revised_map.items() if original_map.get(point_id) != payload]
                timestamp = utc_timestamp().replace(":", "").replace("-", "")
                md_backup = backup_file(storyboard_md_path, f"before-logic-review__{timestamp}")
                json_backup = backup_file(target_json_path if target_json_path.exists() else existing_json_path or target_json_path, f"before-logic-review__{timestamp}")

                markdown = render_seedance_markdown(series_name=series_name, data=revised_package, asset_catalog=asset_catalog)
                storyboard_md_path.write_text(markdown, encoding="utf-8")
                save_json_file(target_json_path, revised_package)
                report_md_path.write_text(
                    render_logic_review_report(
                        series_name=series_name,
                        episode_id=episode_id,
                        model=model,
                        original_package=storyboard_package,
                        revised_package=revised_package,
                        changed_points=changed_points,
                        storyboard_source_path=storyboard_source_path,
                        backup_markdown_path=md_backup,
                        backup_json_path=json_backup,
                    ),
                    encoding="utf-8",
                )
                stamp = {
                    "episode_id": episode_id,
                    "series_name": series_name,
                    "model": model,
                    "storyboard_source_path": str(storyboard_source_path),
                    "backup_markdown_path": str(md_backup) if md_backup else "",
                    "backup_json_path": str(json_backup) if json_backup else "",
                    "changed_point_ids": changed_points,
                    "changed_point_count": len(changed_points),
                    "grounding_warning_count": len(grounding_warnings),
                    "report_markdown_path": str(report_md_path),
                    "applied_at": utc_timestamp(),
                }
                save_json_file(stamp_json_path, stamp)
                episode_telemetry.context["changed_point_count"] = len(changed_points)
                step["metadata"]["changed_point_count"] = len(changed_points)
                step["metadata"]["backup_markdown_path"] = str(md_backup) if md_backup else ""
                step["metadata"]["backup_json_path"] = str(json_backup) if json_backup else ""
                if grounding_warnings:
                    step["metadata"]["grounding_warnings"] = grounding_warnings[:10]

            results.append(
                {
                    "episode_id": episode_id,
                    "storyboard_markdown_path": str(storyboard_md_path),
                    "storyboard_json_path": str(target_json_path),
                    "logic_review_stamp_path": str(stamp_json_path),
                    "logic_review_report_markdown_path": str(report_md_path),
                    "changed_point_count": len(changed_points),
                    "generated_at": utc_timestamp(),
                }
            )
            print_status(
                f"{episode_id} 轻审核完成：修改分镜 {results[-1]['changed_point_count']} 条，原文件已原地更新为 02-seedance-prompts。"
            )
            episode_telemetry.context["final_status"] = "completed"
        except Exception as exc:
            episode_telemetry.context["final_status"] = "failed"
            episode_telemetry.context["error"] = str(exc).strip()
            raise
        finally:
            if write_review_metrics:
                metrics_report = save_metrics(episode_telemetry, metrics_json_path, metrics_md_path)
                print_status(
                    f"{episode_id} 轻审核统计：耗时 {metrics_report['totals']['duration_seconds']}s | "
                    f"tokens in/out/total = {metrics_report['totals']['input_tokens']}/{metrics_report['totals']['output_tokens']}/{metrics_report['totals']['total_tokens']}"
                )
                print_status(f"{episode_id} 轻审核统计报告：{metrics_json_path}")
            merge_telemetry_recorders(telemetry, episode_telemetry)

    if dry_run:
        preview = {
            "series_name": series_name,
            "model": model,
            "assets_dir": str(assets_dir),
            "episodes": previews,
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return preview

    summary = {
        "series_name": series_name,
        "model": model,
        "assets_dir": str(assets_dir),
        "results": results,
        "generated_at": utc_timestamp(),
    }
    summary_path = episode_output_dir.parent / "seedance-prompt-logic-review-summary.json"
    save_json_file(summary_path, summary)
    print_status(f"Seedance Prompt 轻审核链路完成：{summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight logic review for 02-seedance-prompts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_runtime_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
