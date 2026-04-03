from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_telemetry import TelemetryRecorder, telemetry_span
from pipelines.preprocess_episode import EpisodePreprocessConfig, EpisodePreprocessor
from pipelines.video_to_script_pipeline import PipelineConfig, VideoToScriptPipeline
from providers.base import (
    EpisodeInputBundle,
    FrameReference,
    build_provider_model_tag,
    derive_series_folder_name,
    load_json_file,
    read_text_file,
    save_json_file,
    save_text_file,
)
from scripts.generate_art_assets import run_pipeline as run_art_assets_pipeline
from scripts.generate_director_analysis import run_pipeline as run_director_analysis_pipeline
from scripts.generate_nano_banana_assets import run_pipeline as run_nano_banana_pipeline
from scripts.generate_seedance_prompts import run_pipeline as run_storyboard_pipeline


DEFAULT_VIDEO_PATH = Path("videos/总裁/男朋友最穷时候我分手/第1集.mp4")
DEFAULT_VIDEO_CONFIG_PATH = Path("config/video_pipeline.local.json")
DEFAULT_OPENAI_FLOW_CONFIG_PATH = Path("config/openai_agent_flow.local.json")
DEFAULT_NANO_CONFIG_PATH = Path("config/nano_banana_assets.local.json")


def print_status(message: str) -> None:
    print(f"[workflow-benchmark] {message}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the full workflow from video to storyboard and image generation.")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--episode-id", default="ep01")
    parser.add_argument("--title", default="第1集")
    parser.add_argument("--video-provider", choices=["openai", "gemini", "qwen"], default="")
    parser.add_argument("--video-model", default="")
    parser.add_argument("--video-timeout-seconds", type=int, default=600)
    parser.add_argument("--nano-only", action="store_true")
    parser.add_argument("--video-config", default=str(DEFAULT_VIDEO_CONFIG_PATH))
    parser.add_argument("--openai-flow-config", default=str(DEFAULT_OPENAI_FLOW_CONFIG_PATH))
    parser.add_argument("--nano-config", default=str(DEFAULT_NANO_CONFIG_PATH))
    return parser


def ensure_provider_env(video_config: dict[str, Any], flow_config: dict[str, Any], nano_config: dict[str, Any]) -> None:
    openai_key = (
        str(video_config.get("providers", {}).get("openai", {}).get("api_key", "")).strip()
        or str(flow_config.get("provider", {}).get("api_key", "")).strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )
    gemini_key = (
        str(video_config.get("providers", {}).get("gemini", {}).get("api_key", "")).strip()
        or str(nano_config.get("provider", {}).get("gemini", {}).get("api_key", "")).strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
    )
    qwen_key = (
        str(video_config.get("providers", {}).get("qwen", {}).get("api_key", "")).strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    if qwen_key:
        os.environ["DASHSCOPE_API_KEY"] = qwen_key
    if not openai_key:
        raise RuntimeError("缺少 OPENAI_API_KEY，无法完成视频理解/导演/分镜/服化道 benchmark。")
    if not gemini_key:
        raise RuntimeError("缺少 GEMINI_API_KEY，无法完成 Nano Banana benchmark。")


def build_bundle_from_manifest(
    *,
    episode_id: str,
    title: str,
    video_path: Path,
    preprocess_manifest: dict[str, Any],
) -> EpisodeInputBundle:
    corrected_transcript_path = preprocess_manifest.get("corrected_transcript_text_path")
    transcript_text = (
        read_text_file(corrected_transcript_path) if corrected_transcript_path else ""
    ) or read_text_file(preprocess_manifest["transcript_text_path"]) or ""
    ocr_text = read_text_file(preprocess_manifest["ocr_text_path"]) or ""
    frames = [
        FrameReference(
            path=item.get("model_frame_path", item["frame_path"]),
            timestamp=str(item.get("midpoint_seconds", "")),
            note=item.get("scene_id"),
        )
        for item in preprocess_manifest.get("keyframes", [])
    ]
    return EpisodeInputBundle(
        episode_id=episode_id,
        title=title,
        video_path=str(video_path),
        transcript_text=transcript_text or None,
        ocr_text=ocr_text or None,
        frames=frames,
        language="zh-CN",
        metadata={"source_series": derive_series_folder_name(video_path=video_path)},
    )


def build_director_config(
    *,
    series_name: str,
    script_path: Path,
    flow_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "script": {
            "series_dir": str(script_path.parent),
            "script_path": str(script_path),
            "episode_id": "ep01",
            "preferred_filename_suffixes": [script_path.name],
        },
        "series": {
            "series_name": series_name,
            "start_episode": 1,
            "end_episode": 1,
            "episode_id_prefix": "ep",
            "episode_id_padding": 2,
        },
        "provider": {
            "openai": {
                "api_key": str(flow_config.get("provider", {}).get("api_key", "")).strip(),
                "model": str(flow_config.get("provider", {}).get("model", "gpt-5.4")).strip(),
            }
        },
        "quality": {
            "visual_style": flow_config.get("quality", {}).get(
                "visual_style",
                "真人写实，移动端9:16竖屏电影感构图，适合漫剧前期开发",
            ),
            "target_medium": flow_config.get("quality", {}).get("target_medium", "漫剧"),
            "frame_orientation": flow_config.get("quality", {}).get("frame_orientation", "9:16竖屏"),
            "extra_rules": flow_config.get("quality", {}).get("director_extra_rules", []),
        },
        "output": {
            "outputs_root": flow_config.get("output", {}).get("outputs_root", "outputs"),
            "assets_series_suffix": flow_config.get("quality", {}).get("assets_series_suffix", "-gpt"),
            "assets_series_name": flow_config.get("quality", {}).get("assets_series_name", f"{series_name}-gpt"),
        },
        "run": {
            "temperature": flow_config.get("runtime", {}).get("temperature", 0.25),
            "timeout_seconds": flow_config.get("runtime", {}).get("timeout_seconds", 300),
            "enable_review_pass": flow_config.get("runtime", {}).get("enable_review_pass", True),
            "dry_run": False,
        },
    }


def build_art_config(
    *,
    series_name: str,
    script_path: Path,
    video_provider: str,
    video_model: str,
    flow_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "script": {
            "script_path": str(script_path),
            "episode_id": "ep01",
        },
        "series": {
            "series_name": series_name,
            "start_episode": 1,
            "end_episode": 1,
            "episode_id_prefix": "ep",
            "episode_id_padding": 2,
        },
        "provider": {
            "selected_provider": flow_config.get("quality", {}).get("art_selected_provider", "openai"),
            "openai": {
                "api_key": str(flow_config.get("provider", {}).get("api_key", "")).strip(),
                "model": str(flow_config.get("provider", {}).get("model", "gpt-5.4")).strip(),
            },
            "gemini": {
                "api_key": str(flow_config.get("quality", {}).get("gemini_api_key", "")).strip(),
                "model": flow_config.get("quality", {}).get("art_gemini_model", "gemini-3-pro-preview"),
            },
        },
        "sources": {
            "analysis_provider": video_provider,
            "analysis_model": video_model,
            "analysis_path": "",
            "director_analysis_path": "",
            "director_outputs_root": flow_config.get("output", {}).get("outputs_root", "outputs"),
        },
        "quality": {
            "visual_style": flow_config.get("quality", {}).get(
                "visual_style",
                "真人写实，移动端9:16竖屏电影感构图，适合后续参考图工作流",
            ),
            "target_medium": flow_config.get("quality", {}).get("target_medium", "漫剧"),
            "frame_orientation": flow_config.get("quality", {}).get("frame_orientation", "9:16竖屏"),
            "extra_rules": flow_config.get("quality", {}).get("art_extra_rules", []),
        },
        "output": {
            "assets_series_name": flow_config.get("quality", {}).get("assets_series_name", f"{series_name}-gpt"),
            "assets_series_suffix": flow_config.get("quality", {}).get("assets_series_suffix", "-gpt"),
        },
        "run": {
            "analysis_root": flow_config.get("output", {}).get("analysis_root", "analysis"),
            "temperature": flow_config.get("runtime", {}).get("temperature", 0.25),
            "timeout_seconds": flow_config.get("runtime", {}).get("timeout_seconds", 300),
            "enable_review_pass": flow_config.get("runtime", {}).get("enable_review_pass", True),
            "dry_run": False,
        },
    }


def build_storyboard_config(
    *,
    series_name: str,
    script_path: Path,
    flow_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "script": {
            "series_dir": str(script_path.parent),
            "script_path": str(script_path),
            "episode_id": "ep01",
        },
        "series": {
            "series_name": series_name,
            "start_episode": 1,
            "end_episode": 1,
            "episode_id_prefix": "ep",
            "episode_id_padding": 2,
        },
        "provider": {
            "openai": {
                "api_key": str(flow_config.get("provider", {}).get("api_key", "")).strip(),
                "model": str(flow_config.get("provider", {}).get("model", "gpt-5.4")).strip(),
            }
        },
        "sources": {
            "director_analysis_path": "",
        },
        "quality": {
            "visual_style": flow_config.get("quality", {}).get(
                "visual_style",
                "真人写实，移动端9:16竖屏电影感构图，适合 Seedance 2.0 动态生成",
            ),
            "target_medium": flow_config.get("quality", {}).get("target_medium", "漫剧"),
            "frame_orientation": flow_config.get("quality", {}).get("frame_orientation", "9:16竖屏"),
            "extra_rules": flow_config.get("quality", {}).get("storyboard_extra_rules", []),
        },
        "output": {
            "outputs_root": flow_config.get("output", {}).get("outputs_root", "outputs"),
            "assets_series_name": flow_config.get("quality", {}).get("assets_series_name", f"{series_name}-gpt"),
            "assets_series_suffix": flow_config.get("quality", {}).get("assets_series_suffix", "-gpt"),
        },
        "run": {
            "temperature": flow_config.get("runtime", {}).get("temperature", 0.2),
            "timeout_seconds": flow_config.get("runtime", {}).get("timeout_seconds", 300),
            "enable_review_pass": flow_config.get("runtime", {}).get("enable_review_pass", True),
            "dry_run": False,
        },
    }


def build_nano_config(
    *,
    series_name: str,
    script_path: Path,
    flow_config: dict[str, Any],
    nano_config: dict[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(nano_config)
    assets_series_name = str(flow_config.get("quality", {}).get("assets_series_name", "")).strip() or f"{series_name}-gpt"
    config["script"]["script_path"] = str(script_path)
    config["script"]["series_name"] = series_name
    config["script"]["episode_id"] = "ep01"
    config["sources"]["character_prompts_path"] = str(PROJECT_ROOT / "assets" / assets_series_name / "character-prompts.md")
    config["sources"]["scene_prompts_path"] = str(PROJECT_ROOT / "assets" / assets_series_name / "scene-prompts.md")
    config["selection"]["include_all_history_assets"] = False
    config["run"]["dry_run"] = False
    config["run"]["skip_existing_images"] = False
    config["output"]["output_root"] = str(PROJECT_ROOT / "assets" / assets_series_name / "generated" / "benchmark" / "ep01")
    return config


def render_markdown_report(
    telemetry: TelemetryRecorder,
    *,
    series_name: str,
    episode_id: str,
    video_path: Path,
    video_provider: str,
    video_model: str,
    output_paths: dict[str, str],
) -> str:
    report = telemetry.to_dict()
    lines = [
        "# Workflow Benchmark Report",
        "",
        f"- 剧名：{series_name}",
        f"- 集数：{episode_id}",
        f"- 输入视频：{video_path}",
        f"- 视频理解 Provider：{video_provider}/{video_model}",
        "- 说明：当前视频理解链路采用 transcript / OCR / keyframes 聚合理解，不是把视频切成若干片段逐段送模型。",
        "",
        "## 总计",
        "",
        f"- 总步骤数：{report['totals']['step_count']}",
        f"- 总耗时（按步骤累计）：{report['totals']['duration_seconds']} 秒",
        f"- 总输入 tokens：{report['totals']['input_tokens']}",
        f"- 总输出 tokens：{report['totals']['output_tokens']}",
        f"- 总 tokens：{report['totals']['total_tokens']}",
        "",
        "## 分阶段汇总",
        "",
        "| 阶段 | 步骤数 | 耗时(秒) | 输入tokens | 输出tokens | 总tokens | 状态分布 |",
        "|------|--------|---------:|-----------:|-----------:|---------:|----------|",
    ]
    for stage, totals in report["stage_totals"].items():
        statuses = ", ".join(f"{key}:{value}" for key, value in sorted(totals["statuses"].items()))
        lines.append(
            f"| {stage} | {totals['step_count']} | {totals['duration_seconds']} | "
            f"{totals['input_tokens']} | {totals['output_tokens']} | {totals['total_tokens']} | {statuses} |"
        )

    lines.extend(
        [
            "",
            "## 关键产物",
            "",
            *[f"- {label}：{path}" for label, path in output_paths.items()],
            "",
            "## 细粒度步骤",
            "",
            "| Step ID | 阶段 | 名称 | 状态 | 耗时(秒) | 输入tokens | 输出tokens | 总tokens |",
            "|---------|------|------|------|---------:|-----------:|-----------:|---------:|",
        ]
    )
    for step in report["steps"]:
        lines.append(
            f"| {step['step_id']} | {step['stage']} | {step['name']} | {step['status']} | "
            f"{step['duration_seconds']} | {step['input_tokens']} | {step['output_tokens']} | {step['total_tokens']} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_known_output_paths(
    *,
    series_name: str,
    episode_id: str,
    video_provider: str,
    video_model: str,
    assets_series_name: str,
) -> dict[str, str]:
    provider_tag = build_provider_model_tag(video_provider, video_model)
    return {
        "preprocess_manifest": str(PROJECT_ROOT / "analysis" / series_name / episode_id / "preprocess" / "preprocess_manifest.json"),
        "episode_analysis": str(PROJECT_ROOT / "analysis" / series_name / episode_id / f"episode_analysis__{provider_tag}.json"),
        "script": str(PROJECT_ROOT / "script" / series_name / f"{episode_id}__{provider_tag}.md"),
        "director_markdown": str(PROJECT_ROOT / "outputs" / series_name / episode_id / "01-director-analysis.md"),
        "director_json": str(PROJECT_ROOT / "outputs" / series_name / episode_id / "01-director-analysis__openai__gpt-5.4.json"),
        "character_prompts": str(PROJECT_ROOT / "assets" / assets_series_name / "character-prompts.md"),
        "scene_prompts": str(PROJECT_ROOT / "assets" / assets_series_name / "scene-prompts.md"),
        "storyboard_markdown": str(PROJECT_ROOT / "outputs" / series_name / episode_id / "02-seedance-prompts.md"),
        "storyboard_json": str(PROJECT_ROOT / "outputs" / series_name / episode_id / "02-seedance-prompts__openai__gpt-5.4.json"),
        "nano_manifest": str(PROJECT_ROOT / "assets" / assets_series_name / "generated" / "benchmark" / episode_id / "generation_manifest.json"),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在：{video_path}")

    video_config = load_json_file(args.video_config)
    flow_config = load_json_file(args.openai_flow_config)
    nano_config = load_json_file(args.nano_config)
    ensure_provider_env(video_config, flow_config, nano_config)

    series_name = derive_series_folder_name(video_path=video_path)
    episode_id = args.episode_id
    title = args.title
    benchmark_dir = PROJECT_ROOT / "analysis" / series_name / episode_id / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    selected_video_provider = args.video_provider or str(video_config.get("run", {}).get("selected_provider", "openai")).strip()
    video_model = args.video_model or str(video_config.get("providers", {}).get(selected_video_provider, {}).get("model", "gpt-5.4")).strip()
    if selected_video_provider == "openai":
        os.environ["OPENAI_API_KEY"] = str(video_config.get("providers", {}).get("openai", {}).get("api_key", "")).strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if selected_video_provider == "gemini":
        os.environ["GEMINI_API_KEY"] = str(video_config.get("providers", {}).get("gemini", {}).get("api_key", "")).strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if selected_video_provider == "qwen":
        os.environ["DASHSCOPE_API_KEY"] = str(video_config.get("providers", {}).get("qwen", {}).get("api_key", "")).strip() or os.getenv("DASHSCOPE_API_KEY", "").strip()

    report_json_path = benchmark_dir / "workflow_benchmark_report.json"
    report_md_path = benchmark_dir / "workflow_benchmark_report.md"
    if args.nano_only and report_json_path.exists():
        existing_report = load_json_file(report_json_path)
        recorder = TelemetryRecorder(
            run_name=str(existing_report.get("run_name") or f"workflow-benchmark-{series_name}-{episode_id}"),
            context=dict(existing_report.get("context", {})),
            steps=list(existing_report.get("steps", [])),
            started_at=str(existing_report.get("started_at") or ""),
        )
    else:
        recorder = TelemetryRecorder(
            run_name=f"workflow-benchmark-{series_name}-{episode_id}",
            context={
                "series_name": series_name,
                "episode_id": episode_id,
                "video_path": str(video_path),
                "video_provider": selected_video_provider,
                "video_model": video_model,
            },
        )

    assets_series_name = str(flow_config.get("quality", {}).get("assets_series_name", "")).strip() or f"{series_name}-gpt"
    output_paths: dict[str, str] = {}
    output_paths.update(
        build_known_output_paths(
            series_name=series_name,
            episode_id=episode_id,
            video_provider=selected_video_provider,
            video_model=video_model,
            assets_series_name=assets_series_name,
        )
    )
    try:
        flow_config = copy.deepcopy(flow_config)
        flow_config.setdefault("provider", {})
        flow_config["provider"]["api_key"] = os.getenv("OPENAI_API_KEY", "").strip()
        flow_config["provider"]["model"] = flow_config.get("provider", {}).get("model", "gpt-5.4")
        flow_config.setdefault("quality", {})
        flow_config["quality"]["assets_series_name"] = assets_series_name
        flow_config.setdefault("output", {})
        flow_config["output"]["analysis_root"] = flow_config["output"].get("analysis_root", "analysis")
        flow_config["output"]["outputs_root"] = flow_config["output"].get("outputs_root", "outputs")
        script_path = Path(output_paths["script"]).expanduser().resolve()
        if not args.nano_only:
            preprocess_config = EpisodePreprocessConfig(
                output_root=Path(video_config["preprocess"]["output_root"]),
                asr_model_size=video_config["preprocess"]["asr_model_size"],
                asr_language=video_config["preprocess"]["asr_language"],
                asr_device=video_config["preprocess"]["asr_device"],
                asr_compute_type=video_config["preprocess"]["asr_compute_type"],
                asr_beam_size=int(video_config["preprocess"]["asr_beam_size"]),
                asr_best_of=int(video_config["preprocess"].get("asr_best_of", video_config["preprocess"]["asr_beam_size"])),
                asr_patience=float(video_config["preprocess"].get("asr_patience", 1.0)),
                asr_condition_on_previous_text=bool(video_config["preprocess"].get("asr_condition_on_previous_text", True)),
                asr_initial_prompt=str(video_config["preprocess"].get("asr_initial_prompt", "") or ""),
                asr_hotwords=str(video_config["preprocess"].get("asr_hotwords", "") or ""),
                asr_vad_filter=bool(video_config["preprocess"].get("asr_vad_filter", True)),
                asr_enable_dual_track_fusion=bool(video_config["preprocess"].get("asr_enable_dual_track_fusion", False)),
                asr_vad_threshold=float(video_config["preprocess"].get("asr_vad_threshold", 0.5)),
                asr_vad_neg_threshold=(
                    None
                    if video_config["preprocess"].get("asr_vad_neg_threshold") is None
                    else float(video_config["preprocess"].get("asr_vad_neg_threshold"))
                ),
                asr_vad_min_speech_duration_ms=int(video_config["preprocess"].get("asr_vad_min_speech_duration_ms", 0)),
                asr_vad_max_speech_duration_seconds=float(
                    video_config["preprocess"].get("asr_vad_max_speech_duration_seconds", 8.0)
                ),
                asr_vad_min_silence_duration_ms=int(video_config["preprocess"].get("asr_vad_min_silence_duration_ms", 700)),
                asr_vad_speech_pad_ms=int(video_config["preprocess"].get("asr_vad_speech_pad_ms", 320)),
                asr_no_speech_threshold=float(video_config["preprocess"].get("asr_no_speech_threshold", 0.45)),
                asr_log_prob_threshold=float(video_config["preprocess"].get("asr_log_prob_threshold", -1.0)),
                asr_enable_second_pass=bool(video_config["preprocess"].get("asr_enable_second_pass", True)),
                asr_second_pass_trigger_tolerance_seconds=float(
                    video_config["preprocess"].get("asr_second_pass_trigger_tolerance_seconds", 1.0)
                ),
                asr_second_pass_window_padding_seconds=float(
                    video_config["preprocess"].get("asr_second_pass_window_padding_seconds", 1.2)
                ),
                asr_second_pass_max_window_seconds=float(
                    video_config["preprocess"].get("asr_second_pass_max_window_seconds", 12.0)
                ),
                transcript_refine_enabled=bool(video_config["preprocess"].get("transcript_refine_enabled", False)),
                transcript_refine_provider=str(video_config["preprocess"].get("transcript_refine_provider", "qwen") or "qwen"),
                transcript_refine_model=str(video_config["preprocess"].get("transcript_refine_model", "") or ""),
                transcript_refine_endpoint=str(video_config["preprocess"].get("transcript_refine_endpoint", "") or ""),
                transcript_refine_timeout_seconds=int(video_config["preprocess"].get("transcript_refine_timeout_seconds", 180)),
                transcript_refine_batch_size=int(video_config["preprocess"].get("transcript_refine_batch_size", 10)),
                shot_threshold=float(video_config["preprocess"]["shot_threshold"]),
                shot_min_scene_len=int(video_config["preprocess"]["shot_min_scene_len"]),
                max_keyframes=video_config["preprocess"].get("max_keyframes"),
                ocr_crop_bottom_ratio=float(video_config["preprocess"]["ocr_crop_bottom_ratio"]),
            )
            preprocessor = EpisodePreprocessor(preprocess_config)
            print_status(f"开始 benchmark：{video_path}")
            preprocess_result = preprocessor.run(video_path, episode_id, series_name, telemetry=recorder)
            preprocess_manifest = load_json_file(preprocess_result.manifest_path)
            output_paths["preprocess_manifest"] = str(preprocess_result.manifest_path)

            with telemetry_span(
                recorder,
                stage="video_to_script",
                name="build_episode_input_bundle",
                metadata={"episode_id": episode_id, "manifest_path": str(preprocess_result.manifest_path)},
            ) as step:
                bundle = build_bundle_from_manifest(
                    episode_id=episode_id,
                    title=title,
                    video_path=video_path,
                    preprocess_manifest=preprocess_manifest,
                )
                step["metadata"]["frame_count"] = len(bundle.frames)
                step["metadata"]["transcript_chars"] = len(bundle.transcript_text or "")
                step["metadata"]["ocr_chars"] = len(bundle.ocr_text or "")

            pipeline = VideoToScriptPipeline(
                PipelineConfig(
                    provider=selected_video_provider,
                    model=video_model,
                    schema_path=Path("schemas/episode_analysis.schema.json"),
                    analysis_root=Path(video_config["run"].get("analysis_root", "analysis")),
                    script_root=Path(video_config["run"].get("script_root", "script")),
                    temperature=float(video_config["run"].get("temperature", 0.2)),
                    timeout_seconds=int(args.video_timeout_seconds),
                    telemetry=recorder,
                )
            )
            video_result = pipeline.run(bundle)
            output_paths["episode_analysis"] = str(video_result.analysis_path)
            output_paths["script"] = str(video_result.script_path)
            script_path = video_result.script_path

            director_summary = run_director_analysis_pipeline(
                build_director_config(series_name=series_name, script_path=script_path, flow_config=flow_config),
                telemetry=recorder,
            )
            first_director = director_summary.get("results", [{}])[0]
            output_paths["director_markdown"] = str(first_director.get("director_markdown_path", ""))
            output_paths["director_json"] = str(first_director.get("director_json_path", ""))

            art_summary = run_art_assets_pipeline(
                build_art_config(
                    series_name=series_name,
                    script_path=script_path,
                    video_provider=selected_video_provider,
                    video_model=video_model,
                    flow_config=flow_config,
                ),
                telemetry=recorder,
            )
            output_paths["character_prompts"] = str(art_summary.get("character_prompts_path", ""))
            output_paths["scene_prompts"] = str(art_summary.get("scene_prompts_path", ""))

            storyboard_summary = run_storyboard_pipeline(
                build_storyboard_config(series_name=series_name, script_path=script_path, flow_config=flow_config),
                telemetry=recorder,
            )
            first_storyboard = storyboard_summary.get("results", [{}])[0]
            output_paths["storyboard_markdown"] = str(first_storyboard.get("storyboard_markdown_path", ""))
            output_paths["storyboard_json"] = str(first_storyboard.get("storyboard_json_path", ""))

        nano_summary = run_nano_banana_pipeline(
            build_nano_config(
                series_name=series_name,
                script_path=script_path,
                flow_config=flow_config,
                nano_config=nano_config,
            ),
            telemetry=recorder,
        )
        output_paths["nano_manifest"] = str(Path(nano_summary["output_root"]).resolve() / "generation_manifest.json")
        recorder.context["final_status"] = "completed"
    except Exception as exc:
        recorder.context["final_status"] = "failed"
        recorder.context["failure_type"] = type(exc).__name__
        recorder.context["failure_message"] = str(exc)
        raise
    finally:
        recorder.save_json(report_json_path)
        save_text_file(
            report_md_path,
            render_markdown_report(
                recorder,
                series_name=series_name,
                episode_id=episode_id,
                video_path=video_path,
                video_provider=selected_video_provider,
                video_model=video_model,
                output_paths=output_paths,
            ),
        )

    print_status(f"benchmark 完成：{report_json_path}")
    print(json.dumps({"report_json_path": str(report_json_path), "report_md_path": str(report_md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
