from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from seedance_learning import generate_episode_beat_catalog, is_seedance_learning_enabled


DEFAULT_CONFIG_PATH = Path("config/video_pipeline.local.json")


class PipelineExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        report: dict[str, Any] | None = None,
        json_path: Path | None = None,
        markdown_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.report = report or {}
        self.json_path = str(json_path) if json_path else ""
        self.markdown_path = str(markdown_path) if markdown_path else ""


def print_status(message: str) -> None:
    print(f"[video-pipeline] {message}", flush=True)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_runtime_config(path: str | Path) -> dict[str, Any]:
    config = load_json_file(path)
    base_path = config.get("base_config")
    if base_path:
        base_config = load_json_file(base_path)
        return deep_merge(base_config, config)
    return config


def parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def derive_qwen_input_mode(bundle: EpisodeInputBundle | None) -> str:
    if not bundle:
        return ""
    if bundle.frames:
        return "frames_sequence"
    if bundle.video_path:
        return "raw_video"
    return "text_only"


def derive_qwen_selected_frame_count(
    bundle: EpisodeInputBundle | None,
    max_analysis_frames: int | None,
) -> int | None:
    if not bundle:
        return None
    available = len(bundle.frames)
    if available <= 0:
        return 0
    if max_analysis_frames is None or max_analysis_frames <= 0:
        return available
    return min(available, int(max_analysis_frames))


def build_frame_note(item: dict[str, Any]) -> str:
    parts: list[str] = [str(item.get("scene_id", "")).strip()]
    linked_ocr_text = str(item.get("linked_ocr_text", "")).strip()
    if linked_ocr_text:
        shortened = linked_ocr_text if len(linked_ocr_text) <= 72 else linked_ocr_text[:71].rstrip() + "…"
        parts.append(f"OCR:{shortened}")
    return " | ".join(part for part in parts if part)


def _truncate_text(text: str | None, max_chars: int = 240) -> str:
    clean = str(text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"


def print_resize_feedback(preprocess_manifest: dict[str, Any] | None) -> None:
    if not preprocess_manifest:
        return
    resize_summary = dict(preprocess_manifest.get("keyframe_resize_summary", {}))
    purged_artifacts = dict(preprocess_manifest.get("purged_artifacts", {}))
    if not resize_summary:
        return
    original_frames_dir = str(resize_summary.get("original_frames_dir", "")).strip()
    model_frames_dir = str(resize_summary.get("model_frames_dir", "")).strip()
    example = dict(resize_summary.get("example", {}))
    resized_count = int(resize_summary.get("resized_count", 0) or 0)
    scale_ratio = resize_summary.get("scale_ratio", 1.0)
    if resized_count > 0:
        print_status(
            "关键帧缩放反馈："
            f"scale_ratio={scale_ratio} | "
            f"示例={example.get('original', '')} -> {example.get('model', '')}"
        )
        if original_frames_dir:
            suffix = "（已清理）" if purged_artifacts.get("original_frames_deleted") else ""
            print_status(f"关键帧原图目录{suffix}：{original_frames_dir}")
        if model_frames_dir:
            print_status(f"关键帧模型输入图目录：{model_frames_dir}")
    else:
        print_status(
            "关键帧缩放反馈：未改变分辨率，但模型已切换为使用专用模型图目录。"
            + (f" 原图目录{'（已清理）' if purged_artifacts.get('original_frames_deleted') else ''}：{original_frames_dir}" if original_frames_dir else "")
        )
        if model_frames_dir:
            print_status(f"关键帧模型输入图目录：{model_frames_dir}")
    ocr_frames_dir = str(preprocess_manifest.get("ocr_frames_dir", "")).strip()
    if ocr_frames_dir and purged_artifacts.get("ocr_sample_frames_deleted"):
        print_status(f"OCR 取样图目录（已清理）：{ocr_frames_dir}")


def configure_provider_env(config: dict[str, Any]) -> tuple[str, str]:
    run_config = config["run"]
    providers = config["providers"]
    selected_provider = run_config["selected_provider"]
    provider_config = providers[selected_provider]
    api_key = provider_config.get("api_key", "").strip()
    if not run_config.get("only_preprocess", False) and not api_key:
        raise RuntimeError(
            f"{selected_provider} 的 api_key 为空。请先编辑 config/video_pipeline.local.json，或者把 run.only_preprocess 设为 true。"
        )
    if selected_provider == "openai" and api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if selected_provider == "gemini" and api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    if selected_provider == "qwen" and api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key
    return selected_provider, provider_config["model"]


def preprocess_if_needed(
    config: dict[str, Any],
    telemetry: TelemetryRecorder | None = None,
) -> dict[str, Any] | None:
    run_config = config["run"]
    if run_config.get("skip_preprocess", False):
        print_status("已跳过预处理，直接进入模型分析阶段。")
        with telemetry_span(
            telemetry,
            stage="preprocess",
            name="skip_preprocess",
            metadata={"episode_id": run_config.get("episode_id", ""), "video_path": run_config.get("video_path", "")},
        ) as step:
            step["status"] = "skipped"
        return None

    series_folder = derive_series_folder_name(
        video_path=run_config["video_path"],
        explicit_series_name=run_config.get("series_name"),
    )
    manifest_path = (
        Path(config["preprocess"]["output_root"])
        / series_folder
        / run_config["episode_id"]
        / "preprocess"
        / "preprocess_manifest.json"
    )
    if run_config.get("reuse_preprocess_if_exists", True) and manifest_path.exists():
        print_status(f"检测到已有预处理缓存，直接复用：{manifest_path}")
        with telemetry_span(
            telemetry,
            stage="preprocess",
            name="reuse_preprocess_cache",
            metadata={
                "episode_id": run_config.get("episode_id", ""),
                "video_path": run_config.get("video_path", ""),
                "manifest_path": str(manifest_path),
            },
        ) as step:
            step["status"] = "cached"
        preprocess_manifest = load_json_file(manifest_path)
        preprocess_manifest["manifest_path"] = str(manifest_path)
        print_resize_feedback(preprocess_manifest)
        return preprocess_manifest

    preprocess_config = config["preprocess"]
    print_status(
        "开始预处理：抽音频 / ASR / 分镜切分 / 关键场景选取 / OCR / 关键帧。"
    )
    preprocessor = EpisodePreprocessor(
        EpisodePreprocessConfig(
            output_root=Path(preprocess_config["output_root"]),
            asr_model_size=preprocess_config["asr_model_size"],
            asr_language=preprocess_config["asr_language"],
            asr_device=preprocess_config["asr_device"],
            asr_compute_type=preprocess_config["asr_compute_type"],
            asr_beam_size=int(preprocess_config["asr_beam_size"]),
            asr_best_of=int(preprocess_config.get("asr_best_of", preprocess_config["asr_beam_size"])),
            asr_patience=float(preprocess_config.get("asr_patience", 1.0)),
            asr_condition_on_previous_text=bool(preprocess_config.get("asr_condition_on_previous_text", True)),
            asr_initial_prompt=str(preprocess_config.get("asr_initial_prompt", "") or ""),
            asr_hotwords=str(preprocess_config.get("asr_hotwords", "") or ""),
            asr_vad_filter=bool(preprocess_config.get("asr_vad_filter", True)),
            asr_enable_dual_track_fusion=bool(preprocess_config.get("asr_enable_dual_track_fusion", False)),
            asr_vad_threshold=float(preprocess_config.get("asr_vad_threshold", 0.5)),
            asr_vad_neg_threshold=(
                None
                if preprocess_config.get("asr_vad_neg_threshold") is None
                else float(preprocess_config.get("asr_vad_neg_threshold"))
            ),
            asr_vad_min_speech_duration_ms=int(preprocess_config.get("asr_vad_min_speech_duration_ms", 0)),
            asr_vad_max_speech_duration_seconds=float(
                preprocess_config.get("asr_vad_max_speech_duration_seconds", 8.0)
            ),
            asr_vad_min_silence_duration_ms=int(preprocess_config.get("asr_vad_min_silence_duration_ms", 700)),
            asr_vad_speech_pad_ms=int(preprocess_config.get("asr_vad_speech_pad_ms", 320)),
            asr_no_speech_threshold=float(preprocess_config.get("asr_no_speech_threshold", 0.45)),
            asr_log_prob_threshold=float(preprocess_config.get("asr_log_prob_threshold", -1.0)),
            asr_enable_second_pass=bool(preprocess_config.get("asr_enable_second_pass", True)),
            asr_second_pass_trigger_tolerance_seconds=float(
                preprocess_config.get("asr_second_pass_trigger_tolerance_seconds", 1.0)
            ),
            asr_second_pass_window_padding_seconds=float(
                preprocess_config.get("asr_second_pass_window_padding_seconds", 1.2)
            ),
            asr_second_pass_max_window_seconds=float(
                preprocess_config.get("asr_second_pass_max_window_seconds", 12.0)
            ),
            transcript_refine_enabled=bool(preprocess_config.get("transcript_refine_enabled", False)),
            transcript_refine_provider=str(preprocess_config.get("transcript_refine_provider", "qwen") or "qwen"),
            transcript_refine_model=str(preprocess_config.get("transcript_refine_model", "") or ""),
            transcript_refine_endpoint=str(preprocess_config.get("transcript_refine_endpoint", "") or ""),
            transcript_refine_timeout_seconds=int(preprocess_config.get("transcript_refine_timeout_seconds", 180)),
            transcript_refine_batch_size=int(preprocess_config.get("transcript_refine_batch_size", 10)),
            shot_threshold=float(preprocess_config["shot_threshold"]),
            shot_min_scene_len=int(preprocess_config["shot_min_scene_len"]),
            max_keyframes=parse_optional_int(preprocess_config.get("max_keyframes")),
            ocr_crop_bottom_ratio=float(preprocess_config["ocr_crop_bottom_ratio"]),
            ocr_sample_interval_seconds=float(preprocess_config.get("ocr_sample_interval_seconds", 0.5)),
            keyframe_scale_ratio=float(preprocess_config.get("keyframe_scale_ratio", 1.0)),
        )
    )
    result = preprocessor.run(
        run_config["video_path"],
        run_config["episode_id"],
        run_config.get("series_name"),
        telemetry=telemetry,
    )
    print_status(f"预处理完成：{result.manifest_path}")
    preprocess_manifest = load_json_file(result.manifest_path)
    preprocess_manifest["manifest_path"] = str(result.manifest_path)
    print_resize_feedback(preprocess_manifest)
    return preprocess_manifest


def build_bundle(config: dict[str, Any], preprocess_manifest: dict[str, Any] | None) -> EpisodeInputBundle:
    run_config = config["run"]

    transcript_text = ""
    ocr_text = ""
    frames: list[FrameReference] = []
    context_notes: list[str] = []

    if preprocess_manifest:
        corrected_transcript_path = preprocess_manifest.get("corrected_transcript_text_path")
        transcript_text = (
            read_text_file(corrected_transcript_path) if corrected_transcript_path else ""
        ) or read_text_file(preprocess_manifest["transcript_text_path"]) or ""
        ocr_text = read_text_file(preprocess_manifest["ocr_text_path"]) or ""
        correction_summary = dict(preprocess_manifest.get("dialogue_correction_summary", {}))
        if correction_summary.get("enabled"):
            context_notes.append(
                "对白证据策略：以 ASR Transcript 为主，仅在 OCR 与 ASR 高度重合且明显属于个别错别字时，"
                f"才做保守校字。本次自动校正段数：{int(correction_summary.get('corrected_segment_count', 0))}。"
            )
        frames = [
            FrameReference(
                path=item.get("model_frame_path", item["frame_path"]),
                timestamp=str(item["midpoint_seconds"]),
                note=build_frame_note(item),
            )
            for item in preprocess_manifest.get("keyframes", [])
        ]

    genre_hints = dict(run_config.get("genre_hints", {}))
    library_keys = [str(item).strip() for item in genre_hints.get("library_keys", []) if str(item).strip()]
    custom_tokens = [str(item).strip() for item in genre_hints.get("custom_tokens", []) if str(item).strip()]
    ai_suggested_keys = [str(item).strip() for item in genre_hints.get("ai_suggested_keys", []) if str(item).strip()]
    if library_keys or custom_tokens:
        context_notes.append(
            "用户已确认本剧题材："
            + ("、".join(library_keys) if library_keys else "自定义 " + "、".join(custom_tokens))
            + "。这些题材是前置提示，只能作为先验参考；若与当前视频证据冲突，必须以当前视频为准。"
        )

    return EpisodeInputBundle(
        episode_id=run_config["episode_id"],
        title=run_config.get("title"),
        video_path=run_config["video_path"],
        transcript_text=transcript_text or None,
        ocr_text=ocr_text or None,
        frames=frames,
        context_notes=context_notes,
        language=run_config.get("language", "zh-CN"),
        metadata={
            "source_series": run_config.get("series_name", ""),
            "user_genre_hints": library_keys,
            "user_custom_genre_hints": custom_tokens,
            "ai_suggested_genre_hints": ai_suggested_keys,
            "genre_hint_source": str(genre_hints.get("source", "")).strip(),
        },
    )


def run_pipeline(
    config: dict[str, Any],
    provider: str,
    model: str,
    bundle: EpisodeInputBundle,
    telemetry: TelemetryRecorder | None = None,
) -> dict[str, Any]:
    run_config = config["run"]
    series_folder = derive_series_folder_name(
        video_path=run_config["video_path"],
        explicit_series_name=run_config.get("series_name"),
    )
    output_tag = build_provider_model_tag(provider, model)
    pipeline = VideoToScriptPipeline(
        PipelineConfig(
            provider=provider,
            model=model,
            schema_path=Path("schemas/episode_analysis.schema.json"),
            analysis_root=Path(run_config.get("analysis_root", "analysis")),
            script_root=Path(run_config.get("script_root", "script")),
            temperature=float(run_config.get("temperature", 0.2)),
            timeout_seconds=int(run_config.get("timeout_seconds", 180)),
            provider_endpoint=str(config.get("providers", {}).get(provider, {}).get("endpoint", "")).strip(),
            openai_image_detail=str(
                config.get("providers", {}).get("openai", {}).get(
                    "image_detail",
                    run_config.get("openai_image_detail", "auto"),
                )
            ).strip()
            or "auto",
            openai_max_analysis_frames=parse_optional_int(
                config.get("providers", {}).get("openai", {}).get(
                    "max_analysis_frames",
                    run_config.get("openai_max_analysis_frames", 20),
                )
            ),
            qwen_max_analysis_frames=parse_optional_int(
                config.get("providers", {}).get("qwen", {}).get(
                    "max_analysis_frames",
                    run_config.get("qwen_max_analysis_frames", 20),
                )
            ),
            qwen_video_fps=float(
                config.get("providers", {}).get("qwen", {}).get(
                    "video_fps",
                    run_config.get("qwen_video_fps", 2.0),
                )
            ),
            qwen_structured_output_mode=str(
                config.get("providers", {}).get("qwen", {}).get(
                    "structured_output_mode",
                    run_config.get("qwen_structured_output_mode", "json_schema"),
                )
            ).strip()
            or "json_schema",
            telemetry=telemetry,
        )
    )
    print_status(f"开始调用 {provider} / {model} 做视频理解与剧本重建。")
    if provider == "openai":
        print_status(
            "OpenAI 图片输入设置：detail="
            + str(
                config.get("providers", {}).get("openai", {}).get(
                    "image_detail",
                    run_config.get("openai_image_detail", "auto"),
                )
            )
        )
        print_status(
            "OpenAI 分析帧上限："
            + str(
                config.get("providers", {}).get("openai", {}).get(
                    "max_analysis_frames",
                    run_config.get("openai_max_analysis_frames", 20),
                )
            )
        )
    if provider == "qwen":
        qwen_max_analysis_frames = parse_optional_int(
            config.get("providers", {}).get("qwen", {}).get(
                "max_analysis_frames",
                run_config.get("qwen_max_analysis_frames", 20),
            )
        )
        qwen_video_fps = float(
            config.get("providers", {}).get("qwen", {}).get(
                "video_fps",
                run_config.get("qwen_video_fps", 2.0),
            )
        )
        qwen_structured_output_mode = (
            str(
                config.get("providers", {}).get("qwen", {}).get(
                    "structured_output_mode",
                    run_config.get("qwen_structured_output_mode", "json_schema"),
                )
            ).strip()
            or "json_schema"
        )
        print_status(
            "Qwen 视频理解设置：max_analysis_frames="
            + str(qwen_max_analysis_frames)
            + " | video_fps="
            + str(qwen_video_fps)
            + " | structured_output_mode="
            + qwen_structured_output_mode
        )
        print_status(
            "Qwen 输入模式："
            + derive_qwen_input_mode(bundle)
            + " | selected_frames="
            + str(derive_qwen_selected_frame_count(bundle, qwen_max_analysis_frames))
            + " | available_frames="
            + str(len(bundle.frames))
        )
    result = pipeline.run(bundle)
    summary = {
        "provider": provider,
        "model": model,
        "analysis_path": str(result.analysis_path),
        "script_path": str(result.script_path),
        "continuity_paths": {key: str(value) for key, value in result.continuity_paths.items()},
        "genre_debug_paths": {key: str(value) for key, value in result.genre_debug_paths.items()},
        "genre_debug_summary": {
            "primary_genre": result.genre_debug_summary.get("primary_genre", ""),
            "secondary_genres": result.genre_debug_summary.get("secondary_genres", []),
            "confirmed_user_genres": result.genre_debug_summary.get("confirmed_user_genres", []),
            "genre_resolution_mode": result.genre_debug_summary.get("genre_resolution_mode", ""),
            "genre_override_request": result.genre_debug_summary.get("genre_override_request", {}),
            "matched_playbooks": result.genre_debug_summary.get("matched_playbooks", []),
        },
    }
    summary_path = (
        Path(run_config.get("analysis_root", "analysis"))
        / series_folder
        / bundle.episode_id
        / f"run_summary__{output_tag}.json"
    )
    with telemetry_span(
        telemetry,
        stage="video_to_script",
        name="save_video_pipeline_run_summary",
        metadata={"summary_path": str(summary_path), "episode_id": bundle.episode_id},
    ):
        save_json_file(summary_path, summary)
    genre_name = str(result.genre_debug_summary.get("primary_genre", "") or "未明确")
    secondary = "、".join(result.genre_debug_summary.get("secondary_genres", []))
    pre_route = dict(result.genre_debug_summary.get("pre_analysis_routing", {}))
    post_route = dict(result.genre_debug_summary.get("post_analysis_routing", {}))
    print_status(
        f"题材反馈：主题材={genre_name}"
        + (f" | 副题材={secondary}" if secondary else "")
    )
    confirmed = result.genre_debug_summary.get("confirmed_user_genres", [])
    if confirmed:
        print_status("题材反馈：用户确认题材=" + "、".join(confirmed))
        print_status("题材反馈：决议模式=" + str(result.genre_debug_summary.get("genre_resolution_mode", "")))
    print_status(
        f"题材路由：基础 skill={result.genre_debug_summary.get('core_skill_path', '')}"
    )
    print_status(
        f"题材路由：playbook 源目录={result.genre_debug_summary.get('playbook_library_path', '')}"
    )
    print_status(
        "题材路由：预判线索="
        + ("、".join(pre_route.get("route_tokens", [])) if pre_route.get("route_tokens") else "未命中")
    )
    print_status(
        "题材路由：题材 skill="
        + ("；".join(pre_route.get("genre_skill_paths", [])) if pre_route.get("genre_skill_paths") else "未加载补充 skill")
    )
    matched = [
        str(item.get("genre_key", "")).strip()
        for item in result.genre_debug_summary.get("matched_playbooks", [])
        if isinstance(item, dict) and str(item.get("genre_key", "")).strip()
    ]
    print_status(
        f"题材经验：{'、'.join(matched) if matched else '未命中具体 playbook，使用全库参考'}"
    )
    if post_route.get("matched_playbook_keys"):
        print_status(
            f"题材经验：分析后最终命中={ '、'.join(post_route.get('matched_playbook_keys', [])) }"
        )
    override_request = dict(result.genre_debug_summary.get("genre_override_request", {}))
    if override_request.get("needs_user_confirmation"):
        proposed = [
            str(item).strip()
            for item in [
                override_request.get("proposed_primary_genre", ""),
                *override_request.get("proposed_secondary_genres", []),
                *override_request.get("proposed_new_genres", []),
            ]
            if str(item).strip()
        ]
        print_status("题材修正建议：AI 建议新增/修正题材=" + "、".join(proposed))
        print_status("题材修正建议：" + str(override_request.get("reason", "")))
    print_status(f"题材调试报告：{result.genre_debug_paths['markdown_path']}")
    if is_seedance_learning_enabled(config):
        try:
            learning_artifacts = generate_episode_beat_catalog(
                project_root=PROJECT_ROOT,
                series_name=series_folder,
                episode_id=bundle.episode_id,
                analysis_path=result.analysis_path,
                config=config,
                progress_callback=print_status,
            )
            summary["seedance_learning"] = learning_artifacts
            print_status(
                f"Seedance 分段：{learning_artifacts['beat_segmentation_json_path']} "
                f"｜学习目录：{learning_artifacts['catalog_json_path']} "
                f"（raw_shots={learning_artifacts.get('shot_count', 0)}｜beats={learning_artifacts['beat_count']}）"
            )
            print_status(f"Seedance shot 帧目录：{learning_artifacts.get('shot_frame_dir', '')}")
            print_status(f"Seedance beat 帧目录：{learning_artifacts.get('beat_frame_dir', '')}")
            print_status(f"Seedance 二次视觉目录：{learning_artifacts.get('visual_second_pass_dir', '')}")
        except Exception as exc:  # pragma: no cover - learning artifacts are best-effort
            summary["seedance_learning_error"] = str(exc)
            print_status(f"Seedance 学习产物生成失败，但不影响主流程：{exc}")
    return summary


def build_failure_report_paths(config: dict[str, Any], provider: str, model: str) -> tuple[Path, Path]:
    run_config = config["run"]
    series_folder = derive_series_folder_name(
        video_path=run_config["video_path"],
        explicit_series_name=run_config.get("series_name"),
    )
    output_tag = build_provider_model_tag(provider, model)
    base_dir = (
        Path(run_config.get("analysis_root", "analysis"))
        / series_folder
        / run_config["episode_id"]
        / "failures"
    ).expanduser().resolve()
    return (
        base_dir / f"pipeline_failure__{output_tag}.json",
        base_dir / f"pipeline_failure__{output_tag}.md",
    )


def build_failure_report(
    config: dict[str, Any],
    provider: str,
    model: str,
    preprocess_manifest: dict[str, Any] | None,
    bundle: EpisodeInputBundle | None,
    exc: Exception,
) -> dict[str, Any]:
    run_config = config["run"]
    provider_config = dict(config.get("providers", {}).get(provider, {}))
    qwen_max_analysis_frames = parse_optional_int(
        provider_config.get("max_analysis_frames", run_config.get("qwen_max_analysis_frames", 20))
    )
    keyframes = list((preprocess_manifest or {}).get("keyframes", []))
    timing_breakdown = dict((preprocess_manifest or {}).get("timing_breakdown_seconds", {}))
    resize_summary = dict((preprocess_manifest or {}).get("keyframe_resize_summary", {}))
    transcript_text = bundle.transcript_text if bundle else None
    ocr_text = bundle.ocr_text if bundle else None
    hints: list[str] = []
    if provider == "openai" and len(keyframes) >= 30:
        hints.append("当前关键帧数量较多，OpenAI 路线更容易因多图输入导致超时；可尝试降低 max_keyframes。")
    if provider == "openai":
        hints.append("如仍然超时，可提高 run.timeout_seconds，或先切到 Gemini 做视频理解。")
    if provider == "qwen" and len(keyframes) >= 30:
        hints.append("当前关键帧数量较多，Qwen 路线会把关键帧作为时序视频片段送入模型；可尝试降低 qwen.max_analysis_frames 或 preprocess.max_keyframes。")
    if resize_summary.get("scale_ratio", 1.0) >= 0.95:
        hints.append("当前关键帧几乎未缩放；若仍担心上传负担，可适当降低 keyframe_scale_ratio。")
    if not hints:
        hints.append("优先检查 timeout_seconds、max_keyframes、shot_threshold 以及 OCR/ASR 产物是否异常。")
    return {
        "status": "failed",
        "series_name": run_config.get("series_name") or derive_series_folder_name(video_path=run_config["video_path"]),
        "episode_id": run_config["episode_id"],
        "title": run_config.get("title", ""),
        "provider": provider,
        "model": model,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "runtime_config": {
            "timeout_seconds": int(run_config.get("timeout_seconds", 180)),
            "temperature": float(run_config.get("temperature", 0.2)),
            "provider_endpoint": str(provider_config.get("endpoint", "")),
            "openai_image_detail": str(provider_config.get("image_detail", "")) if provider == "openai" else "",
            "openai_max_analysis_frames": (
                parse_optional_int(provider_config.get("max_analysis_frames", run_config.get("openai_max_analysis_frames", 20)))
                if provider == "openai"
                else None
            ),
            "qwen_max_analysis_frames": (
                qwen_max_analysis_frames
                if provider == "qwen"
                else None
            ),
            "qwen_video_fps": (
                float(provider_config.get("video_fps", run_config.get("qwen_video_fps", 2.0)))
                if provider == "qwen"
                else None
            ),
            "qwen_structured_output_mode": (
                str(
                    provider_config.get(
                        "structured_output_mode",
                        run_config.get("qwen_structured_output_mode", "json_schema"),
                    )
                ).strip()
                if provider == "qwen"
                else ""
            ),
            "qwen_input_mode": derive_qwen_input_mode(bundle) if provider == "qwen" else "",
            "qwen_selected_frame_count": (
                derive_qwen_selected_frame_count(bundle, qwen_max_analysis_frames)
                if provider == "qwen"
                else None
            ),
        },
        "input_summary": {
            "video_path": run_config.get("video_path", ""),
            "transcript_chars": len(transcript_text or ""),
            "ocr_chars": len(ocr_text or ""),
            "frame_count": len(bundle.frames) if bundle else 0,
            "frame_examples": [
                {
                    "file_name": Path(frame.path).name,
                    "timestamp": frame.timestamp,
                    "note": frame.note,
                }
                for frame in (bundle.frames[:5] if bundle else [])
            ],
            "context_note_count": len(bundle.context_notes) if bundle else 0,
        },
        "preprocess_summary": {
            "manifest_path": str((preprocess_manifest or {}).get("manifest_path", "")),
            "duration_seconds": (preprocess_manifest or {}).get("duration_seconds"),
            "scene_count": len(load_json_file((preprocess_manifest or {}).get("scene_list_path", ""))["scenes"])
            if preprocess_manifest and preprocess_manifest.get("scene_list_path")
            else None,
            "keyframe_count": len(keyframes),
            "ocr_sample_interval_seconds": (preprocess_manifest or {}).get("ocr_sample_interval_seconds"),
            "keyframe_scale_ratio": (preprocess_manifest or {}).get("keyframe_scale_ratio"),
            "resize_summary": resize_summary,
            "timing_breakdown_seconds": timing_breakdown,
        },
        "artifact_paths": {
            "analysis_root": str(Path(run_config.get("analysis_root", "analysis")).expanduser().resolve()),
            "script_root": str(Path(run_config.get("script_root", "script")).expanduser().resolve()),
            "model_frames_dir": str((preprocess_manifest or {}).get("model_frames_dir", "")),
            "transcript_text_path": str((preprocess_manifest or {}).get("transcript_text_path", "")),
            "corrected_transcript_text_path": str(
                (preprocess_manifest or {}).get("corrected_transcript_text_path", "")
            ),
            "ocr_text_path": str((preprocess_manifest or {}).get("ocr_text_path", "")),
        },
        "debug_hints": hints,
    }


def render_failure_markdown(report: dict[str, Any]) -> str:
    runtime = dict(report.get("runtime_config", {}))
    input_summary = dict(report.get("input_summary", {}))
    preprocess_summary = dict(report.get("preprocess_summary", {}))
    artifact_paths = dict(report.get("artifact_paths", {}))
    lines = [
        "# 视频到剧本失败调试报告",
        "",
        f"- 剧名：{report.get('series_name', '')}",
        f"- 集数：{report.get('episode_id', '')}",
        f"- 标题：{report.get('title', '')}",
        f"- provider/model：{report.get('provider', '')}/{report.get('model', '')}",
        f"- 错误类型：{report.get('error_type', '')}",
        f"- 错误信息：{report.get('error_message', '')}",
        "",
        "## 运行配置",
        "",
        f"- timeout_seconds：{runtime.get('timeout_seconds', '')}",
        f"- temperature：{runtime.get('temperature', '')}",
        f"- provider_endpoint：{runtime.get('provider_endpoint', '') or '默认'}",
        f"- openai_image_detail：{runtime.get('openai_image_detail', '') or '无'}",
        f"- openai_max_analysis_frames：{runtime.get('openai_max_analysis_frames', '') or '无'}",
        f"- qwen_max_analysis_frames：{runtime.get('qwen_max_analysis_frames', '') or '无'}",
        f"- qwen_video_fps：{runtime.get('qwen_video_fps', '') or '无'}",
        f"- qwen_structured_output_mode：{runtime.get('qwen_structured_output_mode', '') or '无'}",
        f"- qwen_input_mode：{runtime.get('qwen_input_mode', '') or '无'}",
        f"- qwen_selected_frame_count：{runtime.get('qwen_selected_frame_count', '') or '无'}",
        "",
        "## 输入概况",
        "",
        f"- video_path：{input_summary.get('video_path', '')}",
        f"- transcript_chars：{input_summary.get('transcript_chars', 0)}",
        f"- ocr_chars：{input_summary.get('ocr_chars', 0)}",
        f"- frame_count：{input_summary.get('frame_count', 0)}",
        "",
        "## 预处理概况",
        "",
        f"- preprocess_manifest：{preprocess_summary.get('manifest_path', '')}",
        f"- scene_count：{preprocess_summary.get('scene_count', '')}",
        f"- keyframe_count：{preprocess_summary.get('keyframe_count', '')}",
        f"- ocr_sample_interval_seconds：{preprocess_summary.get('ocr_sample_interval_seconds', '')}",
        f"- keyframe_scale_ratio：{preprocess_summary.get('keyframe_scale_ratio', '')}",
        f"- resize_example：{dict(preprocess_summary.get('resize_summary', {})).get('example', {})}",
        "",
        "## 产物路径",
        "",
        f"- model_frames_dir：{artifact_paths.get('model_frames_dir', '')}",
        f"- transcript_text_path：{artifact_paths.get('transcript_text_path', '')}",
        f"- corrected_transcript_text_path：{artifact_paths.get('corrected_transcript_text_path', '')}",
        f"- ocr_text_path：{artifact_paths.get('ocr_text_path', '')}",
        "",
        "## 排查建议",
        "",
    ]
    for hint in report.get("debug_hints", []):
        lines.append(f"- {hint}")
    lines.extend(
        [
            "",
            "## Traceback",
            "",
            "```text",
            _truncate_text(report.get("traceback", ""), 6000),
            "```",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def save_failure_report(
    config: dict[str, Any],
    provider: str,
    model: str,
    preprocess_manifest: dict[str, Any] | None,
    bundle: EpisodeInputBundle | None,
    exc: Exception,
) -> tuple[dict[str, Any], Path, Path]:
    report = build_failure_report(config, provider, model, preprocess_manifest, bundle, exc)
    json_path, md_path = build_failure_report_paths(config, provider, model)
    save_json_file(json_path, report)
    save_text_file(md_path, render_failure_markdown(report))
    return report, json_path, md_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run preprocess + video_to_script pipeline from one config file.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print_status(f"加载配置：{args.config}")
    config = load_runtime_config(args.config)
    provider, model = configure_provider_env(config)
    print_status(f"当前 provider={provider} model={model}")
    preprocess_manifest: dict[str, Any] | None = None
    bundle: EpisodeInputBundle | None = None
    try:
        preprocess_manifest = preprocess_if_needed(config)
        if config["run"].get("only_preprocess", False):
            print(
                json.dumps(
                    {
                        "status": "preprocess_only_completed",
                        "provider": provider,
                        "model": model,
                        "preprocess_manifest": preprocess_manifest,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        bundle = build_bundle(config, preprocess_manifest)
        summary = run_pipeline(config, provider, model, bundle)
        print_status("整条链路运行完成。")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        report, json_path, md_path = save_failure_report(
            config,
            provider,
            model,
            preprocess_manifest,
            bundle,
            exc,
        )
        print_status(f"运行失败：{report['error_message']}")
        print_status(
            "失败上下文："
            f"frame_count={report['input_summary'].get('frame_count', 0)} | "
            f"transcript_chars={report['input_summary'].get('transcript_chars', 0)} | "
            f"ocr_chars={report['input_summary'].get('ocr_chars', 0)} | "
            f"timeout={report['runtime_config'].get('timeout_seconds', '')}s"
        )
        print_status(f"失败调试报告：{md_path}")
        raise PipelineExecutionError(
            f"{report['error_message']} | 调试报告：{md_path}",
            report=report,
            json_path=json_path,
            markdown_path=md_path,
        ) from exc


if __name__ == "__main__":
    main()
