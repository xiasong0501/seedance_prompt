from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
from faster_whisper.audio import decode_audio
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from imageio_ffmpeg import count_frames_and_secs, get_ffmpeg_exe
from rapidocr_onnxruntime import RapidOCR
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

from pipeline_telemetry import TelemetryRecorder, telemetry_span
from prompt_utils import render_prompt
from providers.base import (
    ProviderResponseError,
    derive_series_folder_name,
    extract_json_from_text,
    save_json_file,
    save_text_file,
)
from providers.qwen_adapter import QwenAdapter


@dataclass
class EpisodePreprocessConfig:
    output_root: Path = Path("analysis")
    asr_model_size: str = "small"
    asr_language: str = "zh"
    asr_device: str = "auto"
    asr_compute_type: str = "auto"
    asr_beam_size: int = 5
    asr_best_of: int = 5
    asr_patience: float = 1.0
    asr_condition_on_previous_text: bool = True
    asr_initial_prompt: str = ""
    asr_hotwords: str = ""
    asr_vad_filter: bool = True
    asr_enable_dual_track_fusion: bool = False
    asr_vad_threshold: float = 0.5
    asr_vad_neg_threshold: float | None = None
    asr_vad_min_speech_duration_ms: int = 0
    asr_vad_max_speech_duration_seconds: float = 8.0
    asr_vad_min_silence_duration_ms: int = 700
    asr_vad_speech_pad_ms: int = 320
    asr_no_speech_threshold: float = 0.45
    asr_log_prob_threshold: float = -1.0
    asr_hallucination_silence_threshold: float = 0.8
    asr_vad_chunk_merge_gap_seconds: float = 0.2
    asr_vad_chunk_max_seconds: float = 12.0
    asr_enable_second_pass: bool = True
    asr_second_pass_trigger_tolerance_seconds: float = 1.0
    asr_second_pass_window_padding_seconds: float = 1.2
    asr_second_pass_max_window_seconds: float = 12.0
    transcript_refine_enabled: bool = False
    transcript_refine_provider: str = "qwen"
    transcript_refine_model: str = ""
    transcript_refine_endpoint: str = ""
    transcript_refine_timeout_seconds: int = 180
    transcript_refine_batch_size: int = 10
    shot_threshold: float = 27.0
    shot_min_scene_len: int = 15
    max_keyframes: int | None = None
    ocr_crop_bottom_ratio: float = 0.35
    ocr_sample_interval_seconds: float = 0.5
    keyframe_scale_ratio: float = 1.0
    ocr_dialogue_correction_enabled: bool = True
    ocr_dialogue_min_similarity: float = 0.82
    ocr_dialogue_max_mismatch_chars: int = 2
    ocr_dialogue_timing_tolerance_seconds: float = 0.45


@dataclass
class EpisodePreprocessResult:
    preprocess_dir: Path
    audio_path: Path
    transcript_json_path: Path
    transcript_text_path: Path
    corrected_transcript_json_path: Path
    corrected_transcript_text_path: Path
    scene_list_path: Path
    frames_dir: Path
    ocr_json_path: Path
    ocr_text_path: Path
    manifest_path: Path


class EpisodePreprocessor:
    NON_DIALOGUE_BLACKLIST = (
        "欢迎订阅",
        "感谢订阅",
        "请订阅",
        "需要您的支持",
        "明镜",
        "本歌曲来自",
        "工作室",
        "云上",
        "栏目",
        "点赞",
        "关注",
        "转发",
        "收藏",
    )
    SHORT_DIALOGUE_ALLOWLIST = {
        "啊",
        "嗯",
        "哦",
        "喔",
        "呃",
        "诶",
        "欸",
        "哎",
        "唉",
        "哈",
        "呵",
        "呀",
        "喂",
        "哼",
        "滚",
        "走",
        "来",
        "上",
        "是",
        "好",
        "谁",
    }
    WHISPER_SAMPLING_RATE = 16000
    OCR_ASSISTED_SECOND_PASS_SOURCE = "ocr_assisted_second_pass"

    def __init__(self, config: EpisodePreprocessConfig) -> None:
        self.config = config
        self.last_asr_runtime: dict[str, Any] | None = None

    @staticmethod
    def _log(message: str) -> None:
        print(f"[preprocess] {message}", flush=True)

    def run(
        self,
        video_path: str | Path,
        episode_id: str,
        series_name: str | None = None,
        telemetry: TelemetryRecorder | None = None,
    ) -> EpisodePreprocessResult:
        resolved_video = Path(video_path).expanduser().resolve()
        if not resolved_video.exists():
            raise FileNotFoundError(f"视频文件不存在：{resolved_video}")
        ffmpeg_binary = get_ffmpeg_exe()
        if not ffmpeg_binary:
            raise RuntimeError("未找到 ffmpeg 可执行文件。请先运行 scripts/install_video_pipeline.sh")
        series_folder = derive_series_folder_name(
            video_path=resolved_video,
            explicit_series_name=series_name,
        )

        preprocess_dir = self.config.output_root / series_folder / episode_id / "preprocess"
        preprocess_dir = preprocess_dir.expanduser().resolve()
        frames_dir = preprocess_dir / "frames"
        model_frames_dir = preprocess_dir / "frames_for_model"
        ocr_frames_dir = preprocess_dir / "ocr_samples"
        frames_dir.mkdir(parents=True, exist_ok=True)
        model_frames_dir.mkdir(parents=True, exist_ok=True)
        ocr_frames_dir.mkdir(parents=True, exist_ok=True)

        audio_path = preprocess_dir / "audio.wav"
        transcript_json_path = preprocess_dir / "transcript_segments.json"
        transcript_text_path = preprocess_dir / "transcript.txt"
        corrected_transcript_json_path = preprocess_dir / "corrected_transcript_segments.json"
        corrected_transcript_text_path = preprocess_dir / "corrected_transcript.txt"
        scene_list_path = preprocess_dir / "scene_list.json"
        ocr_json_path = preprocess_dir / "ocr_segments.json"
        ocr_text_path = preprocess_dir / "ocr.txt"
        manifest_path = preprocess_dir / "preprocess_manifest.json"

        timings: dict[str, float] = {}
        started = time.perf_counter()
        with telemetry_span(
            telemetry,
            stage="preprocess",
            name="probe_video_duration",
            metadata={"video_path": str(resolved_video)},
        ) as step:
            duration_seconds = self._probe_duration_seconds(resolved_video)
            step["metadata"]["duration_seconds_detected"] = round(duration_seconds, 3)
        timings["probe_video_duration"] = round(time.perf_counter() - started, 3)
        self._log(f"视频时长约 {duration_seconds:.1f}s，开始抽取音频。")
        started = time.perf_counter()
        with telemetry_span(
            telemetry,
            stage="preprocess",
            name="extract_audio",
            metadata={"video_path": str(resolved_video), "audio_path": str(audio_path)},
        ) as step:
            self._extract_audio(ffmpeg_binary, resolved_video, audio_path)
            step["metadata"]["audio_exists"] = audio_path.exists()
        timings["extract_audio"] = round(time.perf_counter() - started, 3)
        self._log(
            f"开始 ASR：model={self.config.asr_model_size} device={self.config.asr_device} compute={self.config.asr_compute_type}"
        )
        started = time.perf_counter()
        with telemetry_span(
            telemetry,
            stage="preprocess",
            name="transcribe_audio_asr",
            metadata={
                "audio_path": str(audio_path),
                "model_size": self.config.asr_model_size,
                "requested_device": self.config.asr_device,
                "requested_compute_type": self.config.asr_compute_type,
                "beam_size": self.config.asr_beam_size,
                "best_of": self.config.asr_best_of,
                "vad_filter": self.config.asr_vad_filter,
            },
        ) as step:
            transcript_segments, transcript_text, dual_track_summary = self._transcribe_audio(audio_path)
            step["metadata"]["segment_count"] = len(transcript_segments)
            step["metadata"]["transcript_chars"] = len(transcript_text)
            step["metadata"]["dual_track_summary"] = dual_track_summary
            if self.last_asr_runtime:
                step["metadata"]["asr_runtime"] = dict(self.last_asr_runtime)
        timings["transcribe_audio_asr"] = round(time.perf_counter() - started, 3)
        self._log("开始镜头切分。")
        started = time.perf_counter()
        with telemetry_span(
            telemetry,
            stage="preprocess",
            name="detect_scenes",
            metadata={
                "video_path": str(resolved_video),
                "shot_threshold": self.config.shot_threshold,
                "shot_min_scene_len": self.config.shot_min_scene_len,
            },
        ) as step:
            scenes = self._detect_scenes(resolved_video, duration_seconds)
            step["metadata"]["scene_count"] = len(scenes)
        timings["detect_scenes"] = round(time.perf_counter() - started, 3)
        started = time.perf_counter()
        with telemetry_span(
            telemetry,
            stage="preprocess",
            name="select_keyframe_scenes",
            metadata={"scene_count": len(scenes), "max_keyframes": self.config.max_keyframes},
        ) as step:
            keyframe_scenes = self._select_keyframe_scenes(scenes)
            step["metadata"]["selected_scene_count"] = len(keyframe_scenes)
        timings["select_keyframe_scenes"] = round(time.perf_counter() - started, 3)
        self._log("开始 OCR 取样（仅针对已选关键场景）。")
        started = time.perf_counter()
        with telemetry_span(
            telemetry,
            stage="preprocess",
            name="extract_ocr_sample_frames",
            metadata={
                "video_path": str(resolved_video),
                "ocr_sample_interval_seconds": self.config.ocr_sample_interval_seconds,
                "selected_scene_count": len(keyframe_scenes),
                "ocr_frames_dir": str(ocr_frames_dir),
            },
        ) as step:
            ocr_samples = self._extract_ocr_sample_frames(
                ffmpeg_binary,
                resolved_video,
                keyframe_scenes,
                ocr_frames_dir,
            )
            step["metadata"]["ocr_sample_count"] = len(ocr_samples)
        timings["extract_ocr_sample_frames"] = round(time.perf_counter() - started, 3)
        self._log(f"开始 OCR，共取样 {len(ocr_samples)} 帧。")
        started = time.perf_counter()
        with telemetry_span(
            telemetry,
            stage="preprocess",
            name="run_ocr_on_samples",
            metadata={
                "ocr_sample_count": len(ocr_samples),
                "ocr_crop_bottom_ratio": self.config.ocr_crop_bottom_ratio,
            },
        ) as step:
            ocr_segments, ocr_text = self._run_ocr(ocr_samples)
            step["metadata"]["ocr_segment_count"] = len(ocr_segments)
            step["metadata"]["ocr_chars"] = len(ocr_text)
        timings["run_ocr_on_samples"] = round(time.perf_counter() - started, 3)
        started = time.perf_counter()
        with telemetry_span(
            telemetry,
            stage="preprocess",
            name="build_corrected_transcript",
            metadata={
                "transcript_segment_count": len(transcript_segments),
                "ocr_segment_count": len(ocr_segments),
                "enabled": self.config.ocr_dialogue_correction_enabled,
            },
        ) as step:
            transcript_segments, transcript_text, supplement_summary = self._augment_transcript_with_ocr_guided_second_pass(
                ffmpeg_binary=ffmpeg_binary,
                audio_path=audio_path,
                transcript_segments=transcript_segments,
                ocr_segments=ocr_segments,
            )
            transcript_segments, transcript_sanitation_summary = self._sanitize_transcript_segments(
                transcript_segments,
                ocr_segments,
            )
            transcript_text = self._render_transcript_text(transcript_segments)
            corrected_transcript_segments, corrected_transcript_text, correction_summary = (
                self._build_corrected_transcript(transcript_segments, ocr_segments)
            )
            corrected_transcript_segments, corrected_transcript_text, refine_summary = self._refine_transcript_with_llm(
                corrected_transcript_segments,
                ocr_segments,
            )
            step["metadata"]["second_pass_summary"] = supplement_summary
            step["metadata"]["transcript_sanitation_summary"] = transcript_sanitation_summary
            step["metadata"]["transcript_refine_summary"] = refine_summary
            step["metadata"].update(correction_summary)
        timings["build_corrected_transcript"] = round(time.perf_counter() - started, 3)
        self._purge_directory(ocr_frames_dir)
        ocr_segments = self._mark_source_artifacts_deleted(ocr_segments)
        self._log(f"开始抽关键帧，共输出 {len(keyframe_scenes)} 个场景关键帧。")
        started = time.perf_counter()
        with telemetry_span(
            telemetry,
            stage="preprocess",
            name="extract_keyframes",
            metadata={
                "video_path": str(resolved_video),
                "selected_scene_count": len(keyframe_scenes),
                "frames_dir": str(frames_dir),
            },
        ) as step:
            keyframes = self._extract_keyframes(ffmpeg_binary, resolved_video, keyframe_scenes, frames_dir)
            step["metadata"]["keyframe_count"] = len(keyframes)
        timings["extract_keyframes"] = round(time.perf_counter() - started, 3)
        started = time.perf_counter()
        with telemetry_span(
            telemetry,
            stage="preprocess",
            name="prepare_model_keyframes",
            metadata={
                "keyframe_count": len(keyframes),
                "keyframe_scale_ratio": self.config.keyframe_scale_ratio,
                "model_frames_dir": str(model_frames_dir),
            },
        ) as step:
            keyframes, resize_summary = self._prepare_model_keyframes(keyframes, model_frames_dir)
            step["metadata"]["resized_count"] = resize_summary.get("resized_count", 0)
            step["metadata"]["example"] = resize_summary.get("example", {})
        timings["prepare_model_keyframes"] = round(time.perf_counter() - started, 3)
        if resize_summary.get("resized_count", 0) > 0:
            example = dict(resize_summary.get("example", {}))
            self._log(
                "关键帧缩放完成："
                f"{example.get('original', '')} -> {example.get('model', '')} | "
                f"模型输入图目录={model_frames_dir}"
            )
        self._purge_directory(frames_dir)
        keyframes = self._mark_source_artifacts_deleted(keyframes)
        started = time.perf_counter()
        with telemetry_span(
            telemetry,
            stage="preprocess",
            name="attach_ocr_to_keyframes",
            metadata={"keyframe_count": len(keyframes), "ocr_segment_count": len(ocr_segments)},
        ) as step:
            keyframes = self._attach_ocr_segments_to_keyframes(keyframes, ocr_segments)
            step["metadata"]["linked_keyframe_count"] = sum(
                1 for item in keyframes if item.get("linked_ocr_segments")
            )
        timings["attach_ocr_to_keyframes"] = round(time.perf_counter() - started, 3)

        save_json_file(transcript_json_path, {"segments": transcript_segments})
        save_text_file(transcript_text_path, transcript_text)
        save_json_file(
            corrected_transcript_json_path,
            {"segments": corrected_transcript_segments, "correction_summary": correction_summary},
        )
        save_text_file(corrected_transcript_text_path, corrected_transcript_text)
        save_json_file(
            scene_list_path,
            {
                "scenes": scenes,
                "sampled_scenes": keyframe_scenes,
                "keyframes": keyframes,
            },
        )
        save_json_file(ocr_json_path, {"segments": ocr_segments})
        save_text_file(ocr_text_path, ocr_text)

        manifest = {
            "series_folder": series_folder,
            "episode_id": episode_id,
            "video_path": str(resolved_video),
            "duration_seconds": duration_seconds,
            "timing_breakdown_seconds": timings,
            "asr_runtime": self.last_asr_runtime,
            "asr_config": self._asr_config_summary(),
            "asr_dual_track_summary": dual_track_summary,
            "asr_second_pass_summary": supplement_summary,
            "ocr_sample_interval_seconds": self.config.ocr_sample_interval_seconds,
            "keyframe_scale_ratio": self.config.keyframe_scale_ratio,
            "audio_path": str(audio_path),
            "transcript_json_path": str(transcript_json_path),
            "transcript_text_path": str(transcript_text_path),
            "corrected_transcript_json_path": str(corrected_transcript_json_path),
            "corrected_transcript_text_path": str(corrected_transcript_text_path),
            "dialogue_correction_summary": correction_summary,
            "transcript_refine_summary": refine_summary,
            "scene_list_path": str(scene_list_path),
            "frames_dir": str(frames_dir),
            "model_frames_dir": str(model_frames_dir),
            "ocr_frames_dir": str(ocr_frames_dir),
            "ocr_json_path": str(ocr_json_path),
            "ocr_text_path": str(ocr_text_path),
            "purged_artifacts": {
                "original_frames_deleted": True,
                "ocr_sample_frames_deleted": True,
            },
            "keyframe_resize_summary": resize_summary,
            "keyframes": keyframes,
        }
        save_json_file(manifest_path, manifest)
        self._log(f"预处理产物已写入：{manifest_path}")

        return EpisodePreprocessResult(
            preprocess_dir=preprocess_dir,
            audio_path=audio_path,
            transcript_json_path=transcript_json_path,
            transcript_text_path=transcript_text_path,
            corrected_transcript_json_path=corrected_transcript_json_path,
            corrected_transcript_text_path=corrected_transcript_text_path,
            scene_list_path=scene_list_path,
            frames_dir=frames_dir,
            ocr_json_path=ocr_json_path,
            ocr_text_path=ocr_text_path,
            manifest_path=manifest_path,
        )

    def _probe_duration_seconds(self, video_path: Path) -> float:
        _, duration_seconds = count_frames_and_secs(str(video_path))
        return float(duration_seconds)

    def _extract_audio(self, ffmpeg_binary: str, video_path: Path, audio_path: Path) -> None:
        command = [
            ffmpeg_binary,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ]
        self._run_command(command)

    def _transcribe_audio(self, audio_path: Path) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        model = self._build_whisper_model()
        primary_segments, primary_text = self._transcribe_with_model(
            model,
            audio_path,
            vad_filter=self.config.asr_vad_filter,
            condition_on_previous_text=self.config.asr_condition_on_previous_text,
        )
        summary = {
            "enabled": bool(self.config.asr_enable_dual_track_fusion),
            "mode": "full_audio_recall_experimental" if self.config.asr_enable_dual_track_fusion else "single_pass_primary_only",
            "primary_vad_filter": bool(self.config.asr_vad_filter),
            "primary_segment_count": len(primary_segments),
            "recall_segment_count": 0,
            "supplemented_segment_count": 0,
            "fused_segment_count": len(primary_segments),
        }
        if not self.config.asr_enable_dual_track_fusion:
            return primary_segments, primary_text, summary

        recall_segments, _ = self._transcribe_with_model(
            model,
            audio_path,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        accepted_recall_segments = [
            segment
            for segment in recall_segments
            if self._is_plausible_dialogue_text(self._normalize_dialogue_text(segment.get("text", "")))
        ]
        summary["recall_segment_count"] = len(accepted_recall_segments)
        if not accepted_recall_segments:
            return primary_segments, primary_text, summary

        fused_segments = self._merge_transcript_segments(
            primary_segments,
            accepted_recall_segments,
            supplemental_source="asr_recall_no_vad",
        )
        summary["fused_segment_count"] = len(fused_segments)
        summary["supplemented_segment_count"] = max(0, len(fused_segments) - len(primary_segments))
        return fused_segments, self._render_transcript_text(fused_segments), summary

    def _transcribe_with_model(
        self,
        model: WhisperModel,
        audio_path: Path,
        *,
        start_offset_seconds: float = 0.0,
        vad_filter: bool | None = None,
        condition_on_previous_text: bool | None = None,
        initial_prompt_override: str | None = None,
        no_speech_threshold_override: float | None = None,
        log_prob_threshold_override: float | None = None,
        hallucination_silence_threshold_override: float | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        resolved_vad_filter = self.config.asr_vad_filter if vad_filter is None else bool(vad_filter)
        resolved_condition = (
            self.config.asr_condition_on_previous_text
            if condition_on_previous_text is None
            else bool(condition_on_previous_text)
        )
        if resolved_vad_filter:
            return self._transcribe_with_vad_chunks(
                model,
                audio_path,
                start_offset_seconds=start_offset_seconds,
            )

        return self._transcribe_audio_chunk(
            model,
            str(audio_path),
            start_offset_seconds=start_offset_seconds,
            condition_on_previous_text=resolved_condition,
            initial_prompt_override=initial_prompt_override,
            no_speech_threshold_override=no_speech_threshold_override,
            log_prob_threshold_override=log_prob_threshold_override,
            hallucination_silence_threshold_override=hallucination_silence_threshold_override,
        )

    def _transcribe_audio_chunk(
        self,
        model: WhisperModel,
        audio_input: str | np.ndarray,
        *,
        start_offset_seconds: float = 0.0,
        condition_on_previous_text: bool,
        initial_prompt_override: str | None = None,
        no_speech_threshold_override: float | None = None,
        log_prob_threshold_override: float | None = None,
        hallucination_silence_threshold_override: float | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        segments_iter, _ = model.transcribe(
            audio_input,
            language=self.config.asr_language,
            beam_size=self.config.asr_beam_size,
            best_of=max(self.config.asr_best_of, self.config.asr_beam_size),
            patience=max(1.0, float(self.config.asr_patience)),
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=(initial_prompt_override if initial_prompt_override is not None else self.config.asr_initial_prompt) or None,
            hotwords=self.config.asr_hotwords or None,
            no_speech_threshold=float(
                self.config.asr_no_speech_threshold
                if no_speech_threshold_override is None
                else no_speech_threshold_override
            ),
            log_prob_threshold=float(
                self.config.asr_log_prob_threshold
                if log_prob_threshold_override is None
                else log_prob_threshold_override
            ),
            hallucination_silence_threshold=float(
                self.config.asr_hallucination_silence_threshold
                if hallucination_silence_threshold_override is None
                else hallucination_silence_threshold_override
            ),
            vad_filter=False,
            vad_parameters=None,
        )

        segments: list[dict[str, Any]] = []
        transcript_lines: list[str] = []
        next_segment_index = 1
        for segment in segments_iter:
            text = (segment.text or "").strip()
            if not text:
                continue
            start_seconds = round(float(segment.start) + start_offset_seconds, 3)
            end_seconds = round(float(segment.end) + start_offset_seconds, 3)
            for part in self._split_transcript_segment_by_punctuation(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                text=text,
            ):
                item = {
                    "segment_id": f"seg-{next_segment_index:04d}",
                    "start": float(part["start"]),
                    "end": float(part["end"]),
                    "text": str(part["text"]).strip(),
                }
                next_segment_index += 1
                segments.append(item)
                transcript_lines.append(f"[{float(item['start']):.3f}-{float(item['end']):.3f}] {item['text']}")
        return segments, "\n".join(transcript_lines).strip()

    def _transcribe_with_vad_chunks(
        self,
        model: WhisperModel,
        audio_path: Path,
        *,
        start_offset_seconds: float = 0.0,
    ) -> tuple[list[dict[str, Any]], str]:
        audio = decode_audio(str(audio_path), sampling_rate=self.WHISPER_SAMPLING_RATE)
        vad_options = VadOptions(
            threshold=float(self.config.asr_vad_threshold),
            neg_threshold=(
                None
                if self.config.asr_vad_neg_threshold is None
                else float(self.config.asr_vad_neg_threshold)
            ),
            min_speech_duration_ms=int(self.config.asr_vad_min_speech_duration_ms),
            max_speech_duration_s=float(self.config.asr_vad_max_speech_duration_seconds),
            min_silence_duration_ms=int(self.config.asr_vad_min_silence_duration_ms),
            speech_pad_ms=int(self.config.asr_vad_speech_pad_ms),
        )
        speech_chunks = get_speech_timestamps(
            audio,
            vad_options=vad_options,
            sampling_rate=self.WHISPER_SAMPLING_RATE,
        )
        merged_windows = self._merge_vad_speech_chunks(speech_chunks)
        if not merged_windows:
            return [], ""

        all_segments: list[dict[str, Any]] = []
        for window in merged_windows:
            chunk_audio = audio[int(window["start_sample"]): int(window["end_sample"])]
            chunk_offset_seconds = start_offset_seconds + float(window["start_seconds"])
            chunk_segments, _ = self._transcribe_audio_chunk(
                model,
                chunk_audio,
                start_offset_seconds=chunk_offset_seconds,
                condition_on_previous_text=False,
            )
            all_segments.extend(chunk_segments)

        all_segments.sort(key=lambda item: (float(item.get("start", 0.0) or 0.0), float(item.get("end", 0.0) or 0.0)))
        normalized_segments: list[dict[str, Any]] = []
        transcript_lines: list[str] = []
        for index, item in enumerate(all_segments, start=1):
            normalized_item = {**item, "segment_id": f"seg-{index:04d}"}
            normalized_segments.append(normalized_item)
            transcript_lines.append(
                f"[{float(normalized_item['start']):.3f}-{float(normalized_item['end']):.3f}] {normalized_item['text']}"
            )
        return normalized_segments, "\n".join(transcript_lines).strip()

    def _merge_vad_speech_chunks(self, speech_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not speech_chunks:
            return []
        merged: list[dict[str, Any]] = []
        current_start = int(speech_chunks[0]["start"])
        current_end = int(speech_chunks[0]["end"])
        for chunk in speech_chunks[1:]:
            next_start = int(chunk["start"])
            next_end = int(chunk["end"])
            gap_seconds = (next_start - current_end) / self.WHISPER_SAMPLING_RATE
            proposed_duration = (next_end - current_start) / self.WHISPER_SAMPLING_RATE
            if (
                gap_seconds <= float(self.config.asr_vad_chunk_merge_gap_seconds)
                and proposed_duration <= float(self.config.asr_vad_chunk_max_seconds)
            ):
                current_end = next_end
                continue
            merged.append(
                {
                    "start_sample": current_start,
                    "end_sample": current_end,
                    "start_seconds": round(current_start / self.WHISPER_SAMPLING_RATE, 3),
                    "end_seconds": round(current_end / self.WHISPER_SAMPLING_RATE, 3),
                }
            )
            current_start = next_start
            current_end = next_end
        merged.append(
            {
                "start_sample": current_start,
                "end_sample": current_end,
                "start_seconds": round(current_start / self.WHISPER_SAMPLING_RATE, 3),
                "end_seconds": round(current_end / self.WHISPER_SAMPLING_RATE, 3),
            }
        )
        return merged

    @staticmethod
    def _split_transcript_segment_by_punctuation(
        *,
        start_seconds: float,
        end_seconds: float,
        text: str,
    ) -> list[dict[str, Any]]:
        normalized_text = str(text or "").strip()
        duration_seconds = max(0.001, float(end_seconds) - float(start_seconds))
        compact_text = re.sub(r"\s+", "", normalized_text)
        if not compact_text:
            return []

        clause_candidates = [
            part.strip()
            for part in re.findall(r"[^，,。！？；、]+[，,。！？；、]?", normalized_text)
            if part and part.strip()
        ]
        if (
            len(clause_candidates) <= 1
            or duration_seconds < 3.6
            or len(compact_text) < 16
        ):
            return [
                {
                    "start": round(float(start_seconds), 3),
                    "end": round(float(end_seconds), 3),
                    "text": normalized_text,
                }
            ]

        weights = [
            max(1, len(re.sub(r"[，,。！？；、\s]", "", clause)))
            for clause in clause_candidates
        ]
        total_weight = max(1, sum(weights))
        min_piece_duration = 0.55
        if duration_seconds / max(1, len(clause_candidates)) < min_piece_duration:
            return [
                {
                    "start": round(float(start_seconds), 3),
                    "end": round(float(end_seconds), 3),
                    "text": normalized_text,
                }
            ]

        pieces: list[dict[str, Any]] = []
        cursor = float(start_seconds)
        for index, clause in enumerate(clause_candidates):
            if index == len(clause_candidates) - 1:
                clause_end = float(end_seconds)
            else:
                clause_duration = max(min_piece_duration, duration_seconds * (weights[index] / total_weight))
                remaining_min = min_piece_duration * (len(clause_candidates) - index - 1)
                clause_end = min(float(end_seconds) - remaining_min, cursor + clause_duration)
            clause_end = max(cursor + 0.2, clause_end)
            pieces.append(
                {
                    "start": round(cursor, 3),
                    "end": round(clause_end, 3),
                    "text": clause.strip(),
                }
            )
            cursor = clause_end

        if pieces:
            pieces[-1]["end"] = round(float(end_seconds), 3)
        return pieces

    def _build_corrected_transcript(
        self,
        transcript_segments: list[dict[str, Any]],
        ocr_segments: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        corrected_segments: list[dict[str, Any]] = []
        applied_count = 0

        for segment in transcript_segments:
            corrected_text, correction_note = self._apply_ocr_dialogue_correction(segment, ocr_segments)
            corrected_segment = dict(segment)
            corrected_segment["text"] = corrected_text
            if correction_note:
                corrected_segment["ocr_correction"] = correction_note
                applied_count += 1
            corrected_segments.append(corrected_segment)

        sanitized_segments, sanitation_summary = self._sanitize_corrected_transcript(corrected_segments, ocr_segments)
        corrected_lines = [
            f"[{float(segment['start']):.3f}-{float(segment['end']):.3f}] {str(segment.get('text') or '').strip()}"
            for segment in sanitized_segments
            if str(segment.get("text") or "").strip()
        ]

        summary = {
            "enabled": self.config.ocr_dialogue_correction_enabled,
            "segment_count": len(sanitized_segments),
            "ocr_segment_count": len(ocr_segments),
            "corrected_segment_count": applied_count,
            "min_similarity": self.config.ocr_dialogue_min_similarity,
            "max_mismatch_chars": self.config.ocr_dialogue_max_mismatch_chars,
            "timing_tolerance_seconds": self.config.ocr_dialogue_timing_tolerance_seconds,
            "policy": "asr_primary_ocr_typo_correction_only",
        }
        summary.update(sanitation_summary)
        return sanitized_segments, "\n".join(corrected_lines).strip(), summary

    def _augment_transcript_with_ocr_guided_second_pass(
        self,
        *,
        ffmpeg_binary: str,
        audio_path: Path,
        transcript_segments: list[dict[str, Any]],
        ocr_segments: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        summary = {
            "enabled": bool(self.config.asr_enable_second_pass),
            "window_count": 0,
            "supplemented_segment_count": 0,
            "windows": [],
        }
        if not self.config.asr_enable_second_pass or not ocr_segments:
            return transcript_segments, self._render_transcript_text(transcript_segments), summary

        windows = self._find_ocr_guided_asr_windows(transcript_segments, ocr_segments)
        summary["window_count"] = len(windows)
        if not windows:
            return transcript_segments, self._render_transcript_text(transcript_segments), summary

        model = self._build_whisper_model()
        supplemental_segments: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="seedance_asr_second_pass_", dir="/tmp") as tmpdir:
            tmp_root = Path(tmpdir)
            for index, window in enumerate(windows, start=1):
                clip_path = tmp_root / f"window_{index:02d}.wav"
                start_seconds = float(window["start_seconds"])
                end_seconds = float(window["end_seconds"])
                self._extract_audio_clip(
                    ffmpeg_binary=ffmpeg_binary,
                    source_audio_path=audio_path,
                    clip_audio_path=clip_path,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )
                clip_segments, _ = self._transcribe_with_model(
                    model,
                    clip_path,
                    start_offset_seconds=start_seconds,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    initial_prompt_override=self._build_ocr_guided_second_pass_prompt(str(window.get("ocr_text") or "")),
                    no_speech_threshold_override=min(0.2, float(self.config.asr_no_speech_threshold)),
                    log_prob_threshold_override=min(float(self.config.asr_log_prob_threshold), -1.5),
                    hallucination_silence_threshold_override=max(
                        0.3,
                        min(0.8, float(self.config.asr_hallucination_silence_threshold)),
                    ),
                )
                accepted_segments = [
                    segment
                    for segment in clip_segments
                    if self._is_plausible_dialogue_text(self._normalize_dialogue_text(segment.get("text", "")))
                    and self._transcript_segment_matches_ocr_text(segment, str(window.get("ocr_text") or ""))
                ]
                if not accepted_segments:
                    fallback_segment = self._build_ocr_fallback_segment(
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        ocr_text=str(window.get("ocr_text") or ""),
                    )
                    if fallback_segment is not None:
                        accepted_segments = [fallback_segment]
                summary["windows"].append(
                    {
                        "start_seconds": round(start_seconds, 3),
                        "end_seconds": round(end_seconds, 3),
                        "ocr_text": str(window.get("ocr_text") or ""),
                        "accepted_segment_count": len(accepted_segments),
                    }
                )
                supplemental_segments.extend(accepted_segments)

        merged_segments = self._merge_transcript_segments(
            transcript_segments,
            supplemental_segments,
            supplemental_source=self.OCR_ASSISTED_SECOND_PASS_SOURCE,
        )
        summary["supplemented_segment_count"] = max(0, len(merged_segments) - len(transcript_segments))
        return merged_segments, self._render_transcript_text(merged_segments), summary

    def _build_ocr_guided_second_pass_prompt(self, ocr_text: str) -> str:
        ocr_hint = str(ocr_text or "").strip()
        if not ocr_hint:
            return self.config.asr_initial_prompt or ""
        base_prompt = str(self.config.asr_initial_prompt or "").strip()
        hint = f"当前局部窗口字幕接近：{ocr_hint}。请优先识别与这句字幕相近的真实中文对白，避免把背景广告、片头片尾宣传词识别进来。"
        return f"{base_prompt} {hint}".strip()

    def _build_ocr_fallback_segment(
        self,
        *,
        start_seconds: float,
        end_seconds: float,
        ocr_text: str,
    ) -> dict[str, Any] | None:
        normalized_text = self._normalize_dialogue_text(ocr_text)
        if not self._is_plausible_dialogue_text(normalized_text):
            return None
        return {
            "segment_id": "",
            "start": round(float(start_seconds), 3),
            "end": round(float(end_seconds), 3),
            "text": str(ocr_text).strip(),
            "source": "ocr_fallback_dialogue",
        }

    def _find_ocr_guided_asr_windows(
        self,
        transcript_segments: list[dict[str, Any]],
        ocr_segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        tolerance = max(0.2, float(self.config.asr_second_pass_trigger_tolerance_seconds))
        padding = max(0.2, float(self.config.asr_second_pass_window_padding_seconds))
        max_window = max(2.0, float(self.config.asr_second_pass_max_window_seconds))
        candidate_times: list[tuple[float, str]] = []
        for item in ocr_segments:
            timestamp = float(item.get("timestamp", 0.0) or 0.0)
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            if not self._is_plausible_dialogue_text(self._normalize_dialogue_text(text)):
                continue
            covered = False
            matched = False
            for segment in transcript_segments:
                start_seconds = float(segment.get("start", 0.0) or 0.0)
                end_seconds = float(segment.get("end", start_seconds) or start_seconds)
                if start_seconds - tolerance <= timestamp <= end_seconds + tolerance:
                    covered = True
                    if self._transcript_segment_matches_ocr_text(segment, text):
                        matched = True
                        break
            if not covered or not matched:
                candidate_times.append((timestamp, text))
        if not candidate_times:
            return []

        windows: list[dict[str, Any]] = []
        current_times: list[float] = [candidate_times[0][0]]
        current_texts: list[str] = [candidate_times[0][1]]
        for timestamp, text in candidate_times[1:]:
            proposed_start = current_times[0] - padding
            proposed_end = timestamp + padding
            if timestamp - current_times[-1] <= max(2.5, tolerance * 2.5) and proposed_end - proposed_start <= max_window:
                current_times.append(timestamp)
                current_texts.append(text)
                continue
            windows.append(
                {
                    "start_seconds": round(max(0.0, current_times[0] - padding), 3),
                    "end_seconds": round(current_times[-1] + padding, 3),
                    "ocr_text": " / ".join(current_texts[:4]),
                }
            )
            current_times = [timestamp]
            current_texts = [text]
        windows.append(
            {
                "start_seconds": round(max(0.0, current_times[0] - padding), 3),
                "end_seconds": round(current_times[-1] + padding, 3),
                "ocr_text": " / ".join(current_texts[:4]),
            }
        )
        return windows

    def _extract_audio_clip(
        self,
        *,
        ffmpeg_binary: str,
        source_audio_path: Path,
        clip_audio_path: Path,
        start_seconds: float,
        end_seconds: float,
    ) -> None:
        duration = max(0.5, float(end_seconds) - float(start_seconds))
        command = [
            ffmpeg_binary,
            "-y",
            "-ss",
            f"{max(0.0, start_seconds):.3f}",
            "-i",
            str(source_audio_path),
            "-t",
            f"{duration:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(clip_audio_path),
        ]
        self._run_command(command)

    def _merge_transcript_segments(
        self,
        primary_segments: list[dict[str, Any]],
        supplemental_segments: list[dict[str, Any]],
        *,
        supplemental_source: str,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = [dict(item) for item in primary_segments]
        for segment in supplemental_segments:
            normalized = self._normalize_dialogue_text(segment.get("text", ""))
            if not normalized:
                continue
            start_seconds = float(segment.get("start", 0.0) or 0.0)
            end_seconds = float(segment.get("end", start_seconds) or start_seconds)
            duplicate = self._is_duplicate_against_primary_window(
                primary_segments=merged,
                normalized_text=normalized,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
            for existing in merged:
                existing_start = float(existing.get("start", 0.0) or 0.0)
                existing_end = float(existing.get("end", existing_start) or existing_start)
                existing_normalized = self._normalize_dialogue_text(existing.get("text", ""))
                overlap = min(existing_end, end_seconds) - max(existing_start, start_seconds)
                overlap_ratio = 0.0
                shorter_duration = min(
                    max(0.001, existing_end - existing_start),
                    max(0.001, end_seconds - start_seconds),
                )
                if overlap > 0:
                    overlap_ratio = overlap / shorter_duration
                similarity = difflib.SequenceMatcher(None, existing_normalized, normalized).ratio()
                texts_substantially_match = (
                    existing_normalized == normalized
                    or (existing_normalized and normalized and (
                        existing_normalized in normalized or normalized in existing_normalized
                    ))
                    or similarity >= 0.68
                )
                if texts_substantially_match and (
                    abs(existing_start - start_seconds) <= 1.2 or overlap > 0.25 or overlap_ratio >= 0.75
                ):
                    duplicate = True
                    break
            if duplicate:
                continue
            merged.append(
                {
                    "segment_id": str(segment.get("segment_id") or ""),
                    "start": round(start_seconds, 3),
                    "end": round(end_seconds, 3),
                    "text": str(segment.get("text") or "").strip(),
                    "source": supplemental_source,
                }
            )

        merged.sort(key=lambda item: (float(item.get("start", 0.0) or 0.0), float(item.get("end", 0.0) or 0.0)))
        normalized_segments: list[dict[str, Any]] = []
        for index, segment in enumerate(merged, start=1):
            normalized_segments.append({**segment, "segment_id": f"seg-{index:04d}"})
        return normalized_segments

    @staticmethod
    def _render_transcript_text(segments: list[dict[str, Any]]) -> str:
        lines = []
        for item in segments:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"[{float(item['start']):.3f}-{float(item['end']):.3f}] {text}")
        return "\n".join(lines).strip()

    def _apply_ocr_dialogue_correction(
        self,
        transcript_segment: dict[str, Any],
        ocr_segments: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any] | None]:
        original_text = str(transcript_segment.get("text", "")).strip()
        if not original_text or not self.config.ocr_dialogue_correction_enabled or not ocr_segments:
            return original_text, None

        normalized_original = self._normalize_dialogue_text(original_text)
        if not self._is_plausible_dialogue_text(normalized_original):
            return original_text, None

        start_seconds = float(transcript_segment.get("start", 0.0))
        end_seconds = float(transcript_segment.get("end", start_seconds))
        tolerance = max(0.1, float(self.config.ocr_dialogue_timing_tolerance_seconds))
        candidates = [
            item
            for item in ocr_segments
            if start_seconds - tolerance <= float(item.get("timestamp", 0.0)) <= end_seconds + tolerance
        ]
        if not candidates:
            return original_text, None

        best_score = 0.0
        best_candidate: dict[str, Any] | None = None
        best_normalized = ""
        for candidate in candidates:
            candidate_text = self._normalize_dialogue_text(candidate.get("text", ""))
            if not self._is_plausible_dialogue_text(candidate_text):
                continue
            if len(candidate_text) != len(normalized_original):
                continue
            mismatch_count = self._count_char_mismatches(normalized_original, candidate_text)
            similarity = difflib.SequenceMatcher(None, normalized_original, candidate_text).ratio()
            if mismatch_count > int(self.config.ocr_dialogue_max_mismatch_chars):
                continue
            if similarity < float(self.config.ocr_dialogue_min_similarity):
                continue
            if similarity > best_score:
                best_score = similarity
                best_candidate = candidate
                best_normalized = candidate_text

        if not best_candidate or not best_normalized or best_normalized == normalized_original:
            return original_text, None

        return best_normalized, {
            "source": "ocr_typo_correction",
            "sample_id": str(best_candidate.get("sample_id", "")),
            "scene_id": str(best_candidate.get("scene_id", "")),
            "timestamp": float(best_candidate.get("timestamp", 0.0)),
            "original_text": original_text,
            "ocr_text": str(best_candidate.get("text", "")).strip(),
            "normalized_ocr_text": best_normalized,
            "similarity": round(best_score, 4),
            "mismatch_chars": self._count_char_mismatches(normalized_original, best_normalized),
        }

    def _normalize_dialogue_text(self, text: Any) -> str:
        normalized = unicodedata.normalize("NFKC", str(text or ""))
        normalized = re.sub(r"\s+", "", normalized)
        normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", normalized)
        return normalized.strip()

    def _is_plausible_dialogue_text(self, text: str) -> bool:
        if not text or len(text) > 30:
            return False
        lowered = str(text or "").lower()
        for blocked in self.NON_DIALOGUE_BLACKLIST:
            if blocked and blocked in text:
                return False
            blocked_lower = blocked.lower()
            if blocked_lower and blocked_lower in lowered:
                return False
        if text in self.SHORT_DIALOGUE_ALLOWLIST:
            return True
        if len(text) == 1:
            char = text[0]
            return "\u4e00" <= char <= "\u9fff"
        cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        alnum_count = sum(1 for char in text if char.isalnum())
        if alnum_count == 0:
            return False
        if len(text) <= 2 and cjk_count >= 1:
            return True
        return cjk_count >= 2 or (cjk_count >= 1 and alnum_count >= 4)

    def _is_duplicate_against_primary_window(
        self,
        *,
        primary_segments: list[dict[str, Any]],
        normalized_text: str,
        start_seconds: float,
        end_seconds: float,
    ) -> bool:
        tolerance_seconds = 1.4
        candidates: list[dict[str, Any]] = []
        for existing in primary_segments:
            existing_start = float(existing.get("start", 0.0) or 0.0)
            existing_end = float(existing.get("end", existing_start) or existing_start)
            if existing_end < start_seconds - tolerance_seconds or existing_start > end_seconds + tolerance_seconds:
                continue
            candidates.append(existing)
        if not candidates:
            return False

        candidates.sort(key=lambda item: (float(item.get("start", 0.0) or 0.0), float(item.get("end", 0.0) or 0.0)))
        normalized_candidates = [
            self._normalize_dialogue_text(str(item.get("text") or ""))
            for item in candidates
            if self._normalize_dialogue_text(str(item.get("text") or ""))
        ]
        for left in range(len(normalized_candidates)):
            combined = ""
            for right in range(left, min(len(normalized_candidates), left + 4)):
                combined += normalized_candidates[right]
                similarity = difflib.SequenceMatcher(None, combined, normalized_text).ratio()
                if (
                    combined == normalized_text
                    or normalized_text in combined
                    or combined in normalized_text
                    or similarity >= 0.78
                ):
                    return True
        return False

    def _sanitize_corrected_transcript(
        self,
        transcript_segments: list[dict[str, Any]],
        ocr_segments: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        normalized_ocr = []
        for item in ocr_segments:
            text = self._normalize_dialogue_text(item.get("text", ""))
            if not text:
                continue
            normalized_ocr.append(
                {
                    "timestamp": float(item.get("timestamp", 0.0) or 0.0),
                    "text": text,
                }
            )

        dropped_blacklist = 0
        dropped_misaligned_duplicates = 0
        kept: list[dict[str, Any]] = []
        for segment in transcript_segments:
            raw_text = str(segment.get("text") or "").strip()
            normalized = self._normalize_dialogue_text(raw_text)
            if not normalized:
                continue
            if not self._is_plausible_dialogue_text(normalized):
                dropped_blacklist += 1
                continue
            if self._should_drop_segment_via_ocr_alignment(segment, transcript_segments, normalized_ocr):
                dropped_misaligned_duplicates += 1
                continue
            kept.append(segment)

        kept.sort(key=lambda item: (float(item.get("start", 0.0) or 0.0), float(item.get("end", 0.0) or 0.0)))
        normalized_segments: list[dict[str, Any]] = []
        for index, segment in enumerate(kept, start=1):
            normalized_segments.append({**segment, "segment_id": f"seg-{index:04d}"})
        return normalized_segments, {
            "sanitized_segment_count": len(normalized_segments),
            "removed_blacklist_or_implausible_count": dropped_blacklist,
            "removed_misaligned_duplicate_count": dropped_misaligned_duplicates,
        }

    def _sanitize_transcript_segments(
        self,
        transcript_segments: list[dict[str, Any]],
        ocr_segments: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        sanitized_segments, summary = self._sanitize_corrected_transcript(transcript_segments, ocr_segments)
        return sanitized_segments, summary

    def _refine_transcript_with_llm(
        self,
        transcript_segments: list[dict[str, Any]],
        ocr_segments: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        summary = {
            "enabled": bool(self.config.transcript_refine_enabled),
            "provider": str(self.config.transcript_refine_provider or ""),
            "model": str(self.config.transcript_refine_model or ""),
            "batch_count": 0,
            "updated_segment_count": 0,
            "failed_batch_count": 0,
        }
        if not self.config.transcript_refine_enabled:
            return transcript_segments, self._render_transcript_text(transcript_segments), summary
        if str(self.config.transcript_refine_provider or "").strip().lower() != "qwen":
            summary["error"] = f"暂不支持的 transcript_refine_provider：{self.config.transcript_refine_provider}"
            return transcript_segments, self._render_transcript_text(transcript_segments), summary

        adapter = QwenAdapter(
            model=str(self.config.transcript_refine_model or "qwen3-vl-plus").strip() or "qwen3-vl-plus",
            endpoint=(
                str(self.config.transcript_refine_endpoint or "").strip()
                or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            ),
            temperature=0.1,
            timeout_seconds=int(self.config.transcript_refine_timeout_seconds or 180),
            structured_output_mode="json_object",
        )
        batch_size = max(4, int(self.config.transcript_refine_batch_size or 10))
        batches = [
            transcript_segments[index: index + batch_size]
            for index in range(0, len(transcript_segments), batch_size)
        ]
        refined_segments: list[dict[str, Any]] = []
        updated_segment_count = 0
        failed_batch_count = 0
        for batch in batches:
            summary["batch_count"] += 1
            try:
                batch_result = self._refine_transcript_batch_with_qwen(adapter, batch, ocr_segments)
            except Exception:
                failed_batch_count += 1
                refined_segments.extend(dict(item) for item in batch)
                continue
            for original, refined in zip(batch, batch_result):
                original_text = str(original.get("text") or "").strip()
                refined_text = str(refined.get("text") or "").strip()
                if refined_text and refined_text != original_text:
                    updated_segment_count += 1
                refined_segments.append(refined)
        summary["updated_segment_count"] = updated_segment_count
        summary["failed_batch_count"] = failed_batch_count
        return refined_segments, self._render_transcript_text(refined_segments), summary

    def _refine_transcript_batch_with_qwen(
        self,
        adapter: QwenAdapter,
        batch: list[dict[str, Any]],
        ocr_segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        system_prompt = render_prompt(
            "preprocess_transcript_refine/system.md",
            {},
            strict=False,
        )
        user_prompt = render_prompt(
            "preprocess_transcript_refine/user.md",
            {
                "segments_block": self._render_transcript_refine_batch(batch),
                "ocr_block": self._render_transcript_refine_ocr_block(batch, ocr_segments),
            },
            strict=False,
        )
        payload = {
            "model": adapter.config.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        response = adapter.request_json(
            adapter.config.endpoint,
            headers={"Authorization": f"Bearer {adapter.require_api_key()}"},
            payload=payload,
        )
        output_text = adapter._extract_output_text(response)
        data = extract_json_from_text(output_text)
        items = data.get("items")
        if not isinstance(items, list) or len(items) != len(batch):
            raise ProviderResponseError("transcript refine 返回 items 数量不匹配。")
        refined_batch: list[dict[str, Any]] = []
        for original, item in zip(batch, items):
            if not isinstance(item, dict):
                raise ProviderResponseError("transcript refine items 必须是对象数组。")
            if str(item.get("segment_id") or "").strip() != str(original.get("segment_id") or "").strip():
                raise ProviderResponseError("transcript refine segment_id 对不齐。")
            refined_text = str(item.get("text") or "").strip() or str(original.get("text") or "").strip()
            refined_batch.append(
                {
                    **original,
                    "text": refined_text,
                    "refined_by_llm": refined_text != str(original.get("text") or "").strip(),
                }
            )
        return refined_batch

    def _render_transcript_refine_batch(self, batch: list[dict[str, Any]]) -> str:
        lines = []
        for item in batch:
            lines.append(
                f"- {item.get('segment_id', '')}｜[{float(item.get('start', 0.0)):.3f}-{float(item.get('end', 0.0)):.3f}] {str(item.get('text') or '').strip()}"
            )
        return "\n".join(lines)

    def _render_transcript_refine_ocr_block(
        self,
        batch: list[dict[str, Any]],
        ocr_segments: list[dict[str, Any]],
    ) -> str:
        if not batch:
            return "- 无 OCR 参考"
        batch_start = float(batch[0].get("start", 0.0) or 0.0) - 0.8
        batch_end = float(batch[-1].get("end", 0.0) or 0.0) + 0.8
        lines: list[str] = []
        for item in ocr_segments:
            timestamp = float(item.get("timestamp", 0.0) or 0.0)
            if not (batch_start <= timestamp <= batch_end):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"- [{timestamp:.3f}] {text}")
        return "\n".join(lines) if lines else "- 无 OCR 参考"

    def _should_drop_segment_via_ocr_alignment(
        self,
        segment: dict[str, Any],
        all_segments: list[dict[str, Any]],
        normalized_ocr: list[dict[str, Any]],
    ) -> bool:
        segment_text = self._normalize_dialogue_text(segment.get("text", ""))
        if not segment_text:
            return False
        start_seconds = float(segment.get("start", 0.0) or 0.0)
        end_seconds = float(segment.get("end", start_seconds) or start_seconds)
        tolerance = 0.75
        matching_ocr_times = []
        for item in normalized_ocr:
            similarity = difflib.SequenceMatcher(None, segment_text, str(item["text"])).ratio()
            if (
                segment_text == item["text"]
                or segment_text in str(item["text"])
                or str(item["text"]) in segment_text
                or similarity >= 0.8
            ):
                matching_ocr_times.append(float(item["timestamp"]))
        if not matching_ocr_times:
            return False
        if any(start_seconds - tolerance <= timestamp <= end_seconds + tolerance for timestamp in matching_ocr_times):
            return False

        for other in all_segments:
            if other is segment:
                continue
            other_text = self._normalize_dialogue_text(other.get("text", ""))
            if not other_text:
                continue
            similarity = difflib.SequenceMatcher(None, other_text, segment_text).ratio()
            if not (
                other_text == segment_text
                or other_text in segment_text
                or segment_text in other_text
                or similarity >= 0.72
            ):
                continue
            other_start = float(other.get("start", 0.0) or 0.0)
            other_end = float(other.get("end", other_start) or other_start)
            if any(other_start - tolerance <= timestamp <= other_end + tolerance for timestamp in matching_ocr_times):
                return True
        return False

    def _transcript_segment_matches_ocr_text(self, segment: dict[str, Any], ocr_text: str) -> bool:
        segment_text = self._normalize_dialogue_text(segment.get("text", ""))
        normalized_ocr = self._normalize_dialogue_text(ocr_text)
        if not segment_text or not normalized_ocr:
            return False
        similarity = difflib.SequenceMatcher(None, segment_text, normalized_ocr).ratio()
        return (
            segment_text == normalized_ocr
            or normalized_ocr in segment_text
            or segment_text in normalized_ocr
            or similarity >= 0.78
        )

    @staticmethod
    def _count_char_mismatches(left: str, right: str) -> int:
        if len(left) != len(right):
            return max(len(left), len(right))
        return sum(1 for left_char, right_char in zip(left, right) if left_char != right_char)

    def _detect_scenes(self, video_path: Path, duration_seconds: float) -> list[dict[str, Any]]:
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=self.config.shot_threshold,
                min_scene_len=self.config.shot_min_scene_len,
            )
        )
        scene_manager.detect_scenes(video)
        raw_scene_list = scene_manager.get_scene_list()

        if not raw_scene_list:
            return [
                {
                    "scene_id": "scene-0001",
                    "start_seconds": 0.0,
                    "end_seconds": round(duration_seconds, 3),
                }
            ]

        scenes: list[dict[str, Any]] = []
        for index, (start_tc, end_tc) in enumerate(raw_scene_list, start=1):
            scenes.append(
                {
                    "scene_id": f"scene-{index:04d}",
                    "start_seconds": round(start_tc.get_seconds(), 3),
                    "end_seconds": round(end_tc.get_seconds(), 3),
                }
            )
        return scenes

    def _build_whisper_model(self) -> WhisperModel:
        preferred_device = self.config.asr_device.strip().lower()
        preferred_compute = self.config.asr_compute_type.strip().lower()

        candidates: list[tuple[str, str]] = []
        if preferred_device in {"auto", ""}:
            candidates = [
                ("cuda", "float16"),
                ("cuda", "int8_float16"),
                ("cpu", "int8"),
            ]
        elif preferred_compute in {"auto", ""}:
            if preferred_device == "cuda":
                candidates = [
                    ("cuda", "float16"),
                    ("cuda", "int8_float16"),
                    ("cpu", "int8"),
                ]
            else:
                candidates = [(preferred_device, "int8")]
        else:
            candidates = [(preferred_device, preferred_compute)]
            if preferred_device == "cuda":
                candidates.append(("cpu", "int8"))

        last_error: Exception | None = None
        for device, compute_type in candidates:
            try:
                model = WhisperModel(
                    self.config.asr_model_size,
                    device=device,
                    compute_type=compute_type,
                )
                self.last_asr_runtime = {
                    "device": device,
                    "compute_type": compute_type,
                    "model_size": self.config.asr_model_size,
                }
                self._log(f"ASR 已加载：device={device} compute={compute_type}")
                return model
            except Exception as exc:
                last_error = exc
                self._log(f"ASR 加载失败，尝试下一个后端：device={device} compute={compute_type} error={exc}")

        raise RuntimeError(f"Whisper ASR 初始化失败：{last_error}")

    def _select_keyframe_scenes(self, scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.config.max_keyframes is None or self.config.max_keyframes <= 0:
            return scenes
        if len(scenes) <= self.config.max_keyframes:
            return scenes

        sampled: list[dict[str, Any]] = []
        last_index = len(scenes) - 1
        for sample_index in range(self.config.max_keyframes):
            mapped = round(sample_index * last_index / max(self.config.max_keyframes - 1, 1))
            sampled.append(scenes[mapped])
        return sampled

    def _extract_ocr_sample_frames(
        self,
        ffmpeg_binary: str,
        video_path: Path,
        scenes: list[dict[str, Any]],
        ocr_frames_dir: Path,
    ) -> list[dict[str, Any]]:
        for old_frame in ocr_frames_dir.glob("*.jpg"):
            old_frame.unlink()

        samples: list[dict[str, Any]] = []
        interval = max(0.3, float(self.config.ocr_sample_interval_seconds))
        sample_index = 1
        for scene in scenes:
            start_seconds = float(scene["start_seconds"])
            end_seconds = float(scene["end_seconds"])
            duration = max(0.0, end_seconds - start_seconds)
            sample_count = min(3, max(1, int(duration / interval) + 1))
            if sample_count == 1:
                timestamps = [round((start_seconds + end_seconds) / 2.0, 3)]
            else:
                timestamps = []
                for offset_index in range(sample_count):
                    mapped = start_seconds + (offset_index + 1) * duration / (sample_count + 1)
                    timestamps.append(round(mapped, 3))
            for timestamp in timestamps:
                frame_path = ocr_frames_dir / f"{sample_index:04d}_{scene['scene_id']}.jpg"
                command = [
                    ffmpeg_binary,
                    "-y",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    str(frame_path),
                ]
                self._run_command(command)
                samples.append(
                    {
                        "sample_id": f"ocr-{sample_index:04d}",
                        "scene_id": scene["scene_id"],
                        "timestamp": timestamp,
                        "frame_path": str(frame_path.resolve()),
                    }
                )
                sample_index += 1
        return samples

    def _prepare_model_keyframes(
        self,
        keyframes: list[dict[str, Any]],
        model_frames_dir: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        for old_frame in model_frames_dir.glob("*.jpg"):
            old_frame.unlink()

        ratio = float(self.config.keyframe_scale_ratio)
        resized_count = 0
        copied_count = 0
        example: dict[str, str] = {}
        prepared: list[dict[str, Any]] = []
        for keyframe in keyframes:
            original_frame_path = Path(keyframe["frame_path"]).expanduser().resolve()
            with Image.open(original_frame_path) as image:
                original_size = image.size
                model_frame_path = model_frames_dir / original_frame_path.name
                if ratio > 0 and abs(ratio - 1.0) >= 0.001:
                    resized_width = max(2, int(original_size[0] * ratio))
                    resized_height = max(2, int(original_size[1] * ratio))
                    resized = image.resize((resized_width, resized_height), Image.LANCZOS)
                    resized.save(model_frame_path, quality=88)
                    model_size = (resized_width, resized_height)
                    resized_count += 1
                else:
                    image.copy().save(model_frame_path, quality=92)
                    model_size = original_size
                    copied_count += 1
            if not example:
                example = {
                    "original": f"{original_size[0]}x{original_size[1]}",
                    "model": f"{model_size[0]}x{model_size[1]}",
                }
            prepared.append(
                {
                    **keyframe,
                    "model_frame_path": str(model_frame_path.resolve()),
                    "original_resolution": {"width": original_size[0], "height": original_size[1]},
                    "model_resolution": {"width": model_size[0], "height": model_size[1]},
                    "resize_applied": str(model_frame_path.resolve()) != str(original_frame_path),
                }
            )
        return prepared, {
            "scale_ratio": ratio,
            "resized_count": resized_count,
            "copied_without_resize_count": copied_count,
            "original_frames_dir": str((model_frames_dir.parent / "frames").resolve()),
            "model_frames_dir": str(model_frames_dir.resolve()),
            "original_frames_deleted": True,
            "example": example,
        }

    def _extract_keyframes(
        self,
        ffmpeg_binary: str,
        video_path: Path,
        scenes: list[dict[str, Any]],
        frames_dir: Path,
    ) -> list[dict[str, Any]]:
        for old_frame in frames_dir.glob("*.jpg"):
            old_frame.unlink()

        extracted: list[dict[str, Any]] = []
        for index, scene in enumerate(scenes, start=1):
            midpoint = (scene["start_seconds"] + scene["end_seconds"]) / 2.0
            frame_path = frames_dir / f"{index:04d}_{scene['scene_id']}.jpg"
            command = [
                ffmpeg_binary,
                "-y",
                "-ss",
                f"{midpoint:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(frame_path),
            ]
            self._run_command(command)
            extracted.append(
                {
                    **scene,
                    "midpoint_seconds": round(midpoint, 3),
                    "frame_path": str(frame_path.resolve()),
                }
            )
        return extracted

    def _run_ocr(self, keyframes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
        ocr_engine = RapidOCR()
        ocr_segments: list[dict[str, Any]] = []
        ocr_lines: list[str] = []
        previous_text = ""

        for item in keyframes:
            frame_path = Path(item["frame_path"])
            with Image.open(frame_path) as image:
                width, height = image.size
                crop_top = max(0, int(height * (1.0 - self.config.ocr_crop_bottom_ratio)))
                cropped = image.crop((0, crop_top, width, height))
                np_image = np.array(cropped)

            result, _ = ocr_engine(np_image)
            texts: list[str] = []
            for row in result or []:
                if len(row) < 2:
                    continue
                text = str(row[1]).strip()
                if text:
                    texts.append(text)

            merged_text = " ".join(texts).strip()
            if not merged_text or merged_text == previous_text:
                continue

            previous_text = merged_text
            entry = {
                "sample_id": item.get("sample_id", ""),
                "scene_id": item.get("scene_id", ""),
                "timestamp": item.get("timestamp", item.get("midpoint_seconds", 0.0)),
                "frame_path": item["frame_path"],
                "text": merged_text,
            }
            ocr_segments.append(entry)
            ocr_lines.append(f"[{float(entry['timestamp']):.3f}] {merged_text}")

        return ocr_segments, "\n".join(ocr_lines).strip()

    def _attach_ocr_segments_to_keyframes(
        self,
        keyframes: list[dict[str, Any]],
        ocr_segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        interval_threshold = max(float(self.config.ocr_sample_interval_seconds) * 1.5, 0.75)
        for keyframe in keyframes:
            start_seconds = float(keyframe["start_seconds"])
            end_seconds = float(keyframe["end_seconds"])
            midpoint_seconds = float(keyframe["midpoint_seconds"])
            linked = [
                segment
                for segment in ocr_segments
                if start_seconds <= float(segment.get("timestamp", 0.0)) <= end_seconds
            ]
            if not linked and ocr_segments:
                nearest = min(
                    ocr_segments,
                    key=lambda segment: abs(float(segment.get("timestamp", 0.0)) - midpoint_seconds),
                )
                if abs(float(nearest.get("timestamp", 0.0)) - midpoint_seconds) <= interval_threshold:
                    linked = [nearest]
            linked_texts: list[str] = []
            for segment in linked:
                text = str(segment.get("text", "")).strip()
                if text and text not in linked_texts:
                    linked_texts.append(text)
            result.append(
                {
                    **keyframe,
                    "linked_ocr_segments": linked,
                    "linked_ocr_text": " / ".join(linked_texts),
                }
            )
        return result

    def _asr_config_summary(self) -> dict[str, Any]:
        return {
            "model_size": self.config.asr_model_size,
            "device": self.config.asr_device,
            "compute_type": self.config.asr_compute_type,
            "beam_size": self.config.asr_beam_size,
            "best_of": self.config.asr_best_of,
            "patience": self.config.asr_patience,
            "vad_filter": self.config.asr_vad_filter,
            "enable_dual_track_fusion": self.config.asr_enable_dual_track_fusion,
            "vad_min_silence_duration_ms": self.config.asr_vad_min_silence_duration_ms,
            "vad_speech_pad_ms": self.config.asr_vad_speech_pad_ms,
            "no_speech_threshold": self.config.asr_no_speech_threshold,
            "log_prob_threshold": self.config.asr_log_prob_threshold,
            "hallucination_silence_threshold": self.config.asr_hallucination_silence_threshold,
            "vad_chunk_merge_gap_seconds": self.config.asr_vad_chunk_merge_gap_seconds,
            "vad_chunk_max_seconds": self.config.asr_vad_chunk_max_seconds,
            "condition_on_previous_text": self.config.asr_condition_on_previous_text,
            "initial_prompt": self.config.asr_initial_prompt,
            "hotwords": self.config.asr_hotwords,
            "enable_second_pass": self.config.asr_enable_second_pass,
            "second_pass_trigger_tolerance_seconds": self.config.asr_second_pass_trigger_tolerance_seconds,
            "second_pass_window_padding_seconds": self.config.asr_second_pass_window_padding_seconds,
            "second_pass_max_window_seconds": self.config.asr_second_pass_max_window_seconds,
            "transcript_refine_enabled": self.config.transcript_refine_enabled,
            "transcript_refine_provider": self.config.transcript_refine_provider,
            "transcript_refine_model": self.config.transcript_refine_model,
            "transcript_refine_endpoint": self.config.transcript_refine_endpoint,
            "transcript_refine_timeout_seconds": self.config.transcript_refine_timeout_seconds,
            "transcript_refine_batch_size": self.config.transcript_refine_batch_size,
        }

    def _purge_directory(self, directory: Path) -> None:
        if directory.exists():
            shutil.rmtree(directory)

    def _mark_source_artifacts_deleted(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        marked: list[dict[str, Any]] = []
        for item in items:
            marked.append(
                {
                    **item,
                    "source_frame_deleted": True,
                }
            )
        return marked

    def _run_command(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess a single episode video into ASR/OCR/shot assets.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--series-name")
    parser.add_argument("--output-root", default="analysis")
    parser.add_argument("--asr-model-size", default="small")
    parser.add_argument("--asr-language", default="zh")
    parser.add_argument("--asr-device", default="auto")
    parser.add_argument("--asr-compute-type", default="auto")
    parser.add_argument("--asr-beam-size", type=int, default=5)
    parser.add_argument("--asr-best-of", type=int, default=5)
    parser.add_argument("--asr-patience", type=float, default=1.0)
    parser.add_argument("--asr-condition-on-previous-text", action="store_true")
    parser.add_argument("--no-asr-condition-on-previous-text", action="store_true")
    parser.add_argument("--asr-initial-prompt", default="")
    parser.add_argument("--asr-hotwords", default="")
    parser.add_argument("--asr-vad-filter", action="store_true")
    parser.add_argument("--no-asr-vad-filter", action="store_true")
    parser.add_argument("--asr-enable-dual-track-fusion", action="store_true")
    parser.add_argument("--no-asr-enable-dual-track-fusion", action="store_true")
    parser.add_argument("--asr-vad-threshold", type=float, default=0.5)
    parser.add_argument("--asr-vad-neg-threshold", type=float, default=None)
    parser.add_argument("--asr-vad-min-speech-duration-ms", type=int, default=0)
    parser.add_argument("--asr-vad-max-speech-duration-seconds", type=float, default=8.0)
    parser.add_argument("--asr-vad-min-silence-duration-ms", type=int, default=700)
    parser.add_argument("--asr-vad-speech-pad-ms", type=int, default=320)
    parser.add_argument("--asr-no-speech-threshold", type=float, default=0.45)
    parser.add_argument("--asr-log-prob-threshold", type=float, default=-1.0)
    parser.add_argument("--asr-hallucination-silence-threshold", type=float, default=0.8)
    parser.add_argument("--asr-vad-chunk-merge-gap-seconds", type=float, default=0.2)
    parser.add_argument("--asr-vad-chunk-max-seconds", type=float, default=12.0)
    parser.add_argument("--asr-enable-second-pass", action="store_true")
    parser.add_argument("--no-asr-enable-second-pass", action="store_true")
    parser.add_argument("--asr-second-pass-trigger-tolerance-seconds", type=float, default=1.0)
    parser.add_argument("--asr-second-pass-window-padding-seconds", type=float, default=1.2)
    parser.add_argument("--asr-second-pass-max-window-seconds", type=float, default=12.0)
    parser.add_argument("--transcript-refine-enabled", action="store_true")
    parser.add_argument("--no-transcript-refine-enabled", action="store_true")
    parser.add_argument("--transcript-refine-provider", default="qwen")
    parser.add_argument("--transcript-refine-model", default="")
    parser.add_argument("--transcript-refine-endpoint", default="")
    parser.add_argument("--transcript-refine-timeout-seconds", type=int, default=180)
    parser.add_argument("--transcript-refine-batch-size", type=int, default=10)
    parser.add_argument("--shot-threshold", type=float, default=27.0)
    parser.add_argument("--shot-min-scene-len", type=int, default=15)
    parser.add_argument("--max-keyframes", type=int)
    parser.add_argument("--ocr-crop-bottom-ratio", type=float, default=0.35)
    parser.add_argument("--ocr-sample-interval-seconds", type=float, default=0.5)
    parser.add_argument("--keyframe-scale-ratio", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    asr_condition_on_previous_text = True
    if args.no_asr_condition_on_previous_text:
        asr_condition_on_previous_text = False
    elif args.asr_condition_on_previous_text:
        asr_condition_on_previous_text = True

    asr_vad_filter = True
    if args.no_asr_vad_filter:
        asr_vad_filter = False
    elif args.asr_vad_filter:
        asr_vad_filter = True

    asr_enable_dual_track_fusion = False
    if args.no_asr_enable_dual_track_fusion:
        asr_enable_dual_track_fusion = False
    elif args.asr_enable_dual_track_fusion:
        asr_enable_dual_track_fusion = True

    asr_enable_second_pass = True
    if args.no_asr_enable_second_pass:
        asr_enable_second_pass = False
    elif args.asr_enable_second_pass:
        asr_enable_second_pass = True

    transcript_refine_enabled = False
    if args.no_transcript_refine_enabled:
        transcript_refine_enabled = False
    elif args.transcript_refine_enabled:
        transcript_refine_enabled = True

    preprocessor = EpisodePreprocessor(
        EpisodePreprocessConfig(
            output_root=Path(args.output_root),
            asr_model_size=args.asr_model_size,
            asr_language=args.asr_language,
            asr_device=args.asr_device,
            asr_compute_type=args.asr_compute_type,
            asr_beam_size=args.asr_beam_size,
            asr_best_of=args.asr_best_of,
            asr_patience=args.asr_patience,
            asr_condition_on_previous_text=asr_condition_on_previous_text,
            asr_initial_prompt=args.asr_initial_prompt,
            asr_hotwords=args.asr_hotwords,
            asr_vad_filter=asr_vad_filter,
            asr_enable_dual_track_fusion=asr_enable_dual_track_fusion,
            asr_vad_threshold=args.asr_vad_threshold,
            asr_vad_neg_threshold=args.asr_vad_neg_threshold,
            asr_vad_min_speech_duration_ms=args.asr_vad_min_speech_duration_ms,
            asr_vad_max_speech_duration_seconds=args.asr_vad_max_speech_duration_seconds,
            asr_vad_min_silence_duration_ms=args.asr_vad_min_silence_duration_ms,
            asr_vad_speech_pad_ms=args.asr_vad_speech_pad_ms,
            asr_no_speech_threshold=args.asr_no_speech_threshold,
            asr_log_prob_threshold=args.asr_log_prob_threshold,
            asr_hallucination_silence_threshold=args.asr_hallucination_silence_threshold,
            asr_vad_chunk_merge_gap_seconds=args.asr_vad_chunk_merge_gap_seconds,
            asr_vad_chunk_max_seconds=args.asr_vad_chunk_max_seconds,
            asr_enable_second_pass=asr_enable_second_pass,
            asr_second_pass_trigger_tolerance_seconds=args.asr_second_pass_trigger_tolerance_seconds,
            asr_second_pass_window_padding_seconds=args.asr_second_pass_window_padding_seconds,
            asr_second_pass_max_window_seconds=args.asr_second_pass_max_window_seconds,
            transcript_refine_enabled=transcript_refine_enabled,
            transcript_refine_provider=args.transcript_refine_provider,
            transcript_refine_model=args.transcript_refine_model,
            transcript_refine_endpoint=args.transcript_refine_endpoint,
            transcript_refine_timeout_seconds=args.transcript_refine_timeout_seconds,
            transcript_refine_batch_size=args.transcript_refine_batch_size,
            shot_threshold=args.shot_threshold,
            shot_min_scene_len=args.shot_min_scene_len,
            max_keyframes=args.max_keyframes,
            ocr_crop_bottom_ratio=args.ocr_crop_bottom_ratio,
            ocr_sample_interval_seconds=args.ocr_sample_interval_seconds,
            keyframe_scale_ratio=args.keyframe_scale_ratio,
        )
    )
    result = preprocessor.run(args.video, args.episode_id, args.series_name)
    print(f"manifest={result.manifest_path}")


if __name__ == "__main__":
    main()
