from __future__ import annotations

import json
import re
from typing import Any, Mapping

from genre_routing import (
    build_confirmed_genre_block,
    enforce_user_genre_alignment,
    resolve_pre_analysis_genre_routing,
)
from pipeline_telemetry import TelemetryRecorder, apply_provider_usage, telemetry_span
from prompt_utils import load_prompt, render_prompt
from providers.base import (
    EpisodeAnalysisProvider,
    EpisodeInputBundle,
    ProviderCapabilities,
    ProviderConfig,
    ProviderResponseError,
    SchemaValidationError,
    ensure_object_field,
    extract_json_from_text,
    file_to_data_url,
    truncate_text,
    utc_timestamp,
    validate_against_schema,
)

ANALYSIS_SYSTEM_PROMPT = load_prompt("video_pipeline/openai_analysis_system.md")
SCRIPT_SYSTEM_PROMPT = load_prompt("video_pipeline/openai_script_system.md")


LEGACY_ROOT_SERIES_LEARNING_KEYS = {
    "reusable_playbook_rules",
    "reusable_skill_rules",
    "character_appeal_patterns",
    "scene_staging_patterns",
    "dialogue_patterns",
    "camera_language_patterns",
    "costume_image_patterns",
    "prop_visual_patterns",
    "storyboard_execution_patterns",
}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _flatten_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, item in value.items():
            nested = _flatten_texts(item)
            if nested:
                result.extend(
                    f"{_as_text(key)}：{entry}" if _as_text(key) else entry
                    for entry in nested
                )
            else:
                text = _as_text(item)
                if text:
                    result.append(f"{_as_text(key)}：{text}" if _as_text(key) else text)
        return result
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_texts(item))
        return result
    text = _as_text(value)
    return [text] if text else []


def _as_string_list(value: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in _flatten_texts(value):
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _parse_time_range(raw: Any) -> tuple[str, str]:
    text = _as_text(raw)
    if not text:
        return "", ""
    match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*[-~—–]+\s*([0-9]+(?:\.[0-9]+)?)\s*$", text)
    if match:
        return match.group(1), match.group(2)
    return text, text


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_evidence_list(value: Any) -> list[dict[str, str]]:
    allowed_source_types = {"video", "frame", "transcript", "ocr", "metadata"}
    items = value if isinstance(value, list) else [value]
    result: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        source_type = _as_text(item.get("source_type"))
        if source_type not in allowed_source_types:
            source_type = "metadata"
        quote_or_description = _as_text(item.get("quote_or_description"))
        timestamp = _as_text(item.get("timestamp"))
        if not quote_or_description:
            continue
        result.append(
            {
                "source_type": source_type,
                "timestamp": timestamp,
                "quote_or_description": quote_or_description,
            }
        )
    return result


class QwenAdapter(EpisodeAnalysisProvider):
    capabilities = ProviderCapabilities(
        supports_structured_output=True,
        supports_image_inputs=True,
        supports_video_inputs=True,
    )

    def __init__(
        self,
        model: str = "qwen3-vl-plus",
        *,
        api_key_env: str = "DASHSCOPE_API_KEY",
        endpoint: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        temperature: float = 0.2,
        timeout_seconds: int = 300,
        max_retries: int = 2,
        max_analysis_frames: int | None = 20,
        video_fps: float = 2.0,
        structured_output_mode: str = "json_schema",
        telemetry: TelemetryRecorder | None = None,
    ) -> None:
        super().__init__(
            ProviderConfig(
                name="qwen",
                model=model,
                api_key_env=api_key_env,
                endpoint=endpoint,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        )
        self.telemetry = telemetry
        self.max_analysis_frames = (
            None if max_analysis_frames is None or int(max_analysis_frames) <= 0 else int(max_analysis_frames)
        )
        self.video_fps = max(float(video_fps), 0.1)
        normalized_mode = str(structured_output_mode or "json_schema").strip().lower()
        if normalized_mode not in {"json_schema", "json_object"}:
            raise ValueError(f"不支持的 Qwen structured_output_mode：{structured_output_mode}")
        self.structured_output_mode = normalized_mode

    def analyze_episode(
        self,
        bundle: EpisodeInputBundle,
        schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        bundle.validate()
        input_mode, selected_frames = self._resolve_analysis_input(bundle)

        with telemetry_span(
            self.telemetry,
            stage="video_analysis",
            name="build_qwen_episode_analysis_request",
            provider="qwen",
            model=self.config.model,
            metadata={
                "episode_id": bundle.episode_id,
                "frame_count": len(bundle.frames),
                "transcript_chars": len(bundle.transcript_text or ""),
                "ocr_chars": len(bundle.ocr_text or ""),
                "has_video": bool(bundle.video_path),
                "input_mode": input_mode,
                "selected_frame_count": len(selected_frames),
            },
        ) as step:
            content = self._build_analysis_content(bundle, input_mode=input_mode, selected_frames=selected_frames)
            payload = {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "messages": [
                    {
                        "role": "system",
                        "content": ANALYSIS_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
                "response_format": self._build_analysis_response_format(schema, self.structured_output_mode),
            }
            step["metadata"]["content_parts"] = len(content)
            step["metadata"]["structured_output_mode"] = self.structured_output_mode

        with telemetry_span(
            self.telemetry,
            stage="video_analysis",
            name="qwen_episode_analysis_model_call",
            provider="qwen",
            model=self.config.model,
            metadata={
                "episode_id": bundle.episode_id,
                "input_mode": input_mode,
                "selected_frame_count": len(selected_frames),
            },
        ) as step:
            response = self.request_json(
                self.config.endpoint,
                headers={"Authorization": f"Bearer {self.require_api_key()}"},
                payload=payload,
            )
            apply_provider_usage(step, "qwen", response)
            step["metadata"]["structured_output_mode"] = self.structured_output_mode

        output_text = self._extract_output_text(response)
        try:
            return self._finalize_analysis_output(bundle, output_text, schema)
        except (ProviderResponseError, SchemaValidationError) as exc:
            if self.structured_output_mode != "json_schema":
                raise
            fallback_payload = dict(payload)
            fallback_payload["response_format"] = self._build_analysis_response_format(schema, "json_object")
            with telemetry_span(
                self.telemetry,
                stage="video_analysis",
                name="qwen_episode_analysis_json_object_fallback",
                provider="qwen",
                model=self.config.model,
                metadata={
                    "episode_id": bundle.episode_id,
                    "reason": str(exc),
                    "input_mode": input_mode,
                    "selected_frame_count": len(selected_frames),
                },
            ) as step:
                fallback_response = self.request_json(
                    self.config.endpoint,
                    headers={"Authorization": f"Bearer {self.require_api_key()}"},
                    payload=fallback_payload,
                )
                apply_provider_usage(step, "qwen", fallback_response)
            fallback_output_text = self._extract_output_text(fallback_response)
            return self._finalize_analysis_output(bundle, fallback_output_text, schema)

    def reconstruct_script(
        self,
        bundle: EpisodeInputBundle,
        analysis: Mapping[str, Any],
    ) -> str:
        with telemetry_span(
            self.telemetry,
            stage="script_reconstruction",
            name="build_qwen_script_reconstruction_request",
            provider="qwen",
            model=self.config.model,
            metadata={"episode_id": bundle.episode_id},
        ):
            payload = {
                "model": self.config.model,
                "temperature": max(self.config.temperature, 0.4),
                "messages": [
                    {
                        "role": "system",
                        "content": SCRIPT_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": self._build_script_prompt(bundle, analysis),
                    },
                ],
            }

        with telemetry_span(
            self.telemetry,
            stage="script_reconstruction",
            name="qwen_script_reconstruction_model_call",
            provider="qwen",
            model=self.config.model,
            metadata={"episode_id": bundle.episode_id},
        ) as step:
            response = self.request_json(
                self.config.endpoint,
                headers={"Authorization": f"Bearer {self.require_api_key()}"},
                payload=payload,
            )
            apply_provider_usage(step, "qwen", response)
        return self._extract_output_text(response).strip()

    def _resolve_analysis_input(self, bundle: EpisodeInputBundle) -> tuple[str, list[Any]]:
        selected_frames = self._select_analysis_frames(bundle.frames)
        if selected_frames:
            return "frames_sequence", selected_frames
        if bundle.video_path:
            return "raw_video", []
        return "text_only", []

    def _build_analysis_content(
        self,
        bundle: EpisodeInputBundle,
        *,
        input_mode: str,
        selected_frames: list[Any],
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []

        if input_mode == "frames_sequence":
            content.append(
                {
                    "type": "video",
                    "video": [
                        file_to_data_url(frame.resolved_path(), frame.detected_mime_type())
                        for frame in selected_frames
                    ],
                    "fps": self.video_fps,
                }
            )
        elif input_mode == "raw_video":
            resolved_video = bundle.resolved_video_path()
            if resolved_video:
                content.append(
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": file_to_data_url(resolved_video),
                        },
                        "fps": self.video_fps,
                    }
                )

        content.append(
            {
                "type": "text",
                "text": self._build_analysis_prompt(bundle),
            }
        )
        return content

    def _select_analysis_frames(self, frames: list[Any]) -> list[Any]:
        if not self.max_analysis_frames or len(frames) <= self.max_analysis_frames:
            return list(frames)
        if self.max_analysis_frames == 1:
            return [frames[0]]
        last_index = len(frames) - 1
        selected_indices = {
            round(position * last_index / (self.max_analysis_frames - 1))
            for position in range(self.max_analysis_frames)
        }
        return [frame for index, frame in enumerate(frames) if index in selected_indices]

    def _build_analysis_prompt(self, bundle: EpisodeInputBundle) -> str:
        transcript_block = ""
        if bundle.transcript_text:
            transcript_block = "ASR Transcript（对白主证据，可能已做 OCR 保守校字）：\n" + truncate_text(bundle.transcript_text, 25000)
        ocr_block = ""
        if bundle.ocr_text:
            ocr_block = "OCR / 屏幕文字（仅作校字与识别画面文字参考）：\n" + truncate_text(bundle.ocr_text, 12000)
        routing = resolve_pre_analysis_genre_routing(bundle)
        return render_prompt(
            "video_pipeline/openai_analysis_user.md",
            {
                "input_summary": bundle.as_prompt_summary(),
                "transcript_block": transcript_block,
                "ocr_block": ocr_block,
                "skill_text": truncate_text(routing.combined_skill_text(), 12000),
                "genre_routing_note": truncate_text(routing.routing_note_text(), 2000),
                "confirmed_genre_block": truncate_text(build_confirmed_genre_block(bundle), 2000),
                "playbook_reference_block": truncate_text(routing.playbook_reference_text(), 9000),
            },
        )

    def _build_analysis_response_format(
        self,
        schema: Mapping[str, Any],
        mode: str,
    ) -> dict[str, Any]:
        if mode == "json_object":
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "episode_analysis",
                "schema": schema,
                "strict": True,
            },
        }

    def _finalize_analysis_output(
        self,
        bundle: EpisodeInputBundle,
        output_text: str,
        schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        analysis = extract_json_from_text(output_text)
        self._normalize_analysis_payload(bundle, analysis, schema)
        self._hydrate_provider_metadata(bundle, analysis)
        enforce_user_genre_alignment(bundle, analysis)
        validate_against_schema(analysis, schema)
        return analysis

    def _normalize_analysis_payload(
        self,
        bundle: EpisodeInputBundle,
        analysis: dict[str, Any],
        schema: Mapping[str, Any],
    ) -> None:
        episode = ensure_object_field(analysis, "episode")
        if "episode_id" in analysis and not _as_text(episode.get("episode_id")):
            episode["episode_id"] = _as_text(analysis.pop("episode_id"))
        else:
            analysis.pop("episode_id", None)
        if "title" in analysis and not _as_text(episode.get("title")):
            episode["title"] = _as_text(analysis.pop("title"))
        else:
            analysis.pop("title", None)
        if "language" in analysis and not _as_text(episode.get("language")):
            episode["language"] = _as_text(analysis.pop("language"))
        else:
            analysis.pop("language", None)
        analysis.pop("source_series", None)

        learning = self._normalize_series_learning_extraction(analysis)
        analysis["series_learning_extraction"] = learning
        analysis["characters"] = self._normalize_characters(analysis.get("characters"))
        analysis["locations"] = self._normalize_locations(analysis.get("locations"))
        analysis["dialogue_segments"] = self._normalize_dialogue_segments(analysis.get("dialogue_segments"))
        analysis["story_beats"] = self._normalize_story_beats(analysis.get("story_beats"))
        analysis["genre_classification"] = self._normalize_genre_classification(analysis.get("genre_classification"))
        analysis["genre_override_request"] = self._normalize_genre_override_request(analysis.get("genre_override_request"))
        analysis["hook_profile"] = self._normalize_hook_profile(analysis.get("hook_profile"))
        analysis["camera_language_analysis"] = self._normalize_camera_language_analysis(
            analysis.get("camera_language_analysis")
        )
        analysis["art_direction_analysis"] = self._normalize_art_direction_analysis(analysis.get("art_direction_analysis"))
        analysis["storyboard_blueprint"] = self._normalize_storyboard_blueprint(analysis.get("storyboard_blueprint"))
        analysis["downstream_design_guidance"] = self._normalize_downstream_design_guidance(
            analysis.get("downstream_design_guidance")
        )
        analysis["quality_assessment"] = self._normalize_quality_assessment(
            analysis.get("quality_assessment"),
            analysis.get("genre_classification"),
        )
        analysis["continuity_notes"] = _as_string_list(analysis.get("continuity_notes"))
        analysis["adaptation_hints"] = _as_string_list(analysis.get("adaptation_hints"))
        analysis["synopsis"] = _as_text(analysis.get("synopsis"))

        allowed_top_level = set(dict(schema.get("properties", {})).keys())
        for key in list(analysis.keys()):
            if key not in allowed_top_level:
                analysis.pop(key, None)

    def _normalize_series_learning_extraction(self, analysis: dict[str, Any]) -> dict[str, Any]:
        raw_learning = analysis.get("series_learning_extraction")
        learning = ensure_object_field(
            analysis,
            "series_learning_extraction",
            {
                "episode_strengths": [],
                "why_it_works": [],
                "character_design_rules": [],
                "costume_makeup_rules": [],
                "scene_design_rules": [],
                "camera_language_rules": [],
                "storyboard_execution_rules": [],
                "dialogue_timing_rules": [],
                "continuity_guardrails": [],
                "negative_patterns": [],
                "reusable_playbook_rules": [],
                "reusable_skill_rules": [],
                "character_appeal_patterns": [],
                "scene_staging_patterns": [],
                "dialogue_patterns": [],
                "camera_language_patterns": [],
                "costume_image_patterns": [],
                "prop_visual_patterns": [],
                "storyboard_execution_patterns": [],
            },
        )
        if isinstance(raw_learning, Mapping):
            for key in list(learning.keys()):
                learning[key] = _as_string_list(raw_learning.get(key, learning.get(key, [])))
        else:
            for key in list(learning.keys()):
                learning[key] = _as_string_list(learning.get(key, []))

        for key in LEGACY_ROOT_SERIES_LEARNING_KEYS:
            if key in analysis:
                merged = list(learning.get(key, [])) + _as_string_list(analysis.pop(key))
                learning[key] = _as_string_list(merged)
        return learning

    def _normalize_characters(self, value: Any) -> list[dict[str, Any]]:
        items = value if isinstance(value, list) else []
        result: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, Mapping):
                continue
            name = _as_text(item.get("name") or item.get("canonical_name"))
            aliases = _as_string_list(item.get("aliases"))
            if name and name not in aliases:
                aliases.insert(0, name)
            result.append(
                {
                    "character_id": _as_text(item.get("character_id") or item.get("id") or f"C{index:02d}"),
                    "name": name,
                    "aliases": aliases,
                    "role": _as_text(item.get("role") or item.get("identity")),
                    "relationship_to_protagonist": _as_text(
                        item.get("relationship_to_protagonist") or item.get("relationship")
                    ),
                    "visual_profile": _as_text(item.get("visual_profile") or item.get("visual_traits")),
                    "current_state": _as_text(item.get("current_state") or item.get("state") or item.get("latest_state")),
                    "evidence": _normalize_evidence_list(item.get("evidence")),
                }
            )
        return result

    def _normalize_locations(self, value: Any) -> list[dict[str, Any]]:
        items = value if isinstance(value, list) else []
        result: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, Mapping):
                continue
            result.append(
                {
                    "location_id": _as_text(item.get("location_id") or item.get("id") or f"L{index:02d}"),
                    "name": _as_text(item.get("name")),
                    "time_of_day": _as_text(item.get("time_of_day") or item.get("time")),
                    "visual_profile": _as_text(item.get("visual_profile") or item.get("description")),
                    "props": _as_string_list(item.get("props")),
                    "evidence": _normalize_evidence_list(item.get("evidence")),
                }
            )
        return result

    def _normalize_dialogue_segments(self, value: Any) -> list[dict[str, Any]]:
        items = value if isinstance(value, list) else []
        result: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, Mapping):
                continue
            start_time = _as_text(item.get("start_time"))
            end_time = _as_text(item.get("end_time"))
            if (not start_time or not end_time) and item.get("timestamp"):
                start_time, end_time = _parse_time_range(item.get("timestamp"))
            result.append(
                {
                    "line_id": _as_text(item.get("line_id") or item.get("id") or f"D{index:02d}"),
                    "start_time": start_time,
                    "end_time": end_time,
                    "speaker": _as_text(item.get("speaker")),
                    "text": _as_text(item.get("text") or item.get("line")),
                    "confidence": max(0.0, min(1.0, _as_float(item.get("confidence"), 0.8))),
                }
            )
        return result

    def _normalize_story_beats(self, value: Any) -> list[dict[str, Any]]:
        items = value if isinstance(value, list) else []
        result: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, Mapping):
                continue
            start_time = _as_text(item.get("start_time"))
            end_time = _as_text(item.get("end_time"))
            if (not start_time or not end_time) and item.get("timestamp"):
                start_time, end_time = _parse_time_range(item.get("timestamp"))
            result.append(
                {
                    "beat_id": _as_text(item.get("beat_id") or f"B{index:02d}"),
                    "title": _as_text(item.get("title") or item.get("beat_name")),
                    "summary": _as_text(item.get("summary") or item.get("description")),
                    "start_time": start_time,
                    "end_time": end_time,
                    "location": _as_text(item.get("location")),
                    "time_of_day": _as_text(item.get("time_of_day")),
                    "characters": _as_string_list(item.get("characters")),
                    "key_actions": _as_string_list(item.get("key_actions")),
                    "plot_function": _as_text(item.get("plot_function")),
                    "emotional_turn": _as_text(item.get("emotional_turn")),
                    "visual_focus": _as_string_list(item.get("visual_focus")),
                    "camera_language": _as_string_list(item.get("camera_language")),
                    "art_direction_cues": _as_string_list(item.get("art_direction_cues")),
                    "storyboard_value": _as_string_list(item.get("storyboard_value")),
                    "dialogue_line_ids": _as_string_list(item.get("dialogue_line_ids")),
                    "evidence": _normalize_evidence_list(item.get("evidence")),
                }
            )
        return result

    def _normalize_genre_classification(self, value: Any) -> dict[str, Any]:
        data = dict(value) if isinstance(value, Mapping) else {}
        return {
            "primary_genre": _as_text(data.get("primary_genre")),
            "secondary_genres": _as_string_list(data.get("secondary_genres")),
            "confirmed_user_genres": _as_string_list(data.get("confirmed_user_genres")),
            "genre_resolution_mode": _as_text(data.get("genre_resolution_mode")),
            "narrative_device": _as_text(data.get("narrative_device")),
            "setting_era": _as_text(data.get("setting_era")),
            "audience_expectation": _as_text(data.get("audience_expectation")),
            "confidence": max(0.0, min(1.0, _as_float(data.get("confidence"), 0.0))),
            "evidence": _normalize_evidence_list(data.get("evidence")),
        }

    def _normalize_genre_override_request(self, value: Any) -> dict[str, Any]:
        data = dict(value) if isinstance(value, Mapping) else {}
        return {
            "needs_user_confirmation": bool(data.get("needs_user_confirmation", False)),
            "proposed_primary_genre": _as_text(data.get("proposed_primary_genre")),
            "proposed_secondary_genres": _as_string_list(data.get("proposed_secondary_genres")),
            "proposed_new_genres": _as_string_list(data.get("proposed_new_genres")),
            "reason": _as_text(data.get("reason")),
        }

    def _normalize_hook_profile(self, value: Any) -> dict[str, Any]:
        data = dict(value) if isinstance(value, Mapping) else {}
        return {
            "opening_hook": _as_text(data.get("opening_hook")),
            "episode_hook_types": _as_string_list(data.get("episode_hook_types") or data.get("hook_types")),
            "emotional_payoff_points": _as_string_list(
                data.get("emotional_payoff_points") or data.get("emotional_fulfillment")
            ),
            "cliffhanger_strategy": _as_text(data.get("cliffhanger_strategy") or data.get("ending_button")),
            "viral_moments": _as_string_list(data.get("viral_moments")),
        }

    def _normalize_camera_language_analysis(self, value: Any) -> dict[str, Any]:
        data = dict(value) if isinstance(value, Mapping) else {}
        return {
            "dominant_shot_types": _as_string_list(
                data.get("dominant_shot_types") or data.get("most_effective_lens_type")
            ),
            "camera_motion_patterns": _as_string_list(
                data.get("camera_motion_patterns") or data.get("dominant_camera_movement")
            ),
            "composition_patterns": _as_string_list(
                data.get("composition_patterns") or data.get("composition_center")
            ),
            "visual_emphasis_rules": _as_string_list(data.get("visual_emphasis_rules")),
            "transition_rhythm": _as_text(data.get("transition_rhythm")),
            "climax_visual_strategy": _as_text(data.get("climax_visual_strategy")),
            "cliffhanger_visual_pattern": _as_text(
                data.get("cliffhanger_visual_pattern") or data.get("ending_cliffhanger_visual_pattern")
            ),
        }

    def _normalize_art_direction_analysis(self, value: Any) -> dict[str, Any]:
        data = dict(value) if isinstance(value, Mapping) else {}
        character_costume = dict(data.get("character_costume", {})) if isinstance(data.get("character_costume"), Mapping) else {}
        space_design = dict(data.get("space_design", {})) if isinstance(data.get("space_design"), Mapping) else {}
        return {
            "costume_signatures": _as_string_list(
                [
                    character_costume.get("female_lead", {}).get("silhouette") if isinstance(character_costume.get("female_lead"), Mapping) else None,
                    character_costume.get("male_lead", {}).get("silhouette") if isinstance(character_costume.get("male_lead"), Mapping) else None,
                    character_costume.get("supporting"),
                ]
            ),
            "hair_makeup_signatures": _as_string_list(
                [
                    character_costume.get("female_lead", {}).get("hair_makeup") if isinstance(character_costume.get("female_lead"), Mapping) else None,
                    character_costume.get("male_lead", {}).get("hair_makeup") if isinstance(character_costume.get("male_lead"), Mapping) else None,
                ]
            ),
            "prop_signatures": _as_string_list(data.get("prop_signatures") or data.get("props")),
            "set_signatures": _as_string_list(
                data.get("set_signatures") or [space_design.get("primary_location")]
            ),
            "lighting_signatures": _as_string_list(
                data.get("lighting_signatures") or [space_design.get("lighting_scheme")]
            ),
            "color_mood_patterns": _as_string_list(
                data.get("color_mood_patterns") or [space_design.get("color_palette")]
            ),
            "texture_material_patterns": _as_string_list(
                data.get("texture_material_patterns")
                or [space_design.get("texture_memory_points"), data.get("visual_motif")]
            ),
        }

    def _normalize_storyboard_blueprint(self, value: Any) -> dict[str, Any]:
        data = dict(value) if isinstance(value, Mapping) else {}
        return {
            "opening_hook_blueprint": _as_string_list(
                data.get("opening_hook_blueprint") or data.get("opening_hook")
            ),
            "conflict_escalation_blueprint": _as_string_list(
                data.get("conflict_escalation_blueprint") or data.get("conflict_escalation")
            ),
            "emotional_payoff_blueprint": _as_string_list(
                data.get("emotional_payoff_blueprint") or data.get("emotional_fulfillment")
            ),
            "ending_button_blueprint": _as_string_list(
                data.get("ending_button_blueprint") or data.get("ending_button")
            ),
            "seedance_emphasis_points": _as_string_list(
                data.get("seedance_emphasis_points") or data.get("seedance_emphasis")
            ),
            "avoid_patterns": _as_string_list(data.get("avoid_patterns") or data.get("seedance_avoid")),
        }

    def _normalize_downstream_design_guidance(self, value: Any) -> dict[str, Any]:
        data = dict(value) if isinstance(value, Mapping) else {}
        return {
            "script_reconstruction_focus": _as_string_list(
                data.get("script_reconstruction_focus") or data.get("script_reconstruction_priority")
            ),
            "character_design_focus": _as_string_list(
                data.get("character_design_focus") or data.get("character_design_priority")
            ),
            "scene_design_focus": _as_string_list(
                data.get("scene_design_focus") or data.get("scene_design_priority")
            ),
            "storyboard_focus": _as_string_list(
                data.get("storyboard_focus") or data.get("storyboard_design_priority")
            ),
            "adaptation_priorities": _as_string_list(
                data.get("adaptation_priorities") or data.get("continuity_critical_points")
            ),
        }

    def _normalize_quality_assessment(
        self,
        value: Any,
        genre_classification: Any,
    ) -> dict[str, Any]:
        data = dict(value) if isinstance(value, Mapping) else {}
        genre_data = dict(genre_classification) if isinstance(genre_classification, Mapping) else {}
        return {
            "overall_confidence": max(
                0.0,
                min(1.0, _as_float(data.get("overall_confidence"), _as_float(genre_data.get("confidence"), 0.0))),
            ),
            "needs_human_review": bool(data.get("needs_human_review", False)),
            "known_gaps": _as_string_list(data.get("known_gaps")),
            "risky_facts": _as_string_list(data.get("risky_facts")),
        }

    def _build_script_prompt(
        self,
        bundle: EpisodeInputBundle,
        analysis: Mapping[str, Any],
    ) -> str:
        routing = resolve_pre_analysis_genre_routing(bundle)
        compact_analysis = self._compact_analysis_for_script(analysis)
        return render_prompt(
            "video_pipeline/openai_script_user.md",
            {
                "episode_id": bundle.episode_id,
                "title_hint": bundle.title or "",
                "analysis_json": json.dumps(compact_analysis, ensure_ascii=False, indent=2),
                "skill_text": truncate_text(routing.combined_skill_text(), 10000),
                "genre_routing_note": truncate_text(routing.routing_note_text(), 2000),
                "playbook_reference_block": truncate_text(routing.playbook_reference_text(), 7000),
            },
        )

    def _compact_analysis_for_script(self, analysis: Mapping[str, Any]) -> dict[str, Any]:
        compact_beats: list[dict[str, Any]] = []
        for beat in list(analysis.get("story_beats", []))[:10]:
            if not isinstance(beat, Mapping):
                continue
            compact_beats.append(
                {
                    "beat_id": beat.get("beat_id", ""),
                    "time_range": beat.get("time_range", ""),
                    "summary": beat.get("summary", ""),
                    "emotional_turn": beat.get("emotional_turn", ""),
                    "key_actions": list(beat.get("key_actions", []))[:3],
                    "dialogue_highlights": list(beat.get("dialogue_highlights", []))[:3],
                    "visual_focus": list(beat.get("visual_focus", []))[:3],
                    "camera_language": list(beat.get("camera_language", []))[:3],
                    "art_direction_cues": list(beat.get("art_direction_cues", []))[:3],
                    "storyboard_value": list(beat.get("storyboard_value", []))[:3],
                }
            )

        compact_characters = []
        for item in list(analysis.get("characters", []))[:12]:
            if not isinstance(item, Mapping):
                continue
            compact_characters.append(
                {
                    "name": item.get("name", ""),
                    "role": item.get("role", ""),
                    "relationship_to_protagonist": item.get("relationship_to_protagonist", ""),
                    "visual_profile": item.get("visual_profile", ""),
                    "current_state": item.get("current_state", ""),
                }
            )

        compact_locations = []
        for item in list(analysis.get("locations", []))[:8]:
            if not isinstance(item, Mapping):
                continue
            compact_locations.append(
                {
                    "name": item.get("name", ""),
                    "time_of_day": item.get("time_of_day", ""),
                    "visual_profile": item.get("visual_profile", ""),
                    "props": list(item.get("props", []))[:4],
                }
            )

        return {
            "episode": dict(analysis.get("episode", {})),
            "synopsis": analysis.get("synopsis", ""),
            "genre_classification": dict(analysis.get("genre_classification", {})),
            "hook_profile": dict(analysis.get("hook_profile", {})),
            "camera_language_analysis": dict(analysis.get("camera_language_analysis", {})),
            "art_direction_analysis": dict(analysis.get("art_direction_analysis", {})),
            "storyboard_blueprint": dict(analysis.get("storyboard_blueprint", {})),
            "downstream_design_guidance": dict(analysis.get("downstream_design_guidance", {})),
            "characters": compact_characters,
            "locations": compact_locations,
            "story_beats": compact_beats,
            "continuity_notes": list(analysis.get("continuity_notes", []))[:8],
            "adaptation_hints": list(analysis.get("adaptation_hints", []))[:8],
        }

    def _extract_output_text(self, response: Mapping[str, Any]) -> str:
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = dict(choices[0].get("message", {}))
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                texts = [
                    str(item.get("text", "")).strip()
                    for item in content
                    if isinstance(item, Mapping) and str(item.get("text", "")).strip()
                ]
                if texts:
                    return "\n".join(texts).strip()

        output = dict(response.get("output", {}))
        native_choices = output.get("choices")
        if isinstance(native_choices, list) and native_choices:
            message = dict(native_choices[0].get("message", {}))
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                texts = [
                    str(item.get("text", "")).strip()
                    for item in content
                    if isinstance(item, Mapping) and str(item.get("text", "")).strip()
                ]
                if texts:
                    return "\n".join(texts).strip()

        raise ProviderResponseError(f"Qwen 响应中没有可提取的文本内容：{response}")

    def _hydrate_provider_metadata(
        self,
        bundle: EpisodeInputBundle,
        analysis: dict[str, Any],
    ) -> None:
        analysis.setdefault("schema_version", "1.0.0")

        provider = ensure_object_field(analysis, "provider")
        provider["name"] = "qwen"
        provider["model"] = self.config.model
        provider["generated_at"] = utc_timestamp()
        provider["run_id"] = provider.get("run_id")

        episode = ensure_object_field(analysis, "episode")
        episode.setdefault("episode_id", bundle.episode_id)
        episode.setdefault("title", bundle.title or bundle.episode_id)
        episode.setdefault("language", bundle.language)
        episode.setdefault("source_video", bundle.video_path or "")
        episode.setdefault("estimated_duration_seconds", None)

        analysis.setdefault(
            "series_learning_extraction",
            {
                "episode_strengths": [],
                "why_it_works": [],
                "character_design_rules": [],
                "costume_makeup_rules": [],
                "scene_design_rules": [],
                "camera_language_rules": [],
                "storyboard_execution_rules": [],
            },
        )
        ensure_object_field(
            analysis,
            "genre_override_request",
            {
                "needs_user_confirmation": False,
                "proposed_primary_genre": "",
                "proposed_secondary_genres": [],
                "proposed_new_genres": [],
                "reason": "",
            },
        )
        genre_profile = ensure_object_field(analysis, "genre_classification")
        genre_profile.setdefault("confirmed_user_genres", [])
        genre_profile.setdefault("genre_resolution_mode", "freeform")
