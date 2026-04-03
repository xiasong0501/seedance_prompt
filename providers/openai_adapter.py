from __future__ import annotations

import json
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
    UnsupportedInputError,
    ensure_object_field,
    extract_json_from_text,
    file_to_data_url,
    truncate_text,
    utc_timestamp,
    validate_against_schema,
)

ANALYSIS_SYSTEM_PROMPT = load_prompt("video_pipeline/openai_analysis_system.md")
SCRIPT_SYSTEM_PROMPT = load_prompt("video_pipeline/openai_script_system.md")


class OpenAIAdapter(EpisodeAnalysisProvider):
    capabilities = ProviderCapabilities(
        supports_structured_output=True,
        supports_image_inputs=True,
        supports_video_inputs=False,
    )

    def __init__(
        self,
        model: str = "gpt-5",
        *,
        api_key_env: str = "OPENAI_API_KEY",
        endpoint: str = "https://api.openai.com/v1/responses",
        temperature: float = 0.2,
        timeout_seconds: int = 180,
        max_retries: int = 1,
        image_detail: str = "auto",
        max_analysis_frames: int | None = 20,
        telemetry: TelemetryRecorder | None = None,
    ) -> None:
        super().__init__(
            ProviderConfig(
                name="openai",
                model=model,
                api_key_env=api_key_env,
                endpoint=endpoint,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        )
        self.telemetry = telemetry
        normalized_detail = str(image_detail or "auto").strip().lower()
        self.image_detail = normalized_detail if normalized_detail in {"low", "high", "auto"} else "auto"
        self.max_analysis_frames = (
            None if max_analysis_frames is None or int(max_analysis_frames) <= 0 else int(max_analysis_frames)
        )

    def analyze_episode(
        self,
        bundle: EpisodeInputBundle,
        schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        bundle.validate()
        if bundle.video_path and not bundle.frames and not bundle.transcript_text and not bundle.ocr_text:
            raise UnsupportedInputError(
                "OpenAI adapter 当前不直接吃原始视频。请先提供 transcript、OCR 或关键帧后再分析。"
            )

        with telemetry_span(
            self.telemetry,
            stage="video_analysis",
            name="build_openai_episode_analysis_request",
            provider="openai",
            model=self.config.model,
            metadata={
                "episode_id": bundle.episode_id,
                "frame_count": len(bundle.frames),
                "transcript_chars": len(bundle.transcript_text or ""),
                "ocr_chars": len(bundle.ocr_text or ""),
            },
        ) as step:
            analysis_content = self._build_analysis_content(bundle)
            payload = {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": ANALYSIS_SYSTEM_PROMPT}],
                    },
                    {"role": "user", "content": analysis_content},
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "episode_analysis",
                        "schema": schema,
                        "strict": True,
                    }
                },
            }
            step["metadata"]["content_parts"] = len(analysis_content)
        with telemetry_span(
            self.telemetry,
            stage="video_analysis",
            name="openai_episode_analysis_model_call",
            provider="openai",
            model=self.config.model,
            metadata={"episode_id": bundle.episode_id},
        ) as step:
            response = self.request_json(
                self.config.endpoint,
                headers={"Authorization": f"Bearer {self.require_api_key()}"},
                payload=payload,
            )
            apply_provider_usage(step, "openai", response)

        output_text = self._extract_output_text(response)
        analysis = extract_json_from_text(output_text)
        self._hydrate_provider_metadata(bundle, analysis)
        enforce_user_genre_alignment(bundle, analysis)
        validate_against_schema(analysis, schema)
        return analysis

    def reconstruct_script(
        self,
        bundle: EpisodeInputBundle,
        analysis: Mapping[str, Any],
    ) -> str:
        with telemetry_span(
            self.telemetry,
            stage="script_reconstruction",
            name="build_openai_script_reconstruction_request",
            provider="openai",
            model=self.config.model,
            metadata={"episode_id": bundle.episode_id},
        ):
            payload = {
                "model": self.config.model,
                "temperature": max(self.config.temperature, 0.4),
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": SCRIPT_SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": self._build_script_prompt(bundle, analysis)}],
                    },
                ],
                "text": {"format": {"type": "text"}},
            }
        with telemetry_span(
            self.telemetry,
            stage="script_reconstruction",
            name="openai_script_reconstruction_model_call",
            provider="openai",
            model=self.config.model,
            metadata={"episode_id": bundle.episode_id},
        ) as step:
            response = self.request_json(
                self.config.endpoint,
                headers={"Authorization": f"Bearer {self.require_api_key()}"},
                payload=payload,
            )
            apply_provider_usage(step, "openai", response)
        return self._extract_output_text(response).strip()

    def _build_analysis_content(self, bundle: EpisodeInputBundle) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": self._build_analysis_prompt(bundle),
            }
        ]

        selected_frames = self._select_analysis_frames(bundle.frames)
        for index, frame in enumerate(selected_frames, start=1):
            descriptor_parts = [part for part in [frame.timestamp, frame.note] if part]
            descriptor = f"关键帧 {index}"
            if descriptor_parts:
                descriptor += f"：{'; '.join(descriptor_parts)}"
            content.append({"type": "input_text", "text": descriptor})
            content.append(
                {
                    "type": "input_image",
                    "image_url": file_to_data_url(frame.resolved_path(), frame.detected_mime_type()),
                    "detail": self.image_detail,
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
        texts: list[str] = []
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    texts.append(content["text"])
        if texts:
            return "\n".join(texts).strip()
        raise ProviderResponseError(f"OpenAI 响应中没有可提取的 output_text：{response}")

    def _hydrate_provider_metadata(
        self,
        bundle: EpisodeInputBundle,
        analysis: dict[str, Any],
    ) -> None:
        analysis.setdefault("schema_version", "1.0.0")

        provider = ensure_object_field(analysis, "provider")
        provider["name"] = "openai"
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

        analysis.setdefault(
            "camera_language_analysis",
            {
                "dominant_shot_types": [],
                "camera_motion_patterns": [],
                "composition_patterns": [],
                "visual_emphasis_rules": [],
                "transition_rhythm": "",
                "climax_visual_strategy": "",
                "cliffhanger_visual_pattern": "",
            },
        )
        analysis.setdefault(
            "art_direction_analysis",
            {
                "costume_signatures": [],
                "hair_makeup_signatures": [],
                "prop_signatures": [],
                "set_signatures": [],
                "lighting_signatures": [],
                "color_mood_patterns": [],
                "texture_material_patterns": [],
            },
        )
        analysis.setdefault(
            "storyboard_blueprint",
            {
                "opening_hook_blueprint": [],
                "conflict_escalation_blueprint": [],
                "emotional_payoff_blueprint": [],
                "ending_button_blueprint": [],
                "seedance_emphasis_points": [],
                "avoid_patterns": [],
            },
        )

        for beat in analysis.get("story_beats", []):
            if not isinstance(beat, dict):
                continue
            beat.setdefault("visual_focus", [])
            beat.setdefault("camera_language", [])
            beat.setdefault("art_direction_cues", [])
            beat.setdefault("storyboard_value", [])
