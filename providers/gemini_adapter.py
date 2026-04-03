from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

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
    ensure_object_field,
    extract_json_from_text,
    file_to_base64,
    guess_mime_type,
    truncate_text,
    utc_timestamp,
    validate_against_schema,
)

ANALYSIS_SYSTEM_PROMPT = load_prompt("video_pipeline/gemini_analysis_system.md")
SCRIPT_SYSTEM_PROMPT = load_prompt("video_pipeline/gemini_script_system.md")


class GeminiAdapter(EpisodeAnalysisProvider):
    capabilities = ProviderCapabilities(
        supports_structured_output=True,
        supports_image_inputs=True,
        supports_video_inputs=True,
    )

    def __init__(
        self,
        model: str = "gemini-3-pro-preview",
        *,
        api_key_env: str = "GEMINI_API_KEY",
        endpoint: str | None = None,
        temperature: float = 0.2,
        timeout_seconds: int = 300,
        max_retries: int = 1,
        poll_interval_seconds: int = 5,
        max_poll_seconds: int = 300,
        telemetry: TelemetryRecorder | None = None,
    ) -> None:
        super().__init__(
            ProviderConfig(
                name="gemini",
                model=model,
                api_key_env=api_key_env,
                endpoint=endpoint or f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                extra_config={
                    "poll_interval_seconds": poll_interval_seconds,
                    "max_poll_seconds": max_poll_seconds,
                },
            )
        )
        self.telemetry = telemetry

    def analyze_episode(
        self,
        bundle: EpisodeInputBundle,
        schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        bundle.validate()
        with telemetry_span(
            self.telemetry,
            stage="video_analysis",
            name="build_gemini_episode_analysis_prompt",
            provider="gemini",
            model=self.config.model,
            metadata={
                "episode_id": bundle.episode_id,
                "frame_count": len(bundle.frames),
                "transcript_chars": len(bundle.transcript_text or ""),
                "ocr_chars": len(bundle.ocr_text or ""),
                "has_video": bool(bundle.video_path),
            },
        ):
            parts = [{"text": self._build_analysis_prompt(bundle)}]

        if bundle.video_path:
            with telemetry_span(
                self.telemetry,
                stage="video_analysis",
                name="gemini_upload_video_file",
                provider="gemini",
                model=self.config.model,
                metadata={"episode_id": bundle.episode_id, "video_path": str(bundle.resolved_video_path())},
            ) as step:
                file_info = self._upload_file(bundle.resolved_video_path())
                step["metadata"]["uploaded_file_name"] = file_info.get("file", {}).get("name", "")
            with telemetry_span(
                self.telemetry,
                stage="video_analysis",
                name="gemini_wait_video_active",
                provider="gemini",
                model=self.config.model,
                metadata={"episode_id": bundle.episode_id},
            ) as step:
                file_uri = self._wait_until_active(file_info)
                step["metadata"]["file_uri"] = file_uri
            parts.append(
                {
                    "file_data": {
                        "mime_type": guess_mime_type(bundle.resolved_video_path(), fallback="video/mp4"),
                        "file_uri": file_uri,
                    }
                }
            )

        if bundle.transcript_text:
            parts.append({"text": f"Transcript:\n{truncate_text(bundle.transcript_text, 40000)}"})
        if bundle.ocr_text:
            parts.append({"text": f"OCR:\n{truncate_text(bundle.ocr_text, 12000)}"})

        for index, frame in enumerate(bundle.frames, start=1):
            descriptor_parts = [part for part in [frame.timestamp, frame.note] if part]
            descriptor = f"关键帧 {index}"
            if descriptor_parts:
                descriptor += f"：{'; '.join(descriptor_parts)}"
            parts.append({"text": descriptor})
            parts.append(
                {
                    "inline_data": {
                        "mime_type": frame.detected_mime_type(),
                        "data": file_to_base64(frame.resolved_path()),
                    }
                }
            )

        with telemetry_span(
            self.telemetry,
            stage="video_analysis",
            name="build_gemini_episode_analysis_request",
            provider="gemini",
            model=self.config.model,
            metadata={"episode_id": bundle.episode_id, "part_count": len(parts)},
        ):
            payload = {
                "systemInstruction": {"parts": [{"text": ANALYSIS_SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": self.config.temperature,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": schema,
                },
            }
        with telemetry_span(
            self.telemetry,
            stage="video_analysis",
            name="gemini_episode_analysis_model_call",
            provider="gemini",
            model=self.config.model,
            metadata={"episode_id": bundle.episode_id},
        ) as step:
            response = self.request_json(
                self.config.endpoint,
                headers={"x-goog-api-key": self.require_api_key()},
                payload=payload,
            )
            apply_provider_usage(step, "gemini", response)
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
            name="build_gemini_script_reconstruction_request",
            provider="gemini",
            model=self.config.model,
            metadata={"episode_id": bundle.episode_id},
        ):
            payload = {
                "systemInstruction": {"parts": [{"text": SCRIPT_SYSTEM_PROMPT}]},
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": self._build_script_prompt(bundle, analysis),
                            }
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": max(self.config.temperature, 0.4),
                },
            }
        with telemetry_span(
            self.telemetry,
            stage="script_reconstruction",
            name="gemini_script_reconstruction_model_call",
            provider="gemini",
            model=self.config.model,
            metadata={"episode_id": bundle.episode_id},
        ) as step:
            response = self.request_json(
                self.config.endpoint,
                headers={"x-goog-api-key": self.require_api_key()},
                payload=payload,
            )
            apply_provider_usage(step, "gemini", response)
        return self._extract_output_text(response).strip()

    def _build_analysis_prompt(self, bundle: EpisodeInputBundle) -> str:
        routing = resolve_pre_analysis_genre_routing(bundle)
        return render_prompt(
            "video_pipeline/gemini_analysis_user.md",
            {
                "input_summary": bundle.as_prompt_summary(),
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
            "video_pipeline/gemini_script_user.md",
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

    def _upload_file(self, file_path: Path | None) -> dict[str, Any]:
        if not file_path:
            raise ProviderResponseError("缺少待上传的视频文件路径。")

        api_key = self.require_api_key()
        mime_type = guess_mime_type(file_path, fallback="application/octet-stream")
        file_bytes = file_path.read_bytes()
        start_headers = {
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(file_bytes)),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }
        _, response_headers, _ = self.request(
            "https://generativelanguage.googleapis.com/upload/v1beta/files",
            headers=start_headers,
            data=json.dumps({"file": {"display_name": file_path.name}}).encode("utf-8"),
        )
        upload_url = response_headers.get("x-goog-upload-url")
        if not upload_url:
            raise ProviderResponseError("Gemini 文件上传初始化成功，但响应里没有 x-goog-upload-url。")

        _, _, raw_body = self.request(
            upload_url,
            headers={
                "Content-Length": str(len(file_bytes)),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            data=file_bytes,
        )
        file_info = json.loads(raw_body.decode("utf-8"))
        if "file" not in file_info:
            raise ProviderResponseError(f"Gemini 文件上传返回缺少 file 字段：{file_info}")
        return file_info

    def _wait_until_active(self, file_info: Mapping[str, Any]) -> str:
        api_key = self.require_api_key()
        file_record = dict(file_info.get("file", {}))
        file_name = file_record.get("name")
        file_uri = file_record.get("uri")
        if not file_name or not file_uri:
            raise ProviderResponseError(f"Gemini 文件上传结果缺少 name 或 uri：{file_info}")

        started_at = time.time()
        poll_interval = int(self.config.extra_config.get("poll_interval_seconds", 5))
        max_poll_seconds = int(self.config.extra_config.get("max_poll_seconds", 300))

        while True:
            state = file_record.get("state")
            state_name = state.get("name") if isinstance(state, dict) else state
            if state_name in {"ACTIVE", None, ""}:
                return str(file_uri)
            if state_name not in {"PROCESSING"}:
                raise ProviderResponseError(f"Gemini 文件状态异常：{state_name!r}")
            if time.time() - started_at > max_poll_seconds:
                raise ProviderResponseError("等待 Gemini 处理视频超时。")

            time.sleep(poll_interval)
            get_url = f"https://generativelanguage.googleapis.com/v1beta/{quote(str(file_name), safe='/')}?key={api_key}"
            status = self.request_json(
                get_url,
                method="GET",
                headers={"x-goog-api-key": api_key},
                payload=None,
            )
            file_record = status.get("file", status)
            file_uri = file_record.get("uri", file_uri)

    def _extract_output_text(self, response: Mapping[str, Any]) -> str:
        texts: list[str] = []
        for candidate in response.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text")
                if text:
                    texts.append(text)
        if texts:
            return "\n".join(texts).strip()
        raise ProviderResponseError(f"Gemini 响应中没有可提取文本：{response}")

    def _hydrate_provider_metadata(
        self,
        bundle: EpisodeInputBundle,
        analysis: dict[str, Any],
    ) -> None:
        analysis.setdefault("schema_version", "1.0.0")

        provider = ensure_object_field(analysis, "provider")
        provider["name"] = "gemini"
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
