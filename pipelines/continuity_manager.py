from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genre_routing import load_genre_playbook_library
from providers.base import EpisodeInputBundle, save_json_file, save_text_file, utc_timestamp


def _normalize_identity(raw: str) -> str:
    clean = re.sub(r"\s+", "", (raw or "").strip()).lower()
    clean = re.sub(r"[^\w\u4e00-\u9fff]+", "", clean)
    return clean


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        text = (item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _dedupe_state_history(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for item in values:
        episode_id = str(item.get("episode_id", "")).strip()
        state = str(item.get("state", "")).strip()
        key = (episode_id, state)
        if not episode_id or key in seen:
            continue
        seen.add(key)
        result.append({"episode_id": episode_id, "state": state})
    return result


def _episode_sort_key(episode_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", episode_id or "")
    if match:
        return int(match.group(1)), episode_id
    return 10**9, episode_id


@dataclass
class ContinuityContext:
    enriched_bundle: EpisodeInputBundle
    series_bible_path: Path
    character_registry_path: Path
    location_registry_path: Path
    plot_timeline_path: Path
    series_context_path: Path
    previous_episode_summary_path: Path | None


@dataclass
class ContinuityUpdateResult:
    series_bible_path: Path
    character_registry_path: Path
    location_registry_path: Path
    plot_timeline_path: Path
    series_context_path: Path
    episode_summary_path: Path
    series_strength_playbook_json_path: Path
    series_strength_playbook_markdown_path: Path


class SeriesContinuityManager:
    def __init__(
        self,
        *,
        analysis_root: Path,
        series_folder: str,
        provider: str,
        model: str,
    ) -> None:
        self.analysis_root = Path(analysis_root)
        self.series_folder = series_folder
        self.provider = provider
        self.model = model

        self.series_dir = self.analysis_root / self.series_folder
        self.episode_summaries_dir = self.series_dir / "episode_summaries"
        self.series_bible_path = self.series_dir / "series_bible.json"
        self.character_registry_path = self.series_dir / "character_registry.json"
        self.location_registry_path = self.series_dir / "location_registry.json"
        self.plot_timeline_path = self.series_dir / "plot_timeline.json"
        self.series_context_path = self.series_dir / "series_context.json"
        self.series_strength_playbook_json_path = self.series_dir / "series_strength_playbook_draft.json"
        self.series_strength_playbook_markdown_path = self.series_dir / "series_strength_playbook_draft.md"

    def build_context(self, bundle: EpisodeInputBundle) -> ContinuityContext:
        series_bible = self._load_or_default(
            self.series_bible_path,
            {
                "series_name": self.series_folder,
                "updated_at": None,
                "episode_count": 0,
                "latest_episode_id": None,
                "premise": "",
                "major_arcs": [],
                "unresolved_threads": [],
                "active_characters": [],
                "recurring_locations": [],
                "genre_profile": {},
                "genre_playbooks": [],
                "camera_language_profile": self._default_camera_language_profile(),
                "art_direction_profile": self._default_art_direction_profile(),
                "storyboard_profile": self._default_storyboard_profile(),
                "character_design_profile": self._default_character_design_profile(),
                "scene_design_profile": self._default_scene_design_profile(),
                "dialogue_timing_profile": self._default_dialogue_timing_profile(),
                "production_guardrails": self._default_production_guardrails(),
                "downstream_design_guidance": {},
                "series_learning_profile": self._default_series_learning_profile(),
                "continuity_rules": self._default_continuity_rules(),
                "source_episodes": [],
            },
        )
        character_registry = self._load_or_default(
            self.character_registry_path,
            {
                "series_name": self.series_folder,
                "updated_at": None,
                "characters": [],
            },
        )
        location_registry = self._load_or_default(
            self.location_registry_path,
            {
                "series_name": self.series_folder,
                "updated_at": None,
                "locations": [],
            },
        )
        plot_timeline = self._load_or_default(
            self.plot_timeline_path,
            {
                "series_name": self.series_folder,
                "updated_at": None,
                "episodes": [],
            },
        )
        previous_summary = self._find_previous_episode_summary(bundle.episode_id)

        notes = list(bundle.context_notes)
        notes.extend(
            self._build_context_notes(
                previous_summary=previous_summary,
                series_bible=series_bible,
                character_registry=character_registry,
                location_registry=location_registry,
                plot_timeline=plot_timeline,
            )
        )

        metadata = dict(bundle.metadata)
        metadata["continuity_context"] = {
            "series_folder": self.series_folder,
            "previous_episode_id": (previous_summary or {}).get("episode_id"),
            "registered_character_count": len(character_registry.get("characters", [])),
            "registered_location_count": len(location_registry.get("locations", [])),
            "timeline_episode_count": len(plot_timeline.get("episodes", [])),
            "genre_profile": dict(series_bible.get("genre_profile", {})),
            "genre_playbooks": list(series_bible.get("genre_playbooks", [])),
            "camera_language_profile": dict(series_bible.get("camera_language_profile", {})),
            "art_direction_profile": dict(series_bible.get("art_direction_profile", {})),
            "storyboard_profile": dict(series_bible.get("storyboard_profile", {})),
            "character_design_profile": dict(series_bible.get("character_design_profile", {})),
            "scene_design_profile": dict(series_bible.get("scene_design_profile", {})),
            "dialogue_timing_profile": dict(series_bible.get("dialogue_timing_profile", {})),
            "production_guardrails": dict(series_bible.get("production_guardrails", {})),
        }

        enriched_bundle = EpisodeInputBundle(
            episode_id=bundle.episode_id,
            title=bundle.title,
            video_path=bundle.video_path,
            transcript_text=bundle.transcript_text,
            ocr_text=bundle.ocr_text,
            synopsis_text=bundle.synopsis_text,
            frames=bundle.frames,
            context_notes=notes,
            language=bundle.language,
            metadata=metadata,
        )
        return ContinuityContext(
            enriched_bundle=enriched_bundle,
            series_bible_path=self.series_bible_path,
            character_registry_path=self.character_registry_path,
            location_registry_path=self.location_registry_path,
            plot_timeline_path=self.plot_timeline_path,
            series_context_path=self.series_context_path,
            previous_episode_summary_path=(
                self._episode_summary_path_for(previous_summary["episode_id"])
                if previous_summary
                else None
            ),
        )

    def update_from_episode(
        self,
        *,
        bundle: EpisodeInputBundle,
        analysis: Mapping[str, Any],
        analysis_path: Path,
        script_path: Path,
    ) -> ContinuityUpdateResult:
        episode_summary = self._build_episode_summary(bundle, analysis, analysis_path, script_path)

        series_bible = self._load_or_default(
            self.series_bible_path,
            {
                "series_name": self.series_folder,
                "updated_at": None,
                "episode_count": 0,
                "latest_episode_id": None,
                "premise": "",
                "major_arcs": [],
                "unresolved_threads": [],
                "active_characters": [],
                "recurring_locations": [],
                "genre_profile": {},
                "genre_playbooks": [],
                "camera_language_profile": self._default_camera_language_profile(),
                "art_direction_profile": self._default_art_direction_profile(),
                "storyboard_profile": self._default_storyboard_profile(),
                "character_design_profile": self._default_character_design_profile(),
                "scene_design_profile": self._default_scene_design_profile(),
                "dialogue_timing_profile": self._default_dialogue_timing_profile(),
                "production_guardrails": self._default_production_guardrails(),
                "downstream_design_guidance": {},
                "series_learning_profile": self._default_series_learning_profile(),
                "continuity_rules": self._default_continuity_rules(),
                "source_episodes": [],
            },
        )
        character_registry = self._load_or_default(
            self.character_registry_path,
            {
                "series_name": self.series_folder,
                "updated_at": None,
                "characters": [],
            },
        )
        location_registry = self._load_or_default(
            self.location_registry_path,
            {
                "series_name": self.series_folder,
                "updated_at": None,
                "locations": [],
            },
        )
        plot_timeline = self._load_or_default(
            self.plot_timeline_path,
            {
                "series_name": self.series_folder,
                "updated_at": None,
                "episodes": [],
            },
        )

        self._merge_characters(character_registry, analysis, bundle.episode_id)
        self._merge_locations(location_registry, analysis, bundle.episode_id)
        self._merge_timeline(plot_timeline, episode_summary)
        self._merge_series_bible(series_bible, analysis, episode_summary, character_registry, location_registry, plot_timeline)

        series_context = self._build_series_context(series_bible, character_registry, location_registry, plot_timeline)
        series_strength_playbook = self._build_series_strength_playbook(series_bible)

        episode_summary_path = self._episode_summary_path_for(bundle.episode_id)
        save_json_file(episode_summary_path, episode_summary)
        save_json_file(self.series_bible_path, series_bible)
        save_json_file(self.character_registry_path, character_registry)
        save_json_file(self.location_registry_path, location_registry)
        save_json_file(self.plot_timeline_path, plot_timeline)
        save_json_file(self.series_context_path, series_context)
        save_json_file(self.series_strength_playbook_json_path, series_strength_playbook)
        save_text_file(
            self.series_strength_playbook_markdown_path,
            self._render_series_strength_playbook_markdown(series_strength_playbook),
        )

        return ContinuityUpdateResult(
            series_bible_path=self.series_bible_path.resolve(),
            character_registry_path=self.character_registry_path.resolve(),
            location_registry_path=self.location_registry_path.resolve(),
            plot_timeline_path=self.plot_timeline_path.resolve(),
            series_context_path=self.series_context_path.resolve(),
            episode_summary_path=episode_summary_path.resolve(),
            series_strength_playbook_json_path=self.series_strength_playbook_json_path.resolve(),
            series_strength_playbook_markdown_path=self.series_strength_playbook_markdown_path.resolve(),
        )

    def _build_context_notes(
        self,
        *,
        previous_summary: Mapping[str, Any] | None,
        series_bible: Mapping[str, Any],
        character_registry: Mapping[str, Any],
        location_registry: Mapping[str, Any],
        plot_timeline: Mapping[str, Any],
    ) -> list[str]:
        notes = [
            "连续性参考只用于实体对齐和承接判断；若与当前视频直接证据冲突，必须以当前视频为准，并把冲突写入 continuity_notes。",
        ]
        plot_lines: list[str] = []
        if previous_summary:
            previous_synopsis = str(previous_summary.get("synopsis", "")).strip()
            continuity_hooks = previous_summary.get("continuity_hooks", [])[:5]
            lines = [f"上一集摘要（{previous_summary.get('episode_id', '')}）：{previous_synopsis}"]
            if continuity_hooks:
                lines.append("上一集留到下一集的承接点：" + "；".join(continuity_hooks))
            plot_lines.extend(lines)

        active_characters = character_registry.get("characters", [])[:12]
        world_lines: list[str] = []
        if active_characters:
            char_lines = []
            for item in active_characters[:8]:
                char_lines.append(
                    f"{item.get('canonical_name', '')}｜身份:{item.get('role', '')}｜关系:{item.get('relationship_to_protagonist', '')}｜最近状态:{item.get('latest_state', '')}"
                )
            world_lines.append("当前人物状态卡：")
            world_lines.extend(char_lines)

        active_locations = location_registry.get("locations", [])[:8]
        if active_locations:
            location_lines = []
            for item in active_locations[:6]:
                location_lines.append(
                    f"{item.get('canonical_name', '')}｜时段:{item.get('time_of_day', '')}｜特征:{item.get('visual_profile', '')}"
                )
            world_lines.append("当前场景库：")
            world_lines.extend(location_lines)

        unresolved_threads = _unique_strings(
            list(series_bible.get("unresolved_threads", [])) + self._latest_timeline_threads(plot_timeline)
        )[:8]
        if unresolved_threads:
            plot_lines.append("当前未解决剧情线：" + "；".join(unresolved_threads[:6]))

        genre_profile = series_bible.get("genre_profile", {})
        genre_lines: list[str] = []
        if genre_profile:
            genre_lines.append(
                "当前题材判断："
                f"{genre_profile.get('primary_genre', '')}"
                f"｜副题材:{'、'.join(genre_profile.get('secondary_genres', []))}"
                f"｜叙事装置:{genre_profile.get('narrative_device', '')}"
                f"｜观众期待:{genre_profile.get('audience_expectation', '')}"
            )

        genre_playbooks = list(series_bible.get("genre_playbooks", []))[:4]
        if genre_playbooks:
            playbook_lines = []
            for item in genre_playbooks:
                playbook_lines.append(
                    f"{item.get('genre_key', '')}｜剧本抓点:{'；'.join(item.get('script_hooks', [])[:2])}"
                )
            genre_lines.append("当前题材经验：")
            genre_lines.extend(playbook_lines[:3])

        downstream_guidance = series_bible.get("downstream_design_guidance", {})
        if downstream_guidance:
            guidance_lines: list[str] = []
            if downstream_guidance.get("script_reconstruction_focus"):
                guidance_lines.append(
                    "剧本重建重点：" + "；".join(downstream_guidance.get("script_reconstruction_focus", [])[:4])
                )
            if downstream_guidance.get("character_design_focus"):
                guidance_lines.append(
                    "人物设计重点：" + "；".join(downstream_guidance.get("character_design_focus", [])[:4])
                )
            if downstream_guidance.get("scene_design_focus"):
                guidance_lines.append(
                    "场景设计重点：" + "；".join(downstream_guidance.get("scene_design_focus", [])[:4])
                )
            if downstream_guidance.get("storyboard_focus"):
                guidance_lines.append(
                    "分镜重点：" + "；".join(downstream_guidance.get("storyboard_focus", [])[:4])
                )
            if guidance_lines:
                genre_lines.append("当前下游设计指导：")
                genre_lines.extend(guidance_lines[:4])

        camera_profile = series_bible.get("camera_language_profile", {})
        visual_lines: list[str] = []
        if camera_profile:
            camera_lines: list[str] = []
            if camera_profile.get("dominant_shot_types"):
                camera_lines.append(
                    "镜头主类型：" + "；".join(camera_profile.get("dominant_shot_types", [])[:4])
                )
            if camera_profile.get("camera_motion_patterns"):
                camera_lines.append(
                    "运镜模式：" + "；".join(camera_profile.get("camera_motion_patterns", [])[:4])
                )
            if camera_profile.get("cliffhanger_visual_pattern"):
                camera_lines.append(
                    "集尾视觉按钮：" + str(camera_profile.get("cliffhanger_visual_pattern", "")).strip()
                )
            if camera_lines:
                visual_lines.append("当前镜头语言画像：")
                visual_lines.extend(camera_lines)

        art_profile = series_bible.get("art_direction_profile", {})
        if art_profile:
            art_lines: list[str] = []
            if art_profile.get("costume_signatures"):
                art_lines.append(
                    "服装识别点：" + "；".join(art_profile.get("costume_signatures", [])[:4])
                )
            if art_profile.get("set_signatures"):
                art_lines.append(
                    "空间识别点：" + "；".join(art_profile.get("set_signatures", [])[:4])
                )
            if art_profile.get("lighting_signatures"):
                art_lines.append(
                    "灯光气质：" + "；".join(art_profile.get("lighting_signatures", [])[:4])
                )
            if art_lines:
                visual_lines.append("当前服化道画像：")
                visual_lines.extend(art_lines)

        storyboard_profile = series_bible.get("storyboard_profile", {})
        if storyboard_profile:
            storyboard_lines: list[str] = []
            if storyboard_profile.get("opening_hook_blueprint"):
                storyboard_lines.append(
                    "开头钩子打法：" + "；".join(storyboard_profile.get("opening_hook_blueprint", [])[:4])
                )
            if storyboard_profile.get("ending_button_blueprint"):
                storyboard_lines.append(
                    "结尾按钮打法：" + "；".join(storyboard_profile.get("ending_button_blueprint", [])[:4])
                )
            if storyboard_profile.get("seedance_emphasis_points"):
                storyboard_lines.append(
                    "Seedance 重点：" + "；".join(storyboard_profile.get("seedance_emphasis_points", [])[:4])
                )
            if storyboard_lines:
                visual_lines.append("当前分镜蓝图：")
                visual_lines.extend(storyboard_lines)

        character_design_profile = series_bible.get("character_design_profile", {})
        scene_design_profile = series_bible.get("scene_design_profile", {})
        dialogue_timing_profile = series_bible.get("dialogue_timing_profile", {})
        production_guardrails = series_bible.get("production_guardrails", {})
        production_lines: list[str] = []
        if character_design_profile:
            character_lines: list[str] = []
            if character_design_profile.get("character_design_rules"):
                character_lines.append(
                    "人物设计规则：" + "；".join(character_design_profile.get("character_design_rules", [])[:4])
                )
            if character_design_profile.get("costume_makeup_rules"):
                character_lines.append(
                    "服化道统一：" + "；".join(character_design_profile.get("costume_makeup_rules", [])[:4])
                )
            if character_lines:
                production_lines.append("当前人物与服化道生产规则：")
                production_lines.extend(character_lines)

        if scene_design_profile and scene_design_profile.get("scene_design_rules"):
            production_lines.append(
                "场景设计规则：" + "；".join(scene_design_profile.get("scene_design_rules", [])[:4])
            )

        if dialogue_timing_profile and dialogue_timing_profile.get("dialogue_timing_rules"):
            production_lines.append(
                "台词时间规则：" + "；".join(dialogue_timing_profile.get("dialogue_timing_rules", [])[:4])
            )

        if production_guardrails:
            if production_guardrails.get("continuity_guardrails"):
                production_lines.append(
                    "连续性红线：" + "；".join(production_guardrails.get("continuity_guardrails", [])[:4])
                )
            if production_guardrails.get("negative_patterns"):
                production_lines.append(
                    "高风险踩坑：" + "；".join(production_guardrails.get("negative_patterns", [])[:4])
                )

        learning_profile = series_bible.get("series_learning_profile", {})
        learning_lines: list[str] = []
        if learning_profile:
            if learning_profile.get("episode_strengths"):
                learning_lines.append(
                    "当前已识别优点：" + "；".join(learning_profile.get("episode_strengths", [])[:4])
                )
            if learning_profile.get("why_it_works"):
                learning_lines.append(
                    "有效原因：" + "；".join(learning_profile.get("why_it_works", [])[:4])
                )
            if learning_profile.get("reusable_playbook_rules"):
                learning_lines.append(
                    "已沉淀玩法经验：" + "；".join(learning_profile.get("reusable_playbook_rules", [])[:4])
                )
            if learning_profile.get("reusable_skill_rules"):
                learning_lines.append(
                    "已沉淀技能经验：" + "；".join(learning_profile.get("reusable_skill_rules", [])[:4])
                )
            if learning_profile.get("camera_language_rules"):
                learning_lines.append(
                    "镜头语言规则：" + "；".join(learning_profile.get("camera_language_rules", [])[:4])
                )
            if learning_profile.get("storyboard_execution_rules"):
                learning_lines.append(
                    "分镜执行规则：" + "；".join(learning_profile.get("storyboard_execution_rules", [])[:4])
                )

        grouped_sections = [
            plot_lines,
            world_lines,
            genre_lines,
            visual_lines,
            production_lines,
            learning_lines,
        ]
        for section in grouped_sections:
            if section:
                notes.append("\n".join(section))

        return notes[:6]

    def _latest_timeline_threads(self, plot_timeline: Mapping[str, Any]) -> list[str]:
        episodes = list(plot_timeline.get("episodes", []))
        episodes.sort(key=lambda item: _episode_sort_key(str(item.get("episode_id", ""))))
        latest = episodes[-2:]
        values: list[str] = []
        for episode in latest:
            values.extend(episode.get("unresolved_threads", []))
        return values

    def _build_episode_summary(
        self,
        bundle: EpisodeInputBundle,
        analysis: Mapping[str, Any],
        analysis_path: Path,
        script_path: Path,
    ) -> dict[str, Any]:
        synopsis = str(analysis.get("synopsis", "")).strip()
        key_events = _unique_strings(
            [
                *[str(beat.get("summary", "")).strip() for beat in analysis.get("story_beats", [])[:12]],
                *[str(action).strip() for beat in analysis.get("story_beats", [])[:6] for action in beat.get("key_actions", [])[:2]],
            ]
        )[:12]
        character_updates = _unique_strings(
            [
                f"{item.get('name', '')}：{item.get('current_state', '')}"
                for item in analysis.get("characters", [])
                if str(item.get("name", "")).strip()
            ]
        )[:12]
        location_updates = _unique_strings(
            [
                f"{item.get('name', '')}：{item.get('time_of_day', '')}｜{item.get('visual_profile', '')}"
                for item in analysis.get("locations", [])
                if str(item.get("name", "")).strip()
            ]
        )[:12]
        continuity_hooks = _unique_strings(
            list(analysis.get("continuity_notes", [])) + list(analysis.get("adaptation_hints", []))
        )[:12]
        unresolved_threads = _unique_strings(list(analysis.get("adaptation_hints", [])) + list(analysis.get("continuity_notes", [])))[:12]

        return {
            "episode_id": bundle.episode_id,
            "title": analysis.get("episode", {}).get("title") or bundle.title or bundle.episode_id,
            "provider": self.provider,
            "model": self.model,
            "generated_at": utc_timestamp(),
            "synopsis": synopsis,
            "genre_profile": analysis.get("genre_classification", {}),
            "hook_profile": analysis.get("hook_profile", {}),
            "downstream_design_guidance": analysis.get("downstream_design_guidance", {}),
            "camera_language_analysis": analysis.get("camera_language_analysis", {}),
            "art_direction_analysis": analysis.get("art_direction_analysis", {}),
            "storyboard_blueprint": analysis.get("storyboard_blueprint", {}),
            "series_learning_extraction": analysis.get("series_learning_extraction", {}),
            "key_events": key_events,
            "character_updates": character_updates,
            "location_updates": location_updates,
            "continuity_hooks": continuity_hooks,
            "unresolved_threads": unresolved_threads,
            "involved_characters": _unique_strings([str(item.get("name", "")).strip() for item in analysis.get("characters", [])])[:20],
            "analysis_path": str(analysis_path.resolve()),
            "script_path": str(script_path.resolve()),
        }

    def _merge_characters(
        self,
        registry: dict[str, Any],
        analysis: Mapping[str, Any],
        episode_id: str,
    ) -> None:
        characters = registry.setdefault("characters", [])
        registry["updated_at"] = utc_timestamp()

        for item in analysis.get("characters", []):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            aliases = _unique_strings([name, *item.get("aliases", [])])
            existing = self._find_character_entry(characters, aliases)
            if existing is None:
                existing = {
                    "canonical_id": item.get("character_id") or f"char-{len(characters) + 1:04d}",
                    "canonical_name": name,
                    "aliases": aliases,
                    "role": item.get("role", ""),
                    "relationship_to_protagonist": item.get("relationship_to_protagonist", ""),
                    "visual_profile": item.get("visual_profile", ""),
                    "latest_state": item.get("current_state", ""),
                    "first_episode": episode_id,
                    "latest_episode": episode_id,
                    "seen_episodes": [episode_id],
                    "state_history": [],
                }
                characters.append(existing)

            existing["aliases"] = _unique_strings(list(existing.get("aliases", [])) + aliases)
            existing["role"] = item.get("role", "") or existing.get("role", "")
            existing["relationship_to_protagonist"] = (
                item.get("relationship_to_protagonist", "") or existing.get("relationship_to_protagonist", "")
            )
            existing["visual_profile"] = item.get("visual_profile", "") or existing.get("visual_profile", "")
            existing["latest_state"] = item.get("current_state", "") or existing.get("latest_state", "")
            existing["latest_episode"] = episode_id
            existing["seen_episodes"] = _unique_strings(list(existing.get("seen_episodes", [])) + [episode_id])
            state_history = list(existing.get("state_history", []))
            state_history.append(
                {
                    "episode_id": episode_id,
                    "state": item.get("current_state", ""),
                }
            )
            existing["state_history"] = _dedupe_state_history(state_history)[-12:]

        characters.sort(key=lambda item: (_episode_sort_key(str(item.get("first_episode", ""))), item.get("canonical_name", "")))

    def _merge_locations(
        self,
        registry: dict[str, Any],
        analysis: Mapping[str, Any],
        episode_id: str,
    ) -> None:
        locations = registry.setdefault("locations", [])
        registry["updated_at"] = utc_timestamp()

        for item in analysis.get("locations", []):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            existing = self._find_location_entry(locations, name)
            if existing is None:
                existing = {
                    "canonical_id": item.get("location_id") or f"loc-{len(locations) + 1:04d}",
                    "canonical_name": name,
                    "time_of_day": item.get("time_of_day", ""),
                    "visual_profile": item.get("visual_profile", ""),
                    "props": item.get("props", []),
                    "first_episode": episode_id,
                    "latest_episode": episode_id,
                    "seen_episodes": [episode_id],
                }
                locations.append(existing)

            existing["time_of_day"] = item.get("time_of_day", "") or existing.get("time_of_day", "")
            existing["visual_profile"] = item.get("visual_profile", "") or existing.get("visual_profile", "")
            existing["props"] = _unique_strings(list(existing.get("props", [])) + list(item.get("props", [])))
            existing["latest_episode"] = episode_id
            existing["seen_episodes"] = _unique_strings(list(existing.get("seen_episodes", [])) + [episode_id])

        locations.sort(key=lambda item: (_episode_sort_key(str(item.get("first_episode", ""))), item.get("canonical_name", "")))

    def _merge_timeline(self, plot_timeline: dict[str, Any], episode_summary: Mapping[str, Any]) -> None:
        episodes = plot_timeline.setdefault("episodes", [])
        plot_timeline["updated_at"] = utc_timestamp()
        episodes = [item for item in episodes if item.get("episode_id") != episode_summary.get("episode_id")]
        episodes.append(
            {
                "episode_id": episode_summary.get("episode_id"),
                "title": episode_summary.get("title"),
                "synopsis": episode_summary.get("synopsis"),
                "genre_profile": episode_summary.get("genre_profile", {}),
                "hook_profile": episode_summary.get("hook_profile", {}),
                "camera_language_analysis": episode_summary.get("camera_language_analysis", {}),
                "art_direction_analysis": episode_summary.get("art_direction_analysis", {}),
                "storyboard_blueprint": episode_summary.get("storyboard_blueprint", {}),
                "key_events": episode_summary.get("key_events", []),
                "unresolved_threads": episode_summary.get("unresolved_threads", []),
                "continuity_hooks": episode_summary.get("continuity_hooks", []),
                "involved_characters": episode_summary.get("involved_characters", []),
            }
        )
        episodes.sort(key=lambda item: _episode_sort_key(str(item.get("episode_id", ""))))
        plot_timeline["episodes"] = episodes

    def _merge_series_bible(
        self,
        series_bible: dict[str, Any],
        analysis: Mapping[str, Any],
        episode_summary: Mapping[str, Any],
        character_registry: Mapping[str, Any],
        location_registry: Mapping[str, Any],
        plot_timeline: Mapping[str, Any],
    ) -> None:
        series_bible["updated_at"] = utc_timestamp()
        series_bible["latest_episode_id"] = episode_summary.get("episode_id")
        series_bible["source_episodes"] = _unique_strings(list(series_bible.get("source_episodes", [])) + [episode_summary.get("episode_id", "")])
        series_bible["episode_count"] = len(series_bible["source_episodes"])
        if not str(series_bible.get("premise", "")).strip():
            series_bible["premise"] = str(episode_summary.get("synopsis", "")).strip()
        series_bible["major_arcs"] = _unique_strings(
            list(series_bible.get("major_arcs", [])) + list(episode_summary.get("key_events", []))
        )[:12]
        series_bible["unresolved_threads"] = _unique_strings(
            list(series_bible.get("unresolved_threads", [])) + list(episode_summary.get("unresolved_threads", []))
        )[:12]
        series_bible["active_characters"] = [
            item.get("canonical_name", "")
            for item in list(character_registry.get("characters", []))[:12]
            if str(item.get("canonical_name", "")).strip()
        ]
        series_bible["recurring_locations"] = [
            item.get("canonical_name", "")
            for item in list(location_registry.get("locations", []))[:12]
            if str(item.get("canonical_name", "")).strip()
        ]
        series_bible["genre_profile"] = self._merge_genre_profile(
            existing=series_bible.get("genre_profile", {}),
            current=analysis.get("genre_classification", {}),
            episode_id=str(episode_summary.get("episode_id", "")),
        )
        series_bible["genre_playbooks"] = self._match_genre_playbooks(series_bible.get("genre_profile", {}))
        series_bible["camera_language_profile"] = self._merge_camera_language_profile(
            existing=series_bible.get("camera_language_profile", self._default_camera_language_profile()),
            current=analysis.get("camera_language_analysis", {}),
        )
        series_bible["art_direction_profile"] = self._merge_art_direction_profile(
            existing=series_bible.get("art_direction_profile", self._default_art_direction_profile()),
            current=analysis.get("art_direction_analysis", {}),
        )
        series_bible["storyboard_profile"] = self._merge_storyboard_profile(
            existing=series_bible.get("storyboard_profile", self._default_storyboard_profile()),
            current=analysis.get("storyboard_blueprint", {}),
        )
        series_bible["character_design_profile"] = self._merge_character_design_profile(
            existing=series_bible.get("character_design_profile", self._default_character_design_profile()),
            current=analysis.get("series_learning_extraction", {}),
        )
        series_bible["scene_design_profile"] = self._merge_scene_design_profile(
            existing=series_bible.get("scene_design_profile", self._default_scene_design_profile()),
            current=analysis.get("series_learning_extraction", {}),
        )
        series_bible["dialogue_timing_profile"] = self._merge_dialogue_timing_profile(
            existing=series_bible.get("dialogue_timing_profile", self._default_dialogue_timing_profile()),
            current=analysis.get("series_learning_extraction", {}),
        )
        series_bible["production_guardrails"] = self._merge_production_guardrails(
            existing=series_bible.get("production_guardrails", self._default_production_guardrails()),
            current=analysis.get("series_learning_extraction", {}),
        )
        series_bible["downstream_design_guidance"] = self._merge_downstream_design_guidance(
            existing=series_bible.get("downstream_design_guidance", {}),
            current=analysis.get("downstream_design_guidance", {}),
        )
        series_bible["series_learning_profile"] = self._merge_series_learning_profile(
            existing=series_bible.get("series_learning_profile", self._default_series_learning_profile()),
            current=analysis.get("series_learning_extraction", {}),
        )
        series_bible["continuity_rules"] = self._default_continuity_rules()
        series_bible["timeline_snapshot"] = [
            {
                "episode_id": item.get("episode_id"),
                "synopsis": item.get("synopsis"),
            }
            for item in list(plot_timeline.get("episodes", []))[-5:]
        ]

    def _build_series_context(
        self,
        series_bible: Mapping[str, Any],
        character_registry: Mapping[str, Any],
        location_registry: Mapping[str, Any],
        plot_timeline: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "series_name": self.series_folder,
            "updated_at": utc_timestamp(),
            "premise": series_bible.get("premise", ""),
            "latest_episode_id": series_bible.get("latest_episode_id"),
            "genre_profile": series_bible.get("genre_profile", {}),
            "genre_playbooks": series_bible.get("genre_playbooks", []),
            "camera_language_profile": series_bible.get(
                "camera_language_profile", self._default_camera_language_profile()
            ),
            "art_direction_profile": series_bible.get(
                "art_direction_profile", self._default_art_direction_profile()
            ),
            "storyboard_profile": series_bible.get(
                "storyboard_profile", self._default_storyboard_profile()
            ),
            "character_design_profile": series_bible.get(
                "character_design_profile", self._default_character_design_profile()
            ),
            "scene_design_profile": series_bible.get(
                "scene_design_profile", self._default_scene_design_profile()
            ),
            "dialogue_timing_profile": series_bible.get(
                "dialogue_timing_profile", self._default_dialogue_timing_profile()
            ),
            "production_guardrails": series_bible.get(
                "production_guardrails", self._default_production_guardrails()
            ),
            "downstream_design_guidance": series_bible.get("downstream_design_guidance", {}),
            "series_learning_profile": series_bible.get("series_learning_profile", self._default_series_learning_profile()),
            "continuity_rules": series_bible.get("continuity_rules", []),
            "active_characters": [
                {
                    "name": item.get("canonical_name", ""),
                    "role": item.get("role", ""),
                    "relationship_to_protagonist": item.get("relationship_to_protagonist", ""),
                    "latest_state": item.get("latest_state", ""),
                }
                for item in list(character_registry.get("characters", []))[:12]
            ],
            "active_locations": [
                {
                    "name": item.get("canonical_name", ""),
                    "time_of_day": item.get("time_of_day", ""),
                    "visual_profile": item.get("visual_profile", ""),
                }
                for item in list(location_registry.get("locations", []))[:12]
            ],
            "unresolved_threads": series_bible.get("unresolved_threads", []),
            "recent_timeline": list(plot_timeline.get("episodes", []))[-5:],
        }

    def _merge_genre_profile(
        self,
        *,
        existing: Mapping[str, Any],
        current: Mapping[str, Any],
        episode_id: str,
    ) -> dict[str, Any]:
        merged = dict(existing or {})
        current_primary = str(current.get("primary_genre", "")).strip()
        if current_primary:
            merged["primary_genre"] = current_primary
        merged["secondary_genres"] = _unique_strings(
            list(existing.get("secondary_genres", [])) + list(current.get("secondary_genres", []))
        )[:8]
        current_device = str(current.get("narrative_device", "")).strip()
        if current_device:
            merged["narrative_device"] = current_device
        current_era = str(current.get("setting_era", "")).strip()
        if current_era:
            merged["setting_era"] = current_era
        current_expectation = str(current.get("audience_expectation", "")).strip()
        if current_expectation:
            merged["audience_expectation"] = current_expectation
        confidence = current.get("confidence")
        if isinstance(confidence, (int, float)):
            merged["confidence"] = confidence
        merged["latest_episode"] = episode_id
        merged["source_evidence"] = list(current.get("evidence", []))[:6]
        return merged

    def _match_genre_playbooks(self, genre_profile: Mapping[str, Any]) -> list[dict[str, Any]]:
        library = load_genre_playbook_library()
        tokens = _unique_strings(
            [
                str(genre_profile.get("primary_genre", "")).strip(),
                str(genre_profile.get("narrative_device", "")).strip(),
                *[str(item).strip() for item in genre_profile.get("secondary_genres", [])],
            ]
        )
        normalized_tokens = {_normalize_identity(item) for item in tokens if _normalize_identity(item)}
        matches: list[dict[str, Any]] = []
        for item in library:
            keys = {_normalize_identity(str(item.get("genre_key", "")))}
            keys.update(_normalize_identity(str(alias)) for alias in item.get("aliases", []))
            if normalized_tokens & keys:
                matches.append(item)
        return matches[:4]

    def _default_camera_language_profile(self) -> dict[str, Any]:
        return {
            "dominant_shot_types": [],
            "camera_motion_patterns": [],
            "composition_patterns": [],
            "visual_emphasis_rules": [],
            "transition_rhythm": "",
            "climax_visual_strategy": "",
            "cliffhanger_visual_pattern": "",
        }

    def _default_art_direction_profile(self) -> dict[str, Any]:
        return {
            "costume_signatures": [],
            "hair_makeup_signatures": [],
            "prop_signatures": [],
            "set_signatures": [],
            "lighting_signatures": [],
            "color_mood_patterns": [],
            "texture_material_patterns": [],
        }

    def _default_storyboard_profile(self) -> dict[str, Any]:
        return {
            "opening_hook_blueprint": [],
            "conflict_escalation_blueprint": [],
            "emotional_payoff_blueprint": [],
            "ending_button_blueprint": [],
            "seedance_emphasis_points": [],
            "avoid_patterns": [],
        }

    def _default_character_design_profile(self) -> dict[str, Any]:
        return {
            "character_design_rules": [],
            "costume_makeup_rules": [],
        }

    def _default_scene_design_profile(self) -> dict[str, Any]:
        return {
            "scene_design_rules": [],
        }

    def _default_dialogue_timing_profile(self) -> dict[str, Any]:
        return {
            "dialogue_timing_rules": [],
        }

    def _default_production_guardrails(self) -> dict[str, Any]:
        return {
            "continuity_guardrails": [],
            "negative_patterns": [],
        }

    def _merge_camera_language_profile(
        self,
        *,
        existing: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "dominant_shot_types": _unique_strings(
                list(existing.get("dominant_shot_types", []))
                + list(current.get("dominant_shot_types", []))
            )[:12],
            "camera_motion_patterns": _unique_strings(
                list(existing.get("camera_motion_patterns", []))
                + list(current.get("camera_motion_patterns", []))
            )[:12],
            "composition_patterns": _unique_strings(
                list(existing.get("composition_patterns", []))
                + list(current.get("composition_patterns", []))
            )[:12],
            "visual_emphasis_rules": _unique_strings(
                list(existing.get("visual_emphasis_rules", []))
                + list(current.get("visual_emphasis_rules", []))
            )[:12],
            "transition_rhythm": str(current.get("transition_rhythm", "")).strip()
            or str(existing.get("transition_rhythm", "")).strip(),
            "climax_visual_strategy": str(current.get("climax_visual_strategy", "")).strip()
            or str(existing.get("climax_visual_strategy", "")).strip(),
            "cliffhanger_visual_pattern": str(current.get("cliffhanger_visual_pattern", "")).strip()
            or str(existing.get("cliffhanger_visual_pattern", "")).strip(),
        }

    def _merge_art_direction_profile(
        self,
        *,
        existing: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "costume_signatures": _unique_strings(
                list(existing.get("costume_signatures", []))
                + list(current.get("costume_signatures", []))
            )[:12],
            "hair_makeup_signatures": _unique_strings(
                list(existing.get("hair_makeup_signatures", []))
                + list(current.get("hair_makeup_signatures", []))
            )[:12],
            "prop_signatures": _unique_strings(
                list(existing.get("prop_signatures", []))
                + list(current.get("prop_signatures", []))
            )[:12],
            "set_signatures": _unique_strings(
                list(existing.get("set_signatures", []))
                + list(current.get("set_signatures", []))
            )[:12],
            "lighting_signatures": _unique_strings(
                list(existing.get("lighting_signatures", []))
                + list(current.get("lighting_signatures", []))
            )[:12],
            "color_mood_patterns": _unique_strings(
                list(existing.get("color_mood_patterns", []))
                + list(current.get("color_mood_patterns", []))
            )[:12],
            "texture_material_patterns": _unique_strings(
                list(existing.get("texture_material_patterns", []))
                + list(current.get("texture_material_patterns", []))
            )[:12],
        }

    def _merge_storyboard_profile(
        self,
        *,
        existing: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "opening_hook_blueprint": _unique_strings(
                list(existing.get("opening_hook_blueprint", []))
                + list(current.get("opening_hook_blueprint", []))
            )[:12],
            "conflict_escalation_blueprint": _unique_strings(
                list(existing.get("conflict_escalation_blueprint", []))
                + list(current.get("conflict_escalation_blueprint", []))
            )[:12],
            "emotional_payoff_blueprint": _unique_strings(
                list(existing.get("emotional_payoff_blueprint", []))
                + list(current.get("emotional_payoff_blueprint", []))
            )[:12],
            "ending_button_blueprint": _unique_strings(
                list(existing.get("ending_button_blueprint", []))
                + list(current.get("ending_button_blueprint", []))
            )[:12],
            "seedance_emphasis_points": _unique_strings(
                list(existing.get("seedance_emphasis_points", []))
                + list(current.get("seedance_emphasis_points", []))
            )[:12],
            "avoid_patterns": _unique_strings(
                list(existing.get("avoid_patterns", []))
                + list(current.get("avoid_patterns", []))
            )[:12],
        }

    def _merge_character_design_profile(
        self,
        *,
        existing: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "character_design_rules": _unique_strings(
                list(existing.get("character_design_rules", []))
                + list(current.get("character_design_rules", []))
            )[:16],
            "costume_makeup_rules": _unique_strings(
                list(existing.get("costume_makeup_rules", []))
                + list(current.get("costume_makeup_rules", []))
            )[:16],
        }

    def _merge_scene_design_profile(
        self,
        *,
        existing: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "scene_design_rules": _unique_strings(
                list(existing.get("scene_design_rules", []))
                + list(current.get("scene_design_rules", []))
            )[:16],
        }

    def _merge_dialogue_timing_profile(
        self,
        *,
        existing: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "dialogue_timing_rules": _unique_strings(
                list(existing.get("dialogue_timing_rules", []))
                + list(current.get("dialogue_timing_rules", []))
            )[:16],
        }

    def _merge_production_guardrails(
        self,
        *,
        existing: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "continuity_guardrails": _unique_strings(
                list(existing.get("continuity_guardrails", []))
                + list(current.get("continuity_guardrails", []))
            )[:16],
            "negative_patterns": _unique_strings(
                list(existing.get("negative_patterns", []))
                + list(current.get("negative_patterns", []))
            )[:16],
        }

    def _merge_downstream_design_guidance(
        self,
        *,
        existing: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "script_reconstruction_focus": _unique_strings(
                list(existing.get("script_reconstruction_focus", []))
                + list(current.get("script_reconstruction_focus", []))
            )[:10],
            "character_design_focus": _unique_strings(
                list(existing.get("character_design_focus", []))
                + list(current.get("character_design_focus", []))
            )[:10],
            "scene_design_focus": _unique_strings(
                list(existing.get("scene_design_focus", []))
                + list(current.get("scene_design_focus", []))
            )[:10],
            "storyboard_focus": _unique_strings(
                list(existing.get("storyboard_focus", []))
                + list(current.get("storyboard_focus", []))
            )[:10],
            "adaptation_priorities": _unique_strings(
                list(existing.get("adaptation_priorities", []))
                + list(current.get("adaptation_priorities", []))
            )[:10],
        }

    def _default_series_learning_profile(self) -> dict[str, Any]:
        return {
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
        }

    def _merge_series_learning_profile(
        self,
        *,
        existing: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "episode_strengths": _unique_strings(
                list(existing.get("episode_strengths", [])) + list(current.get("episode_strengths", []))
            )[:16],
            "why_it_works": _unique_strings(
                list(existing.get("why_it_works", [])) + list(current.get("why_it_works", []))
            )[:16],
            "character_design_rules": _unique_strings(
                list(existing.get("character_design_rules", [])) + list(current.get("character_design_rules", []))
            )[:16],
            "costume_makeup_rules": _unique_strings(
                list(existing.get("costume_makeup_rules", [])) + list(current.get("costume_makeup_rules", []))
            )[:16],
            "scene_design_rules": _unique_strings(
                list(existing.get("scene_design_rules", [])) + list(current.get("scene_design_rules", []))
            )[:16],
            "camera_language_rules": _unique_strings(
                list(existing.get("camera_language_rules", [])) + list(current.get("camera_language_rules", []))
            )[:16],
            "storyboard_execution_rules": _unique_strings(
                list(existing.get("storyboard_execution_rules", [])) + list(current.get("storyboard_execution_rules", []))
            )[:16],
            "dialogue_timing_rules": _unique_strings(
                list(existing.get("dialogue_timing_rules", [])) + list(current.get("dialogue_timing_rules", []))
            )[:16],
            "continuity_guardrails": _unique_strings(
                list(existing.get("continuity_guardrails", [])) + list(current.get("continuity_guardrails", []))
            )[:16],
            "negative_patterns": _unique_strings(
                list(existing.get("negative_patterns", [])) + list(current.get("negative_patterns", []))
            )[:16],
            "reusable_playbook_rules": _unique_strings(
                list(existing.get("reusable_playbook_rules", [])) + list(current.get("reusable_playbook_rules", []))
            )[:16],
            "reusable_skill_rules": _unique_strings(
                list(existing.get("reusable_skill_rules", [])) + list(current.get("reusable_skill_rules", []))
            )[:16],
            "character_appeal_patterns": _unique_strings(
                list(existing.get("character_appeal_patterns", [])) + list(current.get("character_appeal_patterns", []))
            )[:16],
            "scene_staging_patterns": _unique_strings(
                list(existing.get("scene_staging_patterns", [])) + list(current.get("scene_staging_patterns", []))
            )[:16],
            "dialogue_patterns": _unique_strings(
                list(existing.get("dialogue_patterns", [])) + list(current.get("dialogue_patterns", []))
            )[:16],
            "camera_language_patterns": _unique_strings(
                list(existing.get("camera_language_patterns", []))
                + list(current.get("camera_language_patterns", []))
            )[:16],
            "costume_image_patterns": _unique_strings(
                list(existing.get("costume_image_patterns", []))
                + list(current.get("costume_image_patterns", []))
            )[:16],
            "prop_visual_patterns": _unique_strings(
                list(existing.get("prop_visual_patterns", []))
                + list(current.get("prop_visual_patterns", []))
            )[:16],
            "storyboard_execution_patterns": _unique_strings(
                list(existing.get("storyboard_execution_patterns", []))
                + list(current.get("storyboard_execution_patterns", []))
            )[:16],
        }

    def _build_series_strength_playbook(self, series_bible: Mapping[str, Any]) -> dict[str, Any]:
        learning = dict(series_bible.get("series_learning_profile", {}))
        return {
            "series_name": self.series_folder,
            "updated_at": utc_timestamp(),
            "episode_count": series_bible.get("episode_count", 0),
            "episode_strengths": learning.get("episode_strengths", []),
            "why_it_works": learning.get("why_it_works", []),
            "character_design_rules": learning.get("character_design_rules", []),
            "costume_makeup_rules": learning.get("costume_makeup_rules", []),
            "scene_design_rules": learning.get("scene_design_rules", []),
            "camera_language_rules": learning.get("camera_language_rules", []),
            "storyboard_execution_rules": learning.get("storyboard_execution_rules", []),
            "dialogue_timing_rules": learning.get("dialogue_timing_rules", []),
            "continuity_guardrails": learning.get("continuity_guardrails", []),
            "negative_patterns": learning.get("negative_patterns", []),
            "reusable_playbook_rules": learning.get("reusable_playbook_rules", []),
            "reusable_skill_rules": learning.get("reusable_skill_rules", []),
            "character_appeal_patterns": learning.get("character_appeal_patterns", []),
            "scene_staging_patterns": learning.get("scene_staging_patterns", []),
            "dialogue_patterns": learning.get("dialogue_patterns", []),
            "camera_language_patterns": learning.get("camera_language_patterns", []),
            "costume_image_patterns": learning.get("costume_image_patterns", []),
            "prop_visual_patterns": learning.get("prop_visual_patterns", []),
            "storyboard_execution_patterns": learning.get("storyboard_execution_patterns", []),
        }

    def _render_series_strength_playbook_markdown(self, data: Mapping[str, Any]) -> str:
        lines = [
            "# 本剧优点经验沉淀草稿",
            "",
            f"- 剧名：{data.get('series_name', '')}",
            f"- 集数累计：{data.get('episode_count', 0)}",
            "",
            "## 这部剧当前做得好的地方",
            "",
            *[f"- {item}" for item in data.get("episode_strengths", [])],
            "",
            "## 为什么这些点有效",
            "",
            *[f"- {item}" for item in data.get("why_it_works", [])],
            "",
            "## 人物设计规则",
            "",
            *[f"- {item}" for item in data.get("character_design_rules", [])],
            "",
            "## 服化道统一规则",
            "",
            *[f"- {item}" for item in data.get("costume_makeup_rules", [])],
            "",
            "## 场景设计规则",
            "",
            *[f"- {item}" for item in data.get("scene_design_rules", [])],
            "",
            "## 镜头语言规则",
            "",
            *[f"- {item}" for item in data.get("camera_language_rules", [])],
            "",
            "## 分镜执行规则",
            "",
            *[f"- {item}" for item in data.get("storyboard_execution_rules", [])],
            "",
            "## 台词时间规则",
            "",
            *[f"- {item}" for item in data.get("dialogue_timing_rules", [])],
            "",
            "## 连续性红线",
            "",
            *[f"- {item}" for item in data.get("continuity_guardrails", [])],
            "",
            "## 高风险负面模式",
            "",
            *[f"- {item}" for item in data.get("negative_patterns", [])],
            "",
            "## 兼容旧字段：可复用 Playbook 规则",
            "",
            *[f"- {item}" for item in data.get("reusable_playbook_rules", [])],
            "",
            "## 兼容旧字段：可复用 Skill 规则",
            "",
            *[f"- {item}" for item in data.get("reusable_skill_rules", [])],
            "",
            "## 兼容旧字段：人物吸引力模式",
            "",
            *[f"- {item}" for item in data.get("character_appeal_patterns", [])],
            "",
            "## 兼容旧字段：场景调度模式",
            "",
            *[f"- {item}" for item in data.get("scene_staging_patterns", [])],
            "",
            "## 兼容旧字段：台词模式",
            "",
            *[f"- {item}" for item in data.get("dialogue_patterns", [])],
            "",
            "## 兼容旧字段：镜头语言模式",
            "",
            *[f"- {item}" for item in data.get("camera_language_patterns", [])],
            "",
            "## 兼容旧字段：服装与妆造意象",
            "",
            *[f"- {item}" for item in data.get("costume_image_patterns", [])],
            "",
            "## 兼容旧字段：道具视觉模式",
            "",
            *[f"- {item}" for item in data.get("prop_visual_patterns", [])],
            "",
            "## 兼容旧字段：分镜执行模式",
            "",
            *[f"- {item}" for item in data.get("storyboard_execution_patterns", [])],
            "",
        ]
        return "\n".join(lines)

    def _find_previous_episode_summary(self, current_episode_id: str) -> dict[str, Any] | None:
        if not self.episode_summaries_dir.exists():
            return None
        candidates: list[dict[str, Any]] = []
        for path in sorted(self.episode_summaries_dir.glob("*.json")):
            try:
                item = self._load_or_default(path, {})
            except Exception:
                continue
            episode_id = str(item.get("episode_id", "")).strip()
            if not episode_id or episode_id == current_episode_id:
                continue
            candidates.append(item)
        if not candidates:
            return None
        candidates.sort(key=lambda item: _episode_sort_key(str(item.get("episode_id", ""))))
        current_key = _episode_sort_key(current_episode_id)
        previous = [item for item in candidates if _episode_sort_key(str(item.get("episode_id", ""))) < current_key]
        if previous:
            return previous[-1]
        return candidates[-1]

    def _find_character_entry(self, characters: list[dict[str, Any]], aliases: list[str]) -> dict[str, Any] | None:
        alias_keys = {_normalize_identity(item) for item in aliases if _normalize_identity(item)}
        for character in characters:
            existing_keys = {
                _normalize_identity(character.get("canonical_name", "")),
                *[_normalize_identity(item) for item in character.get("aliases", [])],
            }
            if alias_keys & existing_keys:
                return character
        return None

    def _find_location_entry(self, locations: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
        key = _normalize_identity(name)
        for location in locations:
            if _normalize_identity(location.get("canonical_name", "")) == key:
                return location
        return None

    def _load_or_default(self, path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return dict(default)
        import json

        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _episode_summary_path_for(self, episode_id: str) -> Path:
        return self.episode_summaries_dir / f"{episode_id}.json"

    def _default_continuity_rules(self) -> list[str]:
        return [
            "历史上下文只用于实体对齐、称呼统一、状态承接和悬念延续判断。",
            "若历史记录与当前视频直接证据冲突，优先采用当前视频，并在 continuity_notes 标明冲突。",
            "未被当前视频再次验证的旧推断只能作为参考，不能当作硬事实强行延续。",
        ]
