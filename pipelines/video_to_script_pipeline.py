from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genre_routing import build_genre_debug_report, render_genre_debug_markdown
from pipeline_telemetry import TelemetryRecorder, telemetry_span
from providers.base import (
    EpisodeInputBundle,
    FrameReference,
    ProviderConfigurationError,
    build_provider_model_tag,
    coerce_mapping,
    derive_series_folder_name,
    load_json_file,
    read_text_file,
    save_json_file,
    save_text_file,
)
from pipelines.continuity_manager import SeriesContinuityManager
from providers.gemini_adapter import GeminiAdapter
from providers.openai_adapter import OpenAIAdapter
from providers.qwen_adapter import QwenAdapter


DEFAULT_SCHEMA_PATH = Path("schemas/episode_analysis.schema.json")


@dataclass
class PipelineResult:
    analysis_path: Path
    script_path: Path
    analysis: dict[str, Any]
    script_markdown: str
    continuity_paths: dict[str, Path]
    genre_debug_paths: dict[str, Path]
    genre_debug_summary: dict[str, Any]


@dataclass
class PipelineConfig:
    provider: str
    model: str
    schema_path: Path = DEFAULT_SCHEMA_PATH
    analysis_root: Path = Path("analysis")
    script_root: Path = Path("script")
    temperature: float = 0.2
    timeout_seconds: int = 180
    provider_endpoint: str = ""
    openai_image_detail: str = "auto"
    openai_max_analysis_frames: int | None = 20
    qwen_max_analysis_frames: int | None = 20
    qwen_video_fps: float = 2.0
    qwen_structured_output_mode: str = "json_schema"
    telemetry: TelemetryRecorder | None = None


class VideoToScriptPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.provider = self._build_provider()
        self.schema = load_json_file(self.config.schema_path)

    def run(self, bundle: EpisodeInputBundle) -> PipelineResult:
        series_folder = derive_series_folder_name(
            video_path=bundle.video_path,
            explicit_series_name=str(bundle.metadata.get("source_series", "")).strip() or None,
        )
        continuity_manager = SeriesContinuityManager(
            analysis_root=self.config.analysis_root,
            series_folder=series_folder,
            provider=self.config.provider,
            model=self.config.model,
        )
        with telemetry_span(
            self.config.telemetry,
            stage="video_to_script",
            name="build_series_continuity_context",
            metadata={"series_folder": series_folder, "episode_id": bundle.episode_id},
        ):
            continuity_context = continuity_manager.build_context(bundle)
        bundle = continuity_context.enriched_bundle

        analysis = self.provider.analyze_episode(bundle, self.schema)
        output_tag = build_provider_model_tag(self.config.provider, self.config.model)

        analysis_path = (
            self.config.analysis_root
            / series_folder
            / bundle.episode_id
            / f"episode_analysis__{output_tag}.json"
        )
        with telemetry_span(
            self.config.telemetry,
            stage="video_to_script",
            name="save_episode_analysis_json",
            metadata={"analysis_path": str(analysis_path)},
        ):
            save_json_file(analysis_path, analysis)

        script_markdown = self.provider.reconstruct_script(bundle, analysis).strip()
        if not script_markdown:
            script_markdown = self._render_script_fallback(bundle, analysis)

        script_path = (
            self.config.script_root
            / series_folder
            / f"{bundle.episode_id}__{output_tag}.md"
        )
        with telemetry_span(
            self.config.telemetry,
            stage="video_to_script",
            name="save_reconstructed_script_markdown",
            metadata={"script_path": str(script_path)},
        ):
            save_text_file(script_path, script_markdown)

        with telemetry_span(
            self.config.telemetry,
            stage="video_to_script",
            name="update_series_continuity_state",
            metadata={"series_folder": series_folder, "episode_id": bundle.episode_id},
        ):
            continuity_update = continuity_manager.update_from_episode(
                bundle=bundle,
                analysis=analysis,
                analysis_path=analysis_path,
                script_path=script_path,
            )

        genre_debug_json_path = (
            self.config.analysis_root
            / series_folder
            / bundle.episode_id
            / f"genre_routing_debug__{output_tag}.json"
        )
        genre_debug_markdown_path = (
            self.config.analysis_root
            / series_folder
            / bundle.episode_id
            / f"genre_routing_debug__{output_tag}.md"
        )
        with telemetry_span(
            self.config.telemetry,
            stage="video_to_script",
            name="save_genre_routing_debug_report",
            metadata={"episode_id": bundle.episode_id, "genre_debug_json_path": str(genre_debug_json_path)},
        ):
            series_context = load_json_file(continuity_update.series_context_path)
            genre_debug_report = build_genre_debug_report(
                bundle=bundle,
                analysis=analysis,
                series_context=series_context,
            )
            save_json_file(genre_debug_json_path, genre_debug_report)
            save_text_file(genre_debug_markdown_path, render_genre_debug_markdown(genre_debug_report))

        return PipelineResult(
            analysis_path=analysis_path.resolve(),
            script_path=script_path.resolve(),
            analysis=analysis,
            script_markdown=script_markdown,
            continuity_paths={
                "series_bible_path": continuity_update.series_bible_path,
                "character_registry_path": continuity_update.character_registry_path,
                "location_registry_path": continuity_update.location_registry_path,
                "plot_timeline_path": continuity_update.plot_timeline_path,
                "series_context_path": continuity_update.series_context_path,
                "episode_summary_path": continuity_update.episode_summary_path,
                "series_strength_playbook_json_path": continuity_update.series_strength_playbook_json_path,
                "series_strength_playbook_markdown_path": continuity_update.series_strength_playbook_markdown_path,
            },
            genre_debug_paths={
                "json_path": genre_debug_json_path.resolve(),
                "markdown_path": genre_debug_markdown_path.resolve(),
            },
            genre_debug_summary={
                "primary_genre": str(coerce_mapping(analysis.get("genre_classification")).get("primary_genre", "")).strip(),
                "secondary_genres": list(coerce_mapping(analysis.get("genre_classification")).get("secondary_genres", [])),
                "confirmed_user_genres": list(coerce_mapping(analysis.get("genre_classification")).get("confirmed_user_genres", [])),
                "genre_resolution_mode": str(coerce_mapping(analysis.get("genre_classification")).get("genre_resolution_mode", "")).strip(),
                "genre_override_request": coerce_mapping(analysis.get("genre_override_request")),
                "core_skill_path": genre_debug_report.get("core_skill_path", ""),
                "playbook_library_path": genre_debug_report.get("playbook_library_path", ""),
                "pre_analysis_routing": dict(genre_debug_report.get("pre_analysis_routing", {})),
                "post_analysis_routing": dict(genre_debug_report.get("post_analysis_routing", {})),
                "matched_playbooks": list(series_context.get("genre_playbooks", [])),
                "json_path": genre_debug_json_path.resolve(),
                "markdown_path": genre_debug_markdown_path.resolve(),
            },
        )

    def _build_provider(self):
        provider = self.config.provider.lower().strip()
        if provider == "openai":
            return OpenAIAdapter(
                model=self.config.model,
                endpoint=self.config.provider_endpoint or "https://api.openai.com/v1/responses",
                temperature=self.config.temperature,
                timeout_seconds=self.config.timeout_seconds,
                image_detail=self.config.openai_image_detail,
                max_analysis_frames=self.config.openai_max_analysis_frames,
                telemetry=self.config.telemetry,
            )
        if provider == "gemini":
            return GeminiAdapter(
                model=self.config.model,
                endpoint=self.config.provider_endpoint or None,
                temperature=self.config.temperature,
                timeout_seconds=self.config.timeout_seconds,
                telemetry=self.config.telemetry,
            )
        if provider == "qwen":
            return QwenAdapter(
                model=self.config.model,
                endpoint=(
                    self.config.provider_endpoint
                    or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
                ),
                temperature=self.config.temperature,
                timeout_seconds=self.config.timeout_seconds,
                max_analysis_frames=self.config.qwen_max_analysis_frames,
                video_fps=self.config.qwen_video_fps,
                structured_output_mode=self.config.qwen_structured_output_mode,
                telemetry=self.config.telemetry,
            )
        raise ProviderConfigurationError(f"不支持的 provider：{self.config.provider}")

    def _render_script_fallback(
        self,
        bundle: EpisodeInputBundle,
        analysis: Mapping[str, Any],
    ) -> str:
        title = (
            analysis.get("episode", {}).get("title")
            or bundle.title
            or bundle.episode_id
        )
        lines = [str(title), ""]

        for beat in analysis.get("story_beats", []):
            location = beat.get("location", "未命名场景")
            time_of_day = beat.get("time_of_day", "未知时间")
            lines.append(f"※ {location}，{time_of_day}")
            lines.append("")

            summary = beat.get("summary")
            if summary:
                lines.append(f"△ {summary}")

            for action in beat.get("key_actions", []):
                lines.append(f"△ {action}")

            dialogue_ids = set(beat.get("dialogue_line_ids", []))
            for segment in analysis.get("dialogue_segments", []):
                if segment.get("line_id") not in dialogue_ids:
                    continue
                speaker = segment.get("speaker", "人物")
                text = segment.get("text", "").strip()
                if not text:
                    continue
                if speaker in {"独白", "旁白"}:
                    lines.append(f"【{speaker}】{text}")
                else:
                    lines.append(f"{speaker}：{text}")

            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines).strip()


def parse_frame_arg(raw: str) -> FrameReference:
    note = None
    timestamp = None
    path_part = raw

    if "|" in raw:
        path_part, note = raw.split("|", 1)
    if "@" in path_part:
        path_part, timestamp = path_part.split("@", 1)

    return FrameReference(path=path_part, timestamp=timestamp, note=note)


def build_bundle_from_args(args: argparse.Namespace) -> EpisodeInputBundle:
    transcript_text = args.transcript_text or read_text_file(args.transcript_file)
    ocr_text = args.ocr_text or read_text_file(args.ocr_file)
    synopsis_text = args.synopsis_text or read_text_file(args.synopsis_file)
    frames = [parse_frame_arg(item) for item in args.frame]

    return EpisodeInputBundle(
        episode_id=args.episode_id,
        title=args.title,
        video_path=args.video,
        transcript_text=transcript_text,
        ocr_text=ocr_text,
        synopsis_text=synopsis_text,
        frames=frames,
        context_notes=args.note,
        language=args.language,
        metadata={"source_series": args.series_name} if args.series_name else {},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run video -> episode_analysis -> script pipeline.")
    parser.add_argument("--provider", required=True, choices=["openai", "gemini", "qwen"])
    parser.add_argument("--model", required=True, help="Provider model id, e.g. gpt-5 or gemini-2.0-flash")
    parser.add_argument("--episode-id", required=True, help="Episode identifier, e.g. ep01")
    parser.add_argument("--title", help="Optional episode title")
    parser.add_argument("--series-name", help="Optional series name")
    parser.add_argument("--language", default="zh-CN")
    parser.add_argument("--video", help="Raw video path. Gemini adapter can use this directly.")
    parser.add_argument("--transcript-file", help="Transcript text file path")
    parser.add_argument("--transcript-text", help="Transcript text inline")
    parser.add_argument("--ocr-file", help="OCR text file path")
    parser.add_argument("--ocr-text", help="OCR text inline")
    parser.add_argument("--synopsis-file", help="Optional synopsis text file")
    parser.add_argument("--synopsis-text", help="Optional synopsis text inline")
    parser.add_argument(
        "--frame",
        action="append",
        default=[],
        help="Frame input. Format: path[@timestamp][|note]. Repeatable.",
    )
    parser.add_argument("--note", action="append", default=[], help="Extra context note. Repeatable.")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--analysis-root", default="analysis")
    parser.add_argument("--script-root", default="script")
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    bundle = build_bundle_from_args(args)
    pipeline = VideoToScriptPipeline(
        PipelineConfig(
            provider=args.provider,
            model=args.model,
            schema_path=Path(args.schema),
            analysis_root=Path(args.analysis_root),
            script_root=Path(args.script_root),
            temperature=args.temperature,
        )
    )
    result = pipeline.run(bundle)
    print(f"analysis_json={result.analysis_path}")
    print(f"script_md={result.script_path}")


if __name__ == "__main__":
    main()
