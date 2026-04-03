from __future__ import annotations

import base64
import copy
import json
import mimetypes
import os
import re
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class ProviderError(RuntimeError):
    """Base error for provider failures."""


class ProviderConfigurationError(ProviderError):
    """Raised when provider config is incomplete or invalid."""


class ProviderAPIError(ProviderError):
    """Raised when the upstream model API returns an error."""


class ProviderResponseError(ProviderError):
    """Raised when the model response cannot be parsed safely."""


class UnsupportedInputError(ProviderError):
    """Raised when the selected provider cannot handle the given inputs."""


class SchemaValidationError(ProviderError):
    """Raised when the provider result does not satisfy the schema."""


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_structured_output: bool = True
    supports_image_inputs: bool = True
    supports_video_inputs: bool = False


@dataclass(frozen=True)
class FrameReference:
    path: str
    timestamp: str | None = None
    note: str | None = None
    mime_type: str | None = None

    def resolved_path(self) -> Path:
        return Path(self.path).expanduser().resolve()

    def detected_mime_type(self) -> str:
        return self.mime_type or guess_mime_type(self.resolved_path(), fallback="image/jpeg")


@dataclass
class EpisodeInputBundle:
    episode_id: str
    title: str | None = None
    video_path: str | None = None
    transcript_text: str | None = None
    ocr_text: str | None = None
    synopsis_text: str | None = None
    frames: list[FrameReference] = field(default_factory=list)
    context_notes: list[str] = field(default_factory=list)
    language: str = "zh-CN"
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolved_video_path(self) -> Path | None:
        if not self.video_path:
            return None
        return Path(self.video_path).expanduser().resolve()

    def validate(self) -> None:
        if not self.episode_id.strip():
            raise ProviderConfigurationError("episode_id 不能为空。")

        has_any_content = any(
            [
                self.video_path,
                self.transcript_text and self.transcript_text.strip(),
                self.ocr_text and self.ocr_text.strip(),
                self.frames,
                self.synopsis_text and self.synopsis_text.strip(),
            ]
        )
        if not has_any_content:
            raise UnsupportedInputError(
                "EpisodeInputBundle 至少需要提供 video_path、transcript_text、ocr_text、frames 或 synopsis_text 之一。"
            )

    def as_prompt_summary(self) -> str:
        lines = [
            f"- episode_id: {self.episode_id}",
            f"- title: {self.title or ''}",
            f"- language: {self.language}",
            f"- video_path: {self.video_path or ''}",
        ]
        if self.synopsis_text:
            lines.append(f"- synopsis_hint: {truncate_text(self.synopsis_text, 1200)}")
        if self.context_notes:
            lines.append("- context_notes:")
            lines.extend(f"  - {truncate_text(note, 1200)}" for note in self.context_notes[:6])
        if self.metadata:
            lines.append(
                f"- metadata: {json.dumps(self._prompt_safe_metadata(), ensure_ascii=False, sort_keys=True)}"
            )
        if self.frames:
            lines.append("- frames:")
            for index, frame in enumerate(self.frames, start=1):
                label = frame.resolved_path().name
                suffix_parts = [part for part in [frame.timestamp, frame.note] if part]
                suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
                lines.append(f"  - [{index}] {label}{suffix}")
        return "\n".join(lines)

    def _prompt_safe_metadata(self) -> dict[str, Any]:
        metadata = dict(self.metadata or {})
        continuity = dict(metadata.pop("continuity_context", {}))
        prompt_metadata: dict[str, Any] = {}
        for key in [
            "source_series",
            "user_genre_hints",
            "user_custom_genre_hints",
            "ai_suggested_genre_hints",
            "genre_hint_source",
        ]:
            if key in metadata and metadata[key]:
                prompt_metadata[key] = metadata[key]
        if continuity:
            prompt_metadata["continuity_context_summary"] = {
                "series_folder": continuity.get("series_folder", ""),
                "previous_episode_id": continuity.get("previous_episode_id", ""),
                "registered_character_count": continuity.get("registered_character_count", 0),
                "registered_location_count": continuity.get("registered_location_count", 0),
                "timeline_episode_count": continuity.get("timeline_episode_count", 0),
                "primary_genre": dict(continuity.get("genre_profile", {})).get("primary_genre", ""),
                "matched_playbooks": [
                    str(item.get("genre_key", "")).strip()
                    for item in continuity.get("genre_playbooks", [])
                    if isinstance(item, dict) and str(item.get("genre_key", "")).strip()
                ][:4],
            }
        return prompt_metadata


@dataclass
class ProviderConfig:
    name: str
    model: str
    api_key_env: str
    endpoint: str
    timeout_seconds: int = 180
    temperature: float = 0.2
    max_retries: int = 1
    extra_config: dict[str, Any] = field(default_factory=dict)


class EpisodeAnalysisProvider(ABC):
    capabilities = ProviderCapabilities()

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    def analyze_episode(
        self,
        bundle: EpisodeInputBundle,
        schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def reconstruct_script(
        self,
        bundle: EpisodeInputBundle,
        analysis: Mapping[str, Any],
    ) -> str:
        raise NotImplementedError

    def require_api_key(self) -> str:
        api_key = os.getenv(self.config.api_key_env, "").strip()
        if not api_key:
            raise ProviderConfigurationError(
                f"环境变量 {self.config.api_key_env} 未设置，无法调用 {self.config.name} provider。"
            )
        return api_key

    def request(
        self,
        url: str,
        *,
        method: str = "POST",
        headers: Mapping[str, str] | None = None,
        data: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        effective_timeout = timeout_seconds or self.config.timeout_seconds
        retryable_http_codes = {408, 409, 425, 429, 500, 502, 503, 504}
        total_attempts = max(int(self.config.max_retries or 0), 0) + 1

        for attempt in range(1, total_attempts + 1):
            request = urllib.request.Request(url=url, data=data, method=method)
            for key, value in (headers or {}).items():
                request.add_header(key, value)
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=effective_timeout,
                ) as response:
                    status = getattr(response, "status", response.getcode())
                    raw_headers = {key.lower(): value for key, value in response.headers.items()}
                    body = response.read()
                    return status, raw_headers, body
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if attempt < total_attempts and exc.code in retryable_http_codes:
                    time.sleep(min(float(attempt), 3.0))
                    continue
                raise ProviderAPIError(
                    f"{self.config.name} API 请求失败，状态码 {exc.code}，attempt={attempt}/{total_attempts}，响应：{error_body}"
                ) from exc
            except (TimeoutError, socket.timeout) as exc:
                if attempt < total_attempts:
                    time.sleep(min(float(attempt), 3.0))
                    continue
                raise ProviderAPIError(
                    f"{self.config.name} API 请求超时：timeout={effective_timeout}s method={method} url={url} attempts={total_attempts}"
                ) from exc
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", exc)
                if isinstance(reason, (TimeoutError, socket.timeout)):
                    if attempt < total_attempts:
                        time.sleep(min(float(attempt), 3.0))
                        continue
                    raise ProviderAPIError(
                        f"{self.config.name} API 请求超时：timeout={effective_timeout}s method={method} url={url} attempts={total_attempts}"
                    ) from exc
                if attempt < total_attempts:
                    time.sleep(min(float(attempt), 3.0))
                    continue
                raise ProviderAPIError(
                    f"{self.config.name} API 网络请求失败：attempt={attempt}/{total_attempts} {exc}"
                ) from exc

        raise ProviderAPIError(
            f"{self.config.name} API 请求失败：method={method} url={url} attempts={total_attempts}"
        )

    def request_json(
        self,
        url: str,
        *,
        method: str = "POST",
        headers: Mapping[str, str] | None = None,
        payload: Mapping[str, Any] | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        merged_headers: dict[str, str] = {}
        body: bytes | None = None
        if payload is not None:
            merged_headers["Content-Type"] = "application/json"
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        merged_headers.update(headers or {})
        _, _, raw_body = self.request(
            url,
            method=method,
            headers=merged_headers,
            data=body,
            timeout_seconds=timeout_seconds,
        )
        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                f"{self.config.name} 返回了无法解析的 JSON：{raw_body[:500]!r}"
            ) from exc


def load_json_file(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_file(path: str | Path, data: Mapping[str, Any]) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output_path


def save_text_file(path: str | Path, text: str) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
    return output_path


def read_text_file(path: str | Path | None) -> str | None:
    if not path:
        return None
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"文件不存在：{target}")
    return target.read_text(encoding="utf-8")


def guess_mime_type(path: str | Path, fallback: str = "application/octet-stream") -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or fallback


def file_to_base64(path: str | Path) -> str:
    with Path(path).expanduser().resolve().open("rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def file_to_data_url(path: str | Path, mime_type: str | None = None) -> str:
    resolved = Path(path).expanduser().resolve()
    detected_mime = mime_type or guess_mime_type(resolved)
    return f"data:{detected_mime};base64,{file_to_base64(resolved)}"


def truncate_text(text: str | None, max_chars: int) -> str:
    if not text:
        return ""
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"


def extract_json_from_text(text: str) -> dict[str, Any]:
    clean = text.strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(clean):
            if char not in "[{":
                continue
            try:
                data, _ = decoder.raw_decode(clean[index:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise ProviderResponseError(f"未能从模型输出中提取 JSON：{truncate_text(clean, 500)}")
    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], dict):
            return data[0]
        raise ProviderResponseError(
            "模型返回的结构化结果不是 JSON object，而是 JSON array："
            + truncate_text(clean, 500)
        )
    if not isinstance(data, dict):
        raise ProviderResponseError(
            f"模型返回的结构化结果不是 JSON object，而是 {type(data).__name__}："
            + truncate_text(clean, 500)
        )
    return data


def coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def ensure_object_field(
    document: dict[str, Any],
    key: str,
    default: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = document.get(key)
    if isinstance(value, dict):
        result = value
    elif isinstance(value, Mapping):
        result = dict(value)
        document[key] = result
    else:
        result = {}
        document[key] = result
    for default_key, default_value in (default or {}).items():
        if default_key not in result:
            result[default_key] = copy.deepcopy(default_value)
    return result


def validate_against_schema(document: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError:
        return

    try:
        jsonschema.validate(document, schema)
    except jsonschema.ValidationError as exc:
        raise SchemaValidationError(f"结果未通过 schema 校验：{exc.message}") from exc


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sanitize_folder_name(raw: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "-", raw.strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or "未命名剧集"


def sanitize_filename_component(raw: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", raw.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-.")
    return cleaned or "unknown"


def build_provider_model_tag(provider: str, model: str) -> str:
    return f"{sanitize_filename_component(provider)}__{sanitize_filename_component(model)}"


def derive_series_folder_name(
    *,
    video_path: str | Path | None = None,
    explicit_series_name: str | None = None,
) -> str:
    explicit = (explicit_series_name or "").strip()
    if explicit and explicit not in {"待填写剧名", "未填写剧名"}:
        return sanitize_folder_name(explicit)

    if video_path:
        parent_name = Path(video_path).expanduser().resolve().parent.name
        if parent_name:
            return sanitize_folder_name(parent_name)

    return "未命名剧集"
