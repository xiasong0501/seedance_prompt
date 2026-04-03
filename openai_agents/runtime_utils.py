from __future__ import annotations

import copy
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from pipeline_telemetry import TelemetryRecorder, apply_provider_usage, telemetry_span

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 520}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_runtime_config(path: str | Path) -> dict[str, Any]:
    config = load_json(path)
    base_path = config.get("base_config")
    if not base_path:
        return config
    return deep_merge(load_json(base_path), config)


def request_json(
    *,
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    total_attempts = 3
    last_error: Exception | None = None
    for attempt in range(1, total_attempts + 1):
        request = urllib.request.Request(url=url, data=data, method="POST")
        for key, value in headers.items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"模型请求失败，状态码 {exc.code}，响应：{body}")
            if exc.code in RETRYABLE_HTTP_STATUS_CODES and attempt < total_attempts:
                time.sleep(min(6.0, 1.5 * attempt))
                continue
            raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"模型网络请求失败：{exc}")
            if attempt < total_attempts:
                time.sleep(min(6.0, 1.5 * attempt))
                continue
            raise last_error from exc
    if last_error:
        raise last_error
    raise RuntimeError("模型请求失败：未知错误。")


def extract_openai_text(response: Mapping[str, Any]) -> str:
    texts: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                texts.append(content["text"])
    if texts:
        return "\n".join(texts).strip()
    raise RuntimeError(f"OpenAI 响应中没有 output_text：{response}")


def openai_json_completion(
    *,
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: Mapping[str, Any],
    temperature: float,
    timeout_seconds: int,
    telemetry: TelemetryRecorder | None = None,
    stage: str = "openai_stage",
    step_name: str = "openai_json_completion",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from providers.base import extract_json_from_text

    payload = {
        "model": model,
        "temperature": temperature,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_prompt,
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        },
    }
    with telemetry_span(
        telemetry,
        stage=stage,
        name=step_name,
        provider="openai",
        model=model,
        metadata=metadata,
    ) as step:
        response = request_json(
            url="https://api.openai.com/v1/responses",
            payload=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout_seconds=timeout_seconds,
        )
        apply_provider_usage(step, "openai", response)
        step["metadata"]["temperature"] = temperature
        step["metadata"]["schema_name"] = schema_name
    return extract_json_from_text(extract_openai_text(response))


def configure_openai_api(config: Mapping[str, Any], *, provider_key: str = "openai") -> tuple[str, str]:
    provider_config = dict(config.get("provider", {}).get(provider_key, {}))
    model = str(provider_config.get("model") or config.get("provider", {}).get("model") or "gpt-5.4").strip()
    api_key = str(provider_config.get("api_key") or "").strip()
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        fallback_path = PROJECT_ROOT / "config/video_pipeline.local.json"
        if fallback_path.exists():
            fallback = load_json(fallback_path)
            api_key = str(fallback.get("providers", {}).get("openai", {}).get("api_key", "")).strip()
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY。")
    return model, api_key


def build_episode_ids(series_config: Mapping[str, Any]) -> list[str]:
    prefix = str(series_config.get("episode_id_prefix", "ep"))
    padding = int(series_config.get("episode_id_padding", 2))
    start_episode = int(series_config["start_episode"])
    end_episode = int(series_config["end_episode"])
    return [f"{prefix}{index:0{padding}d}" for index in range(start_episode, end_episode + 1)]


def read_text(path: str | Path | None) -> str:
    if not path:
        return ""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return ""
    return resolved.read_text(encoding="utf-8")
