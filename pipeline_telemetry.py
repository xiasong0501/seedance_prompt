from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json_value(item) for item in value]
    return str(value)


def extract_provider_usage(provider: str, response: Mapping[str, Any] | None) -> dict[str, Any]:
    if not response:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "raw_usage": {},
        }

    normalized = provider.strip().lower()
    if normalized in {"openai", "qwen"}:
        usage = dict(response.get("usage", {}))
        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "raw_usage": _safe_json_value(usage),
        }

    if normalized == "gemini":
        usage = dict(response.get("usageMetadata", {}))
        input_tokens = int(usage.get("promptTokenCount") or 0)
        output_tokens = int(
            usage.get("candidatesTokenCount")
            or max(int(usage.get("totalTokenCount") or 0) - input_tokens, 0)
        )
        total_tokens = int(usage.get("totalTokenCount") or (input_tokens + output_tokens))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "raw_usage": _safe_json_value(usage),
        }

    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "raw_usage": {},
    }


def apply_provider_usage(step: dict[str, Any], provider: str, response: Mapping[str, Any] | None) -> None:
    usage = extract_provider_usage(provider, response)
    step["input_tokens"] = int(usage["input_tokens"])
    step["output_tokens"] = int(usage["output_tokens"])
    step["total_tokens"] = int(usage["total_tokens"])
    if usage["raw_usage"]:
        step.setdefault("metadata", {})
        step["metadata"]["raw_usage"] = usage["raw_usage"]


@dataclass
class TelemetryRecorder:
    run_name: str
    context: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None

    @contextmanager
    def span(
        self,
        *,
        stage: str,
        name: str,
        provider: str | None = None,
        model: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        started_at = utc_now_iso()
        started_clock = time.perf_counter()
        step = {
            "step_id": f"step-{len(self.steps) + 1:04d}",
            "stage": stage,
            "name": name,
            "status": "running",
            "provider": provider or "",
            "model": model or "",
            "started_at": started_at,
            "ended_at": None,
            "duration_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "metadata": _safe_json_value(dict(metadata or {})),
        }
        try:
            yield step
        except Exception as exc:
            step["status"] = "failed"
            step["error_type"] = type(exc).__name__
            step["error_message"] = str(exc)
            raise
        finally:
            ended_at = utc_now_iso()
            step["ended_at"] = ended_at
            step["duration_seconds"] = round(time.perf_counter() - started_clock, 3)
            if step.get("status") == "running":
                step["status"] = "completed"
            step["metadata"] = _safe_json_value(step.get("metadata", {}))
            self.steps.append(step)

    def stage_totals(self) -> dict[str, dict[str, Any]]:
        totals: dict[str, dict[str, Any]] = {}
        for step in self.steps:
            bucket = totals.setdefault(
                step["stage"],
                {
                    "step_count": 0,
                    "duration_seconds": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "statuses": {},
                },
            )
            bucket["step_count"] += 1
            bucket["duration_seconds"] = round(bucket["duration_seconds"] + float(step["duration_seconds"]), 3)
            bucket["input_tokens"] += int(step["input_tokens"])
            bucket["output_tokens"] += int(step["output_tokens"])
            bucket["total_tokens"] += int(step["total_tokens"])
            status = str(step["status"])
            bucket["statuses"][status] = int(bucket["statuses"].get(status, 0)) + 1
        return totals

    def totals(self) -> dict[str, Any]:
        return {
            "step_count": len(self.steps),
            "duration_seconds": round(sum(float(step["duration_seconds"]) for step in self.steps), 3),
            "input_tokens": sum(int(step["input_tokens"]) for step in self.steps),
            "output_tokens": sum(int(step["output_tokens"]) for step in self.steps),
            "total_tokens": sum(int(step["total_tokens"]) for step in self.steps),
        }

    def finalize(self) -> None:
        self.finished_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        self.finalize()
        return {
            "run_name": self.run_name,
            "context": _safe_json_value(self.context),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "totals": self.totals(),
            "stage_totals": self.stage_totals(),
            "steps": self.steps,
        }

    def save_json(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target


@contextmanager
def telemetry_span(
    recorder: TelemetryRecorder | None,
    *,
    stage: str,
    name: str,
    provider: str | None = None,
    model: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    if recorder is None:
        yield {
            "stage": stage,
            "name": name,
            "status": "completed",
            "provider": provider or "",
            "model": model or "",
            "metadata": _safe_json_value(dict(metadata or {})),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        return
    with recorder.span(
        stage=stage,
        name=name,
        provider=provider,
        model=model,
        metadata=metadata,
    ) as step:
        yield step
