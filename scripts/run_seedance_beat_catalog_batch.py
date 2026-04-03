from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
TASK_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
DEFAULT_MODEL = "doubao-seedance-2-0-260128"
DEFAULT_RATIO = "9:16"
DEFAULT_RESOLUTION = "480p"
DEFAULT_GENERATE_AUDIO = True
DEFAULT_WATERMARK = False
DEFAULT_MAX_WORKERS = 20
DEFAULT_POLL_INTERVAL = 10
DEFAULT_MAX_POLLS = 120


@dataclass
class BeatPrompt:
    beat_id: str
    title: str
    prompt_text: str
    duration_seconds: float
    source_path: Path


@dataclass
class BeatJob:
    series_name: str
    episode_id: str
    beat: BeatPrompt
    output_dir: Path


@dataclass
class BeatTaskResult:
    beat_id: str
    title: str
    status: str
    task_id: str
    video_url: str
    cover_url: str
    local_video_path: str
    submit_response_path: str
    poll_response_path: str
    task_id_path: str
    error_message: str
    duration_seconds: float


def print_status(message: str) -> None:
    print(f"[seedance-beat-batch] {message}", flush=True)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def episode_sort_key(raw: str) -> tuple[int, str]:
    match = re.search(r"ep(\d+)", str(raw), flags=re.IGNORECASE)
    if not match:
        return (10**9, str(raw))
    return (int(match.group(1)), str(raw))


def beat_sort_key(raw: str) -> tuple[int, str]:
    match = re.search(r"(?:SB|BS|P)(\d+)", str(raw), flags=re.IGNORECASE)
    if not match:
        return (10**9, str(raw))
    return (int(match.group(1)), str(raw))


def parse_beat_index(raw: str, *, arg_name: str) -> int:
    text = str(raw or "").strip()
    match = re.fullmatch(r"(?:SB|BS|P)?(\d+)", text, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError(f"{arg_name} 格式无效：{raw}（支持 SB03 或 3）")
    return int(match.group(1))


def parse_beat_range(
    *,
    beat_start_raw: str | None,
    beat_end_raw: str | None,
    beat_range_raw: str | None,
) -> tuple[int | None, int | None]:
    start: int | None = parse_beat_index(beat_start_raw, arg_name="--beat-start") if beat_start_raw else None
    end: int | None = parse_beat_index(beat_end_raw, arg_name="--beat-end") if beat_end_raw else None

    if beat_range_raw:
        range_text = str(beat_range_raw).strip()
        full_match = re.fullmatch(
            r"\s*(?:SB|BS|P)?(\d+)\s*[-:~]\s*(?:SB|BS|P)?(\d+)\s*",
            range_text,
            flags=re.IGNORECASE,
        )
        left_open_match = re.fullmatch(r"\s*(?:SB|BS|P)?(\d+)\s*[-:~]\s*", range_text, flags=re.IGNORECASE)
        right_open_match = re.fullmatch(r"\s*[-:~]\s*(?:SB|BS|P)?(\d+)\s*", range_text, flags=re.IGNORECASE)
        single_match = re.fullmatch(r"\s*(?:SB|BS|P)?(\d+)\s*", range_text, flags=re.IGNORECASE)
        if full_match:
            if start is None:
                start = int(full_match.group(1))
            if end is None:
                end = int(full_match.group(2))
        elif left_open_match:
            if start is None:
                start = int(left_open_match.group(1))
        elif right_open_match:
            if end is None:
                end = int(right_open_match.group(1))
        elif single_match:
            point = int(single_match.group(1))
            if start is None:
                start = point
            if end is None:
                end = point
        else:
            raise RuntimeError(
                f"--beat-range 格式无效：{beat_range_raw}（示例：SB03-SB08、SB03-、-SB08 或 SB08）"
            )

    if start is not None and end is not None and start > end:
        raise RuntimeError(f"beat 区间无效：起始 SB{start:02d} 大于结束 SB{end:02d}")
    return start, end


def filter_beats(beats: list[BeatPrompt], *, beat_start: int | None, beat_end: int | None) -> list[BeatPrompt]:
    selected: list[BeatPrompt] = []
    for beat in beats:
        beat_index = beat_sort_key(beat.beat_id)[0]
        if beat_index >= 10**9:
            continue
        if beat_start is not None and beat_index < beat_start:
            continue
        if beat_end is not None and beat_index > beat_end:
            continue
        selected.append(beat)
    return selected


def prompt_beat_range(beats: list[BeatPrompt]) -> tuple[int | None, int | None]:
    if not beats:
        return None, None
    options = [f"{beat.beat_id}｜{beat.title}" for beat in beats]
    start_idx, end_idx = choose_range_from_list(
        "请选择要提交的 beat 范围：",
        options,
        default_start=0,
        default_end=len(options) - 1,
    )
    return beat_sort_key(beats[start_idx].beat_id)[0], beat_sort_key(beats[end_idx].beat_id)[0]


def choose_from_list(title: str, options: list[str], default_index: int = 0) -> int:
    if not options:
        raise RuntimeError(f"没有可选项：{title}")
    print(title)
    for index, option in enumerate(options, start=1):
        suffix = "  [默认]" if index - 1 == default_index else ""
        print(f"  {index}. {option}{suffix}")
    raw = input(f"请输入序号（默认 {default_index + 1}）：").strip()
    if not raw:
        return default_index
    if not raw.isdigit():
        raise RuntimeError(f"输入无效：{raw}")
    selected = int(raw) - 1
    if selected < 0 or selected >= len(options):
        raise RuntimeError(f"输入超出范围：{raw}")
    return selected


def choose_range_from_list(
    title: str,
    options: list[str],
    *,
    default_start: int = 0,
    default_end: int | None = None,
) -> tuple[int, int]:
    if not options:
        raise RuntimeError(f"没有可选项：{title}")
    if default_end is None:
        default_end = len(options) - 1
    print(title)
    for index, option in enumerate(options, start=1):
        start_suffix = " [默认起点]" if index - 1 == default_start else ""
        end_suffix = " [默认终点]" if index - 1 == default_end else ""
        print(f"  {index}. {option}{start_suffix}{end_suffix}")
    raw_start = input(f"请输入起始序号（默认 {default_start + 1}）：").strip()
    raw_end = input(f"请输入结束序号（默认 {default_end + 1}）：").strip()
    start = default_start if not raw_start else int(raw_start) - 1
    end = default_end if not raw_end else int(raw_end) - 1
    if start < 0 or end < 0 or start >= len(options) or end >= len(options):
        raise RuntimeError("起止序号超出范围。")
    if start > end:
        raise RuntimeError("起始序号不能大于结束序号。")
    return start, end


def list_series_dirs() -> list[Path]:
    if not ANALYSIS_ROOT.exists():
        return []
    result: list[Path] = []
    for child in sorted(ANALYSIS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("ep*/seedance_beat_catalog.json")) or any(child.glob("ep*/seedance_beat_catalog.md")):
            result.append(child)
    return result


def list_episode_dirs(series_dir: Path) -> list[Path]:
    candidates = [
        path
        for path in series_dir.iterdir()
        if path.is_dir() and ((path / "seedance_beat_catalog.json").exists() or (path / "seedance_beat_catalog.md").exists())
    ]
    return sorted(candidates, key=lambda item: episode_sort_key(item.name))


def normalize_duration_seconds(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 10.0
    return max(5.0, value)


def api_duration_from_seconds(raw: Any) -> int:
    value = normalize_duration_seconds(raw)
    return max(5, min(60, int(math.ceil(value))))


def load_beat_catalog_json(path: Path) -> list[BeatPrompt]:
    data = json.loads(path.read_text(encoding="utf-8"))
    beats = list(data.get("beats") or [])
    result: list[BeatPrompt] = []
    for item in beats:
        if not isinstance(item, dict):
            continue
        beat_id = str(item.get("beat_id") or "").strip()
        title = str(item.get("display_title") or item.get("beat_summary") or beat_id).strip()
        prompt_text = str(item.get("restored_seedance_prompt") or "").strip()
        if not beat_id or not prompt_text:
            continue
        result.append(
            BeatPrompt(
                beat_id=beat_id,
                title=title or beat_id,
                prompt_text=prompt_text,
                duration_seconds=normalize_duration_seconds(item.get("restored_duration_seconds") or item.get("duration_seconds")),
                source_path=path,
            )
        )
    return sorted(result, key=lambda item: beat_sort_key(item.beat_id))


def load_beat_catalog_markdown(path: Path) -> list[BeatPrompt]:
    text = path.read_text(encoding="utf-8")
    section_pattern = re.compile(r"^##\s+(SB\d+)\s*[｜|]\s*(.+?)\n", flags=re.MULTILINE)
    matches = list(section_pattern.finditer(text))
    result: list[BeatPrompt] = []
    for index, match in enumerate(matches):
        beat_id = match.group(1).strip()
        title = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[start:end]
        duration_match = re.search(r"-\s*建议还原时长：\s*([0-9.]+)\s*秒", section)
        prompt_match = re.search(r"###\s*还原版 Prompt\s*\n+(.+?)(?:\n###\s|\Z)", section, flags=re.S)
        prompt_text = prompt_match.group(1).strip() if prompt_match else ""
        if not prompt_text:
            continue
        result.append(
            BeatPrompt(
                beat_id=beat_id,
                title=title or beat_id,
                prompt_text=prompt_text,
                duration_seconds=normalize_duration_seconds(duration_match.group(1) if duration_match else None),
                source_path=path,
            )
        )
    return sorted(result, key=lambda item: beat_sort_key(item.beat_id))


def load_episode_beat_prompts(episode_dir: Path) -> list[BeatPrompt]:
    json_path = episode_dir / "seedance_beat_catalog.json"
    if json_path.exists():
        beats = load_beat_catalog_json(json_path)
        if beats:
            return beats
    md_path = episode_dir / "seedance_beat_catalog.md"
    if md_path.exists():
        beats = load_beat_catalog_markdown(md_path)
        if beats:
            return beats
    raise RuntimeError(f"{episode_dir} 下未解析到任何 beat prompt。")


def build_payload(beat: BeatPrompt) -> dict[str, Any]:
    return {
        "model": os.environ.get("SEEDANCE_MODEL", DEFAULT_MODEL),
        "content": [
            {
                "type": "text",
                "text": beat.prompt_text,
            }
        ],
        "generate_audio": str(os.environ.get("SEEDANCE_GENERATE_AUDIO", str(DEFAULT_GENERATE_AUDIO).lower())).strip().lower() == "true",
        "ratio": os.environ.get("SEEDANCE_RATIO", DEFAULT_RATIO),
        "resolution": os.environ.get("SEEDANCE_RESOLUTION", DEFAULT_RESOLUTION),
        "duration": api_duration_from_seconds(beat.duration_seconds),
        "watermark": str(os.environ.get("SEEDANCE_WATERMARK", str(DEFAULT_WATERMARK).lower())).strip().lower() == "true",
    }


def request_json(url: str, *, api_key: str, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"请求失败：{exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"接口返回非 JSON：{body[:400]}") from exc


def download_file(url: str, output_path: Path) -> None:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=120) as response:
        output_path.write_bytes(response.read())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_existing_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_single_beat(job: BeatJob, *, api_key: str, force: bool) -> BeatTaskResult:
    started = time.time()
    beat_dir = job.output_dir / job.beat.beat_id
    beat_dir.mkdir(parents=True, exist_ok=True)

    payload_path = beat_dir / f"{job.beat.beat_id}__seedance_payload.json"
    submit_response_path = beat_dir / f"{job.beat.beat_id}__seedance_submit_response.json"
    poll_response_path = beat_dir / f"{job.beat.beat_id}__seedance_poll_response.json"
    task_id_path = beat_dir / f"{job.beat.beat_id}__seedance_task_id.txt"
    output_video_path = beat_dir / f"{job.beat.beat_id}__seedance_output.mp4"
    output_cover_path = beat_dir / f"{job.beat.beat_id}__seedance_last_frame.jpg"

    payload = build_payload(job.beat)
    write_json(payload_path, payload)

    task_id = ""
    if not force and task_id_path.exists():
        task_id = task_id_path.read_text(encoding="utf-8").strip()

    submit_response: dict[str, Any] | None = None
    if not task_id:
        print_status(f"{job.beat.beat_id}｜{job.beat.title}：开始提交任务。")
        submit_response = request_json(TASK_URL, api_key=api_key, method="POST", payload=payload)
        write_json(submit_response_path, submit_response)
        task_id = str(submit_response.get("id") or "").strip()
        if not task_id:
            raise RuntimeError(f"{job.beat.beat_id} 提交成功但未返回任务 ID：{submit_response}")
        task_id_path.write_text(task_id, encoding="utf-8")
        print_status(f"{job.beat.beat_id}｜{job.beat.title}：提交成功，task_id={task_id}")
    else:
        submit_response = load_existing_json(submit_response_path)
        print_status(f"{job.beat.beat_id}｜{job.beat.title}：复用已有 task_id={task_id}，开始继续轮询。")

    poll_interval = int(os.environ.get("SEEDANCE_POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL)))
    max_polls = int(os.environ.get("SEEDANCE_MAX_POLLS", str(DEFAULT_MAX_POLLS)))
    task_query_url = f"{TASK_URL}/{task_id}"

    last_payload: dict[str, Any] = {}
    for poll_index in range(1, max_polls + 1):
        last_payload = request_json(task_query_url, api_key=api_key, method="GET")
        write_json(poll_response_path, last_payload)
        status = str(last_payload.get("status") or "").strip()
        print_status(
            f"{job.beat.beat_id}｜{job.beat.title}：轮询 {poll_index}/{max_polls}，状态={status or 'unknown'}"
        )
        if status == "succeeded":
            content = dict(last_payload.get("content") or {})
            video_url = str(content.get("video_url") or "").strip()
            cover_url = str(content.get("last_frame_url") or "").strip()
            if video_url:
                try:
                    download_file(video_url, output_video_path)
                except Exception:
                    pass
            if cover_url:
                try:
                    download_file(cover_url, output_cover_path)
                except Exception:
                    pass
            return BeatTaskResult(
                beat_id=job.beat.beat_id,
                title=job.beat.title,
                status=status,
                task_id=task_id,
                video_url=video_url,
                cover_url=cover_url,
                local_video_path=str(output_video_path) if output_video_path.exists() else "",
                submit_response_path=str(submit_response_path),
                poll_response_path=str(poll_response_path),
                task_id_path=str(task_id_path),
                error_message="",
                duration_seconds=time.time() - started,
            )
        if status in {"failed", "expired"}:
            error = dict(last_payload.get("error") or {})
            return BeatTaskResult(
                beat_id=job.beat.beat_id,
                title=job.beat.title,
                status=status,
                task_id=task_id,
                video_url="",
                cover_url="",
                local_video_path="",
                submit_response_path=str(submit_response_path),
                poll_response_path=str(poll_response_path),
                task_id_path=str(task_id_path),
                error_message=str(error.get("message") or last_payload),
                duration_seconds=time.time() - started,
            )
        time.sleep(poll_interval)

    return BeatTaskResult(
        beat_id=job.beat.beat_id,
        title=job.beat.title,
        status="timeout",
        task_id=task_id,
        video_url="",
        cover_url="",
        local_video_path="",
        submit_response_path=str(submit_response_path),
        poll_response_path=str(poll_response_path),
        task_id_path=str(task_id_path),
        error_message=f"轮询超时，已达到 {max_polls} 次。",
        duration_seconds=time.time() - started,
    )


def render_markdown_report(
    *,
    series_name: str,
    episode_id: str,
    catalog_path: Path,
    output_dir: Path,
    results: list[BeatTaskResult],
) -> str:
    succeeded = [item for item in results if item.status == "succeeded"]
    failed = [item for item in results if item.status != "succeeded"]
    lines = [
        f"# Seedance Beat Batch Submit：{series_name} {episode_id}",
        "",
        f"- 生成时间：{now_iso()}",
        f"- 来源 catalog：`{catalog_path}`",
        f"- 输出目录：`{output_dir}`",
        f"- 总 beat 数：{len(results)}",
        f"- 成功：{len(succeeded)}",
        f"- 非成功：{len(failed)}",
        "",
    ]
    for item in results:
        lines.extend(
            [
                f"## {item.beat_id}｜{item.title}",
                "",
                f"- 状态：`{item.status}`",
                f"- 任务 ID：`{item.task_id or '未返回'}`",
                f"- 视频链接：{item.video_url or '无'}",
                f"- 封面链接：{item.cover_url or '无'}",
                f"- 本地视频：`{item.local_video_path or '无'}`",
                f"- 提交响应：`{item.submit_response_path}`",
                f"- 轮询响应：`{item.poll_response_path}`",
                f"- 耗时：`{item.duration_seconds:.1f}s`",
            ]
        )
        if item.error_message:
            lines.append(f"- 错误：`{item.error_message}`")
        lines.append("")
    return "\n".join(lines)


def choose_series_and_episode(
    *,
    series_name: str | None,
    episode_id: str | None,
) -> tuple[Path, Path]:
    series_dirs = list_series_dirs()
    if not series_dirs:
        raise RuntimeError("analysis/ 下没有找到任何带 seedance_beat_catalog 的剧。")

    selected_series_dir: Path
    if series_name:
        selected_series_dir = ANALYSIS_ROOT / series_name
        if selected_series_dir not in series_dirs:
            raise RuntimeError(f"未找到剧：{series_name}")
    else:
        series_idx = choose_from_list(
            "请选择要提交 Seedance beat 视频的剧：",
            [path.name for path in series_dirs],
            default_index=0,
        )
        selected_series_dir = series_dirs[series_idx]

    episode_dirs = list_episode_dirs(selected_series_dir)
    if not episode_dirs:
        raise RuntimeError(f"{selected_series_dir.name} 下没有任何可用集数。")

    selected_episode_dir: Path
    if episode_id:
        selected_episode_dir = selected_series_dir / episode_id
        if selected_episode_dir not in episode_dirs:
            raise RuntimeError(f"{selected_series_dir.name} 下未找到集数：{episode_id}")
    else:
        episode_idx = choose_from_list(
            f"请选择 {selected_series_dir.name} 的集数：",
            [path.name for path in episode_dirs],
            default_index=0,
        )
        selected_episode_dir = episode_dirs[episode_idx]

    return selected_series_dir, selected_episode_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 analysis/<剧名>/<epXX>/seedance_beat_catalog.* 批量提交 Seedance 视频任务。")
    parser.add_argument("--series", help="剧名目录，位于 analysis/<剧名>/")
    parser.add_argument("--episode", help="集数目录，例如 ep01")
    parser.add_argument("--beat-start", help="起始 beat，例如 SB03")
    parser.add_argument("--beat-end", help="结束 beat，例如 SB08")
    parser.add_argument("--beat-range", help="beat 范围，例如 SB03-SB08 / SB03- / -SB08 / SB08")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="并发提交/轮询的最大 worker 数")
    parser.add_argument("--force", action="store_true", help="忽略已有 task_id，强制重新提交")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = str(os.environ.get("ARK_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("请先设置 ARK_API_KEY，例如 export ARK_API_KEY=...")

    series_dir, episode_dir = choose_series_and_episode(series_name=args.series, episode_id=args.episode)
    beats = load_episode_beat_prompts(episode_dir)
    if not beats:
        raise RuntimeError(f"{episode_dir} 未找到可提交的 beat。")
    beat_start, beat_end = parse_beat_range(
        beat_start_raw=args.beat_start,
        beat_end_raw=args.beat_end,
        beat_range_raw=args.beat_range,
    )
    if args.beat_start or args.beat_end or args.beat_range:
        beats = filter_beats(beats, beat_start=beat_start, beat_end=beat_end)
    else:
        beat_start, beat_end = prompt_beat_range(beats)
        beats = filter_beats(beats, beat_start=beat_start, beat_end=beat_end)
    if not beats:
        raise RuntimeError("按当前 beat 范围没有筛到任何可提交项。")

    catalog_path = episode_dir / "seedance_beat_catalog.json"
    if not catalog_path.exists():
        catalog_path = episode_dir / "seedance_beat_catalog.md"

    output_dir = episode_dir / "seedance_beat_api"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = timestamp_slug()
    report_md_path = output_dir / f"seedance_beat_batch_submit__{run_id}.md"
    report_json_path = output_dir / f"seedance_beat_batch_submit__{run_id}.json"

    print_status(f"开始提交：{series_dir.name} {episode_dir.name}，共 {len(beats)} 个 beat。")
    jobs = [
        BeatJob(
            series_name=series_dir.name,
            episode_id=episode_dir.name,
            beat=beat,
            output_dir=output_dir,
        )
        for beat in beats
    ]

    results: list[BeatTaskResult] = []
    max_workers = max(1, int(args.max_workers or DEFAULT_MAX_WORKERS))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_single_beat, job, api_key=api_key, force=bool(args.force)): job
            for job in jobs
        }
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = BeatTaskResult(
                    beat_id=job.beat.beat_id,
                    title=job.beat.title,
                    status="runner_error",
                    task_id="",
                    video_url="",
                    cover_url="",
                    local_video_path="",
                    submit_response_path="",
                    poll_response_path="",
                    task_id_path="",
                    error_message=str(exc),
                    duration_seconds=0.0,
                )
            results.append(result)
            print_status(
                f"{result.beat_id}｜{result.title} -> {result.status}"
                + (f" | {result.video_url}" if result.video_url else "")
            )

    results.sort(key=lambda item: beat_sort_key(item.beat_id))
    report_payload = {
        "series_name": series_dir.name,
        "episode_id": episode_dir.name,
        "catalog_path": str(catalog_path),
        "output_dir": str(output_dir),
        "generated_at": now_iso(),
        "results": [result.__dict__ for result in results],
    }
    report_json_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md_path.write_text(
        render_markdown_report(
            series_name=series_dir.name,
            episode_id=episode_dir.name,
            catalog_path=catalog_path,
            output_dir=output_dir,
            results=results,
        ),
        encoding="utf-8",
    )

    print()
    print("视频下载链接：")
    for item in results:
        if item.video_url:
            print(f"- {item.beat_id}｜{item.title}: {item.video_url}")
    print()
    print(f"批量报告：{report_md_path}")
    print(f"结构化结果：{report_json_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_status("已中断。")
        raise
    except Exception as exc:
        print_status(f"失败：{exc}")
        sys.exit(1)
