from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_MAX_WORKERS = 20


@dataclass
class ScriptJob:
    series_name: str
    episode_id: str
    scene_id: str
    scene_title: str
    script_path: Path
    episode_dir: Path
    log_path: Path


@dataclass
class TaskResult:
    job: ScriptJob
    return_code: int
    started_at: str
    finished_at: str
    duration_seconds: float
    task_id: str
    final_status: str
    video_url: str
    last_frame_url: str
    local_video_path: str
    local_last_frame_path: str
    submit_response_path: str
    poll_response_path: str
    task_id_path: str
    check_script_path: str
    references_env_path: str
    log_path: str
    error_code: str
    error_message: str


def is_success_result(result: TaskResult) -> bool:
    return result.final_status == "succeeded" and result.return_code == 0


def is_not_ready_result(result: TaskResult) -> bool:
    return result.final_status == "not_ready"


def is_failed_result(result: TaskResult) -> bool:
    if is_not_ready_result(result):
        return False
    return result.final_status in {"failed", "expired", "runner_error"} or result.return_code != 0


def print_status(message: str) -> None:
    print(f"[seedance-batch] {message}", flush=True)


def now_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def episode_sort_key(raw: str) -> tuple[int, str]:
    match = re.search(r"ep(\d+)", raw, flags=re.IGNORECASE)
    if not match:
        return (10**9, raw)
    return (int(match.group(1)), raw)


def scene_sort_key(raw: str) -> tuple[int, str]:
    match = re.search(r"(?:^|[^A-Za-z])P?(\d+)(?:$|[^0-9])", raw, flags=re.IGNORECASE)
    if not match:
        return (10**9, raw)
    return (int(match.group(1)), raw)


def parse_scene_index(raw: str, *, arg_name: str) -> int:
    text = str(raw or "").strip()
    match = re.fullmatch(r"[Pp]?(\d+)", text)
    if not match:
        raise RuntimeError(f"{arg_name} 格式无效：{raw}（支持 P03 或 3）")
    return int(match.group(1))


def parse_scene_range(
    *,
    p_start_raw: str | None,
    p_end_raw: str | None,
    p_range_raw: str | None,
) -> tuple[int | None, int | None]:
    start: int | None = parse_scene_index(p_start_raw, arg_name="--p-start") if p_start_raw else None
    end: int | None = parse_scene_index(p_end_raw, arg_name="--p-end") if p_end_raw else None

    if p_range_raw:
        range_text = str(p_range_raw).strip()
        full_match = re.fullmatch(r"\s*[Pp]?(\d+)\s*[-:~]\s*[Pp]?(\d+)\s*", range_text)
        left_open_match = re.fullmatch(r"\s*[Pp]?(\d+)\s*[-:~]\s*", range_text)
        right_open_match = re.fullmatch(r"\s*[-:~]\s*[Pp]?(\d+)\s*", range_text)
        single_match = re.fullmatch(r"\s*[Pp]?(\d+)\s*", range_text)

        if full_match:
            range_start = int(full_match.group(1))
            range_end = int(full_match.group(2))
            if start is None:
                start = range_start
            if end is None:
                end = range_end
        elif left_open_match:
            range_start = int(left_open_match.group(1))
            if start is None:
                start = range_start
        elif right_open_match:
            range_end = int(right_open_match.group(1))
            if end is None:
                end = range_end
        elif single_match:
            point = int(single_match.group(1))
            if start is None:
                start = point
            if end is None:
                end = point
        else:
            raise RuntimeError(
                f"--p-range 格式无效：{p_range_raw}（示例：P03-P08、P03-、-P08 或 P08）"
            )

    if start is not None and end is not None and start > end:
        raise RuntimeError(f"P 区间无效：起始 P{start:02d} 大于结束 P{end:02d}")
    return start, end


def normalize_episode_id(raw: str) -> str:
    text = str(raw or "").strip().lower()
    match = re.fullmatch(r"ep(\d+)", text)
    if match:
        return f"ep{int(match.group(1)):02d}"
    if text.isdigit():
        return f"ep{int(text):02d}"
    raise RuntimeError(f"--episode 格式无效：{raw}（示例：ep01 或 1）")


def filter_jobs(
    jobs: list[ScriptJob],
    *,
    episode_filter: str | None,
    p_start: int | None,
    p_end: int | None,
) -> list[ScriptJob]:
    selected: list[ScriptJob] = []
    for job in jobs:
        if episode_filter and job.episode_id.lower() != episode_filter.lower():
            continue
        scene_index = scene_sort_key(job.scene_id)[0]
        if scene_index >= 10**9:
            continue
        if p_start is not None and scene_index < p_start:
            continue
        if p_end is not None and scene_index > p_end:
            continue
        selected.append(job)
    return selected


def prompt_episode_filter(jobs: list[ScriptJob]) -> str | None:
    episode_ids = sorted({job.episode_id for job in jobs}, key=episode_sort_key)
    options = ["全部 episode"] + episode_ids
    selected_index = choose_from_list(
        "请选择要执行的集数：",
        options,
        default_index=0,
    )
    if selected_index == 0:
        return None
    return options[selected_index]


def prompt_scene_range_filter() -> tuple[int | None, int | None]:
    print("请输入分镜范围（回车=全部，示例：P03-P12 / P03- / -P12 / P08）")
    while True:
        raw = input("分镜范围：").strip()
        if not raw:
            return None, None
        try:
            return parse_scene_range(
                p_start_raw=None,
                p_end_raw=None,
                p_range_raw=raw,
            )
        except RuntimeError as exc:
            print(f"输入无效：{exc}")


def list_series_dirs() -> list[Path]:
    if not OUTPUTS_ROOT.exists():
        return []
    result: list[Path] = []
    for child in sorted(OUTPUTS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("ep*/P*__seedance_api.sh")) or any(child.glob("ep*/[0-9]*__seedance_api.sh")):
            result.append(child)
    return result


def parse_scene_title(script_path: Path) -> str:
    try:
        text = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return script_path.stem
    match = re.search(r"^#\s*场景：\s*(.+)$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return script_path.stem.replace("__seedance_api", "")


def build_jobs(series_dir: Path, run_id: str) -> list[ScriptJob]:
    jobs: list[ScriptJob] = []
    for episode_dir in sorted(
        [item for item in series_dir.iterdir() if item.is_dir()],
        key=lambda item: episode_sort_key(item.name),
    ):
        matched: dict[str, Path] = {}
        for pattern in ("P*__seedance_api.sh", "[0-9]*__seedance_api.sh"):
            for script_path in episode_dir.glob(pattern):
                matched[script_path.name] = script_path
        script_paths = sorted(matched.values(), key=lambda item: scene_sort_key(item.stem))
        for script_path in script_paths:
            scene_id = script_path.name.split("__", 1)[0]
            jobs.append(
                ScriptJob(
                    series_name=series_dir.name,
                    episode_id=episode_dir.name,
                    scene_id=scene_id,
                    scene_title=parse_scene_title(script_path),
                    script_path=script_path.resolve(),
                    episode_dir=episode_dir.resolve(),
                    log_path=(episode_dir / f"{scene_id}__seedance_batch_run__{run_id}.log").resolve(),
                )
            )
    return jobs


def safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def derive_related_paths(job: ScriptJob) -> dict[str, Path]:
    prefix = job.scene_id
    episode_dir = job.episode_dir
    return {
        "submit_response": episode_dir / f"{prefix}__seedance_submit_response.json",
        "poll_response": episode_dir / f"{prefix}__seedance_poll_response.json",
        "task_id": episode_dir / f"{prefix}__seedance_task_id.txt",
        "check_script": episode_dir / f"{prefix}__seedance_check_last_task.sh",
        "references_env": episode_dir / f"{prefix}__seedance_api_urls.env",
        "local_video": episode_dir / f"{prefix}__seedance_output.mp4",
        "local_last_frame": episode_dir / f"{prefix}__seedance_last_frame.jpg",
    }


def parse_required_reference_vars(script_path: Path) -> list[str]:
    try:
        text = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return re.findall(r"""\$\{(REF_IMAGE_\d+_URL):\?""", text)


def parse_env_exports(env_path: Path) -> dict[str, str]:
    exports: dict[str, str] = {}
    if not env_path.exists():
        return exports
    for raw_line in safe_read_text(env_path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith("export "):
            continue
        body = line[len("export ") :]
        if "=" not in body:
            continue
        key, value = body.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            exports[key] = value
    return exports


def build_skip_result(
    *,
    job: ScriptJob,
    references_env_path: Path,
    error_code: str,
    error_message: str,
) -> TaskResult:
    timestamp = now_label()
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.log_path.write_text(error_message.rstrip() + "\n", encoding="utf-8")
    return TaskResult(
        job=job,
        return_code=0,
        started_at=timestamp,
        finished_at=timestamp,
        duration_seconds=0.0,
        task_id="",
        final_status="not_ready",
        video_url="",
        last_frame_url="",
        local_video_path="",
        local_last_frame_path="",
        submit_response_path="",
        poll_response_path="",
        task_id_path="",
        check_script_path="",
        references_env_path=str(references_env_path.resolve()) if references_env_path.exists() else "",
        log_path=str(job.log_path),
        error_code=error_code,
        error_message=error_message,
    )


def preflight_job(job: ScriptJob) -> TaskResult | None:
    related = derive_related_paths(job)
    references_env_path = related["references_env"]
    required_vars = parse_required_reference_vars(job.script_path)
    if not required_vars:
        return None
    if not references_env_path.exists():
        return build_skip_result(
            job=job,
            references_env_path=references_env_path,
            error_code="references_env_missing",
            error_message=(
                f"引用 env 不存在：{references_env_path}。"
                " 先运行 run_upload_seedance_refs.sh 生成完整引用，再进行批量提交。"
            ),
        )
    exports = parse_env_exports(references_env_path)
    missing_vars = [name for name in required_vars if not str(exports.get(name) or "").strip()]
    if missing_vars:
        return build_skip_result(
            job=job,
            references_env_path=references_env_path,
            error_code="references_env_incomplete",
            error_message=(
                f"引用 env 不完整：缺少 {', '.join(missing_vars)}。"
                f" env={references_env_path}"
            ),
        )
    return None


def parse_result(job: ScriptJob, return_code: int, started_at: str, finished_at: str, duration_seconds: float) -> TaskResult:
    related = derive_related_paths(job)
    submit_payload = safe_read_json(related["submit_response"])
    poll_payload = safe_read_json(related["poll_response"])
    task_id = ""
    if related["task_id"].exists():
        task_id = safe_read_text(related["task_id"]).strip()
    if not task_id:
        task_id = str(submit_payload.get("id") or poll_payload.get("id") or "").strip()

    final_status = str(poll_payload.get("status") or "").strip()
    if not final_status:
        final_status = "succeeded" if return_code == 0 else "unknown"

    content = poll_payload.get("content") or {}
    error_payload = poll_payload.get("error") or submit_payload.get("error") or {}
    video_url = str(content.get("video_url") or "").strip()
    last_frame_url = str(content.get("last_frame_url") or "").strip()
    error_code = str(error_payload.get("code") or "").strip()
    error_message = str(error_payload.get("message") or "").strip()

    if not error_message and return_code != 0:
        log_excerpt = safe_read_text(job.log_path).strip().splitlines()
        if log_excerpt:
            error_message = log_excerpt[-1].strip()

    return TaskResult(
        job=job,
        return_code=return_code,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=round(duration_seconds, 2),
        task_id=task_id,
        final_status=final_status or "unknown",
        video_url=video_url,
        last_frame_url=last_frame_url,
        local_video_path=str(related["local_video"].resolve()) if related["local_video"].exists() else "",
        local_last_frame_path=str(related["local_last_frame"].resolve()) if related["local_last_frame"].exists() else "",
        submit_response_path=str(related["submit_response"].resolve()) if related["submit_response"].exists() else "",
        poll_response_path=str(related["poll_response"].resolve()) if related["poll_response"].exists() else "",
        task_id_path=str(related["task_id"].resolve()) if related["task_id"].exists() else "",
        check_script_path=str(related["check_script"].resolve()) if related["check_script"].exists() else "",
        references_env_path=str(related["references_env"].resolve()) if related["references_env"].exists() else "",
        log_path=str(job.log_path),
        error_code=error_code,
        error_message=error_message,
    )


def run_job(job: ScriptJob) -> TaskResult:
    preflight_result = preflight_job(job)
    if preflight_result is not None:
        return preflight_result

    env = os.environ.copy()
    env.setdefault("SEEDANCE_RESOLUTION", "480p")
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    started_wall = time.time()
    started_at = now_label()
    with job.log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            ["bash", str(job.script_path)],
            cwd=str(job.episode_dir),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    finished_at = now_label()
    return parse_result(
        job=job,
        return_code=process.returncode,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=time.time() - started_wall,
    )


def markdown_link(label: str, target: str) -> str:
    if not target:
        return "-"
    return f"[{label}]({target})"


def render_task_detail(result: TaskResult | None, job: ScriptJob) -> str:
    if result is None:
        return textwrap.dedent(
            f"""\
            ### {job.scene_id} {job.scene_title}

            - 状态：等待执行
            - API 脚本：`{job.script_path}`
            - 运行日志：`{job.log_path}`
            """
        ).strip()

    lines = [
        f"### {job.scene_id} {job.scene_title}",
        "",
        f"- shell 退出码：`{result.return_code}`",
        f"- task id：`{result.task_id or '未取到'}`",
        f"- 最终轮询状态：`{result.final_status}`",
        f"- 开始时间：`{result.started_at}`",
        f"- 结束时间：`{result.finished_at}`",
        f"- 总耗时：`{result.duration_seconds}s`",
        f"- 视频下载链接：{result.video_url or '-'}",
        f"- 尾帧下载链接：{result.last_frame_url or '-'}",
        f"- 本地视频：`{result.local_video_path or '-'}`",
        f"- 本地尾帧：`{result.local_last_frame_path or '-'}`",
        f"- 提交响应：`{result.submit_response_path or '-'}`",
        f"- 轮询响应：`{result.poll_response_path or '-'}`",
        f"- task id 文件：`{result.task_id_path or '-'}`",
        f"- 手动查验脚本：`{result.check_script_path or '-'}`",
        f"- 引用 env：`{result.references_env_path or '-'}`",
        f"- 运行日志：`{result.log_path}`",
    ]
    if result.error_code or result.error_message:
        lines.append(f"- 错误码：`{result.error_code or '-'}`")
        lines.append(f"- 错误信息：{result.error_message or '-'}")
    return "\n".join(lines)


def render_episode_report(
    *,
    series_name: str,
    episode_id: str,
    jobs: list[ScriptJob],
    results_by_scene: dict[str, TaskResult],
    run_id: str,
    max_workers: int,
    started_at: str,
) -> str:
    completed = [results_by_scene[job.scene_id] for job in jobs if job.scene_id in results_by_scene]
    success_count = sum(1 for item in completed if is_success_result(item))
    not_ready_count = sum(1 for item in completed if is_not_ready_result(item))
    failed_count = sum(1 for item in completed if is_failed_result(item))
    pending_count = len(jobs) - len(completed)

    lines = [
        "# Seedance 批量执行报告",
        "",
        f"- 剧名：`{series_name}`",
        f"- 集数：`{episode_id}`",
        f"- 批次 ID：`{run_id}`",
        f"- 开始时间：`{started_at}`",
        f"- 并发数：`{max_workers}`",
        f"- 总任务数：`{len(jobs)}`",
        f"- 已完成：`{len(completed)}`",
        f"- 成功：`{success_count}`",
        f"- 失败：`{failed_count}`",
        f"- 未就绪：`{not_ready_count}`",
        f"- 等待中：`{pending_count}`",
        "",
        "## 汇总",
        "",
        "| 分镜 | task id | 最终状态 | 视频下载链接 | 本地视频 | 日志 |",
        "|---|---|---|---|---|---|",
    ]
    for job in jobs:
        result = results_by_scene.get(job.scene_id)
        if result is None:
            lines.append(
                f"| {job.scene_id} | - | 等待执行 | - | - | {markdown_link('log', str(job.log_path))} |"
            )
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    job.scene_id,
                    result.task_id or "-",
                    result.final_status,
                    result.video_url or "-",
                    markdown_link("video", result.local_video_path),
                    markdown_link("log", result.log_path),
                ]
            )
            + " |"
        )

    lines.extend(["", "## 详情", ""])
    for job in jobs:
        lines.append(render_task_detail(results_by_scene.get(job.scene_id), job))
        lines.extend(["", "---", ""])

    return "\n".join(lines).rstrip() + "\n"


def render_series_report(
    *,
    series_name: str,
    jobs: list[ScriptJob],
    results_by_scene_key: dict[tuple[str, str], TaskResult],
    run_id: str,
    max_workers: int,
    started_at: str,
    episode_report_paths: dict[str, Path],
) -> str:
    completed = list(results_by_scene_key.values())
    success_count = sum(1 for item in completed if is_success_result(item))
    not_ready_count = sum(1 for item in completed if is_not_ready_result(item))
    failed_count = sum(1 for item in completed if is_failed_result(item))
    pending_count = len(jobs) - len(completed)

    lines = [
        "# Seedance 系列批量执行总报告",
        "",
        f"- 剧名：`{series_name}`",
        f"- 批次 ID：`{run_id}`",
        f"- 开始时间：`{started_at}`",
        f"- 并发数：`{max_workers}`",
        f"- 总任务数：`{len(jobs)}`",
        f"- 已完成：`{len(completed)}`",
        f"- 成功：`{success_count}`",
        f"- 失败：`{failed_count}`",
        f"- 未就绪：`{not_ready_count}`",
        f"- 等待中：`{pending_count}`",
        "",
        "## 集数报告",
        "",
    ]
    for episode_id in sorted(episode_report_paths, key=episode_sort_key):
        lines.append(f"- `{episode_id}`: `{episode_report_paths[episode_id]}`")

    lines.extend(["", "## 全任务汇总", "", "| 集数 | 分镜 | task id | 最终状态 | 视频下载链接 |", "|---|---|---|---|---|"])
    for job in sorted(jobs, key=lambda item: (episode_sort_key(item.episode_id), scene_sort_key(item.scene_id))):
        result = results_by_scene_key.get((job.episode_id, job.scene_id))
        if result is None:
            lines.append(f"| {job.episode_id} | {job.scene_id} | - | 等待执行 | - |")
            continue
        lines.append(
            f"| {job.episode_id} | {job.scene_id} | {result.task_id or '-'} | {result.final_status} | {result.video_url or '-'} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_reports(
    *,
    series_dir: Path,
    jobs: list[ScriptJob],
    results_by_scene_key: dict[tuple[str, str], TaskResult],
    run_id: str,
    max_workers: int,
    started_at: str,
    root_report_path: Path,
    episode_report_paths: dict[str, Path],
) -> None:
    jobs_by_episode: dict[str, list[ScriptJob]] = {}
    for job in jobs:
        jobs_by_episode.setdefault(job.episode_id, []).append(job)

    for episode_id, episode_jobs in jobs_by_episode.items():
        results_by_scene = {
            job.scene_id: results_by_scene_key[(episode_id, job.scene_id)]
            for job in episode_jobs
            if (episode_id, job.scene_id) in results_by_scene_key
        }
        report_text = render_episode_report(
            series_name=series_dir.name,
            episode_id=episode_id,
            jobs=episode_jobs,
            results_by_scene=results_by_scene,
            run_id=run_id,
            max_workers=max_workers,
            started_at=started_at,
        )
        episode_report_paths[episode_id].write_text(report_text, encoding="utf-8")

    root_report_text = render_series_report(
        series_name=series_dir.name,
        jobs=jobs,
        results_by_scene_key=results_by_scene_key,
        run_id=run_id,
        max_workers=max_workers,
        started_at=started_at,
        episode_report_paths=episode_report_paths,
    )
    root_report_path.write_text(root_report_text, encoding="utf-8")


def resolve_max_workers(raw: str | None) -> int:
    if not raw:
        return DEFAULT_MAX_WORKERS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_WORKERS
    return max(1, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="并发执行某个剧下所有 Pxx__seedance_api.sh，并写入 markdown 报告。")
    parser.add_argument("--series", help="直接指定 outputs 下的剧目录名，例如 与天同寿-gpt。")
    parser.add_argument("--interactive", action="store_true", help="开启交互筛选：可选 episode 与 Pxx 区间。")
    parser.add_argument("--episode", help="只执行指定集数，例如 ep01 或 1。")
    parser.add_argument("--p-start", help="起始分镜编号，支持 P03 或 3。")
    parser.add_argument("--p-end", help="结束分镜编号，支持 P12 或 12。")
    parser.add_argument("--p-range", help="分镜区间，支持 P03-P12 或 3-12。")
    parser.add_argument("--max-workers", type=int, default=resolve_max_workers(os.environ.get("SEEDANCE_BATCH_MAX_WORKERS")))
    parser.add_argument("--dry-run", action="store_true", help="只列出会执行的脚本并生成初始报告，不真正调用 shell。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not OUTPUTS_ROOT.exists():
        raise RuntimeError(f"outputs 目录不存在：{OUTPUTS_ROOT}")

    series_dirs = list_series_dirs()
    if not series_dirs:
        raise RuntimeError("outputs/ 下没有找到任何可批量执行的 Pxx__seedance_api.sh。")

    selected_series_dir: Path | None = None
    if args.series:
        candidate = OUTPUTS_ROOT / args.series
        if not candidate.exists():
            raise RuntimeError(f"未找到剧目录：{candidate}")
        selected_series_dir = candidate.resolve()
    else:
        selected_index = choose_from_list(
            "请选择要批量提交 Seedance 任务的剧：",
            [item.name for item in series_dirs],
            default_index=0,
        )
        selected_series_dir = series_dirs[selected_index].resolve()

    jobs = build_jobs(selected_series_dir, run_id=timestamp_slug())
    if not jobs:
        raise RuntimeError(f"{selected_series_dir.name} 下没有找到任何 Pxx__seedance_api.sh。")

    episode_filter = normalize_episode_id(args.episode) if args.episode else None
    p_start, p_end = parse_scene_range(
        p_start_raw=args.p_start,
        p_end_raw=args.p_end,
        p_range_raw=args.p_range,
    )
    interactive_filter = bool(args.interactive) or (
        sys.stdin.isatty()
        and not args.episode
        and not args.p_start
        and not args.p_end
        and not args.p_range
    )
    if interactive_filter:
        if not args.episode:
            episode_filter = prompt_episode_filter(jobs)
        if not args.p_start and not args.p_end and not args.p_range:
            p_start, p_end = prompt_scene_range_filter()
    jobs = filter_jobs(
        jobs,
        episode_filter=episode_filter,
        p_start=p_start,
        p_end=p_end,
    )
    if not jobs:
        scope_desc = []
        if episode_filter:
            scope_desc.append(f"episode={episode_filter}")
        if p_start is not None or p_end is not None:
            left = f"P{p_start:02d}" if p_start is not None else "最小"
            right = f"P{p_end:02d}" if p_end is not None else "最大"
            scope_desc.append(f"scene_range={left}..{right}")
        scope_text = "，".join(scope_desc) if scope_desc else "当前筛选条件"
        raise RuntimeError(f"筛选后没有可执行任务：{scope_text}")

    if not args.dry_run and not os.environ.get("ARK_API_KEY"):
        raise RuntimeError("请先设置 ARK_API_KEY，再执行批量 Seedance 提交。")

    run_id = timestamp_slug()
    jobs = build_jobs(selected_series_dir, run_id=run_id)
    jobs = filter_jobs(
        jobs,
        episode_filter=episode_filter,
        p_start=p_start,
        p_end=p_end,
    )
    started_at = now_label()
    root_report_path = (selected_series_dir / f"seedance_batch_submit__{run_id}.md").resolve()
    episode_report_paths = {
        episode_id: (selected_series_dir / episode_id / f"seedance_batch_submit__{run_id}.md").resolve()
        for episode_id in sorted({job.episode_id for job in jobs}, key=episode_sort_key)
    }
    results_by_scene_key: dict[tuple[str, str], TaskResult] = {}

    write_reports(
        series_dir=selected_series_dir,
        jobs=jobs,
        results_by_scene_key=results_by_scene_key,
        run_id=run_id,
        max_workers=args.max_workers,
        started_at=started_at,
        root_report_path=root_report_path,
        episode_report_paths=episode_report_paths,
    )

    print_status(
        f"已选中 {selected_series_dir.name}，共 {len(jobs)} 个任务，涉及 {len(episode_report_paths)} 个 episode，"
        f"并发数 {args.max_workers}。"
    )
    print_status(f"总报告：{root_report_path}")
    for episode_id in sorted(episode_report_paths, key=episode_sort_key):
        print_status(f"{episode_id} 报告：{episode_report_paths[episode_id]}")

    if args.dry_run:
        for job in jobs:
            print_status(f"[dry-run] {job.episode_id}/{job.scene_id} -> {job.script_path}")
        return

    runnable_jobs: list[ScriptJob] = []
    for job in jobs:
        preflight_result = preflight_job(job)
        if preflight_result is None:
            runnable_jobs.append(job)
            continue
        results_by_scene_key[(job.episode_id, job.scene_id)] = preflight_result
        print_status(
            f"{job.episode_id}/{job.scene_id} 跳过：status={preflight_result.final_status}"
            f" error={preflight_result.error_message}"
        )

    write_reports(
        series_dir=selected_series_dir,
        jobs=jobs,
        results_by_scene_key=results_by_scene_key,
        run_id=run_id,
        max_workers=args.max_workers,
        started_at=started_at,
        root_report_path=root_report_path,
        episode_report_paths=episode_report_paths,
    )

    if not runnable_jobs:
        print_status("当前筛选范围内没有可执行的就绪任务；已把未就绪原因写入报告。")
        print_status(f"总报告：{root_report_path}")
        return

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_map = {executor.submit(run_job, job): job for job in runnable_jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                finished_at = now_label()
                result = TaskResult(
                    job=job,
                    return_code=1,
                    started_at=finished_at,
                    finished_at=finished_at,
                    duration_seconds=0.0,
                    task_id="",
                    final_status="runner_error",
                    video_url="",
                    last_frame_url="",
                    local_video_path="",
                    local_last_frame_path="",
                    submit_response_path="",
                    poll_response_path="",
                    task_id_path="",
                    check_script_path="",
                    references_env_path="",
                    log_path=str(job.log_path),
                    error_code="runner_error",
                    error_message=str(exc),
                )
            results_by_scene_key[(job.episode_id, job.scene_id)] = result
            write_reports(
                series_dir=selected_series_dir,
                jobs=jobs,
                results_by_scene_key=results_by_scene_key,
                run_id=run_id,
                max_workers=args.max_workers,
                started_at=started_at,
                root_report_path=root_report_path,
                episode_report_paths=episode_report_paths,
            )
            summary = f"{job.episode_id}/{job.scene_id} 完成：status={result.final_status} task_id={result.task_id or '-'}"
            if result.video_url:
                summary += f" video_url={result.video_url}"
            if result.return_code != 0 and result.error_message:
                summary += f" error={result.error_message}"
            print_status(summary)

    print_status("全部任务已执行完成。")
    print_status(f"总报告：{root_report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_status("用户中断。")
        sys.exit(130)
    except Exception as exc:
        print_status(str(exc))
        sys.exit(1)
