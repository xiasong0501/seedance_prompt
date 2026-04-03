from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import random
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_seedance_api_script import (
    choose_from_list,
    choose_range_from_list,
    episode_sort_key,
    find_storyboard_path,
    list_series_dirs,
    load_reference_workflow_config,
    choose_reference_mode,
    save_reference_workflow_config,
)
from upload_seedance_refs_to_tos import (
    DEFAULT_BUCKET,
    DEFAULT_ENDPOINT,
    DEFAULT_MODE,
    DEFAULT_REGION,
    DEFAULT_VALIDITY,
    choose_mode,
    env_file_text,
    file_sha256,
    load_references,
    load_upload_cache,
    patch_api_script_if_needed,
    prompt_with_default,
    rebuild_references_if_needed,
    resolve_credentials_flags,
    save_upload_cache,
    upload_one_reference,
)

ASSETS_ROOT = PROJECT_ROOT / "assets"
ASSET_REVIEW_DIRNAME = "seedance_asset_refs"
ASSET_RESULT_INBOX_DIRNAME = "_review_results"
ASSET_SUBMISSION_FILENAME = "submission_manifest.json"
ASSET_SUBMISSION_SCENE_SUFFIX = "__seedance_asset_submission.json"
ASSET_RESULT_CACHE_FILENAME = "_seedance_asset_id_cache.json"
ASSET_RESULT_SUFFIXES = {".zip", ".json", ".csv", ".tsv", ".txt", ".md"}
ASSET_ID_PATTERN = re.compile(r"(asset-\d{14}-[A-Za-z0-9]+)")
ASSET_URI_PREFIX = "asset://"
MODEL_GATE_BASE_URL = "https://mgate.zhiqungj.com"
MODEL_GATE_GROUP_TYPE = "AIGC"
MODEL_GATE_PROVIDER_MANUAL = "manual_review"
MODEL_GATE_PROVIDER_GATEWAY = "mgate"
MODEL_GATE_SIGNATURE_SCRIPT_ENV = "MODEL_GATE_SIGNATURE_SCRIPT"
MODEL_GATE_SIGNATURE_MODE_ENV = "MODEL_GATE_SIGNATURE_MODE"
MODEL_GATE_SIGNATURE_DEFAULT_MODE = "timestamp_body_hmac_sha256_hex"
MODEL_GATE_SIGNATURE_MODES = {
    "timestamp_body_hmac_sha256_hex",
    "timestamp_method_path_body_hmac_sha256_hex",
    "timestamp_method_path_body_hmac_sha256_base64",
}
MODEL_GATE_STATE_SUFFIX = "__seedance_mgate_submission.json"
MODEL_GATE_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MODEL_GATE_MAX_RETRIES = max(1, int(os.environ.get("MODEL_GATE_MAX_RETRIES", "6") or "6"))
MODEL_GATE_BACKOFF_BASE_SECONDS = max(
    0.5,
    float(os.environ.get("MODEL_GATE_BACKOFF_BASE_SECONDS", "2") or "2"),
)
MODEL_GATE_BACKOFF_MAX_SECONDS = max(
    MODEL_GATE_BACKOFF_BASE_SECONDS,
    float(os.environ.get("MODEL_GATE_BACKOFF_MAX_SECONDS", "20") or "20"),
)
MODEL_GATE_BACKOFF_JITTER_SECONDS = max(
    0.0,
    float(os.environ.get("MODEL_GATE_BACKOFF_JITTER_SECONDS", "0.8") or "0.8"),
)
MODEL_GATE_CREATE_ASSETS_MIN_INTERVAL_SECONDS = max(
    0.0,
    float(os.environ.get("MODEL_GATE_CREATE_ASSETS_MIN_INTERVAL_SECONDS", "1.5") or "1.5"),
)
MODEL_GATE_CREATE_ASSETS_CONCURRENCY_BACKOFF_BASE_SECONDS = max(
    MODEL_GATE_BACKOFF_BASE_SECONDS,
    float(os.environ.get("MODEL_GATE_CREATE_ASSETS_CONCURRENCY_BACKOFF_BASE_SECONDS", "4") or "4"),
)
MODEL_GATE_CREATE_ASSETS_CONCURRENCY_BACKOFF_MAX_SECONDS = max(
    MODEL_GATE_BACKOFF_MAX_SECONDS,
    float(os.environ.get("MODEL_GATE_CREATE_ASSETS_CONCURRENCY_BACKOFF_MAX_SECONDS", "60") or "60"),
)
MODEL_GATE_POLL_MAX_WAIT_SECONDS = max(
    0.0,
    float(os.environ.get("MODEL_GATE_POLL_MAX_WAIT_SECONDS", "3600") or "3600"),
)
MODEL_GATE_MAX_PENDING_TASKS = int(os.environ.get("MODEL_GATE_MAX_PENDING_TASKS", "0") or "0")
MODEL_GATE_POLL_MAX_STAGNANT_ROUNDS = max(
    1,
    int(os.environ.get("MODEL_GATE_POLL_MAX_STAGNANT_ROUNDS", "8") or "8"),
)
ASSET_FILENAME_KEYS = {
    "filename",
    "file_name",
    "name",
    "basename",
    "original_filename",
    "original_file_name",
    "source_filename",
    "source_file_name",
    "path",
    "file_path",
    "relative_path",
    "uri",
    "url",
    "local_path",
    "staged_basename",
    "staged_relative_path",
}
ASSET_ID_KEYS = {
    "asset_id",
    "assetid",
    "id",
}
ASSET_URI_KEYS = {
    "asset_uri",
    "asseturi",
    "uri",
    "url",
}
ASSET_PROVIDER_OPTIONS = [
    (MODEL_GATE_PROVIDER_MANUAL, "导出送审包 / 导入审核结果（现有人工回包模式）"),
    (MODEL_GATE_PROVIDER_GATEWAY, "Model Gate 自动建组 / 提交素材 / 查询 asset_id"),
]
_MODEL_GATE_NEXT_ALLOWED_AT_BY_PATH: dict[str, float] = {}
_MODEL_GATE_LAST_REQUEST_AT_BY_PATH: dict[str, float] = {}
_MODEL_GATE_DYNAMIC_MIN_INTERVAL_SECONDS_BY_PATH: dict[str, float] = {}


class ModelGateConcurrencyLimitError(RuntimeError):
    pass


def env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


FORCE_SCENE_REFS_REUPLOAD = env_flag("SEEDANCE_SCENE_REFS_FORCE_REUPLOAD", default=False)
FORCE_SCENE_REFS_REREVIEW = env_flag("SEEDANCE_SCENE_REFS_FORCE_REREVIEW", default=False)


def print_status(message: str) -> None:
    print(f"[seedance-refs] {message}", flush=True)


def load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def load_project_env_files() -> None:
    for candidate in [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / ".env.local",
        PROJECT_ROOT / "config" / ".env",
        PROJECT_ROOT / "config" / ".env.local",
    ]:
        load_env_file(candidate)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def reference_material_type(ref: dict[str, Any]) -> str:
    return str(ref.get("material_type") or "").strip()


def is_scene_reference(ref: dict[str, Any]) -> bool:
    return reference_material_type(ref) == "场景参考"


def should_force_reupload_reference(ref: dict[str, Any]) -> bool:
    return FORCE_SCENE_REFS_REUPLOAD and is_scene_reference(ref)


def should_force_rereview_reference(ref: dict[str, Any]) -> bool:
    return FORCE_SCENE_REFS_REREVIEW and is_scene_reference(ref)


def normalize_candidate(value: str | Path | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace("\\", "/").strip().lower()


def candidate_names_for_value(value: str | Path | None) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    names = {normalize_candidate(text)}
    names.add(normalize_candidate(Path(text).name))
    return {item for item in names if item}


def extract_asset_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = ASSET_ID_PATTERN.search(text)
    return match.group(1) if match else ""


def to_asset_uri(value: str) -> str:
    asset_id = extract_asset_id(value)
    if not asset_id:
        return ""
    return f"{ASSET_URI_PREFIX}{asset_id}"


def scene_id_from_reference_path(path: Path) -> str:
    return path.name.split("__", 1)[0]


def list_reference_files_for_episode(episode_dir: Path) -> list[Path]:
    matched: dict[str, Path] = {}
    for pattern in (
        "[0-9]*__seedance_api_references.json",
        "P*__seedance_api_references.json",
        "p*__seedance_api_references.json",
        "SP*__seedance_api_references.json",
        "sp*__seedance_api_references.json",
    ):
        for path in episode_dir.glob(pattern):
            matched[path.name] = path
    return sorted(matched.values(), key=lambda path: episode_sort_key(scene_id_from_reference_path(path)))


def resolve_assets_series_dir(series_dir: Path) -> Path:
    exact = ASSETS_ROOT / series_dir.name
    if exact.exists():
        return exact.resolve()
    fallback = ASSETS_ROOT / series_dir.name.replace("-claude", "").replace("-gpt", "")
    if fallback.exists():
        return fallback.resolve()
    return exact.resolve()


def choose_reference_scope() -> tuple[Path, Path, list[Path], str, Path]:
    series_dirs = list_series_dirs()
    if not series_dirs:
        raise RuntimeError("outputs/ 下没有找到可用剧。")
    series_idx = choose_from_list(
        "请选择要处理 Seedance 引用素材的剧：",
        [path.name for path in series_dirs],
        default_index=0,
    )
    series_dir = series_dirs[series_idx]

    episode_dirs = sorted(
        [path for path in series_dir.iterdir() if path.is_dir() and find_storyboard_path(path) is not None],
        key=lambda item: episode_sort_key(item.name),
    )
    if not episode_dirs:
        raise RuntimeError(f"{series_dir.name} 下没有找到任何带 02-seedance-prompts*.md 的集数。")
    episode_idx = choose_from_list(
        f"请选择 {series_dir.name} 的集数：",
        [path.name for path in episode_dirs],
        default_index=0,
    )
    episode_dir = episode_dirs[episode_idx]

    workflow_payload = load_reference_workflow_config(episode_dir)
    default_mode = str(workflow_payload.get("reference_mode") or "tos").strip().lower() or "tos"
    reference_mode = choose_reference_mode(default_mode=default_mode)

    reference_files = list_reference_files_for_episode(episode_dir)
    if not reference_files:
        raise RuntimeError(
            f"{episode_dir} 下没有找到任何可用引用文件。"
            "预期文件名类似 01__seedance_api_references.json、P01__seedance_api_references.json 或 sp01__seedance_api_references.json。"
        )

    start_idx, end_idx = choose_range_from_list(
        "请选择要处理的场景范围：",
        [f"{episode_dir.name} / {path.name.replace('__seedance_api_references.json', '')}" for path in reference_files],
        default_start=0,
        default_end=0,
    )
    selected_reference_files = reference_files[start_idx : end_idx + 1]
    workflow_path = save_reference_workflow_config(
        episode_dir=episode_dir,
        series_name=series_dir.name,
        episode_id=episode_dir.name,
        reference_mode=reference_mode,
        selected_scene_ids=[scene_id_from_reference_path(path) for path in selected_reference_files],
    )
    print_status(f"引用模式已保存：{workflow_path} -> {reference_mode}")
    return series_dir, episode_dir, selected_reference_files, reference_mode, workflow_path


def run_tos_flow(
    *,
    series_dir: Path,
    episode_dir: Path,
    reference_files: list[Path],
) -> dict[str, Any]:
    bucket = prompt_with_default("请输入 TOS bucket", DEFAULT_BUCKET)
    region = prompt_with_default("请输入 region", DEFAULT_REGION)
    endpoint = prompt_with_default("请输入 endpoint", DEFAULT_ENDPOINT)
    mode = choose_mode()
    validity = DEFAULT_VALIDITY if mode == "presign" else ""
    if mode == "presign":
        validity = prompt_with_default("请输入预签名有效期", DEFAULT_VALIDITY)

    flags = resolve_credentials_flags()
    if flags:
        print_status("检测到环境变量中的 AK/SK，将直接用于本次上传。")
    else:
        print_status("未检测到环境变量 AK/SK，将使用 tosutil 已配置的默认凭证。")

    upload_cache = load_upload_cache(episode_dir)
    if upload_cache:
        print_status(f"检测到 {len(upload_cache)} 条已上传引用缓存，将优先复用同图 URL。")

    summaries: list[dict[str, Any]] = []
    for references_path in reference_files:
        scene_id = scene_id_from_reference_path(references_path)
        api_script_path = episode_dir / references_path.name.replace("__seedance_api_references.json", "__seedance_api.sh")
        refs = rebuild_references_if_needed(series_dir, episode_dir, references_path)
        if not refs:
            print_status(f"{scene_id} 引用清单为空，已跳过上传：{references_path}")
            summaries.append(
                {
                    "scene_id": scene_id,
                    "references_path": str(references_path),
                    "uploaded_count": 0,
                    "skipped": True,
                }
            )
            continue

        uploaded: list[dict[str, Any]] = []
        uploaded_count = 0
        reused_count = 0
        for ref in refs:
            uploaded_item = upload_one_reference(
                bucket=bucket,
                region=region,
                endpoint=endpoint,
                validity=validity,
                mode=mode,
                series_name=series_dir.name,
                episode_id=episode_dir.name,
                scene_id=scene_id,
                ref=ref,
                flags=flags,
                upload_cache=upload_cache,
                force_reupload=should_force_reupload_reference(ref),
            )
            if uploaded_item.get("upload_status") == "reused_cached":
                reused_count += 1
            else:
                uploaded_count += 1
            uploaded.append(uploaded_item)

        references_path.write_text(json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")
        env_path = episode_dir / f"{scene_id}__seedance_api_urls.env"
        manifest_path = episode_dir / f"{scene_id}__seedance_api_uploaded_refs.json"
        env_path.write_text(env_file_text(uploaded), encoding="utf-8")
        manifest_path.write_text(json.dumps(uploaded, ensure_ascii=False, indent=2), encoding="utf-8")
        patched = patch_api_script_if_needed(api_script_path, env_path.name)

        print_status(f"{scene_id} URL env 已写入：{env_path}")
        print_status(f"{scene_id} 上传清单已写入：{manifest_path}")
        if patched:
            print_status(f"{scene_id} 已补丁现有 API 脚本以自动读取：{api_script_path}")
        else:
            print_status(f"{scene_id} API 脚本已支持或无需补丁：{api_script_path}")
        summaries.append(
            {
                "scene_id": scene_id,
                "references_path": str(references_path),
                "env_path": str(env_path),
                "manifest_path": str(manifest_path),
                "api_script_path": str(api_script_path),
                "uploaded_count": uploaded_count,
                "reference_count": len(uploaded),
                "reused_cached_count": reused_count,
                "skipped": False,
            }
        )

    cache_path = save_upload_cache(episode_dir, upload_cache)
    print_status(f"episode 级上传缓存已写入：{cache_path}")
    return {
        "series_name": series_dir.name,
        "episode_id": episode_dir.name,
        "scene_range": {
            "start": scene_id_from_reference_path(reference_files[0]),
            "end": scene_id_from_reference_path(reference_files[-1]),
            "count": len(reference_files),
        },
        "reference_mode": "tos",
        "bucket": bucket,
        "region": region,
        "endpoint": endpoint,
        "mode": mode,
        "upload_cache_path": str(cache_path),
        "results": summaries,
    }


def asset_review_root(series_dir: Path, episode_dir: Path) -> Path:
    return resolve_assets_series_dir(series_dir) / ASSET_REVIEW_DIRNAME / episode_dir.name


def asset_result_inbox_dir(series_dir: Path, episode_dir: Path) -> Path:
    path = asset_review_root(series_dir, episode_dir) / ASSET_RESULT_INBOX_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_asset_result_candidate(series_dir: Path, episode_dir: Path) -> Path | None:
    inbox_dir = asset_result_inbox_dir(series_dir, episode_dir)
    candidates = [
        path for path in inbox_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ASSET_RESULT_SUFFIXES
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def scene_submission_state_path(episode_dir: Path, scene_id: str) -> Path:
    return episode_dir / f"{scene_id}{ASSET_SUBMISSION_SCENE_SUFFIX}"


def load_asset_result_cache(episode_dir: Path) -> dict[str, dict[str, Any]]:
    try:
        current_episode_dir = episode_dir.resolve()
    except Exception:
        current_episode_dir = episode_dir
    series_dir = current_episode_dir.parent
    if series_dir.exists():
        episode_dirs = [
            path.resolve()
            for path in series_dir.iterdir()
            if path.is_dir()
        ]
        episode_dirs.sort(key=lambda path: episode_sort_key(path.name))
    else:
        episode_dirs = []
    ordered_episode_dirs = [path for path in episode_dirs if path != current_episode_dir]
    ordered_episode_dirs.append(current_episode_dir)

    cache: dict[str, dict[str, Any]] = {}
    for related_episode_dir in ordered_episode_dirs:
        cache_path = related_episode_dir / ASSET_RESULT_CACHE_FILENAME
        if not cache_path.exists():
            continue
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                local_path = str(item.get("local_path") or "").strip()
                if not local_path:
                    continue
                cache[str(Path(local_path).expanduser().resolve())] = item
                continue
        if isinstance(payload, dict):
            for key, item in payload.items():
                if not isinstance(item, dict):
                    continue
                local_path = str(item.get("local_path") or key or "").strip()
                if not local_path:
                    continue
                cache[str(Path(local_path).expanduser().resolve())] = item
    return cache


def save_asset_result_cache(episode_dir: Path, cache: dict[str, dict[str, Any]]) -> Path:
    cache_path = episode_dir / ASSET_RESULT_CACHE_FILENAME
    items = sorted(cache.values(), key=lambda item: str(item.get("local_path") or ""))
    cache_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_path


def stage_reference_file(package_dir: Path, scene_id: str, ref: dict[str, Any], local_path: Path) -> tuple[Path, str]:
    files_root = package_dir / "files" / scene_id
    files_root.mkdir(parents=True, exist_ok=True)
    staged_name = f"{ref['env_var']}__{local_path.name}"
    staged_path = files_root / staged_name
    shutil.copy2(local_path, staged_path)
    staged_relative_path = str(staged_path.relative_to(package_dir))
    return staged_path, staged_relative_path


def export_asset_review_package(
    *,
    series_dir: Path,
    episode_dir: Path,
    reference_files: list[Path],
) -> dict[str, Any]:
    review_root = asset_review_root(series_dir, episode_dir)
    inbox_dir = asset_result_inbox_dir(series_dir, episode_dir)
    review_root.mkdir(parents=True, exist_ok=True)
    range_start = scene_id_from_reference_path(reference_files[0])
    range_end = scene_id_from_reference_path(reference_files[-1])
    bundle_slug = f"{range_start}-{range_end}__{timestamp_slug()}"
    package_dir = review_root / bundle_slug
    package_dir.mkdir(parents=True, exist_ok=True)

    bundle_items: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    for references_path in reference_files:
        scene_id = scene_id_from_reference_path(references_path)
        refs = rebuild_references_if_needed(series_dir, episode_dir, references_path)
        scene_items: list[dict[str, Any]] = []
        for ref in refs:
            local_path_raw = str(ref.get("local_path") or "").strip()
            if not local_path_raw:
                raise FileNotFoundError(f"{scene_id} 缺少本地引用路径：{ref.get('env_var')} / {ref.get('label')}")
            local_path = Path(local_path_raw).expanduser().resolve()
            if not local_path.exists():
                raise FileNotFoundError(f"{scene_id} 本地引用素材不存在：{local_path}")
            staged_path, staged_relative_path = stage_reference_file(package_dir, scene_id, ref, local_path)
            item = {
                "scene_id": scene_id,
                "references_path": str(references_path.resolve()),
                "env_var": ref.get("env_var"),
                "token": ref.get("token"),
                "token_number": ref.get("token_number"),
                "material_type": ref.get("material_type"),
                "label": ref.get("label"),
                "source_local_path": str(local_path),
                "source_basename": local_path.name,
                "staged_path": str(staged_path),
                "staged_relative_path": staged_relative_path,
                "staged_basename": staged_path.name,
                "local_sha256": file_sha256(local_path),
                "local_size_bytes": local_path.stat().st_size,
                "asset_id": "",
                "asset_uri": "",
            }
            scene_items.append(item)
            bundle_items.append(item)

        scene_submission_payload = {
            "series_name": series_dir.name,
            "episode_id": episode_dir.name,
            "scene_id": scene_id,
            "reference_mode": "asset",
            "asset_provider": MODEL_GATE_PROVIDER_MANUAL,
            "package_dir": str(package_dir),
            "package_path": str(package_dir.with_suffix(".zip")),
            "created_at": now_iso(),
            "items": scene_items,
        }
        scene_state_path = scene_submission_state_path(episode_dir, scene_id)
        scene_state_path.write_text(json.dumps(scene_submission_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        scene_summaries.append(
            {
                "scene_id": scene_id,
                "references_path": str(references_path),
                "scene_submission_path": str(scene_state_path),
                "reference_count": len(scene_items),
            }
        )

    bundle_manifest = {
        "series_name": series_dir.name,
        "episode_id": episode_dir.name,
        "reference_mode": "asset",
        "asset_provider": MODEL_GATE_PROVIDER_MANUAL,
        "created_at": now_iso(),
        "scene_range": {
            "start": range_start,
            "end": range_end,
            "count": len(reference_files),
        },
        "items": bundle_items,
    }
    manifest_path = package_dir / ASSET_SUBMISSION_FILENAME
    manifest_path.write_text(json.dumps(bundle_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    package_zip_path = package_dir.with_suffix(".zip")
    with zipfile.ZipFile(package_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for file_path in sorted(package_dir.rglob("*")):
            if file_path.is_dir():
                continue
            handle.write(file_path, arcname=str(file_path.relative_to(package_dir)))

    print_status(f"Asset 送审包目录已生成：{package_dir}")
    print_status(f"Asset 送审压缩包已生成：{package_zip_path}")
    print_status(f"审核返回包推荐放置目录：{inbox_dir}")
    return {
        "series_name": series_dir.name,
        "episode_id": episode_dir.name,
        "reference_mode": "asset",
        "asset_provider": MODEL_GATE_PROVIDER_MANUAL,
        "scene_range": {
            "start": range_start,
            "end": range_end,
            "count": len(reference_files),
        },
        "package_dir": str(package_dir),
        "package_path": str(package_zip_path),
        "result_inbox_dir": str(inbox_dir),
        "manifest_path": str(manifest_path),
        "results": scene_summaries,
    }


def extract_candidate_names_from_text(text: str) -> set[str]:
    names: set[str] = set()
    for token in re.findall(r"[^\s,;|]+", text):
        names.update(candidate_names_for_value(token))
    return names


def build_asset_record(asset_id: str, candidate_values: list[str], source: str) -> dict[str, Any]:
    asset_uri = to_asset_uri(asset_id)
    candidate_names: set[str] = set()
    for value in candidate_values:
        candidate_names.update(candidate_names_for_value(value))
    return {
        "asset_id": extract_asset_id(asset_id),
        "asset_uri": asset_uri,
        "candidate_names": sorted(candidate_names),
        "source": source,
    }


def asset_id_from_mapping(mapping: dict[str, Any]) -> str:
    for key, value in mapping.items():
        normalized_key = str(key).strip().lower().replace("-", "_")
        if normalized_key in ASSET_ID_KEYS:
            asset_id = extract_asset_id(value)
            if asset_id:
                return asset_id
    for key, value in mapping.items():
        normalized_key = str(key).strip().lower().replace("-", "_")
        if normalized_key in ASSET_URI_KEYS:
            asset_id = extract_asset_id(value)
            if asset_id:
                return asset_id
    return ""


def records_from_json_payload(payload: Any, source: str, inherited_candidates: list[str] | None = None) -> list[dict[str, Any]]:
    inherited_candidates = inherited_candidates or []
    records: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        asset_id = asset_id_from_mapping(payload)
        candidate_values = list(inherited_candidates)
        for key, value in payload.items():
            normalized_key = str(key).strip().lower().replace("-", "_")
            if normalized_key in ASSET_FILENAME_KEYS and isinstance(value, str):
                candidate_values.append(value)
            elif isinstance(value, str):
                candidate_values.extend(list(candidate_names_for_value(value)))
        if asset_id:
            records.append(build_asset_record(asset_id, candidate_values, source))
        for value in payload.values():
            records.extend(records_from_json_payload(value, source, candidate_values))
    elif isinstance(payload, list):
        for item in payload:
            records.extend(records_from_json_payload(item, source, inherited_candidates))
    elif isinstance(payload, str):
        asset_id = extract_asset_id(payload)
        if asset_id:
            records.append(build_asset_record(asset_id, inherited_candidates + [payload], source))
    return records


def records_from_delimited_text(text: str, source: str, delimiter: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    records: list[dict[str, Any]] = []
    for row in reader:
        if not isinstance(row, dict):
            continue
        asset_id = asset_id_from_mapping(row)
        if not asset_id:
            continue
        candidate_values = [value for value in row.values() if isinstance(value, str)]
        records.append(build_asset_record(asset_id, candidate_values, source))
    return records


def records_from_plain_text(text: str, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        asset_id = extract_asset_id(line)
        if not asset_id:
            continue
        candidate_values = [line]
        candidate_values.extend(list(extract_candidate_names_from_text(line)))
        records.append(build_asset_record(asset_id, candidate_values, source))
    return records


def parse_asset_result_file(path: Path, raw_bytes: bytes | None = None) -> list[dict[str, Any]]:
    data = raw_bytes if raw_bytes is not None else path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(data.decode("utf-8", errors="replace"))
        return records_from_json_payload(payload, str(path))
    if suffix == ".csv":
        return records_from_delimited_text(data.decode("utf-8", errors="replace"), str(path), ",")
    if suffix == ".tsv":
        return records_from_delimited_text(data.decode("utf-8", errors="replace"), str(path), "\t")
    if suffix in {".txt", ".log", ".md"}:
        return records_from_plain_text(data.decode("utf-8", errors="replace"), str(path))
    return []


def parse_asset_result_zip(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(path, "r") as handle:
        for member in handle.infolist():
            if member.is_dir():
                continue
            member_name = member.filename
            member_bytes = handle.read(member)
            member_path = Path(member_name)
            records.extend(parse_asset_result_file(member_path, member_bytes))
            member_asset_id = extract_asset_id(member_name)
            if member_asset_id:
                records.append(
                    build_asset_record(
                        member_asset_id,
                        [member_name, member_path.name],
                        f"{path}::{member_name}",
                    )
                )
    return records


def load_asset_result_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".zip":
        return parse_asset_result_zip(path)
    return parse_asset_result_file(path)


def build_asset_lookup(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for record in records:
        asset_uri = str(record.get("asset_uri") or "").strip()
        if not asset_uri:
            continue
        for candidate in record.get("candidate_names") or []:
            normalized = normalize_candidate(candidate)
            if normalized and normalized not in lookup:
                lookup[normalized] = record
    return lookup


def choose_asset_action() -> str:
    options = [
        ("export", "导出送审包"),
        ("import", "导入审核结果并生成 asset:// 映射"),
        ("both", "先导出送审包；若已收到回包则继续导入"),
    ]
    selected_index = choose_from_list(
        "请选择 Asset 模式要执行的动作：",
        [label for _, label in options],
        default_index=0,
    )
    return options[selected_index][0]


def choose_asset_provider(default_provider: str = MODEL_GATE_PROVIDER_MANUAL) -> str:
    default_index = next(
        (index for index, (provider, _) in enumerate(ASSET_PROVIDER_OPTIONS) if provider == default_provider),
        0,
    )
    selected_index = choose_from_list(
        "请选择 Asset 模式的素材入库方式：",
        [label for _, label in ASSET_PROVIDER_OPTIONS],
        default_index=default_index,
    )
    return ASSET_PROVIDER_OPTIONS[selected_index][0]


def scene_mgate_state_path(episode_dir: Path, scene_id: str) -> Path:
    return episode_dir / f"{scene_id}{MODEL_GATE_STATE_SUFFIX}"


def prompt_review_result_path(
    *,
    series_dir: Path,
    episode_dir: Path,
    default_path: Path | None = None,
) -> Path | None:
    inbox_dir = asset_result_inbox_dir(series_dir, episode_dir)
    if default_path is not None:
        print_status(f"检测到最近审核回包：{default_path}")
        print_status("回车可直接使用这个文件；如需别的文件，也可以手动输入完整路径。")
    else:
        print_status(f"暂未检测到审核回包。推荐把平台返回 zip/json 放到：{inbox_dir}")
        print_status("也可以直接输入任意已有的审核返回文件完整路径。")

    while True:
        prompt = "请输入审核返回压缩包或映射文件路径"
        if default_path is not None:
            prompt += "（回车使用默认）"
        prompt += "："
        raw = input(prompt).strip()
        if not raw:
            if default_path is not None:
                return default_path.resolve()
            print(f"当前还没有默认回包可用。建议先把回包放到：{inbox_dir}")
            return None
        path = Path(raw).expanduser()
        if not path.exists():
            print(f"路径不存在：{path}")
            continue
        if path.is_dir():
            candidates = [
                child for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in ASSET_RESULT_SUFFIXES
            ]
            if not candidates:
                print(f"目录里未找到可用回包文件：{path}")
                continue
            selected = max(candidates, key=lambda child: child.stat().st_mtime_ns)
            print_status(f"已自动选择目录里最新回包：{selected}")
            return selected.resolve()
        return path.resolve()


def load_scene_submission_state(episode_dir: Path, scene_id: str) -> dict[str, Any]:
    path = scene_submission_state_path(episode_dir, scene_id)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def find_matching_asset_record(
    *,
    ref: dict[str, Any],
    local_path: Path,
    submission_state: dict[str, Any],
    lookup: dict[str, dict[str, Any]],
    asset_cache: dict[str, dict[str, Any]],
    allow_asset_cache_fallback: bool = True,
) -> dict[str, Any] | None:
    candidate_names: set[str] = set()
    candidate_names.update(candidate_names_for_value(local_path))
    candidate_names.update(candidate_names_for_value(ref.get("env_var")))
    candidate_names.update(candidate_names_for_value(ref.get("label")))
    for item in submission_state.get("items") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("env_var") or "") != str(ref.get("env_var") or ""):
            continue
        candidate_names.update(candidate_names_for_value(item.get("staged_basename")))
        candidate_names.update(candidate_names_for_value(item.get("staged_relative_path")))
        candidate_names.update(candidate_names_for_value(item.get("source_basename")))

    for candidate in candidate_names:
        record = lookup.get(normalize_candidate(candidate))
        if record is not None:
            return record

    if not allow_asset_cache_fallback:
        return None

    local_path_key = str(local_path.resolve())
    cached_entry = asset_cache.get(local_path_key)
    if cached_entry is None:
        return None
    cached_sha256 = str(cached_entry.get("local_sha256") or "").strip()
    current_sha256 = file_sha256(local_path)
    if cached_sha256 and cached_sha256 != current_sha256:
        return None
    asset_uri = str(cached_entry.get("asset_uri") or "").strip()
    if not asset_uri:
        return None
    return {
        "asset_id": extract_asset_id(asset_uri),
        "asset_uri": asset_uri,
        "candidate_names": sorted(candidate_names),
        "source": "asset-cache",
    }


def import_asset_result(
    *,
    series_dir: Path,
    episode_dir: Path,
    reference_files: list[Path],
    result_source_path: Path,
) -> dict[str, Any]:
    records = load_asset_result_records(result_source_path)
    if not records:
        raise RuntimeError(f"未能从审核结果中解析到任何 asset_id：{result_source_path}")
    lookup = build_asset_lookup(records)
    asset_cache = load_asset_result_cache(episode_dir)
    summaries: list[dict[str, Any]] = []

    for references_path in reference_files:
        scene_id = scene_id_from_reference_path(references_path)
        api_script_path = episode_dir / references_path.name.replace("__seedance_api_references.json", "__seedance_api.sh")
        refs = rebuild_references_if_needed(series_dir, episode_dir, references_path)
        submission_state = load_scene_submission_state(episode_dir, scene_id)
        uploaded: list[dict[str, Any]] = []
        imported_count = 0
        reused_count = 0

        for ref in refs:
            local_path_raw = str(ref.get("local_path") or "").strip()
            if not local_path_raw:
                raise FileNotFoundError(f"{scene_id} 缺少本地引用路径：{ref.get('env_var')} / {ref.get('label')}")
            local_path = Path(local_path_raw).expanduser().resolve()
            if not local_path.exists():
                raise FileNotFoundError(f"{scene_id} 本地引用素材不存在：{local_path}")

            matched_record = find_matching_asset_record(
                ref=ref,
                local_path=local_path,
                submission_state=submission_state,
                lookup=lookup,
                asset_cache=asset_cache,
                allow_asset_cache_fallback=not should_force_rereview_reference(ref),
            )
            if matched_record is None:
                raise RuntimeError(
                    f"{scene_id} 未找到审核资产 ID：{ref.get('env_var')} / {local_path.name}。"
                    "如果返回包结构与当前解析器不一致，可把返回 zip 的样例给我，我再补适配。"
                )

            asset_uri = str(matched_record.get("asset_uri") or "").strip()
            local_sha256 = file_sha256(local_path)
            local_path_key = str(local_path)
            cached_entry = asset_cache.get(local_path_key)
            upload_status = "imported_asset"
            if (
                not should_force_rereview_reference(ref)
                and cached_entry
                and str(cached_entry.get("asset_uri") or "").strip() == asset_uri
            ):
                upload_status = "reused_asset_cache"
                reused_count += 1
            else:
                imported_count += 1

            uploaded_item = {
                **ref,
                "mode": "asset",
                "remote_url": asset_uri,
                "asset_id": extract_asset_id(asset_uri),
                "asset_uri": asset_uri,
                "upload_status": upload_status,
                "scene_id": scene_id,
                "source_path": str(result_source_path),
                "matched_from": str(matched_record.get("source") or ""),
                "local_path": local_path_key,
                "local_sha256": local_sha256,
                "local_size_bytes": local_path.stat().st_size,
                "local_mtime_ns": local_path.stat().st_mtime_ns,
            }
            uploaded.append(uploaded_item)
            asset_cache[local_path_key] = {
                "local_path": local_path_key,
                "local_sha256": local_sha256,
                "local_size_bytes": local_path.stat().st_size,
                "local_mtime_ns": local_path.stat().st_mtime_ns,
                "asset_uri": asset_uri,
                "asset_id": extract_asset_id(asset_uri),
                "scene_id": scene_id,
                "source_path": str(result_source_path),
                "updated_at": now_iso(),
            }

        env_path = episode_dir / f"{scene_id}__seedance_api_urls.env"
        manifest_path = episode_dir / f"{scene_id}__seedance_api_uploaded_refs.json"
        env_path.write_text(env_file_text(uploaded), encoding="utf-8")
        manifest_path.write_text(json.dumps(uploaded, ensure_ascii=False, indent=2), encoding="utf-8")
        patched = patch_api_script_if_needed(api_script_path, env_path.name)
        print_status(f"{scene_id} Asset env 已写入：{env_path}")
        print_status(f"{scene_id} Asset 引用清单已写入：{manifest_path}")
        if patched:
            print_status(f"{scene_id} 已补丁现有 API 脚本以自动读取：{api_script_path}")
        else:
            print_status(f"{scene_id} API 脚本已支持或无需补丁：{api_script_path}")
        summaries.append(
            {
                "scene_id": scene_id,
                "references_path": str(references_path),
                "env_path": str(env_path),
                "manifest_path": str(manifest_path),
                "api_script_path": str(api_script_path),
                "imported_count": imported_count,
                "reused_asset_cache_count": reused_count,
                "reference_count": len(uploaded),
            }
        )

    cache_path = save_asset_result_cache(episode_dir, asset_cache)
    print_status(f"episode 级 Asset 缓存已写入：{cache_path}")
    return {
        "series_name": series_dir.name,
        "episode_id": episode_dir.name,
        "reference_mode": "asset",
        "asset_provider": MODEL_GATE_PROVIDER_MANUAL,
        "scene_range": {
            "start": scene_id_from_reference_path(reference_files[0]),
            "end": scene_id_from_reference_path(reference_files[-1]),
            "count": len(reference_files),
        },
        "result_source_path": str(result_source_path),
        "asset_cache_path": str(cache_path),
        "results": summaries,
    }


def load_mgate_scene_state(episode_dir: Path, scene_id: str) -> dict[str, Any]:
    candidates = [
        scene_mgate_state_path(episode_dir, scene_id),
        scene_submission_state_path(episode_dir, scene_id),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and str(payload.get("asset_provider") or "") == MODEL_GATE_PROVIDER_GATEWAY:
            return payload
    return {}


def extract_mgate_state_items(state_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in list(state_payload.get("items") or []) if isinstance(item, dict)]


def has_pollable_mgate_state(episode_dir: Path, scene_id: str) -> bool:
    state_payload = load_mgate_scene_state(episode_dir, scene_id)
    return bool(extract_mgate_state_items(state_payload))


def mgate_item_has_asset(item: dict[str, Any]) -> bool:
    return bool(str(item.get("asset_uri") or "").strip())


def mgate_item_display_ref(item: dict[str, Any]) -> str:
    return str(item.get("asset_uri") or "").strip() or str(item.get("task_id") or "").strip()


def is_mgate_terminal_failed_status(status: str) -> bool:
    return status.strip().lower() in {"failed", "error"}


def remove_if_exists(path: Path) -> bool:
    if not path.exists():
        return False
    path.unlink()
    return True


def write_scene_state(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def scene_state_identity(item: dict[str, Any]) -> tuple[str, str, str, str]:
    local_path = str(item.get("local_path") or "").strip()
    if local_path:
        try:
            local_path = str(Path(local_path).expanduser().resolve())
        except Exception:
            local_path = str(item.get("local_path") or "").strip()
    remote_url = str(item.get("remote_url") or "").strip()
    token = str(item.get("token") or "").strip()
    label = str(item.get("label") or "").strip()
    return (local_path, remote_url, token, label)


def detect_existing_mgate_group_id(episode_dir: Path, scene_ids: list[str]) -> str:
    candidates: list[tuple[float, str, Path]] = []
    for scene_id in scene_ids:
        for path in (scene_mgate_state_path(episode_dir, scene_id), scene_submission_state_path(episode_dir, scene_id)):
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if str(payload.get("asset_provider") or "") != MODEL_GATE_PROVIDER_GATEWAY:
                continue
            group_id = str(payload.get("group_id") or "").strip()
            if not group_id:
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            candidates.append((mtime, group_id, path))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen_group_id = candidates[0][1]
    distinct_group_ids = sorted({group_id for _, group_id, _ in candidates})
    if len(distinct_group_ids) > 1:
        print_status(
            "检测到多个历史 Model Gate 素材组，优先复用最近一次的 group_id："
            f"{chosen_group_id}"
        )
    return chosen_group_id


def merge_uploaded_item_with_existing_mgate_state(
    *,
    uploaded_item: dict[str, Any],
    existing_item: dict[str, Any],
    group_id: str,
) -> dict[str, Any]:
    merged = {
        **uploaded_item,
        "reference_mode": "asset",
        "asset_provider": MODEL_GATE_PROVIDER_GATEWAY,
        "group_id": group_id,
        "task_id": str(existing_item.get("task_id") or "").strip(),
        "asset_id": str(existing_item.get("asset_id") or "").strip(),
        "asset_uri": str(existing_item.get("asset_uri") or "").strip(),
        "model_gate_status": str(existing_item.get("model_gate_status") or "").strip()
        or ("Active" if str(existing_item.get("asset_uri") or "").strip() else "submitted"),
        "submitted_at": str(existing_item.get("submitted_at") or "").strip() or now_iso(),
    }
    updated_at = str(existing_item.get("updated_at") or "").strip()
    if updated_at:
        merged["updated_at"] = updated_at
    result_payload = existing_item.get("result_payload")
    if result_payload is not None:
        merged["result_payload"] = result_payload
    return merged


def build_mgate_scene_state_payload(
    *,
    series_name: str,
    episode_id: str,
    scene_id: str,
    group_id: str,
    tos_summary: dict[str, Any],
    submitted_at: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "series_name": series_name,
        "episode_id": episode_id,
        "scene_id": scene_id,
        "reference_mode": "asset",
        "asset_provider": MODEL_GATE_PROVIDER_GATEWAY,
        "group_id": group_id,
        "submitted_at": submitted_at,
        "tos_summary": tos_summary,
        "items": items,
    }


def persist_mgate_scene_state(episode_dir: Path, scene_id: str, payload: dict[str, Any]) -> None:
    write_scene_state(scene_submission_state_path(episode_dir, scene_id), payload)
    write_scene_state(scene_mgate_state_path(episode_dir, scene_id), payload)


def normalize_local_path_key(value: str | Path | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve())
    except Exception:
        return text


def mgate_reuse_keys_for_item(item: dict[str, Any]) -> list[str]:
    local_path = normalize_local_path_key(item.get("local_path"))
    local_sha256 = str(item.get("local_sha256") or "").strip()
    remote_url = str(item.get("remote_url") or "").strip()
    keys: list[str] = []
    if local_path and local_sha256:
        keys.append(f"path_sha::{local_path}::{local_sha256}")
    if remote_url:
        keys.append(f"remote_url::{remote_url}")
    if local_path:
        keys.append(f"path::{local_path}")
    return keys


def build_cached_mgate_asset_item(
    *,
    uploaded_item: dict[str, Any],
    asset_uri: str,
    group_id: str,
    source: str,
) -> dict[str, Any]:
    asset_id = extract_asset_id(asset_uri)
    payload = {
        **uploaded_item,
        "reference_mode": "asset",
        "asset_provider": MODEL_GATE_PROVIDER_GATEWAY,
        "group_id": group_id,
        "task_id": "",
        "asset_id": asset_id,
        "asset_uri": asset_uri,
        "model_gate_status": "Active",
        "submitted_at": now_iso(),
        "matched_from": source,
    }
    return payload


def find_reusable_asset_cache_entry(
    *,
    uploaded_item: dict[str, Any],
    asset_cache: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if should_force_rereview_reference(uploaded_item):
        return None
    local_path_key = normalize_local_path_key(uploaded_item.get("local_path"))
    if not local_path_key:
        return None
    candidates: list[dict[str, Any]] = []
    for key in (local_path_key, str(uploaded_item.get("local_path") or "").strip()):
        if key and key in asset_cache and isinstance(asset_cache[key], dict):
            candidates.append(asset_cache[key])
    if not candidates:
        return None

    uploaded_sha256 = str(uploaded_item.get("local_sha256") or "").strip()
    for candidate in candidates:
        asset_uri = str(candidate.get("asset_uri") or "").strip()
        if not asset_uri:
            continue
        candidate_sha256 = str(candidate.get("local_sha256") or "").strip()
        candidate_local_path = normalize_local_path_key(candidate.get("local_path"))
        if candidate_local_path and candidate_local_path != local_path_key:
            continue
        if uploaded_sha256 and candidate_sha256 and uploaded_sha256 != candidate_sha256:
            continue
        return candidate
    return None


def build_episode_mgate_reuse_indexes(
    *,
    episode_dir: Path,
    preferred_group_id: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    completed_asset_index: dict[str, dict[str, Any]] = {}
    pending_task_index: dict[str, dict[str, Any]] = {}
    state_paths = sorted(
        {
            *episode_dir.glob(f"P*{MODEL_GATE_STATE_SUFFIX}"),
            *episode_dir.glob(f"P*{ASSET_SUBMISSION_SCENE_SUFFIX}"),
        },
        key=lambda path: path.stat().st_mtime_ns if path.exists() else 0,
    )
    for path in state_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("asset_provider") or "").strip() != MODEL_GATE_PROVIDER_GATEWAY:
            continue
        payload_group_id = str(payload.get("group_id") or "").strip()
        items = [dict(item) for item in list(payload.get("items") or []) if isinstance(item, dict)]
        for item in items:
            keys = mgate_reuse_keys_for_item(item)
            if not keys:
                continue
            asset_uri = str(item.get("asset_uri") or "").strip()
            task_id = str(item.get("task_id") or "").strip()
            status = str(item.get("model_gate_status") or "").strip()
            if asset_uri:
                for key in keys:
                    completed_asset_index[key] = item
                continue
            if task_id and payload_group_id and payload_group_id == preferred_group_id and not is_mgate_terminal_failed_status(status):
                for key in keys:
                    pending_task_index[key] = item
    return completed_asset_index, pending_task_index


def find_reusable_mgate_item(
    *,
    uploaded_item: dict[str, Any],
    reuse_index: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for key in mgate_reuse_keys_for_item(uploaded_item):
        item = reuse_index.get(key)
        if item is not None:
            return item
    return None


def register_mgate_reuse_item(index: dict[str, dict[str, Any]], item: dict[str, Any]) -> None:
    for key in mgate_reuse_keys_for_item(item):
        index[key] = item


def count_pending_mgate_task_ids(index: dict[str, dict[str, Any]]) -> int:
    task_ids = {
        str(item.get("task_id") or "").strip()
        for item in index.values()
        if isinstance(item, dict)
        and str(item.get("task_id") or "").strip()
        and not str(item.get("asset_uri") or "").strip()
    }
    return len(task_ids)


def model_gate_access_key() -> str:
    value = str(os.environ.get("MODEL_GATE_ACCESS_KEY") or "").strip()
    if not value:
        raise RuntimeError("未设置环境变量 MODEL_GATE_ACCESS_KEY。")
    return value


def model_gate_secret_key() -> str:
    value = str(os.environ.get("MODEL_GATE_SECRET_KEY") or "").strip()
    if not value:
        raise RuntimeError("未设置环境变量 MODEL_GATE_SECRET_KEY。")
    return value


def compute_model_gate_signature(*, method: str, path: str, timestamp: str, body_text: str) -> str:
    signer_script = str(os.environ.get(MODEL_GATE_SIGNATURE_SCRIPT_ENV) or "").strip()
    if signer_script:
        payload = {
            "method": method.upper(),
            "path": path,
            "timestamp": timestamp,
            "body": body_text,
            "access_key": model_gate_access_key(),
            "secret_key": model_gate_secret_key(),
        }
        completed = subprocess.run(
            [signer_script],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=False,
        )
        signature = completed.stdout.strip()
        if completed.returncode != 0 or not signature:
            stderr = completed.stderr.strip()
            raise RuntimeError(
                f"Model Gate 签名脚本执行失败：{signer_script}"
                + (f" / {stderr}" if stderr else "")
            )
        return signature

    mode = str(os.environ.get(MODEL_GATE_SIGNATURE_MODE_ENV) or "").strip() or MODEL_GATE_SIGNATURE_DEFAULT_MODE
    if mode not in MODEL_GATE_SIGNATURE_MODES:
        raise RuntimeError(
            "缺少可用的 Model Gate 签名实现。请设置 "
            f"{MODEL_GATE_SIGNATURE_SCRIPT_ENV}=<可执行脚本>，"
            "或设置 "
            f"{MODEL_GATE_SIGNATURE_MODE_ENV}="
            "timestamp_body_hmac_sha256_hex / "
            "timestamp_method_path_body_hmac_sha256_hex / "
            "timestamp_method_path_body_hmac_sha256_base64。"
        )

    if mode == "timestamp_body_hmac_sha256_hex":
        canonical = "\n".join([timestamp, body_text])
    else:
        canonical = "\n".join([timestamp, method.upper(), path, body_text])
    digest = hmac.new(
        model_gate_secret_key().encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    if mode.endswith("_base64"):
        return base64.b64encode(digest).decode("utf-8")
    return digest.hex()


def model_gate_request_json(path: str, payload: dict[str, Any], timeout_seconds: int = 30) -> dict[str, Any]:
    body_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    for attempt in range(1, MODEL_GATE_MAX_RETRIES + 1):
        now = time.time()
        next_allowed_at = _MODEL_GATE_NEXT_ALLOWED_AT_BY_PATH.get(path, 0.0)
        if next_allowed_at > now:
            wait_seconds = next_allowed_at - now
            print_status(f"Model Gate {path} 正在冷却，等待 {wait_seconds:.1f} 秒后继续。")
            time.sleep(wait_seconds)

        if path == "/ai_router/create_assets" and MODEL_GATE_CREATE_ASSETS_MIN_INTERVAL_SECONDS > 0:
            effective_min_interval_seconds = max(
                MODEL_GATE_CREATE_ASSETS_MIN_INTERVAL_SECONDS,
                _MODEL_GATE_DYNAMIC_MIN_INTERVAL_SECONDS_BY_PATH.get(path, 0.0),
            )
            last_request_at = _MODEL_GATE_LAST_REQUEST_AT_BY_PATH.get(path, 0.0)
            elapsed = time.time() - last_request_at
            if elapsed < effective_min_interval_seconds:
                wait_seconds = effective_min_interval_seconds - elapsed
                print_status(f"Model Gate create_assets 主动节流 {wait_seconds:.1f} 秒，减少撞限流。")
                time.sleep(wait_seconds)

        timestamp = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            "X-Access-Key": model_gate_access_key(),
            "X-Access-Timestamp": timestamp,
            "X-Access-Signature": compute_model_gate_signature(
                method="POST",
                path=path,
                timestamp=timestamp,
                body_text=body_text,
            ),
        }
        request = urllib.request.Request(
            f"{MODEL_GATE_BASE_URL}{path}",
            data=body_text.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            _MODEL_GATE_LAST_REQUEST_AT_BY_PATH[path] = time.time()
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                if path == "/ai_router/create_assets":
                    dynamic_interval_seconds = _MODEL_GATE_DYNAMIC_MIN_INTERVAL_SECONDS_BY_PATH.get(path, 0.0)
                    if dynamic_interval_seconds > MODEL_GATE_CREATE_ASSETS_MIN_INTERVAL_SECONDS:
                        relaxed_interval_seconds = max(
                            MODEL_GATE_CREATE_ASSETS_MIN_INTERVAL_SECONDS,
                            dynamic_interval_seconds * 0.85,
                        )
                        _MODEL_GATE_DYNAMIC_MIN_INTERVAL_SECONDS_BY_PATH[path] = relaxed_interval_seconds
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            concurrency_limit_hit = (
                exc.code == 429
                and path == "/ai_router/create_assets"
                and "并发数超限" in error_body
            )
            if exc.code in MODEL_GATE_RETRYABLE_STATUS_CODES and attempt < MODEL_GATE_MAX_RETRIES:
                retry_after_header = ""
                if exc.headers is not None:
                    retry_after_header = str(exc.headers.get("Retry-After") or "").strip()
                try:
                    retry_after_seconds = float(retry_after_header) if retry_after_header else 0.0
                except ValueError:
                    retry_after_seconds = 0.0
                backoff_base_seconds = (
                    MODEL_GATE_CREATE_ASSETS_CONCURRENCY_BACKOFF_BASE_SECONDS
                    if concurrency_limit_hit
                    else MODEL_GATE_BACKOFF_BASE_SECONDS
                )
                backoff_cap_seconds = (
                    MODEL_GATE_CREATE_ASSETS_CONCURRENCY_BACKOFF_MAX_SECONDS
                    if concurrency_limit_hit
                    else MODEL_GATE_BACKOFF_MAX_SECONDS
                )
                jitter_seconds = (
                    random.uniform(0.0, MODEL_GATE_BACKOFF_JITTER_SECONDS)
                    if MODEL_GATE_BACKOFF_JITTER_SECONDS > 0
                    else 0.0
                )
                backoff_seconds = min(
                    backoff_cap_seconds,
                    max(
                        retry_after_seconds,
                        backoff_base_seconds * (2 ** (attempt - 1)),
                    )
                    + jitter_seconds,
                )
                print_status(
                    f"Model Gate 请求被限流/暂不可用（状态码 {exc.code}，第 {attempt}/{MODEL_GATE_MAX_RETRIES} 次），"
                    f"{backoff_seconds:.1f} 秒后重试：{path}"
                    + (" [并发数超限保护]" if concurrency_limit_hit else "")
                )
                if concurrency_limit_hit:
                    current_dynamic_interval_seconds = _MODEL_GATE_DYNAMIC_MIN_INTERVAL_SECONDS_BY_PATH.get(path, 0.0)
                    increased_interval_seconds = min(
                        10.0,
                        max(
                            current_dynamic_interval_seconds * 1.5 if current_dynamic_interval_seconds > 0 else 0.0,
                            MODEL_GATE_CREATE_ASSETS_MIN_INTERVAL_SECONDS + 1.5,
                            3.0,
                        ),
                    )
                    if increased_interval_seconds > current_dynamic_interval_seconds:
                        _MODEL_GATE_DYNAMIC_MIN_INTERVAL_SECONDS_BY_PATH[path] = increased_interval_seconds
                        print_status(
                            f"Model Gate create_assets 后续最小提交间隔提升到 {increased_interval_seconds:.1f} 秒，"
                            "继续避开并发保护。"
                        )
                _MODEL_GATE_NEXT_ALLOWED_AT_BY_PATH[path] = time.time() + backoff_seconds
                time.sleep(backoff_seconds)
                continue
            if concurrency_limit_hit:
                raise ModelGateConcurrencyLimitError(
                    "Model Gate create_assets 仍处于并发受限状态；"
                    "当前更适合先轮询已有任务，等部分任务完成后再继续提交新素材。"
                ) from exc
            raise RuntimeError(f"Model Gate 请求失败，状态码 {exc.code}，响应：{error_body}") from exc
        except urllib.error.URLError as exc:
            if attempt < MODEL_GATE_MAX_RETRIES:
                jitter_seconds = (
                    random.uniform(0.0, MODEL_GATE_BACKOFF_JITTER_SECONDS)
                    if MODEL_GATE_BACKOFF_JITTER_SECONDS > 0
                    else 0.0
                )
                backoff_seconds = min(
                    MODEL_GATE_BACKOFF_MAX_SECONDS,
                    MODEL_GATE_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + jitter_seconds,
                )
                print_status(
                    f"Model Gate 网络请求失败（第 {attempt}/{MODEL_GATE_MAX_RETRIES} 次）：{exc}；"
                    f"{backoff_seconds:.1f} 秒后重试。"
                )
                _MODEL_GATE_NEXT_ALLOWED_AT_BY_PATH[path] = time.time() + backoff_seconds
                time.sleep(backoff_seconds)
                continue
            raise RuntimeError(f"Model Gate 网络请求失败：{exc}") from exc
    raise RuntimeError(f"Model Gate 请求重试后仍失败：{path}")


def create_model_gate_group(*, series_dir: Path, episode_dir: Path) -> str:
    group_name = f"{series_dir.name}_{episode_dir.name}_seedance_refs"
    payload = {
        "Name": group_name,
        "Description": f"{series_dir.name} {episode_dir.name} Seedance 真人素材组",
        "GroupType": MODEL_GATE_GROUP_TYPE,
    }
    response = model_gate_request_json("/ai_router/create_asset_groups", payload)
    group_id = extract_asset_id(str((response.get("Result") or {}).get("Id") or "")) or str((response.get("Result") or {}).get("Id") or "").strip()
    if not group_id:
        raise RuntimeError(f"创建 Model Gate 素材组失败，未返回 group id：{json.dumps(response, ensure_ascii=False)}")
    return group_id


def create_model_gate_asset_task(*, group_id: str, url: str, asset_name: str) -> str:
    payload = {
        "GroupId": group_id,
        "URL": url,
        "AssetType": "Image",
        "Name": asset_name,
    }
    response = model_gate_request_json("/ai_router/create_assets", payload)
    task_id = str(response.get("Id") or (response.get("Result") or {}).get("Id") or "").strip()
    if not task_id:
        raise RuntimeError(f"提交 Model Gate 素材失败，未返回 task id：{json.dumps(response, ensure_ascii=False)}")
    return task_id


def get_model_gate_asset_result(task_id: str) -> dict[str, Any]:
    response = model_gate_request_json("/ai_router/get_assets", {"task_id": task_id})
    result = response.get("Result")
    return result if isinstance(result, dict) else {}


def choose_mgate_action() -> str:
    options = [
        ("submit", "创建/复用素材组，并提交当前引用图"),
        ("poll", "查询 task_id 并生成 asset:// 映射"),
        ("both", "提交后立即轮询，尽量直接生成 asset:// 映射"),
    ]
    selected_index = choose_from_list(
        "请选择 Model Gate 模式要执行的动作：",
        [label for _, label in options],
        default_index=2,
    )
    return options[selected_index][0]


def prompt_mgate_poll_settings() -> tuple[int, float]:
    max_attempts = int(prompt_with_default("请输入 Model Gate 查询轮询次数（输入 0 表示一直查到完成）", "0"))
    interval_seconds = float(prompt_with_default("请输入 Model Gate 轮询间隔秒数", "5"))
    return max(0, max_attempts), max(1.0, interval_seconds)


def collect_tos_uploads_for_mgate(
    *,
    series_dir: Path,
    episode_dir: Path,
    reference_files: list[Path],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    bucket = prompt_with_default("请输入 TOS bucket", DEFAULT_BUCKET)
    region = prompt_with_default("请输入 region", DEFAULT_REGION)
    endpoint = prompt_with_default("请输入 endpoint", DEFAULT_ENDPOINT)
    mode = choose_mode()
    validity = DEFAULT_VALIDITY if mode == "presign" else ""
    if mode == "presign":
        validity = prompt_with_default("请输入预签名有效期", DEFAULT_VALIDITY)

    flags = resolve_credentials_flags()
    if flags:
        print_status("检测到环境变量中的 AK/SK，将直接用于 TOS 中转上传。")
    else:
        print_status("未检测到环境变量 AK/SK，将使用 tosutil 已配置的默认凭证。")

    upload_cache = load_upload_cache(episode_dir)
    uploaded_by_scene: dict[str, list[dict[str, Any]]] = {}
    for references_path in reference_files:
        scene_id = scene_id_from_reference_path(references_path)
        refs = rebuild_references_if_needed(series_dir, episode_dir, references_path)
        uploaded_items: list[dict[str, Any]] = []
        for ref in refs:
            uploaded_items.append(
                upload_one_reference(
                    bucket=bucket,
                    region=region,
                    endpoint=endpoint,
                    validity=validity,
                    mode=mode,
                    series_name=series_dir.name,
                    episode_id=episode_dir.name,
                    scene_id=scene_id,
                    ref=ref,
                    flags=flags,
                    upload_cache=upload_cache,
                    force_reupload=should_force_reupload_reference(ref),
                )
            )
        uploaded_by_scene[scene_id] = uploaded_items

    cache_path = save_upload_cache(episode_dir, upload_cache)
    return (
        {
            "bucket": bucket,
            "region": region,
            "endpoint": endpoint,
            "mode": mode,
            "validity": validity,
            "upload_cache_path": str(cache_path),
        },
        uploaded_by_scene,
    )


def normalize_asset_name(raw: str) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", raw).strip("_")
    return slug[:80] or "seedance_asset"


def submit_mgate_assets(
    *,
    series_dir: Path,
    episode_dir: Path,
    reference_files: list[Path],
) -> dict[str, Any]:
    workflow_payload = load_reference_workflow_config(episode_dir)
    selected_scene_ids = [scene_id_from_reference_path(path) for path in reference_files]
    group_id = str(workflow_payload.get("mgate_group_id") or "").strip()
    if group_id:
        print_status(f"复用已有 Model Gate 素材组：{group_id}")
    else:
        group_id = detect_existing_mgate_group_id(episode_dir, selected_scene_ids)
        if group_id:
            print_status(f"从已落盘的场景状态恢复 Model Gate 素材组：{group_id}")
        else:
            group_id = create_model_gate_group(series_dir=series_dir, episode_dir=episode_dir)
            print_status(f"已创建 Model Gate 素材组：{group_id}")
    save_reference_workflow_config(
        episode_dir=episode_dir,
        series_name=series_dir.name,
        episode_id=episode_dir.name,
        reference_mode="asset",
        selected_scene_ids=selected_scene_ids,
        asset_provider=MODEL_GATE_PROVIDER_GATEWAY,
        extra_fields={"mgate_group_id": group_id},
    )

    tos_summary, uploaded_by_scene = collect_tos_uploads_for_mgate(
        series_dir=series_dir,
        episode_dir=episode_dir,
        reference_files=reference_files,
    )
    asset_cache = load_asset_result_cache(episode_dir)
    completed_asset_reuse_index, pending_task_reuse_index = build_episode_mgate_reuse_indexes(
        episode_dir=episode_dir,
        preferred_group_id=group_id,
    )

    scene_summaries: list[dict[str, Any]] = []
    submission_paused = False
    paused_scene_id = ""
    paused_reason = ""
    for references_path in reference_files:
        scene_id = scene_id_from_reference_path(references_path)
        uploaded_items = uploaded_by_scene.get(scene_id) or []
        existing_state_payload = load_mgate_scene_state(episode_dir, scene_id)
        existing_items = [
            dict(item)
            for item in list(existing_state_payload.get("items") or [])
            if isinstance(item, dict)
        ]
        existing_by_identity = {
            scene_state_identity(item): item
            for item in existing_items
            if any(scene_state_identity(item))
        }
        state_items: list[dict[str, Any]] = []
        scene_submitted_at = str(existing_state_payload.get("submitted_at") or "").strip() or now_iso()
        submitted_task_count = 0
        reused_task_count = 0
        reused_asset_count = 0
        for uploaded_item in uploaded_items:
            force_rereview = should_force_rereview_reference(uploaded_item)
            existing_item = existing_by_identity.get(scene_state_identity(uploaded_item))
            if force_rereview and existing_item:
                print_status(
                    f"{scene_id} 场景参考强制重新送审：跳过历史状态复用 "
                    f"{existing_item.get('label') or uploaded_item.get('label')}"
                )
            if not force_rereview and existing_item and (
                str(existing_item.get("task_id") or "").strip()
                or str(existing_item.get("asset_uri") or "").strip()
            ) and not is_mgate_terminal_failed_status(str(existing_item.get("model_gate_status") or "")):
                merged_item = merge_uploaded_item_with_existing_mgate_state(
                    uploaded_item=uploaded_item,
                    existing_item=existing_item,
                    group_id=group_id,
                )
                state_items.append(merged_item)
                if mgate_item_has_asset(merged_item):
                    reused_asset_count += 1
                    register_mgate_reuse_item(completed_asset_reuse_index, merged_item)
                elif str(merged_item.get("task_id") or "").strip():
                    reused_task_count += 1
                    register_mgate_reuse_item(pending_task_reuse_index, merged_item)
                print_status(
                    f"{scene_id} 复用已有 Model Gate 提交：{merged_item.get('label') or uploaded_item.get('label')}"
                    f" -> {mgate_item_display_ref(merged_item)}"
                )
                continue

            if not force_rereview and existing_item and is_mgate_terminal_failed_status(str(existing_item.get("model_gate_status") or "")):
                print_status(
                    f"{scene_id} 检测到历史失败任务，改为重新提交：{existing_item.get('label') or uploaded_item.get('label')}"
                )

            cached_asset_entry = find_reusable_asset_cache_entry(
                uploaded_item=uploaded_item,
                asset_cache=asset_cache,
            )
            if cached_asset_entry is not None:
                cached_asset_item = build_cached_mgate_asset_item(
                    uploaded_item=uploaded_item,
                    asset_uri=str(cached_asset_entry.get("asset_uri") or "").strip(),
                    group_id=group_id,
                    source="episode_asset_cache",
                )
                state_items.append(cached_asset_item)
                reused_asset_count += 1
                register_mgate_reuse_item(completed_asset_reuse_index, cached_asset_item)
                print_status(
                    f"{scene_id} 复用已缓存 asset_id：{cached_asset_item.get('label') or uploaded_item.get('label')}"
                    f" -> {cached_asset_item.get('asset_uri')}"
                )
                continue

            reusable_asset_item = None if force_rereview else find_reusable_mgate_item(
                uploaded_item=uploaded_item,
                reuse_index=completed_asset_reuse_index,
            )
            if reusable_asset_item is not None and str(reusable_asset_item.get("asset_uri") or "").strip():
                merged_item = merge_uploaded_item_with_existing_mgate_state(
                    uploaded_item=uploaded_item,
                    existing_item=reusable_asset_item,
                    group_id=group_id,
                )
                state_items.append(merged_item)
                reused_asset_count += 1
                register_mgate_reuse_item(completed_asset_reuse_index, merged_item)
                print_status(
                    f"{scene_id} 复用已存在 asset_id：{merged_item.get('label') or uploaded_item.get('label')}"
                    f" -> {merged_item.get('asset_uri')}"
                )
                continue

            reusable_pending_item = None if force_rereview else find_reusable_mgate_item(
                uploaded_item=uploaded_item,
                reuse_index=pending_task_reuse_index,
            )
            if reusable_pending_item is not None and str(reusable_pending_item.get("task_id") or "").strip():
                merged_item = merge_uploaded_item_with_existing_mgate_state(
                    uploaded_item=uploaded_item,
                    existing_item=reusable_pending_item,
                    group_id=group_id,
                )
                state_items.append(merged_item)
                reused_task_count += 1
                register_mgate_reuse_item(pending_task_reuse_index, merged_item)
                print_status(
                    f"{scene_id} 复用已有待处理任务：{merged_item.get('label') or uploaded_item.get('label')}"
                    f" -> {merged_item.get('task_id')}"
                )
                continue

            current_pending_task_count = count_pending_mgate_task_ids(pending_task_reuse_index)
            if MODEL_GATE_MAX_PENDING_TASKS > 0 and current_pending_task_count >= MODEL_GATE_MAX_PENDING_TASKS:
                submission_paused = True
                paused_scene_id = scene_id
                paused_reason = (
                    f"当前已有 {current_pending_task_count} 个 Model Gate 待处理任务，"
                    f"达到本地安全上限 {MODEL_GATE_MAX_PENDING_TASKS}；先暂停继续提交，建议先执行 poll。"
                )
                print_status(paused_reason)
                break

            asset_name = normalize_asset_name(
                Path(str(uploaded_item.get("local_path") or uploaded_item.get("label") or scene_id)).stem
            )
            try:
                task_id = create_model_gate_asset_task(
                    group_id=group_id,
                    url=str(uploaded_item.get("remote_url") or ""),
                    asset_name=asset_name,
                )
            except ModelGateConcurrencyLimitError as exc:
                submission_paused = True
                paused_scene_id = scene_id
                paused_reason = str(exc)
                print_status(f"{scene_id} 暂停继续提交：{paused_reason}")
                break
            state_items.append(
                {
                    **uploaded_item,
                    "reference_mode": "asset",
                    "asset_provider": MODEL_GATE_PROVIDER_GATEWAY,
                    "group_id": group_id,
                    "task_id": task_id,
                    "asset_id": "",
                    "asset_uri": "",
                    "model_gate_status": "submitted",
                    "submitted_at": now_iso(),
                }
            )
            submitted_task_count += 1
            register_mgate_reuse_item(pending_task_reuse_index, state_items[-1])
            state_payload = build_mgate_scene_state_payload(
                series_name=series_dir.name,
                episode_id=episode_dir.name,
                scene_id=scene_id,
                group_id=group_id,
                tos_summary=tos_summary,
                submitted_at=scene_submitted_at,
                items=state_items,
            )
            persist_mgate_scene_state(episode_dir, scene_id, state_payload)

        if submission_paused and uploaded_items and not state_items:
            print_status(
                f"{scene_id} 本轮尚未提交任何 Model Gate 引用，保留当前场景状态不变；"
                "可先 poll 已提交任务，稍后再从该场景继续 submit。"
            )
            break

        state_payload = build_mgate_scene_state_payload(
            series_name=series_dir.name,
            episode_id=episode_dir.name,
            scene_id=scene_id,
            group_id=group_id,
            tos_summary=tos_summary,
            submitted_at=scene_submitted_at,
            items=state_items,
        )
        persist_mgate_scene_state(episode_dir, scene_id, state_payload)
        scene_summaries.append(
            {
                "scene_id": scene_id,
                "group_id": group_id,
                "reference_count": len(state_items),
                "submitted_task_count": submitted_task_count,
                "reused_task_count": reused_task_count,
                "reused_asset_count": reused_asset_count,
                "task_ids": [
                    str(item.get("task_id") or "")
                    for item in state_items
                    if str(item.get("task_id") or "").strip() and not mgate_item_has_asset(item)
                ],
                "state_path": str(scene_mgate_state_path(episode_dir, scene_id)),
            }
        )
        print_status(
            f"{scene_id} Model Gate 引用处理完成：共 {len(state_items)} 张；"
            f"新提交 {submitted_task_count} 张，复用待处理任务 {reused_task_count} 张，"
            f"复用既有 asset_id {reused_asset_count} 张。"
        )
        if submission_paused:
            break

    save_reference_workflow_config(
        episode_dir=episode_dir,
        series_name=series_dir.name,
        episode_id=episode_dir.name,
        reference_mode="asset",
        selected_scene_ids=selected_scene_ids,
        asset_provider=MODEL_GATE_PROVIDER_GATEWAY,
        extra_fields={"mgate_group_id": group_id},
    )
    return {
        "series_name": series_dir.name,
        "episode_id": episode_dir.name,
        "reference_mode": "asset",
        "asset_provider": MODEL_GATE_PROVIDER_GATEWAY,
        "group_id": group_id,
        "tos_summary": tos_summary,
        "submission_paused": submission_paused,
        "paused_scene_id": paused_scene_id,
        "paused_reason": paused_reason,
        "pending_task_count": count_pending_mgate_task_ids(pending_task_reuse_index),
        "scene_range": {
            "start": scene_id_from_reference_path(reference_files[0]),
            "end": scene_id_from_reference_path(reference_files[-1]),
            "count": len(reference_files),
        },
        "results": scene_summaries,
    }


def poll_mgate_assets(
    *,
    series_dir: Path,
    episode_dir: Path,
    reference_files: list[Path],
    max_attempts: int,
    interval_seconds: float,
) -> dict[str, Any]:
    asset_cache = load_asset_result_cache(episode_dir)
    summaries: list[dict[str, Any]] = []
    pending_scene_ids: list[str] = []
    failed_scene_ids: list[str] = []
    scene_records: list[dict[str, Any]] = []
    skipped_scenes: list[dict[str, Any]] = []

    for references_path in reference_files:
        scene_id = scene_id_from_reference_path(references_path)
        state_payload = load_mgate_scene_state(episode_dir, scene_id)
        if not state_payload:
            reason = "未找到 Model Gate 提交状态，说明该场景还没有提交到 Model Gate。"
            skipped_scenes.append({"scene_id": scene_id, "reason": reason})
            print_status(f"{scene_id} 跳过查询：{reason}")
            continue

        items = extract_mgate_state_items(state_payload)
        if not items:
            reason = (
                "提交状态为空，通常表示该场景尚未真正提交任何素材，"
                "例如上一轮在待处理任务上限前被暂停。"
            )
            skipped_scenes.append({"scene_id": scene_id, "reason": reason})
            print_status(f"{scene_id} 跳过查询：{reason}")
            continue

        scene_records.append(
            {
                "references_path": references_path,
                "scene_id": scene_id,
                "state_payload": state_payload,
                "items": items,
            }
        )

    if not scene_records:
        detail = "；".join(f"{item['scene_id']}: {item['reason']}" for item in skipped_scenes[:3])
        if len(skipped_scenes) > 3:
            detail += "；..."
        raise RuntimeError(
            "当前所选场景没有可查询的 Model Gate 提交状态。"
            + (f" {detail}" if detail else "")
        )

    pending_slots: list[tuple[int, int]] = []
    for scene_index, record in enumerate(scene_records):
        for item_index, item in enumerate(record["items"]):
            if not str(item.get("asset_uri") or "").strip():
                pending_slots.append((scene_index, item_index))

    started_at = time.time()
    attempt = 0
    stagnation_detected = False
    stagnation_reason = ""
    while pending_slots:
        attempt += 1
        if max_attempts > 0 and attempt > max_attempts:
            break
        elapsed_seconds = time.time() - started_at
        if MODEL_GATE_POLL_MAX_WAIT_SECONDS > 0 and elapsed_seconds >= MODEL_GATE_POLL_MAX_WAIT_SECONDS:
            print_status(
                f"Model Gate 轮询已达到安全等待上限 {MODEL_GATE_POLL_MAX_WAIT_SECONDS:.0f} 秒，"
                "本轮先停止，保留 pending 状态，稍后可继续 poll。"
            )
            break
        if not pending_slots:
            break
        print_status(
            f"Model Gate 轮询第 {attempt}/"
            f"{max_attempts if max_attempts > 0 else '直到完成'} 轮：待查询任务 {len(pending_slots)} 个。"
        )
        next_pending_slots: list[tuple[int, int]] = []
        result_payload_by_task_id: dict[str, dict[str, Any]] = {}
        for scene_index, item_index in pending_slots:
            record = scene_records[scene_index]
            item = record["items"][item_index]
            asset_uri = str(item.get("asset_uri") or "").strip()
            if asset_uri:
                continue
            task_id = str(item.get("task_id") or "").strip()
            if not task_id:
                item["model_gate_status"] = str(item.get("model_gate_status") or "").strip() or "Pending"
                item["updated_at"] = now_iso()
                next_pending_slots.append((scene_index, item_index))
                continue
            result_payload = result_payload_by_task_id.get(task_id)
            if result_payload is None:
                result_payload = get_model_gate_asset_result(task_id)
                result_payload_by_task_id[task_id] = result_payload
            status = str(result_payload.get("Status") or "").strip()
            asset_id = extract_asset_id(result_payload.get("Id"))
            remote_update_time = str(result_payload.get("UpdateTime") or "").strip()
            previous_payload = item.get("result_payload")
            previous_status = ""
            previous_asset_id = ""
            previous_remote_update_time = ""
            if isinstance(previous_payload, dict):
                previous_status = str(previous_payload.get("Status") or "").strip()
                previous_asset_id = extract_asset_id(previous_payload.get("Id"))
                previous_remote_update_time = str(previous_payload.get("UpdateTime") or "").strip()
            if asset_id:
                item["asset_id"] = asset_id
                item["asset_uri"] = to_asset_uri(asset_id)
                item["model_gate_status"] = status or "Active"
                item["result_payload"] = result_payload
                item["mgate_poll_stagnant_rounds"] = 0
                item["updated_at"] = now_iso()
                continue
            item["model_gate_status"] = status or "Pending"
            item["result_payload"] = result_payload
            if is_mgate_terminal_failed_status(status):
                item["mgate_poll_stagnant_rounds"] = 0
                item["updated_at"] = now_iso()
                continue
            if (
                previous_status == status
                and previous_asset_id == asset_id
                and previous_remote_update_time == remote_update_time
                and (status or remote_update_time or asset_id)
            ):
                item["mgate_poll_stagnant_rounds"] = int(item.get("mgate_poll_stagnant_rounds") or 0) + 1
            else:
                item["mgate_poll_stagnant_rounds"] = 0
            item["updated_at"] = now_iso()
            next_pending_slots.append((scene_index, item_index))

        pending_slots = next_pending_slots
        if pending_slots:
            stagnant_pending_slots: list[tuple[int, int]] = []
            for scene_index, item_index in pending_slots:
                item = scene_records[scene_index]["items"][item_index]
                if int(item.get("mgate_poll_stagnant_rounds") or 0) >= MODEL_GATE_POLL_MAX_STAGNANT_ROUNDS:
                    stagnant_pending_slots.append((scene_index, item_index))
            if stagnant_pending_slots and len(stagnant_pending_slots) == len(pending_slots):
                labels = [
                    str(scene_records[scene_index]["items"][item_index].get("label") or scene_records[scene_index]["scene_id"])
                    for scene_index, item_index in stagnant_pending_slots[:3]
                ]
                stagnation_detected = True
                stagnation_reason = (
                    f"连续 {MODEL_GATE_POLL_MAX_STAGNANT_ROUNDS} 轮查询均未看到远端状态推进，"
                    "当前任务疑似停滞在 Model Gate 未完成状态。"
                )
                print_status(
                    stagnation_reason
                    + (" 示例：" + " / ".join(labels) if labels else "")
                    + " 本轮先停止，稍后可继续 poll 或改走其他路径。"
                )
                break
        if pending_slots and (max_attempts <= 0 or attempt < max_attempts):
            time.sleep(interval_seconds)

    for record in scene_records:
        references_path = Path(record["references_path"])
        scene_id = str(record["scene_id"])
        state_payload = dict(record["state_payload"])
        api_script_path = episode_dir / references_path.name.replace("__seedance_api_references.json", "__seedance_api.sh")
        env_path = episode_dir / f"{scene_id}__seedance_api_urls.env"
        manifest_path = episode_dir / f"{scene_id}__seedance_api_uploaded_refs.json"
        expected_refs = rebuild_references_if_needed(series_dir, episode_dir, references_path)
        items = [dict(item) for item in list(record["items"])]
        failed_items = [
            item
            for item in items
            if not mgate_item_has_asset(item)
            and is_mgate_terminal_failed_status(str(item.get("model_gate_status") or ""))
        ]
        pending_items = [
            item
            for item in items
            if not mgate_item_has_asset(item)
            and not is_mgate_terminal_failed_status(str(item.get("model_gate_status") or ""))
        ]
        failed_count = len(failed_items)
        pending_count = len(pending_items)
        incomplete_reference_count = max(0, len(expected_refs) - len(items))
        uploaded: list[dict[str, Any]] = []
        for item in items:
            asset_uri = str(item.get("asset_uri") or "").strip()
            if not asset_uri:
                uploaded.append(item)
                continue

            local_path = Path(str(item.get("local_path") or "")).expanduser().resolve()
            if local_path.exists():
                local_sha256 = file_sha256(local_path)
                item["local_sha256"] = local_sha256
                item["local_size_bytes"] = local_path.stat().st_size
                item["local_mtime_ns"] = local_path.stat().st_mtime_ns
                asset_cache[str(local_path)] = {
                    "local_path": str(local_path),
                    "local_sha256": local_sha256,
                    "local_size_bytes": local_path.stat().st_size,
                    "local_mtime_ns": local_path.stat().st_mtime_ns,
                    "asset_uri": asset_uri,
                    "asset_id": extract_asset_id(asset_uri),
                    "scene_id": scene_id,
                    "source_path": f"mgate:{item.get('task_id')}",
                    "updated_at": now_iso(),
                }

            uploaded.append(
                {
                    **item,
                    "mode": "asset",
                    "remote_url": asset_uri,
                    "upload_status": "imported_asset" if str(item.get("model_gate_status") or "") == "Active" else "pending_asset",
                    "matched_from": "mgate:get_assets",
                }
            )

        state_payload["items"] = items
        state_payload["updated_at"] = now_iso()
        write_scene_state(scene_submission_state_path(episode_dir, scene_id), state_payload)
        write_scene_state(scene_mgate_state_path(episode_dir, scene_id), state_payload)

        if pending_count or failed_count or incomplete_reference_count:
            removed_env = remove_if_exists(env_path)
            removed_manifest = remove_if_exists(manifest_path)
            if pending_count:
                pending_scene_ids.append(scene_id)
            if failed_count:
                failed_scene_ids.append(scene_id)
            summaries.append(
                {
                    "scene_id": scene_id,
                    "reference_count": len(expected_refs),
                    "resolved_reference_count": len(items),
                    "pending_count": pending_count,
                    "failed_count": failed_count,
                    "incomplete_reference_count": incomplete_reference_count,
                    "completed_count": len(expected_refs) - pending_count - failed_count - incomplete_reference_count,
                    "stagnant_reference_count": sum(
                        1 for item in items if int(item.get("mgate_poll_stagnant_rounds") or 0) >= MODEL_GATE_POLL_MAX_STAGNANT_ROUNDS
                    ),
                    "failed_labels": [str(item.get("label") or "") for item in failed_items],
                    "state_path": str(scene_mgate_state_path(episode_dir, scene_id)),
                    "stale_env_removed": removed_env,
                    "stale_manifest_removed": removed_manifest,
                }
            )
            if incomplete_reference_count:
                print_status(
                    f"{scene_id} 还有 {incomplete_reference_count} 张引用素材尚未进入 Model Gate 状态；"
                    "先保留场景状态，待继续 submit 后再生成完整 env。"
                )
            if failed_count:
                labels = " / ".join(str(item.get("label") or scene_id) for item in failed_items[:3])
                print_status(
                    f"{scene_id} 有 {failed_count} 张素材返回 Failed，已保留状态；"
                    + (f"示例：{labels}。" if labels else "")
                    + "建议重新提交这些素材或改走其他路径。"
                )
            if pending_count:
                print_status(f"{scene_id} 仍有 {pending_count} 张素材未返回 asset_id，已保留状态，稍后可继续查询。")
            if removed_env:
                print_status(f"{scene_id} 已移除过期的 Model Gate Asset env：{env_path}")
            if removed_manifest:
                print_status(f"{scene_id} 已移除过期的 Model Gate 引用清单：{manifest_path}")
            continue

        env_path.write_text(env_file_text(uploaded), encoding="utf-8")
        manifest_path.write_text(json.dumps(uploaded, ensure_ascii=False, indent=2), encoding="utf-8")
        patched = patch_api_script_if_needed(api_script_path, env_path.name)
        print_status(f"{scene_id} Model Gate Asset env 已写入：{env_path}")
        print_status(f"{scene_id} Model Gate 引用清单已写入：{manifest_path}")
        if patched:
            print_status(f"{scene_id} 已补丁现有 API 脚本以自动读取：{api_script_path}")
        summaries.append(
            {
                "scene_id": scene_id,
                "reference_count": len(expected_refs),
                "resolved_reference_count": len(uploaded),
                "pending_count": 0,
                "env_path": str(env_path),
                "manifest_path": str(manifest_path),
            }
        )

    cache_path = save_asset_result_cache(episode_dir, asset_cache)
    print_status(f"episode 级 Asset 缓存已写入：{cache_path}")
    return {
        "series_name": series_dir.name,
        "episode_id": episode_dir.name,
        "reference_mode": "asset",
        "asset_provider": MODEL_GATE_PROVIDER_GATEWAY,
        "scene_range": {
            "start": scene_id_from_reference_path(reference_files[0]),
            "end": scene_id_from_reference_path(reference_files[-1]),
            "count": len(reference_files),
        },
        "asset_cache_path": str(cache_path),
        "pending_scene_ids": pending_scene_ids,
        "failed_scene_ids": failed_scene_ids,
        "stagnation_detected": stagnation_detected,
        "stagnation_reason": stagnation_reason,
        "skipped_scenes": skipped_scenes,
        "results": summaries,
    }


def run_mgate_asset_flow(
    *,
    series_dir: Path,
    episode_dir: Path,
    reference_files: list[Path],
) -> dict[str, Any]:
    action = choose_mgate_action()
    submit_summary: dict[str, Any] | None = None
    poll_summary: dict[str, Any] | None = None
    skipped_poll_scene_ids: list[str] = []
    deferred_scene_ids: list[str] = []

    if action in {"submit", "both"}:
        submit_summary = submit_mgate_assets(
            series_dir=series_dir,
            episode_dir=episode_dir,
            reference_files=reference_files,
        )
        submitted_scene_ids = {
            str(item.get("scene_id") or "").strip()
            for item in list(submit_summary.get("results") or [])
            if isinstance(item, dict) and str(item.get("scene_id") or "").strip()
        }
        deferred_scene_ids = [
            scene_id_from_reference_path(path)
            for path in reference_files
            if scene_id_from_reference_path(path) not in submitted_scene_ids
        ]
        if deferred_scene_ids:
            submit_summary["deferred_scene_ids"] = deferred_scene_ids
        if action == "submit":
            return {
                "reference_mode": "asset",
                "asset_provider": MODEL_GATE_PROVIDER_GATEWAY,
                "submit_summary": submit_summary,
                "poll_summary": None,
            }

    poll_reference_files = reference_files
    if action == "both" and submit_summary is not None:
        pollable_scene_ids_from_submit = {
            str(item.get("scene_id") or "").strip()
            for item in list(submit_summary.get("results") or [])
            if isinstance(item, dict) and int(item.get("reference_count") or 0) > 0
        }
        poll_reference_files = [
            path
            for path in reference_files
            if scene_id_from_reference_path(path) in pollable_scene_ids_from_submit
            and has_pollable_mgate_state(episode_dir, scene_id_from_reference_path(path))
        ]
        skipped_poll_scene_ids = [
            scene_id_from_reference_path(path)
            for path in reference_files
            if scene_id_from_reference_path(path) not in {
                scene_id_from_reference_path(poll_path) for poll_path in poll_reference_files
            }
        ]
        if not poll_reference_files:
            print_status("本次没有可立即查询的 Model Gate 提交状态，已跳过轮询。")
            return {
                "reference_mode": "asset",
                "asset_provider": MODEL_GATE_PROVIDER_GATEWAY,
                "submit_summary": submit_summary,
                "poll_summary": {
                    "results": [],
                    "pending_scene_ids": [],
                    "skipped_scenes": [
                        {
                            "scene_id": scene_id,
                            "reason": "本轮尚未生成可查询的 Model Gate 提交状态。",
                        }
                        for scene_id in skipped_poll_scene_ids
                    ],
                    "deferred_scene_ids": deferred_scene_ids,
                    "next_step": "待已有任务完成后，可重新运行并选择“查询 task_id 并生成 asset:// 映射”。",
                },
            }
    elif action == "poll":
        poll_reference_files = [
            path
            for path in reference_files
            if has_pollable_mgate_state(episode_dir, scene_id_from_reference_path(path))
        ]
        skipped_poll_scene_ids = [
            scene_id_from_reference_path(path)
            for path in reference_files
            if not has_pollable_mgate_state(episode_dir, scene_id_from_reference_path(path))
        ]
        if not poll_reference_files:
            raise RuntimeError(
                "当前所选范围内没有可查询的 Model Gate 提交状态。"
                " 请先执行提交动作，或把场景范围缩到已提交的场景。"
            )

    max_attempts, interval_seconds = prompt_mgate_poll_settings()
    poll_summary = poll_mgate_assets(
        series_dir=series_dir,
        episode_dir=episode_dir,
        reference_files=poll_reference_files,
        max_attempts=max_attempts,
        interval_seconds=interval_seconds,
    )
    true_skipped_poll_scene_ids = [
        scene_id for scene_id in skipped_poll_scene_ids if scene_id not in set(deferred_scene_ids)
    ]
    if deferred_scene_ids:
        poll_summary["deferred_scene_ids"] = deferred_scene_ids
    if true_skipped_poll_scene_ids:
        poll_summary["requested_but_skipped_scene_ids"] = true_skipped_poll_scene_ids
    return {
        "reference_mode": "asset",
        "asset_provider": MODEL_GATE_PROVIDER_GATEWAY,
        "submit_summary": submit_summary,
        "poll_summary": poll_summary,
    }


def run_asset_flow(
    *,
    series_dir: Path,
    episode_dir: Path,
    reference_files: list[Path],
) -> dict[str, Any]:
    workflow_payload = load_reference_workflow_config(episode_dir)
    default_provider = str(workflow_payload.get("asset_provider") or MODEL_GATE_PROVIDER_MANUAL).strip() or MODEL_GATE_PROVIDER_MANUAL
    asset_provider = choose_asset_provider(default_provider=default_provider)
    save_reference_workflow_config(
        episode_dir=episode_dir,
        series_name=series_dir.name,
        episode_id=episode_dir.name,
        reference_mode="asset",
        selected_scene_ids=[scene_id_from_reference_path(path) for path in reference_files],
        asset_provider=asset_provider,
    )
    print_status(f"Asset 提供方已保存：{asset_provider}")

    if asset_provider == MODEL_GATE_PROVIDER_GATEWAY:
        return run_mgate_asset_flow(
            series_dir=series_dir,
            episode_dir=episode_dir,
            reference_files=reference_files,
        )

    action = choose_asset_action()
    export_summary: dict[str, Any] | None = None
    import_summary: dict[str, Any] | None = None
    inbox_dir = asset_result_inbox_dir(series_dir, episode_dir)

    if action in {"export", "both"}:
        export_summary = export_asset_review_package(
            series_dir=series_dir,
            episode_dir=episode_dir,
            reference_files=reference_files,
        )
        if action == "export":
            return export_summary

    default_result_path = latest_asset_result_candidate(series_dir, episode_dir)
    if action == "both" and default_result_path is None:
        print_status("当前还没有检测到审核回包，已先完成送审包导出。")
        print_status(f"等平台返回后，把回包放到这里最省事：{inbox_dir}")
        print_status("之后重新运行本脚本，选择“导入审核结果并生成 asset:// 映射”即可。")
        return {
            "reference_mode": "asset",
            "asset_provider": MODEL_GATE_PROVIDER_MANUAL,
            "export_summary": export_summary,
            "import_summary": None,
            "result_inbox_dir": str(inbox_dir),
            "next_step": "将审核返回 zip/json 放入 result_inbox_dir 后，重新运行并选择导入。",
        }

    result_source_path = prompt_review_result_path(
        series_dir=series_dir,
        episode_dir=episode_dir,
        default_path=default_result_path,
    )
    if result_source_path is None:
        return {
            "reference_mode": "asset",
            "asset_provider": MODEL_GATE_PROVIDER_MANUAL,
            "export_summary": export_summary,
            "import_summary": None,
            "result_inbox_dir": str(inbox_dir),
            "next_step": "将审核返回 zip/json 放入 result_inbox_dir 后，重新运行并选择导入。",
        }
    import_summary = import_asset_result(
        series_dir=series_dir,
        episode_dir=episode_dir,
        reference_files=reference_files,
        result_source_path=result_source_path,
    )
    return {
        "reference_mode": "asset",
        "asset_provider": MODEL_GATE_PROVIDER_MANUAL,
        "export_summary": export_summary,
        "import_summary": import_summary,
        "result_inbox_dir": str(inbox_dir),
    }


def main() -> None:
    load_project_env_files()
    if FORCE_SCENE_REFS_REUPLOAD or FORCE_SCENE_REFS_REREVIEW:
        print_status(
            "当前策略：场景参考默认强制重传/重审，不复用历史 URL、task_id 或 asset_id；人物参考仍可复用。"
        )
    series_dir, episode_dir, reference_files, reference_mode, workflow_path = choose_reference_scope()
    print_status(
        f"已选中 {series_dir.name} / {episode_dir.name} / "
        f"{scene_id_from_reference_path(reference_files[0])}-{scene_id_from_reference_path(reference_files[-1])} / "
        f"mode={reference_mode}"
    )

    if reference_mode == "tos":
        result = run_tos_flow(
            series_dir=series_dir,
            episode_dir=episode_dir,
            reference_files=reference_files,
        )
    else:
        result = run_asset_flow(
            series_dir=series_dir,
            episode_dir=episode_dir,
            reference_files=reference_files,
        )

    if isinstance(result, dict):
        result = {
            "reference_workflow_config_path": str(workflow_path),
            **result,
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
