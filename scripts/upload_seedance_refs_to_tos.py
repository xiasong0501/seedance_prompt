from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_seedance_api_script import (
    MaterialReference,
    OUTPUTS_ROOT,
    build_scene_reference_payload,
    choose_from_list,
    choose_range_from_list,
    episode_sort_key,
    find_storyboard_path,
    list_series_dirs,
    parse_material_table,
    parse_scene_prompts,
    resolve_local_reference_path,
)

TOSUTIL_PATH = PROJECT_ROOT / "tosutil"
DEFAULT_BUCKET = "xiasongseedance"
DEFAULT_REGION = "cn-beijing"
DEFAULT_ENDPOINT = "tos-cn-beijing.volces.com"
DEFAULT_VALIDITY = "7d"
DEFAULT_MODE = "public"
UPLOAD_CACHE_FILENAME = "_seedance_ref_upload_cache.json"


def print_status(message: str) -> None:
    print(f"[seedance-tos] {message}", flush=True)


def find_reference_files(series_dir: Path) -> list[Path]:
    return sorted(series_dir.glob("ep*/P*__seedance_api_references.json"), key=lambda p: (episode_sort_key(p.parent.name), p.name))


def choose_reference_files() -> tuple[Path, Path, list[Path]]:
    series_dirs = list_series_dirs()
    if not series_dirs:
        raise RuntimeError("outputs/ 下没有找到可用剧。")
    series_idx = choose_from_list(
        "请选择要上传 Seedance 参考图的剧：",
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

    ref_files = sorted(
        episode_dir.glob("P*__seedance_api_references.json"),
        key=lambda p: p.name,
    )
    if not ref_files:
        raise RuntimeError(f"{episode_dir} 下没有找到任何 Pxx__seedance_api_references.json。")

    start_idx, end_idx = choose_range_from_list(
        "请选择要上传参考图的场景范围：",
        [f"{episode_dir.name} / {path.name.replace('__seedance_api_references.json', '')}" for path in ref_files],
        default_start=0,
        default_end=0,
    )
    return series_dir, episode_dir, ref_files[start_idx : end_idx + 1]

def prompt_with_default(label: str, default: str) -> str:
    raw = input(f"{label}（默认 {default}）：").strip()
    return raw or default


def choose_mode() -> str:
    options = ["预签名 URL（私有桶可用）", "公开读 URL（默认，更快；对象会设为 public-read）"]
    default_index = 1 if DEFAULT_MODE == "public" else 0
    idx = choose_from_list(
        "请选择 URL 生成方式：",
        options,
        default_index=default_index,
    )
    return "presign" if idx == 0 else "public"


def load_references(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"引用文件格式不正确：{path}")
    return data


def resolve_credentials_flags() -> list[str]:
    mapping = [
        ("VOLC_ACCESS_KEY_ID", "VOLC_SECRET_ACCESS_KEY"),
        ("TOS_ACCESS_KEY_ID", "TOS_SECRET_ACCESS_KEY"),
        ("TOS_AK", "TOS_SK"),
    ]
    for ak_key, sk_key in mapping:
        ak = os.environ.get(ak_key)
        sk = os.environ.get(sk_key)
        if ak and sk:
            return [f"-i={ak}", f"-k={sk}"]
    return []


def run_cmd(command: list[str]) -> str:
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    return proc.stdout


def normalize_local_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_key_for_reference(series_name: str, episode_id: str, scene_id: str, ref: dict[str, Any], local_path: Path) -> str:
    filename = local_path.name
    return f"seedance_refs/{series_name}/{episode_id}/shared/{filename}"


def build_public_url(bucket: str, region: str, object_key: str) -> str:
    return f"https://{bucket}.tos-{region}.volces.com/{quote(object_key, safe='/')}"


def extract_url(text: str) -> str:
    match = re.search(r"https?://\S+", text)
    if not match:
        raise RuntimeError(f"未能从 tosutil 输出中解析 URL：\n{text}")
    return match.group(0).strip()


def recover_local_path(series_name: str, episode_id: str, ref: dict[str, Any]) -> str:
    material = MaterialReference(
        token=ref.get("token", ""),
        token_number=int(ref.get("token_number") or 0),
        material_type=ref.get("material_type", "未知素材"),
        label=ref.get("label", ""),
    )
    recovered = resolve_local_reference_path(series_name, episode_id, material)
    return str(recovered) if recovered else ""


def related_episode_dirs(episode_dir: Path) -> list[Path]:
    series_dir = episode_dir.parent
    current = episode_dir.resolve()
    if not series_dir.exists():
        return [current]
    candidates = [
        path.resolve()
        for path in series_dir.iterdir()
        if path.is_dir()
    ]
    candidates.sort(key=lambda path: episode_sort_key(path.name))
    ordered = [path for path in candidates if path != current]
    ordered.append(current)
    return ordered


def load_upload_cache(episode_dir: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    manifest_paths: list[Path] = []
    for related_episode_dir in related_episode_dirs(episode_dir):
        cache_path = related_episode_dir / UPLOAD_CACHE_FILENAME
        if cache_path.exists():
            manifest_paths.append(cache_path)
        manifest_paths.extend(sorted(related_episode_dir.glob("P*__seedance_api_uploaded_refs.json")))

    seen_paths: set[Path] = set()
    for manifest_path in manifest_paths:
        resolved_manifest_path = manifest_path.resolve()
        if resolved_manifest_path in seen_paths:
            continue
        seen_paths.add(resolved_manifest_path)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        scene_id = manifest_path.name.split("__", 1)[0] if manifest_path.name.startswith("P") else ""
        for item in payload:
            if not isinstance(item, dict):
                continue
            local_path_raw = str(item.get("local_path") or "").strip()
            object_key = str(item.get("object_key") or "").strip()
            if not local_path_raw or not object_key:
                continue
            cache[normalize_local_path(local_path_raw)] = {
                "local_path": normalize_local_path(local_path_raw),
                "local_sha256": str(item.get("local_sha256") or "").strip(),
                "local_size_bytes": item.get("local_size_bytes"),
                "local_mtime_ns": item.get("local_mtime_ns"),
                "bucket": str(item.get("bucket") or "").strip(),
                "region": str(item.get("region") or "").strip(),
                "endpoint": str(item.get("endpoint") or "").strip(),
                "mode": str(item.get("mode") or "").strip(),
                "object_key": object_key,
                "remote_url": str(item.get("remote_url") or "").strip(),
                "scene_id": str(item.get("scene_id") or scene_id).strip(),
                "source_manifest": str(manifest_path),
                "source_episode_id": manifest_path.parent.name,
            }
    return cache


def save_upload_cache(episode_dir: Path, cache: dict[str, dict[str, Any]]) -> Path:
    cache_path = episode_dir / UPLOAD_CACHE_FILENAME
    items = sorted(cache.values(), key=lambda item: (str(item.get("local_path") or ""), str(item.get("object_key") or "")))
    cache_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_path


def build_cached_upload_result(
    *,
    ref: dict[str, Any],
    cache_entry: dict[str, Any],
    bucket: str,
    region: str,
    endpoint: str,
    validity: str,
    mode: str,
    flags: list[str],
) -> dict[str, Any] | None:
    if cache_entry.get("bucket") and cache_entry.get("bucket") != bucket:
        return None
    if cache_entry.get("region") and cache_entry.get("region") != region:
        return None
    if cache_entry.get("endpoint") and cache_entry.get("endpoint") != endpoint:
        return None

    object_key = str(cache_entry.get("object_key") or "").strip()
    if not object_key:
        return None

    if mode == "public":
        remote_url = build_public_url(bucket, region, object_key)
        url_source = "cache-public"
    else:
        target = f"tos://{bucket}/{object_key}"
        presign_cmd = [str(TOSUTIL_PATH), "presign", target, f"-vp={validity}", f"-e={endpoint}", f"-re={region}"]
        presign_cmd.extend(flags)
        presign_output = run_cmd(presign_cmd)
        remote_url = extract_url(presign_output)
        url_source = "cache-presign"

    return {
        **ref,
        "bucket": bucket,
        "region": region,
        "endpoint": endpoint,
        "mode": mode,
        "url_source": url_source,
        "object_key": object_key,
        "remote_url": remote_url,
        "upload_status": "reused_cached",
        "reused_from_scene_id": cache_entry.get("scene_id") or "",
        "local_sha256": cache_entry.get("local_sha256") or ref.get("local_sha256") or "",
        "local_size_bytes": cache_entry.get("local_size_bytes"),
        "local_mtime_ns": cache_entry.get("local_mtime_ns"),
    }


def upload_one_reference(
    *,
    bucket: str,
    region: str,
    endpoint: str,
    validity: str,
    mode: str,
    series_name: str,
    episode_id: str,
    scene_id: str,
    ref: dict[str, Any],
    flags: list[str],
    upload_cache: dict[str, dict[str, Any]],
    force_reupload: bool = False,
) -> dict[str, Any]:
    existing_local_path = (ref.get("local_path") or "").strip()
    recovered_local_path = recover_local_path(series_name, episode_id, ref)
    local_path_raw = existing_local_path
    if recovered_local_path and recovered_local_path != existing_local_path:
        local_path_raw = recovered_local_path
        ref["local_path"] = recovered_local_path
        if existing_local_path:
            print_status(f"已修正 {ref.get('env_var')} 的本地素材路径：{existing_local_path} -> {recovered_local_path}")
        else:
            print_status(f"已自动补全 {ref.get('env_var')} 的本地素材路径：{recovered_local_path}")
    elif not local_path_raw and recovered_local_path:
        local_path_raw = recovered_local_path
        ref["local_path"] = recovered_local_path
        print_status(f"已自动补全 {ref.get('env_var')} 的本地素材路径：{recovered_local_path}")
    local_path = Path(local_path_raw) if local_path_raw else None
    if local_path is None or not local_path.exists():
        raise FileNotFoundError(
            f"{ref.get('env_var')} 未找到可上传的本地素材。标签={ref.get('label')} token={ref.get('token')}"
        )
    local_path = local_path.expanduser().resolve()
    local_stat = local_path.stat()
    local_sha256 = file_sha256(local_path)
    local_path_key = normalize_local_path(local_path)
    ref["local_path"] = local_path_key
    ref["local_sha256"] = local_sha256
    ref["local_size_bytes"] = local_stat.st_size
    ref["local_mtime_ns"] = local_stat.st_mtime_ns

    cached_entry = upload_cache.get(local_path_key)
    if cached_entry is not None and not force_reupload:
        cached_sha256 = str(cached_entry.get("local_sha256") or "").strip()
        cached_size = cached_entry.get("local_size_bytes")
        cached_mtime_ns = cached_entry.get("local_mtime_ns")
        same_file = False
        if cached_sha256:
            same_file = cached_sha256 == local_sha256
        elif cached_size is not None and cached_mtime_ns is not None:
            same_file = int(cached_size) == local_stat.st_size and int(cached_mtime_ns) == local_stat.st_mtime_ns
        else:
            same_file = True
        if same_file:
            reused = build_cached_upload_result(
                ref=ref,
                cache_entry=cached_entry,
                bucket=bucket,
                region=region,
                endpoint=endpoint,
                validity=validity,
                mode=mode,
                flags=flags,
            )
            if reused is not None:
                print_status(
                    f"复用已上传 {ref['env_var']} <- "
                    f"{cached_entry.get('scene_id') or 'cache'} / {Path(local_path_key).name}"
                )
                upload_cache[local_path_key] = {
                    **cached_entry,
                    "local_path": local_path_key,
                    "local_sha256": local_sha256,
                    "local_size_bytes": local_stat.st_size,
                    "local_mtime_ns": local_stat.st_mtime_ns,
                    "bucket": bucket,
                    "region": region,
                    "endpoint": endpoint,
                    "scene_id": str(cached_entry.get("scene_id") or scene_id),
                }
                return reused

    object_key = object_key_for_reference(series_name, episode_id, scene_id, ref, local_path)
    target = f"tos://{bucket}/{object_key}"

    cp_cmd = [str(TOSUTIL_PATH), "cp", str(local_path), target, f"-e={endpoint}", f"-re={region}"]
    cp_cmd.extend(flags)
    if mode == "public":
        cp_cmd.append("-acl=public-read")

    print_status(f"上传 {ref['env_var']} -> {target}")
    run_cmd(cp_cmd)

    if mode == "public":
        url = build_public_url(bucket, region, object_key)
        source = "public-read"
    else:
        presign_cmd = [str(TOSUTIL_PATH), "presign", target, f"-vp={validity}", f"-e={endpoint}", f"-re={region}"]
        presign_cmd.extend(flags)
        presign_output = run_cmd(presign_cmd)
        url = extract_url(presign_output)
        source = "presign"

    uploaded_item = {
        **ref,
        "bucket": bucket,
        "region": region,
        "endpoint": endpoint,
        "mode": mode,
        "url_source": source,
        "object_key": object_key,
        "remote_url": url,
        "upload_status": "uploaded",
        "scene_id": scene_id,
        "local_sha256": local_sha256,
        "local_size_bytes": local_stat.st_size,
        "local_mtime_ns": local_stat.st_mtime_ns,
    }
    upload_cache[local_path_key] = {
        "local_path": local_path_key,
        "local_sha256": local_sha256,
        "local_size_bytes": local_stat.st_size,
        "local_mtime_ns": local_stat.st_mtime_ns,
        "bucket": bucket,
        "region": region,
        "endpoint": endpoint,
        "mode": mode,
        "object_key": object_key,
        "remote_url": url,
        "scene_id": scene_id,
        "source_manifest": "",
    }
    return uploaded_item


def env_file_text(uploaded: list[dict[str, Any]]) -> str:
    lines = [
        "# 自动生成：Seedance 参考图 URL",
        "# 可直接被 Pxx__seedance_api.sh 自动读取",
        "",
    ]
    for item in uploaded:
        lines.append(f"export {item['env_var']}={shlex.quote(item['remote_url'])}")
    lines.append("")
    return "\n".join(lines)


def patch_api_script_if_needed(api_script_path: Path, env_filename: str) -> bool:
    if not api_script_path.exists():
        return False
    text = api_script_path.read_text(encoding="utf-8")
    if "SEEDANCE_URL_ENV=" in text:
        return False
    marker = 'TASK_URL="https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"\n'
    block = (
        marker
        + f'SEEDANCE_URL_ENV="$SCRIPT_DIR/{env_filename}"\n\n'
        + 'if [[ -f "$SEEDANCE_URL_ENV" ]]; then\n'
        + '  # shellcheck disable=SC1090\n'
        + '  set -a\n'
        + '  source "$SEEDANCE_URL_ENV"\n'
        + '  set +a\n'
        + 'fi\n'
    )
    if marker not in text:
        return False
    api_script_path.write_text(text.replace(marker, block, 1), encoding="utf-8")
    return True


def rebuild_references_if_needed(series_dir: Path, episode_dir: Path, references_path: Path) -> list[dict[str, Any]]:
    refs = load_references(references_path)
    if refs:
        changed = False
        for ref in refs:
            existing_local_path = str(ref.get("local_path") or "").strip()
            recovered_local_path = recover_local_path(series_dir.name, episode_dir.name, ref)
            if recovered_local_path and recovered_local_path != existing_local_path:
                ref["local_path"] = recovered_local_path
                changed = True
        if changed:
            references_path.write_text(json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")
            print_status(f"已自动补全旧引用清单中的本地素材路径：{references_path}")
        return refs

    scene_id = references_path.name.split("__", 1)[0]
    storyboard_path = find_storyboard_path(episode_dir)
    if storyboard_path is None:
        return refs

    scenes = parse_scene_prompts(storyboard_path)
    scene = next((item for item in scenes if item.scene_id == scene_id), None)
    if scene is None:
        return refs

    material_table = parse_material_table(storyboard_path)
    rebuilt = build_scene_reference_payload(
        series_name=series_dir.name,
        episode_id=episode_dir.name,
        scene=scene,
        material_table=material_table,
    )
    if rebuilt:
        references_path.write_text(json.dumps(rebuilt, ensure_ascii=False, indent=2), encoding="utf-8")
        print_status(f"已自动重建引用清单：{references_path}")
    return rebuilt



def main() -> None:
    if not TOSUTIL_PATH.exists():
        raise RuntimeError(f"未找到 tosutil：{TOSUTIL_PATH}")

    series_dir, episode_dir, reference_files = choose_reference_files()
    bucket = prompt_with_default("请输入 TOS bucket", DEFAULT_BUCKET)
    region = prompt_with_default("请输入 region", DEFAULT_REGION)
    endpoint = prompt_with_default("请输入 endpoint", f"tos-{region}.volces.com")
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
        scene_id = references_path.name.split("__", 1)[0]
        api_script_path = episode_dir / references_path.name.replace("__seedance_api_references.json", "__seedance_api.sh")
        refs = rebuild_references_if_needed(series_dir, episode_dir, references_path)
        if not refs:
            print_status(f"{scene_id} 引用清单为空，已跳过上传：{references_path}")
            summaries.append({
                "scene_id": scene_id,
                "references_path": str(references_path),
                "uploaded_count": 0,
                "skipped": True,
            })
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
        summaries.append({
            "scene_id": scene_id,
            "references_path": str(references_path),
            "env_path": str(env_path),
            "manifest_path": str(manifest_path),
            "api_script_path": str(api_script_path),
            "uploaded_count": uploaded_count,
            "reference_count": len(uploaded),
            "reused_cached_count": reused_count,
            "skipped": False,
        })

    cache_path = save_upload_cache(episode_dir, upload_cache)
    print_status(f"episode 级上传缓存已写入：{cache_path}")

    print(json.dumps({
        "series_name": series_dir.name,
        "episode_id": episode_dir.name,
        "scene_range": {
            "start": reference_files[0].name.split("__", 1)[0],
            "end": reference_files[-1].name.split("__", 1)[0],
            "count": len(reference_files),
        },
        "bucket": bucket,
        "region": region,
        "endpoint": endpoint,
        "mode": mode,
        "upload_cache_path": str(cache_path),
        "results": summaries,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
