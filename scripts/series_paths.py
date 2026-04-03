from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


EPISODE_ID_PATTERN = re.compile(r"(ep\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class SeriesPaths:
    project_root: Path
    series_name: str
    episode_id: str | None
    script_path: Path | None
    assets_dir: Path
    outputs_dir: Path
    analysis_dir: Path
    script_dir: Path

    def episode_output_dir(self) -> Path | None:
        if not self.episode_id:
            return None
        return self.outputs_dir / self.episode_id


def infer_episode_id_from_name(name: str) -> str | None:
    match = EPISODE_ID_PATTERN.search(name or "")
    if not match:
        return None
    return match.group(1).lower()


def infer_series_name_from_script_path(script_path: str | Path) -> str:
    resolved = Path(script_path).expanduser().resolve()
    if not resolved.parent.name:
        raise ValueError(f"无法从脚本路径推导剧名：{resolved}")
    return resolved.parent.name


def build_series_paths(
    *,
    project_root: str | Path,
    script_path: str | Path | None = None,
    series_name: str | None = None,
    episode_id: str | None = None,
) -> SeriesPaths:
    root = Path(project_root).expanduser().resolve()
    resolved_script = Path(script_path).expanduser().resolve() if script_path else None
    resolved_series_name = (series_name or "").strip()
    if not resolved_series_name:
        if not resolved_script:
            raise ValueError("series_name 和 script_path 不能同时为空。")
        resolved_series_name = infer_series_name_from_script_path(resolved_script)

    resolved_episode_id = (episode_id or "").strip().lower() or None
    if not resolved_episode_id and resolved_script:
        resolved_episode_id = infer_episode_id_from_name(resolved_script.name)

    return SeriesPaths(
        project_root=root,
        series_name=resolved_series_name,
        episode_id=resolved_episode_id,
        script_path=resolved_script,
        assets_dir=root / "assets" / resolved_series_name,
        outputs_dir=root / "outputs" / resolved_series_name,
        analysis_dir=root / "analysis" / resolved_series_name,
        script_dir=root / "script" / resolved_series_name,
    )
