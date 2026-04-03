from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SKILLS_ROOT = PROJECT_ROOT / "skills"


def load_skill(relative_path: str | Path) -> str:
    path = relative_path if isinstance(relative_path, Path) else SKILLS_ROOT / relative_path
    return path.read_text(encoding="utf-8")

