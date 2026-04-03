#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAUDE_DIR = PROJECT_ROOT / ".claude"
ACTIVE_PATH = CLAUDE_DIR / "CLAUDE.md"
FAST_PATH = CLAUDE_DIR / "CLAUDE-fast.md"
DEFAULT_BACKUP_PATH = CLAUDE_DIR / "CLAUDE.default.md"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_default_backup() -> None:
    if not DEFAULT_BACKUP_PATH.exists():
        shutil.copyfile(ACTIVE_PATH, DEFAULT_BACKUP_PATH)


def status() -> None:
    if not ACTIVE_PATH.exists():
        raise SystemExit(f"缺少活动配置：{ACTIVE_PATH}")

    active_hash = file_hash(ACTIVE_PATH)
    active_label = "custom"
    if FAST_PATH.exists() and active_hash == file_hash(FAST_PATH):
        active_label = "fast"
    elif DEFAULT_BACKUP_PATH.exists() and active_hash == file_hash(DEFAULT_BACKUP_PATH):
        active_label = "default"

    print(f"active={active_label}")
    print(f"active_path={ACTIVE_PATH}")
    print(f"fast_path={FAST_PATH if FAST_PATH.exists() else 'missing'}")
    print(f"default_backup_path={DEFAULT_BACKUP_PATH if DEFAULT_BACKUP_PATH.exists() else 'missing'}")


def switch_to_fast() -> None:
    if not FAST_PATH.exists():
        raise SystemExit(f"缺少 fast 配置：{FAST_PATH}")
    ensure_default_backup()
    shutil.copyfile(FAST_PATH, ACTIVE_PATH)
    print(f"已切换到 fast：{ACTIVE_PATH}")


def switch_to_default() -> None:
    if not DEFAULT_BACKUP_PATH.exists():
        raise SystemExit(f"缺少默认备份：{DEFAULT_BACKUP_PATH}")
    shutil.copyfile(DEFAULT_BACKUP_PATH, ACTIVE_PATH)
    print(f"已恢复默认配置：{ACTIVE_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Switch active .claude/CLAUDE.md between default and fast variants.")
    parser.add_argument("mode", choices=["status", "fast", "default"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "status":
        status()
        return
    if args.mode == "fast":
        switch_to_fast()
        return
    switch_to_default()


if __name__ == "__main__":
    main()
