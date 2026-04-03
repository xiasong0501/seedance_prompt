from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPISODE_DIR_PATTERN = re.compile(r'^ep\d+$', re.IGNORECASE)
START_TMPL = '<!-- episode: {episode_id} start -->'
END_TMPL = '<!-- episode: {episode_id} end -->'

HEADERS = {
    'character': '# 人物提示词',
    'scene': '# 场景道具提示词',
}
FILENAMES = {
    'character': 'character-prompts.md',
    'scene': 'scene-prompts.md',
}
SUMMARY_FILENAME = 'claude-assets-migration-summary.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Migrate Claude episode-scoped asset prompts into shared series-level prompt files.')
    parser.add_argument('--assets-dir', action='append', default=[], help='具体的 assets/<剧名>-claude 目录，可多次传入。')
    parser.add_argument('--series-name', action='append', default=[], help='剧名目录名，例如 红糖姜汁-claude，可多次传入。')
    parser.add_argument('--all', action='store_true', help='迁移所有 assets/*-claude 目录。默认在未指定路径时也会扫描全部。')
    parser.add_argument('--dry-run', action='store_true', help='只预览，不写文件。')
    return parser.parse_args()


def discover_assets_dirs(args: argparse.Namespace) -> list[Path]:
    discovered: list[Path] = []
    for raw in args.assets_dir:
        path = Path(raw).expanduser().resolve()
        if path.is_dir() and path not in discovered:
            discovered.append(path)
    for raw in args.series_name:
        path = (PROJECT_ROOT / 'assets' / raw).resolve()
        if path.is_dir() and path not in discovered:
            discovered.append(path)
    if discovered:
        return discovered
    assets_root = PROJECT_ROOT / 'assets'
    for child in sorted(assets_root.iterdir()):
        if child.is_dir() and child.name.endswith('-claude'):
            discovered.append(child)
    return discovered


def episode_sort_key(episode_id: str) -> tuple[int, str]:
    match = re.search(r'(\d+)', episode_id)
    return (int(match.group(1)) if match else 10**9, episode_id)


def iter_episode_dirs(assets_dir: Path) -> list[Path]:
    items = [child for child in assets_dir.iterdir() if child.is_dir() and EPISODE_DIR_PATTERN.match(child.name)]
    return sorted(items, key=lambda item: episode_sort_key(item.name.lower()))


def strip_header(text: str, header: str) -> str:
    body = text.replace('\ufeff', '').strip()
    if not body:
        return ''
    lines = body.splitlines()
    if lines and lines[0].strip() == header:
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return '\n'.join(lines).strip()


def build_episode_block(episode_id: str, body: str) -> str:
    if START_TMPL.format(episode_id=episode_id) in body and END_TMPL.format(episode_id=episode_id) in body:
        return body.strip() + '\n'
    cleaned = body.strip()
    if not cleaned:
        cleaned = f'<!-- {episode_id} 无内容 -->'
    return (
        f"{START_TMPL.format(episode_id=episode_id)}\n\n"
        f"<!-- {episode_id} -->\n\n"
        f"{cleaned}\n\n"
        f"{END_TMPL.format(episode_id=episode_id)}\n"
    )


def merge_episode_block(existing: str, header: str, episode_id: str, block: str) -> str:
    start_marker = START_TMPL.format(episode_id=episode_id)
    end_marker = END_TMPL.format(episode_id=episode_id)
    if not existing.strip():
        return f'{header}\n\n{block}'.rstrip() + '\n'
    pattern = re.compile(re.escape(start_marker) + r'.*?' + re.escape(end_marker) + r'\n?', flags=re.DOTALL)
    if pattern.search(existing):
        new_content = pattern.sub(block, existing)
    else:
        new_content = existing.rstrip() + '\n\n' + block
    if not new_content.lstrip().startswith(header):
        new_content = f'{header}\n\n' + new_content.lstrip()
    return new_content.rstrip() + '\n'


def migrate_kind(assets_dir: Path, kind: str, dry_run: bool) -> dict:
    header = HEADERS[kind]
    filename = FILENAMES[kind]
    target_path = assets_dir / filename
    content = target_path.read_text(encoding='utf-8') if target_path.exists() else ''
    migrated_blocks: list[dict[str, str]] = []
    for episode_dir in iter_episode_dirs(assets_dir):
        source_path = episode_dir / filename
        if not source_path.exists():
            continue
        raw = source_path.read_text(encoding='utf-8')
        body = strip_header(raw, header)
        block = build_episode_block(episode_dir.name.lower(), body)
        content = merge_episode_block(content, header, episode_dir.name.lower(), block)
        migrated_blocks.append({
            'episode_id': episode_dir.name.lower(),
            'source_path': str(source_path),
            'target_path': str(target_path),
        })
    if migrated_blocks and not dry_run:
        target_path.write_text(content, encoding='utf-8')
    return {
        'kind': kind,
        'target_path': str(target_path),
        'migrated_blocks': migrated_blocks,
        'written': bool(migrated_blocks) and not dry_run,
    }


def migrate_assets_dir(assets_dir: Path, dry_run: bool) -> dict:
    result = {
        'assets_dir': str(assets_dir),
        'series_name': assets_dir.name,
        'episode_dirs': [item.name.lower() for item in iter_episode_dirs(assets_dir)],
        'character': migrate_kind(assets_dir, 'character', dry_run),
        'scene': migrate_kind(assets_dir, 'scene', dry_run),
        'dry_run': dry_run,
    }
    if not dry_run:
        (assets_dir / SUMMARY_FILENAME).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main() -> None:
    args = parse_args()
    assets_dirs = discover_assets_dirs(args)
    if not assets_dirs:
        raise SystemExit('未找到可迁移的 assets/*-claude 目录。')
    results = [migrate_assets_dir(path, args.dry_run) for path in assets_dirs]
    print(json.dumps({'results': results}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
