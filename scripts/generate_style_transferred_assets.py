from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_nano_banana_assets import generate_image, load_json, print_status  # type: ignore
from providers.base import save_json_file

DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'config' / 'style_transfer_assets.local.json'
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}

STYLE_PRESETS: list[dict[str, str]] = [
    {
        'label': '国漫厚涂动画',
        'key': 'guoman_painted_animation',
        'style_prompt': '高完成度国漫厚涂动画风格，非真人、非摄影、非写实照片；保留角色身份与服装结构，强化动画体积感、分层光影、干净边缘与影视级配色。',
    },
    {
        'label': '赛璐璐动画',
        'key': 'cel_animation',
        'style_prompt': '高质量赛璐璐动画风格，非真人、非摄影、非写实照片；清晰线稿、平滑块面、动画光影、统一色彩设计，角色辨识度必须保持一致。',
    },
    {
        'label': '3D动画电影',
        'key': 'stylized_3d_animation',
        'style_prompt': '高质量风格化3D动画电影风格，非真人、非摄影；保持同一角色脸型与服装轮廓，强调动画电影级材质、柔和面部建模与统一美术风格。',
    },
    {
        'label': '水墨插画',
        'key': 'ink_illustration',
        'style_prompt': '东方水墨插画风格，非真人、非摄影；保留角色身份与服装层次，强调墨色层次、留白、笔触与意境，不做照片感渲染。',
    },
    {
        'label': '绘本插画',
        'key': 'storybook_illustration',
        'style_prompt': '高质量绘本插画风格，非真人、非摄影；保持角色身份稳定，色彩统一柔和，强调插画造型语言与叙事感。',
    },
]

NEGATIVE_PROMPT = (
    '避免真人、避免摄影、避免真实演员脸、避免照片感、避免皮肤毛孔摄影质感、避免超写实人像、'
    '避免多余人物、避免改变年龄和性别、避免改变服装结构、避免发型发色漂移、避免角色变成另一个人'
)


def python_choice(prompt: str, options: list[tuple[str, Any]], default_index: int = 0) -> tuple[str, Any]:
    if not options:
        raise RuntimeError('没有可选项。')
    print(prompt)
    for index, (label, _) in enumerate(options, start=1):
        default_text = '  [默认]' if index - 1 == default_index else ''
        print(f'  {index}. {label}{default_text}')
    while True:
        raw = input(f'请输入序号（默认 {default_index + 1}）：').strip()
        if not raw:
            return options[default_index]
        if raw.isdigit():
            value = int(raw)
            if 1 <= value <= len(options):
                return options[value - 1]
        print('输入无效，请重新输入。')


def prompt_bool(prompt: str, default_value: bool) -> bool:
    default_text = 'true' if default_value else 'false'
    while True:
        raw = input(f'{prompt}（true/false，默认 {default_text}）：').strip().lower()
        if not raw:
            return default_value
        if raw in {'true', '1', 'yes', 'y', 'on'}:
            return True
        if raw in {'false', '0', 'no', 'n', 'off'}:
            return False
        print('输入无效，请输入 true 或 false。')


def list_series(script_root: Path) -> list[tuple[str, Path]]:
    options: list[tuple[str, Path]] = []
    for path in sorted(script_root.iterdir(), key=lambda item: item.name):
        if path.is_dir():
            options.append((path.name, path))
    return options


def episode_sort_key(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if name.startswith('ep') and len(name) >= 4 and name[2:].isdigit():
        return (int(name[2:]), name)
    digits = ''.join(ch for ch in name if ch.isdigit())
    return (int(digits) if digits else 10**9, name)


def find_generated_model_root(series_name: str, model: str) -> Path:
    preferred = PROJECT_ROOT / 'assets' / f'{series_name}-gpt' / 'generated' / model
    if preferred.exists():
        return preferred
    fallback = PROJECT_ROOT / 'assets' / series_name / 'generated' / model
    if fallback.exists():
        return fallback
    return preferred


def list_episode_dirs(model_root: Path) -> list[tuple[str, Path]]:
    if not model_root.exists():
        return []
    options: list[tuple[str, Path]] = []
    for path in sorted(model_root.iterdir(), key=episode_sort_key):
        if not path.is_dir() or not path.name.lower().startswith('ep'):
            continue
        character_dir = path / 'characters'
        character_real_dir = path / 'characters-real'
        if character_dir.exists() or character_real_dir.exists():
            options.append((path.name, path))
    return options


def load_config(config_path: Path) -> dict[str, Any]:
    if config_path.exists():
        return load_json(config_path)
    return {
        'provider': {'gemini': {'api_key': '', 'model': 'gemini-3-pro-image-preview'}},
        'run': {'dry_run': False, 'skip_existing_images': True, 'timeout_seconds': 300, 'delay_seconds': 1.0},
        'output': {'characters_source_dirname': 'characters-real', 'characters_target_dirname': 'characters'},
        'quality': {'negative_prompt': NEGATIVE_PROMPT},
    }


def resolve_api_key(config: dict[str, Any]) -> str:
    api_key = str(config.get('provider', {}).get('gemini', {}).get('api_key', '') or '').strip()
    if api_key:
        return api_key
    env_key = (Path('/dev/null') and __import__('os').environ.get('GEMINI_API_KEY', '').strip())
    if env_key:
        return env_key
    try:
        video_config = load_json(PROJECT_ROOT / 'config' / 'video_pipeline.local.json')
        api_key = str(video_config.get('providers', {}).get('gemini', {}).get('api_key', '') or '').strip()
        if api_key:
            return api_key
    except Exception:
        pass
    raise RuntimeError('未找到 Gemini API key，请在 config/style_transfer_assets.local.json 或 GEMINI_API_KEY 中配置。')


def ensure_dirs(episode_root: Path, dry_run: bool, source_name: str, target_name: str) -> tuple[Path, Path, bool]:
    source_dir = episode_root / source_name
    target_dir = episode_root / target_name
    renamed = False
    if source_dir.exists():
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        return source_dir, target_dir, renamed
    original_dir = episode_root / target_name
    if not original_dir.exists():
        raise FileNotFoundError(f'未找到可转换的人物目录：{original_dir}')
    if dry_run:
        print_status(f'预览：将把 {original_dir} 重命名为 {source_dir}，并在 {target_dir} 生成风格转换结果。')
        return source_dir, target_dir, True
    original_dir.rename(source_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    renamed = True
    print_status(f'已将原始人物目录重命名为：{source_dir}')
    return source_dir, target_dir, renamed


def find_existing_output(target_dir: Path, source_image: Path) -> Path | None:
    stem = source_image.stem
    for candidate in sorted(target_dir.glob(f'{stem}.*')):
        if candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate
    return None


def build_style_prompt(style_label: str, style_prompt: str) -> str:
    return (
        '请将参考图中的同一角色转换为指定的非真实艺术风格。必须严格保留同一角色的身份一致性：'
        '脸型、五官比例、发型发色、服装结构、配饰位置、视角布局和角色气质尽量不变；'
        '如果参考图本身是设定图版式，请继续保持同类版式。'
        '只改变画面最终渲染风格，不允许把角色变成另一个人，不允许改变性别、年龄感、民族气质、服装类别。\n\n'
        f'目标风格：{style_label}。{style_prompt}\n\n'
        '结果必须是非真人、非摄影、非现实人像，不允许出现真实演员照片感。'
    )


def collect_images(source_dir: Path) -> list[Path]:
    return [path for path in sorted(source_dir.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def interactive_config(config_path: Path, mode: str) -> dict[str, Any]:
    config = load_config(config_path)
    model = str(config.get('provider', {}).get('gemini', {}).get('model', 'gemini-3-pro-image-preview') or 'gemini-3-pro-image-preview')
    script_root = PROJECT_ROOT / 'script'
    series_label, series_path = python_choice('请选择要做风格转换的剧本：', list_series(script_root), default_index=0)
    model_root = find_generated_model_root(series_label, model)
    episode_options = list_episode_dirs(model_root)
    if not episode_options:
        raise RuntimeError(f'未找到可转换的人物目录：{model_root}')
    episode_label, episode_root = python_choice('请选择要做风格转换的章节：', episode_options, default_index=0)
    style_label, style_key = python_choice(
        '请选择目标非真实风格：',
        [(item['label'], item['key']) for item in STYLE_PRESETS],
        default_index=0,
    )
    style_prompt = next(item['style_prompt'] for item in STYLE_PRESETS if item['key'] == style_key)
    custom_style = input('可选：输入更细的风格补充说明（直接回车跳过）：').strip()
    if custom_style:
        style_prompt = f'{style_prompt} 补充要求：{custom_style}'

    dry_run_default = bool(config.get('run', {}).get('dry_run', False))
    if mode == 'preview':
        dry_run = True
    elif mode == 'run':
        dry_run = False
    else:
        dry_run = dry_run_default
    if mode == 'interactive':
        skip_existing = prompt_bool('是否跳过已存在的风格转换图片', bool(config.get('run', {}).get('skip_existing_images', True)))
    else:
        skip_existing = bool(config.get('run', {}).get('skip_existing_images', True))

    config.setdefault('selection', {})
    config['selection'].update(
        {
            'series_name': series_label,
            'series_path': str(series_path),
            'episode_id': episode_label,
            'episode_root': str(episode_root),
            'style_key': style_key,
            'style_label': style_label,
            'style_prompt': style_prompt,
        }
    )
    config.setdefault('run', {})
    config['run']['dry_run'] = dry_run
    config['run']['skip_existing_images'] = skip_existing
    return config


def run_pipeline(config: dict[str, Any]) -> None:
    selection = config.get('selection', {})
    series_name = str(selection.get('series_name') or '').strip()
    episode_id = str(selection.get('episode_id') or '').strip()
    episode_root = Path(str(selection.get('episode_root') or '')).expanduser().resolve()
    style_label = str(selection.get('style_label') or '').strip()
    style_key = str(selection.get('style_key') or '').strip()
    style_prompt = str(selection.get('style_prompt') or '').strip()
    if not all([series_name, episode_id, style_label, style_key, style_prompt]) or not str(episode_root):
        raise RuntimeError('风格转换配置不完整，请先通过交互选择剧本、章节与风格。')

    api_key = resolve_api_key(config)
    model = str(config.get('provider', {}).get('gemini', {}).get('model', 'gemini-3-pro-image-preview') or 'gemini-3-pro-image-preview')
    dry_run = bool(config.get('run', {}).get('dry_run', False))
    skip_existing = bool(config.get('run', {}).get('skip_existing_images', True))
    timeout_seconds = int(config.get('run', {}).get('timeout_seconds', 300) or 300)
    delay_seconds = float(config.get('run', {}).get('delay_seconds', 1.0) or 1.0)
    source_name = str(config.get('output', {}).get('characters_source_dirname', 'characters-real') or 'characters-real')
    target_name = str(config.get('output', {}).get('characters_target_dirname', 'characters') or 'characters')
    negative_prompt = str(config.get('quality', {}).get('negative_prompt', NEGATIVE_PROMPT) or NEGATIVE_PROMPT)

    print_status(f'剧名：{series_name}')
    print_status(f'章节：{episode_id}')
    print_status(f'目标风格：{style_label}')
    source_dir, target_dir, renamed = ensure_dirs(episode_root, dry_run, source_name, target_name)
    print_status(f'原始人物目录：{source_dir}')
    print_status(f'风格转换输出目录：{target_dir}')
    target_will_be_empty = False
    if renamed and dry_run:
        target_will_be_empty = True
        print_status('当前为预览模式，实际运行时才会执行目录重命名。')

    images = collect_images(source_dir if source_dir.exists() else (episode_root / target_name))
    if not images:
        raise RuntimeError(f'未找到可转换的人物图片：{source_dir}')

    manifest_items: list[dict[str, Any]] = []
    generated_count = 0
    skipped_count = 0
    blocked_count = 0
    print_status(f'待转换：{len(images)} 张人物图')

    for image_path in images:
        existing = None
        if not target_will_be_empty and target_dir.exists():
            existing = find_existing_output(target_dir, image_path)
        if skip_existing and existing is not None:
            skipped_count += 1
            print_status(f'跳过已有风格图：{image_path.name} -> {existing.name}')
            manifest_items.append(
                {
                    'source_path': str(image_path),
                    'status': 'skipped_existing',
                    'output_path': str(existing),
                    'style_key': style_key,
                    'style_label': style_label,
                }
            )
            continue

        ext = image_path.suffix if image_path.suffix.lower() in IMAGE_SUFFIXES else '.jpg'
        output_path = target_dir / f'{image_path.stem}{ext}'
        if dry_run:
            print_status(f'预览：将转换 {image_path.name} -> {output_path.name}')
            manifest_items.append(
                {
                    'source_path': str(image_path),
                    'status': 'dry_run',
                    'output_path': str(output_path),
                    'style_key': style_key,
                    'style_label': style_label,
                }
            )
            continue

        print_status(f'开始风格转换：{image_path.name}')
        try:
            image_bytes, mime_type, raw_response = generate_image(
                api_key=api_key,
                model=model,
                prompt=build_style_prompt(style_label, style_prompt),
                negative_prompt=negative_prompt,
                image_size_hint='保持原设定图版式与构图',
                timeout_seconds=timeout_seconds,
                reference_images=[image_path],
            )
        except Exception as exc:
            blocked_count += 1
            print_status(f'已跳过：{image_path.name}，原因：{exc}')
            manifest_items.append(
                {
                    'source_path': str(image_path),
                    'status': 'blocked',
                    'error': str(exc),
                    'style_key': style_key,
                    'style_label': style_label,
                }
            )
            time.sleep(delay_seconds)
            continue

        actual_ext = '.png' if 'png' in mime_type.lower() else '.jpg'
        if output_path.suffix.lower() != actual_ext:
            output_path = target_dir / f'{image_path.stem}{actual_ext}'
        output_path.write_bytes(image_bytes)
        generated_count += 1
        manifest_items.append(
            {
                'source_path': str(image_path),
                'status': 'generated',
                'output_path': str(output_path),
                'mime_type': mime_type,
                'style_key': style_key,
                'style_label': style_label,
                'raw_response_excerpt': json.dumps(raw_response, ensure_ascii=False)[:1200],
            }
        )
        time.sleep(delay_seconds)

    manifest = {
        'series_name': series_name,
        'episode_id': episode_id,
        'episode_root': str(episode_root),
        'source_dir': str(source_dir),
        'target_dir': str(target_dir),
        'style_key': style_key,
        'style_label': style_label,
        'style_prompt': style_prompt,
        'dry_run': dry_run,
        'generated_count': generated_count,
        'skipped_count': skipped_count,
        'blocked_count': blocked_count,
        'items': manifest_items,
    }
    manifest_path = episode_root / f'style_transfer_manifest__{style_key}.json'
    save_json_file(manifest_path, manifest)
    print_status(f'风格转换清单已写入：{manifest_path}')



def main() -> None:
    parser = argparse.ArgumentParser(description='把现有人物图批量转换为指定的非真实风格。')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH), help='配置文件路径')
    parser.add_argument('--mode', default='interactive', choices=['interactive', 'preview', 'run', 'config'])
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if args.mode == 'config':
        config = load_config(config_path)
    else:
        config = interactive_config(config_path, args.mode)
    run_pipeline(config)


if __name__ == '__main__':
    main()
