# Seedance2.0_AI 精简上 Git 清单

这个目录 `seedance2.0_AI_upd` 是按 [`WORKFLOW_QUICKSTART.md`](./WORKFLOW_QUICKSTART.md) 反推后整理出的“可上 git 的核心包”。

## 1. 已保留的核心功能文件

- 主说明文档：
  - `WORKFLOW_QUICKSTART.md`
  - `WORKFLOW_COMMANDS.md`
  - `requirements-video-pipeline.txt`
- 主入口脚本：
  - `run_video_pipeline.sh`
  - `run_series_pipeline.sh`
  - `run_openai_agent_flow.sh`
  - `run_seedance_style_transfer.sh`
  - `run_seedance_api_generation.sh`
  - `run_upload_seedance_refs.sh`
  - `run_seedance_batch_submit.sh`
  - `run_seedance_beat_catalog_submit.sh`
  - 以及同目录下其他 `run_*.sh` 辅助脚本
- 核心 Python 代码：
  - `scripts/`
  - `pipelines/`
  - `providers/`
  - `seedance_learning.py`
  - `genre_routing.py`
  - `genre_reference_bundle.py`
  - `prompt_utils.py`
  - `skill_utils.py`
  - `workflow_context_compaction.py`
  - `pipeline_telemetry.py`
- 提示词与 skill：
  - `prompts/`
  - `skills/`
  - `openai_agents/`
- 运行依赖资源：
  - `schemas/`
  - `config/*.example.json`
  - `config/参数理解.md`
  - `prompt_library/`
  - `tosutil`

## 2. 这次明确不复制的大体积 / 非核心内容

- 原始视频与压缩包：
  - `videos/`
- 全部生成产物：
  - `outputs/`
  - `assets/`
  - `analysis/`
  - `analysis-old/`
  - `script/`
  - `project/`
- 本地环境和缓存：
  - `.venv/`
  - `.claude/`
  - `.vscode/`
  - `__pycache__/`
- 本地状态与私有配置：
  - `.env.local`
  - `.agent-state.json`
  - `config/*.local.json`
  - `config/archive/`
- 过程分析 / 冗余审计 / 优化笔记类文档：
  - 根目录下大量 `*ANALYSIS*.md`
  - `OPTIMIZATION*.md`
  - `SEEDANCE_*_ANALYSIS.md`
  - 其他阶段性总结文档

## 3. 为什么这样裁剪

- `WORKFLOW_QUICKSTART.md` 里定义的四条主链，真正依赖的是脚本、Python 代码、prompts、skills、示例配置和 `prompt_library/`。
- 体积最大的目录几乎全是样本视频、分析中间件和运行输出，不属于“项目功能本体”。
- `prompt_library/SEARCH_INDEX.json` 是 `run_seedance_style_transfer.sh` 的直接依赖，所以保留。
- `tosutil` 是参考图上传链的一部分，虽然有体积，但属于功能依赖，所以保留。
- `config/*.local.json` 明显是机器相关 / 路径相关配置，不适合直接上 git，所以只保留 `*.example.json`。

## 4. 如果你后面想继续某一部剧，而不是只保留代码

下面这些可以按“单剧”小范围补拷，不建议整仓全带：

- 从 `analysis/<剧名>/` 里按需补：
  - `series_context.json`
  - `series_bible.json`
  - `character_registry.json`
  - `location_registry.json`
  - `plot_timeline.json`
  - `series_strength_playbook_draft.json`
  - `series_strength_playbook_draft.md`
  - `seedance_purpose_skill_library.json`
  - `seedance_purpose_skill_library.md`
  - `seedance_purpose_template_library.json`
  - `seedance_purpose_template_library.md`
  - `openai_agent_flow/genre_reference_bundle.json`
  - `openai_agent_flow/genre_reference_bundle.md`
- 从 `script/<剧名>/` 里补该剧剧本
- 从 `assets/<剧名>-gpt/` 里补该剧的：
  - `character-prompts.md`
  - `scene-prompts.md`

## 5. 现在这个精简包适合做什么

- 上 git，作为核心工作流仓库
- 给别人交接“怎么跑”
- 保留 prompt、skills、workflow 逻辑
- 后面按具体剧目再增量补数据

## 6. 现在这个精简包不包含什么

- 可直接复现历史全部剧目的原始素材
- 历史跑过的完整结果
- 你本机的私有配置和运行环境
