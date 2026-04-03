# OpenAI Agent Flow For Seedance2.0 AI

This folder is the OpenAI-native version of the current `.claude` workflow.

It is intentionally not a one-to-one prompt rename. Instead, it follows the OpenAI agent model:

- specialized `Agent` objects
- `handoffs` between agents
- custom `function_tool` tools for controlled file I/O
- a single orchestrator agent that decides when to delegate

## Agents

- `producer_agent`: entry point and overall coordinator
- `explosive_agent`: scores and upgrades scripts for retention and click-through
- `director_agent`: produces `01-director-analysis.md`
- `art_agent`: produces `character-prompts.md` and `scene-prompts.md`
- `storyboard_agent`: produces `02-seedance-prompts.md`

## Tools

- read repository files
- write repository files
- resolve series-scoped paths
- list source scripts
- run the repo-scoped production pipeline through `scripts/run_openai_agent_flow.py`

## What This Enables

With Codex in VSCode, use `AGENTS.md` as the repo-native instruction file.

With OpenAI API / Agents SDK, use the files in this folder plus:

- `scripts/run_openai_agent_flow.py`
- `config/openai_agent_flow.local.json`
- `scripts/generate_director_analysis.py`
- `scripts/generate_seedance_prompts.py`

## Current Scope

This is now a hybrid of:

1. an official OpenAI agent scaffold for Codex / Agents SDK
2. a deterministic repo pipeline that materializes the exact files this project needs

The main value today is:

1. mapping the repo to official OpenAI agent concepts
2. giving you a clean migration path from `.claude`
3. preserving the exact output targets you care about:
   - `outputs/<剧名>/<集数>/01-director-analysis.md`
   - `outputs/<剧名>/<集数>/02-seedance-prompts.md`
   - `assets/<剧名>-gpt/character-prompts.md`
   - `assets/<剧名>-gpt/scene-prompts.md`
