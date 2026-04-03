# Prompt Catalog

这个目录集中存放项目里主要工作流的提示词模板，便于统一维护。

目录约定：

- `agents/`：OpenAI Agents / Codex 编排层的角色提示词
- `video_pipeline/`：视频理解与剧本重建
- `explosive_rewrite/`：爆款分析与改稿
- `director_analysis/`：导演讲戏本生成与复审
- `art_assets/`：人物/场景提示词生成与复审
- `seedance_storyboard/`：Seedance 分镜提示词生成与复审

模板规则：

- 使用 `{{variable_name}}` 占位
- 由项目内的 `prompt_utils.py` 统一加载和渲染
- 长上下文（如 `.claude` skill、`series_context`、剧本文本）由调用脚本注入

