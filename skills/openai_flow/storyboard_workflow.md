# 分镜 Agent 工作流 Skill

> 使用位置：
> - 由 [flow.py](/mnt/myvg/Agent/Seedance2.0_AI/openai_agents/flow.py) 注入给 `Storyboard Agent`
> - 属于 `./run_openai_agent_flow.sh` 的分镜阶段编排说明
> - 作用方式：定义如何把导演分析和素材提示词转成 `02-seedance-prompts.md`

## 目标

产出尽可能强的 `02-seedance-prompts.md`。

## 输入

- `01-director-analysis.md`
- `character-prompts.md`
- `scene-prompts.md`
- 方法论与生成约束

## 提示词构建规则

- 一个剧情点应对应一个清晰的生成任务
- 使用动态、可执行的视觉语言
- 忠实保留导演的情绪与视觉意图
- 不要在一条提示词中塞入太多不相关动作
- 一旦存在素材引用，要稳定一致地使用

## 质量标准

- 运动应符合物理直觉
- 时序密度应合理
- 转场应清楚
- 提示词应优先服务生成，而不是只追求文学感

