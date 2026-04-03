# Skills 总览

这个目录统一存放当前项目真正使用到的 skill，避免继续在 `.claude/skills/`、`openai_agents/skills/` 两边来回找。

目录分为两类：

- `production/`：生产方法论 skill，主要来自原 `.claude/skills/`
- `openai_flow/`：OpenAI / Codex 编排层 skill，主要来自原 `openai_agents/skills/`

当前项目已切换到优先从这个目录读取 skill 内容。

## Production Skills

### `production/video-script-reconstruction-skill/`
- 作用：视频理解、题材分类、爆点抽取、剧本重建的总方法论
- 被谁使用：
  - [openai_adapter.py](/mnt/myvg/Agent/Seedance2.0_AI/providers/openai_adapter.py)
  - [gemini_adapter.py](/mnt/myvg/Agent/Seedance2.0_AI/providers/gemini_adapter.py)
- 在流程中的哪一步被使用：
  - `./run_video_pipeline.sh` 的单集视频理解阶段
  - `./run_series_pipeline.sh` 的整剧批量视频理解阶段
  - 视频分析完成后的剧本重建阶段
- 怎么被使用：
  - 作为视频理解 prompt 和剧本重建 prompt 的核心方法论来源，要求模型不仅提取事实，还要产出题材分类、爆点画像和下游设计指导
  - 当前真实维护入口是 `production/video-script-reconstruction-skill/genres/<题材>/playbook.json + skill.md`
  - 如果命中特定题材，还会额外加载对应题材目录下的 `skill.md`
  - 对应的题材经验库来自各题材目录下的 `playbook.json`
  - `genre_skills/` 和 `playbooks/genre-playbook-library.json` 现在保留为兼容/认知说明层，不再是推荐维护入口
  - 每集会额外产出 `genre_routing_debug__*.md/.json`，方便调试“题材识别、skill 选择、playbook 命中”

### `production/director-skill/`
- 作用：导演讲戏拆解标准、人物/场景提取、素材状态判断
- 被谁使用：
  - [generate_director_analysis.py](/mnt/myvg/Agent/Seedance2.0_AI/scripts/generate_director_analysis.py)
- 在流程中的哪一步被使用：
  - `./run_openai_agent_flow.sh` 的导演分析阶段
- 怎么被使用：
  - 作为导演分析 draft prompt 的方法论参考，指导模型怎么切剧情点、怎么写导演讲戏、怎么判断素材状态

### `production/script-analysis-review-skill/`
- 作用：导演分析复审，检查剧情点遗漏、拆解质量、是否适合下游继续生产
- 被谁使用：
  - [generate_director_analysis.py](/mnt/myvg/Agent/Seedance2.0_AI/scripts/generate_director_analysis.py)
- 在流程中的哪一步被使用：
  - 导演分析复审阶段
- 怎么被使用：
  - 注入到 director review prompt，要求模型用它的标准修订导演分析初稿

### `production/compliance-review-skill/`
- 作用：合规与风险表达收敛，避免过曝、软色情、真人高风险等描述
- 被谁使用：
  - [generate_director_analysis.py](/mnt/myvg/Agent/Seedance2.0_AI/scripts/generate_director_analysis.py)
  - [generate_art_assets.py](/mnt/myvg/Agent/Seedance2.0_AI/scripts/generate_art_assets.py)
  - [generate_seedance_prompts.py](/mnt/myvg/Agent/Seedance2.0_AI/scripts/generate_seedance_prompts.py)
- 在流程中的哪一步被使用：
  - 导演分析复审
  - 服化道复审
  - Seedance 分镜复审
- 怎么被使用：
  - 在 review prompt 里作为“安全收敛层”，把高风险表述改成更稳但不失张力的表达

### `production/art-design-skill/`
- 作用：人物提示词、场景宫格提示词、美术统一风格与图像提示词规范
- 被谁使用：
  - [generate_art_assets.py](/mnt/myvg/Agent/Seedance2.0_AI/scripts/generate_art_assets.py)
- 在流程中的哪一步被使用：
  - 服化道 draft 阶段
- 怎么被使用：
  - 结合示例、模板、图像指南一起注入 art draft prompt，约束人物提示词和场景提示词的写法

### `production/art-direction-review-skill/`
- 作用：服化道设计复审，纠正人物/场景可执行性与稳定性问题
- 被谁使用：
  - [generate_art_assets.py](/mnt/myvg/Agent/Seedance2.0_AI/scripts/generate_art_assets.py)
- 在流程中的哪一步被使用：
  - 服化道 review 阶段
- 怎么被使用：
  - 用来检查人物辨识度、场景空间清晰度、图像可执行性和命名规范

### `production/seedance-storyboard-skill/`
- 作用：Seedance 分镜提示词主方法论、示例与模板
- 被谁使用：
  - [generate_seedance_prompts.py](/mnt/myvg/Agent/Seedance2.0_AI/scripts/generate_seedance_prompts.py)
- 在流程中的哪一步被使用：
  - Seedance draft 阶段
- 怎么被使用：
  - 与方法论文档、示例、模板一起注入 storyboard draft prompt，指导如何从导演讲戏本生成动态提示词

### `production/seedance-prompt-review-skill/`
- 作用：Seedance 分镜提示词复审，检查忠实度、镜头可执行性、引用准确性
- 被谁使用：
  - [generate_seedance_prompts.py](/mnt/myvg/Agent/Seedance2.0_AI/scripts/generate_seedance_prompts.py)
- 在流程中的哪一步被使用：
  - Seedance review 阶段
- 怎么被使用：
  - 用于修订分镜提示词初稿，尤其检查引用、节拍密度和镜头可执行性

### `production/explosive-screenwriter-skill/`
- 作用：爆款评分、改稿策略、钩子玩法、整剧爆款玩法沉淀
- 被谁使用：
  - [generate_explosive_rewrites.py](/mnt/myvg/Agent/Seedance2.0_AI/scripts/generate_explosive_rewrites.py)
- 在流程中的哪一步被使用：
  - 爆款评分阶段
  - 爆款改稿阶段
  - 整剧玩法手册沉淀阶段
- 怎么被使用：
  - 作为爆款链的核心方法论来源，同时配合 hook playbook、analysis template、rewrite template 一起注入 prompt

## OpenAI Flow Skills

注意：

- 这一组 skill 主要不是“直接生成最终产物”的业务方法论，而是 `OpenAI Agent 编排层` 的阶段说明。
- 它们由 [flow.py](/mnt/myvg/Agent/Seedance2.0_AI/openai_agents/flow.py) 通过 `common_docs + stage skill` 的方式拼进 agent instructions。
- 真正运行入口是 [run_openai_agent_flow.sh](/mnt/myvg/Agent/Seedance2.0_AI/run_openai_agent_flow.sh)。
- 如果你想改“某个 Agent 在编排层如何理解自己的职责”，优先改这里。
- 如果你想改“导演分析 / 服化道 / Seedance / 爆款改稿的方法论细节”，优先改 `skills/production/`。

### `openai_flow/common/path_scope.md`
- 作用：统一路径作用域，确保输出按剧名/集数分仓
- 被谁使用：
  - [openai_agents/flow.py](/mnt/myvg/Agent/Seedance2.0_AI/openai_agents/flow.py)
- 在流程中的哪一步被使用：
  - 所有 OpenAI Agent 的通用约束阶段
- 怎么被使用：
  - 在 agent 开始思考前先约束路径与剧名作用域，防止写错目录或跨剧串档

### `openai_flow/common/output_contracts.md`
- 作用：约束导演分析、服化道、分镜产物的目标形态
- 被谁使用：
  - [openai_agents/flow.py](/mnt/myvg/Agent/Seedance2.0_AI/openai_agents/flow.py)
- 在流程中的哪一步被使用：
  - 所有 OpenAI Agent 的通用输出标准阶段
- 怎么被使用：
  - 在 agent 生成内容前，先告诉它不同产物最低需要包含什么

### `openai_flow/common/review_rubric.md`
- 作用：统一评审标准，帮助 producer/director/art/storyboard agent 对齐质量要求
- 被谁使用：
  - [openai_agents/flow.py](/mnt/myvg/Agent/Seedance2.0_AI/openai_agents/flow.py)
- 在流程中的哪一步被使用：
  - 所有 OpenAI Agent 的自检阶段
- 怎么被使用：
  - 作为写入前的复审标尺，减少“能写出来但下游不好用”的内容

### `openai_flow/producer_workflow.md`
- 作用：Producer Agent 的调度逻辑
- 被谁使用：
  - [openai_agents/flow.py](/mnt/myvg/Agent/Seedance2.0_AI/openai_agents/flow.py)
- 在流程中的哪一步被使用：
  - OpenAI Agent Flow 的入口总调度阶段
- 怎么被使用：
  - 决定先爆款改稿、还是直接导演分析、还是继续服化道/分镜

### `openai_flow/explosive_workflow.md`
- 作用：Explosive Agent 的阶段任务标准
- 被谁使用：
  - [openai_agents/flow.py](/mnt/myvg/Agent/Seedance2.0_AI/openai_agents/flow.py)
- 在流程中的哪一步被使用：
  - 可选的爆款改稿阶段
- 怎么被使用：
  - 指导 Explosive Agent 如何比较同集多个版本、如何写强化版剧本

### `openai_flow/director_workflow.md`
- 作用：Director Agent 的阶段任务标准
- 被谁使用：
  - [openai_agents/flow.py](/mnt/myvg/Agent/Seedance2.0_AI/openai_agents/flow.py)
- 在流程中的哪一步被使用：
  - 导演分析阶段
- 怎么被使用：
  - 规定 Director Agent 应如何拆剧情点、写导演讲戏、提人物和场景

### `openai_flow/art_workflow.md`
- 作用：Art Agent 的阶段任务标准
- 被谁使用：
  - [openai_agents/flow.py](/mnt/myvg/Agent/Seedance2.0_AI/openai_agents/flow.py)
- 在流程中的哪一步被使用：
  - 服化道设计阶段
- 怎么被使用：
  - 规定 Art Agent 应如何生成角色提示词和场景宫格提示词

### `openai_flow/storyboard_workflow.md`
- 作用：Storyboard Agent 的阶段任务标准
- 被谁使用：
  - [openai_agents/flow.py](/mnt/myvg/Agent/Seedance2.0_AI/openai_agents/flow.py)
- 在流程中的哪一步被使用：
  - Seedance 分镜提示词阶段
- 怎么被使用：
  - 规定 Storyboard Agent 应如何把导演分析和素材转成可执行的动态提示词

## Source Mapping

- 原 `.claude/skills/` 中当前被主流程用到的 skill 已复制到 `skills/production/`
- 原 `openai_agents/skills/` 中当前被 OpenAI Agent Flow 用到的 skill 已复制到 `skills/openai_flow/`

后续维护建议：

- 生产质量方法论优先改 `skills/production/`
- OpenAI agent 编排规则优先改 `skills/openai_flow/`
- Prompt 模板优先改 [prompts/](/mnt/myvg/Agent/Seedance2.0_AI/prompts)

一个简单判断方法：

- 想改“这个阶段应该做什么、优先级怎么排、输出大致应该长什么样”：
  改 `skills/openai_flow/`
- 想改“这个阶段内部的专业方法论、检查标准、示例、模板内容”：
  改 `skills/production/`
- 想改“实际送给模型的任务提示词文本结构”：
  改 [prompts/](/mnt/myvg/Agent/Seedance2.0_AI/prompts)
