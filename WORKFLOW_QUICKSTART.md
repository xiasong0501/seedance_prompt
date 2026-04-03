# Workflow Quickstart

这份文档按当前项目真实能力重写，不再按“最短跑法”组织，而是按 4 个核心大类组织：

1. `run_series_pipeline.sh`
   通过视频学习剧本/分镜经验，以及学习 Seedance prompt 模板与经验
2. `run_seedance_style_transfer.sh`
   利用当前 Seedance 提示词去搜索模板库，并做“锁事实、重写表现层”的风格迁移
3. `run_openai_agent_flow.sh`
   从剧本继续生成导演讲戏本、服化道提示词、Seedance 提示词
4. Seedance 任务提交链
   从 `02-seedance-prompts.md` / 素材图走到真正的视频任务提交

如果只想知道“每个脚本怎么敲”，再看 `WORKFLOW_COMMANDS.md`。
这份文档重点解释：

- 每个 workflow 的真实职责
- 关键提示词、skills、agents 在哪里
- 学到的经验和模板落在哪
- 下游到底是怎么消费这些产物的

---

## 0. 项目总关系图

先用一句话概括当前主干：

`视频 -> run_series_pipeline.sh -> 剧本经验/分镜经验/Seedance 模板库 -> run_openai_agent_flow.sh -> 02-seedance-prompts -> run_seedance_style_transfer.sh（可选增强）-> Seedance 提交链`

更细一点：

- `run_series_pipeline.sh`
  从视频中学习剧情结构、题材判断、连续性、剧本写法、镜头和分镜经验，同时抽出 Seedance 可复用模板
- `run_openai_agent_flow.sh`
  把剧本转成导演讲戏本、人物/场景提示词、Seedance 分镜
- `run_seedance_style_transfer.sh`
  用前面学到的模板库反向增强已有 `02-seedance-prompts`
- 任务提交链
  用人物图、场景图、`Pxx__seedance_api.sh`、TOS URL 和 ARK API 去真正出视频

---

## 1. 第一大类：`run_series_pipeline.sh`

这是当前项目最重要的上游学习入口。

它不只是“视频转剧本”，而是同时做两件事：

- 从视频里学习剧本结构、分镜/镜头经验、题材经验
- 从视频里学习 Seedance prompt 的模板和经验

所以交接时，不能把它理解成一个普通转写脚本。

### 1.1 入口与主调用链

入口脚本：

```bash
./run_series_pipeline.sh run
```

主调用链：

```text
run_series_pipeline.sh
-> scripts/interactive_pipeline_launcher.py
-> scripts/run_series_pipeline.py
-> scripts/run_video_pipeline.py
-> pipelines/video_to_script_pipeline.py
-> pipelines/continuity_manager.py
-> seedance_learning.py
```

这条链里实际有两层学习系统：

- A. 视频理解与剧本/分镜经验沉淀
- B. Seedance beat 学习与 prompt template library 构建

---

### 1.2 A 类学习：通过视频学习剧本、分镜、题材和连续性经验

这一层的核心目标是把原视频还原成可继续生产的结构化分析和剧本，并把“这部剧为什么好看、下游该怎么学”沉淀下来。

#### 1.2.1 主要提示词在哪里

视频分析与剧本重建的主要 prompt 在：

- `prompts/video_pipeline/openai_analysis_system.md`
- `prompts/video_pipeline/openai_analysis_user.md`
- `prompts/video_pipeline/openai_script_system.md`
- `prompts/video_pipeline/openai_script_user.md`
- `prompts/video_pipeline/gemini_analysis_system.md`
- `prompts/video_pipeline/gemini_analysis_user.md`
- `prompts/video_pipeline/gemini_script_system.md`
- `prompts/video_pipeline/gemini_script_user.md`

其中最重要的分析系统提示词是：

- `prompts/video_pipeline/openai_analysis_system.md`

它明确要求模型不仅做剧情拆解，还要额外输出：

- 题材分类
- 爆点画像
- 下游设计指导
- “这部剧好在哪、哪些优点值得沉淀成 skill / playbook”的经验抽取

也就是说，`run_series_pipeline.sh` 的上游分析结果，从设计上就不是为了“看懂这一集”而已，而是为了给后续所有 workflow 供经验。

#### 1.2.2 这一层的关键产物

单集层：

- `analysis/<剧名>/<epXX>/episode_analysis__<provider>__<model>.json`
- `analysis/<剧名>/<epXX>/genre_routing_debug__<provider>__<model>.json`
- `analysis/<剧名>/<epXX>/genre_routing_debug__<provider>__<model>.md`
- `script/<剧名>/epXX__<provider>__<model>.md`

整剧连续性层：

- `analysis/<剧名>/series_bible.json`
- `analysis/<剧名>/character_registry.json`
- `analysis/<剧名>/location_registry.json`
- `analysis/<剧名>/plot_timeline.json`
- `analysis/<剧名>/series_context.json`
- `analysis/<剧名>/series_strength_playbook_draft.json`
- `analysis/<剧名>/series_strength_playbook_draft.md`

#### 1.2.3 “剧本分镜经验”具体在哪里

如果你要向别人解释“run_series_pipeline.sh 学到了什么剧本/分镜经验”，最核心的是这几个文件：

##### `analysis/<剧名>/series_strength_playbook_draft.json`

这是当前项目里“本剧优点经验沉淀草稿”的主文件。

它由 `pipelines/continuity_manager.py` 里的 `_build_series_strength_playbook()` 生成，字段包括：

- `episode_strengths`
- `why_it_works`
- `character_design_rules`
- `costume_makeup_rules`
- `scene_design_rules`
- `camera_language_rules`
- `storyboard_execution_rules`
- `dialogue_timing_rules`
- `continuity_guardrails`
- `negative_patterns`
- `reusable_playbook_rules`
- `reusable_skill_rules`

这就是“剧本经验”和“分镜经验”最直接的沉淀结果。

其中尤其重要的几组字段：

- `camera_language_rules`
  偏镜头语言经验
- `storyboard_execution_rules`
  偏分镜执行经验
- `dialogue_timing_rules`
  偏对白节奏经验
- `negative_patterns`
  偏踩坑经验

##### `analysis/<剧名>/series_bible.json`

这是整剧世界观、角色、地点、题材画像、下游设计偏好的汇总。

里面会持续累积：

- `genre_profile`
- `genre_playbooks`
- `camera_language_profile`
- `art_direction_profile`
- `storyboard_profile`
- `character_design_profile`
- `scene_design_profile`
- `dialogue_timing_profile`
- `production_guardrails`
- `series_learning_profile`

它更像“整剧知识库”。

##### `analysis/<剧名>/series_context.json`

这是下游最常被直接消费的整剧上下文包之一。

它把 `series_bible`、角色注册表、地点注册表、时间线等压成更适合后续导演、服化道、分镜继续使用的上下文。

#### 1.2.4 这些经验后面怎么被调用

这一层经验主要走两条路：

##### 路线 1：给后续题材库同步使用

相关入口：

- `./run_sync_series_learning.sh`
- `./run_review_genre_drafts.sh`

也就是把 `series_strength_playbook_draft` 里的规则，进一步整理成题材库 draft。

##### 路线 2：给 OpenAI 生产链间接使用

`run_openai_agent_flow.sh` 本身不直接把 `series_strength_playbook_draft.json` 原样塞给分镜阶段。
它更常走的是：

- `series_context.json`
- `genre_reference_bundle`
- 以及后面 B 类学习里生成的 Seedance purpose libraries

换句话说：

- A 类学习负责沉淀“这部剧整体好在哪、镜头和分镜经验是什么”
- B 类学习负责把这些经验进一步变成“可检索、可迁移的 Seedance 模板资产”

---

### 1.3 B 类学习：通过视频学习 Seedance prompt 模板和经验

这是 `run_series_pipeline.sh` 的第二条核心能力，也是你这次特别强调要写清楚的部分。

它的目的不是生成最终 `02-seedance-prompts.md`，而是：

- 先把视频切成适合 Seedance 学习的 beat
- 为每个 beat 总结镜头链、动作链、受光材质、声音床、尾帧交棒
- 再从整剧层面聚合成 purpose skill library 和 template library
- 最后导出成 prompt library，供后续检索和复用

#### 1.3.1 这一层的主要代码入口

单集 beat 学习：

- `seedance_learning.generate_episode_beat_catalog()`

整剧聚合：

- `seedance_learning.build_series_purpose_libraries()`

这两个函数都由 `run_series_pipeline.py` 在主流程跑完后继续触发。

#### 1.3.2 这条学习链做了什么

单集阶段会：

1. 读取 `episode_analysis`
2. 构建 beat segmentation
3. 为每个 beat 抽关键帧
4. 如启用 second-pass，多模态复盘镜头结构
5. 产出 `seedance_beat_catalog.json`

整剧阶段会：

1. 读取所有 `ep*/seedance_beat_catalog.json`
2. 按 purpose 聚合
3. 生成 skill library
4. 生成 template library
5. 导出 prompt library
6. 导出全局搜索索引 `SEARCH_INDEX.json`

#### 1.3.3 这条链的关键提示词在哪里

最关键的是 second-pass 复盘提示词：

- `prompts/seedance_learning/second_pass_system.md`
- `prompts/seedance_learning/second_pass_user.md`

它的职责是：

- 根据同一段 beat 的局部关键帧、对白、OCR 和 shot_chain 初稿
- 重新还原“这段 beat 实际是怎么拍的”
- 修正每个镜头的镜头语言、动作链、光线、声音、切镜触发

这一步是模板学习链非常关键的质量来源，因为它不是只靠文字分析，而是重新看局部图包来校正镜头结构。

#### 1.3.4 单集学习产物在哪

单集最重要的产物在：

- `analysis/<剧名>/<epXX>/beat_segmentation.json`
- `analysis/<剧名>/<epXX>/beat_segmentation.md`
- `analysis/<剧名>/<epXX>/seedance_beat_catalog.json`
- `analysis/<剧名>/<epXX>/seedance_beat_catalog.md`

以及中间学习目录：

- `analysis/<剧名>/<epXX>/seedance_learning/shot_frames/`
- `analysis/<剧名>/<epXX>/seedance_learning/beat_frames/`
- `analysis/<剧名>/<epXX>/seedance_learning/visual_second_pass/`

其中 `seedance_beat_catalog.json` 是最关键的单集模板学习资产。

它本质上是在说：

- 这个 beat 的 primary purpose 是什么
- 这段真正的 dramatic goal 是什么
- 镜头链怎么推进
- 对白落点在哪里
- 受光、材质、声音床、transition trigger 是什么

#### 1.3.5 整剧模板库和技能库在哪

整剧聚合后，核心文件是：

- `analysis/<剧名>/seedance_purpose_skill_library.json`
- `analysis/<剧名>/seedance_purpose_skill_library.md`
- `analysis/<剧名>/seedance_purpose_template_library.json`
- `analysis/<剧名>/seedance_purpose_template_library.md`

这两个文件的职责不同：

##### `seedance_purpose_skill_library.json`

偏“规则库 / 经验库”。

它按 purpose 汇总：

- `design_skill`
- `beat_rules`
- `action_rules`
- `dialogue_rules`
- `continuity_rules`
- `negative_patterns`

更像“这种目的的分镜应该怎么写”。

##### `seedance_purpose_template_library.json`

偏“模板库 / 示例库”。

它按 purpose 保留代表性模板，适合后续直接检索和迁移。

更像“这种目的可以参考哪些成熟模板”。

#### 1.3.6 最终导出的 prompt library 在哪

全局 prompt library 在：

- `prompt_library/`

最关键的检索索引：

- `prompt_library/SEARCH_INDEX.json`
- `prompt_library/SEARCH_INDEX.md`

这就是后面 `run_seedance_style_transfer.sh` 直接搜索的模板库入口。

也就是说：

- `analysis/<剧名>/seedance_purpose_template_library.json`
  是当前剧的整剧模板聚合结果
- `prompt_library/SEARCH_INDEX.json`
  是为了跨剧、跨 purpose 检索方便而导出的统一索引

#### 1.3.7 这些模板库后面怎么被调用

目前主要有两个下游消费者：

##### 下游 1：`run_openai_agent_flow.sh` 的 Seedance 分镜阶段

`scripts/generate_seedance_prompts.py` 会读取：

- `analysis/<剧名>/seedance_purpose_skill_library.json`
- `analysis/<剧名>/seedance_purpose_template_library.json`

然后构建 `seedance_story_point_guidance`，再把它喂进分镜生成 prompt。

也就是说，`run_openai_agent_flow.sh` 生成 `02-seedance-prompts.md` 时，已经在吃 `run_series_pipeline.sh` 学出来的模板知识。

##### 下游 2：`run_seedance_style_transfer.sh`

它会直接读：

- `prompt_library/SEARCH_INDEX.json`

从里面检索适合当前 point 的模板，再执行模板化重写。

#### 1.3.8 这一类 workflow 最应该怎么向别人解释

一句最准确的话是：

`run_series_pipeline.sh` 不是简单“视频转剧本”，而是项目的上游学习引擎。它同时产出剧本、连续性知识、整剧经验草稿、Seedance beat catalog、purpose skill library、purpose template library 和可检索 prompt library。后面的分镜生成和 style transfer 都依赖它。`

---

## 2. 第二大类：`run_seedance_style_transfer.sh`

这是“用已有 Seedance 提示词反查模板库，再做模板化增强”的 workflow。

它的定位非常明确：

- 输入：现有 `02-seedance-prompts`
- 知识源：`prompt_library/SEARCH_INDEX.json`
- 目标：不改剧情事实，不改引用关系，只重写表现层

### 2.1 它不是做什么

它不是：

- 重新生成一版剧情
- 重新分配素材编号
- 改人物、改场景、改对白事实

它是：

- 从模板库里找“结构适配”的强模板
- 把模板的镜头组织方式迁移到当前 point

所以最准确的理解是：

`这是一个对现有 Seedance 分镜做“模板检索 + 结构迁移 + 表现层重写”的后处理 workflow。`

---

### 2.2 入口与主调用链

入口脚本：

```bash
./run_seedance_style_transfer.sh
```

主调用链：

```text
run_seedance_style_transfer.sh
-> scripts/generate_seedance_style_transfer.py
-> 读取 storyboard json/md
-> 读取 prompt_library/SEARCH_INDEX.json
-> 候选模板召回
-> 模型 rerank
-> style plan
-> style transfer
-> 引用/对白/密度保护
-> sidecar 或 overwrite 落盘
```

---

### 2.3 它用到的系统提示词在哪里

这条链有 3 组核心 prompt：

#### A. 模板匹配阶段

- `prompts/seedance_style_template_match/system.md`
- `prompts/seedance_style_template_match/user.md`

这个阶段的角色是 `Seedance Prompt Template Matcher`。

它做的事情是：

- 对单条 point 和若干候选 prompt template 计算结构适配度
- 输出 `match_score`
- 标出最值得学习的 `learning_focus`

它强调看的不是表面关键词像不像，而是：

- 镜头结构
- 段落骨架
- 动作链
- 空间调度
- 受光/材质/声音床

#### B. 结构骨架规划阶段

- `prompts/seedance_style_transfer_plan/system.md`
- `prompts/seedance_style_transfer_plan/user.md`

这个阶段的角色是 `Seedance Style Transfer Planner`。

它不直接改分镜，而是先输出每条 point 的：

- `rewrite_goals`
- `skeleton_steps`
- `dialogue_freeze_notes`

也就是先把“这条应该怎么学模板”抽成可执行骨架。

#### C. 正式改写阶段

- `prompts/seedance_style_transfer/system.md`
- `prompts/seedance_style_transfer/user.md`

这个阶段的角色是 `Seedance Style Transfer Actor`。

它负责真正输出 delta，改写字段主要是：

- `continuity_bridge`
- `master_timeline`
- `audio_design`
- `prompt_text`

---

### 2.4 它的工作原理

这条链不是“拿一个 prompt 直接改”，而是分 6 步：

#### 第 1 步：读取当前 storyboard

脚本先定位：

- `02-seedance-prompts.md`
- `02-seedance-prompts__*.json`

这两个文件是 base truth。

#### 第 2 步：读取模板检索索引

默认读取：

- `prompt_library/SEARCH_INDEX.json`

并解析其中每个模板关联的 markdown 内容。

#### 第 3 步：先做启发式召回，再做模型 rerank

脚本不会一上来就把全库塞给模型。

它会先做 heuristic candidate selection，再用模型做 rerank。

也就是：

1. 基于 purpose、关键词、结构化 tag、时长等做粗召回
2. 用 `seedance_style_template_match` prompt 做精排
3. 取每条 point 的 top 模板

#### 第 4 步：先做 style plan

根据选中的模板，先为每条 point 生成结构骨架计划。

这一步的作用是强制模型先想清楚：

- 学什么
- 怎么学
- 对白怎么冻结
- 镜头骨架怎么重组

#### 第 5 步：正式改写

正式改写阶段遵守一个硬原则：

- `base point` 是唯一事实真源

模板只提供：

- 镜头语法
- 段落骨架
- 动作推进方式
- 光线材质写法
- 声音床
- 尾帧交棒

模板绝不提供：

- 剧情事实
- 人物名
- 场景名
- 道具名
- 对白事实

#### 第 6 步：结果保护和质量校验

改完后脚本还会执行几轮保护：

- `freeze_ref_integrity`
  锁 `@图片N` 的编号和映射
- `freeze_dialogue_integrity`
  锁对白事实
- `validate_storyboard_density`
  检查分镜密度
- `validate_scene_reference_presence`
  检查场景引用

所以这条链不是一个“随便润色”的脚本，而是一个带强保护的结构化重写器。

---

### 2.5 它到底改哪些东西，不改哪些东西

#### 它优先改

- `master_timeline`
- `prompt_text`
- `continuity_bridge`
- `audio_design`

#### 它明确不改

- point 顺序
- `@图片N` 映射关系
- 人物身份
- 场景事实
- 对白事实
- 剧情因果
- 剧情结论

所以交接时可以把它描述成：

`锁事实层、重写表现层`

这是最准确的 6 个字。

---

### 2.6 怎么运行

最常见是交互式运行：

```bash
./run_seedance_style_transfer.sh
```

它会交互选择：

- 哪个剧
- 哪一集
- 哪些 `point_id`
- 每批处理多少条
- 输出模式是 `sidecar` 还是 `overwrite`

也支持非交互参数，例如：

- `--storyboard-json`
- `--storyboard-md`
- `--point-ids`
- `--output-mode sidecar|overwrite`
- `--non-interactive`

---

### 2.7 最终产物是什么

如果走 `sidecar`，典型产物是：

- `outputs/<剧名>-gpt/epXX/02-seedance-prompts.style-transfer__openai__gpt-5.4.json`
- `outputs/<剧名>-gpt/epXX/02-seedance-prompts.style-transfer__openai__gpt-5.4.md`
- `outputs/<剧名>-gpt/epXX/02-seedance-prompts.style-transfer__openai__gpt-5.4.report.md`
- `outputs/<剧名>-gpt/epXX/02-seedance-prompts.style-transfer__openai__gpt-5.4.plan.json`

其中：

- `.md/.json`
  是改写后的分镜包
- `.report.md`
  是改动报告和 diff 摘要
- `.plan.json`
  是本轮模板映射、point 选择、输出模式、warning 的记录

如果走 `overwrite`：

- 原 `02-seedance-prompts.md/json` 会先备份
- 新结果会直接覆盖原文件

#### 很重要的一点

下游 `run_seedance_api_generation.sh` 默认读取的仍然是：

- `outputs/<剧名>-gpt/epXX/02-seedance-prompts.md`

所以：

- 只想看效果，用 `sidecar`
- 真要影响下游提交链，通常用 `overwrite`

---

## 3. 第三大类：`run_openai_agent_flow.sh`

这是从剧本走到导演讲戏本、服化道提示词和 Seedance 提示词的主生产链。

一句话定义：

`它是当前项目从剧本到 01-director-analysis.md / character-prompts.md / scene-prompts.md / 02-seedance-prompts.md 的核心生产 orchestrator。`

---

### 3.1 入口与主调用链

入口脚本：

```bash
./run_openai_agent_flow.sh run
```

主调用链：

```text
run_openai_agent_flow.sh
-> scripts/interactive_pipeline_launcher.py
-> scripts/run_openai_agent_flow.py
-> generate_director_analysis.py
-> generate_art_assets.py
-> generate_seedance_prompts.py
-> generate_seedance_prompt_refine.py（可选）
```

默认阶段顺序：

1. `explosive_rewrite` 可选
2. `director_analysis`
3. `art_design`
4. `storyboard`
5. `seedance_prompt_refine` 可选

---

### 3.2 先分清两个层面：agents 层 vs 实际执行层

这是交接时最容易讲混的地方。

#### A. 概念 / 编排层：OpenAI agents 映射

相关文件在：

- `openai_agents/README.md`
- `skills/openai_flow/producer_workflow.md`
- `skills/openai_flow/director_workflow.md`
- `skills/openai_flow/art_workflow.md`
- `skills/openai_flow/storyboard_workflow.md`
- `prompts/agents/producer.md`
- `prompts/agents/director.md`
- `prompts/agents/art.md`
- `prompts/agents/storyboard.md`
- `prompts/agents/explosive.md`

这层的作用更偏：

- OpenAI-native agent 概念映射
- 角色职责说明
- 从旧 `.claude` 工作流迁移到 OpenAI agent 思维方式

也就是说，这一层主要回答：

- Producer Agent 管总调度
- Director Agent 做导演分析
- Art Agent 做服化道
- Storyboard Agent 做分镜

#### B. 当前真实落盘执行层：repo pipeline

真正把文件写到仓库里的，当前主要还是这几个脚本：

- `scripts/generate_director_analysis.py`
- `scripts/generate_art_assets.py`
- `scripts/generate_seedance_prompts.py`
- `scripts/generate_seedance_prompt_refine.py`

也就是说：

- `openai_agents/` 和 `skills/openai_flow/` 更像“编排概念层”
- `generate_*.py` 才是“当前真实生产执行层”

交接时最好明确说：

`现在 run_openai_agent_flow.sh 的主体依然是 deterministic repo pipeline；OpenAI agents 目录是当前工作流的 OpenAI-native 角色映射和迁移层。`

---

### 3.3 这个 workflow 用到了哪些 skill

这一条链里，真正会被阶段 prompt 吃进去的 production skill 主要是：

#### 导演阶段

- `skills/production/director-skill/SKILL.md`
- `skills/production/script-analysis-review-skill/SKILL.md`
- `skills/production/compliance-review-skill/SKILL.md`

#### 服化道阶段

- `skills/production/art-design-skill/SKILL.md`
- `skills/production/art-direction-review-skill/SKILL.md`
- `skills/production/compliance-review-skill/SKILL.md`

#### Seedance 分镜阶段

- `skills/production/seedance-storyboard-skill/SKILL.md`
- `skills/production/seedance-storyboard-skill/seedance-prompt-methodology.md`
- `skills/production/seedance-prompt-review-skill/SKILL.md`
- `skills/production/compliance-review-skill/SKILL.md`

这些 skill 不是摆设，而是被各阶段 `generate_*.py` 在构造 prompt 时真实读取的。

---

### 3.4 阶段一：导演讲戏本

执行脚本：

- `scripts/generate_director_analysis.py`

主要 prompt：

- `prompts/director_analysis/draft_system.md`
- `prompts/director_analysis/draft_user.md`
- `prompts/director_analysis/review_system.md`
- `prompts/director_analysis/review_user.md`

主要 skill：

- `skills/production/director-skill/SKILL.md`
- `skills/production/script-analysis-review-skill/SKILL.md`
- `skills/production/compliance-review-skill/SKILL.md`

这个阶段的职责是：

- 把剧本拆成剧情点
- 给每个剧情点写导演讲戏
- 提取人物清单
- 提取场景清单
- 标记素材状态：新增 / 复用 / 变体

主要产物：

- `outputs/<剧名>-gpt/epXX/01-director-analysis.md`
- `outputs/<剧名>-gpt/epXX/01-director-analysis__openai__gpt-5.4.json`

这个阶段还会读取：

- `series_context.json`
- `genre_reference_bundle`
- 当前已有 `assets/.../character-prompts.md`
- 当前已有 `assets/.../scene-prompts.md`

所以它不是纯单集写作，而是带连续性和复用判断的导演阶段。

---

### 3.5 阶段二：服化道

执行脚本：

- `scripts/generate_art_assets.py`

主要 prompt：

- `prompts/art_assets/system.md`
- `prompts/art_assets/draft_user.md`
- `prompts/art_assets/review_system.md`
- `prompts/art_assets/review_user.md`

主要 skill：

- `skills/production/art-design-skill/SKILL.md`
- `skills/production/art-direction-review-skill/SKILL.md`
- `skills/production/compliance-review-skill/SKILL.md`

这个阶段的职责是：

- 生成人物提示词
- 生成场景提示词
- 只为新增/变体项补设计
- 保持跨集可复用

主要产物：

- `assets/<剧名>-gpt/character-prompts.md`
- `assets/<剧名>-gpt/scene-prompts.md`

这一阶段也可能读取：

- `episode_analysis`
- `01-director-analysis`
- `series_context.json`
- `genre_reference_bundle`

所以服化道阶段本身也是吃上下游经验的，不是只吃导演分析。

---

### 3.6 阶段三：Seedance 分镜

执行脚本：

- `scripts/generate_seedance_prompts.py`

主要 prompt：

- `prompts/seedance_storyboard/draft_system.md`
- `prompts/seedance_storyboard/draft_user.md`
- `prompts/seedance_storyboard/review_system.md`
- `prompts/seedance_storyboard/review_user.md`

主要 skill：

- `skills/production/seedance-storyboard-skill/SKILL.md`
- `skills/production/seedance-storyboard-skill/seedance-prompt-methodology.md`
- `skills/production/seedance-prompt-review-skill/SKILL.md`
- `skills/production/compliance-review-skill/SKILL.md`

这个阶段的职责是：

- 把导演剧情点转成 `prompt_entries`
- 建立素材对应表
- 分配 `@图片N`
- 生成 `master_timeline`
- 生成 `prompt_text`
- 做 review pass 和规则校验

主要产物：

- `outputs/<剧名>-gpt/epXX/02-seedance-prompts.md`
- `outputs/<剧名>-gpt/epXX/02-seedance-prompts__openai__gpt-5.4.json`

#### 这一阶段如何调用第一大类学出来的模板库

这是文档里必须写清楚的关键点。

`generate_seedance_prompts.py` 会读取：

- `analysis/<剧名>/seedance_purpose_skill_library.json`
- `analysis/<剧名>/seedance_purpose_template_library.json`

然后构造：

- `seedance_story_point_guidance`

再把这份 guidance 注入到 `seedance_storyboard/draft_user.md` 和 review prompt 里。

所以：

- `run_series_pipeline.sh`
  负责学模板
- `run_openai_agent_flow.sh`
  负责真正用模板去生成 `02-seedance-prompts`

这是当前项目里最核心的上下游关系之一。

---

### 3.7 阶段四：Seedance Prompt Refine

执行脚本：

- `scripts/generate_seedance_prompt_refine.py`

主要 prompt：

- `prompts/seedance_prompt_refine/system.md`
- `prompts/seedance_prompt_refine/user.md`
- `prompts/seedance_prompt_refine/techniques.md`

作用：

- 不改剧情
- 在已有 `02-seedance-prompts` 基础上继续优化可执行性和表现质量

这个阶段是 `run_openai_agent_flow.sh` 的可选收尾增强，不是主生成阶段。

---

### 3.8 这个 workflow 最应该怎么解释

最准确的说法是：

`run_openai_agent_flow.sh` 是当前从剧本到导演讲戏本、服化道提示词、Seedance 提示词的主生产 orchestrator。它一边使用 production skills 和阶段 prompt 进行 deterministic 落盘，一边在概念层保留了 OpenAI agents 的角色映射。`

---

## 4. 第四大类：任务提交链

这一类你前面已经提过，这里只保留当前版的最小结构摘要。

### 4.1 从 `02-seedance-prompts` 到 API 脚本

入口：

- `./run_seedance_api_generation.sh`

产物：

- `outputs/<剧名>-gpt/epXX/Pxx__seedance_api.sh`
- `outputs/<剧名>-gpt/epXX/Pxx__seedance_api_payload.template.json`
- `outputs/<剧名>-gpt/epXX/Pxx__seedance_api_references.json`

### 4.2 上传参考图到 TOS

入口：

- `./run_upload_seedance_refs.sh`

产物：

- `outputs/<剧名>-gpt/epXX/Pxx__seedance_api_urls.env`
- `outputs/<剧名>-gpt/epXX/Pxx__seedance_api_uploaded_refs.json`

### 4.3 真正提交 Seedance 视频任务

单条：

```bash
export ARK_API_KEY=你的火山方舟APIKey
bash outputs/<剧名>-gpt/epXX/P01__seedance_api.sh
```

批量：

```bash
export ARK_API_KEY=你的火山方舟APIKey
SEEDANCE_BATCH_MAX_WORKERS=20 bash ./run_seedance_batch_submit.sh
```

### 4.4 另一条并行提交链：直接消费 beat catalog

入口：

- `./run_seedance_beat_catalog_submit.sh`

用途：

- 直接从 `analysis/<剧名>/<epXX>/seedance_beat_catalog.json`
  提交视频任务
- 不经过 `02-seedance-prompts -> Pxx__seedance_api.sh` 这条链

适合实验 beat learning 结果本身，而不是完整生产链。

---

## 5. 交接时最值得强调的 8 句话

- `run_series_pipeline.sh` 是项目的上游学习引擎，不只是视频转剧本。
- 它学出来的“剧本分镜经验”主要沉淀在 `series_strength_playbook_draft`、`series_bible`、`series_context`。
- 它学出来的“Seedance 模板经验”主要沉淀在 `seedance_beat_catalog`、`seedance_purpose_skill_library`、`seedance_purpose_template_library`、`prompt_library/SEARCH_INDEX.json`。
- `run_openai_agent_flow.sh` 的分镜阶段会直接读取 `seedance_purpose_skill_library.json` 和 `seedance_purpose_template_library.json`。
- `run_seedance_style_transfer.sh` 不改剧情事实，它做的是“锁事实层、重写表现层”。
- `run_seedance_style_transfer.sh` 的三组关键 prompt 分别是：模板匹配、结构计划、正式改写。
- `openai_agents/` 和 `skills/openai_flow/` 更偏概念编排层；当前真实落盘主体还是 `scripts/generate_*.py`。
- 如果 style transfer 的结果要进入下游任务提交，通常要用 `overwrite`，否则下游默认还是读原始 `02-seedance-prompts.md`。
