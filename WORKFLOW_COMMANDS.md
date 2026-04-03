# 工作流运行手册

这份手册按当前仓库里的真实入口整理，重点说明：

- 每条命令适合什么场景
- 是否交互
- 常用运行方式
- 主要配置文件
- 关键产物落在哪里
- Seedance 图片切分、上传到 TOS、再调用视频 API 的完整链路

建议先记住三条主线：

- `视频 -> 剧本`：`./run_video_pipeline.sh` 或 `./run_series_pipeline.sh`
- `剧本 -> 导演/服化道/分镜`：`./run_openai_agent_flow.sh`
- `分镜 -> 参考图 -> TOS -> Seedance 视频 API`：`./run_nano_banana_assets.sh`、`./run_seedance_api_generation.sh`、`./run_upload_seedance_refs.sh`

---

## 1. 单集：从视频生成剧本

适用场景：
- 已有单集视频，想先验证这一集的视频理解和剧本重建效果

入口脚本：
- `./run_video_pipeline.sh`

主要配置：
- `config/video_pipeline.local.json`

说明：
- 这条链目前是配置驱动，不弹交互
- 会先做本地预处理，再调用 OpenAI 或 Gemini 做视频理解与剧本重建
- 还会写入题材路由 debug、连续性上下文和单集分析结果

常用方式：

```bash
./run_video_pipeline.sh
```

主要产物：
- `analysis/<剧名>/<epXX>/preprocess/...`
- `analysis/<剧名>/<epXX>/episode_analysis__<provider>__<model>.json`
- `analysis/<剧名>/<epXX>/genre_routing_debug__<provider>__<model>.json`
- `analysis/<剧名>/<epXX>/genre_routing_debug__<provider>__<model>.md`
- `script/<剧名>/epXX__<provider>__<model>.md`

---

## 2. 批量：从视频生成剧本

适用场景：
- 已有整部剧的视频目录，想批量处理 `ep01-ep20`
- 兼容：
  - `videos/<剧名>/`
  - `videos/<分类>/<剧名>/`

入口脚本：
- `./run_series_pipeline.sh`

主要配置：
- `config/series_pipeline.local.json`

交互模式说明：
- 默认就是交互模式
- 会先列出 `videos/` 下可选的视频目录
- 再输入起始 / 结束集数
- 再确认题材
- 可选择是否记录时间和 token 统计报告
- 跑完后还会追问是否需要“积累经验”

常用方式：

```bash
./run_series_pipeline.sh
```

只预览，不真正调用模型：

```bash
./run_series_pipeline.sh preview
```

交互正式运行：

```bash
./run_series_pipeline.sh run
```

交互正式运行，并强制开启统计：

```bash
./run_series_pipeline.sh run true
```

交互正式运行，并强制关闭统计：

```bash
./run_series_pipeline.sh run false
```

完全按配置文件运行：

```bash
./run_series_pipeline.sh config
```

完全按配置文件运行，并强制开启统计：

```bash
./run_series_pipeline.sh config true
```

主要产物：
- `analysis/<剧名>/<epXX>/...`
- `script/<剧名>/epXX__<provider>__<model>.md`
- 整剧题材修正建议：
  - `analysis/<剧名>/genre_override_request__<provider>__<model>.json`
  - `analysis/<剧名>/genre_override_request__<provider>__<model>.md`

如果开启统计，还会额外生成：
- 单集统计：
  - `analysis/<剧名>/<epXX>/metrics/episode_metrics__*.json`
  - `analysis/<剧名>/<epXX>/metrics/episode_metrics__*.md`
- 整批统计：
  - `analysis/<剧名>/batch_runs/metrics_summary__*.json`
  - `analysis/<剧名>/batch_runs/metrics_summary__*.md`

---

## 3. 爆款剧本改稿

适用场景：
- 同一集已经有多个剧本版本，例如 GPT / Gemini 各自生成的版本
- 想先评分比较，再生成更抓眼球的强化版
- 想结合现有题材库，把剧本往某个风格题材上强化，例如 `萌宝团宠`、`爱情`、`重生`

入口脚本：
- `./run_explosive_actor.sh`

主要配置：
- `config/explosive_rewrite_pipeline.local.json`

交互模式说明：
- 默认就是交互模式
- 会先列出 `script/` 下已有剧本目录供你选择
- 再输入起始 / 结束集数
- 再选择“希望改向的爆款风格题材”
- 目标风格题材优先从 `skills/production/video-script-reconstruction-skill/genres/` 里选
- 正式改稿前会先建立一份“整剧叙事上下文卡”
- 默认只比较“基础原始剧本版本”，不会把历史 `__explosive` 版本再次混入本轮输入

常用方式：

```bash
./run_explosive_actor.sh
./run_explosive_actor.sh preview
./run_explosive_actor.sh run
./run_explosive_actor.sh run true
./run_explosive_actor.sh config true
```

主要产物：
- `script/<剧名>/epXX__openai__gpt-5.4__<目标风格>__explosive.md`
- `analysis/<剧名>/explosive-actor-gpt/<epXX>/episode_explosive_report__openai__<model>__<目标风格>.md`
- `analysis/<剧名>/explosive-actor-gpt/<epXX>/episode_explosive_change_log__openai__<model>__<目标风格>.md`
- `analysis/<剧名>/explosive-actor-gpt/series_narrative_context__openai__<model>.json`
- `analysis/<剧名>/explosive-actor-gpt/series_narrative_context__openai__<model>.md`

如果开启统计，还会额外生成：
- `analysis/<剧名>/explosive-actor-gpt/metrics_summary__openai__<model>__epXX-epYY__<目标风格>.json`
- `analysis/<剧名>/explosive-actor-gpt/metrics_summary__openai__<model>__epXX-epYY__<目标风格>.md`

---

## 4. OpenAI-native 完整生产链

适用场景：
- 已经有剧本，希望继续生成：
  - 导演讲戏本
  - 人物提示词
  - 场景提示词
  - Seedance 分镜提示词

入口脚本：
- `./run_openai_agent_flow.sh`

主要配置：
- `config/openai_agent_flow.local.json`

竖横屏说明：
- `quality.frame_orientation` 控制上游导演 / 服化道 / 分镜的目标画幅思路
- 当前默认是 `9:16竖屏`
- 如果要改回横屏，优先改这里，再配合下游 `SEEDANCE_RATIO=16:9`

交互模式说明：
- 默认交互运行
- 会先让你选择剧本目录，再选择起止集数
- 会让你选择后续阶段使用的剧本来源：
  - `爆改版优先`
  - `原始基础版优先`
  - `仅爆改版`
- 会让你确认题材，优先从现有题材库里选择
- 会让你选择目标媒介 / 风格，例如：
  - `漫剧`
  - `电影`
  - `短剧`
  - `电视剧`
- 会问你是否先运行爆款改稿阶段
- 会问你是否记录时间和 token 统计
- 对于没有现成 `analysis/<剧名>/series_context.json` 的新剧本，也能基于题材库冷启动
- 会自动生成或复用：
  - `analysis/<剧名>/openai_agent_flow/genre_reference_bundle.json`
  - `analysis/<剧名>/openai_agent_flow/genre_reference_bundle.md`
- 已有上游产物时，会自动跳过对应阶段

常用方式：

```bash
./run_openai_agent_flow.sh
./run_openai_agent_flow.sh preview
./run_openai_agent_flow.sh run
./run_openai_agent_flow.sh run true
./run_openai_agent_flow.sh run false
./run_openai_agent_flow.sh config
./run_openai_agent_flow.sh config true
```

主要产物：
- `outputs/<剧名>/<epXX>/01-director-analysis.md`
- `outputs/<剧名>/<epXX>/01-director-analysis__openai__gpt-5.4.json`
- `assets/<剧名>-gpt/character-prompts.md`
- `assets/<剧名>-gpt/scene-prompts.md`
- `outputs/<剧名>/<epXX>/02-seedance-prompts.md`
- `outputs/<剧名>/<epXX>/02-seedance-prompts__openai__gpt-5.4.json`

如果开启统计，还会额外生成：
- `analysis/<剧名>/openai_agent_flow/metrics_summary__openai__<model>__epXX-epYY.json`
- `analysis/<剧名>/openai_agent_flow/metrics_summary__openai__<model>__epXX-epYY.md`

---

## 5. 单独生成服化道提示词

适用场景：
- 只想重跑角色提示词 / 场景提示词
- 不想重跑整条 OpenAI 生产链

入口脚本：
- `./run_art_assets_pipeline.sh`

主要配置：
- `config/art_assets_pipeline.local.json`

常用方式：

```bash
./run_art_assets_pipeline.sh
```

主要产物：
- `assets/<剧名>-gpt/character-prompts.md`
- `assets/<剧名>-gpt/scene-prompts.md`

---

## 6. 生成 Nano Banana 角色图 / 场景图

适用场景：
- 已经有：
  - `assets/<剧名>-gpt/character-prompts.md`
  - `assets/<剧名>-gpt/scene-prompts.md`
- 希望用 Gemini 图片模型出角色图和场景图

入口脚本：
- `./run_nano_banana_assets.sh`

主要配置：
- `config/nano_banana_assets.local.json`

交互模式说明：
- 会先列出 `script/` 下已有剧本目录
- 再输入起始 / 结束集数
- 会自动优先选：
  - `__explosive.md`
  - `__openai__gpt-5.4.md`
  - `__gemini__gemini-3-pro-preview.md`
- 会自动优先读取：
  - `assets/<剧名>-gpt/character-prompts.md`
  - `assets/<剧名>-gpt/scene-prompts.md`
- 场景图生成完成后，会自动读取：
  - `outputs/<剧名>/<epXX>/02-seedance-prompts.md`
  - 其中的 `## 素材对应表`
- 然后自动把场景宫格切成单张场景图并命名

常用方式：

```bash
./run_nano_banana_assets.sh
./run_nano_banana_assets.sh preview
./run_nano_banana_assets.sh run
./run_nano_banana_assets.sh config
```

主要产物：
- `assets/<剧名>-gpt/generated/<模型>/<epXX>/characters/...`
- `assets/<剧名>-gpt/generated/<模型>/<epXX>/scenes/...`
- `assets/<剧名>-gpt/generated/<模型>/<epXX>/generation_manifest.json`
- 自动切分后的单场景图：
  - `assets/<剧名>-gpt/generated/<模型>/<epXX>/scene_materials/...`
- 场景切分清单：
  - `assets/<剧名>-gpt/generated/<模型>/<epXX>/scene_material_manifest.json`

说明：
- 同角色变体会优先参考已有角色图
- 同名同设定资产会跨集复用，不会重复生成
- 场景图会强约束为背景 / 环境图，不应出现主体人物

---

## 7. 生成 Seedance API 调用脚本

适用场景：
- 已经有：
  - `outputs/<剧名>/<epXX>/02-seedance-prompts.md`
  - `assets/<剧名>-gpt/generated/.../characters/...`
  - `assets/<剧名>-gpt/generated/.../scene_materials/...`
- 想针对某个 `Pxx` 场景，生成可直接调用火山方舟 Seedance 2.0 API 的脚本

入口脚本：
- `./run_seedance_api_generation.sh`

说明：
- 默认交互模式
- 会从 `outputs/` 下列出已有 `02-seedance-prompts.md` 的剧
- 让你选择 `epXX`
- 再让你选择 `Pxx`
- 自动读取该场景的：
  - Seedance 提示词
  - `主要引用`
  - `素材对应表`
- 自动从 `assets/` 里找对应人物图和场景图
- 会生成调用 API 所需的 shell 脚本和 payload 模板

常用方式：

```bash
./run_seedance_api_generation.sh
```

主要产物：
- `outputs/<剧名>/<epXX>/Pxx__seedance_api.sh`
- `outputs/<剧名>/<epXX>/Pxx__seedance_api_payload.template.json`
- `outputs/<剧名>/<epXX>/Pxx__seedance_api_references.json`

说明：
- 这一步只生成 API 调用脚本，不会自动上传参考图
- 生成的 shell 脚本会自动读取同目录下的：
  - `Pxx__seedance_api_urls.env`
  如果该文件存在，就直接把其中的公网 URL 注入 payload

---

## 8. 上传 Seedance 参考图到 TOS 并回填 URL

适用场景：
- 已经通过 `./run_seedance_api_generation.sh` 生成了：
  - `Pxx__seedance_api_references.json`
- 想把这些本地参考图上传到 TOS bucket
- 想自动生成可被 `Pxx__seedance_api.sh` 直接读取的 URL env 文件

入口脚本：
- `./run_upload_seedance_refs.sh`

说明：
- 默认交互模式
- 会从 `outputs/` 下列出已有 `Pxx__seedance_api_references.json` 的剧
- 让你选择 `epXX / Pxx`
- 默认 bucket：`xiasongseedance`
- 默认 region：`cn-beijing`
- 默认 endpoint：`tos-cn-beijing.volces.com`
- 可选两种 URL 方式：
  - `公开读 URL（默认，更快；对象设为 public-read）`
  - `预签名 URL（私有桶也可用，但每张图会额外跑一次签名）`
- 支持两种凭证方式：
  - 先运行 `./tosutil config ...`
  - 或提前导出环境变量：
    - `VOLC_ACCESS_KEY_ID / VOLC_SECRET_ACCESS_KEY`
    - `TOS_ACCESS_KEY_ID / TOS_SECRET_ACCESS_KEY`
    - `TOS_AK / TOS_SK`

常用方式：

```bash
./run_upload_seedance_refs.sh
```

主要产物：
- URL env：
  - `outputs/<剧名>/<epXX>/Pxx__seedance_api_urls.env`
- 上传结果清单：
  - `outputs/<剧名>/<epXX>/Pxx__seedance_api_uploaded_refs.json`

说明：
- `Pxx__seedance_api.sh` 会自动读取 `Pxx__seedance_api_urls.env`
- 如果当前 `Pxx__seedance_api.sh` 还是旧版本，上传脚本会自动补丁它，让它支持读取 `.env`

---

## 9. 真正调用 Seedance 2.0 API 生成视频

适用场景：
- 已经有：
  - `Pxx__seedance_api.sh`
  - `Pxx__seedance_api_urls.env`
- 已经拿到火山方舟 API key

运行方式：

```bash
export ARK_API_KEY=你的火山方舟APIKey
bash outputs/<剧名>/<epXX>/Pxx__seedance_api.sh
```

现在生成出的 `Pxx__seedance_api.sh` 会自动：
- 提交 Seedance 任务
- 轮询任务状态
- 成功后自动下载 `Pxx__seedance_output.mp4`
- 如有尾帧则自动下载 `Pxx__seedance_last_frame.jpg`

说明：
- 脚本会自动读取：
  - `Pxx__seedance_api_urls.env`
- 然后渲染出：
  - `Pxx__seedance_api_payload.rendered.json`
- 最后向：
  - `https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks`
  发起请求

常用可选环境变量：
- `SEEDANCE_MODEL`
- `SEEDANCE_RATIO`
- `SEEDANCE_RESOLUTION`
- `SEEDANCE_DURATION`
- `SEEDANCE_GENERATE_AUDIO`
- `SEEDANCE_WATERMARK`

当前脚本默认以 `9:16` 竖屏提交；如果你要改回横屏，可手动覆盖 `SEEDANCE_RATIO=16:9`。

例如：

```bash
export ARK_API_KEY=你的火山方舟APIKey
export SEEDANCE_DURATION=15
export SEEDANCE_RATIO=9:16
export SEEDANCE_RESOLUTION=480p
bash outputs/成为全球女首富/ep01/P01__seedance_api.sh
```

---

## 10. 将整剧经验同步成题材库更新草稿

适用场景：
- 某一部剧已经完整或阶段性跑完
- 已经生成：
  - `analysis/<剧名>/series_strength_playbook_draft.json`
  - `analysis/<剧名>/genre_override_request__<provider>__<model>.json`
- 希望把这部剧学到的可复用经验整理成题材库更新建议

入口脚本：
- `./run_sync_series_learning.sh`

主要配置：
- `config/sync_series_learning.local.json`

说明：
- 默认进入交互模式，会先从 `analysis/` 下列出已有 `series_strength_playbook_draft.json` 的剧给你选
- 默认安全模式，不会直接覆盖正式题材库
- 会把建议写到 `genres/__drafts__/` 和兼容层 `playbooks/__drafts__/`
- 会按置信度排序，并尽量只保留更短、更有代表性的规则

常用方式：

```bash
./run_sync_series_learning.sh         # 交互选择 analysis 下已有整剧经验的剧
./run_sync_series_learning.sh run     # 同上，run 为 interactive 别名
./run_sync_series_learning.sh config  # 不交互，固定按 config 里的 series_name 运行
```

主要产物：
- `analysis/<剧名>/genre_library_update_plan__<provider>__<model>.json`
- `analysis/<剧名>/genre_library_update_plan__<provider>__<model>.md`
- `skills/production/video-script-reconstruction-skill/genres/__drafts__/sync__<剧名>__<题材>/playbook.json`
- `skills/production/video-script-reconstruction-skill/genres/__drafts__/sync__<剧名>__<题材>/skill.md`

---

## 11. 审核并入题材库 Draft

适用场景：
- `run_sync_series_learning.sh` 已经生成了 draft
- 想逐条人工审核后，再并入正式题材库

入口脚本：
- `./run_review_genre_drafts.sh`

说明：
- 会扫描：
  - `skills/production/video-script-reconstruction-skill/genres/__drafts__/`
  - `skills/production/video-script-reconstruction-skill/playbooks/__drafts__/`
- 会逐条显示：
  - 题材
  - 类型
  - 置信度
  - 候选内容
  - 原因
  - 相似的现有规则
- 输入：
  - `1` 并入
  - `0` 跳过
- 审核完成后会删除对应 draft

常用方式：

```bash
./run_review_genre_drafts.sh
```

---

## 12. 体检正式题材包

适用场景：
- 题材库越积越多，怀疑某个正式题材包里有：
  - 重复经验
  - 不适配当前题材的误学内容
  - 被截断或空泛的规则
  - `playbook.json` / `skill.md` 职责重叠
- 希望先让 LLM 给出体检报告，再由人逐条做二次判断

入口脚本：
- `./run_genre_package_audit.sh`

主要配置：
- `config/genre_package_audit.local.json`

说明：
- 默认进入交互模式，会先从正式题材目录里让你选择一个题材
- 会先做程序预检，再调用 `GPT-5.4` 生成结构化体检报告
- 报告会指出：
  - 高风险问题
  - 重复和跨题材重合
  - `playbook` 修改建议
  - `skill` 修改建议
  - 需要人工判断的问题
- 然后会在终端里逐条让你选择：
  - `1` 采纳
  - `0` 跳过
  - `2` 稍后
- 只会基于你采纳的建议生成修订草稿，默认不直接覆盖正式题材包
- 交互模式下可额外选择：
  - 先体检，再把采纳建议直接并入正式题材包
  - 直接使用已有 draft 并入正式题材包
  - 只体检并保留草稿
- 当正式并入成功后，默认会自动删除对应 draft，避免重复应用

常用方式：

```bash
./run_genre_package_audit.sh
./run_genre_package_audit.sh preview
./run_genre_package_audit.sh config
```

主要产物：
- `skills/production/video-script-reconstruction-skill/genres/__audits__/<题材>/audit__*.json`
- `skills/production/video-script-reconstruction-skill/genres/__audits__/<题材>/audit__*.md`
- `skills/production/video-script-reconstruction-skill/genres/__audits__/<题材>/decision__*.json`
- `skills/production/video-script-reconstruction-skill/genres/__audits__/<题材>/decision__*.md`
- `skills/production/video-script-reconstruction-skill/genres/__drafts__/audit__<题材>__<时间戳>/playbook.json`
- `skills/production/video-script-reconstruction-skill/genres/__drafts__/audit__<题材>__<时间戳>/skill.md`

---

## 13. 推荐使用顺序

### A. 从视频开始

1. 先生成剧本

```bash
./run_series_pipeline.sh run
```

或

```bash
./run_video_pipeline.sh
```

2. 如果希望剧本更抓人，再做爆款改稿

```bash
./run_explosive_actor.sh run
```

3. 再跑 OpenAI-native 完整生产链

```bash
./run_openai_agent_flow.sh run
```

4. 如果要出角色图 / 场景图，再跑 Nano Banana

```bash
./run_nano_banana_assets.sh run
```

5. 如果要生成 Seedance 视频，再继续：

```bash
./run_seedance_api_generation.sh
./run_upload_seedance_refs.sh
export ARK_API_KEY=你的火山方舟APIKey
bash outputs/<剧名>/<epXX>/Pxx__seedance_api.sh
```

### B. 从已有剧本开始

1. 直接跑完整生产链

```bash
./run_openai_agent_flow.sh run
```

2. 出角色图 / 场景图

```bash
./run_nano_banana_assets.sh run
```

3. 自动切分场景图、生成 Seedance API 脚本、上传参考图、再出视频

```bash
./run_seedance_api_generation.sh
./run_upload_seedance_refs.sh
export ARK_API_KEY=你的火山方舟APIKey
bash outputs/<剧名>/<epXX>/Pxx__seedance_api.sh
```

## 人物风格转换（把真人写实角色图转成非真实风格）

入口：`./run_style_transfer_assets.sh`

用途：当人物参考图过于真人写实、不适合直接作为 Seedance 参考时，先把某一集 `characters/` 下的人物图转换成统一的非真实风格。

运行方式：
```bash
./run_style_transfer_assets.sh
./run_style_transfer_assets.sh preview
./run_style_transfer_assets.sh run
./run_style_transfer_assets.sh config
```

交互内容：
- 选择剧本
- 选择章节 `epXX`
- 选择目标非真实风格
  - 国漫厚涂动画
  - 赛璐璐动画
  - 3D动画电影
  - 水墨插画
  - 绘本插画
- 可选补充风格说明

目录规则：
- 首次正式运行时，会把 `assets/<剧名>-gpt/generated/<模型>/<epXX>/characters` 重命名为 `characters-real`
- 新生成的风格化人物图写回 `characters`
- 后续 Seedance 参考图流程默认继续读取 `characters`，因此会优先使用风格化后的非真实人物图

产物：
- `assets/<剧名>-gpt/generated/<模型>/<epXX>/characters-real/`
- `assets/<剧名>-gpt/generated/<模型>/<epXX>/characters/`
- `assets/<剧名>-gpt/generated/<模型>/<epXX>/style_transfer_manifest__<style_key>.json`

配置文件：
- `config/style_transfer_assets.local.json`

典型用途：
1. 先对人物图做风格转换，避免 Seedance 把真人写实角色图判成真实人物
2. 再执行 `./run_seedance_api_generation.sh` 和 `./run_upload_seedance_refs.sh`
3. 最后运行 `outputs/<剧名>/<epXX>/Pxx__seedance_api.sh`


## `./run_migrate_claude_assets.sh`

用途：把旧的 Claude 工作流 `assets/<剧名>-claude/epXX/character-prompts.md`、`scene-prompts.md` 合并成和 GPT workflow 一样的系列级共享文件：
- `assets/<剧名>-claude/character-prompts.md`
- `assets/<剧名>-claude/scene-prompts.md`
- 每一集内容使用 `<!-- episode: epXX start/end -->` block 组织

常用：
```bash
./run_migrate_claude_assets.sh --series-name 红糖姜汁-claude
./run_migrate_claude_assets.sh --all
./run_migrate_claude_assets.sh --all --dry-run
```

说明：
- 旧的 `epXX/` prompt 文件默认保留，作为兼容历史输入
- 新的主读取入口改为顶层共享文件，便于跨集复用同一人物/场景提示词
