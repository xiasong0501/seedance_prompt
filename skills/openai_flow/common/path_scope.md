# 路径与作用域 Skill

> 使用位置：
> - 由 [flow.py](/mnt/myvg/Agent/Seedance2.0_AI/openai_agents/flow.py) 作为 `common_docs` 注入给全部 Agent
> - 作用方式：限制 agent 的文件写入范围，确保所有产物按剧名和集数正确分仓

## 剧名作用域

- 所有输出都必须按剧名隔离
- 严禁把不同剧的 assets 或 outputs 混在一起
- 只要可能，就从 `script/<剧名>/...` 反推剧名

## 标准目标路径

- 导演分析：`outputs/<剧名>/<集数>/01-director-analysis.md`
- 分镜提示词：`outputs/<剧名>/<集数>/02-seedance-prompts.md`
- 人物提示词：`assets/<剧名>-gpt/character-prompts.md`
- 场景提示词：`assets/<剧名>-gpt/scene-prompts.md`
- 爆款改稿：`script/<剧名>/<集数>__openai__<model>__explosive.md`

## 上游产物优先级

按以下顺序使用上游材料：

1. 如果用户强调留存和点击率，优先用爆款改稿版剧本
2. 服化道和分镜阶段优先吃导演讲戏本
3. 再回退到当前集分析与整剧连续性上下文
4. 同剧已有素材优先复用

## 连续性规则

- 尊重同剧之前已经写出的系列文件
- 优先在同剧 assets 上增量扩写，而不是重写全部历史
- 只要已有文件存在，就把它当成连续性的锚点

