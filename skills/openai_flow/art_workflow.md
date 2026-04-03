# 服化道 Agent 工作流 Skill

> 使用位置：
> - 由 [flow.py](/mnt/myvg/Agent/Seedance2.0_AI/openai_agents/flow.py) 注入给 `Art Agent`
> - 属于 `./run_openai_agent_flow.sh` 的 OpenAI Agent 编排层说明
> - 作用方式：作为 agent instructions 的一部分，帮助模型判断人物/场景提示词应该达到什么标准

## 目标

产出尽可能强的：

- `assets/<剧名>-gpt/character-prompts.md`
- `assets/<剧名>-gpt/scene-prompts.md`

## 上游输入优先级

1. 导演讲戏本
2. 同剧已有 assets
3. 单集分析与整剧连续性上下文

## 人物提示词标准

每条人物提示词都应覆盖：

- 年龄与身份感
- 脸型与五官
- 发型
- 身形体态
- 服装轮廓
- 材质面料
- 配饰
- 鞋履
- 整体气质
- 设定图版式要求

## 场景提示词标准

每条场景宫格提示词都应覆盖：

- 空间布局
- 时间段
- 光照方向与颜色
- 主色调
- 材质纹理
- 道具与锚点物
- 景深与层次
- 氛围

## 生产规则

- 尽量只为新增或变体项写新内容
- 不重复重写可直接复用的历史素材

