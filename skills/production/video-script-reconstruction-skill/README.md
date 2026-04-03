# 视频理解与剧本重建题材资产说明

这个目录下与题材相关的内容，现在分成三层：

## 1. 推荐维护入口：`genres/`

这是当前真正推荐维护的入口。

结构是：

- `genres/<题材目录>/playbook.json`
- `genres/<题材目录>/skill.md`

说明：

- 现在推荐直接用中文题材名做目录，例如：
  - `genres/先婚后爱/`
  - `genres/宅斗权谋/`
  - `genres/重生/`

其中：

- `playbook.json`
  负责存放结构化题材经验，便于程序匹配、连续性沉淀、调试落盘
- `skill.md`
  负责存放给大模型直接注入的长文本题材方法论

也就是说，如果你后续要改“某个题材到底该怎么分析、怎么重建、怎么指导下游”，优先改 `genres/`。

## 2. 旧概念一：`playbooks/genre-playbook-library.json`

这个文件代表“结构化题材经验总表”的旧组织方式。

它的特点是：

- 结构固定
- 适合程序读取
- 适合做题材匹配
- 适合被连续性层沉淀和复用

你可以把它理解为：

- 更像“题材知识库”
- 更像“规则清单”

## 3. 旧概念二：`genre_skills/`

这个目录代表“题材长文本方法论”的旧组织方式。

它的特点是：

- 每个题材一份长文本说明
- 更适合直接注入 prompt
- 更适合描述该题材在分析和剧本重建时的注意事项

你可以把它理解为：

- 更像“专家操作手册”
- 更像“给模型看的题材说明书”

## 两者区别

`genre-playbook-library.json` 和 `genre_skills/` 都在服务题材理解，但职责不同：

- `genre-playbook-library.json` 偏结构化、偏程序可读、偏状态沉淀
- `genre_skills/` 偏长文本、偏模型可读、偏方法论约束

因此它们“概念相关”，但并不是完全同一种东西。

## 为什么现在改成 `genres/<题材>/playbook.json + skill.md`

因为这样最方便维护：

- 一个题材的结构化经验和长文本经验放在一起
- 后续新增题材时不需要同时在多个目录来回改
- 更容易定位“这个题材到底有哪些资产”

## 当前代码实际如何使用

- `genre_routing.py`
  会优先从 `genres/` 读取所有题材的 `playbook.json` 和 `skill.md`
- `providers/openai_adapter.py`
  和 `providers/gemini_adapter.py`
  会把命中的题材 `skill.md` 注入到分析/剧本重建 prompt 里
- `pipelines/continuity_manager.py`
  会把命中的 `playbook.json` 沉淀进整剧连续性状态

## 维护建议

- 要新增题材：在 `genres/` 下新增一个题材目录
- 要改题材经验：优先改 `genres/<题材>/playbook.json`
- 要改题材方法论：优先改 `genres/<题材>/skill.md`
- 当前正式题材包和扁平浏览文件都已经中文化，方便维护和检索
- `playbooks/` 和 `genre_skills/` 目前主要保留为兼容与认知说明层，不建议再作为主维护入口
