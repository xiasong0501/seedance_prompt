请为下面这条 Seedance point 评估候选 prompt templates 的适配度。

任务目标：
- 给每个候选模板一个 0-1 的 `match_score`
- 分数要尽量精确
- 评分主要看“结构是否适配”，不是看文本表面相似度
- `rationale` 要简洁说明为什么高分或低分
- `learning_focus` 要指出这条 point 最值得从该模板学习什么

当前 point：
{{point_json}}

候选模板：
{{candidate_templates_json}}
