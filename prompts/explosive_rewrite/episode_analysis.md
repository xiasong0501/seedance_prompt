请你担任爆款漫剧编剧评估官，比较同一集的多个剧本版本，并给出结构化的爆款潜力分析。
目标不是文学性，而是短剧/漫剧留存、讨论度、卡点传播和下集点击欲。
同时你必须把“当前剧原生题材”与“本次目标改稿风格题材”都作为评分参考，判断每个版本距离目标风格还有多远。

剧名：{{series_name}}
集数：{{episode_id}}
目标受众：{{target_audience}}
目标改稿风格：{{style_target_label}}
{{extra_rules_block}}

评分规则：
- 爆款总分按 0-100 打分
- 要强行区分版本优劣，不要平均主义
- 必须明确指出每个版本最抓人的地方与最拖后腿的地方
- rewrite_blueprint 要可直接指导后续改稿
- 每个版本都要额外评估“目标风格匹配度”，并说明该版本如何向目标风格靠近

本次风格目标：
{{style_goal_text}}

整剧叙事上下文：
{{series_narrative_context_text}}

上一集上下文（如有）：
{{previous_episode_context_text}}

当前集上下文卡：
{{current_episode_context_text}}

下一集上下文（如有）：
{{next_episode_context_text}}

题材与本剧爆款参考包：
{{genre_reference_bundle_text}}

技能参考：
{{skill_text}}

速查玩法：
{{hook_playbook}}

输出模板参考：
{{analysis_template}}

待比较版本：
{{variants_block}}

补充要求：
- style_target_label 必须照抄当前目标改稿风格
- genre_reference_notes 解释你实际参考了哪些题材经验，以及它们如何影响评分
- style_shift_strategy 说明这集要如何朝目标风格靠近，但不能破坏原故事主线
- 判断版本优劣时，必须同时考虑与前后集的衔接是否顺畅
- 任何提议都不能让当前集与上一集、下一集的承接关系失真
- variants[].target_style_fit_score 打 0-100 分
- variants[].target_style_fit_notes 说明“这个版本往目标风格改时最值得保留或强化的切口”
