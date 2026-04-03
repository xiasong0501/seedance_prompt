请你根据同集多个剧本版本和已完成的爆款诊断，写出更强的强化版剧本。
要求保留故事主线和人物连续性，但显著加强开篇钩子、冲突效率、人物欲望、情绪兑现和结尾卡点。
同时要参考当前剧原生题材与目标风格题材，把改稿往目标风格上推，但不能硬改到失真。

剧名：{{series_name}}
集数：{{episode_id}}
目标受众：{{target_audience}}
目标改稿风格：{{style_target_label}}
{{extra_rules_block}}

硬性要求：
1. 输出的剧本仍使用当前项目已有的剧本格式，便于后续 director-skill 继续处理。
2. 不要为了猎奇而破坏人物可信度，不要添加不存在的核心剧情支线。
3. 旁白要更利落、更有钩子，避免重复解释同一信息。
4. 每个分场都要服务于推进局势，而不是平铺信息。
5. 集尾必须比原稿更像“点击下一集的临界点”。
6. 如原稿中有明显过曝、软色情或易误杀表述，要改得更安全但仍保留张力。
7. top_improvements 写“这版整体比原稿更强的三到五个总变化”。
8. applied_plan 写“你具体怎么改”，例如前置哪句旁白、压缩哪段解释、把哪个冲突提前。
9. explosive_insertions 写“爆款点具体加在了哪里”，例如开头第1场、结尾卡点、某段台词、某个动作反差。
10. style_adaptation_notes 写“为了靠近目标风格，你采用了哪些题材打法”。
11. change_log 写“你具体改了哪些地方”，要让人工审核的人能顺着查。
12. style_target_label 必须照抄当前目标改稿风格。
13. 任何改动都不能破坏与上一集、下一集的承接关系；若某个桥段必须保留以服务后续集数，就算不够炸，也要保留。
14. 如果当前集上下文卡或整剧上下文明确标出“必须保留”或“不可改坏”的项，必须遵守。

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

改稿模板参考：
{{rewrite_template}}

爆款诊断结果：
{{analysis_result_json}}

原始版本：
{{variants_block}}
