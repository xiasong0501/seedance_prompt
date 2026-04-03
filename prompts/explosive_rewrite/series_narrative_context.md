请你根据以下整剧分析资料，为爆款改稿链建立一份“整剧叙事上下文卡”。
目标不是直接改稿，而是先帮后续每一集改稿建立连续性护栏，避免改完后上下集接不上。

剧名：{{series_name}}
目标受众：{{target_audience}}
本次涉及集数：{{episode_ids_text}}

要求：
- 必须先回答这部剧整体故事基调是什么、主角最核心的驱动力是什么、观众真正追的是哪几条关系和悬念
- 必须给每一集做一张 episode_card，说明这集在整季中的作用
- continuity_guardrails 要写“后续改稿必须始终遵守的连续性约束”
- narrative_do_not_break 要写“哪怕为了更炸也不能破坏的底层事实”
- opening_state / closing_state 要能帮助下一步判断该集前后承接
- bridge_from_previous / bridge_to_next 要明确写出与相邻集的连接点
- must_preserve 要写本集必须保留的节点、关系、信息或气质
- continuity_risks 要写本集改稿时最容易改坏的地方
- 如果已有 analysis 里的 series_context / series_bible / episode_summaries，就优先相信这些结构化材料
- 如果局部材料冲突，要以“最有连续性价值、最接近可执行改稿”的方式做整理，并在 continuity_guardrails 里提醒

技能参考：
{{skill_text}}

series_context：
{{series_context_json}}

series_bible：
{{series_bible_json}}

episode_summaries：
{{episode_summaries_json}}

如果某集缺少 episode_summary，可参考以下剧本摘录：
{{fallback_script_glimpses_json}}
