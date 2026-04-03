请基于下列输入，输出高质量 `episode_analysis` JSON。

核心任务：
1. 还原剧情、人物关系、关键场景、对白和情绪转折。
2. 判断本集所属题材、叙事装置和观众期待。
3. 抽取本集最抓人的开头、情绪支付点和结尾卡点策略。
4. 深入分析本集的镜头语言、服化道与空间视觉母题。
5. 给出可直接影响后续剧本、人物设计、场景设计和分镜设计的下游指导。
6. 分析这部剧当前做得好的地方，并把优点抽象成可复用的 playbook / skill 经验。

硬性要求：
1. 尽量还原剧情、人物关系、关键场景、对白、情绪转折。
2. 所有 `story_beats` 必须按时间顺序排列。
3. 关键判断附 `evidence`。
4. 无法确认的信息放入 `known_gaps`，不要臆造。
5. 如果提供了跨集连续性参考，只能用于实体对齐和承接判断；若与当前视频证据冲突，必须以当前视频为准，并把冲突写进 `continuity_notes`。
6. `genre_classification`、`hook_profile`、`camera_language_analysis`、`art_direction_analysis`、`storyboard_blueprint`、`downstream_design_guidance` 都必须写实用结论，不要写空话。
7. `story_beats` 必须按时间顺序输出，并额外补充每个 beat 的 `visual_focus`、`camera_language`、`art_direction_cues`、`storyboard_value`。
8. 如能匹配题材经验库，可将其作为参考；但若与当前视频事实冲突，必须服从当前视频。
9. `series_learning_extraction` 必须优先说明：后续人物设计、服化道、场景设计、分镜设计、对白时间安排和 Seedance 生成最该复用什么；并至少填写 `character_design_rules`、`costume_makeup_rules`、`scene_design_rules`、`camera_language_rules`、`storyboard_execution_rules`、`dialogue_timing_rules`、`continuity_guardrails`、`negative_patterns`。
10. `series_learning_extraction` 中的每条规则都必须去除本剧专属名字和一次性设定，优先抽象成题材级、角色类型级、场景类型级或镜头执行级经验；像“黄雨晗必须…”“叶枫必须…”“某府邸必须…”这种点名式规则不能直接写入最终经验字段。
10. 为兼容旧流程，你仍需同步填写现有兼容字段，如 `reusable_playbook_rules`、`reusable_skill_rules`、`camera_language_patterns`、`storyboard_execution_patterns`，但这些兼容字段应从上述下游规则中提炼，不要另外发散。
11. 如果给了“用户已确认题材”，那么 `genre_classification.primary_genre` 必须从用户已确认题材里精确选择一个，`secondary_genres` 也只能从用户已确认题材里精确选择，不能擅自改写成自由发挥的新标签。
12. 如果你强烈认为当前视频还体现了用户未确认的新题材，不要直接改写 `primary_genre` 或 `secondary_genres`，而要写入 `genre_override_request`，等待用户确认。

视频到剧本方法论参考：
{{skill_text}}

题材路由说明：
{{genre_routing_note}}

用户已确认题材约束：
{{confirmed_genre_block}}

题材经验参考：
{{playbook_reference_block}}

输入摘要：
{{input_summary}}
