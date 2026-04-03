请根据下列材料，为当前这一集生成高质量 `episode_analysis` 结构化 JSON。

你的输出必须同时满足“事实准确”和“后续生产可用”。

核心任务：
1. 还原人物、场景、对白、剧情节点和情绪推进。
2. 判断本集所属题材、叙事装置和观众期待。
3. 抽取本集最抓人的开头、情绪支付点和集尾卡点策略。
4. 深入分析本集的镜头语言、服化道与空间视觉母题。
5. 给出后续剧本重建、人物设计、场景设计、分镜设计可以直接使用的指导信息。
6. 分析这部剧当前做得好的地方，并把这些优点优先抽象为后续导演、服化道、场景设计、分镜设计和 Seedance 生成可直接复用的经验。

硬性要求：
1. 人物、场景、剧情节点、对白、证据都要尽量完整。
1.1 对白内容必须默认以 ASR Transcript 为主证据；OCR 只能在与 ASR 高度重合、明显只是同音错别字或专有名词误写时，作为谨慎校字参考。
1.2 禁止用 OCR 去补写 ASR 中不存在的整句对白，禁止因为 OCR 噪声制造重复对白、缺失对白或新增奇怪内容。
1.3 如果 OCR 与 ASR 冲突，且无法从画面明确验证只是个别字词误差，则必须保留 ASR 版本，并把分歧写入 `quality_assessment.known_gaps` 或相关 evidence 描述。
2. 如果只有画面没有明确台词，不要捏造完整对白。
3. `story_beats` 要按时间顺序排列，时间戳尽量精确。
4. 每个关键判断都要给 `evidence`。
5. 如果信息不完整，把不确定项写进 `quality_assessment.known_gaps`。
6. 如果提供了跨集连续性参考，只能用于实体对齐和承接判断；若与当前视频证据冲突，必须以当前视频为准，并把冲突写进 `continuity_notes`。
7. `genre_classification` 不能只写空泛大类，必须回答：主题材、副题材、核心叙事装置、时代语境、观众期待。
8. `hook_profile` 必须明确开头钩子、集内抓点、情绪支付点、集尾卡点策略。
9. `camera_language_analysis` 必须回答：最常用且最有效的镜头类型、运镜模式、构图重心、转场节奏、高潮视觉策略、集尾卡点视觉模式。
10. `art_direction_analysis` 必须回答：人物服装、妆发、道具、空间、灯光、色彩和材质的稳定辨识特征，且要能服务后续出图。
11. `storyboard_blueprint` 必须回答：开头钩子、冲突升级、情绪兑现、集尾按钮分别适合怎样的分镜组织，以及后续 Seedance 最应强调和避免的点。
12. `story_beats` 不能只写剧情摘要；每个 beat 还必须补 `visual_focus`、`camera_language`、`art_direction_cues`、`storyboard_value`。
13. `downstream_design_guidance` 必须能指导后续剧本、人物、场景和分镜设计，不要写成空泛鸡汤。
14. 如果题材经验库中有可匹配类型，可以参考其玩法；但若与当前视频证据冲突，必须以当前视频为准。
15. `series_learning_extraction` 必须优先产出下游生产规则，并至少补充：`character_design_rules`、`costume_makeup_rules`、`scene_design_rules`、`camera_language_rules`、`storyboard_execution_rules`、`dialogue_timing_rules`、`continuity_guardrails`、`negative_patterns`；每类尽量控制在 3-5 条，必须贴合当前视频证据，不能写空话。
16. `series_learning_extraction` 中的每条规则都必须去除本剧专属名字和一次性设定，优先抽象成题材级、角色类型级、场景类型级或镜头执行级经验；像“黄雨晗必须…”“叶枫必须…”“某府邸必须…”这种点名式规则不能直接写入最终经验字段。
16. 为兼容旧流程，你仍需同步填写现有兼容字段，如 `reusable_playbook_rules`、`reusable_skill_rules`、`camera_language_patterns`、`storyboard_execution_patterns`，但这些兼容字段应从上述下游规则中提炼，不要单独发散。
16.1 这些兼容字段只能放在 `series_learning_extraction` 对象内部，绝不能放在顶层。
16.2 `episode_id`、`title`、`language`、`source_video` 只能写在 `episode` 对象内部，绝不能在顶层重复输出。
17. 如果给了“用户已确认题材”，那么 `genre_classification.primary_genre` 必须从用户已确认题材里精确选择一个，`secondary_genres` 也只能从用户已确认题材里精确选择，不能擅自改写成自由发挥的新标签。
18. 如果你强烈认为当前视频还体现了用户未确认的新题材，不要直接改写 `primary_genre` 或 `secondary_genres`，而要写入 `genre_override_request`，等待用户确认。

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
{{transcript_block}}
{{ocr_block}}
