请基于下面输入，对选中的 Seedance point 做“锁死事实层、迁移写法层”的风格优化。

执行重点：
- base point 是事实真源。
- selected prompt templates 是本轮表现层重写的主要学习源，提供镜头语言、段落骨架、动作组织、光效质感、声音设计、转场写法、场景描述密度与拍摄手法。
- 你必须像“事实锁死的模板导演”而不是“谨慎润色器”。
- 本轮被选中的 point 默认都应出现“肉眼可见”的增强；不要只加几个修饰词就结束。
- 对高分模板，必须把它们当成“主要学习源”，至少吸收 5 类具体优点进 `master_timeline` 和 `prompt_text`。
- 除事实层外，原 prompt 的句式、分段、镜头顺序、场景描述方式都可以被彻底重写。

必须保留：
1. `point_id`、顺序、剧情职责。
2. `primary_refs` / `secondary_refs` 原样不动。
3. base point 已有的人物身份、场景事实、对白事实、关键剧情节点、因果逻辑、剧情结果。
4. 与上下条 point 的连续关系。

允许优化：
1. 先参考 style plan，把每条 point 拆成更强的段落骨架，再落正文。
2. 镜头入口与镜头推进。
3. 动作分段与交棒。
4. 光线、材质、空气感、声音床。
5. 句式节奏、空间调度、前后景层次。
6. 冗余压缩与可执行性提升。
7. 对于被选中的 point，优先把 `prompt_text` 与 `master_timeline` 一起改强。
8. 优先把原 prompt 中偏平、偏泛、偏摘要化的句子重写成更具体的镜头执行语言，不要保留原句式骨架。
9. 允许围绕同一事实重新安排镜头层动作顺序、对白前后的包裹镜头、切镜时点、尾帧交棒，只要不改变剧情因果与结果。
10. 运镜、镜头语法、拍摄手法、场景描述、材质受光、声音床，应尽量直接学习模板，不要只学词汇表面。

绝对禁止：
- 借用模板里的剧情事实、人物名、场景名、道具名。
- 改动对白事实与剧情结论。
- 新增未声明 ref。
- 因为模板很强就把 base point 改写成另一场戏。

输出前请先完成以下内部步骤：
1. 读取每条 point 的 base_entry 与 compact_preview，确认剧情职责和上下条承接。
2. 读取该 point 匹配到的 1-3 个 prompt templates，优先学习它们的通用模板 Prompt 和还原版 Prompt 的“结构骨架”。
3. 读取 style plan，把它作为本轮重构蓝图。
4. 先重建 master_timeline 的镜头结构，再让 prompt_text 跟随新的 master_timeline 成形；不要让 prompt_text 只是旧稿改几个词。
5. `base_entry` 与 `compact_preview` 里的原 prompt 摘要只用于校对事实，不得沿用原句式、原段落节奏或原场景描述方式。
6. 在不改变剧情因果与结果的前提下，优先按模板重建镜头入口、动作链、空间调度、景别变化、光效材质、声音床与尾帧交棒。
7. 对白内容与说话人必须保持 base truth，不得改写；但允许围绕对白重新组织镜头。
8. 如果改完后仍然和原 prompt_text 表层句式高度相似，或仍主要沿用原来的场景描述方式，说明改写失败，需要继续重写。

输入 1：当前任务元信息
{{job_context_json}}

输入 2：选中的 base point 包（只含本轮需要处理的 point；其中每条都包含 matched prompt templates）
{{selected_points_package_json}}

输入 3：本轮候选 prompt templates 池（只作结构与写法参考，不作事实来源）
{{selected_prompt_templates_json}}

输入 4：style plan（第一阶段已经抽象出的重构骨架）
{{style_plan_json}}

输入 5：参考剧本节选（只用于校验忠实度，可为空）
{{source_script_text}}
