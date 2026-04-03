请先为选中的 Seedance point 生成“结构骨架计划”，供第二阶段正式改写使用。

要求：
- 每条 point 都要先看 base_entry，再看 matched prompt templates。
- 优先吸收模板里的：
  - 段落切分方式
  - 镜头入口
  - 动作链推进
  - 前中后景调度
  - 受光/材质/声音床
  - 尾帧交棒
  - 场景描述方式
  - 运镜与拍摄手法
- 不要复制模板里的剧情事实。
- `rewrite_goals` 要说明本条最值得提升的 2-4 个点。
- `skeleton_steps` 要写成“先……再……最后……”这种可执行骨架，并明确体现对高分模板的学习结果。
- `skeleton_steps` 应优先规划“表现层重写”，而不是原稿轻微修整。
- 可以在不改变剧情因果与结果的前提下，重组镜头层动作顺序、对白前后的包裹镜头、切镜位置与尾帧收束方式。
- `dialogue_freeze_notes` 要明确写出：对白内容冻结、说话人冻结、可围绕对白重组镜头。

输入 1：当前任务元信息
{{job_context_json}}

输入 2：选中的 base point 包（每条都带匹配到的 prompt templates）
{{selected_points_package_json}}

输入 3：本轮候选 prompt templates 池
{{selected_prompt_templates_json}}

输入 4：参考剧本节选（可为空）
{{source_script_text}}
