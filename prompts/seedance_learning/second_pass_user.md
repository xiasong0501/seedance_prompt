请基于下面这个 beat 的上下文和图像清单，先判断这段 beat 实际在拍什么，再修正每个 shot 的镜头语言。

要求：
- 图像按 `frame_manifest_json` 的顺序输入。
- `frame_role=beat_primary` 是这个 beat 的主证据，必须优先用它们判断真实时间推进、主体变化和动作连续性。
- `frame_role=shot_anchor` 只用于帮助你把观察结果对齐回本地 shot，不要让少量 anchor 覆盖整段 beat 的真实画面。
- 优先相信图像证据，其次参考对白、OCR 和当前初稿。
- 这次输出不只是修 shot，还要先把 beat 级理解说准：当前是谁在场、空间是什么、动作如何推进、对白大意落在哪一段。
- 如果上游 `source_story_beats` 和 `beat_primary` 画面冲突，必须以 `beat_primary` 为准，不要沿用错误剧情摘要。
- `transcript_text` 和 `dialogue_windows` 请优先当作这段 beat 的对白参考，尤其要对齐人物动作和说话时机，不要忽略它们。
- `episode_context_window` 只用于理解前后承接、人物关系和场景连续性；它是轻量参考，不能覆盖当前 beat 的图像事实。
- 如果当前初稿明显在重复、过泛或与图像不符，请直接修正。
- 如果某个 shot 对应的信息不足，可以保留简短保守描述，但不要把同一句话复读给多个 shot。
- 目标不是写摘要，而是为后续还原高细节 Seedance prompt 提供 shot card。
- 优先补足：镜头切入方式、主体站位、动作分段、受光与材质、背景连续性、声音层、切镜触发。
- 输出必须严格符合约定 JSON 结构。

## Beat Context

```json
{{beat_context_json}}
```

## Frame Manifest

```json
{{frame_manifest_json}}
```
