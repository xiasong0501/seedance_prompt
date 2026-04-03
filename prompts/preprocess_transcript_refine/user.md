请对下面这批 transcript segment 做校字 refine。

要求：
- 保持 segment_id 一一对应
- 只能修正 text
- 不允许改时间
- 优先参考 OCR
- 如果某条明显是字幕署名、翻译鸣谢、水印、平台口播、社群署名或非剧情语音，请把 text 直接输出为空字符串 `""`
- 如果 OCR 没帮助或你不确定，就保留原文

当前 segment：
{{segments_block}}

同时间窗 OCR：
{{ocr_block}}
