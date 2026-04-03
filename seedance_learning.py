from __future__ import annotations

import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from imageio_ffmpeg import get_ffmpeg_exe

from prompt_utils import load_prompt, render_prompt
from providers.base import (
    FrameReference,
    ProviderError,
    extract_json_from_text,
    file_to_data_url,
    load_json_file,
    save_json_file,
    save_text_file,
    utc_timestamp,
    validate_against_schema,
)
from providers.openai_adapter import OpenAIAdapter
from providers.qwen_adapter import QwenAdapter


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TAXONOMY_VERSION = "seedance-purpose-v1"
DEFAULT_MIN_BEAT_DURATION = 8.0
DEFAULT_TARGET_BEAT_DURATION = 12.0
DEFAULT_MAX_BEAT_DURATION = 15.0
DEFAULT_MAX_TEMPLATES_PER_PURPOSE = 8
DEFAULT_MAX_RULES_PER_PURPOSE = 8
DEFAULT_MAX_TEMPLATES_PER_SERIES = 40
DEFAULT_MIN_BEAT_FRAMES = 4
DEFAULT_MAX_BEAT_FRAMES = 8
DEFAULT_VISUAL_SECOND_PASS_MAX_IMAGES = 12
DEFAULT_VISUAL_SECOND_PASS_CONTEXT_FRAMES = 2
DEFAULT_VISUAL_SECOND_PASS_TIMEOUT_SECONDS = 180
DEFAULT_RESTORED_PROMPT_CHAR_LIMIT = 2200
DEFAULT_GENERALIZED_PROMPT_CHAR_LIMIT = 720
DEFAULT_MAX_KEY_PROMPT_SHOTS = 6
DEFAULT_DIALOGUE_WINDOW_GAP_SECONDS = 0.18
DEFAULT_DIALOGUE_WINDOW_MAX_MERGED_CHARS = 18
DEFAULT_DIALOGUE_WINDOW_RENDER_LIMIT = 8
GENERIC_SHOT_ROLE_LABELS = {
    "开场建立",
    "动作触发",
    "对白推进",
    "信息落点",
    "结果反应",
    "反应承接",
    "张力推进",
    "异动扩散",
    "尾帧收束",
    "中段推进",
    "单拍完成",
}
VISUAL_HINT_KEYWORDS = (
    "镜头",
    "画面",
    "视线",
    "目光",
    "眼神",
    "瞳孔",
    "侧脸",
    "面部",
    "脸",
    "嘴角",
    "手",
    "手指",
    "指节",
    "肩",
    "背",
    "腰",
    "脚",
    "步",
    "衣摆",
    "发梢",
    "身影",
    "剪影",
    "站位",
    "前景",
    "中景",
    "后景",
    "近景",
    "特写",
    "远景",
    "大全景",
    "轴线",
    "纵深",
    "高位",
    "低位",
    "中轴",
    "台阶",
    "门",
    "窗",
    "桌",
    "地面",
    "石面",
    "雨",
    "风",
    "烟",
    "尘",
    "火",
    "光",
    "阴影",
    "反光",
    "边缘光",
    "材质",
    "器物",
    "道具",
    "倒影",
    "叠化",
    "推近",
    "拉开",
    "切近",
    "跟拍",
    "摇",
    "抬手",
    "转头",
    "回头",
    "起身",
    "跪",
    "走入",
    "逼近",
    "后撤",
    "停住",
    "落下",
    "压住",
    "震",
    "闪",
    "滴",
    "血",
    "泪",
    "呼吸",
)
STRONG_VISUAL_HINT_KEYWORDS = (
    "镜头",
    "画面",
    "视线",
    "目光",
    "眼神",
    "侧脸",
    "面部",
    "手",
    "肩",
    "背",
    "脚",
    "身影",
    "站位",
    "前景",
    "中景",
    "后景",
    "近景",
    "特写",
    "远景",
    "大全景",
    "轴线",
    "纵深",
    "高位",
    "低位",
    "中轴",
    "台阶",
    "门",
    "窗",
    "桌",
    "地面",
    "石面",
    "倒影",
    "叠化",
    "推近",
    "拉开",
    "切近",
    "跟拍",
    "摇",
    "抬手",
    "转头",
    "回头",
    "起身",
    "跪",
    "走入",
    "逼近",
    "后撤",
    "停住",
    "落下",
    "压住",
    "显露",
    "出现",
    "呼吸",
)
DIALOGUE_HEAVY_HINT_KEYWORDS = (
    "我",
    "你",
    "他",
    "她",
    "它",
    "我们",
    "你们",
    "他们",
    "怎么",
    "为什么",
    "是不是",
    "不是",
    "就是",
    "之前",
    "一直",
    "直到",
    "对啊",
    "名字",
    "回来",
    "听说",
    "知道",
    "觉得",
    "带我",
    "过着",
    "日子",
)
GENERIC_TRANSITION_HINTS = {
    "对白句尾或停顿触发切镜",
    "对手或听者反应带出下一镜",
    "异常信号扩散到下一镜",
    "尾帧把结果反应或下一拍触发物留下",
    "物件或字面锚点出现后切回人物反应",
}

PROVIDER_API_ENV_MAP = {
    "openai": "OPENAI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
}

SECOND_PASS_VISUAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "beat_title": {"type": "string"},
        "beat_summary": {"type": "string"},
        "scene_state": {"type": "string"},
        "subject_identity": {"type": "string"},
        "dialogue_summary": {"type": "string"},
        "timeflow_summary": {"type": "string"},
        "beat_observation": {"type": "string"},
        "primary_purpose_observed": {"type": "string"},
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "shot_id": {"type": "string"},
                    "story_function": {"type": "string"},
                    "visual_focus": {"type": "string"},
                    "camera_language": {"type": "string"},
                    "camera_entry": {"type": "string"},
                    "subject_blocking": {"type": "string"},
                    "action_timeline": {"type": "string"},
                    "lighting_and_texture": {"type": "string"},
                    "background_continuity": {"type": "string"},
                    "dialogue_timing": {"type": "string"},
                    "sound_bed": {"type": "string"},
                    "art_direction_hint": {"type": "string"},
                    "transition_trigger": {"type": "string"},
                },
                "required": [
                    "shot_id",
                    "story_function",
                    "visual_focus",
                    "camera_language",
                    "transition_trigger",
                ],
            },
        },
    },
    "required": ["shots"],
}

PURPOSE_ORDER = [
    "爱情",
    "思念",
    "痛苦",
    "告别",
    "守护",
    "羞辱",
    "反击",
    "报仇",
    "对峙",
    "揭示",
    "权力",
    "规则",
    "觉醒",
    "特效",
    "群像",
    "危险",
    "牺牲",
    "尾钩",
]

PURPOSE_KEYWORDS: dict[str, Sequence[str]] = {
    "爱情": ("爱", "心动", "喜欢", "暧昧", "拥抱", "亲吻", "情愫", "凝视", "靠近", "命定", "相认"),
    "思念": ("想念", "思念", "回忆", "旧梦", "故人", "念起", "回望", "追忆", "挂念", "相思"),
    "痛苦": ("痛", "疼", "哭", "崩溃", "绝望", "濒死", "受伤", "咳血", "窒息", "压迫", "狼狈"),
    "告别": ("告别", "再见", "离开", "诀别", "送别", "离场", "转身", "分别", "别过", "永别"),
    "守护": ("守护", "护", "挡", "救", "保护", "替", "拦", "护主", "不退", "挡住", "救下"),
    "羞辱": ("羞辱", "辱", "贱", "踩", "狗", "不配", "污名", "逼问", "处刑", "公开", "看不起"),
    "反击": ("反击", "回击", "打脸", "反咬", "顶嘴", "回怼", "翻盘", "认领", "冷笑", "质问", "为何怕"),
    "报仇": ("报仇", "复仇", "仇", "讨债", "血债", "清算", "垫背", "灭门", "杀你", "偿命"),
    "对峙": ("对峙", "相持", "对骂", "对立", "剑拔弩张", "互看", "逼近", "停住", "压场"),
    "揭示": ("揭示", "真相", "原来", "发现", "认知", "显影", "揭秘", "曝光", "说破", "身份", "物证"),
    "权力": ("家主", "王", "帝", "宗门", "权威", "命令", "下令", "裁判", "高位", "宣判", "资格"),
    "规则": ("规则", "资格", "代价", "结果", "只认", "不准", "必须", "入宗门", "武比", "法则"),
    "觉醒": ("觉醒", "心跳", "发光", "灵力", "苏醒", "共鸣", "符文", "异动", "涌动", "感应"),
    "特效": ("雷", "光", "爆", "炸", "法阵", "异变", "特效", "镜门", "冲击", "符文", "电光", "火"),
    "群像": ("众人", "三家", "三义子", "群臣", "部队", "围观", "队伍", "全场", "多人", "子弟"),
    "危险": ("危险", "刀", "剑", "杀", "追杀", "威胁", "压住", "逼近", "死令", "悬刀", "处死"),
    "牺牲": ("牺牲", "献祭", "替死", "垫背", "燃命", "赴死", "不亏", "愿意死", "代价换命"),
    "尾钩": ("卡点", "尾钩", "黑屏", "未落", "将要", "下一秒", "即将", "异变将至", "心跳先炸"),
}

PURPOSE_PROFILES: dict[str, dict[str, Any]] = {
    "爱情": {
        "description": "用 8-15 秒建立情感吸引、克制靠近或命定张力。",
        "required_slots": ["【情感主体】", "【被凝视者】", "【关系触发物】"],
        "camera_rules": ["先给视线与距离，再给身体细节，不要一上来就把情绪说满。"],
        "beat_rules": ["常用‘靠近前停顿 -> 目光确认 -> 微动作回应 -> 尾帧留念’四拍结构。"],
        "action_rules": ["动作要轻而具体，优先手指、呼吸、肩颈、眼神而不是大幅肢体动作。"],
        "dialogue_rules": ["对白宜短句，重点放在停顿、吞咽、视线回避和压低声线。"],
        "continuity_rules": ["相邻分镜必须保持两人间距、朝向和谁先看谁的关系。"],
        "negative_patterns": ["避免把爱情 beat 写成纯表白或纯文学抒情，缺少可拍动作。"],
        "when_to_use": "适合命定吸引、克制靠近、未说破的情感确认。",
        "template_opening": "前段先用【情感主体】和【被凝视者】之间的距离与视线差建立情绪，",
        "template_middle": "中段把重点放在一个能被镜头看见的微动作或关系触发物上，",
        "template_tail": "尾段停在目光未收、手位未落或一句未说满的话上，把余韵交给下一拍。",
    },
    "思念": {
        "description": "用 8-15 秒表达想念、追忆或对逝去关系的回看。",
        "required_slots": ["【思念主体】", "【记忆触发物】", "【回望方向】"],
        "camera_rules": ["先给现实中的触发物，再切记忆或情绪回流，不要直接铺满回忆。"],
        "beat_rules": ["常用‘现实静止 -> 触发物进入 -> 情绪回流 -> 回到现实’结构。"],
        "action_rules": ["思念不能只靠眼泪，要给停手、回头、握物、呼吸停顿这些动作桥。"],
        "dialogue_rules": ["对白能省则省，优先让记忆靠物件、方向和表情成立。"],
        "continuity_rules": ["现实与记忆切换必须有同一锚点承接，如光、手势、器物或风声。"],
        "negative_patterns": ["避免大段回忆说明，导致当前时空失去连续性。"],
        "when_to_use": "适合怀念故人、回望旧恩或借记忆强化情绪支付。",
        "template_opening": "前段先在现实里落下【记忆触发物】或【回望方向】，",
        "template_middle": "中段让情绪从手势、呼吸或短闪回里自然回流，",
        "template_tail": "尾段必须回到现实，把思念压成一个仍会延续的动作或目光。",
    },
    "痛苦": {
        "description": "用 8-15 秒承接伤痛、绝望、濒死或心理崩塌。",
        "required_slots": ["【受苦主体】", "【疼痛来源】", "【身体反应】"],
        "camera_rules": ["先拍受力结果，再拍呼吸、手位和眼神变化，避免只拍大表情。"],
        "beat_rules": ["常用‘受力 -> 停顿 -> 身体反馈 -> 情绪认领’四层推进。"],
        "action_rules": ["痛苦要拆成起势、到位、余波，呼吸和重心变化必须可见。"],
        "dialogue_rules": ["痛苦段对白要克制，宁可用喘息、吞咽、停顿代替长台词。"],
        "continuity_rules": ["伤态和体力状态要跨条连续，不能前一条濒死后一条瞬间站稳。"],
        "negative_patterns": ["避免只拍结果性痛苦，没有动作过程和身体反馈。"],
        "when_to_use": "适合受虐、濒死、崩塌、心碎或代价显形的段落。",
        "template_opening": "前段先把【受苦主体】当前的伤态或压迫结果立住，",
        "template_middle": "中段重点给呼吸、肩颈、手位和重心的细碎变化，",
        "template_tail": "尾段把痛苦压成一个仍未结束的身体状态或求生动作。",
    },
    "告别": {
        "description": "用 8-15 秒完成离场、诀别、关系切断或有余温的分离。",
        "required_slots": ["【离场主体】", "【被留下者】", "【离场方向】"],
        "camera_rules": ["优先用空间方向、转身和回头组织离场，而不是只靠说再见。"],
        "beat_rules": ["常用‘停住 -> 看一眼 -> 转身离开 -> 留下空位’结构。"],
        "action_rules": ["转身、松手、退后、迈步都要给完整过程，不能一句带过。"],
        "dialogue_rules": ["告别台词要短，尾字后必须留空让离场动作完成。"],
        "continuity_rules": ["离场后必须明确谁还留在场上、空间空位在哪里。"],
        "negative_patterns": ["避免告别段只有台词，没有空间关系和离场方向。"],
        "when_to_use": "适合诀别、送别、关系切断、角色转身离场。",
        "template_opening": "前段先稳住【离场主体】和【被留下者】的相对站位，",
        "template_middle": "中段把话和动作压在同一条线上，让转身、松手或离场成立，",
        "template_tail": "尾段留出空位或背影，把告别后的余波交给下一条。",
    },
    "守护": {
        "description": "用 8-15 秒表现护主、挡刀、替身承压或弱者硬顶。",
        "required_slots": ["【守护者】", "【被保护者】", "【威胁源】"],
        "camera_rules": ["先给冲入或挡位，再确认前后站位，最后落到不退的身体状态。"],
        "beat_rules": ["常用‘威胁落下 -> 守护者冲入 -> 站稳/抱住/挡住 -> 尾帧不退’结构。"],
        "action_rules": ["守护段必须写清路径和受力，不能只写结果性挡住。"],
        "dialogue_rules": ["对白宜短且咬字重，常用誓言句或喝止句，但不能挤掉动作。"],
        "continuity_rules": ["守护者插入空间后，要重新确认三者前后位和安全区。"],
        "negative_patterns": ["避免守护者凭空入画或挡位没有清楚路径。"],
        "when_to_use": "适合冲台护主、挡刀、挡咒、替他受压等强关系段落。",
        "template_opening": "前段先让【威胁源】落下，再用【守护者】的入画路径和冲刺动作完成打断，",
        "template_middle": "中段把挡位、受力和不退的身体状态写足，",
        "template_tail": "尾段停在守护姿态仍成立、危机尚未完全解除的瞬间。",
    },
    "羞辱": {
        "description": "用 8-15 秒把公开贬低、踩踏、污名化或上下位羞辱拍清楚。",
        "required_slots": ["【施压者】", "【受辱者】", "【公开空间/旁观层】"],
        "camera_rules": ["先拍高低位和压迫关系，再给羞辱台词或动作，不要先说后摆位。"],
        "beat_rules": ["常用‘压迫姿态 -> 羞辱语言/动作 -> 受辱反应 -> 尾帧加压’结构。"],
        "action_rules": ["羞辱动作要写清接触点、压实过程和受辱者的身体反馈。"],
        "dialogue_rules": ["羞辱台词短、狠、带身份差，常配一个停顿让反应落地。"],
        "continuity_rules": ["要持续确认施压者在上位、受辱者在下位，旁观层是否在看。"],
        "negative_patterns": ["避免只剩恶毒台词，没有空间压迫和身体受力。"],
        "when_to_use": "适合公开羞辱、踩头逼问、身份贬低、当众处刑。",
        "template_opening": "前段先用【施压者】对【受辱者】的高低位和接触点建立羞辱秩序，",
        "template_middle": "中段再用最值钱的台词或动作把羞辱说满，",
        "template_tail": "尾段停在加压未完、反应未散或下一重羞辱将落的状态上。",
    },
    "反击": {
        "description": "用 8-15 秒完成主角回击、冷反问、认领优势或打脸启动。",
        "required_slots": ["【反击主体】", "【被反击者】", "【反击触发物】"],
        "camera_rules": ["先拍压住情绪的冷反应，再给出手或说话，不要一上来就爆。"],
        "beat_rules": ["常用‘承压 -> 冷停顿 -> 反问/出手 -> 对手失衡’结构。"],
        "action_rules": ["反击前最好有一个静止或视线锁定，出手后必须拍到对手反应。"],
        "dialogue_rules": ["反击台词要短句、硬句、少解释，尾字后给半拍看对手。"],
        "continuity_rules": ["要保留前序压迫痕迹，反击才有支付感。"],
        "negative_patterns": ["避免反击一上来就完全胜利，缺少压抑后的释放。"],
        "when_to_use": "适合女主冷反击、公开打脸、话语权回收。",
        "template_opening": "前段先保留上一拍留下的压迫痕迹，让【反击主体】冷住一拍，",
        "template_middle": "中段用一个反问、认领动作或短促出手完成夺权，",
        "template_tail": "尾段把结果落到【被反击者】的失衡、失语或表情裂开上。",
    },
    "报仇": {
        "description": "用 8-15 秒把仇恨兑现、清算启动或血债认领拍出重压。",
        "required_slots": ["【复仇主体】", "【清算对象】", "【仇恨证据/旧账】"],
        "camera_rules": ["先立旧伤或旧账，再落清算动作，不要让报仇像普通争吵。"],
        "beat_rules": ["常用‘旧账点题 -> 复仇主体认领 -> 清算动作/命令 -> 后果悬停’结构。"],
        "action_rules": ["报仇段动作要稳狠准，最好带一个回收旧账的物件或视线。"],
        "dialogue_rules": ["台词要带旧账和代价，不要只写情绪化怒骂。"],
        "continuity_rules": ["前后条要持续确认报仇对象、旧账证据和结果落点。"],
        "negative_patterns": ["避免报仇段只剩口号，没有旧账证据和结果回收。"],
        "when_to_use": "适合复仇启动、公开清算、旧债认领、垫背威胁。",
        "template_opening": "前段先把【仇恨证据/旧账】钉在画面里，",
        "template_middle": "中段由【复仇主体】认领这笔账并把清算动作或命令落下，",
        "template_tail": "尾段停在后果将至而未完全兑现的瞬间，给下一条继续清算。",
    },
    "对峙": {
        "description": "用 8-15 秒把双方对峙、卡位、对骂或停手瞬间拍得紧。",
        "required_slots": ["【阵营A】", "【阵营B】", "【空间轴线】"],
        "camera_rules": ["先立轴线和站位，再拍目光与手位，避免镜头漂移。"],
        "beat_rules": ["常用‘阵营落位 -> 视线/手位绷紧 -> 短句碰撞 -> 尾帧悬停’结构。"],
        "action_rules": ["对峙不等于不动，要写重心前压、手位变化、眼神锁定和停住。"],
        "dialogue_rules": ["对峙台词一来一回即可，重点是停顿和反应。"],
        "continuity_rules": ["空间轴线、屏幕方向和高低位必须持续稳定。"],
        "negative_patterns": ["避免对峙段只剩对白，没有手位、站位和镜头锚点。"],
        "when_to_use": "适合剑拔弩张、悬刀未落、两方卡位、临界停手。",
        "template_opening": "前段先把【阵营A】和【阵营B】在【空间轴线】上的位置立住，",
        "template_middle": "中段用手位、视线和一句短碰撞把张力抬起来，",
        "template_tail": "尾段停在动作未落、视线未收或下一拍将爆的临界点。",
    },
    "揭示": {
        "description": "用 8-15 秒完成真相揭开、身份显影或认知翻转。",
        "required_slots": ["【揭示主体】", "【被揭示对象/真相】", "【证据物/证据动作】"],
        "camera_rules": ["先给证据物或异常点，再拍人物反应和认知变化。"],
        "beat_rules": ["常用‘证据出现 -> 认知停顿 -> 真相落地 -> 反应扩散’结构。"],
        "action_rules": ["揭示段的关键不是说出来，而是证据被看见、被认出、被命名。"],
        "dialogue_rules": ["揭示台词要准，常用一句点破，后面让反应镜头接。"],
        "continuity_rules": ["证据物的朝向、位置和谁先看到要连续清楚。"],
        "negative_patterns": ["避免只有解释，没有证据物和反应镜头。"],
        "when_to_use": "适合身份曝光、真相揭晓、物证显影、认知逆转。",
        "template_opening": "前段先把【证据物/证据动作】送入画面，",
        "template_middle": "中段给【揭示主体】或关键人物一个认知停顿，再用一句话点破，",
        "template_tail": "尾段把结果落到群体或对手反应上，让真相继续发酵。",
    },
    "权力": {
        "description": "用 8-15 秒建立高位者压场、命令生效或秩序被改写。",
        "required_slots": ["【高位者】", "【低位群体】", "【权力触发动作】"],
        "camera_rules": ["先拍高位位置和视线高度，再拍命令，不要把权力拍成平视对白。"],
        "beat_rules": ["常用‘高位立住 -> 命令落下 -> 群体反应 -> 尾帧秩序成形’结构。"],
        "action_rules": ["权力段动作可以少，但抬手、点地、俯视、停顿必须准。"],
        "dialogue_rules": ["命令台词应短、稳、不可被背景噪音吞掉。"],
        "continuity_rules": ["高位者位置、群体朝向和谁被压住要跨条保持清楚。"],
        "negative_patterns": ["避免高位者动作过多、情绪过满，削弱威压。"],
        "when_to_use": "适合裁判压场、天帝宣判、家主拍板、秩序重写。",
        "template_opening": "前段先把【高位者】在空间中的高差和主光位置立住，",
        "template_middle": "中段用一个【权力触发动作】和一句命令改写场上秩序，",
        "template_tail": "尾段停在低位群体被迫接受新秩序的状态上。",
    },
    "规则": {
        "description": "用 8-15 秒把世界规则、资格门槛或残酷法则说死。",
        "required_slots": ["【规则宣告者】", "【规则内容】", "【规则受体】"],
        "camera_rules": ["规则段先摆位置关系，再用清晰稳定的镜头容纳整句信息。"],
        "beat_rules": ["常用‘规则宣告者入场/开口 -> 规则内容落地 -> 受体反应 -> 静默消化’结构。"],
        "action_rules": ["规则段动作不求多，但要有稳固的站位和明确的视线压迫。"],
        "dialogue_rules": ["规则台词要完整、短句化、层级清楚，避免长篇解释。"],
        "continuity_rules": ["规则一旦说出，后续分镜必须尊重其结果，不可立刻打破。"],
        "negative_patterns": ["避免规则段堆术语，导致镜头价值被台词完全吞掉。"],
        "when_to_use": "适合宗门规则、武比规则、代价规则、资格切断。",
        "template_opening": "前段先把【规则宣告者】和【规则受体】放入同一空间逻辑里，",
        "template_middle": "中段用 1-2 句把【规则内容】说死，",
        "template_tail": "尾段留一拍静默让规则真正压到场上。",
    },
    "觉醒": {
        "description": "用 8-15 秒把能力苏醒、异能共鸣或身份觉醒拍成可执行的镜头链。",
        "required_slots": ["【觉醒主体】", "【觉醒信号】", "【环境反馈】"],
        "camera_rules": ["先拍异常信号，再确认主体，再扩到环境反馈，避免上来就大特效。"],
        "beat_rules": ["常用‘异常细节 -> 主体反应 -> 共鸣扩散 -> 尾帧锁定源头’结构。"],
        "action_rules": ["觉醒段重点写瞳孔、心跳、手位、皮肤受光、器物共振。"],
        "dialogue_rules": ["对白可有可无，优先让声音和反应承担信息。"],
        "continuity_rules": ["异常信号必须有清楚源头，后续结果要沿同一方向扩散。"],
        "negative_patterns": ["避免一上来就大范围异能爆发，没有前置异动和源头确认。"],
        "when_to_use": "适合心跳异变、灵力苏醒、器物共鸣、身份觉醒起势。",
        "template_opening": "前段先把【觉醒信号】压在小范围细节上，",
        "template_middle": "中段再让【觉醒主体】和【环境反馈】一起被镜头确认，",
        "template_tail": "尾段停在异变尚未完全爆发、但源头已经被锁定的瞬间。",
    },
    "特效": {
        "description": "用 8-15 秒完成法术、雷击、爆闪、机关或奇观兑现。",
        "required_slots": ["【特效源头】", "【传播路径】", "【环境/人物反馈】"],
        "camera_rules": ["奇观必须拍成‘源头 -> 传播 -> 结果’，不能只给抽象特效词。"],
        "beat_rules": ["常用‘源头点亮 -> 路径扩散 -> 环境反馈 -> 结果悬停’结构。"],
        "action_rules": ["特效要带体积感、受光变化、碎屑/烟尘/风压等反馈。"],
        "dialogue_rules": ["特效段对白要少，让声音设计和反应镜头承担信息。"],
        "continuity_rules": ["特效传播方向和受击范围必须跨拍一致。"],
        "negative_patterns": ["避免只写抽象法术名，没有源头、路径和反馈。"],
        "when_to_use": "适合法术释放、雷击压场、爆闪、阵法、机关齐鸣。",
        "template_opening": "前段先在小范围立住【特效源头】，",
        "template_middle": "中段明确写出【传播路径】和对空间的改写，",
        "template_tail": "尾段一定要把【环境/人物反馈】落到可继续接拍的状态上。",
    },
    "群像": {
        "description": "用 8-15 秒组织多人站位、群体反应或公开场合秩序。",
        "required_slots": ["【核心人物】", "【群体层】", "【纵深/高差结构】"],
        "camera_rules": ["群像段优先前中后景、高低位和纵深，不靠横向排排站。"],
        "beat_rules": ["常用‘规模镜头 -> 核心人物落点 -> 群体反应 -> 尾帧秩序’结构。"],
        "action_rules": ["群像不是人人都动，最多 1-4 个核心主体，其余做秩序和反应层。"],
        "dialogue_rules": ["群像对白宜少，重点保留最能定性关系的短句。"],
        "continuity_rules": ["队列方向、主位位置和谁在看谁必须稳定。"],
        "negative_patterns": ["避免群像段写成所有人都在同时做事，导致主次不清。"],
        "when_to_use": "适合入场、围观、群体震惊、队列压场、公开场合。",
        "template_opening": "前段先用【纵深/高差结构】把【群体层】和【核心人物】的关系立住，",
        "template_middle": "中段再把视线或动作收回到真正承担事件推进的 1-4 个主体，",
        "template_tail": "尾段停在群体秩序已经成形、下一拍将被打破的时刻。",
    },
    "危险": {
        "description": "用 8-15 秒组织悬刀、追杀、死令、迫近或临界危险。",
        "required_slots": ["【危险源】", "【受威胁者】", "【危险触发动作】"],
        "camera_rules": ["先确认危险源和受威胁者的距离，再让危险逼近。"],
        "beat_rules": ["常用‘危险显形 -> 逼近/悬停 -> 受威胁者反应 -> 尾帧未落’结构。"],
        "action_rules": ["危险段的关键是未完成动作，如悬刀、抬脚、瞄准、逼近。"],
        "dialogue_rules": ["台词只保留最值钱的威胁句或喝止句。"],
        "continuity_rules": ["危险方向和安全区必须明确，不能每拍都换边。"],
        "negative_patterns": ["避免危险段一上来就出结果，失去临界紧张感。"],
        "when_to_use": "适合悬刀、逼问、追杀、死令、开场高危处境。",
        "template_opening": "前段先让【危险源】和【受威胁者】同时进入画面逻辑，",
        "template_middle": "中段把【危险触发动作】推进到临界但不立刻落下，",
        "template_tail": "尾段停在一拍未落的危险上，逼下一条接手。",
    },
    "牺牲": {
        "description": "用 8-15 秒表现替人受压、赴死、献祭或代价认领。",
        "required_slots": ["【牺牲者】", "【受益者/守护对象】", "【代价形态】"],
        "camera_rules": ["先立牺牲者为何顶上去，再拍代价落到身体或命运上。"],
        "beat_rules": ["常用‘主动顶上 -> 代价落身 -> 受益者反应 -> 尾帧残留’结构。"],
        "action_rules": ["牺牲要有主动性，至少要拍到迈步、伸手、挡住、硬顶这些动作。"],
        "dialogue_rules": ["牺牲台词宜短，常用誓言句、认命句或代价句。"],
        "continuity_rules": ["牺牲后的伤态和代价必须后续延续，不能下一条清零。"],
        "negative_patterns": ["避免牺牲段只拍受伤，不拍主动选择。"],
        "when_to_use": "适合替人挡刀、燃命护人、垫背、赴死认领。",
        "template_opening": "前段先让【牺牲者】主动把自己送进代价位置，",
        "template_middle": "中段把【代价形态】拍到身体和空间上，",
        "template_tail": "尾段保留一个未完的守护姿态或濒死动作，让牺牲延续到下一拍。",
    },
    "尾钩": {
        "description": "用 8-15 秒把一集或一段的尾帧卡点停在最想点下一条的位置。",
        "required_slots": ["【尾帧主体】", "【异常信号/未落动作】", "【下一条触发物】"],
        "camera_rules": ["先让画面压静，再把真正的钩子丢进去，不要尾段还在解释。"],
        "beat_rules": ["常用‘静止收束 -> 异常信号出现 -> 反应锁定 -> 黑屏前悬停’结构。"],
        "action_rules": ["尾钩段最值钱的是未完成动作、异常声响、微光、心跳、视线锁定。"],
        "dialogue_rules": ["台词可有可无，若有必须极短，且说完后留空。"],
        "continuity_rules": ["尾钩一定要能直接成为下一条首帧的触发物。"],
        "negative_patterns": ["避免尾段继续解释剧情，没有真正的未完成状态。"],
        "when_to_use": "适合集尾按钮、未落一脚、异常心跳、道具发光、下一秒将爆。",
        "template_opening": "前段先把场面压到几乎静止，",
        "template_middle": "中段只送入一个真正值钱的【异常信号/未落动作】，",
        "template_tail": "尾段必须把【下一条触发物】锁定在画面里，让观众自然想点下一条。",
    },
}

FALLBACK_PURPOSE = "对峙"
TIMEPOINT_PATTERN = re.compile(r"(\d+):(\d+):(\d+(?:\.\d+)?)")
MINSEC_TIMEPOINT_PATTERN = re.compile(r"(?<!\d)(\d+):(\d+(?:\.\d+)?)(?!\d)")
RANGE_SEPARATORS = ("—", "–", "-", "~", "至")


def seedance_learning_settings(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = dict(config or {})
    raw = dict(root.get("seedance_learning") or {})
    run_config = dict(root.get("run") or {})
    providers = dict(root.get("providers") or {})
    requested_provider = str(
        raw.get("visual_second_pass_provider")
        or run_config.get("selected_provider")
        or ""
    ).strip().lower()

    def _provider_has_api_key(provider_name: str) -> bool:
        provider_name = str(provider_name or "").strip().lower()
        if not provider_name:
            return False
        env_name = PROVIDER_API_ENV_MAP.get(provider_name, "")
        provider_config = dict(providers.get(provider_name) or {})
        configured_key = str(provider_config.get("api_key") or "").strip()
        env_key = str(os.getenv(env_name, "")).strip() if env_name else ""
        return bool(configured_key or env_key)

    provider_candidates = [
        requested_provider,
        "openai",
        "qwen",
    ]
    second_pass_provider = ""
    for candidate in provider_candidates:
        candidate = str(candidate or "").strip().lower()
        if not candidate:
            continue
        if candidate not in {"openai", "qwen"}:
            continue
        if candidate != requested_provider and not _provider_has_api_key(candidate):
            continue
        if candidate == requested_provider and not _provider_has_api_key(candidate):
            continue
        second_pass_provider = candidate
        break
    provider_config = dict(providers.get(second_pass_provider) or {})
    return {
        "enabled": bool(raw.get("enabled", True)),
        "taxonomy_version": str(raw.get("taxonomy_version") or DEFAULT_TAXONOMY_VERSION).strip()
        or DEFAULT_TAXONOMY_VERSION,
        "min_beat_duration": float(raw.get("min_beat_duration", DEFAULT_MIN_BEAT_DURATION)),
        "target_beat_duration": float(raw.get("target_beat_duration", DEFAULT_TARGET_BEAT_DURATION)),
        "max_beat_duration": float(raw.get("max_beat_duration", DEFAULT_MAX_BEAT_DURATION)),
        "max_templates_per_purpose": int(raw.get("max_templates_per_purpose", DEFAULT_MAX_TEMPLATES_PER_PURPOSE)),
        "max_rules_per_purpose": int(raw.get("max_rules_per_purpose", DEFAULT_MAX_RULES_PER_PURPOSE)),
        "max_templates_per_series": int(raw.get("max_templates_per_series", DEFAULT_MAX_TEMPLATES_PER_SERIES)),
        "min_beat_frames": int(raw.get("min_beat_frames", DEFAULT_MIN_BEAT_FRAMES)),
        "max_beat_frames": int(raw.get("max_beat_frames", DEFAULT_MAX_BEAT_FRAMES)),
        "visual_second_pass_enabled": bool(raw.get("visual_second_pass_enabled", True)),
        "visual_second_pass_provider": second_pass_provider,
        "visual_second_pass_model": str(
            raw.get("visual_second_pass_model") or provider_config.get("model") or ""
        ).strip(),
        "visual_second_pass_endpoint": str(
            raw.get("visual_second_pass_endpoint") or provider_config.get("endpoint") or ""
        ).strip(),
        "visual_second_pass_timeout_seconds": int(
            raw.get(
                "visual_second_pass_timeout_seconds",
                run_config.get("timeout_seconds", DEFAULT_VISUAL_SECOND_PASS_TIMEOUT_SECONDS),
            )
        ),
        "visual_second_pass_max_images": int(
            raw.get("visual_second_pass_max_images", DEFAULT_VISUAL_SECOND_PASS_MAX_IMAGES)
        ),
        "visual_second_pass_context_frames": int(
            raw.get("visual_second_pass_context_frames", DEFAULT_VISUAL_SECOND_PASS_CONTEXT_FRAMES)
        ),
        "visual_second_pass_api_key": str(
            raw.get("visual_second_pass_api_key") or provider_config.get("api_key") or ""
        ).strip(),
        "openai_image_detail": str(
            provider_config.get("image_detail") or run_config.get("openai_image_detail", "auto")
        ).strip()
        or "auto",
        "qwen_video_fps": float(
            provider_config.get("video_fps") or run_config.get("qwen_video_fps", 1.0) or 1.0
        ),
        "qwen_structured_output_mode": str(
            provider_config.get("structured_output_mode")
            or run_config.get("qwen_structured_output_mode", "json_object")
        ).strip()
        or "json_object",
    }


def is_seedance_learning_enabled(config: Mapping[str, Any] | None = None) -> bool:
    return bool(seedance_learning_settings(config).get("enabled", True))


def _emit_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(message)
    except Exception:
        return


def purpose_score_breakdown(text: str, *, is_last: bool = False) -> list[dict[str, Any]]:
    normalized = _normalize_spaces(text)
    scores = _score_purposes(normalized, is_last=is_last)
    if not scores:
        scores[FALLBACK_PURPOSE] = 1.0
    return sorted(
        [{"purpose": purpose, "score": round(score, 2)} for purpose, score in scores.items()],
        key=lambda item: (
            -float(item["score"]),
            PURPOSE_ORDER.index(item["purpose"]) if item["purpose"] in PURPOSE_ORDER else 999,
        ),
    )


def infer_primary_purpose(
    text: str,
    *,
    is_last: bool = False,
    fallback: str = FALLBACK_PURPOSE,
) -> str:
    ranked = purpose_score_breakdown(text, is_last=is_last)
    if not ranked:
        return fallback
    return str(ranked[0].get("purpose") or fallback)


def infer_primary_purpose_from_parts(
    parts: Sequence[Any],
    *,
    is_last: bool = False,
    fallback: str = FALLBACK_PURPOSE,
) -> str:
    text = _normalize_spaces(" ".join(_normalize_spaces(str(part or "")) for part in parts if _normalize_spaces(str(part or ""))))
    if not text:
        return fallback
    return infer_primary_purpose(text, is_last=is_last, fallback=fallback)


def generate_episode_beat_catalog(
    *,
    project_root: Path,
    series_name: str,
    episode_id: str,
    analysis_path: str | Path,
    config: Mapping[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    settings = seedance_learning_settings(config)
    analysis_json_path = Path(analysis_path).expanduser().resolve()
    analysis = load_json_file(analysis_json_path)
    episode_dir = analysis_json_path.parent
    preprocess_dir = episode_dir / "preprocess"
    shot_frame_dir = episode_dir / "seedance_learning" / "shot_frames"
    beat_frame_dir = episode_dir / "seedance_learning" / "beat_frames"
    visual_second_pass_dir = episode_dir / "seedance_learning" / "visual_second_pass"
    source_video_raw = str(dict(analysis.get("episode") or {}).get("source_video") or "").strip()
    source_video_path = Path(source_video_raw).expanduser() if source_video_raw else None
    scene_list = _load_optional_json(preprocess_dir / "scene_list.json")
    transcript_segments = _load_optional_json(preprocess_dir / "corrected_transcript_segments.json")
    if not list(transcript_segments.get("segments") or []):
        transcript_segments = _load_optional_json(preprocess_dir / "transcript_segments.json")
    ocr_segments = _load_optional_json(preprocess_dir / "ocr_segments.json")
    strength_playbook = _load_optional_json((project_root / "analysis" / series_name / "series_strength_playbook_draft.json"))
    _emit_progress(
        progress_callback,
        f"Seedance 学习：开始处理 {series_name} {episode_id}｜分析文件={analysis_json_path}",
    )

    normalized_story_beats = _normalize_story_beats(analysis)
    beat_segmentation = _build_episode_beat_segmentation(
        series_name=series_name,
        episode_id=episode_id,
        analysis_path=analysis_json_path,
        normalized_story_beats=normalized_story_beats,
        scene_list=scene_list,
        transcript_segments=transcript_segments,
        ocr_segments=ocr_segments,
        episode_dir=episode_dir,
        source_video_path=source_video_path,
        settings=settings,
        progress_callback=progress_callback,
    )
    episode_catalog = _build_episode_catalog(
        series_name=series_name,
        episode_id=episode_id,
        analysis=analysis,
        analysis_path=analysis_json_path,
        scene_list=scene_list,
        transcript_segments=transcript_segments,
        ocr_segments=ocr_segments,
        normalized_story_beats=normalized_story_beats,
        beat_segmentation=beat_segmentation,
        strength_playbook=strength_playbook,
        settings=settings,
        source_video_path=source_video_path,
        beat_frame_root=beat_frame_dir,
        episode_dir=episode_dir,
        progress_callback=progress_callback,
    )
    _emit_progress(progress_callback, "Seedance 学习：所有分镜处理完成，开始写入分段和学习产物。")

    beat_segmentation_json_path = episode_dir / "beat_segmentation.json"
    beat_segmentation_md_path = episode_dir / "beat_segmentation.md"
    json_path = episode_dir / "seedance_beat_catalog.json"
    md_path = episode_dir / "seedance_beat_catalog.md"
    save_json_file(beat_segmentation_json_path, beat_segmentation)
    save_text_file(beat_segmentation_md_path, render_beat_segmentation_markdown(beat_segmentation))
    save_json_file(json_path, episode_catalog)
    save_text_file(md_path, render_episode_catalog_markdown(episode_catalog))
    _emit_progress(
        progress_callback,
        f"Seedance 学习：beat 分段已写入 {beat_segmentation_json_path}｜目录={beat_frame_dir.resolve()}",
    )
    _emit_progress(
        progress_callback,
        f"Seedance 学习：学习目录已写入 {json_path}｜shot 帧目录={shot_frame_dir.resolve()}",
    )
    return {
        "beat_segmentation_json_path": str(beat_segmentation_json_path),
        "beat_segmentation_markdown_path": str(beat_segmentation_md_path),
        "catalog_json_path": str(json_path),
        "catalog_markdown_path": str(md_path),
        "beat_count": int(episode_catalog.get("beat_count", 0)),
        "shot_count": int(beat_segmentation.get("shot_count", 0)),
        "shot_frame_dir": str(shot_frame_dir.resolve()),
        "beat_frame_dir": str(beat_frame_dir.resolve()),
        "visual_second_pass_dir": str(visual_second_pass_dir.resolve()),
        "episode_id": episode_id,
        "series_name": series_name,
    }


def build_series_purpose_libraries(
    *,
    project_root: Path,
    series_name: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = seedance_learning_settings(config)
    series_dir = project_root / "analysis" / series_name
    episode_catalog_paths = sorted(series_dir.glob("ep*/seedance_beat_catalog.json"))
    catalogs = [load_json_file(path) for path in episode_catalog_paths if path.exists()]
    if not catalogs:
        return {}

    strength_playbook = _load_optional_json(series_dir / "series_strength_playbook_draft.json")
    skill_library = _build_skill_library(
        series_name=series_name,
        catalogs=catalogs,
        strength_playbook=strength_playbook,
        settings=settings,
    )
    template_library = _build_template_library(
        series_name=series_name,
        catalogs=catalogs,
        settings=settings,
    )

    prompt_library_artifacts = _export_prompt_library(
        project_root=project_root,
        series_name=series_name,
        template_library=template_library,
    )
    search_index_artifacts = _export_prompt_library_search_index(
        project_root=project_root,
        current_series_name=series_name,
        current_template_library=template_library,
    )
    template_library["prompt_library_index_markdown_path"] = _normalize_project_relative_path(
        project_root,
        search_index_artifacts.get("prompt_library_index_markdown_path") or "",
    )
    template_library["prompt_library_index_json_path"] = _normalize_project_relative_path(
        project_root,
        search_index_artifacts.get("prompt_library_index_json_path") or "",
    )

    skill_json_path = series_dir / "seedance_purpose_skill_library.json"
    skill_md_path = series_dir / "seedance_purpose_skill_library.md"
    template_json_path = series_dir / "seedance_purpose_template_library.json"
    template_md_path = series_dir / "seedance_purpose_template_library.md"
    save_json_file(skill_json_path, skill_library)
    save_text_file(skill_md_path, render_skill_library_markdown(skill_library))
    save_json_file(template_json_path, template_library)
    save_text_file(template_md_path, render_template_library_markdown(template_library))
    return {
        "skill_library_json_path": str(skill_json_path),
        "skill_library_markdown_path": str(skill_md_path),
        "template_library_json_path": str(template_json_path),
        "template_library_markdown_path": str(template_md_path),
        "episode_catalog_count": len(catalogs),
        **prompt_library_artifacts,
        **search_index_artifacts,
    }


def render_episode_catalog_markdown(catalog: Mapping[str, Any]) -> str:
    lines = [
        f"# Seedance Beat Catalog：{catalog.get('series_name', '')} {catalog.get('episode_id', '')}",
        "",
        f"- 生成时间：{catalog.get('generated_at', '')}",
        f"- beat 数量：{catalog.get('beat_count', 0)}",
        f"- 来源分析：{catalog.get('source_analysis_path', '')}",
        "",
    ]
    for beat in list(catalog.get("beats") or []):
        lines.extend(
            [
                f"## {beat.get('beat_id', '')}｜{beat.get('display_title', '') or beat.get('primary_purpose', '')}",
                "",
                f"- 来源时间：{beat.get('time_range', '')}",
                f"- 来源时长：{beat.get('duration_seconds', 0)} 秒",
                f"- 建议还原时长：{beat.get('restored_duration_seconds', beat.get('duration_seconds', 0))} 秒",
                f"- 叙事目的：{beat.get('primary_purpose', '')}",
                f"- 剧情概述：{beat.get('display_summary', '') or beat.get('beat_summary', '')}",
                f"- 采样帧：{beat.get('frame_sample_count', len(beat.get('frame_samples', [])))} 张"
                f"｜落盘 {beat.get('saved_frame_count', beat.get('frame_sample_count', len(beat.get('frame_samples', []))))} 张",
                f"- 采样目录：{beat.get('frame_sample_dir', '') or '无'}",
                f"- 来源 story_beats：{'、'.join(beat.get('source_story_beat_ids', [])) or '无'}",
                f"- 二次视觉复盘：{'已启用' if beat.get('visual_second_pass_used') else '未启用/未命中'}",
                f"- 叙事目标：{beat.get('dramatic_goal', '')}",
                f"- 场景锚点：{beat.get('scene_anchor', {}).get('summary', '')}",
                f"- 承接入口：{beat.get('continuity_bridge_in', '')}",
                f"- 尾帧交棒：{beat.get('continuity_bridge_out', '')}",
                f"- 质量分：{beat.get('quality_score', 0)}",
                "",
                "### 镜头链",
                "",
            ]
        )
        if beat.get("visual_second_pass_summary"):
            lines.extend(
                [
                    "### 二次视觉观察",
                    "",
                    str(beat.get("visual_second_pass_summary") or ""),
                    "",
                ]
            )
        for shot in list(beat.get("shot_chain") or []):
            lines.append(f"- {shot.get('time_range', '')}｜{shot.get('story_function', '')}")
            if shot.get("camera_entry"):
                lines.append(f"  机位入口：{shot.get('camera_entry', '')}")
            if shot.get("subject_blocking"):
                lines.append(f"  站位关系：{shot.get('subject_blocking', '')}")
            if shot.get("action_timeline"):
                lines.append(f"  动作分段：{shot.get('action_timeline', '')}")
            if shot.get("visual_focus"):
                lines.append(f"  视觉焦点：{shot.get('visual_focus', '')}")
            if shot.get("camera_language"):
                lines.append(f"  镜头推进：{shot.get('camera_language', '')}")
            if shot.get("lighting_and_texture"):
                lines.append(f"  受光质感：{shot.get('lighting_and_texture', '')}")
            if shot.get("background_continuity"):
                lines.append(f"  连续性：{shot.get('background_continuity', '')}")
            if shot.get("dialogue_timing"):
                lines.append(f"  对白节奏：{shot.get('dialogue_timing', '')}")
            if shot.get("sound_bed"):
                lines.append(f"  声音层：{shot.get('sound_bed', '')}")
            if shot.get("transition_trigger"):
                lines.append(f"  切镜触发：{shot.get('transition_trigger', '')}")
        if beat.get("dialogue_windows"):
            lines.extend(["", "### 对白窗", ""])
            for item in beat.get("dialogue_windows", []):
                lines.append(
                    f"- {item.get('time_range', '')}｜{item.get('text', '')}"
                )
        lines.extend(
            [
                "",
                "### 还原版 Prompt",
                "",
                beat.get("restored_seedance_prompt", ""),
                "",
                "### 通用模板 Prompt",
                "",
                beat.get("generalized_template_prompt", ""),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_beat_segmentation_markdown(segmentation: Mapping[str, Any]) -> str:
    lines = [
        f"# Beat Segmentation：{segmentation.get('series_name', '')} {segmentation.get('episode_id', '')}",
        "",
        f"- 生成时间：{segmentation.get('generated_at', '')}",
        f"- 分段方法：{segmentation.get('segmentation_method', '')}",
        f"- raw shots：{segmentation.get('shot_count', 0)}",
        f"- beats：{segmentation.get('beat_count', 0)}",
        "",
    ]
    for beat in list(segmentation.get("beats") or []):
        lines.extend(
            [
                f"## {beat.get('beat_id', '')}",
                "",
                f"- 时间：{beat.get('time_range', '')}",
                f"- 时长：{beat.get('duration_seconds', 0)} 秒",
                f"- shot 数：{beat.get('shot_count', 0)}",
                f"- 采样帧：{beat.get('frame_sample_count', len(beat.get('frame_samples', [])))} 张"
                f"｜落盘 {beat.get('saved_frame_count', beat.get('frame_sample_count', len(beat.get('frame_samples', []))))} 张",
                f"- 采样目录：{beat.get('frame_sample_dir', '') or '无'}",
                f"- source story_beats：{'、'.join(beat.get('source_story_beat_ids', [])) or '无'}",
                f"- 目的提示：{beat.get('dominant_purpose_hint', '') or '无'}",
                f"- 切点原因：{'；'.join(beat.get('end_boundary_reasons', [])) or '无'}",
                "",
            ]
        )
        if beat.get("dialogue_windows"):
            lines.append("### 对白窗")
            lines.append("")
            for item in beat.get("dialogue_windows", []):
                lines.append(f"- {item.get('time_range', '')}｜{item.get('text', '')}")
            lines.append("")
        for shot in list(beat.get("shot_timeline") or []):
            lines.append(
                f"- {shot.get('shot_id', '')}｜{shot.get('time_range', '')}｜{shot.get('ocr_hint', '') or '无 OCR'}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_skill_library_markdown(library: Mapping[str, Any]) -> str:
    shared = dict(library.get("shared_series_rules") or {})
    lines = [
        f"# Seedance 镜头语言技能库：{library.get('series_name', '')}",
        "",
        f"- 生成时间：{library.get('generated_at', '')}",
        f"- 分类维度：{library.get('taxonomy_version', '')}",
        f"- 覆盖集数：{library.get('episode_count', 0)}",
        f"- 覆盖 beat：{library.get('beat_count', 0)}",
        "",
        "## 整剧共享规则",
        "",
    ]
    for title, key in [
        ("镜头语言规则", "camera_language_rules"),
        ("分镜执行规则", "storyboard_execution_rules"),
        ("对白时间规则", "dialogue_timing_rules"),
        ("连续性护栏", "continuity_guardrails"),
        ("负面模式", "negative_patterns"),
    ]:
        values = list(shared.get(key) or [])
        lines.append(f"### {title}")
        if not values:
            lines.append("- <空>")
        else:
            lines.extend(f"- {item}" for item in values)
        lines.append("")
    for purpose in list(library.get("purposes") or []):
        design_skill = dict(purpose.get("design_skill") or {})
        lines.extend(
            [
                f"## {purpose.get('purpose', '')}",
                "",
                f"- 说明：{purpose.get('description', '')}",
                f"- beat 数量：{purpose.get('beat_count', 0)}",
                f"- 覆盖集数：{purpose.get('episode_count', 0)}",
                f"- 时长画像：{purpose.get('duration_profile', {}).get('min_seconds', 0)}"
                f"-{purpose.get('duration_profile', {}).get('max_seconds', 0)} 秒"
                f"｜均值 {purpose.get('duration_profile', {}).get('avg_seconds', 0)} 秒",
                f"- 适用场景：{purpose.get('when_to_use', '')}",
                "",
            ]
        )
        for title, key in [
            ("叙事目标", "narrative_goals"),
            ("镜头设计规则", "camera_rules"),
            ("beat 设计规则", "beat_rules"),
            ("动作设计规则", "action_rules"),
            ("对白规则", "dialogue_rules"),
            ("连续性规则", "continuity_rules"),
            ("负面模式", "negative_patterns"),
        ]:
            values = list(design_skill.get(key) or [])
            lines.append(f"### {title}")
            if not values:
                lines.append("- <空>")
            else:
                lines.extend(f"- {item}" for item in values)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_template_library_markdown(library: Mapping[str, Any]) -> str:
    lines = [
        f"# Seedance 模板库：{library.get('series_name', '')}",
        "",
        f"- 生成时间：{library.get('generated_at', '')}",
        f"- 分类维度：{library.get('taxonomy_version', '')}",
        f"- 覆盖模板：{library.get('template_count', 0)}",
        f"- Prompt Library 根目录：{library.get('prompt_library_root', '') or '未导出'}",
        f"- Prompt Library 检索索引：{library.get('prompt_library_index_markdown_path', '') or '未导出'}",
        f"- Prompt Library 检索 JSON：{library.get('prompt_library_index_json_path', '') or '未导出'}",
        "",
    ]
    for purpose in list(library.get("purposes") or []):
        lines.extend(
            [
                f"## {purpose.get('purpose', '')}",
                "",
                f"- 模板数量：{purpose.get('template_count', 0)}",
                "",
            ]
        )
        for template in list(purpose.get("templates") or []):
            lines.extend(
                [
                    f"### {template.get('template_id', '')}｜{template.get('retrieval_title', '') or template.get('purpose', '')}",
                    "",
                    f"- 来源：{template.get('source_series_name', '')} {template.get('source_episode_id', '')} {template.get('source_beat_id', '')}",
                    f"- 时长：{template.get('duration_seconds', 0)} 秒",
                    f"- 质量分：{template.get('quality_score', 0)}",
                    f"- 主类：{template.get('primary_purpose', '') or template.get('purpose', '')}",
                    f"- 辅类：{'、'.join(template.get('secondary_purposes', [])) or '无'}",
                    f"- 分类置信度：{template.get('classification_confidence', 0)}",
                    f"- 分类说明：{template.get('ambiguity_note', '') or '无'}",
                    f"- 检索建议：{template.get('search_hint', '') or template.get('retrieval_summary', '') or template.get('when_to_use', '')}",
                    f"- 检索标题：{template.get('retrieval_title', '') or '无'}",
                    f"- 检索关键词：{'、'.join(template.get('search_keywords', [])) or '无'}",
                    f"- 场景标签：{'、'.join(template.get('scene_tags', [])) or '无'}",
                    f"- 关系标签：{'、'.join(template.get('relation_tags', [])) or '无'}",
                    f"- 调度标签：{'、'.join(template.get('staging_tags', [])) or '无'}",
                    f"- 镜头标签：{'、'.join(template.get('camera_tags', [])) or '无'}",
                    f"- 情绪标签：{'、'.join(template.get('emotion_tags', [])) or '无'}",
                    f"- 叙事标签：{'、'.join(template.get('narrative_tags', [])) or '无'}",
                    f"- 必填槽位：{'、'.join(template.get('required_slots', [])) or '无'}",
                    f"- Prompt Library 文件：{template.get('prompt_library_path', '') or '未导出'}",
                    "",
                    "#### 还原版 Prompt",
                    "",
                    template.get("restored_seedance_prompt", ""),
                    "",
                    "#### 通用模板 Prompt",
                    "",
                    template.get("generalized_template_prompt", ""),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _build_episode_beat_segmentation(
    *,
    series_name: str,
    episode_id: str,
    analysis_path: Path,
    normalized_story_beats: Sequence[Mapping[str, Any]],
    scene_list: Mapping[str, Any],
    transcript_segments: Mapping[str, Any],
    ocr_segments: Mapping[str, Any],
    episode_dir: Path,
    source_video_path: Path | None,
    settings: Mapping[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    transcript = list(transcript_segments.get("segments") or [])
    ocr = list(ocr_segments.get("segments") or [])
    story_timing_profile = _story_beat_timing_profile(normalized_story_beats, settings=settings)
    shot_frame_root = episode_dir / "seedance_learning" / "shot_frames"
    raw_shots = _normalize_scene_shots(
        scene_list=scene_list,
        ocr_segments=ocr,
        source_video_path=source_video_path,
        shot_frame_root=shot_frame_root,
    )
    resolved_shot_frame_root = str(shot_frame_root.resolve())
    sampled_shot_frames = sum(
        1
        for shot in raw_shots
        if str(shot.get("keyframe_path") or "").strip().startswith(resolved_shot_frame_root)
    )
    _emit_progress(
        progress_callback,
        f"Seedance 学习：发现 {len(raw_shots)} 个 raw shots；shot 中点帧缓存 {sampled_shot_frames} 张｜目录={resolved_shot_frame_root}",
    )
    if not raw_shots:
        fallback = _fallback_segmentation_from_story_beats(
            episode_id=episode_id,
            normalized_story_beats=normalized_story_beats,
            transcript_segments=transcript,
        )
        _emit_progress(
            progress_callback,
            f"Seedance 学习：未检测到 raw shots，回退为 story beat 分段，共 {len(fallback)} 个分镜。",
        )
        return {
            "series_name": series_name,
            "episode_id": episode_id,
            "generated_at": utc_timestamp(),
            "taxonomy_version": str(settings.get("taxonomy_version", DEFAULT_TAXONOMY_VERSION)),
            "source_analysis_path": str(analysis_path),
            "segmentation_method": "fallback_from_story_beats",
            "story_timing_profile": story_timing_profile,
            "shot_count": 0,
            "beat_count": len(fallback),
            "beats": fallback,
        }

    beats = _segment_shots_into_beats(
        shots=raw_shots,
        normalized_story_beats=normalized_story_beats,
        transcript_segments=transcript,
        settings=settings,
        story_timing_profile=story_timing_profile,
    )
    if not beats:
        beats = _fallback_segmentation_from_story_beats(
            episode_id=episode_id,
            normalized_story_beats=normalized_story_beats,
            transcript_segments=transcript,
        )
        _emit_progress(
            progress_callback,
            f"Seedance 学习：hybrid 分段为空，回退为 story beat 分段，共 {len(beats)} 个分镜。",
        )
    _emit_progress(
        progress_callback,
        f"Seedance 学习：视频已切成 {len(beats)} 个分镜 / beats（raw shots={len(raw_shots)}）。",
    )

    return {
        "series_name": series_name,
        "episode_id": episode_id,
        "generated_at": utc_timestamp(),
        "taxonomy_version": str(settings.get("taxonomy_version", DEFAULT_TAXONOMY_VERSION)),
        "source_analysis_path": str(analysis_path),
        "segmentation_method": "hybrid_shots_transcript_story_beats",
        "story_timing_profile": story_timing_profile,
        "shot_count": len(raw_shots),
        "beat_count": len(beats),
        "beats": beats,
    }


def _normalize_scene_shots(
    *,
    scene_list: Mapping[str, Any],
    ocr_segments: Sequence[Mapping[str, Any]],
    source_video_path: Path | None,
    shot_frame_root: Path,
) -> list[dict[str, Any]]:
    scenes = list(scene_list.get("scenes") or [])
    keyframes = list(scene_list.get("keyframes") or [])
    keyframes_by_scene: dict[str, dict[str, Any]] = {}
    for frame in keyframes:
        scene_id = str(frame.get("scene_id") or "").strip()
        if scene_id and scene_id not in keyframes_by_scene:
            keyframes_by_scene[scene_id] = dict(frame)

    shots: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, Mapping):
            continue
        start_seconds = round(float(scene.get("start_seconds", 0) or 0), 3)
        end_seconds = round(float(scene.get("end_seconds", start_seconds) or start_seconds), 3)
        if end_seconds <= start_seconds:
            end_seconds = round(start_seconds + 0.4, 3)
        scene_id = str(scene.get("scene_id") or f"scene-{index:04d}")
        keyframe = keyframes_by_scene.get(scene_id, {})
        ocr_hint = _ocr_hint_for_range(ocr_segments, start_seconds, end_seconds)
        if not ocr_hint:
            ocr_hint = _normalize_spaces(str(keyframe.get("linked_ocr_text") or ""))
        midpoint_seconds = round((start_seconds + end_seconds) / 2.0, 3)
        learning_frame_path = _ensure_reference_frame(
            source_video_path=source_video_path,
            output_path=shot_frame_root / f"{scene_id}.jpg",
            timestamp_seconds=midpoint_seconds,
        )
        resolved_keyframe_path = str(keyframe.get("model_frame_path") or keyframe.get("frame_path") or "").strip()
        shots.append(
            {
                "shot_id": scene_id,
                "source_scene_id": scene_id,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": round(end_seconds - start_seconds, 3),
                "time_range": f"{start_seconds:.2f}-{end_seconds:.2f}s",
                "ocr_hint": ocr_hint[:160],
                "keyframe_path": learning_frame_path or resolved_keyframe_path,
                "fallback_sampled_keyframe_path": resolved_keyframe_path,
                "keyframe_midpoint_seconds": midpoint_seconds,
            }
        )
    return shots


def _attach_beat_frame_samples(
    *,
    beats: Sequence[dict[str, Any]],
    source_video_path: Path | None,
    beat_frame_root: Path,
    settings: Mapping[str, Any],
) -> None:
    for beat in beats:
        _sample_beat_frame_samples(
            beat=beat,
            source_video_path=source_video_path,
            beat_frame_root=beat_frame_root,
            settings=settings,
        )


def _sample_beat_frame_samples(
    *,
    beat: dict[str, Any],
    source_video_path: Path | None,
    beat_frame_root: Path,
    settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    start_seconds = float(beat.get("start_seconds", 0) or 0)
    end_seconds = float(beat.get("end_seconds", start_seconds) or start_seconds)
    sample_timestamps = _compute_beat_sample_timestamps(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        min_frames=int(settings.get("min_beat_frames", DEFAULT_MIN_BEAT_FRAMES)),
        max_frames=int(settings.get("max_beat_frames", DEFAULT_MAX_BEAT_FRAMES)),
    )
    beat_id = str(beat.get("beat_id") or "beat")
    frame_dir = beat_frame_root / beat_id
    frame_samples: list[dict[str, Any]] = []
    for index, timestamp_seconds in enumerate(sample_timestamps, start=1):
        output_path = frame_dir / f"frame_{index:02d}_{timestamp_seconds:.2f}.jpg"
        frame_path = _ensure_reference_frame(
            source_video_path=source_video_path,
            output_path=output_path,
            timestamp_seconds=timestamp_seconds,
        )
        frame_samples.append(
            {
                "frame_id": f"{beat_id}-F{index:02d}",
                "timestamp_seconds": round(timestamp_seconds, 2),
                "path": frame_path,
            }
        )
    beat["frame_samples"] = frame_samples
    beat["frame_sample_count"] = len(frame_samples)
    beat["saved_frame_count"] = sum(1 for item in frame_samples if str(item.get("path") or "").strip())
    beat["frame_sample_dir"] = str(frame_dir.resolve())
    return frame_samples


def _compute_beat_sample_timestamps(
    *,
    start_seconds: float,
    end_seconds: float,
    min_frames: int,
    max_frames: int,
) -> list[float]:
    duration = max(0.6, end_seconds - start_seconds)
    sample_count = int(min(max_frames, max(min_frames, round(duration / 1.8))))
    if sample_count <= 1:
        return [round((start_seconds + end_seconds) / 2.0, 3)]
    timestamps = []
    for index in range(sample_count):
        ratio = (index + 1) / (sample_count + 1)
        timestamps.append(round(start_seconds + duration * ratio, 3))
    return timestamps


def _ensure_reference_frame(
    *,
    source_video_path: Path | None,
    output_path: Path,
    timestamp_seconds: float,
) -> str:
    if not source_video_path or not str(source_video_path):
        return ""
    resolved_video = source_video_path.expanduser().resolve()
    if not resolved_video.exists():
        return ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return str(output_path.resolve())
    ffmpeg_binary = _safe_ffmpeg_binary()
    if not ffmpeg_binary:
        return ""
    try:
        subprocess.run(
            [
                ffmpeg_binary,
                "-y",
                "-ss",
                f"{max(0.0, float(timestamp_seconds)):.3f}",
                "-i",
                str(resolved_video),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return str(output_path.resolve()) if output_path.exists() else ""


def _safe_ffmpeg_binary() -> str:
    try:
        return get_ffmpeg_exe()
    except Exception:
        return ""


def _segment_shots_into_beats(
    *,
    shots: Sequence[Mapping[str, Any]],
    normalized_story_beats: Sequence[Mapping[str, Any]],
    transcript_segments: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    story_timing_profile: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not shots:
        return []
    min_duration = float(settings.get("min_beat_duration", DEFAULT_MIN_BEAT_DURATION))
    target_duration = float(settings.get("target_beat_duration", DEFAULT_TARGET_BEAT_DURATION))
    max_duration = float(settings.get("max_beat_duration", DEFAULT_MAX_BEAT_DURATION))
    story_boundaries_enabled = bool(dict(story_timing_profile or {}).get("reliable_for_boundaries", True))
    beats: list[dict[str, Any]] = []
    current: list[Mapping[str, Any]] = []
    beat_index = 1

    for index, shot in enumerate(shots):
        current.append(shot)
        next_shot = shots[index + 1] if index + 1 < len(shots) else None
        current_start = float(current[0].get("start_seconds", 0) or 0)
        current_end = float(current[-1].get("end_seconds", current_start) or current_start)
        current_duration = round(current_end - current_start, 3)
        should_cut = False
        boundary_score = 0.0
        boundary_reasons: list[str] = []

        if len(current) >= 2:
            previous_shots = current[:-1]
            previous_start = float(previous_shots[0].get("start_seconds", 0) or 0)
            previous_end = float(previous_shots[-1].get("end_seconds", previous_start) or previous_start)
            previous_primary_story = _canonical_story_beat_for_range(
                normalized_story_beats,
                previous_start,
                previous_end,
            )
            current_shot_primary_story = _canonical_story_beat_for_range(
                normalized_story_beats,
                float(shot.get("start_seconds", current_start) or current_start),
                float(shot.get("end_seconds", current_end) or current_end),
            )
            previous_story_id = str(previous_primary_story.get("beat_id") or "").strip()
            current_shot_story_id = str(current_shot_primary_story.get("beat_id") or "").strip()
            if (
                story_boundaries_enabled
                and previous_story_id
                and current_shot_story_id
                and previous_story_id != current_shot_story_id
                and (previous_end - previous_start) >= min_duration
            ):
                beats.append(
                    _build_beat_segment_record(
                        beat_index=beat_index,
                        shots=previous_shots,
                        transcript_segments=transcript_segments,
                        normalized_story_beats=normalized_story_beats,
                        story_timing_reliable=story_boundaries_enabled,
                        boundary_score=1.45,
                        boundary_reasons=[
                            f"当前镜头切入新的主 story beat（{current_shot_story_id}），为避免跨 beat 混写在前一镜收束"
                        ],
                        is_last=False,
                    )
                )
                beat_index += 1
                current = [shot]
                if next_shot is None:
                    beats.append(
                        _build_beat_segment_record(
                            beat_index=beat_index,
                            shots=current,
                            transcript_segments=transcript_segments,
                            normalized_story_beats=normalized_story_beats,
                            story_timing_reliable=story_boundaries_enabled,
                            boundary_score=1.2,
                            boundary_reasons=["视频尾段收束"],
                            is_last=True,
                        )
                    )
                    beat_index += 1
                    current = []
                continue

        if current_duration > max_duration + 0.8 and len(current) >= 2:
            previous_shots = current[:-1]
            previous_start = float(previous_shots[0].get("start_seconds", 0) or 0)
            previous_end = float(previous_shots[-1].get("end_seconds", previous_start) or previous_start)
            previous_duration = previous_end - previous_start
            if previous_duration >= max(5.0, min_duration - 2.0):
                overflow_score, overflow_reasons = _beat_boundary_score(
                    current=previous_shots,
                    next_shot=current[-1],
                    normalized_story_beats=normalized_story_beats,
                    transcript_segments=transcript_segments,
                    min_duration=min_duration,
                    target_duration=target_duration,
                    max_duration=max_duration,
                    use_story_timing=story_boundaries_enabled,
                )
                if overflow_score >= 0.4 or previous_duration >= target_duration - 0.5:
                    beats.append(
                        _build_beat_segment_record(
                            beat_index=beat_index,
                            shots=previous_shots,
                            transcript_segments=transcript_segments,
                            normalized_story_beats=normalized_story_beats,
                            story_timing_reliable=story_boundaries_enabled,
                            boundary_score=max(overflow_score, 0.6),
                            boundary_reasons=[*overflow_reasons, "为避免超长 beat，提前在上一镜头收束"],
                            is_last=False,
                        )
                    )
                    beat_index += 1
                    current = [current[-1]]
                    if next_shot is None:
                        beats.append(
                            _build_beat_segment_record(
                                beat_index=beat_index,
                                shots=current,
                                transcript_segments=transcript_segments,
                                normalized_story_beats=normalized_story_beats,
                                story_timing_reliable=story_boundaries_enabled,
                                boundary_score=1.2,
                                boundary_reasons=["视频尾段收束"],
                                is_last=True,
                            )
                        )
                        beat_index += 1
                        current = []
                    continue

        if next_shot is None:
            should_cut = True
            boundary_score = 1.2
            boundary_reasons = ["视频尾段收束"]
        else:
            boundary_score, boundary_reasons = _beat_boundary_score(
                current=current,
                next_shot=next_shot,
                normalized_story_beats=normalized_story_beats,
                transcript_segments=transcript_segments,
                min_duration=min_duration,
                target_duration=target_duration,
                max_duration=max_duration,
                use_story_timing=story_boundaries_enabled,
            )
            current_story_ids: set[str] = set()
            next_story_ids: set[str] = set()
            introduces_new_story = False
            if story_boundaries_enabled:
                current_story_ids = {
                    str(item.get("beat_id") or "")
                    for item in _significant_story_beats_for_range(normalized_story_beats, current_start, current_end)
                    if str(item.get("beat_id") or "").strip()
                }
                next_story_ids = {
                    str(item.get("beat_id") or "")
                    for item in _significant_story_beats_for_range(
                        normalized_story_beats,
                        float(next_shot.get("start_seconds", current_end) or current_end),
                        float(next_shot.get("end_seconds", current_end) or current_end),
                    )
                    if str(item.get("beat_id") or "").strip()
                }
                introduces_new_story = bool(next_story_ids - current_story_ids)
            if introduces_new_story and current_duration >= max(6.0, min_duration - 2.0):
                boundary_score = max(boundary_score, 0.9)
                boundary_reasons = [*boundary_reasons, "下一镜进入新的 story beat，当前段先收束"]
            if introduces_new_story and len(current_story_ids) >= 2 and current_duration >= max(5.0, min_duration - 2.5):
                should_cut = True
                boundary_score = max(boundary_score, 1.1)
                boundary_reasons = [*boundary_reasons, "当前段已覆盖多个 story beat，避免继续混入新事件"]
            if current_duration >= max_duration:
                should_cut = True
                boundary_score = max(boundary_score, 1.4)
                boundary_reasons = ["达到最大 beat 时长", *boundary_reasons]
            elif current_duration >= min_duration:
                if boundary_score >= 0.95:
                    should_cut = True
                elif current_duration >= target_duration and boundary_score >= 0.55:
                    should_cut = True
                elif current_duration >= target_duration + 1.5:
                    should_cut = True
                    boundary_reasons = [*boundary_reasons, "已超过目标 beat 时长，适合在此收束"]
            elif current_duration >= max(5.0, min_duration - 2.5) and boundary_score >= 1.25:
                should_cut = True
                boundary_reasons = [*boundary_reasons, "边界信号强，提前形成完整 beat"]

        if should_cut:
            beats.append(
                _build_beat_segment_record(
                    beat_index=beat_index,
                    shots=current,
                    transcript_segments=transcript_segments,
                    normalized_story_beats=normalized_story_beats,
                    story_timing_reliable=story_boundaries_enabled,
                    boundary_score=boundary_score,
                    boundary_reasons=boundary_reasons,
                    is_last=next_shot is None,
                )
            )
            beat_index += 1
            current = []

    if current:
        beats.append(
            _build_beat_segment_record(
                beat_index=beat_index,
                shots=current,
                transcript_segments=transcript_segments,
                normalized_story_beats=normalized_story_beats,
                story_timing_reliable=story_boundaries_enabled,
                boundary_score=1.0,
                boundary_reasons=["视频尾段收束"],
                is_last=True,
            )
        )

    beats = _consolidate_underfilled_beats(
        beats=beats,
        min_duration=min_duration,
        max_duration=max_duration,
        allow_cross_story_merge=not story_boundaries_enabled,
    )
    beats = _split_overlong_segment_beats(
        beats=beats,
        transcript_segments=transcript_segments,
        normalized_story_beats=normalized_story_beats,
        story_timing_reliable=story_boundaries_enabled,
        min_duration=min_duration,
        target_duration=target_duration,
        max_duration=max_duration,
    )
    beats = _merge_short_tail_beat(beats=beats, min_duration=min_duration, max_duration=max_duration)
    return beats


def _beat_boundary_score(
    *,
    current: Sequence[Mapping[str, Any]],
    next_shot: Mapping[str, Any],
    normalized_story_beats: Sequence[Mapping[str, Any]],
    transcript_segments: Sequence[Mapping[str, Any]],
    min_duration: float,
    target_duration: float,
    max_duration: float,
    use_story_timing: bool,
) -> tuple[float, list[str]]:
    current_start = float(current[0].get("start_seconds", 0) or 0)
    current_end = float(current[-1].get("end_seconds", current_start) or current_start)
    current_duration = max(0.2, current_end - current_start)
    next_end = float(next_shot.get("end_seconds", current_end) or current_end)
    reasons: list[str] = []
    score = 0.0

    if use_story_timing:
        story_score, story_reasons = _story_boundary_score(
            current_start=current_start,
            current_end=current_end,
            next_end=next_end,
            normalized_story_beats=normalized_story_beats,
        )
        score += story_score
        reasons.extend(story_reasons)

    transcript_score, transcript_reasons = _transcript_boundary_score(
        current_start=current_start,
        current_end=current_end,
        transcript_segments=transcript_segments,
    )
    score += transcript_score
    reasons.extend(transcript_reasons)

    if use_story_timing:
        purpose_score, purpose_reasons = _purpose_shift_score(
            current_start=current_start,
            current_end=current_end,
            next_end=next_end,
            normalized_story_beats=normalized_story_beats,
            transcript_segments=transcript_segments,
        )
        score += purpose_score
        reasons.extend(purpose_reasons)

    last_shot_duration = float(current[-1].get("duration_seconds", 0) or 0)
    next_shot_duration = float(next_shot.get("duration_seconds", 0) or 0)
    if last_shot_duration >= 2.2 and next_shot_duration >= 2.2:
        score += 0.12
        reasons.append("前后镜头都较完整，适合作为 beat 交界")
    if current_duration >= target_duration:
        score += 0.15
        reasons.append("当前 beat 已达到目标时长")
    elif current_duration >= min_duration:
        score += 0.05
    if current_duration >= max_duration - 0.8:
        score += 0.25
    return round(score, 2), _rank_lines(reasons, limit=5)


def _story_boundary_score(
    *,
    current_start: float,
    current_end: float,
    next_end: float,
    normalized_story_beats: Sequence[Mapping[str, Any]],
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    current_story_ids = {
        str(item.get("beat_id") or "")
        for item in _story_beats_for_range(normalized_story_beats, current_start, current_end)
    }
    next_story_ids = {
        str(item.get("beat_id") or "")
        for item in _story_beats_for_range(normalized_story_beats, current_end, next_end)
    }
    if current_story_ids and next_story_ids and current_story_ids != next_story_ids:
        score += 0.55
        reasons.append("story beat 发生切换")

    for item in normalized_story_beats:
        end_seconds = float(item.get("end_seconds", 0) or 0)
        start_seconds = float(item.get("start_seconds", 0) or 0)
        if abs(end_seconds - current_end) <= 1.0:
            score += 0.4
            reasons.append(f"靠近 story beat {item.get('beat_id', '')} 的结尾")
        if abs(start_seconds - current_end) <= 0.8:
            score += 0.25
            reasons.append(f"靠近 story beat {item.get('beat_id', '')} 的起点")
    return round(score, 2), reasons


def _transcript_boundary_score(
    *,
    current_start: float,
    current_end: float,
    transcript_segments: Sequence[Mapping[str, Any]],
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    overlapping = _transcript_segments_for_range(transcript_segments, current_start, current_end)
    if not overlapping:
        return 0.0, reasons

    last_segment = overlapping[-1]
    last_end = float(last_segment.get("end", current_end) or current_end)
    if abs(last_end - current_end) <= 0.45:
        score += 0.25
        reasons.append("边界落在对白句尾附近")
    trailing_gap = current_end - last_end
    if trailing_gap >= 0.35:
        score += 0.18
        reasons.append("对白后留出停顿，适合收束")
    ending_text = str(last_segment.get("text") or "").strip()
    if ending_text and ending_text[-1] in {"了", "吧", "呢", "吗", "啊", "呀", "哇"}:
        score += 0.08
        reasons.append("对白语气完成度较高")
    return round(score, 2), reasons


def _purpose_shift_score(
    *,
    current_start: float,
    current_end: float,
    next_end: float,
    normalized_story_beats: Sequence[Mapping[str, Any]],
    transcript_segments: Sequence[Mapping[str, Any]],
) -> tuple[float, list[str]]:
    current_text = _purpose_source_text(
        story_beats=_story_beats_for_range(normalized_story_beats, current_start, current_end),
        transcript_segments=_transcript_segments_for_range(transcript_segments, current_start, current_end),
    )
    next_text = _purpose_source_text(
        story_beats=_story_beats_for_range(normalized_story_beats, current_end, next_end),
        transcript_segments=_transcript_segments_for_range(transcript_segments, current_end, next_end),
    )
    if not current_text or not next_text:
        return 0.0, []
    current_purpose = infer_primary_purpose(current_text)
    next_purpose = infer_primary_purpose(next_text)
    if current_purpose == next_purpose:
        return 0.0, []
    return 0.22, [f"分镜目的从“{current_purpose}”切向“{next_purpose}”"]


def _build_beat_segment_record(
    *,
    beat_index: int,
    shots: Sequence[Mapping[str, Any]],
    transcript_segments: Sequence[Mapping[str, Any]],
    normalized_story_beats: Sequence[Mapping[str, Any]],
    story_timing_reliable: bool,
    boundary_score: float,
    boundary_reasons: Sequence[str],
    is_last: bool,
) -> dict[str, Any]:
    start_seconds = round(float(shots[0].get("start_seconds", 0) or 0), 2)
    end_seconds = round(float(shots[-1].get("end_seconds", start_seconds) or start_seconds), 2)
    transcript_window = _transcript_segments_for_range(transcript_segments, start_seconds, end_seconds)
    story_window = (
        _canonical_story_window_for_range(normalized_story_beats, start_seconds, end_seconds)
        if story_timing_reliable
        else []
    )
    transcript_text = _normalize_spaces(" ".join(str(item.get("text") or "") for item in transcript_window))
    ocr_hints = [str(shot.get("ocr_hint") or "") for shot in shots]
    ocr_summary = _combine_unique_texts(ocr_hints)
    purpose_text = _purpose_source_text(
        story_beats=story_window,
        transcript_segments=transcript_window,
        extra_parts=[ocr_summary],
    )
    dominant_purpose_hint = infer_primary_purpose(purpose_text, is_last=is_last)
    summary_text = _build_segment_summary(
        story_beats=story_window,
        transcript_text=transcript_text,
        ocr_hints=ocr_hints,
    )
    shot_timeline = []
    for shot in shots:
        shot_timeline.append(
            {
                "shot_id": str(shot.get("shot_id") or ""),
                "source_scene_id": str(shot.get("source_scene_id") or ""),
                "time_range": str(shot.get("time_range") or ""),
                "start_seconds": round(float(shot.get("start_seconds", 0) or 0), 2),
                "end_seconds": round(float(shot.get("end_seconds", 0) or 0), 2),
                "duration_seconds": round(float(shot.get("duration_seconds", 0) or 0), 2),
                "ocr_hint": str(shot.get("ocr_hint") or ""),
                "keyframe_path": str(shot.get("keyframe_path") or ""),
            }
        )
    return {
        "beat_id": f"BS{beat_index:02d}",
        "time_range": f"{start_seconds:.2f}-{end_seconds:.2f}s",
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "duration_seconds": round(max(0.6, end_seconds - start_seconds), 2),
        "shot_ids": [str(shot.get("shot_id") or "") for shot in shots],
        "shot_count": len(shots),
        "shot_timeline": shot_timeline,
        "dialogue_windows": _dialogue_windows_for_range(transcript_segments, start_seconds, end_seconds),
        "transcript_segment_ids": [str(item.get("segment_id") or "") for item in transcript_window],
        "transcript_text": transcript_text,
        "source_story_beat_ids": [str(item.get("beat_id") or "") for item in story_window],
        "source_story_titles": [str(item.get("title") or "") for item in story_window if str(item.get("title") or "").strip()],
        "end_boundary_score": round(boundary_score, 2),
        "end_boundary_reasons": list(boundary_reasons)[:5],
        "dominant_purpose_hint": dominant_purpose_hint,
        "summary_text": summary_text,
    }


def _merge_short_tail_beat(
    *,
    beats: Sequence[Mapping[str, Any]],
    min_duration: float,
    max_duration: float,
) -> list[dict[str, Any]]:
    if len(beats) < 2:
        return [dict(item) for item in beats]
    result = [dict(item) for item in beats]
    last = result[-1]
    prev = result[-2]
    last_duration = float(last.get("duration_seconds", 0) or 0)
    prev_duration = float(prev.get("duration_seconds", 0) or 0)
    if last_duration >= max(5.0, min_duration * 0.8):
        return result
    if prev_duration + last_duration > max_duration + 2.5:
        return result
    prev_story_ids = _dedupe_strings(list(prev.get("source_story_beat_ids") or []))
    last_story_ids = _dedupe_strings(list(last.get("source_story_beat_ids") or []))
    if prev_story_ids and last_story_ids and prev_story_ids != last_story_ids:
        return result
    prev = _merge_adjacent_segment_records(
        prev,
        last,
        merge_reason="尾段过短，合并进上一 beat",
        is_last=True,
    )
    result[-2] = prev
    result.pop()
    return _renumber_segment_beats(result)


def _merge_adjacent_segment_records(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    merge_reason: str,
    is_last: bool,
) -> dict[str, Any]:
    merged = dict(left)
    right_item = dict(right)
    merged_shot_timeline = [*list(merged.get("shot_timeline") or []), *list(right_item.get("shot_timeline") or [])]
    merged_dialogue = [*list(merged.get("dialogue_windows") or []), *list(right_item.get("dialogue_windows") or [])]
    merged_transcript_ids = [
        *list(merged.get("transcript_segment_ids") or []),
        *list(right_item.get("transcript_segment_ids") or []),
    ]
    merged_story_ids = [*list(merged.get("source_story_beat_ids") or []), *list(right_item.get("source_story_beat_ids") or [])]
    merged_story_titles = [*list(merged.get("source_story_titles") or []), *list(right_item.get("source_story_titles") or [])]

    merged["end_seconds"] = round(float(right_item.get("end_seconds", merged.get("end_seconds", 0)) or 0), 2)
    merged["time_range"] = f"{float(merged.get('start_seconds', 0) or 0):.2f}-{float(merged['end_seconds']):.2f}s"
    merged["duration_seconds"] = round(
        max(0.6, float(merged["end_seconds"]) - float(merged.get("start_seconds", 0) or 0)),
        2,
    )
    merged["shot_timeline"] = merged_shot_timeline
    merged["shot_ids"] = [str(item.get("shot_id") or "") for item in merged_shot_timeline]
    merged["shot_count"] = len(merged_shot_timeline)
    merged["dialogue_windows"] = _dedupe_dict_items(merged_dialogue, key="time_range")
    merged["transcript_segment_ids"] = _dedupe_strings(merged_transcript_ids)
    merged["transcript_text"] = _normalize_spaces(
        f"{str(merged.get('transcript_text') or '')} {str(right_item.get('transcript_text') or '')}"
    )
    merged["source_story_beat_ids"] = _dedupe_strings(merged_story_ids)
    merged["source_story_titles"] = _dedupe_strings(merged_story_titles)
    merged["summary_text"] = _normalize_spaces(
        f"{str(merged.get('summary_text') or '')} {str(right_item.get('summary_text') or '')}"
    )[:320]
    merged["end_boundary_score"] = max(
        float(merged.get("end_boundary_score", 0) or 0),
        float(right_item.get("end_boundary_score", 0) or 0),
    )
    merged["end_boundary_reasons"] = _rank_lines(
        [*list(merged.get("end_boundary_reasons") or []), *list(right_item.get("end_boundary_reasons") or []), merge_reason],
        limit=5,
    )
    merged["dominant_purpose_hint"] = infer_primary_purpose_from_parts(
        [
            merged.get("dominant_purpose_hint"),
            right_item.get("dominant_purpose_hint"),
            merged.get("summary_text"),
            merged.get("transcript_text"),
        ],
        is_last=is_last,
    )
    return merged


def _renumber_segment_beats(beats: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    renumbered: list[dict[str, Any]] = []
    for index, item in enumerate(beats, start=1):
        current = dict(item)
        current["beat_id"] = f"BS{index:02d}"
        renumbered.append(current)
    return renumbered


def _consolidate_underfilled_beats(
    *,
    beats: Sequence[Mapping[str, Any]],
    min_duration: float,
    max_duration: float,
    allow_cross_story_merge: bool,
) -> list[dict[str, Any]]:
    if len(beats) < 2:
        return _renumber_segment_beats(beats)
    result = [dict(item) for item in beats]
    target_duration = (min_duration + max_duration) / 2.0

    def _merged_duration(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        return float(right.get("end_seconds", 0) or 0) - float(left.get("start_seconds", 0) or 0)

    def _can_merge(left: Mapping[str, Any], right: Mapping[str, Any], *, allow_soft_overflow: bool) -> bool:
        left_story_ids = _dedupe_strings(list(left.get("source_story_beat_ids") or []))
        right_story_ids = _dedupe_strings(list(right.get("source_story_beat_ids") or []))
        story_conflict = bool(left_story_ids and right_story_ids and left_story_ids != right_story_ids)
        if story_conflict and not allow_cross_story_merge:
            return False
        overflow_limit = max_duration + (1.5 if allow_soft_overflow else 0.0)
        return _merged_duration(left, right) <= overflow_limit

    changed = True
    while changed and len(result) >= 2:
        changed = False
        for index, current in enumerate(list(result)):
            current_duration = float(current.get("duration_seconds", 0) or 0)
            if current_duration >= min_duration:
                continue
            left_candidate = dict(result[index - 1]) if index > 0 else None
            right_candidate = dict(result[index + 1]) if index + 1 < len(result) else None
            options: list[tuple[str, float, bool]] = []
            if left_candidate and _can_merge(left_candidate, current, allow_soft_overflow=False):
                options.append(("left", abs(_merged_duration(left_candidate, current) - target_duration), False))
            if right_candidate and _can_merge(current, right_candidate, allow_soft_overflow=False):
                options.append(("right", abs(_merged_duration(current, right_candidate) - target_duration), False))
            if not options:
                if left_candidate and _can_merge(left_candidate, current, allow_soft_overflow=True):
                    options.append(("left", abs(_merged_duration(left_candidate, current) - target_duration), True))
                if right_candidate and _can_merge(current, right_candidate, allow_soft_overflow=True):
                    options.append(("right", abs(_merged_duration(current, right_candidate) - target_duration), True))
            if not options:
                continue
            direction, _, _ = min(options, key=lambda item: (item[1], 0 if item[0] == "left" else 1))
            if direction == "left" and left_candidate:
                result[index - 1] = _merge_adjacent_segment_records(
                    left_candidate,
                    current,
                    merge_reason="当前 beat 不足 8 秒，回并上一 beat 补足 source 时长",
                    is_last=index == len(result) - 1,
                )
                result.pop(index)
            elif direction == "right" and right_candidate:
                result[index] = _merge_adjacent_segment_records(
                    current,
                    right_candidate,
                    merge_reason="当前 beat 不足 8 秒，向后并段补足 source 时长",
                    is_last=index + 1 >= len(result) - 1,
                )
                result.pop(index + 1)
            else:
                continue
            changed = True
            break
    return _renumber_segment_beats(result)


def _split_overlong_segment_beats(
    *,
    beats: Sequence[Mapping[str, Any]],
    transcript_segments: Sequence[Mapping[str, Any]],
    normalized_story_beats: Sequence[Mapping[str, Any]],
    story_timing_reliable: bool,
    min_duration: float,
    target_duration: float,
    max_duration: float,
) -> list[dict[str, Any]]:
    if not beats:
        return []
    split_result: list[dict[str, Any]] = []
    for item in beats:
        queue: list[list[dict[str, Any]]] = [[dict(shot) for shot in list(item.get("shot_timeline") or [])]]
        if not queue[0]:
            split_result.append(dict(item))
            continue
        while queue:
            shot_chunk = queue.pop(0)
            chunk_duration = float(shot_chunk[-1].get("end_seconds", 0) or 0) - float(shot_chunk[0].get("start_seconds", 0) or 0)
            if chunk_duration <= max_duration + 0.6 or len(shot_chunk) < 2:
                split_result.append(
                    _build_beat_segment_record(
                        beat_index=len(split_result) + 1,
                        shots=shot_chunk,
                        transcript_segments=transcript_segments,
                        normalized_story_beats=normalized_story_beats,
                        story_timing_reliable=story_timing_reliable,
                        boundary_score=1.0,
                        boundary_reasons=["超长 beat 重新按 shot 边界拆分"],
                        is_last=False,
                    )
                )
                continue
            split_index = _choose_shot_split_index(
                shots=shot_chunk,
                min_duration=min_duration,
                target_duration=target_duration,
                max_duration=max_duration,
            )
            if split_index is None:
                split_result.append(
                    _build_beat_segment_record(
                        beat_index=len(split_result) + 1,
                        shots=shot_chunk,
                        transcript_segments=transcript_segments,
                        normalized_story_beats=normalized_story_beats,
                        story_timing_reliable=story_timing_reliable,
                        boundary_score=1.0,
                        boundary_reasons=["超长 beat 无合适 shot 切点，保留原段"],
                        is_last=False,
                    )
                )
                continue
            queue.insert(0, shot_chunk[split_index:])
            queue.insert(0, shot_chunk[:split_index])
    return _renumber_segment_beats(split_result)


def _choose_shot_split_index(
    *,
    shots: Sequence[Mapping[str, Any]],
    min_duration: float,
    target_duration: float,
    max_duration: float,
) -> int | None:
    if len(shots) < 2:
        return None
    overall_start = float(shots[0].get("start_seconds", 0) or 0)
    overall_end = float(shots[-1].get("end_seconds", overall_start) or overall_start)
    overall_duration = max(0.6, overall_end - overall_start)
    desired_left_duration = min(max(target_duration, overall_duration / 2.0), max_duration)
    best_index: int | None = None
    best_penalty = float("inf")
    soft_min_duration = max(7.0, min_duration - 1.0)
    for index in range(1, len(shots)):
        left_end = float(shots[index - 1].get("end_seconds", overall_start) or overall_start)
        right_start = float(shots[index].get("start_seconds", left_end) or left_end)
        left_duration = max(0.6, left_end - overall_start)
        right_duration = max(0.6, overall_end - right_start)
        penalty = abs(left_duration - desired_left_duration) + abs(right_duration - target_duration)
        for duration in (left_duration, right_duration):
            if duration < soft_min_duration:
                penalty += (soft_min_duration - duration) * 4.0
            if duration > max_duration:
                penalty += (duration - max_duration) * 4.0
        if penalty < best_penalty:
            best_penalty = penalty
            best_index = index
    return best_index


def _fallback_segmentation_from_story_beats(
    *,
    episode_id: str,
    normalized_story_beats: Sequence[Mapping[str, Any]],
    transcript_segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fallback: list[dict[str, Any]] = []
    for index, item in enumerate(normalized_story_beats, start=1):
        start_seconds = round(float(item.get("start_seconds", 0) or 0), 2)
        end_seconds = round(float(item.get("end_seconds", start_seconds) or start_seconds), 2)
        fallback.append(
            {
                "beat_id": f"BS{index:02d}",
                "time_range": f"{start_seconds:.2f}-{end_seconds:.2f}s",
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": round(max(0.6, end_seconds - start_seconds), 2),
                "shot_ids": [],
                "shot_count": 0,
                "shot_timeline": [],
                "dialogue_windows": _dialogue_windows_for_range(transcript_segments, start_seconds, end_seconds),
                "transcript_segment_ids": [str(seg.get("segment_id") or "") for seg in _transcript_segments_for_range(transcript_segments, start_seconds, end_seconds)],
                "transcript_text": _normalize_spaces(
                    " ".join(str(seg.get("text") or "") for seg in _transcript_segments_for_range(transcript_segments, start_seconds, end_seconds))
                ),
                "source_story_beat_ids": [str(item.get("beat_id") or "")],
                "source_story_titles": [str(item.get("title") or "")],
                "end_boundary_score": 1.0,
                "end_boundary_reasons": ["回退到上游 story beat 切分"],
                "dominant_purpose_hint": infer_primary_purpose(str(item.get("text_blob") or ""), is_last=index == len(normalized_story_beats)),
                "summary_text": _normalize_spaces(
                    " ".join([str(item.get("title") or ""), str(item.get("summary") or "")])
                ),
            }
        )
    return fallback


def _build_episode_catalog(
    *,
    series_name: str,
    episode_id: str,
    analysis: Mapping[str, Any],
    analysis_path: Path,
    scene_list: Mapping[str, Any],
    transcript_segments: Mapping[str, Any],
    ocr_segments: Mapping[str, Any],
    normalized_story_beats: Sequence[Mapping[str, Any]],
    beat_segmentation: Mapping[str, Any],
    strength_playbook: Mapping[str, Any],
    settings: Mapping[str, Any],
    source_video_path: Path | None,
    beat_frame_root: Path,
    episode_dir: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    beats: list[dict[str, Any]] = []
    transcript = list(transcript_segments.get("segments") or [])
    scenes = list(scene_list.get("scenes") or [])
    keyframes = list(scene_list.get("keyframes") or [])
    segmentation_beats = list(beat_segmentation.get("beats") or [])
    story_timing_reliable = bool(
        dict(beat_segmentation.get("story_timing_profile") or {}).get("reliable_for_prompt_grounding", True)
    )
    visual_second_pass_dir = episode_dir / "seedance_learning" / "visual_second_pass"
    visual_second_pass_client = _build_visual_second_pass_client(
        settings=settings,
        progress_callback=progress_callback,
    )
    if segmentation_beats:
        _emit_progress(
            progress_callback,
            f"Seedance 学习：开始逐分镜处理，共 {len(segmentation_beats)} 个分镜；beat 帧目录={beat_frame_root.resolve()}",
        )

    for index, segment in enumerate(segmentation_beats, start=1):
        start_seconds = round(float(segment.get("start_seconds", 0) or 0), 2)
        end_seconds = round(float(segment.get("end_seconds", start_seconds) or start_seconds), 2)
        duration_seconds = round(max(0.8, end_seconds - start_seconds), 2)
        _emit_progress(
            progress_callback,
            f"Seedance 学习：[{index}/{len(segmentation_beats)}] 开始处理 {segment.get('beat_id', f'BS{index:02d}')} "
            f"{start_seconds:.2f}-{end_seconds:.2f}s",
        )
        segment_frame_samples = _sample_beat_frame_samples(
            beat=segment,
            source_video_path=source_video_path,
            beat_frame_root=beat_frame_root,
            settings=settings,
        )
        segment_frame_dir = str(segment.get("frame_sample_dir") or "")
        _emit_progress(
            progress_callback,
            f"Seedance 学习：[{index}/{len(segmentation_beats)}] {segment.get('beat_id', f'BS{index:02d}')} "
            f"{start_seconds:.2f}-{end_seconds:.2f}s｜已采样 {len(segment_frame_samples)} 帧"
            f"｜落盘 {segment.get('saved_frame_count', 0)} 张｜保存到 {segment_frame_dir}",
        )
        story_window = (
            _canonical_story_window_for_range(normalized_story_beats, start_seconds, end_seconds)
            if story_timing_reliable
            else []
        )
        narrative_window = list(story_window)
        if not narrative_window:
            narrative_window = [_fallback_story_beat_from_segment(segment=segment, beat_index=index)]
        purpose_text = _purpose_source_text(
            story_beats=narrative_window,
            transcript_segments=_transcript_segments_for_range(transcript, start_seconds, end_seconds),
            extra_parts=[segment.get("summary_text"), segment.get("dominant_purpose_hint")],
        )
        purpose_scores = purpose_score_breakdown(purpose_text, is_last=index == len(segmentation_beats))
        primary_purpose = str(segment.get("dominant_purpose_hint") or "")
        if primary_purpose not in PURPOSE_PROFILES:
            primary_purpose = purpose_scores[0]["purpose"] if purpose_scores else FALLBACK_PURPOSE
        dialogue_windows = _dialogue_windows_for_range(transcript, start_seconds, end_seconds)
        restored_duration_seconds = _derive_restored_duration_seconds(
            source_duration_seconds=duration_seconds,
            shot_count=int(segment.get("shot_count", 0) or 0),
            dialogue_window_count=len(dialogue_windows),
        )
        episode_context_window = _build_episode_context_window(
            segmentation_beats=segmentation_beats,
            current_index=index - 1,
        )
        scene_anchor = _scene_anchor_for_range(scenes, keyframes, start_seconds, end_seconds)
        shot_chain = _build_shot_chain_from_segment(
            segment=segment,
            story_beats=narrative_window,
            primary_purpose=primary_purpose,
            dialogue_windows=dialogue_windows,
            target_duration_seconds=restored_duration_seconds,
        )
        visual_second_pass_summary = ""
        visual_second_pass_result: dict[str, Any] = {}
        visual_second_pass_used = False
        if visual_second_pass_client and shot_chain:
            _emit_progress(
                progress_callback,
                f"Seedance 学习：[{index}/{len(segmentation_beats)}] {segment.get('beat_id', f'BS{index:02d}')} 正在做局部多模态复盘｜shots={len(shot_chain)}",
            )
            visual_second_pass = _run_visual_second_pass_for_segment(
                client=visual_second_pass_client,
                series_name=series_name,
                episode_id=episode_id,
                beat_id=str(segment.get("beat_id") or f"BS{index:02d}"),
                segment=segment,
                shot_chain=shot_chain,
                story_window=narrative_window,
                primary_purpose=primary_purpose,
                dialogue_windows=dialogue_windows,
                episode_context_window=episode_context_window,
                visual_second_pass_dir=visual_second_pass_dir,
                progress_callback=progress_callback,
            )
            if visual_second_pass:
                visual_second_pass_result = dict(visual_second_pass)
                shot_chain = _merge_visual_second_pass_into_shot_chain(
                    shot_chain=shot_chain,
                    second_pass_result=visual_second_pass,
                )
                visual_second_pass_summary = str(visual_second_pass.get("beat_observation") or "")
                visual_second_pass_used = bool(list(visual_second_pass.get("shots") or []))
                observed_purpose_text = _normalize_spaces(
                    " ".join(
                        [
                            str(visual_second_pass.get("primary_purpose_observed") or ""),
                            str(visual_second_pass.get("beat_title") or ""),
                            str(visual_second_pass.get("beat_summary") or ""),
                            str(visual_second_pass.get("scene_state") or ""),
                            str(visual_second_pass.get("subject_identity") or ""),
                            str(visual_second_pass.get("dialogue_summary") or ""),
                            str(visual_second_pass.get("timeflow_summary") or ""),
                            visual_second_pass_summary,
                        ]
                    )
                )
                observed_primary_purpose = infer_primary_purpose(observed_purpose_text)
                if observed_primary_purpose in PURPOSE_PROFILES:
                    primary_purpose = observed_primary_purpose
                _emit_progress(
                    progress_callback,
                    f"Seedance 学习：[{index}/{len(segmentation_beats)}] {segment.get('beat_id', f'BS{index:02d}')} 多模态复盘完成｜命中 {sum(1 for item in list(visual_second_pass.get('shots') or []) if str(item.get('shot_id') or '').strip())} 个 shot",
                )
            else:
                _emit_progress(
                    progress_callback,
                    f"Seedance 学习：[{index}/{len(segmentation_beats)}] {segment.get('beat_id', f'BS{index:02d}')} 多模态复盘未产出有效结果，继续使用本地规则",
                )
        purpose_profile = PURPOSE_PROFILES.get(primary_purpose, PURPOSE_PROFILES[FALLBACK_PURPOSE])
        presentation_window = list(narrative_window)
        observed_story = _story_item_from_observation(
            second_pass_result=visual_second_pass_result,
            transcript_text=str(segment.get("transcript_text") or ""),
            beat_index=index,
        )
        if observed_story:
            presentation_window = [observed_story]
        beat = {
            "beat_id": f"SB{index:02d}",
            "episode_id": episode_id,
            "time_range": f"{start_seconds:.2f}-{end_seconds:.2f}s",
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "duration_seconds": duration_seconds,
            "restored_duration_seconds": restored_duration_seconds,
            "primary_purpose": primary_purpose,
            "purpose_scores": purpose_scores[:4],
            "source_story_beat_ids": _dedupe_strings(list(segment.get("source_story_beat_ids") or [])),
            "source_titles": [
                str(item.get("title") or "").strip()
                for item in story_window
                if str(item.get("title") or "").strip()
            ],
            "beat_summary": _build_segment_summary(
                story_beats=presentation_window,
                transcript_text=str(segment.get("transcript_text") or ""),
                ocr_hints=[str(item.get("ocr_hint") or "") for item in list(segment.get("shot_timeline") or [])],
            )[:280],
            "display_title": _derive_catalog_title(primary_purpose, presentation_window),
            "display_summary": _derive_catalog_summary(presentation_window, str(segment.get("transcript_text") or "")),
            "dramatic_goal": _derive_dramatic_goal(primary_purpose, presentation_window),
            "scene_anchor": scene_anchor,
            "entry_state": _derive_entry_state(presentation_window[0], primary_purpose),
            "dialogue_windows": dialogue_windows,
            "episode_context_window": episode_context_window,
            "shot_chain": shot_chain,
            "frame_sample_count": int(segment.get("frame_sample_count", len(segment_frame_samples))),
            "saved_frame_count": int(segment.get("saved_frame_count", 0)),
            "frame_sample_dir": segment_frame_dir,
            "frame_samples": segment_frame_samples,
            "visual_second_pass_used": visual_second_pass_used,
            "visual_second_pass_summary": visual_second_pass_summary,
            "visual_second_pass_beat_title": str(visual_second_pass_result.get("beat_title") or ""),
            "visual_second_pass_beat_summary": str(visual_second_pass_result.get("beat_summary") or ""),
            "visual_second_pass_scene_state": str(visual_second_pass_result.get("scene_state") or ""),
            "visual_second_pass_subject_identity": str(visual_second_pass_result.get("subject_identity") or ""),
            "visual_second_pass_dialogue_summary": str(visual_second_pass_result.get("dialogue_summary") or ""),
            "visual_second_pass_timeflow_summary": str(visual_second_pass_result.get("timeflow_summary") or ""),
            "camera_language_notes": _collect_unique_lines(narrative_window, "camera_language", limit=6),
            "visual_focus_notes": _collect_unique_lines(narrative_window, "visual_focus", limit=6),
            "art_direction_notes": _collect_unique_lines(narrative_window, "art_direction_cues", limit=5),
            "storyboard_value_notes": _collect_unique_lines(narrative_window, "storyboard_value", limit=5),
            "action_design_notes": [purpose_profile.get("action_rules", [""])[0]],
            "dialogue_design_notes": [purpose_profile.get("dialogue_rules", [""])[0]],
            "blocking_notes": [_purpose_blocking_note(primary_purpose, scene_anchor)],
            "negative_patterns": list(purpose_profile.get("negative_patterns", []))[:2],
            "required_slots": list(purpose_profile.get("required_slots", [])),
            "when_to_use": str(purpose_profile.get("when_to_use", "")),
            "source_learning_excerpt": _strength_excerpt_for_purpose(primary_purpose, strength_playbook),
            "segmentation_context": {
                "source_segmentation_beat_id": str(segment.get("beat_id") or ""),
                "shot_count": int(segment.get("shot_count", 0) or 0),
                "end_boundary_score": float(segment.get("end_boundary_score", 0) or 0),
                "end_boundary_reasons": list(segment.get("end_boundary_reasons") or []),
            },
        }
        beat["quality_score"] = _quality_score(beat)
        beats.append(beat)
        _emit_progress(
            progress_callback,
            f"Seedance 学习：[{index}/{len(segmentation_beats)}] {beat['beat_id']} 理解完成｜目的={primary_purpose}｜shot_chain={len(shot_chain)}｜对白窗={len(dialogue_windows)}",
        )

    if beats:
        _emit_progress(progress_callback, "Seedance 学习：逐分镜理解完成，开始补全连续性与 prompt 回译。")
    for index, beat in enumerate(beats):
        previous_beat = beats[index - 1] if index > 0 else None
        next_beat = beats[index + 1] if index + 1 < len(beats) else None
        beat["continuity_bridge_in"] = _continuity_bridge_in(previous_beat, beat)
        beat["continuity_bridge_out"] = _continuity_bridge_out(beat, next_beat)
        beat["restored_seedance_prompt"] = _render_restored_prompt(beat, next_beat)
        beat["generalized_template_prompt"] = _render_generalized_template_prompt(beat, next_beat)

    return {
        "series_name": series_name,
        "episode_id": episode_id,
        "generated_at": utc_timestamp(),
        "taxonomy_version": str(settings.get("taxonomy_version", DEFAULT_TAXONOMY_VERSION)),
        "source_analysis_path": str(analysis_path),
        "episode_title": str(dict(analysis.get("episode") or {}).get("title") or episode_id),
        "beat_count": len(beats),
        "beats": beats,
    }


def _build_visual_second_pass_client(
    *,
    settings: Mapping[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    if not bool(settings.get("visual_second_pass_enabled", True)):
        return None
    provider_name = str(settings.get("visual_second_pass_provider") or "").strip().lower()
    model = str(settings.get("visual_second_pass_model") or "").strip()
    if not provider_name or not model:
        _emit_progress(progress_callback, "Seedance 学习：局部多模态复盘未启用 provider/model，回退本地规则。")
        return None

    api_key = str(settings.get("visual_second_pass_api_key") or "").strip()
    api_key_env = PROVIDER_API_ENV_MAP.get(provider_name, "")
    if api_key_env and api_key and not os.getenv(api_key_env, "").strip():
        os.environ[api_key_env] = api_key

    try:
        if provider_name == "openai":
            adapter = OpenAIAdapter(
                model=model,
                endpoint=str(settings.get("visual_second_pass_endpoint") or "").strip()
                or "https://api.openai.com/v1/responses",
                temperature=0.1,
                timeout_seconds=int(settings.get("visual_second_pass_timeout_seconds", DEFAULT_VISUAL_SECOND_PASS_TIMEOUT_SECONDS)),
                image_detail=str(settings.get("openai_image_detail") or "auto"),
                max_analysis_frames=int(settings.get("visual_second_pass_max_images", DEFAULT_VISUAL_SECOND_PASS_MAX_IMAGES)),
            )
        elif provider_name == "qwen":
            adapter = QwenAdapter(
                model=model,
                endpoint=str(settings.get("visual_second_pass_endpoint") or "").strip()
                or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                temperature=0.1,
                timeout_seconds=int(settings.get("visual_second_pass_timeout_seconds", DEFAULT_VISUAL_SECOND_PASS_TIMEOUT_SECONDS)),
                max_analysis_frames=int(settings.get("visual_second_pass_max_images", DEFAULT_VISUAL_SECOND_PASS_MAX_IMAGES)),
                video_fps=float(settings.get("qwen_video_fps", 1.0) or 1.0),
                structured_output_mode=str(settings.get("qwen_structured_output_mode") or "json_object"),
            )
        else:
            _emit_progress(progress_callback, f"Seedance 学习：局部多模态复盘暂不支持 provider={provider_name}，回退本地规则。")
            return None
        _emit_progress(progress_callback, f"Seedance 学习：已启用局部多模态复盘 provider={provider_name} model={model}")
        return {
            "provider_name": provider_name,
            "model": model,
            "adapter": adapter,
            "max_images": int(settings.get("visual_second_pass_max_images", DEFAULT_VISUAL_SECOND_PASS_MAX_IMAGES)),
            "context_frames": int(settings.get("visual_second_pass_context_frames", DEFAULT_VISUAL_SECOND_PASS_CONTEXT_FRAMES)),
            "timeout_seconds": int(settings.get("visual_second_pass_timeout_seconds", DEFAULT_VISUAL_SECOND_PASS_TIMEOUT_SECONDS)),
            "qwen_video_fps": float(settings.get("qwen_video_fps", 1.0) or 1.0),
            "qwen_structured_output_mode": str(settings.get("qwen_structured_output_mode") or "json_object"),
        }
    except Exception as exc:
        _emit_progress(progress_callback, f"Seedance 学习：局部多模态复盘初始化失败，回退本地规则：{exc}")
        return None


def _run_visual_second_pass_for_segment(
    *,
    client: Mapping[str, Any],
    series_name: str,
    episode_id: str,
    beat_id: str,
    segment: Mapping[str, Any],
    shot_chain: Sequence[Mapping[str, Any]],
    story_window: Sequence[Mapping[str, Any]],
    primary_purpose: str,
    dialogue_windows: Sequence[Mapping[str, Any]],
    episode_context_window: Sequence[Mapping[str, Any]] | None,
    visual_second_pass_dir: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    frames, frame_manifest = _collect_visual_second_pass_frames(
        segment=segment,
        max_images=int(client.get("max_images", DEFAULT_VISUAL_SECOND_PASS_MAX_IMAGES) or DEFAULT_VISUAL_SECOND_PASS_MAX_IMAGES),
        context_frames=int(client.get("context_frames", DEFAULT_VISUAL_SECOND_PASS_CONTEXT_FRAMES) or DEFAULT_VISUAL_SECOND_PASS_CONTEXT_FRAMES),
    )
    if not frames:
        return None
    _emit_progress(
        progress_callback,
        f"Seedance 学习：{beat_id} 局部多模态图包已准备｜images={len(frames)}｜目录={visual_second_pass_dir.resolve()}",
    )

    beat_context = {
        "series_name": series_name,
        "episode_id": episode_id,
        "beat_id": beat_id,
        "time_range": str(segment.get("time_range") or ""),
        "primary_purpose_hint": primary_purpose,
        "purpose_required_slots": list(PURPOSE_PROFILES.get(primary_purpose, PURPOSE_PROFILES[FALLBACK_PURPOSE]).get("required_slots") or []),
        "source_story_beats": [
            {
                "beat_id": str(item.get("beat_id") or ""),
                "title": str(item.get("title") or ""),
                "summary": str(item.get("summary") or ""),
                "visual_focus": _clean_list(item.get("visual_focus"))[:3],
                "camera_language": _clean_list(item.get("camera_language"))[:3],
            }
            for item in story_window
        ],
        "dialogue_windows": [
            {
                "time_range": str(item.get("time_range") or ""),
                "text": str(item.get("text") or ""),
            }
            for item in dialogue_windows
        ],
        "episode_context_window": [
            {
                "relation": str(item.get("relation") or ""),
                "beat_id": str(item.get("beat_id") or ""),
                "time_range": str(item.get("time_range") or ""),
                "purpose": str(item.get("purpose") or ""),
                "summary": str(item.get("summary") or ""),
                "dialogue_hint": str(item.get("dialogue_hint") or ""),
            }
            for item in list(episode_context_window or [])
        ],
        "local_shot_chain": [
            {
                "shot_id": str(item.get("shot_id") or ""),
                "time_range": str(item.get("time_range") or ""),
                "story_function": str(item.get("story_function") or ""),
                "visual_focus": str(item.get("visual_focus") or ""),
                "camera_language": str(item.get("camera_language") or ""),
                "camera_entry": str(item.get("camera_entry") or ""),
                "subject_blocking": str(item.get("subject_blocking") or ""),
                "action_timeline": str(item.get("action_timeline") or ""),
                "lighting_and_texture": str(item.get("lighting_and_texture") or ""),
                "background_continuity": str(item.get("background_continuity") or ""),
                "dialogue_timing": str(item.get("dialogue_timing") or ""),
                "sound_bed": str(item.get("sound_bed") or ""),
                "dialogue_hint": str(item.get("dialogue_hint") or ""),
                "art_direction_hint": str(item.get("art_direction_hint") or ""),
                "transition_trigger": str(item.get("transition_trigger") or ""),
            }
            for item in shot_chain
        ],
        "segment_summary": str(segment.get("summary_text") or ""),
        "transcript_text": str(segment.get("transcript_text") or ""),
        "scene_anchor_summary": str(segment.get("scene_anchor_summary") or ""),
    }
    user_prompt = render_prompt(
        "seedance_learning/second_pass_user.md",
        {
            "beat_context_json": json.dumps(beat_context, ensure_ascii=False, indent=2),
            "frame_manifest_json": json.dumps(frame_manifest, ensure_ascii=False, indent=2),
        },
    )
    system_prompt = load_prompt("seedance_learning/second_pass_system.md")

    request_path = visual_second_pass_dir / f"{beat_id}__request.json"
    prompt_path = visual_second_pass_dir / f"{beat_id}__prompt.md"
    response_path = visual_second_pass_dir / f"{beat_id}__response.json"
    save_json_file(
        request_path,
        {
            "provider": str(client.get("provider_name") or ""),
            "model": str(client.get("model") or ""),
            "beat_id": beat_id,
            "frame_manifest": frame_manifest,
            "beat_context": beat_context,
        },
    )
    save_text_file(
        prompt_path,
        "\n".join(
            [
                "# Second Pass Prompt",
                "",
                "## System",
                "",
                system_prompt,
                "",
                "## User",
                "",
                user_prompt,
                "",
            ]
        ),
    )

    try:
        raw_result = _call_visual_second_pass_provider(
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            frames=frames,
        )
    except Exception as exc:
        save_json_file(
            response_path,
            {
                "beat_id": beat_id,
                "status": "failed",
                "error": str(exc),
            },
        )
        return None

    normalized = _normalize_visual_second_pass_result(raw_result, shot_chain=shot_chain)
    save_json_file(response_path, normalized)
    return normalized


def _collect_visual_second_pass_frames(
    *,
    segment: Mapping[str, Any],
    max_images: int,
    context_frames: int,
) -> tuple[list[FrameReference], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    frames: list[FrameReference] = []
    seen_paths: set[str] = set()
    shot_timeline = list(segment.get("shot_timeline") or [])
    beat_samples = list(segment.get("frame_samples") or [])
    context_limit = max(1, min(max_images, max(0, context_frames)))
    shot_limit = max(1, max_images - context_limit)

    for item in _sample_evenly_spaced_items(beat_samples, context_limit):
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        resolved = str(Path(path).expanduser().resolve())
        if resolved in seen_paths or not Path(resolved).exists():
            continue
        seen_paths.add(resolved)
        note = _normalize_spaces(
            f"beat主帧 {item.get('frame_id', '')} {item.get('timestamp_seconds', '')}s"
        )
        frames.append(
            FrameReference(
                path=resolved,
                timestamp=f"{float(item.get('timestamp_seconds', 0) or 0):.2f}s",
                note=note[:120],
            )
        )
        manifest.append(
            {
                "frame_role": "beat_primary",
                "frame_id": str(item.get("frame_id") or ""),
                "timestamp_seconds": float(item.get("timestamp_seconds", 0) or 0),
                "path": resolved,
                "note": note[:120],
            }
        )

    remaining = max(0, max_images - len(frames))
    if remaining > 0:
        for shot in _sample_evenly_spaced_items(shot_timeline, min(shot_limit, remaining)):
            path = str(shot.get("keyframe_path") or "").strip()
            if not path:
                continue
            resolved = str(Path(path).expanduser().resolve())
            if resolved in seen_paths or not Path(resolved).exists():
                continue
            seen_paths.add(resolved)
            note = _normalize_spaces(
                f"shot锚点 本地{shot.get('shot_id', '')} 源{shot.get('source_scene_id', '')} "
                f"{shot.get('time_range', '')} {str(shot.get('ocr_hint') or '').strip()}"
            )
            frames.append(
                FrameReference(
                    path=resolved,
                    timestamp=str(shot.get("time_range") or ""),
                    note=note[:120],
                )
            )
            manifest.append(
                {
                    "frame_role": "shot_anchor",
                    "shot_id": str(shot.get("shot_id") or ""),
                    "source_scene_id": str(shot.get("source_scene_id") or ""),
                    "time_range": str(shot.get("time_range") or ""),
                    "path": resolved,
                    "note": note[:120],
                }
            )
    return frames, manifest


def _sample_evenly_spaced_items(items: Sequence[Any], limit: int) -> list[Any]:
    sequence = list(items or [])
    if limit <= 0:
        return []
    if len(sequence) <= limit:
        return sequence
    if limit == 1:
        return [sequence[0]]
    last_index = len(sequence) - 1
    selected_indices = sorted(
        {
            round(position * last_index / max(limit - 1, 1))
            for position in range(limit)
        }
    )
    return [sequence[index] for index in selected_indices]


def _call_visual_second_pass_provider(
    *,
    client: Mapping[str, Any],
    system_prompt: str,
    user_prompt: str,
    frames: Sequence[FrameReference],
) -> dict[str, Any]:
    provider_name = str(client.get("provider_name") or "")
    adapter = client.get("adapter")
    if provider_name == "openai" and isinstance(adapter, OpenAIAdapter):
        return _call_openai_visual_second_pass(adapter, system_prompt=system_prompt, user_prompt=user_prompt, frames=frames)
    if provider_name == "qwen" and isinstance(adapter, QwenAdapter):
        return _call_qwen_visual_second_pass(
            adapter,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            frames=frames,
            structured_output_mode=str(client.get("qwen_structured_output_mode") or "json_object"),
            video_fps=float(client.get("qwen_video_fps", 1.0) or 1.0),
        )
    raise ProviderError(f"不支持的局部多模态复盘 provider：{provider_name}")


def _call_openai_visual_second_pass(
    adapter: OpenAIAdapter,
    *,
    system_prompt: str,
    user_prompt: str,
    frames: Sequence[FrameReference],
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
    for index, frame in enumerate(frames, start=1):
        descriptor_parts = [part for part in [frame.timestamp, frame.note] if part]
        descriptor = f"参考图 {index}"
        if descriptor_parts:
            descriptor += f"：{'; '.join(descriptor_parts)}"
        content.append({"type": "input_text", "text": descriptor})
        content.append(
            {
                "type": "input_image",
                "image_url": file_to_data_url(frame.resolved_path(), frame.detected_mime_type()),
                "detail": adapter.image_detail,
            }
        )
    payload = {
        "model": adapter.config.model,
        "temperature": 0.1,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "seedance_visual_second_pass",
                "schema": SECOND_PASS_VISUAL_SCHEMA,
                "strict": True,
            }
        },
    }
    response = adapter.request_json(
        adapter.config.endpoint,
        headers={"Authorization": f"Bearer {adapter.require_api_key()}"},
        payload=payload,
    )
    output_text = adapter._extract_output_text(response)
    result = extract_json_from_text(output_text)
    validate_against_schema(result, SECOND_PASS_VISUAL_SCHEMA)
    return result


def _call_qwen_visual_second_pass(
    adapter: QwenAdapter,
    *,
    system_prompt: str,
    user_prompt: str,
    frames: Sequence[FrameReference],
    structured_output_mode: str,
    video_fps: float,
) -> dict[str, Any]:
    content = [
        {
            "type": "video",
            "video": [file_to_data_url(frame.resolved_path(), frame.detected_mime_type()) for frame in frames],
            "fps": max(0.5, min(video_fps, 2.0)),
        },
        {
            "type": "text",
            "text": user_prompt,
        },
    ]
    payload = {
        "model": adapter.config.model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "response_format": adapter._build_analysis_response_format(
            SECOND_PASS_VISUAL_SCHEMA,
            structured_output_mode if structured_output_mode in {"json_schema", "json_object"} else "json_object",
        ),
    }
    response = adapter.request_json(
        adapter.config.endpoint,
        headers={"Authorization": f"Bearer {adapter.require_api_key()}"},
        payload=payload,
    )
    output_text = adapter._extract_output_text(response)
    result = extract_json_from_text(output_text)
    validate_against_schema(result, SECOND_PASS_VISUAL_SCHEMA)
    return result


def _normalize_visual_second_pass_result(
    result: Mapping[str, Any],
    *,
    shot_chain: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    shot_id_alias_map: dict[str, str] = {}
    for index, shot in enumerate(shot_chain, start=1):
        local_shot_id = str(shot.get("shot_id") or "").strip()
        source_scene_id = str(shot.get("source_scene_id") or "").strip()
        if local_shot_id:
            shot_id_alias_map[local_shot_id] = local_shot_id
            shot_id_alias_map[local_shot_id.lower()] = local_shot_id
            shot_id_alias_map[f"s{index:02d}"] = local_shot_id
            shot_id_alias_map[f"s{index}"] = local_shot_id
        if source_scene_id and local_shot_id:
            shot_id_alias_map[source_scene_id] = local_shot_id
            shot_id_alias_map[source_scene_id.lower()] = local_shot_id
    normalized_shots: list[dict[str, str]] = []
    fallback_ids = [str(item.get("shot_id") or "") for item in shot_chain]
    for index, item in enumerate(list(result.get("shots") or []), start=1):
        if not isinstance(item, Mapping):
            continue
        shot_id = str(item.get("shot_id") or "").strip()
        if shot_id:
            shot_id = shot_id_alias_map.get(shot_id, shot_id_alias_map.get(shot_id.lower(), shot_id))
        if not shot_id and index - 1 < len(fallback_ids):
            shot_id = fallback_ids[index - 1]
        if not shot_id:
            continue
        normalized_shots.append(
            {
                "shot_id": shot_id,
                "story_function": _normalize_spaces(str(item.get("story_function") or "")),
                "visual_focus": _normalize_spaces(str(item.get("visual_focus") or "")),
                "camera_language": _normalize_spaces(str(item.get("camera_language") or "")),
                "camera_entry": _normalize_spaces(str(item.get("camera_entry") or "")),
                "subject_blocking": _normalize_spaces(str(item.get("subject_blocking") or "")),
                "action_timeline": _strip_source_second_markers(str(item.get("action_timeline") or "")),
                "lighting_and_texture": _normalize_spaces(str(item.get("lighting_and_texture") or "")),
                "background_continuity": _normalize_spaces(str(item.get("background_continuity") or "")),
                "dialogue_timing": _normalize_spaces(str(item.get("dialogue_timing") or "")),
                "sound_bed": _normalize_spaces(str(item.get("sound_bed") or "")),
                "art_direction_hint": _normalize_spaces(str(item.get("art_direction_hint") or "")),
                "transition_trigger": _normalize_spaces(str(item.get("transition_trigger") or "")),
            }
        )
    normalized = {
        "beat_title": _normalize_spaces(str(result.get("beat_title") or "")),
        "beat_summary": _normalize_spaces(str(result.get("beat_summary") or "")),
        "scene_state": _normalize_spaces(str(result.get("scene_state") or "")),
        "subject_identity": _normalize_spaces(str(result.get("subject_identity") or "")),
        "dialogue_summary": _normalize_spaces(str(result.get("dialogue_summary") or "")),
        "timeflow_summary": _normalize_spaces(str(result.get("timeflow_summary") or "")),
        "beat_observation": _normalize_spaces(str(result.get("beat_observation") or "")),
        "primary_purpose_observed": _normalize_spaces(str(result.get("primary_purpose_observed") or "")),
        "shots": normalized_shots,
    }
    validate_against_schema(normalized, SECOND_PASS_VISUAL_SCHEMA)
    return normalized


def _merge_visual_second_pass_into_shot_chain(
    *,
    shot_chain: Sequence[Mapping[str, Any]],
    second_pass_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    overrides = [dict(item) for item in list(second_pass_result.get("shots") or []) if isinstance(item, Mapping)]
    by_id = {
        str(item.get("shot_id") or ""): item
        for item in overrides
        if str(item.get("shot_id") or "").strip()
    }
    merged: list[dict[str, Any]] = []
    used_override_ids: set[str] = set()
    fallback_overrides = list(overrides)
    for shot in shot_chain:
        current = dict(shot)
        override = dict(by_id.get(str(current.get("shot_id") or ""), {}))
        if not override:
            source_scene_id = str(current.get("source_scene_id") or "").strip()
            if source_scene_id:
                override = dict(by_id.get(source_scene_id, {}))
        if not override:
            for candidate in fallback_overrides:
                candidate_id = str(candidate.get("shot_id") or "").strip()
                if candidate_id and candidate_id in used_override_ids:
                    continue
                override = dict(candidate)
                break
        if override:
            override_id = str(override.get("shot_id") or "").strip()
            if override_id:
                used_override_ids.add(override_id)
            for key in [
                "story_function",
                "visual_focus",
                "camera_language",
                "camera_entry",
                "subject_blocking",
                "action_timeline",
                "lighting_and_texture",
                "background_continuity",
                "dialogue_timing",
                "sound_bed",
                "art_direction_hint",
                "transition_trigger",
            ]:
                value = _normalize_spaces(str(override.get(key) or ""))
                if value:
                    current[key] = value
        merged.append(current)
    return merged


def _build_skill_library(
    *,
    series_name: str,
    catalogs: Sequence[Mapping[str, Any]],
    strength_playbook: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    all_beats = [beat for catalog in catalogs for beat in list(catalog.get("beats") or [])]
    beats_by_purpose: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for beat in all_beats:
        beats_by_purpose[str(beat.get("primary_purpose") or FALLBACK_PURPOSE)].append(beat)

    purposes: list[dict[str, Any]] = []
    for purpose in PURPOSE_ORDER:
        beats = beats_by_purpose.get(purpose, [])
        if not beats:
            continue
        profile = PURPOSE_PROFILES.get(purpose, PURPOSE_PROFILES[FALLBACK_PURPOSE])
        durations = [float(item.get("duration_seconds", 0) or 0) for item in beats]
        design_skill = {
            "narrative_goals": _rank_lines(
                [str(item.get("dramatic_goal") or "") for item in beats],
                limit=int(settings.get("max_rules_per_purpose", DEFAULT_MAX_RULES_PER_PURPOSE)),
            ),
            "camera_rules": _rank_lines(
                [
                    *profile.get("camera_rules", []),
                    *[
                        note
                        for beat in beats
                        for note in list(beat.get("camera_language_notes") or [])
                    ],
                ],
                limit=int(settings.get("max_rules_per_purpose", DEFAULT_MAX_RULES_PER_PURPOSE)),
            ),
            "beat_rules": _rank_lines(
                [
                    *profile.get("beat_rules", []),
                    *[
                        f"{item.get('time_range', '')}｜{'、'.join(item.get('source_titles', [])[:2])}"
                        for item in beats
                    ],
                ],
                limit=int(settings.get("max_rules_per_purpose", DEFAULT_MAX_RULES_PER_PURPOSE)),
            ),
            "action_rules": _rank_lines(
                [
                    *profile.get("action_rules", []),
                    *[
                        note
                        for beat in beats
                        for note in list(beat.get("visual_focus_notes") or [])
                    ],
                ],
                limit=int(settings.get("max_rules_per_purpose", DEFAULT_MAX_RULES_PER_PURPOSE)),
            ),
            "dialogue_rules": _rank_lines(
                [
                    *profile.get("dialogue_rules", []),
                    *[
                        note
                        for beat in beats
                        for note in list(beat.get("dialogue_design_notes") or [])
                    ],
                ],
                limit=int(settings.get("max_rules_per_purpose", DEFAULT_MAX_RULES_PER_PURPOSE)),
            ),
            "continuity_rules": _rank_lines(
                [
                    *profile.get("continuity_rules", []),
                    *[str(item.get("continuity_bridge_in") or "") for item in beats],
                    *[str(item.get("continuity_bridge_out") or "") for item in beats],
                ],
                limit=int(settings.get("max_rules_per_purpose", DEFAULT_MAX_RULES_PER_PURPOSE)),
            ),
            "negative_patterns": _rank_lines(
                [
                    *profile.get("negative_patterns", []),
                    *[
                        note
                        for beat in beats
                        for note in list(beat.get("negative_patterns") or [])
                    ],
                ],
                limit=max(4, int(settings.get("max_rules_per_purpose", DEFAULT_MAX_RULES_PER_PURPOSE)) - 2),
            ),
        }
        exemplar_templates = sorted(
            beats,
            key=lambda item: (-float(item.get("quality_score", 0) or 0), str(item.get("beat_id") or "")),
        )[:3]
        purposes.append(
            {
                "purpose": purpose,
                "description": profile.get("description", ""),
                "when_to_use": profile.get("when_to_use", ""),
                "beat_count": len(beats),
                "episode_count": len({str(item.get("episode_id") or "") for item in beats}),
                "duration_profile": {
                    "min_seconds": round(min(durations), 2) if durations else 0,
                    "max_seconds": round(max(durations), 2) if durations else 0,
                    "avg_seconds": round(sum(durations) / len(durations), 2) if durations else 0,
                },
                "design_skill": design_skill,
                "top_template_ids": [
                    _template_id(purpose, str(item.get("episode_id") or ""), str(item.get("beat_id") or ""))
                    for item in exemplar_templates
                ],
            }
        )
    return {
        "series_name": series_name,
        "generated_at": utc_timestamp(),
        "taxonomy_version": str(settings.get("taxonomy_version", DEFAULT_TAXONOMY_VERSION)),
        "episode_count": len(catalogs),
        "beat_count": len(all_beats),
        "shared_series_rules": {
            "camera_language_rules": list(strength_playbook.get("camera_language_rules") or [])[:8],
            "storyboard_execution_rules": list(strength_playbook.get("storyboard_execution_rules") or [])[:8],
            "dialogue_timing_rules": list(strength_playbook.get("dialogue_timing_rules") or [])[:8],
            "continuity_guardrails": list(strength_playbook.get("continuity_guardrails") or [])[:8],
            "negative_patterns": list(strength_playbook.get("negative_patterns") or [])[:8],
        },
        "purposes": purposes,
    }


def _build_template_library(
    *,
    series_name: str,
    catalogs: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    all_beats = [beat for catalog in catalogs for beat in list(catalog.get("beats") or [])]
    beats_by_purpose: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for beat in all_beats:
        beats_by_purpose[str(beat.get("primary_purpose") or FALLBACK_PURPOSE)].append(beat)

    purposes: list[dict[str, Any]] = []
    template_count = 0
    per_purpose_limit = int(settings.get("max_templates_per_purpose", DEFAULT_MAX_TEMPLATES_PER_PURPOSE))
    per_series_limit = int(settings.get("max_templates_per_series", DEFAULT_MAX_TEMPLATES_PER_SERIES))
    for purpose in PURPOSE_ORDER:
        beats = beats_by_purpose.get(purpose, [])
        if not beats:
            continue
        selected_beats = sorted(
            beats,
            key=lambda item: (
                -float(item.get("quality_score", 0) or 0),
                -float(item.get("duration_seconds", 0) or 0),
                str(item.get("episode_id") or ""),
                str(item.get("beat_id") or ""),
            ),
        )[:per_purpose_limit]
        templates = []
        for beat in selected_beats:
            if template_count >= per_series_limit:
                break
            retrieval_title = _derive_template_retrieval_title(beat)
            retrieval_summary = _derive_template_retrieval_summary(beat)
            classification_metadata = _derive_template_classification_metadata(
                beat,
                series_name=series_name,
            )
            search_metadata = _derive_template_search_metadata(
                beat,
                series_name=series_name,
                classification_metadata=classification_metadata,
            )
            templates.append(
                {
                    "template_id": _template_id(
                        purpose,
                        str(beat.get("episode_id") or ""),
                        str(beat.get("beat_id") or ""),
                    ),
                    "purpose": purpose,
                    "source_series_name": series_name,
                    "source_episode_id": str(beat.get("episode_id") or ""),
                    "source_beat_id": str(beat.get("beat_id") or ""),
                    "duration_seconds": float(beat.get("duration_seconds", 0) or 0),
                    "time_range": str(beat.get("time_range") or ""),
                    "quality_score": float(beat.get("quality_score", 0) or 0),
                    "when_to_use": str(beat.get("when_to_use") or ""),
                    "required_slots": list(beat.get("required_slots") or []),
                    "source_story_beat_ids": list(beat.get("source_story_beat_ids") or []),
                    "source_titles": list(beat.get("source_titles") or []),
                    "display_title": str(beat.get("display_title") or ""),
                    "display_summary": str(beat.get("display_summary") or beat.get("beat_summary") or ""),
                    "dramatic_goal": str(beat.get("dramatic_goal") or ""),
                    "scene_anchor_summary": str(dict(beat.get("scene_anchor") or {}).get("summary") or ""),
                    "primary_purpose": str(classification_metadata.get("primary_purpose") or purpose),
                    "secondary_purposes": list(classification_metadata.get("secondary_purposes") or []),
                    "purpose_breakdown": list(classification_metadata.get("purpose_breakdown") or []),
                    "classification_confidence": float(classification_metadata.get("classification_confidence") or 0.0),
                    "ambiguity_note": str(classification_metadata.get("ambiguity_note") or ""),
                    "scene_tags": list(classification_metadata.get("scene_tags") or []),
                    "relation_tags": list(classification_metadata.get("relation_tags") or []),
                    "staging_tags": list(classification_metadata.get("staging_tags") or []),
                    "camera_tags": list(classification_metadata.get("camera_tags") or []),
                    "emotion_tags": list(classification_metadata.get("emotion_tags") or []),
                    "narrative_tags": list(classification_metadata.get("narrative_tags") or []),
                    "retrieval_title": retrieval_title,
                    "retrieval_summary": retrieval_summary,
                    "search_hint": str(search_metadata.get("search_hint") or ""),
                    "search_keywords": list(search_metadata.get("search_keywords") or []),
                    "search_text": str(search_metadata.get("search_text") or ""),
                    "shot_outline": [
                        f"{item.get('time_range', '')}｜{item.get('story_function', '')}"
                        for item in list(beat.get("shot_chain") or [])
                    ],
                    "restored_seedance_prompt": str(beat.get("restored_seedance_prompt") or ""),
                    "generalized_template_prompt": str(beat.get("generalized_template_prompt") or ""),
                }
            )
            template_count += 1
        purposes.append(
            {
                "purpose": purpose,
                "template_count": len(templates),
                "templates": templates,
            }
        )
        if template_count >= per_series_limit:
            break

    return {
        "series_name": series_name,
        "generated_at": utc_timestamp(),
        "taxonomy_version": str(settings.get("taxonomy_version", DEFAULT_TAXONOMY_VERSION)),
        "template_count": template_count,
        "prompt_library_root": "",
        "purposes": purposes,
    }


def _normalize_story_beats(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    raw_story_beats = list(analysis.get("story_beats") or [])
    for index, raw in enumerate(raw_story_beats, start=1):
        if not isinstance(raw, Mapping):
            continue
        start_seconds, end_seconds, timing_source = _story_beat_seconds(raw, fallback_index=index)
        text_blob = " ".join(
            [
                str(raw.get("title") or ""),
                str(raw.get("summary") or ""),
                " ".join(_clean_list(raw.get("visual_focus"))),
                " ".join(_clean_list(raw.get("camera_language"))),
                " ".join(_clean_list(raw.get("storyboard_value"))),
            ]
        )
        normalized.append(
            {
                "beat_id": str(raw.get("beat_id") or f"B{index:02d}"),
                "title": str(raw.get("title") or "").strip(),
                "summary": str(raw.get("summary") or "").strip(),
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": round(max(0.6, end_seconds - start_seconds), 2),
                "timing_source": timing_source,
                "visual_focus": _clean_list(raw.get("visual_focus")),
                "camera_language": _clean_list(raw.get("camera_language")),
                "art_direction_cues": _clean_list(raw.get("art_direction_cues")),
                "storyboard_value": _clean_list(raw.get("storyboard_value")),
                "text_blob": _normalize_spaces(text_blob),
            }
        )
    normalized.sort(key=lambda item: (item["start_seconds"], item["end_seconds"]))
    return normalized


def _story_beat_timing_profile(
    story_beats: Sequence[Mapping[str, Any]],
    *,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    beats = [dict(item) for item in story_beats if isinstance(item, Mapping)]
    if not beats:
        return {
            "beat_count": 0,
            "explicit_ratio": 0.0,
            "micro_ratio": 0.0,
            "median_duration": 0.0,
            "reliable_for_boundaries": False,
            "reliable_for_prompt_grounding": False,
            "reason": "no_story_beats",
        }

    min_duration = float(settings.get("min_beat_duration", DEFAULT_MIN_BEAT_DURATION))
    durations = sorted(max(0.6, float(item.get("duration_seconds", 0) or 0)) for item in beats)
    midpoint = len(durations) // 2
    if len(durations) % 2 == 0:
        median_duration = round((durations[midpoint - 1] + durations[midpoint]) / 2.0, 2)
    else:
        median_duration = round(durations[midpoint], 2)
    explicit_count = sum(1 for item in beats if str(item.get("timing_source") or "") != "fallback_uniform")
    explicit_ratio = round(explicit_count / max(1, len(beats)), 2)
    micro_cutoff = max(4.5, min_duration * 0.75)
    micro_ratio = round(
        sum(1 for duration in durations if duration < micro_cutoff) / max(1, len(durations)),
        2,
    )
    reliable_for_boundaries = explicit_ratio >= 0.5 and micro_ratio <= 0.45 and median_duration >= micro_cutoff
    reliable_for_prompt_grounding = explicit_ratio >= 0.5 and micro_ratio <= 0.35 and median_duration >= max(
        6.0, min_duration * 0.8
    )
    if not explicit_count:
        reason = "fallback_uniform_timing"
    elif micro_ratio > 0.45:
        reason = "micro_story_beats"
    elif median_duration < micro_cutoff:
        reason = "median_duration_too_short"
    else:
        reason = "usable"
    return {
        "beat_count": len(beats),
        "explicit_ratio": explicit_ratio,
        "micro_ratio": micro_ratio,
        "median_duration": median_duration,
        "reliable_for_boundaries": reliable_for_boundaries,
        "reliable_for_prompt_grounding": reliable_for_prompt_grounding,
        "reason": reason,
    }


def _story_beats_for_range(
    story_beats: Sequence[Mapping[str, Any]],
    start_seconds: float,
    end_seconds: float,
) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    for item in story_beats:
        beat_start = float(item.get("start_seconds", 0) or 0)
        beat_end = float(item.get("end_seconds", beat_start) or beat_start)
        if beat_end < start_seconds or beat_start > end_seconds:
            continue
        overlaps.append(dict(item))
    return overlaps


def _canonical_story_beat_for_range(
    story_beats: Sequence[Mapping[str, Any]],
    start_seconds: float,
    end_seconds: float,
) -> dict[str, Any]:
    best = _best_story_beat_for_range(story_beats, start_seconds, end_seconds)
    if best:
        return dict(best)
    midpoint = (float(start_seconds) + float(end_seconds)) / 2.0
    nearest: Mapping[str, Any] | None = None
    nearest_distance = float("inf")
    for item in story_beats:
        beat_start = float(item.get("start_seconds", 0) or 0)
        beat_end = float(item.get("end_seconds", beat_start) or beat_start)
        beat_midpoint = (beat_start + beat_end) / 2.0
        distance = min(abs(midpoint - beat_midpoint), abs(start_seconds - beat_end), abs(end_seconds - beat_start))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest = item
    return dict(nearest or {})


def _canonical_story_window_for_range(
    story_beats: Sequence[Mapping[str, Any]],
    start_seconds: float,
    end_seconds: float,
) -> list[dict[str, Any]]:
    canonical = _canonical_story_beat_for_range(story_beats, start_seconds, end_seconds)
    if canonical:
        return [canonical]
    return []


def _significant_story_beats_for_range(
    story_beats: Sequence[Mapping[str, Any]],
    start_seconds: float,
    end_seconds: float,
) -> list[dict[str, Any]]:
    segment_duration = max(0.2, end_seconds - start_seconds)
    overlaps = []
    for item in _story_beats_for_range(story_beats, start_seconds, end_seconds):
        beat_start = float(item.get("start_seconds", 0) or 0)
        beat_end = float(item.get("end_seconds", beat_start) or beat_start)
        overlap = max(0.0, min(end_seconds, beat_end) - max(start_seconds, beat_start))
        beat_duration = max(0.2, beat_end - beat_start)
        if overlap >= 1.2 or overlap / beat_duration >= 0.35 or overlap / segment_duration >= 0.3:
            overlaps.append(dict(item))
    return overlaps or _story_beats_for_range(story_beats, start_seconds, end_seconds)


def _transcript_segments_for_range(
    transcript_segments: Sequence[Mapping[str, Any]],
    start_seconds: float,
    end_seconds: float,
) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    for item in transcript_segments:
        start = float(item.get("start", 0) or 0)
        end = float(item.get("end", start) or start)
        if end < start_seconds or start > end_seconds:
            continue
        overlaps.append(dict(item))
    return overlaps


def _ocr_hint_for_range(
    ocr_segments: Sequence[Mapping[str, Any]],
    start_seconds: float,
    end_seconds: float,
) -> str:
    texts = []
    for item in ocr_segments:
        timestamp = float(item.get("timestamp", 0) or 0)
        if timestamp < start_seconds or timestamp > end_seconds:
            continue
        text = _normalize_spaces(str(item.get("text") or ""))
        if text:
            texts.append(text)
    return _combine_unique_texts(texts)[:180]


def _purpose_source_text(
    *,
    story_beats: Sequence[Mapping[str, Any]],
    transcript_segments: Sequence[Mapping[str, Any]],
    extra_parts: Sequence[Any] | None = None,
) -> str:
    parts: list[str] = []
    for item in story_beats:
        parts.extend(
            [
                str(item.get("title") or ""),
                str(item.get("summary") or ""),
                " ".join(_clean_list(item.get("visual_focus"))),
                " ".join(_clean_list(item.get("camera_language"))),
                " ".join(_clean_list(item.get("storyboard_value"))),
            ]
        )
    parts.extend(str(item.get("text") or "") for item in transcript_segments)
    parts.extend(str(part or "") for part in list(extra_parts or []))
    return _normalize_spaces(" ".join(part for part in parts if _normalize_spaces(part)))


def _build_segment_summary(
    *,
    story_beats: Sequence[Mapping[str, Any]],
    transcript_text: str,
    ocr_hints: Sequence[str],
) -> str:
    story_summary = _combine_unique_texts(
        [item.get("summary") for item in story_beats] + [item.get("title") for item in story_beats]
    )
    ocr_summary = _combine_unique_texts(ocr_hints)
    if story_summary:
        return story_summary[:280]
    if transcript_text:
        return transcript_text[:280]
    if ocr_summary:
        return ("OCR 锚点：" + ocr_summary)[:280]
    return "当前 beat 以镜头推进和动作连续性为主。"


def _build_episode_context_window(
    *,
    segmentation_beats: Sequence[Mapping[str, Any]],
    current_index: int,
) -> list[dict[str, Any]]:
    context_items: list[dict[str, Any]] = []
    for relation, neighbor_index in (("prev", current_index - 1), ("next", current_index + 1)):
        if neighbor_index < 0 or neighbor_index >= len(segmentation_beats):
            continue
        item = dict(segmentation_beats[neighbor_index] or {})
        transcript_text = _normalize_spaces(str(item.get("transcript_text") or ""))
        dialogue_windows = list(item.get("dialogue_windows") or [])
        dialogue_hint = _combine_unique_texts(
            str(window.get("text") or "")
            for window in dialogue_windows[:2]
            if str(window.get("text") or "").strip()
        )
        summary = _combine_unique_texts(
            [
                str(item.get("summary_text") or ""),
                " / ".join(str(title).strip() for title in list(item.get("source_story_titles") or [])[:2] if str(title).strip()),
                transcript_text[:64],
            ]
        )
        context_items.append(
            {
                "relation": relation,
                "beat_id": str(item.get("beat_id") or ""),
                "time_range": str(item.get("time_range") or ""),
                "purpose": str(item.get("dominant_purpose_hint") or ""),
                "summary": summary[:96],
                "dialogue_hint": dialogue_hint[:48],
            }
        )
    return context_items


def _fallback_story_beat_from_segment(
    *,
    segment: Mapping[str, Any],
    beat_index: int,
) -> dict[str, Any]:
    transcript_text = str(segment.get("transcript_text") or "").strip()
    summary_text = str(segment.get("summary_text") or "").strip()
    visual_focus_candidates = _split_freeform_story_chunks(summary_text or transcript_text)
    return {
        "beat_id": f"SEG{beat_index:02d}",
        "title": (visual_focus_candidates[0][:24] if visual_focus_candidates else transcript_text[:24]) or f"分镜 {beat_index}",
        "summary": summary_text or transcript_text or "以当前分镜动作链作为主要信息源。",
        "start_seconds": round(float(segment.get("start_seconds", 0) or 0), 2),
        "end_seconds": round(float(segment.get("end_seconds", 0) or 0), 2),
        "duration_seconds": round(float(segment.get("duration_seconds", 0) or 0), 2),
        "visual_focus": visual_focus_candidates[:6],
        "camera_language": [],
        "art_direction_cues": [],
        "storyboard_value": _split_freeform_story_chunks(transcript_text or summary_text)[:4],
        "text_blob": _normalize_spaces(f"{transcript_text} {summary_text}"),
    }


def _story_item_from_observation(
    *,
    second_pass_result: Mapping[str, Any],
    transcript_text: str,
    beat_index: int,
) -> dict[str, Any]:
    beat_title = _normalize_spaces(str(second_pass_result.get("beat_title") or ""))
    beat_summary = _normalize_spaces(str(second_pass_result.get("beat_summary") or ""))
    scene_state = _normalize_spaces(str(second_pass_result.get("scene_state") or ""))
    subject_identity = _normalize_spaces(str(second_pass_result.get("subject_identity") or ""))
    dialogue_summary = _normalize_spaces(str(second_pass_result.get("dialogue_summary") or ""))
    timeflow_summary = _normalize_spaces(str(second_pass_result.get("timeflow_summary") or ""))
    beat_observation = _normalize_spaces(str(second_pass_result.get("beat_observation") or ""))
    observation_parts = [
        beat_summary,
        scene_state,
        subject_identity,
        dialogue_summary,
        timeflow_summary,
        beat_observation,
    ]
    observation = _normalize_spaces("；".join(part for part in observation_parts if part))
    if not observation and not beat_title:
        return {}
    cleaned_observation = re.sub(
        r"^(?:该\s*beat\s*实际呈现为|该beat实际呈现为|实际画面为|该段实际呈现为|这一段实际呈现为)",
        "",
        observation,
        flags=re.IGNORECASE,
    ).strip("：:，。； ")
    title_source = beat_title or re.split(r"[：:，。；]", cleaned_observation or observation, maxsplit=1)[0].strip()
    title = _normalize_spaces(title_source)[:24] or f"观察分镜{beat_index:02d}"
    storyboard_value = _split_freeform_story_chunks(_normalize_spaces(transcript_text))[:3]
    return {
        "beat_id": f"OBS{beat_index:02d}",
        "title": title,
        "summary": (beat_summary or observation)[:220],
        "start_seconds": 0.0,
        "end_seconds": 0.0,
        "duration_seconds": 0.6,
        "visual_focus": _split_freeform_story_chunks(
            _normalize_spaces(" ".join([scene_state, subject_identity, beat_observation]))
        )[:4],
        "camera_language": [],
        "art_direction_cues": [],
        "storyboard_value": _split_freeform_story_chunks(
            _normalize_spaces(" ".join([dialogue_summary, timeflow_summary, transcript_text]))
        )[:4]
        or storyboard_value,
        "text_blob": _normalize_spaces(
            " ".join(
                [
                    title,
                    beat_summary,
                    scene_state,
                    subject_identity,
                    dialogue_summary,
                    timeflow_summary,
                    beat_observation,
                    transcript_text,
                ]
            )
        ),
    }


def _build_shot_chain_from_segment(
    *,
    segment: Mapping[str, Any],
    story_beats: Sequence[Mapping[str, Any]],
    primary_purpose: str,
    dialogue_windows: Sequence[Mapping[str, Any]],
    target_duration_seconds: float | None = None,
) -> list[dict[str, Any]]:
    shot_timeline = list(segment.get("shot_timeline") or [])
    beat_start_seconds = float(segment.get("start_seconds", 0) or 0)
    source_duration_seconds = max(0.8, float(segment.get("duration_seconds", 0) or 0))
    target_duration = max(source_duration_seconds, float(target_duration_seconds or source_duration_seconds))
    time_scale = target_duration / max(source_duration_seconds, 0.8)
    total_shots = len(shot_timeline)
    prepared_shots: list[dict[str, Any]] = []
    story_bucket_sizes: Counter[str] = Counter()
    for shot in shot_timeline:
        shot_start = float(shot.get("start_seconds", beat_start_seconds) or beat_start_seconds)
        shot_end = float(shot.get("end_seconds", shot_start) or shot_start)
        story_ref = _best_story_beat_for_range(story_beats, shot_start, shot_end)
        bucket_key = str(story_ref.get("beat_id") or f"__shot_{len(prepared_shots) + 1:02d}")
        story_bucket_sizes[bucket_key] += 1
        prepared_shots.append(
            {
                "shot": dict(shot),
                "shot_start": shot_start,
                "shot_end": shot_end,
                "story_ref": dict(story_ref or {}),
                "bucket_key": bucket_key,
            }
        )

    rendered: list[dict[str, Any]] = []
    story_bucket_offsets: Counter[str] = Counter()
    for index, prepared in enumerate(prepared_shots, start=1):
        shot = dict(prepared.get("shot") or {})
        shot_start = float(prepared.get("shot_start", beat_start_seconds) or beat_start_seconds)
        shot_end = float(prepared.get("shot_end", shot_start) or shot_start)
        story_ref = dict(prepared.get("story_ref") or {})
        bucket_key = str(prepared.get("bucket_key") or "__fallback__")
        story_bucket_offsets[bucket_key] += 1
        bucket_index = int(story_bucket_offsets[bucket_key])
        bucket_total = int(story_bucket_sizes.get(bucket_key, 1) or 1)
        relative_start = round(max(0.0, (shot_start - beat_start_seconds) * time_scale), 2)
        relative_end = round(max(relative_start + 0.2, (shot_end - beat_start_seconds) * time_scale), 2)
        if index == total_shots:
            relative_end = round(target_duration, 2)
        overlapping_dialogue = []
        for window in dialogue_windows:
            dialogue_start = float(window.get("start_seconds", 0) or 0)
            dialogue_end = float(window.get("end_seconds", dialogue_start) or dialogue_start)
            if dialogue_end < shot_start or dialogue_start > shot_end:
                continue
            overlapping_dialogue.append(str(window.get("text") or ""))
        dialogue_hint = _compose_dialogue_hint(overlapping_dialogue)
        role_label = _shot_role_label(
            overall_index=index,
            overall_total=total_shots,
            bucket_index=bucket_index,
            bucket_total=bucket_total,
            primary_purpose=primary_purpose,
            dialogue_hint=dialogue_hint,
        )
        focus_text = _shot_specific_field(
            story_ref=story_ref,
            field="visual_focus",
            overall_index=index,
            overall_total=total_shots,
            bucket_index=bucket_index,
            bucket_total=bucket_total,
            shot=shot,
            dialogue_hint=dialogue_hint,
            primary_purpose=primary_purpose,
            fallback=str(segment.get("summary_text") or shot.get("ocr_hint") or ""),
            role_label=role_label,
            segment_summary=str(segment.get("summary_text") or ""),
        )
        camera_text = _shot_specific_field(
            story_ref=story_ref,
            field="camera_language",
            overall_index=index,
            overall_total=total_shots,
            bucket_index=bucket_index,
            bucket_total=bucket_total,
            shot=shot,
            dialogue_hint=dialogue_hint,
            primary_purpose=primary_purpose,
            fallback=_fallback_camera_language_for_purpose(
                primary_purpose,
                index=index,
                total=total_shots,
                role_label=role_label,
                dialogue_hint=dialogue_hint,
                shot=shot,
            ),
            role_label=role_label,
            segment_summary=str(segment.get("summary_text") or ""),
        )
        art_hint = _shot_specific_field(
            story_ref=story_ref,
            field="art_direction_cues",
            overall_index=index,
            overall_total=total_shots,
            bucket_index=bucket_index,
            bucket_total=bucket_total,
            shot=shot,
            dialogue_hint=dialogue_hint,
            primary_purpose=primary_purpose,
            fallback=str(segment.get("summary_text") or shot.get("ocr_hint") or ""),
            role_label=role_label,
            segment_summary=str(segment.get("summary_text") or ""),
        )
        transition_hint = _shot_specific_field(
            story_ref=story_ref,
            field="storyboard_value",
            overall_index=index,
            overall_total=total_shots,
            bucket_index=bucket_index,
            bucket_total=bucket_total,
            shot=shot,
            dialogue_hint=dialogue_hint,
            primary_purpose=primary_purpose,
            fallback=_fallback_transition_trigger_for_shot(
                primary_purpose=primary_purpose,
                overall_index=index,
                overall_total=total_shots,
                role_label=role_label,
                shot=shot,
                dialogue_hint=dialogue_hint,
                segment_summary=str(segment.get("summary_text") or ""),
            ),
            role_label=role_label,
            segment_summary=str(segment.get("summary_text") or ""),
        )
        rendered.append(
            {
                "shot_id": f"S{index:02d}",
                "source_scene_id": str(shot.get("source_scene_id") or ""),
                "time_range": f"{relative_start:.1f}-{relative_end:.1f}秒",
                "start_seconds": relative_start,
                "end_seconds": relative_end,
                "role_label": role_label,
                "story_function": _compose_shot_story_function(
                    story_title=_story_title_for_shot(
                        story_ref=story_ref,
                        segment=segment,
                        shot=shot,
                        overall_index=index,
                        overall_total=total_shots,
                        primary_purpose=primary_purpose,
                        dialogue_hint=dialogue_hint,
                        role_label=role_label,
                    ),
                    role_label=role_label,
                ),
                "visual_focus": focus_text,
                "camera_language": camera_text,
                "camera_entry": _fallback_camera_entry_for_shot(
                    primary_purpose=primary_purpose,
                    overall_index=index,
                    overall_total=total_shots,
                    role_label=role_label,
                    dialogue_hint=dialogue_hint,
                    shot=shot,
                ),
                "subject_blocking": _fallback_subject_blocking_for_shot(
                    primary_purpose=primary_purpose,
                    overall_index=index,
                    overall_total=total_shots,
                    role_label=role_label,
                    shot=shot,
                ),
                "action_timeline": _fallback_action_timeline_for_shot(
                    primary_purpose=primary_purpose,
                    overall_index=index,
                    overall_total=total_shots,
                    role_label=role_label,
                    dialogue_hint=dialogue_hint,
                    focus_text=focus_text,
                    segment_summary=str(segment.get("summary_text") or ""),
                ),
                "lighting_and_texture": _fallback_lighting_and_texture_for_shot(
                    primary_purpose=primary_purpose,
                    overall_index=index,
                    overall_total=total_shots,
                    shot=shot,
                    art_hint=art_hint,
                ),
                "background_continuity": _fallback_background_continuity_for_shot(
                    shot=shot,
                    primary_purpose=primary_purpose,
                    overall_index=index,
                    overall_total=total_shots,
                ),
                "dialogue_timing": _fallback_dialogue_timing_for_shot(
                    overall_index=index,
                    overall_total=total_shots,
                    role_label=role_label,
                    dialogue_hint=dialogue_hint,
                ),
                "sound_bed": _fallback_sound_bed_for_shot(
                    primary_purpose=primary_purpose,
                    role_label=role_label,
                    focus_text=focus_text,
                    camera_text=camera_text,
                    art_hint=art_hint,
                ),
                "dialogue_hint": dialogue_hint,
                "art_direction_hint": art_hint,
                "transition_trigger": transition_hint,
                "frame_path": str(shot.get("keyframe_path") or ""),
            }
        )
    return rendered


def _shot_specific_field(
    *,
    story_ref: Mapping[str, Any],
    field: str,
    overall_index: int,
    overall_total: int,
    bucket_index: int,
    bucket_total: int,
    shot: Mapping[str, Any],
    dialogue_hint: str,
    primary_purpose: str,
    fallback: str,
    role_label: str,
    segment_summary: str,
) -> str:
    lines = _clean_list(story_ref.get(field))
    if field in {"visual_focus", "art_direction_cues"}:
        lines = [line for line in lines if _looks_like_visual_hint(line)]
    elif field == "storyboard_value":
        lines = [line for line in lines if _looks_like_transition_hint(line)]
    if lines:
        sequence: list[str] = []
        for line in lines:
            sequence.extend(_split_shot_detail_sequence(line))
        if field in {"visual_focus", "art_direction_cues"}:
            sequence = [line for line in sequence if _looks_like_visual_hint(line)]
        elif field == "storyboard_value":
            sequence = [line for line in sequence if _looks_like_transition_hint(line)]
        sequence = _dedupe_strings(sequence)
        if sequence:
            picked = _pick_sequence_item(sequence, bucket_index=bucket_index, bucket_total=bucket_total)
            if picked:
                return picked
        picked_line = _pick_sequence_item(lines, bucket_index=bucket_index, bucket_total=bucket_total)
        if picked_line:
            return picked_line
        return lines[0]
    if field == "visual_focus":
        return _fallback_visual_focus_for_shot(
            primary_purpose=primary_purpose,
            overall_index=overall_index,
            overall_total=overall_total,
            fallback=fallback,
            shot=shot,
            dialogue_hint=dialogue_hint,
            role_label=role_label,
        )
    if field == "camera_language":
        return _fallback_camera_language_for_purpose(
            primary_purpose,
            index=overall_index,
            total=overall_total,
            role_label=role_label,
            dialogue_hint=dialogue_hint,
            shot=shot,
        )
    if field == "storyboard_value":
        return _fallback_transition_trigger_for_shot(
            primary_purpose=primary_purpose,
            overall_index=overall_index,
            overall_total=overall_total,
            role_label=role_label,
            shot=shot,
            dialogue_hint=dialogue_hint,
            segment_summary=segment_summary,
        )
    if field == "art_direction_cues":
        fallback_chunks = [
            chunk
            for chunk in _split_freeform_story_chunks(fallback)
            if _looks_like_visual_hint(chunk)
        ]
        if fallback_chunks:
            return _pick_sequence_item(fallback_chunks, bucket_index=overall_index, bucket_total=overall_total)[:100]
    return _normalize_spaces(fallback)


def _split_shot_detail_sequence(text: str) -> list[str]:
    normalized = _normalize_spaces(text)
    if not normalized:
        return []
    parts = re.split(r"\s*(?:→|->|⇒|⟶)\s*", normalized)
    if len(parts) <= 1:
        parts = re.split(r"\s*[；;]\s*", normalized)
    if len(parts) <= 1:
        parts = re.split(r"\s*\|\s*", normalized)
    if len(parts) <= 1:
        parts = re.split(r"\s*\+\s*", normalized)
    cleaned = [_normalize_spaces(part) for part in parts if _normalize_spaces(part)]
    return cleaned or [normalized]


def _split_freeform_story_chunks(text: str) -> list[str]:
    normalized = _normalize_spaces(text)
    if not normalized:
        return []
    parts = re.split(r"\s*[。！？!?；;]\s*", normalized)
    if len(parts) <= 1:
        parts = re.split(r"\s*[，,、/]\s*", normalized)
    cleaned = [_normalize_spaces(part) for part in parts if _normalize_spaces(part)]
    if len(cleaned) <= 1 and len(normalized) >= 18:
        chunk_count = min(4, max(2, round(len(normalized) / 18)))
        step = max(8, math.ceil(len(normalized) / max(chunk_count, 1)))
        cleaned = [_normalize_spaces(normalized[i : i + step]) for i in range(0, len(normalized), step)]
    return _dedupe_strings(cleaned or [normalized])


def _pick_sequence_item(sequence: Sequence[str], *, bucket_index: int, bucket_total: int) -> str:
    if not sequence:
        return ""
    if len(sequence) == 1:
        return sequence[0]
    if bucket_total <= 1:
        return sequence[min(len(sequence) - 1, 0)]
    mapped = round((bucket_index - 1) * (len(sequence) - 1) / max(bucket_total - 1, 1))
    mapped = max(0, min(len(sequence) - 1, mapped))
    return sequence[mapped]


def _fallback_visual_focus_for_shot(
    *,
    primary_purpose: str,
    overall_index: int,
    overall_total: int,
    fallback: str,
    shot: Mapping[str, Any],
    dialogue_hint: str,
    role_label: str,
) -> str:
    visual_dialogue_chunks = [
        chunk
        for chunk in _split_freeform_story_chunks(dialogue_hint)
        if _looks_like_visual_hint(chunk)
    ]
    if visual_dialogue_chunks:
        picked_dialogue = _pick_sequence_item(
            visual_dialogue_chunks,
            bucket_index=overall_index,
            bucket_total=overall_total,
        )
        if picked_dialogue:
            return picked_dialogue[:100]
    normalized_fallback = _normalize_spaces(fallback)
    fallback_chunks = [
        chunk
        for chunk in _split_freeform_story_chunks(normalized_fallback)
        if _looks_like_visual_hint(chunk)
    ]
    if fallback_chunks:
        picked_fallback = _pick_sequence_item(
            fallback_chunks,
            bucket_index=overall_index,
            bucket_total=overall_total,
        )
        if picked_fallback:
            return picked_fallback[:100]
    if overall_index == 1:
        return "当前空间、人物关系与首个动作起点"
    if overall_index == overall_total:
        return "结果反应、尾帧触发物与未完动作"
    if role_label in {"对白推进", "信息落点"}:
        return "说话者口型、听者眼神与关系落点"
    if role_label in {"反应承接", "结果反应"}:
        return "结果方表情变化与肢体停顿"
    if primary_purpose in {"危险", "对峙", "守护"}:
        return "手位、重心和压迫方向变化"
    if primary_purpose in {"揭示", "规则"}:
        return "证据物、表情停顿和信息落点"
    return "动作推进与承接反应"


def _shot_role_label(
    *,
    overall_index: int,
    overall_total: int,
    bucket_index: int,
    bucket_total: int,
    primary_purpose: str,
    dialogue_hint: str,
) -> str:
    if overall_total <= 1:
        return "单拍完成"
    if overall_index == 1:
        return "开场建立"
    if overall_index == overall_total:
        return "尾帧收束"
    if bucket_index == 1 and bucket_total > 1:
        return "动作触发"
    if bucket_index == bucket_total:
        return "结果反应"
    if dialogue_hint:
        if overall_index <= max(2, overall_total // 3):
            return "对白推进"
        if overall_index >= overall_total - 1:
            return "反应承接"
        return "信息落点"
    if primary_purpose in {"特效", "觉醒"}:
        return "异动扩散"
    if primary_purpose in {"危险", "对峙", "守护"}:
        return "张力推进"
    return "中段推进"


def _compose_shot_story_function(*, story_title: str, role_label: str) -> str:
    if not role_label:
        return story_title
    return f"{story_title}·{role_label}"


def _best_story_beat_for_range(
    story_beats: Sequence[Mapping[str, Any]],
    start_seconds: float,
    end_seconds: float,
) -> Mapping[str, Any]:
    best: Mapping[str, Any] | None = None
    best_overlap = -1.0
    for item in story_beats:
        beat_start = float(item.get("start_seconds", 0) or 0)
        beat_end = float(item.get("end_seconds", beat_start) or beat_start)
        overlap = max(0.0, min(end_seconds, beat_end) - max(start_seconds, beat_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best = item
    if best_overlap <= 0.05:
        return {}
    return best or {}


def _fallback_camera_language_for_purpose(
    primary_purpose: str,
    *,
    index: int,
    total: int,
    role_label: str = "",
    dialogue_hint: str = "",
    shot: Mapping[str, Any] | None = None,
) -> str:
    shot = dict(shot or {})
    if total <= 1:
        if primary_purpose in {"特效", "觉醒", "尾钩"}:
            return "先稳住源头，再把异常结果停在尾帧"
        if primary_purpose in {"羞辱", "权力", "规则"}:
            return "稳定景别承接高低位压迫关系"
        return "镜头围绕当前人物关系和动作结果推进"
    if index == 1:
        if primary_purpose in {"群像", "权力", "规则"}:
            return "先用稳定中广景交代空间秩序和主位"
        if dialogue_hint:
            return "先给说话者中近景，再带听者反应"
        return "先立空间锚点和触发动作"
    if index == total:
        if primary_purpose in {"危险", "尾钩", "特效"}:
            return "尾帧悬停在未落动作或异常信号上"
        if dialogue_hint:
            return "句尾停在听者反应或未落动作上"
        return "尾帧收在结果反应或下一拍触发物上"
    if dialogue_hint:
        dialogue_variants = [
            "正反打承接台词，优先保留视线方向",
            "半身对切推进，句尾补一拍听者反应",
            "先给说话者，再切沉默方表情变化",
        ]
        return dialogue_variants[(index - 2) % len(dialogue_variants)]
    if primary_purpose in {"特效", "觉醒"}:
        variants = [
            "先盯异常源头，再切受体反馈",
            "跟着能量或视线方向推进，再补环境反应",
            "近景接细节变化，回人物结果镜",
        ]
        return variants[(index - 2) % len(variants)]
    if primary_purpose in {"危险", "对峙", "守护"}:
        variants = [
            "沿视线轴做正反打，补一拍手位变化",
            "先给施压者，再切受压者反应，轴线保持稳定",
            "中近景接动作推进，补一拍重心变化",
        ]
        return variants[(index - 2) % len(variants)]
    if role_label in {"结果反应", "反应承接"}:
        return "切到结果方反应，停半拍让情绪落下"
    return "先给动作触发，再切承接反应"


def _fallback_camera_entry_for_shot(
    *,
    primary_purpose: str,
    overall_index: int,
    overall_total: int,
    role_label: str,
    dialogue_hint: str,
    shot: Mapping[str, Any],
) -> str:
    if overall_total <= 1:
        if primary_purpose in {"特效", "觉醒", "尾钩"}:
            return "镜头从异常源头附近的近景切入"
        if primary_purpose in {"羞辱", "权力", "规则"}:
            return "镜头从施压关系的侧前方切入"
        return "镜头从当前事件主位的一侧中近景切入"
    if overall_index == 1:
        if primary_purpose in {"羞辱", "权力", "规则"}:
            return "镜头从高低位关系最清楚的一侧切入"
        if primary_purpose in {"群像", "对峙"}:
            return "镜头从能同时看到双方轴线的一侧切入"
        if primary_purpose in {"爱情", "思念", "告别"}:
            return "镜头从人物距离与目光方向最清楚的一侧切入"
        return "镜头从当前动作起点或空间锚点附近切入"
    if overall_index == overall_total:
        return "镜头顺着上一拍动作结果收向尾帧落点"
    if dialogue_hint:
        return "镜头从当前说话者的前侧或肩侧位置承接切入"
    if role_label in {"动作触发", "张力推进", "异动扩散"}:
        return "镜头从动作发起方向的前侧切入"
    if role_label in {"结果反应", "反应承接"}:
        return "镜头从受体或听者一侧切入承接结果"
    return "镜头从当前主体动作最清楚的一侧切入"


def _fallback_subject_blocking_for_shot(
    *,
    primary_purpose: str,
    overall_index: int,
    overall_total: int,
    role_label: str,
    shot: Mapping[str, Any],
) -> str:
    source_scene_id = str(shot.get("source_scene_id") or "").strip()
    scene_hint = f"{source_scene_id}对应的空间轴线" if source_scene_id else "同一空间轴线"
    if primary_purpose in {"羞辱", "权力", "规则"}:
        return f"保持施压者在更高或更前的位置，让受压者留在低位承受画面压力，并沿用{scene_hint}"
    if primary_purpose in {"守护", "危险", "牺牲"}:
        return f"保持威胁源、保护对象与插入者的前后位，动作路径始终沿着{scene_hint}展开"
    if primary_purpose in {"群像", "对峙"}:
        return f"保持双方分占轴线两侧或前后纵深，主位人物始终卡在{scene_hint}上"
    if primary_purpose in {"爱情", "思念", "告别"}:
        return f"保持人物距离、身体朝向和回望方向稳定，关系变化始终落在{scene_hint}里"
    if overall_index == overall_total or role_label == "尾帧收束":
        return f"让结果主体占住主位，其余人物或空间信息退到后景，继续保留{scene_hint}"
    return f"保持当前主体站在主位，听者/对手或关键物件留在侧后或后景，沿用{scene_hint}"


def _fallback_action_timeline_for_shot(
    *,
    primary_purpose: str,
    overall_index: int,
    overall_total: int,
    role_label: str,
    dialogue_hint: str,
    focus_text: str,
    segment_summary: str,
) -> str:
    chunks = [
        chunk
        for chunk in _split_freeform_story_chunks(focus_text or segment_summary)
        if _looks_like_visual_hint(chunk)
    ]
    focus_hint = chunks[0] if chunks else "当前动作焦点"
    if overall_total <= 1:
        return f"先把{focus_hint}立住，再给一个可见的动作落点，最后停在结果或下一拍触发物上"
    if overall_index == 1:
        return f"先把{focus_hint}作为起点立住，再把动作起势或表情停顿送进画面"
    if overall_index == overall_total:
        return "让上一拍结果继续推进半拍，再把未落动作、反应或触发物稳稳停在尾帧"
    if dialogue_hint:
        return "先给一句话前的停顿或手位变化，再让动作和台词同拍推进，句尾落到听者反应"
    if role_label in {"动作触发", "张力推进", "异动扩散"}:
        return "先给动作起点，再推进到峰值，最后补一个受体或环境反馈"
    if primary_purpose in {"特效", "觉醒"}:
        return "先压住异常源头，再让变化扩散，最后把余波落到人物或环境上"
    return "先确认当前动作起势，再推进到结果，最后补一拍承接反应"


def _fallback_lighting_and_texture_for_shot(
    *,
    primary_purpose: str,
    overall_index: int,
    overall_total: int,
    shot: Mapping[str, Any],
    art_hint: str,
) -> str:
    if art_hint and _looks_like_visual_hint(art_hint):
        return art_hint[:120]
    if primary_purpose in {"特效", "觉醒", "尾钩"}:
        return "保留主光源与异常辉光对周边材质的受光变化，强调边缘光和余波质感"
    if primary_purpose in {"羞辱", "权力", "规则", "危险"}:
        return "强调冷硬侧光、衣料摩擦、皮肤受压或石面金属面的硬质触感"
    if primary_purpose in {"爱情", "思念", "告别"}:
        return "保留人物皮肤、衣摆和环境边缘光的柔和层次，避免光线跳变"
    if overall_index == 1:
        return "先把空间主光和主体材质质感交代清楚"
    if overall_index == overall_total:
        return "让尾帧的受光和材质停在最值钱的结果物上"
    return "保持同一光向、材质和环境空气感，不要让中段受光关系跳变"


def _fallback_background_continuity_for_shot(
    *,
    shot: Mapping[str, Any],
    primary_purpose: str,
    overall_index: int,
    overall_total: int,
) -> str:
    source_scene_id = str(shot.get("source_scene_id") or "").strip()
    if source_scene_id:
        return f"画面继续沿用{source_scene_id}对应的空间结构、人物朝向与受光关系"
    if primary_purpose in {"群像", "对峙"}:
        return "后景继续保留双方轴线、纵深层次与旁观者所在方位"
    if overall_index == overall_total:
        return "尾帧继续保留与下一拍相关的空间结构和未消失的触发物"
    return "画面继续沿用同一空间结构、人物场位与受光关系"


def _fallback_dialogue_timing_for_shot(
    *,
    overall_index: int,
    overall_total: int,
    role_label: str,
    dialogue_hint: str,
) -> str:
    if not dialogue_hint:
        return ""
    if overall_total <= 1:
        return "让这句对白压在动作中后段，说完后留半拍看反应"
    if overall_index == 1:
        return "先压一拍表情或手位，再把第一句对白推出"
    if overall_index == overall_total:
        return "把最后一句压在动作落点后，说完停半拍把结果留给尾帧"
    if role_label in {"对白推进", "信息落点"}:
        return "让台词跟着中段动作推进，句尾切去承接反应"
    return "先让动作起势，再把对白压进这一拍，句尾留给下一镜"


def _fallback_sound_bed_for_shot(
    *,
    primary_purpose: str,
    role_label: str,
    focus_text: str,
    camera_text: str,
    art_hint: str,
) -> str:
    text_blob = " ".join([focus_text, camera_text, art_hint, role_label])
    if any(token in text_blob for token in ("刀", "剑", "刃")):
        return "近距离刀刃轻擦、衣料摩擦和被压住的呼吸声"
    if any(token in text_blob for token in ("雷", "光", "法阵", "镜门", "异变", "金丹")):
        return "低频嗡鸣、风压变化和周边材质被带动的细响"
    if primary_purpose in {"羞辱", "权力", "规则"}:
        return "风声压底、衣料摩擦和空间回响保持在下层"
    if primary_purpose in {"守护", "危险", "牺牲"}:
        return "脚步、闷响、粗喘和受力摩擦声压在近处"
    if primary_purpose in {"爱情", "思念", "告别"}:
        return "呼吸、衣摆和被留白的环境底噪轻轻托住情绪"
    if role_label == "尾帧收束":
        return "主动抽空杂音，只留下最值钱的环境声或异常低鸣"
    return ""


def _story_title_for_shot(
    *,
    story_ref: Mapping[str, Any],
    segment: Mapping[str, Any],
    shot: Mapping[str, Any],
    overall_index: int,
    overall_total: int,
    primary_purpose: str,
    dialogue_hint: str,
    role_label: str,
) -> str:
    title = _normalize_spaces(str(story_ref.get("title") or ""))
    if title and (_looks_like_visual_hint(title) or (len(title) <= 12 and _keyword_hit_count(title, DIALOGUE_HEAVY_HINT_KEYWORDS) == 0)):
        return title[:24]
    return _fallback_story_title_for_shot(
        segment=segment,
        shot=shot,
        overall_index=overall_index,
        overall_total=overall_total,
        primary_purpose=primary_purpose,
        dialogue_hint=dialogue_hint,
        role_label=role_label,
    )


def _fallback_story_title_for_shot(
    *,
    segment: Mapping[str, Any],
    shot: Mapping[str, Any],
    overall_index: int,
    overall_total: int,
    primary_purpose: str,
    dialogue_hint: str,
    role_label: str,
) -> str:
    chunks = [
        chunk
        for chunk in _split_freeform_story_chunks(dialogue_hint or str(segment.get("summary_text") or ""))
        if _looks_like_visual_hint(chunk)
    ]
    if chunks:
        picked = _pick_sequence_item(chunks, bucket_index=overall_index, bucket_total=overall_total)
        if picked:
            return picked[:24]
    if role_label and role_label not in {"中段推进", "单拍完成"}:
        return f"{primary_purpose}{role_label}"
    return f"{primary_purpose}镜头{overall_index}"


def _fallback_transition_trigger_for_shot(
    *,
    primary_purpose: str,
    overall_index: int,
    overall_total: int,
    role_label: str,
    shot: Mapping[str, Any],
    dialogue_hint: str,
    segment_summary: str,
) -> str:
    if overall_index == overall_total:
        return "尾帧把结果反应或下一拍触发物留下"
    if dialogue_hint:
        return "对白句尾或停顿触发切镜"
    if primary_purpose in {"特效", "觉醒"}:
        return "异常信号扩散到下一镜"
    if role_label in {"结果反应", "反应承接"}:
        return "对手或听者反应带出下一镜"
    chunks = _split_freeform_story_chunks(segment_summary)
    if chunks:
        picked = _pick_sequence_item(chunks, bucket_index=overall_index, bucket_total=overall_total)
        if picked:
            return picked[:100]
    return "动作起点成立后切到承接镜"


def _group_story_beats(beats: Sequence[dict[str, Any]], settings: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
    if not beats:
        return []
    min_duration = float(settings.get("min_beat_duration", DEFAULT_MIN_BEAT_DURATION))
    max_duration = float(settings.get("max_beat_duration", DEFAULT_MAX_BEAT_DURATION))
    grouped: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for beat in beats:
        if not current:
            current = [beat]
            continue
        current_duration = current[-1]["end_seconds"] - current[0]["start_seconds"]
        merged_duration = beat["end_seconds"] - current[0]["start_seconds"]
        same_purpose_bias = _top_purpose_for_text(current[-1]["text_blob"]) == _top_purpose_for_text(beat["text_blob"])
        should_merge = current_duration < min_duration or (same_purpose_bias and merged_duration <= max_duration)
        if should_merge and merged_duration <= max_duration + 1.5:
            current.append(beat)
            continue
        grouped.append(current)
        current = [beat]
    if current:
        grouped.append(current)

    if len(grouped) >= 2:
        last_duration = grouped[-1][-1]["end_seconds"] - grouped[-1][0]["start_seconds"]
        prev_duration = grouped[-2][-1]["end_seconds"] - grouped[-2][0]["start_seconds"]
        if last_duration < 6.0 and prev_duration + last_duration <= max_duration + 2.0:
            grouped[-2].extend(grouped[-1])
            grouped.pop()
    return grouped


def _story_beat_seconds(raw: Mapping[str, Any], *, fallback_index: int) -> tuple[float, float, str]:
    range_candidates = []
    for key in ("start_time", "end_time"):
        seconds = _parse_seconds_from_time_window(raw.get(key))
        if seconds:
            range_candidates.append(seconds)
    if range_candidates:
        start_seconds = round(min(item[0] for item in range_candidates), 2)
        end_seconds = round(max(item[1] for item in range_candidates), 2)
        if end_seconds <= start_seconds:
            end_seconds = round(start_seconds + 0.8, 2)
        return start_seconds, end_seconds, "explicit_window"

    start_point = _parse_timepoint_seconds(raw.get("start_time"))
    end_point = _parse_timepoint_seconds(raw.get("end_time"))
    if start_point is not None and end_point is not None:
        start_seconds = round(min(start_point, end_point), 2)
        end_seconds = round(max(start_point, end_point), 2)
        if end_seconds <= start_seconds:
            end_seconds = round(start_seconds + 0.8, 2)
        return start_seconds, end_seconds, "explicit_points"
    fallback_start = round((fallback_index - 1) * 8.0, 2)
    return fallback_start, round(fallback_start + 8.0, 2), "fallback_uniform"


def _parse_seconds_from_time_window(raw: Any) -> tuple[float, float] | None:
    points = _extract_timepoints(str(raw or ""))
    if len(points) >= 2:
        return round(min(points), 2), round(max(points), 2)
    return None


def _parse_timepoint_seconds(raw: Any) -> float | None:
    points = _extract_timepoints(str(raw or ""))
    if len(points) == 1:
        return round(float(points[0]), 2)
    if len(points) >= 2:
        return round(float(points[0]), 2)
    return None


def _extract_timepoints(raw: str) -> list[float]:
    text = str(raw or "").strip()
    if not text:
        return []

    points: list[float] = []
    for match in TIMEPOINT_PATTERN.finditer(text):
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        points.append(hours * 3600 + minutes * 60 + seconds)
    if points:
        return points

    for match in MINSEC_TIMEPOINT_PATTERN.finditer(text):
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        points.append(minutes * 60 + seconds)
    if points:
        return points

    normalized = text
    for separator in RANGE_SEPARATORS:
        normalized = normalized.replace(separator, " ")
    number_tokens = re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", normalized)
    if number_tokens:
        return [float(token) for token in number_tokens]
    return []


def _purpose_scores_for_group(group: Sequence[Mapping[str, Any]], *, is_last: bool) -> list[dict[str, Any]]:
    combined_scores: Counter[str] = Counter()
    for item in group:
        for purpose, score in _score_purposes(str(item.get("text_blob") or ""), is_last=is_last).items():
            combined_scores[purpose] += score
    if not combined_scores:
        combined_scores[FALLBACK_PURPOSE] = 1.0
    ranked = sorted(
        [{"purpose": purpose, "score": round(score, 2)} for purpose, score in combined_scores.items()],
        key=lambda item: (-float(item["score"]), PURPOSE_ORDER.index(item["purpose"]) if item["purpose"] in PURPOSE_ORDER else 999),
    )
    return ranked


def _score_purposes(text: str, *, is_last: bool = False) -> Counter[str]:
    normalized = str(text or "")
    scores: Counter[str] = Counter()
    for purpose, keywords in PURPOSE_KEYWORDS.items():
        for token in keywords:
            if token and token in normalized:
                scores[purpose] += 1.0 if len(token) >= 2 else 0.35
    if is_last:
        scores["尾钩"] += 0.5
    if any(token in normalized for token in ("异常", "凌空", "悬浮", "飞身", "腾空", "跃下", "异象")):
        scores["特效"] += 0.9
    if any(token in normalized for token in ("离开", "云游", "送别", "转身", "先走了")):
        scores["告别"] += 0.8
    if any(token in normalized for token in ("救", "挡", "护住", "替他", "替她")):
        scores["守护"] += 0.8
    if any(token in normalized for token in ("心跳", "发光", "苏醒", "共鸣", "灵力")):
        scores["觉醒"] += 1.0
    if any(token in normalized for token in ("雷", "爆", "法阵", "镜门", "冲击", "异变")):
        scores["特效"] += 1.0
    if any(token in normalized for token in ("下令", "家主", "裁判", "宣判", "命令")):
        scores["权力"] += 0.8
    if any(token in normalized for token in ("资格", "代价", "结果", "规则")):
        scores["规则"] += 0.8
    if any(token in normalized for token in ("真相", "揭示", "显影", "认知", "曝光")):
        scores["揭示"] += 0.8
    return scores


def _top_purpose_for_text(text: str) -> str:
    scores = _score_purposes(text)
    if not scores:
        return FALLBACK_PURPOSE
    return max(
        scores.items(),
        key=lambda item: (item[1], -(PURPOSE_ORDER.index(item[0]) if item[0] in PURPOSE_ORDER else 999)),
    )[0]


def _dialogue_windows_for_range(
    transcript_segments: Sequence[Mapping[str, Any]],
    start_seconds: float,
    end_seconds: float,
) -> list[dict[str, Any]]:
    overlaps = []
    for item in transcript_segments:
        start = float(item.get("start", 0) or 0)
        end = float(item.get("end", start) or start)
        if end < start_seconds or start > end_seconds:
            continue
        segment_duration = max(0.001, end - start)
        overlap_duration = max(0.0, min(end_seconds, end) - max(start_seconds, start))
        overlap_ratio = overlap_duration / segment_duration
        center_seconds = start + segment_duration / 2
        include = start_seconds <= center_seconds <= end_seconds
        if not include:
            if segment_duration <= 1.4 and overlap_ratio >= 0.45:
                include = True
            elif overlap_duration >= 1.2 and overlap_ratio >= 0.7:
                include = True
        if not include:
            continue
        overlaps.append(
            {
                "start": round(max(start_seconds, start), 2),
                "end": round(min(end_seconds, end), 2),
                "text": _normalize_spaces(str(item.get("text") or "")),
            }
        )
    if not overlaps:
        return []

    windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [overlaps[0]]
    for item in overlaps[1:]:
        current_text = "".join(str(part.get("text") or "") for part in current)
        next_text = str(item.get("text") or "")
        combined = _normalize_spaces(current_text + next_text)
        gap_seconds = float(item["start"]) - float(current[-1]["end"])
        should_merge = gap_seconds <= DEFAULT_DIALOGUE_WINDOW_GAP_SECONDS and (
            len(combined) <= DEFAULT_DIALOGUE_WINDOW_MAX_MERGED_CHARS
            or ((len(current_text) <= 4 or len(next_text) <= 4) and len(combined) <= DEFAULT_DIALOGUE_WINDOW_MAX_MERGED_CHARS + 4)
        )
        if should_merge:
            current.append(item)
            continue
        windows.append(current)
        current = [item]
    if current:
        windows.append(current)

    rendered: list[dict[str, Any]] = []
    for window in windows:
        start = window[0]["start"]
        end = window[-1]["end"]
        text = _normalize_spaces("".join(item["text"] for item in window if item["text"]))
        if not text:
            continue
        rendered.append(
            {
                "start_seconds": round(start, 2),
                "end_seconds": round(end, 2),
                "time_range": f"{start:.2f}-{end:.2f}s",
                "text": text,
            }
        )
    return rendered[:DEFAULT_DIALOGUE_WINDOW_RENDER_LIMIT]


def _compose_dialogue_hint(
    dialogue_lines: Sequence[str],
    *,
    limit_chars: int = 44,
    max_parts: int = 3,
) -> str:
    parts: list[str] = []
    for raw in dialogue_lines:
        text = _normalize_spaces(str(raw or ""))
        if not text or text == "无对白" or text in parts:
            continue
        projected = " / ".join(parts + [text])
        if parts and len(projected) > limit_chars:
            break
        if not parts and len(text) > limit_chars:
            return _truncate_prompt_fragment(text, limit=limit_chars)
        parts.append(text)
        if len(parts) >= max_parts:
            break
    return " / ".join(parts).strip()


def _scene_anchor_for_range(
    scenes: Sequence[Mapping[str, Any]],
    keyframes: Sequence[Mapping[str, Any]],
    start_seconds: float,
    end_seconds: float,
) -> dict[str, Any]:
    overlapping_scene_ids: list[str] = []
    for scene in scenes:
        scene_start = float(scene.get("start_seconds", 0) or 0)
        scene_end = float(scene.get("end_seconds", scene_start) or scene_start)
        if scene_end < start_seconds or scene_start > end_seconds:
            continue
        scene_id = str(scene.get("scene_id") or "").strip()
        if scene_id and scene_id not in overlapping_scene_ids:
            overlapping_scene_ids.append(scene_id)

    overlapping_keyframes = []
    for frame in keyframes:
        midpoint = float(frame.get("midpoint_seconds", 0) or 0)
        if midpoint < start_seconds or midpoint > end_seconds:
            continue
        overlapping_keyframes.append(
            {
                "scene_id": str(frame.get("scene_id") or "").strip(),
                "ocr_hint": _normalize_spaces(str(frame.get("linked_ocr_text") or "")),
                "frame_path": str(frame.get("model_frame_path") or frame.get("frame_path") or "").strip(),
            }
        )

    summary_parts = []
    if overlapping_scene_ids:
        summary_parts.append("镜头覆盖 " + "、".join(overlapping_scene_ids[:3]))
    ocr_hints = [item.get("ocr_hint", "") for item in overlapping_keyframes if item.get("ocr_hint")]
    if ocr_hints:
        summary_parts.append("OCR 锚点：" + "；".join(ocr_hints[:2]))
    return {
        "scene_ids": overlapping_scene_ids[:6],
        "keyframe_samples": overlapping_keyframes[:3],
        "summary": "；".join(summary_parts) if summary_parts else "依赖当前 beat 的表演与镜头逻辑组织空间。",
    }


def _build_shot_chain(
    *,
    group: Sequence[Mapping[str, Any]],
    beat_start_seconds: float,
    beat_end_seconds: float,
    dialogue_windows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    shots: list[dict[str, Any]] = []
    for index, item in enumerate(group, start=1):
        start_seconds = round(max(0.0, float(item.get("start_seconds", beat_start_seconds)) - beat_start_seconds), 2)
        end_seconds = round(
            min(beat_end_seconds, float(item.get("end_seconds", beat_end_seconds))) - beat_start_seconds,
            2,
        )
        if end_seconds <= start_seconds:
            end_seconds = round(start_seconds + max(0.8, float(item.get("duration_seconds", 0) or 0)), 2)
        overlapping_dialogue = []
        for window in dialogue_windows:
            dialogue_start = float(window.get("start_seconds", 0) or 0)
            dialogue_end = float(window.get("end_seconds", dialogue_start) or dialogue_start)
            shot_abs_start = float(item.get("start_seconds", beat_start_seconds) or beat_start_seconds)
            shot_abs_end = float(item.get("end_seconds", beat_end_seconds) or beat_end_seconds)
            if dialogue_end < shot_abs_start or dialogue_start > shot_abs_end:
                continue
            dialogue_duration = max(0.001, dialogue_end - dialogue_start)
            overlap_duration = max(0.0, min(dialogue_end, shot_abs_end) - max(dialogue_start, shot_abs_start))
            overlap_ratio = overlap_duration / dialogue_duration
            center_seconds = dialogue_start + dialogue_duration / 2
            include = shot_abs_start <= center_seconds <= shot_abs_end
            if not include and dialogue_duration <= 1.2 and overlap_ratio >= 0.45:
                include = True
            if not include:
                continue
            overlapping_dialogue.append(str(window.get("text") or ""))
        shots.append(
            {
                "shot_id": f"S{index:02d}",
                "time_range": f"{start_seconds:.1f}-{end_seconds:.1f}秒",
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "story_function": str(item.get("title") or "").strip(),
                "visual_focus": _best_line(item.get("visual_focus"), fallback=str(item.get("summary") or "")),
                "camera_language": _best_line(item.get("camera_language"), fallback="镜头以当前人物关系为主轴推进"),
                "dialogue_hint": _compose_dialogue_hint(overlapping_dialogue),
                "art_direction_hint": _best_line(item.get("art_direction_cues"), fallback=""),
                "transition_trigger": " ".join(
                    filter(
                        None,
                        [
                            _best_line(item.get("storyboard_value"), fallback=""),
                        ],
                    )
                ).strip(),
            }
        )
    return shots


def _derive_dramatic_goal(primary_purpose: str, group: Sequence[Mapping[str, Any]]) -> str:
    summary = _combine_unique_texts(item.get("summary") for item in group)
    title = _combine_unique_texts(item.get("title") for item in group)
    purpose_descriptions = {
        "爱情": "在一个 beat 内把情感张力和关系推进落到可拍细节上。",
        "思念": "让记忆或挂念通过现实触发物回流，不靠空泛抒情。",
        "痛苦": "让观众看懂痛感如何落到身体和呼吸上，而不是只见结果。",
        "告别": "把关系切断与离场方向交代清楚，留下空位和余波。",
        "守护": "让弱势护主或替人受压的动作路径、站位和誓言成立。",
        "羞辱": "把公开羞辱的高低位、压迫动作和受辱反馈拍清楚。",
        "反击": "在短时长里完成压抑后的冷反击和对手失衡。",
        "报仇": "让旧账被重新点题，并把清算启动的后果压到尾帧。",
        "对峙": "保持双方轴线与临界动作，制造下一拍随时爆发的张力。",
        "揭示": "让真相通过证据和反应镜头落地，而不是纯说明。",
        "权力": "把高位者的秩序改写能力拍成稳定的空间压迫。",
        "规则": "把世界规则、资格门槛和不介入逻辑说死并压到场上。",
        "觉醒": "先让异动显形，再让主体与环境反馈确认觉醒源头。",
        "特效": "把奇观拍成有源头、路径和余波的物理事件。",
        "群像": "在多人场面里分清主次、秩序和谁在真正推动剧情。",
        "危险": "把未落下的危险维持在临界状态，逼下一拍承接。",
        "牺牲": "让主动顶上去的代价与守护对象一起进入画面逻辑。",
        "尾钩": "把最值钱的异常或未落动作停在尾帧，形成点击下一条的冲动。",
    }
    parts = [purpose_descriptions.get(primary_purpose, purpose_descriptions[FALLBACK_PURPOSE])]
    if title:
        parts.append("当前来源段落：" + title[:80])
    if summary:
        parts.append("核心事件：" + summary[:120])
    return " ".join(parts).strip()


def _derive_restored_duration_seconds(
    *,
    source_duration_seconds: float,
    shot_count: int,
    dialogue_window_count: int,
) -> float:
    source_duration = max(0.8, float(source_duration_seconds or 0))
    target = max(
        source_duration,
        6.4 + max(1, int(shot_count or 0)) * 1.1 + min(3, int(dialogue_window_count or 0)) * 0.6,
        8.0,
    )
    return round(min(15.0, target), 2)


def _derive_catalog_title(primary_purpose: str, group: Sequence[Mapping[str, Any]]) -> str:
    title = _combine_unique_texts(item.get("title") for item in group)
    if title:
        simplified = re.split(r"[：:]", title, maxsplit=1)[-1].strip()
        simplified = re.sub(r"\s*(?:vs|VS|Vs)\s*", " ", simplified)
        simplified = _normalize_spaces(simplified)
        if simplified:
            chunks = _split_freeform_story_chunks(simplified)
            if chunks:
                return chunks[0][:12]
            return simplified[:12]
    fallback_map = {
        "爱情": "高燃爱情",
        "思念": "记忆回流",
        "痛苦": "痛感落身",
        "告别": "转身离场",
        "守护": "挡在身前",
        "羞辱": "当众压迫",
        "反击": "冷脸反击",
        "报仇": "旧账清算",
        "对峙": "临界对峙",
        "揭示": "真相显影",
        "权力": "高位压场",
        "规则": "规则压顶",
        "觉醒": "异动觉醒",
        "特效": "奇观兑现",
        "群像": "群像压场",
        "危险": "危险逼近",
        "牺牲": "代价顶上",
        "尾钩": "尾钩卡点",
    }
    return fallback_map.get(primary_purpose, primary_purpose or "剧情推进")


def _derive_catalog_summary(group: Sequence[Mapping[str, Any]], transcript_text: str) -> str:
    summary = _combine_unique_texts(item.get("summary") for item in group)
    if summary:
        return summary[:90]
    transcript = _normalize_spaces(transcript_text)
    if transcript:
        return transcript[:90]
    title = _combine_unique_texts(item.get("title") for item in group)
    if title:
        return title[:90]
    return "围绕当前剧情点补足动作、关系和视觉落点。"


def _derive_entry_state(first_item: Mapping[str, Any], primary_purpose: str) -> str:
    focus = _best_line(first_item.get("visual_focus"), fallback=str(first_item.get("summary") or ""))
    if primary_purpose in {"羞辱", "守护", "危险"}:
        return "开场先确认压迫关系与动作接触点，重点给：" + focus
    if primary_purpose in {"爱情", "思念", "告别"}:
        return "开场先确认人物距离、目光方向或触发物，重点给：" + focus
    if primary_purpose in {"觉醒", "特效", "尾钩"}:
        return "开场先压小范围异常信号，重点给：" + focus
    return "开场先把当前剧情任务立住，重点给：" + focus


def _purpose_blocking_note(primary_purpose: str, scene_anchor: Mapping[str, Any]) -> str:
    if primary_purpose in {"羞辱", "权力", "规则"}:
        return "优先保持高低位和前后压迫关系，旁观层要明确在看谁。"
    if primary_purpose in {"守护", "危险", "牺牲"}:
        return "优先交代威胁源、保护对象与插入者的前后位，确保动作路径清楚。"
    if primary_purpose in {"群像", "对峙"}:
        return "优先用纵深、高差和中轴组织多人站位，收束 1-4 个核心主体。"
    if primary_purpose in {"爱情", "思念", "告别"}:
        return "优先稳定人物距离、身体朝向和离场/靠近方向，避免关系跳位。"
    if primary_purpose in {"觉醒", "特效", "尾钩"}:
        return "优先把源头和传播方向钉在同一空间逻辑里，别让异动失去位置。"
    return "优先让核心人物、触发动作和结果落点处在同一可追踪空间。"


def _strength_excerpt_for_purpose(primary_purpose: str, strength_playbook: Mapping[str, Any]) -> dict[str, list[str]]:
    if not strength_playbook:
        return {}
    selected_keys = {
        "镜头语言": "camera_language_rules",
        "分镜执行": "storyboard_execution_rules",
        "对白节奏": "dialogue_timing_rules",
        "连续性": "continuity_guardrails",
    }
    result: dict[str, list[str]] = {}
    for title, key in selected_keys.items():
        values = list(strength_playbook.get(key) or [])
        if values:
            result[title] = values[:3]
    return result


def _quality_score(beat: Mapping[str, Any]) -> float:
    score = 0.35
    duration = float(beat.get("restored_duration_seconds", beat.get("duration_seconds", 0)) or 0)
    if 8.0 <= duration <= 15.0:
        score += 0.15
    elif 6.0 <= duration <= 16.0:
        score += 0.08
    if len(list(beat.get("shot_chain") or [])) >= 2:
        score += 0.12
    if beat.get("dialogue_windows"):
        score += 0.08
    if beat.get("visual_focus_notes"):
        score += 0.08
    if beat.get("camera_language_notes"):
        score += 0.08
    if beat.get("scene_anchor", {}).get("scene_ids"):
        score += 0.06
    if beat.get("continuity_bridge_in") or beat.get("continuity_bridge_out"):
        score += 0.04
    if beat.get("restored_seedance_prompt"):
        score += 0.04
    return round(min(0.99, score), 2)


def _continuity_bridge_in(previous_beat: Mapping[str, Any] | None, current_beat: Mapping[str, Any]) -> str:
    if previous_beat is None:
        return "首拍直接落在本条最值钱的动作或关系锚点上，不重开无关空间。"
    return (
        f"承接上一拍的{previous_beat.get('primary_purpose', '')}尾态，"
        f"继续沿同一空间关系推进到{current_beat.get('primary_purpose', '')}目的。"
    )


def _continuity_bridge_out(current_beat: Mapping[str, Any], next_beat: Mapping[str, Any] | None) -> str:
    if next_beat is None:
        return "尾帧停在本条最强的未完成动作或异常信号上，作为本集卡点。"
    return (
        f"尾帧把画面收在可继续承接{next_beat.get('primary_purpose', '')}的触发物上，"
        "避免下一拍重新开戏。"
    )


def _render_restored_prompt(current_beat: Mapping[str, Any], next_beat: Mapping[str, Any] | None) -> str:
    shot_chain = list(current_beat.get("shot_chain") or [])
    if not shot_chain:
        return ""
    full_rendered = _render_full_restored_prompt(current_beat=current_beat, next_beat=next_beat, shot_chain=shot_chain)
    if full_rendered:
        return full_rendered
    attempts = [
        {"key_shots": DEFAULT_MAX_KEY_PROMPT_SHOTS, "non_key_detail": "lite", "include_audio": "edge", "include_continuity": True, "include_lighting": True},
        {"key_shots": DEFAULT_MAX_KEY_PROMPT_SHOTS, "non_key_detail": "lite", "include_audio": "edge", "include_continuity": False, "include_lighting": True},
        {"key_shots": 4, "non_key_detail": "micro", "include_audio": "edge", "include_continuity": False, "include_lighting": True},
        {"key_shots": 4, "non_key_detail": "micro", "include_audio": "none", "include_continuity": False, "include_lighting": False},
        {"key_shots": 3, "non_key_detail": "micro", "include_audio": "none", "include_continuity": False, "include_lighting": False},
    ]
    best = ""
    for attempt in attempts:
        rendered = _render_compact_restored_prompt(
            current_beat=current_beat,
            next_beat=next_beat,
            shot_chain=shot_chain,
            key_shots=int(attempt["key_shots"]),
            non_key_detail=str(attempt["non_key_detail"]),
            include_audio=str(attempt["include_audio"]),
            include_continuity=bool(attempt["include_continuity"]),
            include_lighting=bool(attempt["include_lighting"]),
        )
        if rendered and (not best or len(rendered) < len(best)):
            best = rendered
        if rendered and len(rendered) <= DEFAULT_RESTORED_PROMPT_CHAR_LIMIT:
            return rendered
    return best


def _render_generalized_template_prompt(current_beat: Mapping[str, Any], next_beat: Mapping[str, Any] | None) -> str:
    purpose = str(current_beat.get("primary_purpose") or FALLBACK_PURPOSE)
    profile = PURPOSE_PROFILES.get(purpose, PURPOSE_PROFILES[FALLBACK_PURPOSE])
    scene_summary = str(current_beat.get("scene_anchor", {}).get("summary") or "")
    shot_chain = list(current_beat.get("shot_chain") or [])
    if shot_chain:
        selected_shots = _select_prompt_shots(shot_chain, max_shots=4)
        parts = [
            _render_generalized_shot_clause(
                current_beat,
                shot,
                index=index,
                next_beat=next_beat if index == len(selected_shots) - 1 else None,
            )
            for index, shot in enumerate(selected_shots)
        ]
        if scene_summary:
            parts.append("空间保持【同一轴线、人物朝向与受光关系】")
        rendered = "；".join(part for part in parts if part).strip("；")
        return rendered
    next_purpose = str((next_beat or {}).get("primary_purpose") or "下一拍")
    parts = [str(profile.get("template_opening") or ""), str(profile.get("template_middle") or ""), str(profile.get("template_tail") or "")]
    if scene_summary:
        parts.append("空间组织参考：" + scene_summary)
    parts.append("下一拍通常承接到：" + next_purpose)
    return "".join(parts).strip()


def _render_full_restored_prompt(
    *,
    current_beat: Mapping[str, Any],
    next_beat: Mapping[str, Any] | None,
    shot_chain: Sequence[Mapping[str, Any]],
) -> str:
    clauses: list[str] = []
    for index, shot in enumerate(list(shot_chain or [])):
        parts = [
            f"{shot.get('time_range', '')}，"
            + _render_full_restored_visual_sentence(shot)
        ]
        if index == 0:
            context_sentence = _render_restored_context_sentence(current_beat)
            if context_sentence:
                parts.append(context_sentence)
        continuity = _render_background_continuity_sentence(current_beat, shot, index=index)
        if continuity:
            parts.append(_ensure_sentence_prefix(continuity, preferred_prefix="空间承接"))
        dialogue_sentence = _render_dialogue_sentence(current_beat, shot)
        if dialogue_sentence:
            parts.append(dialogue_sentence)
        audio_sentence = _render_sound_sentence(current_beat, shot, index=index)
        if audio_sentence:
            parts.append(audio_sentence)
        transition = _transition_sentence(current_beat, shot, index=index, next_beat=next_beat)
        if transition:
            parts.append(transition)
        clauses.append("；".join(part for part in parts if part))
    return "；".join(clause.strip("；") for clause in clauses if clause).strip("；")


def _render_full_restored_visual_sentence(shot: Mapping[str, Any]) -> str:
    summary = _normalize_spaces(str(shot.get("story_function") or ""))
    focus = _normalize_spaces(str(shot.get("visual_focus") or ""))
    camera = _normalize_spaces(str(shot.get("camera_language") or ""))
    camera_entry = _normalize_spaces(str(shot.get("camera_entry") or ""))
    blocking = _normalize_spaces(str(shot.get("subject_blocking") or ""))
    action_timeline = _strip_source_second_markers(str(shot.get("action_timeline") or ""))
    lighting_and_texture = _normalize_spaces(str(shot.get("lighting_and_texture") or ""))
    art = _normalize_spaces(str(shot.get("art_direction_hint") or ""))
    segments: list[str] = []
    entry = _full_camera_entry(camera_entry, camera)
    if entry:
        segments.append(entry)
    summary_text = _full_story_label(summary)
    if summary_text:
        segments.append(f"先立{summary_text}")
    if focus:
        segments.append(f"聚焦{focus}")
    if action_timeline:
        segments.append(f"动作{action_timeline}")
    if blocking:
        segments.append(blocking)
    lighting_summary = lighting_and_texture or art
    if lighting_summary:
        segments.append(_ensure_sentence_prefix(lighting_summary, preferred_prefix="光感"))
    return "，".join(_dedupe_strings(segment for segment in segments if segment)) or "镜头继续围绕当前事件推进"


def _render_compact_restored_prompt(
    *,
    current_beat: Mapping[str, Any],
    next_beat: Mapping[str, Any] | None,
    shot_chain: Sequence[Mapping[str, Any]],
    key_shots: int,
    non_key_detail: str,
    include_audio: str,
    include_continuity: bool,
    include_lighting: bool,
) -> str:
    ordered_shots = [dict(item) for item in shot_chain]
    if not ordered_shots:
        return ""
    key_indices = sorted(_select_key_shot_indices(ordered_shots, max_key_shots=key_shots))
    selected_shots = [ordered_shots[index] for index in key_indices]
    if not selected_shots:
        selected_shots = ordered_shots[: max(2, min(len(ordered_shots), key_shots))]
    clauses: list[str] = []
    seen_dialogue: set[str] = set()
    seen_audio: set[str] = set()
    for index, shot in enumerate(selected_shots):
        is_first = index == 0
        is_last = index == len(selected_shots) - 1
        detail_level = "full" if is_first or is_last else (non_key_detail if non_key_detail in {"lite", "micro"} else "lite")
        parts = [
            f"{shot.get('time_range', '')}，"
            + _render_restored_visual_sentence(
                shot,
                detail_level=detail_level,
                include_summary=is_first or is_last,
                include_blocking=is_first or is_last,
                include_lighting=include_lighting and (is_first or is_last),
            )
        ]
        if is_first:
            context_sentence = _render_restored_context_sentence(current_beat)
            if context_sentence:
                parts.append(context_sentence)
        if include_continuity and is_first:
            continuity = _render_compact_background_continuity(current_beat, shot, index=index)
            if continuity:
                parts.append(continuity)
        dialogue_sentence = _render_compact_dialogue_sentence(
            current_beat,
            shot,
            seen_dialogue=seen_dialogue,
            detail_level=detail_level,
        )
        if dialogue_sentence:
            parts.append(dialogue_sentence)
        if include_audio == "all" or (include_audio == "edge" and (is_first or is_last)):
            audio_hint = _render_compact_sound_sentence(current_beat, shot, index=index, seen_audio=seen_audio)
            if audio_hint:
                parts.append(audio_hint)
        transition = _render_compact_transition_sentence(
            current_beat=current_beat,
            shot=shot,
            selected_shots=selected_shots,
            index=index,
            next_beat=next_beat,
        )
        if transition:
            parts.append(transition)
        clauses.append("；".join(part for part in parts if part))
    return "；".join(clause.strip("；") for clause in clauses if clause).strip("；")


def _select_prompt_shots(shot_chain: Sequence[Mapping[str, Any]], *, max_shots: int) -> list[dict[str, Any]]:
    shots = [dict(item) for item in shot_chain]
    if len(shots) <= max_shots:
        return shots
    if max_shots <= 1:
        return [shots[0]]
    indices = sorted({round(index * (len(shots) - 1) / (max_shots - 1)) for index in range(max_shots)})
    return [shots[int(index)] for index in indices]


def _render_restored_visual_sentence(
    shot: Mapping[str, Any],
    *,
    detail_level: str,
    include_summary: bool,
    include_blocking: bool,
    include_lighting: bool,
) -> str:
    summary = str(shot.get("story_function") or "").strip()
    focus = str(shot.get("visual_focus") or "").strip()
    camera = str(shot.get("camera_language") or "").strip()
    camera_entry = str(shot.get("camera_entry") or "").strip()
    blocking = str(shot.get("subject_blocking") or "").strip()
    action_timeline = str(shot.get("action_timeline") or "").strip()
    lighting_and_texture = str(shot.get("lighting_and_texture") or "").strip()
    art = str(shot.get("art_direction_hint") or "").strip()
    segments = []
    entry = _compact_camera_entry(camera_entry, camera)
    if entry:
        segments.append(entry)
    if include_summary and summary:
        compact_summary = _compact_story_label(summary)
        if compact_summary:
            segments.append(f"先立{compact_summary}")
    if focus:
        focus_limit = 56 if detail_level == "full" else 40 if detail_level == "lite" else 28
        segments.append(f"聚焦{_truncate_prompt_fragment(focus, limit=focus_limit)}")
    action_summary = _compact_action_timeline(action_timeline)
    if action_summary:
        action_limit = 84 if detail_level == "full" else 60 if detail_level == "lite" else 36
        segments.append(f"动作{_truncate_prompt_fragment(action_summary, limit=action_limit)}")
    if include_blocking:
        blocking_summary = _compact_blocking(blocking)
        if blocking_summary:
            segments.append(blocking_summary)
    if include_lighting:
        lighting_summary = _compact_lighting(lighting_and_texture or art)
        if lighting_summary:
            segments.append(f"光感{lighting_summary}")
    return "，".join(_dedupe_strings(segment for segment in segments if segment)) or "镜头继续围绕当前事件推进"


def _select_key_shot_indices(shot_chain: Sequence[Mapping[str, Any]], *, max_key_shots: int) -> set[int]:
    shots = [dict(item) for item in shot_chain]
    if not shots:
        return set()
    if len(shots) <= max(2, max_key_shots):
        return set(range(len(shots)))
    ranked: list[tuple[float, int]] = []
    for index, shot in enumerate(shots):
        role_label = str(shot.get("role_label") or "").strip() or _generalized_role_label(shot)
        score = 0.0
        if index == 0 or index == len(shots) - 1:
            score += 100.0
        if role_label in {"开场建立", "尾帧收束"}:
            score += 45.0
        elif role_label in {"动作触发", "张力推进", "异动扩散", "信息落点"}:
            score += 30.0
        elif role_label in {"对白推进", "结果反应", "反应承接"}:
            score += 18.0
        if str(shot.get("dialogue_hint") or "").strip():
            score += 6.0
        if str(shot.get("transition_trigger") or "").strip():
            score += 4.0
        if str(shot.get("visual_focus") or "").strip():
            score += 2.0
        ranked.append((score, index))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = {0, len(shots) - 1}
    for _, index in ranked:
        selected.add(index)
        if len(selected) >= max(2, max_key_shots):
            break
    return selected


def _render_background_continuity_sentence(
    current_beat: Mapping[str, Any],
    shot: Mapping[str, Any],
    *,
    index: int,
) -> str:
    continuity = str(shot.get("background_continuity") or "").strip()
    if continuity:
        return continuity if continuity.startswith("画面继续") or continuity.startswith("后景继续") else f"画面继续沿用{continuity}"
    scene_summary = str(current_beat.get("scene_anchor", {}).get("summary") or "").strip()
    if scene_summary and index == 0:
        return f"画面继续沿用当前 beat 的空间结构与受光关系，参考{scene_summary}"
    return ""


def _render_compact_background_continuity(current_beat: Mapping[str, Any], shot: Mapping[str, Any], *, index: int) -> str:
    continuity = str(shot.get("background_continuity") or "").strip()
    if continuity:
        return "空间承接" + _truncate_prompt_fragment(continuity, limit=48)
    scene_summary = str(current_beat.get("scene_anchor", {}).get("summary") or "").strip()
    if "OCR 锚点：" in scene_summary:
        scene_summary = scene_summary.split("OCR 锚点：", 1)[0].rstrip("； ")
    if scene_summary and index == 0:
        return "空间保持" + _truncate_prompt_fragment(scene_summary, limit=42)
    return ""


def _clean_restored_context_text(text: str) -> str:
    normalized = _normalize_spaces(str(text or ""))
    if not normalized:
        return ""
    normalized = re.sub(
        r"^(?:实际拍摄为|实际呈现为|该beat采用|该 beat 采用|该段采用|该段实际呈现为|这一段实际呈现为|此段采用)",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip("：:，,。；; ")
    normalized = normalized.replace("，非", "；非")
    fragments = [
        fragment.strip()
        for fragment in re.split(r"[。]", normalized)
        if fragment.strip()
    ]
    cleaned: list[str] = []
    for fragment in fragments:
        if "镜头通过" in fragment and len(cleaned) >= 1:
            continue
        cleaned.append(fragment)
        if len(cleaned) >= 2:
            break
    return "；".join(cleaned)


def _render_restored_context_sentence(current_beat: Mapping[str, Any]) -> str:
    summary = _clean_restored_context_text(
        str(current_beat.get("display_summary") or current_beat.get("beat_summary") or current_beat.get("display_title") or "")
    )
    if not summary:
        return ""
    return "人物与环境先立住：" + _truncate_prompt_fragment(summary, limit=92)


def _render_dialogue_sentence(current_beat: Mapping[str, Any], shot: Mapping[str, Any]) -> str:
    dialogue_hint = str(shot.get("dialogue_hint") or "").strip()
    dialogue_timing = str(shot.get("dialogue_timing") or "").strip()
    dialogue_hint = _truncate_prompt_fragment(dialogue_hint, limit=56)
    dialogue_prefix = _dialogue_role_prefix(current_beat, dialogue_hint)
    if dialogue_hint and dialogue_timing:
        return f"{dialogue_prefix}{dialogue_timing}落“{dialogue_hint}”"
    if dialogue_hint:
        return f"{dialogue_prefix}落“{dialogue_hint}”"
    if dialogue_timing:
        return f"{dialogue_prefix}{dialogue_timing}推进"
    return ""


def _looks_like_narration_line(text: str) -> bool:
    normalized = _normalize_spaces(str(text or ""))
    if not normalized:
        return False
    narration_markers = ("我是", "我叫", "可在", "直到", "之前", "那天", "后来", "原来", "却有", "给自己", "从那以后")
    return any(marker in normalized for marker in narration_markers) and all(token not in normalized for token in ("你", "吗", "？", "?"))


def _dialogue_role_prefix(current_beat: Mapping[str, Any], dialogue_hint: str) -> str:
    normalized = _normalize_spaces(dialogue_hint)
    if _looks_like_narration_line(normalized):
        return "旁白"
    purpose = str(current_beat.get("primary_purpose") or "")
    if purpose in {"羞辱", "权力", "规则"}:
        return "施压者"
    if purpose in {"对峙", "反击", "守护", "危险"}:
        return "主位人物"
    if purpose in {"思念", "爱情", "告别"}:
        return "人物低声"
    return "人物"


def _render_compact_dialogue_sentence(
    current_beat: Mapping[str, Any],
    shot: Mapping[str, Any],
    *,
    seen_dialogue: set[str],
    detail_level: str,
) -> str:
    dialogue_hint_limit = 48 if detail_level == "full" else 44 if detail_level == "lite" else 24
    dialogue_hint = _truncate_prompt_fragment(str(shot.get("dialogue_hint") or "").strip(), limit=dialogue_hint_limit)
    dialogue_timing = _compact_dialogue_timing(str(shot.get("dialogue_timing") or "").strip())
    if "无对白" in dialogue_hint:
        dialogue_hint = ""
    if dialogue_timing not in {"", "前段", "中段", "后段", "无对白"}:
        dialogue_timing = ""
    if detail_level == "micro" and dialogue_timing not in {"", "无对白"}:
        dialogue_timing = ""
    marker = dialogue_hint or dialogue_timing
    if marker in seen_dialogue:
        return ""
    if dialogue_hint or dialogue_timing:
        seen_dialogue.add(marker)
    dialogue_prefix = _dialogue_role_prefix(current_beat, dialogue_hint)
    if dialogue_timing == "无对白":
        return "无对白，靠动作推进"
    if dialogue_hint and dialogue_timing:
        return f"{dialogue_prefix}{dialogue_timing}落“{dialogue_hint}”"
    if dialogue_hint:
        return f"{dialogue_prefix}落“{dialogue_hint}”"
    if dialogue_timing:
        return f"{dialogue_prefix}{dialogue_timing}推进"
    return ""


def _truncate_prompt_fragment(text: str, *, limit: int) -> str:
    normalized = _normalize_spaces(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(8, limit - 1)].rstrip() + "…"


def _render_sound_sentence(current_beat: Mapping[str, Any], shot: Mapping[str, Any], *, index: int) -> str:
    sound_bed = str(shot.get("sound_bed") or "").strip()
    if not sound_bed:
        sound_bed = _infer_audio_hint(current_beat, shot, index=index)
    if not sound_bed:
        return ""
    return f"同时听见{sound_bed}"


def _render_compact_sound_sentence(
    current_beat: Mapping[str, Any],
    shot: Mapping[str, Any],
    *,
    index: int,
    seen_audio: set[str],
) -> str:
    sound_bed = str(shot.get("sound_bed") or "").strip()
    if not sound_bed:
        sound_bed = _infer_audio_hint(current_beat, shot, index=index)
    dedupe_key = _normalize_spaces(sound_bed)
    compact_sound = _truncate_prompt_fragment(sound_bed, limit=32)
    if not compact_sound or dedupe_key in seen_audio:
        return ""
    seen_audio.add(dedupe_key)
    return f"声场保留{compact_sound}"


def _render_generalized_shot_clause(
    current_beat: Mapping[str, Any],
    shot: Mapping[str, Any],
    *,
    index: int,
    next_beat: Mapping[str, Any] | None,
) -> str:
    purpose = str(current_beat.get("primary_purpose") or FALLBACK_PURPOSE)
    profile = PURPOSE_PROFILES.get(purpose, PURPOSE_PROFILES[FALLBACK_PURPOSE])
    slots = list(profile.get("required_slots") or [])
    role_label = _generalized_role_label(shot)
    lead_slot = slots[0] if slots else "【主体】"
    support_slot = slots[1] if len(slots) >= 2 else "【对手/受体】"
    tail_slot = slots[2] if len(slots) >= 3 else "【触发物/后景锚点】"
    parts = [
        f"{shot.get('time_range', '')}，镜头从【{_generalized_camera_entry(role_label, purpose)}】切入",
        f"主体站位保持【{_generalized_blocking(role_label, purpose, lead_slot, support_slot, tail_slot)}】",
        f"动作按【{_generalized_action_timeline(role_label, purpose, lead_slot, support_slot, tail_slot)}】推进",
        f"受光与质感重点给【{_generalized_lighting(purpose)}】",
        "画面继续沿用【同一空间结构、人物场位与受光关系】",
    ]
    if str(shot.get("dialogue_hint") or "").strip():
        parts.append(f"对白安排为【{_generalized_dialogue_timing(role_label)}】")
    parts.append(f"同时听见【{_generalized_sound_bed(purpose, role_label)}】")
    transition = _generalized_transition_sentence(role_label=role_label, next_beat=next_beat)
    if transition:
        parts.append(transition)
    return "；".join(part for part in parts if part)


def _generalized_role_label(shot: Mapping[str, Any]) -> str:
    role_label = str(shot.get("role_label") or "").strip()
    if role_label:
        return role_label
    story_function = str(shot.get("story_function") or "").strip()
    if "·" in story_function:
        return _normalize_spaces(story_function.rsplit("·", 1)[-1])
    return "中段推进"


def _generalized_camera_entry(role_label: str, purpose: str) -> str:
    if role_label == "开场建立":
        if purpose in {"羞辱", "权力", "规则"}:
            return "高低位关系最清楚的一侧"
        if purpose in {"群像", "对峙"}:
            return "能同时看清双方轴线与主位人物的位置"
        return "空间锚点与主位主体最清楚的位置"
    if role_label in {"动作触发", "张力推进", "异动扩散"}:
        return "动作发起方向的前侧或肩侧"
    if role_label in {"结果反应", "反应承接"}:
        return "受体或听者一侧的承接位"
    if role_label == "尾帧收束":
        return "动作结果与尾帧触发物的收束位"
    return "当前主体最清楚的一侧"


def _generalized_blocking(role_label: str, purpose: str, lead_slot: str, support_slot: str, tail_slot: str) -> str:
    if purpose in {"羞辱", "权力", "规则"}:
        return f"{lead_slot}占上位或前景，{support_slot}留在低位承压，{tail_slot}保留在后景作为秩序锚点"
    if purpose in {"守护", "危险", "牺牲"}:
        return f"{lead_slot}沿路径插入，{support_slot}与威胁源保持前后位，{tail_slot}继续留在可见位置"
    if purpose in {"群像", "对峙"}:
        return f"{lead_slot}与{support_slot}分占轴线两侧或纵深前后位，{tail_slot}作为场面压力层保留"
    if role_label == "尾帧收束":
        return f"{lead_slot}占住主位，{support_slot}或{tail_slot}退到后景继续保留连续性"
    return f"{lead_slot}占主位，{support_slot}留在侧后或对切位，{tail_slot}留在后景支撑连续性"


def _generalized_action_timeline(role_label: str, purpose: str, lead_slot: str, support_slot: str, tail_slot: str) -> str:
    if role_label == "开场建立":
        return f"先把{lead_slot}与{support_slot}的关系立住，再给{tail_slot}或起势动作入画"
    if role_label in {"动作触发", "张力推进", "异动扩散"}:
        return f"先给{lead_slot}的动作起点，再推进到峰值，最后把结果落到{support_slot}或{tail_slot}"
    if role_label in {"结果反应", "反应承接"}:
        return f"先承接上一拍结果，再把反应落到{support_slot}或环境余波上"
    if role_label == "尾帧收束":
        return f"让{lead_slot}的结果继续推进半拍，最后停在{tail_slot}这个未完触发物上"
    if purpose in {"特效", "觉醒"}:
        return f"先压住{tail_slot}这一异常源头，再让变化扩散，最后回到{lead_slot}或{support_slot}"
    return f"先给{lead_slot}起势，再把动作与反应一起推进，最后把结果落到{support_slot}"


def _generalized_lighting(purpose: str) -> str:
    if purpose in {"特效", "觉醒", "尾钩"}:
        return "主光与异常辉光对人物、器物和周边材质的受光变化"
    if purpose in {"羞辱", "权力", "规则", "危险"}:
        return "冷硬侧光、受压皮肤、衣料摩擦与石面/金属面质感"
    if purpose in {"爱情", "思念", "告别"}:
        return "人物皮肤、衣摆和环境边缘光的柔和层次"
    return "同一光向下的人物质感、环境空气感与关键物件反光"


def _generalized_dialogue_timing(role_label: str) -> str:
    if role_label == "开场建立":
        return "先压一拍表情或手位，再把第一句推出"
    if role_label == "尾帧收束":
        return "最后一句压在动作落点后，说完留半拍"
    return "让对白跟着动作推进，句尾留给承接反应"


def _generalized_sound_bed(purpose: str, role_label: str) -> str:
    if purpose in {"特效", "觉醒", "尾钩"}:
        return "低频嗡鸣、风压变化与环境杂音被主动抽空后的异常底噪"
    if purpose in {"羞辱", "权力", "规则"}:
        return "风声压底、衣料摩擦、空间回响与近距离压迫细响"
    if purpose in {"守护", "危险", "牺牲"}:
        return "脚步、闷响、粗喘与受力摩擦声"
    if purpose in {"爱情", "思念", "告别"}:
        return "呼吸、衣摆和被留白的环境声"
    if role_label == "尾帧收束":
        return "最值钱的环境声或器物低鸣"
    return "近景动作声与环境底噪"


def _infer_audio_hint(current_beat: Mapping[str, Any], shot: Mapping[str, Any], *, index: int) -> str:
    purpose = str(current_beat.get("primary_purpose") or FALLBACK_PURPOSE)
    text_blob = " ".join(
        [
            str(shot.get("story_function") or ""),
            str(shot.get("visual_focus") or ""),
            str(shot.get("camera_language") or ""),
            str(shot.get("sound_bed") or ""),
        ]
    )
    if any(token in text_blob for token in ("刀", "剑", "刃")):
        return "刀刃轻鸣、鞋底蹬地和压住呼吸的细响"
    if any(token in text_blob for token in ("雷", "光", "法阵", "镜门", "异变")):
        return "低频嗡鸣、风压变化和石屑轻颤"
    if purpose in {"羞辱", "权力", "规则"}:
        return "风声压底、衣料轻摩和空间回响"
    if purpose in {"守护", "危险", "牺牲"}:
        return "急促脚步、粗喘和身体受力的闷响"
    if purpose in {"爱情", "思念", "告别"}:
        return "呼吸、衣摆和环境留白形成的轻声场"
    if purpose in {"觉醒", "尾钩"} and index == len(list(current_beat.get("shot_chain") or [])) - 1:
        return "异常心跳、器物低鸣或几乎被抽空的环境音"
    return ""


def _transition_sentence(
    current_beat: Mapping[str, Any],
    shot: Mapping[str, Any],
    *,
    index: int,
    next_beat: Mapping[str, Any] | None,
) -> str:
    shot_chain = list(current_beat.get("shot_chain") or [])
    trigger = str(shot.get("transition_trigger") or "").strip()
    if index < len(shot_chain) - 1:
        next_shot = shot_chain[index + 1]
        if trigger:
            return f"{trigger}，把镜头顺势带向{next_shot.get('story_function', '下一动作')}"
        return f"这一拍把镜头顺势带向{next_shot.get('story_function', '下一动作')}"
    if next_beat is None:
        if trigger:
            return f"{trigger}，尾帧停住，留出本集最后一下未完成的悬念"
        return "尾帧停住，留出本集最后一下未完成的悬念"
    if trigger:
        return f"{trigger}，把触发物交给下一拍的{next_beat.get('primary_purpose', '后续推进')}"
    return f"尾帧把触发物交给下一拍的{next_beat.get('primary_purpose', '后续推进')}"


def _render_compact_transition_sentence(
    *,
    current_beat: Mapping[str, Any],
    shot: Mapping[str, Any],
    selected_shots: Sequence[Mapping[str, Any]],
    index: int,
    next_beat: Mapping[str, Any] | None,
) -> str:
    trigger = _compact_transition_trigger(str(shot.get("transition_trigger") or ""))
    if index < len(selected_shots) - 1:
        next_shot = selected_shots[index + 1]
        next_label = _compact_story_label(str(next_shot.get("story_function") or ""))
        if trigger and next_label:
            return f"{trigger}，镜头顺势切到{next_label}"
        if trigger:
            return trigger
        return ""
    if next_beat is None:
        return "尾帧留悬"
    if trigger:
        return f"{trigger}，尾帧把钩子交给下一拍"
    return f"尾帧钩到下一拍{str(next_beat.get('primary_purpose') or '推进')}"


def _ensure_sentence_prefix(text: str, *, preferred_prefix: str) -> str:
    normalized = _normalize_spaces(text)
    if not normalized:
        return ""
    if normalized.startswith(preferred_prefix):
        return normalized
    return preferred_prefix + normalized


def _full_camera_entry(camera_entry: str, camera: str) -> str:
    raw = _normalize_spaces(camera_entry or "")
    if raw:
        if raw.startswith("镜头"):
            return raw
        if raw.startswith("从"):
            return "镜头" + raw
        return "镜头从" + raw
    camera = _normalize_spaces(camera or "")
    if not camera:
        return ""
    if camera.startswith("镜头"):
        return camera
    return "镜头" + camera


def _full_story_label(text: str) -> str:
    normalized = _normalize_spaces(str(text or ""))
    if "·" in normalized:
        normalized = normalized.rsplit("·", 1)[-1]
    for prefix in (
        "开场建立：",
        "动作触发：",
        "张力推进：",
        "反应承接：",
        "尾帧收束：",
        "信息落点：",
        "空间展开：",
        "世界切换：",
    ):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    if normalized in GENERIC_SHOT_ROLE_LABELS:
        return ""
    return normalized


def _compact_camera_entry(camera_entry: str, camera: str) -> str:
    raw = _normalize_spaces(camera_entry or "")
    if raw:
        if raw.startswith("镜头"):
            return _truncate_prompt_fragment(raw, limit=42)
        if raw.startswith("从"):
            return "镜头" + _truncate_prompt_fragment(raw, limit=38)
        return "镜头从" + _truncate_prompt_fragment(raw, limit=34)
    camera = _normalize_spaces(camera or "")
    if not camera:
        return ""
    return "镜头" + _truncate_prompt_fragment(camera, limit=38)


def _compact_story_label(text: str) -> str:
    normalized = _normalize_spaces(str(text or ""))
    if "·" in normalized:
        normalized = normalized.rsplit("·", 1)[-1]
    for prefix in ("开场建立：", "动作触发：", "张力推进：", "反应承接：", "尾帧收束：", "信息落点：", "空间展开：", "世界切换："):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    if normalized in GENERIC_SHOT_ROLE_LABELS:
        return ""
    return _truncate_prompt_fragment(normalized, limit=24)


def _compact_action_timeline(text: str) -> str:
    normalized = _strip_source_second_markers(str(text or ""))
    if not normalized:
        return ""
    normalized = normalized.replace("起点：", "").replace("推进：", "→").replace("峰值：", "→")
    normalized = normalized.replace("结果：", "→").replace("持续：", "→")
    normalized = re.sub(r"[；;,，]\s*→", "→", normalized)
    normalized = re.sub(r"→+", "→", normalized)
    normalized = re.sub(r"\s*→\s*", "→", normalized)
    return _truncate_prompt_fragment(normalized, limit=64)


def _strip_source_second_markers(text: str) -> str:
    normalized = _normalize_spaces(str(text or ""))
    if not normalized:
        return ""
    normalized = re.sub(
        r"[（(]\s*\d+(?:\.\d+)?\s*(?:[—\-–~到]\s*\d+(?:\.\d+)?)?\s*s\s*[)）]",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"(?<![\d.])\d+(?:\.\d+)?\s*(?:[—\-–~到]\s*\d+(?:\.\d+)?)?\s*s(?![\w])",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[，；,;]\s*[，；,;]", "，", normalized)
    return _normalize_spaces(normalized.strip(" ，；,;"))


def _compact_blocking(text: str) -> str:
    normalized = _normalize_spaces(str(text or ""))
    if not normalized:
        return ""
    for chunk in re.split(r"[；。]", normalized):
        candidate = chunk.strip()
        if any(token in candidate for token in ("前景", "中景", "后景", "居", "轴线", "左", "右", "纵深", "高位", "低位", "中轴")):
            return _truncate_prompt_fragment(candidate, limit=68)
    return _truncate_prompt_fragment(normalized, limit=56)


def _compact_lighting(text: str) -> str:
    normalized = _normalize_spaces(str(text or ""))
    if not normalized:
        return ""
    return _truncate_prompt_fragment(normalized, limit=40)


def _compact_dialogue_timing(text: str) -> str:
    normalized = _normalize_spaces(str(text or ""))
    if not normalized:
        return ""
    normalized = normalized.removeprefix("对白").removeprefix("台词").strip("：:，, ")
    if "无对白" in normalized or "纯动作" in normalized:
        return "无对白"
    if any(token in normalized for token in ("前段", "前1/3", "先压")):
        return "前段"
    if "中段" in normalized:
        return "中段"
    if any(token in normalized for token in ("后段", "句尾", "尾", "收尾")):
        return "后段"
    return _truncate_prompt_fragment(normalized, limit=8)


def _compact_transition_trigger(text: str) -> str:
    normalized = _normalize_spaces(str(text or ""))
    if not normalized or _is_generic_transition_hint(normalized):
        return ""
    return _truncate_prompt_fragment(normalized, limit=30)


def _generalized_transition_sentence(*, role_label: str, next_beat: Mapping[str, Any] | None) -> str:
    if role_label == "尾帧收束":
        if next_beat is None:
            return "尾帧停在【未落动作/异常信号】上，给出本集最后一拍悬念"
        return f"尾帧把【触发物】交给下一拍的【{next_beat.get('primary_purpose', '后续推进')}】"
    return "由【对白句尾/手位变化/视线落点/物件显露/结果反应】触发切向下一镜"


def _template_id(purpose: str, episode_id: str, beat_id: str) -> str:
    return f"{_slugify(purpose)}__{episode_id}__{beat_id}"


def _derive_template_retrieval_title(beat: Mapping[str, Any]) -> str:
    purpose = _normalize_spaces(str(beat.get("primary_purpose") or "")) or FALLBACK_PURPOSE
    candidates = [
        str(beat.get("display_summary") or ""),
        str(beat.get("display_title") or ""),
        "；".join(str(item or "") for item in list(beat.get("source_titles") or [])[:2]),
        str(beat.get("dramatic_goal") or ""),
    ]
    fragments: list[str] = []
    for candidate in candidates:
        for chunk in _split_freeform_story_chunks(candidate):
            text = _normalize_template_description_chunk(chunk)
            if not text or text in fragments:
                continue
            fragments.append(text)
            if len(fragments) >= 2:
                break
        if len(fragments) >= 2:
            break
    if not fragments:
        fallback_title = _normalize_spaces(str(beat.get("display_title") or "")) or purpose
        return _truncate_prompt_fragment(fallback_title, limit=24)
    return _truncate_prompt_fragment("，".join(fragments[:2]), limit=30)


def _derive_template_retrieval_summary(beat: Mapping[str, Any]) -> str:
    summary = _normalize_template_description_chunk(str(beat.get("display_summary") or beat.get("beat_summary") or ""))
    dramatic_goal = _normalize_template_description_chunk(str(beat.get("dramatic_goal") or ""))
    scene_summary = _normalize_template_description_chunk(str(dict(beat.get("scene_anchor") or {}).get("summary") or ""))
    when_to_use = _normalize_template_description_chunk(str(beat.get("when_to_use") or ""))
    parts: list[str] = []
    if summary:
        parts.append(f"重点拍 {summary}。")
    if dramatic_goal:
        parts.append(f"镜头目标是 {dramatic_goal}。")
    elif scene_summary:
        parts.append(f"场面通常落在 {scene_summary}。")
    if when_to_use:
        parts.append(f"适用场景：{when_to_use}")
    return _normalize_spaces(" ".join(parts[:3]))


def _normalize_template_description_chunk(text: str) -> str:
    normalized = _normalize_spaces(text)
    if not normalized:
        return ""
    normalized = re.sub(r"\s*(当前来源段落|核心事件|剧情概述|叙事目标|适用场景)[:：].*$", "", normalized)
    normalized = re.sub(r"^(当前来源段落|核心事件|剧情概述|叙事目标|适用场景)[:：]\s*", "", normalized)
    normalized = re.sub(r"^该beat(?:实际为|实为|以|通过|围绕|将|把)?", "", normalized).strip(" ：:，。；;")
    normalized = re.sub(r"^(在一个 beat 内|围绕当前剧情点)", "", normalized).strip(" ，。；;")
    if not normalized:
        return ""
    if normalized in {
        "围绕当前剧情点补足动作、关系和视觉落点",
        "剧情推进",
        FALLBACK_PURPOSE,
    }:
        return ""
    return normalized[:36]


def _normalize_template_search_fragment(text: str) -> str:
    normalized = _normalize_template_description_chunk(text)
    if not normalized:
        return ""
    normalized = re.sub(r"\s*(OCR 锚点|镜头覆盖)[:：].*$", "", normalized)
    normalized = re.sub(r"^适合", "", normalized).strip(" ，。；;：:、")
    if "：" in normalized:
        head, tail = normalized.split("：", 1)
        if len(head) <= 10 and tail:
            normalized = tail
    normalized = normalized.strip(" ，。；;：:、")
    if not normalized or normalized.startswith("scene-"):
        return ""
    if normalized in {"剧情推进", FALLBACK_PURPOSE}:
        return ""
    return normalized[:24].rstrip(" ，。；;：:、")


def _extract_template_search_terms(text: str, *, limit: int) -> list[str]:
    normalized = _normalize_spaces(text)
    if not normalized:
        return []
    candidates: list[str] = []
    chunks = _split_freeform_story_chunks(normalized)
    if not chunks:
        chunks = [normalized]
    for chunk in chunks:
        for part in re.split(r"\s*(?:\+|｜|\||、|，|,|/)\s*", chunk):
            cleaned = _normalize_template_search_fragment(part)
            if not cleaned:
                continue
            candidates.append(cleaned)
            if len(candidates) >= limit * 2:
                break
        if len(candidates) >= limit * 2:
            break
    return _dedupe_strings(candidates)[:limit]


def _template_story_outline_terms(source: Mapping[str, Any], *, max_total: int) -> list[str]:
    terms: list[str] = []
    for outline in list(source.get("shot_outline") or []):
        text = _normalize_spaces(str(outline or ""))
        if not text:
            continue
        _, _, remainder = text.partition("｜")
        terms.extend(_extract_template_search_terms(remainder or text, limit=1))
        if len(terms) >= max_total:
            break
    return _dedupe_strings(terms)[:max_total]


def _template_shot_terms(
    source: Mapping[str, Any],
    key: str,
    *,
    per_shot_limit: int,
    max_total: int,
) -> list[str]:
    shots = list(source.get("shot_chain") or [])
    if not shots:
        if key == "story_function":
            return _template_story_outline_terms(source, max_total=max_total)
        return []
    terms: list[str] = []
    for shot in shots:
        terms.extend(_extract_template_search_terms(str(shot.get(key) or ""), limit=per_shot_limit))
        if len(terms) >= max_total:
            break
    return _dedupe_strings(terms)[:max_total]


TEMPLATE_SCENE_TAG_RULES: dict[str, Sequence[str]] = {
    "殿堂审判": ("殿", "神宫", "凌霄殿", "高台", "群仙", "天帝", "神尊", "审判", "庭审", "宣判"),
    "拍卖囚笼": ("拍卖", "竞拍", "展柜", "囚笼", "锁链", "主持人", "竞价牌"),
    "战场废墟": ("战后", "废墟", "营地", "烟尘", "残旗", "甲士", "血迹", "兵戈"),
    "庭院台阶": ("庭院", "台阶", "门楼", "殿前", "阶前", "石阶", "前庭"),
    "枫林秘境": ("红枫", "枫林", "幻境", "秘境", "红叶", "雾气", "溪涧"),
    "夜街车旁": ("夜街", "街边", "车旁", "车门", "夜色", "路灯"),
    "室内宴席": ("客厅", "室内", "宴席", "闺房", "包厢", "内室"),
}

TEMPLATE_RELATION_TAG_RULES: dict[str, Sequence[str]] = {
    "命定吸引": ("命定", "凝视", "靠近", "情愫", "暧昧", "鼻尖", "将触未触"),
    "公开审压": ("当众", "公开", "群仙", "众人", "围观", "审判", "宣判", "逼问"),
    "母族施压": ("母亲", "养母", "家主", "宗门长辈", "弟弟", "未婚夫", "家族"),
    "强弱对峙": ("对峙", "质问", "逼近", "互看", "停住", "沉默张力"),
    "护主承压": ("守护", "护", "替", "挡", "救", "替他", "替她", "护主"),
    "群体围观": ("围观", "群臣", "众仙", "宾客", "全场", "多人"),
    "权力俯压": ("高位", "帝", "王", "命令", "下令", "资格", "裁判", "神尊"),
}

TEMPLATE_STAGING_TAG_RULES: dict[str, Sequence[str]] = {
    "中轴高低位": ("中轴", "高位", "低位", "高台", "台阶", "高差"),
    "前后景遮挡": ("前景", "后景", "遮挡", "门框", "肩后", "缝隙"),
    "双人近距": ("双人", "靠近", "俯身", "对面", "鼻尖", "肩后"),
    "群像压场": ("群像", "全场", "多人", "队伍", "群臣", "众仙", "宾客"),
    "跪姿受压": ("跪", "单膝", "双膝", "跪地", "俯首"),
    "动作触发切镜": ("触发", "切至", "顺延", "承接", "动作推进", "尾帧"),
    "仪式绑定": ("法器", "红线", "符文", "结印", "共鸣", "绑定"),
}

TEMPLATE_CAMERA_TAG_RULES: dict[str, Sequence[str]] = {
    "特写微反应": ("特写", "极近", "面部", "眼神", "瞳孔", "嘴角", "指尖"),
    "肩后视角": ("肩后", "右肩后", "左肩后", "越肩"),
    "高位压拍": ("高位", "俯角", "俯拍", "45度高位"),
    "低机位仰拍": ("低角度", "仰拍", "地面", "贴近地面"),
    "推近压迫": ("推入", "慢推", "推进", "拉近"),
    "横移跟拍": ("横移", "跟拍", "跟移", "滑入"),
    "视线承接": ("视线落点", "顺延切入", "承接上一镜", "视线方向"),
    "全景建立": ("大全景", "广角", "全景", "空间建立"),
}

TEMPLATE_EMOTION_TAG_RULES: dict[str, Sequence[str]] = {
    "克制暧昧": ("克制", "暧昧", "凝视", "情感张力", "未说破"),
    "羞辱压迫": ("羞辱", "不配", "丑陋", "贱", "踩头", "处刑"),
    "痛感崩塌": ("痛", "濒死", "崩溃", "绝望", "血", "受虐"),
    "权威宣判": ("宣判", "命令", "资格", "规则", "裁定"),
    "认知揭示": ("真相", "揭示", "曝光", "认知", "身份", "物证"),
    "觉醒异动": ("觉醒", "异动", "共鸣", "发光", "灵力", "苏醒"),
    "危险临界": ("危险", "追杀", "威胁", "悬刀", "死令", "临界"),
    "尾钩悬念": ("尾钩", "未落", "黑屏", "下一秒", "即将", "悬念"),
}

TEMPLATE_NARRATIVE_TAG_RULES: dict[str, Sequence[str]] = {
    "身份揭晓": ("身份", "真千金", "原来", "曝光", "揭晓"),
    "关系推进": ("靠近", "凝视", "命定", "关系推进", "未说破"),
    "公开对质": ("公开", "当众", "质问", "逼问", "对峙"),
    "规则宣告": ("规则", "资格", "不准", "必须", "宣示"),
    "反击翻盘": ("反击", "打脸", "回怼", "翻盘", "冷笑"),
    "代价显形": ("代价", "受虐", "濒死", "献祭", "牺牲"),
    "仪式绑定": ("法器", "红线", "共鸣", "绑定", "结印"),
    "尾帧交棒": ("尾帧", "交给下一拍", "留念", "余波", "悬停"),
}


def _rank_rule_tags(
    text: str,
    rules: Mapping[str, Sequence[str]],
    *,
    limit: int,
) -> list[str]:
    normalized = _normalize_spaces(text)
    if not normalized:
        return []
    scored: list[tuple[int, int, str]] = []
    for tag, keywords in rules.items():
        hits = [word for word in keywords if word and word in normalized]
        if not hits:
            continue
        longest = max((len(word) for word in hits), default=0)
        scored.append((len(hits), longest, tag))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [tag for _, _, tag in scored[:limit]]


def _derive_template_classification_metadata(
    source: Mapping[str, Any],
    *,
    series_name: str = "",
) -> dict[str, Any]:
    declared_purpose = _normalize_spaces(
        str(source.get("primary_purpose") or source.get("purpose") or "")
    ) or FALLBACK_PURPOSE
    content_text = _normalize_spaces(
        " ".join(
            [
                str(source.get("dramatic_goal") or ""),
                str(source.get("scene_anchor_summary") or dict(source.get("scene_anchor") or {}).get("summary") or ""),
                str(source.get("display_summary") or source.get("beat_summary") or ""),
                str(source.get("display_title") or ""),
                str(source.get("retrieval_title") or ""),
                str(source.get("retrieval_summary") or ""),
                str(source.get("search_text") or ""),
                " ".join(str(item or "") for item in list(source.get("shot_outline") or [])),
            ]
        )
    )
    support_text = _normalize_spaces(
        " ".join(
            [
                str(source.get("when_to_use") or ""),
                str(source.get("search_hint") or ""),
            ]
        )
    )
    ranked_purposes = purpose_score_breakdown(content_text or support_text, is_last=declared_purpose == "尾钩")
    primary_purpose = declared_purpose
    top_ranked_purpose = str(ranked_purposes[0].get("purpose") or primary_purpose) if ranked_purposes else primary_purpose
    top_score = float(ranked_purposes[0].get("score") or 0.0) if ranked_purposes else 0.0
    if top_ranked_purpose in PURPOSE_ORDER and top_score >= 0.8:
        primary_purpose = top_ranked_purpose

    second_score = float(ranked_purposes[1].get("score") or 0.0) if len(ranked_purposes) > 1 else 0.0
    score_gap = max(0.0, top_score - second_score)
    normalized_top = min(top_score / 6.0, 1.0)
    normalized_gap = min(score_gap / 3.0, 1.0)
    source_match_bonus = 0.12 if declared_purpose == primary_purpose else -0.06
    classification_confidence = round(
        max(0.35, min(0.98, 0.46 + normalized_top * 0.24 + normalized_gap * 0.18 + source_match_bonus)),
        2,
    )

    secondary_purposes: list[str] = []
    threshold = max(1.2, top_score * 0.42)
    for item in ranked_purposes:
        purpose = str(item.get("purpose") or "").strip()
        score = float(item.get("score") or 0.0)
        if not purpose or purpose == primary_purpose:
            continue
        if len(secondary_purposes) >= 3:
            break
        if score >= threshold:
            secondary_purposes.append(purpose)
    if declared_purpose and declared_purpose != primary_purpose and declared_purpose not in secondary_purposes:
        secondary_purposes.insert(0, declared_purpose)
    secondary_purposes = _dedupe_strings(secondary_purposes)[:3]

    ambiguity_note = ""
    if declared_purpose != primary_purpose:
        ambiguity_note = f"原始大类标记为「{declared_purpose}」，但内容更接近「{primary_purpose}」。"
    elif second_score and score_gap <= 0.75 and secondary_purposes:
        ambiguity_note = f"该模板同时带有「{' / '.join(secondary_purposes[:2])}」特征，检索时不宜只按单一大类过滤。"

    combined_text = _normalize_spaces(" ".join([declared_purpose, support_text, content_text]))
    return {
        "primary_purpose": primary_purpose,
        "secondary_purposes": secondary_purposes,
        "purpose_breakdown": ranked_purposes[:4],
        "classification_confidence": classification_confidence,
        "ambiguity_note": ambiguity_note,
        "scene_tags": _rank_rule_tags(combined_text, TEMPLATE_SCENE_TAG_RULES, limit=3),
        "relation_tags": _rank_rule_tags(combined_text, TEMPLATE_RELATION_TAG_RULES, limit=3),
        "staging_tags": _rank_rule_tags(combined_text, TEMPLATE_STAGING_TAG_RULES, limit=4),
        "camera_tags": _rank_rule_tags(combined_text, TEMPLATE_CAMERA_TAG_RULES, limit=4),
        "emotion_tags": _rank_rule_tags(combined_text, TEMPLATE_EMOTION_TAG_RULES, limit=3),
        "narrative_tags": _rank_rule_tags(combined_text, TEMPLATE_NARRATIVE_TAG_RULES, limit=4),
        "series_name_hint": _normalize_spaces(str(source.get("source_series_name") or source.get("series_name") or series_name)),
    }


def _derive_template_search_metadata(
    source: Mapping[str, Any],
    *,
    series_name: str = "",
    classification_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    classification = dict(classification_metadata or {})
    purpose = _normalize_spaces(
        str(classification.get("primary_purpose") or source.get("primary_purpose") or source.get("purpose") or "")
    ) or FALLBACK_PURPOSE
    series_label = _normalize_spaces(
        str(source.get("source_series_name") or source.get("series_name") or series_name)
    )
    episode_id = _normalize_spaces(str(source.get("source_episode_id") or source.get("episode_id") or ""))
    beat_id = _normalize_spaces(str(source.get("source_beat_id") or source.get("beat_id") or ""))
    when_to_use_terms = _extract_template_search_terms(str(source.get("when_to_use") or ""), limit=6)
    summary_terms = _extract_template_search_terms(
        str(
            source.get("retrieval_title")
            or source.get("display_summary")
            or source.get("display_title")
            or source.get("beat_summary")
            or ""
        ),
        limit=3,
    )
    goal_terms = _extract_template_search_terms(str(source.get("dramatic_goal") or ""), limit=2)
    scene_terms = _extract_template_search_terms(
        str(
            source.get("scene_anchor_summary")
            or dict(source.get("scene_anchor") or {}).get("summary")
            or ""
        ),
        limit=2,
    )
    story_terms = _template_shot_terms(source, "story_function", per_shot_limit=1, max_total=4)
    visual_terms = _template_shot_terms(source, "visual_focus", per_shot_limit=2, max_total=4)
    camera_terms = _dedupe_strings(
        _template_shot_terms(source, "camera_entry", per_shot_limit=1, max_total=2)
        + _template_shot_terms(source, "camera_language", per_shot_limit=1, max_total=2)
    )[:3]

    parts: list[str] = []
    if when_to_use_terms:
        parts.append(f"适合{'、'.join(when_to_use_terms[:4])}")
    if story_terms:
        parts.append(f"重点场面：{'、'.join(story_terms[:3])}")
    elif summary_terms:
        parts.append(f"重点场面：{'、'.join(summary_terms[:2])}")
    if visual_terms:
        parts.append(f"关键视觉：{'、'.join(visual_terms[:3])}")
    elif len(summary_terms) >= 2:
        parts.append(f"关键视觉：{'、'.join(summary_terms[1:3])}")
    if camera_terms:
        parts.append(f"镜头抓手：{'、'.join(camera_terms[:3])}")
    if goal_terms:
        parts.append(f"情绪目标：{'、'.join(goal_terms[:1])}")
    elif scene_terms:
        parts.append(f"场景线索：{'、'.join(scene_terms[:2])}")

    search_hint = _normalize_spaces("；".join(parts[:5]))
    if not search_hint:
        fallback_hint = _normalize_spaces(
            str(source.get("retrieval_summary") or source.get("when_to_use") or "")
        )
        search_hint = fallback_hint or purpose

    secondary_purposes = [str(item).strip() for item in list(classification.get("secondary_purposes") or []) if str(item).strip()]
    scene_tags = [str(item).strip() for item in list(classification.get("scene_tags") or []) if str(item).strip()]
    relation_tags = [str(item).strip() for item in list(classification.get("relation_tags") or []) if str(item).strip()]
    staging_tags = [str(item).strip() for item in list(classification.get("staging_tags") or []) if str(item).strip()]
    camera_tags = [str(item).strip() for item in list(classification.get("camera_tags") or []) if str(item).strip()]
    emotion_tags = [str(item).strip() for item in list(classification.get("emotion_tags") or []) if str(item).strip()]
    narrative_tags = [str(item).strip() for item in list(classification.get("narrative_tags") or []) if str(item).strip()]

    search_keywords = _dedupe_strings(
        [
            purpose,
            *secondary_purposes,
            *when_to_use_terms,
            *summary_terms,
            *story_terms,
            *visual_terms,
            *camera_terms,
            *goal_terms,
            *scene_terms,
            *scene_tags,
            *relation_tags,
            *staging_tags,
            *camera_tags,
            *emotion_tags,
            *narrative_tags,
        ]
    )[:16]
    search_text_parts = [purpose, series_label, episode_id, beat_id, search_hint]
    if secondary_purposes:
        search_text_parts.append("辅类：" + "、".join(secondary_purposes[:3]))
    tag_parts = _dedupe_strings(scene_tags + relation_tags + staging_tags + camera_tags + emotion_tags + narrative_tags)
    if tag_parts:
        search_text_parts.append("标签：" + "、".join(tag_parts[:8]))
    if search_keywords:
        search_text_parts.append("关键词：" + "、".join(search_keywords))
    search_text = " ｜ ".join(part for part in search_text_parts if part)
    return {
        "search_hint": search_hint,
        "search_keywords": search_keywords,
        "search_text": search_text,
    }


def _safe_prompt_library_filename(text: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", " ", _normalize_spaces(text))
    cleaned = cleaned.strip(" .")
    if not cleaned:
        cleaned = "模板剧情描述"
    return cleaned[:72].rstrip()


def _render_prompt_library_entry(template: Mapping[str, Any]) -> str:
    lines = [
        f"# {template.get('retrieval_title', '') or template.get('template_id', '')}",
        "",
        f"- 主类：{template.get('primary_purpose', '') or template.get('purpose', '')}",
        f"- 辅类：{'、'.join(template.get('secondary_purposes', [])) or '无'}",
        f"- 分类置信度：{template.get('classification_confidence', 0)}",
        f"- 分类说明：{template.get('ambiguity_note', '') or '无'}",
        f"- 剧名：{template.get('source_series_name', '')}",
        f"- 模板 ID：{template.get('template_id', '')}",
        f"- 来源：{template.get('source_episode_id', '')} {template.get('source_beat_id', '')}",
        f"- 时长：{template.get('duration_seconds', 0)} 秒",
        f"- 质量分：{template.get('quality_score', 0)}",
        f"- 检索建议：{template.get('search_hint', '') or template.get('retrieval_summary', '') or template.get('when_to_use', '')}",
        f"- 检索关键词：{'、'.join(template.get('search_keywords', [])) or '无'}",
        f"- 检索文本：{template.get('search_text', '') or '无'}",
        f"- 场景标签：{'、'.join(template.get('scene_tags', [])) or '无'}",
        f"- 关系标签：{'、'.join(template.get('relation_tags', [])) or '无'}",
        f"- 调度标签：{'、'.join(template.get('staging_tags', [])) or '无'}",
        f"- 镜头标签：{'、'.join(template.get('camera_tags', [])) or '无'}",
        f"- 情绪标签：{'、'.join(template.get('emotion_tags', [])) or '无'}",
        f"- 叙事标签：{'、'.join(template.get('narrative_tags', [])) or '无'}",
        f"- 必填槽位：{'、'.join(template.get('required_slots', [])) or '无'}",
        f"- 来源标题：{'；'.join(template.get('source_titles', [])) or '无'}",
        f"- 场景锚点：{template.get('scene_anchor_summary', '') or '无'}",
        f"- 文件位置：{template.get('prompt_library_path', '') or '未导出'}",
        "",
        "## 还原版 Prompt",
        "",
        str(template.get("restored_seedance_prompt") or ""),
        "",
        "## 通用模板 Prompt",
        "",
        str(template.get("generalized_template_prompt") or ""),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _export_prompt_library(
    *,
    project_root: Path,
    series_name: str,
    template_library: dict[str, Any],
) -> dict[str, Any]:
    prompt_library_root = project_root / "prompt_library"
    export_count = 0
    used_paths: set[Path] = set()
    for purpose_item in list(template_library.get("purposes") or []):
        purpose = _normalize_spaces(str(purpose_item.get("purpose") or "")) or FALLBACK_PURPOSE
        series_dir = prompt_library_root / purpose / series_name
        for template in list(purpose_item.get("templates") or []):
            file_stem = _safe_prompt_library_filename(
                str(template.get("retrieval_title") or template.get("display_summary") or template.get("template_id") or "模板剧情描述")
            )
            target = series_dir / f"{file_stem}.md"
            if target in used_paths:
                target = series_dir / f"{file_stem}__{template.get('template_id', 'template')}.md"
            used_paths.add(target)
            template["prompt_library_path"] = str(target.relative_to(project_root))
            save_text_file(target, _render_prompt_library_entry(template))
            export_count += 1
    template_library["prompt_library_root"] = str(prompt_library_root.relative_to(project_root))
    return {
        "prompt_library_root": str(prompt_library_root),
        "prompt_library_template_count": export_count,
    }


def _iter_template_library_templates(library: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for purpose_item in list(library.get("purposes") or []):
        purpose = _normalize_spaces(str(purpose_item.get("purpose") or "")) or FALLBACK_PURPOSE
        for template in list(purpose_item.get("templates") or []):
            item = dict(template or {})
            item.setdefault("purpose", purpose)
            item.setdefault("primary_purpose", item.get("purpose") or purpose)
            yield item


def _normalize_project_relative_path(project_root: Path, raw: Any) -> str:
    text = _normalize_spaces(str(raw or ""))
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute():
        try:
            return str(path.relative_to(project_root))
        except ValueError:
            return str(path)
    return str(path)


def _purpose_sort_key(purpose: str) -> tuple[int, str]:
    return (
        PURPOSE_ORDER.index(purpose) if purpose in PURPOSE_ORDER else len(PURPOSE_ORDER),
        purpose,
    )


def render_prompt_library_search_index_markdown(index: Mapping[str, Any]) -> str:
    lines = [
        "# Prompt Library 检索索引",
        "",
        f"- 生成时间：{index.get('generated_at', '')}",
        f"- 素材根目录：{index.get('prompt_library_root', '')}",
        f"- 覆盖剧数：{index.get('series_count', 0)}",
        f"- 覆盖模板：{index.get('template_count', 0)}",
        f"- 覆盖主类：{index.get('purpose_count', 0)}",
        "",
    ]
    for purpose_item in list(index.get("purposes") or []):
        lines.extend(
            [
                f"## {purpose_item.get('purpose', '')}",
                "",
                f"- 模板数量：{purpose_item.get('template_count', 0)}",
                "",
            ]
        )
        for template in list(purpose_item.get("templates") or []):
            lines.extend(
                [
                    f"### {template.get('template_id', '')}｜{template.get('retrieval_title', '') or template.get('purpose', '')}",
                    "",
                    f"- 来源：{template.get('source_series_name', '')} {template.get('source_episode_id', '')} {template.get('source_beat_id', '')}",
                    f"- 时长：{template.get('duration_seconds', 0)} 秒",
                    f"- 质量分：{template.get('quality_score', 0)}",
                    f"- 主类：{template.get('primary_purpose', '') or template.get('purpose', '')}",
                    f"- 辅类：{'、'.join(template.get('secondary_purposes', [])) or '无'}",
                    f"- 分类置信度：{template.get('classification_confidence', 0)}",
                    f"- 分类说明：{template.get('ambiguity_note', '') or '无'}",
                    f"- 场景标签：{'、'.join(template.get('scene_tags', [])) or '无'}",
                    f"- 关系标签：{'、'.join(template.get('relation_tags', [])) or '无'}",
                    f"- 调度标签：{'、'.join(template.get('staging_tags', [])) or '无'}",
                    f"- 镜头标签：{'、'.join(template.get('camera_tags', [])) or '无'}",
                    f"- 情绪标签：{'、'.join(template.get('emotion_tags', [])) or '无'}",
                    f"- 叙事标签：{'、'.join(template.get('narrative_tags', [])) or '无'}",
                    f"- 检索建议：{template.get('search_hint', '') or '无'}",
                    f"- 检索关键词：{'、'.join(template.get('search_keywords', [])) or '无'}",
                    f"- 文件位置：{template.get('prompt_library_path', '') or '未导出'}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _export_prompt_library_search_index(
    *,
    project_root: Path,
    current_series_name: str,
    current_template_library: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_library_root = project_root / "prompt_library"
    analysis_root = project_root / "analysis"
    libraries_by_series: dict[str, Mapping[str, Any]] = {}
    for path in sorted(analysis_root.glob("*/seedance_purpose_template_library.json")):
        library = _load_optional_json(path)
        if not library:
            continue
        series_name = _normalize_spaces(str(library.get("series_name") or path.parent.name))
        if not series_name:
            continue
        libraries_by_series[series_name] = library
    current_series_key = _normalize_spaces(current_series_name) or _normalize_spaces(
        str(current_template_library.get("series_name") or "")
    )
    if current_series_key:
        libraries_by_series[current_series_key] = current_template_library

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    flat_templates: list[dict[str, Any]] = []
    series_names: set[str] = set()
    template_count = 0
    for series_name in sorted(libraries_by_series):
        library = libraries_by_series[series_name]
        for template in _iter_template_library_templates(library):
            prompt_library_path = _normalize_project_relative_path(project_root, template.get("prompt_library_path") or "")
            if not prompt_library_path:
                continue
            classification_metadata = _derive_template_classification_metadata(
                template,
                series_name=series_name,
            )
            search_metadata = _derive_template_search_metadata(
                template,
                series_name=series_name,
                classification_metadata=classification_metadata,
            )
            purpose = _normalize_spaces(
                str(classification_metadata.get("primary_purpose") or template.get("primary_purpose") or template.get("purpose") or "")
            ) or FALLBACK_PURPOSE
            entry = {
                "template_id": str(template.get("template_id") or ""),
                "purpose": purpose,
                "primary_purpose": purpose,
                "secondary_purposes": list(classification_metadata.get("secondary_purposes") or template.get("secondary_purposes") or []),
                "purpose_breakdown": list(classification_metadata.get("purpose_breakdown") or template.get("purpose_breakdown") or []),
                "classification_confidence": float(classification_metadata.get("classification_confidence") or 0.0),
                "ambiguity_note": str(classification_metadata.get("ambiguity_note") or ""),
                "source_series_name": str(template.get("source_series_name") or series_name),
                "source_episode_id": str(template.get("source_episode_id") or ""),
                "source_beat_id": str(template.get("source_beat_id") or ""),
                "duration_seconds": float(template.get("duration_seconds", 0) or 0),
                "quality_score": float(template.get("quality_score", 0) or 0),
                "retrieval_title": str(template.get("retrieval_title") or template.get("display_summary") or ""),
                "search_hint": str(template.get("search_hint") or search_metadata.get("search_hint") or ""),
                "search_keywords": list(template.get("search_keywords") or search_metadata.get("search_keywords") or []),
                "search_text": str(template.get("search_text") or search_metadata.get("search_text") or ""),
                "scene_tags": list(classification_metadata.get("scene_tags") or template.get("scene_tags") or []),
                "relation_tags": list(classification_metadata.get("relation_tags") or template.get("relation_tags") or []),
                "staging_tags": list(classification_metadata.get("staging_tags") or template.get("staging_tags") or []),
                "camera_tags": list(classification_metadata.get("camera_tags") or template.get("camera_tags") or []),
                "emotion_tags": list(classification_metadata.get("emotion_tags") or template.get("emotion_tags") or []),
                "narrative_tags": list(classification_metadata.get("narrative_tags") or template.get("narrative_tags") or []),
                "prompt_library_path": prompt_library_path,
            }
            grouped[purpose].append(entry)
            flat_templates.append(entry)
            series_names.add(entry["source_series_name"])
            template_count += 1

    purposes: list[dict[str, Any]] = []
    for purpose in sorted(grouped, key=_purpose_sort_key):
        templates = sorted(
            grouped[purpose],
            key=lambda item: (
                -float(item.get("quality_score", 0) or 0),
                str(item.get("source_series_name") or ""),
                str(item.get("source_episode_id") or ""),
                str(item.get("source_beat_id") or ""),
                str(item.get("template_id") or ""),
            ),
        )
        purposes.append(
            {
                "purpose": purpose,
                "template_count": len(templates),
                "templates": templates,
            }
        )

    index_payload = {
        "generated_at": utc_timestamp(),
        "prompt_library_root": str(prompt_library_root.relative_to(project_root)),
        "series_count": len(series_names),
        "template_count": template_count,
        "purpose_count": len(purposes),
        "purposes": purposes,
        "templates": sorted(
            flat_templates,
            key=lambda item: (
                -float(item.get("quality_score", 0) or 0),
                str(item.get("primary_purpose") or ""),
                str(item.get("source_series_name") or ""),
                str(item.get("source_episode_id") or ""),
                str(item.get("template_id") or ""),
            ),
        ),
    }
    index_json_path = prompt_library_root / "SEARCH_INDEX.json"
    index_md_path = prompt_library_root / "SEARCH_INDEX.md"
    save_json_file(index_json_path, index_payload)
    save_text_file(index_md_path, render_prompt_library_search_index_markdown(index_payload))
    return {
        "prompt_library_index_json_path": str(index_json_path),
        "prompt_library_index_markdown_path": str(index_md_path),
    }


def _rank_lines(lines: Iterable[str], *, limit: int) -> list[str]:
    cleaned = [_normalize_spaces(item) for item in lines if _normalize_spaces(item)]
    if not cleaned:
        return []
    counter = Counter(cleaned)
    ranked = sorted(cleaned, key=lambda item: (-counter[item], cleaned.index(item)))
    result: list[str] = []
    seen: set[str] = set()
    for item in ranked:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def _collect_unique_lines(group: Sequence[Mapping[str, Any]], key: str, *, limit: int) -> list[str]:
    values: list[str] = []
    for item in group:
        values.extend(_clean_list(item.get(key)))
    return _rank_lines(values, limit=limit)


def _combine_unique_texts(values: Iterable[Any]) -> str:
    unique: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = _normalize_spaces(str(raw or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return "；".join(unique)


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = _normalize_spaces(str(raw or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _dedupe_dict_items(values: Iterable[Mapping[str, Any]], *, key: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in values:
        item = dict(raw or {})
        marker = _normalize_spaces(str(item.get(key) or ""))
        if marker and marker in seen:
            continue
        if marker:
            seen.add(marker)
        result.append(item)
    return result


def _best_line(values: Any, *, fallback: str = "") -> str:
    cleaned = _clean_list(values)
    if cleaned:
        return cleaned[0]
    return _normalize_spaces(fallback)


def _clean_list(values: Any) -> list[str]:
    result: list[str] = []
    for item in list(values or []):
        text = _normalize_spaces(str(item or ""))
        if text:
            result.append(text)
    return result


def _keyword_hit_count(text: str, keywords: Sequence[str]) -> int:
    normalized = _normalize_spaces(text)
    return sum(1 for keyword in keywords if keyword and keyword in normalized)


def _looks_like_visual_hint(text: str) -> bool:
    normalized = _normalize_spaces(text)
    if not normalized:
        return False
    strong_visual_hits = _keyword_hit_count(normalized, STRONG_VISUAL_HINT_KEYWORDS)
    visual_hits = _keyword_hit_count(normalized, VISUAL_HINT_KEYWORDS)
    dialogue_hits = _keyword_hit_count(normalized, DIALOGUE_HEAVY_HINT_KEYWORDS)
    if any(token in normalized for token in ("“", "”", "\"", "？", "?")):
        dialogue_hits += 2
    if strong_visual_hits <= 0:
        return len(normalized) <= 12 and visual_hits >= 2 and dialogue_hits == 0
    if dialogue_hits >= strong_visual_hits + 2 and "镜头" not in normalized and "画面" not in normalized:
        return False
    return True


def _is_generic_transition_hint(text: str) -> bool:
    normalized = _normalize_spaces(text)
    if not normalized:
        return True
    if normalized in GENERIC_TRANSITION_HINTS:
        return True
    return any(
        normalized.startswith(prefix)
        for prefix in (
            "对白句尾",
            "对手或听者反应",
            "异常信号扩散",
            "尾帧把结果反应",
            "物件或字面锚点",
        )
    )


def _looks_like_transition_hint(text: str) -> bool:
    normalized = _normalize_spaces(text)
    if not normalized:
        return False
    if _is_generic_transition_hint(normalized):
        return False
    return any(
        token in normalized
        for token in (
            "切",
            "转",
            "停",
            "带向",
            "带出",
            "落到",
            "句尾",
            "尾帧",
            "顺着",
            "顺势",
            "触发",
            "引到",
            "交给",
            "视线",
            "动作",
            "反应",
            "声响",
        )
    )


def _normalize_spaces(raw: str) -> str:
    return re.sub(r"\s+", " ", str(raw or "")).strip()


def _slugify(raw: str) -> str:
    cleaned = re.sub(r"\s+", "_", str(raw).strip())
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "template"


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json_file(path)
