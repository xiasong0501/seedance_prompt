from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_telemetry import TelemetryRecorder, telemetry_span
from openai_agents.runtime_utils import (
    build_episode_ids,
    configure_openai_api,
    load_runtime_config,
    openai_json_completion,
    read_text,
)
from prompt_utils import (
    build_frame_composition_guidance,
    is_portrait_frame_orientation,
    load_prompt,
    normalize_frame_orientation,
    render_bullets,
    render_prompt,
)
from skill_utils import load_skill
import workflow_context_compaction as ctx_compact
from providers.base import build_provider_model_tag, load_json_file, save_json_file, save_text_file, utc_timestamp
from seedance_learning import PURPOSE_ORDER, PURPOSE_PROFILES, infer_primary_purpose_from_parts
from seedance_logic_review_rules import render_generalized_logic_rules_prompt
try:
    from scripts.series_paths import build_series_paths
except ModuleNotFoundError:
    from series_paths import build_series_paths


DEFAULT_CONFIG_PATH = Path("config/seedance_storyboard_pipeline.local.json")
PANEL_LINE_PATTERN = re.compile(r"^(?P<slot>(?:(?:格|Panel)\s*)?\d+)[—-]+【(?P<scene>.+?)】")
FULL_FRAME_PANEL_LINE_PATTERN = re.compile(r"^(?P<slot>全幅|Full(?:\s*Frame)?)\s*[—-]+【(?P<scene>.+?)】", flags=re.IGNORECASE)
BEAT_TIME_WINDOW_PATTERN = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*(?:-|–|—|~|至)\s*\d+(?:\.\d+)?\s*(?:秒|s)\s*[:：]"
)
HIGH_VALUE_ENTRY_PATTERNS = (
    "翻盘", "反击", "揭示", "清算", "认领", "觉醒", "突破", "压制", "命令",
    "对峙", "公开", "打脸", "发难", "拍板", "封锁", "齐射", "法阵", "火力", "崩塌",
)
REACTION_SIGNAL_PATTERNS = (
    "反应", "看向", "抬眼", "回头", "一怔", "一顿", "停顿", "愣住", "呼吸", "眼神",
    "视线", "沉默", "僵住", "后退", "退半步", "收住", "迟疑", "屏息",
)
TAIL_HOOK_SIGNAL_PATTERNS = (
    "尾帧", "停在", "收在", "定在", "停住", "交给下一条", "交给下一点", "即将",
    "将至", "未完", "未落", "看向", "余波", "悬而未决", "下一秒",
)
NARRATION_SPEAKER_MARKERS = ("旁白", "画外音", "内心", "心声", "OS", "VO")
FUTURE_CHARACTER_HOOK_PREFIXES = (
    "交给下一条",
    "给下一条",
    "接下一条",
    "下一条",
    "下一镜",
    "下条",
    "下一秒",
)
STATE_UPDATE_SCENE_PREFIX = "场"
STATE_UPDATE_SCENE_KEYS = ("loc", "cam", "light")
STATE_UPDATE_CHARACTER_KEYS = ("zone", "pose", "face", "hold")
STATE_MAX_UPDATE_LINES = 4
STATE_MAX_SNAPSHOT_LINES = 5
STATE_CAMERA_PATTERNS: list[tuple[str, str]] = [
    ("极近景", "极近景"),
    ("近景", "近景"),
    ("中近景", "中近景"),
    ("中景", "中景"),
    ("半大全景", "半大全景"),
    ("大全景", "大全景"),
    ("竖屏中轴", "竖屏中轴"),
]
STATE_LIGHT_PATTERNS: list[tuple[str, str]] = [
    ("右上冷白", "右上冷白"),
    ("冷白天光", "冷白天光"),
    ("银青法光", "银青法光"),
    ("冷灰天光", "冷灰天光"),
    ("暖金反光", "暖金反光"),
]
STATE_ZONE_PATTERNS: list[tuple[str, str]] = [
    ("斩龙台记忆化闪回区", "斩龙台记忆化闪回区"),
    ("斩龙台远山火力阵地", "斩龙台远山火力阵地"),
    ("传送镜门位", "传送镜门位"),
    ("斩龙台", "斩龙台"),
    ("赛场", "赛场"),
    ("龙辇中轴", "龙辇中轴"),
    ("龙辇前缘", "龙辇前缘"),
    ("龙辇", "龙辇"),
    ("高台栏位", "高台栏位"),
    ("高处栏位", "高台栏位"),
    ("高台前缘", "高台前缘"),
    ("高台", "高台"),
    ("传送镜前", "传送镜前"),
    ("传送镜", "传送镜前"),
    ("入场道", "入场道"),
    ("石阶", "石阶"),
    ("台中央", "台中央"),
    ("赛场中央", "赛场中央"),
    ("台面", "台面"),
    ("裂缝边", "裂缝边"),
    ("裂缝", "裂缝边"),
]
STATE_POSE_PATTERNS: list[tuple[str, str]] = [
    ("半跪", "半跪"),
    ("跪", "跪"),
    ("伏地", "伏地"),
    ("趴", "伏地"),
    ("俯身", "俯身"),
    ("坐", "坐"),
    ("站", "站"),
    ("立", "站"),
    ("躺", "躺"),
    ("倒地", "倒地"),
    ("倒在", "倒地"),
]
STATE_HOLD_PATTERNS: list[tuple[str, str]] = [
    ("持刀", "持刀"),
    ("握刀", "持刀"),
    ("托丹", "托丹"),
    ("抱拳", "抱拳"),
    ("扶手", "扶手"),
    ("扶炉", "扶炉"),
    ("撑地", "撑地"),
    ("按住", "按物"),
    ("压着", "压制"),
    ("空手", "空手"),
]
STATE_BRIDGE_CUE_PATTERNS = (
    "切到", "切向", "切回", "拉开", "回到", "顺着", "跟着", "带到", "转到", "入画", "入镜", "镜头",
)
STATE_TRANSITION_CUE_PATTERNS = (
    "起身", "站起", "坐下", "落座", "跪下", "半跪", "伏地", "趴下", "起步", "迈", "走", "登", "上前",
    "后退", "退", "回头", "转身", "冲入", "扑", "挪", "移到", "靠近", "抬手", "举起", "放下", "扶着", "撑地",
)
STATE_LOCATIVE_CUE_PATTERNS = (
    "在", "于", "位于", "停在", "落在", "留在", "守在", "稳在", "回到", "切到", "切向", "来到", "转到",
    "拉到", "移到", "走到", "退到", "站在", "站于", "立于", "坐在", "坐于", "跪在", "跪于", "伏在",
)
STATE_CHARACTER_ZONE_RULES: list[tuple[tuple[str, ...], str, bool]] = [
    (("龙辇前左三步", "龙辇前右三步", "龙辇前三步", "龙辇前下方", "龙辇前侧", "龙辇前缘", "龙辇边缘"), "龙辇前缘", False),
    (("龙辇中轴", "龙辇中央偏高位", "龙辇中央", "龙辇上"), "龙辇中轴", False),
    (("龙辇",), "龙辇中轴", True),
    (("高台前缘", "高台栏位", "高处栏位"), "高台前缘", False),
    (("高台中央", "高台上方中轴", "高台中央位"), "高台", False),
    (("高台",), "高台", True),
    (("中央赛场", "赛场中央", "赛场边缘", "赛场另一侧", "赛场空地", "中央空地"), "赛场", False),
    (("赛场",), "赛场", True),
    (("入场道",), "入场道", False),
    (("石阶",), "石阶", False),
    (("四门出口", "四门"), "四门", False),
    (("传送镜门位",), "传送镜门位", False),
    (("传送镜前", "传送镜"), "传送镜前", False),
    (("斩龙台记忆化闪回区",), "斩龙台记忆化闪回区", False),
]
STATE_SCENE_LOC_RULES: list[tuple[tuple[str, ...], str, bool]] = [
    (("龙辇前左三步", "龙辇前右三步", "龙辇前三步", "龙辇前下方", "龙辇前侧", "龙辇前缘", "龙辇边缘"), "龙辇", False),
    (("龙辇中轴", "龙辇中央偏高位", "龙辇中央", "龙辇上", "龙辇"), "龙辇", True),
    (("高台前缘", "高台中央", "高台上方中轴", "高台中央位", "高台栏位", "高处栏位", "高台"), "高台", True),
    (("中央赛场", "赛场中央", "赛场边缘", "赛场另一侧", "赛场空地", "中央空地", "赛场"), "赛场", True),
    (("传送镜门位", "传送镜前", "传送镜"), "传送镜", False),
    (("入场道",), "入场道", False),
    (("石阶",), "石阶", False),
    (("四门出口", "四门"), "四门", False),
    (("斩龙台记忆化闪回区",), "斩龙台记忆化闪回区", False),
    (("斩龙台",), "斩龙台", False),
]
STATE_POSE_REGEX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(半跪|单膝跪地)"), "半跪"),
    (re.compile(r"(跪在|跪于|跪稳|双膝[^，。；]{0,4}跪|齐齐跪稳)"), "跪"),
    (re.compile(r"(伏地|贴地|趴下|伏到地面|俯伏)"), "伏地"),
    (re.compile(r"(俯身|弯身)"), "俯身"),
    (re.compile(r"(坐在|坐于|坐着|坐稳|坐回|仍坐|端坐|落座|并未起身)"), "坐"),
    (re.compile(r"(站在|站于|站定|站稳|立于|立在|仍站|停住[^，。；]{0,4}站姿)"), "站"),
    (re.compile(r"(倒地|倒在|摔倒在)"), "倒地"),
    (re.compile(r"(躺在|躺倒)"), "躺"),
]
STATE_HOLD_REGEX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(持刀|握刀)"), "持刀"),
    (re.compile(r"(托着?丹|托丹)"), "托丹"),
    (re.compile(r"(抱拳)"), "抱拳"),
    (re.compile(r"(扶着?扶手|扶手)"), "扶手"),
    (re.compile(r"(扶着?香炉|扶炉|按住香炉|压住香炉)"), "扶炉"),
    (re.compile(r"(撑地)"), "撑地"),
    (re.compile(r"(按住|按着)"), "按物"),
    (re.compile(r"(压着|压住)"), "压制"),
    (re.compile(r"(空着手|空手)"), "空手"),
]

SENSITIVE_TEXT_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(鲜血喷溅|血液喷溅|血浆喷溅|鲜血四溅|血花四溅|喷血|血雾)"), "强冲击带起的尘雾与碎屑"),
    (re.compile(r"(血肉模糊|皮开肉绽|开膛破肚|肠子外露|器官外露)"), "重创后的狼狈状态"),
    (re.compile(r"(断肢|断臂|断手|断腿|断头|斩首|人头落地|枭首)"), "一击制敌后的压制结果"),
    (re.compile(r"(尸体|尸首|死尸|横尸)"), "倒地身影"),
    (re.compile(r"(掐住脖子|扼住咽喉|掐脖子)"), "强行压制对方行动"),
    (re.compile(r"(刑讯|拷打|凌虐|施虐|鞭打)"), "高压逼供"),
    (re.compile(r"(强吻|舌吻)"), "强行逼近"),
    (re.compile(r"(撕衣|扯开衣襟|扒开衣服)"), "粗暴拉扯衣袖"),
    (re.compile(r"(乳沟|酥胸|胸部半裸|裸露胸口|赤裸上身|裸露大腿|湿身贴体)"), "服装轮廓与情绪张力"),
    (re.compile(r"(呻吟|娇喘|喘息声)"), "压抑的呼吸与情绪波动"),
    (re.compile(r"(往地上啐一口唾沫|往地上啐了一口唾沫|往地上啐一口|往地上啐了一口|啐了一口唾沫|啐一口唾沫|啐地|吐唾沫|朝[^\n，。；]*吐口水)"), "以明显嫌弃的动作避开对方，并把手中器物重重顿回原位"),
    (re.compile(r"(唾沫落地声|唾液落地声)"), "衣摆与器物碰石的闷响"),
    (re.compile(r"(酒水和唾沫混在石面上|酒水与唾沫混在石面上)"), "酒水洒在石面上，嫌弃意味已经足够明显"),
    (re.compile(r"(纳的一房小妾|新纳的小妾|那房小妾|小妾)"), "昨夜新收进府的那个人"),
    (re.compile(r"(受用啊|受用)"), "满意"),
    (re.compile(r"(一枝梨花压海棠)"), "昨夜倒是得意得很"),
    (re.compile(r"(恶疾|染病|传病|病气)"), "后手"),
    (re.compile(r"(细菌战)"), "借势布局"),
    (re.compile(r"(捂住下腹|捂着下腹|捂住小腹|捂着小腹|下意识捂住下腹|下意识捂住小腹)"), "身形骤僵，站姿一时失稳"),
    (re.compile(r"(AK小队)"), "火器小队"),
    (re.compile(r"(?<![A-Za-z])AK(?![A-Za-z])"), "异质火器"),
    (re.compile(r"(坦克)"), "重装铁甲车"),
    (re.compile(r"(迫击炮)"), "远程重型火器"),
    (re.compile(r"(无人机)"), "飞行机关"),
    (re.compile(r"(火力覆盖)"), "机械齐鸣压场"),
    (re.compile(r"(炮弹)"), "强冲击弹"),
    (re.compile(r"(一个不留)"), "全部压住，一个都别想再上前"),
    (re.compile(r"(别放一个人出去)"), "控制住场面，别让任何人再冲出场外"),
]

SENSITIVE_OUTPUT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(鲜血喷溅|血液喷溅|血浆喷溅|鲜血四溅|血花四溅|喷血|血雾)"), "避免喷溅血液类直观描写"),
    (re.compile(r"(血肉模糊|皮开肉绽|开膛破肚|肠子外露|器官外露|断肢|断臂|断手|断腿|断头|斩首|人头落地|枭首)"), "避免身体损毁或肢体分离的直观描写"),
    (re.compile(r"(尸体|尸首|死尸|横尸)"), "避免尸体特写，改成结果或反应镜头"),
    (re.compile(r"(刑讯|拷打|凌虐|施虐|鞭打|掐住脖子|扼住咽喉|掐脖子)"), "避免刑讯或受虐细节，改成压制结果"),
    (re.compile(r"(强吻|舌吻|撕衣|扯开衣襟|扒开衣服|乳沟|酥胸|胸部半裸|裸露胸口|赤裸上身|裸露大腿|湿身贴体|呻吟|娇喘)"), "避免软色情或强性暗示表达"),
    (re.compile(r"(啐地|吐唾沫|唾沫落地|唾液落地|朝[^\n，。；]*吐口水)"), "避免低俗唾液羞辱动作，改成站位、眼神或器物动作"),
    (re.compile(r"(小妾|纳妾|受用|一枝梨花压海棠|恶疾|染病|传病|细菌战|捂住下腹|捂着下腹|捂住小腹|捂着小腹)"), "避免性羞辱或疾病传播联想，改成更安全的后手/失态表达"),
    (re.compile(r"(?<![A-Za-z])(AK|坦克|迫击炮|无人机)(?![A-Za-z])|火力覆盖|炮弹|一个不留|别放一个人出去"), "避免军武直白杀伤表述，改成异质机关、冲击结果或压制结果"),
]

PROMPT_TEXT_META_BLOCKS_TO_REMOVE = [
    "竖屏构图要求：按9:16手机竖屏安全区组织画面，主体尽量位于中轴或中轴偏上/偏下，优先用中近景、上下高差、前后景遮挡和纵深调度承载冲突，避免超宽横向铺排。",
    "统一风格要求：延续上一条已建立的主光方向、色温、材质质感、人物体量和镜头高度，不要把本条拍成另一种画风。",
    "运镜要求：全条以1-2个有动机的主运动完成，转向跟着角色视线、动作或关键道具走，起势到收束要平滑。",
    "节奏要求：前20%时长内进入有效剧情，每2-3秒给出新的动作、反应或信息，不做重复空转。",
    "合规要求：避免直观伤口、喷溅血液、裸露身体、刑讯细节与软色情表达，以非直观结果和情绪/构图传达冲突。",
    "侮辱动作要求：不要拍唾液或朝人吐口水，改用顿杯、偏头避让、冷笑、站位排斥和手停在半空这类安全表达。",
    "反击表达要求：保留后手揭晓与对方失态，不拍暧昧细节、疾病传播细节或身体羞辱细节。",
    "战斗表达要求：优先拍远景机械阵列、强光、烟尘、冲击波与震退结果，不拍逐个击中、逐个爆炸或逐个处决。",
]

PROMPT_TEXT_PREFIX_LINES_TO_REMOVE = [
    "承接说明：",
    "承接关系：",
    "首帧承接上一条尾态：",
    "首帧承接要求：",
    "开场第一拍：",
    "尾帧交棒要求：",
    "镜头节拍：",
    "对白时间线：",
    "声音设计：",
    "风险提示：",
    "可直接投喂正文：",
    "竖屏构图要求：",
    "统一风格要求：",
    "运镜要求：",
    "节奏要求：",
    "合规要求：",
    "侮辱动作要求：",
    "反击表达要求：",
    "战斗表达要求：",
    "所有对白必须串行出现，前一句完全结束后下一句再进入，不允许双声叠台词。",
]

PROMPT_TEXT_EXACT_LINES_TO_REMOVE = {
    "**镜头节拍拆解**：",
    "**对白时间线**：",
    "**声音设计**：",
    "**统一复合提示词（主时间线）**：",
    "**统一复合提示词**：",
}

SEEDANCE_DIALOGUE_BLOCK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["speaker", "line", "start_second", "end_second", "delivery_note"],
    "properties": {
        "speaker": {"type": "string"},
        "line": {"type": "string"},
        "start_second": {"type": "number"},
        "end_second": {"type": "number"},
        "delivery_note": {"type": "string"},
    },
}

SEEDANCE_TIMELINE_ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "start_second",
        "end_second",
        "visual_beat",
        "speaker",
        "spoken_line",
        "delivery_note",
        "dialogue_blocks",
        "audio_cues",
        "transition_hook",
    ],
    "properties": {
        "start_second": {"type": "number"},
        "end_second": {"type": "number"},
        "visual_beat": {"type": "string"},
        "speaker": {"type": "string"},
        "spoken_line": {"type": "string"},
        "delivery_note": {"type": "string"},
        "dialogue_blocks": {"type": "array", "items": SEEDANCE_DIALOGUE_BLOCK_SCHEMA},
        "audio_cues": {"type": "string"},
        "transition_hook": {"type": "string"},
    },
}

SEEDANCE_PROMPT_ENTRY_MODEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "point_id",
        "title",
        "pace_label",
        "density_strategy",
        "duration_hint",
        "continuity_bridge",
        "primary_refs",
        "secondary_refs",
        "master_timeline",
        "prompt_text",
        "risk_notes",
    ],
    "properties": {
        "point_id": {"type": "string"},
        "title": {"type": "string"},
        "pace_label": {"type": "string", "enum": ["快压推进", "中速推进", "舒缓铺陈"]},
        "density_strategy": {"type": "string"},
        "duration_hint": {"type": "string"},
        "continuity_bridge": {"type": "string"},
        "primary_refs": {"type": "array", "items": {"type": "string"}},
        "secondary_refs": {"type": "array", "items": {"type": "string"}},
        "master_timeline": {"type": "array", "items": SEEDANCE_TIMELINE_ENTRY_SCHEMA},
        "prompt_text": {"type": "string"},
        "risk_notes": {"type": "array", "items": {"type": "string"}},
    },
}

SEEDANCE_PROMPTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["episode_id", "episode_title", "materials_overview", "prompt_entries", "global_notes"],
    "properties": {
        "episode_id": {"type": "string"},
        "episode_title": {"type": "string"},
        "materials_overview": {"type": "string"},
        "prompt_entries": {"type": "array", "items": SEEDANCE_PROMPT_ENTRY_MODEL_SCHEMA},
        "global_notes": {"type": "array", "items": {"type": "string"}},
    },
}

SEEDANCE_REVIEW_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["materials_overview", "global_notes", "replace_prompt_entries", "delete_point_ids"],
    "properties": {
        "materials_overview": {"type": "string"},
        "global_notes": {"type": "array", "items": {"type": "string"}},
        "replace_prompt_entries": {"type": "array", "items": SEEDANCE_PROMPT_ENTRY_MODEL_SCHEMA},
        "delete_point_ids": {"type": "array", "items": {"type": "string"}},
    },
}


def validate_openai_strict_schema(schema: Mapping[str, Any], *, path: str = "root") -> None:
    if schema.get("type") == "object" and isinstance(schema.get("properties"), Mapping):
        properties = schema["properties"]
        required = schema.get("required")
        if not isinstance(required, list):
            raise ValueError(f"{path}: missing required list for strict object schema")
        missing = sorted(str(key) for key in properties.keys() if key not in required)
        if missing:
            raise ValueError(f"{path}: required is missing keys {missing}")
        for key, value in properties.items():
            if isinstance(value, Mapping):
                validate_openai_strict_schema(value, path=f"{path}.properties.{key}")
        return
    if schema.get("type") == "array" and isinstance(schema.get("items"), Mapping):
        validate_openai_strict_schema(schema["items"], path=f"{path}.items")


validate_openai_strict_schema(SEEDANCE_PROMPTS_SCHEMA)
validate_openai_strict_schema(SEEDANCE_REVIEW_PATCH_SCHEMA)


def print_status(message: str) -> None:
    print(f"[seedance-prompts] {message}", flush=True)


def normalize_spaces(raw: str) -> str:
    return re.sub(r"\s+", " ", str(raw or "")).strip()


def normalize_storyboard_point_id(raw: Any, *, fallback_index: int | None = None) -> str:
    text = normalize_spaces(str(raw or ""))
    numeric_match = re.fullmatch(r"[Pp]?0*(\d{1,3})", text)
    if numeric_match:
        return f"P{int(numeric_match.group(1)):02d}"
    story_match = re.fullmatch(r"(?:sp|story_point)[_-]?0*(\d{1,3})", text, flags=re.IGNORECASE)
    if story_match:
        return f"P{int(story_match.group(1)):02d}"
    embedded_match = re.search(r"\b[Pp]0*(\d{1,3})\b", text)
    if embedded_match:
        return f"P{int(embedded_match.group(1)):02d}"
    if fallback_index is not None:
        return f"P{int(fallback_index):02d}"
    return text


def strip_storyboard_title_prefix(title: Any, point_id: str) -> str:
    clean_title = normalize_spaces(str(title or ""))
    if not clean_title:
        return ""
    point_number = ""
    point_match = re.fullmatch(r"P(\d+)", str(point_id or "").strip(), flags=re.IGNORECASE)
    if point_match:
        point_number = str(int(point_match.group(1)))
    prefix_patterns = [rf"^{re.escape(str(point_id or '').strip())}(?:[\s:：、.\-]+|$)"]
    if point_number:
        prefix_patterns.extend(
            [
                rf"^0*{re.escape(point_number)}(?:[\s:：、.\-]+|$)",
                rf"^[Pp]0*{re.escape(point_number)}(?:[\s:：、.\-]+|$)",
                rf"^(?:sp|story_point)[_-]?0*{re.escape(point_number)}(?:[\s:：、.\-]+|$)",
            ]
        )
    for pattern in prefix_patterns:
        updated = re.sub(pattern, "", clean_title, count=1, flags=re.IGNORECASE).strip()
        if updated and updated != clean_title:
            clean_title = updated
            break
    return clean_title


def normalize_episode_key(raw: str | None) -> str | None:
    if not raw:
        return None
    match = re.search(r"ep(\d+)", str(raw), flags=re.IGNORECASE)
    if not match:
        return None
    return f"ep{int(match.group(1)):02d}"


def episode_sort_key(raw: str | None) -> tuple[int, str]:
    episode_id = normalize_episode_key(raw)
    if not episode_id:
        return (10**9, str(raw or ""))
    return (int(re.search(r"(\d+)", episode_id).group(1)), episode_id)


def split_character_title(title: str) -> tuple[str, str]:
    normalized = normalize_spaces(title)
    if "｜" in normalized:
        name, variant = normalized.split("｜", 1)
    else:
        name, variant = normalized, ""
    name = re.split(r"[（(]", name, maxsplit=1)[0].strip()
    return name or normalized, variant.strip()


def character_variant_priority(label: str | None) -> tuple[int, int]:
    raw = normalize_spaces(label or "")
    if not raw:
        return (9, 0)
    priorities = [
        ("首集正式出场", 0),
        ("正式出场", 0),
        ("首集基础", 0),
        ("基础主形象", 0),
        ("基础", 1),
        ("主形象", 1),
        ("主设", 1),
        ("标准", 2),
    ]
    for keyword, priority in priorities:
        if keyword in raw:
            return (priority, len(raw))
    return (5, len(raw))


def ref_token_regex(ref_id: str) -> re.Pattern[str]:
    return re.compile(rf"{re.escape(str(ref_id or '').strip())}(?!\d)")


def text_contains_ref_token(text: str, ref_id: str) -> bool:
    if not text or not ref_id:
        return False
    return ref_token_regex(ref_id).search(str(text)) is not None


def replace_first_ref_token(text: str, ref_id: str, replacement: str) -> tuple[str, bool]:
    if not text or not ref_id:
        return str(text or ""), False
    replaced, count = ref_token_regex(ref_id).subn(replacement, str(text), count=1)
    return replaced, bool(count)


NAME_TOKEN_NEIGHBOR_CLASS = r"[\u4e00-\u9fffA-Za-z0-9_]"


def lookup_name_token_regex(lookup_name: str) -> re.Pattern[str]:
    escaped = re.escape(str(lookup_name or "").strip())
    return re.compile(rf"(?<!{NAME_TOKEN_NEIGHBOR_CLASS}){escaped}(?!{NAME_TOKEN_NEIGHBOR_CLASS})")


def text_contains_lookup_name_token(text: str, lookup_name: str) -> bool:
    if not text or not lookup_name:
        return False
    return lookup_name_token_regex(lookup_name).search(str(text)) is not None


def replace_first_lookup_name_token(text: str, lookup_name: str, replacement: str) -> tuple[str, bool]:
    if not text or not lookup_name:
        return str(text or ""), False
    replaced, count = lookup_name_token_regex(lookup_name).subn(replacement, str(text), count=1)
    return replaced, bool(count)


def resolve_series_name(config: Mapping[str, Any]) -> str:
    explicit = str(config.get("series", {}).get("series_name") or "").strip()
    if explicit:
        return explicit
    script_path = str(config.get("script", {}).get("script_path") or "").strip()
    if script_path:
        return build_series_paths(project_root=PROJECT_ROOT, script_path=script_path).series_name
    series_dir = str(config.get("script", {}).get("series_dir") or "").strip()
    if series_dir:
        return Path(series_dir).expanduser().resolve().name
    raise RuntimeError("无法推导剧名，请提供 series.series_name 或 script.series_dir。")


def resolve_assets_dir(config: Mapping[str, Any], series_name: str) -> Path:
    output_config = config.get("output", {})
    explicit_assets_series_name = str(output_config.get("assets_series_name") or "").strip()
    assets_series_suffix = str(output_config.get("assets_series_suffix") or "-gpt").strip()
    target_series_name = explicit_assets_series_name or f"{series_name}{assets_series_suffix}"
    return (PROJECT_ROOT / "assets" / target_series_name).resolve()


def resolve_outputs_root(config: Mapping[str, Any]) -> Path:
    outputs_root = Path(config.get("output", {}).get("outputs_root") or "outputs").expanduser()
    if not outputs_root.is_absolute():
        outputs_root = (PROJECT_ROOT / outputs_root).resolve()
    return outputs_root


def resolve_output_series_name(config: Mapping[str, Any], series_name: str) -> str:
    output_config = config.get("output", {})
    explicit = str(output_config.get("outputs_series_name") or "").strip()
    suffix = str(output_config.get("outputs_series_suffix") or "-gpt").strip()
    return explicit or f"{series_name}{suffix}"


def resolve_episode_output_dir(config: Mapping[str, Any], series_name: str, episode_id: str) -> Path:
    return resolve_outputs_root(config) / resolve_output_series_name(config, series_name) / episode_id


def resolve_storyboard_profile(config: Mapping[str, Any]) -> str:
    raw = (
        config.get("run", {}).get("storyboard_profile")
        or config.get("runtime", {}).get("storyboard_profile")
        or config.get("quality", {}).get("storyboard_profile")
        or "normal"
    )
    return ctx_compact.normalize_storyboard_profile(str(raw))


def storyboard_profile_settings(profile: str) -> dict[str, Any]:
    normalized = ctx_compact.normalize_storyboard_profile(profile)
    if normalized == "fast":
        return {
            "profile": "fast",
            "label": "极速版",
            "draft_character_limit": 560,
            "draft_scene_limit": 1300,
            "draft_storyboard_agent_limit": 220,
            "draft_storyboard_skill_limit": 560,
            "draft_methodology_limit": 520,
            "draft_examples_limit": 160,
            "draft_template_limit": 160,
            "review_methodology_limit": 320,
            "review_skill_limit": 220,
            "review_compliance_limit": 140,
            "max_shot_beats": 6,
            "continuity_max_chars": 84,
            "continuity_max_clauses": 2,
            "max_dialogue_entries": 4,
            "dialogue_gap_seconds": 0.25,
        }
    return {
        "profile": "normal",
        "label": "Normal",
        "draft_character_limit": 1100,
        "draft_scene_limit": 3600,
        "draft_storyboard_agent_limit": 320,
        "draft_storyboard_skill_limit": 1200,
        "draft_methodology_limit": 1200,
        "draft_examples_limit": 320,
        "draft_template_limit": 260,
        "review_methodology_limit": 700,
        "review_skill_limit": 520,
        "review_compliance_limit": 220,
        "max_shot_beats": 12,
        "continuity_max_chars": 220,
        "continuity_max_clauses": 4,
        "max_dialogue_entries": 6,
        "dialogue_gap_seconds": 0.2,
    }


def storyboard_mode_guidance(profile: str) -> str:
    normalized = ctx_compact.normalize_storyboard_profile(profile)
    if normalized == "fast":
        return (
            "当前 storyboard 模式：fast（极速版）。目标是尽快保住最关键、最有效的信息，"
            "优先保留人物关系、主要动作、关键道具、核心场景锚点和最重要的高能反馈。"
            " `prompt_text` 以 1 句总起 + 2-4 个时间段短段为主，节奏直接，不追求极致铺陈。"
            " 只需把镜头节拍、对白窗和声音触发写进 `master_timeline`，派生字段由系统自动生成。"
            " 静态说明与弱过场要更快带过，把篇幅集中在动作、关系推进和最重要的爆点上。"
        )
    return (
        "当前 storyboard 模式：normal（细节优先）。目标是在不写废话、不重复字段标签的前提下，"
        "尽量保留最多、最有效的成片信息。`prompt_text` 应明显厚于 fast，优先保留人物身体状态变化、"
        "前中后景站位、镜头路径、主光方向、关键道具手位、特效源头与传播路径、环境反馈、对白前后的反应停顿与尾帧交棒。"
        " 同时允许保留更完整的宏大场景基底，例如主建筑体量、纵深轴线、外圈阵列、远端天际线、镜门/高台/长阶等层级关系，"
        "让同一场戏在后续分镜里更容易保持稳定的空间一致性。"
        " 单条建议写成 1 句总起 + 4-8 个时间段短段，尽量做到语义丰满、逻辑顺滑、细节可执行。"
        " 只需把镜头节拍、对白窗和声音触发写进 `master_timeline`，派生字段由系统自动生成。"
        " 对不重要的静态描述和重复站位要主动加快，对动作推进、专业分镜调度、特效兑现和大场面逻辑要主动写得更足。"
        " 宁可删掉重复修饰和作者评论，也不要删掉能提升成片质量的具体信息；当宏大空间本身就是剧情压力来源时，"
        "可以额外保留 1-2 组高价值环境锚点，而不必一味压回局部表演区。"
    )


def storyboard_metrics_paths(episode_output_dir: Path, provider_tag: str) -> tuple[Path, Path]:
    base = episode_output_dir / f"02-seedance-prompts.metrics__{provider_tag}"
    return Path(f"{base}.json"), Path(f"{base}.md")


def dump_prompt_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_prompt_with_debug_metadata(
    relative_path: str | Path,
    context: Mapping[str, Any],
    *,
    top_n: int = 10,
) -> tuple[str, dict[str, Any]]:
    prompt = render_prompt(relative_path, context)
    section_chars = {
        key: len(str(value))
        for key, value in context.items()
        if str(value)
    }
    total_chars = max(len(prompt), 1)
    top_sections = [
        {
            "name": key,
            "chars": chars,
            "pct": round(chars / total_chars * 100, 2),
        }
        for key, chars in sorted(section_chars.items(), key=lambda item: item[1], reverse=True)[:top_n]
    ]
    template_static_chars = len(render_prompt(relative_path, {key: "" for key in context}))
    return (
        prompt,
        {
            "template_static_chars": template_static_chars,
            "prompt_section_chars": section_chars,
            "prompt_top_sections": top_sections,
        },
    )


def render_storyboard_metrics_markdown(report: Mapping[str, Any]) -> str:
    context = dict(report.get("context", {}) or {})
    totals = dict(report.get("totals", {}) or {})
    stage_totals = dict(report.get("stage_totals", {}) or {})
    error_message = str(context.get("error") or "").strip()
    lines = [
        "# Storyboard 统计报告",
        "",
        f"- run_name：{report.get('run_name', '')}",
        f"- series_name：{context.get('series_name', '')}",
        f"- episode_id：{context.get('episode_id', '')}",
        f"- model：{context.get('model', '')}",
        f"- storyboard_profile：{context.get('storyboard_profile', '')}",
        f"- final_status：{context.get('final_status', '')}",
    ]
    if error_message:
        lines.append(f"- error：{error_message}")
    lines.extend(
        [
            "",
            "## 总计",
            "",
            f"- steps：{totals.get('step_count', 0)}",
            f"- duration_seconds：{totals.get('duration_seconds', 0)}",
            f"- input_tokens：{totals.get('input_tokens', 0)}",
            f"- output_tokens：{totals.get('output_tokens', 0)}",
            f"- total_tokens：{totals.get('total_tokens', 0)}",
            "",
            "## 阶段汇总",
            "",
            "| 阶段 | 步骤数 | 耗时(秒) | 输入tokens | 输出tokens | 总tokens |",
            "|------|--------|---------:|-----------:|-----------:|---------:|",
        ]
    )
    for stage, bucket in stage_totals.items():
        lines.append(
            f"| {stage} | {bucket.get('step_count', 0)} | {bucket.get('duration_seconds', 0)} | "
            f"{bucket.get('input_tokens', 0)} | {bucket.get('output_tokens', 0)} | {bucket.get('total_tokens', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 细粒度步骤",
            "",
            "| Step ID | 阶段 | 名称 | 状态 | 耗时(秒) | 输入tokens | 输出tokens | 总tokens | 备注 |",
            "|---------|------|------|------|---------:|-----------:|-----------:|---------:|------|",
        ]
    )
    for step in report.get("steps", []):
        metadata = dict(step.get("metadata", {}) or {})
        note_parts: list[str] = []
        if "prompt_chars" in metadata:
            note_parts.append(f"prompt_chars={metadata['prompt_chars']}")
        if "template_static_chars" in metadata:
            note_parts.append(f"template_static_chars={metadata['template_static_chars']}")
        if "storyboard_profile" in metadata:
            note_parts.append(f"profile={metadata['storyboard_profile']}")
        top_sections = list(metadata.get("prompt_top_sections") or [])
        if top_sections:
            preview = ",".join(
                f"{str(item.get('name') or '')}:{int(item.get('chars') or 0)}"
                for item in top_sections[:3]
                if str(item.get("name") or "").strip()
            )
            if preview:
                note_parts.append(f"top={preview}")
        lines.append(
            f"| {step.get('step_id', '')} | {step.get('stage', '')} | {step.get('name', '')} | {step.get('status', '')} | "
            f"{step.get('duration_seconds', 0)} | {step.get('input_tokens', 0)} | {step.get('output_tokens', 0)} | {step.get('total_tokens', 0)} | "
            f"{'；'.join(note_parts)} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def save_storyboard_metrics(recorder: TelemetryRecorder, json_path: Path, md_path: Path) -> dict[str, Any]:
    report = recorder.to_dict()
    save_json_file(json_path, report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_storyboard_metrics_markdown(report), encoding="utf-8")
    return report


def merge_telemetry_recorders(target: TelemetryRecorder | None, source: TelemetryRecorder | None) -> None:
    if target is None or source is None:
        return
    base_index = len(target.steps)
    for offset, step in enumerate(source.steps, start=1):
        copied = copy.deepcopy(step)
        copied["step_id"] = f"step-{base_index + offset:04d}"
        target.steps.append(copied)


def load_genre_reference_bundle(config: Mapping[str, Any], series_name: str) -> dict[str, Any]:
    explicit = str(config.get("sources", {}).get("genre_reference_bundle_path") or "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.exists():
            return load_json_file(path)
    candidate = PROJECT_ROOT / "analysis" / series_name / "openai_agent_flow" / "genre_reference_bundle.json"
    if candidate.exists():
        return load_json_file(candidate)
    return {}


def resolve_analysis_root(config: Mapping[str, Any]) -> Path:
    analysis_root = (
        str(config.get("run", {}).get("analysis_root") or "").strip()
        or str(config.get("output", {}).get("analysis_root") or "").strip()
        or "analysis"
    )
    path = Path(analysis_root).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def load_seedance_purpose_skill_library(config: Mapping[str, Any], series_name: str) -> dict[str, Any]:
    explicit = str(config.get("sources", {}).get("seedance_purpose_skill_library_path") or "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.exists():
            return load_json_file(path)
    candidate = resolve_analysis_root(config) / series_name / "seedance_purpose_skill_library.json"
    if candidate.exists():
        return load_json_file(candidate)
    return {}


def load_seedance_purpose_template_library(config: Mapping[str, Any], series_name: str) -> dict[str, Any]:
    explicit = str(config.get("sources", {}).get("seedance_purpose_template_library_path") or "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.exists():
            return load_json_file(path)
    candidate = resolve_analysis_root(config) / series_name / "seedance_purpose_template_library.json"
    if candidate.exists():
        return load_json_file(candidate)
    return {}


def normalize_story_point_primary_purpose(point: Mapping[str, Any], *, is_last: bool = False) -> str:
    raw = str(point.get("primary_purpose") or "").strip()
    if raw in PURPOSE_ORDER:
        return raw
    return infer_primary_purpose_from_parts(
        [
            point.get("title"),
            point.get("shot_group"),
            point.get("narrative_function"),
            point.get("continuity_hook_in"),
            point.get("opening_visual_state"),
            *list(point.get("micro_beats") or []),
            *list(point.get("detail_anchor_lines") or []),
            *list(point.get("key_dialogue_beats") or []),
            point.get("director_statement"),
            point.get("closing_visual_state"),
            point.get("continuity_hook_out"),
        ],
        is_last=is_last,
        fallback="对峙",
    )


def build_seedance_story_point_guidance(
    director_json: Mapping[str, Any] | None,
    skill_library: Mapping[str, Any],
    template_library: Mapping[str, Any],
    *,
    profile: str = "normal",
) -> dict[str, Any]:
    normalized_profile = ctx_compact.normalize_storyboard_profile(profile)
    is_fast = normalized_profile == "fast"
    story_points = list((director_json or {}).get("story_points") or [])
    skill_index = {
        str(item.get("purpose") or "").strip(): dict(item)
        for item in list(skill_library.get("purposes") or [])
        if str(item.get("purpose") or "").strip()
    }
    template_index = {
        str(item.get("purpose") or "").strip(): list(item.get("templates") or [])
        for item in list(template_library.get("purposes") or [])
        if str(item.get("purpose") or "").strip()
    }
    shared_rules = dict(skill_library.get("shared_series_rules") or {})

    guidance_points: list[dict[str, Any]] = []
    for index, point in enumerate(story_points):
        purpose = normalize_story_point_primary_purpose(point, is_last=index == len(story_points) - 1)
        skill_entry = dict(skill_index.get(purpose) or {})
        fallback_profile = PURPOSE_PROFILES.get(purpose, PURPOSE_PROFILES.get("对峙", {}))
        design_skill = {
            "camera_rules": list(fallback_profile.get("camera_rules") or []),
            "beat_rules": list(fallback_profile.get("beat_rules") or []),
            "action_rules": list(fallback_profile.get("action_rules") or []),
            "dialogue_rules": list(fallback_profile.get("dialogue_rules") or []),
            "continuity_rules": list(fallback_profile.get("continuity_rules") or []),
            "negative_patterns": list(fallback_profile.get("negative_patterns") or []),
        }
        design_skill.update(dict(skill_entry.get("design_skill") or {}))
        template_examples = []
        for template in list(template_index.get(purpose) or [])[:1 if is_fast else 2]:
            template_examples.append(
                {
                    "template_id": str(template.get("template_id") or ""),
                    "when_to_use": ctx_compact.shorten_text(str(template.get("when_to_use") or ""), 72 if is_fast else 108),
                    "required_slots": list(template.get("required_slots") or [])[:3 if is_fast else 5],
                    "shot_outline": [
                        ctx_compact.shorten_text(str(item or ""), 72 if is_fast else 108)
                        for item in list(template.get("shot_outline") or [])[:2 if is_fast else 4]
                    ],
                    "generalized_template_prompt": ctx_compact.shorten_text(
                        str(template.get("generalized_template_prompt") or ""),
                        220 if is_fast else 420,
                    ),
                }
            )
        guidance_points.append(
            {
                "point_id": str(point.get("point_id") or ""),
                "title": ctx_compact.shorten_text(str(point.get("title") or ""), 64 if is_fast else 96),
                "primary_purpose": purpose,
                "purpose_description": ctx_compact.shorten_text(
                    str(skill_entry.get("description") or fallback_profile.get("description") or ""),
                    88 if is_fast else 128,
                ),
                "when_to_use": ctx_compact.shorten_text(
                    str(skill_entry.get("when_to_use") or fallback_profile.get("when_to_use") or ""),
                    84 if is_fast else 128,
                ),
                "purpose_rules": {
                    "camera_rules": [
                        ctx_compact.shorten_text(str(item or ""), 84 if is_fast else 112)
                        for item in list(design_skill.get("camera_rules") or [])[:2 if is_fast else 3]
                    ],
                    "beat_rules": [
                        ctx_compact.shorten_text(str(item or ""), 84 if is_fast else 112)
                        for item in list(design_skill.get("beat_rules") or [])[:2 if is_fast else 3]
                    ],
                    "action_rules": [
                        ctx_compact.shorten_text(str(item or ""), 84 if is_fast else 112)
                        for item in list(design_skill.get("action_rules") or [])[:2 if is_fast else 3]
                    ],
                    "dialogue_rules": [
                        ctx_compact.shorten_text(str(item or ""), 84 if is_fast else 112)
                        for item in list(design_skill.get("dialogue_rules") or [])[:2 if is_fast else 3]
                    ],
                    "continuity_rules": [
                        ctx_compact.shorten_text(str(item or ""), 84 if is_fast else 112)
                        for item in list(design_skill.get("continuity_rules") or [])[:2 if is_fast else 3]
                    ],
                    "negative_patterns": [
                        ctx_compact.shorten_text(str(item or ""), 84 if is_fast else 112)
                        for item in list(design_skill.get("negative_patterns") or [])[:2 if is_fast else 3]
                    ],
                },
                "template_examples": template_examples,
            }
        )

    return {
        "series_name": str(skill_library.get("series_name") or template_library.get("series_name") or ""),
        "taxonomy_version": str(skill_library.get("taxonomy_version") or template_library.get("taxonomy_version") or ""),
        "shared_series_rules": {
            "camera_language_rules": [
                ctx_compact.shorten_text(str(item or ""), 82 if is_fast else 104)
                for item in list(shared_rules.get("camera_language_rules") or [])[:3]
            ],
            "storyboard_execution_rules": [
                ctx_compact.shorten_text(str(item or ""), 82 if is_fast else 104)
                for item in list(shared_rules.get("storyboard_execution_rules") or [])[:3]
            ],
            "dialogue_timing_rules": [
                ctx_compact.shorten_text(str(item or ""), 82 if is_fast else 104)
                for item in list(shared_rules.get("dialogue_timing_rules") or [])[:3]
            ],
            "continuity_guardrails": [
                ctx_compact.shorten_text(str(item or ""), 82 if is_fast else 104)
                for item in list(shared_rules.get("continuity_guardrails") or [])[:3]
            ],
            "negative_patterns": [
                ctx_compact.shorten_text(str(item or ""), 82 if is_fast else 104)
                for item in list(shared_rules.get("negative_patterns") or [])[:3]
            ],
        },
        "story_points": guidance_points,
    }


def compact_seedance_story_point_guidance_for_prompt(
    data: Mapping[str, Any] | None,
    *,
    profile: str = "normal",
) -> dict[str, Any]:
    normalized_profile = ctx_compact.normalize_storyboard_profile(profile)
    is_fast = normalized_profile == "fast"
    payload = dict(data or {})
    shared_rules = dict(payload.get("shared_series_rules") or {})

    used_purposes: dict[str, dict[str, Any]] = {}
    compact_story_points: list[dict[str, Any]] = []
    for point in list(payload.get("story_points") or []):
        if not isinstance(point, Mapping):
            continue
        primary_purpose = str(point.get("primary_purpose") or "").strip()
        purpose_rules_payload = dict(point.get("purpose_rules") or {})
        compact_purpose_rules: dict[str, list[str]] = {}
        for field in [
            "camera_rules",
            "beat_rules",
            "action_rules",
            "dialogue_rules",
            "continuity_rules",
            "negative_patterns",
        ]:
            entries = [
                ctx_compact.shorten_text(str(item or ""), 72 if is_fast else 96)
                for item in list(purpose_rules_payload.get(field) or [])[:1 if is_fast else 2]
                if str(item or "").strip()
            ]
            if entries:
                compact_purpose_rules[field] = entries

        compact_template_examples: list[dict[str, Any]] = []
        for template in list(point.get("template_examples") or [])[:1]:
            if not isinstance(template, Mapping):
                continue
            compact_template_examples.append(
                {
                    "template_id": str(template.get("template_id") or ""),
                    "when_to_use": ctx_compact.shorten_text(
                        str(template.get("when_to_use") or ""),
                        56 if is_fast else 80,
                    ),
                    "shot_outline": [
                        ctx_compact.shorten_text(str(item or ""), 68 if is_fast else 92)
                        for item in list(template.get("shot_outline") or [])[:1 if is_fast else 2]
                        if str(item or "").strip()
                    ],
                    "generalized_template_prompt": ctx_compact.shorten_text(
                        str(template.get("generalized_template_prompt") or ""),
                        120 if is_fast else 220,
                    ),
                }
            )

        if primary_purpose and primary_purpose not in used_purposes:
            used_purposes[primary_purpose] = {
                "purpose_rules": compact_purpose_rules,
                "template_examples": compact_template_examples,
            }

        compact_story_points.append(
            {
                "point_id": str(point.get("point_id") or ""),
                "title": ctx_compact.shorten_text(str(point.get("title") or ""), 52 if is_fast else 80),
                "primary_purpose": primary_purpose,
            }
        )

    return {
        "series_name": str(payload.get("series_name") or ""),
        "taxonomy_version": str(payload.get("taxonomy_version") or ""),
        "shared_series_rules": {
            "camera_language_rules": [
                ctx_compact.shorten_text(str(item or ""), 68 if is_fast else 88)
                for item in list(shared_rules.get("camera_language_rules") or [])[:2]
                if str(item or "").strip()
            ],
            "storyboard_execution_rules": [
                ctx_compact.shorten_text(str(item or ""), 68 if is_fast else 88)
                for item in list(shared_rules.get("storyboard_execution_rules") or [])[:2]
                if str(item or "").strip()
            ],
            "dialogue_timing_rules": [
                ctx_compact.shorten_text(str(item or ""), 68 if is_fast else 88)
                for item in list(shared_rules.get("dialogue_timing_rules") or [])[:2]
                if str(item or "").strip()
            ],
            "continuity_guardrails": [
                ctx_compact.shorten_text(str(item or ""), 68 if is_fast else 88)
                for item in list(shared_rules.get("continuity_guardrails") or [])[:2]
                if str(item or "").strip()
            ],
            "negative_patterns": [
                ctx_compact.shorten_text(str(item or ""), 68 if is_fast else 88)
                for item in list(shared_rules.get("negative_patterns") or [])[:2]
                if str(item or "").strip()
            ],
        },
        "purpose_rulebook": used_purposes,
        "story_points": compact_story_points,
    }


def parse_episode_prompt_blocks(text: str) -> list[tuple[str | None, str]]:
    start_pattern = re.compile(r"<!--\s*episode:\s*(ep\d+)\s+start\s*-->", flags=re.IGNORECASE)
    matches = list(start_pattern.finditer(text))
    if not matches:
        return [(None, text)]

    blocks: list[tuple[str | None, str]] = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        blocks.append((None, text[: matches[0].start()]))

    for index, match in enumerate(matches):
        episode_id = f"ep{int(match.group(1)[2:]):02d}"
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end]
        end_pattern = re.compile(
            rf"<!--\s*episode:\s*{re.escape(episode_id)}\s+end\s*-->",
            flags=re.IGNORECASE,
        )
        body = end_pattern.sub("", body)
        blocks.append((episode_id, body))
    return blocks


def parse_character_assets(text: str, *, episode_id: str | None = None) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for block_episode_id, block in parse_episode_prompt_blocks(text):
        if not block_episode_id:
            continue
        normalized_block_episode_id = normalize_episode_key(block_episode_id)
        if episode_id and normalized_block_episode_id and episode_sort_key(normalized_block_episode_id) > episode_sort_key(episode_id):
            continue
        sections = re.split(r"^##\s+", block, flags=re.MULTILINE)
        for section in sections[1:]:
            lines = section.splitlines()
            if not lines:
                continue
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            if not title or title.startswith("ep") or "**提示词**" not in body:
                continue
            base_name, variant_label = split_character_title(title)
            if ctx_compact.is_mixed_crowd_character_asset(
                base_name or title,
                appearance_keywords=body,
                reuse_note=title,
            ):
                continue
            records.append(
                {
                    "asset_type": "人物参考",
                    "display_name": title,
                    "lookup_name": base_name or title,
                    "variant_label": variant_label or title,
                    "episode_id": normalized_block_episode_id or block_episode_id,
                }
            )
    if not episode_id:
        return [
            {
                "asset_type": str(item.get("asset_type") or ""),
                "display_name": str(item.get("display_name") or ""),
                "lookup_name": str(item.get("lookup_name") or ""),
            }
            for item in records
        ]

    selected_by_name: dict[str, dict[str, str]] = {}
    ordered_names: list[str] = []
    for item in records:
        lookup_name = str(item.get("lookup_name") or "").strip()
        if not lookup_name:
            continue
        key = normalize_spaces(lookup_name)
        current = selected_by_name.get(key)
        if current is None:
            selected_by_name[key] = item
            ordered_names.append(key)
            continue
        candidate_episode_key = episode_sort_key(item.get("episode_id"))
        current_episode_key = episode_sort_key(current.get("episode_id"))
        candidate_priority = character_variant_priority(item.get("variant_label") or item.get("display_name"))
        current_priority = character_variant_priority(current.get("variant_label") or current.get("display_name"))
        if candidate_episode_key > current_episode_key or (
            candidate_episode_key == current_episode_key and candidate_priority <= current_priority
        ):
            selected_by_name[key] = item

    assets: list[dict[str, str]] = []
    for key in ordered_names:
        item = selected_by_name.get(key)
        if not item:
            continue
        assets.append(
            {
                "asset_type": "人物参考",
                "display_name": str(item.get("display_name") or ""),
                "lookup_name": str(item.get("lookup_name") or ""),
            }
        )
    return assets


def parse_scene_assets(text: str, *, episode_id: str | None = None) -> list[dict[str, str]]:
    assets: list[dict[str, str]] = []
    for block_episode_id, block in parse_episode_prompt_blocks(text):
        if not block_episode_id:
            continue
        if episode_id and block_episode_id != episode_id:
            continue
        sections = re.split(r"^##\s+", block, flags=re.MULTILINE)
        for section in sections[1:]:
            lines = section.splitlines()
            if not lines:
                continue
            current_grid_title = lines[0].strip()
            if not current_grid_title:
                continue
            for line in lines[1:]:
                stripped = line.strip()
                if not stripped:
                    continue
                match = PANEL_LINE_PATTERN.match(stripped)
                if match:
                    slot_raw = match.group("slot").strip()
                    slot_number_match = re.search(r"(\d+)", slot_raw)
                    slot = f"格{int(slot_number_match.group(1))}" if slot_number_match else slot_raw
                    scene_name = match.group("scene").strip()
                else:
                    full_frame_match = FULL_FRAME_PANEL_LINE_PATTERN.match(stripped)
                    if not full_frame_match:
                        continue
                    slot = "格1"
                    scene_name = full_frame_match.group("scene").strip()
                assets.append(
                    {
                        "asset_type": "场景参考",
                        "display_name": f"{scene_name}（{current_grid_title} {slot}）".strip(),
                        "lookup_name": scene_name,
                    }
            )
    return assets


def extract_reference_number(raw: str | Path | None) -> int | None:
    text = str(raw or "").strip()
    token_match = re.search(r"@图片(?P<number>\d+)", text)
    if token_match:
        return int(token_match.group("number"))
    stem = Path(text).stem
    match = re.match(r"(?P<number>\d+)__", stem)
    if not match:
        return None
    return int(match.group("number"))


def find_latest_episode_generated_manifest(
    assets_dir: Path | None,
    *,
    episode_id: str,
    filename: str,
) -> Path | None:
    if assets_dir is None:
        return None
    generated_root = assets_dir / "generated"
    if not generated_root.exists():
        return None
    candidates = sorted(
        generated_root.glob(f"*/{episode_id}/{filename}"),
        key=lambda item: (item.stat().st_mtime_ns, str(item)),
    )
    return candidates[-1] if candidates else None


def load_generated_character_assets(
    assets_dir: Path | None,
    *,
    episode_id: str,
) -> list[dict[str, str]]:
    manifest_path = find_latest_episode_generated_manifest(
        assets_dir,
        episode_id=episode_id,
        filename="generation_manifest.json",
    )
    if manifest_path is None:
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    selected_by_name: dict[str, dict[str, str]] = {}
    for item in list(payload.get("items") or []):
        if str(item.get("prompt_type") or "") != "character":
            continue
        status = str(item.get("status") or "").strip()
        if status not in {"generated", "generated_with_reference", "reused_from_previous_episode", "skipped_existing"}:
            continue
        output_path = str(item.get("output_path") or "").strip()
        ref_number = extract_reference_number(output_path)
        source_label = str(item.get("source_label") or item.get("name") or "").strip()
        lookup_name, variant_label = split_character_title(source_label)
        if ref_number is None or not lookup_name:
            continue
        if ctx_compact.is_mixed_crowd_character_asset(
            lookup_name,
            appearance_keywords=source_label,
            reuse_note=str(item.get("output_path") or ""),
        ):
            continue
        source_episode_id = normalize_episode_key(source_label) or episode_id
        candidate = {
            "asset_type": "人物参考",
            "ref_id": f"@图片{ref_number}",
            "display_name": source_label or lookup_name,
            "lookup_name": lookup_name,
            "variant_label": variant_label or source_label or lookup_name,
            "source_episode_id": source_episode_id or "",
        }
        key = normalize_spaces(lookup_name)
        current = selected_by_name.get(key)
        if current is None:
            selected_by_name[key] = candidate
            continue
        candidate_episode_key = episode_sort_key(candidate.get("source_episode_id"))
        current_episode_key = episode_sort_key(current.get("source_episode_id"))
        candidate_priority = character_variant_priority(candidate.get("variant_label"))
        current_priority = character_variant_priority(current.get("variant_label"))
        candidate_ref_number = extract_reference_number(candidate.get("ref_id")) or 0
        current_ref_number = extract_reference_number(current.get("ref_id")) or 0
        if candidate_episode_key > current_episode_key or (
            candidate_episode_key == current_episode_key
            and (
                candidate_priority < current_priority
                or (candidate_priority == current_priority and candidate_ref_number >= current_ref_number)
            )
        ):
            selected_by_name[key] = candidate

    selected = list(selected_by_name.values())
    selected.sort(key=lambda item: extract_reference_number(item.get("ref_id")) or 10**9)
    return selected


def load_generated_scene_assets(
    assets_dir: Path | None,
    *,
    episode_id: str,
) -> list[dict[str, str]]:
    manifest_path = find_latest_episode_generated_manifest(
        assets_dir,
        episode_id=episode_id,
        filename="scene_material_manifest.json",
    )
    if manifest_path is None:
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    assets: list[dict[str, str]] = []
    for item in list(payload.get("items") or []):
        ref_number = item.get("reference_number")
        scene_name = str(item.get("scene_name") or "").strip()
        if not isinstance(ref_number, int) or not scene_name:
            continue
        grid_title = str(item.get("grid_title") or "").strip()
        panel_label = str(item.get("panel_label") or "").strip()
        suffix = " ".join(part for part in [grid_title, panel_label] if part).strip()
        display_name = f"{scene_name}（{suffix}）" if suffix else scene_name
        assets.append(
            {
                "asset_type": "场景参考",
                "ref_id": f"@图片{ref_number}",
                "display_name": display_name,
                "lookup_name": scene_name,
            }
        )
    assets.sort(key=lambda item: extract_reference_number(item.get("ref_id")) or 10**9)
    return assets


def build_asset_catalog(
    character_text: str,
    scene_text: str,
    *,
    episode_id: str | None = None,
    assets_dir: Path | None = None,
) -> list[dict[str, str]]:
    catalog: list[dict[str, str]] = []
    used_numbers: set[int] = set()

    parsed_character_assets = parse_character_assets(character_text, episode_id=episode_id)
    generated_character_assets = load_generated_character_assets(assets_dir, episode_id=episode_id) if episode_id else []
    if generated_character_assets and parsed_character_assets:
        generated_by_name: dict[str, dict[str, str]] = {}
        for item in generated_character_assets:
            lookup_name = normalize_spaces(item.get("lookup_name"))
            if lookup_name and lookup_name not in generated_by_name:
                generated_by_name[lookup_name] = item
        next_index = 1
        for parsed_item in parsed_character_assets:
            while next_index in used_numbers:
                next_index += 1
            used_numbers.add(next_index)
            lookup_name = str(parsed_item.get("lookup_name") or "").strip()
            generated_item = generated_by_name.get(normalize_spaces(lookup_name))
            catalog.append(
                {
                    "ref_id": f"@图片{next_index}",
                    "asset_type": "人物参考",
                    "display_name": str(
                        (generated_item or {}).get("display_name")
                        or parsed_item.get("display_name")
                        or lookup_name
                    ),
                    "lookup_name": lookup_name,
                }
            )
            next_index += 1
    elif generated_character_assets:
        for item in generated_character_assets:
            ref_number = extract_reference_number(item.get("ref_id"))
            if ref_number is not None:
                used_numbers.add(ref_number)
            catalog.append(
                {
                    "ref_id": str(item.get("ref_id") or ""),
                    "asset_type": "人物参考",
                    "display_name": str(item.get("display_name") or ""),
                    "lookup_name": str(item.get("lookup_name") or ""),
                }
            )
    else:
        next_index = 1
        for item in parsed_character_assets:
            while next_index in used_numbers:
                next_index += 1
            used_numbers.add(next_index)
            catalog.append({"ref_id": f"@图片{next_index}", **item})
            next_index += 1

    generated_scene_assets = load_generated_scene_assets(assets_dir, episode_id=episode_id) if episode_id else []
    if generated_scene_assets:
        for item in generated_scene_assets:
            ref_number = extract_reference_number(item.get("ref_id"))
            if ref_number is not None:
                used_numbers.add(ref_number)
            catalog.append(
                {
                    "ref_id": str(item.get("ref_id") or ""),
                    "asset_type": "场景参考",
                    "display_name": str(item.get("display_name") or ""),
                    "lookup_name": str(item.get("lookup_name") or ""),
                }
            )
    else:
        next_index = (max(used_numbers) + 1) if used_numbers else 1
        for item in parse_scene_assets(scene_text, episode_id=episode_id):
            while next_index in used_numbers:
                next_index += 1
            used_numbers.add(next_index)
            catalog.append({"ref_id": f"@图片{next_index}", **item})
            next_index += 1

    catalog.sort(key=lambda item: extract_reference_number(item.get("ref_id")) or 10**9)
    return catalog


def build_catalog_text(catalog: list[Mapping[str, str]]) -> str:
    if not catalog:
        return "<空>"
    lines = []
    for item in catalog:
        lines.append(f"- {item['ref_id']} | {item['asset_type']} | {item['display_name']}")
    return "\n".join(lines)


def normalize_scene_match_text(raw: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", normalize_spaces(raw))


def chinese_bigrams(raw: str) -> list[str]:
    chars = [char for char in str(raw or "") if "\u4e00" <= char <= "\u9fff"]
    if len(chars) < 2:
        return chars
    return ["".join(chars[index:index + 2]) for index in range(len(chars) - 1)]


def extract_scene_match_phrases(raw: str) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    raw_candidates = [normalize_spaces(raw)] + re.split(r"[、，,/]|与|及|和|并", raw)
    expanded_candidates: list[str] = []
    for candidate in raw_candidates:
        phrase = normalize_spaces(candidate)
        if not phrase:
            continue
        expanded_candidates.append(phrase)
        if phrase.startswith("斩龙台") and len(phrase) > len("斩龙台") + 1:
            expanded_candidates.append(phrase[len("斩龙台"):])
        if phrase.startswith("远处") and len(phrase) > len("远处") + 1:
            expanded_candidates.append(phrase[len("远处"):])
        if phrase.endswith("区域") and len(phrase) > len("区域") + 1:
            expanded_candidates.append(phrase[:-len("区域")])
    for candidate in expanded_candidates:
        phrase = normalize_spaces(candidate)
        normalized = normalize_scene_match_text(phrase)
        if len(normalized) < 2:
            continue
        if normalized in {"场景", "区域", "空间", "位置", "地点", "画面", "环境"}:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        phrases.append(phrase)
    return phrases


def infer_scene_refs_for_entry(
    item: Mapping[str, Any],
    scene_catalog: list[Mapping[str, str]],
    *,
    extra_search_text: str = "",
) -> list[str]:
    extra_haystack = normalize_scene_match_text(extra_search_text)
    search_text = " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("prompt_text") or ""),
            str(item.get("continuity_bridge") or ""),
            str(item.get("audio_design") or ""),
            " ".join(str(x).strip() for x in list(item.get("shot_beat_plan") or []) if str(x).strip()),
            str(extra_search_text or ""),
        ]
    )
    haystack = normalize_scene_match_text(search_text)
    if not haystack:
        return []

    scored_matches: list[tuple[int, str]] = []
    for scene_item in scene_catalog:
        ref_id = str(scene_item.get("ref_id") or "").strip()
        if not ref_id:
            continue
        scene_name = str(scene_item.get("lookup_name") or scene_item.get("display_name") or "").strip()
        if not scene_name:
            continue
        score = 0
        normalized_scene_name = normalize_scene_match_text(scene_name)
        if normalized_scene_name and normalized_scene_name in haystack:
            score += 120
        if normalized_scene_name and extra_haystack and normalized_scene_name in extra_haystack:
            score += 90
        scene_bigrams = {token for token in chinese_bigrams(scene_name) if token}
        bigram_overlap = sum(1 for token in scene_bigrams if token in haystack)
        if bigram_overlap >= 2:
            score += 20 + min(bigram_overlap, 4) * 12
        elif bigram_overlap == 1:
            score += 8
        matched_phrases = 0
        for phrase in extract_scene_match_phrases(scene_name):
            normalized_phrase = normalize_scene_match_text(phrase)
            if not normalized_phrase or normalized_phrase not in haystack:
                continue
            matched_phrases += 1
            score += 40 if normalized_phrase == normalized_scene_name else 28
        if matched_phrases >= 2:
            score += 16
        if score > 0:
            scored_matches.append((score, ref_id))

    if not scored_matches:
        return []

    scored_matches.sort(key=lambda item: (-item[0], item[1]))
    top_score = scored_matches[0][0]
    selected: list[str] = []
    follow_score_floor = max(40, top_score - 120)
    for score, ref_id in scored_matches:
        if score < 40:
            break
        if score < follow_score_floor:
            continue
        if ref_id not in selected:
            selected.append(ref_id)
        if len(selected) >= 4:
            break
    return selected


def infer_character_refs_for_entry(
    item: Mapping[str, Any],
    character_catalog: list[Mapping[str, str]],
    *,
    max_refs: int = 6,
) -> list[str]:
    text_buckets: list[tuple[str, int]] = [
        (str(item.get("title") or ""), 50),
        (str(item.get("continuity_bridge") or ""), 35),
        (str(item.get("prompt_text") or ""), 90),
        (str(item.get("audio_design") or ""), 8),
    ]
    master_timeline = [dict(entry) for entry in list(item.get("master_timeline") or []) if isinstance(entry, Mapping)]
    for entry in master_timeline:
        text_buckets.extend(
            [
                (str(entry.get("visual_beat") or ""), 48),
                (str(entry.get("transition_hook") or ""), 28),
                (str(entry.get("speaker") or ""), 36),
                (str(entry.get("spoken_line") or ""), 6),
                (" ".join(
                    str(block.get("speaker") or "").strip()
                    for block in list(entry.get("dialogue_blocks") or [])
                    if isinstance(block, Mapping) and str(block.get("speaker") or "").strip()
                ), 24),
            ]
        )

    scored_matches: list[tuple[int, str]] = []
    for character_item in character_catalog:
        ref_id = str(character_item.get("ref_id") or "").strip()
        lookup_name = str(character_item.get("lookup_name") or character_item.get("display_name") or "").strip()
        if not ref_id or not lookup_name:
            continue
        score = 0
        lookup_pattern = lookup_name_token_regex(lookup_name)
        for text, weight in text_buckets:
            if not text or lookup_pattern.search(str(text)) is None:
                continue
            sanitized = str(text)
            for prefix in FUTURE_CHARACTER_HOOK_PREFIXES:
                sanitized = re.sub(rf"{re.escape(prefix)}\s*{re.escape(lookup_name)}", prefix, sanitized)
            mention_count = len(lookup_pattern.findall(sanitized))
            if mention_count <= 0:
                continue
            score += min(mention_count, 3) * weight
        if score > 0:
            scored_matches.append((score, ref_id))

    if not scored_matches:
        return []

    scored_matches.sort(key=lambda item: (-item[0], item[1]))
    selected: list[str] = []
    for score, ref_id in scored_matches:
        if score < 30:
            break
        if ref_id not in selected:
            selected.append(ref_id)
        if len(selected) >= max_refs:
            break
    return selected


def build_storyboard_asset_scope(
    director_json: Mapping[str, Any] | None,
    *,
    draft_package: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(director_json or {})
    story_points = [dict(item) for item in list(payload.get("story_points") or []) if isinstance(item, Mapping)]
    text_parts: list[str] = [
        str(payload.get("episode_title") or ""),
        str(payload.get("structure_overview") or ""),
        str(payload.get("emotional_curve") or ""),
    ]
    character_names: list[str] = []
    scene_names: list[str] = []

    for item in list(payload.get("characters") or []):
        if isinstance(item, Mapping):
            name = str(item.get("name") or "").strip()
            if name and not ctx_compact.is_mixed_crowd_character_asset(
                name,
                appearance_keywords=str(item.get("appearance_keywords") or ""),
                reuse_note=str(item.get("reuse_note") or ""),
            ):
                character_names.append(name)
    for item in list(payload.get("scenes") or []):
        if isinstance(item, Mapping):
            name = str(item.get("name") or "").strip()
            if name:
                scene_names.append(name)

    for point in story_points:
        text_parts.extend(
            [
                str(point.get("point_id") or ""),
                str(point.get("title") or ""),
                str(point.get("shot_group") or ""),
                str(point.get("narrative_function") or ""),
                str(point.get("entry_state") or point.get("continuity_hook_in") or ""),
                str(point.get("opening_visual_state") or ""),
                str(point.get("director_statement") or ""),
                str(point.get("closing_visual_state") or ""),
                str(point.get("exit_state") or point.get("continuity_hook_out") or ""),
                " ".join(str(x).strip() for x in list(point.get("micro_beats") or []) if str(x).strip()),
                " ".join(str(x).strip() for x in list(point.get("detail_anchor_lines") or []) if str(x).strip()),
                " ".join(str(x).strip() for x in list(point.get("key_dialogue_beats") or []) if str(x).strip()),
            ]
        )
        character_names.extend(
            str(x).strip()
            for x in list(point.get("characters") or [])
            if str(x).strip() and not ctx_compact.is_mixed_crowd_character_asset(str(x).strip())
        )
        scene_names.extend(str(x).strip() for x in list(point.get("scenes") or []) if str(x).strip())

    draft_text = ""
    draft_refs: set[str] = set()
    if draft_package:
        try:
            draft_text = json.dumps(draft_package, ensure_ascii=False)
        except TypeError:
            draft_text = str(draft_package)
        draft_refs = {str(ref).strip() for ref in used_ref_ids(draft_package) if str(ref).strip()}

    unique_character_names: list[str] = []
    for item in character_names:
        if item and item not in unique_character_names:
            unique_character_names.append(item)
    unique_scene_names: list[str] = []
    for item in scene_names:
        if item and item not in unique_scene_names:
            unique_scene_names.append(item)

    return {
        "director_text": " ".join(part for part in text_parts if part).strip(),
        "draft_text": draft_text,
        "draft_refs": draft_refs,
        "character_names": unique_character_names,
        "scene_names": unique_scene_names,
    }


def prioritize_storyboard_prompt_asset_catalog(
    asset_catalog: Sequence[Mapping[str, Any]],
    director_json: Mapping[str, Any] | None,
    *,
    draft_package: Mapping[str, Any] | None = None,
    profile: str = "normal",
) -> list[dict[str, str]]:
    normalized_profile = ctx_compact.normalize_storyboard_profile(profile)
    is_fast = normalized_profile == "fast"
    character_quota = 8 if is_fast else 12
    scene_quota = 6 if is_fast else 8
    scope = build_storyboard_asset_scope(director_json, draft_package=draft_package)
    director_text = str(scope.get("director_text") or "")
    draft_text = str(scope.get("draft_text") or "")
    draft_refs = {str(ref).strip() for ref in set(scope.get("draft_refs") or set()) if str(ref).strip()}
    character_names = {normalize_spaces(name) for name in list(scope.get("character_names") or []) if normalize_spaces(name)}
    scene_names = {
        normalize_scene_match_text(name)
        for name in list(scope.get("scene_names") or [])
        if normalize_scene_match_text(name)
    }
    normalized_director_scene_text = normalize_scene_match_text(director_text)
    normalized_draft_scene_text = normalize_scene_match_text(draft_text)

    def _base_ref_sort_key(item: Mapping[str, Any]) -> tuple[int, str]:
        return (extract_reference_number(item.get("ref_id")) or 10**9, str(item.get("ref_id") or ""))

    def _scored_character_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
        scored: list[tuple[int, tuple[int, str], dict[str, str]]] = []
        for item in items:
            ref_id = str(item.get("ref_id") or "").strip()
            lookup_name = str(item.get("lookup_name") or item.get("display_name") or "").strip()
            if not ref_id or not lookup_name:
                continue
            score = 0
            normalized_name = normalize_spaces(lookup_name)
            lookup_pattern = lookup_name_token_regex(lookup_name)
            if normalized_name in character_names:
                score += 260
            score += min(len(lookup_pattern.findall(director_text)), 3) * 60
            score += min(len(lookup_pattern.findall(draft_text)), 3) * 80
            if ref_id in draft_refs:
                score += 320
            scored.append((score, _base_ref_sort_key(item), item))
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        selected = [item for score, _, item in scored if score > 0][:character_quota]
        selected_refs = {str(item.get("ref_id") or "").strip() for item in selected}
        for item in items:
            ref_id = str(item.get("ref_id") or "").strip()
            if ref_id and ref_id not in selected_refs and len(selected) < character_quota:
                selected.append(item)
                selected_refs.add(ref_id)
        return selected

    def _scored_scene_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
        scored: list[tuple[int, tuple[int, str], dict[str, str]]] = []
        for item in items:
            ref_id = str(item.get("ref_id") or "").strip()
            lookup_name = str(item.get("lookup_name") or item.get("display_name") or "").strip()
            if not ref_id or not lookup_name:
                continue
            score = 0
            normalized_name = normalize_scene_match_text(lookup_name)
            if normalized_name in scene_names:
                score += 260
            if normalized_name and normalized_name in normalized_director_scene_text:
                score += 140
            if normalized_name and normalized_name in normalized_draft_scene_text:
                score += 170
            scene_bigrams = {token for token in chinese_bigrams(lookup_name) if token}
            bigram_overlap = sum(1 for token in scene_bigrams if token in normalized_director_scene_text)
            if bigram_overlap >= 2:
                score += 30 + min(bigram_overlap, 4) * 16
            elif bigram_overlap == 1:
                score += 10
            if ref_id in draft_refs:
                score += 320
            scored.append((score, _base_ref_sort_key(item), item))
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        selected = [item for score, _, item in scored if score > 0][:scene_quota]
        selected_refs = {str(item.get("ref_id") or "").strip() for item in selected}
        for item in items:
            ref_id = str(item.get("ref_id") or "").strip()
            if ref_id and ref_id not in selected_refs and len(selected) < scene_quota:
                selected.append(item)
                selected_refs.add(ref_id)
        return selected

    characters = [
        {
            "ref_id": str(item.get("ref_id") or ""),
            "asset_type": str(item.get("asset_type") or ""),
            "display_name": str(item.get("display_name") or ""),
            "lookup_name": str(item.get("lookup_name") or ""),
        }
        for item in list(asset_catalog)
        if str(item.get("asset_type") or "") == "人物参考"
    ]
    scenes = [
        {
            "ref_id": str(item.get("ref_id") or ""),
            "asset_type": str(item.get("asset_type") or ""),
            "display_name": str(item.get("display_name") or ""),
            "lookup_name": str(item.get("lookup_name") or ""),
        }
        for item in list(asset_catalog)
        if str(item.get("asset_type") or "") == "场景参考"
    ]

    selected_refs = {
        str(item.get("ref_id") or "").strip()
        for item in (_scored_character_items(characters) + _scored_scene_items(scenes))
        if str(item.get("ref_id") or "").strip()
    }
    prioritized = [
        {
            "ref_id": str(item.get("ref_id") or ""),
            "asset_type": str(item.get("asset_type") or ""),
            "display_name": str(item.get("display_name") or ""),
            "lookup_name": str(item.get("lookup_name") or ""),
        }
        for item in list(asset_catalog)
        if str(item.get("ref_id") or "").strip() in selected_refs
    ]
    return prioritized


def find_director_markdown_path(config: Mapping[str, Any], series_name: str, episode_id: str) -> Path:
    explicit = str(config.get("sources", {}).get("director_analysis_path") or "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"指定的导演讲戏本不存在：{path}")
        return path
    candidate = resolve_episode_output_dir(config, series_name, episode_id) / "01-director-analysis.md"
    if candidate.exists():
        return candidate
    alternate = resolve_outputs_root(config) / series_name / episode_id / "01-director-analysis.md"
    if alternate.exists():
        return alternate
    legacy = resolve_outputs_root(config) / episode_id / "01-director-analysis.md"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"未找到导演讲戏本：{candidate}")


def find_director_json_path(config: Mapping[str, Any], series_name: str, episode_id: str) -> Path | None:
    episode_dir = resolve_episode_output_dir(config, series_name, episode_id)
    preferred_model = str(config.get("provider", {}).get("openai", {}).get("model") or "gpt-5.4").strip()
    preferred_tag = build_provider_model_tag("openai", preferred_model)
    preferred = episode_dir / f"01-director-analysis__{preferred_tag}.json"
    if preferred.exists():
        return preferred
    candidates = sorted(episode_dir.glob("01-director-analysis__*.json"))
    return candidates[-1] if candidates else None


def build_episode_prompt_context(
    *,
    config: Mapping[str, Any],
    series_name: str,
    episode_id: str,
    director_markdown: str,
    director_json: Mapping[str, Any] | None,
    genre_reference_bundle: Mapping[str, Any],
    seedance_story_point_guidance: Mapping[str, Any],
    character_prompts_text: str,
    scene_prompts_text: str,
    asset_catalog: list[Mapping[str, str]],
) -> str:
    profile = resolve_storyboard_profile(config)
    settings = storyboard_profile_settings(profile)
    storyboard_agent = read_text(PROJECT_ROOT / ".claude/agents/storyboard-artist.md")
    storyboard_skill = load_skill("production/seedance-storyboard-skill/SKILL.md")
    methodology = read_text(PROJECT_ROOT / "skills/production/seedance-storyboard-skill/seedance-prompt-methodology.md")
    compact_director_brief = ctx_compact.compact_director_brief_for_storyboard(
        director_json,
        director_markdown,
        profile=profile,
    )
    compact_director_json = ctx_compact.compact_director_json_for_storyboard(director_json, profile=profile)
    compact_genre_reference_bundle = ctx_compact.compact_genre_reference_bundle_for_storyboard(genre_reference_bundle, profile=profile)
    compact_seedance_story_point_guidance = compact_seedance_story_point_guidance_for_prompt(
        seedance_story_point_guidance,
        profile=profile,
    )
    compact_character_prompts_text = ctx_compact.compact_episode_scoped_prompt_library(character_prompts_text, episode_id, limit=settings["draft_character_limit"], max_recent_blocks=2)
    compact_scene_prompts_text = ctx_compact.compact_episode_scoped_prompt_library(scene_prompts_text, episode_id, limit=settings["draft_scene_limit"], max_recent_blocks=2)
    prioritized_asset_catalog = prioritize_storyboard_prompt_asset_catalog(
        asset_catalog,
        director_json,
        profile=profile,
    )
    compact_asset_catalog = ctx_compact.compact_asset_catalog_for_storyboard(prioritized_asset_catalog, profile=profile)
    style = str(config.get("quality", {}).get("visual_style") or "").strip()
    medium = str(config.get("quality", {}).get("target_medium") or "").strip()
    frame_orientation = normalize_frame_orientation(config.get("quality", {}).get("frame_orientation"))
    frame_composition_guidance = build_frame_composition_guidance(frame_orientation)
    extra_rules = list(config.get("quality", {}).get("extra_rules", []))
    extra_rules_block = ""
    if extra_rules:
        extra_rules_block = "补充要求：\n" + render_bullets(extra_rules)
    story_points = list((director_json or {}).get("story_points") or [])
    storyboard_density_guidance = (
        f"分镜密度要求：本集导演剧情点共 {len(story_points)} 个，最终必须输出完全等量的 prompt_entries。"
        " 整体节奏需比旧版 storyboard 流程明显提快，目标约快 25%-35%。"
        " 每条 prompt_entry 默认优先 8-11 秒，高价值动作/对峙/翻盘/奇观兑现可放宽到 9-12 秒；前 20%-30% 时长内必须进入有效剧情。"
        " 静态说明、重复反应和弱过场要快速带过，把时长让给动作、关系推进、分镜调度、特效兑现和大场面逻辑。"
        "任何一条如果仍需要 12 秒以上才能完整表达，说明导演拆点还不够细，应按导演剧情点中的 detail_anchor_lines 和 micro_beats 把细节拆清，而不是省略。"
    )
    return {
        "series_name": series_name,
        "episode_id": episode_id,
        "visual_style": style or "按导演讲戏本与现有 assets 统一",
        "target_medium": medium or "漫剧",
        "frame_orientation": frame_orientation,
        "frame_composition_guidance": frame_composition_guidance,
        "extra_rules_block": extra_rules_block,
        "storyboard_density_guidance": storyboard_density_guidance,
        "storyboard_mode_guidance": storyboard_mode_guidance(profile),
        "storyboard_agent": ctx_compact.compact_reference_text(storyboard_agent, settings["draft_storyboard_agent_limit"]),
        "storyboard_skill": ctx_compact.compact_reference_text(storyboard_skill, settings["draft_storyboard_skill_limit"]),
        "methodology": ctx_compact.compact_reference_text(methodology, settings["draft_methodology_limit"]),
        "director_brief": compact_director_brief,
        "director_json": dump_prompt_json(compact_director_json),
        "genre_reference_bundle_json": dump_prompt_json(compact_genre_reference_bundle),
        "seedance_story_point_guidance_json": dump_prompt_json(compact_seedance_story_point_guidance),
        "character_prompts_text": compact_character_prompts_text or "<空>",
        "scene_prompts_text": compact_scene_prompts_text or "<空>",
        "asset_catalog_text": build_catalog_text(compact_asset_catalog),
    }


def build_episode_prompt(
    *,
    config: Mapping[str, Any],
    series_name: str,
    episode_id: str,
    director_markdown: str,
    director_json: Mapping[str, Any] | None,
    genre_reference_bundle: Mapping[str, Any],
    seedance_story_point_guidance: Mapping[str, Any],
    character_prompts_text: str,
    scene_prompts_text: str,
    asset_catalog: list[Mapping[str, str]],
) -> str:
    return render_prompt(
        "seedance_storyboard/draft_user.md",
        build_episode_prompt_context(
            config=config,
            series_name=series_name,
            episode_id=episode_id,
            director_markdown=director_markdown,
            director_json=director_json,
            genre_reference_bundle=genre_reference_bundle,
            seedance_story_point_guidance=seedance_story_point_guidance,
            character_prompts_text=character_prompts_text,
            scene_prompts_text=scene_prompts_text,
            asset_catalog=asset_catalog,
        ),
    )


def build_review_prompt_context(
    *,
    config: Mapping[str, Any],
    series_name: str,
    episode_id: str,
    director_json: Mapping[str, Any] | None,
    genre_reference_bundle: Mapping[str, Any],
    seedance_story_point_guidance: Mapping[str, Any],
    asset_catalog: list[Mapping[str, str]],
    draft_package: Mapping[str, Any],
    draft_defects: Sequence[str] | None = None,
) -> str:
    profile = resolve_storyboard_profile(config)
    settings = storyboard_profile_settings(profile)
    review_skill = load_skill("production/seedance-prompt-review-skill/SKILL.md")
    compliance_skill = load_skill("production/compliance-review-skill/SKILL.md")
    review_focus_point_ids = select_storyboard_review_focus_point_ids(draft_package, draft_defects, director_json)
    director_review_checklist = ctx_compact.compact_director_checklist_for_storyboard_review(
        director_json,
        profile=profile,
        focus_point_ids=review_focus_point_ids,
    )
    prioritized_asset_catalog = prioritize_storyboard_prompt_asset_catalog(
        asset_catalog,
        director_json,
        draft_package=draft_package,
        profile=profile,
    )
    compact_asset_catalog = ctx_compact.compact_asset_catalog_for_storyboard_review(
        prioritized_asset_catalog,
        profile=profile,
    )
    compact_draft_package = ctx_compact.compact_storyboard_draft_package_for_review(
        draft_package,
        profile=profile,
        focus_point_ids=review_focus_point_ids,
    )
    style = str(config.get("quality", {}).get("visual_style") or "").strip()
    medium = str(config.get("quality", {}).get("target_medium") or "").strip()
    frame_orientation = normalize_frame_orientation(config.get("quality", {}).get("frame_orientation"))
    frame_composition_guidance = build_frame_composition_guidance(frame_orientation)
    story_points = list((director_json or {}).get("story_points") or [])
    storyboard_density_guidance = (
        f"分镜密度要求：本集导演剧情点共 {len(story_points)} 个，最终必须输出完全等量的 prompt_entries。"
        " 整体节奏需比旧版 storyboard 流程明显提快，目标约快 25%-35%。"
        " 每条 prompt_entry 默认优先 8-11 秒，高价值动作/对峙/翻盘/奇观兑现可放宽到 9-12 秒；前 20%-30% 时长内必须进入有效剧情。"
        " 静态说明、重复反应和弱过场要快速带过，把时长让给动作、关系推进、分镜调度、特效兑现和大场面逻辑。"
        " 如果某条超过 12 秒，必须通过更细的镜头节拍和 detail anchor 保持细节，不得用概括性跳写。"
    )
    unique_defects: list[str] = []
    for defect in list(draft_defects or []):
        text = str(defect or "").strip()
        if text and text not in unique_defects:
            unique_defects.append(text)
        if len(unique_defects) >= 12:
            break
    draft_defect_report = ""
    if unique_defects:
        draft_defect_report = "【初稿已检测到以下问题，复审必须优先修复】\n" + "\n".join(
            f"- {defect}" for defect in unique_defects
        )
    review_focus_scope = ""
    total_entries = len(list((draft_package or {}).get("prompt_entries") or []))
    if review_focus_point_ids and len(review_focus_point_ids) < total_entries:
        review_focus_scope = (
            "【本轮送审范围】\n"
            f"- 仅提供 {len(review_focus_point_ids)} 条需要修补的分镜及其相邻条："
            + "、".join(review_focus_point_ids)
            + "\n- 未出现在当前 draft_package_json 的条目默认保持不变，不要重写整集。"
        )
    return {
        "series_name": series_name,
        "episode_id": episode_id,
        "visual_style": style or "按当前项目统一",
        "target_medium": medium or "漫剧",
        "frame_orientation": frame_orientation,
        "frame_composition_guidance": frame_composition_guidance,
        "storyboard_density_guidance": storyboard_density_guidance,
        "storyboard_mode_guidance": storyboard_mode_guidance(profile),
        "draft_defect_report": draft_defect_report,
        "review_focus_scope": review_focus_scope,
        "generalized_review_rules": render_generalized_logic_rules_prompt(),
        "director_review_checklist": director_review_checklist or "<空>",
        "asset_catalog_text": build_catalog_text(compact_asset_catalog),
        "draft_package_json": dump_prompt_json(compact_draft_package),
        "review_skill": ctx_compact.compact_reference_text(review_skill, settings["review_skill_limit"]),
        "compliance_skill": ctx_compact.compact_reference_text(compliance_skill, settings["review_compliance_limit"]),
    }


def build_review_prompt(
    *,
    config: Mapping[str, Any],
    series_name: str,
    episode_id: str,
    director_json: Mapping[str, Any] | None,
    genre_reference_bundle: Mapping[str, Any],
    seedance_story_point_guidance: Mapping[str, Any],
    asset_catalog: list[Mapping[str, str]],
    draft_package: Mapping[str, Any],
    draft_defects: Sequence[str] | None = None,
) -> str:
    return render_prompt(
        "seedance_storyboard/review_user.md",
        build_review_prompt_context(
            config=config,
            series_name=series_name,
            episode_id=episode_id,
            director_json=director_json,
            genre_reference_bundle=genre_reference_bundle,
            seedance_story_point_guidance=seedance_story_point_guidance,
            asset_catalog=asset_catalog,
            draft_package=draft_package,
            draft_defects=draft_defects,
        ),
    )


def ordered_storyboard_point_ids(
    director_json: Mapping[str, Any] | None,
    draft_package: Mapping[str, Any] | None = None,
) -> list[str]:
    ordered: list[str] = []
    for index, item in enumerate(list((director_json or {}).get("story_points") or []), start=1):
        point_id = normalize_storyboard_point_id(item.get("point_id"), fallback_index=index)
        if point_id and point_id not in ordered:
            ordered.append(point_id)
    for index, item in enumerate(list((draft_package or {}).get("prompt_entries") or []), start=1):
        if not isinstance(item, Mapping):
            continue
        point_id = normalize_storyboard_point_id(item.get("point_id"), fallback_index=index)
        if point_id and point_id not in ordered:
            ordered.append(point_id)
    return ordered


def select_storyboard_review_focus_point_ids(
    draft_package: Mapping[str, Any] | None,
    draft_defects: Sequence[str] | None,
    director_json: Mapping[str, Any] | None,
) -> list[str]:
    ordered_ids = ordered_storyboard_point_ids(director_json, draft_package)
    if not ordered_ids:
        return []
    unique_defects = [str(item or "").strip() for item in list(draft_defects or []) if str(item or "").strip()]
    if not unique_defects:
        return ordered_ids
    focus_indices: set[int] = set()
    for defect in unique_defects:
        for index, point_id in enumerate(ordered_ids):
            if point_id and point_id in defect:
                focus_indices.add(index)
    if not focus_indices:
        return ordered_ids
    expanded_indices: set[int] = set()
    for index in focus_indices:
        expanded_indices.add(index)
        if index - 1 >= 0:
            expanded_indices.add(index - 1)
        if index + 1 < len(ordered_ids):
            expanded_indices.add(index + 1)
    return [ordered_ids[index] for index in sorted(expanded_indices)]


def merge_storyboard_review_patch(
    draft_package: Mapping[str, Any],
    review_patch: Mapping[str, Any],
    *,
    director_json: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(draft_package or {}))
    materials_overview = str(review_patch.get("materials_overview") or "").strip()
    if materials_overview:
        merged["materials_overview"] = materials_overview
    global_notes = [str(item or "").strip() for item in list(review_patch.get("global_notes") or []) if str(item or "").strip()]
    if global_notes:
        merged["global_notes"] = global_notes

    # review 只负责修补，不允许删除已有分镜；delete_point_ids 一律忽略。
    delete_ids: set[str] = set()
    replacement_map: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(list(review_patch.get("replace_prompt_entries") or []), start=1):
        if not isinstance(item, Mapping):
            continue
        point_id = normalize_storyboard_point_id(item.get("point_id"), fallback_index=index)
        if not point_id:
            continue
        replacement_item = copy.deepcopy(dict(item))
        replacement_item["point_id"] = point_id
        replacement_item["title"] = strip_storyboard_title_prefix(replacement_item.get("title"), point_id)
        replacement_map[point_id] = replacement_item

    existing_entries = [dict(item) for item in list(merged.get("prompt_entries") or []) if isinstance(item, Mapping)]
    merged_entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(existing_entries, start=1):
        point_id = normalize_storyboard_point_id(item.get("point_id"), fallback_index=index)
        item["point_id"] = point_id
        item["title"] = strip_storyboard_title_prefix(item.get("title"), point_id)
        if point_id in delete_ids:
            continue
        if point_id in replacement_map:
            merged_entries.append(copy.deepcopy(replacement_map[point_id]))
            seen_ids.add(point_id)
            continue
        merged_entries.append(item)
        if point_id:
            seen_ids.add(point_id)

    for point_id, item in replacement_map.items():
        if point_id in seen_ids or point_id in delete_ids:
            continue
        merged_entries.append(copy.deepcopy(item))

    director_order = ordered_storyboard_point_ids(director_json, merged)
    if director_order:
        order_index = {point_id: index for index, point_id in enumerate(director_order)}
        merged_entries.sort(
            key=lambda item: order_index.get(normalize_storyboard_point_id(item.get("point_id")), len(order_index))
        )

    # 双保险：即使 review patch 异常，也不允许把 draft 里已有的 point 丢掉。
    base_entries = [dict(item) for item in list(draft_package.get("prompt_entries") or []) if isinstance(item, Mapping)]
    merged_ids = {
        normalize_storyboard_point_id(item.get("point_id"), fallback_index=index + 1)
        for index, item in enumerate(merged_entries)
        if isinstance(item, Mapping)
    }
    for index, item in enumerate(base_entries, start=1):
        point_id = normalize_storyboard_point_id(item.get("point_id"), fallback_index=index)
        if not point_id or point_id in merged_ids:
            continue
        preserved_item = copy.deepcopy(item)
        preserved_item["point_id"] = point_id
        preserved_item["title"] = strip_storyboard_title_prefix(preserved_item.get("title"), point_id)
        merged_entries.append(preserved_item)
        merged_ids.add(point_id)

    if director_order:
        order_index = {point_id: index for index, point_id in enumerate(director_order)}
        merged_entries.sort(
            key=lambda item: order_index.get(normalize_storyboard_point_id(item.get("point_id")), len(order_index))
        )

    merged["prompt_entries"] = merged_entries
    return merged


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def parse_duration_hint_seconds(duration_hint: str) -> float:
    values = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(duration_hint or ""))]
    if not values:
        return 10.0
    return max(values)


def normalize_duration_hint_text(raw: str) -> str:
    values = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(raw or ""))]
    if not values:
        return "8-11秒"
    start = values[0]
    end = values[1] if len(values) > 1 else values[0] + 2.0
    if start > end:
        start, end = end, start
    start = max(6.0, min(start, 11.0))
    end = min(12.5, end)
    if end <= start:
        end = min(12.5, start + 1.5)
    if abs(start - round(start)) < 1e-6 and abs(end - round(end)) < 1e-6:
        return f"{int(round(start))}-{int(round(end))}秒"
    return f"{start:.1f}-{end:.1f}秒"


def format_beat_time_window(start_second: float, end_second: float) -> str:
    start = max(0.0, round(start_second, 1))
    end = max(start + 0.3, round(end_second, 1))
    return f"{start:.1f}-{end:.1f}秒："


def ensure_timed_shot_beats(beats: list[str], duration_hint: str) -> list[str]:
    cleaned = [str(item).strip() for item in beats if str(item).strip()]
    if not cleaned:
        return []
    if all(BEAT_TIME_WINDOW_PATTERN.match(item) for item in cleaned):
        return cleaned

    total_duration = max(6.0, parse_duration_hint_seconds(duration_hint))
    count = len(cleaned)
    timed_beats: list[str] = []
    cursor = 0.0
    for index, beat in enumerate(cleaned):
        raw = beat.strip()
        if BEAT_TIME_WINDOW_PATTERN.match(raw):
            timed_beats.append(raw)
            continue
        remaining = count - index
        remaining_duration = max(0.6, total_duration - cursor)
        slot = max(0.8, remaining_duration / remaining)
        end = total_duration if index == count - 1 else min(total_duration, cursor + slot)
        timed_beats.append(f"{format_beat_time_window(cursor, end)}{raw}")
        cursor = end
    return timed_beats


def parse_timed_beat_line(beat: str) -> tuple[float, float, str] | None:
    match = re.match(
        r"^\s*(?P<start>\d+(?:\.\d+)?)\s*(?:-|–|—|~|至)\s*(?P<end>\d+(?:\.\d+)?)\s*(?:秒|s)\s*[:：]\s*(?P<body>.+?)\s*$",
        str(beat or ""),
    )
    if not match:
        return None
    start_second = _safe_float(match.group("start"), 0.0)
    end_second = _safe_float(match.group("end"), start_second + 0.8)
    if end_second <= start_second:
        end_second = start_second + 0.8
    return (round(start_second, 1), round(end_second, 1), match.group("body").strip())


def is_metadata_beat_text(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return True
    metadata_prefixes = (
        "承接上一条",
        "承接前一条",
        "承接上一拍",
        "承接前一拍",
        "尾帧停在",
        "结尾直接",
        "首尾镜承接",
    )
    if any(normalized.startswith(prefix) for prefix in metadata_prefixes):
        return True
    if normalized.startswith("承接") and "尾帧" in normalized:
        return True
    return False


def normalize_master_timeline(
    entries: list[Mapping[str, Any]] | None,
    duration_hint: str,
    *,
    max_entries: int = 10,
    gap_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    duration_seconds = parse_duration_hint_seconds(duration_hint)
    hard_end = max(1.2, duration_seconds - 0.2)
    cursor = 0.0
    normalized: list[dict[str, Any]] = []
    for raw_item in list(entries or [])[:max_entries]:
        visual_beat = str(
            raw_item.get("visual_beat")
            or raw_item.get("beat_text")
            or raw_item.get("visual")
            or raw_item.get("shot")
            or ""
        ).strip()
        if is_metadata_beat_text(visual_beat):
            continue
        start_second = max(_safe_float(raw_item.get("start_second"), cursor), cursor)
        end_second = _safe_float(raw_item.get("end_second"), start_second + 1.0)
        end_second = max(end_second, start_second + 0.6)
        if start_second >= hard_end:
            break
        if end_second > hard_end:
            end_second = hard_end
        dialogue_blocks = normalize_dialogue_blocks(
            raw_item.get("dialogue_blocks"),
            start_second,
            end_second,
            fallback_speaker=str(raw_item.get("speaker") or "").strip(),
            fallback_line=str(raw_item.get("spoken_line") or raw_item.get("line") or "").strip(),
            fallback_note=str(raw_item.get("delivery_note") or "").strip(),
        )
        compat_speaker, compat_line, compat_note = build_compat_dialogue_fields(dialogue_blocks)
        normalized.append(
            {
                "start_second": round(start_second, 1),
                "end_second": round(end_second, 1),
                "visual_beat": visual_beat,
                "speaker": compat_speaker,
                "spoken_line": compat_line,
                "delivery_note": compat_note,
                "dialogue_blocks": dialogue_blocks,
                "audio_cues": str(raw_item.get("audio_cues") or "").strip(),
                "transition_hook": str(raw_item.get("transition_hook") or raw_item.get("tail_hook") or "").strip(),
            }
        )
        cursor = min(hard_end, end_second + gap_seconds)
    return normalized


def normalize_dialogue_blocks(
    raw_dialogue_blocks: Any,
    beat_start: float,
    beat_end: float,
    *,
    fallback_speaker: str = "",
    fallback_line: str = "",
    fallback_note: str = "",
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_block in list(raw_dialogue_blocks or []):
        if not isinstance(raw_block, Mapping):
            continue
        line = str(raw_block.get("line") or raw_block.get("spoken_line") or "").strip()
        if not line:
            continue
        start_second = max(beat_start, _safe_float(raw_block.get("start_second"), beat_start))
        max_start = max(beat_start, beat_end - 0.2)
        start_second = min(start_second, max_start)
        end_second = _safe_float(raw_block.get("end_second"), start_second + 0.9)
        end_second = max(min(end_second, beat_end), start_second + 0.2)
        normalized.append(
            {
                "speaker": str(raw_block.get("speaker") or fallback_speaker or "角色").strip() or "角色",
                "line": line,
                "start_second": round(start_second, 1),
                "end_second": round(end_second, 1),
                "delivery_note": str(raw_block.get("delivery_note") or fallback_note).strip(),
            }
        )
    if not normalized and fallback_line:
        start_second = round(beat_start, 1)
        end_second = round(max(beat_start + 0.2, beat_end), 1)
        normalized.append(
            {
                "speaker": fallback_speaker or "角色",
                "line": fallback_line,
                "start_second": start_second,
                "end_second": end_second,
                "delivery_note": fallback_note,
            }
        )
    normalized.sort(key=lambda item: (_safe_float(item.get("start_second"), beat_start), _safe_float(item.get("end_second"), beat_end)))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float, float, str]] = set()
    for item in normalized:
        key = (
            str(item.get("speaker") or ""),
            str(item.get("line") or ""),
            round(_safe_float(item.get("start_second"), beat_start), 1),
            round(_safe_float(item.get("end_second"), beat_end), 1),
            str(item.get("delivery_note") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_compat_dialogue_fields(dialogue_blocks: list[Mapping[str, Any]]) -> tuple[str, str, str]:
    if len(dialogue_blocks) != 1:
        return ("", "", "")
    block = dialogue_blocks[0]
    return (
        str(block.get("speaker") or "").strip(),
        str(block.get("line") or "").strip(),
        str(block.get("delivery_note") or "").strip(),
    )


def extract_dialogue_blocks_from_entry(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    start_second = _safe_float(entry.get("start_second"), 0.0)
    end_second = _safe_float(entry.get("end_second"), start_second + 0.8)
    return normalize_dialogue_blocks(
        entry.get("dialogue_blocks"),
        start_second,
        end_second,
        fallback_speaker=str(entry.get("speaker") or "").strip(),
        fallback_line=str(entry.get("spoken_line") or "").strip(),
        fallback_note=str(entry.get("delivery_note") or "").strip(),
    )


def build_dialogue_timeline_from_master_timeline(entries: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in entries:
        for dialogue in extract_dialogue_blocks_from_entry(entry):
            start_second = round(_safe_float(dialogue.get("start_second"), 0.0), 1)
            end_second = round(_safe_float(dialogue.get("end_second"), start_second + 0.9), 1)
            if end_second <= start_second:
                end_second = round(start_second + 0.9, 1)
            result.append(
                {
                    "speaker": str(dialogue.get("speaker") or "角色").strip() or "角色",
                    "line": str(dialogue.get("line") or "").strip(),
                    "start_second": start_second,
                    "end_second": end_second,
                    "delivery_note": str(dialogue.get("delivery_note") or "").strip(),
                }
            )
    return result


def build_shot_beat_plan_from_master_timeline(entries: list[Mapping[str, Any]]) -> list[str]:
    beats: list[str] = []
    for entry in entries:
        visual_beat = str(entry.get("visual_beat") or "").strip()
        if not visual_beat:
            continue
        start_second = _safe_float(entry.get("start_second"), 0.0)
        end_second = _safe_float(entry.get("end_second"), start_second + 0.8)
        beats.append(f"{format_beat_time_window(start_second, end_second)}{visual_beat}")
    return beats


def build_audio_design_from_master_timeline(entries: list[Mapping[str, Any]]) -> str:
    cues = _unique_ordered_lines([str(entry.get("audio_cues") or "").strip() for entry in entries])
    if not cues:
        return ""
    return "；".join(cues[:4])


def _normalize_prompt_overlap_text(text: str) -> str:
    return re.sub(r"[\s\"'“”‘’：:；;，,。.!！？?、（）()【】\[\]《》<>]+", "", str(text or ""))


def _prompt_text_contains(haystack: str, needle: str) -> bool:
    normalized_haystack = _normalize_prompt_overlap_text(haystack)
    normalized_needle = _normalize_prompt_overlap_text(needle)
    if len(normalized_needle) < 2:
        return False
    return normalized_needle in normalized_haystack


def _is_generic_transition_hook(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return True
    generic_prefixes = (
        "切到开场建立",
        "切到对白推进",
        "切到信息落点",
        "切到反应承接",
        "切到尾帧收束",
        "切到下一拍",
    )
    if normalized in {"尾帧留悬", "尾帧钩到下一拍推进"}:
        return True
    return any(normalized.startswith(prefix) for prefix in generic_prefixes)


def _normalize_prompt_context_text(text: str) -> str:
    normalized = normalize_spaces(text)
    if not normalized:
        return ""
    normalized = re.sub(
        r"^(?:该beat|该 beat|该段|这一段|本段)(?:采用|为|实际呈现为)?",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip("：:，,。；; ")
    return normalized


def _looks_like_narration_speaker(speaker: str, line: str) -> bool:
    normalized_speaker = normalize_spaces(speaker)
    normalized_line = normalize_spaces(line)
    if any(marker.lower() in normalized_speaker.lower() for marker in NARRATION_SPEAKER_MARKERS):
        return True
    exposition_markers = ("我是", "我叫", "可在", "那天", "后来", "直到", "原来", "给自己", "却有一位")
    return any(marker in normalized_line for marker in exposition_markers) and all(token not in normalized_line for token in ("你", "吗", "？", "?"))


def _render_dialogue_clause_for_prompt(speaker: str, line: str, delivery_note: str) -> str:
    speaker_text = normalize_spaces(speaker) or "角色"
    spoken_line = normalize_spaces(line)
    delivery = f"（{delivery_note}）" if delivery_note else ""
    if _looks_like_narration_speaker(speaker_text, spoken_line):
        return f"旁白{delivery}念出“{spoken_line}”"
    return f"{speaker_text}{delivery}开口说“{spoken_line}”"


def build_prompt_text_from_master_timeline(
    entries: list[Mapping[str, Any]],
    *,
    title: str = "",
    continuity_bridge: str = "",
) -> str:
    segments: list[str] = []
    seen_dialogue_lines: set[str] = set()
    for entry in entries:
        visual_beat = str(entry.get("visual_beat") or "").strip()
        if not visual_beat:
            continue
        start_second = _safe_float(entry.get("start_second"), 0.0)
        end_second = _safe_float(entry.get("end_second"), start_second + 0.8)
        segment_parts = [f"{start_second:.1f}-{end_second:.1f}秒，{visual_beat}"]
        dialogue_clauses: list[str] = []
        for dialogue in extract_dialogue_blocks_from_entry(entry):
            speaker_prefix = str(dialogue.get("speaker") or "角色").strip() or "角色"
            delivery_note = str(dialogue.get("delivery_note") or "").strip()
            spoken_line = str(dialogue.get("line") or "").strip()
            line_marker = _normalize_prompt_overlap_text(spoken_line)
            if not spoken_line or line_marker in seen_dialogue_lines:
                continue
            if _prompt_text_contains(visual_beat, spoken_line):
                continue
            seen_dialogue_lines.add(line_marker)
            dialogue_clauses.append(_render_dialogue_clause_for_prompt(speaker_prefix, spoken_line, delivery_note))
        if dialogue_clauses:
            if len(dialogue_clauses) == 1:
                segment_parts.append(dialogue_clauses[0])
            else:
                segment_parts.append("对白依次落出：" + "；".join(dialogue_clauses[:2]))
        audio_cues = str(entry.get("audio_cues") or "").strip()
        if audio_cues and not _prompt_text_contains(visual_beat, audio_cues):
            segment_parts.append(f"同时听见{audio_cues}")
        transition_hook = str(entry.get("transition_hook") or "").strip()
        if transition_hook and not _is_generic_transition_hook(transition_hook) and not _prompt_text_contains(visual_beat, transition_hook):
            segment_parts.append(transition_hook)
        segments.append("；".join(part for part in segment_parts if part))
    return "；".join(segment.strip("；") for segment in segments if segment).strip("；")


def authored_prompt_text_is_usable(text: str) -> bool:
    cleaned = strip_prompt_text_meta_blocks(str(text or "").strip())
    if not cleaned:
        return False
    if len(cleaned) < 100:
        return False
    sentence_like_breaks = len(re.findall(r"[。；！？?!]", cleaned))
    if sentence_like_breaks >= 2:
        return True
    return len(cleaned) >= 180


def _extract_prompt_refs(text: str) -> set[str]:
    return {match.strip() for match in re.findall(r"@图片\d+", str(text or "")) if match.strip()}


def _normalize_coverage_line(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip())


def _collect_master_timeline_dialogue_lines(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    lines: list[str] = []
    for entry in entries:
        for dialogue in extract_dialogue_blocks_from_entry(entry):
            line = str(dialogue.get("line") or "").strip()
            normalized = _normalize_coverage_line(line)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            lines.append(line)
    return lines


def _collect_master_timeline_audio_cues(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    cues: list[str] = []
    for entry in entries:
        cue = str(entry.get("audio_cues") or "").strip()
        normalized = _normalize_coverage_line(cue)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cues.append(cue)
    return cues


def authored_prompt_text_should_yield_to_master_timeline(
    authored_text: str,
    synthesized_text: str,
    master_timeline: Sequence[Mapping[str, Any]],
) -> bool:
    authored_clean = strip_prompt_text_meta_blocks(str(authored_text or "").strip())
    synthesized_clean = strip_prompt_text_meta_blocks(str(synthesized_text or "").strip())
    if not synthesized_clean:
        return False
    if not authored_prompt_text_is_usable(authored_clean):
        return True

    authored_refs = _extract_prompt_refs(authored_clean)
    required_refs = _extract_prompt_refs(synthesized_clean)
    if required_refs and not required_refs.issubset(authored_refs):
        return True

    required_dialogue_lines = _collect_master_timeline_dialogue_lines(master_timeline)
    for line in required_dialogue_lines:
        normalized = _normalize_coverage_line(line)
        if normalized and normalized not in _normalize_coverage_line(authored_clean):
            return True

    audio_cues = _collect_master_timeline_audio_cues(master_timeline)
    authored_mentions_sound = any(token in authored_clean for token in ("听见", "声", "响", "嗡鸣", "风", "呼吸", "脚步", "衣料"))
    if audio_cues and not authored_mentions_sound:
        return True

    authored_len = len(authored_clean)
    synthesized_len = len(synthesized_clean)
    authored_sentence_like_breaks = len(re.findall(r"[。；！？?!]", authored_clean))
    beat_count = len(list(master_timeline or []))
    minimum_expected_length = max(220, beat_count * 90)

    if authored_len < minimum_expected_length:
        return True
    if synthesized_len >= 240 and authored_len < int(synthesized_len * 0.9):
        return True
    if beat_count >= 4 and authored_sentence_like_breaks < max(2, beat_count - 2):
        return True
    return False


def build_master_timeline_from_legacy(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    beats = [str(x).strip() for x in list(item.get("shot_beat_plan") or []) if str(x).strip()]
    timed_beats = ensure_timed_shot_beats(beats, str(item.get("duration_hint") or ""))
    dialogue_entries = [dict(x) for x in list(item.get("dialogue_timeline") or []) if isinstance(x, Mapping)]
    normalized_dialogues = normalize_dialogue_timeline(dialogue_entries, str(item.get("duration_hint") or ""))
    master_timeline: list[dict[str, Any]] = []
    used_dialogue_indexes: set[int] = set()
    for index, beat in enumerate(timed_beats):
        parsed = parse_timed_beat_line(beat)
        if not parsed:
            continue
        start_second, end_second, visual_beat = parsed
        if is_metadata_beat_text(visual_beat):
            continue
        dialogues_in_beat: list[dict[str, Any]] = []
        local_used_indexes: list[int] = []
        for dialogue_index, dialogue in enumerate(normalized_dialogues):
            if dialogue_index in used_dialogue_indexes:
                continue
            dialogue_start = _safe_float(dialogue.get("start_second"), start_second)
            dialogue_end = _safe_float(dialogue.get("end_second"), dialogue_start + 0.9)
            overlaps = dialogue_start < end_second + 0.05 and dialogue_end > start_second - 0.05
            if not overlaps:
                continue
            local_used_indexes.append(dialogue_index)
            dialogues_in_beat.append(
                {
                    "speaker": str(dialogue.get("speaker") or "角色").strip() or "角色",
                    "line": str(dialogue.get("line") or "").strip(),
                    "start_second": round(max(start_second, dialogue_start), 1),
                    "end_second": round(min(end_second, max(dialogue_end, dialogue_start + 0.2)), 1),
                    "delivery_note": str(dialogue.get("delivery_note") or "").strip(),
                }
            )
        audio_cues = ""
        if index == 0:
            audio_cues = str(item.get("audio_design") or "").strip()
        compat_speaker, compat_line, compat_note = build_compat_dialogue_fields(dialogues_in_beat)
        master_timeline.append(
            {
                "start_second": start_second,
                "end_second": end_second,
                "visual_beat": visual_beat,
                "speaker": compat_speaker,
                "spoken_line": compat_line,
                "delivery_note": compat_note,
                "dialogue_blocks": dialogues_in_beat,
                "audio_cues": audio_cues,
                "transition_hook": "",
            }
        )
        used_dialogue_indexes.update(local_used_indexes)
    unused_dialogues = [
        normalized_dialogues[index]
        for index in range(len(normalized_dialogues))
        if index not in used_dialogue_indexes
    ]
    for dialogue in unused_dialogues:
        if not master_timeline:
            break
        dialogue_start = _safe_float(dialogue.get("start_second"), 0.0)
        dialogue_end = _safe_float(dialogue.get("end_second"), dialogue_start + 0.9)
        dialogue_mid = (dialogue_start + dialogue_end) / 2
        target_entry = min(
            master_timeline,
            key=lambda entry: abs(((_safe_float(entry.get("start_second"), 0.0) + _safe_float(entry.get("end_second"), 0.0)) / 2) - dialogue_mid),
        )
        dialogue_blocks = list(target_entry.get("dialogue_blocks") or [])
        dialogue_blocks.append(
            {
                "speaker": str(dialogue.get("speaker") or "角色").strip() or "角色",
                "line": str(dialogue.get("line") or "").strip(),
                "start_second": round(max(_safe_float(target_entry.get("start_second"), 0.0), dialogue_start), 1),
                "end_second": round(min(_safe_float(target_entry.get("end_second"), dialogue_end), max(dialogue_end, dialogue_start + 0.2)), 1),
                "delivery_note": str(dialogue.get("delivery_note") or "").strip(),
            }
        )
        dialogue_blocks = normalize_dialogue_blocks(
            dialogue_blocks,
            _safe_float(target_entry.get("start_second"), 0.0),
            _safe_float(target_entry.get("end_second"), dialogue_end),
        )
        compat_speaker, compat_line, compat_note = build_compat_dialogue_fields(dialogue_blocks)
        target_entry["dialogue_blocks"] = dialogue_blocks
        target_entry["speaker"] = compat_speaker
        target_entry["spoken_line"] = compat_line
        target_entry["delivery_note"] = compat_note
    return master_timeline


def materialize_storyboard_item_from_master_timeline(item: dict[str, Any]) -> dict[str, Any]:
    master_timeline = [dict(entry) for entry in list(item.get("master_timeline") or []) if isinstance(entry, Mapping)]
    if not master_timeline:
        return item
    authored_prompt_text = strip_prompt_text_meta_blocks(str(item.get("prompt_text") or "").strip())
    for entry in master_timeline:
        dialogue_blocks = extract_dialogue_blocks_from_entry(entry)
        compat_speaker, compat_line, compat_note = build_compat_dialogue_fields(dialogue_blocks)
        entry["dialogue_blocks"] = dialogue_blocks
        entry["speaker"] = compat_speaker
        entry["spoken_line"] = compat_line
        entry["delivery_note"] = compat_note
    item["master_timeline"] = master_timeline
    item["shot_beat_plan"] = build_shot_beat_plan_from_master_timeline(master_timeline)
    item["dialogue_timeline"] = build_dialogue_timeline_from_master_timeline(master_timeline)
    if not str(item.get("audio_design") or "").strip():
        item["audio_design"] = build_audio_design_from_master_timeline(master_timeline)
    synthesized_prompt_text = build_prompt_text_from_master_timeline(
        master_timeline,
        title=str(item.get("title") or "").strip(),
        continuity_bridge=str(item.get("continuity_bridge") or "").strip(),
    )
    if authored_prompt_text_should_yield_to_master_timeline(authored_prompt_text, synthesized_prompt_text, master_timeline):
        item["prompt_text"] = synthesized_prompt_text
    else:
        item["prompt_text"] = authored_prompt_text
    return item


def render_master_timeline_markdown_lines(entries: list[Mapping[str, Any]]) -> list[str]:
    if not entries:
        return ["- 无"]
    lines: list[str] = []
    for entry in entries:
        start_second = round(_safe_float(entry.get("start_second"), 0.0), 1)
        end_second = round(_safe_float(entry.get("end_second"), start_second), 1)
        pieces = [f"{start_second:.1f}-{end_second:.1f}秒", f"画面：{str(entry.get('visual_beat') or '').strip()}"]
        dialogue_blocks = extract_dialogue_blocks_from_entry(entry)
        if dialogue_blocks:
            dialogue_texts: list[str] = []
            for dialogue in dialogue_blocks:
                dialogue_start = round(_safe_float(dialogue.get("start_second"), start_second), 1)
                dialogue_end = round(_safe_float(dialogue.get("end_second"), dialogue_start + 0.9), 1)
                speaker = str(dialogue.get("speaker") or "角色").strip() or "角色"
                delivery_note = str(dialogue.get("delivery_note") or "").strip()
                delivery = f"（{delivery_note}）" if delivery_note else ""
                spoken_line = str(dialogue.get("line") or "").strip()
                dialogue_texts.append(f"{dialogue_start:.1f}-{dialogue_end:.1f}秒 {speaker}{delivery}“{spoken_line}”")
            pieces.append(f"对白窗：{'；'.join(dialogue_texts)}")
        audio_cues = str(entry.get("audio_cues") or "").strip()
        if audio_cues:
            pieces.append(f"声音：{audio_cues}")
        transition_hook = str(entry.get("transition_hook") or "").strip()
        if transition_hook:
            pieces.append(f"交棒：{transition_hook}")
        lines.append("- " + "｜".join(piece for piece in pieces if piece))
    return lines


def normalize_pace_label(raw: str) -> str:
    value = str(raw or "").strip()
    if value in {"快压推进", "中速推进", "舒缓铺陈"}:
        return value
    if any(token in value for token in ("快", "急", "压", "高压")):
        return "快压推进"
    if any(token in value for token in ("慢", "缓", "铺陈", "静")):
        return "舒缓铺陈"
    return "中速推进"


def build_density_strategy(pace_label: str, beat_count: int) -> str:
    if pace_label == "快压推进":
        return f"保持高密度推进，约 {beat_count} 个镜头节拍内完成动作、对白、反应的连续推进；前20%时长内进入冲突，每2-3秒给出新信息，不写重复静态描述。"
    if pace_label == "舒缓铺陈":
        return f"保持舒缓但充实的推进，约 {beat_count} 个镜头节拍承载情绪、视线、环境和关系变化；前段尽快进入有效信息，不允许发空。"
    return f"保持中速推进，约 {beat_count} 个镜头节拍平衡信息交代、关系变化与情绪起伏；前20%时长内必须落下关键信息。"


def _first_nonempty_text(values: list[str]) -> str:
    for item in values:
        text = str(item or "").strip()
        if text:
            return text
    return ""


def simplify_continuity_bridge(text: str, *, max_length: int = 96, max_clauses: int = 2) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    normalized = re.sub(r"\s+", " ", raw).strip()
    sentence_parts = [
        part.strip(" ，；。")
        for part in re.split(r"[。！？\n]+", normalized)
        if part.strip(" ，；。")
    ]
    selected: list[str] = []
    for part in sentence_parts:
        clauses = [chunk.strip(" ，；。") for chunk in re.split(r"[；;]", part) if chunk.strip(" ，；。")]
        for clause in clauses:
            if clause and clause not in selected:
                selected.append(clause)
            if len(selected) >= max_clauses:
                break
        if len(selected) >= max_clauses:
            break
    if not selected and normalized:
        comma_clauses = [chunk.strip(" ，；。") for chunk in re.split(r"[，,]", normalized) if chunk.strip(" ，；。")]
        selected = comma_clauses[:max_clauses] or [normalized]
    simplified = "；".join(selected[:max_clauses]).strip(" ，；。")
    if not simplified:
        return ""
    if len(simplified) > max_length:
        simplified = simplified[:max_length].rstrip(" ，；。")
    return simplified + "。"


def build_continuity_handoff_note(
    item: Mapping[str, Any],
    next_item: Mapping[str, Any] | None,
) -> str:
    next_title = ""
    if isinstance(next_item, Mapping):
        next_title = str(next_item.get("title") or next_item.get("point_id") or "").strip()
    bridge = simplify_continuity_bridge(str(item.get("continuity_bridge") or "").strip())
    last_beat = _first_nonempty_text(list(item.get("shot_beat_plan") or [])[-1:])
    if last_beat and next_title:
        handoff = f"尾帧停在“{last_beat}”，直接接《{next_title}》。"
    elif last_beat:
        handoff = f"尾帧停在“{last_beat}”，给下一条留清楚落点。"
    elif next_title:
        handoff = f"结尾直接把情绪和站位交给《{next_title}》。"
    else:
        handoff = ""
    if bridge and handoff:
        return f"{bridge} {handoff}"
    return bridge or handoff


def normalize_dialogue_timeline(
    entries: list[Mapping[str, Any]] | None,
    duration_hint: str,
    *,
    max_entries: int = 5,
    gap_seconds: float = 0.25,
) -> list[dict[str, Any]]:
    duration_seconds = parse_duration_hint_seconds(duration_hint)
    hard_end = max(1.2, duration_seconds - 0.5)
    cursor = 0.5
    normalized: list[dict[str, Any]] = []
    for item in list(entries or [])[:max_entries]:
        line = str(item.get("line") or "").strip()
        if not line:
            continue
        speaker = str(item.get("speaker") or "角色").strip() or "角色"
        delivery_note = str(item.get("delivery_note") or "").strip()
        start_second = max(_safe_float(item.get("start_second"), cursor), cursor)
        min_span = 1.2 if len(line) > 16 else 0.9
        end_second = _safe_float(item.get("end_second"), start_second + min_span)
        end_second = max(end_second, start_second + min_span)
        if start_second >= hard_end:
            break
        if end_second > hard_end:
            end_second = hard_end
        if end_second <= start_second:
            continue
        normalized.append({
            "speaker": speaker,
            "line": line,
            "start_second": round(start_second, 1),
            "end_second": round(end_second, 1),
            "delivery_note": delivery_note,
        })
        cursor = min(hard_end, end_second + gap_seconds)
    return normalized


def build_dialogue_timeline_note(entries: list[Mapping[str, Any]]) -> str:
    if not entries:
        return ""
    parts = []
    for item in entries:
        delivery = f"（{item['delivery_note']}）" if str(item.get('delivery_note') or '').strip() else ""
        parts.append(
            f"{item['start_second']:.1f}-{item['end_second']:.1f}秒，{item['speaker']}{delivery}说\"{item['line']}\""
        )
    return "对白时间线：" + "；".join(parts) + "。所有对白必须串行出现，前一句完全结束后下一句再进入，不允许双声叠台词。"


def render_dialogue_timeline_markdown_lines(entries: list[Mapping[str, Any]]) -> list[str]:
    if not entries:
        return ["- 无"]
    lines: list[str] = []
    for item in entries:
        speaker = str(item.get("speaker") or "角色").strip() or "角色"
        delivery = str(item.get("delivery_note") or "").strip()
        delivery_suffix = f"（{delivery}）" if delivery else ""
        line = str(item.get("line") or "").strip()
        start_second = round(_safe_float(item.get("start_second"), 0.0), 1)
        end_second = round(_safe_float(item.get("end_second"), start_second), 1)
        lines.append(f"- {start_second:.1f}s - {end_second:.1f}s：{speaker}{delivery_suffix}“{line}”")
    return lines


def _unique_ordered_lines(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def empty_state_snapshot() -> dict[str, Any]:
    return {"scene": {}, "characters": {}}


def clone_state_snapshot(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    scene = dict((snapshot or {}).get("scene") or {})
    characters = {
        str(ref): dict(attrs or {})
        for ref, attrs in dict((snapshot or {}).get("characters") or {}).items()
        if str(ref).strip()
    }
    return {"scene": scene, "characters": characters}


def parse_state_update_line(raw_line: str) -> tuple[str, dict[str, str]] | None:
    parts = [part.strip() for part in str(raw_line or "").split("|") if part.strip()]
    if not parts:
        return None
    anchor = parts[0]
    attrs: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        attrs[key] = value
    if anchor == STATE_UPDATE_SCENE_PREFIX:
        filtered = {key: attrs[key] for key in STATE_UPDATE_SCENE_KEYS if attrs.get(key)}
        return (anchor, filtered) if filtered else None
    if re.fullmatch(r"@图片\d+", anchor):
        filtered = {key: attrs[key] for key in STATE_UPDATE_CHARACTER_KEYS if attrs.get(key)}
        return (anchor, filtered) if filtered else None
    return None


def render_state_update_line(anchor: str, attrs: Mapping[str, str]) -> str:
    key_order = STATE_UPDATE_SCENE_KEYS if anchor == STATE_UPDATE_SCENE_PREFIX else STATE_UPDATE_CHARACTER_KEYS
    parts = [anchor]
    for key in key_order:
        value = str(attrs.get(key) or "").strip()
        if value:
            parts.append(f"{key}={value}")
    return "|".join(parts)


def normalize_state_update_lines(value: Any, *, max_lines: int = STATE_MAX_UPDATE_LINES) -> list[str]:
    raw_lines = value if isinstance(value, list) else [value]
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_line in raw_lines:
        parsed = parse_state_update_line(str(raw_line or "").strip())
        if not parsed:
            continue
        line = render_state_update_line(*parsed)
        if line in seen:
            continue
        seen.add(line)
        normalized.append(line)
        if len(normalized) >= max_lines:
            break
    return normalized


def merge_state_snapshot(
    base: Mapping[str, Any] | None,
    overlay: Mapping[str, Any] | None,
    *,
    explicit_fields: Mapping[str, Any] | None = None,
    compare_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = clone_state_snapshot(base)
    overlay_scene = dict((overlay or {}).get("scene") or {})
    compare_scene = dict((compare_snapshot or {}).get("scene") or {})
    explicit_scene_fields = set(dict(explicit_fields or {}).get("scene") or set())
    for key in STATE_UPDATE_SCENE_KEYS:
        value = str(overlay_scene.get(key) or "").strip()
        if not value:
            continue
        if key in explicit_scene_fields:
            continue
        if compare_scene and value == str(compare_scene.get(key) or "").strip():
            continue
        merged["scene"][key] = value

    overlay_characters = dict((overlay or {}).get("characters") or {})
    compare_characters = dict((compare_snapshot or {}).get("characters") or {})
    explicit_characters = dict(dict(explicit_fields or {}).get("characters") or {})
    for ref, attrs in overlay_characters.items():
        ref_text = str(ref).strip()
        if not ref_text:
            continue
        target = dict(merged["characters"].get(ref_text) or {})
        compare_attrs = dict(compare_characters.get(ref_text) or {})
        explicit_attr_fields = set(explicit_characters.get(ref_text) or set())
        for key in STATE_UPDATE_CHARACTER_KEYS:
            value = str(dict(attrs or {}).get(key) or "").strip()
            if not value:
                continue
            if key in explicit_attr_fields:
                continue
            if compare_attrs and value == str(compare_attrs.get(key) or "").strip():
                continue
            target[key] = value
        if target:
            merged["characters"][ref_text] = target
    return merged


def apply_state_update_lines(
    snapshot: Mapping[str, Any] | None,
    lines: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = clone_state_snapshot(snapshot)
    explicit_fields: dict[str, Any] = {"scene": set(), "characters": {}}
    for line in lines:
        parsed = parse_state_update_line(line)
        if not parsed:
            continue
        anchor, attrs = parsed
        if anchor == STATE_UPDATE_SCENE_PREFIX:
            for key, value in attrs.items():
                updated["scene"][key] = value
                explicit_fields["scene"].add(key)
            continue
        target = dict(updated["characters"].get(anchor) or {})
        explicit_fields["characters"].setdefault(anchor, set())
        for key, value in attrs.items():
            target[key] = value
            explicit_fields["characters"][anchor].add(key)
        updated["characters"][anchor] = target
    return updated, explicit_fields


def build_asset_ref_sets(
    asset_catalog: Sequence[Mapping[str, Any]] | None,
) -> tuple[set[str], set[str]]:
    character_ref_ids: set[str] = set()
    scene_ref_ids: set[str] = set()
    for item in list(asset_catalog or []):
        ref_id = str(item.get("ref_id") or "").strip()
        if not ref_id:
            continue
        asset_type = str(item.get("asset_type") or "").strip()
        if asset_type == "人物参考":
            character_ref_ids.add(ref_id)
        elif asset_type == "场景参考":
            scene_ref_ids.add(ref_id)
    return character_ref_ids, scene_ref_ids


def split_state_clauses(text: str) -> list[str]:
    clauses: list[str] = []
    for raw in re.split(r"[。；\n]+", str(text or "")):
        clause = normalize_spaces(raw)
        if clause:
            clauses.append(clause)
    return clauses


def has_locative_cue(text: str) -> bool:
    normalized = str(text or "")
    return any(cue in normalized for cue in STATE_LOCATIVE_CUE_PATTERNS)


def infer_last_regex_value(text: str, patterns: Sequence[tuple[re.Pattern[str], str]]) -> str:
    normalized = str(text or "")
    best_index = -1
    best_value = ""
    for pattern, value in patterns:
        for match in pattern.finditer(normalized):
            if match.start() >= best_index:
                best_index = match.start()
                best_value = value
    return best_value


def infer_zone_from_rules(
    text: str,
    rules: Sequence[tuple[tuple[str, ...], str, bool]],
) -> str:
    normalized = str(text or "")
    if not normalized:
        return ""
    best_index = -1
    best_value = ""
    for keywords, value, require_cue in rules:
        for keyword in keywords:
            position = normalized.rfind(keyword)
            if position < 0:
                continue
            local_context = normalized[max(0, position - 8): min(len(normalized), position + len(keyword) + 8)]
            if require_cue and not has_locative_cue(local_context):
                continue
            if position >= best_index:
                best_index = position
                best_value = value
    return best_value


def coarse_scene_loc_from_zone(zone: str) -> str:
    normalized = str(zone or "").strip()
    if not normalized:
        return ""
    if normalized.startswith("龙辇"):
        return "龙辇"
    if normalized.startswith("高台"):
        return "高台"
    if normalized.startswith("赛场"):
        return "赛场"
    if normalized.startswith("传送镜"):
        return "传送镜"
    return normalized


def build_ref_match_pattern(ref: str) -> re.Pattern[str]:
    return re.compile(rf"{re.escape(str(ref or ''))}(?!\d)")


def extract_ref_local_snippets(ref: str, text: str, *, max_snippets: int = 4) -> list[str]:
    snippets: list[str] = []
    ref_pattern = build_ref_match_pattern(ref)
    for clause in split_state_clauses(text):
        if ref_pattern.search(clause):
            snippets.append(clause)
        if len(snippets) >= max_snippets:
            return snippets
    normalized = str(text or "")
    if snippets or not normalized:
        return snippets
    for match in ref_pattern.finditer(normalized):
        start = max(0, match.start() - 18)
        end = min(len(normalized), match.end() + 36)
        snippet = normalize_spaces(normalized[start:end])
        if snippet:
            snippets.append(snippet)
        if len(snippets) >= max_snippets:
            break
    return snippets


def extract_ref_local_windows(
    ref: str,
    text: str,
    *,
    before: int = 10,
    after: int = 42,
    max_windows: int = 4,
) -> list[str]:
    windows: list[str] = []
    normalized = str(text or "")
    if not normalized:
        return windows
    ref_pattern = build_ref_match_pattern(ref)
    for match in ref_pattern.finditer(normalized):
        start = max(0, match.start() - before)
        end = min(len(normalized), match.end() + after)
        snippet = normalize_spaces(normalized[start:end])
        if snippet:
            windows.append(snippet)
        if len(windows) >= max_windows:
            break
    return windows


def infer_character_zone_value(window_text: str, full_text: str, *, pose: str = "") -> str:
    explicit = infer_zone_from_rules(window_text, STATE_CHARACTER_ZONE_RULES)
    if explicit:
        return explicit
    normalized = str(full_text or "")
    if any(flag in normalized for flag in ("龙辇前左三步", "龙辇前右三步", "龙辇前三步", "龙辇前下方", "龙辇前侧", "龙辇前缘")):
        return "龙辇前缘"
    if "龙辇" in normalized and (pose == "坐" or any(flag in normalized for flag in ("龙辇上", "龙辇中央", "龙辇中轴", "扶手", "扶炉"))):
        return "龙辇中轴"
    if any(flag in normalized for flag in ("高台前缘", "高台栏位", "高处栏位")):
        return "高台前缘"
    if any(flag in normalized for flag in ("高台中央", "高台上方中轴")):
        return "高台"
    if any(flag in normalized for flag in ("中央赛场", "赛场中央", "赛场边缘", "赛场另一侧", "中央空地")):
        return "赛场"
    if "传送镜门位" in normalized:
        return "传送镜门位"
    if "传送镜前" in normalized or ("传送镜" in normalized and "镜门" in normalized):
        return "传送镜前"
    if "入场道" in normalized and "龙辇" not in normalized:
        return "入场道"
    return ""


def clean_face_target(raw: str) -> str:
    target = normalize_spaces(str(raw or ""))
    target = re.sub(r"(方向|那边|一侧|一端)$", "", target).strip()
    if not target:
        return ""
    if len(target) > 6:
        return ""
    disallowed = ("侧脸", "低声", "镜头", "动作", "视线", "一句", "半步", "半拍", "一瞬", "开", "@", "同时", "回到", "转头")
    if any(flag in target for flag in disallowed):
        return ""
    return target


def infer_face_direction(text: str) -> str:
    best_index = -1
    best_target = ""
    patterns = (
        r"(?:看向|望向|盯向|盯着|看着|望着|钉在|看回)([^，。；、]{1,6})",
        r"(?:朝|回身朝)([^，。；、]{1,6})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, str(text or "")):
            target = clean_face_target(match.group(1))
            if not target:
                continue
            if match.start() >= best_index:
                best_index = match.start()
                best_target = target
    return f"朝{best_target}" if best_target else ""


def infer_state_value_from_patterns(text: str, patterns: Sequence[tuple[str, str]]) -> str:
    normalized = str(text or "")
    best_index = -1
    best_value = ""
    for needle, value in patterns:
        position = normalized.rfind(needle)
        if position >= best_index and position >= 0:
            best_index = position
            best_value = value
    return best_value


def infer_character_state_from_text(ref: str, text: str) -> dict[str, str]:
    snippets = extract_ref_local_snippets(ref, text)
    if not snippets:
        return {}
    snippet_text = " ".join(snippets)
    window_text = " ".join(extract_ref_local_windows(ref, text))
    pose = infer_last_regex_value(snippet_text, STATE_POSE_REGEX_PATTERNS)
    inferred = {
        "zone": infer_character_zone_value(window_text, snippet_text, pose=pose),
        "pose": pose,
        "face": infer_face_direction(snippet_text),
        "hold": infer_last_regex_value(snippet_text, STATE_HOLD_REGEX_PATTERNS),
    }
    return {key: value for key, value in inferred.items() if value}


def infer_scene_state_from_text(text: str, *, anchor_zone: str = "") -> dict[str, str]:
    normalized = str(text or "")
    scene_loc = infer_zone_from_rules(normalized, STATE_SCENE_LOC_RULES)
    if not scene_loc and anchor_zone:
        scene_loc = coarse_scene_loc_from_zone(anchor_zone)
    inferred = {
        "loc": scene_loc,
        "cam": infer_state_value_from_patterns(normalized, STATE_CAMERA_PATTERNS),
        "light": infer_state_value_from_patterns(normalized, STATE_LIGHT_PATTERNS),
    }
    return {key: value for key, value in inferred.items() if value}


def build_state_update_fallback_lines(
    opening_snapshot: Mapping[str, Any] | None,
    closing_snapshot: Mapping[str, Any] | None,
    *,
    max_lines: int = STATE_MAX_UPDATE_LINES,
) -> list[str]:
    lines: list[str] = []
    opening_scene = dict((opening_snapshot or {}).get("scene") or {})
    closing_scene = dict((closing_snapshot or {}).get("scene") or {})
    scene_delta = {
        key: str(closing_scene.get(key) or "").strip()
        for key in STATE_UPDATE_SCENE_KEYS
        if str(closing_scene.get(key) or "").strip()
        and str(closing_scene.get(key) or "").strip() != str(opening_scene.get(key) or "").strip()
    }
    if scene_delta:
        lines.append(render_state_update_line(STATE_UPDATE_SCENE_PREFIX, scene_delta))

    opening_characters = dict((opening_snapshot or {}).get("characters") or {})
    closing_characters = dict((closing_snapshot or {}).get("characters") or {})
    for ref, attrs in sorted(
        closing_characters.items(),
        key=lambda item: int(re.search(r"(\d+)", str(item[0])).group(1)) if re.search(r"(\d+)", str(item[0])) else 999,
    ):
        if len(lines) >= max_lines:
            break
        opening_attrs = dict(opening_characters.get(ref) or {})
        delta = {
            key: str(dict(attrs or {}).get(key) or "").strip()
            for key in STATE_UPDATE_CHARACTER_KEYS
            if str(dict(attrs or {}).get(key) or "").strip()
            and str(dict(attrs or {}).get(key) or "").strip() != str(opening_attrs.get(key) or "").strip()
        }
        if not delta:
            continue
        if not any(key in delta for key in ("zone", "pose", "hold")):
            continue
        lines.append(render_state_update_line(ref, delta))
    return lines[:max_lines]


def infer_state_snapshot_from_text(text: str, refs: Sequence[str], *, anchor_ref: str = "") -> dict[str, Any]:
    snapshot = empty_state_snapshot()
    for ref in refs:
        ref_text = str(ref).strip()
        if not ref_text:
            continue
        inferred = infer_character_state_from_text(ref_text, text)
        if inferred:
            snapshot["characters"][ref_text] = inferred
    anchor_zone = str(snapshot.get("characters", {}).get(anchor_ref, {}).get("zone") or "").strip()
    if not anchor_zone:
        for attrs in snapshot.get("characters", {}).values():
            anchor_zone = str(dict(attrs or {}).get("zone") or "").strip()
            if anchor_zone:
                break
    scene_state = infer_scene_state_from_text(text, anchor_zone=anchor_zone)
    if scene_state:
        snapshot["scene"] = scene_state
    return snapshot


def build_inferred_state_snapshots(
    item: Mapping[str, Any],
    *,
    character_ref_ids: Collection[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    all_refs = _unique_ordered_lines(
        [str(ref).strip() for ref in list(item.get("primary_refs") or []) + list(item.get("secondary_refs") or []) if str(ref).strip()]
    )
    refs = [ref for ref in all_refs if not character_ref_ids or ref in character_ref_ids]
    master_timeline = [dict(entry) for entry in list(item.get("master_timeline") or []) if isinstance(entry, Mapping)]
    continuity_bridge = str(item.get("continuity_bridge") or "").strip()
    anchor_ref = next(
        (
            ref
            for ref in [str(ref).strip() for ref in list(item.get("primary_refs") or []) if str(ref).strip()]
            if not character_ref_ids or ref in character_ref_ids
        ),
        refs[0] if refs else "",
    )
    first_text = " ".join(
        part
        for part in [
            continuity_bridge,
            str(master_timeline[0].get("visual_beat") or "").strip() if master_timeline else "",
            str(master_timeline[0].get("transition_hook") or "").strip() if master_timeline else "",
        ]
        if part
    )
    last_text = " ".join(
        part
        for part in [
            str(master_timeline[-2].get("visual_beat") or "").strip() if len(master_timeline) >= 2 else "",
            str(master_timeline[-1].get("visual_beat") or "").strip() if master_timeline else "",
            str(master_timeline[-1].get("transition_hook") or "").strip() if master_timeline else "",
        ]
        if part
    )
    if not first_text or not last_text:
        prompt_text = str(item.get("prompt_text") or "").strip()
        if not first_text:
            first_text = prompt_text[:180]
        if not last_text:
            last_text = prompt_text[-180:]
    return (
        infer_state_snapshot_from_text(first_text, refs, anchor_ref=anchor_ref),
        infer_state_snapshot_from_text(last_text, refs, anchor_ref=anchor_ref),
    )


def format_state_snapshot_lines(snapshot: Mapping[str, Any], *, max_lines: int = STATE_MAX_SNAPSHOT_LINES) -> list[str]:
    lines: list[str] = []
    scene = dict(snapshot.get("scene") or {})
    if any(str(scene.get(key) or "").strip() for key in STATE_UPDATE_SCENE_KEYS):
        lines.append(render_state_update_line(STATE_UPDATE_SCENE_PREFIX, scene))
    character_items = sorted(
        dict(snapshot.get("characters") or {}).items(),
        key=lambda item: int(re.search(r"(\d+)", str(item[0])).group(1)) if re.search(r"(\d+)", str(item[0])) else 999,
    )
    for ref, attrs in character_items:
        if len(lines) >= max_lines:
            break
        rendered = render_state_update_line(str(ref), dict(attrs or {}))
        if rendered:
            lines.append(rendered)
    return lines[:max_lines]


def parse_state_snapshot_lines(lines: Sequence[str]) -> dict[str, Any]:
    snapshot = empty_state_snapshot()
    updated, _ = apply_state_update_lines(snapshot, list(lines))
    return updated


def attach_resolved_state_snapshots(
    prompt_entries: Sequence[Mapping[str, Any]],
    *,
    character_ref_ids: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    resolved_entries: list[dict[str, Any]] = []
    previous_closing = empty_state_snapshot()
    for raw_item in prompt_entries:
        item = dict(raw_item)
        state_update_lines = normalize_state_update_lines(item.get("state_update"))
        opening_inferred, closing_inferred = build_inferred_state_snapshots(
            item,
            character_ref_ids=character_ref_ids,
        )

        resolved_opening = merge_state_snapshot(previous_closing, opening_inferred)
        item["state_update"] = state_update_lines
        resolved_closing, explicit_fields = apply_state_update_lines(resolved_opening, state_update_lines)
        resolved_closing = merge_state_snapshot(
            resolved_closing,
            closing_inferred,
            explicit_fields=explicit_fields,
            compare_snapshot=resolved_opening,
        )

        item["resolved_opening_state"] = format_state_snapshot_lines(resolved_opening)
        item["resolved_closing_state"] = format_state_snapshot_lines(resolved_closing)
        resolved_entries.append(item)
        previous_closing = resolved_closing
    return resolved_entries


def has_state_transition_cue(text: str, patterns: Sequence[str]) -> bool:
    normalized = str(text or "")
    return any(pattern in normalized for pattern in patterns)


def summarize_character_state(ref: str, attrs: Mapping[str, str]) -> str:
    zone = str(attrs.get("zone") or "").strip()
    pose = str(attrs.get("pose") or "").strip()
    pieces = [piece for piece in [zone, pose] if piece]
    return f"{ref}({ '/'.join(pieces) })" if pieces else ref


def state_attrs_changed(before: Mapping[str, str], after: Mapping[str, str], *, keys: Sequence[str]) -> bool:
    for key in keys:
        before_value = str(before.get(key) or "").strip()
        after_value = str(after.get(key) or "").strip()
        if before_value and after_value and before_value != after_value:
            return True
    return False


def validate_state_continuity(final_result: Mapping[str, Any], *, episode_id: str) -> list[str]:
    warnings: list[str] = []
    previous_closing = empty_state_snapshot()
    previous_has_state_update = False
    prompt_entries = [dict(item) for item in list(final_result.get("prompt_entries") or []) if isinstance(item, Mapping)]
    for item in prompt_entries:
        point_id = str(item.get("point_id") or "未知分镜").strip() or "未知分镜"
        current_state_update = normalize_state_update_lines(item.get("state_update"))
        current_has_state_update = bool(current_state_update)
        opening = parse_state_snapshot_lines(list(item.get("resolved_opening_state") or []))
        closing = parse_state_snapshot_lines(list(item.get("resolved_closing_state") or []))
        master_timeline = [dict(entry) for entry in list(item.get("master_timeline") or []) if isinstance(entry, Mapping)]
        opening_text = " ".join(
            part for part in [
                str(item.get("continuity_bridge") or "").strip(),
                str(master_timeline[0].get("visual_beat") or "").strip() if master_timeline else "",
            ] if part
        )
        full_text = " ".join(
            [str(item.get("continuity_bridge") or "").strip(), str(item.get("prompt_text") or "").strip()]
            + [str(entry.get("visual_beat") or "").strip() for entry in master_timeline]
        ).strip()

        previous_scene_loc = coarse_scene_loc_from_zone(str(previous_closing.get("scene", {}).get("loc") or "").strip())
        opening_scene_loc = coarse_scene_loc_from_zone(str(opening.get("scene", {}).get("loc") or "").strip())
        if previous_has_state_update or current_has_state_update:
            if previous_scene_loc and opening_scene_loc and previous_scene_loc != opening_scene_loc:
                if not has_state_transition_cue(opening_text, STATE_BRIDGE_CUE_PATTERNS):
                    warnings.append(
                        f"{episode_id} 的 {point_id} 开场空间跳变：上一条在“{previous_scene_loc}”，本条首帧到了“{opening_scene_loc}”，但缺少明确切镜/带视线/空间打开的桥接。"
                    )

        shared_refs = sorted(set(previous_closing.get("characters", {}).keys()) & set(opening.get("characters", {}).keys()))
        if previous_has_state_update or current_has_state_update:
            for ref in shared_refs:
                prev_attrs = dict(previous_closing.get("characters", {}).get(ref) or {})
                open_attrs = dict(opening.get("characters", {}).get(ref) or {})
                changed = state_attrs_changed(prev_attrs, open_attrs, keys=("zone", "pose"))
                if changed and not has_state_transition_cue(opening_text, STATE_BRIDGE_CUE_PATTERNS + STATE_TRANSITION_CUE_PATTERNS):
                    warnings.append(
                        f"{episode_id} 的 {point_id} 首帧人物跳位：{summarize_character_state(ref, prev_attrs)} -> {summarize_character_state(ref, open_attrs)}，但缺少可见承接。"
                    )

        shared_internal_refs = sorted(set(opening.get("characters", {}).keys()) & set(closing.get("characters", {}).keys()))
        if current_has_state_update:
            for ref in shared_internal_refs:
                opening_attrs = dict(opening.get("characters", {}).get(ref) or {})
                closing_attrs = dict(closing.get("characters", {}).get(ref) or {})
                changed = state_attrs_changed(opening_attrs, closing_attrs, keys=("zone", "pose"))
                if changed and not has_state_transition_cue(full_text, STATE_TRANSITION_CUE_PATTERNS):
                    warnings.append(
                        f"{episode_id} 的 {point_id} 条内状态变化未写清：{summarize_character_state(ref, opening_attrs)} -> {summarize_character_state(ref, closing_attrs)}，建议补起身/移动/落位过程。"
                    )

        previous_closing = closing
        previous_has_state_update = current_has_state_update
    return warnings


def repair_storyboard_density(data: Mapping[str, Any], *, max_shot_beats: int = 10) -> dict[str, Any]:
    normalized = dict(data)
    repaired_entries: list[dict[str, Any]] = []
    raw_entries = list(data.get("prompt_entries") or [])
    for index, raw_item in enumerate(raw_entries):
        item = dict(raw_item)
        has_master_timeline = bool(list(item.get("master_timeline") or []))
        pace_label = normalize_pace_label(item.get("pace_label"))
        minimum_beats = 6 if pace_label == "快压推进" else 5 if pace_label == "中速推进" else 4
        beats = _unique_ordered_lines(list(item.get("shot_beat_plan") or []))[:max_shot_beats]
        continuity_bridge = str(item.get("continuity_bridge") or "").strip()
        audio_design = str(item.get("audio_design") or "").strip()
        dialogue_entries = list(item.get("dialogue_timeline") or [])
        dialogue_lines: list[str] = []
        for dialogue in dialogue_entries:
            speaker = str(dialogue.get("speaker") or "").strip()
            line = str(dialogue.get("line") or "").strip()
            if not line:
                continue
            if speaker:
                dialogue_lines.append(f"{speaker}说出关键台词：{line}")
            else:
                dialogue_lines.append(f"关键台词落下：{line}")
        next_item = raw_entries[index + 1] if index + 1 < len(raw_entries) else None
        next_title = ""
        if isinstance(next_item, Mapping):
            next_title = str(next_item.get("title") or next_item.get("point_id") or "").strip()
        supplement_pool = _unique_ordered_lines(
            [
                continuity_bridge,
                *dialogue_lines,
                audio_design,
                f"用补充反应镜头或空间落点承接{item.get('title') or item.get('point_id') or '当前分镜'}的情绪推进。",
                (
                    f"尾帧稳住角色姿态、视线与空间关系，自然交棒给下一条《{next_title}》。"
                    if next_title
                    else "尾帧稳住当前情绪与空间落点，为下一条留下明确承接点。"
                ),
            ]
        )
        if not has_master_timeline:
            for candidate in supplement_pool:
                if len(beats) >= minimum_beats:
                    break
                if candidate and candidate not in beats:
                    beats.append(candidate)

        if len(beats) < minimum_beats:
            if pace_label == "快压推进" and len(beats) >= 4:
                pace_label = "中速推进"
            elif pace_label == "中速推进" and len(beats) >= 3:
                pace_label = "舒缓铺陈"

        item["pace_label"] = pace_label
        if has_master_timeline:
            item = materialize_storyboard_item_from_master_timeline(item)
        else:
            item["shot_beat_plan"] = ensure_timed_shot_beats(beats[:max_shot_beats], str(item.get("duration_hint") or ""))
        repaired_entries.append(item)
    normalized["prompt_entries"] = repaired_entries
    return normalized


def sanitize_sensitive_text(text: str) -> tuple[str, list[str]]:
    sanitized = str(text or "")
    risk_notes: list[str] = []
    for pattern, note in SENSITIVE_OUTPUT_PATTERNS:
        if pattern.search(sanitized):
            risk_notes.append(note)
    for pattern, replacement in SENSITIVE_TEXT_REPLACEMENTS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized, risk_notes


def strip_prompt_text_meta_blocks(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    for block in PROMPT_TEXT_META_BLOCKS_TO_REMOVE:
        cleaned = cleaned.replace(f"\n\n{block}", "")
        cleaned = cleaned.replace(f"\n{block}", "")
        cleaned = cleaned.replace(block, "")
    filtered_lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if line in PROMPT_TEXT_EXACT_LINES_TO_REMOVE:
            continue
        if any(line.startswith(prefix) for prefix in PROMPT_TEXT_PREFIX_LINES_TO_REMOVE):
            continue
        filtered_lines.append(raw_line)
    cleaned = "\n".join(filtered_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def sanitize_storyboard_item(item: dict[str, Any]) -> dict[str, Any]:
    prompt_text, prompt_risks = sanitize_sensitive_text(str(item.get("prompt_text") or "").strip())
    audio_design, audio_risks = sanitize_sensitive_text(str(item.get("audio_design") or "").strip())
    item["prompt_text"] = strip_prompt_text_meta_blocks(prompt_text)
    item["audio_design"] = audio_design

    sanitized_master_timeline: list[dict[str, Any]] = []
    master_timeline_risks: list[str] = []
    for entry in list(item.get("master_timeline") or []):
        if not isinstance(entry, Mapping):
            continue
        entry_dict = dict(entry)
        clean_visual, visual_risks = sanitize_sensitive_text(str(entry_dict.get("visual_beat") or "").strip())
        clean_audio, audio_entry_risks = sanitize_sensitive_text(str(entry_dict.get("audio_cues") or "").strip())
        clean_hook, hook_risks = sanitize_sensitive_text(str(entry_dict.get("transition_hook") or "").strip())
        clean_line, line_risks = sanitize_sensitive_text(str(entry_dict.get("spoken_line") or "").strip())
        sanitized_dialogue_blocks: list[dict[str, Any]] = []
        dialogue_block_risks: list[str] = []
        for raw_block in list(entry_dict.get("dialogue_blocks") or []):
            if not isinstance(raw_block, Mapping):
                continue
            block_dict = dict(raw_block)
            clean_block_line, block_line_risks = sanitize_sensitive_text(str(block_dict.get("line") or "").strip())
            block_dict["line"] = clean_block_line
            sanitized_dialogue_blocks.append(block_dict)
            dialogue_block_risks.extend(block_line_risks)
        entry_dict["visual_beat"] = clean_visual
        entry_dict["audio_cues"] = clean_audio
        entry_dict["transition_hook"] = clean_hook
        entry_dict["spoken_line"] = clean_line
        if sanitized_dialogue_blocks:
            entry_dict["dialogue_blocks"] = sanitized_dialogue_blocks
        sanitized_master_timeline.append(entry_dict)
        master_timeline_risks.extend(visual_risks + audio_entry_risks + hook_risks + line_risks + dialogue_block_risks)
    item["master_timeline"] = sanitized_master_timeline

    sanitized_dialogues: list[dict[str, Any]] = []
    dialogue_risks: list[str] = []
    for dialogue in list(item.get("dialogue_timeline") or []):
        dialogue_dict = dict(dialogue)
        clean_line, line_risks = sanitize_sensitive_text(str(dialogue_dict.get("line") or "").strip())
        dialogue_dict["line"] = clean_line
        sanitized_dialogues.append(dialogue_dict)
        dialogue_risks.extend(line_risks)
    item["dialogue_timeline"] = sanitized_dialogues

    existing_risks = [str(x).strip() for x in list(item.get("risk_notes") or []) if str(x).strip()]
    merged_risks: list[str] = []
    for risk in existing_risks + prompt_risks + audio_risks + dialogue_risks + master_timeline_risks:
        if risk and risk not in merged_risks:
            merged_risks.append(risk)
    if merged_risks:
        item["risk_notes"] = merged_risks[:6]
    return item


def normalize_storyboard_result(
    data: Mapping[str, Any],
    *,
    frame_orientation: str = "9:16竖屏",
    storyboard_profile: str = "normal",
    asset_catalog: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized = dict(data)
    profile_settings = storyboard_profile_settings(storyboard_profile)
    normalized_frame_orientation = normalize_frame_orientation(frame_orientation)
    portrait_mode = is_portrait_frame_orientation(normalized_frame_orientation)
    prompt_entries = []
    raw_items = list(data.get("prompt_entries") or [])
    for index, raw_item in enumerate(raw_items):
        item = dict(raw_item)
        item["point_id"] = normalize_storyboard_point_id(item.get("point_id"), fallback_index=index + 1)
        item["title"] = strip_storyboard_title_prefix(item.get("title"), item["point_id"])
        item.pop("state_update", None)
        item.pop("resolved_opening_state", None)
        item.pop("resolved_closing_state", None)
        item["pace_label"] = normalize_pace_label(item.get("pace_label"))
        beats_target = 6 if item["pace_label"] == "快压推进" else 5 if item["pace_label"] == "中速推进" else 4
        item["density_strategy"] = str(item.get("density_strategy") or "").strip() or build_density_strategy(item["pace_label"], beats_target)
        item["duration_hint"] = normalize_duration_hint_text(str(item.get("duration_hint") or "10-13秒"))
        item["continuity_bridge"] = simplify_continuity_bridge(
            str(item.get("continuity_bridge") or "").strip(),
            max_length=profile_settings["continuity_max_chars"],
            max_clauses=profile_settings["continuity_max_clauses"],
        )
        master_timeline = normalize_master_timeline(
            item.get("master_timeline"),
            str(item.get("duration_hint") or ""),
            max_entries=profile_settings["max_shot_beats"],
            gap_seconds=profile_settings["dialogue_gap_seconds"],
        )
        if not master_timeline:
            master_timeline = build_master_timeline_from_legacy(item)
        item["master_timeline"] = master_timeline
        item["shot_beat_plan"] = ensure_timed_shot_beats(
            [
                str(x).strip()
                for x in list(item.get("shot_beat_plan") or [])
                if str(x).strip()
            ][: profile_settings["max_shot_beats"]],
            item["duration_hint"],
        )
        item["dialogue_timeline"] = normalize_dialogue_timeline(
            item.get("dialogue_timeline"),
            str(item.get("duration_hint") or ""),
            max_entries=profile_settings["max_dialogue_entries"],
            gap_seconds=profile_settings["dialogue_gap_seconds"],
        )
        item["audio_design"] = str(item.get("audio_design") or "").strip()
        item["risk_notes"] = [str(x).strip() for x in list(item.get("risk_notes") or []) if str(x).strip()][:6]
        prompt_text = str(item.get("prompt_text") or "").strip()
        item["prompt_text"] = strip_prompt_text_meta_blocks(prompt_text)
        item = sanitize_storyboard_item(item)
        if item.get("master_timeline"):
            item = materialize_storyboard_item_from_master_timeline(item)
        prompt_entries.append(item)
    normalized["prompt_entries"] = prompt_entries
    sanitized_global_notes: list[str] = []
    for note in list(data.get("global_notes") or []):
        clean_note, _ = sanitize_sensitive_text(str(note).strip())
        if clean_note:
            sanitized_global_notes.append(clean_note)
    default_notes = [
        "全局执行要求：同一集内保持统一主光方向、色温、材质质感、镜头高度与人物体量，不把每条分镜拍成独立短片。",
        "全局执行要求：运镜少而准，优先 1-2 个主运动，转场与镜头转向由动作、视线和道具驱动。",
        "全局执行要求：整体叙事速度较旧版常规模式提快约 25%-35%，前 20% 时长内进入有效剧情，每 2-3 秒给出新信息。",
    ]
    if portrait_mode:
        default_notes.insert(
            0,
            "全局执行要求：默认按9:16手机竖屏构图，主体尽量沿中轴或中轴偏上/偏下组织，优先使用中近景、纵深层次、上下高差与前后景遮挡承载戏剧信息。",
        )
    else:
        default_notes.insert(
            0,
            f"全局执行要求：当前目标画幅为 {normalized_frame_orientation}，所有镜头需遵守对应画幅安全区与主体集中原则。",
        )
    for default_note in default_notes:
        if default_note not in sanitized_global_notes:
            sanitized_global_notes.append(default_note)
    normalized["global_notes"] = sanitized_global_notes
    return normalized


def validate_storyboard_density(final_result: Mapping[str, Any], *, episode_id: str) -> list[str]:
    warnings: list[str] = []
    for item in list(final_result.get("prompt_entries") or []):
        beats = [str(x).strip() for x in list(item.get("shot_beat_plan") or []) if str(x).strip()]
        pace_label = normalize_pace_label(item.get("pace_label"))
        minimum_beats = 5 if pace_label == "快压推进" else 4 if pace_label == "中速推进" else 3
        if len(beats) < minimum_beats:
            warnings.append(
                f"{episode_id} 的 {item.get('point_id') or '未知分镜'} 镜头节拍过少："
                f"{pace_label} 至少需要 {minimum_beats} 个，当前只有 {len(beats)} 个。"
            )
        density_strategy = str(item.get("density_strategy") or "").strip()
        if not density_strategy:
            warnings.append(f"{episode_id} 的 {item.get('point_id') or '未知分镜'} 缺少 density_strategy。")
    return warnings


CHARACTER_CONTINUITY_KEYWORDS = (
    "前景",
    "中景",
    "后景",
    "近景",
    "远景",
    "中轴",
    "偏上",
    "偏下",
    "高位",
    "低位",
    "站位",
    "纵深",
    "遮挡",
    "入画",
    "退场",
    "逼近",
    "后撤",
    "转头",
    "回身",
    "抬手",
    "垂手",
    "手位",
    "视线",
    "目光",
    "对视",
    "停在",
    "压向",
    "逼到",
    "贴近",
    "错身",
    "侧身",
    "俯身",
    "起身",
)

CAMERA_STAGING_KEYWORDS = (
    "镜头",
    "推",
    "拉",
    "摇",
    "移",
    "跟",
    "切",
    "抬",
    "俯",
    "仰",
    "环",
    "扫",
    "掠过",
    "带到",
    "顺着",
    "跟到",
    "稳在",
    "回到",
    "切到",
    "转到",
    "推进",
    "回拉",
    "下切",
    "横移",
    "升起",
    "压低",
    "台阶",
    "通道",
    "门框",
    "高台",
    "中轴",
    "纵深",
    "前景",
    "后景",
)

FX_TRIGGER_KEYWORDS = (
    "雷",
    "雷光",
    "雷击",
    "火焰",
    "火光",
    "法术",
    "术式",
    "阵",
    "阵纹",
    "阵列",
    "符",
    "符纹",
    "禁制",
    "灵气",
    "剑气",
    "气浪",
    "冲击波",
    "爆闪",
    "炸开",
    "爆开",
    "爆裂",
    "余波",
    "烟尘",
    "烟浪",
    "机关",
    "异能",
    "能量",
    "光柱",
)

FX_SOURCE_KEYWORDS = (
    "亮起",
    "凝起",
    "汇聚",
    "自",
    "从",
    "由",
    "落下",
    "劈下",
    "劈落",
    "冲出",
    "爆起",
    "升起",
    "起于",
    "源头",
)

FX_PROPAGATION_KEYWORDS = (
    "沿",
    "顺着",
    "扩散",
    "席卷",
    "横扫",
    "蔓延",
    "冲开",
    "穿过",
    "掠过",
    "压向",
    "逼近",
    "卷向",
    "震开",
    "传播",
)

FX_REACTION_KEYWORDS = (
    "石屑",
    "碎屑",
    "焦痕",
    "裂纹",
    "火星",
    "烟尘",
    "震退",
    "扑倒",
    "受光",
    "反光",
    "衣袂",
    "发梢",
    "墙面",
    "地面",
    "人物反应",
    "环境反馈",
    "余震",
    "颤动",
    "摇晃",
)


def count_keyword_hits(text: str, keywords: Sequence[str]) -> int:
    normalized = str(text or "").strip()
    if not normalized:
        return 0
    return sum(1 for keyword in keywords if keyword and keyword in normalized)


def audit_normal_storyboard_richness(final_result: Mapping[str, Any], *, episode_id: str) -> list[str]:
    warnings: list[str] = []
    for item in list(final_result.get("prompt_entries") or []):
        point_id = str(item.get("point_id") or "未知分镜").strip() or "未知分镜"
        beats = [str(x).strip() for x in list(item.get("shot_beat_plan") or []) if str(x).strip()]
        prompt_text = str(item.get("prompt_text") or "").strip()
        continuity_bridge = str(item.get("continuity_bridge") or "").strip()
        audio_design = str(item.get("audio_design") or "").strip()
        dialogue_lines = [str(x.get("line") or "").strip() for x in list(item.get("dialogue_timeline") or []) if str(x.get("line") or "").strip()]

        continuity_text = "\n".join([continuity_bridge, prompt_text] + beats + dialogue_lines)
        continuity_hits = count_keyword_hits(continuity_text, CHARACTER_CONTINUITY_KEYWORDS)
        if continuity_hits < 2:
            warnings.append(
                f"{episode_id} 的 {point_id} 人物连续性描写偏薄：当前缺少足够的站位/视线/动作承接锚点。"
                "normal 模式至少应写清人物在前后景或中轴的关系、动作延续和情绪/视线交棒。"
            )

        staging_text = "\n".join([prompt_text, continuity_bridge] + beats)
        staging_hits = count_keyword_hits(staging_text, CAMERA_STAGING_KEYWORDS)
        if staging_hits < 3:
            warnings.append(
                f"{episode_id} 的 {point_id} 镜头调度描写偏薄：当前缺少足够的运镜路径或空间调度信息。"
                "normal 模式至少应交代镜头如何推拉摇移切，以及人物在前中后景、通道、高台、中轴等空间中的调度。"
            )

        fx_text = "\n".join([prompt_text, continuity_bridge, audio_design] + beats)
        fx_trigger_hits = count_keyword_hits(fx_text, FX_TRIGGER_KEYWORDS)
        if fx_trigger_hits:
            source_hits = count_keyword_hits(fx_text, FX_SOURCE_KEYWORDS)
            propagation_hits = count_keyword_hits(fx_text, FX_PROPAGATION_KEYWORDS)
            reaction_hits = count_keyword_hits(fx_text, FX_REACTION_KEYWORDS)
            missing_dimensions: list[str] = []
            if source_hits == 0:
                missing_dimensions.append("源头")
            if propagation_hits == 0:
                missing_dimensions.append("传播路径")
            if reaction_hits == 0:
                missing_dimensions.append("人物/环境反馈")
            if missing_dimensions:
                warnings.append(
                    f"{episode_id} 的 {point_id} 特效分层描写偏薄：已出现高能特效信号，但缺少"
                    f"{'、'.join(missing_dimensions)}。normal 模式需要把特效写成“触发 -> 扩散 -> 反馈”的完整链路。"
                )
    return warnings


def validate_scene_reference_presence(
    final_result: Mapping[str, Any],
    asset_catalog: list[Mapping[str, str]],
    *,
    episode_id: str,
) -> list[str]:
    return []


def _contains_any_pattern(text: str, patterns: Sequence[str]) -> bool:
    haystack = str(text or "")
    return any(pattern in haystack for pattern in patterns)


def _director_story_point_map(director_json: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if not isinstance(director_json, Mapping):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for item in list(director_json.get("story_points") or []):
        point_id = str(item.get("point_id") or "").strip()
        if point_id:
            result[point_id] = item
    return result


def validate_storyboard_execution_rules(
    final_result: Mapping[str, Any],
    director_json: Mapping[str, Any] | None,
    asset_catalog: list[Mapping[str, str]],
    *,
    episode_id: str,
) -> list[str]:
    story_point_map = _director_story_point_map(director_json)
    scene_ref_ids = {
        str(item.get("ref_id") or "").strip()
        for item in asset_catalog
        if str(item.get("asset_type") or "") == "场景参考" and str(item.get("ref_id") or "").strip()
    }
    warnings: list[str] = []
    for item in list(final_result.get("prompt_entries") or []):
        point_id = str(item.get("point_id") or "未知分镜").strip() or "未知分镜"
        prompt_text = str(item.get("prompt_text") or "").strip()
        continuity_bridge = str(item.get("continuity_bridge") or "").strip()
        beat_text = " ".join(str(x).strip() for x in list(item.get("shot_beat_plan") or []) if str(x).strip())
        combined_text = " ".join([str(item.get("title") or ""), prompt_text, continuity_bridge, beat_text]).strip()
        director_story_point = story_point_map.get(point_id, {})
        director_text = " ".join(
            [
                str(director_story_point.get("title") or ""),
                str(director_story_point.get("director_statement") or ""),
                " ".join(str(x).strip() for x in list(director_story_point.get("detail_anchor_lines") or []) if str(x).strip()),
                " ".join(str(x).strip() for x in list(director_story_point.get("key_dialogue_beats") or []) if str(x).strip()),
            ]
        ).strip()
        high_value = _contains_any_pattern(combined_text + " " + director_text, HIGH_VALUE_ENTRY_PATTERNS)
        if high_value and not _contains_any_pattern(combined_text, REACTION_SIGNAL_PATTERNS):
            warnings.append(
                f"{episode_id} 的 {point_id} 疑似缺少反应 beat：高价值段落里没有明显反应/停顿/视线承接信号，容易只剩事件播报。"
            )

        director_scenes = [str(scene).strip() for scene in list(director_story_point.get("scenes") or []) if str(scene).strip()]
        used_scene_refs = [
            ref
            for ref in list(item.get("primary_refs") or []) + list(item.get("secondary_refs") or [])
            if str(ref).strip() in scene_ref_ids
        ]
        if len(director_scenes) >= 2 and len(set(used_scene_refs)) < 2:
            warnings.append(
                f"{episode_id} 的 {point_id} 可能把多空间事件压扁成单空间：导演场景={director_scenes}，当前场景 refs={list(dict.fromkeys(used_scene_refs)) or '无'}。"
            )

        last_beat = str(list(item.get("shot_beat_plan") or [])[-1] if list(item.get("shot_beat_plan") or []) else "").strip()
        tail_text = " ".join([continuity_bridge, last_beat, prompt_text[-120:]]).strip()
        if not _contains_any_pattern(tail_text, TAIL_HOOK_SIGNAL_PATTERNS):
            warnings.append(
                f"{episode_id} 的 {point_id} 尾帧交棒信号偏弱：当前文本里缺少明确的停住/看向/将至/余波/尾帧落点提示。"
            )
    return warnings


def used_ref_ids(data: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []
    for item in data.get("prompt_entries", []):
        refs.extend(item.get("primary_refs", []))
        refs.extend(item.get("secondary_refs", []))
    ordered: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        ordered.append(ref)
    return ordered


def storyboard_entry_combined_text(item: Mapping[str, Any]) -> str:
    parts: list[str] = [str(item.get("continuity_bridge") or "").strip(), str(item.get("prompt_text") or "").strip()]
    for entry in list(item.get("master_timeline") or []):
        if not isinstance(entry, Mapping):
            continue
        parts.append(str(entry.get("visual_beat") or "").strip())
        parts.append(str(entry.get("transition_hook") or "").strip())
    return " ".join(part for part in parts if part)


def build_ref_grounding_note(ref_id: str, catalog_item: Mapping[str, Any]) -> str:
    asset_type = str(catalog_item.get("asset_type") or "").strip()
    lookup_name = str(catalog_item.get("lookup_name") or catalog_item.get("display_name") or ref_id).strip()
    if asset_type == "人物参考":
        return f"人物沿用参考{ref_id}的{lookup_name}既定造型，脸部、发式与胸前层次保持统一。"
    return ""


def repair_storyboard_ref_grounding(
    final_result: Mapping[str, Any],
    asset_catalog: list[Mapping[str, str]],
    *,
    episode_id: str,
    director_json: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    catalog_by_ref = {str(item.get("ref_id") or ""): item for item in asset_catalog if str(item.get("ref_id") or "").strip()}
    character_catalog = [item for item in asset_catalog if str(item.get("asset_type") or "") == "人物参考"]
    warnings: list[str] = []
    normalized = dict(final_result)
    repaired_entries: list[dict[str, Any]] = []
    for raw_item in list(final_result.get("prompt_entries") or []):
        item = dict(raw_item)
        point_id = str(item.get("point_id") or "未知分镜").strip() or "未知分镜"
        prompt_text = str(item.get("prompt_text") or "").strip()
        master_timeline = [dict(entry) for entry in list(item.get("master_timeline") or []) if isinstance(entry, Mapping)]
        declared_refs: list[str] = []
        seen_refs: set[str] = set()
        for ref in list(item.get("primary_refs") or []) + list(item.get("secondary_refs") or []):
            ref_text = str(ref).strip()
            if not ref_text or ref_text in seen_refs:
                continue
            seen_refs.add(ref_text)
            declared_refs.append(ref_text)

        if character_catalog:
            inferred_character_refs = infer_character_refs_for_entry(
                item,
                character_catalog,
                max_refs=max(0, 9 - len(declared_refs)),
            )
            if inferred_character_refs:
                secondary_refs = [str(ref).strip() for ref in list(item.get("secondary_refs") or []) if str(ref).strip()]
                added_character_refs: list[str] = []
                for ref in inferred_character_refs:
                    if ref in seen_refs:
                        continue
                    seen_refs.add(ref)
                    declared_refs.append(ref)
                    added_character_refs.append(ref)
                    if ref not in secondary_refs:
                        secondary_refs.append(ref)
                if added_character_refs:
                    item["secondary_refs"] = secondary_refs
                    labels = [
                        str(catalog_by_ref.get(ref, {}).get("display_name") or ref).strip()
                        for ref in added_character_refs
                    ]
                    warnings.append(
                        f"{episode_id} 的 {point_id} 自动补入了正文已出镜但未绑定的人物引用 {added_character_refs}（{'；'.join(labels)}）。"
                    )

        declared_character_refs = [
            ref for ref in declared_refs if str(catalog_by_ref.get(ref, {}).get("asset_type") or "") == "人物参考"
        ]
        if declared_character_refs:
            item, repaired_pairs = repair_character_visual_bindings_for_item(
                item,
                character_refs=declared_character_refs,
                catalog_by_ref=catalog_by_ref,
            )
            if repaired_pairs:
                master_timeline = [dict(entry) for entry in list(item.get("master_timeline") or []) if isinstance(entry, Mapping)]
                prompt_text = str(item.get("prompt_text") or "").strip()
                warnings.append(
                    f"{episode_id} 的 {point_id} 自动把人物引用收敛为“编号+人名”成对写法：{'；'.join(repaired_pairs[:6])}。"
                )

        combined_text = storyboard_entry_combined_text(item)
        for ref in declared_character_refs:
            if text_contains_ref_token(combined_text, ref):
                continue
            catalog_item = catalog_by_ref.get(ref, {})
            display_name = str(catalog_item.get("display_name") or ref).strip()
            binding_note = build_ref_grounding_note(ref, catalog_item)
            if binding_note not in prompt_text:
                if master_timeline:
                    first_entry = dict(master_timeline[0])
                    visual_beat = str(first_entry.get("visual_beat") or "").strip()
                    if binding_note not in visual_beat:
                        first_entry["visual_beat"] = (visual_beat + " " + binding_note).strip()
                        master_timeline[0] = first_entry
                        item["master_timeline"] = master_timeline
                else:
                    prompt_text = prompt_text.rstrip() + "\n\n" + binding_note
                    item["prompt_text"] = prompt_text
            warnings.append(
                f"{episode_id} 的 {point_id} 自动补写了 {ref} 的显式引用绑定（{display_name}）。"
            )
            combined_text = storyboard_entry_combined_text(item)

        mentioned_refs: list[str] = []
        mentioned_seen: set[str] = set()
        for ref in re.findall(r"@图片\d+", prompt_text):
            if ref in mentioned_seen:
                continue
            mentioned_seen.add(ref)
            mentioned_refs.append(ref)
        undeclared_refs = [
            ref
            for ref in mentioned_refs
            if ref not in seen_refs and str(catalog_by_ref.get(ref, {}).get("asset_type") or "") == "人物参考"
        ]
        if undeclared_refs:
            secondary_refs = [str(ref).strip() for ref in list(item.get("secondary_refs") or []) if str(ref).strip()]
            for ref in undeclared_refs:
                if ref not in secondary_refs:
                    secondary_refs.append(ref)
            item["secondary_refs"] = secondary_refs
            warnings.append(
                f"{episode_id} 的 {point_id} 自动把未声明引用 {undeclared_refs} 补入 secondary_refs。"
            )
        if master_timeline:
            item["master_timeline"] = master_timeline
            item = materialize_storyboard_item_from_master_timeline(item)
        else:
            item["prompt_text"] = prompt_text
        repaired_entries.append(item)

    normalized["prompt_entries"] = repaired_entries
    if warnings:
        global_notes = [str(x).strip() for x in list(normalized.get("global_notes") or []) if str(x).strip()]
        global_notes.extend([f"自动修正：{message}" for message in warnings[:8]])
        normalized["global_notes"] = _unique_ordered_lines(global_notes)[:20]
    return normalized, warnings


def has_explicit_character_ref_name_pair(text: str, ref_id: str, lookup_name: str) -> bool:
    raw_text = str(text or "")
    if not raw_text or not ref_id or not lookup_name:
        return False
    ref_fragment = rf"{re.escape(ref_id)}(?!\d)"
    lookup_fragment = rf"(?<!{NAME_TOKEN_NEIGHBOR_CLASS}){re.escape(lookup_name)}(?!{NAME_TOKEN_NEIGHBOR_CLASS})"
    pattern = (
        rf"{ref_fragment}[^。；，、,.!?\n]{{0,12}}{lookup_fragment}"
        rf"|{lookup_fragment}[^。；，、,.!?\n]{{0,8}}{ref_fragment}"
    )
    return re.search(pattern, raw_text) is not None


def repair_character_ref_name_pair_in_text(text: str, *, ref_id: str, lookup_name: str) -> tuple[str, bool]:
    raw_text = str(text or "").strip()
    if not raw_text or not ref_id or not lookup_name:
        return raw_text, False
    ref_present = text_contains_ref_token(raw_text, ref_id)
    lookup_name_present = text_contains_lookup_name_token(raw_text, lookup_name)
    if not ref_present and not lookup_name_present:
        return raw_text, False
    if has_explicit_character_ref_name_pair(raw_text, ref_id, lookup_name):
        return raw_text, False

    repaired = raw_text
    changed = False
    ref_fragment = rf"{re.escape(ref_id)}(?!\d)"
    generic_patterns = [
        rf"参考\s*{ref_fragment}(?:所示|对应)?的?(?:人物|角色|人影|身影|身形|身躯|人)",
        rf"{ref_fragment}(?:所示|对应)?的?(?:人物|角色|人影|身影|身形|身躯|人)",
    ]
    for pattern in generic_patterns:
        repaired, count = re.subn(pattern, f"参考{ref_id}的{lookup_name}，", repaired, count=1)
        if count:
            changed = True
            break
    repaired = re.sub(r"，([，。；：、,.!?])", r"\1", repaired)

    if not has_explicit_character_ref_name_pair(repaired, ref_id, lookup_name) and text_contains_ref_token(repaired, ref_id):
        repaired, count = re.subn(
            rf"参考\s*{ref_fragment}(?![^。；，、,.!?\n]{{0,12}}{re.escape(lookup_name)})",
            f"参考{ref_id}的{lookup_name}",
            repaired,
            count=1,
        )
        if not count:
            repaired, count = re.subn(
                rf"{ref_fragment}(?![^。；，、,.!?\n]{{0,12}}{re.escape(lookup_name)})",
                f"参考{ref_id}的{lookup_name}",
                repaired,
                count=1,
            )
        if count:
            changed = True

    if (
        not has_explicit_character_ref_name_pair(repaired, ref_id, lookup_name)
        and text_contains_lookup_name_token(repaired, lookup_name)
        and not text_contains_ref_token(repaired, ref_id)
    ):
        repaired, count = replace_first_lookup_name_token(
            repaired,
            lookup_name,
            f"参考{ref_id}的{lookup_name}",
        )
        if count:
            changed = True

    if not has_explicit_character_ref_name_pair(repaired, ref_id, lookup_name) and (
        text_contains_lookup_name_token(repaired, lookup_name) or text_contains_ref_token(repaired, ref_id)
    ):
        prefix = f"参考{ref_id}的{lookup_name}。"
        if prefix not in repaired:
            repaired = prefix + repaired
            changed = True

    return repaired.strip(), changed


def repair_character_visual_bindings_for_item(
    item: dict[str, Any],
    *,
    character_refs: Sequence[str],
    catalog_by_ref: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    if not character_refs:
        return item, []

    updated_item = dict(item)
    touched_pairs: list[str] = []
    master_timeline = [dict(entry) for entry in list(updated_item.get("master_timeline") or []) if isinstance(entry, Mapping)]
    if master_timeline:
        for ref in character_refs:
            catalog_item = catalog_by_ref.get(ref, {})
            lookup_name = str(catalog_item.get("lookup_name") or catalog_item.get("display_name") or "").strip()
            if not lookup_name:
                continue
            already_bound = any(
                has_explicit_character_ref_name_pair(str(entry.get("visual_beat") or "").strip(), ref, lookup_name)
                for entry in master_timeline
            )
            if already_bound:
                continue
            for index, entry in enumerate(master_timeline):
                visual_beat = str(entry.get("visual_beat") or "").strip()
                if not visual_beat:
                    continue
                if not (text_contains_ref_token(visual_beat, ref) or text_contains_lookup_name_token(visual_beat, lookup_name)):
                    continue
                repaired_text, changed = repair_character_ref_name_pair_in_text(
                    visual_beat,
                    ref_id=ref,
                    lookup_name=lookup_name,
                )
                if changed:
                    entry["visual_beat"] = repaired_text
                    master_timeline[index] = entry
                    pair_label = f"{ref}={lookup_name}"
                    if pair_label not in touched_pairs:
                        touched_pairs.append(pair_label)
                break
        updated_item["master_timeline"] = master_timeline
        if touched_pairs:
            updated_item = materialize_storyboard_item_from_master_timeline(updated_item)
        return updated_item, touched_pairs

    prompt_text = str(updated_item.get("prompt_text") or "").strip()
    if not prompt_text:
        return updated_item, []
    repaired_prompt = prompt_text
    for ref in character_refs:
        catalog_item = catalog_by_ref.get(ref, {})
        lookup_name = str(catalog_item.get("lookup_name") or catalog_item.get("display_name") or "").strip()
        if not lookup_name:
            continue
        repaired_prompt, changed = repair_character_ref_name_pair_in_text(
            repaired_prompt,
            ref_id=ref,
            lookup_name=lookup_name,
        )
        if changed:
            pair_label = f"{ref}={lookup_name}"
            if pair_label not in touched_pairs:
                touched_pairs.append(pair_label)
    if touched_pairs:
        updated_item["prompt_text"] = repaired_prompt
    return updated_item, touched_pairs


def validate_character_ref_name_pairing(
    final_result: Mapping[str, Any],
    asset_catalog: Sequence[Mapping[str, Any]],
    *,
    episode_id: str,
) -> list[str]:
    catalog_by_ref = {
        str(item.get("ref_id") or "").strip(): item
        for item in asset_catalog
        if str(item.get("ref_id") or "").strip()
    }
    warnings: list[str] = []
    for item in list(final_result.get("prompt_entries") or []):
        point_id = str(item.get("point_id") or "未知分镜").strip() or "未知分镜"
        character_refs = [
            ref
            for ref in list(item.get("primary_refs") or []) + list(item.get("secondary_refs") or [])
            if str(catalog_by_ref.get(str(ref).strip(), {}).get("asset_type") or "") == "人物参考"
        ]
        if not character_refs:
            continue
        combined_text = storyboard_entry_combined_text(item)
        issue_count = 0
        for ref in character_refs:
            catalog_item = catalog_by_ref.get(str(ref).strip(), {})
            lookup_name = str(catalog_item.get("lookup_name") or catalog_item.get("display_name") or "").strip()
            if not lookup_name:
                continue
            ref_present = text_contains_ref_token(combined_text, ref)
            name_present = text_contains_lookup_name_token(combined_text, lookup_name)
            if not ref_present and not name_present:
                continue
            if has_explicit_character_ref_name_pair(combined_text, str(ref), lookup_name):
                continue
            if ref_present and not name_present:
                warnings.append(
                    f"{episode_id} 的 {point_id} 人物绑定过泛：{ref} 已出现，但没有显式写出角色名 {lookup_name}。"
                )
            elif name_present and not ref_present:
                warnings.append(
                    f"{episode_id} 的 {point_id} 人物绑定缺 ref：{lookup_name} 已出现，但缺少对应的 {ref}。"
                )
            else:
                warnings.append(
                    f"{episode_id} 的 {point_id} 人物绑定未成对出现：{ref} 与 {lookup_name} 没有在同一分镜里显式配对。"
                )
            issue_count += 1
            if issue_count >= 3:
                break
    return warnings


def validate_storyboard_coverage(
    director_json: Mapping[str, Any] | None,
    final_result: Mapping[str, Any],
    *,
    episode_id: str,
) -> list[str]:
    if not isinstance(director_json, Mapping):
        return []
    expected = [
        normalize_storyboard_point_id(item.get("point_id"), fallback_index=index + 1)
        for index, item in enumerate(list(director_json.get("story_points") or []))
        if isinstance(item, Mapping)
    ]
    expected = [item for item in expected if item]
    if not expected:
        return []
    actual = [
        normalize_storyboard_point_id(item.get("point_id"), fallback_index=index + 1)
        for index, item in enumerate(list(final_result.get("prompt_entries") or []))
        if isinstance(item, Mapping)
    ]
    actual = [item for item in actual if item]
    missing = [item for item in expected if item not in actual]
    extras = [item for item in actual if item not in expected]
    if expected != actual:
        return [
            f"{episode_id} 的 Seedance 分镜覆盖不完整或顺序不一致："
            f"导演剧情点={expected}，实际输出={actual}，缺失={missing or '无'}，额外={extras or '无'}"
        ]
    return []


def render_seedance_markdown(
    *,
    series_name: str,
    data: Mapping[str, Any],
    asset_catalog: list[Mapping[str, str]],
) -> str:
    used_refs = set(used_ref_ids(data))
    visible_catalog = [item for item in asset_catalog if item["ref_id"] in used_refs]

    episode_label = str(data.get("episode_id") or "").upper() or "EP"
    lines = [
        f"# Seedance 2.0 提示词 -- {series_name} {episode_label}",
        "",
        "## 素材对应表",
        "",
        "| 引用编号 | 素材类型 | 对应素材 |",
        "|---------|---------|---------|",
    ]
    for item in visible_catalog:
        lines.append(f"| {item['ref_id']} | {item['asset_type']} | {item['display_name']} |")
    if not visible_catalog:
        lines.append("| 无 | - | 本集未引用现成素材，均需文字描述 |")

    lines.extend(["", "---", ""])
    for item in data.get("prompt_entries", []):
        all_refs = item.get("primary_refs", []) + item.get("secondary_refs", [])
        ref_line = " ".join(all_refs)
        lines.extend(
            [
                f"## {item['point_id']} {item['title']}",
                "",
                f"- 节奏档位：{item['pace_label']}",
                f"- 内容密度策略：{item['density_strategy']}",
                f"- 建议时长：{item['duration_hint']}",
                f"- 主要引用：{ref_line or '无'}",
            ]
        )
        continuity_bridge = str(item.get("continuity_bridge") or "").strip()
        audio_design = str(item.get("audio_design") or "").strip()
        risk_notes = [str(x).strip() for x in list(item.get("risk_notes") or []) if str(x).strip()]
        if continuity_bridge:
            lines.append(f"- 承接关系：{continuity_bridge}")
        if audio_design:
            lines.append(f"- 声音设计：{audio_design}")
        if risk_notes:
            lines.append(f"- 风险提示：{'；'.join(risk_notes)}")
        master_timeline = [dict(entry) for entry in list(item.get("master_timeline") or []) if isinstance(entry, Mapping)]
        lines.extend(["", "**统一复合提示词（主时间线）**：", ""])
        if master_timeline:
            lines.extend(render_master_timeline_markdown_lines(master_timeline))
            prompt_text = str(item.get("prompt_text") or "").strip()
            if prompt_text:
                lines.extend(["", "可直接投喂正文：", "", prompt_text])
        else:
            lines.append("- 无")
        lines.extend(["", "---", ""])
    return "\n".join(lines).rstrip() + "\n"


def save_storyboard_draft_outputs(
    *,
    series_name: str,
    final_result: Mapping[str, Any],
    asset_catalog: list[Mapping[str, str]],
    storyboard_md_path: Path,
    storyboard_json_path: Path,
    pending_marker_path: Path,
) -> None:
    draft_package = dict(final_result)
    global_notes = [str(x).strip() for x in list(draft_package.get("global_notes") or []) if str(x).strip()]
    if STORYBOARD_DRAFT_PENDING_NOTICE not in global_notes:
        global_notes.insert(0, STORYBOARD_DRAFT_PENDING_NOTICE)
    draft_package["global_notes"] = global_notes[:20]
    markdown = render_seedance_markdown(series_name=series_name, data=draft_package, asset_catalog=asset_catalog)
    save_text_file(storyboard_md_path, markdown)
    save_json_file(storyboard_json_path, draft_package)
    pending_marker_path.parent.mkdir(parents=True, exist_ok=True)
    pending_marker_path.write_text("review_pending\n", encoding="utf-8")


STORYBOARD_DRAFT_PENDING_NOTICE = "当前文件为已落盘初稿，复审仍在继续；若流程中断，这份文件可先作为可读草稿使用。"


def strip_storyboard_draft_pending_notice(data: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    global_notes = [
        str(x).strip()
        for x in list(normalized.get("global_notes") or [])
        if str(x).strip() and str(x).strip() != STORYBOARD_DRAFT_PENDING_NOTICE
    ]
    normalized["global_notes"] = global_notes
    return normalized

def run_pipeline(config: Mapping[str, Any], telemetry: TelemetryRecorder | None = None) -> dict[str, Any]:
    model, api_key = configure_openai_api(config)
    series_name = resolve_series_name(config)
    episode_ids = build_episode_ids(config.get("series", {}))
    assets_dir = resolve_assets_dir(config, series_name)
    character_prompts_path = assets_dir / "character-prompts.md"
    scene_prompts_path = assets_dir / "scene-prompts.md"
    character_prompts_text = read_text(character_prompts_path)
    scene_prompts_text = read_text(scene_prompts_path)
    # 优化方案1：优先使用预计算的compact bundle，避免重复加载和压缩
    genre_reference_bundle = config.get("_precomputed_bundle_cache", {}).get("storyboard") or load_genre_reference_bundle(config, series_name)
    seedance_purpose_skill_library = load_seedance_purpose_skill_library(config, series_name)
    seedance_purpose_template_library = load_seedance_purpose_template_library(config, series_name)
    timeout_seconds = int(config.get("run", {}).get("timeout_seconds", 300))
    temperature = float(config.get("run", {}).get("temperature", 0.4))
    enable_review_pass = bool(config.get("run", {}).get("enable_review_pass", True))
    dry_run = bool(config.get("run", {}).get("dry_run", False))
    storyboard_profile = resolve_storyboard_profile(config)
    profile_settings = storyboard_profile_settings(storyboard_profile)
    storyboard_profile_label = profile_settings["label"]
    write_storyboard_metrics = bool(config.get("run", {}).get("write_storyboard_metrics", True))
    provider_tag = build_provider_model_tag("openai", model)

    print_status(f"剧名：{series_name}")
    print_status(f"素材目录：{assets_dir}")
    print_status(f"storyboard 模式：{storyboard_profile_label}")

    previews: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        asset_catalog = build_asset_catalog(
            character_prompts_text,
            scene_prompts_text,
            episode_id=episode_id,
            assets_dir=assets_dir,
        )
        director_md_path = find_director_markdown_path(config, series_name, episode_id)
        director_json_path = find_director_json_path(config, series_name, episode_id)
        episode_output_dir = resolve_episode_output_dir(config, series_name, episode_id)
        storyboard_md_path = episode_output_dir / "02-seedance-prompts.md"
        storyboard_json_path = episode_output_dir / f"02-seedance-prompts__{provider_tag}.json"
        storyboard_pending_marker_path = episode_output_dir / "02-seedance-prompts.review-pending"

        previews.append(
            {
                "episode_id": episode_id,
                "director_markdown_path": str(director_md_path),
                "director_json_path": str(director_json_path) if director_json_path else None,
                "storyboard_markdown_path": str(storyboard_md_path),
                "storyboard_json_path": str(storyboard_json_path),
                "asset_catalog_size": len(asset_catalog),
            }
        )
        if dry_run:
            continue

        episode_telemetry = TelemetryRecorder(
            run_name="seedance-storyboard",
            context={
                "series_name": series_name,
                "episode_id": episode_id,
                "model": model,
                "storyboard_profile": storyboard_profile,
            },
        )
        metrics_json_path, metrics_md_path = storyboard_metrics_paths(episode_output_dir, provider_tag)
        try:
            with telemetry_span(
                episode_telemetry,
                stage="storyboard",
                name="load_storyboard_stage_inputs",
                metadata={
                    "episode_id": episode_id,
                    "director_markdown_path": str(director_md_path),
                    "storyboard_profile": storyboard_profile,
                },
            ) as step:
                director_markdown = director_md_path.read_text(encoding="utf-8")
                director_json = None
                if director_json_path and director_json_path.exists():
                    with director_json_path.open("r", encoding="utf-8") as handle:
                        director_json = json.load(handle)
                seedance_story_point_guidance = build_seedance_story_point_guidance(
                    director_json,
                    seedance_purpose_skill_library,
                    seedance_purpose_template_library,
                    profile=storyboard_profile,
                )
                compact_director_brief = ctx_compact.compact_director_brief_for_storyboard(
                    director_json,
                    director_markdown,
                    profile=storyboard_profile,
                )
                compact_director_json = ctx_compact.compact_director_json_for_storyboard(
                    director_json,
                    profile=storyboard_profile,
                )
                compact_genre_reference_bundle = ctx_compact.compact_genre_reference_bundle_for_storyboard(
                    genre_reference_bundle,
                    profile=storyboard_profile,
                )
                compact_asset_catalog = ctx_compact.compact_asset_catalog_for_storyboard(
                    asset_catalog,
                    profile=storyboard_profile,
                )
                compact_character_prompts_text = ctx_compact.compact_episode_scoped_prompt_library(
                    character_prompts_text,
                    episode_id,
                    limit=profile_settings["draft_character_limit"],
                    max_recent_blocks=2,
                )
                compact_scene_prompts_text = ctx_compact.compact_episode_scoped_prompt_library(
                    scene_prompts_text,
                    episode_id,
                    limit=profile_settings["draft_scene_limit"],
                    max_recent_blocks=2,
                )
                step["metadata"]["director_markdown_chars"] = len(director_markdown)
                step["metadata"]["director_brief_chars"] = len(compact_director_brief)
                step["metadata"]["director_json_chars"] = len(json.dumps(director_json or {}, ensure_ascii=False))
                step["metadata"]["director_json_compact_chars"] = len(json.dumps(compact_director_json, ensure_ascii=False))
                step["metadata"]["genre_reference_bundle_chars"] = len(json.dumps(genre_reference_bundle, ensure_ascii=False))
                step["metadata"]["genre_reference_bundle_compact_chars"] = len(json.dumps(compact_genre_reference_bundle, ensure_ascii=False))
                step["metadata"]["seedance_purpose_skill_library_chars"] = len(
                    json.dumps(seedance_purpose_skill_library, ensure_ascii=False)
                )
                step["metadata"]["seedance_purpose_template_library_chars"] = len(
                    json.dumps(seedance_purpose_template_library, ensure_ascii=False)
                )
                step["metadata"]["seedance_story_point_guidance_chars"] = len(
                    json.dumps(seedance_story_point_guidance, ensure_ascii=False)
                )
                step["metadata"]["character_prompts_chars"] = len(character_prompts_text)
                step["metadata"]["character_prompts_compact_chars"] = len(compact_character_prompts_text)
                step["metadata"]["scene_prompts_chars"] = len(scene_prompts_text)
                step["metadata"]["scene_prompts_compact_chars"] = len(compact_scene_prompts_text)
                step["metadata"]["asset_catalog_size"] = len(asset_catalog)
                step["metadata"]["asset_catalog_compact_size"] = len(compact_asset_catalog)

            resume_review_from_existing_draft = (
                enable_review_pass
                and storyboard_pending_marker_path.exists()
                and storyboard_json_path.exists()
            )

            if resume_review_from_existing_draft:
                print_status(f"检测到 {episode_id} 已有待复审初稿，跳过 draft，直接继续 review 修订。")
                with telemetry_span(
                    episode_telemetry,
                    stage="storyboard",
                    name="load_existing_storyboard_draft",
                    metadata={
                        "episode_id": episode_id,
                        "storyboard_profile": storyboard_profile,
                        "storyboard_json_path": str(storyboard_json_path),
                    },
                ) as step:
                    with storyboard_json_path.open("r", encoding="utf-8") as handle:
                        draft_result = json.load(handle)
                    step["metadata"]["loaded_prompt_entries"] = len(list(draft_result.get("prompt_entries") or []))
                final_result = draft_result
            else:
                if storyboard_pending_marker_path.exists() and storyboard_md_path.exists():
                    print_status(f"检测到 {episode_id} 已有待复审初稿，但缺少可续审的 draft JSON，将重新生成初稿。")
                else:
                    print_status(f"开始生成 {episode_id} 的 Seedance 提示词。")
                with telemetry_span(
                    episode_telemetry,
                    stage="storyboard",
                    name="build_storyboard_draft_prompt",
                    metadata={"episode_id": episode_id, "storyboard_profile": storyboard_profile},
                ) as step:
                    draft_prompt_context = build_episode_prompt_context(
                        config=config,
                        series_name=series_name,
                        episode_id=episode_id,
                        director_markdown=director_markdown,
                        director_json=director_json,
                        genre_reference_bundle=genre_reference_bundle,
                        seedance_story_point_guidance=seedance_story_point_guidance,
                        character_prompts_text=character_prompts_text,
                        scene_prompts_text=scene_prompts_text,
                        asset_catalog=asset_catalog,
                    )
                    draft_prompt, prompt_debug_metadata = render_prompt_with_debug_metadata(
                        "seedance_storyboard/draft_user.md",
                        draft_prompt_context,
                    )
                    step["metadata"]["prompt_chars"] = len(draft_prompt)
                    step["metadata"].update(prompt_debug_metadata)
                draft_result = openai_json_completion(
                    model=model,
                    api_key=api_key,
                    system_prompt=load_prompt("seedance_storyboard/draft_system.md"),
                    user_prompt=draft_prompt,
                    schema_name="seedance_prompt_package",
                    schema=SEEDANCE_PROMPTS_SCHEMA,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                    telemetry=episode_telemetry,
                    stage="storyboard",
                    step_name="storyboard_draft_model_call",
                    metadata={"episode_id": episode_id, "storyboard_profile": storyboard_profile},
                )
                final_result = draft_result

                draft_materialized_result = normalize_storyboard_result(
                    draft_result,
                    frame_orientation=str(config.get("quality", {}).get("frame_orientation") or "9:16竖屏"),
                    storyboard_profile=storyboard_profile,
                    asset_catalog=asset_catalog,
                )
                draft_materialized_result = repair_storyboard_density(
                    draft_materialized_result,
                    max_shot_beats=profile_settings["max_shot_beats"],
                )
                draft_materialized_result, draft_grounding_warnings = repair_storyboard_ref_grounding(
                    draft_materialized_result,
                    asset_catalog,
                    episode_id=episode_id,
                    director_json=director_json,
                )
                if draft_grounding_warnings:
                    print_status(
                        f"{episode_id} 的初稿存在引用绑定小问题，已自动修正 {len(draft_grounding_warnings)} 处并先行落盘。"
                    )
                draft_review_defects = validate_storyboard_coverage(
                    director_json,
                    draft_materialized_result,
                    episode_id=episode_id,
                ) + validate_storyboard_density(
                    draft_materialized_result,
                    episode_id=episode_id,
                )
                if storyboard_profile == "normal":
                    draft_review_defects += audit_normal_storyboard_richness(
                        draft_materialized_result,
                        episode_id=episode_id,
                    )
                draft_review_defects += validate_storyboard_execution_rules(
                    draft_materialized_result,
                    director_json,
                    asset_catalog,
                    episode_id=episode_id,
                ) + validate_character_ref_name_pairing(
                    draft_materialized_result,
                    asset_catalog,
                    episode_id=episode_id,
                )
                save_storyboard_draft_outputs(
                    series_name=series_name,
                    final_result=draft_materialized_result,
                    asset_catalog=asset_catalog,
                    storyboard_md_path=storyboard_md_path,
                    storyboard_json_path=storyboard_json_path,
                    pending_marker_path=storyboard_pending_marker_path,
                )
                print_status(f"{episode_id} 的 Seedance 初稿已写入：{storyboard_md_path}")

            review_input_package = (
                normalize_storyboard_result(
                    final_result,
                    frame_orientation=str(config.get("quality", {}).get("frame_orientation") or "9:16竖屏"),
                    storyboard_profile=storyboard_profile,
                    asset_catalog=asset_catalog,
                )
                if resume_review_from_existing_draft
                else draft_materialized_result
            )

            if enable_review_pass:
                print_status(f"开始复审并修订 {episode_id} 的 Seedance 提示词。")
                with telemetry_span(
                    episode_telemetry,
                    stage="storyboard",
                    name="build_storyboard_review_prompt",
                    metadata={"episode_id": episode_id, "storyboard_profile": storyboard_profile},
                ) as step:
                    review_prompt_context = build_review_prompt_context(
                        config=config,
                        series_name=series_name,
                        episode_id=episode_id,
                        director_json=director_json,
                        genre_reference_bundle=genre_reference_bundle,
                        seedance_story_point_guidance=seedance_story_point_guidance,
                        asset_catalog=asset_catalog,
                        draft_package=review_input_package,
                        draft_defects=draft_review_defects if not resume_review_from_existing_draft else None,
                    )
                    review_prompt, prompt_debug_metadata = render_prompt_with_debug_metadata(
                        "seedance_storyboard/review_user.md",
                        review_prompt_context,
                    )
                    step["metadata"]["prompt_chars"] = len(review_prompt)
                    step["metadata"].update(prompt_debug_metadata)
                review_patch = openai_json_completion(
                    model=model,
                    api_key=api_key,
                    system_prompt=load_prompt("seedance_storyboard/review_system.md"),
                    user_prompt=review_prompt,
                    schema_name="seedance_prompt_review_patch",
                    schema=SEEDANCE_REVIEW_PATCH_SCHEMA,
                    temperature=max(0.15, min(temperature, 0.25)),
                    timeout_seconds=timeout_seconds,
                    telemetry=episode_telemetry,
                    stage="storyboard",
                    step_name="storyboard_review_model_call",
                    metadata={"episode_id": episode_id, "storyboard_profile": storyboard_profile},
                )
                final_result = merge_storyboard_review_patch(
                    review_input_package,
                    review_patch,
                    director_json=director_json,
                )

            with telemetry_span(
                episode_telemetry,
                stage="storyboard",
                name="render_and_save_storyboard_outputs",
                metadata={
                    "episode_id": episode_id,
                    "storyboard_markdown_path": str(storyboard_md_path),
                    "storyboard_json_path": str(storyboard_json_path),
                    "storyboard_profile": storyboard_profile,
                },
            ) as step:
                final_result = normalize_storyboard_result(
                    final_result,
                    frame_orientation=str(config.get("quality", {}).get("frame_orientation") or "9:16竖屏"),
                    storyboard_profile=storyboard_profile,
                    asset_catalog=asset_catalog,
                )
                final_result = strip_storyboard_draft_pending_notice(final_result)
                final_result = repair_storyboard_density(
                    final_result,
                    max_shot_beats=profile_settings["max_shot_beats"],
                )
                coverage_warnings = validate_storyboard_coverage(director_json, final_result, episode_id=episode_id)
                density_warnings = validate_storyboard_density(final_result, episode_id=episode_id)
                quality_warnings = coverage_warnings + density_warnings
                if quality_warnings:
                    step.setdefault("metadata", {})
                    step["metadata"]["quality_warning_count"] = len(quality_warnings)
                    step["metadata"]["quality_warnings"] = quality_warnings[:20]
                    if episode_telemetry is not None:
                        episode_telemetry.context["storyboard_validation_warning_count"] = len(quality_warnings)
                        episode_telemetry.context["storyboard_validation_warnings"] = quality_warnings[:20]
                    print_status(
                        f"{episode_id} 的分镜结构检查提示 {len(quality_warnings)} 条：结果将继续保存，不再因覆盖/密度阈值中断。"
                    )
                if storyboard_profile == "normal":
                    richness_warnings = audit_normal_storyboard_richness(final_result, episode_id=episode_id)
                    if richness_warnings:
                        step.setdefault("metadata", {})
                        existing_warning_count = int(step["metadata"].get("quality_warning_count", 0) or 0)
                        existing_warnings = list(step["metadata"].get("quality_warnings") or [])
                        merged_warnings = existing_warnings + richness_warnings
                        step["metadata"]["quality_warning_count"] = existing_warning_count + len(richness_warnings)
                        step["metadata"]["quality_warnings"] = merged_warnings[:20]
                        if episode_telemetry is not None:
                            existing_context_count = int(
                                episode_telemetry.context.get("storyboard_quality_warning_count", 0) or 0
                            )
                            existing_context_warnings = list(
                                episode_telemetry.context.get("storyboard_quality_warnings") or []
                            )
                            episode_telemetry.context["storyboard_quality_warning_count"] = (
                                existing_context_count + len(richness_warnings)
                            )
                            episode_telemetry.context["storyboard_quality_warnings"] = (
                                existing_context_warnings + richness_warnings
                            )[:20]
                        print_status(
                            f"{episode_id} 的 normal 模式质量审计提示 {len(richness_warnings)} 条："
                            "人物承接、镜头调度或特效分层仍有变薄风险，结果已保留并继续保存。"
                        )
                final_result, grounding_warnings = repair_storyboard_ref_grounding(
                    final_result,
                    asset_catalog,
                    episode_id=episode_id,
                    director_json=director_json,
                )
                if grounding_warnings:
                    print_status(
                        f"{episode_id} 的引用绑定存在小问题，已自动修正 {len(grounding_warnings)} 处并继续保存结果。"
                    )
                scene_ref_warnings = validate_scene_reference_presence(final_result, asset_catalog, episode_id=episode_id)
                execution_rule_warnings = validate_storyboard_execution_rules(
                    final_result,
                    director_json,
                    asset_catalog,
                    episode_id=episode_id,
                )
                character_binding_warnings = validate_character_ref_name_pairing(
                    final_result,
                    asset_catalog,
                    episode_id=episode_id,
                )
                storyboard_validation_warnings = (
                    scene_ref_warnings + execution_rule_warnings + character_binding_warnings
                )
                if storyboard_validation_warnings:
                    step.setdefault("metadata", {})
                    existing_warning_count = int(step["metadata"].get("quality_warning_count", 0) or 0)
                    merged_warnings = list(step["metadata"].get("quality_warnings") or []) + storyboard_validation_warnings
                    step["metadata"]["quality_warning_count"] = existing_warning_count + len(storyboard_validation_warnings)
                    step["metadata"]["quality_warnings"] = merged_warnings[:20]
                    if episode_telemetry is not None:
                        existing_context_count = int(
                            episode_telemetry.context.get("storyboard_validation_warning_count", 0) or 0
                        )
                        existing_context_warnings = list(
                            episode_telemetry.context.get("storyboard_validation_warnings") or []
                        )
                        episode_telemetry.context["storyboard_validation_warning_count"] = (
                            existing_context_count + len(storyboard_validation_warnings)
                        )
                        episode_telemetry.context["storyboard_validation_warnings"] = (
                            existing_context_warnings + storyboard_validation_warnings
                        )[:20]
                    print_status(
                        f"{episode_id} 的 storyboard 规则检查提示 {len(storyboard_validation_warnings)} 条：结果已保留并继续保存。"
                    )
                markdown = render_seedance_markdown(series_name=series_name, data=final_result, asset_catalog=asset_catalog)
                save_text_file(storyboard_md_path, markdown)
                save_json_file(storyboard_json_path, final_result)
                if storyboard_pending_marker_path.exists():
                    storyboard_pending_marker_path.unlink()
            episode_telemetry.context["final_status"] = "completed"
            results.append(
                {
                    "episode_id": episode_id,
                    "director_markdown_path": str(director_md_path),
                    "director_json_path": str(director_json_path) if director_json_path else None,
                    "storyboard_markdown_path": str(storyboard_md_path),
                    "storyboard_json_path": str(storyboard_json_path),
                    "prompt_entries": len(final_result.get("prompt_entries", [])),
                    "used_refs": len(used_ref_ids(final_result)),
                    "review_pass_enabled": enable_review_pass,
                    "storyboard_profile": storyboard_profile,
                    "generated_at": utc_timestamp(),
                }
            )
            print_status(
                f"{episode_id} 完成：提示词 {results[-1]['prompt_entries']} 条，使用素材 {results[-1]['used_refs']} 个。"
            )
        except Exception as exc:
            episode_telemetry.context["final_status"] = "failed"
            episode_telemetry.context["error"] = str(exc).strip()
            raise
        finally:
            if write_storyboard_metrics:
                metrics_report = save_storyboard_metrics(episode_telemetry, metrics_json_path, metrics_md_path)
                print_status(
                    f"{episode_id} storyboard 统计：耗时 {metrics_report['totals']['duration_seconds']}s | "
                    f"tokens in/out/total = {metrics_report['totals']['input_tokens']}/{metrics_report['totals']['output_tokens']}/{metrics_report['totals']['total_tokens']}"
                )
                print_status(f"{episode_id} storyboard 统计报告：{metrics_json_path}")
            merge_telemetry_recorders(telemetry, episode_telemetry)

    if dry_run:
        preview = {
            "series_name": series_name,
            "model": model,
            "assets_dir": str(assets_dir),
            "episodes": previews,
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return preview

    summary = {
        "series_name": series_name,
        "model": model,
        "assets_dir": str(assets_dir),
        "results": results,
        "generated_at": utc_timestamp(),
    }
    summary_path = resolve_outputs_root(config) / resolve_output_series_name(config, series_name) / "seedance-prompts-summary.json"
    save_json_file(summary_path, summary)
    print_status(f"Seedance 提示词链路完成：{summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 02-seedance-prompts.md from director analysis and assets.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print_status(f"加载配置：{args.config}")
    config = load_runtime_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
