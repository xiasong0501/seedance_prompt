from __future__ import annotations

from typing import Any


GENERALIZED_LOGIC_REVIEW_RULES: list[dict[str, Any]] = [
    {
        "name": "空间安全区与打击落点",
        "focus": "大范围攻击、雷击、爆炸、群体冲击后，必须能解释关键人物、关键道具、关键场位为何仍能继续存在或继续说话。",
        "checks": [
            "若后续仍有对话或递物动作，前文不能把对应位置写成无差别必杀区。",
            "需要交代打击落点、避让区域、掩蔽条件、护罩、遮挡或刻意留空中的至少一种。",
            "若关键人未被直接命中，提示词里应能看出其安全区来源，而不是仅靠默认幸存。",
        ],
    },
    {
        "name": "动作前提与手部占用",
        "focus": "人物在同一拍内的动作必须满足肢体占用与姿态前提，不能一只手同时完成互斥动作。",
        "checks": [
            "持丹、持兵器、扶伤、控物、开门、出手等动作不能在同一只手上无过渡并行发生。",
            "若人物既要持物又要攻击，必须补足“换手、收回、藏在另一只手、往身侧一收、松手”等动作桥。",
            "站位、朝向、距离若影响动作成立，应在提示词中留出足够前提。",
        ],
    },
    {
        "name": "特效源头传播反馈闭环",
        "focus": "特效、法术、机关、冲击效果必须具备起点、传播路径与结果反馈，避免只写空壳奇观。",
        "checks": [
            "高能特效至少要能看出从哪里发出、往哪里扩散、打到了什么空间区域。",
            "环境反馈与人物反馈至少要有其一，如石屑、烟尘、旗幡、护罩、后退、失衡、遮挡、压制等。",
            "若强调大威力，也要说明未波及的安全区或被规避的区域。",
        ],
    },
    {
        "name": "状态连续与道具守恒",
        "focus": "人物伤态、站姿、道具持有关系、场景状态要能在相邻分镜间连续追踪。",
        "checks": [
            "同一角色的伤态、持物、坐立、朝向不能无解释跳变。",
            "同一道具的所在位置、持有人、是否还在场，不能前后矛盾。",
            "若某个小物件在大战后仍被强调存在，需要给出遮挡、掩蔽、留空或未被正面波及的理由。",
        ],
    },
    {
        "name": "镜头叙事的可视化因果",
        "focus": "镜头语言要能让观众顺着画面理解因果，不靠脑补补洞。",
        "checks": [
            "镜头切换前后，主体、空间锚点、权力关系和视线关系要尽量可追踪。",
            "悬念动作如枪口偏转、队列倒向、视线回转，需要先交代原始方向或原始状态。",
            "不要只写结果；若结果会引起疑问，需要补一个足够短但关键的中间桥。",
        ],
    },
    {
        "name": "人物与引用绑定清晰",
        "focus": "角色、reference image、对白主体与动作主体必须清晰对齐，避免观众认错人或模型取错脸。",
        "checks": [
            "核心动作、核心对白附近应出现对应人物名称或明确绑定的 reference。",
            "同条中若涉及多名同类角色，尽量减少悬空的“他/她/对方”指代。",
            "若某人物是主说话人或主动作人，但缺少对应角色 ref，需要优先补正。",
        ],
    },
    {
        "name": "世界观混搭时的可接受性",
        "focus": "现代物件、玄幻机关、写实动作、奇观效果混搭时，要让观众能接受它为何出现在同一世界。",
        "checks": [
            "混搭元素出现时，要通过反应、站位、材质、口令、规则差异或环境反馈说明它对世界的冲击。",
            "不要让混搭元素像凭空跳出的道具展示，应与当前空间和人物目标发生明确关系。",
            "若混搭元素持续作用多条分镜，后续的压制线、避让线、旁观者反应也要保持一致。",
        ],
    },
]


def render_generalized_logic_rules_markdown() -> str:
    lines = ["## 审稿规则", ""]
    for index, rule in enumerate(GENERALIZED_LOGIC_REVIEW_RULES, start=1):
        lines.append(f"{index}. {rule['name']}：{rule['focus']}")
        for item in list(rule.get("checks") or []):
            lines.append(f"   - {item}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_generalized_logic_rules_prompt() -> str:
    lines: list[str] = []
    for index, rule in enumerate(GENERALIZED_LOGIC_REVIEW_RULES, start=1):
        lines.append(f"{index}. {rule['name']}")
        lines.append(f"   审查目标：{rule['focus']}")
        for item in list(rule.get("checks") or []):
            lines.append(f"   - {item}")
    return "\n".join(lines).strip()
