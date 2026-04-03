请体检下面这个题材包，并输出结构化审核结果。

题材：{{genre_key}}
最多 findings：{{max_findings}}
每类编辑建议最多：{{max_suggestions_per_type}}

当前 playbook.json：
{{playbook_json}}

当前 skill.md：
{{skill_text}}

程序预检结果（包含字段统计、包内重复、跨题材高重合、可疑条目）：
{{precheck_report_json}}

请重点回答：
1. 哪些条目明显重复、近重复或职责重叠？
2. 哪些条目不够像“{{genre_key}}”题材，而更像其他题材、某部具体剧或某次误学到的经验？
3. 哪些条目太空、太泛、太碎、太像口号，不适合保留在正式题材包里？
4. `playbook.json` 和 `skill.md` 的分工是否清楚？哪些内容更适合移动位置？
5. 哪些关键题材能力目前缺失，值得补一条高质量规则？

请务必让编辑建议足够具体，便于人工二次判断和后续修订。
