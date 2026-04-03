# 题材补充 Skill 扁平视图

这个目录现在主要保留给读者快速浏览“每个题材大概有哪些补充 skill”，不再是推荐维护入口。

文件命名说明：

- 现在这里也统一改成中文文件名，例如：
  - `先婚后爱.md`
  - `宅斗权谋.md`
  - `重生.md`

当前推荐维护入口：

- `skills/production/video-script-reconstruction-skill/genres/<题材>/skill.md`

当前真实加载逻辑：

- 代码会优先从 `genres/<题材>/skill.md` 加载题材补充 skill
- 如果你后续要新增或修改题材，请优先改 `genres/` 目录

保留这个目录的原因：

- 扁平浏览时比较快
- 方便读者快速对比不同题材 skill 的风格
- 兼容旧认知，不至于一下子找不到老路径
