# dcu-image-hub

DCU 镜像仓库（sourcefind.cn 光合开发者社区）查询与拉取 skill。

- SKILL.md：
  - Web API 三端点（ListImage 列镜像 / Keyword 搜索 / ListTag 列版本 tag），匿名免请求头，直连 `developer.sourcefind.cn`
  - 分页参数大小写坑（`Page`/`PageSize` 大写有效，小写被忽略恒返回 10 条）
  - Harbor Registry 匿名 token 通道（tags/list + manifest 直查，`docker pull` 无需 login）
  - tag 命名规律速查（框架版本-OS-DTK-das-py 版本段怎么读）
  - `scripts/fetch_sourcefind_images.py` 一键刷新全量快照（63 镜像/1234 tag @2026-10-14，约 1min）
  - `data/sourcefind_dcu_images.json` 快照（fetched_at 2026-10-14）

与相邻 skill 的关系：`dcu-das-hub-api` 管 pip 包源（pypi.sourcefind.cn，wheel）；本 skill 管 docker 镜像仓库（harbor.sourcefind.cn:5443）；`dcu-docker` 管拉下来后怎么起容器。

> 安装：本 skill 随 dcu-toolkit 包一起放在 <agent-home>/skills/dcu/ 即可，无需额外步骤。
