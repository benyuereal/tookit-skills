# dcu-modelzoo

光合开发者社区 ModelZoo（GitLab group `modelzoo`，DCU 模型适配仓库集）查询 skill。

- SKILL.md：
  - GitLab API v4 匿名访问（列全量仓库/搜模型/model.properties 元数据/README raw）
  - README 标准模板关键位：环境依赖 DTK 表（适配基线）、推荐镜像、支持卡型+最低卡数、推理启动命令
  - **旧项目上新设备警示**：DTK≤25.04 老项目（README 基线可查）在 26.04 新设备生态上需重新适配/验证
  - 两个镜像域名并存的老/新差异（image.sourcefind.cn:5000 vs harbor.sourcefind.cn:5443）
  - `scripts/fetch_modelzoo.py` 全量快照刷新脚本（622 仓库 + properties + README 提取，--with-readme）
  - `data/modelzoo_index.json` 快照（fetched_at 2026-09-02，含每仓库 frameType/accelerateType/DTK hints）

选型链路：本 skill 查模型适配基线 → `dcu-image-hub` 查镜像 tag/拉取命令 → `dcu-docker` 起容器 → `dcu-das-hub-api` 补 pip 包。

> 安装：本 skill 随 dcu-toolkit 包一起放在 <agent-home>/skills/dcu/ 即可，无需额外步骤。
