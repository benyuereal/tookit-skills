# dcu-das-hub-api

DCU 包源（pypi.sourcefind.cn，DAS Hub）JSON API 用法与踩坑 skill。

- SKILL.md：
  - 关键事实：根域名是 Vue SPA 空壳，机器化查包走 JSON API；所有 API 调用必须带 `X-Das-Source: mirror-web-client-v1` 头
  - 三个 API 端点（搜包 / 列仓库 / 列某仓库某包全部版本）+ repo 在前的 URL 顺序坑
  - 下载 wheel：响应里的内网 URL 改写为公网域名下载 + sha256 校验；容器无外网时本地下载再 scp
  - 跨仓库枚举找"比容器更新的构建"（customer/*、das_nightly/*、das_release/*、release/*）
  - torch ABI 不匹配（`undefined symbol: _ZN3c10...`）诊断：kernel 轮的 torch 标记必须与环境 torch 大版本一致
  - `.pth` monkey-patch 有 bug 的预编译 kernel（env 开关、可回滚、容器重建会丢）

实测来源：2026-08-27 反向工程 DAS Hub SPA 的 JSON API + 下载 URL 内网→公网改写（两个 wheel 下载 + sha256 全匹配）。

与 `dcu-env-packages` 的关系：那个讲 `/release/dtk/` pip 索引（直接 `pip install -i`）+ 适配包清单；本 skill 讲 DAS Hub 的 JSON API（枚举 / 跨仓库查 / 下载 wheel），两者互补。

> 安装：本 skill 随 dcu-toolkit 包一起放在 <agent-home>/skills/dcu/ 即可，无需额外步骤。
> 密码/凭证不进 skill 包。
