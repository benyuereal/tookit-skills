# 贡献指南（open-source 分支）

## 脱敏规程（强制）

本分支对外发布。从内部 master 同步或 cherry-pick 内容到本分支时，必须遵循：

1. **内容与提交信息双重脱敏**：用 `git cherry-pick -n` 拿内容（不保留原提交），跑 `python sanitize.py --fix .`，再手写脱敏后的提交信息。原始提交信息可能含内部主机/用户名，禁止原样带入。
2. **提交前跑门禁**：`python sanitize.py .` 退出码非零即有命中，逐条处理到 0 命中才可提交；提交信息用 `python sanitize.py --msg-file <file>` 预检（CI 的 sanitize-gate 作业对两者都检查）。
3. **占位符约定**：内部 GitLab 主机 → `<internal-gitlab>`；内部组/仓库路径 → `<internal-group>` / `<internal-gitlab>/dcu-knowledge/`；内网 IP → `<internal-ip>`；内网用户名 → `<internal-user>`；内部邮箱 → `<internal-email>`；在线文档指纹 → `<internal-file-id>`；设备号 BDF → `<device-BDF>`；agent 运行时路径 → `<agent-home>`；工单号 → `<internal-ticket>`。
4. **白名单（不算泄露）**：`developer.sourcefind.cn`（公开仓库/文档中心）/ `harbor.sourcefind.cn`（公开镜像仓库与镜像名）/ `pypi.sourcefind.cn`（公开 DCU 包源）/ `download.sourcefind.cn`（z-file 分发站，`/file/1/<名>` 直链分发官方 DTK/驱动等，2026-08-06 已验证可写死）；上游公开 PR 语料中的代码路径（如 hermes_tool_parser）；公开模型名（如 hermes-4.3_vllm）。
5. **二进制文件**：.gz/.sqlite/.pyc/.db/.whl 纳入扫描（.gz 解压、.sqlite 抽字符串）；二进制一律提交原字节，禁止行尾转换（.gitattributes 已标 binary）。
6. **新泄露形态**：先在 sanitize.py 增补规则（敏感值以 b64 存于规则表，规则文件自身不得含明文敏感值），`--selftest` 补样本，全仓 --fix 后再提交。
7. **不要改动占位符形态**：`<...>` 内是约定的机读占位符，下游安装脚本按此解析。

CI：`.gitlab-ci.yml` 的 `sanitize-gate` 作业在 open-source 分支每次 push 自动执行同一门禁。
