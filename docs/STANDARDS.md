# DCU Toolkit 开源版 — Skills 开发规范

> 命名、结构、分类、脱敏红线。贡献新 skill 前必读。

## 命名

- Skill 目录用 `{domain}-{name}` kebab-case（如 `dcu-vllm-deploy`）
- `SKILL.md` frontmatter 必须含 `name`（与目录名一致）、`description`（触发条件 + 不适用反例）、`version`、`license: MIT`

### 前缀例外清单

`dcu-` / `op-` 前缀表示**本平台特有**的技能。以下技能是**通用工程能力**
（上游框架的 CI、重构、profiling、部署等），加 `dcu-` 前缀会给它们
虚假的平台归属，因此**允许不带前缀**，登记在 `skill_lint.py:PREFIX_EXEMPT`：

| 类别 | skill |
|---|---|
| 工程规范与重构 | `ci-workflow-guide`、`clean-startup-log`、`env-var-conventions`、`large-class-init-style`、`mechanical-refactor-verify`、`scripted-runtime-notes` |
| cookbook | `cookbook-add-model`、`cookbook-migrate-model`、`cookbook-review-pr` |
| 上游框架操作 | `sglang-bisect-ci-regression`、`sglang-cherrypick`、`sglang-diffusion-add-model`、`sglang-humanize-review`、`sglang-model-day0-support`、`sglang-prod-incident-triage`、`vllm-pr-summary` |
| 内核接入 | `add-jit-kernel`、`add-sgl-kernel`、`generate-profile` |
| profiling / 分析 | `humanize-gen-plan`、`humanize-rlcr`、`llm-pipeline-analysis`、`llm-serving-auto-benchmark`、`llm-serving-capacity-planner`、`llm-torch-profiler-analysis`、`model-architecture-diagram`、`model-compute-simulation`、`model-pr-optimization-history`、`model-pr-history-knowledge`、`torch-profiler-layer-track` |
| 调试 | `cross-engine-acceptance-gate`、`debug-cuda-crash`、`debug-distributed-hang` |

**新增例外要改两处**：本表 + `skill_lint.py:PREFIX_EXEMPT`。判据只有一条——
「这个技能换成非 DCU 平台是否同样成立？成立就不加 `dcu-`」。

## 结构

```
Plugins/<plugin>/skills/<skill>/
├── SKILL.md          # 主入口：触发条件、流程、路由（正文控制在几百行内，细节下沉）
├── scripts/          # 可执行工具（python3/bash，相对路径自定位）
└── references/       # 领域知识文档（SKILL.md 里建速查表暴露）
```

- 渐进式披露：SKILL.md 只放路由与核心流程，深知识放 references/
- 共享数据放包根 `data/ references/ scripts/`，skill 内用相对路径引用；写明双态路径（仓库开发态在包根，安装态在 `<skills根>/_dcu-toolkit-shared/`，由 install.sh 同步）
- 沉淀落点是**用户知识库**（不是 skill 树）：`~/.config/dcu-toolkit/kb/`（`op-opt/` 算子优化台账、`issue/` 故障记录、`solution/` 部署方案卡片；目录与写入约定见 `data/kb/README.md`）。写入纪律：每条带 source + verified_on（日期/设备/gfx/DTK）、未验证标 unverified、性能数字标 [历史数据] 且不进可复用事实条目。

## 五分类

知识库类（被动调用、只读无副作用）/ 工程模板 / 调试与测试 / 测试开发 / 工具辅助。

## 信息来源红线（开源版强制）

1. **零内部溯源**：不出现内部文档平台链接/文件 ID、内部人名/组织/工单号
2. **零内网地址**：IP 用 `<DCU_NODE_IP>` 等占位符；域名用 `<DCU_CODE_HOST>` `<DCU_REGISTRY_HOST>` 等
3. **零真实凭证**：凭证只允许占位模板；设备清单（IP/机型映射）不进包
4. **上游代码合规**：不内嵌第三方项目源码树（如 vLLM/RCCL/SGLang 的 source-tree）；方法论与 diff 片段需注明上游 license
5. **发布前必跑**：`python3 sanitize.py <repo>` 后 grep 断言 `内部实测记录|<DCU_CODE_HOST>|<ORG>|scnet|内网IP` 零残留

## 版本与文档一致性

- 改动 skill 后：更新该 skill frontmatter `version` + 包根 `CHANGELOG.md`（SemVer）
- 新增/删除 skill：同步更新包根 `SKILL.md` 路由表、`README.md` 插件计数、`marketplace.json`
- 路由表「意图 → skill」必须与 skill 自身 description 声明的能力边界一致

## 评测（改 description / 加 skill 必看）

- 用例放 `<skill>/evals/evals.json`：3 条 true（带 `expect` 要点）+ 2 条 false（形似、应路由他处）。
  完整用法、判分口径与失败修法见 [`docs/EVALS.md`](EVALS.md)。
- `description` 的「触发条件 + 不适用反例」不再只靠人审：`scripts/run_evals.py` 会逐条校验，
  **带 evals 的 skill 缺任一面直接 FAIL**；无 evals 的只统计，`--strict-description` 可全库阻塞。
- 路由问题优先改 `description`（触发词 + 边界），内容问题才改 `SKILL.md` 正文；两条评测线别混。
