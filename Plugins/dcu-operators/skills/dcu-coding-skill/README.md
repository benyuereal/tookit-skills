# DCU Coding Skill

> 本包随 dcu-toolkit-skills 仓库整包分发（operators/dcu-coding-skill）；2026-10-19 自 89 live 包 v1.2.1 反向收编，五个子 skill 与 common/ 全量在库。

面向 `Claude Code`、`Cursor`、`Codex`、`OpenClaw` 的 DCU/HIP 编程多 Skill 系统。  
项目目标是把“API 查询、代码迁移、测试生成、Triton 开发”拆分为可独立安装、可组合协同的能力，并通过分片知识库提供统一检索增强。

---

## 核心能力

- `dcu-dispatcher`：统一入口，负责意图分类、路由分发和 workflow 编排（不持有知识分片）
- `dcu-hip-reference`：HIP runtime / API 查询与示例
- `dcu-code-migrator`：CUDA -> HIP 迁移与报告
- `dcu-ci-testcase`：测试用例与测试套件生成
- `dcu-triton-kernel`：Triton kernel 生成、优化与模板复用

知识检索采用“每个 core Skill 分片 + 运行时聚合”模式：

- 每个 core Skill 维护自己的 `kb_shard` 或 `kb_shard_ext`
- 运行时由 `common/kb` 聚合器进行跨分片检索与排序

---

## 项目结构

```text
dcu-coding-skill/
├─ skills/                         # 5 个 Skill（1 entry + 4 core）
├─ common/                         # kb/spi/intent/utils 共享层
├─ docs/
│  ├─ raw_docs/                    # 原始文档平铺目录 + 同步/路由 manifest
│  └─ specs/                       # 设计与开发文档
├─ scripts/
│  ├─ sync_raw_docs_and_build_shards.py
│  ├─ classify_docs.py
│  ├─ build_shard.py
│  ├─ validate_shards.py
│  └─ validate_skills.py
├─ packaging/
│  ├─ release_builder.py
│  └─ README.md
└─ marketplace.json                # Skill 元数据与依赖关系
```

---

## 快速开始

### 1) 校验项目健康度

```bash
python scripts/validate_skills.py
python scripts/validate_shards.py
```

### 2) 同步线上文档并重建知识分片

```bash
python scripts/sync_raw_docs_and_build_shards.py
```

该命令会自动执行：

1. 从线上知识仓库拉取文档
2. 平铺写入 `docs/raw_docs`
3. 生成 `_routing_manifest.json`
4. 构建各 core Skill 分片
5. 校验分片结构

### 3) 构建发布包

见下文 **[发布打包命令](#发布打包命令)**（整包、单包、版本号与组合示例）。

---

## 发布打包命令

入口脚本：`packaging/release_builder.py`。  
**Release 行为固定**：每次打包前会自动执行文档同步、分类、`kb_shard` / `kb_shard_ext` 构建与校验（不提供跳过开关）；需能访问配置中的知识仓库（及可选鉴权环境变量）。

产物默认输出到 `releases/`：全量包 `dcu-coding-skill-<version>.tar.gz`、单 skill 包 `<skill>-<version>.tar.gz`、`release-index.json`。根目录 `VERSION` 文件为当前语义化版本号。

在项目根目录执行（Windows / Linux 均可，`python` 指向同一环境）：

```bash
# 全量包 + 所有默认可单发的子 skill 包（standalone；不含 dcu-dispatcher 单包）
python packaging/release_builder.py

# 仅全量包
python packaging/release_builder.py --skip-single-skill

# 仅指定单 skill 包（可多个 skill 名，空格分隔）
python packaging/release_builder.py --skip-full --skills dcu-code-migrator
python packaging/release_builder.py --skip-full --skills dcu-code-migrator dcu-hip-reference

# 若只写 --skills 且不打 --skip-full：全量包仍含 marketplace 中全部 skill；
#   --skills 会传给同步脚本，用于缩小「文档同步后的分片重建」范围；单 skill 包只打列表内且允许单发的项。

# 打包前清空 releases/ 下旧 dcu-coding-skill-*.tar.gz 与 release-index.json
python packaging/release_builder.py --clean

# 语义化版本：先改写根目录 VERSION 再按新版本打所选包
#   patch：x.y.(z+1)   例 1.0.1 -> 1.0.2
#   minor：x.(y+1).0  例 1.0.2 -> 1.1.0
#   major：(x+1).0.0  例 1.1.0 -> 2.0.0
python packaging/release_builder.py --bump patch
python packaging/release_builder.py --bump minor
python packaging/release_builder.py --bump major

# 常见组合
python packaging/release_builder.py --clean --bump patch
python packaging/release_builder.py --clean --skip-single-skill --bump minor
python packaging/release_builder.py --skip-full --skills dcu-code-migrator --bump patch
```

**单 skill 包约定**：默认只为 `standalone=true` 的业务型 skill 出单包；`dcu-dispatcher` 默认跳过。若确需为其出单包：

```bash
python packaging/release_builder.py --skip-full --allow-non-standalone-single-skill --skills dcu-dispatcher
```

更多参数（如 `--project-root`、`--release-dir`、`--include-skill-assets`）见 `packaging/README.md`。

---

## 常用命令

```bash
# 仅重建分片（不拉取线上文档）
python scripts/sync_raw_docs_and_build_shards.py --skip-sync

# 全量重建可发现分片 embeddings（整包）
python scripts/rebuild_all_embeddings.py
```

## 向量化开关（整包）

- 用户运行开关：`config.yaml -> knowledge_base.vectorization.enabled`
- 包能力开关：`package-manifest.json -> capabilities.vectorization`（由发版流程写入，用户不改）
- 生效逻辑：`effective = capabilities.vectorization AND knowledge_base.vectorization.enabled`
- 因此当用户把 `enabled=true` 但当前包能力为 false 时，查询会自动降级为关键词检索并提示 warning。

## 图谱开关（Graph RAG）

| 模式 | seed/ 目录 | graph_mvp.enabled | 说明 |
|------|------------|-------------------|------|
| 整包（`--skills all`） | ✅ 包含 | ✅ 自动构建并启用 | 整包安装时自动构建知识图谱并启用 Graph RAG |
| 子 skill 包 | ❌ 排除 | ❌ 不添加 | 子 skill 独立部署时不支持图谱功能 |

- **打包脚本**：`scripts/install_skills.py`
- **构建时机**：整包安装时自动执行 `_build_graph_for_bundle()`，在目标目录内联构建图谱表
- **源文件保护**：子 skill 包不包含 `seed/` 目录，源文件 `shard.manifest.yaml` 不含 `graph_mvp` 配置
- **启用效果**：Graph RAG 作为第三路检索（FTS + vector + graph）与现有 RRF 融合，提升 CUDA→HIP 迁移场景的召回与可解释性

---

## 知识库构建流程（当前实现）

`docs/raw_docs` 采用“平铺 + manifest”模式，不按 skill 目录分层：

- `_sync_manifest.json`：记录线上仓库文件到本地平铺文件的映射
- `_routing_manifest.json`：记录每个文档分配到哪些 Skill（含 `__public__` 桶）

构建链路如下：

1. `sync_raw_docs_and_build_shards.py` 拉取并平铺文档
2. `classify_docs.py` 根据 `scripts/kb_assignment_rules.yaml` 做文档归属
3. `build_shard.py` 将文档按“标题层级 + token 驱动 + 代码块安全”规则切块并写入各 Skill 分片 SQLite
4. `validate_shards.py` 做分片一致性校验

其中 `build_shard.py` 的默认分块参数为：`max_tokens=1500`、`min_tokens=80`、`overlap_tokens=200`；对 fenced code block（```...```）采用原子保护，避免代码示例在分块时被切坏。

---

## 发布说明

命令与版本号策略见上文 **[发布打包命令](#发布打包命令)**。

- 安装形态固定为两种：**全量安装**（整包作为统一入口，由 `dcu-dispatcher` 路由分发）与**子 skill 独立安装**。
- 全量包内含各平台安装脚本，可用安装器自动化安装。
- 单 skill 包为轻量分发，解压后手动复制到目标环境 `skills/<skill_name>/`。
- 支持平台：`claude-code`、`cursor`、`codex`、`openclaw`。

### 全量包一键安装（按智能体）

先解压全量包，在包根目录执行：

```bash
tar -xzf dcu-coding-skill-<version>.tar.gz
cd dcu-coding-skill-<version>
```

- `python install-claude-code.py`
- `python install-cursor.py`
- `python install-codex.py`
- `python install-openclaw.py`

默认安装根为当前工作目录。  
如需指定安装根，可在对应脚本后追加参数：

```bash
python install-cursor.py --target-root /your/agent/root
```

```powershell
python .\install-cursor.py --target-root C:\Users\zhang
```

其余约定与可选参数见 `packaging/README.md`。

---

## 关键文档

- 开发者指南：`docs/specs/developer-guide.md`
- 架构设计：`docs/specs/dcu-coding-skill-design-v4.md`
- 用户指南：`docs/specs/user-guide.md`
- 发布说明：`packaging/README.md`
