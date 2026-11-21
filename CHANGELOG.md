# Changelog

所有重要变更记录于此文件。格式遵循 Keep a Changelog，版本号遵循 SemVer。

## [Unreleased]

### Added
- **`dcu-hytlass-source`（operators 域，第 20 个子 skill）**：hytlass 是 CUTLASS 3.5.0（commit `7d49e6c7`）的
  HCU 移植，C++ 模板库，BSD-3-Clause。覆盖两套编程模型（2.x threadblock/warp + mmac/`ds_read_matrix`；3.x CuTe，
  仓内称 HuTe）、11 个代表性示例的选型、构建门槛（DTK 25.10+/CMake 3.19+/C++17）、与手写 HIP 的分界（示例是
  best practice 不保证最优）、已知边界（上游固定 commit、无 autotune/plan 层）。
- **`op-stack-design/references/libraries.md` 扩到十一库**：新增 hytlass / `Moe`(moe_c) / `BoltOPs` / `TileKernels`
  四个画像；每个库由三行改为五行（加「何时选它 / 何时别选」）。AITER 画像补「moe_c 是独立仓库，MoE 慢先查
  `moe_c_configs/`」；DeepGEMM 画像补 HIPC（`m_..._hipc` + `pack_w4a8_moe_hipc_weight`）与 gfx936 专用 W6
  （`m_..._gfx936` + `pack_w8a8_gfx936_weight_to_w6`）、`shuffle_unique`、`fp8_gemm`/`einsum` 走 hipBLASLt；
  LightOp 画像补「仓内置 `.agents/skills/`，带宽测量用它的脚本不要另写探针」。跨库共同税由三条扩到四条
  （新增「融合的依赖链」：融合未必更快，mHC 的 `pre_big_fuse` 实测慢 1.7×）。
- **五个 `dcu-*-source` 补「何时选它 / 何时别选」小节**：`dcu-aiter-source`、`dcu-triton-source`、`dcu-kernelwiki`、
  `dcu-lightop-fusion`、`dcu-deepgemm-source`（后者同时更新 §3 的 `deepgemm-das` 开源锚点与 `csrc/arch/*` 目录
  布局、§4 路径表补 HIPC / gfx936-W6 / `shuffle_unique` / hipBLASLt 入口）。
- **`op-stack-design`（ops 域选型层 skill）接入路由并登记**：库画像（结构定律 / 可调面 / 已知遗留税 /
  何时选它）+ 十条流水线定律，用于「写代码前排掉结构性不可行的设计」。包根 `SKILL.md` 路由表补入该
  skill 并指向唯一权威表；`dcu-skill-router` §8 算子卡域索引同步补齐（连同此前漏登记的
  `op-hip-compile`），并把过时的「三闸四档」更正为 `op-gate` 的「四闸四档」。
- **`Plugins/ops/SKILL.md` §5.1「唯一权威表」**：同一件事在四个 skill（`op-gate` / `op-optimize` /
  `dcu-op-optimization` / `humanize-rlcr`）里各讲一遍，agent 进来不知道信哪份。新增一张表逐项写明
  「问题 → 唯一权威 → 其他位置的定位」：判定→`op-gate`、量法→`op-bench`、天花板→`op-roofline`、
  算子身份→卡片+registry、候选队列→`op-optimize/loop.py`、PMC→`op-profiling`、选型→`op-stack-design`、
  多轮编排→`dcu-orchestrator`。原索引顺延为 §5.2。
- **`dcu-op-optimization` 顶部加「定位声明」**：明确它是方法学导读、不定义判定标准；判定/口径/天花板
  分别归 `op-gate` / `op-bench` / `op-roofline`；文末 `candidates.jsonl` 标为历史遗留，现行格式是
  `op-optimize` 的 `candidates.yaml`。原有内容（微基准 roofline 五步、踩坑、配方）保留不动。
- 规模数字更正：`SKILL.md`、`README.md`、`docs/EVALS.md`、`HANDOFF.md` 的 144 → **146**（141 顶层 + 5 嵌套）。
- **`scripts/skill_eval.py`（evals 动态执行器）+ `docs/EVALS.md`**：把 `evals/evals.json` 真的跑起来——
  每条用例在独立临时 cwd 起 `claude -p`，**挂仓内全部插件**（路由本来就发生在全库之间），并隔离
  `CLAUDE_CONFIG_DIR` 让 `~/.claude/skills` 里的已安装旧副本不参与竞争（不隔离时测的是上次 install 的描述，
  改工作区 `SKILL.md` 对结果没有影响）；`--judge` 可再加一层内容评分，`--no-isolate` 用于复现用户现场。
  默认只判路由（最客观、最便宜），`--list` / `--skill` / `--case`（只重跑失败项）/ `--domain` / `--jobs` / `--json` / `--dry-run` 覆盖日常用法。
  不进 CI（烧 token），命令、读法与失败修法见 `docs/EVALS.md`。
- **ops 五个方法论 skill 补 evals（各 3 true + 2 false）**：`op-contract` / `op-bench` / `op-gate` /
  `op-optimize` / `op-profiling`。这五个边界最近（「怎么量」「怎么判」「怎么持续优化」互相抢活），
  false 用例取同域最近邻，正好压测路由。
- **`Plugins/ops/tests/test_eval_tooling.py`**：评测工具链自测（不调模型）——stream-json 解析、路由判分、
  带 evals 的 skill 必须能向上找到插件根（缺 `plugin.json` 时 `--plugin-dir` 挂不上，表现是全部 0 命中）、
  description 静态门。
- **`scripts/opcheck.py`（算子卡共用门模板）**：`correctness` / `invariants` / `padding` 三个子命令，
  流程、报告字段、退出码只有一份；判据全读卡片 YAML（`tolerance`、`invariants[].mode`、
  `variants.yaml:padding_support/mutates_inputs`），卡目录只放一个 adapter（默认 `_common.py`）写本算子的
  数学（`build_inputs` / `reference` / `runner_kwargs` / `get_runner` / `run_isolated` + 可选探针）。
  `contract.yaml:gates` 新增 `template:` 字段，`opverify.py` 按 `GATE_TEMPLATES` 校验，未知模板即报错。
  目的：算子卡沉淀方法，不再每开发一个算子就多传一套近似重复的测试脚本。
- **`op-hip-compile`（ops 域方法论 skill）**：手写 HIP 的编译调优 + 两个可复现工具。
  `derive_flags.py` 直接调用 tilelang `get_hcu_compile_flags(arch, pass_configs)` 推导 aicc 命令行
  （含 `TL_ENABLE_FAST_MATH` / `TL_ENABLE_HCU_WDRA` 条件项），aicc 不在 PATH 时以退出码 3 拦「静默空集」；
  `inspect_so.py` 从 `.so` 内嵌设备 ELF 的 AMDGPU metadata 读每 kernel 的 VGPR/SGPR/spill/LDS，
  按 VGPR 算 CTA/CU（本机 GLM 档复现出 VGPR=252 ⇒ 1 CTA/CU）。
- `op-sparse-mla` 新增 `decode-v41` 档位（DeepSeek V4.1 Flash：H=64 / d_qk=576=512+64 / topk=512），
  状态 draft、**暂无实现**；`data/ops/registry.json` 同步。`sm_scale` 口径待核定（本卡 576 宽档约定
  `1/sqrt(640)`，DeepSeek 常规为 `1/sqrt(576)`，差 5.4%）。
- **`Plugins/ops/tests/test_doc_links.py`（文档真值门）**：把「文档里写的路径必须真的存在」变成 CI 断言。
  四类检查——① 各卡 `SKILL.md` 与 `Plugins/ops/SKILL.md` 里的路径 token 可解析（仓外源码按前缀白名单、
  示意路径逐条写理由）；② `contract.yaml:gates` 的 `template:` 必须是 `opcheck.py` 子命令、`script:`
  必须存在（除非 `required: false` + `skip_reason`）；③ `registry.json` ↔ 卡目录 ↔ 域索引三方一致，
  且**索引里的状态列必须等于 registry 的 `status`**；④ 用户知识库指针必须写全
  `~/.config/dcu-toolkit/kb/…`。纯静态、无 GPU 依赖，随 `.gitlab-ci.yml` 的 ops 测试一起跑。

### Added
- **`forbid`：路由对之后再判「有没有说错」**（零额外 token）。true 用例可加 `"forbid": ["错话", "re:正则"]`，
  `skill_eval.py` 在**已经抓到的回答文本**里找错误说法，命中判 `说错` 并归到内容问题。
  命中片段本身或前 24 字里有否定词（不/别/禁止/避免/杜绝…）则跳过——「不要用 `sshpass -p`」是正确说法。
  静态门校验：只能出现在 true 用例、`re:` 需可编译、纯文本条目 ≥4 字（写错话不写关键词）。
  已配两处真实禁令：`dcu-ssh-t1`（`sshpass -p <pw>`）、`op-profiling-t2`（`TA_* 取平均` /
  首次 launch 数据直接用）。上机验证过不误报：真实回答写的是「绝不能出现在 `sshpass -p` 里」「不是取平均」。
- **`docs/RELEASE-CHECKLIST.md`**：评测框架不背的那部分——静态门、`--repeat 3` 动态抽检、上机跑
  `opverify` / `opcheck` / `bench`、版本与留痕、装后抽检，逐条可勾。
- **`docs/EVALS.md` §六点五「能力四级阶梯」**：L0 路由 → L1 要点/错话 → L2 产物判定 → L3 上机。
  **L2 明确暂缓**，并写明重新评估的触发条件（第 3 个有可判定产物的 skill，或一次「产出物不能用」的事故）；
  L3 不进框架，归发版清单。

### Changed
- **`skill_eval.py` 加 `--repeat N` 与失败归类**：同一用例跑 N 次报 `k/N`（全过 PASS、部分过 FLAKY、
  全败 FAIL，`FLAKY` 不算通过），失败再分成 **未拉起**（模型没加载任何 skill——内容问题）/
  **拉错**（邻居抢走——描述问题）/ **误拉**（false 用例被触发——描述过宽），三类修法不同。
  `--json` 里每例保留 `runs[]` 原始尝试。首个用它下的结论：`op-optimize-t2` 连跑 3 次 **1/3**
  （两次 `未拉起`），确认是「模型认为通用知识够答」，不是描述没写清——留作内容侧待办。
- **路由评测第二轮：按实测结果修描述与用例**（把 `docs/EVALS.md` 的「抓手」规则真正落进用例）。
  描述侧：`humanize-rlcr` 1.0.1（不适用：单算子 loop 走 op-optimize）、`dcu-op-optimization` 1.2.3
  （方法学总纲；单算子循环→op-optimize、四闸判定→op-gate）、`dcu-hipprof` 1.0.1（只负责采集，读 PMC 归因→op-profiling）、
  `op-profiling` 1.0.3（首句改为「PMC 采回来怎么读」，采/算/判三分）、`dcu-ssh` 1.1.6（首句限定宿主机）。
  用例侧：true 用例的抓手从抽象问法换成 **skill 独有工具的字段名**——`pmc.py` 的 `bound[]/ratios[]/facts[]`、
  `opverify.py` 的 `gates`、`runs/loop/<日期>.jsonl`、`loop.py status`、`四闸`；`dcu-ssh-f1` 的靶子换成不含名字词的
  容器挂载题（`dcu-ssh` 是横切型 skill，见 `docs/EVALS.md` §二）。
  实测（2026-10-30，整批 `--jobs 5`）：`dcu-ssh` 5/5、ops 21/25；整批里 4 条飘的用例逐条复跑后有 3 条转绿
  （`op-gate-t2` / `op-optimize-t1` / `op-profiling-t2`），`op-optimize-t2`（靠 `loop.py status` 续跑）连续 3 次
  未拉起任何 skill——模型直接凭通用知识作答，留作待办（`--repeat 3` 复测见上一条）。
- **`scripts/run_evals.py` 增加 description 硬门**：按 `docs/STANDARDS.md` 校验「触发条件 + 不适用反例」，
  带 evals 的 skill 缺任一面即 FAIL（这些描述正是被动态评测判分的那段话）；无 evals 的只统计，
  `--strict-description` 可全库阻塞，作为逐步收敛的开关。
- **9 个 skill 的 description 按规范重写**（补触发场景 + 「不适用：…走 `<兄弟 skill>`」边界，各自 bump 版本）：
  `dcu-ssh` 1.1.3（容器间免密→`dcu-docker`、多机通测→`dcu-rccl-test`）、`dcu-docker` 1.2.2、
  `dcu-image-hub` 1.1.2、`dcu-modelzoo` 1.0.1、`dcu-model-deploy-catalog` 1.0.2、`dcu-vllm-deploy` 1.0.1、
  `dcu-ontrack-experience` 1.2.3、`dcu-orchestrator` 1.1.3、`dcu-skill-router` 2.2.2；ops 五篇方法论
  description 同步补「不适用」（1.0.1）。另外三个「最近的邻居」补了边界，避免抢活：
  `dcu-kernel-layer-microbench` 1.0.1（算子卡的 A/B 一律走 op-bench）、`dcu-perf-analysis` 1.0.1
  （单算子 PMC 归因走 op-profiling）、`op-roofline` 1.0.1（读实测计数器归因走 op-profiling）。
- `op-sparse-mla`：`bench/bench.py` 成为单入口，新增 `--mode single|workset|config|split`
  （默认 `single` 行为不变；`workset/config/split` 由原 `ablate.py` 并入，消融不再单列文件）。
  口径仍固定 op_call / hot / event 计时 / 3 轮中位数。
- `op-sparse-mla`：DCU 本机编译配方从 `SKILL.md` 正文移入 `dev/COMPILE.md`，卡面只保留两条可迁移原则
  （镜像对标实现的实际选项、怀疑 codegen 就读二进制）；选型表移除 H=16 测试档两行，其方法价值改在
  表下备注保留。
- `op-sparse-mla`：PMC 采集入口定为 **`bench/pmc_collect.py`**（按「采集计数器」而非「跑一遍」命名，
  与 `bench/pmc_table.py` 成 采集/汇总 一对），一条命令覆盖 `tl` / `fl` / `hip` 三个实现——`hip` 分支用
  ctypes 直调裸 `.so` 的 `hip_sparse_mla_fwd`，三者共用同一份输入构造与 3 次预热口径，天然可同表对照。
  docstring 写明三层分工（`hipprof` 采集 / `op-profiling/pmc.py` 为口径与解读的权威 / 本脚本只提供负载），
  不复述域级 PMC 文档；三处硬编码 `/workspace/...` 路径同时改为 `TILELANG_HYGON_ROOT` + 相对 skill 根解析。
- **`Plugins/ops/SKILL.md` 的目录约定从「固定七件套」改为「最小五件套 + 按需四件」**：最小卡 =
  `SKILL.md` + `contract.yaml` + `variants.yaml` + `shapes.yaml` + `_common.py`；`bench/` 只在需要消融时才写
  （常规计时走包根 `scripts/opbench.py`），`ledger.md` / `candidates.yaml` 进优化闭环后才有。
  新增卡 checklist 不再要求 `mkdir -p .../{bench,references}` 或无条件写 `bench/bench.py`；域索引加
  **状态列**（`tuned` / `draft` / `—`），先看这一列再决定进哪张卡，不必逐个翻 `registry.json`。

### Fixed
- **`run_evals.py` / `skill_eval.py` 把包根与插件根的 `SKILL.md` 当成 skill**：`iter_skill_dirs` 现在只认
  `<...>/skills/<name>/SKILL.md`，规模统计从 153 修正为 144（139 顶层 + 5 嵌套子 skill）；
  `README.md` 与包根 `SKILL.md` 里的「140 个 skill」同步更正（`docs/EVALS.md` 同）。
- **`Plugins/ops` 缺 `.claude-plugin/plugin.json`**：`.claude-plugin/marketplace.json` 早已声明 `ops` 插件，
  但目录里没有 plugin.json，`claude --plugin-dir Plugins/ops` 根本挂不上（动态评测表现为「所有用例 0 命中」）。
  补齐元数据，并在 `test_eval_tooling.py` 里加「带 evals 的 skill 必须能找到插件根」防回归。
- **ops 卡的 5 处死链与 1 处错误口径**（由 `test_doc_links.py` 拦下）：`op-profiling` 的
  `op-sparse-mla/bench/ablate.py`（已并入 `bench.py --mode workset|config`）、`op-hip-compile` 的
  `dev/sparse_mla_hip_glm.hip`（缺 `../op-sparse-mla/` 前缀）、`op-sparse-mla` 的 KB 指针
  `kb/op-opt/…`（必须写全 `~/.config/dcu-toolkit/kb/…`，否则模型会按包内相对路径找）、`op-gemm-fp8`
  引的 `bench/bench.py`、`op-moe-w8a8` 引的 `ledger.md`（两卡都还没有这两个文件）——后两处改为指向
  实际存在的判据（`contract.yaml` 的 `skip_reason`、`registry.json` 的 `open_questions`）。
  `op-sparse-mla/ledger.md` 里指向已删除 `tests/` 的证据列同时标注「随 3a96bc8 移除」。
- **`install.sh` 不再把本地构建物装进各 agent 的 skills 目录**：`sync_tree` 新增 `_skip_rel()`
  统一排除 `.so` / `.o` / `.a` / `.bak` / `runs/` / `__pycache__`（此前只挡后者）与 `evals/`
  （评测是开发侧资产，跑的是包根 `scripts/skill_eval.py`，装到 agent 目录里用不上）；
  升级时按「上游已移除」清掉早版本装过去的同类文件（本地改过的仍保留并告警），并清掉空的 `runs/` 残留目录。
  此前每跑一次重装都会把 `dev/*.so`（单个 68 KB，消融变体共 7 个）与实验目录 `runs/` 复制到每个 agent。
- `op-sparse-mla` 形状归属更正：`dev/sparse_mla_hip.hip` 的 H=16 来自 tilelang 示例默认值
  （`examples/dsa_sparse_finetune/sparse_mla_topk_reducesum.py:204`），不是任何生产形状。按 `/models`
  权重核对，两个稀疏 MLA 生产模型均为 H=64：GLM5.3-Flash（d_qk=512 / topk=2048）与
  DeepSeek V4.1 Flash / V4-Flash-0731（d_qk=576 / topk=512）。相应更正 kernel 头注释、
  `SKILL.md` 变体/档位/选型表、`shapes.yaml`（`decode-m1`/`decode-m32` 不再标「生产主力档」）、
  `ledger.md` 第 9 条与 §一 状态。

### Removed
- **ops 域按「不可替代性」清理中间过程脚本**（判据：删掉它之后，模型下次会不会写出一个看起来对、
  但结论相反的版本？不会就删）。卡内测试全部移出：`op-sparse-mla/tests/`（含 `test_pool_caliber.py`）、
  `op-roofline/tests/`（4 个）、`op-profiling/tests/`（含 `data/*.csv`）——它们的口径断言已在
  `_common.build_inputs` 的运行时 warning、`pmc.py` / `tools/*.py` 的 docstring、`rules.md` 与
  `models/` 注释里；`bench/pmc_data/`（4 个 CSV）是单机历史数据，按 AGENTS.md 不进 skill 树。
  词表改成「一张卡不带 tests/」：门的流程只有一份（`scripts/opcheck.py`）。
- `op-sparse-mla/bench/` 的变体生成链 `variants.py`（15 个变体的文本变换生成器）+ `ab.py` +
  `acc_probe.py`：三者的结论已在 `candidates.yaml:refuted`（37 条，含机制），生成器本身按需现写。
  `loop.py run` 随之对「没有 `bench/ab.py` 适配器」的卡明确拒绝（fail-closed），`status`/`queue` 不变。
- `op-sparse-mla/dev/{test_hip_glm,bench_glm_320k}.py`：已被 `opcheck.py correctness --variants hip-glm`
  与 `bench_long.py --variants hip-glm,...` 等价覆盖（`_common.get_runner` 早支持该变体）。
- `op-sparse-mla/bench/dump_src.py`：读生成代码的四条判读移入 `op-hip-compile/SKILL.md §四`。
- `op-sparse-mla/tests/` 的 `test_correctness.py` / `test_invariants.py` / `test_padding_layout.py`
  （共 406 行）：流程并入 `scripts/opcheck.py`，`to_head_layout` / `permute_valid` 两个哨兵布局变换
  移入 `_common.py` 作为 adapter 探针。
- `op-moe-w8a8` / `op-gemm-fp8` 的 `gates` 从「指向不存在的 `tests/*.py`」改为 `template:` 声明，
  消除死链；缺 adapter 前保持 `required: false` 并在 `skip_reason` 写明缺哪几个符号。
- `op-sparse-mla/dev/` 下 `sparse_mla_hip.hip`（H=16 测试形状）、`sparse_mla_hip_scalar_ref.hip`
  （已证伪的标量对照版）、`test_hip.py`：只保留唯一对上生产模型的 `sparse_mla_hip_glm.hip`。
  源码仍在 1.6.0 提交 `2c9cf34`，需要时 `git show 2c9cf34:<path>` 取回。

## [1.6.0] - 2026-10-29

### Added
- **新增第 8 个插件域 `ops`（算子卡域，`Plugins/ops/`）**：以算子为对象组织，一算子一目录，每张卡固定七件套
  `SKILL.md` + `contract.yaml` + `variants.yaml` + `shapes/` + `bench/` + `tests/` + `ledger.md`；
  状态机 `draft → measured → tuned → at_floor → deprecated`，登记在 `data/ops/registry.json`。
  与 `dcu-operators`（源码/框架索引）、`dcu-perf`（剖析与闭环方法）正交。
- 方法论三篇：`op-contract`（契约字段规范、容差分级、不变量清单、地板依据）、
  `op-bench`（口径三问、六条铁律、MAD 显著性、跨卡 ratio）、`op-gate`（三闸四档、回退与留痕）。
- 共享工具：`scripts/opbench.py`（计时/统计/地板判定）、`scripts/opverify.py`（按 `contract.yaml` 判
  ACCEPTED / REJECTED / INCONCLUSIVE / DIAGNOSTIC-ONLY，缺项一律硬拒绝，退出码可直接进 CI）。
- 算子卡首版三张：`op-sparse-mla`（DSA 稀疏 MLA：q/kv/indices 契约、四变体矩阵、`-1` 布局能力检查）、
  `op-moe-w8a8`（FP8 MoE：M 分档、融合收益判据）、`op-gemm-fp8`（batched/分组 GEMM）。
- **`op-sparse-mla` 手写 HIP 交付物**（`dev/sparse_mla_hip.hip` → DSA 档、`dev/sparse_mla_hip_glm.hip` →
  GLM5.3 档）：主循环逐字节取自 tilelang 本机生成的内核，gather 走硬件 `buffer_load_dwordx4`，
  aicc + 全套 `-mllvm` 编译（含 `TL_ENABLE_FAST_MATH` 追加的 `-enable-hcu-approx-func-fp-math` 条件项）。
  320k 长上下文 DSA 档全领先 flash_mla（S=128~512）；GLM 档小 S 领先、大 S 落后（跨 head KV 复用缺失，
  破局需重写 kernel）。站位数字与证伪清单进用户态 KB 台账，skill 卡只留相对值与可迁移方法（aicc 编译纪律、
  buffer_load 向量化、同体不同速三步排查）。
- **orchestrator 算子优化闭环补全**：新增 `op-roofline` / `op-constrain` 两个任务契约、
  `scripts/kbq.py`（五段式 KB 台账校验）、`scripts/workflow_state.py`（工作流状态机）与配套 tests；
  `op-optimize` 工作流加 roofline 节点（两级天花板判定，区分「当前结构到顶」与「物理到顶」）。

### Changed
- 版本登记 1.6.0：`marketplace.json` 根 + 8 插件、`package.env`、`install.sh` 兜底常量、
  根 `SKILL.md`、`AGENTS.md` 与 `install.sh` 生成的 AGENTS.md 模板；规模 134 → 140 个 skill（135 顶层 + 5 子）。
- 根路由表 / AGENTS.md 新增 ops 行，marketplace 安装命令补 `ops`。

### Fixed
- `op-bench` 示例注释笔误（`trade券` → 判据说明）。

## [1.5.1] - 2026-10-22
### Added
- 拆分 7 个插件域为 7 个 Kits Skill Tool（multi-Tool 父/子流水线）：各域独立 `package.env` 与域级 `SKILL.md`（平台单 skill 归档 512 文件上限，各域均在上限内）。

### Fixed
- 路由表重复行清理：dcu-kernel-layer-microbench / dcu-gdn-prefill-kernel-opt 各 1 行重复登记。
- 脱敏漏网收尾：dcu-model-deploy-catalog 真实内网 IP 占位化为 `<DCU_NODE_IP>`；4 处裸机号 36 占位化为
  `<device-D>`（dtk-layout / topology / kernel-dev / on-device-verification 真机验证引用）；12 个文件的
  内部模型变体名去内部构建标记（Channel-/0731/0528 → 常规量化后缀，如 GLM-5.2-INT4-w4a8）；
  dcu-ontrack 旧名引用 2 处改 dcu-ontrack-experience；pd_sweep / nd_pd_sweep 源码注释内部变体标记去标记。

### Changed
- 版本登记 1.5.1（根 SKILL.md / AGENTS.md / install.sh 兜底常量 / marketplace.json 根 + 7 插件 / package.env）。

## [1.5.0] - 2026-10-22

### Removed
- `dcu-rccl-source/source-tree/`（DTK 26.04.2 custom-rccl 完整源码树，650 文件）与
  `dcu-sglang-source/scripts/sglang_dcu_ref/`（海光 SGLang fork DCU 适配源码，30 文件）：
  按 `docs/STANDARDS.md` 上游代码合规红线（不内嵌第三方源码树）移除；两 skill 文字改为
  「源码未随本包分发，自行从 DTK 包/镜像获取」；`scripts/dspark_upstream/`（SGLang 上游快照）
  保留并补 Apache-2.0 LICENSE/NOTICE。
- `dcu-skill-router/references/sourcefind-zfile-push.md`：含内部分发站 / token push 流程 /
  IP 白名单等维护者侧运维细节，不适合开源包；`dcu-inference-infra` 内引用同步移除。

### Security
- 脱敏修补：`dcu-ssh` 跳板机登录名 ×2、`dcu-hy-smi` 内部 ssh 别名、`dcu-model-deploy-catalog`
  三处真实用户 home 目录 → `<jumphost-user>` / `<ssh-alias>` / `<deploy_home>` 占位；
  `dcu-train-cookbook` 删「内网 dcu_megatron…开源线」元话语。
- `sanitize.py` v4：selftest 样本全部改为运行时从 b64 表合成（本文件含 selftest 在内零明文
  敏感值）；新增内部主机别名与新用户名别名规则（b64 存值）；`CONTRIBUTING.md` 白名单补
  `pypi.sourcefind.cn` / `download.sourcefind.cn`（公开分发主机，非泄露）。

### Fixed
- `dcu-orchestrator`（1.1.1 → 1.1.2）：`render_workflow.py` 门禁化——检测到依赖环 / 不可达节点、
  缺任务文件、缺契约字段时输出问题清单并**非零退出**（此前只警告却退出 0，坏工作流会静默通过
  前置核对门禁）；修复 `depends_on` 键缺失或非字符串时的潜伏崩溃（默认值 `[]` 直接走
  `.strip()` 抛 AttributeError）。
- `dcu-orchestrator` SKILL.md：`render_workflow.py` 描述补充退出码语义；使用建议补
  「建议在包外工作目录运行」约定。
- `.gitignore` 增加 `**/runs/`：工作流运行产生的 `runs/<workflow>/<node-id>/...` 证据目录
  不再污染仓库工作树 / 被误提交。
- 恢复 `references/sourcefind/doc-center-index.md`（1.4.0 同步时被删，致 dcu-docker / dcu-dtk /
  dcu-env / dcu-hy-smi / dcu-hyqual-acceptance 5 个 skill 悬空引用）；恢复
  `model-pr-optimization-history/sglang/` 被 `sglang-model-day0-support` 案例引用的 6 个文件
  （deepseek-v4/v41、kimi 的 README.en/zh，敏感扫描 0 命中）。
- `scripts/README.md` 重写：移除未随本包分发的 `skill_gate*` / `githooks` 工具描述与
  CI/install.sh 联动误述（该组工具仅存在于内部 master）。
- `references/upstream-source-contracts.md` 6 处包根相对路径改完整路径；
  `dcu-image-reverse-rebuild/references/rebuild-playbook.md` 2 处相对路径修正。
- `dcu-skill-router`：补登 3 个漏登 skill（dcu-custom-fusion-kernel / dcu-kernel-layer-microbench /
  dcu-gdn-prefill-kernel-opt）；对已剔除 skill 行（hy-smi-tool / dcu-inference-infra-skill-builder）
  补「未随本包分发」标注；删「89 安装包」过时内述与内部 agent `skill_view` API 措辞（改运行时中立）；
  任务链 † 标注修正（kernel-layer-microbench 已入库、model-engine-runbook 等标本机独有）。
- `dcu-env` SKILL.md：vendor 工具「已随本 skill 分发」误述修正（实为 .gitignore 排除、不随包）
  + 内部布局路径 `<agent-home>/skills/dcu/...` 移除。
- kb_shard 知识分片：往返修复 14 行 GBK 乱码 chunk（4 个分片）并重建 FTS 索引；
  残余乱码片段（含 PUA 不可逆部分）挂账，不影响检索主链路。

### Changed
- `dcu-vllm-wiki` 加「内部仓库依赖」标注（知识库在内部 dcu-adapt 仓库，不随包分发）；
  `dcu-ontrack-experience` 加「ontrack 为内部工单系统名、本 skill 为脱敏经验库」标注。
- README 范围说明修正：`*-source` 系列源码结构索引**在包内**、上游源码本体不随包分发；
  「四条内置工作流」统一为五条（README / AGENTS.md / install.sh 模板，1.4.0 新增调研复现未同步）。
- 版本登记 1.5.0：根 SKILL.md / AGENTS.md / install.sh 兜底常量 / marketplace.json（根 + 7 插件）/
  20 个改动 skill 的 frontmatter patch 版本。

## [1.4.1] - 2026-10-22

### Security
- 补齐 1.4.0 脱敏漏网 238 处（44 文件，门禁 v3 新规则）：内部 GitLab 主机/组路径 120 处、裸设备号 BDF 75 处（domain 形 + lspci 短形上下文）、agent 运行时目录路径 27 处、frontmatter 内部键名 8 个（统一为 toolkit）、内网用户名 4 处、内部 git 路径 2 处、内部邮箱 1 处、在线文档指纹短 token 1 处、裸设备号 1 处。
- dcu-code-migrator 知识分片脱敏：source-manifest.json 71 处 + shard.sqlite 112 行内部 GitLab URI 替换为占位符（FTS 索引重建 + VACUUM，字节级零残留）；还原 1 处 GBK 乱码标题；修复 shard.sqlite 被检出期行尾转换损坏的二进制完整性。

### Added
- `CONTRIBUTING.md` 脱敏规程：内容与提交信息双重脱敏、占位符约定、公开平台白名单、二进制扫描口径、CI 接入。
- `.gitlab-ci.yml` sanitize-gate 作业：open-source 分支每次 push 运行全仓门禁 + 提交信息检查。
- `sanitize.py` v2 → v3：新增内部 GitLab 主机/组/仓库路径/邮箱域、内网用户名拼音别名（@提及豁免）、裸设备号 BDF（domain 全局 + 短形上下文）、agent 运行时路径与 frontmatter 键名、在线文档指纹短 token、GBK 乱码文件名检测、--msg-file 提交信息检查模式；--selftest 补新规则样本。
- `.gitattributes`：二进制文件（.sqlite/.gz/.pyc/.db/.whl/图片）显式标 binary，防止 Windows autocrlf 检出时转换损坏。

### Fixed
- dcu-coding-skill SKILL.md：子 skill frontmatter 风格描述修正（现状为外部包原生 keywords/intents 单一风格）。
## [1.4.0] - 2026-10-21

### Added
- 从内部 master 同步（sanitize 全量通过）7 个 skill：dcu-core 域 +2（`dcu-offline-container-deps`、`dcu-offline-wheel-install`）、
  dcu-env 域 +1（`dcu-hyqual-acceptance`）、dcu-perf 域 +2（`dcu-external-kernel-validation`、`dcu-external-opt-scout`）、
  dcu-troubleshooting 域 +2（`cross-engine-acceptance-gate`、`dcu-ontrack-experience`）；skill 计数 129→136（131 顶层 + 5 子 skill）。
- orchestrator 增量：`tasks/baseline-run.yaml`、`tasks/env-reproduce.yaml`、`tasks/research-collect.yaml`、
  `workflows/research-baseline.yaml`、`scripts/extract_candidate_facts.py`、`scripts/suggest_target.py`、
  `references/promotion-review-convention.md`、`evals/evals.json`（tests/ 含内部脱敏夹具，不随包分发）。
- 包根共享层：`references/dcu-models.md`、`references/dual-machine-verification.md`、`references/sourcefind/`、
  `references/hygon-upstream/` 更新；`scripts/run_evals.py`（evals 静态检查器）、`scripts/json_schema_mini.py`（零依赖 schema 校验）。
- 9 个 evals 套件（dcu-skill-router / dcu-docker / dcu-ssh / dcu-image-hub / dcu-modelzoo /
  dcu-model-deploy-catalog / dcu-vllm-deploy / dcu-ontrack-experience / dcu-orchestrator）。

### Changed
- 多 skill 内容更新：kernelwiki hwprobe P0 四机实测基线、`dcu-models` 四型号实测值、`dcu-dtk-layout` 新基线、
  docker/driver/dtk/ssh 等；`dcu-train-cookbook` 内容更新（mAP-cv/Physical AI 配方）。
- 脱敏规则升级：词表补 实测/容器/真机/环境/现场/凭证/机型/跳板/跑/等/基础/标点 + 111→`<device-D>` 机号映射 + ASCII/全角括号机号。

### Fixed
- `dcu-skill-router/evals/evals.json`：false 案例 f6 去掉 expect（false 只给 id+prompt 规范）。

## [1.3.0] - 2026-10-15

### Added
- 沉淀闭环：新增 `op-distill` 任务契约 + `op-optimize` 工作流第 4 节点（gate，需审批）；新增
  `Plugins/dcu-core/skills/dcu-orchestrator/scripts/dcu_distill.py`——把一次 run 的产物（如
  `runs/op-optimize/validate/result.json`）**确定性**转成知识库台账（.md）+ 机器可读证据（.json，可交
  `verify_evidence.py` 按任务契约校验）+ 晋升提案；数字只来自产物，缺字段标 MISSING，不做推断。
- 用户知识库（沉淀落点，与 skill 树解耦）：`~/.config/dcu-toolkit/kb/{op-opt,issue,solution}`，
  新增 `data/kb/README.md`（目录与写入约定）与三份模板（台账 / 故障记录 / 方案卡片）。

### Changed
- install.sh 同步改**清单式增量**（清单在 `~/.config/dcu-toolkit/manifests/`）：只删「上次装的 + 本轮包内已移除 +
  本地未改过」的文件；用户自己新增的文件永不删除；本地改过的厂商文件被覆盖前先备份 `<file>.local.bak`。
  这修掉了旧「整体 rm -rf + cp -a」在重装/升级时抹掉用户沉淀与本地修改的问题，也修掉了带尾斜杠的源路径
  导致「一个文件都不拷、只留空目录」的裁剪缺陷（现改为硬报错中止，不再静默跳过）。
- 沉淀口径改指用户知识库（原约定写在 skill 树里，重装即失效）：`docs/STANDARDS.md`、`data/issue-kb/`、
  `data/product-matrix/` 改为只读模板 + 指向 `kb/`；`dcu-op-optimization` 增「第 6 步·沉淀（gate）」；
  `dcu-kernelwiki` 增安装态只读红线与晋升路径（候选事实先入 kb，经审核才并入 wiki）；orchestrator
  SKILL/README 同步链路与脚本清单。

## [1.2.0] - 2026-10-15

### Added
- 自内部版迁入 29 个 skill（sanitize 全量通过、内部引用改占位标注后收录）：
  env 域 +8（dcu-adaptation-overview、dcu-software-adaptation-guide、dcu-csv-passthrough、dcu-das-hub-api、
  dcu-image-hub、dcu-modelzoo、dcu-opensource-platform、dcu-rccl-source）、
  inference 域 +8（dcu-pytorch-source、dcu-sglang-source、dcu-vllm-source、dcu-vllm-hcu-source、
  dcu-training-frameworks-source、dcu-mooncake-source、dcu-ort-source、dcu-fastsafetensors-source）、
  operators 域 +10（dcu-aiter-source、dcu-lightop-source、dcu-deepgemm-source、dcu-triton-source、
  dcu-tilelang-source、dcu-mhc-kernels-source、dcu-hipdsa-source、dcu-flashattn-source、dcu-deepep-source、
  dcu-skill-on-device-verification）+ dcu-coding-skill 子包补齐 3 个（dcu-dispatcher、dcu-code-migrator、
  dcu-hip-reference，kb_shard 保留、kb_shard_ext 不随包分发）。
  总数 100 → 129（124 顶层 + 5 子 skill）
- dcu-skill-router 路由表同步：新迁 skill 去「未收录」标注（域前缀路径校正）、真缺席 3 个
  （hy-smi-tool / dcu-ontrack-experience / dcu-inference-infra-skill-builder）改为明确
  「未随包分发 + 包内替代指引」；头部计数 83+5+33=127 更正
- 正文引用缺席 skill 的 23 个文件统一补「未随本包分发」标注，消除断链
- sglang-source / rccl-source / tilelang-source 的内部源码树引用
  （sglang_dcu_ref/、dspark_upstream/、source-tree/）改为「未随本包分发」说明

### Fixed
- sanitize.py 裸机号规则补词（实测/容器/真机/镜像/环境/现场/凭证/机型/跳板/跑/逗号/顿号/基础）、
  internal_ip_<数字> 占位规则；全库 --fix 后 0 命中，人工复核 8 处真机号逐个占位化
  （<device-A>/<device-B> 占位），避免内部机号明文外泄
- sanitize 双重替换产生的 <<占位>> 嵌套损坏修复（9 个文件）
- 修正迁移中断残留的错误域目录 Plugins/env|inference|operators（→ dcu-env 等正确三域）
- dcu-opensource-platform related_skills 去掉缺席的 hy-smi-tool

## [1.1.0] - 2026-10-15

### Changed
- dcu-orchestrator 扩充三条内置工作流（各含 3 个任务契约，probe→normal/gate→verify 三段式）：
  `model-deploy`（deploy-precheck → deploy-launch(gate) → deploy-verify，模型部署验证主线）、
  `fault-triage`（fault-collect → fault-locate → fault-fix(gate)，故障排查主线）、
  `op-optimize`（op-profile → op-optimize(gate) → op-validate，算子优化主线）
- dcu-orchestrator SKILL.md 第五节改为四工作流总表；README 目录表同步 9 个新任务契约
- install.sh 共享层断链修复：新增共享层同步步（包根 data/references/scripts/docs → `<skills根>/_dcu-toolkit-shared/`，
  幂等、排除 __pycache__）；7 处跨 skill 的包根共享资源引用改为双态路径写法（仓库根 / 安装态）
- data/ 空壳激活：issue-kb、product-matrix 增加「谁写入」沉淀约定（由编排工作流收尾写入），
  fault-fix / deploy-verify 任务契约增加沉淀步骤，STANDARDS.md 补共享层双态路径与沉淀约定
- render_workflow.py：tasks/ 目录默认改为相对 workflow 文件自动定位（同级或 skill 根），
  仓库根直跑 `python3 .../render_workflow.py .../workflows/xxx.yaml` 不再报 task file not found
- infer-optimize.yaml：移除占位 explore 节点（其 task 复用 baseline 导致渲染出误导性 goal），
  扩展方式改为基础工作流间串联说明
- install.sh 内嵌 AGENTS.md 模板的 orchestrator 工作流行同步四工作流

## [1.0.0] - 2026-10-08

### Added
- 首个开源版本，100 个 skill（98 顶层 + 2 子 skill，dcu-coding-skill 内嵌 2 个）
- 7 插件 marketplace 结构：dcu-core / dcu-env / dcu-perf / dcu-inference / dcu-operators / dcu-troubleshooting / dcu-engineering
- Claude Code marketplace + install.sh 双通道安装
- `Plugins/dcu-perf/skills/upstream-source-contracts.md`：上游框架源码检视契约（修复 12 处跨文件断链）

### Changed
- 上游来源标注补齐：13 个 BBuf/AI-Infra-Auto-Driven-SKILLS 系 skill（llm-torch-profiler-analysis、model-pr-optimization-history 等）加 MIT 来源头声明；5 个优化闭环 skill（generate-profile、humanize-gen-plan、humanize-rlcr、llm-pipeline-analysis、llm-serving-capacity-planner）加来源注记
- dcu-vllm-precision README 补 benyuereal/llm-optimize（MIT）归属声明（LICENSE 文件已随目录保留）
- dcu-perf-cases 的 fork 名称改为「海光 DCU fork」中性表述

### Removed
- dtk-scripts 目录（12 个 DTK/AMD ROCm 衍生脚本，商业发行版再分发需遵循上游许可；改为文档化 docker cp 提取指引）
- 部署环境相关地址、文档链接、设备清单与凭证（统一替换为占位符模板）
