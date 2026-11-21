# HANDOFF

> 交接对象：接手 `dcu-toolkit-open` 评测线与 ops 算子卡的人。
> 写于 2026-10-30。分支 `feat/ops-operator-cards`，HEAD `8fe3d9d`，工作树干净、已推 origin。

## 一、这次交付了什么

### 评测框架（两层，共用 `<skill>/evals/evals.json`）

| 层 | 入口 | 判什么 | 成本 |
|---|---|---|---|
| 静态 | `scripts/run_evals.py` | JSON 结构、3 true + 2 false、id 连续、description 有「触发条件 + 不适用」 | 毫秒，已进 CI |
| 动态 | `scripts/skill_eval.py` | 真会话里**第一个** `Skill` 调用是不是目标 skill；可选 `--judge` 内容评分；`forbid` 零成本判「说错」 | 35k token/例，不进 CI |

三份提交：`3985c17`（框架 + ops 五套用例 + 描述修复）、`893aa5c`（`--repeat` 与失败归类）、
`8fe3d9d`（`forbid` + 发版清单 + L2 暂缓说明）。

### 配套文档（先读这两份，别从代码猜）

- [`docs/EVALS.md`](docs/EVALS.md)：用例怎么写、怎么跑、结果怎么读、隔离坑、CI 策略、能力四级阶梯。
- [`docs/RELEASE-CHECKLIST.md`](docs/RELEASE-CHECKLIST.md)：发版前的人工步骤（含上机验证）。

### 评测实测结论（2026-10-30）

- 整批：`dcu-ssh` 5/5、ops 21/25。飘的 4 条单跑后 3 条转绿。
- 唯一稳定红项：`op-optimize-t2`（`loop.py status` 续跑），`--repeat 3` 实测 **1/3**，两次是 `未拉起`。
  定性是**内容问题**（模型认为通用知识够答），不是描述问题——修法是给 `SKILL.md` 补 `loop.py` 的硬机制。
- 描述合规：144 个 skill 里 20 个合规（带 evals 的必须合规，否则静态门阻塞）。

### 明确没做的（别再重新讨论）

- **L2（产物判定）暂缓**：全仓只有 `op-contract` 有现成判定器。重新评估的条件写在 `docs/EVALS.md` §六点五。
- **L3（上机）不进框架**：归 `docs/RELEASE-CHECKLIST.md`，人工跑。
- `--strict-description` 未开：等 124 个缺语义面的描述补齐再开。

## 二、跑评测的常用命令

```bash
python3 -I scripts/run_evals.py                     # 静态门（提交前必跑）
python3 scripts/skill_eval.py --list                # 哪些 skill 带 evals
python3 scripts/skill_eval.py --dry-run --domain ops
python3 scripts/skill_eval.py --skill op-gemm-fp8 --repeat 3 --jobs 3
python3 scripts/skill_eval.py --case <id> --repeat 3 --jobs 3   # 只复跑失败项
```

结果读法：`PASS/FLAKY/FAIL k/N`。**`FLAKY` 不算过**；失败归类 `未拉起`（内容）/ `拉错`（描述）/
`误拉`（描述过宽）/ `说错`（`forbid` 命中）。细节见 `docs/EVALS.md` §四。

## 三、下一步：fp8-gemm

### 先确认是「补卡」还是「新卡」

`Plugins/ops/skills/op-gemm-fp8/` **已经存在**，`data/ops/registry.json` 里状态 `draft`，
但目前只有 `SKILL.md` + `contract.yaml`，**缺最小五件套里的三件**：

```
Plugins/ops/skills/op-gemm-fp8/
├── SKILL.md          ✅ 1 行 description + 正文（≤240 行，文档门会卡）
├── contract.yaml     ✅
├── variants.yaml     ❌ 缺
├── shapes.yaml       ❌ 缺
└── _common.py        ❌ 缺
```

所以多数情况下这是**把 draft 卡补完整**，而不是新建目录。真要新建，命名沿用 `op-<算子名>`，
不带 `dcu-`（`Plugins/ops/SKILL.md` §一）。已有卡千万别重复建——路由会劈叉。

### 补卡清单（按顺序，每步都有机器门）

1. **`variants.yaml`**：实现变体矩阵，一条变体 = 来源 → 适用档位 → 状态 → 约束。
   FP8 GEMM 至少覆盖 tilelang（MatrixCorePreshuffle / warp_size=64 / k_pack=2）、hipBLASLt `_scaled_mm`、
   aiter；跑不通的要写清"在哪个 gfx/哪个 dtype 上不可用"，别只写"推荐 X"。
2. **`shapes.yaml`**：一档一个顶层键（如 `bf16-m4096-n4096-k4096`、`fp8-e4m3-m8192...`），
   每档挂 `(算子, 档位, gfx, dtype)` 结论。改字段只改一处。
3. **`_common.py`**：adapter 协议 `REQUIRED = (build_inputs, reference, runner_kwargs, get_runner, run_isolated)`，
   数学写在本文件；不变量（确定性 / batch 独立 / padding 不变）走包根 `scripts/opcheck.py`，**不要建卡内 tests/**。
4. **`contract.yaml` 的 `gates`**：`template:` 必须是 `opcheck.py` 的子命令（`correctness` / `invariants` / `padding`），
   `script:` 指向的文件必须存在（除非 `required: false` + 有 `skip_reason`）——`test_doc_links.py` 会拦。
5. **`data/ops/registry.json` + 域索引**：`name` / `path` / `status` 三方一致，**索引状态列必须等于 registry 的 status**；
   `Plugins/ops/SKILL.md` 的表要同步。
6. **evals（3 true + 2 false）**：新增/改动的 ops 卡必须带。false 用例取同域最近邻；prompt 带卡内独有抓手
   （`contract.yaml` 字段名、`_common.py` 的函数名、`variants.yaml` 的变体 key）。
   路由稳定后再考虑加 `forbid`（例如"用 fp8 e4m3 就行，不用管 fnuz"这类错话）。
7. **description**：必须同时有触发场景 + 「不适用：…走 `<兄弟 skill>`」。带 evals 时缺任一面**静态门直接 FAIL**。

### 完成判据

```bash
python3 -I scripts/run_evals.py
python3 -I -m unittest discover -s Plugins/ops/tests -p 'test_*.py'
python3 scripts/opverify.py --contract Plugins/ops/skills/op-gemm-fp8/contract.yaml
python3 sanitize.py .
bash scripts/install_sync_selftest.sh
python3 scripts/skill_eval.py --skill op-gemm-fp8 --repeat 3 --jobs 3    # 3/3 才算稳
```

上机部分按 `docs/RELEASE-CHECKLIST.md` §三跑 `opcheck.py` 与 bench。

### 记录纪律（最容易违规的三条）

- **绝对性能数字不进 skill 包**：进 `~/.config/dcu-toolkit/kb/op-opt/`，卡里只留相对倍数与结论。
- 每条事实带 `source` + `verified_on`（日期 / 设备 / gfx / DTK）；未验证标 `unverified`。
- 历史数字标 `[历史数据，随硬件/DTK/版本变化]`，不进可复用事实条目。

## 四、这次踩过、别再踩的坑

| 坑 | 表现 | 现状 |
|---|---|---|
| 插件根缺 `.claude-plugin/plugin.json` | `--plugin-dir` 静默挂不上，所有用例 0 命中 | `test_eval_tooling.py` 已拦；新增插件记得补 |
| 不隔离 `CLAUDE_CONFIG_DIR` | 测的是 `~/.claude/skills` 里的旧安装副本，改工作区没效果 | 脚本默认隔离 |
| false 用例撞上 skill 名字词 | 问句里写 `SSH`/`bench` 就会把那个 skill 拉起来，描述再清楚也压不住 | `dcu-ssh` 是横切 skill，只用 true 用例测它 |
| 抽象问法 | 同一题招来 3 个邻居轮流抢 | 抓手落到 skill 独有的口径名（`TA_* 跨 SE 求和`、`首次 launch 虚高`） |
| 单次结果当结论 | 同一用例时红时绿 | `--repeat 3`：3/3 稳、0/3 稳定不拉、1–2/3 摇摆 |

## 五、环境与身份

- 提交身份用**仓库本地** `git config`，不要改全局；推送时逐命令覆盖 SSH 代理：
  `GIT_SSH_COMMAND='ssh -o BatchMode=yes' git push`。
- 包内零密码、零密钥、零内网地址；发版前 `python3 sanitize.py .` 必须 0 命中。
- 用户态知识库 `~/.config/dcu-toolkit/kb/`（`op-opt/` 台账、`issue/` 故障、`solution/` 方案）；
  重装不会删用户新增文件，但**别把沉淀写进 skill 树**。
