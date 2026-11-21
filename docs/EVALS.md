# Skill 评测：静态门 + 动态门

> 一句话：`run_evals.py` 管「用例写得对不对」，`skill_eval.py` 管「这条 prompt 到底会不会拉起它」。
> 前者零成本、进 CI；后者烧 token、按需跑。两者共用同一份 `<skill>/evals/evals.json`。

## 一、两层分工

| | `scripts/run_evals.py`（静态） | `scripts/skill_eval.py`（动态） |
|---|---|---|
| 判什么 | JSON 结构、id 唯一/连续、true≥3 + false≥2、`expect` 只出现在 true、description 是否写了触发条件与不适用反例 | 起一次真会话，看 `Skill` 工具**第一个**拉起来的是谁；可选再判回答内容 |
| 成本 | 毫秒级，零 token | 每例约 2–4 万 token、20–60s |
| 依赖 | 无 | 本机 `claude` 登录态（不需要 DCU、不需要装插件） |
| 何时跑 | 每次提交 / CI | 改过 description、新增 skill、发版前抽检、路由事故复盘 |
| 退出码 | 0 PASS / 1 FAIL | 0 全过 / 1 有失败 / 2 环境问题 |

## 二、用例怎么写（`<skill>/evals/evals.json`）

约定 **3 条 true + 2 条 false**：

- **true**：这句 prompt 只有本 skill 答得对。`expect` 写「应命中的要点」，要点必须来自 `SKILL.md` 正文
  （阈值、顺序、必填字段、禁止做法），**不要把 description 原句抄进 expect**——那样只能测出「描述被读到」。
- **false**：形似、但应路由到兄弟 skill。只给 `id` + `prompt`，不给 `expect`。
  反例的选择顺序：**同域最近邻**（op-bench ↔ op-gate）> 跨域同名概念（dcu-ssh ↔ dcu-docker）> 泛指问题。
- 不要写「完全不相关」的题：谁都拦得住，测不出东西，还白花 token。
- **prompt 要带真实抓手**（产物名、命令名、术语）：「contract.yaml 里的容差写 2e-3 行不行」
  比「两个实现之间容差怎么定」稳得多。纯抽象问法会在多个 skill 之间摇摆，测的是问法而不是路由。
  最稳的抓手是**目标 skill 独有的口径名**：`TA_* 跨 SE 求和`、`首次 launch 虚高 ~40%`、
  `contract.yaml` 的 `tolerance.vs_fp32_ref`、`loop.py queue`、`四闸`。这些词只在一个 skill 里出现，
  邻居抢不走。反例（实测）：问成「PMC 怎么判瓶颈」时 `dcu-hipprof` / `dcu-perf-analysis` / `op-roofline`
  轮流来抢，同一个 skill 连改两版描述都没稳住；把问句落到「TA_* 跨 SE 求和」这种口径名上，一次就固定下来。
- **false 用例的 prompt 里不要出现被评 skill 名字里的词**：`dcu-ssh` 的 false 用例只要写了「SSH」，
  模型按名字命中就会把它拉起来——描述里再清楚的不适用条款也压不住名字。这类题换个近邻，或改写问法
  （「两个容器之间配免密」而不是「配 SSH 免密」）。
- **跨切面 skill 不适合当 false 用例的主语**：`dcu-ssh` 的章程写明「所有子 skill 执行远程命令前均须遵循本规范」，
  它被并载是设计使然。对这类 skill，只测 true（该不该被拉起、答得对不对），不测「绝不能出现」。
- 线上出现过一次误路由，就把它固化成一条 false 用例——用例是事故的墓碑。

### 可选：`forbid`——路由对了，但话说错了

true 用例可以加 `forbid`（false 用例不带）。它不花额外 token：模型回答本来就抓在手里，
`skill_eval.py` 直接在里面找**错误说法**，命中即判 `说错`。

```json
"forbid": ["首次 launch 的数可以直接用",
           "re:TA_\\*[^。；\\n]{0,8}取平均"]
```

三条规则：

1. **写「错话」本身，不写关键词**。写 `"取平均"` 会冤枉「不是取平均，要按 SE 求和」这种正确回答；
   静态门会拦 ≤3 字的条目。需要覆盖多种措辞时用 `re:` 前缀。
2. **命中片段本身或它前面 24 个字里有否定词就跳过**（不/别/禁止/避免/勿/严禁/杜绝…）。
   这是启发式：**宁可漏报，不可误报**——会冤枉正确答案的门，很快就没人看了。
3. 只对「路由已经对」的用例判。路由错了先修路由，别拿内容分掩盖。

实测（2026-10-30）：`dcu-ssh-t1` 的真实回答写了「密码**绝不能**出现在 `sshpass -p` 里」，
`op-profiling-t2` 写了「求和，不是取平均」——两条都正确跳过，`forbid` 没有误报。

## 三、怎么跑

```bash
# ① 静态门（CI 同款，提交前先跑）
python3 -I scripts/run_evals.py
python3 -I scripts/run_evals.py --strict-description   # 全库 description 都阻塞（收敛期用）

# ② 看有哪些 skill 带 evals
python3 scripts/skill_eval.py --list

# ③ 跑一个 / 一个域
python3 scripts/skill_eval.py --skill dcu-ssh --jobs 2
python3 scripts/skill_eval.py --domain ops --jobs 4 --json /tmp/eval-ops.json

# ④ 只跑失败项（改完描述验证）
python3 scripts/skill_eval.py --skill dcu-ssh --jobs 2
python3 scripts/skill_eval.py --case op-gate-t2 --case op-gate-t3 --jobs 2

# ④b 同一例跑 3 次看抖动（报 k/N：全过 PASS、部分过 FLAKY、全败 FAIL）
python3 scripts/skill_eval.py --case op-optimize-t2 --repeat 3 --jobs 3

# ⑤ 连内容一起判（成本翻倍：true 用例再加一次评分调用）
python3 scripts/skill_eval.py --skill op-bench --judge

# ⑥ 先看清要跑什么、不花钱
python3 scripts/skill_eval.py --domain ops --dry-run
```

## 四、怎么读结果

每行 `PASS/FLAKY/FAIL k/N 用例 秒数 细节`（`--repeat` 默认 1 时就是 `1/1`），`细节` 里的
`picked` 就是**模型实际拉起的 skill 列表**——失败时先看它，再决定改谁：

| 现象 | 含义 | 修哪里 |
|---|---|---|
| `FAIL 0/N`，`未拉起` | 模型压根没加载任何 skill：它觉得通用知识够用 | **内容**问题——这个 skill 的不可替代性没被感知，别继续改描述，先补正文里的硬知识/工具 |
| `FAIL 0/N`，`拉错`，`picked` 是兄弟 skill | 描述没把邻居挡住 | 补「触发场景」措辞，并写清边界 |
| `FLAKY k/N`，两种原因混着 | 路由处在两个 skill 之间摇摆 | 先按多数原因修；修完 `--repeat 3` 复测，全过才算 |
| false 命中了本 skill（`误拉`） | 描述过宽（「什么都能答」）**或** prompt 里出现了 skill 名里的词 | 先看是不是名字命中（换问法），再补「不适用：…走 `<兄弟 skill>`」 |
| false 没命中本 skill，但也没拉起该去的那个 | 路由噪声，不是本 skill 的错 | 记录；必要时给目标 skill 也加一条 true 用例 |
| `FAIL`/`FLAKY`，`说错` | 路由对了，但回答里出现了 `forbid` 写死的错误说法 | **内容**问题：`SKILL.md` 里的硬规则没压住模型（阈值写清、给反例），不是改描述 |
| `--judge` 分数 < 4 | 路由对了但正文缺要点 | 改 `SKILL.md` 正文，不是改描述 |

判分是概率事件：`FLAKY` 不是通过，**同一 skill 连续两次失败**才当回归；单次抖动先重跑。
要下结论就 `--repeat 3`：3/3 才算稳，0/3 是稳定不拉或稳定拉错，1~2/3 是摇摆。
结果随模型版本漂移，报告里要记 `--model` 与日期，不承诺跨版本逐例可复现。

**同一个用例连红三次，先怀疑用例，不是描述。** 改动顺序：① 加产物抓手（`TA_* 跨 SE 求和`、
`首次 launch 虚高`、`candidates.yaml`、`四闸`）让问题只有一处答得对；② 换掉名字命中的词；
③ 最后才动 description。前两轮我们连续改了 6 个描述才让 `op-optimize-t2` 转绿，
而 `op-gate-t3`/`op-profiling-t1` 是改成产物抓手后才有确定答案的。

## 五、隔离与坑

- 每例在独立临时 cwd 里跑，禁 `Bash/Write/Edit/WebFetch/Task`，评测过程改不了任何东西。
- **挂仓内全部插件**（8 个），不是只挂被测域：路由本来就发生在全库之间，只挂一个域测不出「被邻居抢走」
  （`op-bench` 被 `dcu-kernel-layer-microbench` 截胡就是这么发现的）。插件挂载的 skill 带命名空间
  （`ops:op-bench`），判分前自动去掉前缀；`--json` 明细里保留原名，便于看它到底从哪来。
- **必须隔离 `CLAUDE_CONFIG_DIR`**（脚本默认做）：否则 `~/.claude/skills` 里上次安装的旧副本会参与竞争，
  你改工作区里的 `SKILL.md` 对结果**毫无影响**。这个坑真踩过——第一轮 ops 评测读的全是 10-28 安装的旧描述。
  想复现用户现场（含已安装 skill）时用 `--no-isolate`，但那时测的就不是工作区这份了。
- 隔离后鉴权走 `ANTHROPIC_AUTH_TOKEN` 等环境变量；若你的环境依赖配置文件里的 OAuth，隔离会失败，
  此时只能退回 `--no-isolate`，并接受「测的是已安装副本」这个偏差。
- **插件根必须有 `.claude-plugin/plugin.json`**：没有它 `--plugin-dir` 挂不上，表现是「所有用例 0 命中」，
  看起来像描述全烂。`test_eval_tooling.py` 会拦这一类问题。
- 必须从环境里去掉 `CLAUDECODE`（脚本已处理），否则嵌套会话直接失败。
- 机器上若残留旧仓库副本（`/tmp/p04` 这类），模型可能读到错的树；动态评测前确认临时目录干净。
- 评测不读 `~/.claude/skills` 里已安装的版本——**测的是工作区里的这份**，这正是我们要的。

## 六、CI 策略

- 静态门进 CI（`.gitlab-ci.yml` 的 `ops-quality-gate`）：`python3 -I scripts/run_evals.py` + ops 单测。
- 动态门**不进 CI**：成本高、需要登录态、结果有随机性。发版前对「改动过的 skill」跑一遍，
  把通过率写进 `CHANGELOG.md` 或 PR 描述。
- `forbid` 零额外成本，跟着动态门一起跑，不需要单独的开关。
- `--strict-description` 是收敛工具：等 144 个 skill 的描述都补齐后，把它打开并写进 CI，
  那时「描述不合规」才从报告项变成硬门。

## 六点五、能力评测的四级阶梯，以及为什么只做到 L1

| 级 | 测什么 | 判据 | 状态 |
|---|---|---|---|
| L0 | 会不会被拉起 | 首个 `Skill` 调用（`--repeat` 采样） | ✅ 已有 |
| L1 | 拉起来后说得对不对 | `expect` 要点 + `forbid` 错话 | ✅ 已有（`--judge` 为可选加强） |
| L2 | 能不能产出合格工件 | 产物过程序判定（`opverify.py` 等） | ⏸ 暂缓 |
| L3 | 端到端真任务 | 上机跑 `opcheck.py` / bench | 人工，见 `docs/RELEASE-CHECKLIST.md` |

**L2 暂缓的理由**：它要「模型写文件 → 程序判产物」，全仓目前只有 `op-contract` 一张卡有现成的
程序判定器（`opverify.py`），其余 skill 的产物是建议和排查步骤，没有机器可判的形式。为一张卡搭
`mode: agentic` + 沙箱 + check 协议，是一次性需求做成通用机制，之后只有一条用例在用。

**重新评估的触发条件**（满足任一即做）：① 出现第 3 个有可判定产物的 skill；② 线上出现一次
「skill 被拉起了，但产出物不能用」的事故——把那次事故固化成第一条 L2 用例。

**L3 不进框架**的理由：要空卡、要权重、一次几十分钟、结果随机器漂，做成自动门只会变成
「经常红、大家学会忽略」的门。它属于发版检查清单。

## 七、覆盖面与扩面顺序

现状：144 个 skill 中 19 个带 evals（dcu-core 2、dcu-env 4、dcu-inference 2、
dcu-troubleshooting 1、ops 10）。ops 的 5 张算子卡（op-gemm-fp8 / op-hip-compile /
op-moe-w8a8 / op-roofline / op-sparse-mla）于 2026-11-13 补齐 3+2 例。最近一次整批
（2026-10-30，`--jobs 5`）：`dcu-ssh` 5/5、ops 21/25。
飘的 4 条逐条复跑后 3 条转绿；`op-optimize-t2`（`loop.py status` 续跑）用 `--repeat 3` 实测
**1/3**，两次是 `未拉起`——模型认为通用知识够答，属内容问题而非描述问题，留作待办
（修法是给 `SKILL.md` 补 `loop.py` 的硬机制，不是继续改描述）。
扩面顺序按「误路由代价」排：

1. **ops 方法论**（已完成）：op-contract / op-bench / op-gate / op-optimize / op-profiling——
   这五个互相抢活最严重（「怎么量」「怎么判」「怎么持续优化」边界很近）。2026-11-13 起
   5 张算子卡（op-gemm-fp8 / op-hip-compile / op-moe-w8a8 / op-roofline / op-sparse-mla）也补齐。
2. **路由入口**：dcu-skill-router / dcu-orchestrator（已有）——它们是所有误路由的第一现场。
3. **高频域**：dcu-env（已有 4 个）、dcu-inference（已有 2 个）、dcu-troubleshooting（已有 1 个）。
4. 其余按事故驱动补齐；新增 skill 建议同时带 3+2 例再进包。

## 八、和「skill 质量」的边界

评测只能证明**路由与要点**这两件事，证明不了：

- 步骤在真实 DCU 上跑得通（那要上机，见各卡的 `bench/` 与 ops 的 `opcheck.py`）；
- 内容是不是最新（那靠 `source` + `verified_on` 与 KB 台账）；
- 描述读完就能解决问题（那靠 `SKILL.md` 自身的结构与 `references/` 渐进披露）。

把这三件事混进评测，只会得到一个「看起来很科学」的分数。评测的价值在于：**它是唯一能自动发现
「这个 skill 根本没被拉起来」的手段**。
