# 发版检查清单

> 评测框架能自动证明的只有「路由」和「要点/错话」（`docs/EVALS.md`）。
> 本清单收的是**框架不背、但发版前必须人做一遍**的事。逐条勾，结果记进 `CHANGELOG.md` 或 KB 台账。

## 一、静态门（必须全绿，跑一遍约 10 秒）

```bash
python3 -I scripts/run_evals.py                 # evals 结构 + description 语义面
python3 -I -m unittest discover -s Plugins/ops/tests -p 'test_*.py'
python3 sanitize.py .                           # 零内部残留（内网地址/密码/密钥）
bash scripts/install_sync_selftest.sh           # 安装清单式增量同步
```

## 二、动态路由抽检（改动过的 skill，`--repeat 3`）

```bash
python3 scripts/skill_eval.py --skill <改过的 skill> --repeat 3 --jobs 3
```

- 3/3 才算稳；`FLAKY` 不算过，`未拉起` 属内容问题（见 `docs/EVALS.md` §四）。
- 通过率与红项写进 `CHANGELOG.md`，别只留在终端里。

## 三、上机验证（L3，需要空闲 DCU）

对**这次改到的算子卡**逐张跑，不要求全量：

```bash
# ① 契约与门模板
python3 scripts/opverify.py --contract Plugins/ops/skills/<card>/contract.yaml

# ② 三道机检（identity / correctness / invariants / padding，按契约声明的来）
python3 -I scripts/opcheck.py correctness --contract Plugins/ops/skills/<card>/contract.yaml --shape <shape>
python3 -I scripts/opcheck.py invariants  --contract Plugins/ops/skills/<card>/contract.yaml --shape <shape>
python3 -I scripts/opcheck.py padding     --contract Plugins/ops/skills/<card>/contract.yaml --shape <shape>

# ③ 性能口径（跑完要留 JSON，结论才可复核）
python3 Plugins/ops/skills/<card>/bench/bench.py --shape <shape> --json /tmp/bench.json
```

记录三件事：**日期 / 设备与 gfx / DTK 版本**，以及每条结论的 `source`。
绝对性能数字不进 skill 包，进 KB（`~/.config/dcu-toolkit/kb/op-opt/`）。

## 四、版本与留痕

- [ ] 每个改动的 skill bump `version`（打补丁级即可），`CHANGELOG.md [Unreleased]` 写清「改了什么/为什么」。
- [ ] 新事实带 `source` + `verified_on`；未验证的标 `unverified`；历史性能数字标 `[历史数据，随硬件/DTK/版本变化]`。
- [ ] 无密码/密钥/内网地址进包（`sanitize.py` 已扫，人工再看一眼新增的示例命令）。
- [ ] 提交身份用仓库本地配置（`git config user.name` 在仓库内生效），不改全局 git 配置。

## 五、发布后

- [ ] `bash install.sh` 在目标机重装一次，确认幂等（不重复拷贝、用户新增文件不被删）。
- [ ] 用装好的副本跑一条 `skill_eval.py --no-isolate` 抽检，确认安装产物与工作区一致。
