---
name: op-source-patch
description: "从源码改算子：定位→取源→编译→改(patch)→回归→留痕六步固定。把「算子卡说哪个实现好」和「这段源码怎么拉下来改掉编出来验回去」接上。触发场景：现成实现差一口气、需要改库源码而不是调用参数、要在本地源码树上做可回滚的改动、要把一次源码改动变成可复现的证据。不适用：只调参数（走 op-optimize）；纯手写新 kernel（走 op-hip-compile）；只量不改（走 op-bench）。"
version: 1.0.0
author: DCU Toolkit Contributors
license: MIT
kind: tool
---

# 源码级改动（op-source-patch）

## 一、为什么要这一层

仓库里有 12 个 `*-source` skill（vllm / sglang / pytorch / flashattn / lightop / aiter…），
它们和算子卡是**两套世界**：

- 算子卡说「哪个实现好」
- `*-source` 说「源码在哪、怎么读」
- **但没人说「怎么把那段源码改掉、编出来、验回去」**

这一步是断的，所以「优化」经常停在「发现该改源码」就没了。

## 二、六步（固定顺序，每步一个产物）

```
① 定位 → ② 取源 → ③ 编译 → ④ 改(patch) → ⑤ 回归 → ⑥ 留痕
```

| 步 | 做什么 | 产物 | 判定 |
|---|---|---|---|
| ① 定位 | 从 `card.variants[].file_hint` 找到产物/库名，再到对应 `*-source` skill 找源码入口 | 一句 `库/文件/函数` | 答不出具体文件就不进下一步 |
| ② 取源 | **优先用本地已有**（见 §三）；没有再拉；记录 ref/commit | `SOURCES.md` 一行 | 必须写 commit/tag，不写就是不可复现 |
| ③ 编译 | gfx938 目标，记录编译器版本与 flags | `build.log` + flags 快照 | 复用 `op-hip-compile` 的 `derive_flags.py` |
| ④ 改 | **patch 形式，不直接改树** | `patches/NNNN-<name>.patch` | `patch -p1 --dry-run` 必须干净套用 |
| ⑤ 回归 | A/B（`opbench` 口径）+ 三闸（`opcheck` 正确性/不变量/padding） | `results/*.json` | 没有这三样不算验证过 |
| ⑥ 留痕 | `reproduce.sh` + `ledger.md` 追加一行 | 可复跑的命令 | 别人照着跑能出同样数字 |

## 三、本地已有源码（**不要重拉**）

| 库 | 位置 | 可做的案例 |
|---|---|---|
| LightOp | `/workspace/lightop/*.tgz`（源码 tar）、`lvfl_glm/`、`lzh_glm_new/`（已解包） | 改 `lightop` 的 attention/mqa 模块 的 persistent 分组 |
| FlashMLA | `/workspace/FlashMLA-das`、`/workspace/FlashMLA-upstream` | 改注意力计算，das 与 upstream 对拍 |
| MHC | `/workspace/mhc_kernels` | 改 MHC 三件套 |
| hbird | `/workspace/hbird-gemm` | 改 grouped GEMM / persistent dispatch |

> 重拉一次 LightOp 源码是 266MB。**先看有没有现成的**，这是 §二 第 ② 步的第一条纪律。

## 四、patch 的三条硬规矩

1. **永不直接改源码树**。所有改动落成 `patches/NNNN-name.patch`，
   用 `patch -p1` 套用、`patch -R` 回滚。
   直接改树的问题不是「不方便」，是**改了什么都说不清**。
2. **改完必须能干净套用**：`patch -p1 --dry-run` 无 fuzz。
   有 fuzz 说明基线漂了，先核对 commit。
3. **一条 patch 只做一件事**。混两个改动的 patch 无法单独回滚，
   回归失败时你不知道是哪一半坏的。

## 五、回归的三件套（缺一件都不算验证过）

| 件 | 命令 | 过不了怎么办 |
|---|---|---|
| 正确性 | `opcheck correctness` | 不算优化，直接回滚 |
| 不变量 | `opcheck invariants` | 同上；determinism 不过尤其严重 |
| A/B | `opbench` + `claim_gate` 三条硬门 | 噪声内 → 记 `INCONCLUSIVE`，不是「快了」 |

## 六、一个完整的真实闭环（LightOp MQA）

```bash
# ① 定位：card.variants[0].file_hint → 目标库的 attention/mqa 入口（paged_mqa_logits）
# ② 取源：本地已有
ls /workspace/lightop/*.tgz        # 源码 tar，不要再 git clone
# ③ 编译：按 gfx 显式指定（漏了会用错架构）
PYTORCH_ROCM_ARCH=gfx938 python3 setup.py install
# ④ 改：写 patch 而不是改树
mkdir -p patches && $EDITOR patches/0001-<name>.patch
cd <src> && patch -p1 --dry-run < patches/0001-<name>.patch
# ⑤ 回归
python3 scripts/opcheck.py correctness --contract <card>/contract.yaml --variants <tag>
python3 scripts/opbench.py ...
# ⑥ 留痕：把上面四步固化成卡目录下的 reproduce.sh，一条命令跑完
<卡目录>/reproduce.sh all
```

## 七、已知的坑

| 坑 | 机制 | 判据 |
|---|---|---|
| 改完忘了重编 | 测的是旧 `.so`，数字全是幻觉 | 编译产物 mtime 必须晚于 patch 时间 |
| 编译 flag 集变了 | flag 是测量的一部分，漏一条能差 16% | 快照 flags，与基线逐字比对 |
| 直接改树 | 无法单独回滚，也说不清改了什么 | 只允许 patch |
| 重拉源码 | 本地已有却去 clone，浪费 266MB 还引入 ref 差异 | 先查 §三 |

## 验收

1. `patches/NNNN-*.patch` 能用 `patch -p1 --dry-run` 干净套用（无 fuzz）。
2. 编译产物 mtime 晚于 patch 应用时间（证明测的是新产物）。
3. `opcheck correctness` 与 `invariants` 都通过。
4. A/B 判定经 `claim_gate` 三条硬门，噪声内如实记 `INCONCLUSIVE`。
5. `reproduce.sh all` 能一条命令跑出同样数字。
6. `SOURCES.md` 记录了源码 ref/commit 与编译器版本。

## 不适用场景

只调参数不改源码（走 `op-optimize` 的候选闭环）；
从零手写新 kernel（走 `op-hip-compile`）；
只测量不改动（走 `op-bench` / `op-profiling`）；
改的是上游社区代码但不需要本平台验证（那样不需要本 skill 的编译与回归两步）。
