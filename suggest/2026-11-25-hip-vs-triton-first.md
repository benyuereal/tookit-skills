# 建议：性能题不该默认上手 Triton，高要求时应先评估 HIP

> 提出人：opbench 评测（三臂对照实验执行者）
> 日期：2026-11-25
> 依据：`norm-fused-add-rmsnorm-l1` 三臂实测（A 裸 CC / C toolkit-skills / D SkillHub）
> 状态：**建议**，请 toolkit-skills 维护者评估

---

## 一、观察到的事实

### 1.1 三臂都没写过一行 HIP

| 臂 | 是否写 HIP kernel | 是否写 Triton | 最终比值 |
|---|---|---|---|
| A（裸 Claude Code） | ❌ 0 个 | ✅ 8 个脚本 | 1.298× |
| C（toolkit-skills） | ❌ 0 个 | ✅ 2 个 | 1.297×（自研 triton-fused） |
| D（SkillHub） | ❌ 0 个 | ✅ 4 个 | 1.303× |

> 注：脚本里出现的 `hip` 字样全部是 `torch.version.hip`（打印版本号），
> **没有任何一处 `__global__` / `hipcc` / `hipLaunchKernel` 的真实 kernel**。

### 1.2 但 skill 里明明有现成的 HIP 资产，而且更快

`/workspace/toolkit_skills/Plugins/ops/skills/op-rmsnorm-fused/runs/forged/`
下有 4 个已生成的 HIP kernel（`auto-fusedadd-h4096-t{64,128,256,512}-v*.hip`），
文件头写得很清楚：

```
// 已做对的（不要改坏，这两条是本实现比厂商快的全部原因）：
//   1. (x+r) 留寄存器 —— 不落 LDS，省掉整行往返
//   2. 16 字节向量读写（uint4 = 8 个 bf16）
```

实测（`runs/loop/20261124.jsonl`，base = lightop 0.2019ms）：

| 变体 | 耗时 | vs base |
|---|---|---|
| `auto-fusedadd-h4096-t64-v8` | **0.1805 ms** | **-10.6%** |

而三臂这次用 Triton 拿到的最好成绩是 **0.0866 ms**（M=4096）——
两个数不可直接比（M 不同、bench 口径不同），但**关键点是：
HIP 这条路在本机是通的、而且有现成可用的产出**，三臂却一次都没走。

### 1.3 这不是某一道题的问题

`op-hip-compile/SKILL.md` 的 description 里写着它解决什么：

> 手写 kernel「数值对、逻辑对，但比对标实现慢 10~30%」，十次里有九次不是算法，
> 是**编译选项没镜像全**。

也就是说：**toolkit-skills 已经积累了「怎么写好 HIP」的知识**，
但没有任何一条规则告诉 agent「**什么时候该考虑 HIP**」。

---

## 二、问题定位

现在的 skill 触发链是：

```
题目（性能优化）
  → op-optimize（选实现 / 调参 / 候选队列）
  → op-bench（计时口径）
  → op-hip-compile（编译调优：**只在已经写了 HIP 之后**才被用到）
```

**缺了最前面一环**：`op-optimize` 在枚举候选时，默认从「已装的库」里找，
找不到就退到 Triton，**从不把「手写 HIP」列为候选方向**。

`op-rmsnorm-fused/candidates.yaml` 里三个方向的 `expect` 全是负的：
```
fuse-residual       -20%
inplace-residual    -11%
hidden-non-pow2      -3%
```
**没有一条是「手写 HIP 可能突破现有天花板」**。

---

## 三、建议的改动

### 3.1 在 `op-optimize` 的候选枚举里加一条判据（核心建议）

建议在 `op-optimize/SKILL.md` 的「方向枚举」一节加：

```
## 何时把「手写 HIP」列为候选

满足任一条，就必须把 HIP 作为候选方向之一，不能只靠 Triton：

1. 达标线要求 > 现有最好实现的 1.2×
   （Triton 在访存受限算子上通常只能到 85~95% 带宽，特化 kernel 可到 115%+）
2. 算子有「寄存器驻留」的天然机会
   （如 elementwise + 归约融合：中间结果不落 LDS/HBM，全程在寄存器）
3. 现有实现对某个固定形状有冗余往返
   （读写同一个张量两次、中间张量可省）
4. Triton 版本已尝试且未达标

反例（不需要 HIP）：
- 纯 GEMM / 纯 attention：库实现已接近硬件上限，手写没有空间
- 被 launch 地板支配的小算子：瓶颈在 launch，不在 kernel 内部
```

**判据要能被机器判**，建议写成可执行形式：

```python
# op-optimize 候选枚举时
if best_ratio_vs_baseline < 1.2 or current_util < 0.9:
    candidates.append({"kind": "hip", "reason": "..."})
```

### 3.2 在 `op-rmsnorm-fused` 这类「校准件」卡里，把 HIP 变体摆出来

`runs/forged/*.hip` 现在藏在 `runs/`（gitignore 的目录），**agent 看不到**。
建议把这些已验证的 HIP 实现挪到卡的正式位置，并在 `variants.yaml` 登记：

```yaml
- name: fused-hip-vec8
  source: runs/forged/auto-fusedadd-h4096-t64-v8.hip（kernel_scaffold.py 生成）
  status: candidate
  measured: "0.1805 ms（base lightop 0.2019，-10.6%）"
  applies_to: [prefill-8k]
  constraints:
    - "(x+r) 必须留寄存器，不落 LDS"
    - "16 字节向量读写（uint4 = 8 个 bf16）"
    - "VEC/THREADS 需按形状扫，改一个就要重编译"
```

**理由**：agent 看不到 `runs/`（被 `.gitignore` 且常被清理），
等于把最有价值的产出藏了起来。

### 3.3 在 CLAUDE.md 的「优化闭环」里加一句分流

现在 C 臂的 CLAUDE.md 只说「用 loop.py 推进」。建议加：

```
## 实现路线选择（先判断再做）

优化一个算子前，先按这个顺序排除：
1. 有没有现成库实现能达到 >90% 带宽/算力？
   → 有，用它，别写 kernel
2. 没有，且 roofline 显示还有 >1.2× 空间？
   → **先评估手写 HIP**（寄存器驻留 / 向量化 / 省往返），再考虑 Triton
3. Triton 达不到，且已尝试？→ op-hip-compile 调编译选项
4. 都被 launch 地板支配？→ 别优化 kernel，做融合/减 launch

**不要默认从 Triton 起步。** Triton 的优势是快速迭代，
劣势是生成代码的寄存器/向量化控制不如手写。
访存受限的 elementwise/归约类算子，手写 HIP 通常能多拿 15~25% 带宽。
```

### 3.4 补一个「HIP 最小可跑模板」

`op-hip-compile` 讲的是「编译调优」，但没有「从零写第一个 HIP kernel 并跑起来」的模板。
建议补一个 `templates/minimal.hip` + `templates/load.py`（ctypes 加载），
让 agent 第一次写 HIP 时不用自己解决编译/加载链路。

现成的可以直接拿：`runs/forged/auto-fusedadd-h4096-t64-v8.hip` + 同名 `.py`。

---

## 四、预期收益与风险

### 收益

- 访存受限算子（norm / act / kv 类）在手写 HIP 后可能多拿 **15~25%** 带宽
- 把已经积累的 HIP 编译知识（`op-hip-compile`）真正接进主流程
- 避免 agent 在 Triton 上反复调参却撞不到天花板

### 风险（必须一起写进 skill）

| 风险 | 缓解 |
|---|---|
| HIP 编译慢，迭代成本高 | 先 Triton 探路，确认方向对了再写 HIP |
| 手写 kernel 容易算错 | 必须过 `done_when` 的 correctness 门，不能只看速度 |
| 例子里的 0.1805ms 是 M=8192 口径，和三臂的 0.0866（M=4096）不可直接比 | 建议文档里注明口径，避免误读 |
| 不是所有算子都值得 | 判据里明确排除纯 GEMM / attention / launch-bound |

---

## 五、建议的优先级

| # | 改动 | 成本 | 收益 |
|---|---|---|---|
| 1 | `op-optimize` 加「何时考虑 HIP」判据（3.1） | 低 | **高**——直接改变 agent 的路线选择 |
| 2 | 把 `runs/forged/*.hip` 挪到卡内正式位置并登记（3.2） | 低 | 高——现成资产从隐藏变可见 |
| 3 | CLAUDE.md 加路线分流（3.3） | 低 | 中——给第一次接触的 agent 一个明确顺序 |
| 4 | 补 HIP 最小模板（3.4） | 中 | 中——降低首次尝试门槛 |

**1、2 两条成本最低、收益最直接，建议先做。**

---

## 六、附录：证据位置

```bash
# 三臂都没写 HIP（只有 torch.version.hip）
grep -rl "__global__\|hipcc" /tmp/opbench-run/arm-*/norm-fused-add-rmsnorm-l1/attempt-1/scripts/
# → 空

# 现成 HIP kernel（4 个变体，带参数扫描）
ls /workspace/toolkit_skills/Plugins/ops/skills/op-rmsnorm-fused/runs/forged/*.hip

# 实测记录（-10.6%）
grep auto-fusedadd /workspace/toolkit_skills/Plugins/ops/skills/op-rmsnorm-fused/runs/loop/20261124.jsonl

# HIP 编译知识（已有，但没接上）
cat /workspace/toolkit_skills/Plugins/ops/skills/op-hip-compile/SKILL.md
```
