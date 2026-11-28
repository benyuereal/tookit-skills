---
name: silu-mul
description: "SiLU 门控乘。输入 x1:bf16[M,H], x2:bf16[M,H]，输出 y:bf16[M,H]。用途：(bootstrap 生成，待补真实场景描述)。触发场景：(待补)。不适用：(待补)。"
---

# silu-mul

y[i,j] = silu(x1[i,j]) * x2[i,j]；silu(v)=v*sigmoid(v)，fp32 计算后转 bf16

## 输入输出

- 输入：x1:bf16[M,H], x2:bf16[M,H]
- 输出：y:bf16[M,H]
- 搬运量：201.3 MB（pattern=elementwise-2in）

## 状态

bootstrap 生成，**尚未实测**。补齐顺序：

1. `_common.py` 里的 `reference()` —— 现在是 `NotImplementedError`。
   参考实现是正确性的唯一锚，不要写一个「看起来能跑」的版本。
2. `build_inputs()` 的数值分布 —— 要覆盖真实场景的取值范围。
3. `VARIANTS` 里填真实实现（库调用或手写 kernel）。
4. 跑一次基线拿到 `t_actual`，再 `floor.py` 出 roofline。
5. `lang_select.py` 判语言路线（**它需要 roofline，所以第 4 步不能省**）。

## 纪律

- 分母来源见 `floor_inputs`，算错方向就反。
- 每次对比必须 base 重测。
- 本卡已登记进 `data/ops/registry.json`、`Plugins/ops/SKILL.md` 索引表、
  `Plugins/ops/op-atlas/INDEX.md`；状态变化要三处一起改，
  `python3 -I scripts/card_sync.py` 会核对。
- `baseline_ref` 现在是 `no_baseline_reason`：只有 ref-heavy 时算不出有意义的倍数。
  补上真基准（competitor / best_known）之前，本卡不许写任何「快多少」。
