---
name: gated-add
description: "门控残差加。输入 x:bf16[M,H], gate:bf16[M,H]，输出 y:bf16[M,H]。用途：(bootstrap 生成，待补真实场景描述)。触发场景：(待补)。不适用：(待补)。"
---

# gated-add

y = x + gate * s ；s 为标量 scale

## 输入输出

- 输入：x:bf16[M,H], gate:bf16[M,H]
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
