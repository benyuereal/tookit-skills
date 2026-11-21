# PMC CSV 解析参考实现

**可执行版在 [`scripts/pmc.py`](../../../../../scripts/pmc.py)**（包根共享脚本，标准库，无第三方依赖）。
本文件只是它的说明页，**不另存一份代码**——同一套规则只有一个权威点，两处各写一份必然漂。

```bash
python3 -I scripts/pmc.py <pmc.csv>                       # 自动排噪声、按单次 GRBM 选主 kernel
python3 -I scripts/pmc.py <pmc.csv> --kernel <name> --cu <N> --json
python3 -I scripts/pmc.py --selftest                      # 不需要 CSV / GPU 的规则自检
```

它只做四件事：**选对行**（单次 launch 中 GRBM 最小，首行虚高 ~40%）、**排噪声**
（`_ZN2at`/`_ZN7rocprim`/`_ZN12_GLOBAL__N_1`/`_ZN2c10`/`_ZN3c10`）、**按实例求和**
（`TA_*`/`TCP_*`/`TCC_*` 的 `[n]` 跨 SE 求和，取 max 会得 352%）、**打印原值**
（CU 数等硬件常数不写死，缺了打 `unknown(<原因>)`）。不解释、不判定；判读口径见
`Plugins/ops/skills/op-profiling/SKILL.md` §四/§五。
