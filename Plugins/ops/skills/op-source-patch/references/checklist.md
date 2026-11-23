# 六步检查清单

| 步 | 检查项 | 不过怎么办 |
|---|---|---|
| ① 定位 | 能说出具体 `库/文件/函数` | 答不出就别进下一步，先回卡片的 `variants[].file_hint` |
| ② 取源 | `SOURCES.md` 有 ref/commit | 没写 = 不可复现，补上 |
| ② 取源 | **先查本地已有** | 重拉一次 LightOp 是 266MB |
| ③ 编译 | 编译产物 mtime 晚于 patch 应用时间 | 否则测的是旧 `.so`，数字全是幻觉 |
| ③ 编译 | flags 快照与基线逐字比对 | 漏一条 flag 能差 16% |
| ④ 改 | `patch -p1 --dry-run` 无 fuzz | 有 fuzz 说明基线漂了 |
| ④ 改 | 一条 patch 只做一件事 | 混多个改动无法单独回滚 |
| ⑤ 回归 | correctness + invariants + A/B 三样齐 | 缺一件不算验证过 |
| ⑤ 回归 | A/B 过 `claim_gate` 三条硬门 | 噪声内记 `INCONCLUSIVE`，不是「快了」 |
| ⑥ 留痕 | `reproduce.sh all` 能跑出同样数字 | 跑不出就是口号 |
