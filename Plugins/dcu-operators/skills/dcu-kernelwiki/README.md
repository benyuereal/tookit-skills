# dcu-kernelwiki — 海光 DCU Kernel 优化知识库

> agent 向规范见 [`SKILL.md`](./SKILL.md)。本文件面向使用者（人/agent 概览）。

## 它是什么

把分散在 内部文档 文档、上游仓库、实测里的 **海光 DCU 算子优化事实与坑** 结构化组织，
让 agent 能像查 `ROCmKernelWiki` 一样"查一下 DCU 上某算子怎么写才对"。

设计参考 `jhinpan/ROCmKernelWiki`（AMD CDNA 版，带证据飞轮与真机验证），此处 **DCU 化**。

## 核心纪律

1. **架构作用域隔离**：DCU 多代架构（K100/Z100→`gfx906`、K100-AI→`gfx928`…），
   每条知识标适用 `gfx`；编译前用 `dcu-kernel-dev/scripts/dcu_detect_arch.sh` 现查，不假设。
2. **溯源 + 证据**：每条"已验证事实"挂来源（内部文档 文件 ID / 上游 commit / 设备+DTK+日期）。
3. **真机验证**：涉及数值/性能/指令布局的结论，须在挂载 `/opt/hyhal:ro` + `/dev/kfd` 的真机验证。
4. **性能数字不入本库**：基线 ns / GB/s 等进 `dcu-perf-cases`，本库只留方法与结论。

## 已建知识条目（`wiki/` 目录）

| 文件 | 内容 |
|---|---|
| `wiki/matrixcore_layout.md` | K100-AI Matrix Core 指令集与 wavefront 数据布局（源自 XDZS2025 反编译） |
| `wiki/lds_bank_conflict.md` | LDS bank conflict 与 padding 规避（BM×(BK+2) padding 法） |
| `wiki/rmsnorm_precision_cliff.md` | RMSNorm bf16 舍入悬崖（block 增大→diff>0.03），判据 <0.03 放行 |
| `wiki/tilelang_dcu_known_issues.md` | TileLang 在 DCU 的 12 项已知问题（target=hip / warp_size=64 / alloc_shared 竞争 等） |
| `wiki/optimization_loop_protocol.md` | profile 驱动优化循环协议（METRIC 行 / profiler 证据 / MAD 噪声判定，DCU 化） |

## 怎么用

- agent 遇到"DCU 上某算子该怎么写/有什么坑"时，先查 `wiki/` 对应条目。
- 发现新事实：在 `wiki/` 建/改文件，frontmatter 填 `source` + `verified_on`，不确定标 `unverified`。
- 性能数据不要写这里，转 `dcu-perf-cases`。
- 提交到 dcu skill 仓（master），保持与源文档/实测同步。

## 相关

- 方法学（开发两条路径、调试工具链、精度陷阱）→ `dcu-kernel-dev`
- 优化闭环/留痕 → `dcu-op-optimization`；度量 → `dcu-perf-profiling`
- 性能案例库 → `dcu-perf-cases`
