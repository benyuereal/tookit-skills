# dcu-vmfault-coredump — HSA VMFault 崩溃取证

从海光 DCU / ROCm HSA VMFault 崩溃（HSA_STATUS_ERROR_EXCEPTION 0x1016 / TRAPSTS=0x80000000）日志里
定位 faulting kernel 名 + args ABI 解码 + gpucore ELF 内存快照读越界值。回答「哪个 kernel、哪个张量、
哪个索引值」——当 hy-coredump faultinfo 只看不出名堂时。

## 核心内容
- HSA QUEUE ANALYSIS 解析 faulting kernel 名 + args hex dump
- args ABI 解码（知道哪个张量）
- gpucore ELF 内存快照读越界值
- 驱动自带取证工具（<device-A> 实测）：`hycoredump`（core 解析，`-r` 指 DTK 目录）、`hycapture`（运行中进程 HCU 状态抓取）、`hiplogdump`（HIP 日志解码）

## 目录
```
dcu-vmfault-coredump/
├── SKILL.md                    取证流程
├── scripts/                   classify_trap.py 等解码脚本
└── references/               crash-patterns/gpucore-snapshot/kernel-abi/queue-analysis
```

## 快速用法
HSA_STATUS_ERROR_EXCEPTION 0x1016 / VMFault / 越界访问崩溃时读 `SKILL.md`；补强 `dcu-hipprof` CoreDump 域。
