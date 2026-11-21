---
name: dcu-hy-smi
description: hy-smi查询监控：温度/功耗/显存/拓扑/进程，DCU巡检必备
version: 1.2.3
author: DCU Engineer
license: MIT
kind: tool
---

# dcu-hy-smi — 海光 DCU 系统管理接口

> 触发词：海光DCU、DCU、HCU、hy-smi、DCU监控、DCU拓扑、卡间亲和性、rocm-smi对应

`hy-smi` 是海光 DCU（C-3000 系列 HCU）的 SMI 工具，**对应 AMD ROCm 生态的 `rocm-smi`**。
它是 **DCU 驱动自带的**（安装驱动后自动位于 `/opt/hyhal/bin/hy-smi`，宿主机与容器内均有），
不一定在 PATH 里，但一定在那个绝对路径。

> **运行语境**：本技能面向 DCU 远端主机（如 `ssh <ssh-alias>`，别名在用户 `~/.ssh/config` 中定义）；Windows 本机无 `/opt/hyhal/bin/hy-smi`，命令一律经 ssh 在远端执行。本机语境的通用巡检/工作流见 `hy-smi-tool`。

## 工具定位

- **巡检/监控**：温度、功耗、利用率、显存、时钟、进程
- **拓扑采集（绑核/集合通信关键）**：卡↔NUMA 亲和性、卡间链接类型、可达性、跳数、权重
- **故障排查**：RAS 错误、XGMI 错误计数、异常信息
- 危险设置类操作（超频/电压/卸载驱动）见原 skill 安全警告，**默认不做**

## 执行入口（实测确认）

```bash
# 宿主机（驱动装好即有，不一定在 PATH）
/opt/hyhal/bin/hy-smi --showtoponuma --json

# 容器内（若宿主机无 hy-smi 在 PATH）
docker exec <dcu容器> hy-smi <options>

# 远程执行（密码经 SSHPASS 环境变量，见 dcu-ssh Method 1b；勿用 sshpass -f 直接喂 JSON）
export SSHPASS=$(python3 -c "import json,os; c=json.load(open(os.path.expanduser('~/.config/dcu-toolkit/credentials.json'))); print(c['<device_name>']['password'])")
sshpass -e ssh -o StrictHostKeyChecking=no root@<device_ip> "/opt/hyhal/bin/hy-smi -a --json"
```

**注意**：宿主机（如 Ubuntu 系统）PATH 无 hy-smi，但 `/opt/hyhal/bin/hy-smi` 存在，直接绝对路径调用即可，
不必非走 docker exec（原 rookie-J/hy-smi README 的 SSH 示例默认走 docker exec，但实际宿主机直跑更省事）。

## 快速常用命令

| 用途 | 命令 |
|------|------|
| 设备概览 | `hy-smi` 或 `hy-smi -a` |
| 设备ID/序列号 | `hy-smi -i --showserial --showuniqueid` |
| 温度/功耗 | `hy-smi -t -P` |
| 利用率(HCU/CU) | `hy-smi -u --showhcuutil --showcuutil` |
| 显存使用 | `hy-smi --showmemuse --showmemavailable` |
| 时钟频率 | `hy-smi -c -g --showclkfrq` |
| 运行进程 | `hy-smi --showpids` |
| JSON 输出 | `hy-smi -a --json` |

## 拓扑与连接（集合通信绑核关键 — 优先用这些）

| 选项 | 含义 |
|------|------|
| `--showtopo` | 硬件拓扑总览 |
| `--showtoponuma` | **卡↔NUMA 节点亲和性**（最常用，解决 lspci 失真问题） |
| `--showtopoaccess` | HCU 间链接可达性 |
| `--showtopoweight` | HCU 间相对权重 |
| `--showtopohops` | HCU 间跳数 |
| `--showtopotype` | 链接类型（XGMI / PCIe） |
| `--shownodeid` | 节点 ID |
| `--showxhclstatus` | xhcl 链接状态 |
| `--showxgmierr` | XGMI 错误计数 |

> ⚠️ **`--showtoponuma` 是获取卡↔NUMA 亲和性的唯一可信源**。`lspci -vvv` 在海光上所有 PCI
> 设备都报 `NUMA node: 0`，失真（实测跨 NUMA 机型的 card4-7 实际在 NUMA4，lspci 却报 0）。
> 详见 dcu-topology skill 的 Pitfall #6/#7。dcu-topology 的 `dcu-topology/../dcu-topology/scripts/collect_topology.sh` 已内置
> 用 hy-smi 采集，无需手动跑。

## 带宽相关

- `-b, --showbw` - PCIe 带宽
- `--showdfbw/--showmembw` - DF/显存读写带宽
- `--showumcbw` - UMC 读写带宽
- `--showxhclbw` - xhcl 带宽
- `--channel/--link/--direction/--delay/--count/--bwdetail` - 带宽细查参数

## 健康与异常

- `--healthcheck` - 健康状态检查
- `--showrasinfo BLOCK` - RAS 使能/错误计数
- `--showexceptioninfo` - 异常信息
- `--showxgmierr` - XGMI 错误计数

## 输出格式选项

- `--json` - JSON 格式（脚本解析首选）
- `--csv` - CSV 格式
- `--loglevel LEVEL` - debug/info/warning/error/critical

## 驱动管理与高级选项（<device-A> 实测，hy-smi 1.24.0 / 驱动 6.3.30）

> 来源：2026-08-28 在 <device-A> 设备（BW100X8，驱动 6.3.30-V1.4.1a）对 `/opt/hyhal/bin/hy-smi --help` 全量扫描 + 安全只读项实测。
> 以下选项在旧版 hy-smi 中可能不存在，**先 `hy-smi --help` 确认本机支持再用**。

### 只读查询（安全，可直接执行）

| 选项 | 含义 | <device-A> 实测 |
|------|------|------|
| `--forchip` | 单行式精简概览（Temp/AvgPwr/Perf/PwrCap/VRAM%/HCU%/Dec%/Enc%/Mode） | ✅ 8 卡全出，巡检最省屏 |
| `--showdriverversion` | 内核驱动版本 | ✅ `Driver Version: 6.3.30-V1.4.1a` |
| `--showfwinfo` | 各 HCU 固件版本（MEC/MEC2/RLC/SDMA/SDMA2…） | ✅ 逐 HCU 列出 |
| `--showhswfw` | HSW（HySwitch）固件版本 | ✅ 无 HSW 机型输出 `No hsw dev found`（正常） |
| `--showmemvendor` | 显存颗粒厂商 | ✅ `samsung` |
| `--showvbios` / `-v` | VBIOS 版本 | 见快速命令表 |
| `--getautoloaddriver` / `--getextendbw` / `--getautorunhyckptd` / `--getdriverparams` / `--getstabledriverparams` | 查驱动自加载/扩展带宽/hyckptd 自启/驱动参数 | ⚠️ <device-A> 上输出为空（该驱动版本未实现或无值），不代表故障 |
| `-idmon <id> -sdmon <c/f/p/u> -cdmon <n>` | dmon 监控：指定卡、选数据（c 时钟/f 风扇/p 功耗温度/u 利用率）、次数 | ✅ `-idmon 0 -sdmon p -cdmon 1` 出 Temp(Hotpot)/Temp(Avg)/Avgpwr |
| `-ddmon/-odmon/-fdmon/-tdmon` | dmon 间隔秒数/带时间戳/落文件/文件格式(text/csv) | 配合上面用 |
| `--finegrain` | 细粒度占用（驱动 6.2.22+） | 见环境要求 |
| `--showevents TYPE --duration N` | 事件列表 | 只读 |

### 驱动加载管理（⚠️ 会改变系统状态，生产机先确认）

| 选项 | 含义 |
|------|------|
| `--loaddriver` / `--unloaddriver` | 手动加载/卸载驱动（重启后驱动未自加载时的补救，见 dcu-driver Pitfall #11） |
| `--setautoloaddriver <0/1>` / `--getautoloaddriver` | 驱动开机自加载开关 |
| `--setdriverparams <v>` / `--getdriverparams` | 驱动加载参数 |
| `--addstabledriverparams <v>` / `--getstabledriverparams` / `--clearstabledriverparams` | 稳定驱动参数增/查/清 |
| `--setextendbw <0-3>` / `--getextendbw` | 扩展带宽级别（0 关/1GB/2GB/4GB） |
| `--hcureset` | 重置指定 HCU（必须配 `-d` 指定卡，**高危**） |
| `--load FILE` / `--save FILE` | 从/向文件加载保存 Clock/Fan/Performance 设置 |
| `--resetclocks` / `--resetfans` / `--resetpoweroverdrive` / `--resethslerr` | 恢复默认时钟/风扇自动/功耗上限/清 HSL 错误计数 |
| `--autorespond <resp>` | 自动应答所有提示（官方标注 NOT RECOMMENDED） |

### hyckptd 管理（checkpoint 守护进程）

`--runhyckptd` / `--stophyckptd` / `--setautorunhyckptd <0/1>` / `--getautorunhyckptd`
——hyckptd 是驱动自带的 checkpoint 守护（`/opt/hyhal/bin/hyckptd`，`--log <path>` 指定日志、`-q` 静默）。

### 子命令：virtual（软件虚拟化 vdevice）

```bash
hy-smi virtual -h
# -create-vdevices <n>   创建 n 个虚拟设备
# -destroy-vdevices      销毁已创建的虚拟设备
# -update-vdevice <cu>,<memory>  更新 vdevice 配置（CU 数,显存 MiB；* 表示不更新该字段，需配 -v 指定目标 vdevice）
# -d <dev> -v <vdev>     指定物理卡/虚拟设备
```

### 子命令：mig（多实例 GPU 管理）

```bash
hy-smi mig -h
# -L 列出 GPU 及可用 MIG 实例；-i <gpu...> 指定设备
# -lgip 列出 GPU instance profiles；-lgipp 列出可能 placement
# -cgi/-ci 创建 GPU/Compute instance（-C 用默认 profile）
```

> ⚠️ virtual/mig 会改变设备切分状态，**只在明确需要虚拟化/多实例隔离的环境操作**，操作前先 `hy-smi virtual` / `hy-smi mig -L` 看现状。

## 实测参考（多机型验证）

```json
// 跨 NUMA 机型（Kylin）: hy-smi --showtoponuma --json
{"card0":{"(Topology) Numa Node":"0","(Topology) Numa Affinity":"0"},
 "card1":{...:"0"}, "card2":{...:"0"}, "card3":{...:"0"},
 "card4":{"(Topology) Numa Node":"4","(Topology) Numa Affinity":"4"},
 "card5":{...:"4"}, "card6":{...:"4"}, "card7":{...:"4"}}
// → 跨 NUMA 机型：卡0-3在NUMA0，卡4-7在NUMA4（跨socket，每socket 4卡）

// 单 NUMA 机型（Ubuntu）: 8卡全在NUMA0
```

**OAM+HSW 机型（BW1100，2026-09-15 <device-alias> 实测算力）**：`hy-smi --showtoponuma --json` 显示
**8 卡横跨 6 个 NUMA 节点（0,1,3,4,5,7）**，如 card0→node3、card1→node1、card2→node1、card3→node0、
card4→node7、card5→node5、card6→node5、card7→node4——**不是**双 socket 前 4 后 4 的分组，而是逐卡绑定到不同 NUMA 节点。
`--showtopotype` 全为 **HSW**、`--showtopohops` 全为 **1**（OAM 形态 8 卡经 HySwitch 全互联、无 PCIe/HSL 分组差异）。
**集合通信绑核不能假设单 NUMA，须按卡↔NUMA 亲和性逐卡绑定。**

解析 tip（bash）：
```bash
hy-smi --showtoponuma --json | grep -oE 'card[0-9]+|"\(Topology\) Numa Node": "[0-9]+"' \
  | awk '/card[0-9]+/{c=$0} /Numa Node/{n=$NF; gsub(/"/,"",n); print c, n}'
```

## 占用进程定位与按容器反查（巡检最实用 ⭐）

DCU 上任务基本跑在 docker 容器里。光看概览只能知道"哪张卡有占用"，要定位"**谁**在占卡"必须查进程并反查容器。

```bash
# 1. 查占用进程（PID / VRAM 占用）
/opt/hyhal/bin/hy-smi --showpids
# 输出每个占用 DCU 的 PID，带 HCU Index / GPUID / VRAM USED(MiB) / VRAM USED(%)。
# 注意 VRAM USED(%) 可能为 inf（相对卡总显存算出的异常值），以 VRAM USED(MiB) 为准。

# 2. 按容器反查占用者：从 hy-smi 拿 PID → 查 cgroup 里的 docker 容器 ID → docker inspect 容器名
/opt/hyhal/bin/hy-smi --showpids 2>/dev/null \
  | awk '/^PID:/ {print $2}' \
  | xargs -I {} sh -c "cat /proc/{}/cgroup 2>/dev/null | grep -oP '(docker/\K[^/]+)|(docker-\K[^.]+)' | head -1" \
  | while read cid; do
      echo "${cid:0:5} -> $(docker inspect --format='{{.Name}}' $cid 2>/dev/null | sed 's/^//')"
    done
```

**实测（2026-07-22）**：直接定位到占用卡 0 的是某 vLLM 容器（PID 2083159，52703 MiB ≈ 80% 显存），比光看概览精确得多。

> 注意：`--showpids` 可能列出多个 PID（含仅占用 1 MiB 的附属进程），同一容器会重复出现，认 VRAM USED(MiB) 大的那个 PID 才是真正占显存的。

**卡片↔容器归属对账（要知道"哪个容器被分到哪几张卡"，哪怕此刻没算）**：`--showpids` 给的是**当前 live 占用**（进程正跑在哪些卡上）；若要看**容器被分配/可见哪些物理卡**，查容器 env 的可见设备：`docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' <c> | grep -E 'ROCR|HIP|HSA_.*VISIBLE_DEVICES'`（如 `ROCR_VISIBLE_DEVICES=0,1`）。⚠ **别用 `fuser /dev/dak*` 或 `fuser /dev/dri/renderD*` 反查卡**——海光 DCU 没有 `/dev/dak*` 节点，`renderD*` 由 kfd 内核态持有、`fuser` 也返回空，两者都不可信；卡上的实际占用以 `hy-smi` VRAM% / `--showpids` 为准，env 可见设备只作"分配意图"参考（进程真用哪张以 hy-smi 为准）。

**归属核实（决定能否停容器前必做）**：容器常带 owner label 直接说明是谁的：`docker inspect -f '{{index .Config.Labels "io.sglang-tsoc.owner"}}' <c>`（label 名随团队而异，不确定就 `docker inspect -f '{{json .Config.Labels}}' <c>` 全量列一遍）。owner 非空且非本方 → **只读操作（ps/logs/inspect/stats），停容器必须用户拍板**。共享测试机常态是多人的服务占卡：VRAM 高但 HCU=0 多是遗留引擎，但遗留引擎也是别人的，不擅自停。

### 典型判断流程（"设备有没有人用"）

```bash
# 步骤1：概览看哪张卡有占用
/opt/hyhal/bin/hy-smi
# 步骤2：谁在线
who            # 看登录会话来源 IP / 时间
w              # 看挂机时长 + 当前命令
# 步骤3：占卡的具体容器（上面反查指令）
```

- `VRAM%` 高但 `HCU%`=0、`AvgPwr`≈60W：进程常驻占显存但当前没计算（如遗留的推理服务）
- 卡 1~7 `VRAM%`=0：完全空闲，可直接抢占

## 封装脚本（本 skill 提供）

`scripts/hy_smi_query.sh` — 自动探测 hy-smi 路径（绝对路径→PATH→docker容器兜底），
一行出结构化拓扑/监控，避免每次手敲 docker exec：
```bash
bash scripts/hy_smi_query.sh topo      # 卡↔NUMA + 卡间跳数/hops（JSON）
bash scripts/hy_smi_query.sh monitor   # 温度/功耗/利用率概览
bash scripts/hy_smi_query.sh <任意hy-smi参数>   # 透传，如 --showpids --json
```

## 环境要求

- 驱动版本 6.2.22+（支持 `--finegrain` 等选项）
- 需要 root 权限（部分操作）；容器内需 `--privileged` 或映射 `/dev/kfd`、`/dev/dri`
- 驱动加载检查：`lsmod | grep -E "hydcu|hycu"`

## Pitfalls（实测确认 — 多机型）

1. **hy-smi 不在 PATH 但驱动自带**：宿主机（如 Ubuntu 系统）PATH 无 `hy-smi`，但 `/opt/hyhal/bin/hy-smi` 存在。优先绝对路径，不必非走 docker exec。
2. **`--showtoponuma` 是唯一可信卡↔NUMA 源**：`lspci -vvv` 在海光上所有 PCI 设备都报 `NUMA node: 0`（失真）。跨 NUMA 机型实测 card4-7 实际在 NUMA4，lspci 却报 0。**绑核/集合通信亲和性必须以 hy-smi 为准**（dcu-topology 已内置）。
3. **仅 `--showtoponuma` 支持 `--json`**：`--showtopohops/--showtopoaccess/--showtopoweight/--showtopotype` 不支持 `--json`（hy-smi 报 `This command does not support to set print format`），输出为表格。解析这些需按表格提取（如 hops 矩阵）。
4. **hop 矩阵印证拓扑**：单 NUMA 机型组内 1 hop、跨组 2 hop（全同 NUMA 但物理分两组）；跨 NUMA 机型组内 2 hop、跨 NUMA 组 3 hop（真跨 socket）。hop 数差异直接反映 NUMA 跨 socket 代价，是 RCCL 绑核策略依据（见 dcu-rccl-test）。
5. **容器内执行需 `--privileged` 或映射 `/dev/kfd` `/dev/dri`**：否则 `Permission denied`。
6. **`--showpids` 的 VRAM USED(%) 可能为 `inf`**——以 `VRAM USED(MiB)` 判断真实占用量。
7. **PID 反查容器依赖 docker**——若任务跑在裸机（非容器），`docker inspect` 会找不到，退回到 `ps -p <pid> -o cmd` 看命令行。
8. **多 PID 同容器重复出现正常**——认显存占用大的那个 PID。
9. **⚠️ PID→容器反查指令只能在物理机（宿主机）执行**：容器内看不到宿主的 docker 层级，`/proc/<pid>/cgroup` 里没有宿主 docker ID，且容器内通常没有 `docker` 命令，这条会失效。正确做法：SSH 到**宿主机**跑 `hy-smi --showpids` + 反查；若当前在容器里，先 `exit` 回宿主。（实测：宿主机直接跑成功定位到 `<container>`。）

> 参考源：https://github.com/rookie-J/hy-smi （本 skill 命令参考基于此仓库，已补充多机型实测与本包集成封装）

## References
- 官方文档中心《HY-SMI CTL 使用手册》版本树 `74da671f-1b65-11f1-a4a0-0242ac150003`（22 叶）。取全文：`GET /api-doc/document/TreeDocument?VersionUUID=<树uuid>` 列目录、`DetailDocument?UUID=<叶uuid>`（匿名，2026-10-14 实测）；总索引见包内 `../_dcu-toolkit-shared/references/sourcefind/doc-center-index.md`。

- 原 skill 仓库：https://github.com/rookie-J/hy-smi （含完整选项参考 + examples.sh，本 skill 的 `scripts/examples.sh` 即来源于此）
- 关联：dcu-topology（拓扑采集已内置 hy-smi，`dcu-topology/../dcu-topology/scripts/collect_topology.sh` 用 --showtoponuma 为权威源）、dcu-rccl-test（集合通信绑核，hops/亲和性为 NCCL 映射依据）
- 互补：`hy-smi-tool`（skills-dcu-autoopt 整合，2026-08-19）——通用 hy-smi 巡检/工作流（健康检查/性能监控/拓扑/进程/JSON 采集五套 workflow + RAS/固件/错误处理），偏「通用工具手册」；本 skill 偏「DCU 实测坑 + 按容器反查 + 封装脚本」。两者命令集高度重叠，按需取用，不冲突。

