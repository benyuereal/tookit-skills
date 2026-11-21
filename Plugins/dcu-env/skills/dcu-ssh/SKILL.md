---
name: dcu-ssh
description: "宿主机级 SSH：远程连 DCU 服务器执行命令、传文件、长任务后台轮询。范围只限宿主机——容器内/容器之间的 SSH 免密一律走 dcu-docker，多机集合通信测试本身走 dcu-rccl-test。触发场景：远程连一台 DCU 宿主机跑 hy-smi 或 benchmark、scp 传脚本、配宿主机免密登录。不适用：两个容器之间（或容器↔宿主机之间）配 SSH 免密走 dcu-docker——哪怕用的就是 SSH；本机环境检测走 dcu-env。"
version: 1.1.6
author: DCU Engineer
license: MIT
---

# DCU SSH 远程连接管理

## Overview

指导智能体如何安全地通过SSH连接海光DCU服务器并执行远程操作。所有其他dcu-toolkit子skill在执行远程命令前，均需遵循本skill的规范。

多步任务（部署验证/环境搭建/故障排查等）先查 `dcu-skill-router` §四 选链；跨域多步编排（门槛审批/evidence 留痕）走 `dcu-orchestrator`。

## When to Use

- 需要SSH到DCU服务器执行任何远程命令
- 需要在DCU服务器上部署脚本、运行测试、收集结果
- 任何dcu-toolkit子skill执行远程操作前的连接准备

Don't use for: 本地操作；非DCU服务器的SSH连接（但规范通用）。

## Security Rules (强制，不可违反)

1. **永远不要在skill包内存储密码或密钥**
2. 凭证文件路径：`~/.config/dcu-toolkit/credentials.json`（必须 `chmod 600`）
3. 密码认证必须用 `sshpass -f` 或 `sshpass -e`，**禁止 `sshpass -p`**（密码会出现在进程列表）
4. 临时密码文件用完立即删除
5. 密钥文件权限必须 `chmod 600`
6. 如果检测到SSH config别名存在，优先使用（无需密码）
7. SSH连接加 `-o StrictHostKeyChecking=no -o ConnectTimeout=10` 避免交互式卡住
8. **用户若在对话中直接贴明文令牌/密码**：仅用于当次操作（克隆/登录），用完即弃，**绝不写进 skill 包、不写进 credentials.json 之外的文件、不回显到输出**；并提示用户去对应平台吊销/重发该令牌（对话/日志会留存明文，视为已泄露）。凭据/令牌卫生总则见包根 SKILL.md「安全说明」（token 只进请求头/env，不进命令行参数与会话日志）。

## Device Inventory

设备信息存储在 `<agent-home>/skills/dcu/data/devices.json`（仅非敏感信息）：

```json
{
  "<device_name>": {
    "host": "<device_ip>",
    "port": 22,
    "user": "<user>",
    "auth_method": "password",
    "dcu_type": "<dcu_type>",
    "dcu_count": 8,
    "cpu": "<cpu_model>",
    "memory": "<memory_size>",
    "note": "示例设备（请替换为真实设备，且勿提交进 git）"
  }
}
```

字段说明：
- `auth_method`: "password" 或 "key" 或 "config"
- `dcu_type`: DCU卡型号（Z100/Z100L/K100-AI/K100等）
- `dcu_count`: DCU卡数量

## Step 0：已知主机快速定位（连设备前先看 known_hosts）

当你只知道设备编号（如“<device-A>”）或部分名，而不清楚完整 IP/别名时，**第一步不是查 ssh config / /etc/hosts / bash 历史，而是直接看 SSH 自己记录的已连主机**——`~/.ssh/known_hosts`。只要之前连过这台机器，它就一定在这里，无需任何额外维护。

实战踩坑（2026-08-11）：用户说"之前登录过 <device-A>"，本机 ssh config / /etc/hosts / bash 历史都找不到，devices.json 又只是空模板。一度以为本机没有该设备。结果 known_hosts 里明摆着 `<<internal-ip>> ecdsa-...`——只是第一段没读 known_hosts 全文、且用 search_files 搜短串 “<device-A>” 没命中，才漏掉。纠正后读全文立刻找到。

正确做法：

1. **直接读全文**：`read_file ~/.ssh/known_hosts`——最快最全，所有历史连过的 IP/别名都在。
2. **检索要用完整 IP，不要搜短串**：`search_files` 对 known_hosts 搜 “<device-A>” 这类短串可能返回 0（实测如此），搜完整 IP 如 `<<internal-ip>>` 才能命中。模糊定位优先直接读全文。
3. **known_hosts 的 key 可能是纯 IP 或 `[host]:port` 形式**——提取出 IP 后，再去 credentials.json 匹配凭证。
4. **credentials.json 的设备 key 通常是 `IP-机型` 格式**（如 `<<internal-ip>>-BW100X8`），不是纯 IP。匹配时用 IP 前缀匹配，不要假设 key 名等于纯 IP。

定位示例（只打印长度，不泄露明文）：
```bash
# 1. 看 known_hosts 里所有已连主机（一眼找到 <device-A>/<device-B> 这类编号对应 IP）
read_file ~/.ssh/known_hosts
# 或搜完整 IP
# search_files pattern="<<internal-ip>>" path="~/.ssh/known_hosts" target="content"

# 2. 凭据按 IP 前缀匹配
DEV_IP=<<internal-ip>>
python3 - <<'PY'
import json, os
c = json.load(open(os.path.expanduser('~/.config/dcu-toolkit/credentials.json')))
keys = [k for k in c if k.startswith(f'{os.environ["DEV_IP"]}-')]
print("匹配到的设备key:", keys)
if keys:
    print("password长度:", len(c[keys[0]]['password']))
else:
    print("未找到该IP前缀的凭证，请确认 credentials.json")
PY
```

排错：若 known_hosts 也没有，说明本机从未连过该设备——此时再回头让用户补全 IP/别名/凭证，**不要凭空猜地址去连**（可能连错机器或触发安全告警）。

## Credential Reading

凭证文件位于skill包外部：`~/.config/dcu-toolkit/credentials.json`（chmod 600）

```json
{
  "<device_name>": { "password": "YOUR_PASSWORD_HERE" },
  "dcu-node-02": { "key_path": "~/.ssh/dcu_keys/node02.key" }
}
```

执行步骤：
1. 读取 `data/devices.json` 获取目标设备连接信息
2. 读取 `~/.config/dcu-toolkit/credentials.json` 获取凭证
3. 根据auth_method选择连接方式（见下文）

如果credentials.json不存在，提示用户创建：
```bash
mkdir -p ~/.config/dcu-toolkit
cat > ~/.config/dcu-toolkit/credentials.json << 'EOF'
{
  "<device_name>": { "password": "YOUR_PASSWORD_HERE" }
}
EOF
chmod 600 ~/.config/dcu-toolkit/credentials.json
```

## Connection Methods

### Method 1: Password Auth (sshpass -f)

```bash
# 1. 从凭证文件读取密码，写入临时文件
PASS=$(python3 -c "import json,os; c=json.load(open(os.path.expanduser('~/.config/dcu-toolkit/credentials.json'))); print(c['<device_name>']['password'])")
echo "$PASS" > /tmp/.dcu_pass_$$ && chmod 600 /tmp/.dcu_pass_$$

# 2. 执行SSH
sshpass -f /tmp/.dcu_pass_$$ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@<device_ip> "hy-smi"

# 3. 清理临时文件
rm -f /tmp/.dcu_pass_$$
```

### Method 1b: Password Auth (sshpass -e, 更简洁)

`sshpass -e` 从环境变量 `SSHPASS` 读密码，免去临时文件创建/清理：

```bash
export SSHPASS=$(python3 -c "import json,os; c=json.load(open(os.path.expanduser('~/.config/dcu-toolkit/credentials.json'))); print(c['<device_name>']['password'])")
sshpass -e ssh -o StrictHostKeyChecking=no root@<device_ip> "hy-smi"
# 批量循环时尤其方便（for hp in ...; do export SSHPASS=...; sshpass -e ssh ...; done）
# 注意：SSHPASS 仅存在于当前 shell 进程环境，不落盘、不留进程参数，比 -p 安全
```

### Method 2: Key Auth

```bash
ssh -i ~/.ssh/dcu_keys/node01.key -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@<device_ip> "hy-smi"
```

### Method 3: SSH Config Alias (推荐，最简洁)

先配置 `~/.ssh/config`：
```
Host dcu-node-01
  HostName <device_ip>
  User root
  Port 22
  IdentityFile ~/.ssh/dcu_keys/node01.key
  StrictHostKeyChecking no
```

然后直接：
```bash
ssh dcu-node-01 "hy-smi"
```

不需要知道IP、密码、密钥路径，全部委托给SSH config。

### Method 4: 经跳板机隧道访问（设备非直连，需经 bastion 跳转）

**适用**：目标 DCU 设备不在本机可达网段，必须经一台跳板机（bastion）中转。典型实例：**<device-alias> (<<internal-ip>>, 8×BW1100)** 经 Windows 跳板 **<internal_ip_jumphost>** 访问。

**原理**：本机 →(SSH)→ 跳板机 →(端口转发)→ DCU 设备。在本机建一条**常驻本地隧道**，把 DCU 的 22 端口映射到本机某端口（如 2236），之后所有访问都打本机隧道端口，等价于直连 DCU。

**Step 1 — 建立常驻隧道**（后台长驻进程，带保活）：
```bash
# 跳板机凭证从 credentials.json 读（key: jump-<internal_ip_jumphost>-windows）
export SSHPASS=$(python3 -c "import json,os; c=json.load(open(os.path.expanduser('~/.config/dcu-toolkit/credentials.json'))); print(c['jump-<internal_ip_jumphost>-windows']['password'])")
# 后台启动（用 terminal background=true，勿用 nohup）：
sshpass -e ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=10 -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N \
  -L 127.0.0.1:2236:<<internal-ip>>:22 <jumphost-user>@<internal_ip_jumphost>
```
验证隧道：`(ss -ltn || netstat -ltn) | grep 2236` 应显示 `LISTEN 127.0.0.1:2236`。

**Step 2 — 经隧道访问 DCU**（端口 2236）：

<device-alias> 已配公钥免密（2026-09-15），**直接用 SSH 别名**（`~/.ssh/config` 里 `Host <device-alias>` → `127.0.0.1:2236` + `IdentityFile ~/.ssh/dcu_keys/<device-alias>.key`）：
```bash
ssh <device-alias> "hy-smi"
```
若未配免密（其他设备），用密码：
```bash
export SSHPASS=$(python3 -c "import json,os; c=json.load(open(os.path.expanduser('~/.config/dcu-toolkit/credentials.json'))); print(c['<<internal-ip>>-BW1100']['password'])")
sshpass -e ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=15 -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  -p 2236 root@127.0.0.1 "hy-smi"
```

**Step 3 — 隧道断了重建**：先 `pgrep -af "2236:<<internal-ip>>"` 查旧进程，精确 PID kill 后重跑 Step 1。

**关键坑（实测 2026-09-15）**：
- **跳板机是 Windows**：`Test-NetConnection -ComputerName <ip> -Port 22` 可先验证跳板→DCU 端口通不通；Windows 自带 OpenSSH 客户端（`ssh -V` 确认）。
- **DCU 密码带特殊字符（分号 `;`）**：<device-alias> 的 root 密码含分号（见 credentials.json key `<<internal-ip>>-BW1100`，**明文不写进本 skill**）。必须用 `sshpass -e`（环境变量）或 `-f`（文件），**绝不能用 `-p`**（分号会被 shell 截断成命令分隔符）。
- **连试错密码触发账号锁定**：<device-alias> 对 root 有失败计数，连试 3-5 次错密码会锁 ~60s，期间正确密码也被拒。**规避**：先单条 `sshpass -e ssh` 验证密码，别一上来就 scp/循环试变体；触发后等满 60s 再试。
- **隧道是长驻进程**：用 `terminal background=true` 启动（agent 跟踪），不要用 `nohup`/`&`（会被拦）。`ServerAliveInterval=30` 保活，跳板机断连会自动退出，需重建。
- **跳板机 22 端口是唯一开口**：<internal_ip_jumphost> 只开 SSH(22)，其他端口（LLM 网关等）不能当跳板。

**Windows 跳板机设置（用户手动执行，agent 给指令）** — 实测 2026-09-15：
- **PowerShell 多行粘贴会倒序执行**：把多行命令块粘进 PowerShell，各行**从下往上**执行（实测 `New-Item` 最后才跑，导致前面的 `icacls`/`Add-Content` 报"找不到路径"）。**规避**：给用户**单行、分号 `;` 连接**的命令，一次粘贴按序执行。例：`New-Item -Force -Path $env:USERPROFILE\.ssh | Out-Null; Add-Content -Path $env:USERPROFILE\.ssh\authorized_keys -Value "ssh-ed25519 <pubkey>"; icacls $env:USERPROFILE\.ssh\authorized_keys /inheritance:r`。
- **Windows EDR 会锁死 `authorized_keys`（即使你是管理员）**：企业安全软件（CrowdStrike/Defender 企业策略）把 `authorized_keys` 当攻击持久化点专门拦截——实测 `Admin: True` 却对自己刚建的该文件 `Get-Content`/`icacls` 全报"拒绝访问"。**别再折腾 ACL**，直接**改用密码认证**（`sshpass -e`）登录跳板，最省事。
- **装 OpenSSH Server（Win10/11）**：`Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0` → `Start-Service sshd; Set-Service -Name sshd -StartupType 'Automatic'` → `New-NetFirewallRule -Name sshd -Direction Inbound -Protocol TCP -LocalPort 22 -Action Allow`。验证跳板→DCU 端口：`Test-NetConnection -ComputerName <dcu_ip> -Port 22`。
- **拿登录名**：让用户跑 `whoami`（返回 `DOMAIN\user`，如 `<domain>\<jumphost-user>`），agent 用它拼 `user@jumphost_ip`。

## Command Execution Patterns

### Short Command (< 30s, 直接执行)

```bash
sshpass -f /tmp/.dcu_pass ssh -o StrictHostKeyChecking=no root@<device_ip> "hy-smi"
```

直接取stdout作为结果。适用于：hy-smi、lscpu、free、cat配置文件等快速命令。

### Long Command (> 30s, 后台执行+轮询)

适用于benchmark、烤机、模型推理测试等长时间任务。

```bash
# 1. 上传脚本到服务器
scp -o StrictHostKeyChecking=no run_bench.sh root@<device_ip>:/tmp/

# 2. 后台启动，记录PID
PID=$(sshpass -f /tmp/.dcu_pass ssh root@<device_ip> "nohup bash /tmp/run_bench.sh > /tmp/dcu_task.log 2>&1 & echo \$!")

# 3. 轮询日志（每隔30-60秒）
sshpass -f /tmp/.dcu_pass ssh root@<device_ip> "tail -20 /tmp/dcu_task.log"

# 4. 检查进程是否结束
sshpass -f /tmp/.dcu_pass ssh root@<device_ip> "ps -p $PID"
# 返回非0 → 进程已结束，任务完成

# 5. 下载结果文件
scp -o StrictHostKeyChecking=no root@<device_ip>:/tmp/result.json ./
```

### File Transfer

```bash
# 上传单个文件
scp -o StrictHostKeyChecking=no local_script.sh root@<device_ip>:/tmp/

# 下载单个文件
scp -o StrictHostKeyChecking=no root@<device_ip>:/tmp/result.json ./

# 批量传输（rsync更高效）
rsync -avz -e "ssh -o StrictHostKeyChecking=no" ./scripts/ root@<device_ip>:/tmp/dcu_scripts/
```

## Connection Verification

首次连接设备时，先验证连接：

```bash
ssh root@<device_ip> "echo CONNECTION_OK && hostname && whoami"
```

输出包含 `CONNECTION_OK` 则连接正常。

## DTK Environment Loading

DCU服务器上的DTK工具需要加载环境变量。远程执行时在命令前加source：

```bash
ssh root@<device_ip> "source /opt/dtk/env.sh && hy-smi"
```

如果DTK路径不确定，先检测：
```bash
ssh root@<device_ip> "ls /opt/dtk*/env.sh 2>/dev/null || find /opt -name 'env.sh' -path '*dtk*' 2>/dev/null"
```

## Migration to Key Auth (推荐)

密码认证能跑但安全性不如密钥认证。建议一次性迁移：

```bash
# 1. 生成密钥
ssh-keygen -t ed25519 -f ~/.ssh/dcu_keys/node01.key -N ""

# 2. 部署到服务器（需密码，一次性操作）
sshpass -f /tmp/.dcu_pass ssh-copy-id -i ~/.ssh/dcu_keys/node01.key root@<device_ip>

# 3. 配置SSH config
cat >> ~/.ssh/config << 'EOF'
Host dcu-node-01
  HostName <device_ip>
  User root
  Port 22
  IdentityFile ~/.ssh/dcu_keys/node01.key
  StrictHostKeyChecking no
EOF

# 4. 测试
ssh dcu-node-01 "echo OK"

# 5. 之后所有操作直接用别名，无需密码
```

## Common Pitfalls

1. **用 sshpass -p 传密码** — 密码出现在 `ps aux` 进程列表中。必须用 `-f` 或 `-e`
2. **忘记清理临时密码文件** — 用完必须 `rm -f /tmp/.dcu_pass_*`
3. **长命令阻塞SSH** — 超过30秒的命令不用直接执行模式，用后台+轮询
4. **忘记source DTK环境** — DTK 工具链（dtk 编译器/hipcc/roc* 等）命令找不到。远程命令前加 `source /opt/dtk/env.sh`（路径不确定时先 `ls /opt/dtk*/env.sh`）。
   **例外：hy-smi 不需要 source DTK** — 它是 DCU 驱动自带的，装驱动后就在 `/opt/hyhal/bin/hy-smi`（宿主机与容器内均有，PATH 可能不含）。直接调绝对路径即可：`/opt/hyhal/bin/hy-smi`。若绝对路径不存在，再考虑 `docker exec <dcu容器> hy-smi`（见 `env/dcu-topology/references/hy-smi.md`）。
5. **StrictHostKeyChecking弹交互** — 首次连接卡住。必须加 `-o StrictHostKeyChecking=no`
6. **credentials.json权限不对** — 必须 `chmod 600`，否则有泄露风险
7. **sshpass未安装** — 检查并安装：`apt install sshpass` 或 `yum install sshpass`
8. **用户在对话贴明文凭证** — 用户可能直接把 root 密码、gitee/gitlab 私人令牌贴进对话。处理：当次用、不落盘、不回显，用完提示吊销令牌；密码如需持久化，只写进 `~/.config/dcu-toolkit/credentials.json`（chmod 600），不进 skill 包

9. **credentials.json 格式错误 → SSH Permission denied（连不上）** — 实测 2026-07-22：credentials.json 被存成错误结构（`devices` 字段是 IP 字符串数组，而非"顶层 key=设备名 + `password` 字段"的字典），导致 `c['<设备名>']['password']` 取空密码、sshpass 传空、服务器回 Permission denied。连设备前务必先验格式。正确格式（顶层 key 必须与 devices.json 的设备名完全一致）：
   ```json
   {
     "_comment": "DCU 设备凭证 — chmod 600",
     "<device_name_1>": { "password": "<from-secure-store>" },
     "<device_name_2>": { "password": "<from-secure-store>" }
   }
   ```
   **连设备前的格式自检/自愈**（只打印长度，不泄露明文）：
   ```python
   import json, os, sys
   p = os.path.expanduser('~/.config/dcu-toolkit/credentials.json')
   c = json.load(open(p)); dev = '<设备名>'   # 取自 devices.json
   if dev not in c or not isinstance(c.get(dev), dict) or 'password' not in c.get(dev, {}):
       print(f"WARN: {dev} 凭证格式错误/缺失，请修正 credentials.json"); sys.exit(2)
   print("password len:", len(c[dev]['password']))
   ```
   修正/重建用 python `json.dump` 写回并 `chmod 600`，**不要**用 echo/cat heredoc 写含密码文件。含密码的 credentials.json 绝不提交进任何 git 仓库（`~/.config` 本就不在 skills git 内）。

10. **不要假设同机房/同批次设备共用同一密码** — 实测发现同网段、同一批交付的设备，实际密码也可能不同。把 A 设备的密码套到 B 设备会导致 `Permission denied`。连一台失败时报 Permission denied，**先核对这台设备自己的凭证**，不要跨设备套用，也不要假设"同一批贴的就一定相同"。每台独立存、独立试。

11. **scp 指定端口用大写 `-P`，不是 `-p`** — 实测 2026-07-22：scp 里 `-p` 是"保留时间戳"，端口参数是大写 `-P`。写成 `scp -p 22 ...` 会被当成保留时间戳、忽略端口，接着因找不到默认22以外的路径或源文件而报 `No such file or directory`（误导性报错）。正确：`scp -o StrictHostKeyChecking=no -P 22 local_file root@host:/tmp/`。ssh/sshpass 的端口参数才是小写 `-p`。注意区分：ssh `-p 22`、scp `-P 22`、sshpass 包 ssh 时透传的是 ssh 的小写 `-p`。

12. **多机测试需免密通信** — 双机/多机 rccl-test（mpirun）依赖节点间 root SSH 免密。物理机免密：关 firewalld/iptables → 各机 `ssh-keygen -t rsa` → `/etc/hosts` 配管理网 IP 与主机名 → `ssh-copy-id` 互信（用管理网 IP，非 IB/ROCE 网）。容器免密见 dcu-docker。来自内部测试手册"基本环境准备"。

13. **连续认证失败会锁账号（44 秒）— 实测 2026-09-07**：Kylin V10 设备对 root 有登录失败计数，连续 3 次密码错误（如 scp 没带 sshpass 导致密码为空、或密码打错）会触发 `The account is locked due to 3 failed logins. (44 seconds left to unlock)`，期间所有 SSH 连接（含正确密码）都被拒。**规避**：① 连新设备前先确认密码正确（用 `sshpass -e ssh` 单条命令验证，不要一上来就 scp）；② scp 必须用 `sshpass -e scp`（scp 不读 SSHPASS 环境变量，裸 scp 会弹密码提示→失败→计数）；③ 触发锁定后**等满 44 秒**再试，不要反复连（会重置计时器延长锁定）。

14. **传文件优先用 ssh stdin 重定向，不要裸 scp — 实测 2026-09-07**：`scp` 需要单独认证，容易触发 pitfall 13 的账号锁定（尤其密码没配 sshpass 时）。**更稳的路径**：`sshpass -e ssh root@host "cat > /remote/path" < /local/file`——复用已验证的 SSH 通道，不额外触发认证。小文件（脚本/配置）这是最稳路径；大文件（模型/权重）优先 `-v` 挂载宿主机目录，不走 scp/stdin。

15. **Windows 本机写的脚本带 CRLF，传到 Linux 远端执行报错 — 实测 2026-10（DSH 会话）**：本机（Windows/DSH 工作机）生成的脚本/多行命令默认 CRLF 换行，传到 Kylin/Ubuntu 远端后 bash 报 `$'\r': command not found`（heredoc、多行脚本最易中招，报错不指明真实原因）。**规避**：传前本机转 LF（PortableGit bash：`sed -i 's/\r$//' script.sh`），或传后在远端 `sed -i 's/\r$//' /tmp/xxx.sh` 再执行；经 `ssh "cat > 远端路径" < 本地文件`（见 pitfall 14）传的脚本同样中招。单行命令不受影响，多行/脚本文件必转。

16. **ssh 里宽 `pkill -f` 会自杀** — 模式串出现在远端 `bash -c` 自身的命令行里，`ssh host 'pkill -f "docker pull xxx"'` 自匹配、把当前会话 shell 杀掉（exit 255、输出全丢、目标任务进程可能还活着）；宽 pkill 还容易触发破坏性命令审批门，用户不在时整个任务被卡住。纪律：**先 `pgrep -af '<pattern>'` 取精确 PID 再 `kill <PID>`**；必须用 `pkill -f` 时用括号模式（`pkill -f 'docker pul[l]'`）或变量拼串避免静态字面量（docker build 场景见 `dcu-docker` 构建清理一节）。

## Verification Checklist

- [ ] devices.json中目标设备信息完整（host/port/user/auth_method）
- [ ] credentials.json存在且权限为600，且格式正确（顶层 key=设备名、`password` 字段存在，非 IP 字符串数组）—— 见 pitfall 9
- [ ] SSH连接验证通过（输出CONNECTION_OK）
- [ ] 临时密码文件已清理（用 `-e` 则无需）
- [ ] DTK环境变量已加载（如需要；hy-smi 不需要）
- [ ] 长命令使用后台+轮询模式
- [ ] 结果文件已下载到本地（如需要）
- [ ] **跨系统验证**：DCU 服务器混用 Ubuntu(5.15) 与 Kylin V10(4.19) 老内核，命令行为差异大（nproc/numactl/lspci 等）。脚本改动需在两种系统都验证，双机 harness 见仓库根 `references/dual-machine-verification.md`
- [ ] 多行脚本传远端前已确认 LF 换行（CRLF 报 `$'\r': command not found`）——见 pitfall 15
