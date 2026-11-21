#!/bin/bash
# apply_arp.sh — 各节点自动配置内核 ARP 参数 + 静态 ARP 全量注入
#
# 用途：部署到每个节点运行。读取 cluster_topology.conf（由 collect_cluster.sh 生成），
#       通过比对「本机 MAC」自动过滤掉本机 IP，只把远端所有节点的 IP/MAC 写入本机
#       的每个网口（静态 permanent 邻居条目）。
#
# 为什么必须静态 ARP 全量注入（见 SKILL.md）：
#   1. 避免 ARP 广播泛滥与欺骗（配合 arp_ignore=2 / rp_filter=2）
#   2. 多路径/网卡容错：节点间有 8 条链路，若某接口缺邻居条目会因发不出 ARP 而丢包/延迟
#
# 前置：
#   - 本机已放置 cluster_topology.conf（与 collect_cluster.sh 同款，含全部节点）
#   - 本机所有 NDR 网口物理接口名正确（IFACES 按实际改写）
#
# 用法：bash apply_arp.sh
# 验证：ip neigh show                                          （应看到大量 PERMANENT 条目）
#       ip neigh show dev <任一同子网网口> | grep -i permanent | wc -l
# 注：重启后 ARP 配置失效，需重新执行本脚本（或做成 systemd 自启）

CONF_FILE="/usr/local/bin/cluster_topology.conf"
IFACES="p66p1 p66p2 p61p1 p61p2 p82p1 p82p2 p72p1 p72p2"   # ← 按实际网口名改写

if [ ! -f "$CONF_FILE" ]; then
    echo "错误: 找不到拓扑文件 $CONF_FILE"
    exit 1
fi

# 1. 设置内核 ARP 参数
sysctl -w net.ipv4.conf.all.arp_ignore=2 >/dev/null
sysctl -w net.ipv4.conf.all.arp_announce=1 >/dev/null
sysctl -w net.ipv4.conf.all.rp_filter=2 >/dev/null
ip neigh flush all

# 2. 获取本机所有 MAC 地址列表，用于自动排除本机 IP
LOCAL_MACS=$(ip link | grep link/ether | awk '{print tolower($2)}')

# 3. 读取配置文件，提取远端节点的 IP 和 MAC 映射
REMOTE_ENTRIES=()
while IFS='|' read -r node dev mac ip; do
    # 跳过注释行和空行
    # ⚠️ bash 中 `A || B && C` 按 `(A||B)&&C` 解析：node 是注释行 或 ip 为空 => continue
    [[ "$node" =~ ^#.* ]] || [ -z "$ip" ] && continue

    mac_lower=$(echo "$mac" | tr '[:upper:]' '[:lower:]')

    # 若该 MAC 不属于本机，则记入远端列表
    if ! echo "$LOCAL_MACS" | grep -q "$mac_lower"; then
        REMOTE_ENTRIES+=("$ip|$mac_lower")
    fi
done < "$CONF_FILE"

echo "解析完毕，获取到远端节点 IP/MAC 条目数量: ${#REMOTE_ENTRIES[@]}"

# 4. 遍历本机网卡接口，写入静态 ARP
for dev in $IFACES; do
    if [ -d "/sys/class/net/$dev" ]; then
        for entry in "${REMOTE_ENTRIES[@]}"; do
            ip=$(echo "$entry" | cut -d'|' -f1)
            mac=$(echo "$entry" | cut -d'|' -f2)
            ip neigh add "$ip" lladdr "$mac" dev "$dev" nud permanent 2>/dev/null
        done
        echo "接口 $dev 静态 ARP 写入成功。"
    else
        echo "警告: 接口 $dev 不存在，跳过。"
    fi
done

echo "静态 ARP 配置完成！"
