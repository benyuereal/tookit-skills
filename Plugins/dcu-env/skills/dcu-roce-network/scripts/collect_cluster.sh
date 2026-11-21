#!/bin/bash
# collect_cluster.sh — 自动收集集群网络拓扑（各节点网口 IP + MAC）
#
# 用途：在一台能 SSH 免密登录所有节点的控制节点/跳板机上运行。
#       自动读取各网卡实际 IP 与 MAC，生成统一的拓扑文件 cluster_topology.conf，
#       供 apply_arp.sh 在每个节点做静态 ARP 全量注入使用。
#
# 运行前必做（见 SKILL.md 踩坑）：
#   1. 在所有节点执行 `ip -br link` 确认 NDR 网卡物理接口名（勿照抄 p66p1 等）
#   2. 用 `ls -l /sys/class/net/<iface>/device/infiniband` 或 ibstat/ibv_devinfo
#      核对 mlx5_x 与系统网卡名的对应关系
#   3. 控制节点已与所有节点 SSH 免密互通
#
# 用法：bash collect_cluster.sh
# 输出：/usr/local/bin/cluster_topology.conf（每行格式 Node|Interface|MAC|IP）

NODES=("node1" "node2" "node3" "node4")   # ← 按实际节点名/主机名改写
IFACES="p66p1 p66p2 p61p1 p61p2 p82p1 p82p2 p72p1 p72p2"   # ← 按实际网口名改写

OUTPUT="/usr/local/bin/cluster_topology.conf"
echo "# Node|Interface|MAC|IP" > "$OUTPUT"

echo "=== 开始收集集群网络信息 ==="
for node in "${NODES[@]}"; do
    echo "正在收集 $node ..."
    ssh -o StrictHostKeyChecking=no "$node" "
        for dev in $IFACES; do
            mac=\$(cat /sys/class/net/\$dev/address 2>/dev/null)
            # 提取纯 IP (去除 /27 等掩码位)
            ip=\$(ip -4 -br addr show dev \$dev 2>/dev/null | awk '{print \$3}' | cut -d'/' -f1)
            if [ -n \"\$mac\" ] && [ -n \"\$ip\" ]; then
                echo \"$node|\$dev|\$mac|\$ip\"
            fi
        done
    " >> "$OUTPUT"
done

echo "=== 收集完成！拓扑文件已保存至 $OUTPUT ==="
echo "预览："
cat "$OUTPUT"
