#!/bin/bash
# set_roce_qos.sh — RoCE QoS 配置（适用于每个节点）
#
# 策略：Trust DSCP + PFC(队列3,6) + 关闭链路流控 + TOS=96 + ECN(队列3,6)
#       适用于 Mellanox NDR 网卡（mlnx_qos / cma_roce_tos 工具）。
#
# 前置：节点已装 Mellanox OFED 网卡工具（mlnx_qos、cma_roce_tos 在 PATH）。
# 用法：bash set_roce_qos.sh
# 自启：见 SKILL.md systemd（roce-qos.service）

IFACES=("p66p1" "p66p2" "p61p1" "p61p2" "p82p1" "p82p2" "p72p1" "p72p2")   # ← 按实际网口名改写
TOS_VALUE=96

echo "================ 开始配置 RoCE QoS ================"

for dev in "${IFACES[@]}"; do
    if [ ! -d "/sys/class/net/$dev" ]; then
        echo "警告: 网卡接口 $dev 不存在，跳过配置。"
        continue
    fi

    echo "--------------------------------------------------"
    echo "正在配置网卡: $dev"

    # 1. 设置 Trust 模式为 dscp
    mlnx_qos -i "$dev" --trust dscp >/dev/null 2>&1

    # 2. 使能 PFC 队列 3 和 6 (--pfc 0,0,0,1,0,0,1,0)
    mlnx_qos -i "$dev" --pfc 0,0,0,1,0,0,1,0 >/dev/null 2>&1

    # 3. 关闭网卡全局链路层流控
    ethtool -A "$dev" tx off rx off >/dev/null 2>&1

    # 4. 动态获取网卡对应的 Mellanox RDMA 设备名 (如 mlx5_0, mlx5_1 等)
    mlx_dev=$(ls /sys/class/net/$dev/device/infiniband 2>/dev/null | head -n 1)
    if [ -n "$mlx_dev" ]; then
        cma_roce_tos -d "$mlx_dev" -t $TOS_VALUE >/dev/null 2>&1
        echo "  [✓] 设置 $dev ($mlx_dev) TOS = $TOS_VALUE"
    else
        echo "  [!] 错误: 未查找到 $dev 绑定的 mlx5 设备"
    fi

    # 5. 在优先级 3 和 6 上使能 ECN (NP / RP)
    for prio in 3 6; do
        if [ -f "/sys/class/net/$dev/ecn/roce_np/enable/$prio" ]; then
            echo 1 > "/sys/class/net/$dev/ecn/roce_np/enable/$prio"
            echo 1 > "/sys/class/net/$dev/ecn/roce_rp/enable/$prio"
        else
            echo "  [!] 警告: /sys/class/net/$dev/ecn 路径不存在，跳过 ECN 配置"
        fi
    done

    echo "  [✓] $dev QoS / PFC / ECN 配置完成"
done

echo "--------------------------------------------------"
echo "================ RoCE QoS 配置结束 ================"
