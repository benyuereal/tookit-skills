#!/bin/bash
# rdma_connectivity_test.sh — RDMA 物理网口点对点连通性验证（同/跨子网 + 带宽）
#
# 用途：RoCE 网络配置（IP/MTU/ARP/QoS）完成后，验证各物理网口间是否真正互通。
#      分三档：①同子网点对点 ping（8 子网两两）②跨子网 ping ③ib_write_bw 带宽。
#
# 运行位置：任意一个节点（或控制节点）执行；被测 IP 按实际网表改写。
# 依赖：ping、ib_write_bw（Mellanox OFED）。带宽测试需对端先起服务端（见 SKILL.md）。
#
# 用法：
#   bash rdma_connectivity_test.sh ping-same     # ① 同子网两两 ping
#   bash rdma_connectivity_test.sh ping-cross     # ② 跨子网 ping（手动示例）
#   bash rdma_connectivity_test.sh bw             # ③ 带宽测试（需对端服务端）

MODE="${1:-ping-same}"

# ============ ① 同子网点对点 ping ============
SUBNET_1=(124.254.0.1 124.254.0.2 124.254.0.3 124.254.0.4)        # p66p1
SUBNET_2=(124.254.0.33 124.254.0.34 124.254.0.35 124.254.0.36)    # p66p2
SUBNET_3=(124.254.0.65 124.254.0.66 124.254.0.67 124.254.0.68)    # p61p1
SUBNET_4=(124.254.0.97 124.254.0.98 124.254.0.99 124.254.0.100)   # p61p2
SUBNET_5=(124.254.0.129 124.254.0.130 124.254.0.131 124.254.0.132) # p82p1
SUBNET_6=(124.254.0.161 124.254.0.162 124.254.0.163 124.254.0.164) # p82p2
SUBNET_7=(124.254.0.193 124.254.0.194 124.254.0.195 124.254.0.196) # p72p1
SUBNET_8=(124.254.0.225 124.254.0.226 124.254.0.227 124.254.0.228) # p72p2

ALL_SUBNETS=(SUBNET_1 SUBNET_2 SUBNET_3 SUBNET_4 SUBNET_5 SUBNET_6 SUBNET_7 SUBNET_8)

ping_same() {
  LOG_FILE="rdma_ping_result_$(date +%Y%m%d_%H%M%S).log"
  echo "===== 开始 RDMA 同网口 Ping 测试 =====" | tee "$LOG_FILE"

  FAIL_COUNT=0
  TEST_COUNT=0

  for subnet_name in "${ALL_SUBNETS[@]}"; do
    eval "CURRENT_IPS=(\"\${${subnet_name}[@]}\")"
    echo ">> 正在测试子网: ${CURRENT_IPS[0]}/27" | tee -a "$LOG_FILE"

    for src_ip in "${CURRENT_IPS[@]}"; do
      for dst_ip in "${CURRENT_IPS[@]}"; do
        if [ "$src_ip" == "$dst_ip" ]; then continue; fi
        ((TEST_COUNT++))
        if ! ping -c 2 -W 1 "$dst_ip" &> /dev/null; then
          echo "[FAIL] $src_ip -> $dst_ip" | tee -a "$LOG_FILE"
          ((FAIL_COUNT++))
        fi
      done
    done
  done

  echo "===== 测试完成 =====" | tee -a "$LOG_FILE"
  echo "总计测试: $TEST_COUNT 次链路" | tee -a "$LOG_FILE"
  if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "🎉 恭喜！所有 RDMA 同网口链路均互通，0 个失败。" | tee -a "$LOG_FILE"
  else
    echo "⚠️ 警告：共有 $FAIL_COUNT 条链路不通，请检查日志 $LOG_FILE" | tee -a "$LOG_FILE"
  fi
}

# ============ ② 跨子网 ping（示例：用指定源接口 ping 目标） ============
ping_cross() {
  echo "===== 跨子网 ping 测试 ====="
  echo "示例（按实际目标网口 IP 替换）："
  echo "  ping 124.254.0.34 -I 124.254.0.1      # 用源 IP .1 去 ping 对端 .34"
  echo "-- 若需全量跨子网，按 SUBNET_x 两两组合自行扩展 --"
}

# ============ ③ ib_write_bw 带宽测试 ============
bw_test() {
  echo "===== ib_write_bw 带宽测试 ====="
  echo "需先在被测节点对端起服务端，再在客户端执行，参数示例："
  echo
  echo "服务端："
  echo "  ib_write_bw -d mlx5_2 -a -F -x 3 -q 1 -s 65536 --report_gbits"
  echo
  echo "客户端："
  echo "  ib_write_bw -d mlx5_3 -a -F -x 3 -q 1 -s 65536 --report_gbits 124.254.0.66"
  echo
  echo "参数说明："
  echo "  -d <mlx5_x>       RDMA 设备名（用 ibstat/ibv_devinfo 或 /sys/class/net/<net>/device/infiniband 确认）"
  echo "  -a                所有端口"
  echo "  -F                force（忽略不兼容警告）"
  echo "  -x 3              传输类型（RoCE 场景固定 3）"
  echo "  -q 1              队列对数量"
  echo "  -s 65536          消息大小（大包测带宽）"
  echo "  --report_gbits    以 Gb/s 输出"
}

case "$MODE" in
  ping-same) ping_same ;;
  ping-cross) ping_cross ;;
  bw) bw_test ;;
  *)
    echo "用法: $0 {ping-same|ping-cross|bw}"
    exit 1
    ;;
esac
