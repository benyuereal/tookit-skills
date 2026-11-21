# 集合通信性能估算方法（判瓶颈用）

> 延迟/吞吐分界、Ring vs Tree、分层 max、Bisection、SHARP、与推理指标关系。
> 2026-10-16 自 `env/dcu-rccl-test/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），内容原文照录。

## 集合通信性能估算方法（判瓶颈用）

> 来源：《深入TPU与GPU集群的集合通信》方法论提炼（2026-09-16）。只沉淀可迁移的估算方法，不照搬 TPU 数字；DCU 数值以本机实测为准。
> 核心纪律：集合算法只有懂底层拓扑才成立；理想模型到真集群之间必须靠 microbenchmark 校准，别照搬文章数字。

### 1. 延迟主导 vs 吞吐主导的分界（最常用）
- **方法**：临界 chunk ≈ 单向链路带宽 × 单跳延迟。消息 < 临界 → 延迟主导（time 恒定、纯带宽近似失效）；> 临界 → 吞吐主导（time 随 size 线性增长、busbw 恒定）。
- **怎么找**：跑 `all_reduce_perf -b 2 -e 2G -f 2 -g <N>` 全 size 曲线，找 busbw 开始爬升的拐点（延迟→吞吐过渡）。
- **DCU 实测例（BW100 8卡 dtk26.04，<device-B>，2026-09-16，"两组4卡跨组3-hop"拓扑，非通用基线）**：
  - all_reduce：延迟 floor ~64μs，临界 chunk ~512B-1KB，峰值 busbw ~24 GB/s
  - alltoall：延迟 floor ~100-150μs，临界 chunk ~64KB-128KB，峰值 busbw ~13.75 GB/s
  - alltoall 临界 chunk 比 all_reduce 大两个数量级（alltoall 小消息启动/路由开销更高）

### 2. Ring vs Tree 选型判据
- 大消息 → ring（好流水、链路常忙）；小消息 → tree（log₂N 步、延迟低）。RCCL 按消息大小 + 拓扑自动选。
- 某档消息慢时，先判它落在哪个 regime（配合 §1 临界 chunk），再看算法选得对不对。

### 3. 分层集合通信是 max 不是相加
- 跨 node 总时延 ≈ `max(D/BW_node内, D/BW_node间)`，不是两段相加（scale-out 与 node 内可流水）。
- 估跨机 All-Reduce 别把 node 内 + IB 两段带宽串行相加（会高估时延）。

### 4. Bisection 带宽估算
- 满 fat tree 下，一侧 N 个 node 可跨分区发 `N × 单node注入带宽`；被切更小的那侧 node 数限制上限。
- 估跨分区别被"总带宽"骗，看小侧 node 数 × 单 node 注入带宽。

### 5. 网内归约（SHARP 类）
- 理论能拉满 ~2×，但实测常见仅 ~1.3×（归约与 multicast 流水无法完美重叠）。
- DCU 若有网内归约能力，别按理论 2× 估，以 microbenchmark 为准。

### 6. 集合通信带宽与推理指标的关系（定性）
- **集合通信带宽与 TTFT（首字延迟）关联性较大，对吞吐性能影响相对较小**——长上下文 prefill 阶段 all-reduce/all-gather 占比高，通信带宽直接拖 TTFT；decode 阶段单步通信量小，吞吐对通信带宽不敏感。
- **EP（专家并行）会带来额外通信开销**，TPOT 存在一个分水岭（量级约 30~35ms）：低于该值时 EP 的通信开销可被计算掩盖，高于时通信开始主导 TPOT。是否上 EP 先看单机 TP 能否满足 TPOT 目标，再评估 EP 的通信代价。
