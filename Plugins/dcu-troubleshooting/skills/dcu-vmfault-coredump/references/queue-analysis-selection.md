# HSA 队列分析：怎么在 1000+ 个块里选出 faulting kernel

一次 VMFault 会让运行时把**所有 HOSTQUEUE（stream）**都 dump 队列分析，每个 queue 再 dump 一滑动窗口的
packet —— 一次崩溃能产生几百上千个 `FIND MATCH KERNEL DETAILS` 块（真实案例 1239 个），绝大部分是无关上下文。
选 faulting 的那个，靠的是 **W/R 指针语义**，不是位置。

配合脚本 `find_faulting_kernel.py`（它自动完成本页第 2、3 步）。本页用真实日志
`sg-t1:/data/debug_queue_error/20260616_qwen3_5_35A3B_eval/log.log` 逐步讲透原理。

## 心智模型：HSA queue = 环形缓冲 + 两个指针

每个 stream（HSA host queue）是一块环形内存，CPU 和 GPU 通过**两个指针**协作：

```
write index W  —— CPU（host）生产端：我提交到第几个 slot 了
read  index R  —— GPU（device）消费端：我执行到第几个 slot 了
```

HSA 保证 `R ≤ W` 恒成立（设备永远不会读走还没提交的 slot）：

| 关系 | 含义 | 日志特征 |
|------|------|---------|
| `R == W` | 队列已排空，GPU 追上了 CPU | `all command executed` |
| `R < W` | 有 `W−R` 个 packet 还没执行（pending） | 不打印 `all command executed` |

**VMFault 的本质**：设备在某个 slot 上执行到坏指令/越界访存，硬件 trap 住、**该队列当场卡死**。此时它的消费指针停在那个
slot 上，CPU 还在继续提交 → `R < W`，挂着一堆 pending。**这就是 faulting queue 的指纹。**

## 第 2 步：faulting queue = 唯一 `write != read` 的那个

一次 VMFault，运行时把所有 HOSTQUEUE 都 dump。真实日志里 trap 时同时活着 5 个 stream：

```
line 1243  stream 4  W/R = (168388, 168388)   → all command executed   ← 排空
line 1250  stream 3  W/R = (168388, 168388)   → all command executed   ← 排空
line 1257  stream 2  W/R = ( 96222,  96222)   → all command executed   ← 排空
line 1264  stream 1  W/R = ( 21916,  21916)   → all command executed   ← 排空
line 1271  stream 0  W/R = (35452708, 35451391) ← 没有这行！           ← 唯一 pending
```

**逻辑链：**
1. 前 4 个 queue 都是 `all command executed`（R==W）→ 在 trap 那一刻空闲/已排干。设备不可能死在已执行完的队列里。**无辜。**
2. 只有 stream 0 `write=35452708, read=35451391`，差 **1317** 个 packet 挂着 → 设备死在一个"还没执行完"的队列里。**就是它。**
3. **交叉验证**：日志锚点（VMFault 回调）点名 faulting queue 的地址，与 stream 0 的 base_address 逐字节对上：

```
line 1231  Callback: Queue 0x7f853c200000 aborting ... code: 0x1016
line 1269  stream 0: base_address 0x7f853c200000        ← 完全一致
```

两个独立证据指向同一个 queue，判决锤死。

> 这就是为什么 `find_faulting_kernel.py` 用**语义**（W≠R + base_address 对上锚点）选队列，而不是"第几个块"。

## 第 3 步：faulting packet = read index 那个 packet

确定 queue 后，还要在它内部选具体哪条 packet 崩的。

**直觉**：read index 是设备 dispatch 引擎**当前正在工作**的位置。trap 瞬间，正在执行的只能是 `R` 这个 slot —— 它是"枪口"。
R 之后的全是还没轮到执行的排队项。

真实日志里 dump 起点正是 read index（line 1273 `DUMP AQL ... PACKET WITH INDEX 35451391`），随后是排队序列：

```
packet 35451391  indexSelectSmallIndex<Half,long,uint,2,2,-2>   ← READ INDEX，正在执行 → 崩的
packet 35451392  MATCH KERNEL COMMAND（无符号匹配，无 kernel name）← 排队
packet 35451393  MATCH BARRIER AND COMMAND                       ← fence/屏障，排队
packet 35451394  MATCH BARRIER AND COMMAND                       ← fence/屏障，排队
packet 35451395  triton_poi_fused_0                              ← 排队
packet 35451396  triton_red_fused_1                              ← 排队
...              Cijk_Ailk_Bljk GEMM ×2、triton_poi_fused_2、direct_copy ...
packet 35451401  _causal_conv1d_update_kernel                    ← 排队（GDN decode conv）
```

关键点：
- **35451391 既是 read index、又是第一个被 dump 的 packet、又恰好有符号名** —— 三者指向同一结论：embed 越界就是崩溃点。
  `c++filt` 后 `indexSelectSmallIndex<c10::Half, long, unsigned int, 2, 2, -2>` 直接告诉我们是 2D fp16 + int64 索引的 index_select。
- 35451392 是 `MATCH KERNEL COMMAND` 但**没有** `kernel name:` 行（符号没匹配上）；35451393/94 是 `MATCH BARRIER AND COMMAND`
  （屏障）。它们天然没有内核名 —— **不能因为"没名字"就说它是 faulting kernel**。
- **R 之后排队的 packet 反而是增值证据**（第 4 层归属用）：`_causal_conv1d_update_kernel`（GDN conv）、
  `fused_sigmoid_gating_delta_rule_update_kernel`（GDN attn）只在 TARGET 模型的 GDN 层存在 → 说明 embed 是
  TARGET decode 入口的第一件事，归因到 `embed_tokens`。见 `crash-patterns.md` 模式 1。

## 一句话总结

> **第 2 步**：崩掉的 queue 是那个 R<W 的（设备死在未排空的队列里），再拿 abort 回调的 base_address 对上；
> **第 3 步**：崩掉的 packet 是 read index 那条（设备 trap 时正执行的那个 slot），dump 从它开始，往后全是"排队证据"。

## 已知坑

- **W/R 数字巨大（千万级）很正常**：它是 packet 总数，不是字节偏移。别被吓到。
- **同一队列的 `hsa queue id` 可能重复**：日志里 stream 4 和 stream 3 都叫 id 4，但 HOSTQUEUE 地址和 base_address 不同 ——
  用 stream id 和 base_address 区分，别用 queue id。
- **`all command executed` 的队列也可能有大量 AQL dump**：那是滑动窗口的"历史上下文"，不代表它执行到那了。
- **faulting packet 不保证有符号名**：Triton/CK/自定义 HIP 内核可能在 `MATCH KERNEL COMMAND` 里没有匹配的
  `kernel name:` 行。此时仍以 read index 定位，再用 `parse_kernel_args.py --raw` 解 args（见 `kernel-abi.md`）。
