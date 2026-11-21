# 三平台迁移对照

## 快速对照表

| 维度 | BW1000 | C500（25% sGPU） | Ascend 910B1 |
|---|---|---|---|
| 后端 | HIP / `gfx936` | HIP / MACA 3.0.0.8 | NPU / CANN 9.0.0 |
| PyTorch | 2.7.1 | mcPyTorch 2.4.0+metax3.0.0.3 | torch_npu 2.10.0 |
| Triton | 3.1.0 | mcTriton 3.0.0+metax3.0.0.3 | Triton-Ascend 3.2.1 |
| warp_size | 64 | 64 | — |
| Python | 3.10 | 3.10 | 3.11.15 |
| 评测参数 | `warmup=200, repeat=500` | 同 | 同 |

---

## 迁移说明分类

### 直接复用（算法和 tile 基本不变）

- **Task01 GroupedTopk**：选择与归一化逻辑完全相同
- **Task02 FusedMoE**：路由与分阶段 GEMM 直接复用
- **Task05 MusicFlamingoRotaryEmbedding**：逐元素融合直接复用
- **Task07 mhc_post**：大张量融合直接复用
- **Task08 hc_split_sinkhorn**：4×4 标量化 Sinkhorn 直接复用
- **Task09 CRA**：随机变换融合直接复用

### tile 仍可调优（算法可迁移，需重新扫参）

- **Task03 FlexAttention**：query tile 调为 32×32（Ascend 910B）
- **Task06 MMEncoderAttention**：query tile 调为 32（Ascend 910B）
- **Task04 SPLADESparsePooler**：矩阵 tile 仍可调优
- **Task10 head_compute_mix_bwd**：归约参数仍可调优

### 特殊处理

- **Task04 Ascend 910B**：需要拆分 decoder 与池化，避开 watchdog（超时限制）
- **Task07 C500**：需设置 NCHW 布局后复用高性能 kernel
- **Task08 C500**：显式标量化 4×4 Sinkhorn，不依赖 shared 归约

---

## Ascend 910B 特殊限制

### AICore watchdog

Ascend 910B 有 watchdog 超时限制，大矩阵或长时间运行的单 kernel 可能被强制中断。

**处理方法：**
- 拆分大矩阵为多个小 kernel
- Task04 在 910B 上把 decoder 与池化拆分两个 kernel

### UB 限制

统一缓冲（UB）大小有限制，大张量无法完全放在 UB 中。

**处理方法：**
- 分块处理，使用 HBM 作为中间缓冲
- 减少单 kernel 内的内存占用

---

## C500 特殊处理

### sGPU 切片

C500 使用 25% sGPU 切片，资源受限。

**处理方法：**
- 减小 BLOCK_SIZE
- 降低 num_warps
- 确保内存布局为 NCHW（而非 NHWC）

### NCHW 布局

Task07 需要显式设置 NCHW 布局以复用 BW1000 高性能 kernel。

---

## 环境变量

```bash
# BW1000/C500
DEVICE_ID=0
HIP_VISIBLE_DEVICES=0

# Ascend 910B
CANN_DEVICE_ID=0
```

---

## 参考文档

- BW1000 环境：`platforms/bw1000/README.md`
- C500 迁移指南：`platforms/muxi_c500/C500_环境与迁移指南.md`
- Ascend 910B 迁移指南：`platforms/ascend910b/Ascend910B_环境与迁移指南.md`