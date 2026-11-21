# P0 microbench 基线（2026-10-20，四卡，单次运行参考值）

> 方法见 `dcu-kernelwiki/wiki/hwprobe_p0_20261020.md`。设备：<device-C> K100-AI 卡2（特权容器 DTK-26.04）、<device-A> BW150 卡4（宿主 DTK-26.04）、<device-B> BW100 卡2（特权容器）、111 BW1100 卡4（特权容器 DTK-26.04-rc4）。
> ⚠ 单次运行、非中位数；copy-kernel 口径（非 tensor.copy_）。

| 指标 | K100-AI (gfx928) | BW150 (gfx936) | BW100 (gfx936) | BW1100 (gfx938) |
|---|---|---|---|---|
| HBM 理论 | 896 GB/s | 1843 GB/s | 1843 GB/s | 未算（HIP bus width 报 256 异常值） |
| HBM D2D copy | 374.1 | 1118.9 | 1131.3 | **1521.7** |
| HBM 纯读 float4 | 414.5（46%） | 1053.2（57%） | 1039.5（56%） | 1058.5 |
| HBM 纯写 | 234.6 | 619.8 | 547.4 | **1226.4（写>读）** |
| HBM 纯读 scalar | 484.5 ⚠(scalar>vector 单跑异常) | 822.8 | 818.8 | 882.5 |
| 向量化收益（读） | 异常待多跑确认 | +28% | +26% | +20% |
| kernel launch（2000 空 kernel） | 2.50 µs | 2.06 µs | 2.09 µs | 2.01 µs |
| LDS 带宽（读写混合） | 18.5 TB/s | 28.7 TB/s | 26.8 TB/s | 27.7 TB/s |
| PwrCap | 400W | 600W | 400W | 800W |

## 判读注意
1. 读写不对称方向随代际变：gfx936 纯读 ≈1.7-1.9×纯写；**gfx938 写>读**（BW1100 1226 > 1059）
2. 向量化（dwordx4 vs dword）在 936/938 上 HBM 读增益 +20~28%；K100-AI 单跑出现 scalar 反快（待多跑/换 kernel 确认，可能是该卡延迟受限下 grid-stride 差异）
3. copy kernel 达理论 46-57%（928/936），非 roofline 满值；精确 roof 另用 tensor.copy_ 口径对照（dcu-kernel-layer-microbench 的标准法）
4. launch 2.0-2.5 µs 为背靠背空 kernel（无同步），配合 `SQ_WAVES<20` 判定 launch 轴问题
5. LDS 18.5-28.7 TB/s 为 64 block×256 thr×32KB 读写混合；bank conflict 影响另用 `SQ_LDS_BANK_CONFLICT` 验证
