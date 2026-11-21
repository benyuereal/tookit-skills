---
name: dcu-train-precision
description: "DCU训练精度排查：NV对齐、逐层定位异常层、NaN修复方法论"
version: 1.0.0
author: DCU Engineer
license: MIT
kind: tool
---

# 海光 DCU 训练精度排查（dcu-train-precision）

适用场景：在 **DCU 平台做深度学习训练**时，出现 Loss 不下降 / Loss NaN / 推理输出 NaN / 反向传播 NaN（Kernel Fault）/ 精度不达标（与 NV 平台有差距）等问题，需要系统化定位与修复。

> **来源与方法论**：内容基于内部文档《DCU平台训练精度排查最佳实践》（主文档 file_id `<internal-file-id>`，**纯自包含、已全文抓取、无内嵌子文档/无 spreadsheet 限制**）。本文只沉淀方法论 + 可复用调试脚本，**不存性能基线/逐行日志**。

## 0. 精度问题根因分类（先对号入座）

| 问题类别 | 典型表现 | 常见场景 |
|---|---|---|
| 混合精度问题 | AMP 训练不收敛 / loss NaN | Transformer 类、FP16 训练 |
| 算子兼容性问题 | 特定层输出异常 | Flash-Attention、自定义算子 |
| 算子反向数值溢出 | softmax 反向溢出 | Paddle 的 softmax 反向传播 |
| 框架版本不兼容 | 精度与预期不符 | Transformers 库版本差异 |
| 数值稳定性问题 | 梯度爆炸/消失 | 深层网络、大学习率 |

## 1. 通用排查四步法 + 核心原则

**四步法**（从问题发现到解决）：
```
问题发现
  → 第一步：环境复现与对比（固定随机种子 / NV vs DCU 对比 / 确定差异出现的 epoch/step）
  → 第二步：逐层数据采集（注册 forward 钩子 / 记录每层输入·输出·权重 / 保存日志）
  → 第三步：差异定位分析（对比两平台同层输出 / 定位首个异常层 / 分析该层算子实现）
  → 第四步：根因确认与修复（隔离验证如切 FP32 / 框架升级或补丁 / 算子替换或参数调整）
  → 问题解决
```

**核心原则**：
1. **控制变量**：固定随机种子，确保数据加载、模型初始化一致
2. **逐层对比**：从输入层到输出层，逐层定位首次出现显著差异的位置
3. **精度隔离**：通过切换 FP32/FP16 确认是否与混合精度相关
4. **最小复现**：构造最小化测试用例，排除无关干扰

## 2. 逐层 forward hook 调试工具（核心武器）

脚本见 `scripts/layer_hook_debug.py`（已 `py_compile` 验证）。用法：
```python
set_seed(42)                      # 必须在模型初始化之前
model = YourModel()
handles = register_hooks(model)    # 给所有叶子模块注册 forward hook
# ... 训练（输出重定向到日志：python train.py 2>&1 | tee debug.log）...
for h in handles: h.remove()       # 用完移除
```
hook 会逐层打印 **Input Shape / 前 10 输入值 / Weight Shape / 前 10 权重值 / Bias / Output Shape / 前 10 输出值（卷积层还打 FLOPs）**。

**日志对比分析法**（NV 与 DCU 同种子各跑一遍，对比 debug.log）：

| 对比项 | 正常表现 | 异常表现 |
|---|---|---|
| 输入值 | 两平台数值相近 | 差异显著（>1e-5） |
| 权重值 | 两平台数值相近 | 某层权重异常增大或为 0 |
| 输出值 | 两平台数值相近 | 出现 NaN/Inf 或全零 |
| 数值范围 | 合理范围内 | 过大（>1e10）或过小（<1e-10） |

→ 从输入层开始逐层检查，**首个出现显著差异或 NaN 的层即为根因所在**。

典型异常日志示例（定位结论）：
```
Conv2d:
  Input Shape: [1, 64, 128, 128]
  First 10 input values: [0.00000, 0.00000, ...]  ← 输入全零
  Output: ⚠ WARNING: Output contains NaN!           ← 输出 NaN
定位结论：输入全零导致后续计算异常，需检查上游层或数据加载
```

## 3. 常见精度问题处理（4 类实战）

### 3.1 AMP 训练 Loss NaN（混合精度）
根因：梯度下溢/上溢；Attention 中 Softmax 数值不稳定；Q/K/V 张量内存布局不连续。
| 方案 | 操作 | 适用 |
|---|---|---|
| ① FP32 兜底 | 关 AMP，纯 FP32 | 快速验证/临时 |
| ② 升级 Transformers | 升级到 `>=5.0.0`（不替换 torch/flash_attn 轮子；若该版本尚未正式发布，用当前最新稳定版） | 首选 |
| ③ 修复 SDPA | 改 `sdpa_attention.py` 加 `contiguous()` | 无法升级时 |

方案③临时 patch（`/usr/local/lib/python3.x/dist-packages/transformers/integrations/sdpa_attention.py`）：
```python
def _sdpa_attention(...):
    query = query.contiguous()   # 加这两行保证 Q/K/V 内存连续
    key = key.contiguous()
    value = value.contiguous()
    # ... 原有代码
```

### 3.2 MMCV 训练精度问题
检测/分割类模型 loss NaN：在 config 里调 `optimizer_config`——FP16 有问题就改 FP32 或设 `loss_scale='dynamic'`：
```python
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# 或禁用 FP16：optimizer_config = dict(type='OptimizerHook')
```

### 3.3 Flash-Attention 兼容性
表现：AMP 训练 loss NaN；NV 训好的 AMP 模型在 DCU 推理输出 NaN。
| 优先级 | 方案 | 操作 |
|---|---|---|
| 首选 | 升级 Transformers | `pip install transformers>=5.0.0` |
| 备选 | 临时 Patch | 改 sdpa_attention.py 加 contiguous() |
| 兜底 | 禁用 Flash-Attention | `attn_implementation="eager"` |

禁用示例：
```python
model = AutoModel.from_pretrained(model_path, attn_implementation="eager")
```

### 3.4 Paddle softmax 反向溢出
表现：uie-x-base FP32 训练报 Kernel VMFault；注册钩子打印反向传播时 inf/NaN（数据溢出）。
根因：Paddle 自带 softmax 算子有溢出风险。
→ 临时手写 stable softmax 绕过：
```python
def stable_softmax(logits):
    max_val = paddle.max(logits, axis=-1, keepdim=True)  # 减最大值保数值稳定
    shifted_logits = logits - max_val
    exp_logits = paddle.exp(shifted_logits)
    return exp_logits / paddle.sum(exp_logits, axis=-1, keepdim=True)
```

## 4. 问题处理快速参考表

| 问题现象 | 优先排查 | 推荐方案 |
|---|---|---|
| Loss NaN + AMP 开启 | 混合精度稳定性 | ①升级 Transformers ②检查 Q/K/V 连续性 |
| 特定层输出全零 | 上游层输出或权重初始化 | 检查激活函数、LayerNorm 参数 |
| 输出值爆炸（>1e10） | 梯度裁剪、学习率 | 降低学习率、加梯度裁剪 |
| 两平台数值差异大 | 算子实现差异 | 替换为等价算子、检查原子操作 |
| 仅推理时 NaN | 模型权重加载 | 检查权重转换精度、禁用 AMP 推理 |
| Paddle 首轮训练 kernel fault | softmax 反向严重溢出→NaN | 手动实现 softmax、替换修复框架 |

## 5. NaN/Inf 诊断脚本

脚本见 `scripts/diagnose_nan.py`（已 `py_compile` 验证）。快速检查模型输出是否含 NaN/Inf：
```python
diagnose_model(model, dataloader, device)   # 遍历前 5 个 batch，打印 Loss/Logits 的 NaN 报警
```

## 6. 与兄弟 skill 关系
- `dcu-ontrack-experience`：报错反查（偏推理/部署报错关键词）；本 skill 专注**训练侧**精度对齐。
- `dcu-pytorch-source`：PyTorch 后端/海光适配索引（算子实现差异排查时用）。
- `dcu-kernel-dev`：自定义算子开发（定位到具体算子实现问题后改 kernel）。
- `dcu-vllm-deploy` / `dcu-inference-optimization`：推理侧（与训练精度是不同环节，但"数值稳定性"原则相通）。

## 脚本运行前提（C4 标注）

本 skill 的 `scripts/layer_hook_debug.py`、`scripts/diagnose_nan.py` 是 **PyTorch 训练调试工具**，运行前提：

- 需 **PyTorch/torch 环境**（模型可加载、可训练/推理）。
- 调试日志由用户本地产生——本 skill 不存日志，只给方法与脚本。
- 两脚本已 `python3 -m py_compile` 验证；在无真实训练环境时仅作语法校验。

> 注：训练精度排查脚本需要真实 PyTorch 环境才能发挥调试作用；无环境时仅静态校验。