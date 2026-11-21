# Task09：CentreRandomAugmentation

## 优化思路

**瓶颈：** Memory-bound，多个小操作链式处理每个 sample。

**核心洞察：语义一致性 + 随机数融合**

同一样本的旋转变换和平移变换必须一致，不能每次调用得到不同结果。随机数生成必须与 PyTorch 参考语义等价。

## 关键代码

```python
# 随机数在 kernel 入口一次性生成，保证同一样本同一结果
@triton.jit
def seed_based_random(seed):
    return (seed * 1103515245 + 12345) & 0x7fffffff

def random_uniform(seed):
    return (seed_based_random(seed) % 1000000) / 1000000.0

# program 映射：grid = (num_samples,)
pid = tl.program_id(0)
off = pid * feat_dim + tl.arange(0, feat_dim)
sample = tl.load(sample_ptr + off)

# 入口生成所有随机数
seed = pid * 12345 + tl.load(seed_ptr)
rand_rot = random_uniform(seed)      # 旋转变换用
rand_trans = random_uniform(seed + 1)
rand_scale = random_uniform(seed + 2)

# 融合所有操作
rotated = rotate(sample, rand_rot)           # 旋转
translated = translated + rand_trans          # 平移
scaled = scaled * (1.0 + rand_scale * 0.1)   # 缩放
result = compose(rotated, translated, scaled)
```

## 陷阱

- 同一样本两次运行结果必须一致（语义一致性）
- 随机数分布和值都必须与 PyTorch 一致，不能只看 atol

## BW1000 结果

PyTorch 0.935ms → Triton 0.168ms，**5.55×**