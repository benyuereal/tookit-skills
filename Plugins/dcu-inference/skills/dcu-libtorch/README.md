# dcu-libtorch — 海光 DCU LibTorch C++ 推理部署

海光 DCU 上用 C++ LibTorch 做模型推理部署：LibTorch 与 PyTorch 区别对照（7 项）、TorchScript 模型互操作（Python 导出 .pt / C++ 加载）、DCU 适配身份（libtorch_hip 非 cuda 版、设备语义 torch::kCUDA、编译用 /opt/dtk/dcc/bin/dcc）、Makefile 依赖库清单、模型加载推理 C++ 代码、关键 dcc 编译命令、FAQ（依赖库链接顺序）。

## 目录

```
dcu-libtorch/
└── SKILL.md   # agent 向：完整部署地图 + C++ 片段 + dcc 编译命令
```

## 快速要点

- DCU 用 **libtorch_hip**（不是 cuda 版 libtorch_cuda）；设备语义复用 `torch::kCUDA`
- 编译用 **`/opt/dtk/dcc/bin/dcc`**（不是 g++），`-std=c++17`，链接顺序 `-ltorch_hip -lc10_hip -ltorch_cpu -lc10`
- Python 训好 → `torch.jit.script` 导出 `.pt` → C++ `torch::jit::load` 加载推理
- 不存性能数字；C++ 片段原样保留

> 详细内容（编译命令、C++ 推理代码、链接顺序）见 `SKILL.md`。
