# dcu-image-to-native-env — DCU 镜像环境复刻为物理机原生 conda 环境

把 DCU 容器镜像里的环境**复刻为物理机原生 conda 环境**（非 Dockerfile 重建路线）：docker cp 拷 dist-packages + DTK + 命令行入口、拍平嵌套目录、补 Ubuntu 系统包、DTK 目录名决定 Triton clang 选择。适合没有容器运行时约束、要直接在物理机跑推理的场景。

## 目录

```
dcu-image-to-native-env/
├── SKILL.md                    # agent 向：复刻步骤 + Triton clang 选择 + 踩坑
├── references/kylin-old-glibc.md  # Kylin 老 glibc 场景的处理
└── templates/activate_dcu_env.sh  # 环境激活脚本模板
```

## 快速用法 / 要点

- 核心路径：容器内打包 → 物理机拍平 → conda 接管 → DTK 环境变量与 clang 对齐。
- Triton 的 clang 路径取决于 DTK 目录名（aillvm/clang-18 vs llvm/clang-17），选错编译崩。
- accelerate 1.13.0 循环导入需 .pth 预导入修复；aiter backend 缺函数改 fa3 + page-size 64。
- 推理模型 `reasoning_content` 字段陷阱与 `pkill` 自杀陷阱见 SKILL.md Pitfalls。

> 详细复刻步骤与验证方法见 `SKILL.md`。