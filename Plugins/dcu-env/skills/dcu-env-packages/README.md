# dcu-env-packages — DCU 适配软件包清单与安装排查

海光 DCU 适配版软件包（torch / torchvision / triton / flash_attn / flash_mla / lightop / lmslim / vllm / aiter / deepgemm / sglang / xformers 等 `+das` 轮）的清单、私有 PyPI 用法、版本绑定矩阵与排障。

数据来自 <device-B> 设备 `<container_vllm015>` 容器 `pip list` 实测 + sourcefind 私有 PyPI 在线枚举（2026-08-03 快照）。

## 关键结论（先看）
- 私有 PyPI 正确索引路径：`https://pypi.sourcefind.cn/release/dtk/<包名>/`（**不是** `/simple/<pkg>/`，那个返回 Hub 网页空壳）
- 按 DTK 版本锁源：`pip install torch -i https://pypi.sourcefind.cn/release/dtk2604/`
- 换 torch 大版本时，triton/flash_attn/flash_mla/lightop/lmslim/vllm 等必须跟着换对应 `torch<标记>` 的轮

## 安装 / 启用
本 skill 随 `dcu-*` 技能包就位，无需额外安装。详细清单/矩阵/排障见 `SKILL.md`。

## 快速用法
```bash
# 搭 DTK 26.04 + torch 2.9.0 基线组合（对齐 <device-B> 容器）
pip install torch torchvision triton flash_attn flash_mla lightop lmslim vllm \
  -i https://pypi.sourcefind.cn/release/dtk2604/
```

> 轮子持续更新，装前用浏览器/该 URL 复核最新版本，勿硬编码版本号进脚本。
