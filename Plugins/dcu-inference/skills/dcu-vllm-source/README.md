# dcu-vllm-source — README

## 定位
vLLM 各版本源码结构与 DCU 适配点索引（合并 0.15/0.18/0.21.0/0.23.0/0.26 五版；历史五版目录保留参考骨架）。按目标版本查结构/适配点。

## references
无独立 references 文件；版本索引总表 + 各版特有要点全在 SKILL.md 正文。

## 使用
遇「查某版本 vLLM 的 DCU 适配落点」时进 SKILL.md 按版本段取路径/适配结论，再与 `dcu-vllm-op-replace` / `dcu-vllm-custom-op` 组合走算子开发闭环。

## 兄弟 skill
- `dcu-vllm-custom-op`（vLLM 自定义算子开发流程）
- `dcu-vllm-op-replace`（vLLM 算子替换流程）
- `dcu-aiter-source` / `dcu-deepgemm-source`（AITER / DeepGEMM 算子级展开）
