# 用户知识库（kb/）— 沉淀落点约定

安装时只把**本 README** 拷到用户知识库根 `~/.config/dcu-toolkit/kb/`（install.sh 只创建、
**永不删除、永不覆盖**）；模板不拷贝，在包内 `templates/`（安装态在 `<skills根>/_dcu-toolkit-shared/` 下只读）查阅。
产出的记录写在那里，不写进 skill 树。

## 为什么不在 skill 树里沉淀

`<skills根>/` 与 `<skills根>/_dcu-toolkit-shared/` 是**厂商维护内容**：重装/升级以包内版本为准。
install.sh 的清单式同步已保证「用户新增的文件不会被删、本地改过的厂商文件被覆盖前备份为
`<file>.local.bak`」，但记录类内容的正确落点仍是 `kb/`——那里的文件从来不参与同步。

## 目录与谁写入

| 目录 | 内容 | 谁写入 |
|---|---|---|
| `kb/op-opt/` | 算子优化台账：算子 / shape / 前后数字 / 加速比 / 正确性 / 环境指纹 / 证据路径 | 自动：`op-optimize` 工作流收尾 `op-distill` 节点（`dcu_distill.py` 生成台账 + 证据 JSON） |
| `kb/issue/` | 故障排查记录（现象、环境、过程、根因、解法、复现条件） | 人工：`fault-triage` 链路收尾（`fault-fix` 通过后）按 `templates/issue-record-template.md` 补写；暂无自动节点 |
| `kb/solution/` | 部署方案卡片（硬件/软件栈/调优参数/性能/适用边界/已知问题） | 人工：`model-deploy` 链路收尾（`deploy-verify` 通过后）按 `templates/product-card-template.md` 补写；暂无自动节点 |

模板：`data/kb/templates/`（台账 / 故障记录 / 方案卡片各一份；安装态在 `<skills根>/_dcu-toolkit-shared/` 下只读）。

## 条目纪律（硬要求）

1. 每条事实带来源：`source`（命令/产物路径）+ `verified_on`（日期 + 设备 + gfx + DTK）。
   不确定的标 `status: unverified`，不得写成结论腔。
2. 数字必须可回溯：写进台账的每个数字要能回到 evidence 文件（如
   `runs/op-optimize/validate/result.json`）的字段或命名的 profiling 证据。
3. 性能数字标 `[历史数据，随硬件/DTK/版本变化]`，优先给相对倍数/百分比；
   **数字不进可复用事实条目**（事实条目只留方法与架构结论）。
4. 脱敏红线见 `docs/STANDARDS.md`：零内网地址、零真实凭证、设备号占位化。
5. **台账条目按五段写**：`结论 / 前提 / 已证伪 / 复现命令 / 口径`。缺 `口径` 的结论不可复用
   ——同一个数换个分母、频率档或标称峰值，结论可能反转。
   自查与检索用 `dcu-orchestrator/scripts/kbq.py`：

   ```bash
   python3 scripts/kbq.py index                # 全部条目 + 缺哪段
   python3 scripts/kbq.py search <关键词>       # 检索，命中处标出所属段落
   ```

   缺段的卡**仍能检索到，但输出会标红**（历史卡不必回补，新卡按五段写）。
   `promotion-*.md` 是晋升提案骨架，不套这条规则。

## 与 skill 的关系（晋升路径）

沉淀是两级的：

- **台账/记录**（本目录）：环境专属，含数字，可自由追加，不参与对外发布。
- **可复用事实**（skill 层）：跨项目复用的手法/坑/架构结论。由 `op-distill` 产出**晋升提案**
  （写明拟并入的 skill 文件路径 + 明确不并入项），**经人工/厂商审核后才并入 skill**，
  不自动改写厂商 skill 内容。

客户侧 agent 不回推厂商仓库（无凭证，也不该改厂商内容）。