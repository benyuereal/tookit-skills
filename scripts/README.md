# DCU Toolkit — 包根 scripts/ 索引

## 本目录存放包级共享脚本

本目录只放**跨子 skill 的包级工具脚本**；各子 skill 自己的可执行脚本仍在其各自的 `scripts/` 目录下，不下沉到这里。

| 文件 | 用途 |
|---|---|
| `probe_stream.py` | 流式带宽实测：read（reduce/sum，分母 1× 字节）/ copy（R+W，分母 2×，另给单边值）/ write（fill）三种口径**分开命名**，扫 16/64/256/1024 MiB 逐尺寸报；`--json` 机器可读；`--check` 与 `op-roofline/constants.yaml` 对账（偏差 >2% 报警，退出码 1）。输出明确标注「实测流式峰值，不是型号标称值」。无 torch/GPU 时明确说无法探测，不静默给默认值 |
| `probe_pattern.py` | 访存 run-length 扫描：固定行跨度（默认 32 KB），扫每行连续 16..2048 B 与 `linear` 参照组，每种 run 读相同总字节数；输出 `run_bytes / achieved_gbs / pct_of_linear`，`--json` 机器可读。**任何「行内连续 X 字节」的 kernel 先跑它拿 pattern ceiling，再用它当分母算达成率**（用流式峰值当分母，分母就错了）。内置探针内核用 hipcc 编译缓存（约 2 s），缺 torch/hipcc 时明确降级 |
| `pmc.py` | PMC CSV 的可执行解析器（`op-profiling` 的唯一权威实现，`references/pmc_reference.py.md` 只是说明页）：稳态取单次 GRBM 最小行、先排 `_ZN2at`/`_ZN7rocprim` 等噪声、`TA_*` 跨 SE 实例求和、CU 数等常数不写死（缺了打 `unknown(<原因>)`）。只打印原值、不下结论；`--json` / `--selftest` |
| `run_evals.py` | evals 静态检查器：扫全库 `*/evals/evals.json`，校验 skill 字段与目录一致、true 案例带 expect / false 案例只给 prompt、id 形如 `<skill>-tN/-fN` 且编号连续、全库 id 唯一；并校验 `SKILL.md` description 的「触发条件 + 不适用反例」（带 evals 的 skill 不合规即 FAIL，`--strict-description` 全库阻塞）。用法 `python scripts/run_evals.py [--root <包根>]`，PASS 退 0、FAIL 退 1 |
| `skill_eval.py` | evals 动态执行器：每条用例起一次 `claude -p --plugin-dir <插件>` 独立会话，判「该不该拉起这个 skill」（true=第一个 Skill 调用是它；false=全程不得出现），`--judge` 再判回答要点。烧 token，不进 CI。用法见 `python3 scripts/skill_eval.py --list` / `--skill <name>` / `--domain <plugin>`，完整流程见 [`docs/EVALS.md`](../docs/EVALS.md) |
| `json_schema_mini.py` | 零依赖 JSON Schema 子集校验器（type/required/properties/items/enum/const/minItems/pattern/minLength/minimum/maximum/additionalProperties）。用法 `python scripts/json_schema_mini.py <schema.json> <data.json>`；现被 dcu-image-hub / dcu-modelzoo 的 fetch 脚本在快照落盘后自动调用，schema 文件放各 skill `data/*.schema.json` |
| `hygon-upstream-sync.py` | HYGON-AI 上游仓库周更同步（35 仓库索引/版本锚点见包根 `references/hygon-upstream/repos-index.md`；配套的内部 skill `dcu-upstream-integration` 未随本包分发） |
| `opgate_checks.py` | ops 门禁（`Plugins/ops/hooks/ops_gate.py`）的检查函数：A 契约 / B 执行身份 / C 方向预算 / D 交付件 / E 编译产物 / F verified 提案。hook 只做动作识别与判定编排，判定本身仍调用 `opverify` / `opmanifest` / `loop.py` |
| `probe_stream.py` | **流式带宽探针**：分 read / copy / write 三种口径测实测峰值，按 buffer 尺寸分别报（小尺寸落 L2，不能当 HBM 峰值），`--check` 与 `op-roofline/constants.yaml` 对账。**任何达成率的百分比都必须用它给出的分母，不能用型号标称值**。先用 `bash -c` 之外的方式确认机器空闲 |
| `install_sync_selftest.sh` | install.sh 清单式增量同步语义自测（7 用例：尾斜杠/用户新增/上游移除/本地改动备份/幂等等），发布前可跑 |

**可执行的 `.sh` 脚本全部在各子 skill 自己的 `scripts/` 目录下**，例如：

- `Plugins/dcu-env/skills/dcu-rccl-test/scripts/rccl_test_single_node.sh` — 单节点 RCCL 通测一键流程
- `Plugins/dcu-env/skills/dcu-env/scripts/check_iommu_acs.sh` — BIOS ACS/IOMMU 检测（**主要针对 PCIe 形态卡：K100-AI/BW10/BW100/BW150 等；OAM 产品 BW1000/BW1100 基本无影响，无需为此改 BIOS/内核**）
- `Plugins/dcu-env/skills/dcu-cpu-affinity/scripts/gen_affinity.sh` — NUMA 绑核方案生成
- `Plugins/dcu-env/skills/dcu-driver/scripts/check_driver.sh` — 驱动加载验证
- `Plugins/dcu-env/skills/dcu-hy-smi/scripts/hy_smi_query.sh` — hy-smi 查询封装
- `Plugins/dcu-env/skills/dcu-hy-smi/scripts/examples.sh` — 原仓库示例脚本（参考）
- `Plugins/dcu-env/skills/dcu-topology/scripts/collect_topology.sh` — 拓扑采集
- `Plugins/dcu-env/skills/dcu-env/scripts/vendor_dcuprofiletools/...` — 海光官方环境检测工具（第三方 vendor，**被 `.gitignore` 排除、不随包分发**，需要时从官方渠道获取，见 `dcu-env` skill）

> 引用约定：文档中凡提及脚本，**一律写完整相对路径**（如 `Plugins/dcu-env/skills/dcu-topology/scripts/collect_topology.sh`），
> 不写裸名（如 `collect_topology.sh`）以免机检误报"文件缺失"。shell 命令示例里"已进入脚本目录后直接 `bash xxx.sh`"
> 属正常操作，不在该约定约束内。
>
> 判断标准：文档叙述里出现 `<某>.sh` 时，先按"完整相对路径 → 子 skill scripts/ → 外部仓库/示例占位"顺序核对，
> 确认都不是才视为真缺失。

## 历史坑（避免重复排查）

早期版本的本 README 曾罗列 `env_detect.sh` / `collect_metrics.sh` / `parse_hy_smi.py` /
`gen_report.py` 等脚本，并标注"\[待创建\]"——但这些脚本**从未创建**，也从未被任何子 skill
引用，属于"文档许诺但未实现"的悬空描述，会让人误以为这里有现成脚本。
现已删除该误述。2026-10-15 起本目录按上文规则正式承载包级共享脚本（质量门/校验器），
此前"占位"定位随之作废——新增包级脚本须同时登记到上方表格。
