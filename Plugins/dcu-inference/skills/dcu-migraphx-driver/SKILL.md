---
name: dcu-migraphx-driver
description: "海光DCU上MIGraphX命令行工具migraphx-driver源码解析与扩展"
version: 1.0.0
author: DCU Engineer
license: MIT
---

# 海光 DCU MIGraphX Driver 工具扩展（dcu-migraphx-driver）

适用场景：在 **DCU 平台用 MIGraphX（AMD/海光推理优化器）的命令行驱动工具 migraphx-driver** 做模型加载/编译/运行/验证/性能测试，并**基于现有源码扩展自定义调试命令**。本文是**源码解析 + 功能扩展实战地图**：先结构化解析 migraphx-driver 原有架构，再讲如何新增 3 个实用命令（info / op_stat / test-op）。

> **来源**：内部文档《migraphx-driver的源码解析和功能增加实践》（file_id `<internal-file-id>`，<internal-author>，纯自包含已全文抓取；含 3 块 base64 使用示例截图，已按纪律剔除、用文字描述图意，不搬运二进制）。
> ⚠ **DCU 适配身份**：MIGraphX 通过 HIP 生态兼容 DCU，海光 DCU 装 MIGraphX 即自带 `migraphx-driver` CLI 工具（无需额外安装）。本文与其姊妹篇 `dcu-migraphx-custom-op`（自定义算子层）互补——本文是**工具/CLI 层**的命令扩展。
> ⚠ **不存性能数字**：本文是开发/调试方法，无基准；实测性能请用 dcu-inference-optimization 的方法学评估。
> ⚠ **完整 C++ 源码外置** `scripts/`（info/op_stat/test_op 三个 command 结构体，做全角归一化，与文档 §七 对应）。本 SKILL.md 只讲架构和关键片段，避免膨胀。

## 0. 一句话机制

migraphx-driver 是 MIGraphX 的 CLI 驱动，核心是「命令注册（command<T>）→ 参数解析（argument_parser）→ 逻辑执行（parse/run）」；扩展新命令只需继承 `command<T>` 实现 parse/run 并复用 loader/compiler 基础模块，模板自动注册。

## 1. 原有源码结构（三大模块）

| 层 | 模块 | 职责 |
|---|---|---|
| 基础工具 | `argument_parser` | 命令行参数解析（位置/选项/分组/帮助），如 `ap(file, {}, ap.metavar("<input file>"), ap.file_exist(), ap.required(), ap.group("input"))` |
| 基础工具 | `loader` | 模型加载（ONNX/TF/MIGraphX 二进制/JSON 自动识别后缀），支持 `--batch`/`--optimize`/`--input-dim`，执行 trim/optimize/mlir 预处理，可 save 为文本/JSON/C++/Python |
| 基础工具 | `program_params` | 生成参数映射（parameter_map），支持 `--fill0`/`--fill1`/`--load-arg`，适配静态/动态维度 |
| 基础工具 | `compiler_target` | 编译目标设备（GPU/CPU/ref 三种），`get_target` 返回 target 对象 |
| 核心功能 | `compiler` | 整合 loader+program_params+compiler_target，完成加载/量化/编译全流程；量化支持 **FP16/FP8/INT8/INT4**（调用 `quantize_fp16`/`quantize_int8` 等）；`program::compile` 自动选 NCHW/NHWC 布局 |
| 命令模块 | `command<T>` | 每个命令继承此模板，实现 `parse`（参数解析）和 `run`（逻辑执行），注册后命令行可调用 |

**原有命令清单**：read（格式转换）/ params（参数名+维度）/ verify（逐指令/简化/二分验证+误差容忍）/ compile（仅编译）/ run_cmd（编译运行）/ time_cmd（测时）/ perf（算子性能报告）/ roctx（rocTX 标记）/ op·onnx·tf（算子列表）/ main_command（默认帮助）。

## 2. 整体运行流程（4 步）

1. **参数解析**：`main` 读命令行参数，判命令类型，调对应 `parse`
2. **模型加载与预处理**：`loader` 加载，执行 trim/optimize
3. **模型编译（可选）**：`compiler` 编译到目标设备，执行量化/布局优化
4. **命令逻辑执行**：按命令类型执行运行/验证/性能测试，输出结果

## 3. 量化支持（DCU 推理优化常用）

`compiler` 结构体支持 **FP16 / FP8 / INT8 / INT4** 四种量化模式（通过 `quantize_fp16` / `quantize_int8` 等方法），是 DCU 上做推理精度/性能权衡的关键路径。

## 4. 新增 3 指令（本文核心，完整源码见 `scripts/`）

### 4.1 info —— 模型基础信息
- **功能**：在原有 `params`（仅参数名+维度）基础上扩展，输出 file/batch/optimized/parameters/parameter_shapes（名称+类型），支持 `--json`（机器友好）与文本两种格式
- **实现**：继承 `command<info>`，复用 `loader` 加载，`program` 对象取信息；`run` 里 `if(json_output)` 走 `to_pretty_json_string` 否则文本打印
- **关键片段**（`scripts/info.hpp`）：
```cpp
if(json_output){
    value v;
    v["name"] = l.file; v["batch"] = l.batch;
    v["parameters"] = p.get_parameter_shapes().size();
    v["optimized"] = l.optimize;
    value params;
    for(auto&& [k,s] : p.get_parameter_shapes()) params[k] = s.type_string();
    v["parameter_shapes"] = params;
    std::cout << to_pretty_json_string(v) << std::endl; return;
}
```

### 4.2 op_stat —— 算子数量统计
- **功能**：遍历主模块所有指令，统计各算子调用次数，按次数降序输出，帮助定位核心算子（重点优化对象）
- **实现**：继承 `command<op_stat>`，复用 `loader`；遍历 `*mod` 用 `ins.name()` 计数，`std::sort` 降序
- **⚠ 易错点**：`ins` 是 `instruction` **对象（非指针）**，必须用 `.` 访问（`ins.name()`）而非 `->`
- **关键片段**（`scripts/op_stat.hpp`）：
```cpp
std::unordered_map<std::string, size_t> op_count;
for (auto&& ins : *mod) {          // ins 是对象，用 .
    std::string op_name = ins.name();
    op_count[op_name]++;
}
```

### 4.3 test-op —— 调试单个算子
- **功能**：按算子名查找模型中所有匹配实例，输出索引+输出维度（`ins.get_shape()`），便于排查维度不匹配/算子实现错误
- **参数**：`--op`（目标算子名）、`--all`（显示所有匹配，默认只第一个）
- **实现**：继承 `command<test_op>`，复用 `compiler`（需编译模型获取输出维度）；遍历 `*module` 匹配 `ins.name() == target_op`
- **关键片段**（`scripts/test_op.hpp`）：
```cpp
for(auto&& ins : *module){
    if(target_op.empty() || ins.name() == target_op){
        std::cout << "[OP] " << ins.name() << " (index: " << idx << ")" << std::endl;
        std::cout << "[OUTPUT SHAPE] " << ins.get_shape() << std::endl;
        if(!show_all) break;
    }
    idx++;
}
```

> 3 个命令的**使用示例截图**（原文档 base64 内联图，已剔除）：info 示例输出模型文件路径/批量大小/优化标志/参数数量及类型列表；op_stat 示例输出 `Total instructions` + 各算子降序列表；test-op 示例输出 `[OP] 算子名 (index: N)` + `[OUTPUT SHAPE] 维度`。

## 5. 集成与兼容

- **集成**：三个命令均继承 `command<T>`，实现 parse/run，注册到命令映射（**模板类自动关联，无需手动注册**），可直接命令行调用
- **兼容保障**：① 代码风格兼容（小写命名/缩进/注释）；② 依赖兼容（仅依赖 loader/compiler/argument_parser，不引入新依赖）；③ 功能兼容（不修改原有命令逻辑，老用户习惯不变）

## 6. 扩展建议

- 增强 info：加输入/输出节点信息、模型大小、优化前后指令数对比
- 增强 op_stat：结合 perf 模块加算子平均耗时
- 增强 test-op：加算子输入/输出数据打印、单独运行验证
- 新增命令：`model_compare`（对比两模型结构差异）、`op_profile`（单算子性能分析）

## 7. 与兄弟 skill 关系
- `dcu-migraphx-custom-op`：同框架（MIGraphX）**算子层**扩展（写 HIP 核函数算子）；本文是**工具/CLI 层**扩展（migraphx-driver 命令），互补
- `dcu-kernel-dev`：HIP/TileLang 算子开发规范（本文调试的算子底层实现见那里）
- `dcu-inference-optimization`：推理优化方法论（量化 FP16/FP8/INT8/INT4 的实测评估见那里）
- `dcu-pytorch-source`：PyTorch 后端适配（MIGraphX 可作为 PyTorch 之外的独立推理优化器）

## 脚本运行前提（C4 标注）

本 skill 的 `scripts/*.hpp` 是 **MIGraphX 命令行工具（migraphx-driver）的 C++ command 结构体源码**（info/op_stat/test_op 三个指令，做全角归一化），运行前提：

- 需 **DCU 真机环境**（有 `/dev/kfd` 或 `/dev/dri`、挂 `/opt/hyhal:ro`）或对应 DTK 版本环境。
- 需可访问 **migraphx-driver 二进制**及源码（本 skill 的 scripts 解析其命令行行为）。
- 无 DCU 真机时，仅能作**结构与语法校验**（C++ 需 DTK + migraphx 源码环境编译，非 `py_compile`），不能验证实际解析动作。

> 注：`migraphx-driver` 的扩展/源码解析脚本并非通用 Python 工具，依赖 DCU/DTK 环境。按用户纪律，脚本运行前提必须标注；无真机环境时只做静态校验。
- 不存性能基线/逐行日志；外链无
- 高危操作谨慎，优先测试机验证
