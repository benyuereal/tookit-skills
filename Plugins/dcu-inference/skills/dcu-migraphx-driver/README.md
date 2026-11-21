# dcu-migraphx-driver — 海光 DCU MIGraphX Driver 工具扩展（CLI 命令开发）

海光 DCU 上用 MIGraphX 自带命令行工具 migraphx-driver 的源码解析与功能扩展：原有源码三大模块（基础工具 argument_parser/loader/program_params/compiler_target、核心功能 compiler 支持 FP16/FP8/INT8/INT4 量化、命令模块 command<T> 继承 parse/run + 原有命令 read/params/verify/compile/run_cmd/time_cmd/perf/roctx/op）、4步运行流程、新增 3 指令（info 模型基础信息+--json / op_stat 算子统计降序注意 ins 对象用. / test-op 按名查算子实例+输出维度 --op/--all）、集成兼容（继承 command<T> 自动注册）、扩展建议（model_compare/op_profile）。完整 C++ 源码外置 `scripts/`（info/op_stat/test_op 3 个 struct）。

详见 `SKILL.md`。
