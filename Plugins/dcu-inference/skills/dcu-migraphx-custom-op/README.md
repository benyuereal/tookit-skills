# dcu-migraphx-custom-op — 海光 DCU MIGraphX 自定义算子（C++ + HIP）

海光 DCU 上用 MIGraphX 推理优化器写自定义算子：4 固定步骤（解析ONNX→添加算子→编译→推理）、3层封装（op_add 添加API / op_add_api 高层API / custom_op_register 注册宏 REGISTER_CUSTOM_OP）、1个核心（算子结构体继承 experimental_custom_op_base + 实现 name/runs_on_offload_target/compute/compute_shape 纯虚函数）、HIP 核函数（hip_resize_kernel 双线性插值+标准化+NHWC→NCHW、hip_crop_center_kernel 中心裁剪）、4步调用完整 C++ 实例、流程总结「3层封装+1个核心+4步调用」。

完整工程源码外置 `scripts/`（7 文件 + Makefile，用 hipcc 编译 + /opt/dtk 头文件/库）。

详见 `SKILL.md`。
