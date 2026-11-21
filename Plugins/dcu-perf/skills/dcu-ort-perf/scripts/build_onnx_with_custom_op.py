# 创建包含自定义算子的 ONNX 模型（两种方式）
# 来源：《OnnxRuntime-DCU 最佳性能实践手册》§5.2
# 说明：方式一用 ONNX Python API 从头构建；方式二从 PyTorch 导出时映射自定义算子。
import onnx
from onnx import helper, TensorProto


# ========== 方式一：使用 ONNX Python API 从头构建模型 ==========
def build_model_from_scratch():
    # 定义算子节点
    custom_node = helper.make_node(
        op_type="MyCustomOp",                    # 算子类型名称
        inputs=["input_a", "input_b"],           # 输入名称列表
        outputs=["output"],                      # 输出名称列表
        name="my_custom_node",                   # 节点名称
        domain="my.domain",                      # 算子域（建议使用自定义域避免冲突）
        bias=2                                   # 算子属性
    )

    # 定义模型输入输出
    input_a = helper.make_tensor_value_info("input_a", TensorProto.FLOAT, [None, 3, 224, 224])
    input_b = helper.make_tensor_value_info("input_b", TensorProto.FLOAT, [None, 3, 224, 224])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 3, 224, 224])

    # 构建计算图
    graph = helper.make_graph(
        nodes=[custom_node],
        name="custom_op_graph",
        inputs=[input_a, input_b],
        outputs=[output]
    )

    # 创建模型（指定算子集版本）
    model = helper.make_model(
        graph,
        producer_name="custom_op_demo",
        opset_imports=[helper.make_operatorsetid("", 19)]  # ONNX opset 版本
    )

    # 保存模型
    onnx.save(model, "model_with_custom_op.onnx")
    print("ONNX 模型已保存")
    return model


# ========== 方式二：从 PyTorch 导出时映射自定义算子 ==========
def export_from_pytorch():
    import torch
    import torch.onnx
    from torch.onnx import register_custom_op_symbolic

    class MyModel(torch.nn.Module):
        def forward(self, x, y):
            # 使用 torch ops 作为占位符
            return x * y + 2  # 将被映射为 MyCustomOp

    model = MyModel()
    dummy_x = torch.randn(1, 3, 224, 224)
    dummy_y = torch.randn(1, 3, 224, 224)

    # 自定义符号函数，将 torch 操作映射到自定义 ONNX 算子
    def my_op_symbolic(g, x, y):
        return g.op("my.domain::MyCustomOp", x, y, bias_i=2)

    register_custom_op_symbolic("::mul", my_op_symbolic, 9)

    # 导出模型
    torch.onnx.export(
        model,
        (dummy_x, dummy_y),
        "model_from_pytorch.onnx",
        opset_version=11,
        input_names=["input_a", "input_b"],
        output_names=["output"]
    )


if __name__ == "__main__":
    build_model_from_scratch()
    # export_from_pytorch()  # 需要 torch 环境
