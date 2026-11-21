# torch_migraphx 集成（DCU 平台）
# 来源：《【DCU】DCU MIGraphX 最佳性能实践手册》§6.2
# 说明：把 PyTorch 模型经 fx.lower_to_mgx 转成 MIGraphX 优化模型后推理。
import torch
import torch_migraphx
import torchvision

model = torchvision.models.resnet50().eval()
sample_input = torch.randn(1, 3, 224, 224)

# 转换为 MIGraphX 优化模型
model_mgx = torch_migraphx.fx.lower_to_mgx(
    model,
    [sample_input],
    lower_precision=torch_migraphx.LowerPrecision.FP16,
)

# 推理
output = model_mgx(sample_input)
