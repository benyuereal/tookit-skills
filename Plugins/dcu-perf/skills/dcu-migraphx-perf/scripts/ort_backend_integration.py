# ONNX Runtime 后端集成 MIGraphX（DCU 平台）
# 来源：《【DCU】DCU MIGraphX 最佳性能实践手册》§6.1
# 说明：通过 MIGraphXExecutionProvider 让 ORT 走 MIGraphX 编译路径，
#       并保存编译引擎避免重复编译。
import onnxruntime as ort

provider_options = [
    {
        "device_id": "0",
        "migraphx_fp16_enable": "true",
        "migraphx_save_compiled_model": "true",  # 保存编译引擎
        "migraphx_save_model_path": "./model.mxr",
    }
]

session = ort.InferenceSession(
    "model.onnx",
    providers=["MIGraphXExecutionProvider"],
    provider_options=provider_options,
)
