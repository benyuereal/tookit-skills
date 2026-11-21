// 自定义算子：C++ 高性能实现（生产环境推荐）
// 来源：《OnnxRuntime-DCU 最佳性能实践手册》§5.1
// 说明：继承 Ort::CustomOpBase 实现自定义算子，编译为动态库后注册。
//       示例：MyCustomOp，计算 out = x * y + bias。
// 注意：示例 Compute 用 CPU 循环做伪代码；实际在 DCU 应使用 HIP kernel 编程实现并行。
//       DCU 上可通过 Ort::Custom::RocmContext 获取 HIP 流和 miopen 句柄。
#include <onnxruntime_cxx_api.h>

struct MyCustomOpKernel {
    int64_t bias_ = 1;

    MyCustomOpKernel(const OrtApi& api, const OrtKernelInfo* info) {
        // 获取算子属性
        int64_t attr = 0;
        OrtStatus* status = api.KernelInfoGetAttribute_int64(info, "bias", &attr);
        if (status == nullptr) {
            bias_ = attr;
        }
        // 默认值 1
    }

    void Compute(const OrtApi& api, OrtKernelContext* context) {
        // 获取输入张量
        const OrtValue* input_x = nullptr;
        const OrtValue* input_y = nullptr;
        api.KernelContext_GetInput(context, 0, &input_x);
        api.KernelContext_GetInput(context, 1, &input_y);

        // 获取张量信息
        OrtTensorTypeAndShapeInfo* info_x = nullptr;
        api.GetTensorTypeAndShape(input_x, &info_x);
        size_t element_count = api.GetTensorShapeElementCount(info_x);

        // 获取数据指针
        float* x_data = nullptr;
        float* y_data = nullptr;
        api.GetTensorMutableData(input_x, (void**)&x_data);
        api.GetTensorMutableData(input_y, (void**)&y_data);

        // 创建输出张量
        OrtValue* output = nullptr;
        api.KernelContext_GetOutput(context, 0, info_x, &output);
        float* out_data = nullptr;
        api.GetTensorMutableData(output, (void**)&out_data);

        // 执行计算: out = x * y + bias（伪代码，实际在 DCU 应使用 kernel 编程实现并行）
        for (size_t i = 0; i < element_count; i++) {
            out_data[i] = x_data[i] * y_data[i] + (float)bias_;
        }
    }
};

// 注册算子
struct MyCustomOp : Ort::CustomOpBase<MyCustomOp, MyCustomOpKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new MyCustomOpKernel(api, info);
    }
    const char* GetName() const { return "MyCustomOp"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 2; }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetInputType(size_t index) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    ONNXTensorElementDataType GetOutputType(size_t index) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

// 导出注册函数（用于动态库）
ORT_EXTERN_C OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                                       const OrtApiBase* api_base) {
    OrtCustomOpDomain* domain = nullptr;
    api_base->GetApi(ORT_API_VERSION)->CreateCustomOpDomain("my.domain", &domain);
    static MyCustomOp op;
    api_base->GetApi(ORT_API_VERSION)->CustomOpDomain_Add(domain, &op);
    api_base->GetApi(ORT_API_VERSION)->AddCustomOpDomain(options, domain);
    return nullptr;
}
