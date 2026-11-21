// MIGraphX 自定义算子 — ResNet50 预处理算子结构体（resnet50_preprocess.hpp）
// 来源：内部文档《MIGRAPHX 自定义预处理算子（C++）实现流程深度解析》§6.5
#ifndef RESNET50_PREPROCESS_HPP
#define RESNET50_PREPROCESS_HPP

#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <migraphx/migraphx.hpp>
#include <hip/hip_runtime.h>

// 修复1：定义 HIP 断言
#define MIGRAPHX_HIP_ASSERT(...) ((void)__VA_ARGS__)

// 修复2：提前定义结构体
struct mSize{
    mSize() = default;
    mSize(size_t h,size_t w):height(h), width(w){}
    size_t width;
    size_t height;
};

// 修复3：提前定义枚举
enum inter_type {
    INTER_LINEAR = 0
};

// 修复4：【关键】提前声明函数
void resnet_resize(hipStream_t stream,
                float*   output_data,
                const uint8_t* input_data,
                float*  res_data,
                float ratio,
                const mSize   input_size,
                const mSize   output_size,
                size_t   channel,
                inter_type itype);

// 算子结构体
struct resnet_preprocess_custom_op final : migraphx::experimental_custom_op_base{

    virtual std::string name() const override { return "resnet_preprocess"; }
    virtual bool runs_on_offload_target() const override { return true; }

    virtual migraphx::argument
    compute(migraphx::context ctx, migraphx::shape, migraphx::arguments inputs) const override{
        auto* input_buffer  = reinterpret_cast<uint8_t*>(inputs[0].data());
        auto* output_buffer = reinterpret_cast<float*>(inputs[1].data());

        std::vector<std::size_t> in_shape = inputs[0].get_shape().lengths();
        assert(in_shape.size() == 4);

        std::vector<std::size_t> out_shape = inputs[1].get_shape().lengths();
        assert(out_shape.size() == 4);

        MIGRAPHX_HIP_ASSERT(hipSetDevice(0));
        mSize input_size{in_shape[2],in_shape[3]};
        mSize output_size{out_shape[2],out_shape[3]};
        size_t channel = in_shape[1];
        float ratio = (float)256 / std::min(input_size.width, input_size.height);
        float* resize_buffer = nullptr;

        size_t resize_bytes = 3 * 256 * 512 * sizeof(float);
        hipMalloc(&resize_buffer, resize_bytes);

        hipStream_t stream = ctx.get_queue<hipStream_t>();
        hipMemsetAsync(resize_buffer, 0, resize_bytes, stream);
        hipMemsetAsync(output_buffer, 0, inputs[1].get_shape().bytes(), stream);

        // 现在能找到了
        resnet_resize(stream,
                output_buffer,
                input_buffer,
                resize_buffer,
                ratio,
                input_size,
                output_size,
                channel,
                INTER_LINEAR);

        hipFree(resize_buffer);
        return inputs[1];
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override{
        if(inputs.size() != 2)
        {
            throw std::runtime_error("invalid input number!");
        }
        return inputs.back();
    }
};

#endif
