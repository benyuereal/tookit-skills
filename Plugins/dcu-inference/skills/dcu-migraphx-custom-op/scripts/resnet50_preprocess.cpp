// MIGraphX 自定义算子 — ResNet50 预处理算子核心计算（resnet50_preprocess.cpp）
// 来源：内部文档《MIGRAPHX 自定义预处理算子（C++）实现流程深度解析》§6.6
#include "resnet50_preprocess.hpp"
#include "custom_op_register.hpp"  // <--- 修复：必须加这个！
#include <hip/hip_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

__global__ void hip_resize_kernel(float* output_data,
    const uint8_t* input_data,
    float ratio,
    const mSize in_size,
    const mSize out_size,
    size_t channel,
    int n,
    inter_type itype)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= out_size.width || dy >= out_size.height)
        return;

    int stride = out_size.width;
    int tid = dy * stride + dx;

    float3 std_vec = make_float3(1.0f / 58.395f, 1.0f / 57.12f, 1.0f / 57.375f);
    float3 mean_vec = make_float3(123.675f, 116.28f, 103.53f);

    float src_x = dx / ratio;
    float src_y = dy / ratio;

    int x0 = floorf(src_x);
    int x1 = min(x0 + 1, (int)in_size.width - 1);
    int y0 = floorf(src_y);
    int y1 = min(y0 + 1, (int)in_size.height - 1);

    float a = src_x - x0;
    float b = src_y - y0;

    int pos00 = channel * (y0 * in_size.width + x0);
    int pos01 = channel * (y0 * in_size.width + x1);
    int pos10 = channel * (y1 * in_size.width + x0);
    int pos11 = channel * (y1 * in_size.width + x1);

    float3 p00 = make_float3(input_data[pos00 + 2], input_data[pos00 + 1], input_data[pos00 + 0]);
    float3 p01 = make_float3(input_data[pos01 + 2], input_data[pos01 + 1], input_data[pos01 + 0]);
    float3 p10 = make_float3(input_data[pos10 + 2], input_data[pos10 + 1], input_data[pos10 + 0]);
    float3 p11 = make_float3(input_data[pos11 + 2], input_data[pos11 + 1], input_data[pos11 + 0]);

    float3 val;
    val.x = ((1 - a) * (1 - b) * p00.x + a * (1 - b) * p01.x + (1 - a) * b * p10.x + a * b * p11.x);
    val.y = ((1 - a) * (1 - b) * p00.y + a * (1 - b) * p01.y + (1 - a) * b * p10.y + a * b * p11.y);
    val.z = ((1 - a) * (1 - b) * p00.z + a * (1 - b) * p01.z + (1 - a) * b * p10.z + a * b * p11.z);

    val.x = (val.x - mean_vec.x) * std_vec.x;
    val.y = (val.y - mean_vec.y) * std_vec.y;
    val.z = (val.z - mean_vec.z) * std_vec.z;

    output_data[tid] = val.x;
    output_data[tid + n] = val.y;
    output_data[tid + 2 * n] = val.z;
}

__global__ void hip_crop_center_kernel(float* output_data,
    const float* input_data,
    int start_x, int start_y,
    mSize in_size, mSize out_size,
    int channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels = out_size.width * out_size.height;
    int c = idx / pixels;
    int pix = idx % pixels;

    if (c >= channels)
        return;

    int y = pix / out_size.width;
    int x = pix % out_size.width;

    int in_y = start_y + y;
    int in_x = start_x + x;

    if (in_y < 0 || in_y >= in_size.height || in_x < 0 || in_x >= in_size.width)
        return;

    int in_idx = c * in_size.height * in_size.width + in_y * in_size.width + in_x;
    int out_idx = c * pixels + pix;

    output_data[out_idx] = input_data[in_idx];
}

void resnet_resize(hipStream_t stream,
    float* output_data,
    const uint8_t* input_data,
    float* res_data,
    float ratio,
    const mSize input_size,
    const mSize output_size,
    size_t channel,
    inter_type itype)
{
    mSize res_size;

    if (input_size.height > input_size.width) {
        res_size.width = 256;
        res_size.height = (int)(input_size.height * ratio);
    } else {
        res_size.width = (int)(input_size.width * ratio);
        res_size.height = 256;
    }

    dim3 block(16, 16);
    dim3 grid((res_size.width + block.x - 1) / block.x,
              (res_size.height + block.y - 1) / block.y);

    hipLaunchKernelGGL(hip_resize_kernel, grid, block, 0, stream,
        res_data,
        input_data,
        ratio,
        input_size,
        res_size,
        channel,
        res_size.width * res_size.height,
        itype);

    int start_x = res_size.width / 2 - output_size.width / 2;
    int start_y = res_size.height / 2 - output_size.height / 2;

    int total = channel * output_size.width * output_size.height;
    dim3 crop_block(256);
    dim3 crop_grid((total + crop_block.x - 1) / crop_block.x);

    hipLaunchKernelGGL(hip_crop_center_kernel, crop_grid, crop_block, 0, stream,
        output_data,
        res_data,
        start_x, start_y,
        res_size,
        output_size,
        channel);

    return;  // <--- 修复：补上 return
}

// 现在这个宏可以正常识别了
REGISTER_CUSTOM_OP(resnet_preprocess_custom_op);
