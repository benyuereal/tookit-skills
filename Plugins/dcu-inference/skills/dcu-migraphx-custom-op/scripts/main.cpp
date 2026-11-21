// MIGraphX 自定义算子 — 推理全流程示例（main.cpp）
// 来源：内部文档《MIGRAPHX 自定义预处理算子（C++）实现流程深度解析》§6.7
#include <cmath>
#include <opencv2/opencv.hpp>
#include "op_add_api.hpp"
#include "resnet50_preprocess.hpp"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

static float sigmoid(float x)
{
    return 1.0/(1.0 + exp(-x));
}

int main(int argc,char* argv[]){
    std::string onnx_path = "resnet50.onnx";
    migraphx::onnx_options onnx_options;
    // 1、onnx 解析
    migraphx::program net = migraphx::parse_onnx(onnx_path.c_str(),onnx_options);
    migraphx::module mm = net.get_main_module();

    // 2、添加预处理算子
    std::string preprocess_op_name = "resnet_preprocess";
    migraphx::add_experimental_custom_op(mm,custom_op_type_preprocess,preprocess_op_name);

    migraphx::compile_options options;
    options.set_offload_copy();
    net.compile(migraphx::target("gpu"), options);

    std::string path = "./Images/*.jpg";
    std::vector<cv::String> file_paths;
    cv::glob(path, file_paths, false);
    if (file_paths.empty()) {
        std::cerr << "No images found in: " << path << std::endl;
        return -1;
    }
    for(auto file : file_paths)
    {
        cv::Mat img = cv::imread(file);

#if 0
// 打印维度信息：OpenCV 的 Mat 维度是 (height, width, channels) → HWC
    std::cout << "图像高度：" << img.rows << std::endl;
    std::cout << "图像宽度：" << img.cols << std::endl;
    std::cout << "通道数：" << img.channels() << std::endl;
    std::cout << "维度顺序：H(" << img.rows << ") × W(" << img.cols << ") × C(" << img.channels() << ") → HWC格式" << std::endl;
#endif

        migraphx::shape s{migraphx_shape_uint8_type, {1, 3,size_t(img.rows),size_t(img.cols)}};
        migraphx::program_parameters input_param;
        std::cout<<"-------------------inference---------------------------"<<std::endl;
        fs::path path_(file);
        std::string filename = path_.filename().string();
        std::cout<<"category file name:"<<filename<<std::endl;

        input_param.add("input", migraphx::argument(s, img.data));
        // 3、更新输入 shape
        std::string update_shape_op_name = "update_shape";
        migraphx::add_experimental_custom_op(mm,custom_op_type_update_shape,update_shape_op_name,s);

        auto results = net.eval(input_param);
        auto result  = results[0];

        float *logits = (float *)result.data();
        float max_conf = 1e-6;
        float label = -1;
        for(int j = 0; j < 1000; ++j)
        {
            if(max_conf<sigmoid(logits[j]))
            {
                label      = j;
                max_conf = sigmoid(logits[j]);
            }
        }
        std::cout<<"label :"<<label<<", conf:"<<max_conf<<std::endl;
    }
    return 0;
}
