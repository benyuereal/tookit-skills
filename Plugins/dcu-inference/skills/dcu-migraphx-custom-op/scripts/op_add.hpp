// MIGraphX 自定义算子 — 通用添加 API 头文件（op_add.hpp）
// 来源：内部文档《MIGRAPHX 自定义预处理算子（C++）实现流程深度解析》§6.1
#ifndef MIGRAPHX_CUSTOM_OP_OP_ADD_API_HPP
#define MIGRAPHX_CUSTOM_OP_OP_ADD_API_HPP

    typedef enum
    {
        custom_op_type_preprocess = 0, // 模型预处理
        custom_op_type_postprocess  ,  // 模型后处理
        custom_op_type_process,        // 其他自定义算子(可能位于 network 中间的位置)
        custom_op_type_update_shape,   // 虚拟自定义算子，不会存在 network 中，功能是修改模型输入的 shape，配合预处理自定义算子可实现不同 size 图像的输入
    } custom_op_type;

    typedef struct migraphx_module* migraphx_module_t;
    typedef struct migraphx_shape* migraphx_shape_t;

    /**
     * @brief add op
     * @param mm  main model
     * @param op_typ custom op type
     * @param args custom op parameters
     * @return 0: successful   -1：false
     */
    int migraphx_experimental_custom_op_add(migraphx_module_t mm,custom_op_type& op_typ,const char* name,void* args =nullptr);

#endif
