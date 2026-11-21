// MIGraphX 自定义算子 — 注册接口（custom_op_register.hpp）
// 来源：内部文档《MIGRAPHX 自定义预处理算子（C++）实现流程深度解析》§6.4
#ifndef MIGRAPHX_CUSTOM_OP_REGISTER_H
#define MIGRAPHX_CUSTOM_OP_REGISTER_H

#include <migraphx/migraphx.hpp>

template <class T, typename = std::enable_if_t<
    std::is_base_of_v<migraphx::experimental_custom_op_base, T> &&
    !std::is_same_v<migraphx::experimental_custom_op_base, T> &&
    std::is_copy_constructible_v<T> &&
    std::is_final_v<T>>>
struct opRegisterer {
    opRegisterer() {
        T op;
        migraphx::register_experimental_custom_op(op);
    }
};
#define REGISTER_CUSTOM_OP(name)                   \
namespace {                                        \
    const ::opRegisterer<name> _registerer_##name; \
}

#endif
