// MIGraphX 自定义算子 — 高层调用 API（op_add_api.hpp）
// 来源：内部文档《MIGRAPHX 自定义预处理算子（C++）实现流程深度解析》§6.3
#ifndef MIGRAPHX_CUSTOM_OP_ADD_API_HPP
#define MIGRAPHX_CUSTOM_OP_ADD_API_HPP
#include "migraphx/migraphx.hpp"
#include "op_add.hpp"
/*custom op add interface*/
namespace migraphx {
    // 带 shape 的 API，主要针对某些新增算子可能和原始输入或者输出 shape 不同，需要修改 shape 的场景，例如 update_shape 算子或者后处理算子
    void add_op(module& obj,custom_op_type op_typ,std::string name,shape s)
    {
        call(&migraphx_experimental_custom_op_add, obj.get_handle_ptr(),op_typ,name.c_str(),(void*)(s.get_handle_ptr()));
    }

    void add_op(module& obj,custom_op_type op_typ,std::string name)
    {
        call(&migraphx_experimental_custom_op_add, obj.get_handle_ptr(),op_typ,name.c_str(),nullptr);
    }

    template<class... Ts>
    void add_experimental_custom_op(module& obj,custom_op_type op_typ,std::string name,Ts&&... args)
    {
        assert(name.length() != 0);
        add_op(obj,op_typ,name,std::forward<Ts>(args)...);
    }
}
#endif
