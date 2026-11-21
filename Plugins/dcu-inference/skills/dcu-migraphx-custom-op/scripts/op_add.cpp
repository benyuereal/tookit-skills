// MIGraphX 自定义算子 — 算子添加逻辑实现（op_add.cpp）
// 来源：内部文档《MIGRAPHX 自定义预处理算子（C++）实现流程深度解析》§6.2
#include <iostream>
#include <assert.h>
#include <op_add.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/onnx.hpp>

namespace migraphx{

    struct customOperator
    {
    public:
        explicit customOperator(std::string name){
            assert(name != "");
            op_name = name;
        }

        ~customOperator(){};
        template <class... Ts>
        void add_op(module* mm ,custom_op_type& op_type,Ts&&... args){
            switch (op_type)
            {
            case custom_op_type_preprocess:
                add_custom_preprocess(mm);
                break;
            case custom_op_type_postprocess:
                add_custom_postprocess(mm,std::forward<Ts>(args)...);
                break;
            case custom_op_type_process:

                break;
            case custom_op_type_update_shape:
                add_custom_update_param_shape(mm,std::forward<Ts>(args)...);
                break;
            default:
                std::cerr<<"Unknown custom op type ["<<op_type<<"]"<<std::endl;
                break;
            }
        };

    private:
        template <class... Ts>
        migraphx::instruction_ref add_custom_update_param_shape(migraphx::module* mm,Ts&&...args){
            std::unordered_map<migraphx::instruction_ref, std::string> inputs;
            migraphx::instruction_ref input_ins;
            migraphx::instruction_ref param_ins;
            migraphx::shape u8_input_shape {std::forward<Ts>(args)...};

            for(auto ins : migraphx::iterator_for(*mm))
            {
                // Get network input instruction if insert preprocess op
                if(ins->name() == "@param")
                {
                    inputs.emplace(ins,ins->name());
                    input_ins = ins;
                    if(ins->get_shape() == u8_input_shape)
                    {
                        param_ins = ins;
                        break;
                    }
                    // 预处理算子的输入类型为 uint8，param 默认输入 float 类型，如果增加自定义算预处理算子，则需要强制修改 param 的输入类型为 uint8
                    if(ins->get_shape().type() != migraphx::shape::uint8_type)
                    {
                        std::string  param = any_cast<builtin::param>(input_ins->get_operator()).parameter;
                        migraphx::shape input_shape = ins->get_shape();
                        auto output_ins = ins->outputs();
                        mm->remove_instruction(ins);

                        migraphx::shape u8_input_shape = input_shape.with_type(migraphx::shape::uint8_type);
                        ins = mm->add_parameter(param, u8_input_shape);
                        for(auto out : output_ins)
                            ins->add_output(out);
                    }

                    for(auto o_ins : ins->outputs())
                    {
                        if(o_ins->get_shape().type() != migraphx::shape::uint8_type)
                        {
                            auto op = o_ins->get_operator();
                            auto shape = o_ins->get_shape().with_type(migraphx::shape::uint8_type);
                            auto in_args = o_ins->inputs();
                            migraphx::instruction::replace(o_ins, op, shape, in_args);
                        }

                        for(auto in_o_ins : o_ins->inputs())
                        {
                            if(in_o_ins->name().compare("load") == 0)
                            {
                                size_t offset = u8_input_shape.bytes();
                                mm->replace_instruction(in_o_ins, make_op("load", {{"shape", to_value(u8_input_shape)}, {"offset", 0}}), in_o_ins->inputs()[0]);
                            }
                        }
                    }
                }
                else if(not inputs.empty())
                {
                    // Replace input instruction that netx node.
                    for(auto in_ins : ins->inputs())
                    {
                        if(in_ins == input_ins and in_ins->name().compare(input_ins->name())==0)
                        {
                            std::string  input_name = any_cast<builtin::param>(input_ins->get_operator()).parameter;
                            mm->remove_instruction(input_ins);
                            param_ins = mm->add_parameter(input_name, u8_input_shape);
                            migraphx::instruction::replace_argument(ins, in_ins, param_ins);
                        }
                    }
                }
            }
            return param_ins;
        }

        migraphx::instruction_ref add_custom_preprocess(migraphx::module* mm){
            if (op_name == "")
            {
                std::cerr << "Please specify a valid op name" << std::endl;
                exit(1);
            }

            std::unordered_map<migraphx::instruction_ref, std::string> inputs;
            migraphx::instruction_ref input_ins;
            migraphx::instruction_ref add_op_ins;
            for(auto ins : migraphx::iterator_for(*mm))
            {
                // Get network input instruction if insert preprocess op
                if(ins->name().compare(op_name) == 0)
                {
                    std::cout << "Op ["<<op_name<<"] already existed!"<<std::endl;
                    continue;
                }
                if(ins->name() == "@param")
                {
                    inputs.emplace(ins,ins->name());
                    input_ins = ins;
                    auto param_name = any_cast<builtin::param>(ins->get_operator()).parameter;
                    std::cout<<"Parameter name: "<<param_name<<"\n";
                }
                else if(not inputs.empty())
                {
                    for(auto in_ins : ins->inputs())
                    {
                        if(in_ins == input_ins and in_ins->name().compare(input_ins->name())==0)
                        {
                            migraphx::operation op = migraphx::make_op(op_name);

                            migraphx::shape input_shape = in_ins->get_shape();
                            migraphx::shape alloc_shape = input_shape.with_type(migraphx::shape::float_type);

                            {
                                std::string  input_name = any_cast<builtin::param>(input_ins->get_operator()).parameter;
                                mm->remove_instruction(input_ins);
                                migraphx::shape u8_input_shape = input_shape.with_type(migraphx::shape::uint8_type);
                                auto u8_input_ins = mm->add_parameter(input_name, u8_input_shape);
                                auto alloc = mm->insert_instruction(ins,migraphx::make_op("allocate", {{"shape", migraphx::to_value(alloc_shape)}}),{});
                                add_op_ins = mm->insert_instruction(ins,op,{u8_input_ins,alloc});
                            }

                            migraphx::instruction::replace_argument(ins, in_ins, add_op_ins);
                            std::cout<<"Adding op ["<<op_name<<"] to ["<<ins->name()<<"]"<<std::endl;
                        }
                    }
                }
            }
            return add_op_ins;
        }

        std::vector<migraphx::instruction_ref> add_custom_postprocess(module* mm){
            if (op_name == "")
            {
                std::cerr << "Please specify a valid op name" << std::endl;
                exit(1);
            }

            std::unordered_map<migraphx::instruction_ref, std::string> inputs;
            std::vector<instruction_ref> ret_ins = mm->get_returns();
            std::vector<instruction_ref> add_op_ins;
            for(auto ins : migraphx::iterator_for(*mm))
            {
                if(ins->name().compare(op_name) == 0)
                {
                    std::cout << "Op ["<<op_name<<"] already existed!"<<std::endl;
                    continue;
                }
                auto it = std::find(ret_ins.begin(), ret_ins.end(), ins);
                if(it != ret_ins.end())
                {
                    std::cout<<"Adding op ["<<op_name<<"] to ["<<ins->name()<<"]"<<std::endl;
                    migraphx::operation op = migraphx::make_op(op_name);
                    auto op_ins = mm->add_instruction(op, {ins});
                    add_op_ins.push_back(op_ins);
                }

                if(ins->name() == "@return")
                {
                    // Remove old return
                    mm->remove_instruction(ins);
                    assert(add_op_ins.size() == ret_ins.size());

                    // Add new return
                    mm->add_return(add_op_ins);
                    break;
                }
            }
            return add_op_ins;
        }

        std::vector<migraphx::instruction_ref> add_custom_postprocess(module* mm,migraphx::shape& s){
            if (op_name == "")
            {
                std::cerr << "Please specify a valid op name" << std::endl;
                exit(1);
            }

            migraphx::shape new_output_shape {s};
            std::unordered_map<migraphx::instruction_ref, std::string> inputs;
            std::vector<instruction_ref> ret_ins = mm->get_returns();
            std::vector<instruction_ref> add_op_ins;
            for(auto ins : migraphx::iterator_for(*mm))
            {
                if(ins->name().compare(op_name) == 0)
                {
                    std::cout << "Op ["<<op_name<<"] already existed!"<<std::endl;
                    continue;
                }
                auto it = std::find(ret_ins.begin(), ret_ins.end(), ins);
                if(it != ret_ins.end())
                {
                    std::cout<<"Adding op ["<<op_name<<"[ to ["<<ins->name()<<"]"<<std::endl;
                    migraphx::operation op = migraphx::make_op(op_name);
                    auto alloc = mm->insert_instruction(ins,migraphx::make_op("allocate", {{"shape", migraphx::to_value(new_output_shape)}}),{});
                    auto op_ins = mm->add_instruction(op, {ins,alloc});
                    add_op_ins.push_back(op_ins);
                }

                if(ins->name() == "@return")
                {
                    // Remove old return
                    mm->remove_instruction(ins);
                    assert(add_op_ins.size() == ret_ins.size());
                    // Add new return
                    mm->add_return(add_op_ins);
                    break;
                }
            }
            return add_op_ins;
        }

        std::string op_name; // custom op name
    };
}

struct migraphx_module
{
    template <class... Ts>
    migraphx_module(Ts&&... xs)
        : object(std::forward<Ts>(xs)...)
    {
    }
    migraphx::module object;
};

struct migraphx_shape
{
    template <class... Ts>
    migraphx_shape(Ts&&... xs)
        : object(std::forward<Ts>(xs)...)
    {
    }
    migraphx::shape object;
};

int migraphx_experimental_custom_op_add(migraphx_module_t mm,custom_op_type& op_typ,const char* name,void* args){
    std::unique_ptr<migraphx::customOperator> add_op_ptr = std::make_unique<migraphx::customOperator>(name);
    if(args != nullptr)
    {
        migraphx_shape_t param = (migraphx_shape_t)args;
        migraphx::shape shape = param->object;
        add_op_ptr->add_op(&(mm->object),op_typ,shape);
    }else{
        add_op_ptr->add_op(&(mm->object),op_typ);
    }
    // std::cout<<"Add custom op ["<<name<<"] successful!"<<std::endl;
    return 0;
}
