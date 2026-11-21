// migraphx-driver 新增命令：test-op —— 调试单个算子（按名查找模型中的算子实例）
// 来源：内部文档《migraphx-driver的源码解析和功能增加实践》§七
// 继承自 command<test_op>，复用 compiler 模块（需编译模型以获取算子输出维度）
// 支持 --op（目标算子名）与 --all（显示所有匹配，默认只显示第一个）
struct test_op : command<test_op>
{
    compiler c;
    std::string target_op;
    bool show_all = false;

    void parse(argument_parser& ap)
    {
        c.parse(ap);
        ap(target_op, {"--op"}, ap.help("Target operator name to test"), ap.metavar("OP_NAME"));
        ap(show_all, {"--all"}, ap.help("Show all matching ops"), ap.set_value(true));
    }

    void run()
    {
        auto p = c.l.load();
        auto t = c.ct.get_target();
        auto m = c.params(p);
        auto&& module = p.get_main_module();
        int idx = 0;
        bool found = false;

        std::cout << "[TEST-OP] Searching for: " << target_op << std::endl;

        for(auto&& ins : *module)
        {
            if(target_op.empty() || ins.name() == target_op)
            {
                found = true;
                std::cout << "=====================================================" << std::endl;
                std::cout << "[OP] " << ins.name() << " (index: " << idx << ")" << std::endl;
                std::cout << "[OUTPUT SHAPE] " << ins.get_shape() << std::endl;
                std::cout << "=====================================================" << std::endl;

                // 只运行一个，除非 --all
                if(!show_all)
                    break;
            }
            idx++;
        }

        if(!found)
            std::cout << "[TEST-OP] Operator not found: " << target_op << std::endl;
    }
};
