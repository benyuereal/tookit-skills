// migraphx-driver 新增命令：info —— 查看模型基础信息
// 来源：内部文档《migraphx-driver的源码解析和功能增加实践》§七
// 继承自 command<info>，复用 loader 模块加载模型，输出文本/JSON 两种格式
struct info : command<info>
{
    loader l;
    bool json_output = false;

    void parse(argument_parser& ap)
    {
        l.parse(ap);
        ap(json_output, {"--json"}, ap.help("Output model info in JSON format"), ap.set_value(true));
    }

    void run()
    {
        auto p = l.load();
        auto&& module = p.get_main_module();

        if(json_output)
        {
            value v;
            v["name"]             = l.file;
            v["batch"]            = l.batch;
            v["parameters"]       = p.get_parameter_shapes().size();
            v["optimized"]        = l.optimize;
            value params;
            for(auto&& [k, s] : p.get_parameter_shapes())
                params[k] = s.type_string();
            v["parameter_shapes"] = params;
            std::cout << to_pretty_json_string(v) << std::endl;
            return;
        }

        std::cout << "=============================================" << std::endl;
        std::cout << "           MIGraphX Model Info              " << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "File:         " << l.file << std::endl;
        std::cout << "Batch:        " << l.batch << std::endl;
        std::cout << "Optimized:    " << std::boolalpha << l.optimize << std::endl;
        std::cout << "Parameters:   " << p.get_parameter_shapes().size() << std::endl;
        std::cout << "---------------------------------------------" << std::endl;
        std::cout << "Parameters & Shapes:" << std::endl;
        for(auto&& [name, s] : p.get_parameter_shapes())
            std::cout << "  " << name << " : " << s << std::endl;
        std::cout << "=============================================" << std::endl;
        return;
    }
};
