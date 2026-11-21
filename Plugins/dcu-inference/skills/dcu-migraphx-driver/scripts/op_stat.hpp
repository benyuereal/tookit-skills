// migraphx-driver 新增命令：op_stat —— 统计算子数量（按调用次数降序）
// 来源：内部文档《migraphx-driver的源码解析和功能增加实践》§七
// 继承自 command<op_stat>，复用 loader 加载模型，遍历主模块指令统计算子名出现次数
struct op_stat : command<op_stat>
{
    loader l;

    void parse(argument_parser& ap)
    {
        l.parse(ap);
    }
void run()
{
    auto p = l.load();
    auto* mod = p.get_main_module(); // 获取主模块

    std::unordered_map<std::string, size_t> op_count;

    // 直接遍历模块指令，ins 是 instruction 对象（非指针）
    for (auto&& ins : *mod)
    {
        // 使用 . 而不是 ->
        std::string op_name = ins.name();
        op_count[op_name]++;
    }

    size_t total_instructions = 0;
    for (auto&& ins : *mod) total_instructions++; // 或者 mod->size() 如果存在

    std::cout << "=============================================" << std::endl;
    std::cout << "            Operator Statistics              " << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Total instructions: " << total_instructions << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    std::vector<std::pair<std::string, size_t>> sorted_ops(op_count.begin(), op_count.end());
    std::sort(sorted_ops.begin(), sorted_ops.end(),
              [](auto& a, auto& b) { return a.second > b.second; });

    for (auto&& [op, cnt] : sorted_ops)
        std::cout << cnt << "\t" << op << std::endl;

    std::cout << "=============================================" << std::endl;
}
};
