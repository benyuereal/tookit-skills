#!/bin/bash
set -e

# 1. 编译 op_test
cd torch_optest
mkdir -p build && cd build
cmake .. && make -j$(nproc)

# 2. 运行全量测试
echo "===== DCU Operator Compatibility Test ====="
echo "Date: $(date)"
echo "Host: $(hostname)"
./op_test 2>&1 | tee op_test_result.log

# 3. 提取结果摘要
echo ""
echo "===== Test Summary ====="
grep -E "PASS" op_test_result.log | wc -l | xargs echo "PASS:"
grep -E "FAIL" op_test_result.log | wc -l | xargs echo "FAIL:"
echo "================================"
