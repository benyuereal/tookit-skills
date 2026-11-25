#!/usr/bin/env bash
# fmla_build —— 打补丁 → 验证补丁在 → 强制重编 → 验证产物，确保编译真的带上改动。
#
# 为什么不用简单的 `python setup.py build_ext`：
#
# 2026-11-25 一天之内，改动「消失」了 4 次：
#   1. 改了 .h 没同步 _hip.h（hipify 产物）
#   2. 仓库被重新 clone 覆盖
#   3. .o 比源文件新，setup.py 跳过重编
#   4. 工作区文件被无声还原
#
# 结果是：我以为在验证 A，实际编译的是 B。每次白等一轮编译，
# 而且「修了还崩」这个观察本身是假的。
#
# 任一步失败都返回非 0，不会出现「编译成功但其实没改」。
#
# 用法：bash scripts/fmla_build.sh
set -u
REPO=${FMLA_REPO:-/workspace/FlashMLA-das}
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=${FMLA_LOG:-/tmp/fmla_build_$(date +%H%M%S).log}

die() { echo "✗ $*" >&2; exit 1; }

echo "=== ① 打补丁 ==="
python3 -I "$HERE/fmla_patch.py" --apply || die "打补丁失败"

echo
echo "=== ② 确认补丁在文件里 ==="
python3 -I "$HERE/fmla_patch.py" --check || die "补丁没打上（或被回滚）—— 拒绝编译，否则白等一轮"

echo
echo "=== ③ 删除相关目标文件 ==="
cd "$REPO" || die "进不去 $REPO"
for o in build/temp.*/csrc/api/api_hip.o \
         build/temp.*/csrc/extension/flash_fwd_mla_fp8_gfx938.o \
         build/temp.*/csrc/extension/flash_fwd_mla_bf16_gfx936.o; do
    [ -e "$o" ] && { rm -f "$o"; echo "  删除 $o"; }
done
rm -f flash_mla/cuda*.so
echo "  已清掉旧产物"

echo
echo "=== ④ 编译（日志 $LOG）==="
FLASH_MLA_OPT=1 python3 setup.py build_ext --inplace > "$LOG" 2>&1 \
    || { grep -E " error:" "$LOG" | head -5; die "编译失败，见 $LOG"; }
if grep -qE " error:" "$LOG"; then
    grep -E " error:" "$LOG" | head -5
    die "编译有错误"
fi

echo
echo "=== ⑤ 确认产物真的比源文件新 ==="
SO=$(ls flash_mla/cuda*.so 2>/dev/null | head -1)
[ -n "$SO" ] || die "没生成 so"

for pair in "build/temp.*/csrc/api/api_hip.o:csrc/api/dense_decode_hip.h" \
            "build/temp.*/csrc/extension/flash_fwd_mla_fp8_gfx938.o:csrc/extension/flash_fwd_mla_kernel_fp8.h"; do
    o=${pair%%:*}
    src=${pair##*:}
    ob=$(ls $o 2>/dev/null | head -1)
    if [ -z "$ob" ]; then
        echo "  （跳过：$o 不存在）"
        continue
    fi
    if [ "$ob" -nt "$src" ]; then
        echo "  ✓ $(basename "$ob") 比 $src 新"
    else
        die "$ob 不比 $src 新 —— 这次编译没带上改动"
    fi
done

echo
echo "✓ 编译完成且改动确认已进入产物：$SO"
echo "  下一步：HIP_VISIBLE_DEVICES=0 python3 -I /tmp/probe.py 32 64 32768"
