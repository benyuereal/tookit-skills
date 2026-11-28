# 带宽阶梯（bw-ladder）—— 由 scripts/bw_calibrate.py 生成，勿手改。
#
# 跳层就是这个文件存在的原因：1724 被直接当「本卡地板」断言，造成五次分母错。
# 判「还能优化多少」必须走 ②→③→④（由 scripts/floor.py 算），不许跳。
#
# confidence: high=多 kernel 互证 / low=单样本 / none=未标定
{{body}}