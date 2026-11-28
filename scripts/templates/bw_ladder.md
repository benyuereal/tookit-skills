# 带宽阶梯（bw-ladder）

> 由 `scripts/bw_calibrate.py` 生成，勿手改。日期 {{date}}。
>
> **跳层就是这个文件的由来。** 1724 的直接断言造成了五次分母错。
> 判「还能优化多少」必须走 ②→③→④，不许跳。

| 层 | 可达 GB/s | 方法 | 置信 |
|---|---|---|---|
{{nominal_row}}
{{mode_rows}}
| ④ 本卡地板 | 由 floor.py 算 | 流量 ÷ ③ | — |

## 用法

```bash
python3 -I scripts/bw_calibrate.py                 # 重新标定
python3 -I scripts/floor.py --card <卡> --shape <档>  # 算本卡地板
```

**confidence 说明**：`high` = 三个独立 kernel 互相印证；
`low` = 单样本观测，不能当定律；`none` = 未标定。
