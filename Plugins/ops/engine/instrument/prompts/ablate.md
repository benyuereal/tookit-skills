# 消融任务{{card_suffix}} / 冠军 = `{{champion_tag}}`

你只负责**生成消融变体**。测量、归因、记账全部由引擎完成；不要在会话里自己跑 bench 或计时。

## 冠军实现用了这 {{method_count}} 项方法

{{methods_block}}

## 你的任务：逐个删掉一项

各生成一个变体（共 {{method_count}} 个），每个变体：

1. **只删这一项**，其它部分与冠军保持逐字节相同；
2. 在 `<卡>/_common.py` 里注册成 `<champion>-no-<method>`（`VARIANTS` / `HIP_VARIANTS`，
   并让 `get_runner` 返回可运行实现）；
3. 保证数值仍正确 —— 删错了就说明这不是可消融的方法，**如实标注**，不要改别的代码去凑。

消融 tag 必须是这 {{method_count}} 个（引擎会逐个测）：

{{tag_lines}}

## 停止条件

**不要测性能。** 删完就停，用一句话说每个变体删了什么。
