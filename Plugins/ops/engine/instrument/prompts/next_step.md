<!-- main -->
下一步：{{label}}（{{action}}）

为什么：{{why}}

照做（复制粘贴即可）：
    {{cmd}}{{self_checks}}{{problems}}{{expect}}{{produces}}{{on_failure}}{{also}}{{spec}}

{{status_line}}

<!-- label_run -->
跑实验

<!-- label_implement -->
实现变体

<!-- label_explore -->
出题

<!-- label_baseline -->
先测基线

<!-- label_stop -->
停止

<!-- self_checks -->
已自检（1 秒，未跑实验）——直接执行不会失败：
{{items}}

<!-- problems -->
预检不通过（先修这些）：
{{items}}

<!-- expect -->
预计耗时：{{value}}

<!-- produces -->
跑完之后你会得到：
{{items}}

<!-- on_failure -->
如果失败：
{{items}}

<!-- also -->
之后还有：{{items}}

<!-- spec -->
这条候选的说明：
{{items}}

<!-- check_item -->
  ✓ {{item}}

<!-- problem_item -->
  ✗ {{item}}

<!-- bullet_item -->
  · {{item}}

<!-- spec_item -->
  {{key}} {{value}}
