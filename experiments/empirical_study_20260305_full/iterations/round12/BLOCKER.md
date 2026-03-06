# Blocker 记录（Round12 追加）

## 现象
在执行 `run_swebench.py` 对照实验时，`claude_output.stdout` 出现：

`You've hit your limit · resets 2am (UTC)`

导致本次新增 run 的 `tool_calls=0`，不可作为有效样本。

## 影响
- 无法在当前时段继续获得有效的新增对照样本。
- 继续运行会产生大量无效运行目录，污染统计。

## 处理
1. 已将该类 run 标记为无效，不并入主统计。
2. 在配额恢复后优先补跑：
   - 无 tracing 基线（starlette/diffcover）
   - 100ms vs 1000ms tracing 对照
3. 论文正文保留当前结论，并在“有效性威胁”中注明该限制。
