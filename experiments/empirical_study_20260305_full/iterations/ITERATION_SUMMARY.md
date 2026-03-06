# 12 轮迭代汇总

## 已执行轮次
- round01 ~ round12：全部完成

## 当前主稿
- ../PAPER_CN_FINAL.md

## 当前主要剩余缺口
1. 跨仓库每仓库仅 1 次运行，统计显著性不足
2. 缺少无 tracing / 低频 tracing 对照组
3. 缺少跨仓库多次重复后的置信区间与检验

## 下一阶段实验优先级
1. 每仓库补 3 次重复（pytorch 至少补 2 次）
2. starlette 与 diffcover 做 tracing 开销对照
3. 更新 build_report.py 增加 CI 与检验结果表
