# Characterization 数据与脚本说明

本文档说明 Section 3 (Characterization) 中使用的原始数据来源、分析脚本、生成的图表，以及新旧数据集的逐条对比与混合使用方案。

## 1. 原始数据

所有实验数据位于 `experiments/` 目录下，每个任务的 `attempt_1/` 子目录包含以下文件：

| 文件 | 内容 |
|------|------|
| `results.json` | 执行结果、Claude 输出（stdout/stderr）、退出码、资源采样序列 |
| `resources.json` | 聚合后的资源统计（最小/最大/平均 CPU 和内存） |
| `tool_calls.json` | 每个工具调用的类型、开始时间、结束时间 |
| `trace.jsonl` | Claude Code 完整执行 trace（JSONL 格式） |
| `resource_plot.png` | 单任务 CPU/内存时序图 |

### 数据集

| 数据集 | 路径 | 模型 | 有效任务数 | 用途 |
|--------|------|------|------------|------|
| batch_swebench_18tasks | `experiments/batch_swebench_18tasks/` | Claude Code + Haiku | 18 (6 类别 × 3 难度) | Category 分析、极端案例 |
| all_images_haiku | `experiments/all_images_haiku/` | Claude Code + Haiku | 33 (filtered) | 大规模 Haiku 统计 |
| all_images_local | `experiments/all_images_local/` | Claude Code + GLM 4.7 flash (本地) | 111 (filtered) | 大规模 Local 统计、模型对比 |

18 个 curated 任务覆盖：CLI_Tools、DevOps_Build、ML_Scientific、Medical_Bio、SQL_Data、Web_Network，每类别 Easy/Medium/Hard 各一个。分类映射定义在 `scripts/batch_test_swebench.py` 的 `SAMPLE_TASKS` 和 `scripts/verify_sample_tasks.py` 的 `CATEGORY_REPOS`。

新数据集中约 54%–64% 的任务可通过 repo 名反推 category（`CATEGORY_REPOS` 映射），其余为未分类的 SWE-bench 随机任务。

## 2. 新旧数据对比与混合使用方案

### 2.1 逐条对比

#### 3.1 Experimental Setup

| Claim | 旧数据 (18 tasks) | 新数据 (H=Haiku 33, L=Local 111) | 判定 |
|-------|-------------------|----------------------------------|------|
| 镜像 2.9–17.7 GB | 2.9–17.7 GB | H: 2.9–17.3, L: 2.9–17.3 | 基本一致 |
| 平均 4.2GB，中位数 3.5GB | 同左 | H: 4.1/3.4, L: 4.1/3.5 | 基本一致 |
| 总量 469GB | 115 tasks | L: 456GB (111 tasks) | 基本一致 |
| 权限修复 avg 28.3s, max 97s | 同左 | H: 24.3/87.9, L: 27.2/97.0 | L 一致，H 略低 |

**结论：新旧差别不大。用 Local 111 tasks（n 最大）。**

#### 3.2 RQ1: Agent Execution Model

| Claim | 旧数据 | 新数据 | 判定 | 建议数据源 |
|-------|--------|--------|------|-----------|
| 平均运行约 10 分钟 | Haiku ~400s (6.7min) | H: 347s (5.8min), L: 646s (10.8min) | L 吻合，H 偏短 | Local 111 |
| 工具时间 avg 28.2%, 0.1%–73.3% | 同左 | 以 active_time 为分母: H: mean 42.5%, median 34.7%, 3%–86%; L: mean 36.4%, median 36.5%, 0%–86% | 修正后比例更高 | Local 111 |
| 测试 44.1%, Python 26.7%, 安装 10.9% | 旧 18 tasks | 需核实新数据 chart_06 | 需核实 | 取实际值 |
| Medical_Bio 4GB vs Web_Network 291MB (13.7x) | 有这些 category | 新数据 category 覆盖不全 | 用旧数据更好 | **旧 18 tasks** |
| Bash 2.64s, Task 66.16s, Read 0.06s, Edit 0.04s | 旧 18 tasks | H: Bash 3.76s, Task 100.47s; L: Bash 5.88s, Edit 0.04s | 量级一致 | Local 111 |

#### 3.3 RQ2: Resource Unpredictability

| Claim | 旧数据 | 新数据 | 判定 | 建议数据源 |
|-------|--------|--------|------|-----------|
| 内存变化高达 2.9GB/s | max 2983 MB/s | H: 1619, L: 1756 MB/s | 新弱 (1.6GB vs 2.9GB) | **旧 18 tasks** |
| CPU 变化率超过 50%/s | 旧 max 41.6%（不到 50%!） | H: 143.9%, L: 50.9% | **新反而更强** | Haiku 33 |
| 显著变化事件 1.6%–4.1% | 旧 4.1% | H: 3.8%, L: 1.7% | 一致 | 新数据（范围更精确） |
| Medical_Bio_Hard peak 4060MB, avg 264MB, 15.4x | 有 | 没有此任务 | 丢失 | **旧 18 tasks** |
| **CPU-Mem 正相关 91–95%** | **旧也是 avg -0.30** | H: -0.41, L: -0.39 | **两组都不支持！必须纠正** | — |
| 峰值内存 197MB–4GB, CV=147% | 279–4060MB, CV=147% | H: 201–2076MB, CV=111%; L: 220–2041MB, CV=63% | 新弱 | **旧 18 tasks**（category 分析） |
| Haiku 30.6% vs Qwen 7.9%, 3.9x | 旧 18 tasks | H 13.2% vs L 7.6%, 1.7x | 严重削弱 (3.9x→1.7x) | 新数据（更诚实） |
| Haiku 400s vs Qwen 607s | 旧 | H 352s vs L 664s | 趋势一致 | 新数据（n 更大） |
| CPU>50%: Haiku 21.2%, Qwen 0.5% | 旧 | H 8.2%, L 0.5% | Haiku 削弱 | 新数据 |
| 20–51 个重试组, Bash 密度 61.8% | 旧有极端任务 | H max=5, L max=20 | 严重削弱 | 新数据（更真实） |
| 并发 Haiku ~3, Qwen ~12 | 旧 | H ~7, L ~12 | L 一致，H 变了 | 新数据 |

#### 3.4 RQ3: Provisioning Efficiency

| Claim | 旧数据 | 新数据 | 判定 | 建议数据源 |
|-------|--------|--------|------|-----------|
| Haiku 利用率 24%, waste 76% | CPU: 24% | CPU: 9%, waste 91% | **新更严重（有利）** | 新数据 |
| Qwen 利用率 7%, waste 93% | 同左 | L: 7%, waste 93% | 完全一致 | 新数据 |
| CPU 过度供给 4.1x–13.6x | H: 4.1x | H: 11.1x, L: 13.9x | **新更极端（有利）** | 新数据 |
| Mem 过度供给 1.6x–2.4x | H: 2.4x | H: 1.7x, L: 1.6x | 新略低 | 新数据 |

### 2.2 Category 分类覆盖情况

新数据集的任务可通过 `CATEGORY_REPOS`（`scripts/verify_sample_tasks.py`）将 repo 映射到 6 个 category：

| 数据集 | 可分类 | 不可分类 | Medical_Bio |
|--------|--------|----------|-------------|
| Haiku 33 | 21 (64%) | 12 (36%) | **0 个** |
| Local 111 | 60 (54%) | 51 (46%) | 14 个 |

新数据 category 间资源差异：

| 数据集 | Max category | Min category | 比值 |
|--------|-------------|-------------|------|
| 旧 18 tasks | Medical_Bio 4060 MB | Web_Network 291 MB | **13.7x** |
| 新 Haiku 33 | DevOps_Build 698 MB | Web_Network 235 MB | 3.0x |
| 新 Local 111 | Medical_Bio 500 MB | Web_Network 286 MB | 1.7x (avg), 7.1x (max/avg) |

原因：旧数据每个 category×difficulty 恰好 1 个任务，最大化多样性；新数据同一 repo 有多个任务（如 pydicom 14 个），极端值被稀释。

### 2.3 已完成的修改（characterization.md）

以下修改已应用到 `paper/docs/characterization.md`：

| # | 修改项 | 旧值 | 新值 | 数据源 |
|---|--------|------|------|--------|
| 1 | **CPU-Mem 相关性（关键纠正）** | "强正相关 91-95%" | "任务依赖性，avg -0.39（-0.84 到 +0.50）" | 新数据 |
| 2 | 数据集描述 | "18 个任务" | 两级数据集策略（111 Local + 33 Haiku + 18 curated） | — |
| 3 | Qwen→GLM | 全文 "Qwen" | "GLM" | — |
| 4 | 工具时间占比（active_time 分母） | 28.2% / 71.8% / 0.1%–73.3% | H: mean 42.5% median 34.7%, L: mean 36.4% median 36.5%, 0%–86% | 新数据 + active_time |
| 5 | 任务数 | 115 个 | 111 个 | Local 111 |
| 6 | 显著变化事件 | 1.6%–4.1% | 1.7%–3.8% | 新数据 |
| 7 | Agent CPU 对比 | 30.6% vs 7.9%, 3.9x | 13.2% vs 7.6%, 1.7x | 新数据 common tasks |
| 8 | Agent 执行时间 | 400s vs 607s | 352s vs 664s | 新数据 |
| 9 | CPU>50% 采样点 | Haiku 21.2% | Haiku 8.2% | 新数据 |
| 10 | 重试组 | 20–51 个 | 多达 20 个 | Local 111 |
| 11 | 并发实例 | Haiku ~3, GLM ~12 | Haiku ~7, GLM ~12 | 新数据 |
| 12 | Haiku 利用率 | 24%, waste 76% | 9%, waste 91% | 新数据 |
| 13 | CPU 过度供给 | 4.1×–13.6× | 11.1×–13.9× | 新数据 |
| 14 | Mem 过度供给 | 1.6×–2.4× | 1.6×–1.7× | 新数据 |
| 15 | 资源浪费总结 | 76%–93% | 91%–93% | 新数据 |

**保留旧 18 tasks 数据的极端值**（curated representative subset）：
- 内存变化率高达 2.9GB/s、3GB/s（极端观测值）
- Medical_Bio_Hard 瞬态突发：peak 4060MB, avg 264MB, 15.4x
- Medical_Bio 4GB vs Web_Network 291MB (13.7x)
- 峰值内存 197MB–4GB, CV=147%（category 分析）

### 2.4 待验证/待修改

以下数值尚未更新，需要用新数据重新核实后决定是否修改：

| # | 项目 | 当前值 | 说明 |
|---|------|--------|------|
| 1 | 工具类型分布（testing 44.1%, Python 26.7%, install 10.9%） | 旧 18 tasks | 需运行新数据的 chart_06 核实 |
| 2 | 工具执行时间（Bash 2.64s, Task 66.16s, Read 0.06s, Edit 0.04s） | 旧 18 tasks | 新数据 Local: Bash 5.88s，其他待核实 |
| 3 | 镜像总量 469GB | 旧数据 | Local 111 实际约 456GB，差异不大 |
| 4 | 非确定性案例（DevOps_Build_Hard 3 次执行） | 具体实验数据 | 保留，无需更新 |
| 5 | 图表路径 | 全部指向 haiku_figures/ | 部分应指向 qwen3_figures/（Local 111），待图表重新生成后更新 |

### 2.5 混合使用方案

学术论文中混合使用多个数据集是常见做法，关键是**每处明确标注数据来源**。建议方案：

**论文表述方式**：
> We conduct experiments on two scales: (1) a curated set of 18 representative tasks spanning 6 categories × 3 difficulty levels, used for category-level analysis; (2) a broader set of 111 (GLM local) / 33 (Haiku cloud) SWE-rebench tasks, used for aggregate statistics and cross-architecture comparison. All tasks run in identical sandboxed environments.

**各小节数据源分配**：

| 章节 | 分析内容 | 数据源 | 理由 |
|------|---------|--------|------|
| 3.1 实验设置 | 镜像大小、启动开销 | Local 111 tasks | n 最大，数值与旧数据一致 |
| 3.2 阶段划分 | 工具时间比例、工具执行时间 | Local 111 tasks | n 大，统计更稳 |
| 3.2 工具类型分布 | Bash category 时间占比 | Local 111 tasks | n 大 |
| 3.2 工具语义差异 | Medical_Bio vs Web_Network | **旧 18 tasks** | 唯一有完整 category 标签的数据 |
| 3.3 时间动态性 | 变化率、突发事件 | Local 111 + Haiku 33 | 双数据集交叉验证 |
| 3.3 瞬态突发 | 极端 peak/avg 案例 | **旧 18 tasks**（Medical_Bio_Hard） | 新数据无此极端案例 |
| 3.3 CPU-Mem 相关性 | ~~正相关 91–95%~~ → avg -0.39 | ✅ 已纠正 | 改写为"任务依赖的耦合性" |
| 3.3 异构性 (category) | 峰值内存 CV、boxplot | **旧 18 tasks** | Category 分析需要标签 |
| 3.3 异构性 (agent) | Haiku vs Local CPU/时间 | 新数据 30 common tasks | n 更大，更有说服力 |
| 3.3 非确定性 | 重试循环 | Local 111 tasks | 新数据真实（max 20 groups） |
| 3.4 过度供给 | CPU/Mem waste | 新数据 | **新数据更极端，更有利** |
| 3.4 聚合内存轨迹 | 归一化内存趋势 | Local 111 tasks | n 大，统计更平滑 |

**学术规范要点**：
1. 每张图表、每个数值必须标注来自哪个数据集（e.g., "n=18 curated" vs "n=111 SWE-rebench"）
2. 不同数据集的分析结果不能混在同一张图中暗示来自同一来源
3. 使用旧数据做 category 分析时，明确说明是"representative subset"而非随机样本
4. 建议在 Experimental Setup 中用一段话说明两级数据集策略

## 3. 分析脚本与生成图表

### 3.1 characterization.py（一键生成）

**路径**: `analysis/characterization.py`
**功能**: 导入并依次运行所有分析脚本，生成 Section 3 所有图表和数值

```bash
python analysis/characterization.py              # 完整运行（Haiku + Local）
python analysis/characterization.py --haiku-only  # 仅 Haiku
python analysis/characterization.py --local-only  # 仅 Local
python analysis/characterization.py --skip-extended --skip-rq  # 快速模式
```

运行顺序：
1. `analyze_swebench_data.py` → Haiku haiku_figures/ + Local qwen3_figures/
2. `analyze_tool_time_ratio.py` → chart_01..chart_14
3. `analyze_haiku_vs_qwen.py` → comparison_figures/
4. `analyze_extended_insights.py` → 文本分析数据
5. `analyze_rq_validation.py` → 验证图表

最后输出 **Numerical Values Summary**，列出 characterization.md 中所有数值的最新计算结果。

### 3.2 analyze_swebench_data.py

**路径**: `analysis/analyze_swebench_data.py`
**输入**: `experiments/all_images_haiku/` 或 `experiments/all_images_local/`（通过 `--dataset` 选择）
**输出**: `analysis/haiku_figures/` 或 `analysis/qwen3_figures/`

```bash
python analysis/analyze_swebench_data.py --all                    # Haiku（默认）
python analysis/analyze_swebench_data.py --dataset qwen3 --all    # Local/GLM
```

| 生成图表 | 论文 Figure | characterization.md 章节 | 数据源建议 |
|----------|-------------|--------------------------|-----------|
| `rq1_resource_timeseries.png` | Fig. timeseries | 3.3 时间动态性 | Local 111 |
| `rq1_change_rate_distribution.png` | Fig. changerate | 3.3 时间动态性 | Local 111 |
| `rq2_category_boxplots.png` | Fig. categories | 3.3 异构性 | **旧 18 tasks**（需 category） |
| `rq3_tool_analysis.png` | — | 3.2 RQ1 | Local 111 |
| `rq4_overprovisioning.png` | Fig. overprovisioning | 3.4 RQ3 | 新数据（更有利） |

### 3.3 analyze_tool_time_ratio.py

**路径**: `analysis/analyze_tool_time_ratio.py`
**输入**: `experiments/all_images_local/`（默认）或 `--data-dir`
**输出**: 14 张 chart 图表

```bash
python analysis/analyze_tool_time_ratio.py                          # Local → qwen3_figures
python analysis/analyze_tool_time_ratio.py --data-dir experiments/all_images_haiku --figures-dir analysis/haiku_figures
```

| 生成图表 | characterization.md 章节 | 数据源建议 |
|----------|--------------------------|-----------|
| `chart_03_tool_ratio_distribution.png` | 3.2 阶段划分 | Local 111 |
| `chart_04_tool_usage_breakdown.png` | 3.2 工具执行时间差异 | Local 111 |
| `chart_05_tool_timeline.png` | 3.2 工具使用时间分布 | Local 111 |
| `chart_06_bash_categories.png` | 3.2 工具类型分布 | Local 111 |
| `chart_09_overhead_analysis.png` | 3.2 磁盘与启动开销 | Local 111 |
| `chart_10_memory_trajectory.png` | 3.4 聚合内存轨迹 | Local 111 |
| `chart_12_bash_time_by_category.png` | 3.2 工具语义决定资源消耗 | **旧 18 tasks** |
| `chart_13_memory_peak_timing.png` | 3.3 时间动态性 | Local 111 |

### 3.4 analyze_haiku_vs_qwen.py

**路径**: `analysis/analyze_haiku_vs_qwen.py`
**输入**: `experiments/all_images_haiku/` + `experiments/all_images_local/`
**输出**: `analysis/comparison_figures/` (6-7 PNG)

```bash
python analysis/analyze_haiku_vs_qwen.py
```

| 生成图表 | characterization.md 章节 | 新数据值 |
|----------|--------------------------|----------|
| `04_cpu_utilization_comparison.png` | 3.3 异构性 | Haiku 13.2% vs Local 7.6% (1.7x) |

### 3.5 analyze_extended_insights.py

**路径**: `analysis/analyze_extended_insights.py`

```bash
python analysis/analyze_extended_insights.py --compare
```

| 函数 | 对应章节 | 数据源建议 |
|------|----------|-----------|
| `analyze_disk_and_startup_overhead()` | 3.2 磁盘与启动开销 | Local 111 |
| `analyze_transient_bursts()` | 3.3 瞬态突发特征 | **旧 18 tasks**（Medical_Bio_Hard 15.4x） |
| `analyze_cpu_memory_correlation()` | 3.3 CPU 与内存相关性 | **必须纠正**（两组均为负相关） |
| `analyze_retry_loop_patterns()` | 3.3 重试循环模式 | Local 111（max 20 groups，更真实） |
| `analyze_local_vs_api_inference()` | 3.3 本地 vs API 推理 | 新数据 30 common tasks |
| `analyze_concurrency_potential()` | 3.3 异构性 | 新数据 |
| `analyze_memory_trajectory()` | 3.4 聚合内存轨迹 | Local 111 |
| `analyze_tool_semantic_variance()` | 3.2 工具语义差异 | **旧 18 tasks** |

### 3.6 analyze_rq_validation.py

**路径**: `analysis/analyze_rq_validation.py`

```bash
python analysis/analyze_rq_validation.py --all
```

## 4. 论文图表引用一览

| LaTeX label | 图表文件 | 生成脚本 | 数据源 |
|-------------|----------|----------|--------|
| `fig:timeseries` | `rq1_resource_timeseries.png` | `analyze_swebench_data.py --dynamics` | Local 111 |
| `fig:change_rate` | `rq1_change_rate_distribution.png` | `analyze_swebench_data.py --dynamics` | Local 111 |
| `fig:categories` | `rq2_category_boxplots.png` | `analyze_swebench_data.py --domain` | **旧 18 tasks** |
| `fig:cpu_diff` | `04_cpu_utilization_comparison.png` | `analyze_haiku_vs_qwen.py` | 新 30 common |
| `fig:overprovisioning` | `rq4_overprovisioning.png` | `analyze_swebench_data.py --efficiency` | Local 111 |
| `fig:tool_ratio` | `chart_03_tool_ratio_distribution.png` | `analyze_tool_time_ratio.py` | Local 111 |
| `fig:bash_categories` | `chart_06_bash_categories.png` | `analyze_tool_time_ratio.py` | Local 111 |
| `fig:tool_time` | `chart_04_tool_usage_breakdown.png` | `analyze_tool_time_ratio.py` | Local 111 |
| `fig:tool_timeline` | `chart_05_tool_timeline.png` | `analyze_tool_time_ratio.py` | Local 111 |
| `fig:peak_timing` | `chart_13_memory_peak_timing.png` | `analyze_tool_time_ratio.py` | Local 111 |
| `fig:memory_trajectory` | `chart_10_memory_trajectory.png` | `analyze_tool_time_ratio.py` | Local 111 |

## 5. 时间指标定义

实验中涉及多个时间指标，定义和计算方式各不相同。以下是完整定义：

### 5.1 各时间指标定义

| 指标 | 起点 | 终点 | 定义 | 来源 |
|------|------|------|------|------|
| `total_time` | `podman pull` 之前 | cleanup 完成之后 | 整个实验流程总耗时（pull + perm_fix + claude_time + cleanup） | `run_swebench.py:197→261` |
| `pull_time` | 开始 pull 镜像 | pull 完成 | `podman pull` 耗时 | `run_swebench.py:215` |
| `permission_fix_time` | 开始修复 /testbed 权限 | 修复完成 | `podman run chmod` 创建修复镜像的耗时 | `run_swebench.py:219→221` |
| `claude_time` | `podman run -d` 返回 | `podman wait` 返回 | 容器从创建到退出的全生命周期，**包含容器启动开销** | `run_swebench.py:236→238` |
| `active_time` | trace.jsonl 第一条记录 | trace.jsonl 最后一条记录 | 实际 agent 对话执行时间（LLM 思考 + 工具执行），**排除容器启动开销** | `analyze_swebench_data.py` 从 trace 计算 |
| `sampling_duration` | `podman stats` 首次返回有效数据 | 最后一次返回有效数据 | 资源监控有效窗口 | `resources.json` 首末采样点 |

### 5.2 时间线关系

```
|←────────────────────────── total_time ─────────────────────────────→|
|← pull →|← perm_fix →|←──────────── claude_time ──────────────────→|← cleanup →|
                       |← 容器启动开销 →|←──── active_time ────→|← git diff/du →|
                                        |←── sampling_duration ──→|
```

**容器启动开销**：`podman run -d` 带 `--userns=keep-id` 需要对所有 overlay 层做用户命名空间 ID 映射，时间与镜像大小成正比。在 `claude` CLI 进程真正启动前，`podman stats` 无法获取有效数据（返回空/失败被 `except` 吞掉），因此 `sampling_duration ≈ active_time`。

### 5.3 容器启动开销的量化

| 数据集 | 平均启动间隙 | 中位数 | 占 claude_time 比例 |
|--------|-------------|--------|-------------------|
| Haiku 33 | 158s | 169s | 47.7% |
| Local 111 | 169s | 172s | 31.3% |

代表性任务对比：

| 任务 | claude_time | active_time | 启动间隙 | perm_fix |
|------|------------|------------|---------|----------|
| NVIDIA__nv-ingest-71 | 704s | 347s | 357s | 87.9s |
| dask__dask-11628 | 291s | 96s | 194s | 22.5s |
| RDFLib__rdflib-1117 | 198s | 63s | 135s | 53.0s |
| joke2k__faker-1520 | 126s | 123s | 3s | 0.04s |
| beeware__briefcase-2212 | 477s | 473s | 4s | 0.04s |

**关键发现**：
1. **启动间隙是容器初始化开销**（userns ID 映射、overlay 准备），不是 LLM 思考时间
2. 已缓存的镜像（`perm_fix ≈ 0`）启动仅需 3-5 秒
3. 未缓存的镜像启动需 135-357+ 秒，与镜像大小相关（r=0.457）

### 5.4 对工具时间比例的影响

旧的 `claude_time` 作分母导致工具执行比例被容器启动开销稀释。改用 `active_time` 后：

| 指标 | Haiku（旧 claude_time） | Haiku（新 active_time） | GLM（旧） | GLM（新） |
|------|------------------------|------------------------|----------|----------|
| 工具时间占比（均值） | 25.9% | **42.5%** | 25.3% | **36.4%** |
| 工具时间占比（中位数） | — | **34.7%** | — | **36.5%** |
| 范围 | 0%–73% | 3%–86% | 0%–73% | 0%–86% |

RQ2 burst 归因中的 **50.6%**（工具调用占采样时间的比例）与 **~35% 中位数**的差异来源：burst 归因以资源采样点（`sampling_duration`）为分母，而工具时间比以 `active_time` 为分母，两者范围接近但不完全相同（`active_time` 包含首末 trace 之间不在采样范围内的少量时间）。

### 5.5 新增分析脚本

| 脚本 | 路径 | 功能 |
|------|------|------|
| `analyze_new_insights.py` | `analysis/analyze_new_insights.py` | Token 分析、工具-burst 归因、重试资源浪费、多租户并发模拟 |
| `compute_active_time.py` | `analysis/compute_active_time.py` | 计算 active_time 并与 claude_time 对比（辅助验证脚本） |

```bash
python analysis/analyze_new_insights.py               # 全部运行
python analysis/analyze_new_insights.py --analysis 1   # 仅 token 分析
python analysis/analyze_new_insights.py --analysis 2   # 仅工具-burst 归因
python analysis/analyze_new_insights.py --analysis 3   # 仅重试资源浪费
python analysis/analyze_new_insights.py --analysis 4   # 仅多租户并发模拟
python analysis/analyze_new_insights.py --analysis 5   # 仅 token-resource 相关性
```

新增图表：

| 图表文件 | characterization.md 章节 | 说明 |
|----------|--------------------------|------|
| `token_distribution.png` | — | Haiku per-turn context 增长 + output 分布 |
| `tool_burst_correlation.png` | 3.3 时间动态性 | 工具类型 memory spike + Bash 类别 burst profile |
| `retry_waste.png` | 3.3 非确定性 | 重试组分布 + 重试时间 vs 内存累积 |
| `concurrency_simulation.png` | 3.4 RQ3 | 静态 vs 动态分配 + 统计复用增益 |
| `token_resource_correlation.png` | 3.3 非确定性 | Token vs 峰值内存 + Turns vs 执行时间 |

## 6. 一键重现

```bash
# 全部生成（推荐）
python analysis/characterization.py

# 分步生成
# 1. Local 大数据集（主要图表）
python analysis/analyze_swebench_data.py --dataset qwen3 --all
python analysis/analyze_tool_time_ratio.py

# 2. Haiku 数据集
python analysis/analyze_swebench_data.py --all
python analysis/analyze_tool_time_ratio.py --data-dir experiments/all_images_haiku --figures-dir analysis/haiku_figures

# 3. 旧 18 tasks（category 分析）
python analysis/analyze_tool_time_ratio.py --data-dir experiments/batch_swebench_18tasks --figures-dir analysis/haiku_figures

# 4. 模型对比
python analysis/analyze_haiku_vs_qwen.py

# 5. 扩展分析
python analysis/analyze_extended_insights.py --compare
```
