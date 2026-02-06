# 3 Characterization

我们对生产级 AI coding agent 的资源使用模式进行了首次系统性测量研究，旨在回答以下研究问题：

- **RQ1**: Agent 的执行模型是什么？
- **RQ2**: Agent 工作负载的资源使用有何特征？为什么难以预测？
- **RQ3**: 静态资源分配的效率如何？

## 3.1 Experimental Setup

**数据集。** 我们从 SWE-rebench 数据集中选取 18 个任务，覆盖六个类别（CLI_Tools、DevOps_Build、ML_Scientific、Medical_Bio、SQL_Data、Web_Network）和三个难度级别（Easy、Medium、Hard）。这些任务涵盖了 AI coding agent 的典型使用场景，包括命令行工具修复、构建系统配置、机器学习代码调试、生物医学数据处理、数据库查询优化和 Web 服务修复。

**Agent 实现。** 我们使用两个不同的 agent 实现执行相同的 18 个任务：Claude Code with Haiku（Anthropic 的生产级 AI coding agent）和基于 Qwen 模型的本地 agent 实现。选择这两个 agent 是为了观察不同架构和推理策略对资源使用的影响。

**数据收集。** 对于每个任务执行，我们以 1 秒间隔采样 CPU 利用率和内存使用量，并记录每个工具调用的类型、开始时间和结束时间。所有任务在相同的沙箱环境中执行，以确保测量的可比性。

## 3.2 RQ1: Agent Execution Model

Agent 的执行过程与传统容器化工作负载存在根本差异。与 serverless/FaaS 处理短暂无状态请求（100ms–2s）不同，每个 agent 任务平均运行约 10 分钟，使用约 10GB 磁盘空间的 Docker 镜像，并执行有状态的多轮推理和工具调用循环。

**阶段划分。** Agent 执行由 LLM 推理和工具调用两个阶段交替组成。在所有任务中，工具执行时间平均占总执行时间的 28.2%，LLM 思考时间占 71.8%。然而，这一比例在不同任务间差异巨大，范围从 0.1% 到 73.3%。

![工具时间占比分布](../../analysis/haiku_figures/chart_03_tool_ratio_distribution.png)

**工具类型分布。** 测试执行（pytest、unittest 等）占 Bash 总时间的 44.1%，是最耗时的操作类别。Python 代码片段执行占 26.7%，包安装占 10.9%。不同类型的工具调用具有截然不同的资源消耗特征：测试执行通常是 CPU 和内存密集型操作，而文件探索（6.2%）和 Git 操作（2.1%）则相对轻量。

![Bash 命令类型时间分布](../../analysis/haiku_figures/chart_06_bash_categories.png)

**工具执行时间差异。** Bash 命令平均执行时间为 2.64 秒，而 Task（子 agent 调用）平均执行时间高达 66.16 秒。相比之下，Read 和 Edit 操作的平均执行时间仅为 0.06 秒和 0.04 秒。这种三个数量级的差异表明，不同工具类型需要不同的资源配置策略。

![工具执行时间对比](../../analysis/haiku_figures/chart_04_tool_usage_breakdown.png)

## 3.3 RQ2: Resource Unpredictability

Agent 工作负载的资源使用难以预测，这种不可预测性源于三个维度：时间动态性、非确定性和异构性。

### 时间动态性

Agent 工作负载的资源使用呈现剧烈的时间波动特征。可以观察到，内存使用在单个采样间隔（1 秒）内变化高达 2.9GB，CPU 利用率出现剧烈波动，峰值超过 100%（多核利用）。资源使用呈现明显的"突发"模式，而非平稳变化。

![资源使用时序图](../../analysis/haiku_figures/rq1_resource_timeseries.png)

我们观察到最大内存变化率达到 3GB/秒，最大 CPU 变化率超过 50%/秒。显著变化事件（CPU 变化超过 20% 或内存变化超过 50MB/秒）占所有采样点的 1.6%–4.1%。

![资源变化率分布](../../analysis/haiku_figures/rq1_change_rate_distribution.png)

内存峰值可能出现在执行的任何阶段——早期（前 1/3）、中期（中间 1/3）或后期（后 1/3）。这种不可预测性进一步增加了静态资源分配的难度。

![内存峰值时机分布](../../analysis/haiku_figures/chart_13_memory_peak_timing.png)

### 非确定性

与传统容器化工作负载不同，agent 工作负载表现出高度的非确定性。即使对完全相同的任务执行多次，资源使用模式和执行结果也会显著不同。我们对同一任务（DevOps_Build_Hard）执行了三次，观察到：执行时间分别为 402 秒、222 秒和 259 秒，差异达到 1.8 倍。更重要的是，三次执行产生了完全不同的解决方案——不同的代码修改、不同的文件变更数量、甚至不同的实现策略。这种非确定性源于 LLM 推理的随机性和 agent 决策路径的多样性，使得基于历史数据预测资源需求变得极其困难。

### 异构性

不同任务和不同 agent 之间的资源需求存在显著差异。峰值内存需求范围从 197MB 到 4GB，变异系数（CV）达到 147%。ML_Scientific 和 Medical_Bio 类别的任务表现出显著高于 CLI_Tools 或 Web_Network 任务的内存需求，但所有任务都在同一个容器中运行。

![不同任务类别的资源需求](../../analysis/haiku_figures/rq2_category_boxplots.png)

Haiku 和 Qwen agent 在相同 18 个任务上表现出 3.9 倍的 CPU 利用率差异（Haiku 平均 30.6%，Qwen 平均 7.9%）。平均执行时间也存在显著差异（Haiku 400 秒，Qwen 607 秒）。这一结果表明，资源需求不仅取决于任务本身，还取决于 agent 的架构和实现。

![不同 Agent 的 CPU 利用率对比](../../analysis/comparison_figures/04_cpu_utilization_comparison.png)

基于平均 CPU 利用率，Haiku agent 理论上可以并发运行约 3 个实例，Qwen agent 可以并发运行约 12 个实例。然而，由于资源突发和非确定性的存在，实际并发能力会受到峰值资源需求的限制。

## 3.4 RQ3: Provisioning Efficiency

静态资源分配在 agent 工作负载上表现出严重的效率问题。如果将 CPU 限制设置为峰值需求，Haiku 数据集的实际利用率仅为 24%，浪费 76%；Qwen 数据集的实际利用率仅为 7%，浪费 93%。过度供给因子在 CPU 上为 4.1×–13.6×，在内存上为 1.6×–2.4×。

![静态资源限制的过度供给分析](../../analysis/haiku_figures/rq4_overprovisioning.png)

静态预算无论如何设置都会产生问题：保守设置（按峰值）导致 76%–93% 的资源浪费；激进设置（按平均）在突发期间导致 OOM 或性能下降。

## 3.5 Summary and Implications

基于以上观察，我们识别出现有资源管理工具在处理 AI agent 工作负载时的两个根本性问题。

**时间尺度不匹配。** 用户空间资源控制器的典型工作流程是监控资源压力指标（如 PSI、memory.events），做出调整决策，然后写入 cgroup 控制文件。这个循环通常需要 10–100ms。然而，RQ2 的测量表明 agent 工作负载的资源变化发生在秒级甚至更快的时间尺度上。当用户空间控制器观察到内存压力并调整限制时，突发已经导致了 reclaim 风暴或运行队列膨胀。任何基于用户空间监控和 cgroup 文件写入的方法都无法及时响应 agent 工作负载的资源突发。

**域不匹配。** 现有资源控制在容器粒度设置静态预算（如 Kubernetes resource limits、Docker --memory/--cpus、systemd ResourceControl），但 agent 工作负载需要工具调用级别的动态控制。RQ1 表明 agent 执行由 LLM 推理和工具调用两个阶段交替组成，不同工具类型的执行时间差异达到三个数量级。RQ2 表明资源使用具有时间动态性、非确定性和异构性三重不可预测性。RQ3 表明静态限制导致 76%–93% 的资源浪费。静态资源限制无法适应 agent 工作负载的动态、多相位、非确定性特性。

此外，现有工具缺乏表达相位感知控制（根据 LLM 推理 vs 工具执行阶段调整资源）、工具类型感知控制（根据工具类型如测试执行 vs 文件读取分配资源）和跨资源协调（CPU 与内存策略联动）的能力。

这些差距共同表明：有效的 agent 资源管理需要内核级执行（控制逻辑在内核执行点实现微秒级响应）和动态细粒度控制（资源域与工具调用边界对齐）。这正是 AgentCgroup 的设计动机。
