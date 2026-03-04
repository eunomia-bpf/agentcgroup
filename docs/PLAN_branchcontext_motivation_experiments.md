# Branch Context Motivation 实验计划

## 背景

本文提出 "branch context"——一种面向 agentic exploration 的新 OS 抽象，提供 copy-on-write 状态隔离、fork/explore/commit 生命周期、first-commit-wins 语义和嵌套支持。论文的 Section 2（Motivation）做出了以下核心声明，需要实验数据支撑：

1. **AI agent 确实在频繁地进行多路径探索**，现有 ad-hoc 方案（git stash、容器克隆、临时目录）不够用
2. **六项需求（R1-R6）是真实存在且未被满足的**
3. **现有 OS 机制有具体的功能缺陷**（论文 Table 1 和 Table 2）

现有 AgentCgroup 项目已有 SWE-bench trace 数据（144 个任务），但**现有 trace 数据不足以支撑所有分析**——缺少文件系统快照、进程列表、/tmp 状态等关键数据。需要重跑一组任务并增强数据收集。

### 论文中需要 motivate 的六项需求

| 需求 | 含义 | 核心挑战 |
|------|------|---------|
| R1: 隔离的并行执行 | 多条探索路径同时运行，各自有独立的文件系统视图、进程空间和内存状态 | 并发修改同一文件/共享内存会导致状态污染 |
| R2: 原子提交 + 单赢者决议 | 成功路径原子应用（文件系统+进程状态），兄弟路径自动失效并释放资源 | 需要事务语义，手动 merge 容易出错，失败路径的内存/进程需可靠回收 |
| R3: 层次嵌套 | Tree-of-Thoughts 等模式需要嵌套分支，每层向父级提交 | 现有机制要么不支持嵌套，要么嵌套后性能退化 |
| R4: 完整状态覆盖 | agent 产生文件系统变更（构建产物、包安装）、内存状态（加载的数据/模型）、临时资源（/tmp、IPC） | git 只追踪已跟踪文件，进程内存和 IPC 状态完全不可回滚 |
| R5: 轻量、无需特权、可移植 | 分支创建必须亚毫秒级、不需要 root、跨文件系统可用 | VM 快照/特权容器/文件系统绑定方案均违反此要求 |
| R6: 进程协调 | 分支内的进程在 commit/abort 时必须可靠终止并释放内存，兄弟分支间相互隔离 | 进程组可逃逸，cgroup 需 root，PID 命名空间有 init 开销 |

### 需要隔离的状态维度

Branch context 不仅仅是文件系统隔离——agent 的探索路径会产生**多维状态**，都需要在 abort 时被干净地回收：

| 状态维度 | 具体内容 | abort 时需要的操作 | 现有方案能否处理 |
|----------|---------|-------------------|-----------------|
| **文件系统** | 源文件修改、包安装、构建产物、缓存 | 丢弃所有 CoW 页面 | git：仅部分；OverlayFS：需 root |
| **进程内存** | 加载的测试数据、编译器内存、运行时堆 | 终止所有进程，释放内存 | kill -PGID：可逃逸；cgroup：需 root |
| **临时文件/IPC** | /tmp 文件、Unix socket、共享内存段、管道 | 清理所有临时资源 | 无统一机制，需逐个追踪 |
| **环境状态** | 环境变量、工作目录、ulimit 设置 | 恢复到分支创建时的快照 | 无机制支持 |
| **网络状态** | 监听端口、建立的连接 | 关闭所有 socket | PID namespace 可以，但开销大 |

---

## 第零部分：增强数据收集——eBPF 内核级追踪

### 现有数据的不足

现有 trace 数据（`tool_calls.json` + `resources.json` + `trace.jsonl`）只记录了：
- 工具调用类型、时间戳、输入参数（Bash 命令文本、Edit 的 file_path/old_string/new_string）
- 工具返回结果的 preview（截断到 500 字符）
- 1 秒粒度的容器级 CPU/内存采样

**关键缺失**：
1. **无文件系统变更追踪**——不知道 `pip install`/`npm install`/`pytest` 到底改了多少文件、哪些目录
2. **无进程生命周期和协调事件**——不知道是否有残留后台进程、进程组逃逸、fork 行为
3. **无网络端口使用**——不知道 agent 是否启动了 server、绑定了哪些端口
4. **无 per-process 内存分解**——只有容器总内存
5. **无共享内存/mmap 追踪**——不知道跨进程状态共享情况

### 增强采集方案：eBPF 内核级追踪（AgentSight）

**弃用旧方案**：bash wrapper + podman exec 周期采集存在以下问题：
- `find /testbed` 在 4GB 工作空间上耗时数秒，干扰 agent 行为
- 10 秒采样粒度遗漏短暂状态变化（如短命后台进程、临时端口绑定）
- `podman exec` 本身有 ~100ms 开销
- 无法追踪 syscall 级别的细粒度行为（谁删了哪个文件、谁 bind 了哪个端口）

**新方案**：使用 AgentSight（`agentsight/` submodule）的增强 `process` tracer，通过 eBPF tracepoints 在**内核层面**捕获所有关键 syscall。

详细设计见 `docs/PLAN_agentsight_workspace_tracer.md`。核心要点：

#### 追踪能力

在现有 process tracer（EXEC/EXIT/FILE_OPEN/BASH_READLINE）基础上，通过 feature flags 新增：

| Flag | 追踪内容 | 对应 syscall tracepoints |
|------|---------|------------------------|
| `--trace-fs` | 文件删除/重命名/目录创建/写入/截断/chdir | unlinkat, renameat2, mkdirat, write/pwrite64, ftruncate, chdir |
| `--trace-net` | 端口绑定/监听/连接 | bind, listen, connect |
| `--trace-signals` | 进程组变更/会话创建/信号发送/fork | setpgid, setsid, kill, sched_process_fork |
| `--trace-mem` | 共享内存映射 | mmap（仅 MAP_SHARED） |
| `--trace-all` | 以上全部 | — |

#### 关键设计

- **零开销 feature flags**：BPF `const volatile` 变量，JIT 优化禁用分支为 nop
- **统一内核聚合**：所有新事件走 BPF hash map 内核聚合（不经 ring buffer），用户空间每 5 秒 flush
- **PID 过滤复用**：现有 pid_tracker 对所有事件统一生效，`-c python` 同时过滤所有事件类型
- **向后兼容**：不加新 flag 时行为、性能完全不变

#### 输出格式

现有事件不变。新增事件统一输出为 SUMMARY 格式：

```jsonl
{"timestamp":260,"event":"SUMMARY","pid":1234,"comm":"pip","type":"DIR_CREATE","detail":"/testbed/venv/lib/requests","count":47,"extra":"/testbed/venv/lib/requests/utils"}
{"timestamp":260,"event":"SUMMARY","pid":1234,"comm":"pip","type":"WRITE","detail":"fd=5","count":1847,"total_bytes":4521984,"extra":"/testbed/venv/lib/requests/api.py"}
{"timestamp":260,"event":"SUMMARY","pid":5678,"comm":"python","type":"NET_BIND","detail":"0.0.0.0:8080","count":1}
{"timestamp":260,"event":"SUMMARY","pid":1234,"comm":"kill","type":"SIGNAL_SEND","detail":"target=5678,sig=9","count":1}
```

### 实验运行方式

在宿主机上用 AgentSight 追踪容器内的 agent 行为：

```bash
# 启动 SWE-bench 容器
podman run -d --name swebench_task ...

# 在宿主机上用 agentsight 追踪容器内进程（需 root/CAP_BPF）
sudo ./agentsight/bpf/process --trace-all -c python -c node -c bash -m 2 \
    > experiments/branchfs_motivation/<task_name>/ebpf_trace.jsonl 2>/dev/null &

# 或通过 Rust collector（支持 SSL 追踪 + Web UI）
sudo ./agentsight record -c python --trace-all \
    --output experiments/branchfs_motivation/<task_name>/
```

eBPF 运行在宿主机内核，天然能看到容器内进程的所有 syscall——无需在容器内安装任何东西。

### 任务选择

不需要重跑所有 144 个任务。选择 **10-15 个代表性任务**：

**选择标准**：
1. 有大量重试的任务（探索频率高）——从已有数据中选 retry 最多的 5 个
2. 有 pip/npm install 的任务——从已有数据中选有 install 命令的 5 个
3. 有 git stash/checkout 回滚操作的任务——从已有数据中选有回滚的 3-5 个

**已有的候选**：

| 任务 | 特征 | 数据集 |
|------|------|--------|
| `tobymao__sqlglot-4415` | 199 个工具调用，123 Bash（大量探索） | GLM |
| `Algebra8__pyopenapi3-91` | 178 个工具调用，107 Bash | GLM |
| `getsentry__sentry-python-2148` | 21 Bash 含大量 pip install + venv 创建 | Haiku |
| `12rambau__sepal_ui-574` | 有 git stash/pop 回滚 | GLM |
| `AzureAD__microsoft-authentication-library-for-python-186` | 有 git stash + 测试回滚 | GLM |
| `facelessuser__pymdown-extensions-2576` | 有 git stash + 测试回滚 | GLM |
| `beeware__briefcase-2212` | 77 工具调用 | Haiku |
| `numba__numba-9636` | 有 __pycache__ 清理 + git checkout 回滚 | GLM |
| `getsentry__sentry-python-1053` | 有 git checkout + git stash 回滚 | GLM |
| `dask__dask-2205` | 两个数据集都有，可做交叉验证 | Both |

### 重跑的输出格式

每个任务在 `experiments/branchfs_motivation/<task_name>/` 下产出：

```
tool_calls.json          # 原有（工具调用序列）
resources.json           # 原有（CPU/内存采样）
trace.jsonl              # 原有（完整 trace）
ebpf_trace.jsonl         # 新增：eBPF 内核级追踪（所有事件）
```

`ebpf_trace.jsonl` 包含以下事件类型：

| 事件类型 | 来源 | 说明 |
|----------|------|------|
| EXEC/EXIT | 现有 ring buffer | 进程启动/退出 + 完整命令行 + 退出码 |
| FILE_OPEN | 现有 ring buffer | 文件打开（带 dedup） |
| BASH_READLINE | 现有 ring buffer | bash 命令输入 |
| SUMMARY: DIR_CREATE/FILE_DELETE/FILE_RENAME | BPF map flush | 文件系统变更聚合（含 count + 目录前缀） |
| SUMMARY: WRITE | BPF map flush | 写入聚合（含 total_bytes） |
| SUMMARY: NET_BIND/NET_LISTEN/NET_CONNECT | BPF map flush | 网络事件（含 addr:port） |
| SUMMARY: PGRP_CHANGE/SESSION_CREATE/SIGNAL_SEND/PROC_FORK | BPF map flush | 进程协调事件 |
| SUMMARY: MMAP_SHARED | BPF map flush | 共享内存映射 |

**单一文件包含所有维度**——比旧方案（7+ 文件散布在多个目录）大幅简化分析。

### 实施步骤

1. **增强 agentsight process tracer**（详见 `docs/PLAN_agentsight_workspace_tracer.md`）
   - Step 1: 重构 process.c 为 header-only 模块化结构
   - Step 2: 分批添加新 tracepoints（fs → net → signals → mem）
   - Step 3: 扩展 CLI + BPF map flush 逻辑
   - Step 4: 单元测试 + 集成测试
2. **扩展 `run_swebench.py`**：在容器启动后自动启动 agentsight 追踪
3. **选择 10-15 个任务**重跑
4. **编写分析脚本**解析 `ebpf_trace.jsonl`（统一 JSON 格式，比旧方案简单得多）

### eBPF 方案 vs 旧方案对比

| | bash wrapper + podman exec（旧） | eBPF agentsight（新） |
|---|---|---|
| 粒度 | 10 秒采样 / 每次 Bash 调用 | syscall 级别（纳秒精度） |
| 开销 | `find` 数秒 + `podman exec` ~100ms | 内核聚合，<1% CPU |
| 遗漏 | 短暂进程/端口/文件操作 | 无——内核 tracepoint 不会漏 |
| 侵入性 | 修改 bash_wrapper + 容器内安装工具 | 零侵入——宿主机内核追踪 |
| 输出 | 7+ 文件，格式各异 | 单一 JSONL，统一格式 |
| 安装 | 无需 root | 需 root/CAP_BPF（宿主机） |

### 预计资源和时间

- 每个任务平均 10 分钟，10 个任务 ≈ 2 小时（串行）
- 使用 Haiku 模型（云端 API），无需本地 GPU
- agentsight 追踪开销 <1%，不影响 agent 行为
- 磁盘：ebpf_trace.jsonl 约 1-10MB/任务（内核聚合大幅减少数据量）
- 可并行跑 2-3 个（agentsight 可同时追踪多个容器的进程）

---

## 第一部分：数据分析——量化探索模式

> 以下分析同时使用**旧数据**（144 个任务，统计频率/比例）和**新数据**（10-15 个任务，详细快照）。

**总目标**：证明真实 agent 频繁进行多路径探索，且每条路径产生大量**多维状态副作用**（文件系统变更、内存积累、进程残留）需要回滚。

### 实验 1.1：探索频率分析

**目标**：量化 agent 在真实任务中尝试多少条不同的解决策略，以及回滚操作的频率。

**数据来源**：144 个 SWE-bench traces（33 Haiku + 111 GLM），位于 `experiments/all_images_haiku/` 和 `experiments/all_images_local/`

**方法**：

1. **识别不同的解决策略**（不仅仅是重试）：
   - 已有发现：85-97% 的任务包含重试循环（连续 3 次以上 Bash 调用）
   - 扩展分析：在重试组之间，检测 agent 是否改变了策略（判定标准：编辑了不同的文件、应用了不同的 patch、尝试了不同的实现路径）
   - 具体做法：提取每个重试组之间的 Edit/Write 操作，比较修改的文件集合和 patch 内容的相似度；若文件集合或 patch 差异超过阈值，则判定为新的探索策略

2. **统计显式回滚操作**：
   - 扫描 trace 中的以下模式：
     - `git checkout -- <file>`、`git stash`、`git restore` 等 git 回滚命令
     - 手动重新编辑以恢复原始内容（Edit 操作的 new_content 与文件的原始内容匹配）
     - `rm -rf` 删除之前创建的文件/目录
   - 这些操作直接 motivate R2（自动回滚需求）

3. **分析并行子 agent**：
   - Haiku 数据中有 17 个 Task（子 agent）调用
   - 检查这些 Task 是否时间上重叠（并发执行）
   - 如果并发，检查它们是否共享文件系统状态

**预期产出**：
- 表格：每个任务的独立探索路径数量、回滚操作次数、每条路径修改的文件数
- 关键数字：例如 "X% 的任务尝试了 ≥2 种不同的解决策略"，"平均每个任务有 Y 次显式回滚操作"

**Motivate 目标**：Section 2.1（agent 模式）、R1（隔离并行执行）、R2（原子提交）

**分析脚本位置**：基于 `analysis/analyze_new_insights.py` 中的重试分析扩展

---

### 实验 1.2：每次探索的文件系统变更范围

**目标**：量化每次探索尝试产生多少文件系统变更，其中多少是 git 无法追踪的。

**数据来源**：
- **旧数据**（144 个任务）：从 trace 中通过命令语义分析推断文件变更
- **新数据**（10-15 个重跑任务）：`ebpf_trace.jsonl` 中的 SUMMARY 事件提供精确的 syscall 级追踪

**方法**：

1. **文件变更统计**（新数据 eBPF 追踪）：
   - 从 `ebpf_trace.jsonl` 中提取 SUMMARY 事件：
     - `DIR_CREATE`：创建的目录数量及路径分布（detail 字段 = 父目录前缀）
     - `FILE_DELETE`：删除的文件数量（`rm -rf` 场景下 count 很高）
     - `FILE_RENAME`：重命名数量（`pip install` 的原子安装模式）
     - `WRITE`：写入的文件数量和总字节数（total_bytes 字段）
   - 按探索尝试（重试组）分组统计

2. **文件变更统计**（旧数据命令分析）：
   - 从 trace 中提取 Edit/Write 工具调用 → 源代码编辑
   - 从 Bash 命令中识别文件变更操作（pip install、npm install、make、pytest 等）

3. **区分 git 可追踪 vs 不可追踪的变更**：
   - **git 可追踪**：源代码文件的编辑（.py, .js, .ts 等）
   - **git 不可追踪**（eBPF 追踪到但 git 看不到的）：
     - `pip install` → `site-packages/` 下大量 DIR_CREATE + FILE_RENAME（eBPF detail 字段可直接看到路径前缀）
     - `npm install` → `node_modules/` 下的变更
     - `make build` → `build/`、`dist/`、`__pycache__/`
     - `pytest` → `.pytest_cache/`、`htmlcov/`
   - 从 eBPF SUMMARY 事件的 detail 字段按路径前缀分类统计

4. **量化"git 盲区"**：
   - 计算：所有文件变更中，有多少比例是 git 不会追踪的
   - eBPF 数据提供精确数字（vs 旧数据的估算）
   - 这直接 motivate R4

**预期产出**：
- 饼图/柱状图：文件变更类型分布（源代码编辑 vs 包安装 vs 构建产物 vs 测试缓存）
- 关键数字：例如 "N% 的文件系统变更不在 git 追踪范围内"
- eBPF 精确数据：如 "`pip install flask` 产生 47 个 DIR_CREATE + 203 个 FILE_RENAME + 4.5MB WRITE"

**Motivate 目标**：R4（完整文件系统覆盖）、Section 2.1（git stashing 的局限性）

---

### 实验 1.3：重试循环中的内存积累与资源泄漏

**目标**：量化 agent 在多次探索尝试中产生的内存积累和进程残留，证明需要 branch-level 的资源回收。

**数据来源**：已有 144 个任务的 `resources.json`（1 秒粒度 CPU/内存采样）+ `tool_calls.json`（工具调用时序）

**方法**：

1. **内存积累分析**：
   - 已有发现：重试循环导致内存逐次积累，最极端案例 502MB 不释放
   - 扩展分析：
     - 对每个任务，识别所有重试组（连续 Bash 调用）
     - 测量每个重试组**结束后**的内存基线（相比重试组开始前）
     - 计算累积的内存"泄漏"：每次重试后基线升高了多少
     - 这些泄漏的内存就是 branch context 在 abort 时应该回收的

2. **内存突发与探索路径的对齐分析**：
   - 已有发现：98.5% 的内存突发发生在工具调用期间，峰均比 15.4×
   - 新分析：
     - 将内存突发按"探索尝试"分组（而非按单个工具调用）
     - 计算每个探索尝试的峰值内存——这就是 branch context 需要隔离的内存量
     - 如果 N 条路径并行，总内存需求 = N × 单路径峰值，而非所有路径共享一个峰值

3. **进程残留分析**：
   - 从 Bash 调用中识别会产生后台进程的命令（如 `nohup`、`&`、daemon 启动）
   - 分析 agent 是否在重试时清理了之前的后台进程
   - 检查是否有 agent 因前一轮残留的进程（如仍在运行的 test server）导致错误

4. **并行探索的资源倍增估算**：
   - 基于已有数据，假设 agent 的 K 次串行重试改为 K 路并行探索
   - 估算并行时的总内存需求：K × 单路径峰值内存
   - 与当前机器内存（128GB）对比，计算可支撑的最大并行度
   - 这 motivate 了 branch context 需要在 abort 时**立即**释放内存的重要性

**预期产出**：
- 图表：重试循环中内存基线的逐步升高（"阶梯状"内存曲线），标注每次重试的增量
- 数字：平均每个任务因重试累积 XMB 未释放内存
- 数字：若并行探索 3 条路径，峰值内存需求增加 Y 倍
- 关键结论：**没有 branch-level 的资源回收，失败的探索路径会持续占用内存，限制并行探索的可行性**

**Motivate 目标**：R2（abort 时释放资源）、R1（并行探索的资源隔离）

**分析脚本位置**：基于 `analysis/analyze_new_insights.py` 中的重试分析 + `analysis/analyze_swebench_data.py` 中的内存分析扩展

---

### 实验 1.4：探索树深度分析

**目标**：证明 agent 的探索具有层次结构，需要嵌套分支支持。

**数据来源**：Haiku traces（含 Task 子 agent 调用）

**方法**：

1. **重建探索树**：
   - 从 trace 中提取 Task（子 agent）调用
   - 分析子 agent 内部是否也有重试/分支行为
   - 测量最大探索深度

2. **调研现有 agent 框架的探索策略**：
   - **Claude Code**：per-file snapshots，不支持并行分支，shell 命令产生的变更无法快照
   - **SWE-agent**：sequential retry，无并行探索
   - **OpenHands**：支持多种 agent 策略，但探索仍是串行的
   - **Tree-of-Thoughts / Graph-of-Thoughts**：设计上需要嵌套搜索，但缺乏 OS 级支持
   - 引用这些框架的文档/代码，说明嵌套探索是被需要但尚未被支持的

3. **统计 Claude Code 的 Task 嵌套深度**：
   - Haiku 数据中 Task 调用的嵌套关系
   - 是否有 Task 内再调用 Task 的情况

**预期产出**：
- 探索树深度分布
- 框架对比表：各框架的探索策略及其局限

**Motivate 目标**：R3（层次嵌套）

---

## 第二部分：基准测试现有机制——证明它们不够用

**总目标**：用具体的性能数据和功能测试，实证验证 Table 1 和 Table 2 的声明。

### 实验 2.1：分支创建与提交开销对比

**目标**：量化各种隔离机制的创建和提交延迟，证明 BranchFS 的 O(1) 创建优势。

**待测机制**：

| # | 机制 | 类别 | 备注 |
|---|------|------|------|
| 1 | `git stash` + `git stash pop` | 版本控制 | 仅追踪文件 |
| 2 | `git checkout -b` + merge | 版本控制 | 仅追踪文件 |
| 3 | `git worktree add` | 版本控制 | 独立工作目录，但仅追踪文件 |
| 4 | `cp -r` + `rsync` 回写 | 完全复制 | 完整覆盖但 O(n) 开销 |
| 5 | `podman run` / `docker run`（容器克隆） | 容器化 | 重量级，需镜像 |
| 6 | OverlayFS mount | 联合文件系统 | 需 root 权限 |
| 7 | Btrfs snapshot（用 loopback 设备测试） | 文件系统级 | 依赖特定文件系统 |
| 8 | BranchFS（FUSE 实现） | 本文方案 | 无需 root，O(1) 创建 |

**测试工作负载**：变化工作空间大小

| 级别 | 大小 | 说明 |
|------|------|------|
| Small | 10MB | 最小项目（几个源文件） |
| Medium | 100MB | 典型代码仓库 |
| Large | 1GB | 仓库 + node_modules |
| XL | 4GB | SWE-bench 镜像工作空间（平均 4.1GB） |

**测量指标**：

1. **分支创建延迟**（μs）：从发起创建到分支可用的时间
2. **提交/合并延迟**（μs）：将分支变更应用回父级的时间（在分支中做少量修改后测量）
3. **每个分支的内存开销**（KB）：分支本身的元数据内存占用
4. **每个分支的磁盘空间开销**（KB）：CoW 前（仅元数据）和 CoW 后（有修改）的磁盘占用
5. **并发分支扩展性**：同时创建 1/10/100/1000 个分支的总时间

**实验设计**：
- 每个配置重复 100 次取中位数和 P99
- 创建分支后修改 1 个文件（1KB），然后测量 commit 时间
- 同时测量修改多个文件（10/100/1000 个）后的 commit 时间

**预期产出**：
- 柱状图：各机制在不同工作空间大小下的创建延迟
- 表格：BranchFS 创建 < 350μs 且与基础文件系统大小无关；cp -r / container clone 线性增长
- 关键结论：BranchFS 比 cp -r 快 N 个数量级，比 container clone 快 M 倍

**Motivate 目标**：R5（轻量级）、Table 1

---

### 实验 2.2：文件系统覆盖范围对比

**目标**：证明 git 基方案无法捕获 agent 产生的全部文件系统变更。

**实验设置**：

1. **准备工作空间**：一个包含以下内容的项目目录
   - git 仓库（有 .gitignore）
   - `node_modules/`（200MB+，被 .gitignore 忽略）
   - `.cache/`（构建缓存）
   - `build/`（构建产物）
   - `.env`（环境配置）

2. **模拟一个 agent 探索路径**：执行以下操作序列
   ```bash
   # 源代码编辑（git 可追踪）
   echo "fix" >> src/main.py

   # 包安装（git 不追踪）
   pip install requests
   npm install lodash

   # 构建（git 不追踪）
   python setup.py build

   # 测试（产生缓存，git 不追踪）
   pytest --cache-clear

   # 环境修改（git 不追踪）
   echo "NEW_VAR=1" >> .env
   ```

3. **用每种机制尝试"回滚"**：
   - `git checkout -- .` + `git clean -fd`：检查哪些变更被恢复、哪些遗漏
   - `git stash` + `git stash pop`：同上
   - OverlayFS：discard upper layer，检查是否完全恢复
   - BranchFS：abort branch，检查是否完全恢复

4. **对比方法**：操作前后用 `find . -newer <timestamp>` 或 `diff -r` 对比文件系统快照

**预期产出**：

| 变更类型 | git stash | git worktree | cp -r | OverlayFS | BranchFS |
|----------|-----------|-------------|-------|-----------|----------|
| 源文件编辑 | ✓ | ✓ | ✓ | ✓ | ✓ |
| pip install | ✗ | ✗ | ✓ | ✓ | ✓ |
| npm install | ✗ | ✗ | ✓ | ✓ | ✓ |
| 构建产物 | ✗ | ✗ | ✓ | ✓ | ✓ |
| 测试缓存 | ✗ | ✗ | ✓ | ✓ | ✓ |
| .env 修改 | ✗ | ✗ | ✓ | ✓ | ✓ |

**Motivate 目标**：R4（完整文件系统覆盖）、Table 1

---

### 实验 2.3：进程隔离对比

**目标**：用具体测试用例验证 Table 2 中各进程管理机制的隔离缺陷。

**测试场景**：

#### 场景 1：进程逃逸测试

```c
// child.c - 一个会逃逸进程组的子进程
#include <unistd.h>
int main() {
    setsid();           // 创建新 session，逃离原进程组
    // 或 setpgid(0, 0);  // 创建新进程组
    while(1) sleep(1);  // 持续运行
    return 0;
}
```

- 在进程组/session 中启动此进程
- 尝试 `kill(-pgid, SIGKILL)` 终止整个组
- 检查逃逸进程是否仍存活

**预期结果**：

| 机制 | 逃逸进程是否存活 |
|------|-----------------|
| 进程组 (pgrp) | ✓ 存活（已逃逸） |
| Session | ✓ 存活（已用 setsid 逃逸） |
| cgroup | ✗ 被终止（cgroup.kill 杀死所有成员） |
| PID namespace | ✗ 被终止（namespace 销毁杀死所有进程） |
| branch() | ✗ 被终止（内核强制） |

#### 场景 2：跨分支信号干扰测试

```python
# 两个并发"探索路径"
# Path A 的进程尝试 kill Path B 的进程
import os, signal
os.kill(path_b_pid, signal.SIGKILL)  # 能否成功？
```

| 机制 | 跨组信号是否可发送 |
|------|------------------|
| 进程组 | ✓ 可以（同 UID 即可） |
| Session | ✓ 可以（同 UID 即可） |
| cgroup | ✓ 可以（cgroup 不阻止跨组信号） |
| PID namespace | ✗ 不可以（不同 namespace 看不到对方） |
| branch() | ✗ 不可以（内核强制兄弟隔离） |

#### 场景 3：孤儿进程清理测试

```python
# 父进程 fork 子进程后退出
import os
pid = os.fork()
if pid == 0:
    # 子进程：继续运行
    while True: time.sleep(1)
else:
    # 父进程：立即退出
    os._exit(0)
```

- 检查父进程退出后，子进程是否仍存活
- 尝试清理：各机制能否可靠终止所有后代进程

**实验实施**：
- 用 C/Python 编写上述测试程序
- 在每种机制下运行，记录结果
- 特别注意 cgroup 的权限问题（需要测试有/无 root 两种情况）

**预期产出**：填充完整的 Table 2，附带实测 pass/fail 结果

**Motivate 目标**：R6（进程协调）、Table 2

---

### 实验 2.4：权限需求验证

**目标**：验证各机制对 root 权限的依赖。

**方法**：以普通用户（无 root、无 CAP_SYS_ADMIN）身份尝试各操作：

```bash
# 1. OverlayFS
mkdir lower upper work merged
mount -t overlay overlay -o lowerdir=lower,upperdir=upper,workdir=work merged
# 预期：mount: permission denied

# 2. Btrfs snapshot
btrfs subvolume snapshot /path/to/subvol /path/to/snap
# 预期：permission denied

# 3. Device-mapper snapshot
dmsetup create snap ...
# 预期：permission denied

# 4. Cgroup 创建
mkdir /sys/fs/cgroup/user.slice/test_cgroup
# 预期：permission denied（除非 systemd 已委派）

# 5. PID namespace
unshare --pid --fork bash
# 预期：需 CAP_SYS_ADMIN 或 user namespace 支持

# 6. BranchFS（FUSE）
branchfs mount /path/to/workspace /path/to/mountpoint
# 预期：成功（FUSE 允许普通用户挂载）
```

**预期产出**：表格对比各机制是否需要 root 权限

**Motivate 目标**：R5（无需特权）、Table 1 & 2

---

## 第三部分：端到端演示——展示问题的实际影响

**总目标**：通过具体案例展示没有 branch context 时会出什么问题，以及有了之后如何解决。

### 实验 3.1：状态污染演示

**目标**：直观展示并发探索在没有隔离的情况下导致状态污染。

**实验设置**：

1. 创建一个简单的 Python 项目，包含 `main.py` 和 `test_main.py`
2. `main.py` 有一个 bug 需要修复
3. 同时启动两个"探索路径"（两个 shell 进程）：
   - **Path A**：应用修复策略 1（修改 `main.py` 的第 10 行），然后运行 `pytest`
   - **Path B**：应用修复策略 2（修改 `main.py` 的第 10 行为不同内容），然后运行 `pytest`

**无隔离场景**：
```
时间线：
T0: Path A 写入 main.py (策略1)
T1: Path B 写入 main.py (策略2) ← 覆盖了 Path A 的修改！
T2: Path A 运行 pytest ← 实际测试的是策略2，不是策略1！
T3: Path B 运行 pytest
结果：Path A 报告"策略1成功"，但实际测试的是策略2的代码 → 错误的结论
```

**有 BranchFS 隔离场景**：
```
T0: 创建 Branch A 和 Branch B（各自有独立的 CoW 视图）
T1: Branch A 写入 main.py (策略1) ← 仅影响 Branch A
T2: Branch B 写入 main.py (策略2) ← 仅影响 Branch B
T3: Branch A 运行 pytest ← 正确测试策略1
T4: Branch B 运行 pytest ← 正确测试策略2
T5: Branch A 测试通过 → commit → Branch B 自动失效
```

**实现方式**：
- 用简单的 shell 脚本模拟两个并发路径
- 无隔离版本：两个进程操作同一目录
- 有隔离版本：两个进程分别操作 BranchFS 的两个分支

**预期产出**：
- 终端日志对比：无隔离时的状态污染 vs 有隔离时的正确执行
- 可用于论文中的示例/figure

**Motivate 目标**：R1（隔离并行执行）、R2（原子提交 + 单赢者）

---

### 实验 3.2：真实 Agent Trace 重放

**目标**：用真实 SWE-bench 任务的 trace 展示 branch context 能带来的加速。

**实验设计**：

1. **选取任务**：从已有数据中选择重试循环最多的任务（如 GLM 最大连续重试 56 次的任务）

2. **串行重放（现状）**：
   - 按 trace 原始顺序回放所有操作
   - 测量花在"回滚"上的时间（git checkout、重新编辑等操作的累计时间）
   - 测量总端到端时间

3. **并行重放（假设有 branch context）**：
   - 在每个重试点创建 branch
   - 并行执行多条探索路径
   - 第一个成功的路径 commit，其余 abort
   - 测量理论加速比

4. **分析**：
   - 回滚时间占总执行时间的比例
   - 并行探索的理论加速比（假设 N 条路径并行，加速比 ≈ N / 串行尝试次数）
   - 被浪费的计算（失败路径执行的工具调用总时间）

**预期产出**：
- 具体数字：例如 "任务 X 串行执行 10 分钟，其中 2 分钟花在回滚上；若并行探索可缩短至 6 分钟"
- 图表：不同任务的潜在加速比分布

**Motivate 目标**：整体 motivation、R1-R6

**注意**：此实验需要 BranchFS 集成，优先级较低（P3），但即使不实际运行，也可以基于 trace 数据进行理论分析。

---

### 实验 3.3：包安装副作用演示

**目标**：具体展示 git 无法回滚包安装等副作用。

**实验步骤**：

```bash
# 1. 初始化工作空间
mkdir workspace && cd workspace
git init
npm init -y
npm install express  # 安装基础依赖
git add . && git commit -m "initial"

# 记录初始状态
find . | wc -l  # 例如：5000 个文件
du -sh .         # 例如：50MB

# 2. 模拟 agent 探索路径
npm install lodash moment axios webpack  # 安装新包
echo "import lodash" >> src/main.js       # 编辑源文件
npm run build                              # 生成构建产物

# 记录变更后状态
find . | wc -l  # 例如：8000 个文件（+3000）
du -sh .         # 例如：120MB（+70MB）

# 3. 尝试用 git 回滚
git checkout -- .
git clean -fd

# 检查残留
find . -newer /tmp/timestamp | wc -l  # 仍有大量未被 git 追踪的文件
du -sh node_modules/                   # node_modules 中的新包仍然存在
ls build/                               # 构建产物仍然存在
```

**对比方案**：

| 回滚方式 | 耗时 | 回滚完整性 | 残留文件 |
|----------|------|-----------|---------|
| git checkout + clean | <1s | 仅源文件 | node_modules 新包、build/、.cache/ |
| cp -r 恢复 | 数秒~数十秒 | 完整 | 无（但创建备份时已付出 O(n) 代价） |
| BranchFS abort | <1ms | 完整 | 无（CoW 丢弃即可） |

**预期产出**：
- 数据：git 回滚后残留了 N 个文件、M MB 的"泄漏"状态
- 关键结论：git 基方案在包安装场景下遗漏 >90% 的文件变更

**Motivate 目标**：R4（完整文件系统覆盖）

---

## 第四部分：内存与进程状态隔离——超越文件系统

**总目标**：证明 branch context 需要隔离的不仅仅是文件系统，还包括进程内存、共享状态和临时资源。文件系统隔离是必要的但不充分的。

### 实验 4.1：内存状态污染演示

**目标**：展示并发探索路径之间通过共享内存/tmpfs/环境变量产生的状态干扰。

**场景设计**：

#### 场景 A：共享 /tmp 导致的干扰

```python
# Path A: 将测试数据写入 /tmp
with open("/tmp/test_config.json", "w") as f:
    json.dump({"strategy": "A", "param": 1}, f)
subprocess.run(["pytest", "test_suite.py"])  # 测试读取 /tmp/test_config.json

# Path B: 同时也写入 /tmp（相同文件名）
with open("/tmp/test_config.json", "w") as f:
    json.dump({"strategy": "B", "param": 2}, f)
subprocess.run(["pytest", "test_suite.py"])  # 覆盖了 Path A 的配置！
```

- 无隔离：两条路径共享 /tmp，配置文件互相覆盖
- 有 branch context：每条路径有独立的 /tmp 视图（通过 mount namespace 或 CoW）

#### 场景 B：共享内存段干扰

```c
// Path A: 创建共享内存用于进程间通信
int fd = shm_open("/agent_state", O_CREAT|O_RDWR, 0666);
ftruncate(fd, 4096);
void *ptr = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
// 写入 Path A 的状态...

// Path B: 打开同一个共享内存段（因为名字相同）
int fd = shm_open("/agent_state", O_RDWR, 0666);  // 读到了 Path A 的数据！
```

#### 场景 C：环境变量和端口冲突

```bash
# Path A: 启动测试服务器在 8080 端口
export PORT=8080
python manage.py runserver 0.0.0.0:$PORT &

# Path B: 也尝试在 8080 端口启动（失败！）
export PORT=8080
python manage.py runserver 0.0.0.0:$PORT &  # Address already in use
```

**预期产出**：
- 3 个具体的状态污染场景 + 日志证据
- 对比表：各隔离机制能否防止每种污染

| 污染类型 | git stash | cp -r | OverlayFS | PID ns | branch context |
|----------|-----------|-------|-----------|--------|---------------|
| /tmp 文件冲突 | ✗ | ✗ | ✗ | ✓ | ✓ |
| 共享内存干扰 | ✗ | ✗ | ✗ | ✓ | ✓ |
| 端口/socket 冲突 | ✗ | ✗ | ✗ | ✓ | ✓ |
| 环境变量泄漏 | ✗ | ✗ | ✗ | ✓ | ✓ |
| 文件系统变更 | 部分 | ✓ | ✓ | ✗ | ✓ |
| **全部覆盖** | ✗ | ✗ | ✗ | ✗ | ✓ |

**关键结论**：OverlayFS 只隔离文件系统，PID namespace 只隔离进程，唯有 branch context 统一解决所有维度。

**Motivate 目标**：R1、R4（完整状态覆盖，不仅是文件系统）

---

### 实验 4.2：内存回收延迟——失败探索路径的资源成本

**目标**：测量当探索路径失败/abort 时，各种机制释放进程内存的速度和完整性。

**实验设计**：

1. **创建一个内存密集型探索路径**：
   ```python
   # 模拟 agent 加载大型测试数据集
   import numpy as np
   data = np.random.randn(100_000_000)  # ~800MB
   # 运行测试...失败了
   # 现在需要 abort 这条探索路径
   ```

2. **测量各机制的 abort + 内存释放时间**：

   | 机制 | abort 操作 | 内存释放方式 |
   |------|-----------|-------------|
   | 手动 kill | `kill -9 <pid>` | 依赖 OS 回收，可能遗漏子进程 |
   | kill 进程组 | `kill -9 -<pgid>` | 进程可逃逸 setsid → 内存泄漏 |
   | cgroup kill | `echo 1 > cgroup.kill` | 可靠，但需 root/delegation |
   | PID namespace 销毁 | 终止 init 进程 | 可靠，但有 init 开销 |
   | branch() abort | 内核终止所有进程 | 可靠 + 立即释放 |

3. **测量指标**：
   - abort 到内存完全释放的延迟（ms）
   - 是否有残留进程仍占用内存
   - 多次 abort 后的累积内存泄漏

**预期产出**：
- 表格：各机制的 abort 延迟和内存释放完整性
- 关键发现：进程组 kill 在有子进程逃逸时留下 X MB 泄漏

**Motivate 目标**：R2（abort 时的资源回收）、R6（进程协调）

---

### 实验 4.3：从 Trace 数据量化多维状态副作用

**目标**：量化 agent 产生的非文件系统状态副作用的频率和规模。

**数据来源**：
- **旧数据**（144 个任务）：从 Bash 命令文本推断状态副作用（粗粒度）
- **新数据**（10-15 个重跑任务）：`ebpf_trace.jsonl` 提供精确的多维状态追踪

**方法**：

1. **从 eBPF 追踪数据直接量化多维状态副作用**（新数据）：

   | eBPF SUMMARY 事件 | 状态维度 | 分析方式 |
   |-------------------|---------|---------|
   | `NET_BIND` + `NET_LISTEN` | 网络端口占用 | 统计 agent 绑定了哪些端口，是否有端口冲突风险 |
   | `NET_CONNECT` | 外部连接 | 统计 API 调用频率（如 pypi.org:443、anthropic API） |
   | `PGRP_CHANGE` + `SESSION_CREATE` | 进程组逃逸 | 直接验证 agent 的子进程是否调用了 setsid/setpgid |
   | `SIGNAL_SEND` | 进程间信号 | agent 是否在重试时 kill 之前的进程 |
   | `PROC_FORK` | 子进程创建 | 量化每次探索尝试创建的进程数 |
   | `MMAP_SHARED` | 共享内存 | 是否有跨进程共享状态 |
   | `DIR_CREATE` + `FILE_DELETE` + `FILE_RENAME` | 文件系统变更 | 精确的 git 盲区量化 |
   | `WRITE` | 写入规模 | total_bytes 量化每次探索的磁盘写入 |

2. **从 Bash 命令推断**（旧数据，作为补充）：

   | 命令模式 | 状态类型 | 频率统计 |
   |----------|---------|---------|
   | `python -m pytest`、`unittest` | 进程内存（加载测试数据） | 已知占 Bash 时间 44-73% |
   | `pip install`、`npm install` | 文件系统 + 进程内存（编译 C 扩展） | 已知占 ~10% |
   | `python manage.py runserver`、`flask run` | 网络端口 + 进程 | 待统计 |
   | `nohup`、`&`（后台进程） | 持久进程状态 | 待统计 |
   | `export`、`source` | 环境变量 | 待统计 |

3. **内存突发规模按探索尝试分组**：
   - 已有数据：P95 内存 spike 518MB（Haiku）/ 234MB（GLM）
   - 新分析：将这些 spike 按探索尝试分组，计算**每条探索路径的独立内存足迹**
   - eBPF 的 PROC_FORK 事件可精确定位每次探索启动了多少子进程

4. **识别"需要回滚但无法回滚"的状态**：
   - eBPF 数据可直接回答：agent 是否在重试时 kill 了之前的进程（SIGNAL_SEND 事件）
   - 是否有进程在 agent 放弃后仍然绑定端口（NET_BIND 事件 vs 进程 EXIT 事件的时序）
   - 是否有进程通过 setsid 逃逸了进程组（SESSION_CREATE 事件）

**预期产出**：
- 表格：各类非文件系统状态副作用的**精确频率**（eBPF 数据，非推断）
- 数字：X% 的任务产生了超出文件系统的状态副作用
- 数字：每个任务平均产生 Y 个进程组变更、Z 个端口绑定
- 数字：并行 N 路探索的内存需求估算
- **关键新数据**：进程组逃逸（setsid/setpgid）的实际发生频率——直接支撑 Table 2

**Motivate 目标**：R1、R4（完整状态覆盖）、R6（进程协调）

---

### 实验 4.4：并发探索的资源扩展性分析

**目标**：估算如果 agent 从串行探索改为并行探索，对系统资源（特别是内存）的影响。

**方法**：

1. **基于已有数据建模**：
   - 从 144 个任务的资源数据中，提取每个任务的：
     - 峰值内存（P_mem）
     - 平均内存（A_mem）
     - 重试次数（N_retry）
     - 框架基线内存（~185MB）

2. **并行探索资源需求模型**：
   ```
   串行模式内存需求 = P_mem（单路径峰值）
   并行 K 路模式内存需求 = K × (P_mem - baseline) + baseline
                          ≈ K × tool_spike + 185MB
   ```
   - 以 Medical_Bio_Hard 任务为例：P_mem = 4060MB，baseline = 264MB
     - 并行 2 路：264 + 2 × (4060-264) = 7856MB
     - 并行 3 路：264 + 3 × (4060-264) = 11652MB
   - 但如果用 branch context，abort 的路径**立即释放**，实际峰值远小于理论值

3. **对比资源回收策略**：

   | 策略 | 并行 3 路峰值内存 | abort 后内存 | 说明 |
   |------|-----------------|-------------|------|
   | 无隔离（共享内存空间） | 混乱——不可预测 | 不可控 | 状态污染 |
   | 独立容器 | 3 × (P_mem + 容器开销) | 需等待容器销毁 | 每个容器额外 200-500MB |
   | branch context | K × tool_spike + baseline | 立即释放 | CoW 共享基线 |

4. **可支撑的并行探索数**：
   - 在 128GB 机器上，不同方案能支持多少并行探索路径
   - branch context 的 CoW 优势：共享基线，仅隔离增量

**预期产出**：
- 图表：不同并行度下的内存需求（串行 vs 独立容器 vs branch context）
- 表格：128GB 机器上各方案支持的最大并行探索数
- 关键结论：branch context 通过 CoW + 快速 abort 回收，可支撑 N 倍于容器方案的并行探索

**Motivate 目标**：R1（资源高效的并行探索）、R5（轻量级）

---

## 实施优先级

| 优先级 | 实验 | 工作量 | 论文影响 | 说明 |
|--------|------|--------|---------|------|
| **P0** | 1.1（探索频率分析） | 低（已有数据） | 高 — 量化核心声明 | 基于已有 trace 数据编写分析脚本 |
| **P0** | 2.1（分支创建基准测试） | 中 | 高 — Table 1 的核心数据 | 需编写 benchmark 脚本 |
| **P0** | 1.2（文件变更范围） | 低（已有数据） | 高 — motivate R4 | 基于已有 trace 数据 |
| **P0** | 1.3（内存积累分析） | 低（已有数据） | 高 — motivate 内存隔离 | 扩展已有内存 + 重试分析 |
| **P0** | 4.3（多维状态副作用频率） | 低（已有数据） | 高 — 证明不只是文件系统 | 扫描 trace 中的命令模式 |
| **P1** | 2.3（进程隔离对比） | 中 | 高 — Table 2 的核心数据 | 需编写 C/Python 测试程序 |
| **P1** | 4.1（内存状态污染演示） | 中 | 高 — 核心论点扩展 | 3 个具体场景的脚本 |
| **P1** | 4.4（并行资源扩展性） | 低（已有数据） | 高 — 量化 CoW 优势 | 基于已有数据建模 |
| **P1** | 3.1（文件系统状态污染演示） | 低 | 中 — 论文中的直观示例 | 简单 shell 脚本 |
| **P1** | 2.2（文件系统覆盖对比） | 中 | 中 — 验证 Table 1 | 需设置多种环境 |
| **P2** | 4.2（内存回收延迟） | 中 | 中 — 量化 abort 效率 | 需编写内存密集测试 |
| **P2** | 3.3（包安装副作用） | 低 | 中 — R4 的具体案例 | 简单脚本 |
| **P2** | 2.4（权限需求验证） | 低 | 低 — 定性验证 | 运行几条命令即可 |
| **P2** | 1.4（探索树深度） | 中 | 中 — motivate R3 | 需分析 Task 嵌套关系 |
| **P3** | 3.2（真实 trace 重放） | 高 | 高 — 但需 BranchFS 集成 | 可先做理论分析 |

> 调研类和文档分析类任务见 `docs/RESEARCH_branchcontext_survey.md`

---

## 关键文件与资源

| 资源 | 路径/URL |
|------|----------|
| 现有 trace 数据（Haiku） | `experiments/all_images_haiku/` |
| 现有 trace 数据（GLM） | `experiments/all_images_local/` |
| 分析脚本（可复用） | `analysis/analyze_swebench_data.py`, `analysis/analyze_new_insights.py` |
| AgentSight submodule | `agentsight/`（eBPF 追踪工具） |
| AgentSight 增强计划 | `docs/PLAN_agentsight_workspace_tracer.md` |
| BranchFS 仓库 | https://github.com/multikernel/branchfs |
| 论文源码 | `paper-repo/main.tex` |

## 实验与 eBPF 追踪事件的对应

| 论文实验 | 需要的 eBPF 事件 | agentsight flag |
|----------|-----------------|----------------|
| 1.2（文件变更范围） | DIR_CREATE + FILE_DELETE + FILE_RENAME + WRITE | `--trace-fs` |
| 2.2（文件系统覆盖对比） | 同上 | `--trace-fs` |
| 2.3（进程隔离对比/Table 2） | PGRP_CHANGE + SESSION_CREATE + SIGNAL_SEND | `--trace-signals` |
| 3.1（状态污染演示） | NET_BIND（端口冲突） | `--trace-net` |
| 3.3（包安装副作用） | DIR_CREATE + FILE_RENAME + WRITE | `--trace-fs` |
| 4.1（多维状态污染） | NET_BIND + SIGNAL_SEND + MMAP_SHARED | `--trace-net --trace-signals --trace-mem` |
| 4.3（多维副作用频率） | 全部事件 | `--trace-all` |

## 验证标准

1. **P0 实验**产出可直接引用在 Section 2 中的具体数字
2. **基准测试**结果用实测数据替代 Table 1 & 2 中的纯定性对比（✓/✗）
3. **至少一个端到端演示**（3.1 或 4.1）提供令人信服的 "before/after" 对比叙事
4. **内存/进程维度**的实验（第四部分）必须有至少 2 个产出，证明 branch context 的价值超越纯文件系统隔离
5. 所有实验可复现，脚本和原始数据一并保存在 `experiments/branchfs_motivation/` 目录下

## 第五部分：补充分析方向

调研类和纯文档分析类的内容已移至 `docs/RESEARCH_branchcontext_survey.md`，包括：
- 5.1 正确性影响量化
- 5.2 首次成功率 / 投机成功率（需从 trace 数据分析）
- 5.3 现有 Agent 框架隔离机制调研
- 5.4 组合竞态窗口演示
- 5.5 跨工作空间修改分析（需从 trace 数据分析）
- 5.6 存储放大分析
- 5.7 不可逆外部副作用边界讨论
- 5.8 数据库事务类比分析

---

## 论文叙事建议

建议 Section 2 的 motivation 按以下逻辑组织：

1. **Agent 确实在做多路径探索**（实验 1.1 的数据）
2. **每条探索路径产生多维状态副作用**（实验 1.2 文件系统 + 1.3 内存 + 4.3 其他状态）
   - 不只是文件修改，还有内存积累、进程残留、临时资源
3. **现有机制各自覆盖部分维度，但没有统一方案**（实验 2.x + 4.1 的对比表）
   - git 只管文件的子集
   - OverlayFS 管文件系统但需 root
   - PID namespace 管进程但有 init 开销
   - 没有一个方案同时覆盖文件系统 + 内存 + 进程 + 临时资源
4. **组合现有机制是脆弱的**（实验 4.1 的具体故障场景）
   - 步骤间存在竞态窗口
   - 部分失败时的清理容易出错
5. **并行探索对资源的影响是可管理的**（实验 4.4 的资源模型）
   - CoW 共享基线，abort 立即释放
   - 相比容器方案支持更高并行度
