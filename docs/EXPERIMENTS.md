# AgentCgroup 实验设计（基于 Trace 回放）

本文档描述 AgentCgroup 的 trace 驱动实验方法。所有实验使用预先收集的 trace 进行回放，以确保可重复性并消除 LLM 的随机性。

## 1. 为什么使用 Trace 回放？

对于 OS/资源控制的评估，trace 驱动回放是首选方法：

| 方法 | 优点 | 缺点 |
|------|------|------|
| **在线执行（带 LLM）** | 真实的 agent 行为 | 不确定性、昂贵、慢 |
| **Trace 回放** | 确定性、快速、可重复、无 LLM 成本 | 需要预先收集的 trace |

**我们的方法**：回放预先收集的 tool call traces，生成可重复的资源压力模式。

## 2. 可用的 Trace 数据源

### 2.1 代码/CLI Traces（SWE-bench 领域）

**SWE-agent Trajectories**（推荐）
- 80k+ trajectories
- 已解决任务：平均 31 步，未解决：平均 58 步
- 约 54% 的 trace 有对应的 Docker 镜像

```bash
# 下载（单个 shard 约 6670 条）
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
  repo_id='nebius/SWE-agent-trajectories',
  repo_type='dataset',
  filename='data/train-00000-of-00012.parquet'
)
"
```

**OpenHands Trajectories**
- 67k+ agent trajectories
- 包含：bash 命令、文件编辑、输出
- 平均 64 轮，最多 100 轮

### 2.2 Trace 数据结构

**SWE-agent trajectory 格式**：
```json
{
  "instance_id": "repo__issue-123",
  "model_name": "swe-agent-llama-70b",
  "exit_status": "submitted",
  "trajectory": [
    {"role": "system", "text": "..."},
    {"role": "user", "text": "issue description..."},
    {"role": "ai", "text": "```\npip install pytest\n```"}
  ]
}
```

**OpenHands trajectory 格式**：
```json
{
  "trajectory": [
    {"role": "system", "content": "..."},
    {"role": "assistant", "tool_calls": [
      {"function": {"name": "bash", "arguments": "{\"command\": \"pip install pytest\"}"}}
    ]}
  ]
}
```

### 2.3 重要发现：Trace 不包含时间信息

**经过验证，现有的 trace 数据集都不包含时间信息：**

| 数据集 | 包含字段 | 有 LLM 调用时间？ | 有 Tool 执行时间？ |
|--------|----------|-------------------|-------------------|
| SWE-agent | `role`, `text`, `mask` | ❌ | ❌ |
| OpenHands | `role`, `content`, `tool_calls` | ❌ | ❌ |

**这意味着**：
- 无法区分 LLM 推理时间 vs Tool 执行时间
- 无法获得原始执行的 latency 分布
- 回放时需要自己测量 Tool 执行时间

**对实验的影响**：
- 我们关注的是 **Tool 执行时的资源消耗**，而非 LLM 推理
- 回放时测量的 latency 是真实的 Tool 执行时间
- 如果需要模拟 LLM 思考间隔，需要额外添加 sleep

### 2.4 Docker 镜像可用性

SWE-rebench 提供预构建的 Docker 镜像：

| 统计 | 数值 |
|------|------|
| 总 instances | ~21,000 |
| 有 Docker 镜像的 | ~3,400 (16%) |
| 镜像格式 | `swerebench/sweb.eval.x86_64.<repo>-<issue>` |

```bash
# 查询某个 instance 的镜像
python -c "
import pandas as pd
from huggingface_hub import hf_hub_download
path = hf_hub_download('nebius/SWE-rebench', 'dataset', 'data/test-00000-of-00002.parquet')
df = pd.read_parquet(path)
print(df[df['instance_id'] == 'xxx']['docker_image'].values)
"

# 拉取镜像
docker pull docker.io/swerebench/sweb.eval.x86_64.hugovk_1776_pypistats-41
```

## 3. Trace 回放架构

### 3.1 统一 Trace IR

将所有 trace 源转换为统一的中间表示：

```json
{
  "trace_id": "hugovk__pypistats-41",
  "source": "sweagent",
  "model": "swe-agent-llama-70b",
  "exit_status": "submitted",
  "steps": [
    {
      "step_id": 0,
      "tool": "bash",
      "command": "pypistats python_minor pylast -m 2018-12",
      "timeout_ms": 60000
    },
    {
      "step_id": 1,
      "tool": "swe_agent_editor",
      "command": "open pypistats/cli.py 246",
      "timeout_ms": 60000
    },
    {
      "step_id": 2,
      "tool": "swe_agent_editor",
      "command": "edit 242:247\ndef _month(yyyy_mm):\n...\nend_of_edit",
      "timeout_ms": 60000
    }
  ]
}
```

### 3.2 已实现的脚本

| 脚本 | 功能 |
|------|------|
| `scripts/convert_sweagent_trace.py` | SWE-agent parquet → Trace IR |
| `scripts/analyze_traces.py` | 分析 trace 统计，找出有 Docker 镜像的 |
| `scripts/run_trace_in_container.py` | 在容器内回放 trace |

**回放脚本功能**：
- 自动拉取 Docker 镜像
- 执行 bash 命令和 SWE-agent editor 命令（open, edit, goto, submit 等）
- 收集每步的 latency_ms
- 运行后可选删除镜像释放空间

```bash
# 使用方法
python scripts/run_trace_in_container.py \
    data/sample_traces/hugovk__pypistats-41.json \
    docker.io/swerebench/sweb.eval.x86_64.hugovk_1776_pypistats-41 \
    --cleanup  # 运行后删除镜像
```

### 3.3 Editor 命令实现

SWE-agent 使用特殊的编辑器命令，我们的回放器实现了以下命令：

| 命令 | 说明 | 实现方式 |
|------|------|----------|
| `create <file>` | 创建文件 | `touch` |
| `open <file> [line]` | 打开文件 | `sed -n` 显示内容 |
| `goto <line>` | 跳转到行 | 更新内部状态 |
| `scroll_down/up` | 滚动 | 更新行号 |
| `edit <range>\n<content>\nend_of_edit` | 编辑文件 | Python 脚本替换行 |
| `search_file <pattern>` | 搜索文件 | `grep -n` |
| `submit` | 提交 | 无操作，标记完成 |

## 4. 实验设计

### 实验 1：Domain Mismatch 验证

**目标**：验证细粒度 tool-call 级别控制 vs 粗粒度环境级别控制的差异

**Trace 选择**：
- 从有 Docker 镜像的 trace 中选择 50 条
- 混合短（10-20 步）和长（50+ 步）trace
- 包含多样的 tool call（install, build, test, grep, edit）

**回放配置**：

| 配置 | Cgroup 结构 | 策略 |
|------|-------------|------|
| Static-Env | 整个 trace 一个 cgroup | 固定限制 |
| Static-ToolCall | 每步一个子 cgroup | 固定每步限制 |
| AgentCgroup | 每步一个子 cgroup | 动态 eBPF 策略 |

**收集的指标**：
- Wall-clock latency
- CPU 时间（user + sys）
- 最大 RSS
- memory.high 突破次数
- memory.max 突破次数
- OOM kills

### 实验 2：Timescale Mismatch 验证

**目标**：验证 in-kernel 控制 vs user-space 控制的响应时间差异

**Trace 选择**：筛选"突发性"trace
- 包含已知高资源使用的步骤（pytest, make, npm install）

**回放配置**：

| 配置 | 控制器 | 响应时间 |
|------|--------|----------|
| User-space | 每 50ms 轮询 PSI，写 cgroup 文件 | 50-100ms |
| In-kernel | eBPF hooks（sched_ext, memcg_bpf_ops） | <1ms |

**指标**：
- 从 memory.high 突破到 throttle 生效的时间
- 对共存 workload 的干扰（测量尾部延迟）

### 实验 3：多租户隔离

**Trace 选择**：
- 租户 A：SWE-agent trace（bash 密集，编译/测试）
- 租户 B：另一个 SWE-agent trace
- 租户 C：合成的 noisy neighbor（fork bomb / malloc stress）

**方法**：同时运行三个租户，测量跨租户干扰

**指标**：
- 每租户的步骤延迟分布
- 有/无 noisy neighbor 时的延迟变化

### 实验 4：开销测量

**目标**：测量 AgentCgroup 本身的开销

**方法**：
1. 无资源控制运行同一 trace（baseline）
2. 使用静态 cgroup 运行（最小开销）
3. 使用 AgentCgroup 运行（测量额外开销）

## 5. Trace 选择指南

### 代表性工作负载

选择覆盖以下类型的 trace：

| 类别 | 特征 | 示例命令 |
|------|------|----------|
| CPU 密集 | 编译、测试 | `make -j4`, `pytest` |
| 内存密集 | 大数据处理 | `npm install`, 数据加载 |
| IO 密集 | 文件操作 | `git clone`, `find`, `grep -r` |
| 突发性 | 短暂的高峰 | 快速 pytest, 浏览器点击 |
| 长时间运行 | 持续负载 | 完整测试套件 |

### 推荐的 Trace 子集

| 用途 | 数量 | 说明 |
|------|------|------|
| 快速冒烟测试 | 5 条 | 每条约 20 步 |
| 开发迭代 | 20 条 | 混合复杂度 |
| 论文主要结果 | 50-100 条 | 分层采样 |
| 完整评估 | 全部 | 可选 |

## 6. 输出格式

### 每步指标 (metrics.jsonl)

```json
{"trace_id": "t1", "step_id": 0, "tool": "bash", "cmd": "pip install", "latency_ms": 5234, "exit_code": 0, "success": true}
{"trace_id": "t1", "step_id": 1, "tool": "swe_agent_editor", "cmd": "edit 1:10...", "latency_ms": 800, "exit_code": 0, "success": true}
{"trace_id": "t1", "step_id": 2, "tool": "bash", "cmd": "pytest", "latency_ms": 12456, "exit_code": 1, "success": false}
```

### 汇总统计 (summary.json)

```json
{
  "config": "agentcgroup",
  "traces": 50,
  "total_steps": 2340,
  "latency_p50_ms": 1234,
  "latency_p95_ms": 8765,
  "latency_p99_ms": 15432,
  "success_rate": 0.85,
  "bash_steps": 1200,
  "editor_steps": 1140
}
```

## 7. 已知限制和讨论

### 7.1 Trace 数据的局限性

1. **无时间信息**：原始 trace 不包含 LLM 调用时间或 tool 执行时间
2. **部分镜像缺失**：只有约 16% 的 instance 有预构建 Docker 镜像
3. **编辑可能失败**：editor 命令的回放可能因为行号变化而失败

### 7.2 回放 vs 真实执行的差异

| 方面 | 真实执行 | Trace 回放 |
|------|----------|------------|
| LLM 调用 | 有，带延迟 | 无 |
| 决策逻辑 | 动态 | 固定序列 |
| Tool 执行 | 真实 | 真实 |
| 资源消耗 | 真实 | 真实 |
| 确定性 | 低 | 高 |

**结论**：对于资源控制实验，trace 回放是合适的方法，因为我们关注的是 tool 执行时的资源消耗，而非 LLM 的决策过程。

### 7.3 模拟 LLM 思考时间（可选）

如果需要更真实地模拟 agent 行为，可以在步骤之间添加延迟：

```python
# 在回放时添加模拟的 LLM 思考时间
import random
import time

for step in trace["steps"]:
    # 模拟 LLM 思考时间（1-5 秒）
    think_time = random.uniform(1.0, 5.0)
    time.sleep(think_time)

    # 执行 tool call
    execute_step(step)
```

## 8. 参考资料

### Trace 数据源
- SWE-agent trajectories: https://huggingface.co/datasets/nebius/SWE-agent-trajectories
- OpenHands trajectories: https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories
- SWE-rebench (Docker 镜像): https://huggingface.co/datasets/nebius/SWE-rebench

### 环境
- Docker 镜像格式: `docker.io/swerebench/sweb.eval.x86_64.<repo>-<issue>`
- 容器内工作目录: `/testbed`
- Conda 环境: `testbed`

### 内核特性
- sched_ext: https://docs.kernel.org/scheduler/sched-ext.html
- memcg_bpf_ops: https://lwn.net/Articles/1055698/
