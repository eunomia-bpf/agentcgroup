# memcg BPF 实验理论分析与论文 Claim 设计

## 1. 当前 Replay 方案的局限性

### 1.1 内存使用来源分析

原始实验（Claude Code 运行）的内存组成：
```
总内存 = Claude Code 进程 (~150-200MB) + 工具执行内存 (变化)
```

Replay 实验的内存组成：
```
总内存 = 工具执行内存 (变化)
```

**问题**：Replay 时没有 Claude Code 进程，内存基线大幅降低。

| 实验 | 内存 avg | 内存 max | 说明 |
|------|---------|---------|------|
| 原始 dask | 198 MB | 321 MB | 包含 Claude Code |
| Replay dask | 8 MB | 78 MB | 仅工具执行 |

### 1.2 并发 Replay 的问题

当前方案：3 个独立容器各自 replay
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Container 1 │  │ Container 2 │  │ Container 3 │
│ (replay A)  │  │ (replay B)  │  │ (replay C)  │
└─────────────┘  └─────────────┘  └─────────────┘
      ↓                ↓                ↓
   独立内存          独立内存          独立内存
   (~80MB)          (~80MB)          (~80MB)
```

**问题**：
1. 各容器内存独立，没有真正的内存竞争
2. 即使设置 cgroup 内存限制，80MB 的峰值远低于任何合理限制
3. 无法触发 memcg BPF 的 `get_high_delay_ms` 限流机制

## 2. 需要什么样的实验才能支持论文 Claim？

### 2.1 核心 Claim 分析

| Claim | 需要的证据 | 当前方案能否提供 |
|-------|-----------|-----------------|
| C1: memcg BPF 可实现优先级隔离 | 高优先级任务不受低优先级影响 | ❌ 需要内存竞争 |
| C2: 比静态限制更灵活 | 动态调整 vs 固定限制的对比 | ❌ 需要突发场景 |
| C3: 减少 OOM 事件 | OOM 计数对比 | ❌ 需要内存压力 |
| C4: 适合 agent 工作负载 | agent 特有的突发模式得到处理 | ⚠️ 需要真实负载 |

### 2.2 实验设计要求

要支持上述 Claim，实验必须满足：

1. **内存竞争场景**
   ```
   总内存限制 < Σ(各任务峰值内存)
   ```
   例如：3 个任务各 500MB 峰值，总限制 1GB

2. **可测量的差异**
   - 完成时间差异 > 10%（统计显著）
   - OOM 事件数差异明显

3. **可复现性**
   - 相同 trace 多次运行结果一致

## 3. 两种可行的实验路径

### 3.1 路径 A：合成内存压力工作负载

**思路**：不 replay 工具命令，而是根据 resources.json 的内存时序分配内存。

```python
# 伪代码
def replay_memory_pattern(resources_json):
    for sample in resources_json['samples']:
        target_mem = parse_mem_usage(sample['mem_usage'])
        current_mem = get_current_allocation()

        if target_mem > current_mem:
            # 分配内存到目标值
            allocate(target_mem - current_mem)
        else:
            # 释放内存到目标值
            free(current_mem - target_mem)

        sleep(1)  # 1秒采样间隔
```

**优点**：
- 可以精确复现原始内存模式
- 可以在任何环境运行（不需要 Docker 镜像）
- 可以创造真实的内存竞争

**缺点**：
- 不是"真实"的 agent 工作负载
- 可能被质疑为合成负载

### 3.2 路径 B：真实 Agent 运行 + 内存限制

**思路**：真正运行多个 Claude Code 实例，但施加内存限制。

```bash
# 在受限 cgroup 中运行真实 agent
sudo cgcreate -g memory:agent_exp
echo "2G" | sudo tee /sys/fs/cgroup/agent_exp/memory.max

# 运行多个 agent session
claude --task "task1" &  # 进入 agent_exp/session1
claude --task "task2" &  # 进入 agent_exp/session2
claude --task "task3" &  # 进入 agent_exp/session3
```

**优点**：
- 完全真实的工作负载
- 最有说服力

**缺点**：
- 成本高（API 调用费用）
- 不可完全复现（LLM 输出有随机性）
- 需要多次运行取统计

### 3.3 路径 C：混合方案（推荐）

**思路**：Replay 工具命令 + 模拟 Claude Code 内存占用

```python
def replay_with_simulated_claude(trace):
    # 1. 分配基线内存模拟 Claude Code (~200MB)
    baseline = allocate_baseline(200 * MB)

    # 2. Replay 工具命令（真实执行）
    for tool_call in trace.tool_calls:
        wait_for_timing(tool_call.timestamp)
        execute_tool(tool_call)

    # 3. 可选：根据 resources.json 调整内存
    # 这样既有真实工具执行，又有合理的内存占用
```

## 4. 定量分析：需要多少内存才能触发 BPF？

### 4.1 BPF 触发条件

从 memcg_ops.bpf.c 分析：
```c
SEC("struct_ops/get_high_delay_ms")
unsigned int get_high_delay_ms_impl(struct mem_cgroup *memcg)
{
    if (local_config.over_high_ms && need_threshold())
        return local_config.over_high_ms;  // 返回延迟（如 2000ms）
    return 0;
}
```

触发条件：
1. 内存使用超过 `memory.high` 阈值
2. `need_threshold()` 返回 true（检查是否需要限流）

### 4.2 实验参数设计

假设 3 个并发任务：

| 参数 | 设置 | 说明 |
|------|------|------|
| 总内存 (memory.max) | 1.5 GB | 产生竞争 |
| HIGH session memory.high | 800 MB | 较宽松 |
| LOW session memory.high | 400 MB | 较紧 |
| 预期 HIGH 峰值 | ~500 MB | 不触发限流 |
| 预期 LOW 峰值 | ~500 MB | 触发限流 |

### 4.3 预期结果计算

**场景：3 个任务竞争 1.5GB 内存**

无 BPF 控制：
```
时间线：
t=0:    A=200MB, B=200MB, C=200MB  (总=600MB, OK)
t=30:   A=500MB, B=400MB, C=300MB  (总=1.2GB, OK)
t=60:   A=500MB, B=500MB, C=500MB  (总=1.5GB, 临界)
t=90:   A 尝试分配更多 → 所有任务竞争 → 随机减速或 OOM
```

有 BPF 控制（A=HIGH, B,C=LOW）：
```
时间线：
t=0:    A=200MB, B=200MB, C=200MB  (总=600MB, OK)
t=30:   A=500MB, B=400MB, C=300MB  (总=1.2GB, OK)
t=60:   B,C 超过 memory.high → BPF 延迟 2s
        A 继续正常运行
t=90:   A 完成，B,C 继续（被限流）
```

**预期差异**：
- HIGH 任务完成时间：BPF 组 ≈ 基准，无 BPF 组 > 1.5x 基准
- OOM 事件：BPF 组 < 无 BPF 组

## 5. 论文 Claim 建议

### 5.1 强 Claim（需要充分实验支持）

> "AgentCgroup's memcg BPF struct_ops mechanism reduces p99 latency of high-priority agent sessions by X% compared to static memory limits, while maintaining Y% higher overall throughput."

需要的实验：
- 多次运行，计算 p99
- 与静态限制对比
- 吞吐量测量

### 5.2 中等 Claim（较易支持）

> "We demonstrate that memcg BPF struct_ops can effectively protect high-priority agent sessions from memory pressure caused by concurrent low-priority sessions, achieving priority isolation at the kernel level."

需要的实验：
- 高/低优先级任务并发
- 完成时间对比
- 定性说明保护效果

### 5.3 保守 Claim（最易支持）

> "We implement and validate a prototype using Linux's new memcg BPF struct_ops interface, showing its potential for fine-grained memory control in multi-tenant agent workloads."

需要的实验：
- 功能验证（test_progs 通过）
- 简单的优先级演示
- 讨论潜在应用

## 6. 建议的实验计划

### Phase 1: 合成负载验证（1-2天）

目标：验证 memcg BPF 机制在内存竞争场景下有效

```python
# memory_stress.py - 合成内存压力工具
def run_memory_stress(cgroup_path, target_mb, duration_s):
    """在指定 cgroup 中分配目标内存"""
    # 加入 cgroup
    write_pid_to_cgroup(cgroup_path)

    # 分配内存
    buf = bytearray(target_mb * 1024 * 1024)
    for i in range(0, len(buf), 4096):
        buf[i] = 1  # 触发物理分配

    # 保持
    time.sleep(duration_s)
```

实验设计：
- 3 个进程，各目标 500MB
- 总限制 1.2GB
- 比较 BPF vs 无 BPF

### Phase 2: Trace-based 内存模式（2-3天）

目标：使用真实 agent trace 的内存模式

```python
# trace_memory_replay.py
def replay_memory_trace(resources_json, speed=1.0):
    """根据 trace 分配/释放内存"""
    samples = load_trace(resources_json)

    for sample in samples:
        target = parse_mem(sample['mem_usage'])
        adjust_memory(target)
        time.sleep(1.0 / speed)
```

### Phase 3: 完整 Replay + 模拟基线（3-5天）

目标：Replay 工具命令 + 模拟 Claude Code 内存

最终实验产出论文结果

## 7. 结论

### 当前状态

- ✅ memcg BPF struct_ops 已验证（test_progs 通过）
- ✅ 基本 replay 框架已实现
- ❌ 当前 replay 无法产生有效内存竞争
- ❌ 无法支持论文 claim

### 建议的下一步

1. **立即**：实现 Phase 1 合成负载测试，验证 BPF 在竞争场景下有效
2. **短期**：实现 Phase 2 trace-based 内存模式回放
3. **中期**：完成 Phase 3，产出论文可用数据

### 论文建议

采用"中等 Claim"策略：
- 展示 memcg BPF struct_ops 在 agent 场景的应用
- 定量展示优先级隔离效果
- 讨论局限性和未来工作
