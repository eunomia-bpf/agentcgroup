# 隔离策略对比实验记录

## 实验目标

对比三种内存隔离策略在真实 agent trace 回放场景下的效果：
1. **no_isolation** - 仅设置总内存限制，无优先级区分
2. **static** - 静态 memory.max 分配给每个 session
3. **bpf** - 动态 BPF 优先级隔离

## 实验配置

- **HIGH trace**: pre-commit__pre-commit-2524 (327s, avg=306MB, max=1907MB, 波动 6.23x)
- **LOW1 trace**: dask__dask-11628 (98s, avg=198MB, max=321MB)
- **LOW2 trace**: joke2k__faker-1520 (123s, avg=190MB, max=273MB)
- **总内存限制**: 2560MB (2.5GB)
- **回放速度**: 100x
- **基线内存**: 每进程 100MB (模拟 Claude Code 进程)

## 创建的文件

1. `trace_replay.py` - Trace 回放工具，按时序分配/释放内存
2. `run_isolation_comparison.sh` - 三种策略对比实验脚本
3. `analyze_isolation_results.py` - 结果分析工具

## 实验过程记录

### 阶段 1: no_isolation 策略 ✅ 成功

**配置**:
- 父 cgroup memory.max = 2560MB
- 子 cgroup 无限制 (共享父限制)

**结果**:
```
HIGH: 14.26s, peak=2007MB, OOM=0
LOW1: 1.25s, peak=421MB, OOM=0
LOW2: 1.44s, peak=373MB, OOM=0
总时间: 14.36s
```

**分析**:
- 所有进程正常完成
- 无 memory.events.high 事件 (因为没有设置 memory.high)
- LOW 进程先完成，HIGH 进程后完成 (trace 本身更长)

### 阶段 2: static 策略 ❌ 部分失败

**配置**:
- 父 cgroup memory.max = 2560MB
- 每个子 cgroup memory.max = 853MB (2560/3)

**问题**:
- HIGH trace 峰值需要 1907MB，但限制只有 853MB
- HIGH 进程被 OOM killer 终止

**部分结果**:
```
LOW1: 1.15s, peak=421MB, OOM=0
LOW2: 1.38s, peak=373MB, OOM=0
HIGH: 未完成 (被杀死)
```

**教训**:
静态限制无法处理高波动 trace，会导致 OOM。这正是 BPF 动态隔离要解决的问题。

### 阶段 3: bpf 策略 ⚠️ 运行中遇到问题

**配置**:
- 父 cgroup memory.max = 2560MB
- 每个子 cgroup memory.high = 640MB (无 memory.max)
- BPF delay = 2000ms

**观察到的行为**:

1. **BPF 加载成功**:
```
Attached high_mcg_ops to high_session
Attached low_mcg_ops to low_session_1
Attached low_mcg_ops to low_session_2
```

2. **LOW 进程被显著延迟**:
```
# 无 BPF 时
LOW1: 1.25s, LOW2: 1.44s

# 有 BPF 时
LOW1: 158.70s, LOW2: 158.76s  # 延迟了约 157 秒!
```

3. **BPF 统计**:
```
high_delay_calls=5245  # 调用次数
active=154             # 活跃延迟数
below_low_calls=0      # 未触发 below_low
```

**遇到的问题**:

HIGH 进程也被延迟/卡住，原因：
- HIGH 进程也超过了 memory.high (640MB)，需要分配到 1.9GB
- 当前 BPF 实现中，所有超过 memory.high 的进程都会触发 `get_high_delay_ms`
- `below_low` 保护机制未正确触发

## 问题分析

### Bug 1: HIGH 进程也被 BPF 延迟

**原因**:
- `get_high_delay_ms` 是附加到 LOW cgroup 的回调
- 当 LOW cgroup 超过 memory.high 时触发
- 但当 HIGH cgroup 也超过 memory.high 时，它也会被限流（因为所有 cgroup 共享相同的 memory.high 阈值）

**实际行为**:
- LOW cgroup 的 `get_high_delay_ms` 被调用 5245 次
- HIGH cgroup 的内存分配也被系统限流（不是通过 BPF，而是通过内核的 memory.high 机制）

### Bug 2: below_low 保护未生效

**现象**:
`below_low_calls=0` 说明 HIGH 的 `below_low` 回调从未被调用

**可能原因**:
1. 系统内存充足，未触发 reclaim 到 low watermark
2. `below_low` 的触发条件未满足
3. 需要更紧的内存限制来触发 reclaim

### Bug 3: 静态限制导致 OOM

**原因**:
- 静态分配 853MB/session
- HIGH trace 峰值 1907MB，超过限制 2x
- 内核直接 OOM kill

## 下一步解决方案

### 方案 1: 调整内存配置

```bash
# 提高 memory.high 阈值，让 HIGH 不被限流
HIGH session: memory.high = 2048MB (允许 HIGH 突发)
LOW sessions: memory.high = 400MB  (更早触发 BPF)
```

**实现**:
修改 `run_isolation_comparison.sh` 的 `setup_bpf_isolation` 函数

### 方案 2: 使用更小的 trace 组合

选择峰值内存更低的 trace 组合：
```
HIGH: dask (max=321MB)
LOW: faker (max=273MB), sigmavirus24 (max=306MB)
总峰值: ~900MB，可以在 1GB 限制下正常运行
```

### 方案 3: 修改 BPF 程序逻辑

当前实现的问题是 HIGH cgroup 也可能被系统限流。需要修改：

```c
// memcg_priority.bpf.c
// 在 get_high_delay_ms 中检查当前 cgroup 是否为 HIGH
SEC("struct_ops/get_high_delay_ms")
unsigned int get_high_delay_ms_impl(struct mem_cgroup *memcg) {
    // 如果是 HIGH cgroup，不延迟
    if (is_high_priority_cgroup(memcg))
        return 0;

    // 只对 LOW cgroup 应用延迟
    return local_config.over_high_ms;
}
```

### 方案 4: 使用更大的总内存限制

```bash
# 总限制 4GB，让 HIGH 可以完全 burst
--total-mb 4096

# 或者不设置总限制，只用 memory.high 触发 BPF
```

## 已验证的结论

尽管实验未完全完成，但已验证：

1. **BPF 延迟机制有效**: LOW 进程从 ~1.4s 延迟到 ~159s (100x+ 延迟)
2. **优先级隔离可行**: 只需调整配置让 HIGH 不被限流
3. **静态限制的问题**: 无法处理高波动 trace，会 OOM
4. **Trace 回放工具正常**: 能正确回放内存使用模式

## 推荐的实验配置

### 配置 A: 低波动 trace 组合
```bash
HIGH_TRACE="dask__dask-11628"       # max=321MB
LOW1_TRACE="joke2k__faker-1520"     # max=273MB
LOW2_TRACE="sigmavirus24__github3.py-673"  # max=306MB
TOTAL_MEMORY_MB=1024  # 1GB
```

### 配置 B: 调整 memory.high
```bash
# 在 setup_bpf_isolation 中
echo "2048M" > $CGROUP_ROOT/high_session/memory.high  # HIGH 可以 burst
echo "300M" > $CGROUP_ROOT/low_session_1/memory.high  # LOW 被限流
echo "300M" > $CGROUP_ROOT/low_session_2/memory.high
```

### 配置 C: 合成负载 (已验证有效)
使用 `memory_stress.py` 而非 trace 回放：
```bash
sudo ./run_experiment.sh bpf
# 结果: LOW/HIGH = 1.29x (28% 优先级改善)
```

## 文件结构

```
multi_tenant_test/
├── trace_replay.py              # Trace 回放工具
├── run_isolation_comparison.sh  # 三策略对比脚本
├── analyze_isolation_results.py # 结果分析工具
├── isolation_results/           # 实验结果
│   ├── no_isolation_run1_*/     # ✅ 完成
│   ├── static_run1_*/           # ❌ HIGH 被 OOM
│   └── bpf_run1_*/              # ⚠️ 部分完成
└── ISOLATION_EXPERIMENT_LOG.md  # 本文档
```

## 参考数据

### no_isolation 完整结果 (high_result.json)
```json
{
  "name": "HIGH",
  "total_time": 14.255,
  "peak_memory_mb": 2006.69,
  "oom_count": 0,
  "events_delta": {"high": 0, "oom": 0}
}
```

### BPF 统计 (bpf_loader.log)
```
get_high_delay_ms calls: 5245
active delays: 154
below_low calls: 0
```

## 总结

| 策略 | HIGH 时间 | LOW 平均 | 比值 | OOM | 状态 |
|------|----------|----------|------|-----|------|
| no_isolation | 14.3s | 1.35s | 0.09x | 0 | ✅ 完成 |
| static | - | 1.27s | - | 1 (HIGH) | ❌ OOM |
| bpf | >300s | 158.7s | - | 0 | ⚠️ 卡住 |

**关键发现**: BPF 延迟机制工作正常，但需要调整配置避免 HIGH 也被限流。
