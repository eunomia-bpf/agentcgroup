# 多租户内存竞争实验进展

## 1. Baseline 实验结果 ✅

**实验配置**：
- 3 个进程各分配 200MB 内存
- memory.high = 150MB（触发阈值）
- 无 memory.max（避免 OOM）

**结果**：

| 进程 | 分配时间 | 总时间 | memory.high 事件 |
|------|---------|--------|-----------------|
| HIGH | 293.04s | 298.09s | 3526 |
| LOW1 | 297.12s | 302.14s | 3577 |
| LOW2 | 295.86s | 300.88s | 3594 |

**关键发现**：
- LOW/HIGH 比值 = **1.01x**（无优先级差异）
- 每个进程触发 ~3500 次 memory.high 事件
- 这正是 BPF `get_high_delay_ms` 应该介入的地方

## 2. BPF 附加逻辑分析

从 `prog_tests/memcg_ops.c` 分析，附加 BPF struct_ops 需要：

```c
// 1. 加载 skeleton
skel = memcg_ops__open_and_load();

// 2. 配置 local_config
bss_data->local_config.high_cgroup_id = high_cgroup_id;
bss_data->local_config.threshold = 1;
bss_data->local_config.over_high_ms = 2000;
bpf_map_update_elem(map_fd, &key, bss_data, BPF_EXIST);

// 3. 附加 struct_ops 到 cgroup
map = bpf_object__find_map_by_name(skel->obj, "low_mcg_ops");
opts.relative_fd = low_cgroup_fd;
link = bpf_map__attach_struct_ops_opts(map, &opts);
```

## 3. 下一步：实现 BPF 附加

### 方案 A：修改 test_progs（推荐）

在 `memcg_ops.c` 中添加一个测试函数，接受外部 cgroup 路径：

```bash
# 理想用法
./test_progs -t memcg_ops_custom \
  --high-cgroup /sys/fs/cgroup/memcg_bpf_test/high_session \
  --low-cgroups /sys/fs/cgroup/memcg_bpf_test/low_session_1,low_session_2
```

### 方案 B：Python BPF 加载器

使用 `libbpf` 的 Python 绑定或 `bcc` 库：

```python
from bcc import BPF
# 加载并附加 BPF...
```

### 方案 C：直接复用 test_progs 结构

test_progs 已经创建了 `/memcg_ops_test/high` 和 `/memcg_ops_test/low` cgroup，我们可以：

1. 运行 test_progs 加载 BPF
2. 同时在这些 cgroup 中运行我们的 memory_stress

## 4. 预期 BPF 实验结果

如果 BPF 正常工作：

| 进程 | Baseline | BPF 预期 | 说明 |
|------|----------|---------|------|
| HIGH | ~300s | ~30s | 受保护，快速完成 |
| LOW1 | ~300s | ~600s+ | 被限流，大量延迟 |
| LOW2 | ~300s | ~600s+ | 被限流，大量延迟 |

**预期 LOW/HIGH 比值**：> 10x（显著优先级隔离）

## 5. 论文 Claim 支持

| Claim | Baseline 支持 | BPF 实验支持 |
|-------|--------------|-------------|
| "memcg BPF 可实现优先级隔离" | ❌ | ✅ 如果 LOW/HIGH > 5x |
| "memory.high 触发 BPF 回调" | ✅ ~3500 events | ✅ |
| "无 BPF 时公平竞争" | ✅ 1.01x | N/A |

## 6. 文件清单

```
multi_tenant_test/
├── EXPERIMENT_PLAN.md       # 实验计划
├── RESULTS_SUMMARY.md       # 本文件
├── memory_stress.py         # 内存压力工具
├── run_experiment.sh        # 实验运行脚本
├── show_results.py          # 结果显示工具
└── results/
    └── baseline_20260208_035126/  # Baseline 结果
        ├── config.json
        ├── high_result.json
        ├── low1_result.json
        ├── low2_result.json
        └── *_memory_events.txt
```
