# AgentSight 增强计划：扩展 process tracer

## 背景

Branch context 论文需要证明 AI agent 的探索路径会产生**多维状态副作用**（文件系统、网络、进程、环境），而现有机制无法统一回滚。AgentSight 现有的 `process` tracer 只追踪了文件打开和进程生命周期，缺少文件删除/重命名、网络端口、进程组变更等关键数据。

**方案**：直接扩展现有 `process` tracer，通过特性开关（feature flags）按需启用新追踪能力。保持 CLI 接口和 JSON 输出格式向后兼容。

## 为什么合并而非独立 tracer

| | 合并到 process | 独立 workspace tracer |
|---|---|---|
| PID 过滤 | ✅ 复用现有 pid_tracker，一处过滤 | ❌ 需要重新实现一套过滤 |
| Ring buffer | ✅ 单个 buffer，事件有序 | ❌ 两个 buffer，需要在用户空间合并排序 |
| 进程管理 | ✅ 一个进程 | ❌ 两个进程需协调启停 |
| 代码复用 | ✅ 复用 event 结构、dedup、rate limit | ❌ 重复实现 |
| 向后兼容 | ✅ 默认不启用新特性=完全相同行为 | ✅ 独立二进制不影响 |

## 当前 process tracer 的能力

```
现有事件类型：
├── EVENT_TYPE_PROCESS (EXEC/EXIT)     ← tp/sched/sched_process_exec, sched_process_exit
├── EVENT_TYPE_BASH_READLINE           ← uretprobe//usr/bin/bash:readline
└── EVENT_TYPE_FILE_OPERATION          ← tp/syscalls/sys_enter_openat, sys_enter_open
```

## 核心设计原则：全部追踪，统一内核聚合

**所有新增事件统一走 BPF hash map 内核聚合**，不经过 ring buffer。

- Ring buffer **只给现有事件**（EXEC/EXIT/FILE_OPEN/BASH_READLINE）——已验证稳定
- 所有新增事件在内核侧按合适的 key 粒度聚合计数
- 用户空间定时（每 N 秒）遍历 map → flush 为 summary JSON
- 低频事件（setsid、bind）自然 count=1，等价于逐条 report，不需要特殊路径

**一条路径，零例外。**

### 统一聚合 map 设计

所有新事件共用一个 map（或按类别分 2-3 个 map）：

```c
/*
 * 统一的聚合 key：按 (pid, event_type, detail) 分组
 * detail 的含义随 event_type 变化：
 *   FS mutations:  dir_prefix（父目录路径前缀）
 *   Write:         fd（文件描述符）
 *   Network:       addr:port（远端/本地地址+端口）
 *   Signals:       target_pid + signal（目标进程+信号）
 *   Pgrp/Session:  new_pgid / new_sid
 *   Fork:          （空，按 pid 聚合子进程数量）
 *   Mmap:          fd
 *   Chdir:         path
 */

#define DETAIL_LEN 64

struct agg_key {
    u32 pid;
    u32 event_type;
    char detail[DETAIL_LEN];    // 语义随 event_type 变化
};

struct agg_value {
    u64 count;
    u64 total_bytes;            // 仅 write/mmap 使用，其余为 0
    u64 first_ts;
    u64 last_ts;
    char extra[MAX_FILENAME_LEN];  // 最后一次操作的额外信息（完整路径/new_path 等）
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, struct agg_key);
    __type(value, struct agg_value);
} event_agg_map SEC(".maps");
```

### 各事件类型的 key 设计

| 事件 | event_type | detail 内容 | extra 内容 | 典型 count |
|------|-----------|-------------|-----------|-----------|
| FILE_DELETE | 10 | dir_prefix（`/testbed/venv/lib`） | last_path | 高（rm -rf） |
| FILE_RENAME | 11 | new_path 的 dir_prefix | old_path:new_path | 高（pip install） |
| DIR_CREATE | 12 | parent_dir | last_dirpath | 高（pip install） |
| FILE_TRUNCATE | 13 | fd（sprintf） | — | 低 |
| CHDIR | 14 | path | — | 极低 |
| WRITE | 15 | fd（sprintf） | — | 极高 |
| NET_BIND | 20 | `addr:port`（`0.0.0.0:8080`） | — | 极低（1-2） |
| NET_LISTEN | 21 | fd（sprintf） | — | 极低 |
| NET_CONNECT | 22 | `addr:port`（`api.anthropic.com:443`） | — | 中（API 调用） |
| PGRP_CHANGE | 30 | `pid:pgid`（`1234:5678`） | — | 极低 |
| SESSION_CREATE | 31 | `sid`（`1234`） | — | 极低 |
| SIGNAL_SEND | 32 | `target:sig`（`5678:9`） | — | 低 |
| PROC_FORK | 33 | 空或 `child_pid` | — | 中 |
| MMAP_SHARED | 40 | fd（sprintf） | — | 中 |

### Flush 输出格式

用户空间每 5 秒遍历 map，对每个非零条目输出一行 JSON：

```jsonl
// 高频事件 → 高 count，按目录聚合
{"timestamp":260,"event":"FS_SUMMARY","pid":1234,"comm":"pip","type":"DIR_CREATE","detail":"/testbed/venv/lib/requests","count":47,"extra":"/testbed/venv/lib/requests/utils"}
{"timestamp":260,"event":"FS_SUMMARY","pid":1234,"comm":"pip","type":"FILE_RENAME","detail":"/testbed/venv/lib","count":203,"extra":"/testbed/venv/lib/urllib3/response.py"}
{"timestamp":260,"event":"WRITE_SUMMARY","pid":1234,"comm":"pip","detail":"fd=5","count":1847,"total_bytes":4521984,"filepath":"/testbed/venv/lib/requests/api.py"}

// 低频事件 → count=1，等价于逐条
{"timestamp":260,"event":"NET_SUMMARY","pid":5678,"comm":"python","type":"NET_BIND","detail":"0.0.0.0:8080","count":1}
{"timestamp":260,"event":"SIGNAL_SUMMARY","pid":1234,"comm":"kill","type":"SIGNAL_SEND","detail":"5678:9","count":1}
{"timestamp":260,"event":"PROC_SUMMARY","pid":1234,"comm":"bash","type":"SESSION_CREATE","detail":"sid=1234","count":1}
```

**关键**：消费者不需要关心 count 是 1 还是 1000——格式统一，逻辑统一。

## 新增事件类型

```
新增事件类型（全部走 BPF map 内核聚合，通过 feature flags 控制）：
│
├── --trace-fs（文件系统 mutations）
│   ├── FILE_DELETE                     ← tp/syscalls/sys_enter_unlinkat
│   ├── FILE_RENAME                     ← tp/syscalls/sys_enter_renameat2
│   ├── DIR_CREATE                      ← tp/syscalls/sys_enter_mkdirat
│   ├── WRITE (bytes+count)             ← tp/syscalls/sys_enter_write + sys_exit_write
│   ├── FILE_TRUNCATE                   ← tp/syscalls/sys_enter_ftruncate
│   └── CHDIR                           ← tp/syscalls/sys_enter_chdir
│
├── --trace-net（网络状态）
│   ├── NET_BIND                        ← tp/syscalls/sys_enter_bind
│   ├── NET_LISTEN                      ← tp/syscalls/sys_enter_listen
│   └── NET_CONNECT                     ← tp/syscalls/sys_enter_connect
│
├── --trace-signals（进程协调）
│   ├── PGRP_CHANGE                     ← tp/syscalls/sys_enter_setpgid
│   ├── SESSION_CREATE                  ← tp/syscalls/sys_enter_setsid
│   ├── SIGNAL_SEND                     ← tp/syscalls/sys_enter_kill
│   └── PROC_FORK                       ← tp/sched/sched_process_fork
│
└── --trace-mem（内存/共享状态）
    └── MMAP_SHARED                     ← tp/syscalls/sys_enter_mmap（内核侧过滤仅 MAP_SHARED）

所有事件统一路径：tracepoint → BPF map 计数 → 用户空间定时 flush → JSON summary
```

## 关键设计

### 1. 零开销特性开关

在 BPF 侧用 `const volatile` 变量控制，JIT 编译后禁用的分支直接被优化掉：

```c
// process.bpf.c 新增
const volatile bool trace_fs_mutations = false;   // --trace-fs
const volatile bool trace_network = false;         // --trace-net
const volatile bool trace_signals = false;         // --trace-signals
const volatile bool trace_memory = false;          // --trace-mem

SEC("tp/syscalls/sys_enter_unlinkat")
int trace_unlinkat(struct trace_event_raw_sys_enter *ctx)
{
    if (!trace_fs_mutations)       // ← JIT 优化为 nop
        return 0;
    // ... 正常处理
}
```

用户空间在 load skeleton 后、attach 前设置：

```c
// process.c
skel->rodata->trace_fs_mutations = env.trace_fs;
skel->rodata->trace_network = env.trace_net;
skel->rodata->trace_signals = env.trace_signals;
skel->rodata->trace_memory = env.trace_mem;
```

**结果**：不加新 flag → 行为完全不变、性能完全不变。

### 2. write() 的 BPF map 内核侧聚合

write/pwrite64 每秒可达数万次，不能逐条发送到 ring buffer。在 BPF 内核侧用 hash map 聚合：

```c
/* BPF 侧：内核中完成聚合，不经过 ring buffer */

struct write_key {
    u32 pid;
    int fd;
};

struct write_stats {
    u64 count;
    u64 total_bytes;
    u64 first_ts;
    u64 last_ts;
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, struct write_key);
    __type(value, struct write_stats);
} write_stats_map SEC(".maps");

/* 需要配对 enter/exit 来同时获得 fd（enter 参数）和实际写入字节数（exit 返回值） */

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, u64);  /* pid_tgid */
    __type(value, int); /* fd from enter */
} write_fd_map SEC(".maps");

SEC("tp/syscalls/sys_enter_write")
int trace_write_enter(struct trace_event_raw_sys_enter *ctx) {
    if (!trace_fs_mutations) return 0;
    u64 id = bpf_get_current_pid_tgid();
    int fd = (int)ctx->args[0];
    bpf_map_update_elem(&write_fd_map, &id, &fd, BPF_ANY);
    return 0;
}

SEC("tp/syscalls/sys_exit_write")
int trace_write_exit(struct trace_event_raw_sys_exit *ctx) {
    if (!trace_fs_mutations) return 0;
    long ret = ctx->ret;
    if (ret <= 0) return 0;

    u64 id = bpf_get_current_pid_tgid();
    int *fd_ptr = bpf_map_lookup_elem(&write_fd_map, &id);
    if (!fd_ptr) return 0;

    struct write_key key = {
        .pid = id >> 32,
        .fd = *fd_ptr,
    };
    bpf_map_delete_elem(&write_fd_map, &id);

    struct write_stats *stats = bpf_map_lookup_elem(&write_stats_map, &key);
    if (stats) {
        __sync_fetch_and_add(&stats->count, 1);
        __sync_fetch_and_add(&stats->total_bytes, (u64)ret);
        stats->last_ts = bpf_ktime_get_ns();
    } else {
        u64 now = bpf_ktime_get_ns();
        struct write_stats new_stats = { .count = 1, .total_bytes = ret,
                                          .first_ts = now, .last_ts = now };
        bpf_map_update_elem(&write_stats_map, &key, &new_stats, BPF_ANY);
    }
    return 0;
}
```

**用户空间 flush 策略**（process.c）：
- 每 5 秒遍历 `write_stats_map`，输出非零条目为 `WRITE_SUMMARY` 事件
- 在进程 EXIT 事件时，flush 该 pid 的所有 write stats
- fd→path 解析：读 `/proc/<pid>/fd/<fd>` 符号链接（进程存活时有效）

```jsonl
{"timestamp":500,"event":"WRITE_SUMMARY","pid":1234,"comm":"pip","fd":5,"filepath":"/testbed/venv/lib/requests/api.py","count":3,"total_bytes":12847,"duration_ms":4200}
{"timestamp":500,"event":"WRITE_SUMMARY","pid":1234,"comm":"pip","fd":7,"filepath":"/testbed/venv/lib/urllib3/response.py","count":1,"total_bytes":45210,"duration_ms":100}
```

### 3. 文件系统 mutations 的 BPF map 内核聚合

FILE_DELETE、FILE_RENAME、DIR_CREATE 同样在**内核侧用 BPF hash map 聚合**，不经过 ring buffer：

```c
/* BPF 侧：按 (pid, event_type, dir_prefix) 聚合 */

struct fs_mut_key {
    u32 pid;
    u32 event_type;           // FILE_DELETE / FILE_RENAME / DIR_CREATE
    char dir_prefix[64];      // 路径截取到父目录（减少 key 空间）
};

struct fs_mut_stats {
    u64 count;
    u64 first_ts;
    u64 last_ts;
    char last_path[MAX_FILENAME_LEN];  // 最后一个操作的完整路径（用于输出参考）
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, struct fs_mut_key);
    __type(value, struct fs_mut_stats);
} fs_mut_stats_map SEC(".maps");

SEC("tp/syscalls/sys_enter_unlinkat")
int trace_unlinkat(struct trace_event_raw_sys_enter *ctx) {
    if (!trace_fs_mutations) return 0;

    // 读取 filepath，提取父目录作为 dir_prefix
    char filepath[MAX_FILENAME_LEN];
    bpf_probe_read_user_str(filepath, sizeof(filepath), (const char *)ctx->args[1]);

    struct fs_mut_key key = { .pid = bpf_get_current_pid_tgid() >> 32, .event_type = EVENT_TYPE_FILE_DELETE };
    extract_dir_prefix(filepath, key.dir_prefix, sizeof(key.dir_prefix));

    struct fs_mut_stats *stats = bpf_map_lookup_elem(&fs_mut_stats_map, &key);
    if (stats) {
        __sync_fetch_and_add(&stats->count, 1);
        stats->last_ts = bpf_ktime_get_ns();
        bpf_probe_read_kernel_str(stats->last_path, sizeof(stats->last_path), filepath);
    } else {
        u64 now = bpf_ktime_get_ns();
        struct fs_mut_stats new_stats = { .count = 1, .first_ts = now, .last_ts = now };
        bpf_probe_read_kernel_str(new_stats.last_path, sizeof(new_stats.last_path), filepath);
        bpf_map_update_elem(&fs_mut_stats_map, &key, &new_stats, BPF_ANY);
    }
    return 0;
}
// renameat2, mkdirat 同理
```

用户空间定时 flush 输出：

```jsonl
{"timestamp":260,"event":"FS_MUT_SUMMARY","pid":1234,"comm":"pip","type":"DIR_CREATE","dir_prefix":"/testbed/venv/lib/requests","count":47,"last_path":"/testbed/venv/lib/requests/utils","duration_ms":4200}
{"timestamp":260,"event":"FS_MUT_SUMMARY","pid":1234,"comm":"pip","type":"FILE_RENAME","dir_prefix":"/testbed/venv/lib","count":203,"last_path":"/testbed/venv/lib/urllib3/response.py","duration_ms":5100}
{"timestamp":260,"event":"FS_MUT_SUMMARY","pid":1234,"comm":"pip","type":"FILE_DELETE","dir_prefix":"/tmp/pip-build-xyz","count":89,"last_path":"/tmp/pip-build-xyz/setup.py","duration_ms":300}
```

**优点**（vs 用户空间 dedup）：
- `pip install` 创建 3000 个文件时，ring buffer 只需 0 条事件（全在 map 里），不会丢失
- 用户空间 dedup 在高频场景下可能来不及处理导致 ring buffer 溢出
- 内核聚合按目录前缀分组，直接得到"哪个目录变化最多"的分布信息

### 4. mmap(MAP_SHARED) 的内核侧过滤

mmap 调用频率高，但只有 `MAP_SHARED` 的才涉及跨进程共享状态。在 BPF 侧过滤：

```c
SEC("tp/syscalls/sys_enter_mmap")
int trace_mmap(struct trace_event_raw_sys_enter *ctx) {
    if (!trace_memory) return 0;
    int flags = (int)ctx->args[3];
    if (!(flags & 0x01))  // MAP_SHARED = 0x01
        return 0;         // 跳过 MAP_PRIVATE
    // ... 只报告 MAP_SHARED 的 mmap
}
```

## 数据结构

### struct event（不修改）

现有 `struct event` + ring buffer 只服务于 4 种现有事件（EXEC/EXIT/FILE_OPEN/BASH_READLINE），**不变**。

### agg_key + agg_value（新增，用于 BPF map 聚合）

所有新增事件共用一套聚合结构：

```c
// process.h 新增

#define DETAIL_LEN 64

/* 聚合 key：(pid, event_type, detail) */
struct agg_key {
    __u32 pid;
    __u32 event_type;          // EVENT_TYPE_FILE_DELETE, NET_BIND, etc.
    char detail[DETAIL_LEN];   // 语义随 event_type 变化（见上文 key 设计表）
};

/* 聚合 value：count + bytes + timestamps + extra info */
struct agg_value {
    __u64 count;
    __u64 total_bytes;         // 仅 write/mmap 使用
    __u64 first_ts;
    __u64 last_ts;
    char comm[TASK_COMM_LEN];  // 最后一次操作的进程名
    char extra[MAX_FILENAME_LEN]; // 最后一次操作的额外信息（完整路径等）
};

/* event_type 编号（新增部分） */
enum event_type {
    // ... 现有 0-2 不变 ...
    EVENT_TYPE_FILE_DELETE = 10,
    EVENT_TYPE_FILE_RENAME = 11,
    EVENT_TYPE_DIR_CREATE = 12,
    EVENT_TYPE_FILE_TRUNCATE = 13,
    EVENT_TYPE_CHDIR = 14,
    EVENT_TYPE_WRITE = 15,
    EVENT_TYPE_NET_BIND = 20,
    EVENT_TYPE_NET_LISTEN = 21,
    EVENT_TYPE_NET_CONNECT = 22,
    EVENT_TYPE_PGRP_CHANGE = 30,
    EVENT_TYPE_SESSION_CREATE = 31,
    EVENT_TYPE_SIGNAL_SEND = 32,
    EVENT_TYPE_PROC_FORK = 33,
    EVENT_TYPE_MMAP_SHARED = 40,
};
```

**BPF map 定义**（process.bpf.c）：

```c
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, struct agg_key);
    __type(value, struct agg_value);
} event_agg_map SEC(".maps");

/* write 专用：暂存 enter 阶段的 fd，exit 阶段查找 */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, __u64);    /* pid_tgid */
    __type(value, int);    /* fd */
} write_fd_map SEC(".maps");
```

## CLI 接口变更

### 现有（保持不变）

```
./process [-v] [-d MS] [-c COMMANDS] [-p PID] [-m MODE]
```

### 新增 flags（纯追加）

```
./process [...现有参数...] [--trace-fs] [--trace-net] [--trace-signals] [--trace-all]
```

| Flag | 启用的追踪 | 默认 |
|------|-----------|------|
| （无新 flag） | EXEC + EXIT + FILE_OPEN + BASH_READLINE | ✅ 与当前完全一致 |
| `--trace-fs` / `-F` | + FILE_DELETE, FILE_RENAME, DIR_CREATE, **WRITE_SUMMARY**, FILE_TRUNCATE, CHDIR | off |
| `--trace-net` / `-N` | + NET_BIND + NET_LISTEN + NET_CONNECT | off |
| `--trace-signals` / `-S` | + PGRP_CHANGE + SESSION_CREATE + SIGNAL_SEND + PROC_FORK | off |
| `--trace-mem` / `-M` | + MMAP_SHARED | off |
| `--trace-all` / `-A` | 以上全部 | off |

### Rust collector 侧

`ProcessRunner` 新增 `with_trace_features()` builder 方法，将对应 flags 转为 CLI 参数传给 BPF 二进制：

```rust
let runner = ProcessRunner::from_binary_extractor(path)
    .with_args(&["--trace-all"])    // 新增
    .with_args(&["-c", "python"]);
```

**不需要修改 Runner trait 或 Event 结构**——新事件类型自动通过 JSON → Event 流水线流入 analyzer chain。

## JSON 输出格式

### 现有事件（不变）

```jsonl
{"timestamp":123,"event":"EXEC","comm":"python","pid":1234,"ppid":1000,"filename":"/usr/bin/python3","full_command":"python3 -m pytest"}
{"timestamp":124,"event":"EXIT","comm":"python","pid":1234,"ppid":1000,"exit_code":0,"duration_ms":5000}
{"timestamp":125,"event":"FILE_OPEN","comm":"python","pid":1234,"count":1,"filepath":"/testbed/main.py","flags":0}
{"timestamp":126,"event":"BASH_READLINE","comm":"bash","pid":1000,"command":"pytest test.py"}
```

### 新增事件（统一 summary 格式，来自 BPF map flush）

```jsonl
// === 文件系统（高频 → 高 count） ===
{"timestamp":260,"event":"SUMMARY","pid":1234,"comm":"pip","type":"DIR_CREATE","detail":"/testbed/venv/lib/requests","count":47,"extra":"/testbed/venv/lib/requests/utils"}
{"timestamp":260,"event":"SUMMARY","pid":1234,"comm":"pip","type":"FILE_RENAME","detail":"/testbed/venv/lib","count":203,"extra":"/testbed/venv/lib/urllib3/response.py"}
{"timestamp":260,"event":"SUMMARY","pid":1234,"comm":"pip","type":"FILE_DELETE","detail":"/tmp/pip-build-xyz","count":89,"extra":"/tmp/pip-build-xyz/setup.py"}
{"timestamp":260,"event":"SUMMARY","pid":1234,"comm":"pip","type":"WRITE","detail":"fd=5","count":1847,"total_bytes":4521984,"extra":"/testbed/venv/lib/requests/api.py"}
{"timestamp":260,"event":"SUMMARY","pid":1234,"comm":"python","type":"FILE_TRUNCATE","detail":"fd=5","count":1}
{"timestamp":260,"event":"SUMMARY","pid":1234,"comm":"bash","type":"CHDIR","detail":"/testbed/src","count":2}

// === 网络（低频 → count 通常 1-2） ===
{"timestamp":260,"event":"SUMMARY","pid":5678,"comm":"python","type":"NET_BIND","detail":"0.0.0.0:8080","count":1}
{"timestamp":260,"event":"SUMMARY","pid":5678,"comm":"python","type":"NET_LISTEN","detail":"fd=3","count":1}
{"timestamp":260,"event":"SUMMARY","pid":9012,"comm":"curl","type":"NET_CONNECT","detail":"127.0.0.1:8080","count":1}
{"timestamp":260,"event":"SUMMARY","pid":1234,"comm":"pip","type":"NET_CONNECT","detail":"pypi.org:443","count":15}

// === 进程协调（低频 → count=1，等价于逐条） ===
{"timestamp":260,"event":"SUMMARY","pid":1234,"comm":"bash","type":"PGRP_CHANGE","detail":"pid=1234,pgid=5678","count":1}
{"timestamp":260,"event":"SUMMARY","pid":1234,"comm":"daemon","type":"SESSION_CREATE","detail":"sid=1234","count":1}
{"timestamp":260,"event":"SUMMARY","pid":1234,"comm":"kill","type":"SIGNAL_SEND","detail":"target=5678,sig=9","count":1}
{"timestamp":260,"event":"SUMMARY","pid":1234,"comm":"python","type":"PROC_FORK","detail":"","count":3}

// === 内存 ===
{"timestamp":260,"event":"SUMMARY","pid":1234,"comm":"python","type":"MMAP_SHARED","detail":"fd=5","count":2,"total_bytes":8192}
```

**所有新事件统一 `event: "SUMMARY"` + `type` 字段**，消费者用 `type` 区分。

**兼容性**：现有消费者（Rust ProcessRunner、analyzer chain）对未知 event type 直接透传，不影响。

## 用户空间 handle_event 扩展

```c
static int handle_event(void *ctx, void *data, size_t data_sz)
{
    const struct event *e = data;
    struct pid_tracker *tracker = (struct pid_tracker *)ctx;
    uint64_t timestamp_ns = e->timestamp_ns;

    // 现有逻辑：PID 过滤对所有事件统一生效
    // 新事件和 FILE_OPEN 一样使用 should_report_file_ops()

    switch (e->type) {
        case EVENT_TYPE_PROCESS:
            // ... 现有代码不变 ...
            break;
        case EVENT_TYPE_BASH_READLINE:
            // ... 现有代码不变 ...
            break;
        case EVENT_TYPE_FILE_OPERATION:
            // ... 现有代码不变 ...
            break;

        /* ========== 新增事件 ========== */

        case EVENT_TYPE_FILE_DELETE:
            if (!should_report_file_ops(tracker, e->pid)) break;
            printf("{\"timestamp\":%llu,\"event\":\"FILE_DELETE\","
                   "\"comm\":\"%s\",\"pid\":%d,"
                   "\"filepath\":\"%s\",\"flags\":%d}\n",
                   timestamp_ns, e->comm, e->pid,
                   e->fs_mut.path, e->fs_mut.flags);
            fflush(stdout);
            break;

        case EVENT_TYPE_FILE_RENAME:
            if (!should_report_file_ops(tracker, e->pid)) break;
            printf("{\"timestamp\":%llu,\"event\":\"FILE_RENAME\","
                   "\"comm\":\"%s\",\"pid\":%d,"
                   "\"old_path\":\"%s\",\"new_path\":\"%s\"}\n",
                   timestamp_ns, e->comm, e->pid,
                   e->rename.old_path, e->rename.new_path);
            fflush(stdout);
            break;

        // NET_BIND, SIGNAL_SEND 等类似模式...
    }
    return 0;
}
```

**关键**：PID 过滤对所有事件统一生效。`-c python` 会同时过滤 EXEC、FILE_OPEN、FILE_DELETE、NET_BIND 等所有事件。

## Ring buffer 大小

现有：256KB。启用所有追踪后，`pip install` 期间 mkdirat/renameat2/unlinkat 可能每秒产生数百事件。

方案：
- 默认保持 256KB（只有现有事件时足够）
- 启用 `--trace-fs` 时自动提升到 1MB
- 启用 `--trace-all` 时提升到 2MB
- 或增加 `--ring-buffer-size` CLI 参数

```c
// process.bpf.c
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);  // 默认值，用户空间可在 load 后调整
} rb SEC(".maps");
```

实际上 ring buffer 大小在 BPF 中是编译时常量。可以通过 `bpf_map__set_max_entries()` 在 load 前调整：

```c
// process.c, load skeleton 后
if (env.trace_all || env.trace_fs) {
    bpf_map__set_max_entries(skel->maps.rb, 1024 * 1024);  // 1MB
}
```

## 文件变更清单

```
agentsight/bpf/
├── process.h           ← 修改：扩展 enum event_type + union 成员
├── process.bpf.c       ← 修改：新增 feature flags + tracepoint handlers
├── process.c           ← 修改：新增 CLI flags + handle_event cases
└── Makefile            ← 不变（process target 已存在）

agentsight/collector/src/framework/runners/
└── process.rs          ← 可选修改：新增 with_trace_features() builder

agentsight/collector/src/main.rs
└──                     ← 可选修改：trace/record 命令传递新 flags
```

**不新增文件**（makefile 不需要改），只修改 3 个现有文件（BPF 侧）。

## 代码组织：header-only 模块化

现有 `process.c` 已超过 700 行，继续往里加会不好维护。重构为 header-only 库的模块化结构：

```
agentsight/bpf/
│
├── process.h                      ← 共享事件结构 + enum（BPF + 用户空间都用）
│
├── ===== BPF 内核侧 =====
├── process.bpf.c                  ← 主 BPF 程序：#include 各模块，定义 maps + feature flags
├── process_bpf_fs.h               ← BPF header-only：文件系统 tracepoints（unlinkat, renameat2, mkdirat, ftruncate, chdir）+ fs_mut_stats_map 内核聚合
├── process_bpf_write.h            ← BPF header-only：write 聚合（enter/exit handlers + write_stats_map + write_fd_map）
├── process_bpf_net.h              ← BPF header-only：网络 tracepoints（bind, listen, connect + sockaddr 解析）
├── process_bpf_signals.h          ← BPF header-only：进程协调 tracepoints（setpgid, setsid, kill, fork）
├── process_bpf_mem.h              ← BPF header-only：内存 tracepoints（mmap MAP_SHARED 过滤）
│
├── ===== 用户空间 =====
├── process.c                      ← 主程序：main() + CLI 解析 + event loop
├── process_utils.h                ← 现有：/proc 读取工具函数
├── process_filter.h               ← 现有：PID 过滤器
├── process_output.h               ← 新增：JSON 输出函数（每种事件类型一个 print 函数）
├── process_dedup.h                ← 重构：现有 FILE_OPEN dedup 逻辑抽出（仍用用户空间 dedup，因为 FILE_OPEN 是现有行为不改）
├── process_map_flush.h            ← 新增：BPF map 用户空间定时遍历 + flush（write_stats_map + fs_mut_stats_map）+ fd→path 解析
├── process_net_fmt.h              ← 新增：sockaddr 格式化（IP/port 字符串转换）
│
└── tests/
    ├── test_process_utils.c       ← 现有（移入）
    ├── test_process_filter.c      ← 现有（移入）
    ├── test_process_output.c      ← 新增：JSON 输出格式
    ├── test_process_dedup.c       ← 新增：FILE_OPEN dedup（现有逻辑的独立测试）
    ├── test_process_net.c         ← 新增：sockaddr 解析（AF_INET/AF_INET6/AF_UNIX）
    └── test_process_map_flush.c   ← 新增：BPF map flush 逻辑 + fd→path
```

**BPF 侧模块化说明**：

`process.bpf.c` 变成一个"胶水"文件，只定义共享资源 + include 各模块：

```c
// process.bpf.c — 精简为胶水文件
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "process.h"

char LICENSE[] SEC("license") = "Dual BSD/GPL";

/* === 共享资源 === */
struct { __uint(type, BPF_MAP_TYPE_RINGBUF); __uint(max_entries, 256 * 1024); } rb SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_HASH); __uint(max_entries, 8192); __type(key, pid_t); __type(value, u64); } exec_start SEC(".maps");

const volatile unsigned long long min_duration_ns = 0;
const volatile bool trace_fs_mutations = false;
const volatile bool trace_network = false;
const volatile bool trace_signals = false;
const volatile bool trace_memory = false;

/* === 现有 handlers（保留在此文件，或也可抽出） === */
// handle_exec, handle_exit, trace_openat, trace_open, bash_readline
// ... 现有代码 ...

/* === 新增模块 === */
#include "process_bpf_fs.h"        // unlinkat, renameat2, mkdirat, ftruncate, chdir
#include "process_bpf_write.h"     // write enter/exit + write_stats_map
#include "process_bpf_net.h"       // bind, listen, connect
#include "process_bpf_signals.h"   // setpgid, setsid, kill, fork
#include "process_bpf_mem.h"       // mmap (MAP_SHARED only)
```

每个 BPF header-only 模块的结构一致：

```c
// process_bpf_fs.h — 文件系统 mutations
#ifndef __PROCESS_BPF_FS_H
#define __PROCESS_BPF_FS_H

// 引用 process.bpf.c 中定义的共享资源（extern 不需要，BPF 同一编译单元）

SEC("tp/syscalls/sys_enter_unlinkat")
int trace_unlinkat(struct trace_event_raw_sys_enter *ctx)
{
    if (!trace_fs_mutations) return 0;
    // ...
}

SEC("tp/syscalls/sys_enter_renameat2")
int trace_renameat2(struct trace_event_raw_sys_enter *ctx)
{
    if (!trace_fs_mutations) return 0;
    // ...
}

SEC("tp/syscalls/sys_enter_mkdirat")
int trace_mkdirat(struct trace_event_raw_sys_enter *ctx)
{
    if (!trace_fs_mutations) return 0;
    // ...
}

SEC("tp/syscalls/sys_enter_ftruncate")
int trace_ftruncate(struct trace_event_raw_sys_enter *ctx)
{
    if (!trace_fs_mutations) return 0;
    // ...
}

SEC("tp/syscalls/sys_enter_chdir")
int trace_chdir(struct trace_event_raw_sys_enter *ctx)
{
    if (!trace_fs_mutations) return 0;
    // ...
}

#endif /* __PROCESS_BPF_FS_H */
```

**好处**：
- 每个模块独立可读（~50-80 行），不用在一个 500+ 行的 .bpf.c 里翻找
- BPF 编译器看到的仍是单个编译单元（通过 #include），maps 和 flags 自然共享
- 可以按模块 review、按模块 disable（注释掉一行 #include）

**重构原则**：
- 每个 `.h` 文件是 header-only（函数带 `static inline` 或 `static`），不生成额外 .o
- `process.c` 的 `#include` 顺序就是依赖顺序
- 测试文件独立编译，不需要 BPF skeleton
- 现有测试 (`test_process_utils`, `test_process_filter`) 不受影响

**从 process.c 抽出的模块**：

| 模块 | 抽出的内容 | 行数（约） |
|------|-----------|-----------|
| `process_output.h` | `print_file_open_event()` + 新事件的 print 函数 | ~150 |
| `process_dedup.h` | `file_hash_entry`, `get_file_open_count()`, `flush_pid_file_opens()`, rate limiting | ~250 |
| `process_write_stats.h` | BPF map iterate, fd→path 解析, WRITE_SUMMARY 输出 | ~100 |
| `process_net.h` | sockaddr 解析, IP/port 格式化 | ~80 |

重构后 `process.c` 的 main/handle_event 约 200 行，清晰可读。

## 测试计划

### 现有测试（不修改）

```bash
make test  # 运行 test_process_utils + test_process_filter
```

### 新增单元测试

#### test_process_output.c
```c
// 测试每种事件类型的 JSON 输出格式
void test_print_file_delete() {
    // 构造 event，capture stdout，验证 JSON 解析正确
    struct event e = { .type = EVENT_TYPE_FILE_DELETE, .pid = 1234, ... };
    char *output = capture_stdout(print_file_delete_event, &e, ts, 1, NULL);
    assert(json_has_key(output, "event", "FILE_DELETE"));
    assert(json_has_key(output, "filepath", "/testbed/old.py"));
}

void test_print_net_bind() {
    // 验证 IPv4 地址 + port 格式化正确
    // 验证 AF_UNIX socket path 格式化正确
}

void test_print_write_summary() {
    // 验证 WRITE_SUMMARY 包含 count, total_bytes, filepath
}
```

#### test_process_dedup.c
```c
// 从现有 process.c 抽出 dedup 逻辑后的独立测试
void test_dedup_same_file() {
    // 同一 (pid, FILE_DELETE, path) 60 秒内只输出一次，count 累加
}

void test_dedup_window_expiry() {
    // 60 秒后窗口到期，flush 带 count + window_expired
}

void test_dedup_process_exit_flush() {
    // 进程退出时 flush 所有 pending 聚合
}

void test_dedup_rename_by_new_path() {
    // FILE_RENAME 按 new_path 做 dedup key
}

void test_rate_limit() {
    // 超过 MAX_DISTINCT_FILES_PER_SEC 后丢弃 + 警告
}
```

#### test_process_net.c
```c
void test_parse_sockaddr_ipv4() {
    struct sockaddr_in addr = { .sin_family = AF_INET, .sin_port = htons(8080), ... };
    char ip[16]; uint16_t port;
    parse_sockaddr((struct sockaddr *)&addr, ip, &port, ...);
    assert(strcmp(ip, "0.0.0.0") == 0);
    assert(port == 8080);
}

void test_parse_sockaddr_ipv6() { ... }
void test_parse_sockaddr_unix() { ... }
```

#### test_process_write_stats.c
```c
void test_write_stats_flush() {
    // 模拟多次 write 聚合，验证 flush 输出正确的 count + total_bytes
}

void test_fd_to_path_resolution() {
    // 测试 /proc/pid/fd/N 读取（用实际 fd 测试）
}
```

### 集成测试（需要 root/eBPF）

```bash
# 回归：默认行为不变
sudo ./process -c python -v    # 应只输出 EXEC/EXIT/FILE_OPEN/BASH_READLINE

# 文件系统追踪
sudo ./process --trace-fs -m 0 &
pip install requests            # 应看到 DIR_CREATE, FILE_RENAME（带 count 聚合）
rm /tmp/test.txt                # 应看到 FILE_DELETE
echo "hello" > /tmp/test.txt    # 5 秒后应看到 WRITE_SUMMARY

# 网络追踪
sudo ./process --trace-net -m 0 &
python -m http.server 8080      # 应看到 NET_BIND + NET_LISTEN
curl localhost:8080             # 应看到 NET_CONNECT

# 进程协调追踪
sudo ./process --trace-signals -m 0 &
setsid sleep 100 &              # 应看到 PROC_FORK + SESSION_CREATE
kill -9 $!                      # 应看到 SIGNAL_SEND

# 全开
sudo ./process --trace-all -c pip -m 2 &
pip install flask               # 应看到所有事件类型，PID 过滤生效
```

## 实施步骤

### Step 1：重构 process.c 为模块化结构（1-2 小时）

**先不加新功能**，只做代码搬移：
- 抽出 `process_output.h`（print 函数）
- 抽出 `process_dedup.h`（dedup + rate limit 逻辑）
- 验证 `make process && make test` 通过，行为不变

### Step 2：扩展 process.h + process.bpf.c（2-3 小时）

分四批，每批加完即可测试：

**批次 A（文件系统 mutations）**：
- 新增 feature flag: `trace_fs_mutations`
- 新增 tracepoints: `unlinkat`, `renameat2`, `mkdirat`, `ftruncate`, `chdir`
- 新增 write 聚合: `write_fd_map` + `write_stats_map` + enter/exit handlers
- 新增 `process_write_stats.h` + `test_process_write_stats.c`

**批次 B（网络）**：
- 新增 feature flag: `trace_network`
- 新增 tracepoints: `bind`, `listen`, `connect`
- 新增 `process_net.h`（sockaddr 解析）+ `test_process_net.c`

**批次 C（进程协调）**：
- 新增 feature flag: `trace_signals`
- 新增 tracepoints: `setpgid`, `setsid`, `kill`, `sched_process_fork`
- 参数都是整数，实现最简单

**批次 D（内存）**：
- 新增 feature flag: `trace_memory`
- 新增 tracepoint: `mmap`（内核侧过滤仅 MAP_SHARED）

### Step 3：扩展 process.c（1 小时）

- 添加 CLI 参数：`--trace-fs`, `--trace-net`, `--trace-signals`, `--trace-mem`, `--trace-all`
- 设置 BPF feature flags
- 扩展 `handle_event()` switch（使用 `process_output.h` 的 print 函数）
- 新增 write stats 的定时 flush 逻辑（在 poll loop 中每 5 秒检查一次）
- 文件系统 mutations 复用 `process_dedup.h` 的聚合框架

### Step 4：新增测试（1 小时）

- 将现有 `test_process_utils.c`、`test_process_filter.c` 移入 `tests/`
- 新增 `tests/test_process_output.c`
- 新增 `tests/test_process_dedup.c`（扩展覆盖新事件类型的 dedup）
- 新增 `tests/test_process_net.c`
- 新增 `tests/test_process_write_stats.c`
- 更新 Makefile：测试编译路径指向 `tests/`，`make test` 运行全部测试

### Step 5：集成测试 + Rust collector（30 分钟）

- 手动运行集成测试
- 可选：给 `ProcessRunner` 加 `with_trace_features()` builder

## 与论文实验的对应

| 论文实验 | 需要的新事件 | 启用 flag |
|----------|-------------|----------|
| 1.2（文件变更范围） | FILE_DELETE + DIR_CREATE + FILE_RENAME | `--trace-fs` |
| 2.2（文件系统覆盖对比） | 同上 | `--trace-fs` |
| 2.3（进程隔离对比/Table 2） | PGRP_CHANGE + SESSION_CREATE + SIGNAL_SEND | `--trace-signals` |
| 3.1（状态污染演示） | NET_BIND（端口冲突） | `--trace-net` |
| 3.3（包安装副作用） | DIR_CREATE + FILE_RENAME + FILE_DELETE | `--trace-fs` |
| 4.1（多维状态污染） | NET_BIND + SIGNAL_SEND | `--trace-net --trace-signals` |
| 4.3（多维副作用频率） | 全部 | `--trace-all` |

## 设计约束

1. **新事件全部内核聚合**——所有新增事件走 BPF hash map，ring buffer 只给现有 4 种事件
2. **struct event 不变**——新事件不用 struct event，不占 ring buffer
3. **mmap() 内核侧过滤**——只聚合 MAP_SHARED 的 mmap，忽略 MAP_PRIVATE
4. **不修改现有接口**——不加新 flag 时行为、性能完全不变
5. **现有 FILE_OPEN 不变**——仍用用户空间 60 秒窗口 dedup（向后兼容）
