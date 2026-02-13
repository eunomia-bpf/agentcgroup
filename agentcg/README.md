# agentcg - Integrated eBPF Agent Cgroup Management

Minimal PoC that integrates three independent eBPF tools to provide comprehensive
resource isolation and monitoring for AI agent workloads.

## Components

| Component | Directory | Description |
|-----------|-----------|-------------|
| **scx_flatcg** | `scheduler/` | eBPF CPU scheduler (sched_ext) - flattened cgroup hierarchy scheduling |
| **memcg_priority** | `memcg/` | eBPF memory isolation (memcg_bpf_ops) - priority-based memory management |
| **process monitor** | `process/` | eBPF process lifecycle monitor - tracks EXEC/EXIT events with filtering |

## Architecture

```
agentcgroupd (shell script)
  ├── Creates cgroup hierarchy (/sys/fs/cgroup/agentcg/)
  │   ├── session_high/   (high-priority agent session)
  │   └── session_low/    (low-priority / throttled)
  ├── Starts scx_flatcg       → CPU scheduling via cgroup weights
  ├── Starts memcg_priority    → Memory isolation via BPF struct_ops
  └── Starts process monitor   → Detects new processes, assigns to cgroups
```

### Inter-component Communication

- **process → agentcgroupd**: JSON events over stdout pipe (EXEC, EXIT, FILE_OPEN)
- **agentcgroupd → cgroups**: Writes PIDs to `cgroup.procs` to assign processes
- **agentcgroupd → memcg**: CLI arguments at startup (`--high`, `--low`)
- **scheduler**: Reads cgroup weights automatically (no explicit communication)
- **Shared state**: cgroup filesystem (`memory.events`, `cpu.weight`)

## Prerequisites

- Linux kernel with sched_ext and memcg_bpf_ops support
- clang/llvm (tested with llvm-19)
- libbpf dependencies: `libelf-dev`, `zlib1g-dev`, `libzstd-dev`
- `jq` (optional, for JSON parsing in agentcgroupd)
- Built `third_party/bpftool` and `third_party/scx`

## Build

```bash
# Build all three components
cd agentcg/
make

# Or build individually
make scheduler
make memcg
make process
```

## Usage

```bash
# Run the integrated daemon (requires root)
sudo ./agentcgroupd [CGROUP_ROOT]

# Default cgroup root: /sys/fs/cgroup/agentcg
sudo ./agentcgroupd

# Custom cgroup root
sudo ./agentcgroupd /sys/fs/cgroup/my_agents
```

### Running Components Individually

```bash
# CPU scheduler
sudo ./scheduler/scx_flatcg -i 5

# Memory isolation
sudo ./memcg/memcg_priority \
    --high /sys/fs/cgroup/agentcg/session_high \
    --low /sys/fs/cgroup/agentcg/session_low \
    --delay-ms 50 --below-low

# Process monitor (JSON output)
sudo ./process/process -m 2 -c "python,bash,pytest"
```

## Directory Structure

```
agentcg/
├── scheduler/              # scx_flatcg CPU scheduler
│   ├── scx_flatcg.bpf.c   # BPF kernel program
│   ├── scx_flatcg.c        # Userspace loader
│   ├── scx_flatcg.h        # Shared header
│   └── Makefile
├── memcg/                  # memcg_priority memory isolation
│   ├── memcg_priority.bpf.c
│   ├── memcg_priority.c
│   ├── memcg_priority.h
│   └── Makefile
├── process/                # Process lifecycle monitor
│   ├── process.bpf.c
│   ├── process.c
│   ├── process.h
│   ├── process_filter.h
│   ├── process_utils.h
│   └── Makefile
├── agentcgroupd            # Wrapper script (coordinates all three)
├── Makefile                # Top-level build
└── README.md
```

## Dependencies

| Component | libbpf | bpftool | vmlinux.h |
|-----------|--------|---------|-----------|
| scheduler | `third_party/bpftool/src/libbpf` | `third_party/bpftool` | scx vmlinux headers |
| memcg | `memcg/linux/tools/lib/bpf` | `third_party/bpftool` | Generated from `/sys/kernel/btf/vmlinux` |
| process | `third_party/bpftool/src/libbpf` | `third_party/bpftool` | Generated from `/sys/kernel/btf/vmlinux` |
