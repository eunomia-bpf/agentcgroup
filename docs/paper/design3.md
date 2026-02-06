下面给你一份**可以直接落地实现**的“工程化设计文档 + 操作手册 + eval plan（含逐步命令/脚本骨架 + 明确 RQ）”，核心以 **flatcg（CPU）+ memcg_bpf_ops / memops（MEM）** 为 data plane，并把“安装后对很多容器生效”的工业化形态（NRI/daemon/config/可观测）也一起纳入。

我会先把“Linux 现有接口与是否必须 sched_ext”讲清楚（你要求的 interface survey），再给完整设计与实现步骤。

---

## 0. 术语与定位

* **flatcg**：指基于 sched_ext 的 `scx_flatcg` 思路：把 cgroup 层级权重折叠/传播，做层级公平（hierarchical fairness），并以“先选 cgroup 再选 task”的两级调度为核心。`scx_flatcg` 自己的注释也明确了这个结构，并提到在 cgroup delegation 边界做分层可以缓解 correctness/突发问题。([GitHub][1])
* **memops**：你这里说的更像是**memcg 的可编程 hooks**——当前公开实现路径是 **memcg_bpf_ops（BPF struct_ops）**，提供 `get_high_delay_ms / below_low / below_min / handle_cgroup_online/offline` 等钩子。([Spinics][2])
* **本系统定位用词**：不要叫“resource manager”（容易被理解成 Borg/YARN 级别），更准确的是：

  * **in-kernel resource governance / in-kernel resource control**（资源治理/控制系统）
  * 或 “agent-aware resource control plane + in-kernel enforcement”
    你论文里现在用的 “resource controller” 是可以的，但最好加限定词：**node-level (host-level) resource controller**，避免被当成 cluster scheduler。

---

## 1. Linux 现状：CPU/cgroup 接口有什么？我们是否必须用 sched_ext？

### 1.1 cgroup v2 CPU controller 现有接口（你必须在论文里明确）

cgroup v2 的 CPU 控制核心是：

* `cpu.weight`：按权重分配 CPU（层级公平）。并且文档明确：它影响 **fair-class scheduler**，以及“带 `cgroup_set_weight` callback 的 BPF scheduler（具体取决于 callback 实现）”。([Kernel Documentation][3])
* `cpu.max` / `cpu.max.burst`：带宽/配额模型（quota/period）。**关键点：文档明确 `cpu.max` 只影响 fair-class scheduler。**([Kernel Documentation][3])
* 其它相关：`cpu.pressure`（PSI）、`cpu.uclamp.min/max`（util clamp）等。([Kernel Documentation][3])

**结论 1（很关键）**：如果你启用 sched_ext 全局接管普通任务，**Docker/K8s 写进 `cpu.max` 的配额并不会自动生效**（因为它只作用于 fair-class）。([Kernel Documentation][3])
所以工业化要么：

* A) 在 sched_ext 里自己实现 cpu.max 语义（或用 sched_ext 提供的 cgroup bandwidth callbacks），
* B) 只用 sched_ext **partial mode** 让特定任务走 SCHED_EXT，其它容器仍走 CFS（这样 `cpu.max` 对那些容器仍有效）。([Kernel Documentation][4])

### 1.2 sched_ext 提供了什么（为什么它是你 CPU 侧的“必要条件”）

kernel 文档明确：sched_ext 是一个可扩展调度类，BPF scheduler 可以实现任意算法；并且它有**fail-safe**：出错/可运行任务 stall/sysrq 会自动回退到默认 fair scheduler。([Kernel Documentation][4])
同时文档也明确：sched_ext 的 BPF API **没有稳定性保证**（ABI instability），所以你必须 pin kernel 版本/commit 来保证可复现。([Kernel.org][5])

**结论 2**：如果你的论文核心主张是“在 kernel enforcement points 做微秒级决策”，CPU 侧要做到这一点，**sched_ext 基本是目前 Linux 上唯一合理的 upstream 路径**（否则你只能：改内核/CFS，或用用户态调参/改变 nice/rt 优先级——这都不是真正的 in-kernel policy）。

### 1.3 你要的 cgroup callbacks（权重/带宽）目前是什么状态？

sched_ext 社区在推进 cgroup 相关 callback，例如：

* `cgroup_set_weight(struct cgroup*, u32 weight)`
* `cgroup_set_bandwidth(struct cgroup*, u64 period_us, u64 quota_us, u64 burst_us)`
  这在 LKML 的 patch 里能看到。([LKML][6])

**工程含义**：

* 如果你想“兼容 K8s/Docker 的 cpu.weight/cpu.max 写法”，最理想是依赖这些 callbacks（或 scx 仓库里相应的兼容层）。
* 论文里要诚实：你到底用的是 mainline 的哪版，还是 linux-next/bpf-next 的哪套 patch。否则审稿人会抓“compatibility/no kernel modification”这句。

### 1.4 内存侧 memcg_bpf_ops（memops）是什么状态？

公开信息显示 memcg_bpf_ops 仍在 RFC/patch series 演进中，但其 hook 语义非常清晰：

* `get_high_delay_ms`：当 cgroup breach `memory.high` 时返回自定义 throttling delay（ms）
* `below_low` / `below_min`：覆盖 memory.low/min 的保护判定
* `handle_cgroup_online/offline`：生命周期回调
  并且这些 hook 会被集成进 charge path（如 `try_charge_memcg`）和 over-high handler 等。([Spinics][2])

**工程含义**：你可以做“内核内即时决策 + 由内核执行 throttling”，这正是你要打的“timescale mismatch”。

---

## 2. AgentCgroup v0.1 设计文档（flatcg + memops）

### 2.1 目标与非目标

#### 目标（论文/系统都需要写成 checklist）

* **G1 Isolation**：邻居 burst（CPU fork storm / mem blow-up）不显著抬高受害者 p99/p999 step latency。
* **G2 Domain alignment**：资源域对齐到 **(container/session) → step/tool-call** 的生命周期边界，而不是只有 container 粒度。
* **G3 Timescale**：控制决策发生在 kernel enforcement point（scheduler dispatch / memcg charge/throttle path），避免 10–100ms userspace loop。
* **G4 Compatibility**：尽量复用 cgroup v2 语义（memory.high/max/events，cpu.weight/max），并提供渐进式部署路径。
* **G5 Fail-safe**：sched_ext 出错/卡死自动回退 CFS；内存侧策略可随时卸载，回到默认 memcg 行为。([Kernel Documentation][4])

#### 非目标

* 不做 cluster-level scheduling（不是 Borg/Mesos）。
* 不追求替代容器 runtime（Docker/containerd 仍负责创建容器与基础 cgroup）。
* 不在 v0.1 里解决 I/O（blkio）与 network（tc）协同（但接口预留）。

---

## 2.2 系统模型与威胁模型

### 系统模型

* Host 上运行多个 **tenant/workload**（K8s Pod/Container 或者 agent sandbox）。
* 每个 workload 内部包含多个 **steps/tool-calls**（短命进程树），由 agent runtime 触发。
* **资源域映射**：每个 workload 对应一个 cgroup 子树；每个 tool-call 对应子 cgroup。

### 威胁模型（审稿人会问）

* **Benign contention**：多个正常 agent 并发，阶段性 burst 造成互扰。
* **Adversarial tool**：某些 step 可能出现：

  * CPU spin（busy loop）
  * fork storm（进程爆炸）
  * memory blow-up（瞬时分配/工作集膨胀）
* 我们目标是：

  * **contain**：问题被限制在对应 step/workload 子树
  * **preserve neighbor SLO**：保护其它 tenant 的 tail latency
  * **recover**：通过 freeze/kill/oom.group 保持完整性与可恢复性

---

## 2.3 架构概览

### 控制面 / 数据面分离

**Data plane（内核态 BPF）**

1. `agentcg_scx`：sched_ext BPF scheduler（flatcg 改造版）
2. `agentcg_mem`：memcg_bpf_ops BPF 程序（memops）
3. 共享 BPF maps：policy state + cross-resource signals + metrics ringbuf

**Control plane（用户态）**

1. `agentcgroupd`：daemon

   * 管理 cgroup 树（或接管/观察 container runtime 的 cgroup）
   * 接收 runtime hints（step start/end，class，deadline，预算）
   * 更新 BPF maps，采集事件并导出指标（Prometheus/JSON）
2. 可选：`agentcg-nri`（containerd/CRI-O 的 NRI 插件）

   * 在容器生命周期事件中注入标注/参数（cgroup path、PID、资源规格）
   * 使“安装后对很多容器生效”成为现实
     NRI 文档明确：插件注册后能订阅 container lifecycle events（creation/starting/updating/stopping/removal 等），并能获得容器的 cgroups path、PID 等属性。([GitHub][7])

---

## 2.4 cgroup 层级与生命周期（Domain 对齐机制）

### 2.4.1 推荐 cgroup 树（host 视角）

```text
/sys/fs/cgroup/
  agentcg.slice/                 # 我们的根（systemd slice 或手动目录）
    tenant_A/                    # 一个容器 / 一个 agent sandbox / 一个 workload
      sess_0001/                 # 可选：一次 agent session
        step_0001/               # tool-call / step
        step_0002/
    tenant_B/
      sess_0100/
        step_0101/
```

### 2.4.2 为什么你要强调 memory.high/max/events 与 freeze/kill

* `memory.high`：超过后会 throttle 并进入强 reclaim，但**不会触发 OOM killer**，适合做“可恢复的软限”。([Kernel Documentation][3])
* `memory.max`：硬限，超过后 cgroup 内触发 OOM。([Kernel Documentation][3])
* `memory.events`：包含 `high/max/oom/oom_kill` 等事件计数，其中 `high` 就是“超过 memory.high 导致进程被 throttle 并 routed to direct reclaim”的次数。([Kernel Documentation][3])
* `cgroup.freeze`：写 1 冻结子树，等冻结完成 `cgroup.events` 里的 `frozen` 会更新并通知；冻结可能需要时间。([Kernel Documentation][3])
* `cgroup.kill`：写 1 会 kill 整个子树，且文档明确能正确处理并发 fork，并防止 migration race。([Kernel Documentation][3])

**论文叙事建议**：
你把 `freeze` 定义为 “soft-pause semantics（暂停等待 runtime 采取动作）”，`kill` 定义为 “integrity-preserving termination”，都能直接用内核文档背书。([Kernel Documentation][3])

---

## 2.5 CPU：基于 flatcg 的 sched_ext 策略设计

### 2.5.1 从 scx_flatcg 出发的关键策略点：你到底要在 eBPF 里编程什么？

一句话：**把“谁先跑/跑多久/谁被降权/谁被限额/谁被优先”这套决策，搬到 kernel scheduler 的 dispatch 点。**

`scx_flatcg` 的核心结构（先选 cgroup 再选 task）以及它对 delegation 边界分层的建议，是你很好的起点。([GitHub][1])

你要在 BPF 里实现的策略可拆成 5 个“可被实验验证”的模块：

1. **Hierarchical fairness（flatcg）**

   * 以 cgroup 为单位做 WFQ/vtime（或类似 CFS vruntime 思路）
   * 维护每个 cgroup 的 `cg_vtime`，以 `delta = slice / weight` 更新
2. **Tool-call latency boost（agent-aware）**

   * 对 “interactive/latency class step” 给短暂 boost（更小 vtime 增量、更高权重或更小 slice 抖动）
3. **cpu.max 语义（bandwidth control）**

   * 对容器/step 的 CPU limit 以 token bucket 执行（period/quota/burst）
   * 这是工业化兼容 K8s 的关键（因为 `cpu.max` 默认不影响 sched_ext）([Kernel Documentation][3])
4. **Anti-burst / anti-thundering herd**

   * 处理“很多 idle step 同时 wakeup”导致短时间不公平/抢占问题（flatcg 自己承认这是 correctness 风险）([GitHub][1])
5. **Cross-resource coupling**（与内存联动）

   * 如果某 step 正在 memcg over-high / direct reclaim（或我们判定其 memory pressure 高），CPU 侧对它**降权/限 slice**，防止 reclaim 抖动放大 tail latency。

### 2.5.2 数据结构（BPF maps）——工程必须写死

**Key 选择：cgroup id**（而不是 pid），避免 fork/exec 变动。

```c
// bpf/agentcg_common.h
enum agentcg_class {
  AGENTCG_INTERACTIVE = 0,
  AGENTCG_LATENCY     = 1,
  AGENTCG_BATCH       = 2,
  AGENTCG_BESTEFFORT  = 3,
};

struct agentcg_cgrp_cfg {
  __u32 weight;          // 1..10000 (align cpu.weight semantics)
  __u32 class;           // agentcg_class
  __u64 deadline_ns;     // optional
  __u64 burst_credit_us; // CPU burst budget for interactive steps

  // bandwidth (cpu.max-like)
  __u64 bw_period_us;
  __u64 bw_quota_us;
  __u64 bw_burst_us;
};

struct agentcg_cgrp_rt {
  __u64 vtime;           // virtual time for WFQ
  __u64 vruntime_bias;   // for boost/hysteresis
  __u64 bw_runtime_us;   // remaining tokens in current period
  __u64 bw_period_start; // ktime of current period start
  __u64 mem_penalty;     // coupling: derived from mem pressure
  __u64 nr_enqueued;
};
```

Maps：

* `BPF_MAP_TYPE_HASH cfg_map[cgrp_id] -> agentcg_cgrp_cfg`
* `BPF_MAP_TYPE_HASH rt_map[cgrp_id]  -> agentcg_cgrp_rt`
* `BPF_MAP_TYPE_RINGBUF events`（导出调度事件/trace）
* `BPF_MAP_TYPE_HASH task_map[pid] -> cgrp_id`（或 task storage）

### 2.5.3 调度算法（可写进论文的伪代码 + 可实现）

#### 2-level pick：tenant/session/step（cgroup）→ task

**核心思路**：

* 维护一个 “runnable cgroup heap/queue”（按最小 `vtime/weight`）
* dispatch 时选最小者，然后从该 cgroup 的 DSQ 中取 task
* slice 决策：base slice ± boost ± mem penalty

伪代码（论文可直接放）：

```text
on enqueue(task p):
  cg = cgroup_of(p)
  push p into cg_runq
  if cg becomes runnable: push cg into runnable_cg_set

pick_next():
  cg = argmin_cg (cg.vtime + penalty(cg.mem_penalty) - boost(cg.class))
  if !cg.bandwidth_has_token(): return pick_other_cg_or_idle()
  p = pop_task(cg)
  slice = slice_us(cg.class) * f(penalty, boost)
  dispatch(p, slice)
  charge_bandwidth(cg, slice)
  cg.vtime += slice / effective_weight(cg)
```

#### cpu.max 的实现（必须补齐，否则 K8s 兼容性站不住）

因为 `cpu.max` 默认只影响 fair scheduler，你在 sched_ext 必须自己 enforce（或者依赖 sched_ext 的 cgroup bandwidth callbacks）。([Kernel Documentation][3])

最可实现的方式：**token bucket**

* 每个 cgroup 有 `quota_us` per `period_us`，可额外 `burst_us`
* 每次 dispatch 扣 token；token 用完则该 cgroup 暂停调度直到下个 period replenish
* 需要避免“所有 cgroup 都耗尽 → sched_ext watchdog 判定 stall”。工程上：

  * 保留 root cgroup（或 housekeeping）无限配额
  * 或者在 “全部耗尽” 时显式让 CPU idle（不要让 runnable task 卡死不运行）
  * 并在 daemon 层监控这种异常作为 policy bug

#### Anti-thundering herd（flatcg 的 correctness 风险）

flatcg 代码注释明确提到 correctness 问题与 thundering herd，并建议在 delegation 边界做分层。([GitHub][1])

你可以在 v0.1 用一个“低成本且可解释”的修正：

* cgroup 从 idle→active 的第一次 enqueue 时，给它一个 `activation_vtime = now_vtime`（或当前最小 vtime）以避免大量 idle cgroup 同时激活时抢跑过多
* 对 interactive boost 使用 **budget**（burst_credit_us），用完回落，避免持续饿死 batch

---

## 2.6 Memory：基于 memcg_bpf_ops（memops）的策略设计

### 2.6.1 memcg_bpf_ops 提供的 hooks（你要用哪些？）

RFC/patch 与 LWN 描述一致：

* `get_high_delay_ms`：当 cgroup breach `memory.high` 时返回 throttle delay（ms）
* `below_low` / `below_min`：覆盖 memory.low/min 保护检查
* `handle_cgroup_online/offline`：用于 state 初始化/清理
  并且 hook 会进入 charge path 与 over-high handler。([Spinics][2])

### 2.6.2 为什么不用纯 userspace memory.high 调参？

cgroup 文档明确 `memory.high` 的设计假设是“外部进程监控并采取措施”。([Kernel Documentation][3])
你论文的 point 是：agent bursts 在 ms 级，userspace loop 10–100ms 太慢（timescale mismatch），所以我们把决策搬进 kernel hook（memops）。

### 2.6.3 可实现的 delay 函数（把“dynamic/fine-grained”写成具体公式）

输入：

* `usage = memory.current`
* `high = memory.high`
* cgroup class（interactive/latency/batch）
* `burst_credit_bytes`（允许短 burst）
* 可选：PGFAULT rate（LWN patch 示例就是按 PGFAULT 触发 below_min/priority）([LWN.net][8])

输出：

* `delay_ms`（给内核 throttling）

建议 v0.1 策略：**burst-aware piecewise + hysteresis**

```text
over = max(0, usage - high)

if over <= burst_credit_bytes:
    delay = 0   // allow short burst
else:
    ratio = (over - burst_credit_bytes) / high
    base = class_factor[class] * ratio
    delay_ms = clamp(base * 1000, 0, Dmax)
    delay_ms = hysteresis_smooth(delay_ms, last_delay_ms)
```

其中：

* `class_factor[interactive] < class_factor[batch]`（interactive 更宽松）
* `Dmax` 建议 50–200ms（不要无限大，避免 scheduler stall）
* hysteresis：避免 usage 在 high 附近抖动导致 delay oscillation

### 2.6.4 below_low/below_min 的用法（谨慎 + 论文要写清楚边界）

memcg hook 支持 `below_low/below_min` override。([Spinics][2])
你可以把它用作**短窗保护**：

* 对 latency-critical step：当它出现 PGFAULT spike 或者被判定为“交互关键窗口”，短暂返回 `below_min=true`，让 reclaim 优先回避它；
* 但必须配合全局 guardrail：如果 system-wide 进入严重内存压力（可用 global watermark 或者 host PSI），禁用该 override，防止把系统推向 OOM。

---

## 2.7 CPU↔MEM 联合策略：效果与必要性（你论文要站住的核心）

### 2.7.1 为什么需要联合策略？

单独做 CPU fairness 或单独做 memory throttling 仍会出现两类 tail amplification：

1. **mem over-high → direct reclaim → CPU 抖动**
   `memory.events.high` 本质上表示该 cgroup 的进程被 throttle 并 routed 去做 direct reclaim。([Kernel Documentation][3])
   在这个状态下，如果 CPU 仍给它高权重，它会把 CPU 花在 reclaim/抖动上，同时把 runqueue 搞大，影响其他 tenant 的 tail。

2. **CPU 爆炸（fork storm）→ 竞争加剧 → 内存回收更糟**
   CPU 调度器若不感知 mem pressure，会让“正在造成内存压力的 step”继续获得 CPU，进一步触发 page faults / reclaim，形成正反馈。

### 2.7.2 联合策略接口：共享 in-kernel state（最简单、可实现、可解释）

设计一个 `pressure_map[cgrp_id]`，由 memops 写、scx 读：

* memops（get_high_delay_ms）在计算 delay 时，同时更新：

  * `mem_penalty = f(over_high_ratio, pgfault_rate)`
  * `mem_overhigh_state = {0/1}`
* scx 在 pick_next 时把 `mem_penalty` 计入 vtime 或 slice：

  * `effective_weight = weight / (1 + mem_penalty)`
  * 或 `slice = slice / (1 + mem_penalty)`

反向（CPU→MEM）可以 v0.2 再做：例如 CPU 侧检测 step backlog 过大时，降低其 burst_credit_bytes。

---

## 3. 工业化形态：安装后对很多容器生效（Docker/K8s 结合点）

你要的“装上就对很多容器生效”，本质是两个问题：

1. **如何发现/标识容器的 cgroup？**
2. **如何把容器的资源规格/语义映射到我们的 policy state？**

### 3.1 推荐方案：containerd/CRI-O 的 NRI 插件 + node daemon

NRI 的 README 明确：

* 插件注册后能订阅容器生命周期事件（creation/starting/updating/stopping/removal 等）
* 插件能拿到容器的 `cgroups path`、`process ID`、以及很多 Linux 资源相关属性，并且可以请求对容器进行调整。([GitHub][7])

因此工程架构：

* `agentcgroupd`：常驻 daemon（加载 BPF、维护 maps、导出 metrics）
* `agentcg-nri`：NRI 插件（把 container metadata + cgroup path 推给 daemon，并可根据 annotations 选择是否启用 SCHED_EXT/分类）

**渐进式部署策略（强烈建议）**：

* sched_ext **partial mode**：只让被标记的容器进程使用 `SCHED_EXT`，其他容器继续走 CFS。文档明确 partial mode 行为。([Kernel Documentation][4])
  这样你不会一下子改变整个节点的调度语义，工程风险可控。

### 3.2 Docker 的落地方式

Docker 默认也是基于 containerd 的 cgroup 配置。最稳的做法仍是：

* 在 K8s 侧用 CRI/containerd + NRI
* 或在纯 Docker 环境里用 “cgroup watcher + label mapping”：daemon 监听 `/sys/fs/cgroup/*` 下新增目录，按命名规则（docker-<id>.scope）识别容器，并应用默认 policy。
  （这条不如 NRI 正规，但可作为 fallback。）

---

## 4. 实现手册：repo 结构、构建与运行（可以按步骤直接做）

下面给你一套 **“最小可运行 MVP”** 的 repo 规划。目标是：先跑通 CPU（flatcg 改造）+ MEM（memops）+ daemon + 基准，再逐步增强。

### 4.1 Repo layout（建议）

```text
agentcgroup/
  bpf/
    agentcg_scx.bpf.c          # sched_ext scheduler (flatcg-based)
    agentcg_mem.bpf.c          # memcg_bpf_ops program
    agentcg_common.h
  daemon/
    agentcgroupd.c             # or Go/Rust
    config.yaml
  runtime/
    libagentcg/                # optional: agent runner integration
    agentcgctl                 # CLI: create session/step cgroups, set hints
  nri/
    agentcg-nri.go             # optional: NRI plugin
  bench/
    setup_env.sh
    workloads/
      code_compile.sh
      data_pandas.sh
      web_playwright.sh
    noisy/
      cpu_spin.sh
      fork_storm.sh
      mem_blowup.sh
    collect/
      collect_metrics.sh
      parse_logs.py
      plot.py
  docs/
    DESIGN.md                  # 这份设计文档
    EVAL.md
  Makefile
```

---

## 5. 核心代码（最小骨架，能按步骤扩展到可跑）

> 说明：sched_ext API 会随内核版本变化（ABI instability），所以你要在 `docs/EVAL.md` 里**固定 kernel 版本/commit**。([Kernel.org][5])
> 下面代码是“结构骨架 + 关键逻辑点”，你按你最终 pin 的内核头文件签名做一次对齐即可。

### 5.1 `bpf/agentcg_scx.bpf.c`（核心：cgroup WFQ + mem penalty + bandwidth token）

```c
// SPDX-License-Identifier: GPL-2.0
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include "agentcg_common.h"
#include "scx/common.bpf.h"   // 来自 tools/sched_ext/include/scx (按你的内核/工具树)

char LICENSE[] SEC("license") = "GPL";

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 65536);
  __type(key, __u64);                 // cgroup id
  __type(value, struct agentcg_cgrp_cfg);
} cfg_map SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 65536);
  __type(key, __u64);                 // cgroup id
  __type(value, struct agentcg_cgrp_rt);
} rt_map SEC(".maps");

// 简化：每个 cgroup 一个 DSQ（实际要做 DSQ lifecycle 管理/上限）
static __u64 dsq_id_for_cgrp(__u64 cgid) {
  return (cgid | (1ULL << 63)); // 避免和内建 DSQ 冲突
}

static __always_inline struct agentcg_cgrp_cfg *get_cfg(__u64 cgid) {
  return bpf_map_lookup_elem(&cfg_map, &cgid);
}

static __always_inline struct agentcg_cgrp_rt *get_rt(__u64 cgid) {
  return bpf_map_lookup_elem(&rt_map, &cgid);
}

static __always_inline void bw_replenish(struct agentcg_cgrp_cfg *cfg,
                                         struct agentcg_cgrp_rt *rt,
                                         __u64 now_ns)
{
  if (!cfg->bw_period_us || cfg->bw_quota_us == (__u64)-1)
    return;

  __u64 period_ns = cfg->bw_period_us * 1000;
  if (rt->bw_period_start == 0)
    rt->bw_period_start = now_ns;

  if (now_ns - rt->bw_period_start >= period_ns) {
    rt->bw_period_start = now_ns;
    rt->bw_runtime_us = cfg->bw_quota_us + cfg->bw_burst_us;
  }
}

static __always_inline bool bw_has_token(struct agentcg_cgrp_cfg *cfg,
                                         struct agentcg_cgrp_rt *rt,
                                         __u64 now_ns)
{
  if (cfg->bw_quota_us == (__u64)-1) // "max"
    return true;
  bw_replenish(cfg, rt, now_ns);
  return rt->bw_runtime_us > 0;
}

static __always_inline void bw_charge(struct agentcg_cgrp_cfg *cfg,
                                      struct agentcg_cgrp_rt *rt,
                                      __u64 slice_us)
{
  if (cfg->bw_quota_us == (__u64)-1)
    return;
  if (rt->bw_runtime_us > slice_us)
    rt->bw_runtime_us -= slice_us;
  else
    rt->bw_runtime_us = 0;
}

static __always_inline __u64 class_base_slice_us(__u32 class)
{
  switch (class) {
  case AGENTCG_INTERACTIVE: return 2000;  // 2ms
  case AGENTCG_LATENCY:     return 4000;  // 4ms
  case AGENTCG_BATCH:       return 8000;  // 8ms
  default:                  return 4000;
  }
}

static __always_inline __u64 effective_weight(struct agentcg_cgrp_cfg *cfg,
                                              struct agentcg_cgrp_rt *rt)
{
  __u64 w = cfg->weight ? cfg->weight : 100;
  // mem_penalty: 0..N，越大越降权（简化：w/(1+penalty)）
  __u64 p = rt->mem_penalty;
  if (p == 0) return w;
  return w / (1 + p);
}

s32 BPF_STRUCT_OPS(agentcg_enqueue, struct task_struct *p, u64 enq_flags)
{
  __u64 cgid = bpf_get_current_cgroup_id(); // 简化：实际需从 p 取 cgroup id
  __u64 dsq = dsq_id_for_cgrp(cgid);

  // 确保 DSQ 存在（真实实现要做 create_dsq + refcount）
  scx_bpf_create_dsq(dsq, -1);

  // 直接把 task 放进对应 cgroup DSQ
  scx_bpf_dispatch(p, dsq, SCX_SLICE_DFL, enq_flags);
  return 0;
}

void BPF_STRUCT_OPS(agentcg_dispatch, s32 cpu, struct task_struct *prev)
{
  // 简化：真实需要 runnable cgroup 集合（min-vtime pick）
  // 这里给一个最小可跑骨架：从本地 DSQ->全局 DSQ 回退
  scx_bpf_consume(SCX_DSQ_LOCAL);
  scx_bpf_consume(SCX_DSQ_GLOBAL);
}

// TODO: 在 running/stopping 里记账 slice, 更新 vtime, bw tokens 等。
// 真实实现里你需要：
// - stopping(): 得到实际运行时间，转成 slice_us
// - 更新 cgroup rt->vtime += slice_us / effective_weight

SCX_OPS_DEFINE(agentcg_ops,
  .enqueue   = (void *)agentcg_enqueue,
  .dispatch  = (void *)agentcg_dispatch,
  .name      = "agentcg_flatcg",
  .flags     = SCX_OPS_SWITCH_PARTIAL, // 工业部署建议先用 partial mode
);
```

**你需要补齐的关键点**（工程 TODO，但路径明确）：

* “runnable cgroup set” 的维护（min-vtime pick）
* stopping 记账 & vtime 更新
* cgroup DSQ lifecycle（避免泄露/上限）
* 从 `task_struct *p` 获取准确的 cgroup id（而不是用 current）
* 接入 `cgroup_set_weight/bandwidth` callback（若你 pin 的内核支持）([LKML][6])

---

### 5.2 `bpf/agentcg_mem.bpf.c`（memops：get_high_delay_ms + burst credit）

```c
// SPDX-License-Identifier: GPL-2.0
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include "agentcg_common.h"

char LICENSE[] SEC("license") = "GPL";

// memcg_bpf_ops 的 struct 定义来自你 pin 的 patch/kernel headers
// 这里用“概念骨架”：你实际要 include 对应 header。

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 65536);
  __type(key, __u64); // cgroup id
  __type(value, struct agentcg_cgrp_cfg);
} cfg_map SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 65536);
  __type(key, __u64); // cgroup id
  __type(value, struct agentcg_cgrp_rt);
} rt_map SEC(".maps");

static __always_inline __u32 class_factor(__u32 class)
{
  switch (class) {
  case AGENTCG_INTERACTIVE: return 1;
  case AGENTCG_LATENCY:     return 2;
  case AGENTCG_BATCH:       return 4;
  default:                  return 3;
  }
}

// 伪接口：真实 hook 里会给你 memcg 指针、current usage、high 等上下文
__u32 BPF_STRUCT_OPS(agentcg_get_high_delay_ms, struct cgroup *cgrp,
                     __u64 usage_bytes, __u64 high_bytes)
{
  __u64 cgid = bpf_get_current_cgroup_id(); // 实际应从 cgrp 派生稳定 id
  struct agentcg_cgrp_cfg *cfg = bpf_map_lookup_elem(&cfg_map, &cgid);
  struct agentcg_cgrp_rt  *rt  = bpf_map_lookup_elem(&rt_map, &cgid);
  if (!cfg || !rt || high_bytes == 0)
    return 0;

  if (usage_bytes <= high_bytes)
    return 0;

  __u64 over = usage_bytes - high_bytes;

  // burst credit（这里偷懒用 cfg->burst_credit_us 当 bytes，实际应有 burst_credit_bytes 字段）
  __u64 credit = cfg->burst_credit_us; // TODO: rename to burst_credit_bytes
  if (over <= credit)
    return 0;

  __u64 ratio_x1000 = ((over - credit) * 1000) / high_bytes; // 0..?
  __u32 fac = class_factor(cfg->class);

  __u64 delay = (ratio_x1000 * fac); // ms 的一个线性基
  if (delay > 200) delay = 200;

  // 写回 mem penalty 给 CPU 调度用（联合策略）
  rt->mem_penalty = (delay >= 50) ? 2 : (delay >= 10) ? 1 : 0;
  bpf_map_update_elem(&rt_map, &cgid, rt, BPF_ANY);

  return (__u32)delay;
}
```

**注意：memcg_bpf_ops hook 名称/签名**你必须跟你 pin 的 patch 对齐。hook 列表与语义可直接引用 RFC/LWN。([Spinics][2])

---

## 6. 构建与运行步骤（一步步命令）

下面我给两套路径：**CPU-only（更容易）** 和 **CPU+MEM（全量，依赖 memcg patch）**。

### 6.1 环境准备（通用）

以 Ubuntu 为例（你可以换发行版，但包名类似）：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential clang llvm lld pahole \
  libelf-dev zlib1g-dev pkg-config \
  linux-tools-common linux-tools-generic \
  stress-ng jq python3 python3-pip
pip3 install pandas matplotlib
```

确保 cgroup v2：

```bash
mount | grep cgroup2 || (echo "cgroup v2 not mounted"; exit 1)
```

### 6.2 Kernel（CPU：sched_ext）

sched_ext 文档给出了必须的 kernel config（BPF、BTF、SCHED_CLASS_EXT 等）。([Kernel Documentation][4])

**方案 A：用你已有带 sched_ext 的内核**
（如果你实验机内核已支持 `CONFIG_SCHED_CLASS_EXT`，可以跳过编译。）

验证：

```bash
grep -q "CONFIG_SCHED_CLASS_EXT=y" /boot/config-$(uname -r) && echo OK
ls /sys/kernel/sched_ext 2>/dev/null && echo "sched_ext present"
```

**方案 B：自己编译内核（更可复现）**

```bash
git clone https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git
cd linux
# 选择你 pin 的版本/tag，论文里写死
git checkout <YOUR_PINNED_TAG_OR_COMMIT>

# 配置：至少确保以下选项
# CONFIG_BPF=y
# CONFIG_BPF_SYSCALL=y
# CONFIG_BPF_JIT=y
# CONFIG_DEBUG_INFO_BTF=y
# CONFIG_SCHED_CLASS_EXT=y
make olddefconfig
scripts/config --enable BPF --enable BPF_SYSCALL --enable BPF_JIT --enable DEBUG_INFO_BTF --enable SCHED_CLASS_EXT
make -j$(nproc)
sudo make modules_install install
sudo reboot
```

### 6.3 Kernel（MEM：memcg_bpf_ops / memops）

如果你要做论文“in-kernel memory policy”，你需要 memcg_bpf_ops patch（当前公开是 RFC 系列）。([Spinics][2])

推荐用 bpf-next / linux-next 并应用 patch series（示例）：

```bash
git clone https://git.kernel.org/pub/scm/linux/kernel/git/bpf/bpf-next.git
cd bpf-next
# checkout 你要固定的日期/commit
git checkout <PINNED_COMMIT>

# 应用 memcg_bpf_ops patch series（示例：用 mbox）
# 你需要把 spinics/patchew 上的 mbox 下载到本地再 git am
# git am < memcg_bpf_ops_series.mbox

make olddefconfig
make -j$(nproc)
sudo make modules_install install
sudo reboot
```

> 论文里必须写清楚：你是基于哪个 commit + 哪个 patch series（RFC vX）。
> hook 列表与集成点可引用公开 patch/LWN（见上）。([Spinics][2])

### 6.4 编译并加载 BPF scheduler（最小路径：用 kernel tools/sched_ext）

kernel 文档说明 `tools/sched_ext` 下有示例 scheduler 和构建方式。([Kernel Documentation][4])

你可以直接在你的 pin 的 kernel tree 里：

```bash
cd linux/tools/sched_ext
make -j$(nproc)
# 你可以先跑 scx_simple 验证 sched_ext OK
sudo ./build/bin/scx_simple
# Ctrl+C 退出
```

然后把 `agentcg_scx` 的用户态 loader（daemon 里实现）加载进内核。

### 6.5 启动 agentcgroupd（daemon）

daemon 的职责：

* 加载 `agentcg_scx.bpf.o` 与 `agentcg_mem.bpf.o`（如果有）
* attach struct_ops（sched_ext_ops + memcg_bpf_ops）
* 读 config.yaml，初始化默认 class/权重/配额映射
* 启动 ringbuf 读 events
* 暴露 HTTP `/metrics`

运行示例：

```bash
sudo ./daemon/agentcgroupd --config ./daemon/config.yaml
```

---

## 7. Eval Plan（可一步步执行 + 脚本骨架 + 明确 RQ）

下面是一套**系统论文可接受**的 eval 结构。你可以照搬到论文里。

### 7.1 Research Questions（RQ）

* **RQ1（Isolation）**：在多租户 agent/workload 并发时，AgentCgroup 是否能显著降低受害者 step/session 的 p99/p999 latency 和 SLO violation，相比：

  1. static cgroup knobs（CFS + cpu.max/memory.high）
  2. userspace controller（PSI/metrics → 改 cgroup 文件）
* **RQ2（Timescale）**：in-kernel enforcement 的 “event→decision” 延迟分布是否显著小于 userspace loop（10–100ms），并且这种差异能解释 tail 改善？
* **RQ3（Joint policy）**：CPU+MEM 联合策略是否优于“只做 CPU 或只做 MEM”，在 reclaim-driven tail amplification、fork storm 等场景下差距多大？
* **RQ4（Overhead & safety）**：BPF 加载/验证成本、运行时开销、以及 fail-safe（sched_ext abort 回退）是否可接受？([Kernel Documentation][4])

### 7.2 Baselines（必须写清楚）

1. **Static**：

   * CFS
   * `cpu.max`, `cpu.weight`, `memory.high`, `memory.max` 固定
   * 不跑任何 BPF scheduler/memops
2. **Userspace control**（Senpai-like）：

   * 10ms/50ms/100ms 周期读取 PSI 或 memory.events
   * 根据阈值写 `memory.high`/`cpu.max` 或调整权重
3. **AgentCgroup-CPU**：只启用 sched_ext（flatcg 改造）
4. **AgentCgroup-Full**：sched_ext + memops + joint coupling
5. **Ablation**：Full 但关闭 coupling（mem_penalty 不回写给 CPU）

### 7.3 Workloads（可复现的“agent tool-call”替身）

你论文里说“compiler/data/web tool calls”，这里把它落实为可脚本化：

* **W1 Code/Compile burst**：clang/gcc 编译一个中型项目 + 单测（fork/exec 多、短 burst 多）

  * 或先用 `stress-ng --class cpu --fork N` 作为替身

* **W2 Data/Memory swing**：Python + pandas 做 groupby/join（或者 `stress-ng --vm --vm-bytes`）

* **W3 Web/Browser**：Playwright/Chromium headless（或用 `stress-ng --iomix`/`--cpu` 混合替身）

* **Noisy neighbors**（对抗测试）：

  * CPU spin：`stress-ng --cpu 1 --cpu-method all --timeout 30s`
  * fork storm：`stress-ng --fork 64 --timeout 30s`
  * mem blow-up：`stress-ng --vm 4 --vm-bytes 80% --vm-keep --timeout 30s`

### 7.4 指标（必须对齐内核语义）

* Latency：step p95/p99/p999，SLO violation rate
* Completion rate：session/step 是否完成
* Interference amplification：在 noisy neighbor 下受害者 slowdown
* Reaction time：

  * **in-kernel**：在 BPF 里用 `bpf_ktime_get_ns()` 打点（event→decision）
  * **userspace**：PSI/event 采样时间戳 + 写文件生效时间戳
* Memory events：`memory.events.local` 的 `high/max/oom/oom_kill`（解释 controlled throttling vs hard failure）([Kernel Documentation][3])
* Overhead：perf stat（context switches、cycles）、BPF load/verify 时间

### 7.5 Step-by-step 执行脚本（骨架）

#### 7.5.1 `bench/setup_env.sh`

```bash
#!/usr/bin/env bash
set -euxo pipefail

CGROOT=/sys/fs/cgroup/agentcg
sudo mkdir -p $CGROOT
# 建议 enable 你需要的 controllers
# echo "+cpu +memory" > /sys/fs/cgroup/cgroup.subtree_control (需要在可写父层级)
```

#### 7.5.2 `bench/run_baseline_static.sh`

```bash
#!/usr/bin/env bash
set -euxo pipefail
CG=/sys/fs/cgroup/agentcg/tenantA
sudo mkdir -p $CG
echo "max 100000" | sudo tee $CG/cpu.max
echo "100"        | sudo tee $CG/cpu.weight
echo $((2*1024*1024*1024)) | sudo tee $CG/memory.high
echo $((3*1024*1024*1024)) | sudo tee $CG/memory.max

# 运行 workload（示例：mem blow-up）
sudo cgexec -g cpu,memory:agentcg/tenantA stress-ng --vm 2 --vm-bytes 2G --vm-keep --timeout 30s --metrics-brief
```

#### 7.5.3 `bench/run_agentcgroup_full.sh`

```bash
#!/usr/bin/env bash
set -euxo pipefail

# 1) 启动 daemon（后台）
sudo ./daemon/agentcgroupd --config ./daemon/config.yaml &

# 2) 创建 tenant/session/step cgroups（可以用 agentcgctl 包一层）
sudo ./runtime/agentcgctl create-tenant tenantA
sudo ./runtime/agentcgctl start-session tenantA sess1
sudo ./runtime/agentcgctl start-step tenantA sess1 step1 --class latency --cpu-weight 2000 --mem-high 2G --mem-max 3G

# 3) 在 step cgroup 中运行 tool-call
sudo cgexec -g cpu,memory:agentcg/tenantA/sess1/step1 ./bench/workloads/code_compile.sh

# 4) 结束 step/session
sudo ./runtime/agentcgctl end-step tenantA sess1 step1
sudo ./runtime/agentcgctl end-session tenantA sess1
```

#### 7.5.4 `bench/collect/collect_metrics.sh`

```bash
#!/usr/bin/env bash
set -euxo pipefail
OUT=${1:-results/$(date +%s)}
mkdir -p "$OUT"

# cgroup 侧：memory.events.local (high/max/oom/oom_kill)
grep . /sys/fs/cgroup/agentcg/tenantA/**/memory.events.local | tee "$OUT/memory_events.txt" || true

# sched_ext 状态
cat /sys/kernel/sched_ext/state | tee "$OUT/scx_state.txt" || true

# daemon metrics（假设 9090）
curl -s http://127.0.0.1:9090/metrics > "$OUT/metrics.prom"
```

#### 7.5.5 `bench/collect/parse_logs.py`（把 ringbuf JSON 解析成 CSV）

你把 ringbuf event 输出成 JSON lines，然后用 pandas 解析出：

* event timestamp
* cgroup id
* decision type（delay、penalty、throttle）
* decision latency 等

---

## 8. 你论文里要怎么写“完整故事线”并与实现对齐

你现在的 story line（agent tool-call → domain/timescale mismatch → in-kernel eBPF enforcement → eval）是对的。把它压实的关键是：

1. **Domain mismatch**：用 cgroup 树 + step 子树把语义落地（并引用 freeze/kill/memory.events 等具体语义）。([Kernel Documentation][3])
2. **Timescale mismatch**：用 RQ2 的 microbench 图（event→decision）证明 userspace loop 慢；然后强调 sched_ext/memops 在 enforcement point 上决策。([Kernel Documentation][4])
3. **兼容性/工业化**：承认 `cpu.max` 在 sched_ext 下默认不生效这一点，并给出两条路：partial rollout + 自己实现 bandwidth。([Kernel Documentation][3])
4. **Fail-safe/风险控制**：sched_ext fail-safe + ABI instability（pin kernel）。([Kernel Documentation][4])
5. **memops 状态**：如果你依赖 memcg_bpf_ops RFC patch，论文中明确“prototype on patched kernel”，并给 artifact。([Spinics][2])

---

## 9. 你下一步最该做的 3 件事（能最快把“宣言式”变成“可复现系统”）

1. **把 CPU 侧做成一个能跑的 sched_ext scheduler**

   * 先不追求全部功能：先跑通 “cgroup DSQ + basic fairness + ringbuf metrics”
   * 再加 mem_penalty coupling
2. **把 memops 侧 pin 到一个明确 patch series + commit**

   * 把 hook 签名对齐
   * 先只实现 `get_high_delay_ms`，不碰 below_low/min（避免过度承诺）
3. **把 eval 的 RQ2 microbench 做出来**

   * 这是你论文最硬的证据：in-kernel decision latency vs userspace loop
   * 先用 synthetic burst（stress-ng）就能出图

---

如果你愿意，我可以在你“pin 的 kernel 版本/commit（以及是否启用 cgroup_set_weight/bandwidth callbacks）”这个前提下，把上面的 BPF 代码骨架**补齐到可编译可运行**（包括：runnable cgroup heap、stopping 记账、DSQ 生命周期、daemon 的 libbpf loader、以及 bench 脚本能直接产出 CSV/图）。但即使你现在不提供 pin 信息，你也可以先按我给的“CPU-only + partial mode + ringbuf metrics”路径把 MVP 跑通，然后再把 memops patch 集成进去。

[1]: https://github.com/systemd/systemd/blob/main/docs/CGROUP_DELEGATION.md "https://github.com/systemd/systemd/blob/main/docs/CGROUP_DELEGATION.md"
[2]: https://www.spinics.net/lists/netdev/msg1155691.html "https://www.spinics.net/lists/netdev/msg1155691.html"
[3]: https://docs.kernel.org/admin-guide/cgroup-v2.html "https://docs.kernel.org/admin-guide/cgroup-v2.html"
[4]: https://docs.kernel.org/scheduler/sched-ext.html "https://docs.kernel.org/scheduler/sched-ext.html"
[5]: https://www.kernel.org/doc/html/next/scheduler/sched-ext.html "https://www.kernel.org/doc/html/next/scheduler/sched-ext.html"
[6]: https://lkml.org/lkml/2025/9/2/2024 "https://lkml.org/lkml/2025/9/2/2024"
[7]: https://github.com/containerd/nri "https://github.com/containerd/nri"
[8]: https://lwn.net/Articles/1055698/ "https://lwn.net/Articles/1055698/"
