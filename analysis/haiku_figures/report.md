# AgentCgroup SWE-Bench Experiment Analysis Report

Generated: 2026-02-05 23:30:44

Data source: `/home/yunwei37/agentcgroup/experiments/batch_swebench_18tasks`

Total tasks analyzed: 18


## Dataset Overview

| Metric | Value |
|--------|-------|
| Total tasks | 18 |
| Successful | 17 (94.4%) |
| Total execution time | 10161.9s (169.4 min) |
| Categories | 6 |
| Difficulty levels | 3 |

## RQ1: Resource Usage Dynamics (Time-scale Mismatch)

**Research Question**: How dynamic are resource changes during AI agent execution?

**Paper Claim**: User-space controllers react in 10-100ms, but resource changes happen at millisecond scale.


### Findings

- **Total burst events detected**: 200
- **Tasks with bursts**: 18 / 18

**CPU Change Rate Statistics (%/sec)**:
- Mean: 0.48
- Max: 34.05
- 95th percentile: 1.18

![Resource Time Series](figures/rq1_resource_timeseries.png)

![Change Rate Distribution](figures/rq1_change_rate_distribution.png)

## RQ2: Resource Usage by Category (Domain Mismatch)

**Research Question**: Do different task categories have significantly different resource needs?

**Paper Claim**: Static resource limits cannot adapt to different workloads.


### Memory Usage by Category

| Category | Avg Memory (MB) | Peak Memory (MB) |
|----------|-----------------|------------------|
| CLI_Tools | 236.3 | 452.1 |
| DevOps_Build | 244.6 | 633.2 |
| ML_Scientific | 236.2 | 1403.0 |
| Medical_Bio | 247.5 | 4060.0 |
| SQL_Data | 261.1 | 392.5 |
| Web_Network | 237.1 | 308.4 |

![Category Box Plots](figures/rq2_category_boxplots.png)

![Category Heatmap](figures/rq2_category_heatmap.png)

## RQ3: Tool Call Patterns

**Research Question**: What is the relationship between tool calls and resource consumption?


### Top Tools by Execution Time

| Tool | Call Count | Total Time (s) | Avg Time (s) |
|------|------------|----------------|--------------|
| Bash | 529 | 1375.86 | 2.60 |
| Task | 18 | 1058.35 | 58.80 |
| TaskOutput | 4 | 210.19 | 52.55 |
| WebFetch | 2 | 9.98 | 4.99 |
| Read | 160 | 7.92 | 0.05 |
| Edit | 71 | 3.41 | 0.05 |
| TodoWrite | 84 | 2.68 | 0.03 |
| Grep | 13 | 0.87 | 0.07 |
| Write | 4 | 0.15 | 0.04 |
| KillShell | 2 | 0.10 | 0.05 |

**Tool Time Ratio**: Mean 37.9%, Median 34.3%

![Tool Analysis](figures/rq3_tool_analysis.png)

## RQ4: Over-provisioning Analysis

**Research Question**: How much over-provisioning would static limits require?


### Over-provisioning Factors

| Metric | CPU Ratio | Memory Ratio |
|--------|-----------|--------------|
| Mean | 4.10x | 2.43x |
| Median | 3.94x | 1.36x |
| Max | 6.02x | 15.39x |
| 95th Percentile | 5.77x | 6.95x |

![Over-provisioning Analysis](figures/rq4_overprovisioning.png)

## Appendix: Tool Time Ratio Calculation Methodology

### Definition

**Tool Time Ratio** measures the proportion of time spent on tool execution versus total Claude execution time:

```
Tool Time Ratio = (Total Tool Execution Time / Claude Total Time) × 100%
```

### Calculation Steps

#### 1. Extract Tool Execution Times from trace.jsonl

Each tool call has two timestamps in the trace:
- **tool_use** (type=assistant): When Claude requests the tool call
- **tool_result** (type=user): When the tool returns its result

```
Tool Duration = tool_result.timestamp - tool_use.timestamp
```

#### 2. Per-Task Calculation

For each task:
```python
total_tool_time = sum(duration for each tool call)
thinking_time = claude_time - total_tool_time
tool_ratio = (total_tool_time / claude_time) × 100%
```

#### 3. Example (CLI_Tools_Easy)

| Metric | Value |
|--------|-------|
| Claude Time | 123.1s |
| Tool Calls | 20 |
| Total Tool Time | ~46.4s |
| **Tool Time Ratio** | **37.7%** |

### Interpretation

| Component | Percentage | Description |
|-----------|------------|-------------|
| Tool Time | 37.9% | Actual tool execution (Bash, Read, Edit, etc.) |
| Thinking Time | 62.1% | LLM inference/generation (waiting for API) |

### Key Insights

1. **Tool-Heavy Operations**: Bash commands dominate tool time (1375.86s total)
2. **Fast Local Operations**: Read (0.05s avg) and Edit (0.05s avg) are nearly instantaneous
3. **Variable Task Patterns**: Tool ratio ranges from 13.2% to 79.5% across tasks
4. **Resource Implications**: High tool-time tasks have more container resource usage bursts

---

## Key Conclusions

1. **Time-scale Mismatch**: Resource usage exhibits significant burstiness that exceeds
   the reaction time of typical user-space controllers.
2. **Domain Mismatch**: Different task categories show distinct resource profiles,
   making static limits suboptimal.
3. **Over-provisioning Waste**: Static provisioning at peak levels wastes significant resources,
   as average usage is typically much lower than peak.
4. **Tool Execution Dominance**: ~38% of execution time is spent on tool calls, with Bash
   commands being the primary resource consumer during active tool execution phases.