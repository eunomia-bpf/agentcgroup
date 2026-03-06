# An Empirical Cross-Repository Study of Container Overheads in Agentic Software Engineering Tasks

## Abstract
For SWE-bench-style agentic software engineering tasks, container images are not just execution environments; they also directly shape resource pressure, experiment cost, and optimization headroom. However, the existing AgentCgroup dataset mainly covers tool-call traces and container-level resource curves, which is not sufficient to explain why different repositories still exhibit order-of-magnitude cost differences under similar task settings. This paper conducts a cross-repository empirical study over four representative repository images (`starlette`, `diffcover`, `azure_msrest`, and `pytorch_ignite`) using existing experiment artifacts only. All main experiments were run with `run_swebench_new.py` and `agentsight/bpf/process_new`, using hard container cgroup filtering, child-cgroup matching, and `100 ms` sampling to collect eBPF events, container resources, tool calls, and image dependency information. The results show that: (1) dynamic overhead is highly heterogeneous across repositories, with `pytorch_ignite` writing `16062.33 MB`, about `971.9x` the smallest sample; (2) the three lightweight repositories sustain `53.02-95.09 events/s` but only `0.09-0.22 MB/s` of write bandwidth, indicating metadata-heavy filesystem churn rather than bulk data writes; (3) cross-image `testbed` dependency overlap is low, with Jaccard similarity only `0.137-0.260`, only 17 packages shared by all four images, and 94 packages unique to `pytorch_ignite`; (4) cleaning only `conda/pip` caches can reclaim `4533 MB`, or `26.43%` of the current total image footprint; and (5) historical repeated runs of `starlette` show CV values of `0.21-0.25`, indicating that single-run observations are insufficient for strong statistical claims. Based on these findings, we propose a layered optimization path: short-term cache cleanup, mid-term image-family partitioning, and long-term repeated runs with phase-level attribution.

**Keywords**: LLM Agent; SWE-bench; eBPF; container image; dependency overlap; runtime overhead; empirical study

## 1. Introduction
The end-to-end latency and resource consumption of AI coding agents are determined not only by model inference, but also by container images, testing pipelines, and filesystem behavior. Existing studies usually report coarse-grained metrics such as success rate, tool-call count, average CPU, or average memory, but these metrics do not answer a more important question: under the same agent framework and similar execution protocol, why do different repositories incur drastically different runtime costs?

The root problem is insufficient observation granularity. If we only observe tool-call traces and container resource curves, we can tell that a run is "slow" or "heavy", but we cannot tell whether the cost comes from test-file churn, dependency write amplification, network connections, memory mappings, or dependency redundancy already embedded in the image. For a system such as AgentCgroup, which aims at OS-level control and isolation, the lack of syscall-level and image-level evidence directly limits the precision of downstream policy design.

Rather than expanding benchmark coverage, this paper focuses on explaining the current data well. Concretely, we jointly analyze cgroup-filtered eBPF events, container resources, tool calls, and image dependency structure from completed experiment artifacts, and construct a closed evidence chain from runtime behavior to actionable optimization guidance.

## 2. Research Questions and Contributions
### 2.1 Research Questions
- **RQ1**: How large are the cross-repository differences in runtime dynamic overhead?
- **RQ2**: What event structures and bottleneck patterns primarily explain these differences?
- **RQ3**: What do cross-repository dependency overlap and reclaimable image space look like?
- **RQ4**: Given the current repeated observations, how stable are the conclusions?

### 2.2 Main Contributions
- We provide syscall-level cross-repository behavioral differences using cgroup-filtered eBPF data, rather than stopping at tool-call or container-average metrics.
- We identify two distinct cost profiles: metadata-heavy filesystem churn in lightweight repositories, and high-volume write pressure mixed with cleanup activity in `pytorch_ignite`.
- We quantify both dependency overlap and storage redundancy, showing that a single unified dependency image has limited upside, while cache cleanup offers immediate and verifiable benefit.
- Using existing repeated runs of `starlette`, we estimate the scale of variability and make explicit which conclusions are robust observations versus which still need stronger statistical support.

## 3. Study Design
### 3.1 Subjects and Data Sources
This paper uses existing experiment artifacts only and adds no new runs. The main analysis covers four SWE-bench repository images:
- `swerebench/sweb.eval.x86_64.encode_1776_starlette-1147`
- `swerebench/sweb.eval.x86_64.bachmann1234_1776_diff_cover-210`
- `swerebench/sweb.eval.x86_64.azure_1776_msrest-for-python-224`
- `swerebench/sweb.eval.x86_64.pytorch_1776_ignite-1077`

Dynamic data comes from `runs/*/run_manifest.json`, `ebpf_trace.jsonl`, `swebench/results.json`, `resources.json`, and `tool_calls.json`. Static data comes from image inspect information, path sizes under `/opt/conda` and the `testbed` environment, and the `testbed` pip dependency sets. Repeatability data comes from five historical valid `starlette` runs under `branchfs_motivation`.

For `azure_msrest` and `pytorch_ignite`, we include only the retry runs with valid container cgroup filtering; earlier attempts without a valid cgroup path are excluded. This choice ensures that all main samples are drawn under comparable tracing conditions.

### 3.2 Tracing Configuration and Analysis Protocol
All main experiments use `run_swebench_new.py` to drive `agentsight/bpf/process_new`, with the following properties:
- The automatically discovered container cgroup is used as a hard filter.
- Child-cgroup matching is enabled to avoid missing execution chains spawned inside the container.
- `--trace-all`, `--trace-resources`, and `--resource-detail` are enabled.
- The eBPF sampling interval is `100 ms`, and container resource monitoring runs at `0.5 s`.
- Only runs with `claude_exit=0` and no main-run error are included.

This means that all dynamic metrics in this paper come from container-local events rather than host-level noise. This matters for cross-repository attribution, because unfiltered traces can easily be polluted by unrelated host or concurrent processes.

### 3.3 Metrics and Interpretation Boundaries
We use three categories of metrics:
- Dynamic metrics: `runtime_s`, `tool_calls`, `event_total`, `write_mb`, `event_per_s`, `write_mb_per_s`, `fs_summary_share`, average CPU, and average memory.
- Event-structure metrics: SUMMARY counts such as `WRITE`, `DIR_CREATE`, `FILE_TRUNCATE`, `FILE_DELETE`, `NET_CONNECT`, and `MMAP_SHARED`, plus average bytes per write.
- Static metrics: image size, layer count, `testbed` pip package count, path-level storage hotspots, reclaimable cache space, and cross-repository Jaccard overlap.

One boundary is important: the main cross-repository dataset currently contains only one valid run per repository. Accordingly, we interpret the results as scale-level observations and engineering attribution evidence, not as statistically significant comparative claims. Likewise, the five historical `starlette` runs are used only to estimate variability scale, not as strict same-configuration repeated trials.

## 4. Results and Analysis
### 4.1 RQ1: Dynamic Overhead Differs Substantially Across Repositories
**Table 1** reports the core dynamic metrics for the four repositories. `pytorch_ignite` is a clear heavyweight outlier in nearly every dimension: total runtime `317.35 s`, total events `67000`, write volume `16062.33 MB`, average CPU `42.30%`, and average memory `476.82 MB`. In contrast, the other three repositories write only `16.53-29.33 MB`.

Table 1: Cross-repository dynamic overhead

| repo | runtime_s | tool_calls | event_total | write_mb | event_per_s | write_mb_per_s | fs_summary_share_% | mem_avg_mb | cpu_avg_% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| azure_msrest | 157.41 | 44 | 9349 | 16.53 | 59.39 | 0.10 | 95.61 | 334.35 | 11.35 |
| diffcover | 132.89 | 49 | 12637 | 29.33 | 95.09 | 0.22 | 95.42 | 335.69 | 19.12 |
| pytorch_ignite | 317.35 | 44 | 67000 | 16062.33 | 211.13 | 50.61 | 79.74 | 476.82 | 42.30 |
| starlette | 190.79 | 39 | 10116 | 18.01 | 53.02 | 0.09 | 94.73 | 333.18 | 11.21 |

Two observations matter here. First, tool-call count does not directly predict cost. For example, `diffcover` has the largest tool-call count (49), yet both runtime and write volume remain far below `pytorch_ignite`. Second, for the three lightweight repositories `starlette`, `diffcover`, and `azure_msrest`, normalized event rates are `53.02-95.09 events/s`, while write bandwidth stays at only `0.09-0.22 MB/s`, with means of `69.17 events/s` and `0.14 MB/s`. This indicates that lightweight repositories are not "writing a lot of data"; they are generating frequent metadata events.

![Figure 1: Runtime and tool-call comparison](figures/fig2_runtime_toolcalls.png)

![Figure 2: Log-scale write volume comparison](figures/fig3_write_volume_log.png)

![Figure 3: Normalized runtime pressure comparison](figures/fig7_normalized_pressure.png)

RQ1 takeaway: the cross-repository difference is a genuine order-of-magnitude difference, and its dominant driver is not tool-call count itself, but the event intensity and write pattern triggered by each repository's internal execution pipeline.

### 4.2 RQ2: Bottleneck Patterns Split into Metadata Churn and High-Volume Data Pressure
To explain RQ1 in more detail, **Table 2** reports the core SUMMARY event structure. The three lightweight repositories show a very similar pattern: `fs_summary_share` stays around `95%`, both `DIR_CREATE` and `FILE_TRUNCATE` are high, and average write size is only `1.4-3.4 KB`. This suggests that their primary cost comes from metadata-heavy churn during testing and environment preparation, such as directory creation, temporary file truncation, and repeated result rewriting, rather than sustained large-file output.

`pytorch_ignite`, in contrast, shows a completely different profile: `WRITE=140078`, `FILE_DELETE=29329`, `MMAP_SHARED=27185`, and `NET_CONNECT=12641`, with an average write size of about `120237 B`, or roughly `35-87x` larger than the other repositories. Its filesystem-related SUMMARY share drops to `79.74%`, indicating that network connections, shared mappings, and process coordination materially contribute alongside filesystem activity.

Table 2: Core event structure and write-amplification characteristics

| repo | WRITE count | avg bytes/write | DIR_CREATE | FILE_TRUNCATE | FILE_DELETE | NET_CONNECT | fs_summary_share_% |
|---|---:|---:|---:|---:|---:|---:|---:|
| azure_msrest | 12597 | 1375.7 | 1524 | 3631 | 81 | 280 | 95.61 |
| diffcover | 9643 | 3188.9 | 3045 | 2607 | 1433 | 172 | 95.42 |
| pytorch_ignite | 140078 | 120237.1 | 8394 | 1710 | 29329 | 12641 | 79.74 |
| starlette | 5505 | 3431.4 | 2888 | 1880 | 136 | 98 | 94.73 |

This qualitative difference directly changes how bottlenecks should be interpreted. For lightweight repositories, the dominant cost looks like filesystem-management noise during test and build phases. For `pytorch_ignite`, the dominant cost is a composite execution pipeline made of high-volume writes, cleanup deletions, shared mappings, and connection activity. In other words, the problem in `pytorch_ignite` is not simply high CPU or high memory; the entire environment and test data path is heavier.

![Figure 4: Event mix comparison across repositories](figures/fig6_event_mix_stacked.png)

![Figure 5: SUMMARY metric heatmap](figures/fig9_summary_heatmap.png)

RQ2 takeaway: lightweight repositories are closer to metadata-heavy filesystem churn, whereas `pytorch_ignite` is closer to high-volume data-path pressure. These two classes need different optimization and control strategies.

### 4.3 RQ3: Dependency Overlap Is Limited, but Cache Cleanup Has Immediate Value
Static image analysis reveals two kinds of optimization space: the upper bound of cross-repository shared space, and the immediate payoff of low-risk redundancy cleanup within each image.

From the shared-space perspective, cross-repository `testbed` pip overlap is generally low, with Jaccard similarity only `0.137-0.260`. All four images share only 17 common packages, mostly generic packaging and testing components such as `attrs`, `pip`, `pytest`, `requests`, and `urllib3`. Meanwhile, `pytorch_ignite` alone contains 94 unique packages, which shows clear divergence between the ML dependency stack and the others. As a result, building one large unified base image does not naturally yield high reuse, and may instead spread heavyweight dependencies into lightweight repositories.

From the immediate-payoff perspective, `/opt/conda` dominates storage in all four images, accounting for `51.91%-59.31%` of total image size. Among the lightweight repositories, the `testbed` environment itself accounts for only `7.37%-9.65%` of image size, while for `pytorch_ignite` it reaches `25.32%`, which further indicates that the ML repository's main cost lies in the real execution environment rather than cache alone. Even so, cleaning only `/opt/conda/pkgs` and `/root/.cache/pip` can reclaim `4533 MB`, or `26.43%` of the total `17152.23 MB` image footprint.

Table 3: Image size, dependency scale, and reclaimable space

| repo | image_gb | layers | pip_count | unique_pip | /opt/conda MB | /opt/conda/envs/testbed MB | reclaim_mb | reclaim_pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| azure_msrest | 3.24 | 6 | 40 | 14 | 1970 | 273 | 1038 | 31.25% |
| diffcover | 3.45 | 17 | 57 | 24 | 1916 | 260 | 978 | 27.72% |
| pytorch_ignite | 6.35 | 6 | 134 | 94 | 3375 | 1646 | 1433 | 22.04% |
| starlette | 3.71 | 6 | 85 | 44 | 2053 | 367 | 1084 | 28.51% |

One subtle point is that `diffcover` has 17 layers, yet its total image size remains close to the other lightweight repositories. This suggests that layer count itself is not the dominant factor; dependency content and environment size matter much more than the raw number of layers.

![Figure 6: Image size comparison](figures/fig1_image_sizes.png)

![Figure 7: Dependency-overlap heatmap](figures/fig4_pip_overlap_heatmap.png)

![Figure 8: Storage hotspot distribution](figures/fig5_space_hotspots.png)

![Figure 9: Reclaimable cache space comparison](figures/fig8_cache_reclaim.png)

RQ3 takeaway: the theoretical gain of a unified dependency layer is limited by low overlap, but cache cleanup and image-family partitioning offer clear, immediate, and low-risk value.

### 4.4 RQ4: Current Repeat Observations Show Moderate Variability
The five historical valid runs of `starlette` provide an important signal: for the same repository, the coefficients of variation are `0.22` for runtime, `0.23` for total events, `0.25` for write volume, and `0.21` for tool-call count. This means a single repository run is not a mechanically stable process; the agent's decision path and execution itself show moderate variability.

This directly constrains the language we should use in the paper. We can robustly claim that cross-repository differences are large in scale, that dependency overlap is generally low, and that reclaimable cache space is real. But for finer-grained claims, such as `10%-20%` runtime differences between repositories or subtle ranking among lightweight repositories, the current sample size is not strong enough to justify strong comparative statements.

![Figure 10: CV statistics for repeated `starlette` runs](figures/fig10_starlette_repeat_cv.png)

RQ4 takeaway: the current repeated observations are enough to force tighter language, but not enough to replace formal repeated cross-repository trials.

## 5. Discussion
### 5.1 What This Adds Beyond the Original AgentCgroup Data
Compared with the earlier "tool trace + container resource" view, this study adds four kinds of information that the original data could not directly provide.

First, it introduces syscall-level behavior evidence under cgroup filtering, which allows us to distinguish metadata-heavy lightweight repositories from heavyweight repositories with large-volume write and cleanup activity. Second, it gives an explicit upper bound on dependency overlap, showing that a unified dependency-layer strategy will not naturally achieve high payoff. Third, it pushes image optimization down to path-level evidence and quantifies `4533 MB` of low-risk reclaimable space. Fourth, it uses historical repeated runs to show the interpretive limitation of single-run observations, preventing us from overstating one-off measurements as stable regularities.

### 5.2 Implications for System and Experiment Design
For engineering optimization, the current evidence supports a three-layer strategy.

First, short-term work should prioritize cache cleanup. It does not alter task semantics, the gain is deterministic, and it applies across all images. Second, mid-term work should split base images by dependency family, at minimum separating ML from non-ML repositories, to avoid spreading the `pytorch_ignite` dependency stack into lightweight repositories. Third, long-term work should add repeated runs and phase-level attribution so that AgentCgroup control policies can distinguish between metadata-churn repositories and high-volume data-path repositories.

For AgentCgroup itself, the study also suggests a more refined design direction: not every repository first needs CPU scheduling enhancements. For lightweight repositories, filesystem behavior is more explanatory; for heavyweight repositories, storage paths, dependency environments, and CPU/memory pressure jointly shape the bottleneck. A future control strategy is therefore better framed as repository-type-aware plus tool-phase-aware, rather than a single uniform control template for all tasks.

## 6. Threats to Validity
There are four main threats to validity.

First, the main cross-repository dataset currently contains only one valid run per repository, so we do not claim statistical significance across repositories. Second, the repeatability analysis for `starlette` comes from historical valid runs; it reflects variability scale, but not strict same-configuration reproduction. Third, the current write analysis is still based mainly on `SUMMARY:WRITE` aggregation, without a full path-level write mapping; this is strong enough to describe write scale and write pattern, but not to attribute every write to exact file paths. Fourth, the study covers only four repositories and one main model (`haiku`), so external validity still requires broader repository, model, and control-condition coverage.

## 7. Conclusion
Using existing experiment artifacts, this paper completes an empirical cross-repository analysis of container overheads for four SWE-bench repository images. The results show that cross-repository runtime differences are not only large, but structurally different: lightweight repositories are dominated by metadata-heavy filesystem churn, while heavyweight repositories exhibit a composite pressure pattern combining large-volume writes, cleanup deletions, mappings, and connections. At the same time, cross-repository dependency overlap is too low to justify a single unified dependency-layer strategy, but cache cleanup alone can immediately reclaim `4533 MB` and therefore has direct engineering value.

The core conclusion is not merely that one repository is heavier than another. It is that repositories become heavy in different ways, and that distinction is exactly the evidence AgentCgroup needs for subsequent system optimization.

## Appendix: Artifact Index
- Main statistics file: `analysis_summary.json`
- Figure 1: `figures/fig2_runtime_toolcalls.png`
- Figure 2: `figures/fig3_write_volume_log.png`
- Figure 3: `figures/fig7_normalized_pressure.png`
- Figure 4: `figures/fig6_event_mix_stacked.png`
- Figure 5: `figures/fig9_summary_heatmap.png`
- Figure 6: `figures/fig1_image_sizes.png`
- Figure 7: `figures/fig4_pip_overlap_heatmap.png`
- Figure 8: `figures/fig5_space_hotspots.png`
- Figure 9: `figures/fig8_cache_reclaim.png`
- Figure 10: `figures/fig10_starlette_repeat_cv.png`
