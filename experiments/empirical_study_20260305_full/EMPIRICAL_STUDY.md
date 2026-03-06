# Comprehensive Empirical Study: Cross-Repo SWE-bench Runtime and Image Overhead

- Generated at: 2026-03-05T17:28:07.603689
- Scope: 4 repo images, eBPF-enabled runs, plus static dependency/layer analysis.

## Abstract
We evaluate cross-repo variability in runtime behavior and storage overhead for SWE-bench container workloads.
Results show substantial heterogeneity in write volume and runtime. Static image analysis indicates large removable cache overhead and low overlap in testbed pip dependencies.

## Method
- Runtime runner: `scripts/run_swebench_new.py`
- Runtime flags: `--trace-all --trace-cgroup-children --trace-resources --resource-detail --sample-interval 100 --resource-monitor-interval 0.5`
- Model: `haiku`
- Dynamic metrics from: `results.json`, `resources.json`, `tool_calls.json`, `ebpf_trace.jsonl`
- Static metrics from: image inspect + `/opt/conda`, `/root/.cache/pip`, testbed pip sets

## What Is New vs Original AgentCgroup Paper Dataset
1. Syscall-level FS/NET/MMAP/SIGNAL process behavior via eBPF SUMMARY is now available (not only tool trace + CPU/memory).
2. Per-repo write amplification and FS pressure differences are directly quantifiable.
3. Image-internal reclaim opportunity can be estimated from real path-level size hotspots.
4. Cross-repo dependency overlap in the actual `testbed` env is measured (Jaccard matrix), informing image strategy.

## RQ1: Runtime and eBPF behavior
| repo | runtime_s | tool_calls | event_total | write_mb | event_per_s | write_mb_per_s | fs_summary_share_% | mem_avg_mb | cpu_avg_pct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| azure_msrest | 157.41 | 44 | 9349 | 16.53 | 59.39 | 0.10 | 95.61 | 334.35 | 11.35 |
| diffcover | 132.89 | 49 | 12637 | 29.33 | 95.09 | 0.22 | 95.42 | 335.69 | 19.12 |
| pytorch_ignite | 317.35 | 44 | 67000 | 16062.33 | 211.13 | 50.61 | 79.74 | 476.82 | 42.30 |
| starlette | 190.79 | 39 | 10116 | 18.01 | 53.02 | 0.09 | 94.73 | 333.18 | 11.21 |

Key findings:
1. Longest runtime is `pytorch_ignite` at 317.35s.
2. Largest write volume is `pytorch_ignite` at 16062.33 MB.
3. Write volume variability is high: max/min ratio is 971.9x.

## RQ2: Static dependency overlap and size hotspots
| repo | image_gb | layers | testbed_pip_count | /opt/conda MB | /opt/conda/pkgs MB | /root/.cache/pip MB | /opt/conda/envs/testbed MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| azure_msrest | 3.24 | 6 | 40 | 1970.00 | 923.00 | 115.00 | 273.00 |
| diffcover | 3.45 | 17 | 57 | 1916.00 | 865.00 | 113.00 | 260.00 |
| pytorch_ignite | 6.35 | 6 | 134 | 3375.00 | 955.00 | 478.00 | 1646.00 |
| starlette | 3.71 | 6 | 85 | 2053.00 | 912.00 | 172.00 | 367.00 |

Pip-overlap (testbed env, Jaccard):
| repo_a | repo_b | jaccard |
| --- | --- | ---: |
| azure_msrest | diffcover | 0.260 |
| azure_msrest | pytorch_ignite | 0.137 |
| azure_msrest | starlette | 0.202 |
| diffcover | pytorch_ignite | 0.165 |
| diffcover | starlette | 0.246 |
| pytorch_ignite | starlette | 0.177 |

Key findings:
1. `/opt/conda/pkgs` cache is consistently large (hundreds of MB per image).
2. `/root/.cache/pip` is non-trivial, especially for heavy ML image.
3. Testbed pip overlap is low-to-moderate, so full dependency unification is limited.

## RQ3: Optimization opportunities
- Low-risk immediate cleanup potential (conda pkg cache + pip cache): about 4533 MB across these 4 images.
- Medium-term: split image families (ML vs non-ML) and tune base layers separately.
- Long-term: standardize build pipeline for better cross-image layer reuse.

## Supplementary Robustness: Starlette Repeated Runs
- Repeated valid runs found: 5
| metric | mean | stddev | cv | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| total_time_s | 98.65 | 21.24 | 0.22 | 67.16 | 127.27 |
| event_total | 4714.60 | 1062.14 | 0.23 | 3738.00 | 6736.00 |
| write_mb | 25.86 | 6.42 | 0.25 | 17.27 | 31.99 |
| tool_calls | 26.80 | 5.71 | 0.21 | 21.00 | 37.00 |

## Figures
- `figures/fig1_image_sizes.png`
- `figures/fig2_runtime_toolcalls.png`
- `figures/fig3_write_volume_log.png`
- `figures/fig4_pip_overlap_heatmap.png`
- `figures/fig5_space_hotspots.png`
- `figures/fig6_event_mix_stacked.png`
- `figures/fig7_normalized_pressure.png`
- `figures/fig8_cache_reclaim.png`
- `figures/fig9_summary_heatmap.png`
- `figures/fig10_starlette_repeat_cv.png`

## Threats to Validity
- Single run per repo in this batch for cross-repo comparison; model behavior stochasticity remains.
- Runtime behavior includes agent decision variability, not only repo complexity.
- Storage metrics are image-level and path-level proxies, not block-level physical dedupe.
- One heavy case (`pytorch_ignite`) may dominate aggregate statistics; we report both raw and normalized views.

