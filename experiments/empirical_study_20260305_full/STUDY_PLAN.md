# Empirical Study Plan (2026-03-05)

## Scope
- Goal: run an end-to-end empirical study across multiple SWE-bench repo images.
- Images (local, preloaded):
  1. swerebench/sweb.eval.x86_64.encode_1776_starlette-1147
  2. swerebench/sweb.eval.x86_64.bachmann1234_1776_diff_cover-210
  3. swerebench/sweb.eval.x86_64.azure_1776_msrest-for-python-224
  4. swerebench/sweb.eval.x86_64.pytorch_1776_ignite-1077

## Runtime Configuration
- Runner: scripts/run_swebench_new.py
- Model: haiku
- Tracing: --trace-all --trace-cgroup-children --trace-resources --resource-detail --sample-interval 100
- Resource monitor interval: 0.5s
- Output root: experiments/empirical_study_20260305_full/runs

## Data Collected
For each run:
- run_manifest.json
- ebpf_trace.jsonl (+ stderr/pid)
- swebench/results.json
- swebench/resources.json
- swebench/tool_calls.json

## Analysis Dimensions
1. Static image dependency overlap:
   - layer overlap, dpkg overlap, testbed pip overlap
   - size hotspots (/opt/conda, /root/.cache, env/testbed)
2. Dynamic runtime behavior:
   - runtime duration, tool-call count
   - eBPF event mix and summary counts/bytes
   - FS pressure, write volume, process hotspots
3. Cross-image comparison and optimization opportunities

## Deliverables
- analysis_summary.json
- figures/
- EMPIRICAL_STUDY.md (paper-style writeup)
