#!/usr/bin/env python3
"""
New Characterization Insights
==============================

Additional analyses for Section 3 of the AgentCgroup paper:
  1. Token consumption analysis  (trace.jsonl → input/output/cache tokens)
  2. Per-tool-call resource delta (tool_calls.json × resources.json alignment)
  3. Retry resource waste         (consecutive Bash failures → memory accumulation)
  4. Multi-tenant concurrency simulation (overlay real traces → peak memory)

Each function operates on a dataset directory (e.g. experiments/all_images_haiku)
and returns a results dict.  Charts are saved to comparison_figures/.

Usage:
    python analysis/analyze_new_insights.py               # full run
    python analysis/analyze_new_insights.py --analysis 1   # token only
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from filter_valid_tasks import get_valid_task_names

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

EXPERIMENTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "experiments"))
HAIKU_DIR = os.path.join(EXPERIMENTS_DIR, "all_images_haiku")
LOCAL_DIR = os.path.join(EXPERIMENTS_DIR, "all_images_local")
FIGURES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "comparison_figures"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_iso(ts_str):
    """Parse ISO timestamp string to datetime."""
    if ts_str is None:
        return None
    ts_str = ts_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return None


def _parse_mem_usage(mem_str):
    """Parse mem_usage string like '195.7MB / 134.5GB' → float (MB)."""
    if not mem_str:
        return 0.0
    try:
        val_part = mem_str.split("/")[0].strip()
        if val_part.upper().endswith("GIB") or val_part.upper().endswith("GB"):
            return float(val_part[:-3]) * 1024
        if val_part.upper().endswith("MIB") or val_part.upper().endswith("MB"):
            return float(val_part[:-3])
        if val_part.upper().endswith("KIB") or val_part.upper().endswith("KB"):
            return float(val_part[:-3]) / 1024
        return float(val_part)
    except (ValueError, IndexError):
        return 0.0


def _load_resource_samples(attempt_dir):
    """Load resource samples from resources.json (or results.json fallback).

    Returns list of dicts: [{epoch, mem_mb, cpu_pct}, ...]
    """
    samples = []
    # Try resources.json first
    rp = os.path.join(attempt_dir, "resources.json")
    if not os.path.exists(rp):
        rp = os.path.join(attempt_dir, "results.json")
    if not os.path.exists(rp):
        return samples

    with open(rp) as f:
        data = json.load(f)

    raw = data.get("samples", [])
    if not raw:
        raw = data.get("resource_samples", {}).get("samples", [])

    for s in raw:
        epoch = s.get("epoch", 0)
        mem_mb = _parse_mem_usage(s.get("mem_usage", ""))
        cpu_str = s.get("cpu_percent", "0")
        if isinstance(cpu_str, str):
            cpu_str = cpu_str.replace("%", "").strip()
        try:
            cpu_pct = float(cpu_str)
        except (ValueError, TypeError):
            cpu_pct = 0.0
        if epoch > 0:
            samples.append({"epoch": epoch, "mem_mb": mem_mb, "cpu_pct": cpu_pct})

    samples.sort(key=lambda x: x["epoch"])
    return samples


def _load_tool_calls(attempt_dir):
    """Load tool_calls.json → list of dicts with parsed datetimes."""
    tc_path = os.path.join(attempt_dir, "tool_calls.json")
    if not os.path.exists(tc_path):
        return []
    with open(tc_path) as f:
        raw = json.load(f)
    calls = []
    for c in raw:
        start = _parse_iso(c.get("timestamp"))
        end = _parse_iso(c.get("end_timestamp"))
        tool = c.get("tool", "Unknown")
        inp = c.get("input", {})
        calls.append({
            "tool": tool,
            "start": start,
            "end": end,
            "start_epoch": start.timestamp() if start else None,
            "end_epoch": end.timestamp() if end else None,
            "duration": (end - start).total_seconds() if start and end else 0,
            "input": inp,
        })
    return calls


def _section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ============================================================================
# 1. Token Consumption Analysis
# ============================================================================

def analyze_tokens(base_dir, label="Haiku"):
    """Parse trace.jsonl for all valid tasks → per-turn token usage stats.

    Focuses on per-turn metrics: context size growth, output per turn, cache
    efficiency.  Designed for Haiku (cloud API) where token data is meaningful.

    Returns dict with per-task, per-turn, and aggregate metrics.
    """
    _section(f"Token Analysis: {label}")

    valid = get_valid_task_names(base_dir)
    print(f"  Valid tasks: {len(valid)}")

    task_tokens = {}
    all_per_turn_context = []   # total context per turn (across all tasks)
    all_per_turn_output = []    # output tokens per turn
    all_per_turn_cache_hit = [] # cache hit rate per turn

    for name in valid:
        trace_path = os.path.join(base_dir, name, "attempt_1", "trace.jsonl")
        if not os.path.exists(trace_path):
            continue

        turns = []  # per-turn records for this task
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "assistant":
                    continue
                usage = rec.get("message", {}).get("usage", {})
                if not usage:
                    continue
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                cache_create = usage.get("cache_creation_input_tokens", 0)
                cache_read = usage.get("cache_read_input_tokens", 0)
                context = inp + cache_create + cache_read  # total context this turn
                total_input = inp + cache_create + cache_read
                cache_rate = cache_read / max(total_input, 1)
                turns.append({
                    "input_tokens": inp,
                    "output_tokens": out,
                    "cache_create": cache_create,
                    "cache_read": cache_read,
                    "context_size": context,
                    "cache_hit_rate": cache_rate,
                })

        if not turns:
            continue

        n_turns = len(turns)
        contexts = [t["context_size"] for t in turns]
        outputs = [t["output_tokens"] for t in turns]
        cache_rates = [t["cache_hit_rate"] for t in turns]
        total_output = sum(outputs)
        total_context = sum(contexts)

        task_tokens[name] = {
            "n_turns": n_turns,
            "total_output": total_output,
            "total_context": total_context,
            "avg_context_per_turn": statistics.mean(contexts),
            "max_context_per_turn": max(contexts),
            "avg_output_per_turn": statistics.mean(outputs),
            "avg_cache_hit_rate": statistics.mean(cache_rates),
            "context_growth": contexts[-1] - contexts[0] if n_turns > 1 else 0,
            "turns": turns,
        }

        all_per_turn_context.extend(contexts)
        all_per_turn_output.extend(outputs)
        all_per_turn_cache_hit.extend(cache_rates)

    if not task_tokens:
        print("  WARNING: No token data found")
        return {}

    # Aggregate
    all_turns = [t["n_turns"] for t in task_tokens.values()]
    all_total_output = [t["total_output"] for t in task_tokens.values()]
    all_avg_ctx = [t["avg_context_per_turn"] for t in task_tokens.values()]
    all_max_ctx = [t["max_context_per_turn"] for t in task_tokens.values()]
    all_avg_out = [t["avg_output_per_turn"] for t in task_tokens.values()]
    all_avg_cache = [t["avg_cache_hit_rate"] for t in task_tokens.values()]

    agg = {
        "n_tasks": len(task_tokens),
        "avg_turns": statistics.mean(all_turns),
        "median_turns": statistics.median(all_turns),
        "min_turns": min(all_turns),
        "max_turns": max(all_turns),
        "avg_context_per_turn": statistics.mean(all_per_turn_context),
        "median_context_per_turn": statistics.median(all_per_turn_context),
        "max_context_per_turn": max(all_per_turn_context),
        "avg_output_per_turn": statistics.mean(all_per_turn_output),
        "median_output_per_turn": statistics.median(all_per_turn_output),
        "avg_total_output": statistics.mean(all_total_output),
        "avg_cache_hit_rate": statistics.mean(all_per_turn_cache_hit),
    }

    print(f"\n  Tasks with token data: {agg['n_tasks']}")
    print(f"\n  --- Per-Turn Statistics (across all {len(all_per_turn_context)} turns) ---")
    print(f"  Context per turn:  avg {agg['avg_context_per_turn']:,.0f},  "
          f"median {agg['median_context_per_turn']:,.0f},  "
          f"max {agg['max_context_per_turn']:,.0f}")
    print(f"  Output per turn:   avg {agg['avg_output_per_turn']:,.0f},  "
          f"median {agg['median_output_per_turn']:,.0f}")
    print(f"  Cache hit rate:    avg {agg['avg_cache_hit_rate']:.1%}")
    print(f"\n  --- Per-Task Statistics ---")
    print(f"  Turns/task:   avg {agg['avg_turns']:.1f},  median {agg['median_turns']:.0f},  "
          f"range {agg['min_turns']}–{agg['max_turns']}")
    print(f"  Total output/task: avg {agg['avg_total_output']:,.0f}")

    # Context growth: how context size evolves across conversation
    # Normalize all tasks to 10 bins and show context growth
    n_bins = 10
    ctx_by_phase = [[] for _ in range(n_bins)]
    for t in task_tokens.values():
        turns = t["turns"]
        for i, turn in enumerate(turns):
            phase = min(int(i / len(turns) * n_bins), n_bins - 1)
            ctx_by_phase[phase].append(turn["context_size"])

    print(f"\n  Context size by conversation phase:")
    print(f"  {'Phase':<12} {'Avg Ctx':>10} {'Median':>10} {'N turns':>8}")
    for i, vals in enumerate(ctx_by_phase):
        if vals:
            print(f"  {i*10}-{(i+1)*10}%       {statistics.mean(vals):>10,.0f} "
                  f"{statistics.median(vals):>10,.0f} {len(vals):>8}")

    # Top tasks by turns
    sorted_tasks = sorted(task_tokens.items(), key=lambda x: x[1]["n_turns"], reverse=True)
    print(f"\n  Top 10 tasks by turns:")
    print(f"  {'Task':<48} {'Turns':>6} {'AvgCtx':>10} {'AvgOut':>8} {'Cache%':>7}")
    for name, t in sorted_tasks[:10]:
        short = name[:46]
        print(f"  {short:<48} {t['n_turns']:>6} {t['avg_context_per_turn']:>10,.0f} "
              f"{t['avg_output_per_turn']:>8,.0f} {t['avg_cache_hit_rate']:>6.1%}")

    return {"per_task": task_tokens, "aggregate": agg, "context_by_phase": ctx_by_phase}


# ============================================================================
# 2. Per-Tool-Call Resource Delta
# ============================================================================

def analyze_tool_burst_correlation(base_dir, label="Dataset"):
    """Analyze correlation between tool calls and CPU/memory bursts.

    For each resource sample, determines if it falls within a tool call.
    Computes:
    - What % of CPU/memory bursts happen during tool calls vs LLM thinking
    - Per-tool-type burst magnitude (peak CPU, peak memory spike)
    - Bash sub-category burst profiles (test exec, pip install, etc.)
    - Burst attribution: which tool types are responsible for extreme spikes
    """
    import re as _re

    _section(f"Tool-Burst Correlation: {label}")

    valid = get_valid_task_names(base_dir)
    print(f"  Valid tasks: {len(valid)}")

    def _categorize_bash(cmd):
        cl = cmd.strip().lower()
        if _re.search(r"\bpytest\b|\bpython\s+-m\s+pytest\b|\btox\b|\bnose\b|\bunittest\b", cl):
            return "Test Execution"
        if _re.search(r"\bpip\s+install\b|\bconda\s+install\b|\bapt\b|\byum\b", cl):
            return "Package Install"
        if _re.search(r"\bgit\s+(diff|log|status|show|add|commit|checkout|stash|branch|reset)\b", cl):
            return "Git Operations"
        if _re.search(r"\bpython\s+-c\b|\bpython3\s+-c\b", cl):
            return "Python Snippet"
        if _re.search(r"\bpython\b|\bpython3\b", cl):
            return "Python Run"
        if _re.search(r"\bls\b|\bfind\b|\btree\b|\bwc\b|\bdu\b|\bdf\b", cl):
            return "File Exploration"
        if _re.search(r"\bcat\b|\bhead\b|\btail\b|\bgrep\b|\bsed\b|\bawk\b", cl):
            return "Text Processing"
        return "Other"

    # Per-tool-type burst stats
    tool_mem_peaks = defaultdict(list)  # tool → [peak_mem_during_call, ...]
    tool_cpu_peaks = defaultdict(list)  # tool → [peak_cpu_during_call, ...]
    tool_mem_spikes = defaultdict(list) # tool → [peak - baseline, ...]
    tool_cpu_spikes = defaultdict(list) # tool → [peak - baseline, ...]

    # Bash sub-category stats
    bash_cat_mem = defaultdict(list)
    bash_cat_cpu = defaultdict(list)

    # Burst attribution: samples during tool vs LLM thinking
    total_samples = 0
    tool_samples = 0
    llm_samples = 0
    tool_burst_mem_samples = 0   # samples during tool calls with mem > threshold
    llm_burst_mem_samples = 0
    tool_burst_cpu_samples = 0   # samples during tool calls with cpu > threshold
    llm_burst_cpu_samples = 0

    # Thresholds for "burst" detection
    MEM_BURST_THRESHOLD = 300   # MB (above ~185MB baseline)
    CPU_BURST_THRESHOLD = 30    # % (above typical ~10% average)

    n_analyzed = 0

    for name in valid:
        attempt = os.path.join(base_dir, name, "attempt_1")
        samples = _load_resource_samples(attempt)
        calls = _load_tool_calls(attempt)

        if len(samples) < 10:
            continue

        epochs = np.array([s["epoch"] for s in samples])
        mems = np.array([s["mem_mb"] for s in samples])
        cpus = np.array([s["cpu_pct"] for s in samples])
        n_analyzed += 1

        # Build a mask: for each sample, is it during a tool call?
        in_tool = np.zeros(len(samples), dtype=bool)
        sample_tool_type = [None] * len(samples)
        sample_bash_cat = [None] * len(samples)

        for tc in calls:
            if tc["start_epoch"] is None or tc["end_epoch"] is None:
                continue
            mask = (epochs >= tc["start_epoch"]) & (epochs <= tc["end_epoch"])
            in_tool |= mask
            for idx in np.where(mask)[0]:
                sample_tool_type[idx] = tc["tool"]
                if tc["tool"] == "Bash":
                    cmd = tc["input"].get("command", "")
                    sample_bash_cat[idx] = _categorize_bash(cmd)

        # Count burst attribution
        for i in range(len(samples)):
            total_samples += 1
            is_burst_mem = mems[i] > MEM_BURST_THRESHOLD
            is_burst_cpu = cpus[i] > CPU_BURST_THRESHOLD

            if in_tool[i]:
                tool_samples += 1
                if is_burst_mem:
                    tool_burst_mem_samples += 1
                if is_burst_cpu:
                    tool_burst_cpu_samples += 1
            else:
                llm_samples += 1
                if is_burst_mem:
                    llm_burst_mem_samples += 1
                if is_burst_cpu:
                    llm_burst_cpu_samples += 1

        # Per-tool-call burst magnitude
        for tc in calls:
            if tc["start_epoch"] is None or tc["end_epoch"] is None:
                continue
            if tc["duration"] < 0.1:
                continue

            during_mask = (epochs >= tc["start_epoch"]) & (epochs <= tc["end_epoch"])
            if not during_mask.any():
                continue

            # Baseline: sample just before the call
            before_mask = epochs < tc["start_epoch"]
            if before_mask.any():
                baseline_mem = mems[before_mask][-1]
                baseline_cpu = cpus[before_mask][-1]
            else:
                baseline_mem = mems[during_mask][0]
                baseline_cpu = cpus[during_mask][0]

            peak_mem = mems[during_mask].max()
            peak_cpu = cpus[during_mask].max()
            mem_spike = peak_mem - baseline_mem
            cpu_spike = peak_cpu - baseline_cpu

            tool_mem_peaks[tc["tool"]].append(float(peak_mem))
            tool_cpu_peaks[tc["tool"]].append(float(peak_cpu))
            tool_mem_spikes[tc["tool"]].append(float(mem_spike))
            tool_cpu_spikes[tc["tool"]].append(float(cpu_spike))

            if tc["tool"] == "Bash":
                cmd = tc["input"].get("command", "")
                cat = _categorize_bash(cmd)
                bash_cat_mem[cat].append(float(mem_spike))
                bash_cat_cpu[cat].append(float(cpu_spike))

    print(f"  Tasks analyzed: {n_analyzed}")
    print(f"  Total resource samples: {total_samples}")

    # ---- Burst Attribution ----
    total_burst_mem = tool_burst_mem_samples + llm_burst_mem_samples
    total_burst_cpu = tool_burst_cpu_samples + llm_burst_cpu_samples
    tool_time_pct = tool_samples / max(total_samples, 1) * 100

    print(f"\n  --- Burst Attribution ---")
    print(f"  Time in tool calls: {tool_samples}/{total_samples} "
          f"({tool_time_pct:.1f}% of samples)")
    print(f"  Time in LLM thinking: {llm_samples}/{total_samples} "
          f"({100 - tool_time_pct:.1f}% of samples)")

    if total_burst_mem > 0:
        tool_mem_pct = tool_burst_mem_samples / total_burst_mem * 100
        print(f"\n  Memory bursts (>{MEM_BURST_THRESHOLD}MB):")
        print(f"    During tool calls:  {tool_burst_mem_samples} ({tool_mem_pct:.1f}%)")
        print(f"    During LLM thinking: {llm_burst_mem_samples} ({100 - tool_mem_pct:.1f}%)")
        print(f"    → Tool calls produce {tool_mem_pct / max(tool_time_pct, 0.1):.1f}x "
              f"more mem bursts than expected by time share")

    if total_burst_cpu > 0:
        tool_cpu_pct = tool_burst_cpu_samples / total_burst_cpu * 100
        print(f"\n  CPU bursts (>{CPU_BURST_THRESHOLD}%):")
        print(f"    During tool calls:  {tool_burst_cpu_samples} ({tool_cpu_pct:.1f}%)")
        print(f"    During LLM thinking: {llm_burst_cpu_samples} ({100 - tool_cpu_pct:.1f}%)")
        print(f"    → Tool calls produce {tool_cpu_pct / max(tool_time_pct, 0.1):.1f}x "
              f"more CPU bursts than expected by time share")

    # ---- Per-Tool Burst Magnitude ----
    print(f"\n  --- Per-Tool Burst Magnitude ---")
    print(f"  {'Tool':<15} {'N':>6} {'MemSpike':>10} {'MemP95':>10} "
          f"{'CpuSpike':>10} {'CpuP95':>10} {'PeakMem':>10} {'PeakCPU':>10}")

    tool_burst_summary = {}
    for tool in sorted(set(tool_mem_spikes.keys()) | set(tool_cpu_spikes.keys())):
        ms = tool_mem_spikes.get(tool, [])
        cs = tool_cpu_spikes.get(tool, [])
        n = max(len(ms), len(cs))
        if n == 0:
            continue
        summary = {
            "count": n,
            "mean_mem_spike": statistics.mean(ms) if ms else 0,
            "p95_mem_spike": float(np.percentile(ms, 95)) if ms else 0,
            "max_mem_spike": max(ms) if ms else 0,
            "mean_cpu_spike": statistics.mean(cs) if cs else 0,
            "p95_cpu_spike": float(np.percentile(cs, 95)) if cs else 0,
            "mean_peak_mem": statistics.mean(tool_mem_peaks.get(tool, [])) if tool_mem_peaks.get(tool) else 0,
            "mean_peak_cpu": statistics.mean(tool_cpu_peaks.get(tool, [])) if tool_cpu_peaks.get(tool) else 0,
        }
        tool_burst_summary[tool] = summary
        s = summary
        print(f"  {tool:<15} {n:>6} {s['mean_mem_spike']:>+10.1f} {s['p95_mem_spike']:>+10.1f} "
              f"{s['mean_cpu_spike']:>+10.1f} {s['p95_cpu_spike']:>+10.1f} "
              f"{s['mean_peak_mem']:>10.0f} {s['mean_peak_cpu']:>10.1f}")

    # ---- Bash Sub-Category Burst Profile ----
    print(f"\n  --- Bash Command Category Burst Profile ---")
    print(f"  {'Category':<20} {'N':>6} {'AvgMemSpike':>12} {'P95MemSpike':>12} "
          f"{'AvgCpuSpike':>12} {'P95CpuSpike':>12}")

    bash_summary = {}
    for cat in sorted(bash_cat_mem.keys()):
        ms = bash_cat_mem[cat]
        cs = bash_cat_cpu.get(cat, [])
        n = len(ms)
        if n < 2:
            continue
        bash_summary[cat] = {
            "count": n,
            "mean_mem_spike": statistics.mean(ms),
            "p95_mem_spike": float(np.percentile(ms, 95)),
            "max_mem_spike": max(ms),
            "mean_cpu_spike": statistics.mean(cs) if cs else 0,
            "p95_cpu_spike": float(np.percentile(cs, 95)) if cs else 0,
        }
        s = bash_summary[cat]
        print(f"  {cat:<20} {n:>6} {s['mean_mem_spike']:>+12.1f} {s['p95_mem_spike']:>+12.1f} "
              f"{s['mean_cpu_spike']:>+12.1f} {s['p95_cpu_spike']:>+12.1f}")

    return {
        "tool_burst_summary": tool_burst_summary,
        "bash_summary": bash_summary,
        "attribution": {
            "total_samples": total_samples,
            "tool_samples": tool_samples,
            "tool_time_pct": tool_time_pct,
            "mem_burst_tool_pct": tool_burst_mem_samples / max(total_burst_mem, 1) * 100,
            "mem_burst_llm_pct": llm_burst_mem_samples / max(total_burst_mem, 1) * 100,
            "cpu_burst_tool_pct": tool_burst_cpu_samples / max(total_burst_cpu, 1) * 100,
            "cpu_burst_llm_pct": llm_burst_cpu_samples / max(total_burst_cpu, 1) * 100,
            "mem_burst_concentration": (tool_burst_mem_samples / max(total_burst_mem, 1) * 100) / max(tool_time_pct, 0.1),
            "cpu_burst_concentration": (tool_burst_cpu_samples / max(total_burst_cpu, 1) * 100) / max(tool_time_pct, 0.1),
        },
        "n_analyzed": n_analyzed,
    }


# ============================================================================
# 3. Retry Resource Waste
# ============================================================================

def analyze_retry_waste(base_dir, label="Dataset"):
    """Quantify resource waste from agent retry cycles.

    A "retry group" = 3+ consecutive Bash calls (indicating test-fail-edit-retest).
    Measures memory accumulation and CPU time spent in retries.
    """
    _section(f"Retry Resource Waste: {label}")

    valid = get_valid_task_names(base_dir)
    print(f"  Valid tasks: {len(valid)}")

    task_retry_data = {}

    for name in valid:
        attempt = os.path.join(base_dir, name, "attempt_1")
        calls = _load_tool_calls(attempt)
        samples = _load_resource_samples(attempt)

        if not calls or len(samples) < 10:
            continue

        epochs = np.array([s["epoch"] for s in samples])
        mems = np.array([s["mem_mb"] for s in samples])

        # Identify retry groups: 3+ consecutive Bash calls
        retry_groups = []
        current_group = []
        for tc in calls:
            if tc["tool"] == "Bash" and tc["start_epoch"] is not None:
                current_group.append(tc)
            else:
                if len(current_group) >= 3:
                    retry_groups.append(current_group)
                current_group = []
        if len(current_group) >= 3:
            retry_groups.append(current_group)

        if not retry_groups:
            continue

        # For each retry group, measure:
        # - Total time consumed
        # - Memory at start of first call vs end of last call (accumulation)
        # - Peak memory within the group
        group_data = []
        for group in retry_groups:
            g_start = group[0]["start_epoch"]
            g_end = group[-1]["end_epoch"] or group[-1]["start_epoch"]
            total_bash_time = sum(tc["duration"] for tc in group)

            # Find memory at start and end
            before_mask = epochs <= g_start
            after_mask = epochs >= g_end
            during_mask = (epochs >= g_start) & (epochs <= g_end)

            mem_start = mems[before_mask][-1] if before_mask.any() else 0
            mem_end = mems[after_mask][0] if after_mask.any() else 0
            mem_peak = mems[during_mask].max() if during_mask.any() else 0
            mem_accum = mem_end - mem_start

            group_data.append({
                "n_calls": len(group),
                "total_time": total_bash_time,
                "elapsed_time": g_end - g_start if g_end > g_start else 0,
                "mem_start": mem_start,
                "mem_end": mem_end,
                "mem_peak": mem_peak,
                "mem_accumulation": mem_accum,
            })

        # Total retry time / total execution time
        results_path = os.path.join(attempt, "results.json")
        claude_time = 0
        if os.path.exists(results_path):
            with open(results_path) as f:
                res = json.load(f)
            claude_time = res.get("claude_time", 0)

        total_retry_time = sum(g["total_time"] for g in group_data)
        total_retry_accum = sum(max(0, g["mem_accumulation"]) for g in group_data)

        task_retry_data[name] = {
            "n_retry_groups": len(retry_groups),
            "total_retry_calls": sum(g["n_calls"] for g in group_data),
            "total_retry_time": total_retry_time,
            "retry_time_pct": total_retry_time / max(claude_time, 1) * 100,
            "total_mem_accumulation": total_retry_accum,
            "max_group_size": max(g["n_calls"] for g in group_data),
            "groups": group_data,
        }

    if not task_retry_data:
        print("  No retry groups found")
        return {}

    # Aggregate
    all_groups = [t["n_retry_groups"] for t in task_retry_data.values()]
    all_retry_time = [t["total_retry_time"] for t in task_retry_data.values()]
    all_retry_pct = [t["retry_time_pct"] for t in task_retry_data.values()]
    all_accum = [t["total_mem_accumulation"] for t in task_retry_data.values()]
    all_max_group = [t["max_group_size"] for t in task_retry_data.values()]

    agg = {
        "tasks_with_retries": len(task_retry_data),
        "avg_retry_groups": statistics.mean(all_groups),
        "max_retry_groups": max(all_groups),
        "avg_retry_time": statistics.mean(all_retry_time),
        "avg_retry_time_pct": statistics.mean(all_retry_pct),
        "avg_mem_accumulation": statistics.mean(all_accum),
        "max_mem_accumulation": max(all_accum),
        "max_group_size": max(all_max_group),
    }

    print(f"\n  Tasks with retry groups: {agg['tasks_with_retries']} / {len(valid)}")
    print(f"  Avg retry groups/task:  {agg['avg_retry_groups']:.1f} (max {agg['max_retry_groups']})")
    print(f"  Max consecutive retries: {agg['max_group_size']}")
    print(f"  Avg retry time/task:    {agg['avg_retry_time']:.1f}s ({agg['avg_retry_time_pct']:.1f}% of exec)")
    print(f"  Avg memory accumulation: {agg['avg_mem_accumulation']:.1f}MB")
    print(f"  Max memory accumulation: {agg['max_mem_accumulation']:.1f}MB")

    # Top tasks by retry impact
    sorted_tasks = sorted(task_retry_data.items(),
                          key=lambda x: x[1]["total_retry_time"], reverse=True)
    print(f"\n  Top 10 tasks by retry time:")
    print(f"  {'Task':<45} {'Groups':>7} {'MaxGrp':>7} {'Time':>8} {'%Exec':>7} {'MemΔ':>8}")
    for name, t in sorted_tasks[:10]:
        short = name[:43]
        print(f"  {short:<45} {t['n_retry_groups']:>7} {t['max_group_size']:>7} "
              f"{t['total_retry_time']:>7.0f}s {t['retry_time_pct']:>6.1f}% "
              f"{t['total_mem_accumulation']:>+7.0f}MB")

    return {"per_task": task_retry_data, "aggregate": agg}


# ============================================================================
# 4. Multi-Tenant Concurrency Simulation
# ============================================================================

def analyze_concurrency_simulation(base_dir_haiku, base_dir_local):
    """Simulate N concurrent agents using real memory traces.

    Approach:
    - Collect all memory traces (normalized to 100 points each)
    - For N = 2, 4, 8, 16, 32, 64: randomly sample N traces and overlay
    - Repeat 100 times for statistical stability
    - Compare: static allocation (N × peak) vs dynamic (actual peak of overlay)
    """
    _section("Multi-Tenant Concurrency Simulation")

    # Collect all memory traces
    all_traces = []
    all_peaks = []
    all_avgs = []

    for ds_name, base_dir in [("Haiku", base_dir_haiku), ("Local", base_dir_local)]:
        if not os.path.exists(base_dir):
            continue
        valid = get_valid_task_names(base_dir)
        for name in valid:
            attempt = os.path.join(base_dir, name, "attempt_1")
            samples = _load_resource_samples(attempt)
            if len(samples) < 10:
                continue
            mems = [s["mem_mb"] for s in samples]
            # Normalize to 100 points
            x_orig = np.linspace(0, 1, len(mems))
            x_new = np.linspace(0, 1, 100)
            interp = np.interp(x_new, x_orig, mems)
            all_traces.append(interp)
            all_peaks.append(max(mems))
            all_avgs.append(statistics.mean(mems))

    if not all_traces:
        print("  No memory traces found")
        return {}

    traces = np.array(all_traces)
    n_traces = len(traces)
    print(f"  Total memory traces: {n_traces}")
    print(f"  Avg peak memory: {statistics.mean(all_peaks):.0f}MB")
    print(f"  Max peak memory: {max(all_peaks):.0f}MB")
    print(f"  Avg avg memory: {statistics.mean(all_avgs):.0f}MB")

    # Simulate
    concurrency_levels = [2, 4, 8, 16, 32, 64]
    n_simulations = 200
    rng = np.random.default_rng(42)

    results = {}
    print(f"\n  {'N':>4} {'StaticAlloc':>12} {'DynPeak(avg)':>13} {'DynPeak(p95)':>13} "
          f"{'Savings%':>9} {'StatMuxGain':>12}")

    for N in concurrency_levels:
        static_allocs = []
        dynamic_peaks = []

        for _ in range(n_simulations):
            # Sample N traces (with replacement if N > n_traces)
            indices = rng.choice(n_traces, size=N, replace=True)
            selected = traces[indices]

            # Each trace starts at a random offset within the timeline
            # This simulates agents starting at different times
            shifted = np.zeros((N, 200))  # double length for shifting
            for i in range(N):
                offset = rng.integers(0, 100)
                shifted[i, offset:offset+100] = selected[i]

            # Peak of the overlaid traces at each time point
            total_mem = shifted.sum(axis=0)
            dynamic_peak = total_mem.max()

            # Static allocation: sum of individual peaks
            static_alloc = sum(traces[idx].max() for idx in indices)

            static_allocs.append(static_alloc)
            dynamic_peaks.append(dynamic_peak)

        avg_static = statistics.mean(static_allocs)
        avg_dynamic = statistics.mean(dynamic_peaks)
        p95_dynamic = float(np.percentile(dynamic_peaks, 95))
        savings = (1 - avg_dynamic / avg_static) * 100
        stat_mux = avg_static / avg_dynamic

        results[N] = {
            "static_alloc_avg": avg_static,
            "dynamic_peak_avg": avg_dynamic,
            "dynamic_peak_p95": p95_dynamic,
            "savings_pct": savings,
            "stat_mux_gain": stat_mux,
        }

        print(f"  {N:>4} {avg_static/1024:>11.1f}GB {avg_dynamic/1024:>12.1f}GB "
              f"{p95_dynamic/1024:>12.1f}GB {savings:>8.1f}% {stat_mux:>11.2f}x")

    # Also compute: at each N, how much memory does average-based allocation need?
    avg_mem = statistics.mean(all_avgs)
    peak_mem = max(all_peaks)
    print(f"\n  Reference points:")
    print(f"    Per-task avg memory: {avg_mem:.0f}MB")
    print(f"    Per-task max peak:   {peak_mem:.0f}MB")
    print(f"    128GB can hold (by peak): {128*1024 / peak_mem:.0f} instances")
    print(f"    128GB can hold (by avg):  {128*1024 / avg_mem:.0f} instances")

    return {
        "n_traces": n_traces,
        "avg_peak": statistics.mean(all_peaks),
        "max_peak": max(all_peaks),
        "avg_avg": avg_mem,
        "simulation": results,
    }


# ============================================================================
# 5. Token-Resource Correlation
# ============================================================================

def analyze_token_resource_correlation(base_dir, label="Dataset"):
    """Correlate token usage with resource consumption per task.

    Returns correlation coefficients between tokens and peak memory, CPU, etc.
    """
    _section(f"Token-Resource Correlation: {label}")

    valid = get_valid_task_names(base_dir)

    task_data = []
    for name in valid:
        attempt = os.path.join(base_dir, name, "attempt_1")

        # Load tokens from trace.jsonl
        trace_path = os.path.join(attempt, "trace.jsonl")
        if not os.path.exists(trace_path):
            continue
        total_output = 0
        n_turns = 0
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "assistant":
                    continue
                usage = rec.get("message", {}).get("usage", {})
                total_output += usage.get("output_tokens", 0)
                n_turns += 1

        if n_turns == 0:
            continue

        # Load resource summary
        rp = os.path.join(attempt, "results.json")
        if not os.path.exists(rp):
            continue
        with open(rp) as f:
            res = json.load(f)

        summary = res.get("resource_samples", {}).get("summary", {})
        mem_info = summary.get("memory_mb", {})
        cpu_info = summary.get("cpu_percent", {})
        claude_time = res.get("claude_time", 0)

        if not mem_info or claude_time <= 0:
            continue

        task_data.append({
            "name": name,
            "output_tokens": total_output,
            "n_turns": n_turns,
            "peak_mem": mem_info.get("max", 0),
            "avg_mem": mem_info.get("avg", 0),
            "avg_cpu": cpu_info.get("avg", 0),
            "peak_cpu": cpu_info.get("max", 0),
            "exec_time": claude_time,
        })

    if len(task_data) < 5:
        print("  Not enough data for correlation analysis")
        return {}

    # Compute correlations
    output_tokens = np.array([d["output_tokens"] for d in task_data])
    n_turns = np.array([d["n_turns"] for d in task_data])
    peak_mem = np.array([d["peak_mem"] for d in task_data])
    avg_mem = np.array([d["avg_mem"] for d in task_data])
    avg_cpu = np.array([d["avg_cpu"] for d in task_data])
    exec_time = np.array([d["exec_time"] for d in task_data])

    correlations = {
        "output_tokens_vs_peak_mem": float(np.corrcoef(output_tokens, peak_mem)[0, 1]),
        "output_tokens_vs_avg_mem": float(np.corrcoef(output_tokens, avg_mem)[0, 1]),
        "output_tokens_vs_avg_cpu": float(np.corrcoef(output_tokens, avg_cpu)[0, 1]),
        "output_tokens_vs_exec_time": float(np.corrcoef(output_tokens, exec_time)[0, 1]),
        "n_turns_vs_peak_mem": float(np.corrcoef(n_turns, peak_mem)[0, 1]),
        "n_turns_vs_exec_time": float(np.corrcoef(n_turns, exec_time)[0, 1]),
    }

    print(f"\n  Tasks with both token & resource data: {len(task_data)}")
    print(f"\n  Correlations:")
    for key, val in correlations.items():
        print(f"    {key:<40} r = {val:+.3f}")

    return {"task_data": task_data, "correlations": correlations}


# ============================================================================
# Charts
# ============================================================================

def generate_charts(token_haiku, burst_haiku, burst_local,
                    retry_haiku, retry_local, sim_results,
                    corr_haiku, corr_local):
    """Generate all new insight figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ---- Chart 1: Token distribution (Haiku only) ----
    if token_haiku:
        _chart_token_distribution(token_haiku)

    # ---- Chart 2: Tool burst correlation ----
    if burst_haiku or burst_local:
        _chart_tool_burst(burst_haiku, burst_local)

    # ---- Chart 3: Retry waste ----
    if retry_haiku or retry_local:
        _chart_retry_waste(retry_haiku, retry_local)

    # ---- Chart 4: Concurrency simulation ----
    if sim_results:
        _chart_concurrency_sim(sim_results)

    # ---- Chart 5: Token-resource correlation ----
    if corr_haiku or corr_local:
        _chart_token_resource_corr(corr_haiku, corr_local)


def _chart_token_distribution(token_h):
    """Token usage chart — Haiku only (2 subplots)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Context size growth across conversation phases
    ctx_by_phase = token_h.get("context_by_phase", [])
    if ctx_by_phase:
        phases = list(range(len(ctx_by_phase)))
        means = [statistics.mean(v) / 1000 if v else 0 for v in ctx_by_phase]
        medians = [statistics.median(v) / 1000 if v else 0 for v in ctx_by_phase]
        p25 = [float(np.percentile(v, 25)) / 1000 if len(v) > 1 else 0 for v in ctx_by_phase]
        p75 = [float(np.percentile(v, 75)) / 1000 if len(v) > 1 else 0 for v in ctx_by_phase]
        labels = [f"{i*10}%" for i in phases]

        ax1.fill_between(phases, p25, p75, alpha=0.3, color="#2196F3", label="P25–P75")
        ax1.plot(phases, means, "o-", color="#2196F3", linewidth=2, label="Mean")
        ax1.plot(phases, medians, "s--", color="#FF9800", linewidth=1.5, label="Median")
        ax1.set_xticks(phases)
        ax1.set_xticklabels(labels, fontsize=11)

    ax1.set_xlabel("Conversation Progress", fontsize=14)
    ax1.set_ylabel("Context Size (K tokens)", fontsize=14)
    ax1.set_title("(a) Context Size Growth", fontsize=15)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=12)

    # (b) Output tokens per turn distribution
    per_task = token_h.get("per_task", {})
    all_outputs = []
    for t in per_task.values():
        for turn in t.get("turns", []):
            all_outputs.append(turn["output_tokens"])

    if all_outputs:
        # Clip for visualization
        clipped = [min(o, 500) for o in all_outputs]
        ax2.hist(clipped, bins=30, alpha=0.75, color="#2196F3", edgecolor="white")
        avg_out = statistics.mean(all_outputs)
        med_out = statistics.median(all_outputs)
        ax2.axvline(avg_out, color="red", ls="--", lw=1.5,
                    label=f"Mean ({avg_out:.0f})")
        ax2.axvline(med_out, color="black", ls=":", lw=1.5,
                    label=f"Median ({med_out:.0f})")

    ax2.set_xlabel("Output Tokens per Turn", fontsize=14)
    ax2.set_ylabel("Number of Turns", fontsize=14)
    ax2.set_title(f"(b) Output Tokens per Turn (n={len(all_outputs)})", fontsize=15)
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=12)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "token_distribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _chart_tool_burst(burst_h, burst_l):
    """Tool-burst correlation chart (2 subplots)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Per-tool burst magnitude (memory spike, grouped bar for Haiku vs GLM)
    all_tools = set()
    for d in [burst_h, burst_l]:
        if d:
            all_tools.update(d.get("tool_burst_summary", {}).keys())
    # Filter to tools with ≥5 calls in either dataset
    tools_sorted = sorted(all_tools)
    tools_sorted = [t for t in tools_sorted
                    if (burst_h and burst_h.get("tool_burst_summary", {}).get(t, {}).get("count", 0) >= 5)
                    or (burst_l and burst_l.get("tool_burst_summary", {}).get(t, {}).get("count", 0) >= 5)]

    if tools_sorted:
        x = np.arange(len(tools_sorted))
        width = 0.35
        for i, (label, data, color) in enumerate([
            ("Haiku", burst_h, "#2196F3"), ("GLM", burst_l, "#4CAF50")
        ]):
            if not data:
                continue
            vals = [data.get("tool_burst_summary", {}).get(t, {}).get("mean_mem_spike", 0)
                    for t in tools_sorted]
            p95s = [data.get("tool_burst_summary", {}).get(t, {}).get("p95_mem_spike", 0)
                    for t in tools_sorted]
            ax1.bar(x + i * width, vals, width, label=f"{label} (mean)",
                    color=color, alpha=0.8)
            # Add P95 markers
            ax1.scatter(x + i * width, p95s, marker="_", color=color,
                        s=200, linewidths=2, zorder=5)

        ax1.set_xlabel("Tool Type", fontsize=14)
        ax1.set_ylabel("Memory Spike (MB)", fontsize=14)
        ax1.set_title("(a) Memory Spike by Tool Type", fontsize=15)
        ax1.set_xticks(x + width / 2)
        ax1.set_xticklabels(tools_sorted, fontsize=10, rotation=30, ha="right")
        ax1.legend(fontsize=12)
        ax1.axhline(0, color="black", lw=0.8)
        ax1.grid(axis="y", alpha=0.3)

    # (b) Bash sub-category burst profile (combined datasets)
    # Merge bash summaries from both datasets
    merged_bash_mem = defaultdict(list)
    merged_bash_cpu = defaultdict(list)
    for data in [burst_h, burst_l]:
        if not data:
            continue
        for cat, summary in data.get("bash_summary", {}).items():
            # We need raw values; approximate from summary stats
            merged_bash_mem[cat].append(summary)
            merged_bash_cpu[cat].append(summary)

    # Use the raw data approach - re-extract from tool_burst_summary if available
    # For simplicity, use the bash_summary directly
    bash_cats = sorted(set(
        list((burst_h or {}).get("bash_summary", {}).keys()) +
        list((burst_l or {}).get("bash_summary", {}).keys())
    ))
    if bash_cats:
        # Combine both datasets
        combined_mem = {}
        combined_cpu = {}
        combined_n = {}
        for cat in bash_cats:
            mem_vals, cpu_vals, n = [], [], 0
            for data in [burst_h, burst_l]:
                if not data:
                    continue
                bs = data.get("bash_summary", {}).get(cat, {})
                if bs:
                    mem_vals.append(bs.get("mean_mem_spike", 0))
                    cpu_vals.append(bs.get("mean_cpu_spike", 0))
                    n += bs.get("count", 0)
            if mem_vals:
                combined_mem[cat] = statistics.mean(mem_vals)
                combined_cpu[cat] = statistics.mean(cpu_vals)
                combined_n[cat] = n

        cats_sorted = sorted(combined_mem.keys(),
                             key=lambda c: combined_mem.get(c, 0), reverse=True)
        x2 = np.arange(len(cats_sorted))
        mem_vals = [combined_mem[c] for c in cats_sorted]
        cpu_vals = [combined_cpu[c] for c in cats_sorted]
        counts = [combined_n[c] for c in cats_sorted]

        bars = ax2.bar(x2, mem_vals, 0.6, color="#F44336", alpha=0.8, label="Mem Spike")
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x2, cpu_vals, "s-", color="#2196F3", linewidth=2,
                      markersize=8, label="CPU Spike")

        # Add count labels
        for i, (bar, n) in enumerate(zip(bars, counts)):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f"n={n}", ha="center", va="bottom", fontsize=9)

        ax2.set_xlabel("Bash Command Category", fontsize=14)
        ax2.set_ylabel("Mean Memory Spike (MB)", fontsize=14, color="#F44336")
        ax2_twin.set_ylabel("Mean CPU Spike (%)", fontsize=14, color="#2196F3")
        ax2.set_title("(b) Burst Profile by Bash Category", fontsize=15)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(cats_sorted, fontsize=9, rotation=35, ha="right")
        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=11)
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "tool_burst_correlation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _chart_retry_waste(retry_h, retry_l):
    """Retry waste chart (2 subplots)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Retry groups per task (histogram)
    for label, data, color in [("Haiku", retry_h, "#2196F3"), ("GLM", retry_l, "#4CAF50")]:
        if not data or not data.get("per_task"):
            continue
        vals = [t["n_retry_groups"] for t in data["per_task"].values()]
        if vals:
            ax1.hist(vals, bins=range(0, max(vals) + 2), alpha=0.55, color=color,
                     label=f"{label} (n={len(vals)})", edgecolor="white")

    ax1.set_xlabel("Retry Groups per Task", fontsize=14)
    ax1.set_ylabel("Number of Tasks", fontsize=14)
    ax1.set_title("(a) Retry Group Distribution", fontsize=15)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(labelsize=12)

    # (b) Retry time % vs memory accumulation (scatter)
    for label, data, color, marker in [
        ("Haiku", retry_h, "#2196F3", "o"), ("GLM", retry_l, "#4CAF50", "^")
    ]:
        if not data or not data.get("per_task"):
            continue
        pcts = [t["retry_time_pct"] for t in data["per_task"].values()]
        accums = [t["total_mem_accumulation"] for t in data["per_task"].values()]
        ax2.scatter(pcts, accums, alpha=0.5, color=color, label=label,
                    marker=marker, s=30)

    ax2.set_xlabel("Retry Time (% of Execution)", fontsize=14)
    ax2.set_ylabel("Memory Accumulation (MB)", fontsize=14)
    ax2.set_title("(b) Retry Cost: Time vs Memory", fontsize=15)
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=12)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "retry_waste.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _chart_concurrency_sim(sim_results):
    """Concurrency simulation chart (2 subplots)."""
    sim = sim_results.get("simulation", {})
    if not sim:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ns = sorted(sim.keys())
    static = [sim[n]["static_alloc_avg"] / 1024 for n in ns]
    dynamic_avg = [sim[n]["dynamic_peak_avg"] / 1024 for n in ns]
    dynamic_p95 = [sim[n]["dynamic_peak_p95"] / 1024 for n in ns]
    savings = [sim[n]["savings_pct"] for n in ns]
    mux_gain = [sim[n]["stat_mux_gain"] for n in ns]

    # (a) Static vs Dynamic allocation
    ax1.plot(ns, static, "o-", color="#F44336", label="Static (sum of peaks)",
             linewidth=2, markersize=8)
    ax1.plot(ns, dynamic_avg, "s-", color="#2196F3", label="Dynamic (avg peak)",
             linewidth=2, markersize=8)
    ax1.fill_between(ns, dynamic_avg, dynamic_p95, alpha=0.2, color="#2196F3",
                     label="Dynamic P95")
    ax1.axhline(128, color="gray", ls="--", lw=1.5, label="128GB limit")
    ax1.set_xlabel("Number of Concurrent Agents", fontsize=14)
    ax1.set_ylabel("Total Memory Required (GB)", fontsize=14)
    ax1.set_title("(a) Static vs Dynamic Allocation", fontsize=15)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xscale("log", base=2)
    ax1.tick_params(labelsize=12)
    ax1.set_xticks(ns)
    ax1.set_xticklabels([str(n) for n in ns])

    # (b) Statistical multiplexing gain & savings
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(ns, mux_gain, "o-", color="#4CAF50", label="Mux Gain",
                     linewidth=2, markersize=8)
    line2 = ax2_twin.plot(ns, savings, "s--", color="#FF9800", label="Savings %",
                          linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Concurrent Agents", fontsize=14)
    ax2.set_ylabel("Statistical Multiplexing Gain (×)", fontsize=14, color="#4CAF50")
    ax2_twin.set_ylabel("Memory Savings (%)", fontsize=14, color="#FF9800")
    ax2.set_title("(b) Multiplexing Benefit", fontsize=15)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_xscale("log", base=2)
    ax2.tick_params(labelsize=12)
    ax2.set_xticks(ns)
    ax2.set_xticklabels([str(n) for n in ns])

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "concurrency_simulation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _chart_token_resource_corr(corr_h, corr_l):
    """Token vs resource scatter plot (2 subplots)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Output tokens vs peak memory
    for label, data, color, marker in [
        ("Haiku", corr_h, "#2196F3", "o"), ("GLM", corr_l, "#4CAF50", "^")
    ]:
        if not data or not data.get("task_data"):
            continue
        tokens = [d["output_tokens"] / 1000 for d in data["task_data"]]
        peak_mem = [d["peak_mem"] for d in data["task_data"]]
        ax1.scatter(tokens, peak_mem, alpha=0.5, color=color, label=label,
                    marker=marker, s=30)
        # Correlation annotation
        r = data.get("correlations", {}).get("output_tokens_vs_peak_mem", 0)
        ax1.annotate(f"r={r:.2f}", xy=(0.05, 0.95 if label == "Haiku" else 0.88),
                     xycoords="axes fraction", fontsize=11, color=color,
                     fontweight="bold")

    ax1.set_xlabel("Output Tokens (K)", fontsize=14)
    ax1.set_ylabel("Peak Memory (MB)", fontsize=14)
    ax1.set_title("(a) Tokens vs Peak Memory", fontsize=15)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=12)

    # (b) Turns vs execution time
    for label, data, color, marker in [
        ("Haiku", corr_h, "#2196F3", "o"), ("GLM", corr_l, "#4CAF50", "^")
    ]:
        if not data or not data.get("task_data"):
            continue
        turns = [d["n_turns"] for d in data["task_data"]]
        exec_t = [d["exec_time"] / 60 for d in data["task_data"]]
        ax2.scatter(turns, exec_t, alpha=0.5, color=color, label=label,
                    marker=marker, s=30)
        r = data.get("correlations", {}).get("n_turns_vs_exec_time", 0)
        ax2.annotate(f"r={r:.2f}", xy=(0.05, 0.95 if label == "Haiku" else 0.88),
                     xycoords="axes fraction", fontsize=11, color=color,
                     fontweight="bold")

    ax2.set_xlabel("Number of LLM Turns", fontsize=14)
    ax2.set_ylabel("Execution Time (min)", fontsize=14)
    ax2.set_title("(b) Turns vs Execution Time", fontsize=15)
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=12)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "token_resource_correlation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="New characterization insights")
    parser.add_argument("--analysis", type=int, nargs="*",
                        help="Run specific analyses (1=tokens, 2=delta, 3=retry, "
                             "4=simulation, 5=token-resource corr). Default: all")
    args = parser.parse_args()

    run_all = args.analysis is None
    analyses = set(args.analysis) if args.analysis else set()

    print("=" * 70)
    print("  New Characterization Insights")
    print("=" * 70)

    token_h = {}
    burst_h, burst_l = {}, {}
    retry_h, retry_l = {}, {}
    sim_results = {}
    corr_h, corr_l = {}, {}

    # 1. Token analysis (Haiku only — GLM local model tokens are not meaningful)
    if run_all or 1 in analyses:
        if os.path.exists(HAIKU_DIR):
            token_h = analyze_tokens(HAIKU_DIR, "Haiku")

    # 2. Tool-burst correlation
    if run_all or 2 in analyses:
        if os.path.exists(HAIKU_DIR):
            burst_h = analyze_tool_burst_correlation(HAIKU_DIR, "Haiku")
        if os.path.exists(LOCAL_DIR):
            burst_l = analyze_tool_burst_correlation(LOCAL_DIR, "GLM (Local)")

    # 3. Retry waste
    if run_all or 3 in analyses:
        if os.path.exists(HAIKU_DIR):
            retry_h = analyze_retry_waste(HAIKU_DIR, "Haiku")
        if os.path.exists(LOCAL_DIR):
            retry_l = analyze_retry_waste(LOCAL_DIR, "GLM (Local)")

    # 4. Concurrency simulation
    if run_all or 4 in analyses:
        sim_results = analyze_concurrency_simulation(HAIKU_DIR, LOCAL_DIR)

    # 5. Token-resource correlation
    if run_all or 5 in analyses:
        if os.path.exists(HAIKU_DIR):
            corr_h = analyze_token_resource_correlation(HAIKU_DIR, "Haiku")
        if os.path.exists(LOCAL_DIR):
            corr_l = analyze_token_resource_correlation(LOCAL_DIR, "GLM (Local)")

    # Generate charts
    _section("Generating Charts")
    generate_charts(token_h, burst_h, burst_l,
                    retry_h, retry_l, sim_results,
                    corr_h, corr_l)

    print(f"\n{'='*70}")
    print("  Done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
