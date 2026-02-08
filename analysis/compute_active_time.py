#!/usr/bin/env python3
"""
Compute active_time (trace duration) for all valid tasks and recalculate tool time ratios.

Compares:
  - Old ratio: tool_time / claude_time  (includes container startup overhead)
  - New ratio: tool_time / active_time  (only actual agent execution from trace timestamps)

active_time = last trace entry timestamp - first trace entry timestamp
tool_time   = sum of (tool_result.timestamp - tool_use.timestamp) for each tool call
"""

import json
import os
import sys
import statistics
from datetime import datetime, timezone

# Add parent dir so we can import from analysis/
sys.path.insert(0, os.path.dirname(__file__))
from filter_valid_tasks import get_valid_task_names

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    "all_images_haiku": os.path.join(REPO_ROOT, "experiments", "all_images_haiku"),
    "all_images_local": os.path.join(REPO_ROOT, "experiments", "all_images_local"),
}


def parse_iso_timestamp(ts_str):
    """Parse ISO format timestamp string to datetime (UTC)."""
    # Handle both 'Z' suffix and '+00:00' or naive timestamps
    if ts_str is None:
        return None
    ts_str = ts_str.strip()
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def get_attempt_dir(base_dir, task_name):
    """Return the latest attempt directory for a task."""
    task_dir = os.path.join(base_dir, task_name)
    import glob
    attempts = glob.glob(os.path.join(task_dir, "attempt_*"))
    if not attempts:
        return None
    return sorted(attempts)[-1]


def compute_active_time_from_trace(trace_path):
    """Read trace.jsonl and return (active_time_seconds, first_ts, last_ts).

    active_time = last timestamp - first timestamp across all trace entries.
    Skips the first 'summary' line which has no timestamp.
    """
    timestamps = []
    try:
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts_str = entry.get("timestamp")
                if ts_str:
                    dt = parse_iso_timestamp(ts_str)
                    if dt:
                        timestamps.append(dt)
    except FileNotFoundError:
        return None, None, None

    if len(timestamps) < 2:
        return None, None, None

    first_ts = min(timestamps)
    last_ts = max(timestamps)
    active_time = (last_ts - first_ts).total_seconds()
    return active_time, first_ts, last_ts


def compute_tool_time_from_trace(trace_path):
    """Extract tool durations from trace.jsonl.

    Approach: find pairs of (assistant tool_use, user tool_result) by matching tool IDs.
    Tool duration = tool_result entry timestamp - tool_use entry timestamp.

    Also tries tool_calls.json in the same directory as a fallback.
    """
    attempt_dir = os.path.dirname(trace_path)

    # First try tool_calls.json which has pre-extracted data with timestamp + end_timestamp
    tool_calls_path = os.path.join(attempt_dir, "tool_calls.json")
    if os.path.exists(tool_calls_path):
        try:
            with open(tool_calls_path) as f:
                tool_calls = json.load(f)
            total_tool_time = 0.0
            count = 0
            for tc in tool_calls:
                start_ts = parse_iso_timestamp(tc.get("timestamp"))
                end_ts = parse_iso_timestamp(tc.get("end_timestamp"))
                if start_ts and end_ts:
                    duration = (end_ts - start_ts).total_seconds()
                    if duration >= 0:
                        total_tool_time += duration
                        count += 1
            if count > 0:
                return total_tool_time, count
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: parse trace.jsonl directly
    tool_use_times = {}  # id -> timestamp
    tool_result_times = {}  # tool_use_id -> timestamp

    try:
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts_str = entry.get("timestamp")
                if not ts_str:
                    continue

                msg = entry.get("message", {})
                content = msg.get("content", [])
                entry_type = entry.get("type")

                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "tool_use" and entry_type == "assistant":
                                tool_id = item.get("id")
                                if tool_id:
                                    tool_use_times[tool_id] = parse_iso_timestamp(ts_str)
                            elif item.get("type") == "tool_result" and entry_type == "user":
                                tool_id = item.get("tool_use_id")
                                if tool_id:
                                    tool_result_times[tool_id] = parse_iso_timestamp(ts_str)
    except FileNotFoundError:
        return 0.0, 0

    total_tool_time = 0.0
    count = 0
    for tool_id, start_dt in tool_use_times.items():
        end_dt = tool_result_times.get(tool_id)
        if start_dt and end_dt:
            duration = (end_dt - start_dt).total_seconds()
            if duration >= 0:
                total_tool_time += duration
                count += 1

    return total_tool_time, count


def analyze_dataset(name, base_dir):
    """Analyze all valid tasks in a dataset. Returns list of per-task results."""
    valid_tasks = get_valid_task_names(base_dir)
    results = []

    for task_name in valid_tasks:
        attempt_dir = get_attempt_dir(base_dir, task_name)
        if not attempt_dir:
            continue

        # Read results.json for claude_time
        results_path = os.path.join(attempt_dir, "results.json")
        try:
            with open(results_path) as f:
                res = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        claude_time = res.get("claude_time", 0)
        total_time = res.get("total_time", 0)

        # Compute active_time from trace
        trace_path = os.path.join(attempt_dir, "trace.jsonl")
        active_time, first_ts, last_ts = compute_active_time_from_trace(trace_path)
        if active_time is None or active_time <= 0:
            continue

        # Compute tool_time
        tool_time, tool_count = compute_tool_time_from_trace(trace_path)

        startup_gap = claude_time - active_time

        old_ratio = tool_time / claude_time if claude_time > 0 else 0
        new_ratio = tool_time / active_time if active_time > 0 else 0

        results.append({
            "task": task_name,
            "claude_time": claude_time,
            "total_time": total_time,
            "active_time": active_time,
            "startup_gap": startup_gap,
            "tool_time": tool_time,
            "tool_count": tool_count,
            "old_ratio": old_ratio,
            "new_ratio": new_ratio,
        })

    return results


def print_summary(name, results):
    """Print summary statistics for a dataset."""
    if not results:
        print(f"\n{'='*80}")
        print(f"  Dataset: {name} -- NO VALID RESULTS")
        print(f"{'='*80}")
        return

    n = len(results)
    claude_times = [r["claude_time"] for r in results]
    active_times = [r["active_time"] for r in results]
    startup_gaps = [r["startup_gap"] for r in results]
    tool_times = [r["tool_time"] for r in results]
    old_ratios = [r["old_ratio"] for r in results]
    new_ratios = [r["new_ratio"] for r in results]

    print(f"\n{'='*80}")
    print(f"  Dataset: {name}")
    print(f"{'='*80}")
    print(f"  Valid tasks analyzed: {n}")
    print()
    print(f"  {'Metric':<25} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
    print(f"  {'-'*75}")

    for label, vals in [
        ("claude_time (s)", claude_times),
        ("active_time (s)", active_times),
        ("startup_gap (s)", startup_gaps),
        ("tool_time (s)", tool_times),
    ]:
        mean = statistics.mean(vals)
        median = statistics.median(vals)
        mn = min(vals)
        mx = max(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0
        print(f"  {label:<25} {mean:>10.1f} {median:>10.1f} {mn:>10.1f} {mx:>10.1f} {std:>10.1f}")

    print()
    print(f"  {'Ratio':<25} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
    print(f"  {'-'*75}")
    for label, vals in [
        ("old: tool/claude", old_ratios),
        ("new: tool/active", new_ratios),
    ]:
        mean = statistics.mean(vals)
        median = statistics.median(vals)
        mn = min(vals)
        mx = max(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0
        print(f"  {label:<25} {mean:>10.4f} {median:>10.4f} {mn:>10.4f} {mx:>10.4f} {std:>10.4f}")

    # Also print gap as percentage of claude_time
    gap_pcts = [r["startup_gap"] / r["claude_time"] * 100 if r["claude_time"] > 0 else 0 for r in results]
    print()
    print(f"  Startup gap as % of claude_time:")
    print(f"    Mean: {statistics.mean(gap_pcts):.1f}%  Median: {statistics.median(gap_pcts):.1f}%  "
          f"Min: {min(gap_pcts):.1f}%  Max: {max(gap_pcts):.1f}%")


def print_per_task_table(name, results):
    """Print per-task details."""
    print(f"\n{'='*80}")
    print(f"  Per-task details: {name}")
    print(f"{'='*80}")
    header = (f"  {'Task':<50} {'claude':>7} {'active':>7} {'gap':>6} {'gap%':>5} "
              f"{'tool_t':>7} {'#tools':>6} {'old_r':>7} {'new_r':>7}")
    print(header)
    print(f"  {'-'*len(header)}")

    for r in sorted(results, key=lambda x: x["startup_gap"], reverse=True):
        task_short = r["task"][:48]
        gap_pct = r["startup_gap"] / r["claude_time"] * 100 if r["claude_time"] > 0 else 0
        print(f"  {task_short:<50} {r['claude_time']:>7.1f} {r['active_time']:>7.1f} "
              f"{r['startup_gap']:>6.1f} {gap_pct:>4.1f}% "
              f"{r['tool_time']:>7.1f} {r['tool_count']:>6} "
              f"{r['old_ratio']:>7.4f} {r['new_ratio']:>7.4f}")


def main():
    print("=" * 80)
    print("  Active Time Analysis: Comparing tool_time / claude_time vs tool_time / active_time")
    print("  active_time = trace duration (last timestamp - first timestamp)")
    print("  startup_gap = claude_time - active_time (container startup overhead)")
    print("=" * 80)

    all_dataset_results = {}

    for name, base_dir in DATASETS.items():
        if not os.path.isdir(base_dir):
            print(f"\n  WARNING: Dataset directory not found: {base_dir}")
            continue
        results = analyze_dataset(name, base_dir)
        all_dataset_results[name] = results
        print_summary(name, results)

    # Per-task details for Haiku
    if "all_images_haiku" in all_dataset_results and all_dataset_results["all_images_haiku"]:
        print_per_task_table("all_images_haiku (Haiku)", all_dataset_results["all_images_haiku"])

    # Per-task details for Local/GLM
    if "all_images_local" in all_dataset_results and all_dataset_results["all_images_local"]:
        print_per_task_table("all_images_local (Local/GLM)", all_dataset_results["all_images_local"])

    # Cross-dataset comparison
    if len(all_dataset_results) == 2:
        print(f"\n{'='*80}")
        print(f"  Cross-Dataset Comparison")
        print(f"{'='*80}")
        for name, results in all_dataset_results.items():
            if not results:
                continue
            gaps = [r["startup_gap"] for r in results]
            gap_pcts = [r["startup_gap"] / r["claude_time"] * 100 if r["claude_time"] > 0 else 0 for r in results]
            old_r = [r["old_ratio"] for r in results]
            new_r = [r["new_ratio"] for r in results]
            print(f"\n  {name}:")
            print(f"    Tasks: {len(results)}")
            print(f"    Avg startup gap:    {statistics.mean(gaps):>7.1f}s ({statistics.mean(gap_pcts):.1f}% of claude_time)")
            print(f"    Median startup gap: {statistics.median(gaps):>7.1f}s ({statistics.median(gap_pcts):.1f}% of claude_time)")
            print(f"    Old ratio (tool/claude): mean={statistics.mean(old_r):.4f}  median={statistics.median(old_r):.4f}")
            print(f"    New ratio (tool/active): mean={statistics.mean(new_r):.4f}  median={statistics.median(new_r):.4f}")
            ratio_increase = (statistics.mean(new_r) - statistics.mean(old_r)) / statistics.mean(old_r) * 100 if statistics.mean(old_r) > 0 else 0
            print(f"    Ratio increase (old->new): {ratio_increase:+.1f}%")


if __name__ == "__main__":
    main()
