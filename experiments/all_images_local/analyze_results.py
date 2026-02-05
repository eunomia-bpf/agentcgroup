#!/usr/bin/env python3
"""
Comprehensive SWE-bench Experiment Results Analysis
Analyzes task results from all_images_local experiment directory.
"""

import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path("/home/yunwei37/workspace/agentcgroup/experiments/all_images_local")


def parse_mem_mb(mem_str):
    """Parse memory string like '192.4MB / 134.5GB' into MB float."""
    if not mem_str:
        return 0.0
    match = re.match(r"([\d.]+)\s*(KB|MB|GB|TB)", mem_str.split("/")[0].strip())
    if not match:
        return 0.0
    val = float(match.group(1))
    unit = match.group(2)
    if unit == "KB":
        return val / 1024
    elif unit == "MB":
        return val
    elif unit == "GB":
        return val * 1024
    elif unit == "TB":
        return val * 1024 * 1024
    return val


def load_json(path):
    """Load JSON file, return None on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def extract_repo_name(task_key):
    """Extract repo name from task key like '12rambau__sepal_ui-411' -> '12rambau/sepal_ui'."""
    parts = task_key.rsplit("-", 1)
    if len(parts) == 2:
        repo_part = parts[0]
        return repo_part.replace("__", "/")
    return task_key


def main():
    # ===== Load progress.json =====
    progress = load_json(BASE_DIR / "progress.json")
    if not progress:
        print("ERROR: Could not load progress.json")
        return

    completed = progress.get("completed", [])
    results_map = progress.get("results", {})

    print("=" * 80)
    print("  SWE-BENCH EXPERIMENT RESULTS ANALYSIS")
    print("  Model: qwen3  |  Tasks: {} completed".format(len(completed)))
    print("=" * 80)

    # ===== 1. Overall Stats =====
    total = len(results_map)
    successes = sum(1 for v in results_map.values() if v.get("success"))
    failures = total - successes
    success_rate = (successes / total * 100) if total > 0 else 0

    print()
    print("=" * 80)
    print("  1. OVERALL STATISTICS")
    print("=" * 80)
    print(f"  Total tasks completed:  {total}")
    print(f"  Passed (resolved):      {successes}")
    print(f"  Failed:                 {failures}")
    print(f"  Success rate:           {success_rate:.1f}%")

    # ===== Gather per-task detailed data =====
    task_dirs = {}
    for d in sorted(BASE_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("task_"):
            match = re.match(r"task_(\d+)_(.*)", d.name)
            if match:
                task_num = int(match.group(1))
                task_key = match.group(2)
                task_dirs[task_key] = d

    success_times = []
    failure_times = []
    success_claude_times = []
    failure_claude_times = []
    success_tool_counts = []
    failure_tool_counts = []

    success_peak_mem = []
    failure_peak_mem = []
    success_avg_cpu = []
    failure_avg_cpu = []
    success_disk = []
    failure_disk = []

    task_details = {}
    error_types = defaultdict(int)
    infrastructure_failures = 0

    for task_key, result in results_map.items():
        success = result.get("success", False)
        total_time = result.get("total_time", 0)

        detail = {
            "success": success,
            "total_time": total_time,
            "repo": extract_repo_name(task_key),
        }

        task_dir = task_dirs.get(task_key)
        if task_dir:
            results_json = load_json(task_dir / "attempt_1" / "results.json")
            if results_json:
                claude_time = results_json.get("claude_time")
                detail["claude_time"] = claude_time
                detail["has_error"] = "error" in results_json
                detail["image_size_mb"] = results_json.get("image_info", {}).get("size_mb", 0)
                detail["disk_usage_mb"] = results_json.get("disk_usage", {}).get("testbed_mb", 0)

                if "error" in results_json:
                    error_msg = results_json["error"]
                    if "container storage" in error_msg or "Failed to start" in error_msg:
                        error_types["Container startup failure"] += 1
                        infrastructure_failures += 1
                    elif "timeout" in error_msg.lower():
                        error_types["Timeout"] += 1
                    else:
                        error_types["Other error"] += 1

                if claude_time:
                    if success:
                        success_claude_times.append(claude_time)
                    else:
                        failure_claude_times.append(claude_time)

                disk_mb = results_json.get("disk_usage", {}).get("testbed_mb", 0)
                if disk_mb:
                    if success:
                        success_disk.append(disk_mb)
                    else:
                        failure_disk.append(disk_mb)

            # Load resources.json
            resources_json = load_json(task_dir / "attempt_1" / "resources.json")
            if resources_json and "samples" in resources_json:
                samples = resources_json["samples"]
                if samples:
                    mem_values = [parse_mem_mb(s.get("mem_usage", "")) for s in samples]
                    mem_values = [m for m in mem_values if m > 0]
                    if mem_values:
                        peak_mem = max(mem_values)
                        avg_mem = statistics.mean(mem_values)
                        detail["peak_mem_mb"] = peak_mem
                        detail["avg_mem_mb"] = avg_mem
                        if success:
                            success_peak_mem.append(peak_mem)
                        else:
                            failure_peak_mem.append(peak_mem)

                    cpu_values = []
                    for s in samples:
                        try:
                            cpu_values.append(float(s.get("cpu_percent", "0").rstrip("%")))
                        except (ValueError, AttributeError):
                            pass
                    if cpu_values:
                        avg_cpu = statistics.mean(cpu_values)
                        detail["avg_cpu"] = avg_cpu
                        if success:
                            success_avg_cpu.append(avg_cpu)
                        else:
                            failure_avg_cpu.append(avg_cpu)

            # Load tool_calls.json
            tool_calls = load_json(task_dir / "attempt_1" / "tool_calls.json")
            if tool_calls and isinstance(tool_calls, list):
                n_calls = len(tool_calls)
                detail["tool_calls"] = n_calls

                tool_types = defaultdict(int)
                for tc in tool_calls:
                    tool_types[tc.get("tool", "unknown")] += 1
                detail["tool_types"] = dict(tool_types)

                if success:
                    success_tool_counts.append(n_calls)
                else:
                    failure_tool_counts.append(n_calls)

        if success:
            success_times.append(total_time)
        else:
            failure_times.append(total_time)

        task_details[task_key] = detail

    # ===== 2. Time Analysis =====
    print()
    print("=" * 80)
    print("  2. TIME ANALYSIS (seconds)")
    print("=" * 80)

    def print_time_stats(label, times):
        if not times:
            print(f"  {label}: No data")
            return
        print(f"  {label}:")
        print(f"    Count:   {len(times)}")
        print(f"    Mean:    {statistics.mean(times):>8.1f}s  ({statistics.mean(times)/60:.1f} min)")
        print(f"    Median:  {statistics.median(times):>8.1f}s  ({statistics.median(times)/60:.1f} min)")
        print(f"    Min:     {min(times):>8.1f}s  ({min(times)/60:.1f} min)")
        print(f"    Max:     {max(times):>8.1f}s  ({max(times)/60:.1f} min)")
        if len(times) > 1:
            print(f"    StdDev:  {statistics.stdev(times):>8.1f}s")

    print_time_stats("Successful tasks (total time)", success_times)
    print()
    print_time_stats("Failed tasks (total time)", failure_times)
    print()
    print_time_stats("Successful tasks (Claude agent time)", success_claude_times)
    print()
    print_time_stats("Failed tasks (Claude agent time)", failure_claude_times)

    # ===== 3. Resource Analysis =====
    print()
    print("=" * 80)
    print("  3. RESOURCE ANALYSIS")
    print("=" * 80)

    def print_resource_stats(label, values, unit=""):
        if not values:
            print(f"  {label}: No data")
            return
        print(f"  {label}:")
        print(f"    Count:   {len(values)}")
        print(f"    Mean:    {statistics.mean(values):>10.1f} {unit}")
        print(f"    Median:  {statistics.median(values):>10.1f} {unit}")
        print(f"    Min:     {min(values):>10.1f} {unit}")
        print(f"    Max:     {max(values):>10.1f} {unit}")

    print()
    print("  -- Peak Memory Usage --")
    print_resource_stats("Successful tasks", success_peak_mem, "MB")
    print()
    print_resource_stats("Failed tasks", failure_peak_mem, "MB")

    print()
    print("  -- Average CPU Usage --")
    print_resource_stats("Successful tasks", success_avg_cpu, "%")
    print()
    print_resource_stats("Failed tasks", failure_avg_cpu, "%")

    print()
    print("  -- Disk Usage (testbed) --")
    print_resource_stats("Successful tasks", success_disk, "MB")
    print()
    print_resource_stats("Failed tasks", failure_disk, "MB")

    # ===== 4. Per-Repo Breakdown =====
    print()
    print("=" * 80)
    print("  4. PER-REPO BREAKDOWN")
    print("=" * 80)

    repo_stats = defaultdict(lambda: {"total": 0, "success": 0, "tasks": []})
    for task_key, detail in task_details.items():
        repo = detail["repo"]
        repo_stats[repo]["total"] += 1
        if detail["success"]:
            repo_stats[repo]["success"] += 1
        repo_stats[repo]["tasks"].append(task_key)

    sorted_repos = sorted(repo_stats.items(), key=lambda x: (-x[1]["total"], x[0]))

    print()
    print(f"  {'Repo':<50} {'Pass':>4} {'Fail':>4} {'Total':>5} {'Rate':>7}")
    print(f"  {'-'*50} {'----':>4} {'----':>4} {'-----':>5} {'-------':>7}")

    for repo, stats in sorted_repos:
        rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
        fail = stats["total"] - stats["success"]
        bar = "#" * stats["success"] + "." * fail
        print(f"  {repo:<50} {stats['success']:>4} {fail:>4} {stats['total']:>5} {rate:>6.0f}%  {bar}")

    repos_with_success = sum(1 for r in repo_stats.values() if r["success"] > 0)
    repos_all_success = sum(1 for r in repo_stats.values() if r["success"] == r["total"])
    repos_all_fail = sum(1 for r in repo_stats.values() if r["success"] == 0)

    print()
    print(f"  Total repos: {len(repo_stats)}")
    print(f"  Repos with at least 1 success: {repos_with_success}")
    print(f"  Repos with 100% success: {repos_all_success}")
    print(f"  Repos with 0% success: {repos_all_fail}")

    # ===== 5. Tool Call Analysis =====
    print()
    print("=" * 80)
    print("  5. TOOL CALL ANALYSIS")
    print("=" * 80)

    print()
    print_resource_stats("Successful tasks (tool calls)", success_tool_counts, "calls")
    print()
    print_resource_stats("Failed tasks (tool calls)", failure_tool_counts, "calls")

    all_tool_types_success = defaultdict(int)
    all_tool_types_failure = defaultdict(int)
    for task_key, detail in task_details.items():
        if "tool_types" in detail:
            target = all_tool_types_success if detail["success"] else all_tool_types_failure
            for tool, count in detail["tool_types"].items():
                target[tool] += count

    print()
    print("  -- Tool Type Usage (aggregated across all tasks) --")
    print()
    all_tools = set(list(all_tool_types_success.keys()) + list(all_tool_types_failure.keys()))
    all_tools_sorted = sorted(all_tools, key=lambda t: -(all_tool_types_success.get(t, 0) + all_tool_types_failure.get(t, 0)))

    print(f"  {'Tool':<25} {'Success':>10} {'Failure':>10} {'Total':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for tool in all_tools_sorted:
        s = all_tool_types_success.get(tool, 0)
        f = all_tool_types_failure.get(tool, 0)
        print(f"  {tool:<25} {s:>10} {f:>10} {s+f:>10}")

    # ===== 6. Correlation: Time vs Success =====
    print()
    print("=" * 80)
    print("  6. CORRELATION: TIME SPENT vs SUCCESS")
    print("=" * 80)

    all_times = [(detail["total_time"], detail["success"]) for detail in task_details.values()]
    all_times.sort(key=lambda x: x[0])

    buckets = [
        (0, 200, "0-200s (< 3.3 min)"),
        (200, 400, "200-400s (3.3-6.7 min)"),
        (400, 600, "400-600s (6.7-10 min)"),
        (600, 800, "600-800s (10-13.3 min)"),
        (800, 1200, "800-1200s (13.3-20 min)"),
        (1200, float("inf"), "1200s+ (20+ min)"),
    ]

    print()
    print(f"  {'Time Bucket':<30} {'Pass':>4} {'Fail':>4} {'Total':>5} {'Rate':>7}")
    print(f"  {'-'*30} {'----':>4} {'----':>4} {'-----':>5} {'-------':>7}")

    for lo, hi, label in buckets:
        bucket_tasks = [(t, s) for t, s in all_times if lo <= t < hi]
        if not bucket_tasks:
            continue
        n_success = sum(1 for _, s in bucket_tasks if s)
        n_fail = len(bucket_tasks) - n_success
        rate = (n_success / len(bucket_tasks) * 100) if bucket_tasks else 0
        print(f"  {label:<30} {n_success:>4} {n_fail:>4} {len(bucket_tasks):>5} {rate:>6.0f}%")

    print()
    if success_times and failure_times:
        all_t = success_times + failure_times
        all_s = [1] * len(success_times) + [0] * len(failure_times)
        n = len(all_t)
        mean_t = statistics.mean(all_t)
        mean_s = statistics.mean(all_s)

        cov = sum((all_t[i] - mean_t) * (all_s[i] - mean_s) for i in range(n)) / n
        std_t = statistics.pstdev(all_t)
        std_s = statistics.pstdev(all_s)

        if std_t > 0 and std_s > 0:
            r = cov / (std_t * std_s)
            print(f"  Point-biserial correlation (time vs success): r = {r:.3f}")
            if abs(r) < 0.1:
                print("  Interpretation: Very weak or no correlation")
            elif abs(r) < 0.3:
                print(f"  Interpretation: Weak {'positive' if r > 0 else 'negative'} correlation")
            elif abs(r) < 0.5:
                print(f"  Interpretation: Moderate {'positive' if r > 0 else 'negative'} correlation")
            else:
                print(f"  Interpretation: Strong {'positive' if r > 0 else 'negative'} correlation")

    if success_tool_counts and failure_tool_counts:
        all_tc = success_tool_counts + failure_tool_counts
        all_s = [1] * len(success_tool_counts) + [0] * len(failure_tool_counts)
        n = len(all_tc)
        mean_tc = statistics.mean(all_tc)
        mean_s = statistics.mean(all_s)

        cov = sum((all_tc[i] - mean_tc) * (all_s[i] - mean_s) for i in range(n)) / n
        std_tc = statistics.pstdev(all_tc)
        std_s = statistics.pstdev(all_s)

        if std_tc > 0 and std_s > 0:
            r = cov / (std_tc * std_s)
            print(f"  Point-biserial correlation (tool calls vs success): r = {r:.3f}")
            if abs(r) < 0.1:
                print("  Interpretation: Very weak or no correlation")
            elif abs(r) < 0.3:
                print(f"  Interpretation: Weak {'positive' if r > 0 else 'negative'} correlation")
            elif abs(r) < 0.5:
                print(f"  Interpretation: Moderate {'positive' if r > 0 else 'negative'} correlation")
            else:
                print(f"  Interpretation: Strong {'positive' if r > 0 else 'negative'} correlation")

    # ===== 7. Error Analysis =====
    print()
    print("=" * 80)
    print("  7. ERROR / FAILURE ANALYSIS")
    print("=" * 80)
    print()
    print(f"  Infrastructure failures (container issues): {infrastructure_failures}")
    print(f"  Tasks that ran but failed tests:            {failures - infrastructure_failures}")
    print()

    if error_types:
        print("  Error type breakdown:")
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"    {err_type}: {count}")

    print()
    print("  Failed tasks detail:")
    print(f"  {'Task':<55} {'Time':>8} {'Claude':>8} {'Tools':>6} {'Reason'}")
    print(f"  {'-'*55} {'-'*8} {'-'*8} {'-'*6} {'-'*25}")
    for task_key in sorted(results_map.keys()):
        if not results_map[task_key].get("success"):
            detail = task_details.get(task_key, {})
            total_t = f"{detail.get('total_time', 0):.0f}s"
            claude_t = f"{detail.get('claude_time', 0):.0f}s" if detail.get("claude_time") else "N/A"
            tools = str(detail.get("tool_calls", "N/A"))
            reason = "infra" if detail.get("has_error") else "test fail"
            print(f"  {task_key:<55} {total_t:>8} {claude_t:>8} {tools:>6} {reason}")

    # ===== 8. Individual Task Summary =====
    print()
    print("=" * 80)
    print("  8. ALL TASKS SUMMARY (sorted by time)")
    print("=" * 80)
    print()
    print(f"  {'#':>3} {'Task':<50} {'Result':>7} {'Total':>9} {'Claude':>9} {'Tools':>6} {'PeakMem':>9}")
    print(f"  {'-'*3} {'-'*50} {'-'*7} {'-'*9} {'-'*9} {'-'*6} {'-'*9}")

    sorted_tasks = sorted(task_details.items(), key=lambda x: x[1].get("total_time", 0))
    for i, (task_key, detail) in enumerate(sorted_tasks, 1):
        result_str = "PASS" if detail["success"] else "FAIL"
        total_t = f"{detail.get('total_time', 0):.0f}s"
        claude_t = f"{detail.get('claude_time', 0):.0f}s" if detail.get("claude_time") else "N/A"
        tools = str(detail.get("tool_calls", "N/A"))
        peak_mem = f"{detail.get('peak_mem_mb', 0):.0f}MB" if detail.get("peak_mem_mb") else "N/A"
        print(f"  {i:>3} {task_key:<50} {result_str:>7} {total_t:>9} {claude_t:>9} {tools:>6} {peak_mem:>9}")

    # ===== Final Summary =====
    print()
    print("=" * 80)
    print("  EXECUTIVE SUMMARY")
    print("=" * 80)
    print()
    effective_total = total - infrastructure_failures
    effective_rate = (successes / effective_total * 100) if effective_total > 0 else 0
    print(f"  Model: qwen3")
    print(f"  Overall success rate: {success_rate:.1f}% ({successes}/{total})")
    print(f"  Effective success rate (excl. infra failures): {effective_rate:.1f}% ({successes}/{effective_total})")
    print()
    if success_times:
        print(f"  Avg time for successful tasks:  {statistics.mean(success_times):.0f}s ({statistics.mean(success_times)/60:.1f} min)")
    if failure_times:
        print(f"  Avg time for failed tasks:      {statistics.mean(failure_times):.0f}s ({statistics.mean(failure_times)/60:.1f} min)")
    if success_tool_counts:
        print(f"  Avg tool calls for success:     {statistics.mean(success_tool_counts):.0f}")
    if failure_tool_counts:
        print(f"  Avg tool calls for failure:     {statistics.mean(failure_tool_counts):.0f}")
    if success_peak_mem:
        print(f"  Avg peak memory (success):      {statistics.mean(success_peak_mem):.0f} MB")
    if failure_peak_mem:
        print(f"  Avg peak memory (failure):      {statistics.mean(failure_peak_mem):.0f} MB")
    print()
    print(f"  Repos tested: {len(repo_stats)}")
    print(f"  Repos with 100% pass: {repos_all_success} / {len(repo_stats)}")
    print(f"  Repos with 0% pass:   {repos_all_fail} / {len(repo_stats)}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
