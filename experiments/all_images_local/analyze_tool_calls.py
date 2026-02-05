#!/usr/bin/env python3
"""
Analyze tool call durations from SWE-bench experiment results.
Reads tool_calls.json and results.json from each task directory.
"""

import json
import glob
import os
import statistics
from datetime import datetime
from collections import defaultdict

BASE_DIR = "/home/yunwei37/workspace/agentcgroup/experiments/all_images_local"


def parse_iso(ts_str):
    """Parse ISO format timestamp, handling Z suffix and fractional seconds."""
    if ts_str is None:
        return None
    ts_str = ts_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return None


def load_json(path):
    """Load JSON file, return None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def get_task_name_from_dir(dirname):
    """Extract task identifier like '12rambau__sepal_ui-814' from dir name."""
    parts = dirname.split("_", 2)
    if len(parts) >= 3:
        return parts[2]
    return dirname


def main():
    # -------------------------------------------------------------------------
    # 1. Load progress.json for success/failure mapping
    # -------------------------------------------------------------------------
    progress = load_json(os.path.join(BASE_DIR, "progress.json"))
    success_map = {}
    if progress and "results" in progress:
        for task_name, info in progress["results"].items():
            success_map[task_name] = info.get("success", False)

    # -------------------------------------------------------------------------
    # 2. Discover all task directories and load data
    # -------------------------------------------------------------------------
    task_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "task_*")))

    # Per-tool aggregation
    tool_total_time = defaultdict(float)
    tool_call_count = defaultdict(int)
    tool_all_durations = defaultdict(list)

    # Global tracking
    all_individual_calls = []
    task_summaries = []

    tasks_loaded = 0
    tasks_skipped = 0

    for task_dir in task_dirs:
        dir_name = os.path.basename(task_dir)
        task_name = get_task_name_from_dir(dir_name)

        tool_calls_path = os.path.join(task_dir, "attempt_1", "tool_calls.json")
        results_path = os.path.join(task_dir, "attempt_1", "results.json")

        tool_calls = load_json(tool_calls_path)
        results = load_json(results_path)

        if tool_calls is None:
            tasks_skipped += 1
            continue

        tasks_loaded += 1

        task_tool_time = defaultdict(float)
        task_tool_count = defaultdict(int)
        task_total_tool_time = 0.0
        valid_calls = 0

        for call in tool_calls:
            tool_name = call.get("tool", "Unknown")
            ts_start = parse_iso(call.get("timestamp"))
            ts_end = parse_iso(call.get("end_timestamp"))

            if ts_start is None or ts_end is None:
                continue

            duration = (ts_end - ts_start).total_seconds()
            if duration < 0:
                continue

            valid_calls += 1
            task_tool_time[tool_name] += duration
            task_tool_count[tool_name] += 1
            task_total_tool_time += duration

            tool_total_time[tool_name] += duration
            tool_call_count[tool_name] += 1
            tool_all_durations[tool_name].append(duration)
            all_individual_calls.append((duration, tool_name, dir_name))

        claude_time = None
        if results:
            claude_time = results.get("claude_time")

        is_success = success_map.get(task_name, None)

        task_summaries.append({
            "dir_name": dir_name,
            "task_name": task_name,
            "total_tool_time": task_total_tool_time,
            "claude_time": claude_time,
            "tool_breakdown": dict(task_tool_time),
            "tool_counts": dict(task_tool_count),
            "valid_calls": valid_calls,
            "total_calls": len(tool_calls),
            "success": is_success,
        })

    # =========================================================================
    # REPORT
    # =========================================================================
    sep = "=" * 78
    sep2 = "-" * 78

    print(sep)
    print("       SWE-BENCH TOOL CALL DURATION ANALYSIS")
    print(sep)
    print(f"  Tasks loaded:  {tasks_loaded}")
    print(f"  Tasks skipped: {tasks_skipped} (missing tool_calls.json)")
    print(f"  Total tool calls analyzed: {sum(tool_call_count.values())}")
    print()

    # -------------------------------------------------------------------------
    # Section 1: Per-tool breakdown
    # -------------------------------------------------------------------------
    print(sep)
    print("  1. PER-TOOL BREAKDOWN")
    print(sep)

    grand_total_time = sum(tool_total_time.values())

    sorted_tools = sorted(tool_total_time.keys(), key=lambda t: tool_total_time[t], reverse=True)

    print(f"  {'Tool':<18} {'Count':>7} {'Total Time':>12} {'Avg (s)':>10} {'Median (s)':>11} {'Max (s)':>10} {'% of Total':>11}")
    print(f"  {sep2}")

    for tool in sorted_tools:
        count = tool_call_count[tool]
        total = tool_total_time[tool]
        durations = tool_all_durations[tool]
        avg = total / count if count > 0 else 0
        med = statistics.median(durations) if durations else 0
        mx = max(durations) if durations else 0
        pct = (total / grand_total_time * 100) if grand_total_time > 0 else 0

        print(f"  {tool:<18} {count:>7} {total:>11.1f}s {avg:>9.2f}s {med:>10.2f}s {mx:>9.2f}s {pct:>10.1f}%")

    print(f"  {sep2}")
    print(f"  {'TOTAL':<18} {sum(tool_call_count.values()):>7} {grand_total_time:>11.1f}s")
    print()

    # -------------------------------------------------------------------------
    # Section 2: Bash deep-dive
    # -------------------------------------------------------------------------
    print(sep)
    print("  2. BASH TOOL DEEP-DIVE")
    print(sep)

    bash_durations = tool_all_durations.get("Bash", [])
    if bash_durations:
        bash_total = tool_total_time["Bash"]
        bash_count = tool_call_count["Bash"]
        bash_avg = bash_total / bash_count
        bash_med = statistics.median(bash_durations)
        bash_max = max(bash_durations)
        bash_min = min(bash_durations)
        bash_stdev = statistics.stdev(bash_durations) if len(bash_durations) > 1 else 0
        bash_pct = (bash_total / grand_total_time * 100) if grand_total_time > 0 else 0

        sorted_bd = sorted(bash_durations)
        n = len(sorted_bd)
        p90 = sorted_bd[int(n * 0.90)] if n >= 10 else bash_max
        p95 = sorted_bd[int(n * 0.95)] if n >= 20 else bash_max
        p99 = sorted_bd[int(n * 0.99)] if n >= 100 else bash_max

        print(f"  Bash call count:           {bash_count}")
        print(f"  Bash total time:           {bash_total:.1f}s ({bash_total/60:.1f} min)")
        print(f"  % of all tool time:        {bash_pct:.1f}%")
        print(f"  Average duration:          {bash_avg:.2f}s")
        print(f"  Median duration:           {bash_med:.2f}s")
        print(f"  Min duration:              {bash_min:.3f}s")
        print(f"  Max duration:              {bash_max:.2f}s")
        print(f"  Std deviation:             {bash_stdev:.2f}s")
        print(f"  P90:                       {p90:.2f}s")
        print(f"  P95:                       {p95:.2f}s")
        print(f"  P99:                       {p99:.2f}s")

        buckets = [(0, 1), (1, 5), (5, 10), (10, 30), (30, 60), (60, 120), (120, 300), (300, float('inf'))]
        print()
        print(f"  Duration distribution:")
        for lo, hi in buckets:
            count_in_bucket = sum(1 for d in bash_durations if lo <= d < hi)
            pct_bucket = count_in_bucket / len(bash_durations) * 100
            hi_label = "inf" if hi == float('inf') else f"{hi:.0f}"
            label = f"{lo:>5.0f}s - {hi_label:>4}s"
            bar = "#" * int(pct_bucket / 2)
            print(f"    {label}: {count_in_bucket:>5} ({pct_bucket:>5.1f}%) {bar}")
    else:
        print("  No Bash tool calls found.")
    print()

    # -------------------------------------------------------------------------
    # Section 3: Tool time vs thinking time
    # -------------------------------------------------------------------------
    print(sep)
    print("  3. TOOL EXECUTION TIME vs THINKING TIME")
    print(sep)

    tasks_with_both = [t for t in task_summaries if t["claude_time"] is not None and t["total_tool_time"] > 0]

    if tasks_with_both:
        total_claude = sum(t["claude_time"] for t in tasks_with_both)
        total_tool = sum(t["total_tool_time"] for t in tasks_with_both)
        total_thinking = total_claude - total_tool

        print(f"  Tasks with both metrics:   {len(tasks_with_both)}")
        print(f"  Total claude_time:         {total_claude:.1f}s ({total_claude/60:.1f} min)")
        print(f"  Total tool exec time:      {total_tool:.1f}s ({total_tool/60:.1f} min)")
        print(f"  Total thinking time:       {total_thinking:.1f}s ({total_thinking/60:.1f} min)")
        print(f"  Tool time / claude_time:   {total_tool/total_claude*100:.1f}%")
        print(f"  Thinking / claude_time:    {total_thinking/total_claude*100:.1f}%")
        print()

        ratios = []
        for t in tasks_with_both:
            ratio = t["total_tool_time"] / t["claude_time"] if t["claude_time"] > 0 else 0
            ratios.append(ratio)

        print(f"  Per-task tool-time ratio (tool_time / claude_time):")
        print(f"    Average:  {statistics.mean(ratios)*100:.1f}%")
        print(f"    Median:   {statistics.median(ratios)*100:.1f}%")
        print(f"    Min:      {min(ratios)*100:.1f}%")
        print(f"    Max:      {max(ratios)*100:.1f}%")

        print()
        print(f"  Top 5 tasks by tool-time ratio:")
        tasks_sorted_ratio = sorted(tasks_with_both, key=lambda t: t["total_tool_time"] / t["claude_time"] if t["claude_time"] > 0 else 0, reverse=True)
        for t in tasks_sorted_ratio[:5]:
            ratio = t["total_tool_time"] / t["claude_time"] * 100 if t["claude_time"] > 0 else 0
            status = "PASS" if t["success"] else ("FAIL" if t["success"] is not None else "???")
            print(f"    {t['dir_name']:<55} {ratio:>5.1f}%  [{status}]")

        print()
        print(f"  Bottom 5 tasks by tool-time ratio:")
        for t in tasks_sorted_ratio[-5:]:
            ratio = t["total_tool_time"] / t["claude_time"] * 100 if t["claude_time"] > 0 else 0
            status = "PASS" if t["success"] else ("FAIL" if t["success"] is not None else "???")
            print(f"    {t['dir_name']:<55} {ratio:>5.1f}%  [{status}]")
    else:
        print("  No tasks with both tool_calls and claude_time data found.")
    print()

    # -------------------------------------------------------------------------
    # Section 4: Successful vs Failed tasks comparison
    # -------------------------------------------------------------------------
    print(sep)
    print("  4. SUCCESSFUL vs FAILED TASKS")
    print(sep)

    successful_tasks = [t for t in task_summaries if t["success"] is True and t["claude_time"] and t["claude_time"] > 0]
    failed_tasks = [t for t in task_summaries if t["success"] is False and t["claude_time"] and t["claude_time"] > 0]

    def summarize_group(tasks, label):
        if not tasks:
            print(f"  {label}: No tasks in this group.")
            return

        total_ct = sum(t["claude_time"] for t in tasks)
        total_tt = sum(t["total_tool_time"] for t in tasks)
        avg_ct = statistics.mean([t["claude_time"] for t in tasks])
        avg_tt = statistics.mean([t["total_tool_time"] for t in tasks])
        avg_calls = statistics.mean([t["valid_calls"] for t in tasks])
        ratios = [t["total_tool_time"] / t["claude_time"] for t in tasks if t["claude_time"] > 0]

        print(f"  {label} ({len(tasks)} tasks):")
        print(f"    Avg claude_time:         {avg_ct:.1f}s ({avg_ct/60:.1f} min)")
        print(f"    Avg tool exec time:      {avg_tt:.1f}s ({avg_tt/60:.1f} min)")
        print(f"    Avg tool calls:          {avg_calls:.1f}")
        print(f"    Agg tool/claude ratio:   {total_tt/total_ct*100:.1f}%")
        if ratios:
            print(f"    Per-task ratio avg:      {statistics.mean(ratios)*100:.1f}%")
            print(f"    Per-task ratio median:   {statistics.median(ratios)*100:.1f}%")

        group_tool_time = defaultdict(float)
        group_tool_count = defaultdict(int)
        for t in tasks:
            for tool, time_val in t["tool_breakdown"].items():
                group_tool_time[tool] += time_val
            for tool, cnt in t["tool_counts"].items():
                group_tool_count[tool] += cnt

        group_total = sum(group_tool_time.values())
        print(f"    Tool breakdown:")
        for tool in sorted(group_tool_time.keys(), key=lambda x: group_tool_time[x], reverse=True):
            pct = group_tool_time[tool] / group_total * 100 if group_total > 0 else 0
            avg_per_call = group_tool_time[tool] / group_tool_count[tool] if group_tool_count[tool] > 0 else 0
            print(f"      {tool:<16} {group_tool_count[tool]:>5} calls, {group_tool_time[tool]:>8.1f}s ({pct:>5.1f}%), avg {avg_per_call:.2f}s/call")
        print()

    summarize_group(successful_tasks, "SUCCESSFUL")
    summarize_group(failed_tasks, "FAILED")

    if successful_tasks and failed_tasks:
        print(f"  KEY DIFFERENCES (Successful vs Failed):")
        s_avg_ct = statistics.mean([t["claude_time"] for t in successful_tasks])
        f_avg_ct = statistics.mean([t["claude_time"] for t in failed_tasks])
        s_avg_tt = statistics.mean([t["total_tool_time"] for t in successful_tasks])
        f_avg_tt = statistics.mean([t["total_tool_time"] for t in failed_tasks])
        s_avg_calls = statistics.mean([t["valid_calls"] for t in successful_tasks])
        f_avg_calls = statistics.mean([t["valid_calls"] for t in failed_tasks])
        s_ratios = [t["total_tool_time"]/t["claude_time"] for t in successful_tasks if t["claude_time"] > 0]
        f_ratios = [t["total_tool_time"]/t["claude_time"] for t in failed_tasks if t["claude_time"] > 0]

        print(f"    {'Metric':<30} {'Successful':>12} {'Failed':>12} {'Delta':>12}")
        print(f"    {'-'*66}")
        print(f"    {'Avg claude_time (s)':<30} {s_avg_ct:>12.1f} {f_avg_ct:>12.1f} {f_avg_ct-s_avg_ct:>+12.1f}")
        print(f"    {'Avg tool time (s)':<30} {s_avg_tt:>12.1f} {f_avg_tt:>12.1f} {f_avg_tt-s_avg_tt:>+12.1f}")
        print(f"    {'Avg tool calls':<30} {s_avg_calls:>12.1f} {f_avg_calls:>12.1f} {f_avg_calls-s_avg_calls:>+12.1f}")
        print(f"    {'Avg tool/claude ratio':<30} {statistics.mean(s_ratios)*100:>11.1f}% {statistics.mean(f_ratios)*100:>11.1f}% {(statistics.mean(f_ratios)-statistics.mean(s_ratios))*100:>+11.1f}%")
    print()

    # -------------------------------------------------------------------------
    # Section 5: Top 10 longest individual tool calls
    # -------------------------------------------------------------------------
    print(sep)
    print("  5. TOP 10 LONGEST INDIVIDUAL TOOL CALLS")
    print(sep)

    all_individual_calls.sort(key=lambda x: x[0], reverse=True)

    print(f"  {'Rank':<6} {'Duration':>10} {'Tool':<18} {'Task'}")
    print(f"  {sep2}")
    for i, (duration, tool, task_dir) in enumerate(all_individual_calls[:10], 1):
        print(f"  {i:<6} {duration:>9.2f}s {tool:<18} {task_dir}")

    print()

    # -------------------------------------------------------------------------
    # Appendix: per-task summary table
    # -------------------------------------------------------------------------
    print(sep)
    print("  APPENDIX: PER-TASK SUMMARY")
    print(sep)

    task_summaries_sorted = sorted(task_summaries, key=lambda t: t["total_tool_time"], reverse=True)

    print(f"  {'Task':<55} {'Tool(s)':>8} {'Claude(s)':>10} {'Ratio':>7} {'Calls':>6} {'Status':>7}")
    print(f"  {sep2}")
    for t in task_summaries_sorted:
        ct_str = f"{t['claude_time']:.0f}" if t["claude_time"] else "N/A"
        ratio_str = f"{t['total_tool_time']/t['claude_time']*100:.0f}%" if t["claude_time"] and t["claude_time"] > 0 else "N/A"
        status = "PASS" if t["success"] else ("FAIL" if t["success"] is not None else "???")
        print(f"  {t['dir_name']:<55} {t['total_tool_time']:>7.1f} {ct_str:>10} {ratio_str:>7} {t['valid_calls']:>6} {status:>7}")

    print()
    print(sep)
    print("  END OF REPORT")
    print(sep)


if __name__ == "__main__":
    main()
