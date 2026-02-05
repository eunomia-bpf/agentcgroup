#!/usr/bin/env python3
"""
Analyze the proportion of tool call time in total execution time.
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

    all_data = []
    tasks_loaded = 0
    tasks_skipped = 0

    for task_dir in task_dirs:
        dir_name = os.path.basename(task_dir)
        task_name = get_task_name_from_dir(dir_name)

        tool_calls_path = os.path.join(task_dir, "attempt_1", "tool_calls.json")
        results_path = os.path.join(task_dir, "attempt_1", "results.json")

        tool_calls = load_json(tool_calls_path)
        results = load_json(results_path)

        if tool_calls is None or results is None:
            tasks_skipped += 1
            continue

        claude_time = results.get("claude_time")
        if claude_time is None or claude_time <= 0:
            tasks_skipped += 1
            continue

        tasks_loaded += 1

        tool_total_time = 0.0
        valid_calls = 0
        tool_call_times = []

        for call in tool_calls:
            ts_start = parse_iso(call.get("timestamp"))
            ts_end = parse_iso(call.get("end_timestamp"))

            if ts_start is None or ts_end is None:
                continue

            duration = (ts_end - ts_start).total_seconds()
            if duration < 0:
                continue

            valid_calls += 1
            tool_total_time += duration
            tool_call_times.append(duration)

        thinking_time = claude_time - tool_total_time
        tool_ratio = (tool_total_time / claude_time * 100) if claude_time > 0 else 0
        thinking_ratio = (thinking_time / claude_time * 100) if claude_time > 0 else 0

        all_data.append(
            {
                "dir_name": dir_name,
                "task_name": task_name,
                "claude_time": claude_time,
                "tool_time": tool_total_time,
                "thinking_time": thinking_time,
                "tool_ratio": tool_ratio,
                "thinking_ratio": thinking_ratio,
                "valid_calls": valid_calls,
                "total_calls": len(tool_calls),
                "success": success_map.get(task_name, None),
            }
        )

    # =========================================================================
    # REPORT
    # =========================================================================
    sep = "=" * 78
    sep2 = "-" * 78

    print(sep)
    print("       TOOL CALL TIME RATIO ANALYSIS")
    print(sep)
    print(f"  Tasks loaded:  {tasks_loaded}")
    print(f"  Tasks skipped: {tasks_skipped}")
    print()

    if not all_data:
        print("  No data to analyze.")
        return

    # -------------------------------------------------------------------------
    # Overall statistics
    # -------------------------------------------------------------------------
    total_claude_time = sum(d["claude_time"] for d in all_data)
    total_tool_time = sum(d["tool_time"] for d in all_data)
    total_thinking_time = sum(d["thinking_time"] for d in all_data)

    print(sep)
    print("  OVERALL STATISTICS")
    print(sep)
    print(
        f"  Total execution time (all tasks):  {total_claude_time:.1f}s ({total_claude_time / 60:.1f} min)"
    )
    print(
        f"  Total tool execution time:        {total_tool_time:.1f}s ({total_tool_time / 60:.1f} min)"
    )
    print(
        f"  Total thinking time:             {total_thinking_time:.1f}s ({total_thinking_time / 60:.1f} min)"
    )
    print()
    print(
        f"  Tool time ratio:                  {total_tool_time / total_claude_time * 100:.1f}%"
    )
    print(
        f"  Thinking time ratio:              {total_thinking_time / total_claude_time * 100:.1f}%"
    )
    print()

    # -------------------------------------------------------------------------
    # Per-task statistics
    # -------------------------------------------------------------------------
    tool_ratios = [d["tool_ratio"] for d in all_data]
    thinking_ratios = [d["thinking_ratio"] for d in all_data]

    print(sep)
    print("  PER-TASK STATISTICS")
    print(sep)
    print(f"  Avg tool time ratio:    {statistics.mean(tool_ratios):.1f}%")
    print(f"  Median tool time ratio: {statistics.median(tool_ratios):.1f}%")
    print(f"  Min tool time ratio:    {min(tool_ratios):.1f}%")
    print(f"  Max tool time ratio:    {max(tool_ratios):.1f}%")
    print()
    print(f"  Avg thinking time ratio:    {statistics.mean(thinking_ratios):.1f}%")
    print(f"  Median thinking time ratio: {statistics.median(thinking_ratios):.1f}%")
    print(f"  Min thinking time ratio:    {min(thinking_ratios):.1f}%")
    print(f"  Max thinking time ratio:    {max(thinking_ratios):.1f}%")
    print()

    # -------------------------------------------------------------------------
    # Distribution buckets
    # -------------------------------------------------------------------------
    buckets = [
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 40),
        (40, 50),
        (50, 60),
        (60, 70),
        (70, 80),
        (80, 90),
        (90, 100),
    ]
    print(sep)
    print("  TOOL TIME RATIO DISTRIBUTION")
    print(sep)

    for lo, hi in buckets:
        count_in_bucket = sum(1 for d in all_data if lo <= d["tool_ratio"] < hi)
        pct_bucket = count_in_bucket / len(all_data) * 100
        label = f"{lo:>2.0f}% - {hi:>3.0f}%"
        bar = "#" * int(pct_bucket / 2)
        print(f"  {label}: {count_in_bucket:>3} ({pct_bucket:>5.1f}%) {bar}")

    count_eq_100 = sum(1 for d in all_data if d["tool_ratio"] >= 100)
    pct_eq_100 = count_eq_100 / len(all_data) * 100
    label = ">= 100%"
    bar = "#" * int(pct_eq_100 / 2)
    print(f"  {label:>7}: {count_eq_100:>3} ({pct_eq_100:>5.1f}%) {bar}")
    print()

    # -------------------------------------------------------------------------
    # Top and bottom tasks by tool time ratio
    # -------------------------------------------------------------------------
    print(sep)
    print("  TOP 10 TASKS BY TOOL TIME RATIO")
    print(sep)

    sorted_by_tool_ratio = sorted(all_data, key=lambda d: d["tool_ratio"], reverse=True)
    print(
        f"  {'Rank':<6} {'Task':<55} {'Tool(s)':>8} {'Think(s)':>9} {'Ratio':>7} {'Status':>7}"
    )
    print(f"  {sep2}")
    for i, d in enumerate(sorted_by_tool_ratio[:10], 1):
        status = (
            "PASS" if d["success"] else ("FAIL" if d["success"] is not None else "???")
        )
        print(
            f"  {i:<6} {d['dir_name']:<55} {d['tool_time']:>7.1f} {d['thinking_time']:>8.1f} {d['tool_ratio']:>5.1f}% {status:>7}"
        )

    print()
    print(sep)
    print("  BOTTOM 10 TASKS BY TOOL TIME RATIO")
    print(sep)

    print(
        f"  {'Rank':<6} {'Task':<55} {'Tool(s)':>8} {'Think(s)':>9} {'Ratio':>7} {'Status':>7}"
    )
    print(f"  {sep2}")
    for i, d in enumerate(sorted_by_tool_ratio[-10:], 1):
        status = (
            "PASS" if d["success"] else ("FAIL" if d["success"] is not None else "???")
        )
        print(
            f"  {i:<6} {d['dir_name']:<55} {d['tool_time']:>7.1f} {d['thinking_time']:>8.1f} {d['tool_ratio']:>5.1f}% {status:>7}"
        )

    print()

    # -------------------------------------------------------------------------
    # Successful vs Failed tasks comparison
    # -------------------------------------------------------------------------
    print(sep)
    print("  SUCCESSFUL vs FAILED TASKS")
    print(sep)

    successful_tasks = [d for d in all_data if d["success"] is True]
    failed_tasks = [d for d in all_data if d["success"] is False]

    def summarize_group(tasks, label):
        if not tasks:
            print(f"  {label}: No tasks in this group.")
            return

        total_ct = sum(d["claude_time"] for d in tasks)
        total_tt = sum(d["tool_time"] for d in tasks)
        avg_tool_ratio = statistics.mean([d["tool_ratio"] for d in tasks])
        median_tool_ratio = statistics.median([d["tool_ratio"] for d in tasks])

        print(f"  {label} ({len(tasks)} tasks):")
        print(
            f"    Total execution time:     {total_ct:.1f}s ({total_ct / 60:.1f} min)"
        )
        print(
            f"    Total tool time:          {total_tt:.1f}s ({total_tt / 60:.1f} min)"
        )
        print(f"    Avg tool time ratio:      {avg_tool_ratio:.1f}%")
        print(f"    Median tool time ratio:   {median_tool_ratio:.1f}%")

    summarize_group(successful_tasks, "SUCCESSFUL")
    print()
    summarize_group(failed_tasks, "FAILED")
    print()

    if successful_tasks and failed_tasks:
        print(f"  KEY DIFFERENCES:")
        s_avg_ratio = statistics.mean([d["tool_ratio"] for d in successful_tasks])
        f_avg_ratio = statistics.mean([d["tool_ratio"] for d in failed_tasks])
        s_median_ratio = statistics.median([d["tool_ratio"] for d in successful_tasks])
        f_median_ratio = statistics.median([d["tool_ratio"] for d in failed_tasks])
        print(
            f"    Avg tool time ratio:  Successful {s_avg_ratio:.1f}% vs Failed {f_avg_ratio:.1f}% (diff: {f_avg_ratio - s_avg_ratio:+.1f}%)"
        )
        print(
            f"    Median tool time ratio: Successful {s_median_ratio:.1f}% vs Failed {f_median_ratio:.1f}% (diff: {f_median_ratio - s_median_ratio:+.1f}%)"
        )

    print()

    # -------------------------------------------------------------------------
    # Per-task summary table
    # -------------------------------------------------------------------------
    print(sep)
    print("  APPENDIX: PER-TASK SUMMARY")
    print(sep)

    all_data_sorted = sorted(all_data, key=lambda d: d["tool_ratio"], reverse=True)

    print(
        f"  {'Task':<55} {'Total(s)':>10} {'Tool(s)':>8} {'Think(s)':>9} {'Tool%':>7} {'Status':>7}"
    )
    print(f"  {sep2}")
    for d in all_data_sorted:
        status = (
            "PASS" if d["success"] else ("FAIL" if d["success"] is not None else "???")
        )
        print(
            f"  {d['dir_name']:<55} {d['claude_time']:>9.1f} {d['tool_time']:>7.1f} {d['thinking_time']:>8.1f} {d['tool_ratio']:>5.1f}% {status:>7}"
        )

    print()
    print(sep)
    print("  END OF REPORT")
    print(sep)


if __name__ == "__main__":
    main()
