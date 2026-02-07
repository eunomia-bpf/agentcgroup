#!/usr/bin/env python3
"""
Filter valid tasks from experiment datasets.

Scans experiment directories, checks each task for valid data, and outputs
a JSON list of tasks that pass all quality filters. The output can be
reused by other analysis scripts.

Filters applied:
1. resources.json must exist with valid summary
2. Resource sampling must have >= MIN_SAMPLES samples
3. Resource sampling duration must be >= MIN_DURATION seconds
4. Optionally require presence in multiple datasets (--common)

Usage:
    # Filter single dataset
    python filter_valid_tasks.py experiments/all_images_local/

    # Filter and find common valid tasks across two datasets
    python filter_valid_tasks.py experiments/all_images_haiku/ experiments/all_images_local/ --common

    # Custom thresholds
    python filter_valid_tasks.py experiments/all_images_local/ --min-duration 90 --min-samples 20

    # Output to JSON file for reuse
    python filter_valid_tasks.py experiments/all_images_haiku/ experiments/all_images_local/ --common -o valid_tasks.json
"""

import argparse
import json
import os
import glob
import sys


MIN_DURATION = 60   # seconds
MIN_SAMPLES = 10    # minimum resource sample points


def get_valid_task_names(base_dir, min_duration=MIN_DURATION, min_samples=MIN_SAMPLES):
    """Convenience: return a sorted list of valid task directory names.

    This is the primary API for other analysis scripts to filter tasks.
    """
    valid, _ = scan_dataset(base_dir, min_duration, min_samples)
    return sorted(t["task"] for t in valid)


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None


def check_task(base_dir, task_name, min_duration, min_samples):
    """Check if a task has valid, usable data.

    Returns a dict with task info if valid, None otherwise.
    """
    task_dir = os.path.join(base_dir, task_name)
    attempts = glob.glob(os.path.join(task_dir, "attempt_*"))
    if not attempts:
        return None, "no attempt directory"

    attempt_dir = sorted(attempts)[-1]
    resources_path = os.path.join(attempt_dir, "resources.json")
    results_path = os.path.join(attempt_dir, "results.json")

    # Check resources.json
    resources = load_json(resources_path)
    if not resources:
        return None, "no resources.json"

    summary = resources.get("summary", {})
    sample_count = summary.get("sample_count", 0)
    duration = summary.get("duration_seconds", 0)

    if sample_count < min_samples:
        return None, f"too few samples ({sample_count} < {min_samples})"

    if duration < min_duration:
        return None, f"duration too short ({duration:.0f}s < {min_duration}s)"

    # Check results.json (optional but useful)
    results = load_json(results_path)
    claude_time = 0
    total_time = 0
    if results:
        claude_time = results.get("claude_time", 0)
        total_time = results.get("total_time", 0)

    # Check tool_calls.json
    tool_calls_path = os.path.join(attempt_dir, "tool_calls.json")
    tool_calls = load_json(tool_calls_path)
    num_tool_calls = len(tool_calls) if tool_calls else 0

    return {
        "task": task_name,
        "attempt_dir": attempt_dir,
        "duration": round(duration, 1),
        "sample_count": sample_count,
        "claude_time": round(claude_time, 1),
        "total_time": round(total_time, 1),
        "peak_mem_mb": round(summary.get("memory_mb", {}).get("max", 0), 1),
        "avg_cpu_pct": round(summary.get("cpu_percent", {}).get("avg", 0), 1),
        "num_tool_calls": num_tool_calls,
    }, None


def scan_dataset(base_dir, min_duration, min_samples):
    """Scan a dataset directory and return valid/invalid task lists."""
    valid = []
    invalid = []

    if not os.path.isdir(base_dir):
        print(f"ERROR: Directory not found: {base_dir}", file=sys.stderr)
        return valid, invalid

    for entry in sorted(os.listdir(base_dir)):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        info, reason = check_task(base_dir, entry, min_duration, min_samples)
        if info:
            valid.append(info)
        else:
            invalid.append({"task": entry, "reason": reason})

    return valid, invalid


def main():
    parser = argparse.ArgumentParser(
        description="Filter valid tasks from experiment datasets"
    )
    parser.add_argument(
        "dirs", nargs="+",
        help="One or more experiment directories to scan"
    )
    parser.add_argument(
        "--common", action="store_true",
        help="Only output tasks valid in ALL specified directories"
    )
    parser.add_argument(
        "--min-duration", type=int, default=MIN_DURATION,
        help=f"Minimum resource sampling duration in seconds (default: {MIN_DURATION})"
    )
    parser.add_argument(
        "--min-samples", type=int, default=MIN_SAMPLES,
        help=f"Minimum number of resource samples (default: {MIN_SAMPLES})"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path (default: print to stdout)"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress summary output, only write JSON"
    )
    args = parser.parse_args()

    all_results = {}

    for d in args.dirs:
        d = os.path.abspath(d)
        dataset_name = os.path.basename(d)

        if not args.quiet:
            print(f"\n{'='*70}", file=sys.stderr)
            print(f"Scanning: {d}", file=sys.stderr)
            print(f"{'='*70}", file=sys.stderr)

        valid, invalid = scan_dataset(d, args.min_duration, args.min_samples)

        if not args.quiet:
            print(f"  Valid:   {len(valid)}", file=sys.stderr)
            print(f"  Invalid: {len(invalid)}", file=sys.stderr)
            if invalid:
                for inv in invalid:
                    print(f"    - {inv['task']}: {inv['reason']}", file=sys.stderr)

        all_results[dataset_name] = {
            "directory": d,
            "valid_tasks": valid,
            "invalid_tasks": invalid,
            "valid_count": len(valid),
            "invalid_count": len(invalid),
        }

    # If --common, find intersection
    if args.common and len(args.dirs) > 1:
        dataset_names = list(all_results.keys())
        valid_sets = []
        for name in dataset_names:
            task_names = set(t["task"] for t in all_results[name]["valid_tasks"])
            valid_sets.append(task_names)

        common_tasks = valid_sets[0]
        for s in valid_sets[1:]:
            common_tasks = common_tasks & s

        common_tasks = sorted(common_tasks)

        if not args.quiet:
            print(f"\n{'='*70}", file=sys.stderr)
            print(f"Common valid tasks across {len(args.dirs)} datasets: {len(common_tasks)}", file=sys.stderr)
            print(f"{'='*70}", file=sys.stderr)
            for t in common_tasks:
                # Show info from each dataset
                parts = []
                for name in dataset_names:
                    for task_info in all_results[name]["valid_tasks"]:
                        if task_info["task"] == t:
                            parts.append(f"{name}: {task_info['duration']:.0f}s")
                            break
                print(f"  {t}  ({', '.join(parts)})", file=sys.stderr)

        # Build combined output
        output = {
            "filter_config": {
                "min_duration": args.min_duration,
                "min_samples": args.min_samples,
                "mode": "common",
            },
            "common_tasks": common_tasks,
            "common_count": len(common_tasks),
            "datasets": {},
        }
        for name in dataset_names:
            per_dataset = {}
            for task_info in all_results[name]["valid_tasks"]:
                if task_info["task"] in common_tasks:
                    per_dataset[task_info["task"]] = task_info
            output["datasets"][name] = {
                "directory": all_results[name]["directory"],
                "tasks": per_dataset,
            }
    else:
        output = {
            "filter_config": {
                "min_duration": args.min_duration,
                "min_samples": args.min_samples,
                "mode": "individual",
            },
            "datasets": {},
        }
        for name, data in all_results.items():
            output["datasets"][name] = {
                "directory": data["directory"],
                "valid_count": data["valid_count"],
                "invalid_count": data["invalid_count"],
                "valid_tasks": data["valid_tasks"],
            }

    # Output
    json_str = json.dumps(output, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
        if not args.quiet:
            print(f"\nOutput saved to: {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
