#!/usr/bin/env python3
"""
One-off empirical study builder.

Generates:
  - analysis_summary.json
  - figures/*.png (multi-figure)
  - EMPIRICAL_STUDY.md
from runs under experiments/empirical_study_20260305_full/runs
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
FIG_DIR = ROOT / "figures"
BRANCHFS_DIR = ROOT.parent / "branchfs_motivation"

SOURCE_IMAGES = {
    "starlette": "swerebench/sweb.eval.x86_64.encode_1776_starlette-1147",
    "diffcover": "swerebench/sweb.eval.x86_64.bachmann1234_1776_diff_cover-210",
    "azure_msrest": "swerebench/sweb.eval.x86_64.azure_1776_msrest-for-python-224",
    "pytorch_ignite": "swerebench/sweb.eval.x86_64.pytorch_1776_ignite-1077",
}

FS_SUMMARY_TYPES = {"WRITE", "DIR_CREATE", "FILE_TRUNCATE", "FILE_DELETE", "FILE_RENAME", "CHDIR"}


def run_cmd(cmd: List[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout


def run_bash_in_image(image: str, script: str) -> str:
    return run_cmd(["podman", "run", "--rm", "--entrypoint", "bash", image, "-lc", script])


def parse_du_single_mb(image: str, path: str) -> float:
    out = run_bash_in_image(image, f"du -sm {path} 2>/dev/null | awk '{{print $1}}' | head -n1 || true").strip()
    try:
        return float(out)
    except Exception:
        return 0.0


def identify_repo_key(image_name: str) -> str:
    if "encode_1776_starlette-1147" in image_name:
        return "starlette"
    if "bachmann1234_1776_diff_cover-210" in image_name:
        return "diffcover"
    if "azure_1776_msrest-for-python-224" in image_name:
        return "azure_msrest"
    if "pytorch_1776_ignite-1077" in image_name:
        return "pytorch_ignite"
    return image_name.replace("/", "_")


def parse_dynamic_run(run_dir: Path, require_success: bool = True) -> Dict | None:
    manifest_path = run_dir / "run_manifest.json"
    results_path = run_dir / "swebench" / "results.json"
    trace_path = run_dir / "ebpf_trace.jsonl"
    if not manifest_path.exists() or not results_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text())
    results = json.loads(results_path.read_text())

    error = manifest.get("error") or results.get("error")
    claude_exit = (results.get("claude_output") or {}).get("exit_code")
    tool_calls = len((results.get("traces") or {}).get("tool_calls", []))
    if require_success and (error or claude_exit != 0 or tool_calls == 0):
        return None

    events = Counter()
    summary_counts = Counter()
    summary_bytes = Counter()
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = obj.get("event", "UNKNOWN")
            events[ev] += 1
            if ev == "SUMMARY":
                t = obj.get("type", "UNKNOWN")
                summary_counts[t] += int(obj.get("count", 1))
                summary_bytes[t] += int(obj.get("total_bytes", 0))

    rs = (results.get("resource_samples") or {}).get("summary", {})
    mem = rs.get("memory_mb", {})
    cpu = rs.get("cpu_percent", {})
    disk_mb = (results.get("disk_usage") or {}).get("testbed_mb")

    image = manifest.get("image", "")
    key = identify_repo_key(image)
    total_time_s = float(results.get("total_time", 0.0) or 0.0)
    write_mb = float(summary_bytes.get("WRITE", 0) / (1024 * 1024))
    summary_total = int(sum(summary_counts.values()))
    fs_summary_total = int(sum(summary_counts.get(t, 0) for t in FS_SUMMARY_TYPES))

    return {
        "run_name": run_dir.name,
        "repo_key": key,
        "image": image,
        "error": error,
        "claude_exit": claude_exit,
        "total_time_s": total_time_s,
        "claude_time_s": float(results.get("claude_time", 0.0) or 0.0),
        "tool_calls": int(tool_calls),
        "mem_avg_mb": float(mem.get("avg", 0.0) or 0.0),
        "mem_max_mb": float(mem.get("max", 0.0) or 0.0),
        "cpu_avg_pct": float(cpu.get("avg", 0.0) or 0.0),
        "cpu_max_pct": float(cpu.get("max", 0.0) or 0.0),
        "disk_testbed_mb": float(disk_mb or 0.0),
        "event_total": int(sum(events.values())),
        "event_counts": dict(events),
        "summary_counts": dict(summary_counts),
        "summary_bytes": dict(summary_bytes),
        "summary_total": summary_total,
        "fs_summary_total": fs_summary_total,
        "fs_summary_share": float(fs_summary_total / summary_total) if summary_total > 0 else 0.0,
        "write_mb": write_mb,
        "event_per_s": float((sum(events.values()) / total_time_s) if total_time_s > 0 else 0.0),
        "write_mb_per_s": float((write_mb / total_time_s) if total_time_s > 0 else 0.0),
        "event_per_tool_call": float((sum(events.values()) / tool_calls) if tool_calls > 0 else 0.0),
        "write_mb_per_tool_call": float((write_mb / tool_calls) if tool_calls > 0 else 0.0),
        "top_summary_types": sorted(summary_counts.items(), key=lambda x: x[1], reverse=True)[:8],
    }


def collect_dynamic() -> List[Dict]:
    rows = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        d = parse_dynamic_run(run_dir, require_success=True)
        if d:
            rows.append(d)
    return rows


def collect_starlette_repeats() -> Dict:
    candidates: List[Dict] = []
    if not BRANCHFS_DIR.exists():
        return {"n": 0, "runs": [], "stats": {}}

    patterns = [
        "swebench_example_*",
        "sweb.eval.x86_64.encode_1776_starlette-1147_*",
        "sweb1147_100ms_*",
    ]
    seen = set()
    for pat in patterns:
        for run_dir in BRANCHFS_DIR.glob(pat):
            if not run_dir.is_dir():
                continue
            if run_dir in seen:
                continue
            seen.add(run_dir)
            d = parse_dynamic_run(run_dir, require_success=True)
            if d and d["repo_key"] == "starlette":
                candidates.append(d)

    metrics = {
        "total_time_s": [x["total_time_s"] for x in candidates],
        "event_total": [x["event_total"] for x in candidates],
        "write_mb": [x["write_mb"] for x in candidates],
        "tool_calls": [x["tool_calls"] for x in candidates],
    }
    stats = {}
    for k, vals in metrics.items():
        if not vals:
            continue
        mu = mean(vals)
        sd = pstdev(vals) if len(vals) > 1 else 0.0
        stats[k] = {
            "mean": mu,
            "stddev": sd,
            "cv": (sd / mu) if mu > 1e-9 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }
    return {"n": len(candidates), "runs": candidates, "stats": stats}


def collect_static() -> Dict[str, Dict]:
    static = {}
    for key, image in SOURCE_IMAGES.items():
        inspect = json.loads(run_cmd(["podman", "image", "inspect", image]))[0]
        size_mb = float(inspect.get("Size", 0) / (1024 * 1024))
        layers = inspect.get("RootFS", {}).get("Layers", [])

        # testbed env pip list
        pip_json = run_bash_in_image(image, "/opt/conda/envs/testbed/bin/python -m pip list --format=json")
        pip_list = json.loads(pip_json)
        pip_pkgs = sorted({x["name"].lower() for x in pip_list})

        static[key] = {
            "image": image,
            "image_size_mb": size_mb,
            "layer_count": len(layers),
            "pip_testbed_count": len(pip_pkgs),
            "pip_testbed_pkgs": pip_pkgs,
            "opt_conda_mb": parse_du_single_mb(image, "/opt/conda"),
            "opt_conda_pkgs_mb": parse_du_single_mb(image, "/opt/conda/pkgs"),
            "opt_conda_env_testbed_mb": parse_du_single_mb(image, "/opt/conda/envs/testbed"),
            "root_pip_cache_mb": parse_du_single_mb(image, "/root/.cache/pip"),
        }
        static[key]["reclaim_mb"] = static[key]["opt_conda_pkgs_mb"] + static[key]["root_pip_cache_mb"]
        static[key]["reclaim_pct"] = (
            static[key]["reclaim_mb"] / static[key]["image_size_mb"] * 100.0 if static[key]["image_size_mb"] > 0 else 0.0
        )
    return static


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def build_overlap_matrix(static: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    keys = list(static.keys())
    mat: Dict[str, Dict[str, float]] = {}
    for a in keys:
        mat[a] = {}
        for b in keys:
            ma = set(static[a]["pip_testbed_pkgs"])
            mb = set(static[b]["pip_testbed_pkgs"])
            mat[a][b] = jaccard(ma, mb)
    return mat


def plot_image_sizes(static: Dict[str, Dict]) -> None:
    labels = list(static.keys())
    vals = [static[k]["image_size_mb"] / 1024.0 for k in labels]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, vals, color="tab:blue")
    plt.ylabel("Image Size (GB)")
    plt.title("Image Size by Repo")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_image_sizes.png", dpi=160)
    plt.close()


def plot_runtime(dynamic: List[Dict]) -> None:
    labels = [d["repo_key"] for d in dynamic]
    t = [d["total_time_s"] for d in dynamic]
    tc = [d["tool_calls"] for d in dynamic]
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.bar(labels, t, color="tab:green", alpha=0.75)
    ax1.set_ylabel("Runtime (s)")
    ax1.set_title("Runtime and Tool Calls by Repo")
    ax2 = ax1.twinx()
    ax2.plot(labels, tc, color="tab:red", marker="o")
    ax2.set_ylabel("Tool Calls")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_runtime_toolcalls.png", dpi=160)
    plt.close(fig)


def plot_write_volume(dynamic: List[Dict]) -> None:
    labels = [d["repo_key"] for d in dynamic]
    w = [max(d["write_mb"], 1e-6) for d in dynamic]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, w, color="tab:purple")
    plt.yscale("log")
    plt.ylabel("WRITE MB (log scale)")
    plt.title("WRITE Volume by Repo (eBPF SUMMARY)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_write_volume_log.png", dpi=160)
    plt.close()


def plot_overlap_heatmap(overlap: Dict[str, Dict[str, float]]) -> None:
    keys = list(overlap.keys())
    mat = [[overlap[a][b] for b in keys] for a in keys]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, vmin=0, vmax=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=25, ha="right")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)
    ax.set_title("Testbed pip overlap (Jaccard)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Jaccard")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_pip_overlap_heatmap.png", dpi=160)
    plt.close(fig)


def plot_space_hotspots(static: Dict[str, Dict]) -> None:
    keys = list(static.keys())
    conda_pkgs = [static[k]["opt_conda_pkgs_mb"] for k in keys]
    pip_cache = [static[k]["root_pip_cache_mb"] for k in keys]
    env = [static[k]["opt_conda_env_testbed_mb"] for k in keys]

    x = list(range(len(keys)))
    w = 0.25
    plt.figure(figsize=(10, 4.5))
    plt.bar([i - w for i in x], conda_pkgs, width=w, label="/opt/conda/pkgs MB")
    plt.bar(x, pip_cache, width=w, label="/root/.cache/pip MB")
    plt.bar([i + w for i in x], env, width=w, label="/opt/conda/envs/testbed MB")
    plt.xticks(x, keys)
    plt.ylabel("MB")
    plt.title("Space Hotspots by Repo Image")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_space_hotspots.png", dpi=160)
    plt.close()


def plot_event_mix(dynamic: List[Dict]) -> None:
    labels = [d["repo_key"] for d in dynamic]
    file_open = [d["event_counts"].get("FILE_OPEN", 0) for d in dynamic]
    summary = [d["event_counts"].get("SUMMARY", 0) for d in dynamic]
    execs = [d["event_counts"].get("EXEC", 0) for d in dynamic]
    exits = [d["event_counts"].get("EXIT", 0) for d in dynamic]
    other = [
        max(
            d["event_total"]
            - d["event_counts"].get("FILE_OPEN", 0)
            - d["event_counts"].get("SUMMARY", 0)
            - d["event_counts"].get("EXEC", 0)
            - d["event_counts"].get("EXIT", 0),
            0,
        )
        for d in dynamic
    ]

    plt.figure(figsize=(10, 4.8))
    plt.bar(labels, file_open, label="FILE_OPEN")
    plt.bar(labels, summary, bottom=file_open, label="SUMMARY")
    b2 = [file_open[i] + summary[i] for i in range(len(labels))]
    plt.bar(labels, execs, bottom=b2, label="EXEC")
    b3 = [b2[i] + execs[i] for i in range(len(labels))]
    plt.bar(labels, exits, bottom=b3, label="EXIT")
    b4 = [b3[i] + exits[i] for i in range(len(labels))]
    plt.bar(labels, other, bottom=b4, label="OTHER")
    plt.ylabel("event count")
    plt.title("eBPF Event Composition by Repo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_event_mix_stacked.png", dpi=160)
    plt.close()


def plot_normalized_pressure(dynamic: List[Dict]) -> None:
    labels = [d["repo_key"] for d in dynamic]
    e_per_s = [d["event_per_s"] for d in dynamic]
    w_per_s = [d["write_mb_per_s"] for d in dynamic]
    fs_share = [d["fs_summary_share"] * 100.0 for d in dynamic]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(labels, e_per_s, color="tab:orange")
    axes[0].set_title("Events/s")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labels, w_per_s, color="tab:purple")
    axes[1].set_title("WRITE MB/s")
    axes[1].tick_params(axis="x", rotation=20)
    axes[2].bar(labels, fs_share, color="tab:green")
    axes[2].set_title("FS SUMMARY share (%)")
    axes[2].tick_params(axis="x", rotation=20)
    fig.suptitle("Normalized Runtime Pressure")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_normalized_pressure.png", dpi=160)
    plt.close(fig)


def plot_cache_reclaim(static: Dict[str, Dict]) -> None:
    labels = list(static.keys())
    reclaim_mb = [static[k]["reclaim_mb"] for k in labels]
    reclaim_pct = [static[k]["reclaim_pct"] for k in labels]

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.bar(labels, reclaim_mb, color="tab:brown", alpha=0.7)
    ax1.set_ylabel("Reclaimable MB")
    ax1.set_title("Estimated Low-Risk Space Reclaim")
    ax2 = ax1.twinx()
    ax2.plot(labels, reclaim_pct, color="tab:red", marker="o")
    ax2.set_ylabel("Reclaim % of image")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_cache_reclaim.png", dpi=160)
    plt.close(fig)


def plot_summary_heatmap(dynamic: List[Dict]) -> None:
    type_counter = Counter()
    for d in dynamic:
        type_counter.update(d["summary_counts"])
    top_types = [x[0] for x in type_counter.most_common(8)]
    labels = [d["repo_key"] for d in dynamic]
    mat = []
    for d in dynamic:
        row = [math.log10(d["summary_counts"].get(t, 0) + 1) for t in top_types]
        mat.append(row)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(top_types)))
    ax.set_xticklabels(top_types, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("SUMMARY type intensity by repo (log10(count+1))")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(count+1)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_summary_heatmap.png", dpi=160)
    plt.close(fig)


def plot_starlette_repeat(stability: Dict) -> None:
    if stability.get("n", 0) < 2:
        return
    stats = stability.get("stats", {})
    keys = ["total_time_s", "event_total", "write_mb", "tool_calls"]
    labels = [k for k in keys if k in stats]
    cv = [stats[k]["cv"] * 100.0 for k in labels]
    plt.figure(figsize=(8, 4.2))
    plt.bar(labels, cv, color="tab:cyan")
    plt.ylabel("CV (%)")
    plt.title("Starlette Repeatability (branchfs_motivation runs)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_starlette_repeat_cv.png", dpi=160)
    plt.close()


def fmt(x: float) -> str:
    return f"{x:.2f}"


def write_markdown(
    dynamic: List[Dict], static: Dict[str, Dict], overlap: Dict[str, Dict[str, float]], stability: Dict
) -> None:
    dynamic = sorted(dynamic, key=lambda d: d["repo_key"])
    lines: List[str] = []
    lines.append("# Comprehensive Empirical Study: Cross-Repo SWE-bench Runtime and Image Overhead")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat()}")
    lines.append("- Scope: 4 repo images, eBPF-enabled runs, plus static dependency/layer analysis.")
    lines.append("")
    lines.append("## Abstract")
    lines.append("We evaluate cross-repo variability in runtime behavior and storage overhead for SWE-bench container workloads.")
    lines.append("Results show substantial heterogeneity in write volume and runtime. Static image analysis indicates large removable cache overhead and low overlap in testbed pip dependencies.")
    lines.append("")
    lines.append("## Method")
    lines.append("- Runtime runner: `scripts/run_swebench_new.py`")
    lines.append("- Runtime flags: `--trace-all --trace-cgroup-children --trace-resources --resource-detail --sample-interval 100 --resource-monitor-interval 0.5`")
    lines.append("- Model: `haiku`")
    lines.append("- Dynamic metrics from: `results.json`, `resources.json`, `tool_calls.json`, `ebpf_trace.jsonl`")
    lines.append("- Static metrics from: image inspect + `/opt/conda`, `/root/.cache/pip`, testbed pip sets")
    lines.append("")
    lines.append("## What Is New vs Original AgentCgroup Paper Dataset")
    lines.append("1. Syscall-level FS/NET/MMAP/SIGNAL process behavior via eBPF SUMMARY is now available (not only tool trace + CPU/memory).")
    lines.append("2. Per-repo write amplification and FS pressure differences are directly quantifiable.")
    lines.append("3. Image-internal reclaim opportunity can be estimated from real path-level size hotspots.")
    lines.append("4. Cross-repo dependency overlap in the actual `testbed` env is measured (Jaccard matrix), informing image strategy.")
    lines.append("")
    lines.append("## RQ1: Runtime and eBPF behavior")
    lines.append("| repo | runtime_s | tool_calls | event_total | write_mb | event_per_s | write_mb_per_s | fs_summary_share_% | mem_avg_mb | cpu_avg_pct |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for d in dynamic:
        lines.append(
            f"| {d['repo_key']} | {fmt(d['total_time_s'])} | {d['tool_calls']} | {d['event_total']} | {fmt(d['write_mb'])} | {fmt(d['event_per_s'])} | {fmt(d['write_mb_per_s'])} | {fmt(d['fs_summary_share']*100)} | {fmt(d['mem_avg_mb'])} | {fmt(d['cpu_avg_pct'])} |"
        )
    lines.append("")
    lines.append("Key findings:")
    max_runtime = max(dynamic, key=lambda d: d["total_time_s"])
    max_write = max(dynamic, key=lambda d: d["write_mb"])
    lines.append(
        f"1. Longest runtime is `{max_runtime['repo_key']}` at {fmt(max_runtime['total_time_s'])}s."
    )
    lines.append(
        f"2. Largest write volume is `{max_write['repo_key']}` at {fmt(max_write['write_mb'])} MB."
    )
    if len(dynamic) >= 2:
        min_write = min(dynamic, key=lambda d: d["write_mb"])
        ratio = max_write["write_mb"] / max(min_write["write_mb"], 1e-6)
        lines.append(
            f"3. Write volume variability is high: max/min ratio is {ratio:.1f}x."
        )
    lines.append("")
    lines.append("## RQ2: Static dependency overlap and size hotspots")
    lines.append("| repo | image_gb | layers | testbed_pip_count | /opt/conda MB | /opt/conda/pkgs MB | /root/.cache/pip MB | /opt/conda/envs/testbed MB |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for k in sorted(static.keys()):
        s = static[k]
        lines.append(
            f"| {k} | {fmt(s['image_size_mb']/1024)} | {s['layer_count']} | {s['pip_testbed_count']} | {fmt(s['opt_conda_mb'])} | {fmt(s['opt_conda_pkgs_mb'])} | {fmt(s['root_pip_cache_mb'])} | {fmt(s['opt_conda_env_testbed_mb'])} |"
        )
    lines.append("")
    lines.append("Pip-overlap (testbed env, Jaccard):")
    keys = sorted(overlap.keys())
    lines.append("| repo_a | repo_b | jaccard |")
    lines.append("| --- | --- | ---: |")
    for a in keys:
        for b in keys:
            if a >= b:
                continue
            lines.append(f"| {a} | {b} | {overlap[a][b]:.3f} |")
    lines.append("")
    lines.append("Key findings:")
    lines.append("1. `/opt/conda/pkgs` cache is consistently large (hundreds of MB per image).")
    lines.append("2. `/root/.cache/pip` is non-trivial, especially for heavy ML image.")
    lines.append("3. Testbed pip overlap is low-to-moderate, so full dependency unification is limited.")
    lines.append("")
    lines.append("## RQ3: Optimization opportunities")
    total_reclaim = sum(s["reclaim_mb"] for s in static.values())
    lines.append(
        f"- Low-risk immediate cleanup potential (conda pkg cache + pip cache): about {total_reclaim:.0f} MB across these 4 images."
    )
    lines.append("- Medium-term: split image families (ML vs non-ML) and tune base layers separately.")
    lines.append("- Long-term: standardize build pipeline for better cross-image layer reuse.")
    lines.append("")
    if stability.get("n", 0) > 0:
        lines.append("## Supplementary Robustness: Starlette Repeated Runs")
        lines.append(f"- Repeated valid runs found: {stability['n']}")
        st = stability.get("stats", {})
        if st:
            lines.append("| metric | mean | stddev | cv | min | max |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for k in ["total_time_s", "event_total", "write_mb", "tool_calls"]:
                if k not in st:
                    continue
                s = st[k]
                lines.append(
                    f"| {k} | {fmt(s['mean'])} | {fmt(s['stddev'])} | {fmt(s['cv'])} | {fmt(s['min'])} | {fmt(s['max'])} |"
                )
        lines.append("")
    lines.append("## Figures")
    lines.append("- `figures/fig1_image_sizes.png`")
    lines.append("- `figures/fig2_runtime_toolcalls.png`")
    lines.append("- `figures/fig3_write_volume_log.png`")
    lines.append("- `figures/fig4_pip_overlap_heatmap.png`")
    lines.append("- `figures/fig5_space_hotspots.png`")
    lines.append("- `figures/fig6_event_mix_stacked.png`")
    lines.append("- `figures/fig7_normalized_pressure.png`")
    lines.append("- `figures/fig8_cache_reclaim.png`")
    lines.append("- `figures/fig9_summary_heatmap.png`")
    if stability.get("n", 0) >= 2:
        lines.append("- `figures/fig10_starlette_repeat_cv.png`")
    lines.append("")
    lines.append("## Threats to Validity")
    lines.append("- Single run per repo in this batch for cross-repo comparison; model behavior stochasticity remains.")
    lines.append("- Runtime behavior includes agent decision variability, not only repo complexity.")
    lines.append("- Storage metrics are image-level and path-level proxies, not block-level physical dedupe.")
    lines.append("- One heavy case (`pytorch_ignite`) may dominate aggregate statistics; we report both raw and normalized views.")
    lines.append("")
    (ROOT / "EMPIRICAL_STUDY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    dynamic = collect_dynamic()
    if len(dynamic) < 4:
        raise RuntimeError(f"expected >=4 valid runs, got {len(dynamic)}")

    static = collect_static()
    overlap = build_overlap_matrix(static)
    stability = collect_starlette_repeats()

    plot_image_sizes(static)
    plot_runtime(dynamic)
    plot_write_volume(dynamic)
    plot_overlap_heatmap(overlap)
    plot_space_hotspots(static)
    plot_event_mix(dynamic)
    plot_normalized_pressure(dynamic)
    plot_cache_reclaim(static)
    plot_summary_heatmap(dynamic)
    plot_starlette_repeat(stability)

    summary = {
        "dynamic_runs": dynamic,
        "static_images": static,
        "pip_overlap_jaccard": overlap,
        "starlette_repeatability": stability,
    }
    (ROOT / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(dynamic, static, overlap, stability)

    print(f"Wrote: {ROOT / 'analysis_summary.json'}")
    print(f"Wrote: {ROOT / 'EMPIRICAL_STUDY.md'}")
    print(f"Wrote figs in: {FIG_DIR}")


if __name__ == "__main__":
    main()
