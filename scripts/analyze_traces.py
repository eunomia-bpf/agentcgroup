#!/usr/bin/env python3
"""
Analyze SWE-agent trajectories and match with available Docker images.

This script:
1. Loads SWE-agent trajectories (traces with steps)
2. Loads SWE-rebench dataset (docker images)
3. Joins by instance_id to find traces with available environments
4. Reports statistics
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


def load_sweagent_trajectories(max_shards=1):
    """Load SWE-agent trajectory data."""
    cache_path = Path.home() / '.cache/huggingface/hub/datasets--nebius--SWE-agent-trajectories/snapshots'
    parquet_files = sorted(cache_path.glob('*/data/train-*.parquet'))

    if not parquet_files:
        print("Downloading SWE-agent trajectories (first shard)...")
        path = hf_hub_download(
            repo_id='nebius/SWE-agent-trajectories',
            repo_type='dataset',
            filename='data/train-00000-of-00012.parquet'
        )
        parquet_files = [Path(path)]

    dfs = []
    for pf in parquet_files[:max_shards]:
        print(f"Loading {pf.name}...")
        dfs.append(pd.read_parquet(pf))

    return pd.concat(dfs, ignore_index=True)


def load_swebench_images():
    """Load SWE-rebench dataset to get docker image mapping."""
    print("Loading SWE-rebench dataset (docker images)...")

    # Load both test shards
    dfs = []
    for i in range(2):
        path = hf_hub_download(
            repo_id='nebius/SWE-rebench',
            repo_type='dataset',
            filename=f'data/test-0000{i}-of-00002.parquet'
        )
        dfs.append(pd.read_parquet(path))

    df = pd.concat(dfs, ignore_index=True)

    # Keep only instance_id and docker_image
    return df[['instance_id', 'docker_image', 'repo']].copy()


def analyze_traces(trajs_df, images_df):
    """Analyze traces and match with docker images."""

    # Calculate trajectory stats
    trajs_df['num_steps'] = trajs_df['trajectory'].apply(len)
    trajs_df['num_bash'] = trajs_df['trajectory'].apply(
        lambda t: sum(1 for m in t if m.get('role') == 'ai')
    )

    # Merge with docker images
    merged = trajs_df.merge(images_df, on='instance_id', how='left')

    # Identify traces with docker images
    merged['has_docker'] = merged['docker_image'].notna() & (merged['docker_image'] != '')

    return merged


def print_report(df):
    """Print analysis report."""
    print("\n" + "=" * 60)
    print("SWE-agent Trajectories Analysis")
    print("=" * 60)

    total = len(df)
    unique_instances = df['instance_id'].nunique()
    with_docker = df['has_docker'].sum()
    unique_with_docker = df[df['has_docker']]['instance_id'].nunique()

    print(f"\nTotal trajectories: {total}")
    print(f"Unique instances: {unique_instances}")
    print(f"Trajectories with Docker image: {with_docker} ({100*with_docker/total:.1f}%)")
    print(f"Unique instances with Docker: {unique_with_docker}")

    # Step statistics
    print("\n--- Step Statistics (all traces) ---")
    print(f"Steps: min={df['num_steps'].min()}, max={df['num_steps'].max()}, "
          f"mean={df['num_steps'].mean():.1f}, median={df['num_steps'].median():.0f}")

    # For traces with docker
    docker_df = df[df['has_docker']]
    if len(docker_df) > 0:
        print("\n--- Step Statistics (traces with Docker) ---")
        print(f"Steps: min={docker_df['num_steps'].min()}, max={docker_df['num_steps'].max()}, "
              f"mean={docker_df['num_steps'].mean():.1f}, median={docker_df['num_steps'].median():.0f}")

    # Exit status distribution
    print("\n--- Exit Status Distribution ---")
    for status, count in df['exit_status'].value_counts().head(5).items():
        docker_count = len(df[(df['exit_status'] == status) & df['has_docker']])
        print(f"  {status}: {count} ({docker_count} with docker)")

    # Model distribution
    print("\n--- Model Distribution ---")
    for model, count in df['model_name'].value_counts().head(5).items():
        docker_count = len(df[(df['model_name'] == model) & df['has_docker']])
        print(f"  {model}: {count} ({docker_count} with docker)")

    # Step distribution buckets
    print("\n--- Step Count Distribution (traces with Docker) ---")
    if len(docker_df) > 0:
        bins = [0, 10, 20, 30, 50, 100, 200, 1000]
        labels = ['1-10', '11-20', '21-30', '31-50', '51-100', '101-200', '200+']
        docker_df = docker_df.copy()
        docker_df['step_bucket'] = pd.cut(docker_df['num_steps'], bins=bins, labels=labels)
        for bucket, count in docker_df['step_bucket'].value_counts().sort_index().items():
            print(f"  {bucket} steps: {count}")

    return docker_df


def export_runnable_traces(df, output_path, limit=100):
    """Export list of traces that can be run (have docker images)."""
    docker_df = df[df['has_docker']].copy()

    # Sort by step count (prefer medium-length traces)
    docker_df['step_score'] = abs(docker_df['num_steps'] - 30)  # prefer ~30 steps
    docker_df = docker_df.sort_values('step_score')

    # Select diverse set
    selected = docker_df.head(limit)

    export_data = []
    for _, row in selected.iterrows():
        export_data.append({
            'instance_id': row['instance_id'],
            'docker_image': row['docker_image'],
            'num_steps': row['num_steps'],
            'model': row['model_name'],
            'exit_status': row['exit_status']
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(export_data).to_csv(output_path, index=False)
    print(f"\nExported {len(export_data)} runnable traces to {output_path}")

    # Also print top 10
    print("\n--- Top 10 Recommended Traces ---")
    print(f"{'Instance ID':<45} {'Steps':>6} {'Docker Image':<50}")
    print("-" * 105)
    for item in export_data[:10]:
        print(f"{item['instance_id']:<45} {item['num_steps']:>6} {item['docker_image']:<50}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SWE-agent trajectories")
    parser.add_argument("--shards", type=int, default=1, help="Number of trajectory shards to load (1-12)")
    parser.add_argument("--output", type=Path, default="data/runnable_traces.csv", help="Output CSV path")
    parser.add_argument("--limit", type=int, default=100, help="Max traces to export")
    args = parser.parse_args()

    # Load data
    trajs_df = load_sweagent_trajectories(max_shards=args.shards)
    images_df = load_swebench_images()

    # Analyze
    merged_df = analyze_traces(trajs_df, images_df)

    # Report
    docker_df = print_report(merged_df)

    # Export runnable traces
    export_runnable_traces(merged_df, args.output, args.limit)


if __name__ == "__main__":
    main()
