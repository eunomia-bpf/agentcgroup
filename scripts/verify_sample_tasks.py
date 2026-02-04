#!/usr/bin/env python3
"""
Verify sample task images exist and find replacements if not.

Usage:
    python scripts/verify_sample_tasks.py
"""

import subprocess
import sys
from datasets import load_dataset

# Current sample tasks from batch_test_swebench.py
SAMPLE_TASKS = {
    # SQL/Data
    ('SQL/Data', 'Easy'): {
        'instance_id': 'sqlfluff__sqlfluff-5362',
        'docker_image': 'swerebench/sweb.eval.x86_64.sqlfluff_1776_sqlfluff-5362',
    },
    ('SQL/Data', 'Medium'): {
        'instance_id': 'tobymao__sqlglot-1177',
        'docker_image': 'swerebench/sweb.eval.x86_64.tobymao_1776_sqlglot-1177',
    },
    ('SQL/Data', 'Hard'): {
        'instance_id': 'reata__sqllineage-438',
        'docker_image': 'swerebench/sweb.eval.x86_64.reata_1776_sqllineage-438',
    },

    # DevOps/Build
    ('DevOps/Build', 'Easy'): {
        'instance_id': 'pre-commit__pre-commit-2524',
        'docker_image': 'swerebench/sweb.eval.x86_64.pre-commit_1776_pre-commit-2524',
    },
    ('DevOps/Build', 'Medium'): {
        'instance_id': 'beeware__briefcase-1525',
        'docker_image': 'swerebench/sweb.eval.x86_64.beeware_1776_briefcase-1525',
    },
    ('DevOps/Build', 'Hard'): {
        'instance_id': 'iterative__dvc-777',
        'docker_image': 'swerebench/sweb.eval.x86_64.iterative_1776_dvc-777',
    },

    # ML/Scientific
    ('ML/Scientific', 'Easy'): {
        'instance_id': 'dask__dask-5510',
        'docker_image': 'swerebench/sweb.eval.x86_64.dask_1776_dask-5510',
    },
    ('ML/Scientific', 'Medium'): {
        'instance_id': 'dask__dask-11628',
        'docker_image': 'swerebench/sweb.eval.x86_64.dask_1776_dask-11628',
    },
    ('ML/Scientific', 'Hard'): {
        'instance_id': 'numba__numba-5721',
        'docker_image': 'swerebench/sweb.eval.x86_64.numba_1776_numba-5721',
    },

    # Web/Network
    ('Web/Network', 'Easy'): {
        'instance_id': 'encode__httpx-2701',
        'docker_image': 'swerebench/sweb.eval.x86_64.encode_1776_httpx-2701',
    },
    ('Web/Network', 'Medium'): {
        'instance_id': 'streamlink__streamlink-3485',
        'docker_image': 'swerebench/sweb.eval.x86_64.streamlink_1776_streamlink-3485',
    },
    ('Web/Network', 'Hard'): {
        'instance_id': 'streamlink__streamlink-2160',
        'docker_image': 'swerebench/sweb.eval.x86_64.streamlink_1776_streamlink-2160',
    },

    # CLI/Tools
    ('CLI/Tools', 'Easy'): {
        'instance_id': 'asottile__pyupgrade-939',
        'docker_image': 'swerebench/sweb.eval.x86_64.asottile_1776_pyupgrade-939',
    },
    ('CLI/Tools', 'Medium'): {
        'instance_id': 'Textualize__textual-3548',
        'docker_image': 'swerebench/sweb.eval.x86_64.textualize_1776_textual-3548',
    },
    ('CLI/Tools', 'Hard'): {
        'instance_id': 'joke2k__faker-1520',
        'docker_image': 'swerebench/sweb.eval.x86_64.joke2k_1776_faker-1520',
    },

    # Medical/Bio
    ('Medical/Bio', 'Easy'): {
        'instance_id': 'pydicom__pydicom-1601',
        'docker_image': 'swerebench/sweb.eval.x86_64.pydicom_1776_pydicom-1601',
    },
    ('Medical/Bio', 'Medium'): {
        'instance_id': 'pydicom__pydicom-1853',
        'docker_image': 'swerebench/sweb.eval.x86_64.pydicom_1776_pydicom-1853',
    },
    ('Medical/Bio', 'Hard'): {
        'instance_id': 'pydicom__pydicom-2022',
        'docker_image': 'swerebench/sweb.eval.x86_64.pydicom_1776_pydicom-2022',
    },
}

# Category to repo mapping for finding replacements
CATEGORY_REPOS = {
    'SQL/Data': ['sqlfluff/sqlfluff', 'tobymao/sqlglot', 'reata/sqllineage', 'narwhals-dev/narwhals'],
    'DevOps/Build': ['pre-commit/pre-commit', 'conan-io/conan', 'iterative/dvc', 'tox-dev/tox', 'beeware/briefcase', 'pdm-project/pdm'],
    'ML/Scientific': ['dask/dask', 'PennyLaneAI/pennylane', 'pytorch/ignite', 'zarr-developers/zarr-python', 'networkx/networkx', 'numba/numba'],
    'Web/Network': ['encode/httpx', 'encode/starlette', 'streamlink/streamlink'],
    'CLI/Tools': ['asottile/pyupgrade', 'Textualize/textual', 'hgrecco/pint', 'python-cmd2/cmd2', 'joke2k/faker'],
    'Medical/Bio': ['pydicom/pydicom'],
}

DIFFICULTY_MAP = {
    'Easy': 1,
    'Medium': 2,
    'Hard': 3,
}


def check_image_exists(image_name: str) -> bool:
    """Check if Docker image exists by trying to pull manifest."""
    result = subprocess.run(
        ["podman", "manifest", "inspect", f"docker.io/{image_name}"],
        capture_output=True,
        timeout=30
    )
    return result.returncode == 0


def find_replacement(category: str, difficulty: str, dataset, exclude_ids: set) -> dict:
    """Find a replacement task from the dataset."""
    difficulty_score = DIFFICULTY_MAP[difficulty]
    repos = CATEGORY_REPOS.get(category, [])

    for row in dataset:
        # Check if matches category repos
        if row['repo'] not in repos:
            continue

        # Check difficulty
        meta = row.get('meta', {})
        llm_score = meta.get('llm_score', {})
        if llm_score.get('difficulty_score') != difficulty_score:
            continue

        # Check if has docker image
        if not row.get('docker_image'):
            continue

        # Skip already used
        if row['instance_id'] in exclude_ids:
            continue

        # Verify image exists
        print(f"    Checking replacement: {row['instance_id']}...")
        if check_image_exists(row['docker_image']):
            return {
                'instance_id': row['instance_id'],
                'repo': row['repo'],
                'docker_image': row['docker_image'],
            }

    return None


def main():
    print("Loading SWE-rebench dataset...")
    dataset = load_dataset("nebius/SWE-rebench", split="filtered")
    print(f"Loaded {len(dataset)} tasks\n")

    results = {}
    used_ids = set()

    for (category, difficulty), task in SAMPLE_TASKS.items():
        key = f"{category} - {difficulty}"
        print(f"Checking: {key}")
        print(f"  Instance: {task['instance_id']}")
        print(f"  Image: {task['docker_image']}")

        exists = check_image_exists(task['docker_image'])

        if exists:
            print(f"  Status: OK")
            results[key] = {
                'status': 'ok',
                'task': task,
            }
            used_ids.add(task['instance_id'])
        else:
            print(f"  Status: NOT FOUND - searching for replacement...")
            replacement = find_replacement(category, difficulty, dataset, used_ids)

            if replacement:
                print(f"  Replacement found: {replacement['instance_id']}")
                results[key] = {
                    'status': 'replaced',
                    'original': task,
                    'replacement': replacement,
                }
                used_ids.add(replacement['instance_id'])
            else:
                print(f"  ERROR: No replacement found!")
                results[key] = {
                    'status': 'failed',
                    'task': task,
                }
        print()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    ok_count = sum(1 for r in results.values() if r['status'] == 'ok')
    replaced_count = sum(1 for r in results.values() if r['status'] == 'replaced')
    failed_count = sum(1 for r in results.values() if r['status'] == 'failed')

    print(f"OK: {ok_count}")
    print(f"Replaced: {replaced_count}")
    print(f"Failed: {failed_count}")

    if replaced_count > 0:
        print("\n" + "="*60)
        print("REPLACEMENTS (update SAMPLE_TASKS with these):")
        print("="*60)
        for key, result in results.items():
            if result['status'] == 'replaced':
                r = result['replacement']
                cat, diff = key.split(' - ')
                print(f"""
    ('{cat}', '{diff}'): {{
        'instance_id': '{r['instance_id']}',
        'repo': '{r['repo']}',
        'docker_image': '{r['docker_image']}',
    }},""")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
