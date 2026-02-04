# SWE-rebench 数据集介绍

## 概述

**SWE-rebench** 是由 Nebius 发布的大规模软件工程基准数据集，用于训练和评估基于 LLM 的代码修复 Agent。该数据集基于 [SWE-bench](https://www.swebench.com/) 扩展，通过自动化流水线从 GitHub 仓库持续提取真实的软件工程任务。

- **论文**: [SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation](https://arxiv.org/abs/2505.20411)
- **数据集**: https://huggingface.co/datasets/nebius/SWE-rebench
- **许可证**: CC-BY-4.0

## 数据集规模

| Split | 任务数 | 说明 |
|-------|--------|------|
| `test` | 21,336 | 完整数据集 |
| `filtered` | 6,542 | 精选子集，包含 Docker 镜像，质量更高 |

`filtered` split 包含 **1,790 个独立仓库** 的任务，每个任务都有：
- 预构建的 Docker 镜像用于复现环境
- 自动化测试验证修复正确性
- LLM 评分的难度等级

## 数据结构

每条记录包含以下字段：

```python
{
    "instance_id": "repo_owner__repo_name-issue_number",  # 唯一标识
    "repo": "owner/repo_name",                            # GitHub 仓库
    "problem_statement": "Issue description...",          # 问题描述
    "hints_text": "Additional hints...",                  # 提示信息
    "patch": "diff content...",                           # 正确的修复补丁
    "test_patch": "test diff...",                         # 测试补丁
    "base_commit": "abc123...",                           # 基准 commit
    "docker_image": "swerebench/sweb.eval.x86_64...",    # Docker 镜像名
    "meta": {
        "llm_score": {
            "difficulty_score": 1-3,      # 难度评分
            "issue_text_score": 1-5,      # Issue 描述质量
            "test_score": 1-5             # 测试覆盖质量
        },
        "num_modified_files": 1,          # 需要修改的文件数
        "is_lite": true/false             # 是否属于 lite 子集
    },
    "FAIL_TO_PASS": ["test_name1", ...],  # 修复后应该通过的测试
    "PASS_TO_PASS": ["test_name2", ...],  # 修复后应该保持通过的测试
}
```

## 难度分布

基于 LLM 评分的难度分布（`filtered` split，4,327 个有评分的任务）：

| 难度等级 | 数量 | 比例 | 说明 |
|----------|------|------|------|
| **Easy (1)** | 3,010 | 69.6% | 简单修复，通常单文件、少量代码改动 |
| **Medium (2)** | 1,209 | 27.9% | 中等难度，可能涉及多处修改 |
| **Hard (3)** | 108 | 2.5% | 复杂问题，需要深入理解代码库 |

```
Easy   ████████████████████████████████████ 69.6%
Medium ██████████████ 27.9%
Hard   █ 2.5%
```

## 领域分类

按仓库领域分类统计：

### SQL/数据处理 (413 tasks)

| 仓库 | 任务数 | 描述 |
|------|--------|------|
| tobymao/sqlglot | 294 | SQL 解析器和转译器 |
| sqlfluff/sqlfluff | 51 | SQL linter 和格式化工具 |
| reata/sqllineage | 40 | SQL 血缘分析 |
| narwhals-dev/narwhals | 25 | DataFrame 库抽象层 |

### DevOps/构建工具 (370 tasks)

| 仓库 | 任务数 | 描述 |
|------|--------|------|
| conan-io/conan | 73 | C/C++ 包管理器 |
| iterative/dvc | 72 | 数据版本控制 |
| tox-dev/tox | 58 | Python 测试自动化 |
| pre-commit/pre-commit | 42 | Git hooks 管理 |
| beeware/briefcase | 40 | Python 应用打包工具 |
| pdm-project/pdm | 28 | Python 包管理器 |

### ML/AI/科学计算 (369 tasks)

| 仓库 | 任务数 | 描述 |
|------|--------|------|
| PennyLaneAI/pennylane | 76 | 量子机器学习框架 |
| dask/dask | 72 | 并行计算库 |
| pytorch/ignite | 63 | PyTorch 训练框架 |
| zarr-developers/zarr-python | 42 | 分块数组存储 |
| networkx/networkx | 35 | 图算法库 |
| numba/numba | 35 | JIT 编译器 |

### Web/网络 (133 tasks)

| 仓库 | 任务数 | 描述 |
|------|--------|------|
| streamlink/streamlink | 73 | 流媒体提取工具 |
| encode/starlette | 27 | ASGI Web 框架 |
| encode/httpx | 25 | 异步 HTTP 客户端 |

### CLI/工具 (180 tasks)

| 仓库 | 任务数 | 描述 |
|------|--------|------|
| Textualize/textual | 49 | TUI 框架 |
| asottile/pyupgrade | 33 | Python 代码升级工具 |
| hgrecco/pint | 30 | 物理单位处理 |
| python-cmd2/cmd2 | 27 | 命令行应用框架 |
| joke2k/faker | 26 | 假数据生成器 |

### 医疗/生物 (42 tasks)

| 仓库 | 任务数 | 描述 |
|------|--------|------|
| pydicom/pydicom | 42 | DICOM 医学影像处理 |

## Docker 镜像使用

每个任务都有预构建的 Docker 镜像，命名格式：
```
swerebench/sweb.eval.x86_64.<org>_<id>_<repo>-<issue>
```

### 拉取镜像
```bash
podman pull docker.io/swerebench/sweb.eval.x86_64.encode_1776_starlette-1147
```

### 镜像内容
- `/testbed/`: 代码仓库（已 checkout 到 base_commit）
- `/issue.md`: Issue 描述文件
- 预安装的 Python 环境和依赖

### 运行示例
```bash
podman run --rm -it \
    docker.io/swerebench/sweb.eval.x86_64.encode_1776_starlette-1147 \
    bash -c "cat /issue.md && cd /testbed && git log -1"
```

## 与其他 SWE-bench 变体的比较

| 数据集 | 任务数 | 仓库数 | 特点 |
|--------|--------|--------|------|
| SWE-bench (原版) | 2,294 | 12 | 最早的基准，仅限特定仓库 |
| SWE-bench Lite | 300 | 12 | 精选子集，评估成本低 |
| SWE-bench Verified | 500 | 12 | 人工验证，高质量 |
| SWE-bench Pro | 1,865 | 41 | 专业仓库，更具挑战性 |
| **SWE-rebench** | **21,336** | **3,400+** | 最大规模，持续更新 |
| SWE-rebench filtered | 6,542 | 1,790 | 有 Docker 镜像的高质量子集 |

## 使用方法

### 加载数据集

```python
from datasets import load_dataset

# 加载完整数据集
ds = load_dataset("nebius/SWE-rebench", split="test")

# 加载 filtered 子集（推荐，有 Docker 镜像）
ds_filtered = load_dataset("nebius/SWE-rebench", split="filtered")

# 按难度筛选
easy_tasks = [
    row for row in ds_filtered
    if row['meta']['llm_score']['difficulty_score'] == 1
]
```

### 按仓库筛选

```python
# 获取特定仓库的任务
starlette_tasks = [
    row for row in ds_filtered
    if row['repo'] == 'encode/starlette'
]
print(f"Starlette tasks: {len(starlette_tasks)}")
```

### 获取 Docker 镜像名

```python
for row in ds_filtered:
    if row['docker_image']:
        print(f"docker.io/{row['docker_image']}")
```

## 评估指标

SWE-rebench 使用以下指标评估 Agent 性能：

1. **Resolve Rate**: 成功修复的任务比例
2. **FAIL_TO_PASS**: 修复后测试从失败变为通过
3. **PASS_TO_PASS**: 修复后原有测试保持通过（无回归）

## 推荐的实验任务

### 入门级（Easy，单文件修改）

| 仓库 | 任务数 | 推荐原因 |
|------|--------|---------|
| encode/starlette | 27 | Web 框架，代码清晰 |
| encode/httpx | 25 | HTTP 客户端，测试完善 |
| hgrecco/pint | 30 | 单位处理，逻辑简单 |

### 中等难度（Medium，多文件修改）

| 仓库 | 任务数 | 推荐原因 |
|------|--------|---------|
| dask/dask | 72 | 并行计算，文档完善 |
| sqlfluff/sqlfluff | 51 | SQL 处理，测试丰富 |
| tox-dev/tox | 58 | 构建工具，模块清晰 |

### 高难度（Hard，复杂架构）

| 仓库 | 任务数 | 推荐原因 |
|------|--------|---------|
| conan-io/conan | 73 | 包管理器，系统复杂 |
| PennyLaneAI/pennylane | 76 | 量子 ML，专业领域 |
| numba/numba | 35 | JIT 编译，底层优化 |

## 任务矩阵 (Category × Difficulty)

### 数量分布

| Category | Easy (1) | Medium (2) | Hard (3) |
|----------|----------|------------|----------|
| SQL/Data | 192 | 83 | 7 |
| DevOps/Build | 183 | 53 | 6 |
| ML/Scientific | 168 | 78 | 8 |
| Web/Network | 63 | 27 | 3 |
| CLI/Tools | 85 | 25 | 3 |
| Other | 2,319 | 943 | 81 |

### 样本任务（每个类别×难度各选一个）

```python
SAMPLE_TASKS = {
    # SQL/Data
    ('SQL/Data', 'Easy'): {
        'instance_id': 'sqlfluff__sqlfluff-5362',
        'repo': 'sqlfluff/sqlfluff',
        'docker_image': 'swerebench/sweb.eval.x86_64.sqlfluff_1776_sqlfluff-5362',
    },
    ('SQL/Data', 'Medium'): {
        'instance_id': 'tobymao__sqlglot-1177',
        'repo': 'tobymao/sqlglot',
        'docker_image': 'swerebench/sweb.eval.x86_64.tobymao_1776_sqlglot-1177',
    },
    ('SQL/Data', 'Hard'): {
        'instance_id': 'reata__sqllineage-438',
        'repo': 'reata/sqllineage',
        'docker_image': 'swerebench/sweb.eval.x86_64.reata_1776_sqllineage-438',
    },

    # DevOps/Build
    ('DevOps/Build', 'Easy'): {
        'instance_id': 'pre-commit__pre-commit-2524',
        'repo': 'pre-commit/pre-commit',
        'docker_image': 'swerebench/sweb.eval.x86_64.pre-commit_1776_pre-commit-2524',
    },
    ('DevOps/Build', 'Medium'): {
        'instance_id': 'beeware__briefcase-1525',
        'repo': 'beeware/briefcase',
        'docker_image': 'swerebench/sweb.eval.x86_64.beeware_1776_briefcase-1525',
    },
    ('DevOps/Build', 'Hard'): {
        'instance_id': 'iterative__dvc-777',
        'repo': 'iterative/dvc',
        'docker_image': 'swerebench/sweb.eval.x86_64.iterative_1776_dvc-777',
    },

    # ML/Scientific
    ('ML/Scientific', 'Easy'): {
        'instance_id': 'dask__dask-5510',
        'repo': 'dask/dask',
        'docker_image': 'swerebench/sweb.eval.x86_64.dask_1776_dask-5510',
    },
    ('ML/Scientific', 'Medium'): {
        'instance_id': 'dask__dask-11628',
        'repo': 'dask/dask',
        'docker_image': 'swerebench/sweb.eval.x86_64.dask_1776_dask-11628',
    },
    ('ML/Scientific', 'Hard'): {
        'instance_id': 'numba__numba-5721',
        'repo': 'numba/numba',
        'docker_image': 'swerebench/sweb.eval.x86_64.numba_1776_numba-5721',
    },

    # Web/Network
    ('Web/Network', 'Easy'): {
        'instance_id': 'encode__httpx-2701',
        'repo': 'encode/httpx',
        'docker_image': 'swerebench/sweb.eval.x86_64.encode_1776_httpx-2701',
    },
    ('Web/Network', 'Medium'): {
        'instance_id': 'streamlink__streamlink-3485',
        'repo': 'streamlink/streamlink',
        'docker_image': 'swerebench/sweb.eval.x86_64.streamlink_1776_streamlink-3485',
    },
    ('Web/Network', 'Hard'): {
        'instance_id': 'streamlink__streamlink-2160',
        'repo': 'streamlink/streamlink',
        'docker_image': 'swerebench/sweb.eval.x86_64.streamlink_1776_streamlink-2160',
    },

    # CLI/Tools
    ('CLI/Tools', 'Easy'): {
        'instance_id': 'asottile__pyupgrade-939',
        'repo': 'asottile/pyupgrade',
        'docker_image': 'swerebench/sweb.eval.x86_64.asottile_1776_pyupgrade-939',
    },
    ('CLI/Tools', 'Medium'): {
        'instance_id': 'Textualize__textual-3548',
        'repo': 'Textualize/textual',
        'docker_image': 'swerebench/sweb.eval.x86_64.textualize_1776_textual-3548',
    },
    ('CLI/Tools', 'Hard'): {
        'instance_id': 'joke2k__faker-1520',
        'repo': 'joke2k/faker',
        'docker_image': 'swerebench/sweb.eval.x86_64.joke2k_1776_faker-1520',
    },
}
```

### 运行样本任务

```bash
# 选择一个任务运行
python scripts/run_swebench.py swerebench/sweb.eval.x86_64.sqlfluff_1776_sqlfluff-5362

# 或者批量运行
for img in "${SAMPLE_IMAGES[@]}"; do
    python scripts/run_swebench.py "$img" --memory 4g --cpus 2
done
```

## 参考资料

- [SWE-rebench 论文](https://arxiv.org/abs/2505.20411)
- [SWE-bench 官网](https://www.swebench.com/)
- [HuggingFace 数据集页面](https://huggingface.co/datasets/nebius/SWE-rebench)
- [SWE-bench GitHub](https://github.com/SWE-bench/SWE-bench)
