# AgentCgroup PoC 实验（基于 Claude Code Trace）

本文档描述使用 Claude Code 自身的 trace 进行快速 PoC 实验的方法。

## 1. 为什么用 Claude Code Trace？

相比 SWE-agent/OpenHands 的 trace，Claude Code 的 trace 有显著优势：

| 特性 | SWE-agent Trace | Claude Code Trace |
|------|-----------------|-------------------|
| 时间信息 | ❌ 无 | ✅ 有精确时间戳 |
| Tool 执行时间 | ❌ 无 | ✅ 可计算 |
| 环境依赖 | 需要 Docker 镜像 | 本地直接运行 |
| 设置复杂度 | 高 | 低 |
| 场景可控 | 固定 | 可自定义 |

**结论**：PoC 阶段使用 Claude Code trace 更快、更简单。

## 2. Claude Code Trace 格式

### 2.1 文件位置

```
~/.claude/projects/<project-path>/<session-id>.jsonl
```

例如：
```
~/.claude/projects/-home-yunwei37-agentcgroup/69a03e75-1672-42de-a4d1-f30917de82e0.jsonl
```

### 2.2 记录类型

| 类型 | 说明 |
|------|------|
| `assistant` | LLM 响应，包含 tool_use |
| `user` | 用户输入或 tool_result |
| `progress` | 进度更新 |
| `system` | 系统消息 |

### 2.3 Tool Call 格式

**tool_use（在 assistant 记录中）**：
```json
{
  "type": "assistant",
  "timestamp": "2026-01-31T01:00:13.001Z",
  "message": {
    "content": [
      {
        "type": "tool_use",
        "id": "toolu_013atJRUNgLgT3bJ7DPYfh8R",
        "name": "Bash",
        "input": {"command": "pytest tests/"}
      }
    ]
  }
}
```

**tool_result（在 user 记录中）**：
```json
{
  "type": "user",
  "timestamp": "2026-01-31T01:00:15.206Z",
  "message": {
    "content": [
      {
        "type": "tool_result",
        "tool_use_id": "toolu_013atJRUNgLgT3bJ7DPYfh8R",
        "content": "test results..."
      }
    ]
  }
}
```

### 2.4 可用的 Tool 类型

| Tool | 说明 | 资源特征 |
|------|------|----------|
| `Bash` | 执行 shell 命令 | CPU/内存/IO 密集 |
| `Read` | 读取文件 | IO 密集 |
| `Write` | 写入文件 | IO 密集 |
| `Edit` | 编辑文件 | IO 密集 |
| `Glob` | 文件搜索 | IO 密集 |
| `Grep` | 内容搜索 | CPU/IO 密集 |
| `WebSearch` | 网络搜索 | 网络 IO |
| `Task` | 子任务 | 递归 |

## 3. 当前 Trace 统计

从本项目的 Claude Code trace 分析：

| Tool | 调用次数 | 平均延迟 | 最大延迟 |
|------|----------|----------|----------|
| Bash | 152 | 37.1s | 794.5s |
| Edit | 220 | 0.76s | 7.6s |
| Read | 67 | 0.57s | 1.3s |
| Write | 14 | 43.8s | 608.8s |
| Grep | 12 | 0.84s | 1.3s |
| Glob | 3 | 0.19s | 0.2s |

**观察**：
- Bash 命令延迟变化大（有些包含用户等待时间）
- 文件操作（Read/Edit/Glob）延迟稳定
- Write 延迟高是因为包含用户确认时间

## 4. PoC 实验设计

### 场景 1：编译/构建任务

**触发方式**：让 Claude Code 编译一个项目
```
请编译这个 LaTeX 文档：pdflatex main.tex
```

**预期 Tool Call**：
- `Bash`: `pdflatex main.tex`
- 资源特征：CPU 密集，短时间突发

**测量指标**：
- 编译时间
- CPU 使用率峰值
- 内存使用

### 场景 2：测试套件执行

**触发方式**：
```
运行项目的测试套件：pytest tests/ -v
```

**预期 Tool Call**：
- `Bash`: `pytest tests/ -v`
- 资源特征：CPU 密集，可能内存密集

**测量指标**：
- 测试执行时间
- 进程数
- 内存峰值

### 场景 3：大规模代码搜索

**触发方式**：
```
在整个代码库中搜索所有使用了 "cgroup" 的地方
```

**预期 Tool Call**：
- `Grep`: 搜索 pattern
- `Read`: 读取匹配文件
- 资源特征：IO 密集

**测量指标**：
- 搜索时间
- IO 带宽
- 文件句柄数

### 场景 4：依赖安装

**触发方式**：
```
安装项目依赖：pip install -r requirements.txt
```

**预期 Tool Call**：
- `Bash`: `pip install -r requirements.txt`
- 资源特征：网络 IO + 磁盘 IO + CPU（编译）

**测量指标**：
- 安装时间
- 网络带宽
- 磁盘写入

## 5. 实验实施步骤

### 5.1 收集 Trace

```bash
# 1. 启动 Claude Code 执行任务
claude

# 2. 执行预定义的场景任务
# （手动输入或使用脚本自动化）

# 3. 结束会话，trace 自动保存到 ~/.claude/projects/
```

### 5.2 解析 Trace

```python
# scripts/parse_claude_trace.py
import json
from datetime import datetime
from pathlib import Path

def parse_claude_trace(trace_path):
    """解析 Claude Code trace，提取 tool call 和时间信息"""
    tool_calls = {}
    results = []

    with open(trace_path) as f:
        for line in f:
            data = json.loads(line)
            timestamp = data.get('timestamp', '')

            if data.get('type') == 'assistant':
                for item in data.get('message', {}).get('content', []):
                    if item.get('type') == 'tool_use':
                        tool_calls[item['id']] = {
                            'name': item['name'],
                            'input': item['input'],
                            'start': timestamp
                        }

            elif data.get('type') == 'user':
                for item in data.get('message', {}).get('content', []):
                    if item.get('type') == 'tool_result':
                        tool_id = item['tool_use_id']
                        if tool_id in tool_calls:
                            call = tool_calls[tool_id]
                            start = datetime.fromisoformat(call['start'].replace('Z', '+00:00'))
                            end = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            results.append({
                                'tool': call['name'],
                                'input': call['input'],
                                'latency_ms': (end - start).total_seconds() * 1000,
                                'start': call['start'],
                                'end': timestamp
                            })

    return results
```

### 5.3 回放 Trace（带资源监控）

```python
# scripts/replay_claude_trace.py
import subprocess
import time
import psutil

def replay_with_monitoring(tool_calls):
    """回放 tool call 并监控资源"""
    results = []

    for call in tool_calls:
        if call['tool'] != 'Bash':
            continue

        command = call['input'].get('command', '')

        # 启动资源监控
        process = subprocess.Popen(
            command, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # 监控资源使用
        max_rss = 0
        max_cpu = 0
        start = time.time()

        while process.poll() is None:
            try:
                p = psutil.Process(process.pid)
                mem = p.memory_info().rss
                cpu = p.cpu_percent()
                max_rss = max(max_rss, mem)
                max_cpu = max(max_cpu, cpu)
            except:
                pass
            time.sleep(0.1)

        latency = (time.time() - start) * 1000

        results.append({
            'command': command,
            'latency_ms': latency,
            'max_rss_mb': max_rss / 1024 / 1024,
            'max_cpu_percent': max_cpu,
            'exit_code': process.returncode
        })

    return results
```

## 6. 与 Cgroup 集成

### 6.1 在 Cgroup 中运行 Tool

```python
def run_in_cgroup(command, cgroup_path):
    """在指定 cgroup 中运行命令"""
    # 创建子 cgroup
    subprocess.run(['sudo', 'mkdir', '-p', cgroup_path])

    # 设置资源限制
    with open(f'{cgroup_path}/memory.max', 'w') as f:
        f.write('512M')
    with open(f'{cgroup_path}/cpu.max', 'w') as f:
        f.write('50000 100000')  # 50% CPU

    # 在 cgroup 中运行
    process = subprocess.Popen(
        f'echo $$ > {cgroup_path}/cgroup.procs && {command}',
        shell=True
    )

    return process
```

### 6.2 收集 Cgroup 指标

```python
def read_cgroup_metrics(cgroup_path):
    """读取 cgroup 资源使用指标"""
    metrics = {}

    # CPU 时间
    with open(f'{cgroup_path}/cpu.stat') as f:
        for line in f:
            if line.startswith('usage_usec'):
                metrics['cpu_usec'] = int(line.split()[1])

    # 内存
    with open(f'{cgroup_path}/memory.current') as f:
        metrics['memory_bytes'] = int(f.read().strip())

    # 内存事件
    with open(f'{cgroup_path}/memory.events') as f:
        for line in f:
            key, val = line.split()
            metrics[f'memory_{key}'] = int(val)

    return metrics
```

## 7. 预期输出

### Per-Tool Metrics
```json
{"tool": "Bash", "cmd": "pdflatex main.tex", "latency_ms": 5234, "cpu_usec": 4800000, "memory_mb": 128, "memory_high_events": 0}
{"tool": "Bash", "cmd": "pytest tests/", "latency_ms": 12456, "cpu_usec": 11000000, "memory_mb": 256, "memory_high_events": 2}
```

### Summary
```json
{
  "scenario": "compile",
  "total_tools": 15,
  "bash_tools": 8,
  "total_latency_ms": 45000,
  "max_memory_mb": 512,
  "memory_high_events": 3,
  "oom_kills": 0
}
```

## 8. 实验对比

| 配置 | 描述 |
|------|------|
| Baseline | 无 cgroup 限制 |
| Static-256M | 固定 256MB 内存限制 |
| Static-512M | 固定 512MB 内存限制 |
| Dynamic | 根据 tool 类型动态调整 |

## 9. 下一步

1. **实现 trace 解析脚本** (`scripts/parse_claude_trace.py`)
2. **实现 cgroup 回放脚本** (`scripts/replay_with_cgroup.py`)
3. **设计 3-5 个可重复场景**
4. **收集 baseline 数据**
5. **对比不同 cgroup 策略**

## 10. 优势总结

| 方面 | SWE-agent 方案 | Claude Code 方案 |
|------|---------------|-----------------|
| 设置时间 | 小时级（下载镜像） | 分钟级 |
| 环境依赖 | Docker + 特定镜像 | 本地环境 |
| 时间数据 | 无 | 有 |
| 场景控制 | 固定 trace | 可自定义 |
| 调试难度 | 高 | 低 |

**建议**：PoC 阶段使用 Claude Code trace，验证方法后再扩展到 SWE-agent trace。
