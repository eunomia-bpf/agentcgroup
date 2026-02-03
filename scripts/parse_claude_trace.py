#!/usr/bin/env python3
"""
解析 Claude Code trace 文件，提取 tool call 和时间信息。

Usage:
    python scripts/parse_claude_trace.py <trace.jsonl>
    python scripts/parse_claude_trace.py ~/.claude/projects/-home-yunwei37-agentcgroup/*.jsonl
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_timestamp(ts: str) -> datetime:
    """解析 ISO 格式时间戳"""
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))


def parse_claude_trace(trace_path: Path) -> list:
    """解析 Claude Code trace，提取 tool call 和时间信息"""
    tool_calls = {}  # id -> {name, input, start}
    results = []

    with open(trace_path) as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            timestamp = data.get('timestamp', '')
            if not timestamp:
                continue

            if data.get('type') == 'assistant':
                msg = data.get('message', {})
                for item in msg.get('content', []):
                    if isinstance(item, dict) and item.get('type') == 'tool_use':
                        tool_calls[item['id']] = {
                            'name': item['name'],
                            'input': item.get('input', {}),
                            'start': timestamp
                        }

            elif data.get('type') == 'user':
                msg = data.get('message', {})
                for item in msg.get('content', []):
                    if isinstance(item, dict) and item.get('type') == 'tool_result':
                        tool_id = item.get('tool_use_id')
                        if tool_id in tool_calls:
                            call = tool_calls[tool_id]
                            try:
                                start = parse_timestamp(call['start'])
                                end = parse_timestamp(timestamp)
                                latency_ms = (end - start).total_seconds() * 1000
                            except:
                                latency_ms = 0

                            results.append({
                                'tool': call['name'],
                                'input': call['input'],
                                'latency_ms': latency_ms,
                                'start': call['start'],
                                'end': timestamp
                            })

    return results


def filter_bash_calls(results: list) -> list:
    """过滤出 Bash 命令，去除包含用户等待的异常值"""
    bash_calls = []
    for r in results:
        if r['tool'] != 'Bash':
            continue
        # 过滤掉超过 60 秒的调用（可能包含用户等待）
        if r['latency_ms'] > 60000:
            continue
        bash_calls.append(r)
    return bash_calls


def print_summary(results: list):
    """打印统计摘要"""
    print(f"总 tool calls: {len(results)}")
    print()

    # 按 tool 分组统计
    by_tool = defaultdict(list)
    for r in results:
        by_tool[r['tool']].append(r['latency_ms'])

    print("按 Tool 类型统计:")
    print(f"{'Tool':<15} {'Count':>6} {'Avg(ms)':>10} {'Max(ms)':>10} {'Min(ms)':>10}")
    print("-" * 55)
    for tool in sorted(by_tool.keys()):
        latencies = by_tool[tool]
        avg = sum(latencies) / len(latencies)
        print(f"{tool:<15} {len(latencies):>6} {avg:>10.1f} {max(latencies):>10.1f} {min(latencies):>10.1f}")


def print_bash_commands(results: list, limit: int = 20):
    """打印 Bash 命令详情"""
    bash_calls = [r for r in results if r['tool'] == 'Bash']

    print(f"\nBash 命令 (前 {limit} 条):")
    print(f"{'Latency(ms)':>12}  Command")
    print("-" * 70)

    for r in bash_calls[:limit]:
        cmd = r['input'].get('command', '')[:50]
        print(f"{r['latency_ms']:>12.1f}  {cmd}")


def export_trace_ir(results: list, output_path: Path):
    """导出为 Trace IR 格式"""
    steps = []
    for i, r in enumerate(results):
        step = {
            'step_id': i,
            'tool': r['tool'].lower(),
            'latency_ms': r['latency_ms'],
            'start': r['start'],
            'end': r['end']
        }

        # 添加工具特定字段
        if r['tool'] == 'Bash':
            step['command'] = r['input'].get('command', '')
        elif r['tool'] in ('Read', 'Write', 'Edit'):
            step['file_path'] = r['input'].get('file_path', '')
        elif r['tool'] == 'Grep':
            step['pattern'] = r['input'].get('pattern', '')
        elif r['tool'] == 'Glob':
            step['pattern'] = r['input'].get('pattern', '')

        steps.append(step)

    trace_ir = {
        'trace_id': output_path.stem,
        'source': 'claude_code',
        'steps': steps
    }

    with open(output_path, 'w') as f:
        json.dump(trace_ir, f, indent=2)

    print(f"\n导出到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="解析 Claude Code trace")
    parser.add_argument("trace_path", type=Path, help="trace.jsonl 文件路径")
    parser.add_argument("--output", "-o", type=Path, help="导出 Trace IR 文件")
    parser.add_argument("--bash-only", action="store_true", help="只显示 Bash 命令")
    parser.add_argument("--limit", type=int, default=20, help="显示条目数限制")
    args = parser.parse_args()

    print(f"解析: {args.trace_path}")
    print("=" * 70)

    results = parse_claude_trace(args.trace_path)

    if args.bash_only:
        results = filter_bash_calls(results)
        print(f"Bash 命令 (过滤后): {len(results)}")
        print_bash_commands(results, args.limit)
    else:
        print_summary(results)
        print_bash_commands(results, args.limit)

    if args.output:
        export_trace_ir(results, args.output)


if __name__ == "__main__":
    main()
