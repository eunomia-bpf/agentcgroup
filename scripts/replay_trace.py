#!/usr/bin/env python3
"""
Trace Replay Tool for SWE-bench Experiments

Replays Bash commands from a Claude Code trace file in a container,
strictly matching the original timing, and records CPU/memory usage.

This helps understand resource consumption patterns without running Claude Code.

Usage:
    # Replay with original timing (default)
    python scripts/replay_trace.py experiments/batch_swebench_18tasks/Web_Network_Easy/attempt_1

    # Speed up replay (2x faster)
    python scripts/replay_trace.py <attempt_dir> --speed 2.0

    # No delay (run commands as fast as possible)
    python scripts/replay_trace.py <attempt_dir> --no-delay
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Import from run_swebench.py
from run_swebench import ResourceMonitor
from plot_resources import plot_from_attempt_dir


class TraceParser:
    """Parse Claude Code trace files to extract Bash commands."""

    def __init__(self, trace_file: Path):
        self.trace_file = trace_file
        self.commands: List[Dict] = []
        self.all_tool_calls: List[Dict] = []  # All tool calls for plotting
        self.start_timestamp: Optional[str] = None

    def parse(self) -> List[Dict]:
        """Parse trace file and extract Bash commands with timing."""
        tool_uses = {}  # tool_use_id -> command info
        tool_results = {}  # tool_use_id -> result info

        with open(self.trace_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    self._process_entry(entry, tool_uses, tool_results)
                except json.JSONDecodeError:
                    continue

        # Match tool_use with tool_result to get execution time
        for tool_id, cmd_info in tool_uses.items():
            if tool_id in tool_results:
                cmd_info['end_timestamp'] = tool_results[tool_id].get('timestamp')
                cmd_info['result'] = tool_results[tool_id].get('result', {})
            self.commands.append(cmd_info)

        # Sort by timestamp
        self.commands.sort(key=lambda x: x.get('timestamp', ''))

        if self.commands:
            self.start_timestamp = self.commands[0].get('timestamp')

        return self.commands

    def _process_entry(self, entry: dict, tool_uses: dict, tool_results: dict):
        """Process a single trace entry."""
        entry_type = entry.get('type')

        # Tool use (command invocation)
        if entry_type == 'assistant' and 'message' in entry:
            msg = entry['message']
            if 'content' in msg:
                for block in msg['content']:
                    if block.get('type') == 'tool_use':
                        tool_id = block.get('id')
                        tool_name = block.get('name')
                        cmd_input = block.get('input', {})

                        # Record all tool calls for plotting
                        self.all_tool_calls.append({
                            'timestamp': entry.get('timestamp'),
                            'tool': tool_name,
                            'id': tool_id
                        })

                        # Only extract Bash commands for replay
                        if tool_name == 'Bash':
                            tool_uses[tool_id] = {
                                'tool_use_id': tool_id,
                                'timestamp': entry.get('timestamp'),
                                'command': cmd_input.get('command', ''),
                                'description': cmd_input.get('description', ''),
                                'timeout': cmd_input.get('timeout'),
                            }

        # Tool result (command output)
        elif entry_type == 'user' and 'message' in entry:
            msg = entry['message']
            if 'content' in msg:
                content = msg['content']
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'tool_result':
                            tool_id = block.get('tool_use_id')
                            tool_results[tool_id] = {
                                'timestamp': entry.get('timestamp'),
                                'result': entry.get('toolUseResult', {}),
                            }


class TraceReplayer:
    """Replay Bash commands in a container with timing."""

    def __init__(self, image_name: str, commands: List[Dict], all_tool_calls: List[Dict],
                 output_dir: Path, speed: float = 1.0, no_delay: bool = False,
                 task_name: str = ""):
        self.image_name = image_name
        self.commands = commands
        self.all_tool_calls = all_tool_calls
        self.output_dir = output_dir
        self.speed = speed
        self.no_delay = no_delay
        self.task_name = task_name
        self.home = Path.home()
        self.container_id: Optional[str] = None
        self.fixed_image_name: Optional[str] = None
        self.replay_tool_calls: List[Dict] = []  # Tool calls with replay timestamps

    def run(self) -> dict:
        """Run the replay."""
        start_time = time.time()
        results = {
            "image": self.image_name,
            "start_time": datetime.now().isoformat(),
            "command_count": len(self.commands),
            "speed": self.speed,
            "no_delay": self.no_delay,
            "task_name": self.task_name,
        }

        resource_data = None
        try:
            # Step 1: Setup container
            print(f"[1/5] Setting up container for image: {self.image_name}")
            self._setup_container()

            # Step 2: Start resource monitoring
            print(f"[2/5] Starting resource monitoring...")
            monitor = ResourceMonitor(self.container_id, interval=1.0)
            monitor.start()

            # Step 3: Replay commands with timing
            print(f"[3/5] Replaying {len(self.commands)} Bash commands (speed: {self.speed}x)...")
            replay_start = time.time()
            replay_results = self._replay_commands(replay_start)
            results["replay_results"] = replay_results

            # Step 4: Collect results
            print(f"[4/5] Collecting results...")
            monitor.stop()

            resource_data = {
                "samples": monitor.samples,
                "summary": monitor.get_summary()
            }
            results["resource_samples"] = resource_data

            # Print summary
            summary = resource_data["summary"]
            print(f"  Collected {len(monitor.samples)} resource samples")
            print(f"  Memory: avg={summary['memory_mb']['avg']:.1f}MB, max={summary['memory_mb']['max']:.1f}MB")
            print(f"  CPU: avg={summary['cpu_percent']['avg']:.1f}%, max={summary['cpu_percent']['max']:.1f}%")

            # Step 5: Save and generate plot
            print(f"[5/5] Saving results and generating plot...")
            self._save_results(results, resource_data)
            self._generate_plot()

        except Exception as e:
            results["error"] = str(e)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

        results["total_time"] = time.time() - start_time
        results["end_time"] = datetime.now().isoformat()

        return results

    def _setup_container(self):
        """Setup the container with fixed permissions."""
        import os

        # Create fixed image name
        safe_name = self.image_name.replace("/", "_").replace(":", "_")
        self.fixed_image_name = f"swebench-fixed-{safe_name}"

        # Check if fixed image exists, if not create it
        result = subprocess.run(
            ["podman", "image", "exists", self.fixed_image_name],
            capture_output=True
        )

        if result.returncode != 0:
            print(f"  Creating fixed image...")
            self._fix_permissions()
        else:
            print(f"  Using existing fixed image: {self.fixed_image_name}")

        # Start container (keep running with sleep)
        container_cmd = [
            "podman", "run", "-d",
            "--userns=keep-id",
            "--network=host",
            "-v", "/usr:/usr:ro",
            "-v", "/lib:/lib:ro",
            "-v", "/lib64:/lib64:ro",
            "-v", "/etc:/etc:ro",
            "-v", "/bin:/bin:ro",
            "-v", "/sbin:/sbin:ro",
            "-v", "/home:/home",
            "-v", "/tmp:/tmp",
            "-v", "/var:/var",
            "-w", "/testbed",
            "-e", f"HOME={self.home}",
            "-e", "PATH=/usr/local/bin:/usr/bin:/bin",
            self.fixed_image_name,
            "sleep", "infinity"
        ]

        result = subprocess.run(container_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")

        self.container_id = result.stdout.strip()
        print(f"  Container started: {self.container_id[:12]}")

        # Initialize git config
        subprocess.run(
            ["podman", "exec", self.container_id, "bash", "-c",
             "git config user.email 'test@test.com' && git config user.name 'Test' && git config --add safe.directory /testbed"],
            capture_output=True
        )

    def _fix_permissions(self):
        """Create a modified image with fixed /testbed permissions."""
        import os
        uid = os.getuid()
        gid = os.getgid()

        # Pull original image if needed
        subprocess.run(
            ["podman", "pull", f"docker.io/{self.image_name}"],
            capture_output=True
        )

        # Create temp container
        result = subprocess.run(
            ["podman", "run", "-d", f"docker.io/{self.image_name}", "sleep", "120"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create temp container: {result.stderr}")

        temp_container = result.stdout.strip()

        try:
            # Fix permissions
            subprocess.run(
                ["podman", "exec", temp_container, "chown", "-R", f"{uid}:{gid}", "/testbed"],
                check=True, capture_output=True
            )

            # Commit as new image
            subprocess.run(
                ["podman", "commit", temp_container, self.fixed_image_name],
                check=True, capture_output=True
            )
            print(f"  Created fixed image: {self.fixed_image_name}")
        finally:
            subprocess.run(["podman", "stop", temp_container], capture_output=True)
            subprocess.run(["podman", "rm", temp_container], capture_output=True)

    def _replay_commands(self, replay_start: float) -> List[Dict]:
        """Replay commands with timing matching original trace."""
        results = []

        if not self.commands:
            return results

        # Calculate original start time
        first_ts = self.commands[0].get('timestamp', '')
        try:
            original_start = datetime.fromisoformat(first_ts.replace('Z', '+00:00')).timestamp()
        except:
            original_start = 0

        # Create mapping of original tool call times to replay times
        tool_call_index = 0

        for i, cmd_info in enumerate(self.commands):
            cmd_ts = cmd_info.get('timestamp', '')

            # Calculate when this command should run relative to start
            try:
                cmd_original_time = datetime.fromisoformat(cmd_ts.replace('Z', '+00:00')).timestamp()
                relative_time = cmd_original_time - original_start
            except:
                relative_time = 0

            # Wait until it's time to run this command
            if not self.no_delay:
                target_time = replay_start + (relative_time / self.speed)
                current_time = time.time()
                wait_time = target_time - current_time

                if wait_time > 0:
                    if wait_time > 1:
                        print(f"  Waiting {wait_time:.1f}s (t={relative_time:.1f}s in original)...")
                    time.sleep(wait_time)

            # Record tool call with replay timestamp
            self.replay_tool_calls.append({
                'timestamp': datetime.now().isoformat(),
                'tool': 'Bash',
                'id': cmd_info.get('tool_use_id', f'replay_{i}')
            })

            # Execute command
            command = cmd_info['command']
            desc = cmd_info.get('description', '')[:50]
            print(f"  [{i+1}/{len(self.commands)}] {desc or command[:50]}...")

            exec_start = time.time()
            try:
                result = subprocess.run(
                    ["podman", "exec", self.container_id, "bash", "-c", command],
                    capture_output=True, text=True, timeout=300
                )
                exec_result = {
                    "index": i,
                    "command": command[:200],
                    "description": cmd_info.get('description', ''),
                    "original_timestamp": cmd_ts,
                    "replay_timestamp": datetime.now().isoformat(),
                    "exit_code": result.returncode,
                    "stdout_len": len(result.stdout),
                    "stderr_len": len(result.stderr),
                    "execution_time": time.time() - exec_start,
                    "success": result.returncode == 0
                }
            except subprocess.TimeoutExpired:
                exec_result = {
                    "index": i,
                    "command": command[:200],
                    "error": "timeout",
                    "execution_time": time.time() - exec_start,
                    "success": False
                }
            except Exception as e:
                exec_result = {
                    "index": i,
                    "command": command[:200],
                    "error": str(e),
                    "execution_time": time.time() - exec_start,
                    "success": False
                }

            results.append(exec_result)

        return results

    def _cleanup(self):
        """Clean up container."""
        if self.container_id:
            subprocess.run(["podman", "stop", self.container_id], capture_output=True)
            subprocess.run(["podman", "rm", self.container_id], capture_output=True)
            print(f"  Removed container: {self.container_id[:12]}")

    def _save_results(self, results: dict, resource_data: Optional[dict]):
        """Save results to output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save resource data
        if resource_data:
            with open(self.output_dir / "resources.json", "w") as f:
                json.dump(resource_data, f, indent=2)

        # Save tool calls for plotting
        with open(self.output_dir / "tool_calls.json", "w") as f:
            json.dump(self.replay_tool_calls, f, indent=2)

        print(f"  Results saved to: {self.output_dir}")

    def _generate_plot(self):
        """Generate resource usage plot."""
        try:
            title = f"Replay - {self.task_name}" if self.task_name else "Trace Replay"
            if self.speed != 1.0:
                title += f" ({self.speed}x speed)"
            plot_from_attempt_dir(self.output_dir, title=title)
            print(f"  Plot saved to: {self.output_dir / 'resource_plot.png'}")
        except Exception as e:
            print(f"  Warning: Failed to generate plot: {e}")


def get_image_from_attempt(attempt_dir: Path) -> Optional[str]:
    """Extract Docker image name from attempt results."""
    results_file = attempt_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
            return results.get("image")
    return None


def get_task_name_from_path(attempt_dir: Path) -> str:
    """Extract task name from attempt directory path."""
    # e.g., experiments/batch_swebench_18tasks/Web_Network_Easy/attempt_1
    parts = attempt_dir.parts
    for i, part in enumerate(parts):
        if part.startswith("batch_swebench"):
            if i + 1 < len(parts):
                return parts[i + 1]  # e.g., "Web_Network_Easy"
    return attempt_dir.parent.name


def main():
    parser = argparse.ArgumentParser(
        description="Replay Bash commands from a Claude Code trace with original timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay with original timing (default)
  python scripts/replay_trace.py experiments/batch_swebench_18tasks/Web_Network_Easy/attempt_1

  # Replay at 2x speed
  python scripts/replay_trace.py experiments/batch_swebench_18tasks/Web_Network_Easy/attempt_1 --speed 2.0

  # Replay with no delay (fast as possible)
  python scripts/replay_trace.py experiments/batch_swebench_18tasks/Web_Network_Easy/attempt_1 --no-delay

Output is saved to: experiments/replays/<task_name>/
"""
    )
    parser.add_argument("attempt_dir", help="Path to attempt directory containing trace.jsonl")
    parser.add_argument("--output-dir", help="Custom output directory")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speed multiplier for delays (default: 1.0, original timing)")
    parser.add_argument("--no-delay", action="store_true",
                        help="Run commands without delays (as fast as possible)")
    parser.add_argument("--image", help="Override Docker image name")

    args = parser.parse_args()

    attempt_dir = Path(args.attempt_dir)
    if not attempt_dir.exists():
        print(f"Error: Attempt directory not found: {attempt_dir}")
        return 1

    trace_file = attempt_dir / "trace.jsonl"
    if not trace_file.exists():
        print(f"Error: Trace file not found: {trace_file}")
        return 1

    # Get image name
    image_name = args.image or get_image_from_attempt(attempt_dir)
    if not image_name:
        print("Error: Could not determine Docker image. Use --image to specify.")
        return 1

    # Get task name
    task_name = get_task_name_from_path(attempt_dir)

    # Setup output directory (separate from original attempt)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: experiments/replays/<task_name>/
        base_dir = Path.home() / "agentcgroup" / "experiments" / "replays"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_dir / f"{task_name}_{timestamp}"

    print("=" * 70)
    print("Trace Replay Tool")
    print("=" * 70)
    print(f"Source: {attempt_dir}")
    print(f"Task: {task_name}")
    print(f"Image: {image_name}")
    print(f"Output: {output_dir}")
    print(f"Speed: {args.speed}x {'(no delay)' if args.no_delay else '(original timing)'}")
    print("=" * 70)

    # Parse trace
    print("\nParsing trace file...")
    trace_parser = TraceParser(trace_file)
    commands = trace_parser.parse()
    print(f"Found {len(commands)} Bash commands")
    print(f"Found {len(trace_parser.all_tool_calls)} total tool calls")

    if not commands:
        print("No Bash commands found in trace")
        return 1

    # Calculate expected duration
    if len(commands) > 1:
        first_ts = commands[0].get('timestamp', '')
        last_ts = commands[-1].get('timestamp', '')
        try:
            first_dt = datetime.fromisoformat(first_ts.replace('Z', '+00:00'))
            last_dt = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
            original_duration = (last_dt - first_dt).total_seconds()
            expected_duration = original_duration / args.speed if not args.no_delay else 0
            print(f"Original trace duration: {original_duration:.1f}s")
            if not args.no_delay:
                print(f"Expected replay duration: {expected_duration:.1f}s")
        except:
            pass

    # Run replay
    print("\nStarting replay...")
    replayer = TraceReplayer(
        image_name=image_name,
        commands=commands,
        all_tool_calls=trace_parser.all_tool_calls,
        output_dir=output_dir,
        speed=args.speed,
        no_delay=args.no_delay,
        task_name=task_name
    )

    results = replayer.run()

    print("\n" + "=" * 70)
    print("Replay Summary")
    print("=" * 70)
    print(f"Total time: {results.get('total_time', 0):.1f}s")
    print(f"Commands executed: {len(commands)}")

    if "resource_samples" in results:
        summary = results["resource_samples"].get("summary", {})
        print(f"Resource samples: {summary.get('sample_count', 0)}")
        if "memory_mb" in summary:
            print(f"Memory (MB): min={summary['memory_mb']['min']:.1f}, "
                  f"max={summary['memory_mb']['max']:.1f}, avg={summary['memory_mb']['avg']:.1f}")
        if "cpu_percent" in summary:
            print(f"CPU (%): min={summary['cpu_percent']['min']:.1f}, "
                  f"max={summary['cpu_percent']['max']:.1f}, avg={summary['cpu_percent']['avg']:.1f}")

    print(f"\nOutput saved to: {output_dir}")

    if "error" in results:
        print(f"\nError: {results['error']}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
