#!/usr/bin/env python3
"""
Run a trace inside a SWE-rebench Docker container.

Features:
- Automatically pulls docker image if not present
- Executes both bash and SWE-agent editor commands
- Cleans up docker image after execution to save space

Usage:
    python scripts/run_trace_in_container.py <trace_json> <docker_image>
    python scripts/run_trace_in_container.py <trace_json> --auto  # Auto-detect image from CSV

Example:
    python scripts/run_trace_in_container.py \
        data/sample_traces/hugovk__pypistats-41.json \
        docker.io/swerebench/sweb.eval.x86_64.hugovk_1776_pypistats-41
"""

import argparse
import json
import subprocess
import time
import uuid
import re
from pathlib import Path


class SWEAgentEditor:
    """Simulates SWE-agent editor commands inside a container."""

    def __init__(self, runner: 'ContainerRunner'):
        self.runner = runner
        self.current_file = None
        self.current_line = 1

    def execute(self, command: str) -> dict:
        """Execute an editor command and return result."""
        command = command.strip()

        if command.startswith('create '):
            return self._create(command[7:].strip())
        elif command.startswith('open '):
            return self._open(command[5:].strip())
        elif command.startswith('goto '):
            return self._goto(command[5:].strip())
        elif command == 'scroll_down':
            return self._scroll_down()
        elif command == 'scroll_up':
            return self._scroll_up()
        elif command.startswith('edit '):
            return self._edit(command[5:])
        elif command.startswith('search_file '):
            return self._search_file(command[12:].strip())
        elif command.startswith('search_dir '):
            return self._search_dir(command[11:].strip())
        elif command == 'submit':
            return self._submit()
        else:
            return {"success": False, "exit_code": 1, "stderr": f"Unknown editor command: {command[:50]}",
                    "stdout": "", "latency_ms": 0}

    def _create(self, filename: str) -> dict:
        """Create a new file."""
        self.current_file = filename
        self.current_line = 1
        return self.runner.exec(f"touch {filename}")

    def _open(self, args: str) -> dict:
        """Open a file, optionally at a specific line."""
        parts = args.split()
        filename = parts[0]
        line_num = int(parts[1]) if len(parts) > 1 else 1

        self.current_file = filename
        self.current_line = line_num

        # Verify file exists and show content
        return self.runner.exec(f"test -f {filename} && sed -n '{max(1, line_num-5)},{line_num+100}p' {filename}")

    def _goto(self, line_str: str) -> dict:
        """Go to a specific line."""
        try:
            self.current_line = int(line_str.strip())
            if self.current_file:
                return self.runner.exec(f"sed -n '{max(1, self.current_line-5)},{self.current_line+100}p' {self.current_file}")
            return {"success": True, "exit_code": 0, "stdout": f"Moved to line {self.current_line}",
                    "stderr": "", "latency_ms": 0}
        except ValueError:
            return {"success": False, "exit_code": 1, "stderr": f"Invalid line number: {line_str}",
                    "stdout": "", "latency_ms": 0}

    def _scroll_down(self) -> dict:
        """Scroll down 100 lines."""
        self.current_line += 100
        if self.current_file:
            return self.runner.exec(f"sed -n '{self.current_line},{self.current_line+100}p' {self.current_file}")
        return {"success": True, "exit_code": 0, "stdout": "", "stderr": "", "latency_ms": 0}

    def _scroll_up(self) -> dict:
        """Scroll up 100 lines."""
        self.current_line = max(1, self.current_line - 100)
        if self.current_file:
            return self.runner.exec(f"sed -n '{self.current_line},{self.current_line+100}p' {self.current_file}")
        return {"success": True, "exit_code": 0, "stdout": "", "stderr": "", "latency_ms": 0}

    def _edit(self, edit_cmd: str) -> dict:
        """Apply an edit to the current file.

        Format: edit <start_line>:<end_line>
        <new_content>
        end_of_edit
        """
        if not self.current_file:
            return {"success": False, "exit_code": 1, "stderr": "No file open",
                    "stdout": "", "latency_ms": 0}

        # Parse edit command: "start:end\ncontent\nend_of_edit"
        lines = edit_cmd.strip().split('\n')
        if not lines:
            return {"success": False, "exit_code": 1, "stderr": "Empty edit command",
                    "stdout": "", "latency_ms": 0}

        # First line should be "start:end" or just line range
        range_line = lines[0].strip()
        match = re.match(r'(\d+):(\d+)', range_line)
        if match:
            start_line = int(match.group(1))
            end_line = int(match.group(2))
            content_lines = lines[1:]
        else:
            # Try to parse as single line number
            try:
                start_line = int(range_line.split()[0].rstrip(':'))
                end_line = start_line
                content_lines = lines[1:]
            except (ValueError, IndexError):
                # Assume content starts immediately
                start_line = self.current_line
                end_line = self.current_line
                content_lines = lines

        # Remove "end_of_edit" marker if present
        if content_lines and content_lines[-1].strip() == 'end_of_edit':
            content_lines = content_lines[:-1]

        new_content = '\n'.join(content_lines)

        # Use sed to replace lines (create temp file approach for safety)
        # Write new content to temp file, then use sed to replace range
        escaped_content = new_content.replace("'", "'\\''").replace('$', '\\$')

        if start_line == end_line and not new_content:
            # Delete single line
            cmd = f"sed -i '{start_line}d' {self.current_file}"
        elif start_line <= end_line:
            # Write new content to a temp file and use Python to apply the edit
            import base64
            encoded_content = base64.b64encode(new_content.encode()).decode()
            cmd = f'''python3 -c "
import base64
with open('{self.current_file}', 'r') as f:
    lines = f.readlines()
new_content = base64.b64decode('{encoded_content}').decode()
new_lines = lines[:{start_line - 1}] + [new_content + '\\n'] + lines[{end_line}:]
with open('{self.current_file}', 'w') as f:
    f.writelines(new_lines)
print('Edit applied: lines {start_line}-{end_line}')
"'''
        else:
            return {"success": False, "exit_code": 1, "stderr": f"Invalid range: {start_line}:{end_line}",
                    "stdout": "", "latency_ms": 0}

        return self.runner.exec(cmd, timeout=30)

    def _search_file(self, pattern: str) -> dict:
        """Search for pattern in current file."""
        pattern = pattern.strip().strip('"\'')
        if self.current_file:
            return self.runner.exec(f"grep -n '{pattern}' {self.current_file} | head -20")
        return {"success": False, "exit_code": 1, "stderr": "No file open",
                "stdout": "", "latency_ms": 0}

    def _search_dir(self, args: str) -> dict:
        """Search for pattern in directory."""
        parts = args.strip().split()
        pattern = parts[0].strip('"\'') if parts else ""
        directory = parts[1] if len(parts) > 1 else "."
        return self.runner.exec(f"grep -rn '{pattern}' {directory} 2>/dev/null | head -30")

    def _submit(self) -> dict:
        """Submit the changes (no-op for replay, just marks completion)."""
        return {"success": True, "exit_code": 0, "stdout": "Submitted",
                "stderr": "", "latency_ms": 0}


class ContainerRunner:
    """Manages a persistent container for running trace commands."""

    def __init__(self, docker_image: str, workdir: str = "/testbed"):
        self.docker_image = docker_image
        self.workdir = workdir
        self.container_id = None

    def pull_image(self) -> bool:
        """Pull the docker image if not present."""
        # Check if image exists locally
        result = subprocess.run(
            ["podman", "image", "exists", self.docker_image],
            capture_output=True
        )
        if result.returncode == 0:
            print(f"Image already exists: {self.docker_image}")
            return True

        print(f"Pulling image: {self.docker_image}...")
        result = subprocess.run(
            ["podman", "pull", self.docker_image],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"Failed to pull image: {result.stderr}")
            return False
        print("Image pulled successfully")
        return True

    def remove_image(self):
        """Remove the docker image to free space."""
        print(f"Removing image: {self.docker_image}...")
        subprocess.run(
            ["podman", "rmi", "-f", self.docker_image],
            capture_output=True
        )
        print("Image removed")

    def start(self):
        """Start the container in detached mode."""
        name = f"trace-runner-{uuid.uuid4().hex[:8]}"
        result = subprocess.run(
            ["podman", "run", "-d", "--name", name, "-w", self.workdir,
             self.docker_image, "sleep", "3600"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")
        self.container_id = result.stdout.strip()
        print(f"Started container: {self.container_id[:12]}")
        return self.container_id

    def stop(self):
        """Stop and remove the container."""
        if self.container_id:
            subprocess.run(["podman", "rm", "-f", self.container_id],
                           capture_output=True)
            print(f"Stopped container: {self.container_id[:12]}")
            self.container_id = None

    def exec(self, command: str, timeout: int = 60) -> dict:
        """Execute a command in the running container."""
        if not self.container_id:
            raise RuntimeError("Container not started")

        # Wrap command to activate conda environment
        full_cmd = f"source /opt/conda/etc/profile.d/conda.sh && conda activate testbed && cd {self.workdir} && {command}"

        start = time.perf_counter()
        try:
            result = subprocess.run(
                ["podman", "exec", self.container_id, "bash", "-c", full_cmd],
                capture_output=True, text=True, timeout=timeout
            )
            elapsed = time.perf_counter() - start
            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
                "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
                "latency_ms": elapsed * 1000
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": "TIMEOUT",
                "latency_ms": timeout * 1000
            }
        except Exception as e:
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "latency_ms": (time.perf_counter() - start) * 1000
            }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def replay_trace(trace_path: Path, docker_image: str, dry_run: bool = False,
                 setup_cmd: str = None, cleanup_image: bool = False):
    """Replay a trace inside a container."""
    with open(trace_path) as f:
        trace = json.load(f)

    print(f"Trace: {trace['trace_id']}")
    print(f"Image: {docker_image}")
    print(f"Steps: {len(trace['steps'])}")
    print("=" * 70)

    results = []

    if dry_run:
        for step in trace["steps"]:
            step_id = step["step_id"]
            tool = step["tool"]
            command = step["command"]
            cmd_display = command.replace('\n', '\\n')[:60]
            print(f"[{step_id:2d}] DRY-RUN {tool}: {cmd_display}...")
            results.append({"step_id": step_id, "tool": tool, "command": command[:100],
                            "success": True, "exit_code": 0, "latency_ms": 0})
        return results

    # Create runner and pull image
    runner = ContainerRunner(docker_image)

    if not runner.pull_image():
        print("ERROR: Failed to pull image")
        return results

    try:
        with runner:
            editor = SWEAgentEditor(runner)

            # Run setup command if provided
            if setup_cmd:
                print(f"[--] SETUP: {setup_cmd[:60]}...")
                result = runner.exec(setup_cmd, timeout=120)
                status = "OK" if result["success"] else f"FAIL({result['exit_code']})"
                print(f"       -> {status}, latency={result['latency_ms']:.0f}ms")
                if not result["success"]:
                    print(f"       -> {result['stderr'][-100:]}")
                print("-" * 70)

            # Execute trace steps
            for step in trace["steps"]:
                step_id = step["step_id"]
                tool = step["tool"]
                command = step["command"]
                cmd_display = command.replace('\n', '\\n')[:60]

                print(f"[{step_id:2d}] {tool}: {cmd_display}...")

                if tool == "bash":
                    result = runner.exec(command, timeout=120)
                elif tool == "swe_agent_editor":
                    result = editor.execute(command)
                else:
                    result = {"success": False, "exit_code": 1,
                              "stderr": f"Unknown tool: {tool}", "stdout": "", "latency_ms": 0}

                results.append({
                    "step_id": step_id,
                    "tool": tool,
                    "command": command[:100],
                    **result
                })

                status = "OK" if result["success"] else f"FAIL({result['exit_code']})"
                print(f"       -> {status}, latency={result['latency_ms']:.0f}ms")

                if result["stdout"] and result["success"]:
                    # Show brief output
                    out_preview = result["stdout"].strip().split('\n')[0][:60]
                    if out_preview:
                        print(f"       -> {out_preview}")

                if result["stderr"] and not result["success"]:
                    last_err = result["stderr"].strip().split('\n')[-1][:70]
                    print(f"       -> {last_err}")

    finally:
        if cleanup_image:
            runner.remove_image()

    print("=" * 70)

    # Summary
    total_steps = len(results)
    success_count = sum(1 for r in results if r["success"])
    total_latency = sum(r["latency_ms"] for r in results)
    bash_count = sum(1 for r in results if r["tool"] == "bash")
    editor_count = sum(1 for r in results if r["tool"] == "swe_agent_editor")

    print(f"Total steps: {total_steps} (bash: {bash_count}, editor: {editor_count})")
    print(f"Success: {success_count}/{total_steps}")
    print(f"Total latency: {total_latency:.0f}ms")

    return results


def get_docker_image_for_trace(trace_path: Path) -> str:
    """Look up docker image from runnable_traces.csv."""
    csv_path = trace_path.parent.parent / "runnable_traces.csv"
    if not csv_path.exists():
        return None

    trace_id = trace_path.stem  # e.g., "hugovk__pypistats-41"
    # Handle suffix like "_0000"
    base_id = re.sub(r'_\d{4}$', '', trace_id)

    import csv
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['instance_id'] == base_id or row['instance_id'] == trace_id:
                img = row['docker_image']
                if not img.startswith('docker.io/'):
                    img = f"docker.io/{img}"
                return img
    return None


def main():
    parser = argparse.ArgumentParser(description="Run trace in container")
    parser.add_argument("trace_path", type=Path, help="Path to trace JSON file")
    parser.add_argument("docker_image", type=str, nargs='?', default=None,
                        help="Docker image name (or --auto to detect)")
    parser.add_argument("--auto", action="store_true", help="Auto-detect docker image from CSV")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually run commands")
    parser.add_argument("--setup", type=str, default="pip install -e . -q 2>/dev/null || true",
                        help="Setup command to run first")
    parser.add_argument("--no-setup", action="store_true", help="Skip setup step")
    parser.add_argument("--cleanup", action="store_true", help="Remove docker image after run")
    parser.add_argument("--output", type=Path, help="Output metrics file (JSON)")
    args = parser.parse_args()

    # Determine docker image
    docker_image = args.docker_image
    if args.auto or docker_image is None:
        docker_image = get_docker_image_for_trace(args.trace_path)
        if not docker_image:
            print(f"ERROR: Could not auto-detect docker image for {args.trace_path}")
            print("Please provide docker image explicitly or ensure runnable_traces.csv exists")
            return

    if not docker_image.startswith('docker.io/'):
        docker_image = f"docker.io/{docker_image}"

    setup_cmd = None if args.no_setup else args.setup
    results = replay_trace(args.trace_path, docker_image, args.dry_run, setup_cmd, args.cleanup)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
