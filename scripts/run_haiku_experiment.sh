#!/bin/bash

# ============================================
# SWE-bench Haiku Experiment Runner
# Runs swebench experiments with Anthropic haiku
# No local server needed
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Config
TASK_LIST="$PROJECT_DIR/task_list.json"
MODEL="haiku"
PROGRESS_FILE="$PROJECT_DIR/experiments/all_images_haiku/progress.json"

# Create log directory
mkdir -p "$LOG_DIR"

# Log files
SWEBENCH_LOG="$LOG_DIR/swebench_haiku_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$SWEBENCH_LOG"
}

# Rotate old logs (keep last 5)
rotate_logs() {
    local count=$(ls -1 "$LOG_DIR"/swebench_haiku_*.log 2>/dev/null | wc -l)
    if [ "$count" -gt 5 ]; then
        ls -1t "$LOG_DIR"/swebench_haiku_*.log | tail -n +6 | xargs -r rm -f
        log "Rotated old logs"
    fi
}

rotate_logs

# Check if swebench is running
check_swebench_running() {
    pgrep -f "run_all_swebench_images.py.*--model haiku" > /dev/null 2>&1
}

# Start swebench experiment
start_swebench() {
    log "Starting swebench haiku experiment..."
    cd "$PROJECT_DIR"

    source .venv/bin/activate

    nohup .venv/bin/python -u scripts/run_all_swebench_images.py \
        --model haiku \
        --task-list "$TASK_LIST" \
        --resume \
        >> "$SWEBENCH_LOG" 2>&1 &

    local pid=$!
    disown $pid
    log "swebench started with PID: $pid"
    return 0
}

# Stop swebench
stop_swebench() {
    log "Stopping swebench..."
    pkill -f "run_all_swebench_images.py.*--model haiku" 2>/dev/null
    podman stop -a 2>/dev/null
    sleep 2
}

# Show progress
show_progress() {
    if [ -f "$PROGRESS_FILE" ]; then
        python3 -c "
import json
p = json.load(open('$PROGRESS_FILE'))
completed = len(p.get('completed', []))
success = sum(1 for r in p.get('results', {}).values() if r.get('success'))
print(f'Progress: {completed} completed, {success} successful')
"
    else
        echo "No progress file yet"
    fi
}

# Parse command
case "${1:-start}" in
    start)
        if check_swebench_running; then
            log "swebench haiku is already running"
            show_progress
        else
            start_swebench
            echo ""
            echo "Log: tail -f $SWEBENCH_LOG"
        fi
        ;;
    stop)
        stop_swebench
        log "Stopped"
        ;;
    restart)
        stop_swebench
        sleep 2
        start_swebench
        echo ""
        echo "Log: tail -f $SWEBENCH_LOG"
        ;;
    status)
        echo "swebench haiku: $(check_swebench_running && echo 'running' || echo 'stopped')"
        show_progress
        ;;
    log)
        # Show latest log
        LATEST_LOG=$(ls -1t "$LOG_DIR"/swebench_haiku_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            tail -f "$LATEST_LOG"
        else
            echo "No log files found"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|log}"
        exit 1
        ;;
esac
