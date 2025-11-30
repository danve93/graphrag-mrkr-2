#!/usr/bin/env bash
# Portable frontend starter: gently terminates any process listening on port 3000
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

echo "ROOT_DIR=$ROOT_DIR"

# Helper: attempt polite termination, then force if needed
terminate_pids() {
  local pids="$1"
  for pid in $pids; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "Terminating pid $pid (SIGTERM)..."
      kill "$pid" 2>/dev/null || true
    fi
  done

  # give them a short grace period
  sleep 3

  # force remaining
  local still
  still=$(lsof -ti:3000 || true)
  if [ -n "$still" ]; then
    echo "Forcing kill of remaining pids: $still"
    kill -9 $still 2>/dev/null || true
  fi
}

echo "Checking for processes listening on port 3000..."
PIDS=$(lsof -ti:3000 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  terminate_pids "$PIDS"
else
  echo "No process found on port 3000"
fi

# Kill existing frontend pid if tracked (gentle)
if [ -f "$ROOT_DIR/frontend-dev.pid" ]; then
  PID=$(cat "$ROOT_DIR/frontend-dev.pid") || true
  if [ -n "${PID:-}" ] && ps -p "$PID" >/dev/null 2>&1; then
    echo "Terminating tracked frontend process $PID"
    terminate_pids "$PID"
  fi
  rm -f "$ROOT_DIR/frontend-dev.pid" || true
fi

echo "Starting frontend (developer mode) on port 3000..."
cd "$ROOT_DIR/frontend"
# Run in foreground so logs are visible; developers can run in a terminal or use their own process manager
npm run dev
