#!/usr/bin/env zsh
# Dev startup helper for frontend (Next) and backend (FastAPI)
# Usage: ./scripts/dev.sh [frontend|backend|all|restart|stop]

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PIDS_FILE="$ROOT_DIR/.dev_pids"
LOG_DIR="$ROOT_DIR/logs"

mkdir -p "$LOG_DIR"

function pid_alive() {
  local pid=$1
  if [ -z "$pid" ]; then
    return 1
  fi
  kill -0 "$pid" >/dev/null 2>&1
}

function read_pids() {
  if [ -f "$PIDS_FILE" ]; then
    source "$PIDS_FILE"
  fi
}

function write_pids() {
  : > "$PIDS_FILE"
  [ -n "${FRONTEND_PID-}" ] && echo "FRONTEND_PID=${FRONTEND_PID}" >> "$PIDS_FILE"
  [ -n "${BACKEND_PID-}" ] && echo "BACKEND_PID=${BACKEND_PID}" >> "$PIDS_FILE"
}

function stop_all() {
  read_pids
  echo "Stopping dev processes..."
  if [ -n "${FRONTEND_PID-}" ] && pid_alive "$FRONTEND_PID"; then
    echo "Killing frontend pid $FRONTEND_PID"
    kill "$FRONTEND_PID" || true
  fi
  if [ -n "${BACKEND_PID-}" ] && pid_alive "$BACKEND_PID"; then
    echo "Killing backend pid $BACKEND_PID"
    kill "$BACKEND_PID" || true
  fi
  rm -f "$PIDS_FILE"
}

function start_backend() {
  read_pids
  if [ -n "${BACKEND_PID-}" ] && pid_alive "$BACKEND_PID"; then
    echo "Backend already running (pid $BACKEND_PID)"
    return
  fi

  echo "Starting backend (uvicorn) in background..."
  # Prefer venv if present
  if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PY_CMD="$ROOT_DIR/.venv/bin/python"
  else
    PY_CMD=$(command -v python3 || command -v python)
  fi

  if [ -z "$PY_CMD" ]; then
    echo "No python executable found in PATH. Install Python or create .venv." >&2
    return 1
  fi

  nohup "$PY_CMD" -m uvicorn api.main:app --reload --port 8000 > "$LOG_DIR/dev-backend.log" 2>&1 &
  BACKEND_PID=$!
  echo "Backend started pid=$BACKEND_PID (logs: $LOG_DIR/dev-backend.log)"
  write_pids
}

function start_frontend_bg() {
  read_pids
  if [ -n "${FRONTEND_PID-}" ] && pid_alive "$FRONTEND_PID"; then
    echo "Frontend already running (pid $FRONTEND_PID)"
    return
  fi

  echo "Starting frontend (npm run dev) in background..."
  nohup sh -c "cd '$ROOT_DIR/frontend' && npm run dev" > "$LOG_DIR/dev-frontend.log" 2>&1 &
  FRONTEND_PID=$!
  echo "Frontend started pid=$FRONTEND_PID (logs: $LOG_DIR/dev-frontend.log)"
  write_pids
}

function start_frontend_fg() {
  echo "Starting frontend (npm run dev) in foreground..."
  cd "$ROOT_DIR/frontend"
  npm run dev
}

COMMAND=${1:-all}

case "$COMMAND" in
  stop)
    stop_all
    ;;
  restart)
    stop_all
    sleep 0.3
    start_backend
    start_frontend_fg
    ;;
  backend)
    start_backend
    ;;
  frontend)
    start_frontend_fg
    ;;
  all|both)
    # start backend in background, frontend in foreground (so logs attach)
    start_backend
    start_frontend_fg
    ;;
  bg)
    # start both in background
    start_backend
    start_frontend_bg
    ;;
  *)
    echo "Usage: $0 [frontend|backend|all|both|bg|restart|stop]"
    exit 1
    ;;
esac
