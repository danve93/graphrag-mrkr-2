#!/usr/bin/env zsh
# Quick frontend starter that kills port 3000 first

set -e

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

# Kill any process on port 3000
echo "Killing any process on port 3000..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
sleep 1

# Kill existing frontend pid if tracked
if [ -f "$ROOT_DIR/frontend-dev.pid" ]; then
  PID=$(cat "$ROOT_DIR/frontend-dev.pid")
  if ps -p $PID >/dev/null 2>&1; then
    echo "Killing tracked frontend process $PID"
    kill $PID || kill -9 $PID
  fi
  rm -f "$ROOT_DIR/frontend-dev.pid"
fi

echo "Starting frontend on port 3000..."
cd "$ROOT_DIR/frontend"
npm run dev
