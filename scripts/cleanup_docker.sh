#!/usr/bin/env bash
set -euo pipefail

# Stops and removes the project's Compose stacks (main and e2e) to free ports
# commonly used by the stack. This helps when a previous run left containers
# or published ports running (7474, 7687, 8000, 3000, 6379).

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not installed or not on PATH" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_FILES=("$ROOT_DIR/docker-compose.yml")
OVERRIDE_FILE="$ROOT_DIR/docker-compose.override.yml"
E2E_FILE="$ROOT_DIR/docker-compose.e2e.yml"

if [ -f "$OVERRIDE_FILE" ]; then
  MAIN_FILES+=("$OVERRIDE_FILE")
fi

# Compose derives a project name from the directory by default, but users may
# override it. Attempt cleanup with both the provided COMPOSE_PROJECT_NAME and
# a sane fallback to catch common cases.
PROJECT_NAMES=("${COMPOSE_PROJECT_NAME:-graphrag-mrkr-2}" "graphrag")

echo "Stopping main stack containers (if any)..."
for project in "${PROJECT_NAMES[@]}"; do
  docker compose -p "$project" -f "$ROOT_DIR/docker-compose.yml" down --remove-orphans --volumes --timeout 30 || true
  if [ -f "$OVERRIDE_FILE" ]; then
    docker compose -p "$project" -f "$ROOT_DIR/docker-compose.yml" -f "$OVERRIDE_FILE" down --remove-orphans --volumes --timeout 30 || true
  fi
  echo "- attempted project: $project"
done

echo "Stopping e2e stack containers (if any)..."
if [ -f "$E2E_FILE" ]; then
  for project in "${PROJECT_NAMES[@]}"; do
    docker compose -p "${project}-e2e" -f "$E2E_FILE" down --remove-orphans --volumes --timeout 30 || true
    echo "- attempted project: ${project}-e2e"
  done
else
  echo "- e2e compose file not present; skipping"
fi

echo
echo "Remaining containers publishing the usual ports (3000, 8000, 7474, 7687, 6379):"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E ':(3000|8000|7474|7687|6379)->' || echo "No containers currently publishing those ports."
