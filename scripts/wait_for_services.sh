#!/usr/bin/env bash
set -euo pipefail

# Wait-for script: waits for Neo4j (and optionally other services) to be reachable
# Reads `NEO4J_URI` environment variable (default: bolt://neo4j:7687)

NEO4J_URI=${NEO4J_URI:-bolt://neo4j:7687}

echo "Detected NEO4J_URI=$NEO4J_URI"

# Extract host and port using Python (robust parsing)
NEO4J_HOST_PORT=$(python3 - <<'PY'
import os, urllib.parse
uri = os.environ.get('NEO4J_URI', 'bolt://neo4j:7687')
u = urllib.parse.urlparse(uri)
host = u.hostname or 'neo4j'
port = u.port or 7687
print(f"{host}:{port}")
PY
)

HOST=${NEO4J_HOST_PORT%%:*}
PORT=${NEO4J_HOST_PORT##*:}

echo "Waiting for Neo4j at ${HOST}:${PORT}"

MAX_WAIT=${WAIT_FOR_MAX_SECONDS:-90}
WAITED=0
SLEEP=2

while true; do
  # Quick auth-sanity check: if NEO4J_AUTH is provided and password length < 8, fail fast
  if [ -n "${NEO4J_AUTH:-}" ]; then
    # parse user/password
    pw="${NEO4J_AUTH#*/}"
    if [ ${#pw} -lt 8 ]; then
      echo "Detected NEO4J_AUTH with short password (${#pw} chars). Neo4j requires passwords >= 8 characters." >&2
      echo "Please set a stronger password in NEO4J_AUTH (format user/password) or export NEO4J_USERNAME/NEO4J_PASSWORD for the backend." >&2
      exit 2
    fi
  fi
  if bash -c "</dev/tcp/${HOST}/${PORT}" >/dev/null 2>&1; then
    echo "Service ${HOST}:${PORT} is reachable"
    break
  fi
  echo "Still waiting for ${HOST}:${PORT}... (${WAITED}s elapsed)"
  sleep ${SLEEP}
  WAITED=$((WAITED + SLEEP))
  if [ ${WAITED} -ge ${MAX_WAIT} ]; then
    echo "Timeout waiting for ${HOST}:${PORT} after ${MAX_WAIT}s"
    echo "Giving up — dependency not reachable. Exiting with error."
    # Print some diagnostics to help debugging
    if command -v nc >/dev/null 2>&1; then
      echo "nc scan:"
      nc -vz ${HOST} ${PORT} || true
    fi
    exit 1
  fi
done

echo "Proceeding to start application: $*"

exec "$@"
