import os
import shutil
import socket
import subprocess
import time
import logging
from pathlib import Path

import pytest

LOG = logging.getLogger("tests.conftest")
DOCKER_AVAILABLE = shutil.which("docker") is not None


def pytest_collection_modifyitems(config, items):
    """Skip service-dependent suites when Docker is unavailable."""

    if DOCKER_AVAILABLE:
        return

    skip_marker = pytest.mark.skip(reason="Docker CLI not available; skipping integration and e2e tests that require services")
    for item in items:
        path = Path(str(getattr(item, "fspath", "")))
        if "integration" in path.parts or "e2e" in path.parts:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session", autouse=True)
def docker_services():
    """Start docker compose services for the integration tests.

    - Runs `docker compose up -d --build` at the start of the test session.
    - Waits for Neo4j to accept Bolt connections on localhost:7687 (configurable
      via `TEST_SERVICES_WAIT` env var, default 120s).
    - Emits the last 200 lines of logs for `neo4j`, `backend`, and `worker` to
      help with readiness debugging.
    - Tears down with `docker compose down --volumes` at the end of the session
      unless `TEST_KEEP_SERVICES` is set to a truthy value.
    """

    if not DOCKER_AVAILABLE:
        LOG.warning("Docker CLI not available; skipping docker compose startup for tests")
        yield
        return

    wait_seconds = int(os.environ.get("TEST_SERVICES_WAIT", "120"))
    keep_services = os.environ.get("TEST_KEEP_SERVICES", "0") not in ("0", "", None)

    # Use an isolated compose project so test containers are unique and
    # removed after the test session. Allow overriding via
    # `TEST_DOCKER_PROJECT` for reproducibility.
    project = os.environ.get("TEST_DOCKER_PROJECT") or f"pytest_{int(time.time())}_{os.getpid()}"

    LOG.info("Ensuring any existing test containers for project=%s are stopped...", project)
    skip_cleanup = os.environ.get("TEST_SKIP_CLEANUP", "0") not in ("0", "", None)
    if not skip_cleanup:
        try:
            # Attempt to remove any containers that may conflict with the compose file
            # We parse common container_name entries from docker-compose.yml (best-effort)
            compose_file = os.path.join(os.getcwd(), "docker-compose.yml")
            if os.path.exists(compose_file):
                with open(compose_file, "r", encoding="utf-8") as fh:
                    content = fh.read()
                # crude parse: look for lines like 'container_name: NAME'
                names = []
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith("container_name:"):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            name = parts[1].strip()
                            if name:
                                names.append(name)
                # Also include the raw service names as fallback (they become <project>_<service>)
                # but we'll attempt removal of explicit container names first.
                for cname in names:
                    # If a container with that name exists, remove it (force)
                    try:
                        # `docker ps -a -q -f name=^/cname$` would be precise, but use --filter name for portability
                        found = subprocess.run(["docker", "ps", "-a", "--filter", f"name=^{cname}$", "-q"], capture_output=True, text=True)
                        if found.stdout.strip():
                            LOG.info("Removing conflicting container: %s", cname)
                            subprocess.run(["docker", "rm", "-f", cname], check=False)
                    except Exception:
                        LOG.exception("Error while removing container %s (continuing)", cname)
        except Exception:
            LOG.exception("Failed to tear down pre-existing compose containers (continuing)")
    else:
        LOG.info("Skipping pre-start cleanup of existing containers (TEST_SKIP_CLEANUP set)")

    LOG.info("Starting docker compose services for integration tests (project=%s)...", project)
    try:
        # If the project already has services (e.g., we started it manually during debugging),
        # skip the `up` step to avoid compose errors. This helps when running pytest after
        # an interactive `docker compose up` during local troubleshooting.
        # Force docker-compose to use container-internal connection strings so that
        # service hostnames (e.g. `neo4j`) are used inside containers. This prevents
        # host-level env vars like `NEO4J_URI=bolt://localhost:7687` from being
        # propagated into containers and causing them to look for services on
        # their own localhost instead of the compose network.
        compose_env = os.environ.copy()
        # Ensure container-internal Neo4j address is used
        compose_env.setdefault("NEO4J_URI", "bolt://neo4j:7687")
        # If username/password are present, set NEO4J_AUTH for the neo4j container init
        if "NEO4J_USERNAME" in compose_env and "NEO4J_PASSWORD" in compose_env:
            compose_env.setdefault("NEO4J_AUTH", f"{compose_env['NEO4J_USERNAME']}/{compose_env['NEO4J_PASSWORD']}")

        ps = subprocess.run(["docker", "compose", "-p", project, "ps", "-q"], capture_output=True, text=True, env=compose_env)
        if ps.stdout.strip():
            LOG.info("Compose project '%s' already has running services; skipping `up`.", project)
        else:
            # Run compose up and capture output to help diagnose failures in CI/local runs.
            try:
                subprocess.run(["docker", "compose", "-p", project, "up", "-d", "--build"], check=True, capture_output=True, text=True, env=compose_env)
            except subprocess.CalledProcessError as cpe:  # pragma: no cover - environment dependent
                # Emit captured stdout/stderr to logs for easier debugging, then try a lighter fallback.
                LOG.error("`docker compose up` failed (stdout):\n%s", cpe.stdout)
                LOG.error("`docker compose up` failed (stderr):\n%s", cpe.stderr)
                LOG.info("Attempting fallback: `docker compose up -d` (no --build)")
                try:
                    subprocess.run(["docker", "compose", "-p", project, "up", "-d"], check=True, capture_output=True, text=True, env=compose_env)
                except subprocess.CalledProcessError as cpe2:  # pragma: no cover - environment dependent
                    LOG.error("Fallback `docker compose up -d` also failed (stdout):\n%s", cpe2.stdout)
                    LOG.error("Fallback `docker compose up -d` also failed (stderr):\n%s", cpe2.stderr)
                    # Provide additional diagnostics before failing hard
                    try:
                        subprocess.run(["docker", "compose", "-p", project, "ps"], check=False, env=compose_env)
                        subprocess.run(["docker", "ps", "-a"], check=False)
                    except Exception:
                        LOG.exception("Failed to emit additional docker diagnostics")
                    raise
    except Exception as exc:  # pragma: no cover - environment dependent
        LOG.exception("Failed to start docker compose: %s", exc)
        raise

    # Wait for Neo4j bolt port
    start = time.time()
    while time.time() - start < wait_seconds:
        try:
            with socket.create_connection(("localhost", 7687), timeout=1):
                LOG.info("Neo4j bolt port 7687 is accepting connections")
                break
        except Exception:
            time.sleep(1)
    else:
        LOG.error("Neo4j did not become ready within %s seconds", wait_seconds)
        # Dump recent logs to help debugging, then raise
        try:
            subprocess.run(["docker", "compose", "-p", project, "logs", "--no-color", "--tail=200", "neo4j", "backend", "worker"], check=False)
        except Exception:
            pass
        raise RuntimeError("Neo4j did not become ready within timeout")

    # Show recent service logs (non-blocking) to aid debugging of readiness
    try:
        subprocess.run(["docker", "compose", "-p", project, "logs", "--no-color", "--tail=200", "neo4j", "backend", "worker"], check=False)
    except Exception:
        LOG.exception("Failed to capture docker service logs")

    # Give a short grace for app-level readiness
    time.sleep(1)

    yield

    # Teardown: bring services down unless the env var indicates to keep them up
    if keep_services:
        LOG.info("Leaving docker services running (TEST_KEEP_SERVICES set); project=%s", project)
        return

    LOG.info("Tearing down docker compose services for integration tests (project=%s)...", project)
    try:
        subprocess.run(["docker", "compose", "-p", project, "down", "--volumes", "--remove-orphans"], check=False)
    except Exception:
        LOG.exception("Failed to shut down docker compose services")
