import os
import time
import secrets
import socket

import pytest

try:
    from testcontainers.core.container import DockerContainer
except Exception as e:
    raise RuntimeError("testcontainers is required for e2e tests. Install with 'pip install testcontainers'") from e


@pytest.fixture(scope="session", autouse=True)
def neo4j_container():
    """Start a temporary Neo4j container for e2e tests and set env vars.

    Uses a randomly generated password to avoid relying on repo defaults.
    """
    password = secrets.token_urlsafe(12)
    auth = f"neo4j/{password}"

    # Check whether the Docker daemon is available to the Python docker client
    can_use_docker = False
    try:
        import docker as _docker
        client = _docker.from_env()
        # ping() will raise if the daemon is unreachable
        client.ping()
        can_use_docker = True
    except Exception:
        can_use_docker = False

    started_container = False
    container = None
    try:
        if can_use_docker:
            try:
                container = DockerContainer("neo4j:5.21").with_env("NEO4J_AUTH", auth).with_exposed_ports(7474, 7687)
                container.start()
                started_container = True
                host = container.get_container_host_ip()
                bolt_port = container.get_exposed_port(7687)
                bolt_uri = f"bolt://{host}:{bolt_port}"
                os.environ["NEO4J_URI"] = bolt_uri
                os.environ["NEO4J_USERNAME"] = "neo4j"
                os.environ["NEO4J_PASSWORD"] = password
            except Exception as e:
                # Fall back to compose-hosted or externally-provided Neo4j instance
                msg = str(e)
                fallback_info = (
                    "Testcontainers could not start a container (Docker may be unavailable). "
                    "Falling back to using an existing Neo4j at neo4j://localhost:7687."
                )
                print(f"[tests/e2e/conftest.py] {fallback_info} (cause: {msg})")

                bolt_uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
                username = os.environ.get("NEO4J_USERNAME")
                pwd = os.environ.get("NEO4J_PASSWORD")

                if not username or not pwd:
                    secrets_path = os.path.join(os.getcwd(), ".secrets", "neo4j_password")
                    if os.path.exists(secrets_path):
                        try:
                            with open(secrets_path, "r") as f:
                                pwd = f.read().strip()
                                username = "neo4j"
                        except Exception:
                            pass

                if (not username) or (not pwd):
                    auth_env = os.environ.get("NEO4J_AUTH")
                    if auth_env and "/" in auth_env:
                        parts = auth_env.split("/", 1)
                        username = parts[0]
                        pwd = parts[1]

                if (not username) or (not pwd):
                    raise RuntimeError(
                        "Unable to determine Neo4j credentials for e2e tests. "
                        "When Docker is unavailable, set `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`, "
                        "or run the tests where Docker is accessible."
                    )

                os.environ["NEO4J_URI"] = bolt_uri
                os.environ["NEO4J_USERNAME"] = username
                os.environ["NEO4J_PASSWORD"] = pwd

        else:
            # No docker access: use provided or default compose-hosted Neo4j
            bolt_uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
            username = os.environ.get("NEO4J_USERNAME")
            pwd = os.environ.get("NEO4J_PASSWORD")
            if not username or not pwd:
                secrets_path = os.path.join(os.getcwd(), ".secrets", "neo4j_password")
                if os.path.exists(secrets_path):
                    try:
                        with open(secrets_path, "r") as f:
                            pwd = f.read().strip()
                            username = "neo4j"
                    except Exception:
                        pass
            if (not username) or (not pwd):
                auth_env = os.environ.get("NEO4J_AUTH")
                if auth_env and "/" in auth_env:
                    parts = auth_env.split("/", 1)
                    username = parts[0]
                    pwd = parts[1]
            if (not username) or (not pwd):
                raise RuntimeError(
                    "Unable to determine Neo4j credentials for e2e tests. "
                    "When Docker is unavailable, set `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`, "
                    "or run the tests where Docker is accessible."
                )
            os.environ["NEO4J_URI"] = bolt_uri
            os.environ["NEO4J_USERNAME"] = username
            os.environ["NEO4J_PASSWORD"] = pwd

        # Wait for bolt port to accept connections
        parsed = os.environ["NEO4J_URI"].replace("neo4j://", "").replace("bolt://", "").split(":")
        host = parsed[0]
        port = int(parsed[1]) if len(parsed) > 1 else 7687
        for _ in range(60):
            try:
                with socket.create_connection((host, int(port)), timeout=1):
                    break
            except Exception:
                time.sleep(1)

        yield
    finally:
        if started_container and container is not None:
            try:
                container.stop()
            except Exception:
                pass
