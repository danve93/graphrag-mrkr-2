import pytest

from neo4j.exceptions import ServiceUnavailable

from core.graph_db import GraphDB


def test_session_scope_retries_and_fails(monkeypatch):
    g = GraphDB()

    # Make ensure_connected always raise ServiceUnavailable to exercise retry path
    def always_fail_connect():
        raise ServiceUnavailable("simulated down")

    monkeypatch.setattr(g, "ensure_connected", always_fail_connect)

    # Use a small max_attempts and backoff to keep test fast
    with pytest.raises(ServiceUnavailable):
        with g.session_scope(max_attempts=2, initial_backoff=0.01):
            pass


def test_session_scope_successful(monkeypatch):
    g = GraphDB()

    # Provide a dummy driver/session that yields a session object and records close
    class DummySession:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class DummyDriver:
        def session(self):
            return DummySession()

        def close(self):
            pass

    def ensure_connect_sets_driver():
        g.driver = DummyDriver()

    monkeypatch.setattr(g, "ensure_connected", ensure_connect_sets_driver)

    # Use the context manager and ensure session object is usable and closed
    with g.session_scope() as session:
        assert hasattr(session, "close")

    # after context exit the DummySession.close should have been called
    # the last created session (session variable) should have closed flag True
    assert getattr(session, "closed", True) is True
