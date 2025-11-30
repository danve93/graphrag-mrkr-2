import pytest

from contextlib import contextmanager

from core.graph_db import GraphDB


def test_get_communities_aggregates_validation():
    g = GraphDB()
    with pytest.raises(ValueError):
        g.get_communities_aggregates_for_level(None)

    with pytest.raises(ValueError):
        g.get_communities_aggregates_for_level('not-an-int')


def test_get_communities_aggregates_returns_expected_shape(monkeypatch):
    g = GraphDB()

    # Create a fake session.run return value (iterable of record-like dicts)
    fake_records = [
        {
            "community_id": 1,
            "entities": [
                {"id": "e1", "name": "Entity 1", "type": "COMPONENT", "importance_score": 0.8}
            ],
            "relationships": [
                {"id": 101, "source": "e1", "target": "e2", "type": "RELATED_TO", "text_unit_ids": ["t1"]}
            ],
        }
    ]

    class FakeSession:
        def run(self, query, level=None):
            return fake_records

    @contextmanager
    def fake_session_scope(max_attempts=3, initial_backoff=0.5):
        yield FakeSession()

    # Monkeypatch the instance's session_scope to use our fake
    monkeypatch.setattr(g, "session_scope", fake_session_scope)

    communities = g.get_communities_aggregates_for_level(0)
    assert isinstance(communities, list)
    assert len(communities) == 1
    c = communities[0]
    assert c["community_id"] == 1
    assert "entities" in c and isinstance(c["entities"], list)
    assert "relationships" in c and isinstance(c["relationships"], list)
    assert c["entity_count"] == len(c["entities"])
    assert c["relationship_count"] == len(c["relationships"])
