import pytest
import sqlite3
import json
import tempfile
from pathlib import Path
from core.chunk_change_log import ChunkChangeLog

@pytest.fixture
def change_log():
    # Use a temporary file for the DB
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    
    log = ChunkChangeLog(db_path=db_path)
    yield log
    
    # Cleanup
    try:
        db_path.unlink()
    except FileNotFoundError:
        pass

def test_log_change_and_get(change_log):
    """Test logging a change and retrieving it."""
    change_id = change_log.log_change(
        document_id="doc1",
        chunk_id="chk1",
        action="edit",
        before_content="foo",
        after_content="bar",
        reasoning="typo fix"
    )
    
    assert change_id > 0
    
    changes = change_log.get_changes(document_id="doc1")
    assert len(changes) == 1
    assert changes[0]["chunk_id"] == "chk1"
    assert changes[0]["action"] == "edit"
    assert changes[0]["reasoning"] == "typo fix"
    assert changes[0]["before_content"] == "foo"
    assert changes[0]["after_content"] == "bar"

def test_get_changes_filters(change_log):
    """Test filtering changes."""
    change_log.log_change("doc1", "c1", "edit")
    change_log.log_change("doc1", "c2", "delete")
    change_log.log_change("doc2", "c3", "edit")
    
    # Filter by doc
    doc1_changes = change_log.get_changes(document_id="doc1")
    assert len(doc1_changes) == 2
    
    # Filter by action
    edits = change_log.get_changes(action="edit")
    assert len(edits) == 2
    
    # Filter by chunk
    c2 = change_log.get_changes(chunk_id="c2")
    assert len(c2) == 1
    assert c2[0]["chunk_id"] == "c2"

def test_export_changes(change_log):
    """Test exporting changes to JSON-friendly format."""
    change_log.log_change("doc1", "c1", "edit", "old", "new")
    
    export = change_log.export_changes(document_id="doc1")
    assert export["document_id"] == "doc1"
    assert export["total_changes"] == 1
    assert export["changes"][0]["before_content"] == "old"
    
    # Test without content
    # Note: export_changes doesn't support omit_content arg in the implementation I read?
    # Let me check the code I read.
    # line 236: include_content: bool = True.
    # Yes it does.
    
    export_no_content = change_log.export_changes(document_id="doc1", include_content=False)
    assert "before_content" not in export_no_content["changes"][0]

def test_delete_changes(change_log):
    """Test deleting old changes."""
    change_log.log_change("doc1", "c1", "edit")
    
    # Delete by doc
    deleted = change_log.delete_changes(document_id="doc1")
    assert deleted == 1
    
    assert len(change_log.get_changes()) == 0

def test_metadata_serialization(change_log):
    """Test that metadata is correctly serialized/deserialized."""
    meta = {"merged_chunks": ["c2", "c3"]}
    change_log.log_change("doc1", "c1", "merge", metadata=meta)
    
    changes = change_log.get_changes()
    assert changes[0]["metadata"] == meta
