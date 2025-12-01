import sys
import tempfile
import os
import pytest

from config import settings


def _make_tmp_file(tmp_path, content="hello world"):
    p = tmp_path / "doc.txt"
    p.write_text(content)
    return p


def test_cli_sync_extraction_flag(monkeypatch, tmp_path):
    # Prepare a temporary file
    tmp = _make_tmp_file(tmp_path)

    # Stub out the actual processing to avoid heavy work
    from ingestion import document_processor as dp_mod

    def fake_process_file(path):
        return {"status": "success", "chunks_created": 1}

    monkeypatch.setattr(dp_mod.document_processor, "process_file", fake_process_file)

    # Ensure starting value is False (or set a known value)
    monkeypatch.setattr(settings, "sync_entity_embeddings", False, raising=False)

    # Run the CLI main with --sync-extraction
    test_argv = ["ingest_documents.py", "--file", str(tmp), "--sync-extraction"]
    monkeypatch.setattr(sys, "argv", test_argv)

    # Import the script and run main; it will call sys.exit which raises SystemExit
    from scripts import ingest_documents

    with pytest.raises(SystemExit) as se:
        ingest_documents.main()

    # Expect successful exit
    assert se.value.code == 0

    # The CLI should have flipped the settings flag for this run
    assert settings.sync_entity_embeddings is True


def test_cli_wait_for_extraction_flag(monkeypatch, tmp_path):
    tmp = _make_tmp_file(tmp_path)

    from ingestion import document_processor as dp_mod

    def fake_process_file(path):
        return {"status": "success", "chunks_created": 1}

    monkeypatch.setattr(dp_mod.document_processor, "process_file", fake_process_file)

    # Create a stateful is_entity_extraction_running that returns True twice then False
    call = {"c": 0}

    def fake_is_running():
        call["c"] += 1
        return call["c"] < 3

    monkeypatch.setattr(dp_mod.document_processor, "is_entity_extraction_running", fake_is_running)

    test_argv = ["ingest_documents.py", "--file", str(tmp), "--wait-for-extraction", "--wait-timeout", "5"]
    monkeypatch.setattr(sys, "argv", test_argv)

    from scripts import ingest_documents

    with pytest.raises(SystemExit) as se:
        ingest_documents.main()

    assert se.value.code == 0
