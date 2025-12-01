import time
import threading

import pytest

from ingestion import document_processor as dp_module
from ingestion.document_processor import document_processor, EntityExtractionState, EntityExtractionStatus


def test_is_entity_extraction_running_false():
    # Ensure clean state
    with document_processor._bg_lock:
        document_processor._bg_entity_threads.clear()
    with document_processor._operations_lock:
        document_processor._entity_extraction_operations.clear()

    assert document_processor.is_entity_extraction_running() is False


def test_is_entity_extraction_running_with_tracked_thread():
    class FakeThread:
        def is_alive(self):
            return True

        daemon = True
        name = "fake-thread"

    fake = FakeThread()
    with document_processor._bg_lock:
        document_processor._bg_entity_threads.append(fake)

    try:
        assert document_processor.is_entity_extraction_running() is True
    finally:
        with document_processor._bg_lock:
            document_processor._bg_entity_threads = [t for t in document_processor._bg_entity_threads if t is not fake]


def test_is_entity_extraction_running_with_active_operation():
    # Add a fake active operation
    op_id = "test-op-1"
    status = EntityExtractionStatus(
        operation_id=op_id,
        document_id="doc-1",
        document_name="doc.txt",
        state=EntityExtractionState.LLM_EXTRACTION,
        started_at=time.time(),
        last_updated=time.time(),
    )

    with document_processor._operations_lock:
        document_processor._entity_extraction_operations[op_id] = status

    try:
        assert document_processor.is_entity_extraction_running() is True
    finally:
        with document_processor._operations_lock:
            document_processor._entity_extraction_operations.pop(op_id, None)


def test_is_entity_extraction_running_detects_legacy_threads(monkeypatch):
    # Create a fake legacy thread returned by threading.enumerate
    class LegacyThread:
        daemon = True

        def __init__(self):
            self.name = "entity-legacy"

        def is_alive(self):
            return True

    def fake_enumerate():
        return [threading.current_thread(), LegacyThread()]

    monkeypatch.setattr("threading.enumerate", fake_enumerate)

    # Ensure no tracked threads/ops confuse the result
    with document_processor._bg_lock:
        document_processor._bg_entity_threads.clear()
    with document_processor._operations_lock:
        document_processor._entity_extraction_operations.clear()

    assert document_processor.is_entity_extraction_running() is True
