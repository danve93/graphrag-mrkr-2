
import os
import sys
import traceback

# Add project root to path
sys.path.insert(0, os.getcwd())

from config.settings import settings
from core.graph_db import graph_db

def debug_details():
    doc_id = 'bb374326496b924f7eb9cc37d121187e'
    print(f"Calling get_document_details for {doc_id}...")
    
    try:
        graph_db.ensure_connected()
        details = graph_db.get_document_details(doc_id)
        print("Success!")
        print(details)
    except Exception as e:
        print("FAILED!")
        traceback.print_exc()

if __name__ == "__main__":
    debug_details()
