
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from config.settings import settings
from core.graph_db import graph_db

def repair_chap8():
    doc_id = 'bb374326496b924f7eb9cc37d121187e'
    print(f"Repairing {doc_id} (Chap8)...")
    
    try:
        graph_db.ensure_connected()
        
        # 1. Repair Relationships
        print("Running repair_contains_entity_relationships_for_document...")
        repair_stats = graph_db.repair_contains_entity_relationships_for_document(doc_id)
        print(f"Repair stats: {repair_stats}")
        
        # 2. Update Stats
        print("Updating precomputed stats...")
        stats = graph_db.update_document_precomputed_summary(doc_id)
        print(f"Stats: {stats}")
        
        # 3. Update Preview
        graph_db.update_document_preview(doc_id)
        print("Preview updated.")
        
        if repair_stats['created'] > 0 or repair_stats['after'] > 0:
            print("SUCCESS: Relationships repaired/confirmed.")
        else:
            print("WARNING: No relationships created. Investigation needed.")

    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    repair_chap8()
