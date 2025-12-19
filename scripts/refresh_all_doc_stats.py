
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from config.settings import settings
from core.graph_db import graph_db

def refresh_stats():
    print("Connecting to Neo4j...")
    try:
        graph_db.ensure_connected()
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return

    print("Fetching all document IDs...")
    with graph_db.driver.session() as session:
        result = session.run("MATCH (d:Document) RETURN d.id as id, d.filename as filename")
        docs = [record.data() for record in result]
    
    print(f"Found {len(docs)} documents.")
    
    for doc in docs:
        doc_id = doc['id']
        filename = doc['filename']
        print(f"Updating stats for {filename} ({doc_id})...")
        try:
            stats = graph_db.update_document_precomputed_summary(doc_id)
            print(f"  -> Stats: {stats}")
            
            # also update preview
            try:
                graph_db.update_document_preview(doc_id)
                print("  -> Preview updated.")
            except Exception as ex:
                print(f"  -> Preview update skipped: {ex}")
                
        except Exception as e:
            print(f"  -> FAILED: {e}")

    print("\nRefresh complete.")

if __name__ == "__main__":
    refresh_stats()
