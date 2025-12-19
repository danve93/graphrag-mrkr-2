
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from config.settings import settings
from core.graph_db import graph_db

def debug_chap8():
    doc_id = 'bb374326496b924f7eb9cc37d121187e'
    print(f"Debugging {doc_id} (Chap8)...")
    
    with graph_db.driver.session() as session:
        # 1. Get Entities
        print("Checking entities...")
        result = session.run(
            "MATCH (e:Entity) WHERE any(x IN e.source_chunks WHERE x STARTS WITH $doc_id) RETURN e.id, e.name, e.source_chunks",
            doc_id=doc_id
        )
        entities = [record.data() for record in result]
        print(f"Found {len(entities)} entities claiming chunks from this doc.")
        
        # 2. Get Chunks
        print("Checking chunks...")
        result = session.run(
            "MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk) RETURN c.id",
            doc_id=doc_id
        )
        chunks = [record["c.id"] for record in result]
        print(f"Found {len(chunks)} chunks linked to document.")
        chunk_set = set(chunks)
        
        # 3. Compare
        missing_links = 0
        for e in entities:
            print(f"Entity: {e['e.name']} ({e['e.id']})")
            sources = e['e.source_chunks']
            print(f"  Source chunks ({len(sources)}):")
            for sc in sources:
                if sc.startswith(doc_id):
                    exists = sc in chunk_set
                    status = "EXISTS" if exists else "MISSING!"
                    print(f"    - {sc} [{status}]")
                    if exists:
                        # Check existences of relationship
                        rel_check = session.run(
                            "MATCH (c:Chunk {id: $cid})-[r:CONTAINS_ENTITY]->(e:Entity {id: $eid}) RETURN count(r)",
                            cid=sc, eid=e['e.id']
                        ).single()['count(r)']
                        print(f"      -> Relationship exists? {rel_check > 0}")
                        if rel_check == 0:
                            missing_links += 1
                else:
                    print(f"    - {sc} (from other doc?)")

    print(f"Total missing relationships found: {missing_links}")

if __name__ == "__main__":
    debug_chap8()
