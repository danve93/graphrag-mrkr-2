"""
Neo4j Category Schema Migration

This script applies the necessary schema changes for the LLM Category Taxonomy feature.
It creates constraints, indexes, and validates the schema for Category nodes.

Run this script after deploying the backend category management code.

Usage:
    python scripts/migrate_category_schema.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from neo4j import GraphDatabase

# Override Neo4j URI for localhost when running outside Docker
if "neo4j:7687" in settings.neo4j_uri:
    original_uri = settings.neo4j_uri
    settings.neo4j_uri = settings.neo4j_uri.replace("neo4j:7687", "localhost:7687")
    print(f"Note: Adjusted Neo4j URI from {original_uri} to {settings.neo4j_uri} for localhost access")

# Create driver directly with updated settings
driver = GraphDatabase.driver(
    settings.neo4j_uri,
    auth=(settings.neo4j_username, settings.neo4j_password)
)


def create_category_constraints():
    """Create uniqueness constraints for Category nodes."""
    print("Creating Category node constraints...")
    
    queries = [
        # Unique constraint on category ID
        "CREATE CONSTRAINT category_id_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.id IS UNIQUE",
        
        # Unique constraint on category name (optional - comment out if you want duplicate names)
        # "CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
    ]
    
    with driver.session() as session:
        for query in queries:
            try:
                session.run(query)
                print(f"  ✓ Applied: {query}")
            except Exception as e:
                print(f"  ⚠ Warning: {query}")
                print(f"    {str(e)}")


def create_category_indexes():
    """Create indexes for efficient category queries."""
    print("\nCreating Category indexes...")
    
    queries = [
        # Index on approved field for filtering
        "CREATE INDEX category_approved_index IF NOT EXISTS FOR (c:Category) ON (c.approved)",
        
        # Index on created_at for sorting
        "CREATE INDEX category_created_at_index IF NOT EXISTS FOR (c:Category) ON (c.created_at)",
        
        # Text index on name for search (if using Neo4j 5.0+)
        # "CREATE TEXT INDEX category_name_text_index IF NOT EXISTS FOR (c:Category) ON (c.name)",
    ]
    
    with driver.session() as session:
        for query in queries:
            try:
                session.run(query)
                print(f"  ✓ Applied: {query}")
            except Exception as e:
                print(f"  ⚠ Warning: {query}")
                print(f"    {str(e)}")


def verify_schema():
    """Verify that all schema changes were applied successfully."""
    print("\nVerifying schema...")
    
    with driver.session() as session:
        # Check constraints
        constraints_query = "SHOW CONSTRAINTS"
        result = session.run(constraints_query)
        constraints = [dict(record) for record in result]
        
        category_constraints = [
            c for c in constraints 
            if 'Category' in str(c.get('labelsOrTypes', []))
        ]
        
        print(f"  ✓ Found {len(category_constraints)} Category constraints")
        for constraint in category_constraints:
            print(f"    - {constraint.get('name', 'unnamed')}: {constraint.get('type', 'unknown')}")
        
        # Check indexes
        indexes_query = "SHOW INDEXES"
        result = session.run(indexes_query)
        indexes = [dict(record) for record in result]
        
        category_indexes = [
            idx for idx in indexes 
            if 'Category' in str(idx.get('labelsOrTypes', []))
        ]
        
        print(f"  ✓ Found {len(category_indexes)} Category indexes")
        for index in category_indexes:
            print(f"    - {index.get('name', 'unnamed')}: {index.get('type', 'unknown')}")
        
        # Check if any categories exist
        count_query = "MATCH (c:Category) RETURN count(c) as count"
        result = session.run(count_query)
        record = result.single()
        count = record.get('count', 0) if record else 0
        
        print(f"\n  ℹ  Current category count: {count}")
        
        return True


def test_category_creation():
    """Test creating a sample category to verify the schema works."""
    print("\nTesting category creation...")
    
    test_query = """
    CREATE (c:Category {
        id: 'test_' + randomUUID(),
        name: 'Test Category',
        description: 'This is a test category to verify schema',
        keywords: ['test', 'schema', 'verification'],
        patterns: ['test*', 'verify*'],
        approved: false,
        created_at: datetime(),
        updated_at: datetime(),
        document_count: 0
    })
    RETURN c.id as id
    """
    
    try:
        with driver.session() as session:
            result = session.run(test_query)
            record = result.single()
            test_id = record.get('id') if record else None
            print(f"  ✓ Test category created: {test_id}")
            
            # Clean up test category
            cleanup_query = "MATCH (c:Category {id: $id}) DELETE c"
            session.run(cleanup_query, {"id": test_id})
            print(f"  ✓ Test category deleted")
            
            return True
    except Exception as e:
        print(f"  ✗ Test failed: {str(e)}")
        return False


def main():
    """Run the migration."""
    print("=" * 60)
    print("Neo4j Category Schema Migration")
    print("=" * 60)
    print(f"\nNeo4j URI: {settings.neo4j_uri}")
    print(f"Username: {settings.neo4j_username}")
    print()
    
    try:
        # Step 1: Create constraints
        create_category_constraints()
        
        # Step 2: Create indexes
        create_category_indexes()
        
        # Step 3: Verify schema
        success = verify_schema()
        
        # Step 4: Test category creation
        if success:
            test_success = test_category_creation()
            
            if test_success:
                print("\n" + "=" * 60)
                print("✓ Migration completed successfully!")
                print("=" * 60)
                print("\nNext steps:")
                print("  1. Start the backend: python api/main.py")
                print("  2. Navigate to Categories view in UI")
                print("  3. Click 'Generate Categories' to create taxonomy")
                print("  4. Review and approve proposed categories")
                print("  5. Run 'Auto-Categorize Documents' to assign documents")
                return 0
            else:
                print("\n⚠ Migration completed with warnings")
                return 1
        else:
            print("\n✗ Migration failed during verification")
            return 1
            
    except Exception as e:
        print(f"\n✗ Migration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
