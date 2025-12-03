#!/usr/bin/env python3
"""
Fix broken links in README files to point to existing files.
"""

import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "documentation"

# Mapping of broken links to correct ones
LINK_MAPPINGS = {
    # 04-features fixes
    "04-features/entity-clustering.md": "04-features/community-detection.md",
    "04-features/reranking.md": "05-data-flows/reranking-flow.md",
    "04-features/document-classification.md": "04-features/document-upload.md",  # Closest match
    "04-features/gleaning.md": "04-features/entity-reasoning.md",  # Closest match
    "04-features/response-caching.md": "02-core-concepts/caching-system.md",
    
    # 05-data-flows fixes
    "05-data-flows/ingestion-flow.md": "05-data-flows/document-ingestion-flow.md",
    "05-data-flows/retrieval-flow.md": "05-data-flows/graph-expansion-flow.md",
    "05-data-flows/entity-deduplication.md": "05-data-flows/entity-extraction-flow.md",
    "05-data-flows/streaming-sse.md": "05-data-flows/streaming-sse-flow.md",
    
    # 06-api-reference fixes
    "06-api-reference/chat.md": "06-api-reference/chat-api.md",
    
    # 08-operations fixes  
    "08-operations/performance-tuning.md": "08-operations/monitoring.md",
    "08-operations/backup-restore.md": "08-operations/maintenance-reindexing.md",
    "08-operations/scaling.md": "08-operations/deployment.md",
    
    # 09-development fixes
    "09-development/testing.md": "09-development/testing-backend.md",
    "09-development/code-conventions.md": "09-development/coding-standards.md",
    "09-development/debugging.md": "09-development/dev-scripts.md",
    "09-development/adding-features.md": "09-development/feature-flag-wiring.md",
    
    # 10-scripts fixes
    "10-scripts/maintenance-scripts.md": "10-scripts/setup-neo4j.md",
    "04-features/entity-clustering.md": "04-features/community-detection.md",
}

def fix_readme(readme_path: Path):
    """Fix links in a README file."""
    content = readme_path.read_text(encoding='utf-8')
    original_content = content
    
    # Replace each broken link
    for old_link, new_link in LINK_MAPPINGS.items():
        # Match the link in markdown format
        pattern = re.escape(f"]({old_link})")
        replacement = f"]({new_link})"
        content = re.sub(pattern, replacement, content)
    
    # Write back if changed
    if content != original_content:
        readme_path.write_text(content, encoding='utf-8')
        return True
    return False

def main():
    """Fix all README files."""
    fixed_count = 0
    
    for readme in DOCS_DIR.rglob("README.md"):
        if fix_readme(readme):
            print(f"Fixed: {readme.relative_to(DOCS_DIR)}")
            fixed_count += 1
    
    print(f"\nTotal README files fixed: {fixed_count}")

if __name__ == "__main__":
    main()
