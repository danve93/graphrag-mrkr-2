#!/usr/bin/env python3
"""
Fix documentation links to use absolute paths from documentation root.
"""

import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "documentation"

def fix_links_in_file(file_path: Path):
    """Fix relative links in a markdown file to use absolute paths."""
    content = file_path.read_text(encoding='utf-8')
    original_content = content
    
    # Get the directory of the current file relative to DOCS_DIR
    relative_dir = file_path.parent.relative_to(DOCS_DIR)
    
    # Find all markdown links: [text](./path.md) or [text](../path.md)
    def replace_link(match):
        link_text = match.group(1)
        link_path = match.group(2)
        
        # Skip external links
        if link_path.startswith('http://') or link_path.startswith('https://'):
            return match.group(0)
        
        # Skip anchors
        if link_path.startswith('#'):
            return match.group(0)
        
        # Convert relative path to absolute
        if link_path.startswith('./') or link_path.startswith('../'):
            # Resolve the path
            absolute_path = (file_path.parent / link_path).resolve()
            # Get path relative to DOCS_DIR
            try:
                new_path = absolute_path.relative_to(DOCS_DIR)
                return f'[{link_text}]({new_path})'
            except ValueError:
                # Path is outside DOCS_DIR, keep as is
                return match.group(0)
        
        # Already absolute or needs no change
        return match.group(0)
    
    # Replace all links
    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_link, content)
    
    # Write back if changed
    if content != original_content:
        file_path.write_text(content, encoding='utf-8')
        return True
    return False

def main():
    """Fix all documentation files."""
    fixed_count = 0
    
    for md_file in DOCS_DIR.rglob("*.md"):
        if fix_links_in_file(md_file):
            print(f"Fixed: {md_file.relative_to(DOCS_DIR)}")
            fixed_count += 1
    
    print(f"\nTotal files fixed: {fixed_count}")

if __name__ == "__main__":
    main()
