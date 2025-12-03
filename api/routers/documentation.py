"""
Documentation router - serves documentation files.
"""

import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documentation", tags=["documentation"])

# Documentation directory
DOCS_DIR = Path(__file__).parent.parent.parent / "documentation"


@router.get("/{file_path:path}", response_class=PlainTextResponse)
async def get_documentation_file(file_path: str):
    """
    Serve a documentation file.
    
    Args:
        file_path: Relative path to the documentation file (with or without .md extension)
        
    Returns:
        File content as plain text
    """
    try:
        # Add .md extension if not present
        if not file_path.endswith('.md'):
            file_path = f"{file_path}.md"
        
        # Resolve the full path
        full_path = (DOCS_DIR / file_path).resolve()
        
        # Security check: ensure the path is within the documentation directory
        if not str(full_path).startswith(str(DOCS_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if file exists
        if not full_path.exists():
            raise HTTPException(status_code=404, detail=f"Documentation file not found: {file_path}")
        
        # Check if it's a file (not a directory)
        if not full_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")
        
        # Read and return the file content
        content = full_path.read_text(encoding="utf-8")
        return content
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving documentation file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Error loading documentation file")


@router.get("/")
async def list_documentation():
    """
    List available documentation files.
    
    Returns:
        Dictionary with documentation structure
    """
    try:
        def scan_directory(path: Path, base_path: Path):
            """Recursively scan directory and build file tree."""
            items = []
            for item in sorted(path.iterdir()):
                if item.name.startswith('.'):
                    continue
                
                relative_path = str(item.relative_to(base_path))
                
                if item.is_dir():
                    items.append({
                        "name": item.name,
                        "path": relative_path,
                        "type": "directory",
                        "children": scan_directory(item, base_path)
                    })
                elif item.suffix in ['.md', '.txt']:
                    items.append({
                        "name": item.name,
                        "path": relative_path,
                        "type": "file"
                    })
            
            return items
        
        if not DOCS_DIR.exists():
            return {"error": "Documentation directory not found"}
        
        structure = scan_directory(DOCS_DIR, DOCS_DIR)
        
        return {
            "documentation": structure
        }
        
    except Exception as e:
        logger.error(f"Error listing documentation: {e}")
        raise HTTPException(status_code=500, detail="Error listing documentation")
