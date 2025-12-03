# Marker Integration Component

High-accuracy document conversion tool integration for PDFs, presentations, and complex documents.

## Overview

Marker is a specialized document conversion tool that provides superior extraction quality for PDFs, presentations, and documents with complex layouts. It supports LLM-assisted extraction, OCR, table detection, and layout preservation. Amber integrates Marker through multiple modes: in-process, CLI, or dedicated server.

**Location**: `ingestion/converters.py`, `docs/marker/`
**Tool**: marker-pdf (external tool)
**Modes**: In-process API, CLI subprocess, HTTP server
**Output**: High-quality Markdown with preserved structure

## Architecture

```
┌──────────────────────────────────────────────────┐
│         Marker Integration Modes                  │
├──────────────────────────────────────────────────┤
│                                                   │
│  Mode 1: In-Process (Python API)                 │
│  ┌─────────────────────────────────────────────┐ │
│  │  import marker                               │ │
│  │  result = marker.convert(pdf_path)          │ │
│  │  → Fast, direct integration                 │ │
│  │  → Requires marker installation             │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Mode 2: CLI (Subprocess)                        │
│  ┌─────────────────────────────────────────────┐ │
│  │  marker_single input.pdf output_dir/        │ │
│  │  → subprocess.run()                          │ │
│  │  → Read generated markdown                   │ │
│  │  → Process isolation                         │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Mode 3: HTTP Server                             │
│  ┌─────────────────────────────────────────────┐ │
│  │  POST /convert with PDF file                │ │
│  │  → Server runs on separate port              │ │
│  │  → Returns markdown via HTTP                 │ │
│  │  → Can run on different machine/GPU         │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
└──────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

```bash
# Marker integration
ENABLE_MARKER=false              # Enable Marker integration
MARKER_MODE=cli                  # "inprocess", "cli", or "server"

# CLI mode settings
MARKER_EXECUTABLE=/path/to/marker_single
MARKER_TEMP_DIR=/tmp/marker

# Server mode settings
MARKER_SERVER_URL=http://localhost:8001

# Conversion settings
MARKER_LANGS=en                  # Language codes (comma-separated)
MARKER_DEVICE=cpu                # "cpu", "cuda", or "mps"
MARKER_BATCH_MULTIPLIER=2        # Batch size multiplier
MARKER_MAX_PAGES=null            # Max pages to convert (null = all)
MARKER_OCR_ALL_PAGES=false       # Force OCR on all pages
```

### Settings Class

```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Marker integration
    enable_marker: bool = False
    marker_mode: str = "cli"  # "inprocess", "cli", "server"
    
    # CLI settings
    marker_executable: str = "marker_single"
    marker_temp_dir: str = "/tmp/marker"
    
    # Server settings
    marker_server_url: str = "http://localhost:8001"
    
    # Conversion settings
    marker_langs: str = "en"
    marker_device: str = "cpu"
    marker_batch_multiplier: int = 2
    marker_max_pages: Optional[int] = None
    marker_ocr_all_pages: bool = False
```

## Converter Interface

### MarkerConverter Class

```python
# ingestion/converters.py
import os
import tempfile
import subprocess
import requests
from pathlib import Path
from typing import Tuple, Dict, Optional
from config.settings import settings

class MarkerConverter:
    """
    High-accuracy document conversion using Marker.
    
    Supports three modes:
        1. In-process: Direct Python API
        2. CLI: Command-line tool via subprocess
        3. Server: HTTP API to dedicated server
    """
    
    def __init__(self, mode: Optional[str] = None):
        self.mode = mode or settings.marker_mode
        
        if self.mode not in ["inprocess", "cli", "server"]:
            raise ValueError(f"Invalid Marker mode: {self.mode}")
        
        # Lazy load marker for in-process mode
        self._marker_module = None
    
    def convert(self, file_path: str) -> Tuple[str, Dict]:
        """
        Convert document to markdown.
        
        Args:
            file_path: Path to document (PDF, PPTX, etc.)
        
        Returns:
            Tuple of (markdown_text, metadata)
        """
        if not settings.enable_marker:
            raise RuntimeError("Marker integration is disabled")
        
        if self.mode == "inprocess":
            return self._convert_inprocess(file_path)
        elif self.mode == "cli":
            return self._convert_cli(file_path)
        elif self.mode == "server":
            return self._convert_server(file_path)
    
    def is_available(self) -> bool:
        """Check if Marker is available in the configured mode."""
        try:
            if self.mode == "inprocess":
                import marker
                return True
            elif self.mode == "cli":
                result = subprocess.run(
                    [settings.marker_executable, "--version"],
                    capture_output=True,
                    timeout=5
                )
                return result.returncode == 0
            elif self.mode == "server":
                response = requests.get(
                    f"{settings.marker_server_url}/health",
                    timeout=5
                )
                return response.status_code == 200
        except Exception:
            return False
        
        return False
```

## In-Process Mode

### Direct API Integration

```python
def _convert_inprocess(self, file_path: str) -> Tuple[str, Dict]:
    """Convert using Marker Python API."""
    # Lazy import
    if self._marker_module is None:
        import marker
        self._marker_module = marker
    
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
    
    # Load models (cached after first call)
    models = load_all_models(
        device=settings.marker_device,
        batch_multiplier=settings.marker_batch_multiplier
    )
    
    # Convert
    result = convert_single_pdf(
        pdf_path=file_path,
        model_dict=models,
        max_pages=settings.marker_max_pages,
        langs=[settings.marker_langs],
        ocr_all_pages=settings.marker_ocr_all_pages
    )
    
    markdown_text = result["markdown"]
    
    metadata = {
        "converter": "marker",
        "mode": "inprocess",
        "page_count": result.get("page_count", 0),
        "images_extracted": result.get("images", 0),
        "tables_detected": result.get("tables", 0)
    }
    
    return markdown_text, metadata
```

## CLI Mode

### Subprocess Execution

```python
import shutil
import json

def _convert_cli(self, file_path: str) -> Tuple[str, Dict]:
    """Convert using Marker CLI via subprocess."""
    # Create temp directory for output
    with tempfile.TemporaryDirectory(prefix="marker_") as temp_dir:
        output_dir = Path(temp_dir)
        
        # Build command
        cmd = [
            settings.marker_executable,
            file_path,
            str(output_dir),
            "--langs", settings.marker_langs,
            "--device", settings.marker_device,
            "--batch_multiplier", str(settings.marker_batch_multiplier)
        ]
        
        if settings.marker_max_pages:
            cmd.extend(["--max_pages", str(settings.marker_max_pages)])
        
        if settings.marker_ocr_all_pages:
            cmd.append("--ocr_all_pages")
        
        # Execute
        logger.info(f"Running Marker CLI: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Marker CLI failed: {result.stderr}")
        
        # Read output markdown
        filename = Path(file_path).stem
        markdown_path = output_dir / f"{filename}.md"
        
        if not markdown_path.exists():
            raise FileNotFoundError(f"Marker output not found: {markdown_path}")
        
        markdown_text = markdown_path.read_text(encoding="utf-8")
        
        # Read metadata if available
        metadata_path = output_dir / f"{filename}_meta.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                file_metadata = json.load(f)
        else:
            file_metadata = {}
        
        metadata = {
            "converter": "marker",
            "mode": "cli",
            "page_count": file_metadata.get("page_count", 0),
            **file_metadata
        }
        
        return markdown_text, metadata
```

## Server Mode

### HTTP API Client

```python
def _convert_server(self, file_path: str) -> Tuple[str, Dict]:
    """Convert using Marker HTTP server."""
    url = f"{settings.marker_server_url}/convert"
    
    # Prepare multipart form data
    with open(file_path, 'rb') as f:
        files = {'file': (Path(file_path).name, f, 'application/pdf')}
        
        data = {
            'langs': settings.marker_langs,
            'device': settings.marker_device,
            'batch_multiplier': settings.marker_batch_multiplier,
            'ocr_all_pages': str(settings.marker_ocr_all_pages).lower()
        }
        
        if settings.marker_max_pages:
            data['max_pages'] = settings.marker_max_pages
        
        # Send request
        response = requests.post(
            url,
            files=files,
            data=data,
            timeout=300  # 5 minute timeout
        )
    
    response.raise_for_status()
    
    result = response.json()
    
    markdown_text = result["markdown"]
    
    metadata = {
        "converter": "marker",
        "mode": "server",
        "page_count": result.get("page_count", 0),
        "processing_time": result.get("processing_time", 0)
    }
    
    return markdown_text, metadata
```

## Marker Server

### Simple Flask Server

```python
# scripts/marker_server.py
from flask import Flask, request, jsonify
import tempfile
from pathlib import Path
import time

app = Flask(__name__)

# Load models once at startup
from marker.convert import convert_single_pdf
from marker.models import load_all_models

models = load_all_models()

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

@app.route('/convert', methods=['POST'])
def convert():
    """Convert uploaded document to markdown."""
    start_time = time.time()
    
    # Get uploaded file
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    # Get parameters
    langs = request.form.get('langs', 'en').split(',')
    max_pages = request.form.get('max_pages', type=int)
    ocr_all_pages = request.form.get('ocr_all_pages', 'false').lower() == 'true'
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name
    
    try:
        # Convert
        result = convert_single_pdf(
            pdf_path=temp_path,
            model_dict=models,
            max_pages=max_pages,
            langs=langs,
            ocr_all_pages=ocr_all_pages
        )
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "markdown": result["markdown"],
            "page_count": result.get("page_count", 0),
            "processing_time": processing_time
        })
    
    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
```

**Start Server**:
```bash
python scripts/marker_server.py
```

## Integration with Document Processor

### Enhanced PDF Loader

```python
# ingestion/loaders/pdf_loader.py
from ingestion.converters import MarkerConverter

class PDFLoader(BaseLoader):
    """PDF loader with optional Marker integration."""
    
    def __init__(self):
        self.marker_converter = None
        
        if settings.enable_marker:
            self.marker_converter = MarkerConverter()
    
    def load(self, file_path: str) -> Tuple[str, Dict]:
        """Load PDF with optional Marker conversion."""
        # Try Marker first if enabled and available
        if self.marker_converter and self.marker_converter.is_available():
            try:
                logger.info(f"Using Marker for {file_path}")
                text, metadata = self.marker_converter.convert(file_path)
                metadata["loader"] = "marker"
                return text, metadata
            
            except Exception as e:
                logger.warning(f"Marker conversion failed: {e}, falling back to PyPDF2")
        
        # Fallback to standard extraction
        return self._extract_with_pdfplumber(file_path)
```

## Performance Comparison

### Extraction Quality

```python
def compare_extraction_methods(pdf_path: str):
    """Compare extraction quality between methods."""
    from ingestion.loaders.pdf_loader import PDFLoader
    from ingestion.converters import MarkerConverter
    
    # Standard extraction
    loader = PDFLoader()
    standard_text, standard_meta = loader._extract_with_pdfplumber(pdf_path)
    
    # Marker extraction
    marker = MarkerConverter(mode="cli")
    marker_text, marker_meta = marker.convert(pdf_path)
    
    print("Standard Extraction:")
    print(f"  Length: {len(standard_text)} chars")
    print(f"  Quality: {standard_meta.get('extraction_quality', 'N/A')}")
    print()
    
    print("Marker Extraction:")
    print(f"  Length: {len(marker_text)} chars")
    print(f"  Tables: {marker_meta.get('tables_detected', 'N/A')}")
    print(f"  Images: {marker_meta.get('images_extracted', 'N/A')}")
```

**Typical Results**:
- **Standard (pdfplumber)**: Fast, good for text-heavy PDFs
- **Marker**: Slower but superior for complex layouts, tables, images

## Installation

### Marker Installation

```bash
# Install marker-pdf
pip install marker-pdf

# GPU support (optional, faster)
pip install marker-pdf[gpu]

# Install system dependencies
# For OCR support
sudo apt-get install tesseract-ocr

# For better PDF handling
sudo apt-get install poppler-utils
```

### CLI Tool

```bash
# Install CLI tool
pip install marker-pdf

# Verify installation
marker_single --version

# Test conversion
marker_single input.pdf output_dir/
```

## Usage Examples

### Basic Conversion

```python
from ingestion.converters import MarkerConverter

converter = MarkerConverter(mode="cli")

# Convert PDF
markdown, metadata = converter.convert("document.pdf")

print(f"Converted {metadata['page_count']} pages")
print(markdown[:500])  # Preview
```

### Batch Conversion

```python
from pathlib import Path

def convert_directory(input_dir: str, output_dir: str):
    """Convert all PDFs in directory."""
    converter = MarkerConverter()
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for pdf_file in input_path.glob("*.pdf"):
        try:
            markdown, metadata = converter.convert(str(pdf_file))
            
            # Save markdown
            output_file = output_path / f"{pdf_file.stem}.md"
            output_file.write_text(markdown)
            
            print(f"{pdf_file.name} → {output_file.name}")
        
        except Exception as e:
            print(f"Error - {pdf_file.name}: {e}")
```

## Testing

### Unit Tests

```python
import pytest
from ingestion.converters import MarkerConverter

@pytest.fixture
def converter():
    return MarkerConverter(mode="cli")

@pytest.mark.skipif(
    not MarkerConverter().is_available(),
    reason="Marker not available"
)
def test_marker_conversion(converter):
    markdown, metadata = converter.convert("tests/fixtures/sample.pdf")
    
    assert markdown
    assert metadata["converter"] == "marker"
    assert metadata["page_count"] > 0

def test_marker_availability():
    converter = MarkerConverter()
    is_available = converter.is_available()
    
    # Should not raise, just return boolean
    assert isinstance(is_available, bool)
```

## Troubleshooting

### Common Issues

**Issue**: `marker_single: command not found`
```bash
# Solution: Install marker-pdf
pip install marker-pdf

# Or specify full path
MARKER_EXECUTABLE=/path/to/venv/bin/marker_single
```

**Issue**: Out of memory errors
```python
# Solution: Reduce batch multiplier
MARKER_BATCH_MULTIPLIER=1

# Or limit pages
MARKER_MAX_PAGES=50
```

**Issue**: Server connection refused
```bash
# Solution: Start marker server
python scripts/marker_server.py

# Verify it's running
curl http://localhost:8001/health
```

**Issue**: Poor quality extraction
```python
# Solution: Enable OCR on all pages
MARKER_OCR_ALL_PAGES=true

# Or use GPU for better models
MARKER_DEVICE=cuda
```

## Related Documentation

- [Document Loaders](03-components/ingestion/loaders.md)
- [Document Processor](03-components/ingestion/document-processor.md)
- [Quality Scoring](03-components/ingestion/quality-scoring.md)
- [Marker Documentation](marker)
