# Document Loaders Component

Format-specific loaders for extracting text and metadata from various document types.

## Overview

The loaders component provides specialized extractors for different document formats. Each loader implements a common interface and handles format-specific quirks, metadata extraction, and optional OCR integration. Loaders convert documents to normalized text/Markdown suitable for chunking and embedding.

**Location**: `ingestion/loaders/`
**Supported Formats**: PDF, DOCX, TXT, MD, PPTX, XLSX, CSV, images
**Output**: Text content + metadata dict

## Architecture

```
┌──────────────────────────────────────────────────┐
│            Loader Interface                       │
├──────────────────────────────────────────────────┤
│                                                   │
│  BaseLoader (Abstract)                           │
│  ┌─────────────────────────────────────────────┐ │
│  │  def load(file_path) → (text, metadata)    │ │
│  │  def detect_encoding(file_path) → str      │ │
│  │  def extract_metadata(file_path) → dict    │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Format-Specific Loaders                         │
│  ┌─────────────────────────────────────────────┐ │
│  │  PDFLoader        → PyPDF2, pdfplumber      │ │
│  │  DOCXLoader       → python-docx             │ │
│  │  TextLoader       → Plain text              │ │
│  │  MarkdownLoader   → Markdown                │ │
│  │  PPTXLoader       → python-pptx             │ │
│  │  ExcelLoader      → openpyxl                │ │
│  │  CSVLoader        → pandas                  │ │
│  │  ImageLoader      → pytesseract (OCR)       │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
└──────────────────────────────────────────────────┘
```

## Base Loader Interface

### Abstract Base Class

```python
from abc import ABC, abstractmethod
from typing import Tuple, Dict

class BaseLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: str) -> Tuple[str, Dict]:
        """
        Load document and extract content.
        
        Args:
            file_path: Path to document file
        
        Returns:
            Tuple of (text_content, metadata_dict)
        """
        pass
    
    def extract_metadata(self, file_path: str) -> Dict:
        """
        Extract document metadata.
        
        Returns:
            Metadata dict with keys like:
                - file_type: MIME type
                - page_count: Number of pages
                - title: Document title
                - author: Document author
                - created_date: Creation timestamp
        """
        import os
        
        return {
            "file_type": self._get_mime_type(file_path),
            "file_size": os.path.getsize(file_path),
            "filename": os.path.basename(file_path)
        }
    
    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type from file extension."""
        import mimetypes
        
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"
```

## PDF Loader

### Implementation

```python
# ingestion/loaders/pdf_loader.py
import PyPDF2
import pdfplumber
from typing import Tuple, Dict

class PDFLoader(BaseLoader):
    """
    PDF document loader with multiple extraction strategies.
    
    Strategies:
        1. PyPDF2 (fast, basic text)
        2. pdfplumber (tables, layout)
        3. OCR fallback (image-based PDFs)
    """
    
    def load(self, file_path: str) -> Tuple[str, Dict]:
        """Load PDF and extract text."""
        metadata = self.extract_metadata(file_path)
        
        # Try pdfplumber first (better quality)
        try:
            text = self._extract_with_pdfplumber(file_path, metadata)
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying PyPDF2")
            text = self._extract_with_pypdf2(file_path, metadata)
        
        # Check if extraction quality is low (might need OCR)
        if self._is_low_quality(text):
            metadata["extraction_quality"] = 0.3
        else:
            metadata["extraction_quality"] = 1.0
        
        return text, metadata
    
    def _extract_with_pdfplumber(self, file_path: str, metadata: Dict) -> str:
        """Extract text using pdfplumber (preserves layout)."""
        with pdfplumber.open(file_path) as pdf:
            metadata["page_count"] = len(pdf.pages)
            
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                
                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    table_text = self._format_table(table)
                    page_text += f"\n\n{table_text}\n\n"
                
                text_parts.append(page_text)
            
            return "\n\n".join(text_parts)
    
    def _extract_with_pypdf2(self, file_path: str, metadata: Dict) -> str:
        """Extract text using PyPDF2 (fallback)."""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            metadata["page_count"] = len(reader.pages)
            
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text())
            
            return "\n\n".join(text_parts)
    
    def _format_table(self, table: list) -> str:
        """Convert table to markdown."""
        if not table:
            return ""
        
        # Build markdown table
        lines = []
        for row in table:
            line = " | ".join(str(cell or "") for cell in row)
            lines.append(f"| {line} |")
            
            # Add separator after header
            if len(lines) == 1:
                separator = " | ".join(["---"] * len(row))
                lines.append(f"| {separator} |")
        
        return "\n".join(lines)
    
    def _is_low_quality(self, text: str) -> bool:
        """Check if extracted text quality is low."""
        if not text or len(text.strip()) < 50:
            return True
        
        # Check for excessive non-alphanumeric characters
        alphanum_ratio = sum(c.isalnum() for c in text) / len(text)
        return alphanum_ratio < 0.5
    
    def extract_metadata(self, file_path: str) -> Dict:
        """Extract PDF metadata."""
        metadata = super().extract_metadata(file_path)
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                info = reader.metadata
                
                if info:
                    metadata.update({
                        "title": info.get("/Title", ""),
                        "author": info.get("/Author", ""),
                        "subject": info.get("/Subject", ""),
                        "creator": info.get("/Creator", ""),
                        "created_date": info.get("/CreationDate", "")
                    })
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")
        
        return metadata
```

## DOCX Loader

### Implementation

```python
# ingestion/loaders/docx_loader.py
from docx import Document
from typing import Tuple, Dict

class DOCXLoader(BaseLoader):
    """Microsoft Word document loader."""
    
    def load(self, file_path: str) -> Tuple[str, Dict]:
        """Load DOCX and extract text."""
        doc = Document(file_path)
        
        # Extract paragraphs
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        
        # Extract tables
        table_texts = []
        for table in doc.tables:
            table_text = self._extract_table(table)
            table_texts.append(table_text)
        
        # Combine
        text = "\n\n".join(paragraphs)
        if table_texts:
            text += "\n\n" + "\n\n".join(table_texts)
        
        # Metadata
        metadata = self.extract_metadata(file_path)
        metadata["page_count"] = len(doc.sections)
        
        return text, metadata
    
    def _extract_table(self, table) -> str:
        """Convert table to markdown."""
        lines = []
        
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip() for cell in row.cells]
            line = " | ".join(cells)
            lines.append(f"| {line} |")
            
            # Add separator after header
            if i == 0:
                separator = " | ".join(["---"] * len(cells))
                lines.append(f"| {separator} |")
        
        return "\n".join(lines)
    
    def extract_metadata(self, file_path: str) -> Dict:
        """Extract DOCX metadata."""
        metadata = super().extract_metadata(file_path)
        
        try:
            doc = Document(file_path)
            core_props = doc.core_properties
            
            metadata.update({
                "title": core_props.title or "",
                "author": core_props.author or "",
                "created_date": core_props.created.isoformat() if core_props.created else ""
            })
        except Exception as e:
            logger.warning(f"Failed to extract DOCX metadata: {e}")
        
        return metadata
```

## Text Loader

### Implementation

```python
# ingestion/loaders/text_loader.py
import chardet
from typing import Tuple, Dict

class TextLoader(BaseLoader):
    """Plain text file loader with encoding detection."""
    
    def load(self, file_path: str) -> Tuple[str, Dict]:
        """Load text file with auto-encoding detection."""
        encoding = self._detect_encoding(file_path)
        
        with open(file_path, 'r', encoding=encoding) as f:
            text = f.read()
        
        metadata = self.extract_metadata(file_path)
        metadata["encoding"] = encoding
        
        return text, metadata
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding."""
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Sample first 10KB
        
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        
        # Fallback to UTF-8
        if not encoding or result['confidence'] < 0.5:
            encoding = 'utf-8'
        
        return encoding
```

## Markdown Loader

### Implementation

```python
# ingestion/loaders/markdown_loader.py
from typing import Tuple, Dict

class MarkdownLoader(TextLoader):
    """
    Markdown file loader.
    
    Inherits from TextLoader but preserves markdown formatting.
    """
    
    def load(self, file_path: str) -> Tuple[str, Dict]:
        """Load markdown preserving formatting."""
        text, metadata = super().load(file_path)
        
        # Extract title from first heading
        lines = text.split('\n')
        for line in lines:
            if line.startswith('# '):
                metadata["title"] = line[2:].strip()
                break
        
        return text, metadata
```

## PowerPoint Loader

### Implementation

```python
# ingestion/loaders/pptx_loader.py
from pptx import Presentation
from typing import Tuple, Dict

class PPTXLoader(BaseLoader):
    """PowerPoint presentation loader."""
    
    def load(self, file_path: str) -> Tuple[str, Dict]:
        """Load PPTX and extract text from slides."""
        prs = Presentation(file_path)
        
        slide_texts = []
        for i, slide in enumerate(prs.slides):
            slide_text = self._extract_slide_text(slide, i + 1)
            if slide_text:
                slide_texts.append(slide_text)
        
        text = "\n\n---\n\n".join(slide_texts)
        
        metadata = self.extract_metadata(file_path)
        metadata["page_count"] = len(prs.slides)
        
        return text, metadata
    
    def _extract_slide_text(self, slide, slide_number: int) -> str:
        """Extract text from a single slide."""
        parts = [f"Slide {slide_number}"]
        
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                parts.append(shape.text.strip())
        
        return "\n\n".join(parts)
```

## Excel Loader

### Implementation

```python
# ingestion/loaders/excel_loader.py
from openpyxl import load_workbook
from typing import Tuple, Dict

class ExcelLoader(BaseLoader):
    """Excel spreadsheet loader."""
    
    def load(self, file_path: str) -> Tuple[str, Dict]:
        """Load Excel and convert sheets to markdown tables."""
        wb = load_workbook(file_path, data_only=True)
        
        sheet_texts = []
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            sheet_text = self._extract_sheet(sheet, sheet_name)
            if sheet_text:
                sheet_texts.append(sheet_text)
        
        text = "\n\n---\n\n".join(sheet_texts)
        
        metadata = self.extract_metadata(file_path)
        metadata["page_count"] = len(wb.sheetnames)
        
        return text, metadata
    
    def _extract_sheet(self, sheet, sheet_name: str) -> str:
        """Convert sheet to markdown table."""
        parts = [f"## {sheet_name}"]
        
        rows = []
        for row in sheet.iter_rows(values_only=True):
            # Skip empty rows
            if any(cell for cell in row):
                rows.append(row)
        
        if not rows:
            return ""
        
        # Build markdown table
        table_lines = []
        for i, row in enumerate(rows):
            cells = [str(cell or "") for cell in row]
            line = " | ".join(cells)
            table_lines.append(f"| {line} |")
            
            # Add separator after header
            if i == 0:
                separator = " | ".join(["---"] * len(cells))
                table_lines.append(f"| {separator} |")
        
        parts.append("\n".join(table_lines))
        
        return "\n\n".join(parts)
```

## CSV Loader

### Implementation

```python
# ingestion/loaders/csv_loader.py
import pandas as pd
from typing import Tuple, Dict

class CSVLoader(BaseLoader):
    """CSV file loader using pandas."""
    
    def load(self, file_path: str) -> Tuple[str, Dict]:
        """Load CSV and convert to markdown table."""
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode CSV with any encoding")
        
        # Convert to markdown
        text = df.to_markdown(index=False)
        
        metadata = self.extract_metadata(file_path)
        metadata["row_count"] = len(df)
        metadata["column_count"] = len(df.columns)
        
        return text, metadata
```

## Image Loader (OCR)

### Implementation

```python
# ingestion/loaders/image_loader.py
from PIL import Image
import pytesseract
from typing import Tuple, Dict

class ImageLoader(BaseLoader):
    """
    Image file loader with OCR.
    
    Requires: tesseract-ocr installed
    """
    
    def load(self, file_path: str) -> Tuple[str, Dict]:
        """Load image and extract text via OCR."""
        try:
            image = Image.open(file_path)
            
            # Run OCR
            text = pytesseract.image_to_string(image)
            
            # Get image dimensions
            width, height = image.size
            
            metadata = self.extract_metadata(file_path)
            metadata.update({
                "width": width,
                "height": height,
                "extraction_method": "tesseract_ocr"
            })
            
            return text, metadata
        
        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {e}")
            return "", self.extract_metadata(file_path)
```

## Loader Registry

### Dynamic Loader Selection

```python
# ingestion/loaders/__init__.py
from pathlib import Path
from typing import Type
from .base_loader import BaseLoader
from .pdf_loader import PDFLoader
from .docx_loader import DOCXLoader
from .text_loader import TextLoader
from .markdown_loader import MarkdownLoader
from .pptx_loader import PPTXLoader
from .excel_loader import ExcelLoader
from .csv_loader import CSVLoader
from .image_loader import ImageLoader

LOADERS: Dict[str, Type[BaseLoader]] = {
    ".pdf": PDFLoader,
    ".docx": DOCXLoader,
    ".doc": DOCXLoader,
    ".txt": TextLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".pptx": PPTXLoader,
    ".ppt": PPTXLoader,
    ".xlsx": ExcelLoader,
    ".xls": ExcelLoader,
    ".csv": CSVLoader,
    ".png": ImageLoader,
    ".jpg": ImageLoader,
    ".jpeg": ImageLoader,
    ".gif": ImageLoader,
    ".bmp": ImageLoader,
    ".tiff": ImageLoader,
}

def get_loader(file_path: str) -> BaseLoader:
    """
    Get appropriate loader for file type.
    
    Args:
        file_path: Path to file
    
    Returns:
        Loader instance
    
    Raises:
        ValueError: If file type is not supported
    """
    extension = Path(file_path).suffix.lower()
    
    loader_class = LOADERS.get(extension)
    if not loader_class:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported: {', '.join(LOADERS.keys())}"
        )
    
    return loader_class()

def list_supported_formats() -> list:
    """Get list of supported file formats."""
    return sorted(LOADERS.keys())
```

## Usage Examples

### Load Single Document

```python
from ingestion.loaders import get_loader

def load_document(file_path: str):
    """Load document using appropriate loader."""
    loader = get_loader(file_path)
    text, metadata = loader.load(file_path)
    
    print(f"Loaded: {metadata['filename']}")
    print(f"Type: {metadata['file_type']}")
    print(f"Length: {len(text)} characters")
    
    return text, metadata
```

### Batch Loading

```python
from pathlib import Path

def load_directory(directory: str):
    """Load all supported documents in directory."""
    results = []
    
    for file_path in Path(directory).rglob("*"):
        if not file_path.is_file():
            continue
        
        try:
            loader = get_loader(str(file_path))
            text, metadata = loader.load(str(file_path))
            
            results.append({
                "file_path": str(file_path),
                "text": text,
                "metadata": metadata
            })
            
            print(f"{file_path.name}")
        
        except ValueError:
            # Unsupported format
            continue
        except Exception as e:
            print(f"Error - {file_path.name}: {e}")
    
    return results
```

## Testing

### Unit Tests

```python
import pytest
from ingestion.loaders import get_loader, PDFLoader, DOCXLoader

def test_get_loader_pdf():
    loader = get_loader("test.pdf")
    assert isinstance(loader, PDFLoader)

def test_get_loader_docx():
    loader = get_loader("test.docx")
    assert isinstance(loader, DOCXLoader)

def test_get_loader_unsupported():
    with pytest.raises(ValueError):
        get_loader("test.xyz")

@pytest.mark.integration
def test_pdf_loader():
    loader = PDFLoader()
    text, metadata = loader.load("tests/fixtures/sample.pdf")
    
    assert text
    assert metadata["file_type"] == "application/pdf"
    assert metadata["page_count"] > 0

@pytest.mark.integration
def test_docx_loader():
    loader = DOCXLoader()
    text, metadata = loader.load("tests/fixtures/sample.docx")
    
    assert text
    assert "title" in metadata
```

## Dependencies

### Required Packages

```bash
# PDF
pip install PyPDF2 pdfplumber

# Office documents
pip install python-docx python-pptx openpyxl

# Data formats
pip install pandas

# Text processing
pip install chardet

# OCR (optional)
pip install pytesseract pillow
# System: sudo apt-get install tesseract-ocr

# Markdown rendering (optional)
pip install tabulate  # For pandas.to_markdown()
```

## Related Documentation

- [Document Processor](03-components/ingestion/document-processor.md)
- [Chunking Strategy](03-components/ingestion/chunking.md)
- [Marker Integration](03-components/ingestion/marker-integration.md)
- [OCR Processing](02-core-concepts/ocr-processing.md)
