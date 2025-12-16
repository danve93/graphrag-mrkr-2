"""
Precompute embeddings for static entities (categories, entity types).

This script generates vector embeddings for static taxonomies that change infrequently,
allowing for fast in-memory vector search without database queries.

Usage:
    uv run python scripts/precompute_static_embeddings.py

Output:
    config/static_embeddings.json.gz - Compressed embeddings file
"""

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.embeddings import embedding_manager
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_categories() -> Dict[str, Dict[str, Any]]:
    """Load category configuration from JSON file."""
    categories_path = project_root / "config" / "document_categories.json"

    if not categories_path.exists():
        raise FileNotFoundError(f"Categories config not found: {categories_path}")

    with open(categories_path, 'r') as f:
        config = json.load(f)

    categories = config.get('categories', {})
    logger.info(f"Loaded {len(categories)} categories from {categories_path}")

    return categories


def build_category_text(category_id: str, category_data: Dict[str, Any]) -> str:
    """
    Build searchable text from category metadata.

    Combines title, description, and keywords into a single text representation
    optimized for embedding generation.

    Args:
        category_id: Category identifier (e.g., "install", "admin")
        category_data: Category metadata (title, description, keywords)

    Returns:
        Combined text for embedding
    """
    parts = [
        category_data.get('title', category_id),
        category_data.get('description', ''),
    ]

    # Add keywords as comma-separated list
    keywords = category_data.get('keywords', [])
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")

    # Join with newlines for better semantic separation
    text = '\n'.join(filter(None, parts))

    logger.debug(f"Category '{category_id}' text ({len(text)} chars): {text[:100]}...")

    return text


async def generate_embeddings_async(categories: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate embeddings for all categories asynchronously.

    Args:
        categories: Dictionary of category_id -> category_data

    Returns:
        Dictionary with embeddings data:
        {
            "version": "1.0",
            "model": "text-embedding-3-small",
            "dimension": 1536,
            "categories": [
                {
                    "id": "install",
                    "title": "Installation",
                    "text": "...",
                    "embedding": [0.1, 0.2, ...]
                }
            ]
        }
    """
    logger.info("Generating embeddings for categories...")

    embeddings_data = {
        "version": "1.0",
        "model": settings.embedding_model,
        "dimension": None,  # Will be set after first embedding
        "categories": []
    }

    for category_id, category_data in categories.items():
        logger.info(f"Processing category: {category_id}")

        # Build category text
        text = build_category_text(category_id, category_data)

        # Generate embedding
        try:
            embedding = await embedding_manager.aget_embedding(text)

            # Set dimension from first embedding
            if embeddings_data["dimension"] is None:
                embeddings_data["dimension"] = len(embedding)
                logger.info(f"Embedding dimension: {embeddings_data['dimension']}")

            # Add to results
            embeddings_data["categories"].append({
                "id": category_id,
                "title": category_data.get('title', category_id),
                "description": category_data.get('description', ''),
                "keywords": category_data.get('keywords', []),
                "text": text,
                "embedding": embedding
            })

            logger.info(f"✓ Generated embedding for '{category_id}' ({len(embedding)} dimensions)")

        except Exception as e:
            logger.error(f"✗ Failed to generate embedding for '{category_id}': {e}")
            raise

    logger.info(f"Successfully generated {len(embeddings_data['categories'])} embeddings")

    return embeddings_data


def generate_embeddings(categories: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Synchronous wrapper for embedding generation."""
    import asyncio
    return asyncio.run(generate_embeddings_async(categories))


def save_embeddings(embeddings_data: Dict[str, Any], output_path: Path, compress: bool = True):
    """
    Save embeddings to file with optional gzip compression.

    Args:
        embeddings_data: Embeddings data dictionary
        output_path: Path to save file
        compress: Whether to gzip compress the output
    """
    if compress and not str(output_path).endswith('.gz'):
        output_path = Path(str(output_path) + '.gz')

    # Convert to JSON
    json_str = json.dumps(embeddings_data, indent=2)
    json_bytes = json_str.encode('utf-8')

    # Calculate sizes
    uncompressed_size = len(json_bytes)

    if compress:
        # Write compressed
        with gzip.open(output_path, 'wb') as f:
            f.write(json_bytes)
        compressed_size = output_path.stat().st_size
        compression_ratio = (1 - compressed_size / uncompressed_size) * 100

        logger.info(
            f"Saved compressed embeddings: {output_path} "
            f"({compressed_size / 1024:.1f}KB, {compression_ratio:.1f}% compression)"
        )
    else:
        # Write uncompressed
        with open(output_path, 'w') as f:
            f.write(json_str)

        logger.info(f"Saved embeddings: {output_path} ({uncompressed_size / 1024:.1f}KB)")


def verify_embeddings(embeddings_path: Path):
    """
    Verify that embeddings file can be loaded correctly.

    Args:
        embeddings_path: Path to embeddings file
    """
    logger.info(f"Verifying embeddings file: {embeddings_path}")

    try:
        if str(embeddings_path).endswith('.gz'):
            with gzip.open(embeddings_path, 'rb') as f:
                data = json.loads(f.read().decode('utf-8'))
        else:
            with open(embeddings_path, 'r') as f:
                data = json.load(f)

        # Verify structure
        assert 'version' in data, "Missing 'version' field"
        assert 'model' in data, "Missing 'model' field"
        assert 'dimension' in data, "Missing 'dimension' field"
        assert 'categories' in data, "Missing 'categories' field"
        assert len(data['categories']) > 0, "No categories in embeddings"

        # Verify each category
        for cat in data['categories']:
            assert 'id' in cat, "Category missing 'id' field"
            assert 'embedding' in cat, "Category missing 'embedding' field"
            assert len(cat['embedding']) == data['dimension'], \
                f"Category '{cat['id']}' embedding dimension mismatch"

        logger.info(
            f"✓ Verification passed: {len(data['categories'])} categories, "
            f"{data['dimension']} dimensions, model: {data['model']}"
        )

        return True

    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        return False


def main():
    """Main script entry point."""
    parser = argparse.ArgumentParser(
        description='Precompute embeddings for static entities (categories, types)'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=Path,
        default=project_root / 'config' / 'static_embeddings.json.gz',
        help='Output path for embeddings file (default: config/static_embeddings.json.gz)'
    )
    parser.add_argument(
        '--no-compress',
        action='store_true',
        help='Disable gzip compression'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing embeddings file'
    )

    args = parser.parse_args()

    try:
        if args.verify_only:
            # Verify mode
            if not args.output.exists():
                logger.error(f"Embeddings file not found: {args.output}")
                return 1

            success = verify_embeddings(args.output)
            return 0 if success else 1

        # Generation mode
        logger.info("=== Static Embeddings Precomputation ===")
        logger.info(f"Embedding model: {settings.embedding_model}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Compression: {'disabled' if args.no_compress else 'enabled'}")

        # Load categories
        categories = load_categories()

        # Generate embeddings
        embeddings_data = generate_embeddings(categories)

        # Save to file
        save_embeddings(
            embeddings_data,
            args.output,
            compress=not args.no_compress
        )

        # Verify output
        verify_embeddings(args.output)

        logger.info("=== Precomputation Complete ===")
        logger.info(f"Generated embeddings for {len(embeddings_data['categories'])} categories")
        logger.info(f"Embedding dimension: {embeddings_data['dimension']}")
        logger.info(f"Output file: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Precomputation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
