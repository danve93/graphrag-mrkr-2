"""
Category Manager for LLM-based document categorization.

Implements automatic category generation through LLM corpus analysis with human oversight.
Categories improve retrieval precision by enabling query routing and context filtering.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

from core.llm import llm_manager
from core.graph_db import graph_db
from config.settings import settings

logger = logging.getLogger(__name__)


class CategoryManager:
    """
    Manages document categories with LLM-generated taxonomy and human approval workflow.
    
    Categories are stored in Neo4j and used for:
    - Query routing (automatic category detection from user questions)
    - Document filtering (restrict retrieval to specific categories)
    - Context organization (category-specific prompting)
    """
    
    def __init__(self):
        """Initialize category manager with graph database connection."""
        self.graph_db = graph_db
        self.llm_manager = llm_manager
    
    async def analyze_corpus_for_categories(
        self, 
        max_categories: int = 10,
        sample_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Analyze document corpus using LLM to propose category taxonomy.
        
        Args:
            max_categories: Maximum number of categories to generate
            sample_size: Number of documents to sample for analysis
            
        Returns:
            List of proposed categories with metadata:
            - name: Category name
            - description: Category description
            - keywords: List of relevant keywords
            - confidence: LLM confidence score (0-1)
            - document_examples: Sample document IDs
        """
        logger.info(f"Analyzing corpus for categories (max={max_categories}, sample={sample_size})")
        
        # Get sample of documents with their chunks
        sample_docs = await self._get_document_sample(sample_size)
        
        if not sample_docs:
            logger.warning("No documents found for category analysis")
            return []
        
        # Prepare prompt for LLM analysis
        prompt = self._build_category_generation_prompt(sample_docs, max_categories)
        
        # Get LLM response
        system_message = """You are a document taxonomy expert. Analyze the provided document samples and generate a hierarchical category taxonomy that will help users find information more efficiently.

Categories should be:
- Mutually exclusive where possible (minimal overlap)
- Collectively exhaustive (cover all document types)
- Clear and intuitive for end users
- Based on document content and purpose, not format

Return your response as valid JSON only, no additional text."""

        try:
            response = self.llm_manager.generate_response(
                prompt=prompt,
                system_message=system_message,
                temperature=0.3,  # Low temperature for consistent categorization
                max_tokens=2000,
                include_usage=True,
            )

            # Track token usage
            if isinstance(response, dict) and "usage" in response:
                try:
                    from core.llm_usage_tracker import usage_tracker
                    usage_tracker.record(
                        operation="rag.category_analysis",
                        provider=getattr(settings, "llm_provider", "openai"),
                        model=settings.openai_model,
                        input_tokens=response["usage"].get("input", 0),
                        output_tokens=response["usage"].get("output", 0),
                    )
                except Exception as track_err:
                    logger.debug(f"Token tracking failed: {track_err}")
                response = (response.get("content") or "").strip()
            else:
                response = (response or "").strip()
            
            # Parse LLM response
            categories_json = response
            
            # Handle markdown code blocks if present
            if categories_json.startswith("```"):
                categories_json = categories_json.split("```")[1]
                if categories_json.startswith("json"):
                    categories_json = categories_json[4:]
            
            proposed_categories = json.loads(categories_json)
            
            logger.info(f"LLM proposed {len(proposed_categories.get('categories', []))} categories")
            return proposed_categories.get('categories', [])
            
        except Exception as e:
            logger.error(f"Failed to generate categories from LLM: {e}")
            raise
    
    def _build_category_generation_prompt(
        self, 
        sample_docs: List[Dict[str, Any]], 
        max_categories: int
    ) -> str:
        """Build prompt for LLM category generation."""
        doc_summaries = []
        for doc in sample_docs[:20]:  # Limit to avoid token overflow
            chunks_preview = " ".join([c.get('text', '')[:200] for c in doc.get('chunks', [])[:3]])
            doc_summaries.append(f"Document: {doc['filename']}\nContent preview: {chunks_preview}...\n")
        
        corpus_summary = "\n".join(doc_summaries)
        
        return f"""Analyze the following document samples from our knowledge base and propose {max_categories} categories for organizing them.

Document Samples:
{corpus_summary}

Generate up to {max_categories} categories that would help users find information efficiently. For each category, provide:

1. name: Clear, concise category name (2-4 words)
2. description: One-sentence description of what belongs in this category
3. keywords: List of 5-10 keywords that indicate documents belong to this category
4. patterns: Text patterns or phrases commonly found in these documents
5. confidence: Your confidence this is a useful category (0.0-1.0)

Return your response as JSON in this exact format:
{{
  "categories": [
    {{
      "name": "Category Name",
      "description": "Brief description of category scope",
      "keywords": ["keyword1", "keyword2", ...],
      "patterns": ["pattern1", "pattern2", ...],
      "confidence": 0.9
    }}
  ]
}}"""
    
    async def _get_document_sample(self, sample_size: int) -> List[Dict[str, Any]]:
        """Get random sample of documents with their chunks for analysis."""
        query = """
        MATCH (d:Document)
        WITH d, rand() as r
        ORDER BY r
        LIMIT $sample_size
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        WITH d, collect({text: c.content, chunk_index: c.chunk_index})[0..3] as chunks
        RETURN d.filename as filename, d.file_path as file_path, d.id as doc_id, chunks
        """
        
        with self.graph_db.driver.session() as session:
            result = session.run(query, sample_size=sample_size)
            return [dict(record) for record in result]
    
    async def create_category(
        self,
        name: str,
        description: str,
        keywords: List[str],
        patterns: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        approved: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new category in Neo4j.
        
        Args:
            name: Category name
            description: Category description
            keywords: Keywords associated with this category
            patterns: Text patterns that indicate this category
            parent_id: Parent category ID for hierarchical structure
            approved: Whether category is approved for use (default: False, requires human approval)
            
        Returns:
            Created category with ID
        """
        # Enforce unique category names
        with self.graph_db.driver.session() as session:
            try:
                existing = session.run(
                    """
                    MATCH (c:Category)
                    WHERE toLower(c.name) = toLower($name)
                    RETURN c.id as id
                    """,
                    name=name,
                ).single()
                if existing:
                    raise ValueError("CATEGORY_NAME_CONFLICT")
            except ValueError:
                raise
            except Exception as e:
                logger.warning(f"Failed uniqueness check for category name '{name}': {e}")

        query = """
        CREATE (c:Category {
            id: randomUUID(),
            name: $name,
            description: $description,
            keywords: $keywords,
            patterns: $patterns,
            approved: $approved,
            created_at: datetime(),
            updated_at: datetime(),
            document_count: 0
        })
        WITH c
        OPTIONAL MATCH (parent:Category {id: $parent_id})
        FOREACH(_ IN CASE WHEN parent IS NOT NULL THEN [1] ELSE [] END |
            CREATE (c)-[:CHILD_OF]->(parent)
        )
        RETURN c
        """
        
        with self.graph_db.driver.session() as session:
            result = session.run(
                query,
                name=name,
                description=description,
                keywords=keywords,
                patterns=patterns or [],
                approved=approved,
                parent_id=parent_id
            )
            record = result.single()
            if not record:
                return None
            c = record["c"]
            return {
                "id": c["id"],
                "name": c.get("name"),
                "description": c.get("description"),
                "keywords": c.get("keywords", []),
                "patterns": c.get("patterns", []),
                "approved": c.get("approved", False),
                "document_count": 0,
                "children": [],
                "created_at": str(c.get("created_at")) if c.get("created_at") is not None else None,
            }

    async def update_category(self, category_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a category's editable fields.

        Allowed fields: name, description, keywords, patterns, parent_id.
        Always updates updated_at.
        """
        allowed = {k: v for k, v in fields.items() if k in {"name", "description", "keywords", "patterns", "parent_id"}}
        if not allowed:
            return None

        # If renaming, enforce uniqueness
        if "name" in allowed and allowed["name"]:
            try:
                with self.graph_db.driver.session() as session:
                    existing = session.run(
                        """
                        MATCH (c:Category)
                        WHERE toLower(c.name) = toLower($name) AND c.id <> $category_id
                        RETURN c.id as id
                        """,
                        name=allowed["name"],
                        category_id=category_id,
                    ).single()
                    if existing:
                        raise ValueError("CATEGORY_NAME_CONFLICT")
            except ValueError:
                raise
            except Exception as e:
                logger.warning(f"Failed uniqueness check for category rename '{allowed['name']}': {e}")

        query = """
        MATCH (c:Category {id: $category_id})
        SET c.updated_at = timestamp()
        """

        params: Dict[str, Any] = {"category_id": category_id}
        for key, val in allowed.items():
            query += f"\nSET c.{key} = ${key}"
            params[key] = val

        query += "\nRETURN c as category"

        try:
            with self.graph_db.driver.session() as session:
                result = session.run(query, **params)
                record = result.single()
                if not record:
                    return None
                c = record["category"]
                return {
                    "id": c["id"],
                    "name": c.get("name"),
                    "description": c.get("description"),
                    "keywords": c.get("keywords", []),
                    "patterns": c.get("patterns", []),
                    "approved": c.get("approved", False),
                    "document_count": 0,  # recalculated on read paths
                    "children": [],
                    "created_at": str(c.get("created_at")) if c.get("created_at") is not None else None,
                }
        except Exception as e:
            logger.exception(f"Failed to update category {category_id}: {e}")
            raise
    
    async def approve_category(self, category_id: str) -> bool:
        """
        Approve a proposed category for use in production.
        
        Args:
            category_id: Category ID to approve
            
        Returns:
            True if successful
        """
        query = """
        MATCH (c:Category {id: $category_id})
        SET c.approved = true, c.approved_at = datetime()
        RETURN c
        """
        
        with self.graph_db.driver.session() as session:
            result = session.run(query, category_id=category_id)
            if result.single():
                logger.info(f"Approved category: {category_id}")
                return True
            return False
    
    async def assign_document_to_category(
        self,
        document_id: str,
        category_id: str,
        confidence: float = 1.0,
        auto_assigned: bool = False
    ) -> bool:
        """
        Assign a document to a category.
        
        Args:
            document_id: Document ID
            category_id: Category ID
            confidence: Confidence score for assignment (0-1)
            auto_assigned: Whether assignment was automatic (vs manual)
            
        Returns:
            True if successful
        """
        query = """
        MATCH (d:Document {id: $document_id})
        MATCH (c:Category {id: $category_id})
        MERGE (d)-[r:BELONGS_TO]->(c)
        SET r.confidence = $confidence,
            r.auto_assigned = $auto_assigned,
            r.assigned_at = datetime()
        WITH c
        MATCH (c)<-[:BELONGS_TO]-(d:Document)
        WITH c, count(d) as doc_count
        SET c.document_count = doc_count
        RETURN c
        """
        
        with self.graph_db.driver.session() as session:
            result = session.run(
                query,
                document_id=document_id,
                category_id=category_id,
                confidence=confidence,
                auto_assigned=auto_assigned
            )
            if result.single():
                logger.info(f"Assigned document {document_id} to category {category_id}")
                return True
            return False
    
    async def get_all_categories(
        self, 
        approved_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all categories.
        
        Args:
            approved_only: If True, only return approved categories
            
        Returns:
            List of categories with metadata
        """
        where_clause = "WHERE c.approved = true" if approved_only else ""
        
        query = f"""
        MATCH (c:Category)
        {where_clause}
        OPTIONAL MATCH (c)<-[:CHILD_OF]-(child:Category)
        OPTIONAL MATCH (c)<-[:BELONGS_TO]-(d:Document)
        WITH c, collect(DISTINCT child.id) as children, count(DISTINCT d) as doc_count
        RETURN c, children, doc_count
        ORDER BY c.name
        """
        
        with self.graph_db.driver.session() as session:
            result = session.run(query)
            categories = []
            for record in result:
                cat = dict(record['c'])
                cat['children'] = record['children']
                cat['document_count'] = record['doc_count']
                categories.append(cat)
            return categories
    
    async def get_category_by_id(self, category_id: str) -> Optional[Dict[str, Any]]:
        """Get category by ID."""
        query = """
        MATCH (c:Category {id: $category_id})
        OPTIONAL MATCH (c)<-[:BELONGS_TO]-(d:Document)
        WITH c, collect(d.filename) as documents
        RETURN c, documents
        """
        
        with self.graph_db.driver.session() as session:
            result = session.run(query, category_id=category_id)
            record = result.single()
            if record:
                cat = dict(record['c'])
                cat['documents'] = record['documents']
                return cat
            return None
    
    async def delete_category(self, category_id: str) -> bool:
        """
        Delete a category and its relationships.
        
        Args:
            category_id: Category ID to delete
            
        Returns:
            True if successful
        """
        query = """
        MATCH (c:Category {id: $category_id})
        OPTIONAL MATCH (c)-[r]-()
        DELETE r, c
        """
        
        with self.graph_db.driver.session() as session:
            result = session.run(query, category_id=category_id)
            logger.info(f"Deleted category: {category_id}")
            return True
    
    async def classify_query(self, query_text: str) -> List[Tuple[str, float]]:
        """
        Classify a user query to determine relevant categories.
        
        Args:
            query_text: User's question or search query
            
        Returns:
            List of (category_id, confidence_score) tuples, sorted by confidence
        """
        # Get approved categories
        categories = await self.get_all_categories(approved_only=True)
        
        if not categories:
            logger.warning("No approved categories available for classification")
            return []
        
        # Build classification prompt
        category_descriptions = "\n".join([
            f"- {cat['name']}: {cat['description']} (Keywords: {', '.join(cat.get('keywords', [])[:5])})"
            for cat in categories
        ])
        
        prompt = f"""Classify the following user query into the most relevant categories:

Query: "{query_text}"

Available Categories:
{category_descriptions}

Return the top 3 most relevant categories with confidence scores (0.0-1.0).
Return as JSON: {{"classifications": [{{"category": "Category Name", "confidence": 0.95}}]}}"""

        system_message = "You are a query classification expert. Analyze user queries and assign them to the most relevant document categories. Return only valid JSON."
        
        try:
            response = self.llm_manager.generate_response(
                prompt=prompt,
                system_message=system_message,
                temperature=0.1,
                max_tokens=500,
                include_usage=True,
            )

            # Track token usage
            if isinstance(response, dict) and "usage" in response:
                try:
                    from core.llm_usage_tracker import usage_tracker
                    usage_tracker.record(
                        operation="rag.category_classification",
                        provider=getattr(settings, "llm_provider", "openai"),
                        model=settings.openai_model,
                        input_tokens=response["usage"].get("input", 0),
                        output_tokens=response["usage"].get("output", 0),
                    )
                except Exception as track_err:
                    logger.debug(f"Token tracking failed: {track_err}")
                response = (response.get("content") or "").strip()
            else:
                response = (response or "").strip()
            
            classifications_json = response
            
            # Handle markdown code blocks
            if classifications_json.startswith("```"):
                classifications_json = classifications_json.split("```")[1]
                if classifications_json.startswith("json"):
                    classifications_json = classifications_json[4:]
            
            classifications = json.loads(classifications_json)
            
            # Map category names to IDs
            category_map = {cat['name']: cat['id'] for cat in categories}
            
            results = []
            for item in classifications.get('classifications', []):
                cat_name = item['category']
                if cat_name in category_map:
                    results.append((category_map[cat_name], item['confidence']))
            
            logger.info(f"Classified query into {len(results)} categories")
            return sorted(results, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to classify query: {e}")
            return []
    
    async def auto_categorize_documents(self, batch_size: int = 10) -> Dict[str, int]:
        """
        Automatically categorize uncategorized documents using LLM.
        
        Args:
            batch_size: Number of documents to process in one batch
            
        Returns:
            Statistics: {categorized: int, skipped: int, failed: int}
        """
        # Get uncategorized documents
        query = """
        MATCH (d:Document)
        WHERE NOT (d)-[:BELONGS_TO]->(:Category)
        WITH d LIMIT $batch_size
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        WITH d, collect(c.content)[0..3] as chunk_texts
        RETURN d.id as doc_id, d.filename as filename, chunk_texts
        """
        
        with self.graph_db.driver.session() as session:
            result = session.run(query, batch_size=batch_size)
            uncategorized = [dict(record) for record in result]
        
        if not uncategorized:
            logger.info("No uncategorized documents found")
            return {"categorized": 0, "skipped": 0, "failed": 0}
        
        stats = {"categorized": 0, "skipped": 0, "failed": 0}
        
        for doc in uncategorized:
            try:
                # Create a pseudo-query from document content
                content_sample = " ".join(doc['chunk_texts'])[:500]
                query_text = f"Document about: {doc['filename']}. Content: {content_sample}"
                
                # Classify
                classifications = await self.classify_query(query_text)
                
                if classifications and classifications[0][1] >= 0.6:  # Confidence threshold
                    category_id, confidence = classifications[0]
                    success = await self.assign_document_to_category(
                        doc['doc_id'],
                        category_id,
                        confidence=confidence,
                        auto_assigned=True
                    )
                    if success:
                        stats["categorized"] += 1
                else:
                    stats["skipped"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to auto-categorize document {doc['doc_id']}: {e}")
                stats["failed"] += 1
        
        logger.info(f"Auto-categorization complete: {stats}")
        return stats
