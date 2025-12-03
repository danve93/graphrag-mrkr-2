"""
API endpoints for structured knowledge graph queries.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from rag.nodes.structured_kg_executor import get_structured_kg_executor
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["structured-kg"])


class StructuredQueryRequest(BaseModel):
    """Request for structured KG query execution."""
    query: str = Field(..., description="Natural language query")
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context (conversation history, etc.)"
    )


class StructuredQueryResponse(BaseModel):
    """Response from structured KG query execution."""
    success: bool
    results: list = Field(default_factory=list)
    cypher: Optional[str] = None
    query_type: Optional[str] = None
    entities: list = Field(default_factory=list)
    corrections: int = 0
    duration_ms: int
    error: Optional[str] = None
    fallback_recommended: bool = False


class CypherValidationRequest(BaseModel):
    """Request for Cypher query validation."""
    cypher: str = Field(..., description="Cypher query to validate")


@router.post("/execute", response_model=StructuredQueryResponse)
async def execute_structured_query(request: StructuredQueryRequest) -> StructuredQueryResponse:
    """
    Execute a structured knowledge graph query via Text-to-Cypher.
    
    Args:
        request: Query request with natural language query and optional context
        
    Returns:
        Query results, generated Cypher, and execution metadata
    """
    try:
        if not settings.enable_structured_kg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Structured KG queries are disabled"
            )
        
        executor = get_structured_kg_executor()
        result = await executor.execute_query(
            query=request.query,
            context=request.context
        )
        
        return StructuredQueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Structured query execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )


@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """
    Get structured KG configuration.
    
    Returns:
        Current configuration settings
    """
    return {
        "enabled": settings.enable_structured_kg,
        "entity_threshold": settings.structured_kg_entity_threshold,
        "max_corrections": settings.structured_kg_max_corrections,
        "timeout_ms": settings.structured_kg_timeout,
        "supported_query_types": settings.structured_kg_query_types
    }


@router.get("/schema")
async def get_schema() -> Dict[str, Any]:
    """
    Get graph database schema information for query construction.
    
    Returns:
        Schema description with node labels and relationship types
    """
    try:
        from core.graph_db import graph_db
        
        with graph_db.driver.session() as session:
            # Get node labels
            labels_result = session.run("CALL db.labels()")
            node_labels = [record[0] for record in labels_result]
            
            # Get relationship types
            rels_result = session.run("CALL db.relationshipTypes()")
            relationship_types = [record[0] for record in rels_result]
            
            # Get sample node properties for each label
            node_properties = {}
            for label in node_labels[:10]:  # Limit to 10 labels
                props_result = session.run(
                    f"MATCH (n:{label}) RETURN keys(n) AS props LIMIT 1"
                )
                record = props_result.single()
                if record:
                    node_properties[label] = record["props"]
        
        return {
            "node_labels": node_labels,
            "relationship_types": relationship_types,
            "node_properties": node_properties,
            "description": {
                "Document": "Document nodes with title, filename, metadata",
                "Chunk": "Text chunks with content, embeddings, and provenance",
                "Entity": "Extracted entities with names, labels, descriptions",
                "Category": "Document categories for classification"
            },
            "relationships": {
                "CONTAINS": "Document contains chunks",
                "MENTIONS": "Chunk mentions entity",
                "RELATED_TO": "Entity related to entity (with strength)",
                "SIMILAR_TO": "Chunk similar to chunk (with similarity score)",
                "BELONGS_TO": "Document belongs to category"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve schema: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Schema retrieval failed: {str(e)}"
        )


@router.post("/validate")
async def validate_cypher(request: CypherValidationRequest) -> Dict[str, Any]:
    """
    Validate a Cypher query without executing it.
    
    Args:
        request: Validation request containing Cypher query
        
    Returns:
        Validation result with success status and error message if invalid
    """
    try:
        from core.graph_db import graph_db
        
        # Use EXPLAIN to validate without executing
        with graph_db.driver.session() as session:
            session.run(f"EXPLAIN {request.cypher}")
        
        return {
            "valid": True,
            "cypher": request.cypher
        }
        
    except Exception as e:
        return {
            "valid": False,
            "cypher": request.cypher,
            "error": str(e)
        }
