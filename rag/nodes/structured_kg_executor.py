"""
Structured Knowledge Graph Query Executor

This module provides Text-to-Cypher translation for structured graph queries.
It enables direct graph database queries for specific relationship queries,
aggregations, and complex graph patterns that benefit from structured execution.

Key features:
- Natural language to Cypher translation
- Entity linking with confidence scoring
- Iterative query correction (max 2 iterations)
- Coreference resolution for conversational queries
- Query validation and error recovery

Query types suitable for structured path:
- Relationship queries: "What does X connect to?"
- Aggregations: "How many documents mention X?"
- Path queries: "What's the relationship between X and Y?"
- Comparison queries: "Which entities are related to both X and Y?"
- Hierarchical queries: "Show me the hierarchy of X"

Usage:
    executor = get_structured_kg_executor()
    result = await executor.execute_query(
        query="How many documents mention Neo4j?",
        context={"conversation_history": [...]}
    )
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from config.settings import settings
from core.llm import llm_manager
from core.graph_db import graph_db
from core.embeddings import embedding_manager

logger = logging.getLogger(__name__)


class StructuredKGExecutor:
    """Executes structured graph queries via Text-to-Cypher translation."""
    
    def __init__(self):
        """Initialize the structured KG executor."""
        self.max_correction_attempts = getattr(settings, 'structured_kg_max_corrections', 2)
        self.entity_linking_threshold = getattr(settings, 'structured_kg_entity_threshold', 0.85)
        self.query_timeout = getattr(settings, 'structured_kg_timeout', 5000)  # milliseconds
        
    async def execute_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a structured graph query via Text-to-Cypher translation.
        
        Args:
            query: Natural language query
            context: Optional context (conversation history, entities, etc.)
            
        Returns:
            Dictionary with results, cypher, metadata
        """
        start_time = time.time()
        context = context or {}
        
        try:
            # Step 1: Detect query type and check if suitable for structured path
            query_type = self._detect_query_type(query)
            if not self._is_suitable_for_structured(query_type):
                return {
                    "success": False,
                    "error": "Query not suitable for structured path",
                    "query_type": query_type,
                    "fallback_recommended": True,
                    "duration_ms": int((time.time() - start_time) * 1000)
                }
            
            # Step 2: Entity linking - find entities mentioned in query
            entities = await self._link_entities(query, context)
            
            # Step 3: Generate Cypher query
            cypher_result = await self._generate_cypher(query, entities, query_type, context)
            
            if not cypher_result["success"]:
                return {
                    "success": False,
                    "error": cypher_result.get("error", "Cypher generation failed"),
                    "query_type": query_type,
                    "entities": entities,
                    "duration_ms": int((time.time() - start_time) * 1000)
                }
            
            cypher = cypher_result["cypher"]
            
            # Step 4: Execute with iterative correction
            execution_result = await self._execute_with_correction(
                cypher=cypher,
                query=query,
                entities=entities,
                query_type=query_type,
                context=context
            )
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            return {
                "success": execution_result["success"],
                "results": execution_result.get("results", []),
                "cypher": execution_result.get("final_cypher", cypher),
                "query_type": query_type,
                "entities": entities,
                "corrections": execution_result.get("corrections", 0),
                "duration_ms": duration_ms,
                "error": execution_result.get("error")
            }
            
        except Exception as e:
            logger.error(f"Structured KG execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_recommended": True,
                "duration_ms": int((time.time() - start_time) * 1000)
            }
    
    def _detect_query_type(self, query: str) -> str:
        """
        Detect the type of graph query from natural language.
        
        Args:
            query: Natural language query
            
        Returns:
            Query type: relationship, aggregation, path, comparison, hierarchical, general
        """
        query_lower = query.lower()
        
        # Aggregation patterns
        if any(word in query_lower for word in ['how many', 'count', 'total', 'number of', 'sum of']):
            return 'aggregation'
        
        # Path/relationship patterns
        if any(word in query_lower for word in ['relationship between', 'connect', 'path from', 'link between', 'relates to']):
            return 'path'
        
        # Comparison patterns
        if any(word in query_lower for word in ['both', 'compare', 'versus', 'vs', 'difference between', 'similar to']):
            return 'comparison'
        
        # Hierarchical patterns
        if any(word in query_lower for word in ['hierarchy', 'parent', 'child', 'descendants', 'ancestors', 'tree']):
            return 'hierarchical'
        
        # Direct relationship patterns
        if any(word in query_lower for word in ['what does', 'which are', 'related to', 'associated with', 'mentions', 'mention']):
            return 'relationship'
        
        # Document search patterns (find/search documents)
        if any(pattern in query_lower for pattern in ['find document', 'search document', 'documents that', 'docs that', 'which document']):
            return 'relationship'
        
        return 'general'
    
    def _is_suitable_for_structured(self, query_type: str) -> bool:
        """
        Check if query type is suitable for structured KG path.
        
        Args:
            query_type: Detected query type
            
        Returns:
            True if suitable for structured execution
        """
        suitable_types = ['aggregation', 'path', 'comparison', 'hierarchical', 'relationship']
        return query_type in suitable_types
    
    async def _link_entities(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Link entities mentioned in the query to graph nodes.
        
        Args:
            query: Natural language query
            context: Optional context with conversation history
            
        Returns:
            List of linked entities with confidence scores
        """
        try:
            # Extract potential entity mentions using LLM
            extraction_prompt = f"""Extract entity names mentioned in this query. Return only the entity names, one per line.

Query: {query}

Entity names:"""
            
            response = llm_manager.generate_response(
                prompt=extraction_prompt,
                system_message="You are an entity extraction system. Extract only explicitly mentioned entity names.",
                temperature=0.1,
                max_tokens=200
            )
            
            # Parse entity names
            entity_names = [
                line.strip().strip('-').strip('*').strip()
                for line in response.strip().split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]
            
            # Link each entity to graph nodes using semantic search
            linked_entities = []
            for entity_name in entity_names[:5]:  # Limit to 5 entities
                # Get entity embedding
                entity_embedding = await embedding_manager.get_embedding(entity_name)
                
                # Search for matching entities in graph
                with graph_db.driver.session() as session:
                    result = session.run("""
                        MATCH (e:Entity)
                        WHERE e.embedding IS NOT NULL
                        RETURN e.name AS name, e.id AS id, e.label AS label, e.embedding AS embedding
                        LIMIT 100
                    """)
                    
                    candidates = []
                    for record in result:
                        if record["embedding"]:
                            # Calculate cosine similarity
                            from numpy import dot
                            from numpy.linalg import norm
                            
                            candidate_embedding = record["embedding"]
                            similarity = dot(entity_embedding, candidate_embedding) / (
                                norm(entity_embedding) * norm(candidate_embedding)
                            )
                            
                            if similarity >= self.entity_linking_threshold:
                                candidates.append({
                                    "name": record["name"],
                                    "id": record["id"],
                                    "label": record["label"],
                                    "confidence": float(similarity),
                                    "query_mention": entity_name
                                })
                    
                    # Take best match
                    if candidates:
                        best_match = max(candidates, key=lambda x: x["confidence"])
                        linked_entities.append(best_match)
                        logger.info(f"Linked '{entity_name}' -> '{best_match['name']}' (confidence: {best_match['confidence']:.2f})")
            
            return linked_entities
            
        except Exception as e:
            logger.error(f"Entity linking failed: {e}")
            return []
    
    async def _generate_cypher(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        query_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate Cypher query from natural language.
        
        Args:
            query: Natural language query
            entities: Linked entities
            query_type: Detected query type
            context: Optional context
            
        Returns:
            Dictionary with cypher query and success status
        """
        try:
            # Build entity context
            entity_context = ""
            if entities:
                entity_context = "\n\nLinked entities:\n"
                for e in entities:
                    entity_context += f"- {e['query_mention']} → {e['name']} (ID: {e['id']}, Label: {e['label']}, Confidence: {e['confidence']:.2f})\n"
            
            # Build schema context
            schema_info = """
Graph schema:
- Document nodes: (d:Document {id, title, filename})
- Chunk nodes: (c:Chunk {id, content, chunk_index, document_id})
- Entity nodes: (e:Entity {id, name, label, description})
- Category nodes: (cat:Category {id, name, description})

Relationships:
- (d)-[:CONTAINS]->(c) - Document contains chunks
- (c)-[:MENTIONS]->(e) - Chunk mentions entity
- (e)-[:RELATED_TO {strength}]->(e) - Entities related
- (c)-[:SIMILAR_TO {similarity}]->(c) - Similar chunks
- (d)-[:BELONGS_TO]->(cat) - Document belongs to category
"""
            
            # Generate Cypher
            cypher_prompt = f"""Generate a Cypher query for Neo4j to answer this question.

{schema_info}
{entity_context}

Query type: {query_type}
Question: {query}

Requirements:
1. Use only the schema above
2. Reference linked entities by their IDs when available
3. Return results with clear aliases
4. Limit results to 50 unless aggregating
5. Handle missing data gracefully (OPTIONAL MATCH when appropriate)

Generate ONLY the Cypher query, no explanations:"""
            
            response = llm_manager.generate_response(
                prompt=cypher_prompt,
                system_message="You are a Cypher query generator for Neo4j. Generate only valid Cypher syntax.",
                temperature=0.1,
                max_tokens=500
            )
            
            # Extract and clean Cypher
            cypher = self._extract_cypher(response)
            
            if not cypher:
                return {"success": False, "error": "Could not extract valid Cypher from response"}
            
            logger.info(f"Generated Cypher:\n{cypher}")
            
            return {"success": True, "cypher": cypher}
            
        except Exception as e:
            logger.error(f"Cypher generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_cypher(self, response: str) -> Optional[str]:
        """
        Extract Cypher query from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Cleaned Cypher query or None
        """
        # Remove markdown code blocks
        response = re.sub(r'```cypher\n?', '', response)
        response = re.sub(r'```\n?', '', response)
        
        # Remove comments
        lines = []
        for line in response.split('\n'):
            # Remove line comments
            if '//' in line:
                line = line[:line.index('//')]
            if line.strip() and not line.strip().startswith('#'):
                lines.append(line)
        
        cypher = '\n'.join(lines).strip()
        
        # Basic validation
        if not cypher or not any(keyword in cypher.upper() for keyword in ['MATCH', 'RETURN', 'WITH', 'CREATE']):
            return None
        
        return cypher
    
    async def _execute_with_correction(
        self,
        cypher: str,
        query: str,
        entities: List[Dict[str, Any]],
        query_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Cypher with iterative error correction.
        
        Args:
            cypher: Initial Cypher query
            query: Original natural language query
            entities: Linked entities
            query_type: Query type
            context: Optional context
            
        Returns:
            Dictionary with results and correction metadata
        """
        corrections = 0
        current_cypher = cypher
        
        for attempt in range(self.max_correction_attempts + 1):
            try:
                # Execute query
                with graph_db.driver.session() as session:
                    result = session.run(current_cypher)
                    records = [dict(record) for record in result]
                
                logger.info(f"Query succeeded on attempt {attempt + 1}, returned {len(records)} results")
                
                return {
                    "success": True,
                    "results": records,
                    "final_cypher": current_cypher,
                    "corrections": corrections
                }
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Query failed on attempt {attempt + 1}: {error_msg}")
                
                # If we've exhausted corrections, return error
                if attempt >= self.max_correction_attempts:
                    return {
                        "success": False,
                        "error": error_msg,
                        "final_cypher": current_cypher,
                        "corrections": corrections
                    }
                
                # Try to correct the query
                correction_result = await self._correct_cypher(
                    cypher=current_cypher,
                    error=error_msg,
                    query=query,
                    entities=entities,
                    query_type=query_type
                )
                
                if not correction_result["success"]:
                    return {
                        "success": False,
                        "error": f"Correction failed: {error_msg}",
                        "final_cypher": current_cypher,
                        "corrections": corrections
                    }
                
                current_cypher = correction_result["corrected_cypher"]
                corrections += 1
                logger.info(f"Attempting correction {corrections}: {current_cypher}")
        
        return {
            "success": False,
            "error": "Max correction attempts exceeded",
            "final_cypher": current_cypher,
            "corrections": corrections
        }
    
    async def _correct_cypher(
        self,
        cypher: str,
        error: str,
        query: str,
        entities: List[Dict[str, Any]],
        query_type: str
    ) -> Dict[str, Any]:
        """
        Attempt to correct a failed Cypher query.
        
        Args:
            cypher: Failed Cypher query
            error: Error message
            query: Original natural language query
            entities: Linked entities
            query_type: Query type
            
        Returns:
            Dictionary with corrected cypher or error
        """
        try:
            correction_prompt = f"""The following Cypher query failed with an error. Fix the query.

Original question: {query}
Query type: {query_type}

Failed Cypher:
{cypher}

Error:
{error}

Generate a corrected Cypher query that fixes the error. Return ONLY the corrected Cypher:"""
            
            response = llm_manager.generate_response(
                prompt=correction_prompt,
                system_message="You are a Cypher query debugger. Fix syntax errors and logic issues in Cypher queries.",
                temperature=0.1,
                max_tokens=500
            )
            
            corrected = self._extract_cypher(response)
            
            if not corrected:
                return {"success": False, "error": "Could not extract corrected Cypher"}
            
            return {"success": True, "corrected_cypher": corrected}
            
        except Exception as e:
            logger.error(f"Cypher correction failed: {e}")
            return {"success": False, "error": str(e)}


# Singleton instance
_executor: Optional[StructuredKGExecutor] = None


def get_structured_kg_executor() -> StructuredKGExecutor:
    """Get or create singleton StructuredKGExecutor instance."""
    global _executor
    if _executor is None:
        _executor = StructuredKGExecutor()
    return _executor
