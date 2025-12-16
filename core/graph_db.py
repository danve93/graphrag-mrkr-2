"""
Neo4j graph database operations for the RAG pipeline.
"""

import asyncio
import logging
import math
import mimetypes
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import ServiceUnavailable, DriverError
from contextlib import contextmanager
import threading

from config.settings import settings
from core.embeddings import embedding_manager
from core.singletons import (
    get_graph_db_driver,
    get_entity_label_cache,
    get_blocking_executor,
    SHUTTING_DOWN,
)
import json

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Entity node in the graph."""

    id: str
    name: str
    type: str
    description: str = ""
    importance_score: float = 0.5
    embedding: Optional[List[float]] = None


@dataclass
class Relationship:
    """Relationship between entities."""

    source_entity_id: str
    target_entity_id: str
    type: str
    description: str = ""
    strength: float = 0.5
    source_chunks: List[str] = field(default_factory=list)
    source_text_units: List[str] = field(default_factory=list)


@dataclass
class PathResult:
    """Result of a multi-hop path traversal."""

    entities: List[Entity]
    relationships: List[Relationship]
    score: float
    supporting_chunk_ids: List[List[str]]  # List of chunk ids per hop


class GraphDB:
    """Neo4j database manager for document storage and retrieval."""

    def __init__(self):
        """Initialize GraphDB with singleton driver and caching."""
        # Initialize caching infrastructure
        self._entity_label_cache = get_entity_label_cache()
        self._entity_label_lock = threading.Lock()
        
        # Use singleton driver for connection pooling
        self.driver: Optional[Driver] = None
        try:
            # Prefer the singleton driver (may perform connectivity verification)
            self.driver = get_graph_db_driver()
        except Exception as e:
            logger.warning(f"get_graph_db_driver failed: {e}. Attempting direct connect with retries.")
            # Try a direct connect with the GraphDB.connect retry logic
            try:
                self.connect()
            except Exception as e2:
                logger.warning(f"Direct connect attempt failed: {e2}. Creating lazy driver instance without immediate verify.")
                # As a final fallback, create a driver object without forcing verify_connectivity
                try:
                    self.driver = GraphDatabase.driver(
                        settings.neo4j_uri,
                        auth=(settings.neo4j_username, settings.neo4j_password),
                        max_connection_pool_size=settings.neo4j_max_connection_pool_size,
                    )
                    logger.info("Created lazy Neo4j driver instance (connectivity not yet verified).")
                except Exception as e3:
                    logger.warning(f"Failed to create lazy Neo4j driver: {e3}")
                    self.driver = None

    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        # Attempt connecting with retries (exponential backoff) because
        # Neo4j service may take longer to become available when starting
        # via Docker Compose. This reduces spurious failures at startup.
        from urllib.parse import urlparse

        uri = settings.neo4j_uri
        username = settings.neo4j_username
        password = settings.neo4j_password

        max_attempts = 8
        delay = 1.0
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                logger.debug("Attempting to connect to Neo4j (attempt %s) at %s", attempt, uri)
                self.driver = GraphDatabase.driver(uri, auth=(username, password))
                # Test the connection
                self.driver.verify_connectivity()
                logger.info("Successfully connected to Neo4j database at %s", uri)
                return
            except Exception as e:
                last_exc = e
                logger.warning("Neo4j connection attempt %s failed: %s", attempt, e)

                # If we see an incomplete TLS/handshake error when talking to Bolt,
                # try a fallback scheme that trusts the server certificate. This
                # helps local Docker setups where Neo4j exposes Bolt with a
                # self-signed certificate and the client needs to use
                # `neo4j+ssc://` to accept it.
                try:
                    msg = str(e).lower()
                    if "handshake" in msg or "incomplete handshake" in msg or "tls" in msg or "ssl" in msg:
                        if uri.startswith("bolt://") or uri.startswith("neo4j://"):
                            alt_uri = uri
                            if uri.startswith("bolt://"):
                                alt_uri = "neo4j+ssc://" + uri.split("//", 1)[1]
                            elif uri.startswith("neo4j://"):
                                alt_uri = "neo4j+ssc://" + uri.split("//", 1)[1]
                            logger.info("Attempting fallback Neo4j URI using neo4j+ssc:// -> %s", alt_uri)
                            try:
                                self.driver = GraphDatabase.driver(alt_uri, auth=(username, password))
                                self.driver.verify_connectivity()
                                settings.neo4j_uri = alt_uri
                                logger.info("Successfully connected to Neo4j using fallback URI %s", alt_uri)
                                return
                            except Exception as ee:
                                logger.warning("Fallback neo4j+ssc connection failed: %s", ee)
                except Exception:
                    # Swallow any errors from fallback logic and continue retry/backoff
                    pass

                # Exponential backoff before retrying
                import time

                time.sleep(delay)
                delay = min(delay * 2, 8.0)

        # If we reach here, all attempts failed
        logger.error("Failed to connect to Neo4j after %s attempts: %s", max_attempts, last_exc)
        raise last_exc

    def ensure_connected(self) -> None:
        """Ensure there is an active Neo4j connection, connecting if necessary.

        This helper is safe to call from request handlers and will raise the
        underlying exception if a connection cannot be established so the
        caller can handle it and return a proper HTTP error rather than
        allowing an implicit fallback to another host.
        """
        # If we don't have a driver at all, attempt to create/connect one.
        if self.driver is None:
            try:
                # Try getting the process-wide singleton driver first (may verify connectivity).
                from core.singletons import get_graph_db_driver

                try:
                    self.driver = get_graph_db_driver()
                    return
                except Exception:
                    # Fall back to direct connect if singleton helper fails
                    pass

                self.connect()
            except Exception as e:
                logger.error("ensure_connected: unable to establish Neo4j connection: %s", e)
                raise

        # If we have a driver object but it might have been closed externally
        # (for example, cleanup_singletons() closed the process-global driver),
        # verify connectivity and recreate the driver when necessary.
        else:
            try:
                # Some test helpers and mocks provide a driver-like object that
                # implements `session()` but may not implement
                # `verify_connectivity()`. In that case, attempt a lightweight
                # validation by opening a session (and closing it) rather than
                # forcing a full connectivity check which would attempt a real
                # network call.
                if hasattr(self.driver, "verify_connectivity") and callable(
                    getattr(self.driver, "verify_connectivity")
                ):
                    # verify_connectivity will raise if the driver is closed or unreachable
                    try:
                        self.driver.verify_connectivity()
                        return
                    except Exception:
                        # Attempt to re-acquire a healthy singleton driver
                        from core.singletons import get_graph_db_driver

                        try:
                            self.driver = get_graph_db_driver()
                            return
                        except Exception:
                            # As a last resort, clear driver and attempt direct connect
                            try:
                                if self.driver:
                                    try:
                                        self.driver.close()
                                    except Exception:
                                        pass
                            finally:
                                self.driver = None
                            self.connect()
                else:
                    # No verify_connectivity method available; try opening a session
                    # to validate that the driver-like object is usable (works for
                    # test fakes that implement `session()`).
                    try:
                        sess = self.driver.session()
                        try:
                            # If session is a context manager, close via .close();
                            # some fakes return simple objects so be defensive.
                            try:
                                sess.close()
                            except Exception:
                                pass
                        finally:
                            pass
                        return
                    except Exception:
                        # Fall through to attempt to reconnect using real driver
                        try:
                            if self.driver:
                                try:
                                    self.driver.close()
                                except Exception:
                                    pass
                        finally:
                            self.driver = None
                        self.connect()
            except Exception as e:
                logger.error("ensure_connected: driver appears unusable and reconnect failed: %s", e)
                raise

    @contextmanager
    def session_scope(self, max_attempts: int = 3, initial_backoff: float = 0.5):
        """Context manager to provide a session with transient reconnect retries.

        Usage:
            with graph_db.session_scope() as session:
                session.run(...)

        This will attempt to ensure a connection, create a session, and on
        transient failures (ServiceUnavailable) will retry establishing a new
        connection up to `max_attempts` times with exponential backoff.
        """
        attempts = 0
        backoff = initial_backoff
        while True:
            attempts += 1
            try:
                # Ensure the driver is available (may raise)
                self.ensure_connected()

                session = self.driver.session()  # type: ignore
                try:
                    yield session
                    return
                finally:
                    try:
                        session.close()
                    except Exception:
                        pass

            except (ServiceUnavailable, DriverError) as e:
                logger.warning(
                    "session_scope: ServiceUnavailable on attempt %s/%s: %s",
                    attempts,
                    max_attempts,
                    e,
                )
                # Close driver and attempt fresh connect on next iteration
                try:
                    if self.driver:
                        try:
                            self.driver.close()
                        except Exception:
                            pass
                        self.driver = None
                except Exception:
                    pass

                if attempts >= max_attempts:
                    logger.exception("session_scope: exceeded max attempts (%s)", max_attempts)
                    raise

                import time

                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
                continue
            except Exception:
                # Non-transient exception - re-raise
                raise

    def close(self) -> None:
        """Close database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def get_entity_label_cached(self, entity_id: str) -> str:
        """
        Get entity label with TTL caching.
        
        Args:
            entity_id: Entity ID to look up
        
        Returns:
            str: Entity name/label
        
        Cache Strategy:
            - Hit: Return cached value (5-minute freshness)
            - Miss: Query Neo4j, cache result, return
        """
        # Check if caching is enabled
        if not settings.enable_caching:
            return self._get_entity_label_direct(entity_id)
        
        # Check cache (thread-safe read)
        # Check cache (thread-safe read)
        # CacheService handles locking internally for diskcache, but we have a lock here anyway.
        cached_label = self._entity_label_cache.get(entity_id)
        if cached_label is not None:
            logger.debug(f"Entity label cache HIT: {entity_id}")
            return cached_label
        
        # Cache miss - query database
        logger.debug(f"Entity label cache MISS: {entity_id}")
        label = self._get_entity_label_direct(entity_id)
        
        # Store in cache (thread-safe write)
        self._entity_label_cache.set(entity_id, label)
        
        return label

    def _get_entity_label_direct(self, entity_id: str) -> str:
        """Direct database query for entity label (no caching)."""
        # Use session_scope which will ensure connectivity and retry as needed.
        with self.session_scope() as session:
            result = session.run(
                "MATCH (e:Entity {id: $entity_id}) RETURN e.name as label",
                entity_id=entity_id,
            )
            record = result.single()
            return record["label"] if record else entity_id

    def create_document_node(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Create a document node in the graph."""
        with self.session_scope() as session:
            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d += $metadata
                """,
                doc_id=doc_id,
                metadata=metadata,
            )

    def list_document_ids(self, limit: Optional[int] = None) -> List[str]:
        """Return list of document ids."""
        with self.session_scope() as session:
            q = "MATCH (d:Document) RETURN d.id AS id"
            if limit:
                q += " LIMIT $limit"
            result = session.run(q, limit=limit)
            return [r["id"] for r in result]

    def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Return document properties as a dict."""
        with self.session_scope() as session:
            rec = session.run(
                "MATCH (d:Document {id: $doc_id}) RETURN d AS doc",
                doc_id=doc_id,
            ).single()
            if not rec:
                return {}
            node = rec["doc"]
            try:
                return dict(node)
            except Exception:
                return {}

    def run_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a read query and return list of record dictionaries."""
        with self.session_scope() as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]

    def get_document_text(self, doc_id: str, max_chars: int = 10000) -> str:
        """Return concatenated chunk contents for a document (truncated)."""
        with self.session_scope() as session:
            res = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.content AS content
                ORDER BY c.chunk_index ASC
                """,
                doc_id=doc_id,
            )
            parts = []
            total = 0
            for r in res:
                s = r["content"] or ""
                if not s:
                    continue
                need = max_chars - total
                if need <= 0:
                    break
                parts.append(s[:need])
                total += len(s[:need])
            return "\n\n".join(parts)

    def update_chunk_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> None:
        """Update chunk metadata (merge properties)."""
        with self.session_scope() as session:
            session.run(
                """
                MATCH (c:Chunk {id: $chunk_id})
                SET c += $metadata
                """,
                chunk_id=chunk_id,
                metadata=metadata,
            )

    def merge_nodes(self, target_id: str, source_ids: List[str]) -> bool:
        """
        Merge source nodes into a target node. 
        Transfers all relationships from sources to target, merges properties, and deletes sources.
        
        Args:
            target_id: ID of the node to keep.
            source_ids: IDs of nodes to merge into target.
            
        Returns:
            bool: True if successful.
        """
        if not source_ids:
            return False
            
        with self.session_scope() as session:
            # 1. Verify existence of all nodes
            all_ids = [target_id] + source_ids
            check = session.run(
                "MATCH (n:Entity) WHERE n.id IN $ids RETURN count(n) as count",
                ids=all_ids
            ).single()
            
            if not check or check["count"] != len(set(all_ids)):
                logger.error(f"Merge failed: Not all nodes found. Expected {len(set(all_ids))}, found {check['count'] if check else 0}")
                raise ValueError("Target or source nodes not found")

            # 2. Transfer relationships
            # Redirect incoming edges: (Other)-[r]->(Source) ==> (Other)-[r]->(Target)
            session.run(
                """
                MATCH (source:Entity)<-[r]-(other)
                WHERE source.id IN $source_ids
                AND other.id <> $target_id
                WITH source, r, other
                MATCH (target:Entity {id: $target_id})
                CALL apoc.refactor.to(r, target) YIELD input, output, error
                RETURN count(*)
                """,
                source_ids=source_ids,
                target_id=target_id
            )
            
            # Redirect outgoing edges: (Source)-[r]->(Other) ==> (Target)-[r]->(Other)
            session.run(
                """
                MATCH (source:Entity)-[r]->(other)
                WHERE source.id IN $source_ids
                AND other.id <> $target_id
                WITH source, r, other
                MATCH (target:Entity {id: $target_id})
                CALL apoc.refactor.from(r, target) YIELD input, output, error
                RETURN count(*)
                """,
                source_ids=source_ids,
                target_id=target_id
            )

            # 3. Merge properties (Simple strategy: Append/Concatenate descriptions)
            # We fetch source details and append meaningful distinct info to target
            sources_data = session.run(
                "MATCH (s:Entity) WHERE s.id IN $source_ids RETURN s.name, s.description",
                source_ids=source_ids
            )
            
            # Get current target desc
            target_data = session.run(
                "MATCH (t:Entity {id: $target_id}) RETURN t.description",
                target_id=target_id
            ).single()
            
            current_desc = target_data["t.description"] if target_data else ""
            
            # Append source info
            new_desc_parts = [current_desc]
            for rec in sources_data:
                src_name = rec["s.name"]
                src_desc = rec["s.description"]
                if src_desc and src_desc not in current_desc:
                    new_desc_parts.append(f"[{src_name}]: {src_desc}")
            
            final_desc = "\n".join(filter(None, new_desc_parts))
            
            session.run(
                "MATCH (t:Entity {id: $target_id}) SET t.description = $desc",
                target_id=target_id,
                desc=final_desc
            )
            
            # 4. Delete source nodes
            session.run(
                "MATCH (s:Entity) WHERE s.id IN $source_ids DETACH DELETE s",
                source_ids=source_ids
            )
            
            # 5. Re-generate embedding for target since description changed
            # (Done asynchronously/lazily or we can force it here)
            # We'll just force a re-embed
            self.heal_node(target_id) # heal_node re-generates embedding if missing, but we might want to force update.
            # Actually, let's just nullify the embedding so heal_node or next search fixes it
            session.run("MATCH (t:Entity {id: $target_id}) SET t.embedding = NULL", target_id=target_id)
            self.heal_node(target_id)
            
            return True

    def heal_node(self, node_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find potential missing connections for a node using vector similarity + LLM judgment.
        
        Two-phase approach:
        1. Vector search to find semantically similar candidates (fast)
        2. LLM judges each candidate for logical relationship validity (accurate)
        
        Args:
            node_id: The ID of the node to heal.
            top_k: Number of final candidates to return.
            
        Returns:
            List of candidate nodes with similarity scores and LLM reasoning.
        """
        from core.llm import llm_manager
        
        with self.session_scope() as session:
            # 1. Get target node info
            result = session.run(
                "MATCH (e:Entity {id: $node_id}) RETURN e.id, e.name, e.description, e.type, e.embedding",
                node_id=node_id
            ).single()
            
            if not result:
                raise ValueError(f"Node {node_id} not found")
            
            embedding = result["e.embedding"]
            source_name = result["e.name"]
            source_description = result["e.description"] or ""
            source_type = result["e.type"] or "Entity"
            
            # 2. If embedding is missing, generate it on the fly
            if not embedding:
                logger.info(f"Generating missing embedding for node {node_id} during healing")
                text = f"{source_name}: {source_description}" if source_description else source_name
                try:
                    embedding = embedding_manager.get_embedding(text)
                    # Save it back
                    session.run(
                        "MATCH (e:Entity {id: $node_id}) SET e.embedding = $embedding",
                        node_id=node_id,
                        embedding=embedding
                    )
                except Exception as e:
                    logger.error(f"Failed to generate embedding for {node_id}: {e}")
                    return []

            # 3. Vector search for similar nodes (get more candidates for LLM to filter)
            # Exclude existing neighbors to find *new* connections
            candidates_result = session.run(
                """
                CALL db.index.vector.queryNodes('entity_embeddings', $candidate_count, $embedding)
                YIELD node, score
                WHERE node.id <> $node_id
                AND NOT (node)-[:RELATED_TO]-(:Entity {id: $node_id})
                AND score > 0.5
                RETURN node.id as id, node.name as name, node.description as description, node.type as type, score
                LIMIT $candidate_count
                """,
                embedding=embedding,
                node_id=node_id,
                candidate_count=top_k * 3  # Get more candidates for LLM to filter
            )
            
            candidates = [dict(record) for record in candidates_result]
            
            if not candidates:
                logger.info(f"No embedding candidates found for node {node_id}")
                return []
            
            # 4. LLM judges each candidate
            logger.info(f"LLM judging {len(candidates)} candidates for node '{source_name}'")
            approved_candidates = []
            
            for candidate in candidates:
                try:
                    # Build prompt for LLM judgment
                    target_name = candidate.get("name", "Unknown")
                    target_description = candidate.get("description", "") or ""
                    target_type = candidate.get("type", "Entity") or "Entity"
                    
                    prompt = f"""Evaluate if these two entities from a knowledge graph should be directly connected:

SOURCE ENTITY:
- Name: {source_name}
- Type: {source_type}
- Description: {source_description[:300] if source_description else 'No description'}

TARGET ENTITY:
- Name: {target_name}
- Type: {target_type}
- Description: {target_description[:300] if target_description else 'No description'}

Should these entities have a direct relationship in a knowledge graph?
Consider: Are they related concepts, processes, components, or have meaningful semantic connection?

Reply with ONLY one of:
- YES: [brief reason why they should be connected]
- NO: [brief reason why they should NOT be connected]"""

                    system_message = "You are a knowledge graph curator. Evaluate entity relationships objectively. A connection should represent a meaningful semantic relationship, not just coincidental similarity."
                    
                    response = llm_manager.generate_response(
                        prompt=prompt,
                        system_message=system_message,
                        temperature=0.1,
                        max_tokens=100
                    ).strip()
                    
                    # Parse response
                    if response.upper().startswith("YES"):
                        reason = response[4:].strip(": ").strip() if len(response) > 4 else "Approved by AI"
                        candidate["approved"] = True
                        candidate["reason"] = reason
                        approved_candidates.append(candidate)
                        logger.debug(f"  ✓ Approved: {source_name} → {target_name}: {reason}")
                    else:
                        logger.debug(f"  ✗ Rejected: {source_name} → {target_name}")
                        
                except Exception as e:
                    logger.warning(f"LLM judgment failed for candidate {candidate.get('name')}: {e}")
                    # On error, include candidate but mark as unverified
                    candidate["approved"] = False
                    candidate["reason"] = "LLM verification failed"
                
                # Stop if we have enough approved candidates
                if len(approved_candidates) >= top_k:
                    break
            
            logger.info(f"LLM approved {len(approved_candidates)}/{len(candidates)} candidates for '{source_name}'")
            return approved_candidates[:top_k]

    def find_orphan_nodes(self, min_cluster_size: int = 5) -> List[str]:
        """
        Find orphan nodes: entities not connected to any other entities.
        
        An orphan is an entity that has no RELATED_TO relationships with other entities.
        This means it's isolated in the knowledge graph.
        
        Args:
            min_cluster_size: (Reserved for future cluster-based detection)
            
        Returns:
            List of node IDs that are orphans.
        """
        with self.session_scope() as session:
            # Find entities with no RELATED_TO relationships to other entities
            # These are truly isolated nodes in the knowledge graph
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE NOT (e)-[:RELATED_TO]-(:Entity)
                RETURN e.id as id
                """
            )
            
            orphan_ids = [record["id"] for record in result]
            logger.info(f"Found {len(orphan_ids)} orphan nodes")
            return orphan_ids

    def update_document_summary(
        self,
        doc_id: str,
        summary: str,
        document_type: str,
        hashtags: List[str]
    ) -> None:
        """Update a document node with summary, document type, and hashtags."""
        with self.session_scope() as session:
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                SET d.summary = $summary,
                    d.document_type = $document_type,
                    d.hashtags = $hashtags
                """,
                doc_id=doc_id,
                summary=summary,
                document_type=document_type,
                hashtags=hashtags,
            )

    def update_document_hashtags(
        self,
        doc_id: str,
        hashtags: List[str]
    ) -> None:
        """Update only the hashtags for a document."""
        with self.session_scope() as session:
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                SET d.hashtags = $hashtags
                """,
                doc_id=doc_id,
                hashtags=hashtags,
            )

    def update_document_metadata(
        self,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Update the metadata (properties) for a document."""
        with self.session_scope() as session:
            # We want to merge the new metadata into the existing document properties.
            # CAUTION: This will overwrite existing keys with the same name.
            # It will strictly add/update the properties provided in `metadata`.
            # To replace the entire metadata, one would need to remove other props first,
            # but typically patch semantics are preferred.
            
            # Since 'metadata' in the request is a dict of arbitrary fields,
            # and we store document properties at the node root level or within
            # a 'metadata' map depending on architecture.
            # Looking at create_document_node: SET d += $metadata
            # so we are storing them at root level or mixed in.
            
            # Use same += operator to merge/update
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                SET d.metadata = $metadata
                """,
                doc_id=doc_id,
                metadata=metadata,
            )

    # Temporal Graph Methods

    def create_temporal_nodes_for_document(
        self, doc_id: str, timestamp: Optional[float] = None
    ) -> None:
        """Create temporal nodes and link document to them.

        Creates a hierarchy of temporal nodes (Date -> Month -> Quarter -> Year)
        and links the document via CREATED_AT relationship.

        Args:
            doc_id: Document ID to link to temporal nodes
            timestamp: Unix timestamp (if None, fetches from document's created_at)
        """
        with self.session_scope() as session:
            # Get timestamp from document if not provided
            if timestamp is None:
                result = session.run(
                    "MATCH (d:Document {id: $doc_id}) RETURN d.created_at AS ts",
                    doc_id=doc_id,
                ).single()
                if not result or not result["ts"]:
                    logger.warning(f"No timestamp found for document {doc_id}, skipping temporal nodes")
                    return
                timestamp = result["ts"]

            # Convert timestamp to datetime
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

            # Extract temporal components
            year = dt.year
            quarter = (dt.month - 1) // 3 + 1
            month = dt.month
            date_str = dt.strftime("%Y-%m-%d")

            # Create temporal node hierarchy and link document
            session.run(
                """
                MATCH (d:Document {id: $doc_id})

                // Create Year node
                MERGE (y:TimeNode:Year {year: $year})
                SET y.type = 'year'

                // Create Quarter node
                MERGE (q:TimeNode:Quarter {year: $year, quarter: $quarter})
                SET q.type = 'quarter',
                    q.label = $year + 'Q' + $quarter

                // Create Month node
                MERGE (m:TimeNode:Month {year: $year, month: $month})
                SET m.type = 'month',
                    m.label = $month_label

                // Create Date node
                MERGE (dt:TimeNode:Date {date: $date_str})
                SET dt.type = 'date',
                    dt.year = $year,
                    dt.month = $month,
                    dt.day = $day

                // Create temporal hierarchy
                MERGE (dt)-[:IN_MONTH]->(m)
                MERGE (m)-[:IN_QUARTER]->(q)
                MERGE (q)-[:IN_YEAR]->(y)

                // Link document to date node
                MERGE (d)-[:CREATED_AT]->(dt)
                """,
                doc_id=doc_id,
                year=year,
                quarter=quarter,
                month=month,
                date_str=date_str,
                month_label=dt.strftime("%Y-%m"),
                day=dt.day,
            )
            logger.debug(f"Created temporal nodes for document {doc_id} at {date_str}")

    def retrieve_chunks_with_temporal_filter(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        after_date: Optional[str] = None,
        before_date: Optional[str] = None,
        time_decay_weight: float = 0.0,
        allowed_document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks with temporal filtering and optional time-decay scoring.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            after_date: ISO date string (YYYY-MM-DD) - only return docs after this date
            before_date: ISO date string (YYYY-MM-DD) - only return docs before this date
            time_decay_weight: Weight for time-decay scoring (0.0 = no decay, 1.0 = full decay)
            allowed_document_ids: Optional list of document IDs to restrict search

        Returns:
            List of chunks with similarity scores (adjusted for time decay if enabled)
        """
        with self.session_scope() as session:
            # Build temporal filter clause
            temporal_filter = ""
            params = {
                "embedding": query_embedding,
                "top_k": top_k,
                "time_decay_weight": time_decay_weight,
            }

            if after_date or before_date:
                temporal_filter = """
                MATCH (d)-[:CREATED_AT]->(dt:Date)
                """
                if after_date:
                    temporal_filter += " WHERE dt.date >= $after_date"
                    params["after_date"] = after_date
                if before_date:
                    connector = " AND" if after_date else " WHERE"
                    temporal_filter += f"{connector} dt.date <= $before_date"
                    params["before_date"] = before_date

            # Build document filter clause
            doc_filter = ""
            if allowed_document_ids:
                doc_filter = " AND d.id IN $allowed_doc_ids"
                params["allowed_doc_ids"] = allowed_document_ids

            # Build time-decay scoring
            time_decay_clause = ""
            if time_decay_weight > 0:
                time_decay_clause = """
                // Calculate time decay (exponential decay based on age in days)
                , duration.between(datetime(d.created_at), datetime()).days AS age_days
                , exp(-0.01 * age_days) AS time_factor
                , similarity * (1 - $time_decay_weight + $time_decay_weight * time_factor) AS adjusted_similarity
                """
                score_field = "adjusted_similarity"
            else:
                time_decay_clause = ", similarity AS adjusted_similarity"
                score_field = "adjusted_similarity"

            query = f"""
            CALL db.index.vector.queryNodes('chunk_embeddings', $top_k * 2, $embedding)
            YIELD node AS c, score AS similarity
            MATCH (d:Document)-[:HAS_CHUNK]->(c)
            {temporal_filter}
            WHERE similarity > 0.3{doc_filter}
            WITH c, d, similarity
            {time_decay_clause}
            RETURN c.id AS chunk_id,
                   c.content AS content,
                   c.metadata AS metadata,
                   d.id AS document_id,
                   d.filename AS filename,
                   {score_field} AS score
            ORDER BY {score_field} DESC
            LIMIT $top_k
            """

            result = session.run(query, **params)
            return [dict(record) for record in result]

    def find_temporally_related_chunks(
        self,
        reference_doc_id: str,
        time_window_days: int = 30,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find chunks from documents created around the same time.

        Args:
            reference_doc_id: Reference document ID
            time_window_days: Time window in days (before and after)
            top_k: Number of results to return

        Returns:
            List of chunks from temporally related documents
        """
        with self.session_scope() as session:
            result = session.run(
                """
                // Find reference document's date
                MATCH (ref:Document {id: $ref_doc_id})-[:CREATED_AT]->(ref_date:Date)

                // Find other documents within time window
                MATCH (d:Document)-[:CREATED_AT]->(dt:Date)
                WHERE d.id <> $ref_doc_id
                  AND duration.between(date(ref_date.date), date(dt.date)).days <= $time_window_days
                  AND duration.between(date(ref_date.date), date(dt.date)).days >= -$time_window_days

                // Get chunks from those documents
                MATCH (d)-[:HAS_CHUNK]->(c:Chunk)

                // Calculate temporal proximity score
                WITH c, d, dt, ref_date,
                     abs(duration.between(date(ref_date.date), date(dt.date)).days) AS days_diff,
                     1.0 / (1.0 + abs(duration.between(date(ref_date.date), date(dt.date)).days)) AS proximity_score

                RETURN c.id AS chunk_id,
                       c.content AS content,
                       c.metadata AS metadata,
                       d.id AS document_id,
                       d.filename AS filename,
                       dt.date AS created_date,
                       days_diff,
                       proximity_score AS score
                ORDER BY proximity_score DESC
                LIMIT $top_k
                """,
                ref_doc_id=reference_doc_id,
                time_window_days=time_window_days,
                top_k=top_k,
            )
            return [dict(record) for record in result]

    def get_temporal_statistics(self) -> Dict[str, Any]:
        """Get statistics about temporal distribution of documents.

        Returns:
            Dictionary with temporal statistics
        """
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (d:Document)-[:CREATED_AT]->(dt:Date)
                WITH dt.date AS date, count(d) AS doc_count
                WITH min(date) AS earliest_date,
                     max(date) AS latest_date,
                     sum(doc_count) AS total_docs,
                     collect({date: date, count: doc_count}) AS date_distribution
                RETURN earliest_date, latest_date, total_docs, date_distribution
                """
            ).single()

            if not result:
                return {
                    "earliest_date": None,
                    "latest_date": None,
                    "total_documents": 0,
                    "date_distribution": [],
                }

            return {
                "earliest_date": result["earliest_date"],
                "latest_date": result["latest_date"],
                "total_documents": result["total_docs"],
                "date_distribution": result["date_distribution"],
            }

    def update_document_precomputed_summary(self, doc_id: str) -> Dict[str, int]:
        """Compute and store lightweight precomputed counts on the Document node.

        Stores the following properties on `:Document`:
          - `precomputed_chunk_count`
          - `precomputed_entity_count`
          - `precomputed_community_count`
          - `precomputed_similarity_count`
          - `precomputed_summary_updated_at` (Neo4j datetime)

        Returns a dict with the computed counts.
        """
        with self.session_scope() as session:
            rec = session.run(
                """
                MATCH (d:Document {id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
                WITH d, count(DISTINCT c) AS chunk_count, count(DISTINCT e) AS entity_count, count(DISTINCT e.community_id) AS community_count
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c1:Chunk)-[s:SIMILAR_TO]-(c2:Chunk)
                WHERE (d)-[:HAS_CHUNK]->(c2) AND c1.id < c2.id
                RETURN chunk_count, entity_count, community_count, count(DISTINCT s) AS similarity_count
                """,
                doc_id=doc_id,
            ).single()

            if not rec:
                # Document not found or no stats
                return {
                    "chunk_count": 0,
                    "entity_count": 0,
                    "community_count": 0,
                    "similarity_count": 0,
                }

            chunk_count = rec["chunk_count"] or 0
            entity_count = rec["entity_count"] or 0
            return {
                "chunk_count": chunk_count,
                "entity_count": entity_count,
                "community_count": rec["community_count"] or 0,
                "similarity_count": rec["similarity_count"] or 0,
            }

    # Snapshot & Restore (Phase 0: Safety)

    def export_graph_snapshot(self) -> Dict[str, Any]:
        """Export the entire graph structure (nodes and edges) directly."""
        with self.session_scope() as session:
            # Fetch all nodes with their properties and labels
            # elementId is only available in newer Neo4j, using id() for compatibility or built-in ID handling if needed.
            # However, standard practice for portable export is to rely on application IDs (d.id, e.id).
            # We will grab all properties.
            nodes_result = session.run(
                """
                MATCH (n)
                RETURN labels(n) as labels, properties(n) as props, id(n) as internal_id
                """
            )
            
            nodes = []
            # We map internal Neo4j IPs to a temporary ID for edge reconstruction if application IDs are missing,
            # but ideally our graph is well-formed with 'id' properties on critical nodes.
            # For a pure backup, we need a reliable way to link edge source/target.
            # Using Neo4j internal IDs for the export map is safest for the edges step.
            
            for record in nodes_result:
                nodes.append({
                    "id": str(record["internal_id"]), # Use internal ID for linking
                    "labels": record["labels"],
                    "properties": record["props"]
                })

            # Fetch all edges
            edges_result = session.run(
                """
                MATCH (s)-[r]->(t)
                RETURN id(s) as source, id(t) as target, type(r) as type, properties(r) as props
                """
            )
            
            edges = []
            for record in edges_result:
                edges.append({
                    "source": str(record["source"]),
                    "target": str(record["target"]),
                    "label": record["type"],
                    "properties": record["props"]
                })
                
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "nodes": nodes,
                "edges": edges,
                "version": "1.0"
            }

    def restore_graph_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Wipe database and restore from snapshot.
        Warning: This is a destructive operation.
        """
        nodes = snapshot.get("nodes", [])
        edges = snapshot.get("edges", [])
        
        with self.session_scope() as session:
            # 1. Wipe Database
            logger.warning("Initiating Graph Restore: Wiping database...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # 2. Recreate Nodes (Batching recommended for large graphs, but simple version for now)
            # We use the 'id' from the snapshot (which was the old internal ID) as a temporary key to link edges later.
            # We will store it in a transient property `_restore_id` and remove it later.
            
            if nodes:
                logger.info(f"Restoring {len(nodes)} nodes...")
                session.run(
                    """
                    UNWIND $batch as row
                    CALL apoc.create.node(row.labels, row.properties) YIELD node
                    SET node._restore_id = row.id
                    """,
                    batch=nodes
                )
            
            # 3. Recreate Edges
            if edges:
                logger.info(f"Restoring {len(edges)} edges...")
                # We match source/target by the temporary `_restore_id`
                session.run(
                    """
                    UNWIND $batch as row
                    MATCH (s), (t)
                    WHERE s._restore_id = row.source AND t._restore_id = row.target
                    CALL apoc.create.relationship(s, row.label, row.properties, t) YIELD rel
                    RETURN count(rel)
                    """,
                    batch=edges
                )

            # 4. Cleanup temporary ID
            session.run("MATCH (n) REMOVE n._restore_id")
            logger.info("Graph Restore completed.")

    # Graph Editor (Phase 1: Manual Curation)

    def create_relationship(self, source_id: str, target_id: str, relation_type: str, properties: Dict[str, Any] = {}) -> bool:
        """Create a relationship between two nodes."""
        # Sanitize relation type (Cypher cannot parameterize relationship types directly)
        import re
        safe_type = re.sub(r'[^A-Z0-9_]', '_', relation_type.upper())
        if not safe_type:
            safe_type = "RELATED_TO"

        with self.session_scope() as session:
            # We assume node IDs are the 'id' property. 
            # If using Neo4j internal IDs, the strategy would differ.
            result = session.run(
                f"""
                MATCH (s), (t)
                WHERE (s.id = $source_id OR id(s) = toInteger($source_id)) 
                  AND (t.id = $target_id OR id(t) = toInteger($target_id))
                MERGE (s)-[r:`{safe_type}`]->(t)
                SET r += $properties
                RETURN count(r) as created
                """,
                source_id=source_id,
                target_id=target_id,
                properties=properties
            )
            rec = result.single()
            return rec["created"] > 0 if rec else False

    def delete_relationship(self, source_id: str, target_id: str, relation_type: str) -> bool:
        """Delete a relationship between two nodes."""
        import re
        safe_type = re.sub(r'[^A-Z0-9_]', '_', relation_type.upper())
        
        with self.session_scope() as session:
           result = session.run(
                f"""
                MATCH (s)-[r:`{safe_type}`]->(t)
                WHERE (s.id = $source_id OR id(s) = toInteger($source_id)) 
                  AND (t.id = $target_id OR id(t) = toInteger($target_id))
                DELETE r
                RETURN count(r) as deleted
                """,
                source_id=source_id,
                target_id=target_id
            )
           rec = result.single()
           return rec["deleted"] > 0 if rec else False

    def update_node_properties(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update properties of a node."""
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE n.id = $node_id OR id(n) = toInteger($node_id)
                SET n += $properties
                RETURN count(n) as updated
                """,
                node_id=node_id,
                properties=properties
            )
            rec = result.single()
            return rec["updated"] > 0 if rec else False
            community_count = rec["community_count"] or 0
            similarity_count = rec["similarity_count"] or 0

            # Persist precomputed values on document node
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                SET d.precomputed_chunk_count = $chunk_count,
                    d.precomputed_entity_count = $entity_count,
                    d.precomputed_community_count = $community_count,
                    d.precomputed_similarity_count = $similarity_count,
                    d.precomputed_summary_updated_at = datetime()
                """,
                doc_id=doc_id,
                chunk_count=chunk_count,
                entity_count=entity_count,
                community_count=community_count,
                similarity_count=similarity_count,
            )

        return {
            "chunk_count": chunk_count,
            "entity_count": entity_count,
            "community_count": community_count,
            "similarity_count": similarity_count,
        }

    def update_document_preview(self, doc_id: str, top_n_communities: int = None, top_n_similarities: int = None) -> Dict[str, Any]:
        """Compute and persist small preview lists for a document.

        - top communities: list of community_id with counts
        - top similarities: list of {chunk1_id, chunk2_id, score}

        Stores JSON-serialized strings on the Document node to keep properties simple.
        Returns a dict containing the preview data.
        """
        if top_n_communities is None:
            top_n_communities = getattr(settings, "document_summary_top_n_communities", 10)
        if top_n_similarities is None:
            top_n_similarities = getattr(settings, "document_summary_top_n_similarities", 20)

        with self.session_scope() as session:
            # Top communities by entity count within this document
            comm_q = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                WHERE e.community_id IS NOT NULL
                RETURN e.community_id AS community_id, count(DISTINCT e) AS cnt
                ORDER BY cnt DESC
                LIMIT $limit
                """,
                doc_id=doc_id,
                limit=top_n_communities,
            )
            top_communities = [ {"community_id": record["community_id"], "count": record["cnt"]} for record in comm_q ]

            # Top chunk-to-chunk similarities by score within this document
            sim_q = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c1:Chunk)-[s:SIMILAR_TO]-(c2:Chunk)
                WHERE (d)-[:HAS_CHUNK]->(c2) AND c1.id < c2.id
                RETURN c1.id AS chunk1_id, c2.id AS chunk2_id, coalesce(s.score, 0) AS score
                ORDER BY score DESC
                LIMIT $limit
                """,
                doc_id=doc_id,
                limit=top_n_similarities,
            )
            top_sims = [ {"chunk1_id": r["chunk1_id"], "chunk2_id": r["chunk2_id"], "score": float(r["score"] or 0)} for r in sim_q ]

            # Persist JSON strings on the Document node
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                SET d.precomputed_top_communities_json = $comm_json,
                    d.precomputed_top_similarities_json = $sims_json,
                    d.precomputed_summary_updated_at = datetime()
                """,
                doc_id=doc_id,
                comm_json=json.dumps(top_communities),
                sims_json=json.dumps(top_sims),
            )

        return {"top_communities": top_communities, "top_similarities": top_sims}

    def get_documents_with_summaries(self) -> List[Dict[str, Any]]:
        """Get all documents that have summaries."""
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (d:Document)
                WHERE d.summary IS NOT NULL AND d.summary <> ''
                RETURN d.id as document_id,
                       d.summary as summary,
                       d.document_type as document_type,
                       d.hashtags as hashtags,
                       d.filename as filename
                """
            )
            return [record.data() for record in result]

    def get_all_hashtags(self) -> List[str]:
        """Get all unique hashtags from all documents."""
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (d:Document)
                WHERE d.hashtags IS NOT NULL
                UNWIND d.hashtags as hashtag
                RETURN DISTINCT hashtag
                ORDER BY hashtag
                """
            )
            return [record["hashtag"] for record in result]

    def create_chunk_node(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Create a chunk node and link it to its document."""
        logger.debug(
            "create_chunk_node called: chunk_id=%s doc_id=%s content_len=%s embedding_len=%s",
            chunk_id,
            doc_id,
            len(content) if content is not None else 0,
            len(embedding) if embedding is not None else 0,
        )

        try:
            # Extract content_hash from metadata to store as top-level property
            content_hash = metadata.get("content_hash", "")
            
            with self.session_scope() as session:
                session.run(
                    """
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.content = $content,
                        c.embedding = $embedding,
                        c.chunk_index = $chunk_index,
                        c.offset = $offset,
                        c.content_hash = $content_hash,
                        c += $metadata
                    WITH c
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=content,
                    embedding=embedding,
                    chunk_index=metadata.get("chunk_index", 0),
                    offset=metadata.get("offset", 0),
                    content_hash=content_hash,
                    metadata=metadata,
                )
        except Exception as e:
            logger.error(
                "Failed to create chunk node %s for document %s: %s",
                chunk_id,
                doc_id,
                e,
            )
            # Re-raise so callers can handle or log at higher level
            raise

    def create_similarity_relationship(
        self, chunk_id1: str, chunk_id2: str, similarity_score: float, rank: int = None
    ) -> None:
        """Create similarity relationship between chunks."""
        with self.session_scope() as session:
            cypher = """
                MATCH (c1:Chunk {id: $chunk_id1})
                MATCH (c2:Chunk {id: $chunk_id2})
                MERGE (c1)-[r:SIMILAR_TO]-(c2)
                SET r.score = $similarity_score
                """
            params = {
                "chunk_id1": chunk_id1,
                "chunk_id2": chunk_id2,
                "similarity_score": similarity_score,
            }
            if rank is not None:
                cypher += "\nSET r.rank = $rank"
                params["rank"] = rank
            session.run(cypher, **params)

    @staticmethod
    def _calculate_cosine_similarity(
        embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same length")

        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(a * a for a in embedding2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def create_chunk_similarities(self, doc_id: str, threshold: float = None) -> int:  # type: ignore
        """Create similarity relationships between chunks of a document."""
        if threshold is None:
            threshold = settings.similarity_threshold

        with self.session_scope() as session:
            # Get all chunks for the document with their embeddings
            result = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.id as chunk_id, c.embedding as embedding
                ORDER BY c.chunk_index ASC
                """,
                doc_id=doc_id,
            )

            chunks_data = [
                (record["chunk_id"], record["embedding"]) for record in result
            ]

            if len(chunks_data) < 2:
                logger.info(
                    f"Skipping similarity creation for document {doc_id}: less than 2 chunks"
                )
                return 0

            relationships_created = 0
            max_connections = settings.max_similarity_connections

            # Calculate similarities between all pairs of chunks
            for i in range(len(chunks_data)):
                chunk_id1, embedding1 = chunks_data[i]
                similarities = []

                for j in range(len(chunks_data)):
                    if i != j:
                        chunk_id2, embedding2 = chunks_data[j]
                        similarity = self._calculate_cosine_similarity(
                            embedding1, embedding2
                        )

                        if similarity >= threshold:
                            similarities.append((chunk_id2, similarity))

                # Sort by similarity and take top connections
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_similarities = similarities[:max_connections]

                # Create relationships with rank
                for rank, (chunk_id2, similarity) in enumerate(top_similarities):
                    self.create_similarity_relationship(
                        chunk_id1, chunk_id2, similarity, rank
                    )
                    relationships_created += 1

            logger.info(
                f"Created {relationships_created} similarity relationships for document {doc_id}"
            )
            return relationships_created

    def create_all_chunk_similarities(self, threshold: float = None, batch_size: int = 10) -> Dict[str, int]:  # type: ignore
        """Create similarity relationships for all documents in the database."""
        if threshold is None:
            threshold = settings.similarity_threshold

        with self.session_scope() as session:
            # Get all document IDs
            result = session.run("MATCH (d:Document) RETURN d.id as doc_id")
            doc_ids = [record["doc_id"] for record in result]

        total_relationships = 0
        processed_docs = 0
        results = {}

        for doc_id in doc_ids:
            try:
                relationships_created = self.create_chunk_similarities(
                    doc_id, threshold
                )
                results[doc_id] = relationships_created
                total_relationships += relationships_created
                processed_docs += 1

                logger.info(
                    f"Processed document {doc_id}: {relationships_created} relationships"
                )

                # Process in batches to avoid memory issues
                if processed_docs % batch_size == 0:
                    logger.info(
                        f"Processed {processed_docs}/{len(doc_ids)} documents so far"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to create similarities for document {doc_id}: {e}"
                )
                results[doc_id] = 0

        logger.info(
            f"Batch processing complete: {total_relationships} total relationships created for {processed_docs} documents"
        )
        return results

    def create_entity_similarities(self, doc_id: str = None, threshold: float = None) -> int:  # type: ignore
        """Create similarity relationships between entities based on their embeddings."""
        if threshold is None:
            threshold = settings.similarity_threshold

        with self.session_scope() as session:
            # Build query based on whether we're processing specific doc or all entities
            if doc_id:
                # Get entities for specific document
                result = session.run(
                    """
                    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                    WHERE e.embedding IS NOT NULL
                    RETURN DISTINCT e.id as entity_id, e.embedding as embedding, e.name as name, e.type as type
                    """,
                    doc_id=doc_id,
                )
            else:
                # Get all entities with embeddings
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.embedding IS NOT NULL
                    RETURN e.id as entity_id, e.embedding as embedding, e.name as name, e.type as type
                    """
                )

            entities_data = [
                (
                    record["entity_id"],
                    record["embedding"],
                    record["name"],
                    record["type"],
                )
                for record in result
            ]

            if len(entities_data) < 2:
                scope = f"document {doc_id}" if doc_id else "database"
                logger.info(
                    f"Skipping entity similarity creation for {scope}: less than 2 entities with embeddings"
                )
                return 0

            relationships_created = 0
            max_connections = settings.max_similarity_connections

            # Calculate similarities between all pairs of entities
            for i in range(len(entities_data)):
                entity_id1, embedding1, name1, type1 = entities_data[i]
                similarities = []

                for j in range(len(entities_data)):
                    if i != j:
                        entity_id2, embedding2, name2, type2 = entities_data[j]

                        # Skip if same entity type and name (likely duplicate)
                        if type1 == type2 and name1 == name2:
                            continue

                        similarity = self._calculate_cosine_similarity(
                            embedding1, embedding2
                        )

                        if similarity >= threshold:
                            similarities.append((entity_id2, similarity))

                # Sort by similarity and take top connections
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_similarities = similarities[:max_connections]

                # Create relationships
                for entity_id2, similarity in top_similarities:
                    self._create_entity_similarity_relationship(
                        entity_id1, entity_id2, similarity
                    )
                    relationships_created += 1

            scope = f"document {doc_id}" if doc_id else "all entities"
            logger.info(
                f"Created {relationships_created} entity similarity relationships for {scope}"
            )
            return relationships_created

    def create_all_entity_similarities(self, threshold: float = None, batch_size: int = 10) -> Dict[str, int]:  # type: ignore
        """Create entity similarity relationships for all documents in the database."""
        if threshold is None:
            threshold = settings.similarity_threshold

        with self.session_scope() as session:
            # Get all document IDs that have entities
            result = session.run(
                """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                RETURN DISTINCT d.id as doc_id
                """
            )
            doc_ids = [record["doc_id"] for record in result]

        if not doc_ids:
            logger.info("No documents with entities found")
            return {}

        total_relationships = 0
        processed_docs = 0
        results = {}

        for doc_id in doc_ids:
            try:
                relationships_created = self.create_entity_similarities(
                    doc_id, threshold
                )
                results[doc_id] = relationships_created
                total_relationships += relationships_created
                processed_docs += 1

                logger.info(
                    f"Processed document {doc_id}: {relationships_created} entity relationships"
                )

                # Process in batches to avoid memory issues
                if processed_docs % batch_size == 0:
                    logger.info(
                        f"Processed {processed_docs}/{len(doc_ids)} documents so far"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to create entity similarities for document {doc_id}: {e}"
                )
                results[doc_id] = 0

        logger.info(
            f"Entity similarity batch processing complete: {total_relationships} total relationships created for {processed_docs} documents"
        )
        return results

    def _create_entity_similarity_relationship(
        self, entity_id1: str, entity_id2: str, similarity: float
    ) -> None:
        """Create a similarity relationship between two entities."""
        with self.session_scope() as session:
            session.run(
                """
                MATCH (e1:Entity {id: $entity_id1})
                MATCH (e2:Entity {id: $entity_id2})
                MERGE (e1)-[r:SIMILAR_TO]-(e2)
                SET r.similarity = $similarity, r.created_at = datetime()
                """,
                entity_id1=entity_id1,
                entity_id2=entity_id2,
                similarity=similarity,
            )

    def vector_similarity_search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search using cosine similarity."""
        try:
            with self.session_scope() as session:
                result = session.run(
                    """
                    CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
                    YIELD node, score
                    MATCH (d:Document)-[:HAS_CHUNK]->(node)
                    RETURN node.id as chunk_id, node.content as content, score as similarity,
                           coalesce(d.original_filename, d.filename) as document_name, d.id as document_id
                    """,
                    query_embedding=query_embedding,
                    top_k=top_k,
                )
                results_list = [record.data() for record in result]
                logger.debug(f"Vector search returned {len(results_list)} results for query (top_k={top_k})")
                return results_list
        except Exception as e:
            # If vector index search fails, fall back to Python-side cosine calculation
            logger.warning("Vector index search failed; falling back to Python cosine computation: %s", e)

            # Query candidate chunks that have embeddings and compute cosine locally
            with self.session_scope() as session:
                result = session.run(
                    """
                    MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                    WHERE c.embedding IS NOT NULL
                    RETURN c.id as chunk_id, c.content as content, c.embedding as embedding,
                           coalesce(d.original_filename, d.filename) as document_name, d.id as document_id
                    """
                )
                candidates = [record.data() for record in result]

            # Compute cosine similarity in Python (fallback, pure-Python implementation)
            def _cosine(a: List[float], b: List[float]) -> float:
                try:
                    # simple dot / (||a|| * ||b||)
                    dot = 0.0
                    na = 0.0
                    nb = 0.0
                    for x, y in zip(a, b):
                        dot += x * y
                        na += x * x
                        nb += y * y
                    if na == 0.0 or nb == 0.0:
                        return 0.0
                    return dot / ((na ** 0.5) * (nb ** 0.5))
                except Exception:
                    return 0.0

            scored = []
            for row in candidates:
                emb = row.get("embedding")
                if not emb:
                    continue
                try:
                    sim = _cosine(query_embedding, emb)
                except Exception:
                    sim = 0.0
                scored.append({
                    "chunk_id": row.get("chunk_id"),
                    "content": row.get("content"),
                    "similarity": sim,
                    "document_name": row.get("document_name"),
                    "document_id": row.get("document_id"),
                })

            scored.sort(key=lambda r: r.get("similarity", 0.0), reverse=True)
            return scored[:top_k]

    def get_related_chunks(
        self,
        chunk_id: str,
        relationship_types: List[str] = None,  # type: ignore
        max_depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """Get chunks related to a given chunk through various relationships."""
        if relationship_types is None:
            relationship_types = ["SIMILAR_TO", "HAS_CHUNK"]

        with self.session_scope() as session:
            # Build the query dynamically since Neo4j doesn't allow parameters in pattern ranges
            query = f"""
                MATCH (start:Chunk {{id: $chunk_id}})
                MATCH path = (start)-[*1..{max_depth}]-(related:Chunk)
                WHERE ALL(r in relationships(path) WHERE type(r) IN $relationship_types)
                OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(related)
                WITH related, d, length(path) as distance,
                     [r in relationships(path) WHERE type(r) = 'SIMILAR_TO' | r.score] as similarity_scores
                WITH related, d, distance,
                     CASE
                         WHEN size(similarity_scores) > 0 THEN
                             reduce(avg = 0.0, s in similarity_scores | avg + s) / size(similarity_scores)
                         ELSE
                             CASE distance
                                 WHEN 1 THEN 0.3
                                 WHEN 2 THEN 0.2
                                 ELSE 0.15
                             END
                     END as calculated_similarity
                RETURN DISTINCT related.id as chunk_id, related.content as content,
                       distance, coalesce(d.original_filename, d.filename) as document_name, d.id as document_id,
                       calculated_similarity as similarity
                ORDER BY distance ASC, calculated_similarity DESC
                """

            result = session.run(
                query,  # type: ignore
                chunk_id=chunk_id,
                relationship_types=relationship_types,
            )
            return [record.data() for record in result]

    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.id as chunk_id, c.content as content, c.embedding as embedding
                ORDER BY c.chunk_index ASC
                """,
                doc_id=doc_id,
            )
            return [record.data() for record in result]

    def delete_document(self, doc_id: str) -> None:
        """Delete a document and all its chunks."""
        with self.session_scope() as session:
            # 1. Collect chunk ids for the document
            result = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN collect(c.id) as chunk_ids
                """,
                doc_id=doc_id,
            )

            record = result.single()
            chunk_ids = (
                record["chunk_ids"]
                if record and record["chunk_ids"] is not None
                else []
            )

            if chunk_ids:
                # 2. Remove references to these chunks from Entity.source_chunks lists
                session.run(
                    """
                    UNWIND $chunk_ids AS cid
                    MATCH (e:Entity)
                    WHERE cid IN coalesce(e.source_chunks, [])
                    SET e.source_chunks = [s IN coalesce(e.source_chunks, []) WHERE s <> cid]
                    """,
                    chunk_ids=chunk_ids,
                )

                # 3. Delete CONTAINS_ENTITY relationships from the chunks (so entities lose relationships to these chunks)
                session.run(
                    """
                    MATCH (c:Chunk)-[r:CONTAINS_ENTITY]->(e:Entity)
                    WHERE c.id IN $chunk_ids
                    DELETE r
                    """,
                    chunk_ids=chunk_ids,
                )

                # 4. Delete entities that are now orphaned: no source_chunks and no incoming CONTAINS_ENTITY relationships
                session.run(
                    """
                    MATCH (e:Entity)
                    WHERE (coalesce(e.source_chunks, []) = [] OR e.source_chunks IS NULL)
                    AND NOT ( ()-[:CONTAINS_ENTITY]->(e) )
                    DETACH DELETE e
                    """,
                )

            # 5. Finally delete chunks and the document
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                DETACH DELETE c, d
                """,
                doc_id=doc_id,
            )

            logger.info(
                f"Deleted document {doc_id} and cleaned up {len(chunk_ids)} chunks and related entities"
            )

    # ========== Incremental Document Update Methods ==========

    def get_chunk_hashes_for_document(self, doc_id: str) -> Dict[str, str]:
        """
        Return {content_hash: chunk_id} mapping for a document.
        
        Used for incremental updates to identify which chunks already exist.
        """
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                WHERE c.content_hash IS NOT NULL
                RETURN c.content_hash as hash, c.id as chunk_id
                """,
                doc_id=doc_id,
            )
            return {record["hash"]: record["chunk_id"] for record in result if record["hash"]}

    def get_document_chunking_params(self, doc_id: str) -> Optional[Dict[str, int]]:
        """
        Get the chunking parameters used when the document was originally ingested.
        
        Returns:
            Dict with chunk_size_used and chunk_overlap_used, or None if not found.
        """
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (d:Document {id: $doc_id})
                RETURN d.chunk_size_used as chunk_size, d.chunk_overlap_used as chunk_overlap
                """,
                doc_id=doc_id,
            )
            record = result.single()
            if record and record["chunk_size"] is not None:
                return {
                    "chunk_size_used": record["chunk_size"],
                    "chunk_overlap_used": record["chunk_overlap"],
                }
            return None

    def delete_chunks_with_entity_cleanup(self, chunk_ids: List[str]) -> Dict[str, int]:
        """
        Delete specific chunks with comprehensive entity cleanup.
        
        This method performs a targeted deletion of chunks while properly cleaning
        up all related entities and relationships:
        
        1. Find entities ONLY referenced by these chunks (orphaned after deletion)
        2. Delete those orphaned entities
        3. Remove CONTAINS_ENTITY relationships from the chunks
        4. Update Entity.source_chunks lists to remove references
        5. Delete SIMILAR_TO relationships involving these chunks
        6. Delete the chunks themselves
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            Dict with cleanup statistics:
                - chunks_deleted: Number of chunks removed
                - entities_deleted: Number of orphaned entities removed
                - relationships_cleaned: Number of relationships removed
                - source_chunks_updated: Number of entities whose source_chunks were updated
        """
        if not chunk_ids:
            return {
                "chunks_deleted": 0,
                "entities_deleted": 0,
                "relationships_cleaned": 0,
                "source_chunks_updated": 0,
            }

        with self.session_scope() as session:
            # 1. Update Entity.source_chunks lists to remove references to these chunks
            source_update_result = session.run(
                """
                UNWIND $chunk_ids AS cid
                MATCH (e:Entity)
                WHERE cid IN coalesce(e.source_chunks, [])
                SET e.source_chunks = [s IN coalesce(e.source_chunks, []) WHERE s <> cid]
                RETURN count(DISTINCT e) as updated_count
                """,
                chunk_ids=chunk_ids,
            ).single()
            source_chunks_updated = source_update_result["updated_count"] if source_update_result else 0

            # 2. Delete CONTAINS_ENTITY relationships from these chunks
            contains_result = session.run(
                """
                MATCH (c:Chunk)-[r:CONTAINS_ENTITY]->(e:Entity)
                WHERE c.id IN $chunk_ids
                DELETE r
                RETURN count(r) as deleted_count
                """,
                chunk_ids=chunk_ids,
            ).single()
            contains_deleted = contains_result["deleted_count"] if contains_result else 0

            # 3. Delete SIMILAR_TO relationships involving these chunks
            similar_result = session.run(
                """
                MATCH (c:Chunk)-[r:SIMILAR_TO]-()
                WHERE c.id IN $chunk_ids
                DELETE r
                RETURN count(r) as deleted_count
                """,
                chunk_ids=chunk_ids,
            ).single()
            similar_deleted = similar_result["deleted_count"] if similar_result else 0

            # 4. Delete orphaned entities (no remaining source_chunks and no CONTAINS_ENTITY)
            orphan_result = session.run(
                """
                MATCH (e:Entity)
                WHERE (coalesce(e.source_chunks, []) = [] OR e.source_chunks IS NULL)
                AND NOT ( ()-[:CONTAINS_ENTITY]->(e) )
                DETACH DELETE e
                RETURN count(e) as deleted_count
                """,
            ).single()
            entities_deleted = orphan_result["deleted_count"] if orphan_result else 0

            # 5. Delete the chunks themselves
            chunks_result = session.run(
                """
                MATCH (c:Chunk)
                WHERE c.id IN $chunk_ids
                DETACH DELETE c
                RETURN count(c) as deleted_count
                """,
                chunk_ids=chunk_ids,
            ).single()
            chunks_deleted = chunks_result["deleted_count"] if chunks_result else 0

            total_relationships = contains_deleted + similar_deleted

            logger.info(
                f"Deleted {chunks_deleted} chunks with cleanup: "
                f"{entities_deleted} orphaned entities removed, "
                f"{total_relationships} relationships removed, "
                f"{source_chunks_updated} entities updated"
            )

            return {
                "chunks_deleted": chunks_deleted,
                "entities_deleted": entities_deleted,
                "relationships_cleaned": total_relationships,
                "source_chunks_updated": source_chunks_updated,
            }

    def reset_document_entities(self, doc_id: str) -> Dict[str, int]:
        """Remove existing entity links for a document so extraction can be rerun cleanly."""

        with self.session_scope() as session:
            record = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN collect(c.id) AS chunk_ids
                """,
                doc_id=doc_id,
            ).single()

            chunk_ids: List[str] = [] if record is None else record.get("chunk_ids", [])

            # Clear previous extraction metrics
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                SET d.entity_extraction_metrics = NULL
                """,
                doc_id=doc_id,
            )

            if not chunk_ids:
                return {"chunk_ids": 0, "entity_relationships": 0, "chunk_entity_relationships": 0, "entities_removed": 0}

            rel_result = session.run(
                """
                MATCH (e1:Entity)-[r:RELATED_TO]-(e2:Entity)
                WHERE any(cid IN coalesce(r.source_chunks, []) WHERE cid IN $chunk_ids)
                DELETE r
                RETURN count(r) AS deleted
                """,
                chunk_ids=chunk_ids,
            ).single()
            rel_deleted = 0 if rel_result is None else rel_result.get("deleted", 0)

            chunk_entity_result = session.run(
                """
                MATCH (c:Chunk)-[rel:CONTAINS_ENTITY]->(e:Entity)
                WHERE c.id IN $chunk_ids
                DELETE rel
                RETURN count(rel) AS deleted
                """,
                chunk_ids=chunk_ids,
            ).single()
            chunk_entity_deleted = (
                0 if chunk_entity_result is None else chunk_entity_result.get("deleted", 0)
            )

            session.run(
                """
                MATCH (e:Entity)
                WHERE any(cid IN coalesce(e.source_chunks, []) WHERE cid IN $chunk_ids)
                SET e.source_chunks = [cid IN coalesce(e.source_chunks, []) WHERE NOT cid IN $chunk_ids]
                """,
                chunk_ids=chunk_ids,
            )

            orphan_result = session.run(
                """
                MATCH (e:Entity)
                WHERE (coalesce(e.source_chunks, []) = [] OR e.source_chunks IS NULL)
                  AND NOT ( ()-[:CONTAINS_ENTITY]->(e) )
                WITH e LIMIT 10000
                DETACH DELETE e
                RETURN count(e) AS deleted
                """,
            ).single()
            orphan_deleted = 0 if orphan_result is None else orphan_result.get("deleted", 0)

        logger.info(
            "Reset entity data for document %s — chunks: %s, entity rels removed: %s, chunk/entity rels removed: %s, orphans deleted: %s",
            doc_id,
            len(chunk_ids),
            rel_deleted,
            chunk_entity_deleted,
            orphan_deleted,
        )

        return {
            "chunk_ids": len(chunk_ids),
            "entity_relationships": rel_deleted,
            "chunk_entity_relationships": chunk_entity_deleted,
            "entities_removed": orphan_deleted,
        }

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents with their metadata, chunk counts, and OCR information."""
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                WITH d, count(c) as chunk_count
                RETURN d.id as document_id,
                       d.filename as filename,
                       d.file_size as file_size,
                       d.file_extension as file_extension,
                       d.created_at as created_at,
                       d.modified_at as modified_at,
                       COALESCE(d.processing_method, '') as processing_method,
                       COALESCE(d.ocr_applied_pages, 0) as ocr_applied_pages,
                       COALESCE(d.readable_text_pages, 0) as readable_text_pages,
                       COALESCE(d.total_pages, 0) as total_pages,
                       COALESCE(d.ocr_items_count, 0) as ocr_items_count,
                       COALESCE(d.summary_total_pages, 0) as summary_total_pages,
                       COALESCE(d.summary_readable_pages, 0) as summary_readable_pages,
                       COALESCE(d.summary_ocr_pages, 0) as summary_ocr_pages,
                       COALESCE(d.summary_image_pages, 0) as summary_image_pages,
                       COALESCE(d.summary_mixed_pages, 0) as summary_mixed_pages,
                       COALESCE(d.content_primary_type, '') as content_primary_type,
                       chunk_count
                ORDER BY d.filename ASC
                """
            )
            return [record.data() for record in result]

    def get_community_levels(self) -> List[int]:
        """Return sorted community levels that have assignments."""

        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.community_id IS NOT NULL AND e.level IS NOT NULL
                RETURN DISTINCT e.level as level
                ORDER BY level ASC
                """
            )

            return [record["level"] for record in result if record["level"] is not None]

    def get_communities_for_level(self, level: int) -> List[Dict[str, Any]]:
        """Return community assignments and entity metadata for a given level."""

        # Basic validation
        if level is None:
            raise ValueError("level must be provided")
        try:
            level_int = int(level)
        except Exception:
            raise ValueError("level must be an integer")

        # Ensure we have an active connection before running queries
        self.ensure_connected()

        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.community_id IS NOT NULL AND e.level = $level
                RETURN e.community_id AS community_id,
                       collect({
                           id: e.id,
                           name: coalesce(e.name, ""),
                           type: coalesce(e.type, ""),
                           description: coalesce(e.description, ""),
                           importance_score: coalesce(e.importance_score, 0.0)
                       }) AS entities
                ORDER BY community_id
                """,
                level=level_int,
            )

            communities = []
            total_entities = 0
            for record in result:
                entities: List[Dict[str, Any]] = record.get("entities", [])
                total_entities += len(entities)
                communities.append(
                    {
                        "community_id": record.get("community_id"),
                        "entities": entities,
                        "entity_count": len(entities),
                    }
                )

            logger.info(
                "get_communities_for_level(level=%s) -> %s communities, %s entities",
                level_int,
                len(communities),
                total_entities,
            )

            return communities

    def get_communities_aggregates_for_level(self, level: int) -> List[Dict[str, Any]]:
        """Return communities with aggregated intra-community relationships and text_unit ids.

        This method performs a single aggregated query to collect relationships that
        exist between entities inside the same community (at the requested level) and
        returns for each community: community_id, entities (list of basic metadata),
        relationships (id, source, target, type, text_unit_ids) and counts.
        """

        if level is None:
            raise ValueError("level must be provided")
        try:
            level_int = int(level)
        except Exception:
            raise ValueError("level must be an integer")

        # Ensure DB connection is available
        self.ensure_connected()

        query = """
        MATCH (e1:Entity)-[r:RELATED_TO]-(e2:Entity)
        WHERE e1.community_id IS NOT NULL
          AND e1.level = $level
          AND e2.level = $level
          AND e1.community_id = e2.community_id
        WITH e1.community_id AS community_id,
             collect(DISTINCT {id: e1.id, name: coalesce(e1.name, ''), type: coalesce(e1.type, ''), importance_score: coalesce(e1.importance_score, 0.0)}) AS members,
             collect(DISTINCT r) AS rels
        WITH community_id, members, rels
        UNWIND rels AS rel
        WITH community_id, members, collect(DISTINCT {
            id: id(rel),
            source: startNode(rel).id,
            target: endNode(rel).id,
            type: type(rel),
            text_unit_ids: coalesce(rel.source_text_units, [])
        }) AS relationships
        RETURN community_id, members AS entities, relationships
        ORDER BY community_id ASC
        """

        with self.session_scope() as session:
            try:
                result = session.run(query, level=level_int)
                communities = []
                total_rels = 0
                for record in result:
                    rels = record.get("relationships") or []
                    total_rels += len(rels)
                    entities = record.get("entities") or []
                    communities.append(
                        {
                            "community_id": record.get("community_id"),
                            "entities": entities,
                            "entity_count": len(entities),
                            "relationships": rels,
                            "relationship_count": len(rels),
                        }
                    )

                logger.info(
                    "get_communities_aggregates_for_level(level=%s) -> %s communities, %s relationships",
                    level_int,
                    len(communities),
                    total_rels,
                )

                return communities
            except Exception as exc:
                logger.exception("Failed to aggregate communities for level %s: %s", level_int, exc)
                return []

    def get_text_units_for_entities(
        self, entity_ids: List[str], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Fetch exemplar TextUnits linked to the provided entities."""

        if not entity_ids:
            return []

        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                WHERE e.id IN $entity_ids
                WITH DISTINCT c, d
                RETURN c.id AS id,
                       coalesce(c.content, "") AS content,
                       d.id AS document_id,
                       coalesce(c.quality_score, 1.0) AS quality_score
                ORDER BY coalesce(c.quality_score, 1.0) DESC,
                         size(c.content) DESC
                LIMIT $limit
                """,
                entity_ids=entity_ids,
                limit=limit,
            )

            return [record.data() for record in result]

    def upsert_community_summary(
        self,
        community_id: int,
        level: int,
        summary: str,
        member_entities: List[Dict[str, Any]],
        exemplar_text_units: List[Dict[str, Any]],
    ) -> None:
        """Persist or update a community summary node."""
        import json
        
        # Convert complex objects to JSON strings for Neo4j compatibility
        member_entities_json = json.dumps(member_entities)
        exemplar_text_units_json = json.dumps(exemplar_text_units)

        with self.session_scope() as session:
            session.run(
                """
                MERGE (s:CommunitySummary {community_id: $community_id, level: $level})
                SET s.summary = $summary,
                    s.member_entities_json = $member_entities_json,
                    s.exemplar_text_units_json = $exemplar_text_units_json,
                    s.generated_at = datetime()
                """,
                community_id=community_id,
                level=level,
                summary=summary,
                member_entities_json=member_entities_json,
                exemplar_text_units_json=exemplar_text_units_json,
            )

    def get_graph_stats(self) -> Dict[str, int]:
        """Get basic statistics about the graph database."""
        with self.session_scope() as session:
            result = session.run(
                """
                OPTIONAL MATCH (d:Document)
                WITH count(d) AS documents
                OPTIONAL MATCH (c:Chunk)
                WITH documents, count(c) AS chunks
                OPTIONAL MATCH (e:Entity)
                WITH documents, chunks, count(e) AS entities
                OPTIONAL MATCH ()-[r]-()
                WITH documents, chunks, entities,
                     sum(CASE WHEN type(r) = 'HAS_CHUNK' THEN 1 ELSE 0 END) AS has_chunk_relations,
                     sum(CASE WHEN type(r) = 'SIMILAR_TO' AND (startNode(r):Chunk OR endNode(r):Chunk) THEN 1 ELSE 0 END) AS similarity_relations,
                     sum(CASE WHEN type(r) = 'RELATED_TO' OR (type(r) = 'SIMILAR_TO' AND startNode(r):Entity AND endNode(r):Entity) THEN 1 ELSE 0 END) AS entity_relations,
                     sum(CASE WHEN type(r) = 'CONTAINS_ENTITY' THEN 1 ELSE 0 END) AS chunk_entity_relations
                RETURN documents, chunks, entities, has_chunk_relations,
                       similarity_relations, entity_relations, chunk_entity_relations
                """
            )
            record = result.single()
            if record is not None:
                return record.data()
            else:
                return {
                    "documents": 0,
                    "chunks": 0,
                    "entities": 0,
                    "has_chunk_relations": 0,
                    "similarity_relations": 0,
                    "entity_relations": 0,
                    "chunk_entity_relations": 0,
                }

    def get_clustered_graph(
        self,
        community_id: Optional[int] = None,
        node_type: Optional[str] = None,
        level: Optional[int] = None,
        limit: int = 300,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return clustered graph data with community and degree metadata."""

        filters = []
        params: Dict[str, Any] = {
            "limit": max(1, min(limit, 1000)),
        }

        # Bound the total number of materialized nodes used in the Cypher
        # to avoid large transactions that can hit Neo4j memory limits.
        # Choose a sensible cap relative to the requested `limit`.
        params["max_nodes"] = min(max(25, params["limit"] * 3), 1000)

        if community_id is not None:
            filters.append("e.community_id = $community_id")
            params["community_id"] = community_id
        if node_type:
            filters.append("e.type = $node_type")
            params["node_type"] = node_type
        if level is not None:
            filters.append("e.level = $community_level")
            params["community_level"] = level
        if document_id:
            # Use an EXISTS block to scope the pattern inside the predicate
            # rather than introducing new variables in the WHERE clause
            filters.append(
                "EXISTS { MATCH (e)<-[:CONTAINS_ENTITY]-(:Chunk)<-[:HAS_CHUNK]-(d:Document) WHERE d.id = $document_id }"
            )
            params["document_id"] = document_id

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        query = f"""
            MATCH (e:Entity)
            {where_clause}
            WITH collect(e)[0..$limit] AS selected
            UNWIND selected AS e
            OPTIONAL MATCH (e)-[r:RELATED_TO]-(n:Entity)
              WITH collect(DISTINCT e) AS selectedNodes,
                  collect(DISTINCT n) AS neighborNodes,
                  collect(DISTINCT r)[0..100] AS relatedEdges
              WITH selectedNodes + neighborNodes AS rawNodes, relatedEdges
              UNWIND rawNodes AS node
              WITH collect(DISTINCT node) AS allNodes, relatedEdges
              WITH allNodes[0..$max_nodes] AS allNodes, relatedEdges
              UNWIND allNodes AS node
              OPTIONAL MATCH (node)-[rel:RELATED_TO]-()
              WITH allNodes, relatedEdges, node, count(DISTINCT rel) AS degree
            OPTIONAL MATCH (node)<-[:CONTAINS_ENTITY]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
            WITH
                allNodes,
                relatedEdges,
                node,
                degree,
                collect(DISTINCT CASE WHEN d.id IS NOT NULL THEN {{doc_id: d.id, doc_name: coalesce(d.original_filename, d.filename)}} ELSE NULL END) AS docs
            WITH
                allNodes,
                relatedEdges,
                collect(DISTINCT {{
                    id: node.id,
                    label: node.name,
                    type: node.type,
                    community_id: node.community_id,
                    level: node.level,
                    degree: degree,
                    documents: [doc IN docs WHERE doc IS NOT NULL]
                }}) AS nodes
            WITH nodes, relatedEdges, [n IN nodes | n.id] AS nodeIds
            UNWIND relatedEdges AS rel
            WITH nodes, nodeIds, rel
            MATCH (s:Entity)-[rel]-(t:Entity)
            WHERE s.id IN nodeIds AND t.id IN nodeIds
            WITH nodes, rel, coalesce(rel.source_text_units, []) AS tus
            WITH nodes, rel, CASE WHEN size(tus) = 0 THEN [NULL] ELSE tus END AS textUnitIds
            UNWIND textUnitIds AS tu
            OPTIONAL MATCH (c:Chunk {{id: tu}})<-[:HAS_CHUNK]-(d:Document)
            WITH
                nodes,
                rel,
                collect(DISTINCT CASE WHEN tu IS NULL THEN NULL ELSE {{id: tu, doc_id: CASE WHEN d.id IS NOT NULL THEN d.id ELSE 'unknown' END, doc_name: CASE WHEN d.id IS NOT NULL THEN coalesce(d.original_filename, d.filename) ELSE 'unknown' END}} END) AS textUnits
            RETURN
                nodes,
                collect(DISTINCT {{
                    source: startNode(rel).id,
                    target: endNode(rel).id,
                    type: rel.type,
                    weight: coalesce(rel.strength, 0.5),
                    description: rel.description,
                    text_units: [tu IN textUnits WHERE tu IS NOT NULL]
                }})[0..200] AS edges
        """

        # Ensure DB connection is available before opening a session
        self.ensure_connected()

        with self.session_scope() as session:
            graph_result = session.run(query, **params).single()
            nodes = graph_result["nodes"] if graph_result else []
            edges = graph_result["edges"] if graph_result else []

            community_result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.community_id IS NOT NULL AND e.level IS NOT NULL
                RETURN DISTINCT e.community_id AS community_id, e.level AS level
                ORDER BY community_id ASC
                """
            ).data()

            node_types_result = session.run(
                """
                MATCH (e:Entity)
                RETURN DISTINCT e.type AS type
                ORDER BY type ASC
                """
            ).data()

            node_types = [record["type"] for record in node_types_result if record.get("type")]

        return {
            "nodes": nodes or [],
            "edges": edges or [],
            "communities": community_result or [],
            "node_types": node_types,
        }

    def count_document_entities(self, document_id: str, community_id: Optional[int] = None) -> int:
        """Count total number of unique entities for a document, optionally filtered by community."""
        with self.session_scope() as session:
            if community_id is not None:
                result = session.run(
                    """
                    MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                    WHERE e.community_id = $community_id
                    RETURN count(DISTINCT e) as entity_count
                    """,
                    document_id=document_id,
                    community_id=community_id
                ).single()
            else:
                result = session.run(
                    """
                    MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                    RETURN count(DISTINCT e) as entity_count
                    """,
                    document_id=document_id
                ).single()
            return result["entity_count"] if result else 0

    def get_entity_extraction_status(self) -> Dict[str, Any]:
        """Get entity extraction status for all documents."""
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
                WITH d, count(DISTINCT c) as total_chunks, count(DISTINCT e) as total_entities,
                     count(DISTINCT CASE WHEN e IS NOT NULL THEN c END) as chunks_with_entities
                RETURN d.id as document_id,
                       d.filename as filename,
                       total_chunks,
                       total_entities,
                       chunks_with_entities,
                       CASE
                           WHEN total_chunks = 0 THEN true
                           WHEN total_entities > 0 AND chunks_with_entities >= (total_chunks * 0.7) THEN true
                           ELSE false
                       END as entities_extracted
                ORDER BY d.filename ASC
                """
            )

            documents = [record.data() for record in result]

            # Calculate overall stats
            total_docs = len(documents)
            docs_with_entities = len([d for d in documents if d["entities_extracted"]])
            docs_without_entities = total_docs - docs_with_entities

            return {
                "documents": documents,
                "total_documents": total_docs,
                "documents_with_entities": docs_with_entities,
                "documents_without_entities": docs_without_entities,
                "all_extracted": docs_without_entities == 0,
            }

    def get_document_entities(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all entities extracted from a specific document."""
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                RETURN DISTINCT e.id as entity_id, e.name as name, e.type as type,
                       e.description as description, e.importance_score as importance_score,
                       count(DISTINCT c) as chunk_count
                ORDER BY e.importance_score DESC, e.name ASC
                """,
                doc_id=doc_id,
            )
            return [record.data() for record in result]

    def setup_indexes(self) -> None:
        """Create necessary indexes for performance."""
        with self.session_scope() as session:
            # Create indexes for faster lookups
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            
            # Create vector index for Entity embeddings
            # Note: 1536 dimensions for text-embedding-ada-002
            try:
                session.run(
                    "CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS "
                    "FOR (e:Entity) ON (e.embedding) "
                    "OPTIONS {indexConfig: { "
                    " `vector.dimensions`: 1536, "
                    " `vector.similarity_function`: 'cosine' "
                    "}}"
                )
                logger.info("Entity vector index created successfully")
            except Exception as e:
                logger.warning(f"Failed to create entity vector index (may already exist or not supported): {e}")
            
            # Index for incremental document updates (content hash matching)
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.content_hash)")

            # Create temporal indexes for performance
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:TimeNode) ON (t.date)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Date) ON (t.date)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Month) ON (t.year, t.month)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Quarter) ON (t.year, t.quarter)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Year) ON (t.year)")

            # Create memory system indexes (User, Fact, Conversation nodes)
            session.run("CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (f:Fact) ON (f.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (conv:Conversation) ON (conv.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (conv:Conversation) ON (conv.user_id, conv.created_at)")

            # Create fulltext index on chunk content for BM25-style keyword search
            try:
                session.run(
                    "CREATE FULLTEXT INDEX chunk_content_fulltext IF NOT EXISTS "
                    "FOR (c:Chunk) ON EACH [c.content]"
                )
                logger.info("Chunk fulltext index created successfully")
            except Exception as e:
                logger.warning(f"Failed to create chunk fulltext index (may already exist): {e}")

            logger.info("Database indexes created successfully (including temporal indexes)")

    def estimate_total_chunks(self) -> int:
        """Estimate total number of chunks in the database.

        Returns:
            Approximate count of chunks
        """
        try:
            with self.session_scope() as session:
                result = session.run(
                    "MATCH (:Chunk) RETURN count(*) AS total"
                ).single()
                return result["total"] if result else 0
        except Exception as e:
            logger.error(f"Failed to estimate chunk count: {e}")
            return 0

    def retrieve_chunks_by_ids_with_similarity(
        self,
        query_embedding: List[float],
        candidate_chunk_ids: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search on a filtered set of candidate chunks.

        This is used in two-stage retrieval where BM25 pre-filters candidates,
        and then vector search runs only on those candidates for efficiency.

        Args:
            query_embedding: Query embedding vector
            candidate_chunk_ids: List of chunk IDs to search within
            top_k: Maximum number of results to return

        Returns:
            List of chunks with similarity scores, sorted by score descending
        """
        if not candidate_chunk_ids:
            return []

        try:
            with self.session_scope() as session:
                result = session.run(
                    """
                    MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                    WHERE c.id IN $candidate_ids
                    WITH c, d, gds.similarity.cosine(c.embedding, $query_embedding) AS similarity
                    RETURN c.id as chunk_id, c.content as content, similarity,
                           coalesce(d.original_filename, d.filename) as document_name, d.id as document_id
                    ORDER BY similarity DESC
                    LIMIT $top_k
                    """,
                    query_embedding=query_embedding,
                    candidate_ids=candidate_chunk_ids,
                    top_k=top_k,
                )
                return [record.data() for record in result]
        except Exception as e:
            # Fallback to Python-side cosine calculation if GDS unavailable
            msg = str(e).lower()
            if "unknown function 'gds.similarity.cosine'" in msg or 'gds.similarity.cosine' in msg:
                logger.warning("GDS cosine function unavailable; falling back to Python cosine computation: %s", e)

                # Query only the candidate chunks
                with self.session_scope() as session:
                    result = session.run(
                        """
                        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                        WHERE c.id IN $candidate_ids AND c.embedding IS NOT NULL
                        RETURN c.id as chunk_id, c.content as content, c.embedding as embedding,
                               coalesce(d.original_filename, d.filename) as document_name, d.id as document_id
                        """,
                        candidate_ids=candidate_chunk_ids,
                    )
                    candidates = [record.data() for record in result]

                # Compute cosine similarity in Python
                def _cosine(a: List[float], b: List[float]) -> float:
                    try:
                        dot = 0.0
                        na = 0.0
                        nb = 0.0
                        for x, y in zip(a, b):
                            dot += x * y
                            na += x * x
                            nb += y * y
                        if na == 0.0 or nb == 0.0:
                            return 0.0
                        return dot / ((na ** 0.5) * (nb ** 0.5))
                    except Exception:
                        return 0.0

                scored = []
                for row in candidates:
                    emb = row.get("embedding")
                    if not emb:
                        continue
                    try:
                        sim = _cosine(query_embedding, emb)
                    except Exception:
                        sim = 0.0
                    scored.append({
                        "chunk_id": row.get("chunk_id"),
                        "content": row.get("content"),
                        "similarity": sim,
                        "document_name": row.get("document_name"),
                        "document_id": row.get("document_id"),
                    })

                scored.sort(key=lambda r: r.get("similarity", 0.0), reverse=True)
                return scored[:top_k]
            # Re-raise for other exceptions
            raise

    def chunk_keyword_search(
        self,
        query: str,
        top_k: int = 10,
        allowed_document_ids: Optional[List[str]] = None,
        fuzzy_distance: int = 0,
    ) -> List[Dict[str, Any]]:
        """Perform BM25-style keyword search on chunk content using fulltext index.

        Args:
            query: Search query text
            top_k: Maximum number of results to return
            allowed_document_ids: Optional list of document IDs to restrict search
            fuzzy_distance: Edit distance for fuzzy matching (0=exact, 1-2=fuzzy).
                          Neo4j supports fuzzy with ~ operator (term~1 or term~2)

        Returns:
            List of chunks with keyword match scores
        """
        try:
            # Apply fuzzy matching if requested
            search_query = query
            if fuzzy_distance > 0:
                # Transform query to add fuzzy operator to each term
                # Example: "authentication system" -> "authentication~2 system~2"
                terms = query.split()
                fuzzy_terms = [f"{term}~{fuzzy_distance}" for term in terms]
                search_query = " ".join(fuzzy_terms)
                logger.debug(f"Fuzzy search: '{query}' -> '{search_query}'")

            with self.session_scope() as session:
                # Build Cypher query with optional document filter
                if allowed_document_ids:
                    cypher = """
                    CALL db.index.fulltext.queryNodes('chunk_content_fulltext', $query)
                    YIELD node, score
                    MATCH (d:Document)-[:HAS_CHUNK]->(node)
                    WHERE d.id IN $allowed_doc_ids
                    RETURN node.id AS chunk_id,
                           node.content AS content,
                           node.chunk_index AS chunk_index,
                           d.id AS document_id,
                           d.filename AS document_name,
                           d.original_filename AS filename,
                           score AS keyword_score
                    ORDER BY score DESC
                    LIMIT $limit
                    """
                    result = session.run(
                        cypher,
                        query=search_query,
                        allowed_doc_ids=allowed_document_ids,
                        limit=top_k,
                    )
                else:
                    cypher = """
                    CALL db.index.fulltext.queryNodes('chunk_content_fulltext', $query)
                    YIELD node, score
                    MATCH (d:Document)-[:HAS_CHUNK]->(node)
                    RETURN node.id AS chunk_id,
                           node.content AS content,
                           node.chunk_index AS chunk_index,
                           d.id AS document_id,
                           d.filename AS document_name,
                           d.original_filename AS filename,
                           score AS keyword_score
                    ORDER BY score DESC
                    LIMIT $limit
                    """
                    result = session.run(cypher, query=search_query, limit=top_k)

                chunks = []
                for record in result:
                    chunk = {
                        "chunk_id": record["chunk_id"],
                        "content": record["content"],
                        "chunk_index": record["chunk_index"],
                        "document_id": record["document_id"],
                        "document_name": record["document_name"],
                        "filename": record["filename"],
                        "keyword_score": float(record["keyword_score"]),
                    }
                    chunks.append(chunk)

                logger.info(
                    "Keyword search returned %d chunks (restricted=%s)",
                    len(chunks),
                    bool(allowed_document_ids),
                )
                return chunks

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    # Entity-related methods

    async def acreate_entity_node(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str,
        importance_score: float = 0.5,
        source_chunks: Optional[List[str]] = None,
        source_text_units: Optional[List[str]] = None,
    ) -> None:
        """Create an entity node in the graph with embedding (async version)."""
        if source_chunks is None:
            source_chunks = []
        if source_text_units is None:
            source_text_units = list(source_chunks)

        # Generate embedding for the entity using name and description
        if getattr(settings, "skip_entity_embeddings", False):
            embedding = []  # store empty embedding for skipped mode
        else:
            entity_text = f"{name}: {description}"
            embedding = await embedding_manager.aget_embedding(entity_text)

        loop = asyncio.get_running_loop()
        try:
            executor = get_blocking_executor()
            await loop.run_in_executor(
                executor,
                self._create_entity_node_sync,
                entity_id,
                name,
                entity_type,
                description,
                importance_score,
                source_chunks,
                source_text_units,
                embedding,
            )
        except RuntimeError as e:
            logger.debug(
                f"Blocking executor unavailable while creating entity {entity_id}: {e}."
            )
            if SHUTTING_DOWN:
                logger.info(
                    "Process shutting down; aborting create_entity_node for %s",
                    entity_id,
                )
                return
            try:
                executor = get_blocking_executor()
                await loop.run_in_executor(
                    executor,
                    self._create_entity_node_sync,
                    entity_id,
                    name,
                    entity_type,
                    description,
                    importance_score,
                    source_chunks,
                    source_text_units,
                    embedding,
                )
            except Exception as e2:
                logger.error(
                    f"Failed to schedule create_entity_node for {entity_id}: {e2}"
                )

    def _create_entity_node_sync(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str,
        importance_score: float,
        source_chunks: List[str],
        source_text_units: List[str],
        embedding: List[float],
    ) -> None:
        """Synchronous helper for creating entity node in database."""
        with self.session_scope() as session:
            session.run(
                """
                MERGE (e:Entity {id: $entity_id})
                SET e.name = $name,
                    e.type = $entity_type,
                    e.description = $description,
                    e.importance_score = $importance_score,
                    e.source_chunks = $source_chunks,
                    e.source_text_units = $source_text_units,
                    e.embedding = $embedding,
                    e.updated_at = timestamp()
                """,
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                description=description,
                importance_score=importance_score,
                source_chunks=source_chunks,
                source_text_units=source_text_units,
                embedding=embedding,
            )

    def create_entity_node(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str,
        importance_score: float = 0.5,
        source_chunks: Optional[List[str]] = None,
        source_text_units: Optional[List[str]] = None,
    ) -> None:
        """Create an entity node in the graph with embedding (sync version kept for compatibility)."""
        if source_chunks is None:
            source_chunks = []
        if source_text_units is None:
            source_text_units = list(source_chunks)

        # Generate embedding for the entity using name and description
        if getattr(settings, "skip_entity_embeddings", False):
            embedding = []
        else:
            entity_text = f"{name}: {description}"
            embedding = embedding_manager.get_embedding(entity_text)

        self._create_entity_node_sync(
            entity_id,
            name,
            entity_type,
            description,
            importance_score,
            source_chunks,
            source_text_units,
            embedding,
        )

    async def aupdate_entities_with_embeddings(self) -> int:
        """Update existing entities that don't have embeddings (async version)."""
        with self.session_scope() as session:
            # Get entities without embeddings
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.embedding IS NULL
                RETURN e.id as entity_id, e.name as name, e.description as description
                """
            )

            entities_to_update = [
                (record["entity_id"], record["name"], record["description"])
                for record in result
            ]

            logger.info(f"Found {len(entities_to_update)} entities without embeddings")

            if not entities_to_update:
                return 0

            # Process entities with parallel embedding generation
            updated_count = 0
            concurrency = getattr(settings, "embedding_concurrency")
            sem = asyncio.Semaphore(concurrency)

            async def _embed_and_update_entity(entity_data):
                nonlocal updated_count
                entity_id, name, description = entity_data

                async with sem:
                    try:
                        # Add small delay to prevent API flooding
                        await asyncio.sleep(0.3)
                        entity_text = f"{name}: {description}" if description else name
                        embedding = await embedding_manager.aget_embedding(entity_text)
                    except Exception as e:
                        logger.error(
                            f"Async embedding failed for entity {entity_id}: {e}"
                        )
                        return None

                # Persist to DB in a thread to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                try:
                    executor = get_blocking_executor()
                    await loop.run_in_executor(
                        executor,
                        self._update_entity_embedding_sync,
                        entity_id,
                        embedding,
                    )
                    updated_count += 1

                    if updated_count % 100 == 0:
                        logger.info(
                            f"Updated {updated_count}/{len(entities_to_update)} entities with embeddings"
                        )
                    return entity_id
                except RuntimeError as e:
                    logger.debug(
                        f"Blocking executor unavailable while updating entity {entity_id}: {e}."
                    )
                    if SHUTTING_DOWN:
                        logger.info(
                            "Process shutting down; aborting update for %s",
                            entity_id,
                        )
                        return None
                    try:
                        executor = get_blocking_executor()
                        await loop.run_in_executor(
                            executor,
                            self._update_entity_embedding_sync,
                            entity_id,
                            embedding,
                        )
                        updated_count += 1
                        return entity_id
                    except Exception as e2:
                        logger.error(
                            f"Failed to update entity {entity_id} with embedding: {e2}"
                        )
                        return None

            tasks = [
                asyncio.create_task(_embed_and_update_entity(entity))
                for entity in entities_to_update
            ]

            for coro in asyncio.as_completed(tasks):
                try:
                    await coro
                except Exception as e:
                    logger.error(f"Error in entity update task: {e}")

            logger.info(
                f"Successfully updated {updated_count} entities with embeddings"
            )
            return updated_count

    def _update_entity_embedding_sync(
        self, entity_id: str, embedding: List[float]
    ) -> None:
        """Synchronous helper for updating entity embedding in database."""
        with self.session_scope() as session:
            session.run(
                """
                MATCH (e:Entity {id: $entity_id})
                SET e.embedding = $embedding
                """,
                entity_id=entity_id,
                embedding=embedding,
            )

    def update_entities_with_embeddings(self) -> int:
        """Update existing entities that don't have embeddings (sync version kept for compatibility)."""
        updated_count = 0

        with self.session_scope() as session:
            # Get entities without embeddings
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.embedding IS NULL
                RETURN e.id as entity_id, e.name as name, e.description as description
                """
            )

            entities_to_update = [
                (record["entity_id"], record["name"], record["description"])
                for record in result
            ]

            logger.info(f"Found {len(entities_to_update)} entities without embeddings")

            # Update entities with embeddings
            for entity_id, name, description in entities_to_update:
                try:
                    entity_text = f"{name}: {description}" if description else name
                    embedding = embedding_manager.get_embedding(entity_text)

                    session.run(
                        """
                        MATCH (e:Entity {id: $entity_id})
                        SET e.embedding = $embedding
                        """,
                        entity_id=entity_id,
                        embedding=embedding,
                    )
                    updated_count += 1

                    if updated_count % 100 == 0:
                        logger.info(
                            f"Updated {updated_count}/{len(entities_to_update)} entities with embeddings"
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to update entity {entity_id} with embedding: {e}"
                    )

            logger.info(
                f"Successfully updated {updated_count} entities with embeddings"
            )
            return updated_count

    def create_entity_relationship(
        self,
        entity_id1: str,
        entity_id2: str,
        relationship_type: str,
        description: str,
        strength: float = 0.5,
        source_chunks: Optional[List[str]] = None,
        source_text_units: Optional[List[str]] = None,
    ) -> None:
        """Create a relationship between two entities."""
        if source_chunks is None:
            source_chunks = []
        if source_text_units is None:
            source_text_units = list(source_chunks)

        with self.session_scope() as session:
            session.run(
                """
                MATCH (e1:Entity {id: $entity_id1})
                MATCH (e2:Entity {id: $entity_id2})
                MERGE (e1)-[r:RELATED_TO]-(e2)
                SET r.type = $relationship_type,
                    r.description = $description,
                    r.strength = $strength,
                    r.source_chunks = $source_chunks,
                    r.source_text_units = $source_text_units,
                    r.updated_at = timestamp()
                """,
                entity_id1=entity_id1,
                entity_id2=entity_id2,
                relationship_type=relationship_type,
                description=description,
                strength=strength,
                source_chunks=source_chunks,
                source_text_units=source_text_units,
            )

    def create_chunk_entity_relationship(self, chunk_id: str, entity_id: str) -> None:
        """Create a relationship between a chunk and an entity it contains."""
        with self.session_scope() as session:
            session.run(
                """
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (e:Entity {id: $entity_id})
                MERGE (c)-[:CONTAINS_ENTITY]->(e)
                """,
                chunk_id=chunk_id,
                entity_id=entity_id,
            )

    def repair_contains_entity_relationships_for_document(self, document_id: str) -> Dict[str, int]:
        """Ensure CONTAINS_ENTITY relationships exist for entities referencing chunks of a document.

        This method is intended to be idempotent and scoped to a single document. It computes
        the number of CONTAINS_ENTITY relationships before and after performing MERGE operations
        for any entity `source_chunks` that reference chunks belonging to the given document.

        Returns a dict with `before`, `after`, and `created` counts.
        """
        with self.session_scope() as session:
            before_rec = session.run(
                "MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)-[r:CONTAINS_ENTITY]->(e:Entity) RETURN count(r) as cnt",
                document_id=document_id,
            ).single()
            before = before_rec["cnt"] if before_rec else 0

            # Create missing relationships by matching entity.source_chunks against the document's chunk ids
            session.run(
                """
                MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
                WITH collect(c.id) as doc_chunk_ids
                MATCH (e:Entity)
                WHERE e.source_chunks IS NOT NULL AND size([x IN e.source_chunks WHERE x IN doc_chunk_ids]) > 0
                UNWIND [x IN e.source_chunks WHERE x IN doc_chunk_ids] as matched_chunk_id
                MATCH (c2:Chunk {id: matched_chunk_id})
                MERGE (c2)-[:CONTAINS_ENTITY]->(e)
                """,
                document_id=document_id,
            )

            after_rec = session.run(
                "MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)-[r:CONTAINS_ENTITY]->(e:Entity) RETURN count(r) as cnt",
                document_id=document_id,
            ).single()
            after = after_rec["cnt"] if after_rec else 0

        created = max(0, after - before)
        return {"before": before, "after": after, "created": created}

    def execute_batch_unwind(
        self,
        entity_query: str,
        entity_params: Dict[str, Any],
        rel_query: str,
        rel_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute batch UNWIND queries for entities and relationships (Phase 2).
        
        This method supports Phase 2 NetworkX implementation by executing
        batch INSERT operations using Neo4j's UNWIND clause.
        
        Args:
            entity_query: UNWIND query for entities
            entity_params: Parameters with $entities array
            rel_query: UNWIND query for relationships
            rel_params: Parameters with $relationships array
        
        Returns:
            Dict with execution metrics (entities_created, relationships_created, batches)
        
        Raises:
            Exception: On query execution failure
        """
        try:
            entity_count = len(entity_params.get("entities", []))
            rel_count = len(rel_params.get("relationships", []))
            
            logger.info(f"Executing batch UNWIND: {entity_count} entities, {rel_count} relationships")
            
            # Check if we need to split batches
            if entity_count > settings.neo4j_unwind_batch_size or rel_count > settings.neo4j_unwind_batch_size:
                logger.warning(
                    f"Batch size exceeds limit ({settings.neo4j_unwind_batch_size}), "
                    f"splitting into smaller batches..."
                )
                return self._execute_batch_split(
                    entity_query, entity_params, rel_query, rel_params
                )
            
            # Execute in single transaction using the resilient session_scope
            try:
                with self.session_scope() as session:
                    # Insert entities
                    entity_result = session.run(entity_query, entity_params)
                    entity_summary = entity_result.consume()
                    entities_created = entity_summary.counters.nodes_created

                    # Insert relationships
                    rel_result = session.run(rel_query, rel_params)
                    rel_summary = rel_result.consume()
                    relationships_created = rel_summary.counters.relationships_created
            except Exception as e:
                # As a last-resort, attempt to re-establish connectivity and retry once
                logger.warning("Batch UNWIND initial attempt failed, retrying after ensure_connected(): %s", e)
                try:
                    self.ensure_connected()
                    with self.session_scope() as session:
                        entity_result = session.run(entity_query, entity_params)
                        entity_summary = entity_result.consume()
                        entities_created = entity_summary.counters.nodes_created

                        rel_result = session.run(rel_query, rel_params)
                        rel_summary = rel_result.consume()
                        relationships_created = rel_summary.counters.relationships_created
                except Exception:
                    logger.error("Batch UNWIND retry failed: %s", e)
                    raise
            
            logger.info(
                f"Batch UNWIND complete: {entities_created} entities, "
                f"{relationships_created} relationships created"
            )
            
            return {
                "entities_created": entities_created,
                "relationships_created": relationships_created,
                "batches": 1,
                "status": "success"
            }
        
        except Exception as e:
            logger.error(f"Batch UNWIND failed: {e}")
            raise
    
    def _execute_batch_split(
        self,
        entity_query: str,
        entity_params: Dict[str, Any],
        rel_query: str,
        rel_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Split large batches into chunks for Neo4j parameter size limits.
        
        Args:
            entity_query: UNWIND query for entities
            entity_params: Parameters with $entities array
            rel_query: UNWIND query for relationships
            rel_params: Parameters with $relationships array
        
        Returns:
            Dict with execution metrics
        """
        batch_size = settings.neo4j_unwind_batch_size
        entities = entity_params.get("entities", [])
        relationships = rel_params.get("relationships", [])
        
        total_entities_created = 0
        total_relationships_created = 0
        batch_count = 0
        
        # Split entities into batches
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            with self.session_scope() as session:
                result = session.run(entity_query, {"entities": batch})
                summary = result.consume()
                total_entities_created += summary.counters.nodes_created
                batch_count += 1
            logger.debug(f"Entity batch {batch_count}: {len(batch)} entities")
        
        # Split relationships into batches
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]
            with self.session_scope() as session:
                result = session.run(rel_query, {"relationships": batch})
                summary = result.consume()
                total_relationships_created += summary.counters.relationships_created
                batch_count += 1
            logger.debug(f"Relationship batch {batch_count}: {len(batch)} relationships")
        
        logger.info(
            f"Batch UNWIND (split) complete: {total_entities_created} entities, "
            f"{total_relationships_created} relationships in {batch_count} batches"
        )
        
        return {
            "entities_created": total_entities_created,
            "relationships_created": total_relationships_created,
            "batches": batch_count,
            "status": "success"
        }

    def get_entities_by_type(
        self, entity_type: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get entities of a specific type."""
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (e:Entity {type: $entity_type})
                RETURN e.id as entity_id, e.name as name, e.description as description,
                       e.importance_score as importance_score, e.source_chunks as source_chunks
                ORDER BY e.importance_score DESC
                LIMIT $limit
                """,
                entity_type=entity_type,
                limit=limit,
            )
            return [record.data() for record in result]

    def get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for a specific entity."""
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (e1:Entity {id: $entity_id})-[r:RELATED_TO]-(e2:Entity)
                RETURN e2.id as related_entity_id,
                       e2.type as related_entity_type, r.type as relationship_type,
                       r.description as relationship_description, r.strength as strength
                ORDER BY r.strength DESC
                """,
                entity_id=entity_id,
            )
            relationships = []
            for record in result:
                rel_data = dict(record)
                # Use cached label lookup instead of query result
                rel_data["related_entity_name"] = self.get_entity_label_cached(
                    record["related_entity_id"]
                )
                relationships.append(rel_data)
            return relationships

    def entity_similarity_search(
        self, query_text: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search entities by text similarity using full-text search."""
        with self.session_scope() as session:
            # Create full-text index if it doesn't exist
            try:
                session.run(
                    "CREATE FULLTEXT INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]"
                )
            except Exception:
                pass  # Index might already exist

            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('entity_text', $query_text)
                YIELD node, score
                RETURN node.id as entity_id, node.name as name, node.type as type,
                       node.description as description, node.importance_score as importance_score,
                       score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                query_text=query_text,
                top_k=top_k,
            )
            return [record.data() for record in result]

    def get_entities_for_chunks(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Get all entities contained in the specified chunks."""
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                WHERE c.id IN $chunk_ids
                RETURN DISTINCT e.id as entity_id, e.name as name, e.type as type,
                       e.description as description, e.importance_score as importance_score,
                       collect(c.id) as source_chunks
                """,
                chunk_ids=chunk_ids,
            )
            return [record.data() for record in result]

    def get_chunks_for_entities(self, entity_ids: List[str]) -> List[Dict[str, Any]]:
        """Get all chunks that contain the specified entities."""
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                WHERE e.id IN $entity_ids
                OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
                RETURN DISTINCT c.id as chunk_id, c.content as content,
                       coalesce(d.original_filename, d.filename) as document_name, d.id as document_id,
                       collect(e.name) as contained_entities
                """,
                entity_ids=entity_ids,
            )
            return [record.data() for record in result]

    def get_entity_graph_neighborhood(
        self, entity_id: str, max_depth: int = 2, max_entities: int = 50
    ) -> Dict[str, Any]:
        """Get a subgraph around a specific entity."""
        with self.session_scope() as session:
            if max_depth == 1:
                query = """
                    MATCH (start:Entity {id: $entity_id})-[r:RELATED_TO]-(related:Entity)
                    RETURN collect(DISTINCT start) + collect(DISTINCT related) as entities,
                           collect({
                               start: startNode(r).id,
                               end: endNode(r).id,
                               type: r.type,
                               description: r.description,
                               strength: r.strength
                           }) as relationships
                    """
            else:
                query = """
                    MATCH (start:Entity {id: $entity_id})-[*1..2]-(related:Entity)
                    WITH start, related
                    MATCH (e1:Entity)-[r:RELATED_TO]-(e2:Entity)
                    WHERE (e1.id = start.id OR e1.id = related.id)
                      AND (e2.id = start.id OR e2.id = related.id)
                    RETURN collect(DISTINCT start) + collect(DISTINCT related) as entities,
                           collect(DISTINCT {
                               start: startNode(r).id,
                               end: endNode(r).id,
                               type: r.type,
                               description: r.description,
                               strength: r.strength
                           }) as relationships
                    """

            result = session.run(query, entity_id=entity_id)
            record = result.single()
            if record:
                entities = []
                for node in record["entities"]:
                    entity_data = {
                        "id": node["id"],
                        "name": self.get_entity_label_cached(node["id"]),  # Use cached lookup
                        "type": node["type"],
                        "description": node["description"],
                        "importance_score": node["importance_score"],
                    }
                    entities.append(entity_data)
                return {"entities": entities, "relationships": record["relationships"]}
            return {"entities": [], "relationships": []}

    def validate_chunk_embeddings(self, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate embeddings for chunks, checking for empty/invalid embeddings.

        Args:
            doc_id: Optional document ID to validate only specific document chunks

        Returns:
            Dictionary with validation results
        """
        with self.session_scope() as session:
            if doc_id:
                query = """
                    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                    RETURN c.id as chunk_id, c.embedding as embedding, c.content as content
                """
                result = session.run(query, doc_id=doc_id)
            else:
                query = """
                    MATCH (c:Chunk)
                    RETURN c.id as chunk_id, c.embedding as embedding, c.content as content
                """
                result = session.run(query)

            total_chunks = 0
            invalid_chunks = []
            empty_embeddings = 0
            wrong_size_embeddings = 0

            # Detect embedding size from existing embeddings instead of hardcoding
            expected_embedding_size = None

            for record in result:
                total_chunks += 1
                chunk_id = record["chunk_id"]
                embedding = record["embedding"]
                content = record["content"]

                # Check for empty or None embeddings
                if not embedding:
                    empty_embeddings += 1
                    invalid_chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "issue": "empty_embedding",
                            "content_preview": (
                                content[:100] + "..." if len(content) > 100 else content
                            ),
                        }
                    )
                    continue

                # Detect expected embedding size from first valid embedding
                if expected_embedding_size is None and embedding:
                    expected_embedding_size = len(embedding)
                    logger.info(f"Detected embedding size: {expected_embedding_size}")

                # Check embedding size consistency (only flag if significantly different)
                if (
                    expected_embedding_size
                    and embedding
                    and len(embedding) != expected_embedding_size
                ):
                    wrong_size_embeddings += 1
                    invalid_chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "issue": f"wrong_size_{len(embedding)}_expected_{expected_embedding_size}",
                            "content_preview": (
                                content[:100] + "..." if len(content) > 100 else content
                            ),
                        }
                    )

            validation_results = {
                "total_chunks": total_chunks,
                "valid_chunks": total_chunks - len(invalid_chunks),
                "invalid_chunks": len(invalid_chunks),
                "empty_embeddings": empty_embeddings,
                "wrong_size_embeddings": wrong_size_embeddings,
                "invalid_chunk_details": invalid_chunks,
                "validation_passed": len(invalid_chunks) == 0,
            }

            logger.info(
                f"Chunk embedding validation: {validation_results['valid_chunks']}/{total_chunks} valid"
            )
            if invalid_chunks:
                logger.warning(f"Found {len(invalid_chunks)} invalid chunk embeddings")

            return validation_results

    def validate_entity_embeddings(
        self, doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate embeddings for entities, checking for empty/invalid embeddings.

        Args:
            doc_id: Optional document ID to validate only entities from specific document

        Returns:
            Dictionary with validation results
        """
        with self.session_scope() as session:
            if doc_id:
                query = """
                    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                    RETURN DISTINCT e.id as entity_id, e.embedding as embedding, e.name as entity_name
                """
                result = session.run(query, doc_id=doc_id)
            else:
                query = """
                    MATCH (e:Entity)
                    RETURN e.id as entity_id, e.embedding as embedding, e.name as entity_name
                """
                result = session.run(query)

            total_entities = 0
            invalid_entities = []
            empty_embeddings = 0
            wrong_size_embeddings = 0
            no_embeddings = 0  # Entities may not have embeddings by design

            # Detect embedding size from existing embeddings instead of hardcoding
            expected_embedding_size = None

            for record in result:
                total_entities += 1
                entity_id = record["entity_id"]
                embedding = record["embedding"]
                entity_name = record["entity_name"]

                # Skip entities without embeddings (they may be designed this way)
                if embedding is None:
                    no_embeddings += 1
                    continue

                # Check for empty embeddings
                if not embedding:
                    empty_embeddings += 1
                    invalid_entities.append(
                        {
                            "entity_id": entity_id,
                            "issue": "empty_embedding",
                            "entity_name": entity_name,
                        }
                    )
                    continue

                # Detect expected embedding size from first valid embedding
                if expected_embedding_size is None and embedding:
                    expected_embedding_size = len(embedding)
                    logger.info(
                        f"Detected entity embedding size: {expected_embedding_size}"
                    )

                # Check embedding size consistency (only flag if significantly different)
                if (
                    expected_embedding_size
                    and embedding
                    and len(embedding) != expected_embedding_size
                ):
                    wrong_size_embeddings += 1
                    invalid_entities.append(
                        {
                            "entity_id": entity_id,
                            "issue": f"wrong_size_{len(embedding)}_expected_{expected_embedding_size}",
                            "entity_name": entity_name,
                        }
                    )

            validation_results = {
                "total_entities": total_entities,
                "entities_with_embeddings": total_entities - no_embeddings,
                "valid_embeddings": (total_entities - no_embeddings)
                - len(invalid_entities),
                "invalid_embeddings": len(invalid_entities),
                "empty_embeddings": empty_embeddings,
                "wrong_size_embeddings": wrong_size_embeddings,
                "no_embeddings": no_embeddings,
                "invalid_entity_details": invalid_entities,
                "validation_passed": len(invalid_entities) == 0,
            }

            logger.info(
                f"Entity embedding validation: {validation_results['valid_embeddings']}/{validation_results['entities_with_embeddings']} valid"
            )
            if invalid_entities:
                logger.warning(
                    f"Found {len(invalid_entities)} invalid entity embeddings"
                )

            return validation_results

    async def afix_invalid_embeddings(
        self,
        chunk_ids: Optional[List[str]] = None,
        entity_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fix invalid embeddings by regenerating them (async version).

        Args:
            chunk_ids: List of chunk IDs to fix (if None, fixes all invalid chunk embeddings)
            entity_ids: List of entity IDs to fix (if None, fixes all invalid entity embeddings)

        Returns:
            Dictionary with fix results
        """
        results = {"chunks_fixed": 0, "entities_fixed": 0, "errors": []}

        # Fix chunk embeddings in parallel
        if chunk_ids is not None:
            concurrency = getattr(settings, "embedding_concurrency")
            sem = asyncio.Semaphore(concurrency)

            async def _fix_chunk_embedding(chunk_id):
                async with sem:
                    try:
                        # Get chunk content in executor to avoid blocking
                        loop = asyncio.get_running_loop()
                        try:
                            executor = get_blocking_executor()
                            content = await loop.run_in_executor(
                                executor,
                                self._get_chunk_content_sync,
                                chunk_id,
                            )
                        except RuntimeError as e:
                            logger.debug(
                                f"Blocking executor unavailable while fetching chunk content {chunk_id}: {e}."
                            )
                            if SHUTTING_DOWN:
                                logger.info(
                                    "Process shutting down; aborting fix for chunk %s",
                                    chunk_id,
                                )
                                return False
                            try:
                                executor = get_blocking_executor()
                                content = await loop.run_in_executor(
                                    executor,
                                    self._get_chunk_content_sync,
                                    chunk_id,
                                )
                            except Exception as e2:
                                logger.error(
                                    f"Failed to fetch chunk content {chunk_id}: {e2}"
                                )
                                return False

                        if content:
                            # Add small delay to prevent API flooding
                            await asyncio.sleep(0.3)
                            # Generate new embedding
                            embedding = await embedding_manager.aget_embedding(content)

                            # Update chunk with new embedding in executor
                            try:
                                executor = get_blocking_executor()
                                await loop.run_in_executor(
                                    executor,
                                    self._update_chunk_embedding_sync,
                                    chunk_id,
                                    embedding,
                                )
                            except RuntimeError as e:
                                logger.debug(
                                    f"Blocking executor unavailable while updating chunk {chunk_id}: {e}."
                                )
                                if SHUTTING_DOWN:
                                    logger.info(
                                        "Process shutting down; aborting update for chunk %s",
                                        chunk_id,
                                    )
                                    return False
                                try:
                                    executor = get_blocking_executor()
                                    await loop.run_in_executor(
                                        executor,
                                        self._update_chunk_embedding_sync,
                                        chunk_id,
                                        embedding,
                                    )
                                except Exception as e2:
                                    logger.error(
                                        f"Failed to update chunk {chunk_id} embedding: {e2}"
                                    )
                                    return False
                            results["chunks_fixed"] += 1
                            logger.info(f"Fixed embedding for chunk {chunk_id}")
                            return True
                    except Exception as e:
                        error_msg = f"Failed to fix embedding for chunk {chunk_id}: {e}"
                        results["errors"].append(error_msg)
                        logger.error(error_msg)
                        return False
                return False

            if chunk_ids:
                tasks = [
                    asyncio.create_task(_fix_chunk_embedding(chunk_id))
                    for chunk_id in chunk_ids
                ]

                for coro in asyncio.as_completed(tasks):
                    try:
                        await coro
                    except Exception as e:
                        logger.error(f"Error in chunk fix task: {e}")

        # Fix entity embeddings in parallel
        if entity_ids is not None:
            concurrency = getattr(settings, "embedding_concurrency")
            sem = asyncio.Semaphore(concurrency)

            async def _fix_entity_embedding(entity_id):
                async with sem:
                    try:
                        # Get entity data in executor to avoid blocking
                        loop = asyncio.get_running_loop()
                        try:
                            executor = get_blocking_executor()
                            entity_data = await loop.run_in_executor(
                                executor,
                                self._get_entity_data_sync,
                                entity_id,
                            )
                        except RuntimeError as e:
                            logger.debug(
                                f"Blocking executor unavailable while fetching entity data {entity_id}: {e}."
                            )
                            if SHUTTING_DOWN:
                                logger.info(
                                    "Process shutting down; aborting entity fix for %s",
                                    entity_id,
                                )
                                return False
                            try:
                                executor = get_blocking_executor()
                                entity_data = await loop.run_in_executor(
                                    executor,
                                    self._get_entity_data_sync,
                                    entity_id,
                                )
                            except Exception as e2:
                                logger.error(
                                    f"Failed to fetch entity data {entity_id}: {e2}"
                                )
                                return False

                        if entity_data:
                            # Add small delay to prevent API flooding
                            await asyncio.sleep(0.3)
                            # Use entity name + description for embedding
                            text = (
                                f"{entity_data['name']}: {entity_data['description']}"
                            )
                            # Generate new embedding
                            embedding = await embedding_manager.aget_embedding(text)

                            # Update entity with new embedding in executor
                            try:
                                executor = get_blocking_executor()
                                await loop.run_in_executor(
                                    executor,
                                    self._update_entity_embedding_sync,
                                    entity_id,
                                    embedding,
                                )
                            except RuntimeError as e:
                                logger.debug(
                                    f"Blocking executor unavailable while updating entity {entity_id}: {e}."
                                )
                                if SHUTTING_DOWN:
                                    logger.info(
                                        "Process shutting down; aborting update for entity %s",
                                        entity_id,
                                    )
                                    return False
                                try:
                                    executor = get_blocking_executor()
                                    await loop.run_in_executor(
                                        executor,
                                        self._update_entity_embedding_sync,
                                        entity_id,
                                        embedding,
                                    )
                                except Exception as e2:
                                    logger.error(
                                        f"Failed to update entity {entity_id} embedding: {e2}"
                                    )
                                    return False
                            results["entities_fixed"] += 1
                            logger.info(f"Fixed embedding for entity {entity_id}")
                            return True
                    except Exception as e:
                        error_msg = (
                            f"Failed to fix embedding for entity {entity_id}: {e}"
                        )
                        results["errors"].append(error_msg)
                        logger.error(error_msg)
                        return False
                return False

            if entity_ids:
                tasks = [
                    asyncio.create_task(_fix_entity_embedding(entity_id))
                    for entity_id in entity_ids
                ]

                for coro in asyncio.as_completed(tasks):
                    try:
                        await coro
                    except Exception as e:
                        logger.error(f"Error in entity fix task: {e}")

        return results

    def _get_chunk_content_sync(self, chunk_id: str) -> Optional[str]:
        """Synchronous helper for getting chunk content."""
        with self.session_scope() as session:
            result = session.run(
                "MATCH (c:Chunk {id: $chunk_id}) RETURN c.content as content",
                chunk_id=chunk_id,
            )
            record = result.single()
            return record["content"] if record else None

    def _update_chunk_embedding_sync(
        self, chunk_id: str, embedding: List[float]
    ) -> None:
        """Synchronous helper for updating chunk embedding."""
        with self.session_scope() as session:
            session.run(
                "MATCH (c:Chunk {id: $chunk_id}) SET c.embedding = $embedding",
                chunk_id=chunk_id,
                embedding=embedding,
            )

    def _get_entity_data_sync(self, entity_id: str) -> Optional[Dict[str, str]]:
        """Synchronous helper for getting entity data."""
        with self.session_scope() as session:
            result = session.run(
                "MATCH (e:Entity {id: $entity_id}) RETURN e.name as name, e.description as description",
                entity_id=entity_id,
            )
            record = result.single()
            return (
                {"name": record["name"], "description": record["description"]}
                if record
                else None
            )

    def fix_invalid_embeddings(
        self,
        chunk_ids: Optional[List[str]] = None,
        entity_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fix invalid embeddings by regenerating them (sync version kept for compatibility).

        Args:
            chunk_ids: List of chunk IDs to fix (if None, fixes all invalid chunk embeddings)
            entity_ids: List of entity IDs to fix (if None, fixes all invalid entity embeddings)

        Returns:
            Dictionary with fix results
        """
        results = {"chunks_fixed": 0, "entities_fixed": 0, "errors": []}

        # Fix chunk embeddings
        if chunk_ids is not None:
            with self.session_scope() as session:
                for chunk_id in chunk_ids:
                    try:
                        # Get chunk content
                        result = session.run(
                            "MATCH (c:Chunk {id: $chunk_id}) RETURN c.content as content",
                            chunk_id=chunk_id,
                        )
                        record = result.single()
                        if record:
                            content = record["content"]
                            # Generate new embedding
                            embedding = embedding_manager.get_embedding(content)
                            # Update chunk with new embedding
                            session.run(
                                "MATCH (c:Chunk {id: $chunk_id}) SET c.embedding = $embedding",
                                chunk_id=chunk_id,
                                embedding=embedding,
                            )
                            results["chunks_fixed"] += 1
                            logger.info(f"Fixed embedding for chunk {chunk_id}")
                    except Exception as e:
                        error_msg = f"Failed to fix embedding for chunk {chunk_id}: {e}"
                        results["errors"].append(error_msg)
                        logger.error(error_msg)

        # Fix entity embeddings (if they're supposed to have embeddings)
        if entity_ids is not None:
            with self.session_scope() as session:
                for entity_id in entity_ids:
                    try:
                        # Get entity name/description for embedding
                        result = session.run(
                            "MATCH (e:Entity {id: $entity_id}) RETURN e.name as name, e.description as description",
                            entity_id=entity_id,
                        )
                        record = result.single()
                        if record:
                            # Use entity name + description for embedding
                            text = f"{record['name']}: {record['description']}"
                            # Generate new embedding
                            embedding = embedding_manager.get_embedding(text)
                            # Update entity with new embedding
                            session.run(
                                "MATCH (e:Entity {id: $entity_id}) SET e.embedding = $embedding",
                                entity_id=entity_id,
                                embedding=embedding,
                            )
                            results["entities_fixed"] += 1
                            logger.info(f"Fixed embedding for entity {entity_id}")
                    except Exception as e:
                        error_msg = (
                            f"Failed to fix embedding for entity {entity_id}: {e}"
                        )
                        results["errors"].append(error_msg)
                        logger.error(error_msg)

        return results

    def find_scored_paths(
        self,
        seed_entity_ids: List[str],
        max_hops: int = 2,
        beam_size: int = 8,
        min_edge_strength: float = 0.0,
        node_filter: Optional[Callable[[Entity], bool]] = None,
    ) -> List[PathResult]:
        """
        Find scored paths from seed entities using beam search.

        Args:
            seed_entity_ids: Starting entity IDs for path traversal
            max_hops: Maximum number of hops to traverse
            beam_size: Number of best paths to keep at each depth
            min_edge_strength: Minimum relationship strength to follow
            node_filter: Optional filter function for entities

        Returns:
            List of PathResult objects sorted by score
        """
        if not seed_entity_ids:
            logger.warning("No seed entities provided for path search")
            return []

        try:
            with self.session_scope() as session:
                # Get seed entities with their data
                seed_entities_data = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.id IN $entity_ids
                    RETURN e.id as id, e.name as name, e.type as type,
                           e.description as description, e.importance_score as importance_score,
                           e.embedding as embedding
                    """,
                    entity_ids=seed_entity_ids,
                ).data()

                if not seed_entities_data:
                    logger.warning(f"No entities found for IDs: {seed_entity_ids}")
                    return []

                # Initialize paths with seed entities
                current_paths = []
                for entity_data in seed_entities_data:
                    entity = Entity(
                        id=entity_data["id"],
                        name=entity_data["name"],
                        type=entity_data["type"],
                        description=entity_data.get("description", ""),
                        importance_score=entity_data.get("importance_score", 0.5),
                        embedding=entity_data.get("embedding"),
                    )

                    # Apply node filter if provided
                    if node_filter and not node_filter(entity):
                        continue

                    # Start with single-entity paths
                    path = PathResult(
                        entities=[entity],
                        relationships=[],
                        score=entity.importance_score,
                        supporting_chunk_ids=[],
                    )
                    current_paths.append(path)

                # Perform beam search up to max_hops
                for hop in range(max_hops):
                    next_paths = []
                    visited_in_hop = set()  # Track what we've expanded this hop

                    for path in current_paths:
                        # Get the last entity in the path
                        last_entity = path.entities[-1]

                        # Skip if we've already expanded from this entity in this hop
                        path_key = (tuple(e.id for e in path.entities), hop)
                        if path_key in visited_in_hop:
                            continue
                        visited_in_hop.add(path_key)

                        # Get relationships from last entity
                        relationships_data = session.run(
                            """
                            MATCH (e1:Entity {id: $entity_id})-[r:RELATED_TO]-(e2:Entity)
                            WHERE r.strength >= $min_strength
                            AND NOT e2.id IN $visited_ids
                            RETURN e2.id as target_id,
                                   e2.type as target_type, e2.description as target_description,
                                   e2.importance_score as target_importance,
                                   e2.embedding as target_embedding,
                                   r.type as rel_type, r.description as rel_description,
                                   r.strength as rel_strength,
                                   coalesce(r.source_chunks, []) as source_chunks,
                                   startNode(r).id as source_id
                            ORDER BY r.strength DESC
                            LIMIT $limit
                            """,
                            entity_id=last_entity.id,
                            min_strength=min_edge_strength,
                            visited_ids=[e.id for e in path.entities],
                            limit=beam_size * 2,  # Get more candidates than beam size
                        ).data()

                        # Expand path with each relationship
                        for rel_data in relationships_data:
                            target_entity = Entity(
                                id=rel_data["target_id"],
                                name=self.get_entity_label_cached(rel_data["target_id"]),  # Use cached lookup
                                type=rel_data["target_type"],
                                description=rel_data.get("target_description", ""),
                                importance_score=rel_data.get("target_importance", 0.5),
                                embedding=rel_data.get("target_embedding"),
                            )

                            # Apply node filter if provided
                            if node_filter and not node_filter(target_entity):
                                continue

                            # Determine direction of relationship
                            if rel_data["source_id"] == last_entity.id:
                                source_id = last_entity.id
                                target_id = target_entity.id
                            else:
                                source_id = target_entity.id
                                target_id = last_entity.id

                            relationship = Relationship(
                                source_entity_id=source_id,
                                target_entity_id=target_id,
                                type=rel_data["rel_type"],
                                description=rel_data.get("rel_description", ""),
                                strength=rel_data.get("rel_strength", 0.5),
                                source_chunks=rel_data.get("source_chunks", []),
                                source_text_units=rel_data.get(
                                    "source_text_units",
                                    rel_data.get("source_chunks", []),
                                ),
                            )

                            # Calculate new path score
                            # Score = average of: path score, relationship strength, target importance
                            new_score = (
                                path.score * 0.5
                                + relationship.strength * 0.3
                                + target_entity.importance_score * 0.2
                            )

                            # Create new path
                            new_path = PathResult(
                                entities=path.entities + [target_entity],
                                relationships=path.relationships + [relationship],
                                score=new_score,
                                supporting_chunk_ids=path.supporting_chunk_ids
                                + [relationship.source_chunks],
                            )
                            next_paths.append(new_path)

                    # Apply beam search: keep only top beam_size paths
                    next_paths.sort(key=lambda p: p.score, reverse=True)
                    current_paths = next_paths[:beam_size]

                    # Stop if no more paths to expand
                    if not current_paths:
                        break

                # Return all paths sorted by score
                current_paths.sort(key=lambda p: p.score, reverse=True)
                logger.info(
                    f"Found {len(current_paths)} paths from {len(seed_entity_ids)} seed entities "
                    f"with max_hops={max_hops}, beam_size={beam_size}"
                )
                return current_paths

        except Exception as e:
            logger.error(f"Path finding failed: {e}")
            return []

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics for the API."""
        with self.session_scope() as session:
            # Get basic stats
            result = session.run(
                """
                MATCH (d:Document)
                WITH count(d) AS total_documents
                OPTIONAL MATCH (c:Chunk)
                WITH total_documents, count(c) AS total_chunks
                OPTIONAL MATCH (e:Entity)
                WITH total_documents, total_chunks, count(e) AS total_entities
                OPTIONAL MATCH ()-[r]->()
                RETURN total_documents, total_chunks, total_entities, count(r) AS total_relationships
                """
            )
            record = result.single()
            stats = record.data() if record else {}

            # Get document list
            doc_result = session.run(
                """
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                WITH d, count(c) as chunk_count
                RETURN d.id as document_id,
                       d.filename as filename,
                       coalesce(d.original_filename, d.filename) as original_filename,
                       d.created_at as created_at,
              coalesce(d.processing_status, 'idle') as processing_status,
              coalesce(d.processing_stage, 'idle') as processing_stage,
              coalesce(d.processing_progress, 0.0) as processing_progress,
              chunk_count,
              d.document_type as document_type
                ORDER BY d.created_at DESC
                """
            )
            documents = [record.data() for record in doc_result]

            return {
                "total_documents": stats.get("total_documents", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "total_entities": stats.get("total_entities", 0),
                "total_relationships": stats.get("total_relationships", 0),
                "documents": documents,
            }

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database."""
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                WITH d, count(c) as chunk_count
                RETURN d.id as document_id,
                       d.filename as filename,
                       coalesce(d.original_filename, d.filename) as original_filename,
                       d.created_at as created_at,
                       d.hashtags as hashtags,
                       chunk_count
                ORDER BY d.created_at DESC
                """
            )
            return [record.data() for record in result]

    def get_document_details(self, doc_id: str) -> Dict[str, Any]:
        """Retrieve detailed metadata for a document."""

        def _timestamp_to_iso(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
            if isinstance(value, str):
                try:
                    numeric = float(value)
                    return datetime.fromtimestamp(numeric, tz=timezone.utc).isoformat()
                except ValueError:
                    return value
            return str(value)

        with self.session_scope() as session:
            doc_record = session.run(
                """
                MATCH (d:Document {id: $doc_id})
                RETURN d
                """,
                doc_id=doc_id,
            ).single()

            if doc_record is None:
                raise ValueError("Document not found")

            doc_node = doc_record["d"]
            doc_data = dict(doc_node)

            chunk_records = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.id as id,
                       c.content as text,
                       c.chunk_index as index,
                       coalesce(c.offset, 0) as offset,
                       c.score as score
                ORDER BY coalesce(c.chunk_index, 0) ASC, c.id ASC
                """,
                doc_id=doc_id,
            )
            chunks = [
                {
                    "id": record["id"],
                    "text": record["text"] or "",
                    "index": record["index"],
                    "offset": record["offset"],
                    "score": record["score"],
                }
                for record in chunk_records
            ]

            entity_records = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                RETURN e.type as type,
                       e.name as text,
                       e.community_id as community_id,
                       e.level as level,
                       count(*) as count,
                       collect(DISTINCT c.chunk_index) as positions
                ORDER BY type ASC, text ASC
                """,
                doc_id=doc_id,
            )
            entities = [
                {
                    "type": record["type"],
                    "text": record["text"],
                    "community_id": record["community_id"],
                    "level": record.get("level"),
                    "count": record["count"],
                    "positions": [pos for pos in (record["positions"] or []) if pos is not None],
                }
                for record in entity_records
            ]

            related_records = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[r:RELATED_TO|SIMILAR_TO]-(other:Document)
                RETURN DISTINCT other.id as id,
                                other.filename as title,
                                coalesce(other.link, '') as link,
                                other.filename as filename
                ORDER BY other.filename ASC
                """,
                doc_id=doc_id,
            )
            related_documents = [
                {
                    "id": record["id"],
                    "title": record["title"] or record["filename"],
                    "link": record["link"] if record["link"] else None,
                }
                for record in related_records
            ]

            uploader_info: Optional[Dict[str, Any]] = None
            uploader_value = doc_data.get("uploader")
            if isinstance(uploader_value, dict):
                uploader_info = {
                    "id": uploader_value.get("id"),
                    "name": uploader_value.get("name"),
                }
            else:
                uploader_id = doc_data.get("uploader_id")
                uploader_name = doc_data.get("uploader_name")
                if uploader_id or uploader_name:
                    uploader_info = {"id": uploader_id, "name": uploader_name}

            uploaded_at = (
                _timestamp_to_iso(doc_data.get("uploaded_at"))
                or _timestamp_to_iso(doc_data.get("created_at"))
            )

            known_keys = {
                "id",
                "title",
                "filename",
                "original_filename",
                "mime_type",
                "preview_url",
                "uploaded_at",
                "created_at",
                "uploader",
                "uploader_id",
                "uploader_name",
                "quality_scores",
                "summary",
                "document_type",
                "hashtags",
            }

            metadata = {
                key: value
                for key, value in doc_data.items()
                if key not in known_keys
            }

            return {
                "id": doc_data.get("id", doc_id),
                "title": doc_data.get("title"),
                "file_name": doc_data.get("filename"),
                "original_filename": doc_data.get("original_filename"),
                "mime_type": doc_data.get("mime_type"),
                "preview_url": doc_data.get("preview_url"),
                "uploaded_at": uploaded_at,
                "uploader": uploader_info,
                "summary": doc_data.get("summary"),
                "document_type": doc_data.get("document_type"),
                "hashtags": doc_data.get("hashtags", []),
                "chunks": chunks,
                "entities": entities,
                "quality_scores": doc_data.get("quality_scores"),
                "related_documents": related_documents or None,
                "metadata": metadata or None,
            }

    def get_document_file_info(self, doc_id: str) -> Dict[str, Any]:
        """Return file metadata for previewing a document."""

        with self.session_scope() as session:
            record = session.run(
                """
                MATCH (d:Document {id: $doc_id})
                RETURN d.filename as file_name,
                       d.file_path as file_path,
                       d.mime_type as mime_type
                """,
                doc_id=doc_id,
            ).single()

            if record is None:
                raise ValueError("Document not found")

            file_path = record["file_path"]
            mime_type = record["mime_type"]
            if not mime_type and file_path:
                mime_type = mimetypes.guess_type(file_path)[0]

            return {
                "file_name": record["file_name"],
                "file_path": file_path,
                "mime_type": mime_type,
            }

    # ============================================================
    # Memory System: User Management
    # ============================================================

    def create_user(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new User node.

        Args:
            user_id: Unique user identifier
            metadata: Optional metadata (name, email, preferences, etc.)

        Returns:
            User data including id, created_at, and metadata
        """
        with self.session_scope() as session:
            result = session.run(
                """
                MERGE (u:User {id: $user_id})
                ON CREATE SET
                    u.created_at = datetime(),
                    u.metadata = $metadata
                ON MATCH SET
                    u.metadata = $metadata
                RETURN u.id as user_id, u.created_at as created_at, u.metadata as metadata
                """,
                user_id=user_id,
                metadata=metadata or {},
            ).single()

            if result:
                return {
                    "user_id": result["user_id"],
                    "created_at": result["created_at"],
                    "metadata": result["metadata"],
                }
            return {}

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get User node data.

        Args:
            user_id: User identifier

        Returns:
            User data or None if not found
        """
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})
                RETURN u.id as user_id, u.created_at as created_at, u.metadata as metadata
                """,
                user_id=user_id,
            ).single()

            if result:
                return {
                    "user_id": result["user_id"],
                    "created_at": result["created_at"],
                    "metadata": result["metadata"] or {},
                }
            return None

    def update_user(self, user_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update User node metadata.

        Args:
            user_id: User identifier
            metadata: Updated metadata

        Returns:
            Updated user data
        """
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})
                SET u.metadata = $metadata
                RETURN u.id as user_id, u.created_at as created_at, u.metadata as metadata
                """,
                user_id=user_id,
                metadata=metadata,
            ).single()

            if result:
                return {
                    "user_id": result["user_id"],
                    "created_at": result["created_at"],
                    "metadata": result["metadata"],
                }
            return {}

    # ============================================================
    # Memory System: User Facts (Preferences)
    # ============================================================

    def create_fact(
        self,
        user_id: str,
        fact_id: str,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a Fact node linked to a User.

        Facts represent user preferences, statements, or important information
        extracted from conversations.

        Args:
            user_id: User identifier
            fact_id: Unique fact identifier
            content: Fact content/description
            importance: Importance score (0.0-1.0)
            metadata: Optional metadata (category, source, etc.)

        Returns:
            Fact data including id, content, importance
        """
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})
                MERGE (f:Fact {id: $fact_id})
                ON CREATE SET
                    f.content = $content,
                    f.importance = $importance,
                    f.created_at = datetime(),
                    f.metadata = $metadata
                ON MATCH SET
                    f.content = $content,
                    f.importance = $importance,
                    f.metadata = $metadata
                MERGE (u)-[r:HAS_PREFERENCE]->(f)
                RETURN f.id as fact_id, f.content as content, f.importance as importance,
                       f.created_at as created_at, f.metadata as metadata
                """,
                user_id=user_id,
                fact_id=fact_id,
                content=content,
                importance=importance,
                metadata=metadata or {},
            ).single()

            if result:
                return {
                    "fact_id": result["fact_id"],
                    "content": result["content"],
                    "importance": result["importance"],
                    "created_at": result["created_at"],
                    "metadata": result["metadata"],
                }
            return {}

    def get_user_facts(
        self, user_id: str, min_importance: float = 0.0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all facts for a user, sorted by importance.

        Args:
            user_id: User identifier
            min_importance: Minimum importance threshold
            limit: Maximum number of facts to return

        Returns:
            List of facts ordered by importance descending
        """
        with self.session_scope() as session:
            results = session.run(
                """
                MATCH (u:User {id: $user_id})-[:HAS_PREFERENCE]->(f:Fact)
                WHERE f.importance >= $min_importance
                RETURN f.id as fact_id, f.content as content, f.importance as importance,
                       f.created_at as created_at, f.metadata as metadata
                ORDER BY f.importance DESC
                LIMIT $limit
                """,
                user_id=user_id,
                min_importance=min_importance,
                limit=limit,
            )

            facts = []
            for record in results:
                facts.append({
                    "fact_id": record["fact_id"],
                    "content": record["content"],
                    "importance": record["importance"],
                    "created_at": record["created_at"],
                    "metadata": record["metadata"] or {},
                })
            return facts

    def update_fact(
        self,
        fact_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update a Fact node.

        Args:
            fact_id: Fact identifier
            content: Updated content (if provided)
            importance: Updated importance (if provided)
            metadata: Updated metadata (if provided)

        Returns:
            Updated fact data
        """
        with self.session_scope() as session:
            # Build SET clause dynamically based on what's provided
            set_clauses = []
            params = {"fact_id": fact_id}

            if content is not None:
                set_clauses.append("f.content = $content")
                params["content"] = content
            if importance is not None:
                set_clauses.append("f.importance = $importance")
                params["importance"] = importance
            if metadata is not None:
                set_clauses.append("f.metadata = $metadata")
                params["metadata"] = metadata

            if not set_clauses:
                # Nothing to update, just return current data
                result = session.run(
                    """
                    MATCH (f:Fact {id: $fact_id})
                    RETURN f.id as fact_id, f.content as content, f.importance as importance,
                           f.created_at as created_at, f.metadata as metadata
                    """,
                    fact_id=fact_id,
                ).single()
            else:
                set_clause = ", ".join(set_clauses)
                result = session.run(
                    f"""
                    MATCH (f:Fact {{id: $fact_id}})
                    SET {set_clause}
                    RETURN f.id as fact_id, f.content as content, f.importance as importance,
                           f.created_at as created_at, f.metadata as metadata
                    """,
                    **params,
                ).single()

            if result:
                return {
                    "fact_id": result["fact_id"],
                    "content": result["content"],
                    "importance": result["importance"],
                    "created_at": result["created_at"],
                    "metadata": result["metadata"] or {},
                }
            return {}

    def delete_fact(self, fact_id: str) -> bool:
        """Delete a Fact node.

        Args:
            fact_id: Fact identifier

        Returns:
            True if deleted, False if not found
        """
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (f:Fact {id: $fact_id})
                DETACH DELETE f
                RETURN count(f) as deleted_count
                """,
                fact_id=fact_id,
            ).single()

            return result and result["deleted_count"] > 0

    # ============================================================
    # Memory System: Conversation Summaries
    # ============================================================

    def create_conversation(
        self,
        user_id: str,
        conversation_id: str,
        title: str = "",
        summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a Conversation summary node linked to a User.

        Args:
            user_id: User identifier
            conversation_id: Unique conversation identifier
            title: Conversation title
            summary: Conversation summary (not full transcript)
            metadata: Optional metadata (topics, key points, etc.)

        Returns:
            Conversation data
        """
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (u:User {id: $user_id})
                MERGE (conv:Conversation {id: $conversation_id})
                ON CREATE SET
                    conv.user_id = $user_id,
                    conv.title = $title,
                    conv.summary = $summary,
                    conv.created_at = datetime(),
                    conv.updated_at = datetime(),
                    conv.metadata = $metadata
                ON MATCH SET
                    conv.title = $title,
                    conv.summary = $summary,
                    conv.updated_at = datetime(),
                    conv.metadata = $metadata
                MERGE (u)-[r:HAS_CONVERSATION]->(conv)
                RETURN conv.id as conversation_id, conv.user_id as user_id,
                       conv.title as title, conv.summary as summary,
                       conv.created_at as created_at, conv.updated_at as updated_at,
                       conv.metadata as metadata
                """,
                user_id=user_id,
                conversation_id=conversation_id,
                title=title,
                summary=summary,
                metadata=metadata or {},
            ).single()

            if result:
                return {
                    "conversation_id": result["conversation_id"],
                    "user_id": result["user_id"],
                    "title": result["title"],
                    "summary": result["summary"],
                    "created_at": result["created_at"],
                    "updated_at": result["updated_at"],
                    "metadata": result["metadata"],
                }
            return {}

    def get_user_conversations(
        self, user_id: str, limit: int = 10, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get recent conversations for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip (for pagination)

        Returns:
            List of conversation summaries ordered by updated_at descending
        """
        with self.session_scope() as session:
            results = session.run(
                """
                MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(conv:Conversation)
                RETURN conv.id as conversation_id, conv.user_id as user_id,
                       conv.title as title, conv.summary as summary,
                       conv.created_at as created_at, conv.updated_at as updated_at,
                       conv.metadata as metadata
                ORDER BY conv.updated_at DESC
                SKIP $offset
                LIMIT $limit
                """,
                user_id=user_id,
                limit=limit,
                offset=offset,
            )

            conversations = []
            for record in results:
                conversations.append({
                    "conversation_id": record["conversation_id"],
                    "user_id": record["user_id"],
                    "title": record["title"],
                    "summary": record["summary"],
                    "created_at": record["created_at"],
                    "updated_at": record["updated_at"],
                    "metadata": record["metadata"] or {},
                })
            return conversations

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific conversation by ID.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Conversation data or None if not found
        """
        with self.session_scope() as session:
            result = session.run(
                """
                MATCH (conv:Conversation {id: $conversation_id})
                RETURN conv.id as conversation_id, conv.user_id as user_id,
                       conv.title as title, conv.summary as summary,
                       conv.created_at as created_at, conv.updated_at as updated_at,
                       conv.metadata as metadata
                """,
                conversation_id=conversation_id,
            ).single()

            if result:
                return {
                    "conversation_id": result["conversation_id"],
                    "user_id": result["user_id"],
                    "title": result["title"],
                    "summary": result["summary"],
                    "created_at": result["created_at"],
                    "updated_at": result["updated_at"],
                    "metadata": result["metadata"] or {},
                }
            return None

    def update_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update a Conversation node.

        Args:
            conversation_id: Conversation identifier
            title: Updated title (if provided)
            summary: Updated summary (if provided)
            metadata: Updated metadata (if provided)

        Returns:
            Updated conversation data
        """
        with self.session_scope() as session:
            # Build SET clause dynamically
            set_clauses = ["conv.updated_at = datetime()"]
            params = {"conversation_id": conversation_id}

            if title is not None:
                set_clauses.append("conv.title = $title")
                params["title"] = title
            if summary is not None:
                set_clauses.append("conv.summary = $summary")
                params["summary"] = summary
            if metadata is not None:
                set_clauses.append("conv.metadata = $metadata")
                params["metadata"] = metadata

            set_clause = ", ".join(set_clauses)
            result = session.run(
                f"""
                MATCH (conv:Conversation {{id: $conversation_id}})
                SET {set_clause}
                RETURN conv.id as conversation_id, conv.user_id as user_id,
                       conv.title as title, conv.summary as summary,
                       conv.created_at as created_at, conv.updated_at as updated_at,
                       conv.metadata as metadata
                """,
                **params,
            ).single()

            if result:
                return {
                    "conversation_id": result["conversation_id"],
                    "user_id": result["user_id"],
                    "title": result["title"],
                    "summary": result["summary"],
                    "created_at": result["created_at"],
                    "updated_at": result["updated_at"],
                    "metadata": result["metadata"] or {},
                }
            return {}

    def clear_database(self) -> None:
        """Clear all data from the database."""
        with self.session_scope() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")


# Global database instance
graph_db = GraphDB()
