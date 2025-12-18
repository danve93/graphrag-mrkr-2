"""
LangGraph-based RAG pipeline implementation.
"""

import logging
import threading
import hashlib
import asyncio
import time
import queue as _queue
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import END, StateGraph

from core.otel_config import get_tracer

from core.singletons import get_response_cache, ResponseKeyLock, hash_response_params
from core.cache_metrics import cache_metrics
from rag.nodes.graph_reasoning import reason_with_graph
from rag.nodes.query_analysis import analyze_query
from rag.nodes.retrieval import retrieve_documents
from core.graph_db import graph_db
from rag.nodes.category_manager import CategoryManager
from rag.nodes.query_router import route_query_to_categories, get_documents_by_categories
from rag.nodes.smart_consolidation import consolidate_chunks
from rag.nodes.prompt_selector import get_prompt_selector
from rag.nodes.structured_kg_executor import get_structured_kg_executor
from core.quality_scorer import quality_scorer
from rag.quality_monitor import quality_monitor
from core.routing_metrics import routing_metrics
from config.settings import settings
from rag.nodes.adaptive_router import get_feedback_learner
from core.conversation_memory import memory_manager

logger = logging.getLogger(__name__)

# In-memory TTL cache for routing decisions: query → (category_id, confidence, timestamp)
# Issue #18: Use TTLCache with max size to prevent unbounded growth
from cachetools import TTLCache
ROUTING_TTL_SECONDS = 3600.0
ROUTING_CACHE_MAX_SIZE = getattr(settings, 'routing_cache_max_size', 1000)
ROUTING_CACHE: TTLCache = TTLCache(maxsize=ROUTING_CACHE_MAX_SIZE, ttl=ROUTING_TTL_SECONDS)


class RAGState:
    """State management for the RAG pipeline."""

    def __init__(self):
        """Initialize RAG state."""
        self.query: str = ""
        self.query_analysis: Dict[str, Any] = {}
        self.retrieved_chunks: List[Dict[str, Any]] = []
        self.graph_context: List[Dict[str, Any]] = []
        self.response: str = ""
        self.sources: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.quality_score: Optional[Dict[str, Any]] = None
        self.context_documents: List[str] = []
        self.stages: List[Dict[str, Any]] = []  # Track stages with timing for UI


class GraphRAG:
    """LangGraph-based RAG pipeline orchestrator."""

    def __init__(self):
        """Initialize the GraphRAG pipeline."""
        # workflow may be a LangGraph compiled graph — keep as Any to avoid static typing issues
        self.workflow: Any = self._build_workflow()

    def _build_workflow(self) -> Any:
        """Build the LangGraph workflow for RAG."""
        # Use plain dict as the runtime state type for LangGraph. Keep as Any to silence type checkers.
        workflow: Any = StateGraph(dict)  # type: ignore

        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("structured_kg_router", self._structured_kg_router_node)
        workflow.add_node("retrieve_documents", self._retrieve_documents_node)
        # Routing is captured within retrieval stage with timing metadata
        workflow.add_node("reason_with_graph", self._reason_with_graph_node)
        workflow.add_node("generate_response", self._generate_response_node)

        # Add edges
        workflow.add_edge("analyze_query", "structured_kg_router")
        workflow.add_conditional_edges(
            "structured_kg_router",
            lambda state: "generate_response" if state.get("structured_kg_complete") else "retrieve_documents",
            {"generate_response": "generate_response", "retrieve_documents": "retrieve_documents"}
        )
        workflow.add_edge("retrieve_documents", "reason_with_graph")
        workflow.add_edge("reason_with_graph", "generate_response")
        workflow.add_edge("generate_response", END)

        # Set entry point
        workflow.set_entry_point("analyze_query")

        return workflow.compile()

    def _analyze_query_node(self, state) -> Any:
        """Analyze the user query (dict-based state for LangGraph)."""
        try:
            query = state.get("query", "")
            chat_history = state.get("chat_history", [])
            logger.info(f"Analyzing query: {query}")
            
            tracer = get_tracer("amber.graphrag")
            with tracer.start_as_current_span("graphrag.analyze_query") as span:
                span.set_attribute("query", query[:200])
            
            # Initialize stages list if not present
            if "stages" not in state:
                state["stages"] = []
            
            # Track stage with timing
            start_time = time.time()
            state["query_analysis"] = analyze_query(query, chat_history)
            duration_ms = int((time.time() - start_time) * 1000)
            
            state["stages"].append({
                "name": "query_analysis",
                "duration_ms": duration_ms,
                "timestamp": time.time(),
            })
            logger.info(f"Stage query_analysis completed in {duration_ms}ms")
            
            return state
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            state["query_analysis"] = {"error": str(e)}
            return state

    def _structured_kg_router_node(self, state) -> Any:
        """Route query to structured KG executor if suitable (dict-based state for LangGraph)."""
        try:
            # Check if structured KG is enabled
            if not settings.enable_structured_kg:
                logger.info("Structured KG disabled, skipping to standard retrieval")
                state["structured_kg_complete"] = False
                return state
            
            query = state.get("query", "")
            logger.info(f"Checking if query is suitable for structured KG: {query}")
            
            # Initialize stages list if not present
            if "stages" not in state:
                state["stages"] = []
            
            # Track stage with timing
            start_time = time.time()
            
            tracer = get_tracer("amber.graphrag")
            with tracer.start_as_current_span("graphrag.structured_kg_router") as span:
                span.set_attribute("query", query[:200])

                # Get structured KG executor
                executor = get_structured_kg_executor()
                
                # Check if query is suitable for structured execution
                if not executor._is_suitable_for_structured(query):
                    logger.info("Query not suitable for structured KG, proceeding to standard retrieval")
                    duration_ms = int((time.time() - start_time) * 1000)
                    state["stages"].append({
                        "name": "structured_kg_routing",
                        "duration_ms": duration_ms,
                        "timestamp": time.time(),
                        "metadata": {
                            "routed_to": "standard_retrieval",
                            "reason": "not_suitable"
                        }
                    })
                    state["structured_kg_complete"] = False
                    return state
                
                # Execute structured query
                logger.info("Executing structured KG query")
                result = executor.execute_query(query)
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                # If structured query succeeded, prepare result and skip standard retrieval
                if result.get("success") and result.get("results"):
                    logger.info(f"Structured KG query succeeded with {len(result['results'])} results")
                    
                    # Format structured results as context chunks for generation
                    structured_chunks = self._format_structured_results_as_chunks(result)
                    state["structured_kg_results"] = result
                    state["graph_context"] = structured_chunks
                    state["retrieved_chunks"] = []  # No standard retrieval
                    state["structured_kg_complete"] = True
                    
                    span.set_attribute("routed_to", "structured_kg")
                    span.set_attribute("query_type", result.get("query_type"))
                    span.set_attribute("results_count", len(result.get("results", [])))
                    
                    state["stages"].append({
                        "name": "structured_kg_execution",
                        "duration_ms": duration_ms,
                        "timestamp": time.time(),
                        "metadata": {
                            "routed_to": "structured_kg",
                            "query_type": result.get("query_type"),
                            "results_count": len(result["results"]),
                            "linked_entities": len(result.get("linked_entities", [])),
                            "cypher_query": result.get("cypher_query", ""),
                            "corrections": result.get("corrections", 0)
                        }
                    })
                    logger.info(f"Stage structured_kg_execution completed in {duration_ms}ms")
                    return state
                else:
                    # Structured query failed, fall back to standard retrieval
                    logger.warning(f"Structured KG query failed: {result.get('error', 'unknown error')}, falling back to standard retrieval")
                    state["stages"].append({
                        "name": "structured_kg_routing",
                        "duration_ms": duration_ms,
                        "timestamp": time.time(),
                        "metadata": {
                            "routed_to": "standard_retrieval",
                            "reason": "execution_failed",
                            "error": result.get("error", "unknown")
                        }
                    })
                    state["structured_kg_complete"] = False
                    return state
                
        except Exception as e:
            logger.error(f"Structured KG routing failed: {e}")
            state["stages"].append({
                "name": "structured_kg_routing",
                "duration_ms": int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0,
                "timestamp": time.time(),
                "metadata": {
                    "routed_to": "standard_retrieval",
                    "reason": "exception",
                    "error": str(e)
                }
            })
            state["structured_kg_complete"] = False
            return state

    def _format_structured_results_as_chunks(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format structured KG results as context chunks for generation."""
        try:
            chunks = []
            
            # Create a summary chunk with query metadata
            query_type = result.get("query_type", "general")
            linked_entities = result.get("linked_entities", [])
            cypher = result.get("cypher_query", "")
            
            summary_content = []
            summary_content.append(f"**Query Type:** {query_type}")
            
            if linked_entities:
                entity_names = ", ".join([e['name'] for e in linked_entities])
                summary_content.append(f"**Linked Entities:** {entity_names}")
            
            if cypher:
                summary_content.append(f"**Cypher Query:**\n```cypher\n{cypher}\n```")
            
            chunks.append({
                "id": "structured_kg_summary",
                "chunk_id": "structured_kg_summary",
                "content": "\n\n".join(summary_content),
                "metadata": {
                    "source": "structured_kg",
                    "query_type": query_type,
                    "type": "summary"
                },
                "similarity": 1.0,
                "hybrid_score": 1.0
            })
            
            # Create chunks for result rows
            results = result.get("results", [])
            if results:
                # Group results into manageable chunks (max 10 rows per chunk)
                chunk_size = 10
                for i in range(0, len(results), chunk_size):
                    batch = results[i:i+chunk_size]
                    
                    rows_content = []
                    rows_content.append(f"**Results {i+1}-{min(i+chunk_size, len(results))} of {len(results)}:**\n")
                    
                    for j, row in enumerate(batch, i+1):
                        row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
                        rows_content.append(f"{j}. {row_str}")
                    
                    chunks.append({
                        "id": f"structured_kg_results_{i//chunk_size}",
                        "chunk_id": f"structured_kg_results_{i//chunk_size}",
                        "content": "\n".join(rows_content),
                        "metadata": {
                            "source": "structured_kg",
                            "query_type": query_type,
                            "type": "results",
                            "batch_index": i//chunk_size,
                            "row_start": i,
                            "row_end": min(i+chunk_size, len(results))
                        },
                        "similarity": 1.0,
                        "hybrid_score": 1.0
                    })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to format structured results: {e}")
            # Return a single error chunk
            return [{
                "id": "structured_kg_error",
                "chunk_id": "structured_kg_error",
                "content": f"Structured query results (formatting error: {e})",
                "metadata": {"source": "structured_kg", "type": "error"},
                "similarity": 1.0,
                "hybrid_score": 1.0
            }]

    def _retrieve_documents_node(self, state) -> Any:
        """Retrieve relevant documents (dict-based state for LangGraph)."""
        try:
            logger.info("Retrieving relevant documents")
            
            tracer = get_tracer("amber.graphrag")
            with tracer.start_as_current_span("graphrag.retrieval") as span:
            
                # Initialize stages list if not present
                if "stages" not in state:
                    state["stages"] = []
                
                # Track stage with timing
                start_time = time.time()
                
                # Pass additional retrieval tuning parameters from state
                chunk_weight = state.get("chunk_weight", 0.5)
                entity_weight = state.get("entity_weight", None)
                path_weight = state.get("path_weight", None)
                graph_expansion = state.get("graph_expansion", True)
                use_multi_hop = state.get("use_multi_hop", False)
                max_hops = state.get("max_hops", None)
                beam_size = state.get("beam_size", None)
                restrict_to_context = state.get("restrict_to_context", True)
                expansion_depth = state.get("graph_expansion_depth", None)
    
                # Optional: classify query and restrict to a category when confident
                routing_category_id = None
                routing_confidence = 0.0
                routing_stage_started = time.time()
                used_cache = False
                routing_categories = []
                
                # Check for category filter override
                category_filter = state.get("category_filter")
                
                try:
                    query_text = state.get("query", "")
                    query_analysis = state.get("query_analysis", {})
                    
                    # If category_filter is provided, use it directly (skip routing)
                    if category_filter and len(category_filter) > 0:
                        routing_categories = category_filter
                        routing_confidence = 1.0  # Manual override = 100% confidence
                        routing_category_id = routing_categories[0]
                        logger.info(f"Using manual category filter: {routing_categories}")
                    # Use new query_router if enabled, otherwise fall back to CategoryManager
                    elif settings.enable_query_routing:
                        routing_result = route_query_to_categories(
                            query=query_text,
                            query_analysis=query_analysis,
                            confidence_threshold=settings.query_routing_confidence_threshold,
                        )
                        routing_categories = routing_result.get("categories", [])
                        routing_confidence = routing_result.get("confidence", 0.0)
                        used_cache = routing_result.get("used_cache", False)
                        should_filter = routing_result.get("should_filter", False)
                        
                        # Use first category as primary (for backward compatibility)
                        if routing_categories and should_filter:
                            routing_category_id = routing_categories[0]
                        
                        logger.info(
                            f"Query router result: categories={routing_categories}, "
                            f"confidence={routing_confidence:.2f}, should_filter={should_filter}"
                        )
                    else:
                        # Fallback to CategoryManager (legacy behavior)
                        # TTLCache automatically expires entries, no manual timestamp check needed
                        cached = ROUTING_CACHE.get(query_text)
                        if cached:
                            routing_category_id, routing_confidence = cached
                            used_cache = True
                            logger.info(
                                f"Routing cache hit for query; category={routing_category_id} conf={routing_confidence}"
                            )
                        else:
                            manager = CategoryManager()
                            classifications = manager.classify_query(query_text)
                            # classify_query may be sync; if coroutine, run it
                            if asyncio.iscoroutine(classifications):
                                classifications = asyncio.run(classifications)
                            if classifications and classifications[0][1] >= 0.6:
                                routing_category_id = classifications[0][0]
                                routing_confidence = classifications[0][1]
                            # Store without timestamp - TTLCache handles expiration
                            ROUTING_CACHE[query_text] = (routing_category_id, routing_confidence)
                    
                    if routing_category_id is not None:
                        state["routing_category_id"] = routing_category_id
                        state["routing_confidence"] = routing_confidence
                except Exception as e:
                    logger.warning(f"Routing classification failed: {e}")
    
                # Record routing stage timing for UI visibility
                try:
                    routing_duration_ms = int((time.time() - routing_stage_started) * 1000)
                    if "stages" not in state:
                        state["stages"] = []
                    state["stages"].append({
                        "name": "routing",
                        "duration_ms": routing_duration_ms,
                        "timestamp": time.time(),
                        "metadata": {
                            "routing_category_id": routing_category_id,
                            "routing_confidence": routing_confidence,
                            "routing_categories": routing_categories,  # Include full list
                            "document_count": len(allowed_docs) if routing_category_id else None,
                        },
                    })
                except Exception:
                    pass
    
                # If we have a confident category, fetch document ids in that category
                allowed_docs = state.get("context_documents", []) or []
                if routing_category_id and not allowed_docs:
                    try:
                        # Use get_documents_by_categories if we have multiple categories from query_router
                        if settings.enable_query_routing and routing_categories:
                            allowed_docs = get_documents_by_categories(routing_categories)
                            logger.info(
                                f"Routing applied: {len(allowed_docs)} docs in categories {routing_categories}"
                            )
                        else:
                            # Fallback to single-category query for CategoryManager
                            with graph_db.driver.session() as session:
                                result = session.run(
                                    """
                                    MATCH (d:Document)-[:BELONGS_TO]->(c:Category {id: $cid})
                                    RETURN d.id as doc_id
                                    """,
                                    cid=routing_category_id,
                                )
                                allowed_docs = [r["doc_id"] for r in result]
                                logger.info(f"Routing applied: {len(allowed_docs)} docs in category {routing_category_id}")
                        
                        state["context_documents"] = allowed_docs
                    except Exception as e:
                        logger.warning(f"Failed to fetch docs for category/categories: {e}")
    
                # Adaptive routing: override weights if enabled
                if getattr(settings, "enable_adaptive_routing", False):
                    feedback_learner = get_feedback_learner()
                    adaptive_weights = feedback_learner.get_weights()
                    chunk_weight = adaptive_weights.get("chunk_weight", chunk_weight)
                    entity_weight = adaptive_weights.get("entity_weight", entity_weight)
                    path_weight = adaptive_weights.get("path_weight", path_weight)
                    state["adaptive_weights"] = adaptive_weights
    
                retrieved_chunks, alternative_chunks = retrieve_documents(
                    state.get("query", ""),
                    state.get("query_analysis", {}),
                    state.get("retrieval_mode", "graph_enhanced"),
                    state.get("top_k", 5),
                    chunk_weight=chunk_weight,
                    entity_weight=entity_weight,
                    path_weight=path_weight,
                    graph_expansion=graph_expansion,
                    use_multi_hop=use_multi_hop,
                    max_hops=max_hops,
                    beam_size=beam_size,
                    restrict_to_context=restrict_to_context,
                    expansion_depth=expansion_depth,
                    embedding_model=state.get("embedding_model", None),
                    context_documents=allowed_docs,
                )
                state["retrieved_chunks"] = retrieved_chunks
                state["alternative_chunks"] = alternative_chunks
    
                span.set_attribute("chunks_retrieved", len(retrieved_chunks))
                span.set_attribute("routing_category_id", str(routing_category_id) if routing_category_id else "none")
                span.set_attribute("routing_confidence", routing_confidence)
    
                duration_ms = int((time.time() - start_time) * 1000)
                
                # Debug: log retrieved chunk summary (count + sample ids/similarities)
                retrieved = state.get("retrieved_chunks", []) or []
                alternatives = state.get("alternative_chunks", []) or []
                try:
                    sample_info = [
                        {
                            "chunk_id": c.get("chunk_id") or c.get("id"),
                            "similarity": c.get("similarity", c.get("hybrid_score", 0.0)),
                        }
                        for c in retrieved[:5]
                    ]
                    logger.info(
                        "Post-retrieval: %d chunks retrieved, %d alternatives. Sample: %s",
                        len(retrieved),
                        len(alternatives),
                        sample_info,
                    )
                except Exception:
                    logger.debug("Failed to log retrieved chunk sample")
                
                chunks_count = len(retrieved)
                meta = {"chunks_retrieved": chunks_count}
                if routing_category_id:
                    meta.update({
                        "routing_category_id": routing_category_id,
                        "routing_confidence": routing_confidence,
                        "routing_categories": routing_categories,
                        "document_count": len(allowed_docs),
                    })
                state["stages"].append({
                    "name": "retrieval",
                    "duration_ms": duration_ms,
                    "timestamp": time.time(),
                    "metadata": meta,
                })
                logger.info(f"Stage retrieval completed in {duration_ms}ms, retrieved {chunks_count} chunks")
    
                # Fallback validation: if insufficient results, expand search scope
                fallback_used = False
                try:
                    min_results = 3
                    if chunks_count < min_results:
                        logger.warning(
                            f"Fallback triggered: only {chunks_count} chunks; expanding to all documents"
                        )
                        fallback_used = True
                        retrieved_chunks, alternative_chunks = retrieve_documents(
                            state.get("query", ""),
                            state.get("query_analysis", {}),
                            state.get("retrieval_mode", "graph_enhanced"),
                            state.get("top_k", 5),
                            chunk_weight=chunk_weight,
                            entity_weight=entity_weight,
                            path_weight=path_weight,
                            graph_expansion=graph_expansion,
                            use_multi_hop=use_multi_hop,
                            max_hops=max_hops,
                            beam_size=beam_size,
                            restrict_to_context=False,
                            expansion_depth=expansion_depth,
                            embedding_model=state.get("embedding_model", None),
                            context_documents=[],
                        )
                        state["retrieved_chunks"] = retrieved_chunks
                        state["alternative_chunks"] = alternative_chunks
                        logger.info(
                            "Fallback retrieval completed: %d chunks",
                            len(retrieved_chunks or []),
                        )
                except Exception:
                    logger.debug("Fallback validation skipped due to error")
    
                # Apply smart consolidation if enabled and multi-category routing
                if settings.consolidation_strategy == "category_aware" and routing_categories and len(routing_categories) > 1:
                    try:
                        consolidation_start = time.time()
                        original_count = len(state.get("retrieved_chunks", []))
                        
                        # Consolidate with category awareness
                        consolidated = asyncio.run(consolidate_chunks(
                            chunks=state.get("retrieved_chunks", []),
                            categories=routing_categories,
                            top_k=state.get("top_k", 5),
                        ))
                        
                        state["retrieved_chunks"] = consolidated
                        consolidation_ms = int((time.time() - consolidation_start) * 1000)
                        
                        logger.info(
                            f"Smart consolidation: {original_count} → {len(consolidated)} chunks "
                            f"({consolidation_ms}ms, categories: {routing_categories})"
                        )
                        
                        # Add consolidation stage to UI
                        if "stages" not in state:
                            state["stages"] = []
                        state["stages"].append({
                            "name": "consolidation",
                            "duration_ms": consolidation_ms,
                            "timestamp": time.time(),
                            "metadata": {
                                "original_count": original_count,
                                "consolidated_count": len(consolidated),
                                "categories": routing_categories,
                            },
                        })
                    except Exception as e:
                        logger.warning(f"Smart consolidation failed: {e}")
    
                # Record routing metrics (track all queries, even those without routing)
                try:
                    routing_latency_ms = int((time.time() - routing_stage_started) * 1000)
                    # Use routing_categories if available (from query_router), otherwise single category
                    categories_list = routing_categories if routing_categories else ([routing_category_id] if routing_category_id else [])
                    routing_metrics.record_routing(
                        categories=categories_list,
                        confidence=routing_confidence,
                        latency_ms=routing_latency_ms,
                        used_cache=used_cache,
                        fallback_used=fallback_used,
                    )
                except Exception as e:
                    logger.debug(f"Failed to record routing metrics: {e}")
    
                return state
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            state["retrieved_chunks"] = []
            return state

    def _reason_with_graph_node(self, state) -> Any:
        """Perform graph-based reasoning (dict-based state for LangGraph)."""
        try:
            logger.info("Performing graph reasoning")
            
            tracer = get_tracer("amber.graphrag")
            with tracer.start_as_current_span("graphrag.graph_reasoning") as span:
            
                # Initialize stages list if not present
                if "stages" not in state:
                    state["stages"] = []
                
                # Track stage with timing
                start_time = time.time()
                state["graph_context"] = reason_with_graph(
                    state.get("query", ""),
                    state.get("retrieved_chunks", []),
                    state.get("query_analysis", {}),
                    state.get("retrieval_mode", "graph_enhanced"),
                )
                duration_ms = int((time.time() - start_time) * 1000)
                
                # Debug: log graph context summary
                graph_ctx = state.get("graph_context", []) or []
                try:
                    sample_graph = [
                        {"chunk_id": c.get("chunk_id") or c.get("id"), "similarity": c.get("similarity", c.get("hybrid_score", 0.0))}
                        for c in graph_ctx[:5]
                    ]
                    logger.info("Post-graph-reasoning: %d items in graph_context. Sample: %s", len(graph_ctx), sample_graph)
                except Exception:
                    logger.debug("Failed to log graph_context sample")
                
                state["stages"].append({
                    "name": "graph_reasoning",
                    "duration_ms": duration_ms,
                    "timestamp": time.time(),
                    "metadata": {"context_items": len(graph_ctx)},
                })
                logger.info(f"Stage graph_reasoning completed in {duration_ms}ms")
                return state
        except Exception as e:
            logger.error(f"Graph reasoning failed: {e}")
            state["graph_context"] = state.get("retrieved_chunks", [])
            return state

    def _generate_response_node(self, state) -> Any:
        """Generate the final response (dict-based state for LangGraph)."""
        try:
            logger.info("Generating response")
            
            tracer = get_tracer("amber.graphrag")
            with tracer.start_as_current_span("graphrag.generation") as span:
            
                # Initialize stages list if not present
                if "stages" not in state:
                    state["stages"] = []
                
                # Debug: log what will be passed to generation
                try:
                    # Check if we have structured KG results
                    structured_kg_results = state.get("structured_kg_results")
                    if structured_kg_results:
                        logger.info(
                            "Using structured KG results for generation — query_type=%s, results_count=%d",
                            structured_kg_results.get("query_type"),
                            len(structured_kg_results.get("results", [])),
                        )
                    else:
                        retrieved = state.get("retrieved_chunks", []) or []
                        graph_ctx = state.get("graph_context", []) or []
                        logger.info(
                            "About to generate response — retrieved_chunks=%d, graph_context=%d",
                            len(retrieved),
                            len(graph_ctx),
                        )
                except Exception:
                    logger.debug("Failed to log pre-generation context sizes")
    
                # Track stage with timing
                start_time = time.time()
                
                # dynamic import so tests can monkeypatch `rag.nodes.generation.generate_response`
                try:
                    from rag.nodes import generation as generation_module
    
                    # Prompt grounding: attach routing category info to query_analysis
                    try:
                        routing_id = state.get("routing_category_id")
                        routing_conf = state.get("routing_confidence")
                        routing_categories = state.get("routing_categories", [])
                        routing_info = None
                        if routing_id:
                            with graph_db.driver.session() as session:
                                rec = session.run("MATCH (c:Category {id: $id}) RETURN c", id=routing_id).single()
                            if rec:
                                c = rec["c"]
                                routing_info = {
                                    "id": routing_id,
                                    "name": c.get("name"),
                                    "keywords": c.get("keywords", []),
                                    "confidence": routing_conf,
                                }
                        qa = state.get("query_analysis", {})
                        qa = {**qa, "routing": routing_info}
                        state["query_analysis"] = qa
                    except Exception:
                        # Non-fatal if routing info not available
                        pass
    
                    # Generate category-specific prompt if enabled
                    custom_prompt = None
                    if settings.enable_category_prompts and routing_categories:
                        try:
                            prompt_selector = get_prompt_selector()
                            # Build full context from graph_context chunks
                            graph_ctx = state.get("graph_context", [])
                            context_text = "\n\n".join([
                                f"[Chunk {i+1}]: {chunk.get('content', '')}"
                                for i, chunk in enumerate(graph_ctx)
                            ])
                            custom_prompt = asyncio.run(prompt_selector.select_generation_prompt(
                                query=state.get("query", ""),
                                categories=routing_categories,
                                context=context_text,
                                conversation_history=state.get("chat_history", [])
                            ))
                            logger.info(f"Using category-specific prompt for categories: {routing_categories}")
                        except Exception as e:
                            logger.warning(f"Failed to select category prompt, using default: {e}")
                            custom_prompt = None
    
                    response_data = generation_module.generate_response(
                        state.get("query", ""),
                        state.get("graph_context", []),
                        state.get("query_analysis", {}),
                        state.get("temperature", 0.7),
                        state.get("chat_history", []),
                        llm_model=state.get("llm_model", None),
                        custom_prompt=custom_prompt,
                        memory_context=state.get("memory_context", None),
                    )
                except Exception:
                    # fallback to direct import if package import fails
                    from rag.nodes.generation import generate_response as _gen
    
                    response_data = _gen(
                        state.get("query", ""),
                        state.get("graph_context", []),
                        state.get("query_analysis", {}),
                        state.get("temperature", 0.7),
                        state.get("chat_history", []),
                        llm_model=state.get("llm_model", None),
                        memory_context=state.get("memory_context", None),
                    )
    
                state["response"] = response_data.get("response", "")
                state["sources"] = response_data.get("sources", [])
                state["metadata"] = response_data.get("metadata", {})
                # Capture quality score computed during generation (if available)
                state["quality_score"] = response_data.get("quality_score", None)
                
                span.set_attribute("response_length", len(state["response"]))
                span.set_attribute("model", str(state.get("llm_model", "default")))
    
                duration_ms = int((time.time() - start_time) * 1000)
                state["stages"].append({
                    "name": "generation",
                    "duration_ms": duration_ms,
                    "timestamp": time.time(),
                    "metadata": {"response_length": len(state["response"])},
                })
                logger.info(f"Stage generation completed in {duration_ms}ms")
    
                return state
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            state["response"] = f"I apologize, but I encountered an error: {str(e)}"
            state["sources"] = []
            state["metadata"] = {"error": str(e)}
            return state

    def query(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_type: Optional[str] = None,
        retrieval_mode: str = "graph_enhanced",
        top_k: int = 5,
        temperature: float = 0.7,
        chunk_weight: float = 0.5,
        entity_weight: Optional[float] = None,
        path_weight: Optional[float] = None,
        graph_expansion: bool = True,
        use_multi_hop: bool = False,
        max_hops: Optional[int] = None,
        beam_size: Optional[int] = None,
        restrict_to_context: bool = True,
        graph_expansion_depth: Optional[int] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        context_documents: Optional[List[str]] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        category_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.

        Args:
            user_query: User's question or request
            retrieval_mode: Retrieval strategy ("simple", "graph_enhanced", "hybrid")
            top_k: Number of chunks to retrieve
            temperature: LLM temperature for response generation
            chunk_weight: Weight for chunk-based results in hybrid mode
            entity_weight: Weight for entity-filtered results in hybrid mode
            path_weight: Weight for path-based results in hybrid mode
            graph_expansion: Whether to use graph expansion
            use_multi_hop: Whether to use multi-hop reasoning
            max_hops: Depth limit for multi-hop traversal
            beam_size: Beam size for multi-hop traversal
            restrict_to_context: Whether to restrict retrieval to provided context documents
            graph_expansion_depth: Optional override for graph expansion depth
            chat_history: Optional conversation history for follow-up questions

        Returns:
            Dictionary containing response and metadata
        """
        tracer = get_tracer("amber.graphrag")
        with tracer.start_as_current_span("graphrag.query") as span:
            span.set_attribute("query", user_query)
            span.set_attribute("retrieval_mode", retrieval_mode)
            span.set_attribute("session_id", str(session_id))
            
            result = self._exec_query(
                user_query=user_query,
                session_id=session_id,
                user_id=user_id,
                user_type=user_type,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                temperature=temperature,
                chunk_weight=chunk_weight,
                entity_weight=entity_weight,
                path_weight=path_weight,
                graph_expansion=graph_expansion,
                use_multi_hop=use_multi_hop,
                max_hops=max_hops,
                beam_size=beam_size,
                restrict_to_context=restrict_to_context,
                graph_expansion_depth=graph_expansion_depth,
                chat_history=chat_history,
                context_documents=context_documents,
                llm_model=llm_model,
                embedding_model=embedding_model,
                category_filter=category_filter,
            )
            
            # Inject trace ID into result
            try:
                if isinstance(result, dict) and span.get_span_context().is_valid:
                    trace_id = f"{span.get_span_context().trace_id:032x}"
                    result["trace_id"] = trace_id
            except Exception:
                pass
                
            return result

    def _exec_query(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_type: Optional[str] = None,
        retrieval_mode: str = "graph_enhanced",
        top_k: int = 5,
        temperature: float = 0.7,
        chunk_weight: float = 0.5,
        entity_weight: Optional[float] = None,
        path_weight: Optional[float] = None,
        graph_expansion: bool = True,
        use_multi_hop: bool = False,
        max_hops: Optional[int] = None,
        beam_size: Optional[int] = None,
        restrict_to_context: bool = True,
        graph_expansion_depth: Optional[int] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        context_documents: Optional[List[str]] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        category_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.

        Args:
            user_query: User's question or request
            retrieval_mode: Retrieval strategy ("simple", "graph_enhanced", "hybrid")
            top_k: Number of chunks to retrieve
            temperature: LLM temperature for response generation
            chunk_weight: Weight for chunk-based results in hybrid mode
            entity_weight: Weight for entity-filtered results in hybrid mode
            path_weight: Weight for path-based results in hybrid mode
            graph_expansion: Whether to use graph expansion
            use_multi_hop: Whether to use multi-hop reasoning
            max_hops: Depth limit for multi-hop traversal
            beam_size: Beam size for multi-hop traversal
            restrict_to_context: Whether to restrict retrieval to provided context documents
            graph_expansion_depth: Optional override for graph expansion depth
            chat_history: Optional conversation history for follow-up questions

        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Initialize state object and convert to dict for the workflow
            state_obj = RAGState()
            state_obj.query = user_query
            state = state_obj.__dict__.copy()

            # Load memory context if user_id and session_id are provided
            memory_context = None
            if user_id and session_id and settings.enable_memory_system:
                try:
                    # Load or create memory context for this session
                    memory_context = memory_manager.get_session_context(session_id)
                    if not memory_context:
                        memory_context = memory_manager.load_memory_context(
                            user_id=user_id,
                            session_id=session_id,
                        )

                    # Add current query to session messages
                    memory_manager.add_message_to_session(
                        session_id=session_id,
                        role="user",
                        content=user_query,
                    )

                    # Store memory context in state for access in generation
                    state["memory_context"] = memory_context
                    state["user_id"] = user_id

                    logger.info(f"Loaded memory context for user={user_id}, session={session_id}")
                except Exception as e:
                    logger.warning(f"Failed to load memory context: {e}")
                    state["memory_context"] = None

            # If session-scoped response cache is enabled, check fast-path
            try:
                cache = get_response_cache()
                # Compute compact chat-history fingerprint
                try:
                    chat_hist = chat_history or []
                    recent = chat_hist[-3:]
                    import json

                    chat_hist_serial = json.dumps(recent, sort_keys=True, ensure_ascii=False)
                    chat_history_hash = hashlib.md5(chat_hist_serial.encode("utf-8")).hexdigest()
                except Exception:
                    chat_history_hash = ""

                cache_key = hash_response_params(
                    query=user_query,
                    retrieval_mode=state.get("retrieval_mode", "graph_enhanced"),
                    top_k=state.get("top_k", 5),
                    chunk_weight=state.get("chunk_weight", 0.5),
                    entity_weight=state.get("entity_weight", None),
                    path_weight=state.get("path_weight", None),
                    graph_expansion=state.get("graph_expansion", True),
                    use_multi_hop=state.get("use_multi_hop", False),
                    llm_model=state.get("llm_model", None),
                    embedding_model=state.get("embedding_model", None),
                    context_documents=state.get("context_documents", []),
                    routing_category_id=state.get("routing_category_id", None),
                    routing_confidence=state.get("routing_confidence", None),
                    session_id=session_id or "__global__",
                    chat_history_hash=chat_history_hash,
                )

                # CacheService handles namespacing, but we pass session_id for extra explicit isolation if needed
                # (though hash_response_params already includes it).
                cached = cache.get(cache_key, workspace_id=session_id)
                if cached is not None:
                    cache_metrics.record_response_hit()
                    try:
                        if isinstance(cached.get("metadata"), dict):
                            cached["metadata"]["cached"] = True
                    except Exception:
                        pass
                    if "stages" in cached and isinstance(cached["stages"], list):
                        if "cache_hit" not in cached["stages"]:
                            cached["stages"].insert(0, "cache_hit")
                    else:
                        cached["stages"] = ["cache_hit"]
                    logger.info("Response cache hit — returning cached response")

                    # Record quality metrics for cache hit
                    try:
                        quality_score_data = cached.get("quality_score")
                        quality_monitor.record_query(
                            query=user_query,
                            query_type=retrieval_mode,
                            retrieval_latency_ms=0.0,  # Cache hit, no retrieval
                            generation_latency_ms=0.0,  # Cache hit, no generation
                            num_chunks_retrieved=len(cached.get("retrieved_chunks", [])),
                            quality_score=quality_score_data.get("total") if quality_score_data else None,
                            quality_breakdown=quality_score_data.get("breakdown") if quality_score_data else None,
                            quality_confidence=quality_score_data.get("confidence") if quality_score_data else None,
                            cache_hit=True,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to record quality metrics for cache hit: {e}")

                    # KEY FIX: Record routing metrics for cache hit so Routing View is accurate
                    try:
                        from core.routing_metrics import routing_metrics
                        meta = cached.get("metadata", {})
                        r_info = meta.get("routing_info", {})
                        
                        # Fallback if routing info missing in older cache entries
                        cats = r_info.get("categories", ["cached"])
                        conf = r_info.get("confidence", 1.0)
                        
                        routing_metrics.record_routing(
                            categories=cats,
                            confidence=conf,
                            latency_ms=0.0, # Zero latency for cache hit
                            used_cache=True,
                            fallback_used=False
                        )
                    except Exception as e:
                        logger.warning(f"Failed to record routing metrics for cache hit: {e}")

                    return cached
            except Exception:
                # proceed without cache
                pass

            # Add RAG parameters to state
            state["retrieval_mode"] = retrieval_mode
            state["top_k"] = top_k
            state["temperature"] = temperature
            # Include hybrid tuning options provided by caller
            state["chunk_weight"] = chunk_weight
            state["entity_weight"] = entity_weight
            state["path_weight"] = path_weight
            state["graph_expansion"] = graph_expansion
            state["use_multi_hop"] = use_multi_hop
            state["max_hops"] = max_hops
            state["beam_size"] = beam_size
            state["restrict_to_context"] = restrict_to_context
            state["graph_expansion_depth"] = graph_expansion_depth
            # Add chat history for follow-up questions
            state["chat_history"] = chat_history or []
            state["context_documents"] = context_documents or []
            # Add category filter override
            state["category_filter"] = category_filter
            # Allow per-request model overrides for LLM and embeddings
            if llm_model:
                state["llm_model"] = llm_model
            if embedding_model:
                state["embedding_model"] = embedding_model

            # Record chat history and context documents in state
            state["chat_history"] = chat_history or []
            state["context_documents"] = context_documents or []

            # Run the workflow with a dict-based state
            logger.info(f"Processing query through RAG pipeline: {user_query}")
            # Acquire singleflight lock for this logical response to avoid duplicate work
            try:
                cache = get_response_cache()
                try:
                    logger.info(f"Acquiring singleflight lock for key: {cache_key}")
                    with ResponseKeyLock(cache_key, timeout=30) as acquired:
                        logger.info(f"Singleflight lock acquired={acquired} for key: {cache_key}")
                        # re-check cache after acquiring lock
                        existing = cache.get(cache_key)
                        if existing is not None:
                            logger.info("Cache populated while waiting for lock — returning cached response")
                            return existing

                        # Invoke workflow while holding the singleflight lock
                        final_state_dict = self.workflow.invoke(state)

                        # Rebuild RAGState object from returned dict for backward compatibility
                        final_state = RAGState()
                        for k, v in (final_state_dict or {}).items():
                            setattr(final_state, k, v)

                        context_docs = getattr(final_state, "context_documents", [])
                        metadata = getattr(final_state, "metadata", {}) or {}
                        history_turns = len(chat_history or [])

                        if context_docs:
                            metadata = {**metadata, "context_documents": context_docs}
                        metadata["chat_history_turns"] = history_turns
                        
                        # Include routing information in metadata
                        routing_categories = getattr(final_state, "routing_categories", [])
                        routing_confidence = getattr(final_state, "routing_confidence", 0.0)
                        if routing_categories:
                            metadata["routing_info"] = {
                                "categories": routing_categories,
                                "confidence": routing_confidence,
                                "category_id": getattr(final_state, "routing_category_id", None),
                            }

                        # Calculate quality score if not present
                        quality_score = getattr(final_state, "quality_score", None)
                        if not quality_score:
                            relevant_chunks = getattr(final_state, "graph_context", []) or getattr(
                                final_state, "retrieved_chunks", []
                            )
                            relevant_chunks = [
                                chunk
                                for chunk in relevant_chunks
                                if chunk.get("similarity", chunk.get("hybrid_score", 0.0)) > 0.0
                            ]
                            quality_score = quality_scorer.calculate_quality_score(
                                answer=getattr(final_state, "response", ""),
                                query=user_query,
                                context_chunks=relevant_chunks,
                                sources=getattr(final_state, "sources", []),
                            )
                            setattr(final_state, "quality_score", quality_score)

                        if quality_score:
                            metadata["quality_score"] = quality_score

                        setattr(final_state, "metadata", metadata)

                        # Extract timing metrics from stages for quality monitoring
                        stages = getattr(final_state, "stages", [])
                        retrieval_time = sum(
                            s.get("duration_ms", 0)
                            for s in stages
                            if "retrieve" in s.get("name", "").lower()
                        )
                        generation_time = sum(
                            s.get("duration_ms", 0)
                            for s in stages
                            if "generat" in s.get("name", "").lower()
                        )

                        # Prepare result dict and write to cache while still holding the lock
                        result = {
                            "query": user_query,
                            "response": getattr(final_state, "response", ""),
                            "sources": getattr(final_state, "sources", []),
                            "retrieved_chunks": getattr(final_state, "retrieved_chunks", []),
                            "alternative_chunks": getattr(final_state, "alternative_chunks", []),
                            "graph_context": getattr(final_state, "graph_context", []),
                            "query_analysis": getattr(final_state, "query_analysis", {}),
                            "metadata": getattr(final_state, "metadata", {}),
                            "quality_score": getattr(final_state, "quality_score", None),
                            "context_documents": context_docs,
                            "stages": getattr(final_state, "stages", []),
                        }

                        try:
                            # Use .set() for CacheService
                            cache.set(cache_key, result, workspace_id=session_id)
                            # We wrote a computed response into the cache — count as a miss that led to compute
                            cache_metrics.record_response_miss()
                        except Exception:
                            logger.warning("Failed to write response to cache; continuing without caching")

                        # Record quality metrics for monitoring
                        try:
                            quality_monitor.record_query(
                                query=user_query,
                                query_type=retrieval_mode,
                                retrieval_latency_ms=retrieval_time,
                                generation_latency_ms=generation_time,
                                num_chunks_retrieved=len(result.get("retrieved_chunks", [])),
                                quality_score=quality_score.get("total") if quality_score else None,
                                quality_breakdown=quality_score.get("breakdown") if quality_score else None,
                                quality_confidence=quality_score.get("confidence") if quality_score else None,
                                cache_hit=False,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to record quality metrics: {e}")

                        # Return result while still inside lock to ensure first writer returns directly
                        return result
                except Exception:
                    # If lock failed or errored, fall back to direct invocation
                    logger.warning("ResponseKeyLock failed or timed out; invoking workflow without lock")
                    final_state_dict = self.workflow.invoke(state)
            except Exception:
                # If cache subsystem unavailable, just run workflow
                final_state_dict = self.workflow.invoke(state)

            # Rebuild RAGState object from returned dict for backward compatibility
            final_state = RAGState()
            for k, v in (final_state_dict or {}).items():
                setattr(final_state, k, v)

            context_docs = getattr(final_state, "context_documents", [])
            metadata = getattr(final_state, "metadata", {}) or {}
            history_turns = len(chat_history or [])

            if context_docs:
                metadata = {**metadata, "context_documents": context_docs}
            metadata["chat_history_turns"] = history_turns

            # Calculate quality score inside the pipeline when available
            quality_score = getattr(final_state, "quality_score", None)
            if not quality_score:
                relevant_chunks = getattr(final_state, "graph_context", []) or getattr(
                    final_state, "retrieved_chunks", []
                )
                relevant_chunks = [
                    chunk
                    for chunk in relevant_chunks
                    if chunk.get("similarity", chunk.get("hybrid_score", 0.0)) > 0.0
                ]
                quality_score = quality_scorer.calculate_quality_score(
                    answer=getattr(final_state, "response", ""),
                    query=user_query,
                    context_chunks=relevant_chunks,
                    sources=getattr(final_state, "sources", []),
                )
                setattr(final_state, "quality_score", quality_score)

            if quality_score:
                metadata["quality_score"] = quality_score

            setattr(final_state, "metadata", metadata)

            # Extract timing metrics from stages for quality monitoring
            stages = getattr(final_state, "stages", [])
            retrieval_time = sum(
                s.get("duration_ms", 0)
                for s in stages
                if "retrieve" in s.get("name", "").lower()
            )
            generation_time = sum(
                s.get("duration_ms", 0)
                for s in stages
                if "generat" in s.get("name", "").lower()
            )

            # Prepare results
            alternative_chunks = getattr(final_state, "alternative_chunks", [])
            logger.info(f"Preparing result with {len(alternative_chunks)} alternative chunks")
            result = {
                "query": user_query,
                "response": getattr(final_state, "response", ""),
                "sources": getattr(final_state, "sources", []),
                "retrieved_chunks": getattr(final_state, "retrieved_chunks", []),
                "alternative_chunks": alternative_chunks,
                "graph_context": getattr(final_state, "graph_context", []),
                "query_analysis": getattr(final_state, "query_analysis", {}),
                "metadata": getattr(final_state, "metadata", {}),
                "quality_score": getattr(final_state, "quality_score", None),
                "context_documents": context_docs,
                "stages": getattr(final_state, "stages", []),
            }

            # Store the result in response cache for fast subsequent retrievals
            try:
                cache = get_response_cache()
                try:
                    cache.set(cache_key, result, workspace_id=session_id)
                    cache_metrics.record_response_miss()
                except Exception:
                    logger.warning("Failed to write response to cache; continuing without caching")
            except Exception:
                # If cache subsystem unavailable, ignore and continue
                pass

            # Record quality metrics for monitoring
            try:
                quality_monitor.record_query(
                    query=user_query,
                    query_type=retrieval_mode,
                    retrieval_latency_ms=retrieval_time,
                    generation_latency_ms=generation_time,
                    num_chunks_retrieved=len(result.get("retrieved_chunks", [])),
                    quality_score=quality_score.get("total") if quality_score else None,
                    quality_breakdown=quality_score.get("breakdown") if quality_score else None,
                    quality_confidence=quality_score.get("confidence") if quality_score else None,
                    cache_hit=False,
                )
            except Exception as e:
                logger.warning(f"Failed to record quality metrics: {e}")

            return result

        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")

            # Record error metrics for monitoring
            try:
                quality_monitor.record_query(
                    query=user_query,
                    query_type=retrieval_mode,
                    retrieval_latency_ms=0.0,
                    generation_latency_ms=0.0,
                    num_chunks_retrieved=0,
                    quality_score=None,
                    quality_breakdown=None,
                    quality_confidence=None,
                    cache_hit=False,
                    error=str(e),
                )
            except Exception as monitor_error:
                logger.warning(f"Failed to record error metrics: {monitor_error}")

            return {
                "query": user_query,
                "response": f"I apologize, but I encountered an error processing your query: {str(e)}",
                "sources": [],
                "retrieved_chunks": [],
                "alternative_chunks": [],
                "graph_context": [],
                "query_analysis": {},
                "metadata": {"error": str(e)},
                "quality_score": None,
                "context_documents": context_documents or [],
                "stages": [],
            }

    async def stream_query(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        retrieval_mode: str = "graph_enhanced",
        top_k: int = 5,
        temperature: float = 0.7,
        chunk_weight: float = 0.5,
        entity_weight: Optional[float] = None,
        path_weight: Optional[float] = None,
        graph_expansion: bool = True,
        use_multi_hop: bool = False,
        max_hops: Optional[int] = None,
        beam_size: Optional[int] = None,
        restrict_to_context: bool = True,
        graph_expansion_depth: Optional[int] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        context_documents: Optional[List[str]] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ):
        """
        Async generator that streams SSE-formatted strings by running the
        retrieval and reasoning nodes synchronously then streaming model tokens
        produced by the generation node.
        """
        try:
            # Run analysis/retrieval/reasoning in thread to avoid blocking the loop
            chat_hist = chat_history or []

            # Analyze query
            analysis = await asyncio.to_thread(analyze_query, user_query, chat_hist)

            # Retrieval (synchronous wrapper will handle internal async)
            retrieved_chunks = await asyncio.to_thread(
                retrieve_documents,
                user_query,
                analysis,
                retrieval_mode,
                top_k,
                chunk_weight,
                entity_weight,
                path_weight,
                graph_expansion,
                use_multi_hop,
                max_hops,
                beam_size,
                restrict_to_context,
                graph_expansion_depth,
                context_documents,
                embedding_model,
            )

            # Graph reasoning
            graph_context = await asyncio.to_thread(
                reason_with_graph, user_query, retrieved_chunks, analysis, retrieval_mode
            )

            # Emit stage events
            stages = ["query_analysis", "retrieval", "graph_reasoning", "generation"]
            for st in stages[:-1]:
                stage_data = {"type": "stage", "content": st}
                yield f"data: {__import__('json').dumps(stage_data)}\n\n"
                # small pause to give frontend chance to render stage updates
                await asyncio.sleep(0.01)

            # Prepare streaming generation
            from rag.nodes import generation as generation_module


            # Create a bounded thread-safe queue to receive token fragments from generator
            # Bounded queue helps provide backpressure to the producer
            q: _queue.Queue = _queue.Queue(maxsize=64)

            stop_event = cancel_event or asyncio.Event()

            def produce():
                try:
                    gen = generation_module.stream_generate_response(
                        user_query,
                        graph_context,
                        analysis,
                        temperature=temperature,
                        chat_history=chat_hist,
                        llm_model=llm_model,
                    )

                    for token in gen:
                        # Respect cancellation from the request side
                        try:
                            if stop_event.is_set():
                                logger.info("Producer detected cancel event; stopping production")
                                break
                        except Exception:
                            pass

                        # Try to put token into the queue without blocking forever
                        try:
                            q.put(token, timeout=1)
                        except Exception:
                            # If we can't enqueue (consumer too slow), check cancellation and continue
                            try:
                                if stop_event.is_set():
                                    break
                            except Exception:
                                pass
                            logger.debug("Dropping token due to full queue")
                            continue
                except Exception as e:
                    logger.error(f"Producer thread error in stream_query: {e}")
                finally:
                    # Signal consumer we are done
                    try:
                        q.put(None, timeout=1)
                    except Exception:
                        # If queue is full or put fails, swallow — consumer will eventually stop
                        pass

            t = threading.Thread(target=produce, daemon=True)
            t.start()

            # Emit generation stage marker
            stage_data = {"type": "stage", "content": "generation"}
            yield f"data: {__import__('json').dumps(stage_data)}\n\n"

            # Stream tokens as they are pushed into the queue
            response_buffer = []
            while True:
                # If client requested cancel, drain and stop
                try:
                    if cancel_event and cancel_event.is_set():
                        logger.info("Stream consumer detected cancel event; stopping consumption")
                        break
                except Exception:
                    pass

                token = await asyncio.to_thread(q.get)
                if token is None:
                    break
                # Append to buffer for post-processing
                response_buffer.append(token)
                chunk_data = {"type": "token", "content": token}
                yield f"data: {__import__('json').dumps(chunk_data)}\n\n"

            # If cancellation requested, attempt to set stop_event to notify producer
            try:
                if cancel_event and cancel_event.is_set():
                    try:
                        # best-effort: set the same event object for producer
                        cancel_event.set()
                    except Exception:
                        pass
            except Exception:
                pass

            # After stream ends, compute sources and quality using buffered response
            response_text = "".join(response_buffer)

            # Reconstruct sources similar to generation node logic
            sources = []
            seen = set()
            for i, chunk in enumerate(graph_context):
                citation = f"{chunk.get('document_name','Unknown Document')}#{chunk.get('chunk_index', i)}"
                if citation in seen:
                    continue
                seen.add(citation)
                src = {
                    "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                    "content": chunk.get("content", "") ,
                    "similarity": chunk.get("similarity", chunk.get("hybrid_score", 0.0)),
                    "document_name": chunk.get("document_name", "Unknown Document"),
                    "document_id": chunk.get("document_id", ""),
                    "filename": chunk.get("filename", chunk.get("document_name", "Unknown Document")),
                    "citation": citation,
                }
                sources.append(src)

            # Quality scoring
            try:
                relevant_chunks = [c for c in graph_context if c.get("similarity", c.get("hybrid_score",0.0))>0.0]
                quality = quality_scorer.calculate_quality_score(
                    answer=response_text,
                    query=user_query,
                    context_chunks=relevant_chunks,
                    sources=sources,
                )
            except Exception:
                quality = None

            # Emit sources, quality, and done signals
            yield f"data: {__import__('json').dumps({'type':'sources','content':sources})}\n\n"
            if quality:
                yield f"data: {__import__('json').dumps({'type':'quality_score','content':quality})}\n\n"

            # Emit metadata
            metadata = {"session_id": session_id or '__stream__'}
            yield f"data: {__import__('json').dumps({'type':'metadata','content':metadata})}\n\n"

            yield f"data: {__import__('json').dumps({'type':'done'})}\n\n"

        except Exception as e:
            logger.error(f"Stream query failed: {e}")
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {__import__('json').dumps(error_data)}\n\n"

    async def aquery(self, user_query: str) -> Dict[str, Any]:
        """
        Async version of query processing.

        Args:
            user_query: User's question or request

        Returns:
            Dictionary containing response and metadata
        """
        # For now, just call the sync version
        # Future enhancement: implement full async pipeline
        return self.query(user_query)


# Global GraphRAG instance
graph_rag = GraphRAG()
