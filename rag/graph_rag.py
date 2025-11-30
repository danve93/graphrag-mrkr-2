"""
LangGraph-based RAG pipeline implementation.
"""

import logging
import threading
import hashlib
import asyncio
import queue as _queue
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from core.singletons import get_response_cache, ResponseKeyLock, hash_response_params
from core.cache_metrics import cache_metrics
from rag.nodes.graph_reasoning import reason_with_graph
from rag.nodes.query_analysis import analyze_query
from rag.nodes.retrieval import retrieve_documents
from core.quality_scorer import quality_scorer

logger = logging.getLogger(__name__)


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
        self.stages: List[str] = []  # Track stages for UI


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
        workflow.add_node("retrieve_documents", self._retrieve_documents_node)
        workflow.add_node("reason_with_graph", self._reason_with_graph_node)
        workflow.add_node("generate_response", self._generate_response_node)

        # Add edges
        workflow.add_edge("analyze_query", "retrieve_documents")
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
            
            # Initialize stages list if not present
            if "stages" not in state:
                state["stages"] = []
            
            # Track stage
            state["stages"].append("query_analysis")
            logger.info(f"Stage query_analysis completed, current stages: {state['stages']}")
            
            state["query_analysis"] = analyze_query(query, chat_history)
            return state
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            state["query_analysis"] = {"error": str(e)}
            return state

    def _retrieve_documents_node(self, state) -> Any:
        """Retrieve relevant documents (dict-based state for LangGraph)."""
        try:
            logger.info("Retrieving relevant documents")
            
            # Initialize stages list if not present
            if "stages" not in state:
                state["stages"] = []
            
            # Track retrieval stage
            state["stages"].append("retrieval")
            logger.info(f"Stage retrieval completed, current stages: {state['stages']}")
            
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

            state["retrieved_chunks"] = retrieve_documents(
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
                context_documents=state.get("context_documents", []),
            )
            # Debug: log retrieved chunk summary (count + sample ids/similarities)
            try:
                retrieved = state.get("retrieved_chunks", []) or []
                sample_info = [
                    {
                        "chunk_id": c.get("chunk_id") or c.get("id"),
                        "similarity": c.get("similarity", c.get("hybrid_score", 0.0)),
                    }
                    for c in retrieved[:5]
                ]
                logger.info(
                    "Post-retrieval: %d chunks retrieved. Sample: %s",
                    len(retrieved),
                    sample_info,
                )
            except Exception:
                logger.debug("Failed to log retrieved chunk sample")

            return state
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            state["retrieved_chunks"] = []
            return state

    def _reason_with_graph_node(self, state) -> Any:
        """Perform graph-based reasoning (dict-based state for LangGraph)."""
        try:
            logger.info("Performing graph reasoning")
            
            # Initialize stages list if not present
            if "stages" not in state:
                state["stages"] = []
            
            # Track stage
            state["stages"].append("graph_reasoning")
            logger.info(f"Stage graph_reasoning completed, current stages: {state['stages']}")
            
            state["graph_context"] = reason_with_graph(
                state.get("query", ""),
                state.get("retrieved_chunks", []),
                state.get("query_analysis", {}),
                state.get("retrieval_mode", "graph_enhanced"),
            )
            # Debug: log graph context summary
            try:
                graph_ctx = state.get("graph_context", []) or []
                sample_graph = [
                    {"chunk_id": c.get("chunk_id") or c.get("id"), "similarity": c.get("similarity", c.get("hybrid_score", 0.0))}
                    for c in graph_ctx[:5]
                ]
                logger.info("Post-graph-reasoning: %d items in graph_context. Sample: %s", len(graph_ctx), sample_graph)
            except Exception:
                logger.debug("Failed to log graph_context sample")
            return state
        except Exception as e:
            logger.error(f"Graph reasoning failed: {e}")
            state["graph_context"] = state.get("retrieved_chunks", [])
            return state

    def _generate_response_node(self, state) -> Any:
        """Generate the final response (dict-based state for LangGraph)."""
        try:
            logger.info("Generating response")
            
            # Initialize stages list if not present
            if "stages" not in state:
                state["stages"] = []
            
            # Track stage
            state["stages"].append("generation")
            logger.info(f"Stage generation completed, current stages: {state['stages']}")
            # Debug: log what will be passed to generation
            try:
                retrieved = state.get("retrieved_chunks", []) or []
                graph_ctx = state.get("graph_context", []) or []
                logger.info(
                    "About to generate response — retrieved_chunks=%d, graph_context=%d",
                    len(retrieved),
                    len(graph_ctx),
                )
            except Exception:
                logger.debug("Failed to log pre-generation context sizes")

            # dynamic import so tests can monkeypatch `rag.nodes.generation.generate_response`
            try:
                from rag.nodes import generation as generation_module

                response_data = generation_module.generate_response(
                    state.get("query", ""),
                    state.get("graph_context", []),
                    state.get("query_analysis", {}),
                    state.get("temperature", 0.7),
                    state.get("chat_history", []),
                    llm_model=state.get("llm_model", None),
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
                )

            state["response"] = response_data.get("response", "")
            state["sources"] = response_data.get("sources", [])
            state["metadata"] = response_data.get("metadata", {})
            # Capture quality score computed during generation (if available)
            state["quality_score"] = response_data.get("quality_score", None)

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
                    session_id=session_id or "__global__",
                    chat_history_hash=chat_history_hash,
                )

                cached = cache.get(cache_key)
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

                        # Prepare result dict and write to cache while still holding the lock
                        result = {
                            "query": user_query,
                            "response": getattr(final_state, "response", ""),
                            "sources": getattr(final_state, "sources", []),
                            "retrieved_chunks": getattr(final_state, "retrieved_chunks", []),
                            "graph_context": getattr(final_state, "graph_context", []),
                            "query_analysis": getattr(final_state, "query_analysis", {}),
                            "metadata": getattr(final_state, "metadata", {}),
                            "quality_score": getattr(final_state, "quality_score", None),
                            "context_documents": context_docs,
                            "stages": getattr(final_state, "stages", []),
                        }

                        try:
                            cache[cache_key] = result
                            # We wrote a computed response into the cache — count as a miss that led to compute
                            cache_metrics.record_response_miss()
                        except Exception:
                            logger.warning("Failed to write response to cache; continuing without caching")

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

            # Prepare results
            result = {
                "query": user_query,
                "response": getattr(final_state, "response", ""),
                "sources": getattr(final_state, "sources", []),
                "retrieved_chunks": getattr(final_state, "retrieved_chunks", []),
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
                    cache[cache_key] = result
                    cache_metrics.record_response_miss()
                except Exception:
                    logger.warning("Failed to write response to cache; continuing without caching")
            except Exception:
                # If cache subsystem unavailable, ignore and continue
                pass

            return result

        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            return {
                "query": user_query,
                "response": f"I apologize, but I encountered an error processing your query: {str(e)}",
                "sources": [],
                "retrieved_chunks": [],
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
