"""
LangGraph-based RAG pipeline implementation.
"""

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from rag.nodes.generation import generate_response
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

            response_data = generate_response(
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

            # Run the workflow with a dict-based state
            logger.info(f"Processing query through RAG pipeline: {user_query}")
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

            # Return results
            return {
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
