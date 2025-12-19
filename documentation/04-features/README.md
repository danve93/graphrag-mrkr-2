# Features

Advanced features and capabilities of the Amber platform.

## Contents

- [README](04-features/README.md) - Features overview

### Retrieval & RAG
- [Hybrid Retrieval](04-features/hybrid-retrieval.md) - Multi-signal retrieval combining vector, graph, and keyword search
- [Multi-Stage Retrieval](04-features/multi-stage-retrieval.md) - BM25 pre-filtering for large corpora speedup
- [Query Expansion](04-features/query-expansion.md) - Synonym and abbreviation expansion for improved recall
- [Fuzzy Matching](04-features/fuzzy-matching.md) - Typo tolerance and technical term matching
- [Temporal Retrieval](04-features/temporal-retrieval.md) - Time-based filtering and recency boosting
- [Client-Side Vector Search](04-features/client-side-vector-search.md) - Low-latency static entity matching
- [Reranking](05-data-flows/reranking-flow.md) - Cross-encoder reranking (FlashRank)
- [Chat Tuning](04-features/chat-tuning.md) - Runtime retrieval parameter controls
- [Query Routing](04-features/query-routing.md) - Intelligent category-based routing
- [Smart Consolidation](04-features/smart-consolidation.md) - Diversity-aware result ranking
- [Category Prompts](04-features/category-prompts.md) - Context-aware system prompts
- [Structured KG](04-features/structured-kg.md) - Text-to-Cypher graph queries
- [Adaptive Routing](04-features/adaptive-routing.md) - Feedback-loop learning for retrieval weights

### Memory & History
- [Layered Memory System](04-features/layered-memory-system.md) - 4-layer context (Session, Facts, Summaries, Messages)
- [Conversation History](04-features/conversation-history.md) - Persistent session management

### Security & Access
- [Role-Based Access Control](04-features/role-based-access-control.md) - User roles (Admin, External, User)
- [API Key Management](04-features/api-key-management.md) - Secure access for external integrations
- [External User Integration](04-features/external-user-integration.md) - Embedded chat for third-party apps
- [Content Filtering](04-features/content-filtering.md) - Pre-ingestion quality gates

### Data Management
- [Graph Curation Workbench](04-features/graph-curation-workbench.md) - Interactive graph editing and healing
- [Incremental Updates](04-features/incremental-updates.md) - Efficient document updating via content hashing
- [Entity Clustering](04-features/community-detection.md) - Leiden community detection
- [Document Classification](04-features/document-upload.md) - Auto-labeling rules
- [Gleaning](04-features/entity-reasoning.md) - Multi-pass entity extraction
- [Orphan Cleanup](04-features/orphan-cleanup.md) - Automatic removal of disconnected chunks and entities

### Monitoring & Caching
- [Routing Metrics](04-features/routing-metrics.md) - Performance tracking
- [Response Caching](02-core-concepts/caching-system.md) - Semantic caching

## Feature Overview

### Retrieval & RAG Features

#### Hybrid Retrieval
**Status**: Production-ready
Combines vector similarity search with graph traversal (SIMILAR_TO, MENTIONS) to utilize both semantic and structural signals.
See [Hybrid Retrieval](04-features/hybrid-retrieval.md).

#### Multi-Stage Retrieval
**Status**: Production-ready
Optimizes performance on large corpora (>5k docs) by using BM25 keyword search to pre-filter candidates before vector search. 5-10x speedup.
See [Multi-Stage Retrieval](04-features/multi-stage-retrieval.md).

#### Query Expansion
**Status**: Production-ready
Expands queries with synonyms and full terms for abbreviations (e.g., "k8s" -> "kubernetes") to improve recall.
See [Query Expansion](04-features/query-expansion.md).

#### Fuzzy Matching
**Status**: Production-ready
Uses approximate string matching to handle typos and variances in technical terms (e.g., snake_case vs camelCase).
See [Fuzzy Matching](04-features/fuzzy-matching.md).

#### Temporal Retrieval
**Status**: Production-ready
Adds time-awareness to retrieval, allowing filtering by date ranges and boosting recent documents via time-decay scoring.
See [Temporal Retrieval](04-features/temporal-retrieval.md).

#### Client-Side Vector Search
**Status**: Production-ready
Ultra-low latency (<10ms) matching for static entities using client-side (in-memory) embeddings.
See [Client-Side Vector Search](04-features/client-side-vector-search.md).

### Memory & History Features

#### Layered Memory System
**Status**: Production-ready
A sophisticated 4-layer memory architecture (Session, User Facts, Summaries, Messages) that provides deep context while reducing token usage by ~80%.
See [Layered Memory System](04-features/layered-memory-system.md).

#### Conversation History
**Status**: Production-ready
Standard chat session persistence allowing users to browse and resume past conversations.
See [Conversation History](04-features/conversation-history.md).

### Security & Access Features

#### Role-Based Access Control (RBAC)
**Status**: Production-ready
Comprehensive permission system separating Admin, User, and External roles.
See [Role-Based Access Control](04-features/role-based-access-control.md).

#### API Key Management
**Status**: Production-ready
Admin interface for generating and managing API keys to allow external applications to securely access chat features.
See [API Key Management](04-features/api-key-management.md).

#### External User Integration
**Status**: Production-ready
Simplified, embeddable chat interface for external users authenticated via API keys.
See [External User Integration](04-features/external-user-integration.md).

#### Content Filtering
**Status**: Production-ready
Heuristic-based filter that rejects low-quality content (spam, repetitive text, mostly special chars) before embedding to save costs.
See [Content Filtering](04-features/content-filtering.md).

### Data Management Features

#### Graph Curation Workbench
**Status**: Production-ready
Interactive visual tool for manual graph refinement, allowing admins to add/remove edges, merge nodes, and use AI to "heal" the graph.
See [Graph Curation Workbench](04-features/graph-curation-workbench.md).

#### Incremental Updates
**Status**: Production-ready
Smart document updating that hashes chunk content to only re-process changed sections, preserving existing embeddings where possible.
See [Incremental Updates](04-features/incremental-updates.md).

#### Entity Clustering
**Status**: Production-ready
Leiden community detection to group entities into semantic clusters for visualization and retrieval.
See [Entity Clustering](04-features/community-detection.md).

## Feature Flags

All features are controlled via environment variables or runtime configuration:

| Feature | Flag | Default |
|---------|------|---------|
| Hybrid Retrieval | `RETRIEVAL_MODE=hybrid` | `hybrid` |
| Two-Stage Retrieval | `ENABLE_TWO_STAGE_RETRIEVAL` | `true` |
| Query Expansion | `ENABLE_QUERY_EXPANSION` | `true` |
| Fuzzy Matching | `ENABLE_FUZZY_MATCHING` | `true` |
| Temporal Filtering | `ENABLE_TEMPORAL_FILTERING` | `true` |
| Memory System | `ENABLE_MEMORY_SYSTEM` | `false` |
| Reranking | `FLASHRANK_ENABLED` | `true` |
| Content Filtering | `ENABLE_CONTENT_FILTERING` | `false` |
| Client-Side Matching | `ENABLE_STATIC_ENTITY_MATCHING` | `true` |

## Related Documentation

- [Configuration Reference](07-configuration)
- [Backend Components](03-components/backend)
- [Operations Guide](08-operations)
