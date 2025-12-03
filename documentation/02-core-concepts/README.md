# Core Concepts

Fundamental concepts underlying the Amber platform architecture.

## Contents

- [README](02-core-concepts/README.md) - Core concepts overview
- [Graph-Enhanced RAG](02-core-concepts/graph-rag-pipeline.md) - What graph-enhanced retrieval-augmented generation is and why it matters
- [Data Model](02-core-concepts/data-model.md) - Neo4j schema: Document, Chunk, and Entity nodes with relationships
- [Entity Types](02-core-concepts/entity-types.md) - Canonical entity taxonomy and relationship patterns
- [Retrieval Strategies](02-core-concepts/retrieval-strategies.md) - Hybrid retrieval combining vector search, graph expansion, and reranking
- [Caching System](02-core-concepts/caching-system.md) - Multi-layer cache architecture for performance optimization

## Overview

Amber implements graph-enhanced Retrieval-Augmented Generation (RAG), which extends traditional vector-based RAG with knowledge graph capabilities. This combination enables:

- **Contextual retrieval**: Vector similarity plus entity relationships
- **Multi-hop reasoning**: Traverse entity connections to find related context
- **Semantic clustering**: Group entities into communities for topic discovery
- **Hybrid scoring**: Combine embedding similarity with graph structure signals

## Key Concepts

### Graph-Enhanced RAG

Traditional RAG retrieves context using embedding similarity alone. Graph-enhanced RAG adds a knowledge graph layer where entities extracted from documents become nodes, and relationships become edges. During retrieval, the system:

1. Finds relevant chunks via vector similarity
2. Expands to related chunks via graph edges
3. Includes entities and their connections in the context
4. Reranks results using both embedding and graph signals

See [Graph-Enhanced RAG](02-core-concepts/graph-rag-pipeline.md) for details.

### Data Model

The Neo4j database stores three primary node types:

- **Document**: Ingested files with metadata and precomputed statistics
- **Chunk**: Text segments with embeddings and quality scores
- **Entity**: Extracted named entities with type and community assignments

Relationships connect these nodes:
- `CONTAINS`: Document → Chunk
- `MENTIONS`: Chunk → Entity
- `SIMILAR_TO`: Chunk ↔ Chunk (cosine similarity)
- `RELATED_TO`: Entity ↔ Entity (co-occurrence and semantic relationships)

See [Data Model](02-core-concepts/data-model.md) for schema details.

### Entity Taxonomy

Entities are classified into canonical types including:

- **Infrastructure**: Component, Service, Node, Domain
- **Access Control**: Account, Role, Resource, Quota
- **Data Objects**: Item, Storage Object, Backup Object
- **Operations**: Task, Procedure, Migration Procedure, CLI Command
- **Documentation**: Concept, Document, Config Option, Security Feature
- **General**: Person, Organization, Location, Event, Technology, Product

See [Entity Types](02-core-concepts/entity-types.md) for complete taxonomy and extraction rules.

### Retrieval Strategies

Amber supports multiple retrieval modes:

- **Vector-only**: Pure embedding similarity (baseline)
- **Hybrid**: Weighted combination of chunks and entities
- **Graph-expansion**: Multi-hop traversal of relationships
- **Entity-focused**: Prioritize entity mentions in context

Optional reranking with FlashRank cross-encoder improves precision.

See [Retrieval Strategies](02-core-concepts/retrieval-strategies.md) for algorithm details.

### Caching System

Multi-layer caching reduces latency by 30-50% for repeated queries:

- **Entity label cache**: TTL cache for entity name lookups (70-80% hit rate)
- **Embedding cache**: LRU cache for text embeddings (40-60% hit rate)
- **Retrieval cache**: TTL cache for query results (20-30% hit rate)
- **Response cache**: TTL cache for complete pipeline responses (configurable)

All caches are feature-flagged and monitored via `/api/database/cache-stats`.

See [Caching System](02-core-concepts/caching-system.md) for implementation details.

## Design Principles

1. **Modularity**: Each component (retrieval, extraction, generation) is independently testable and replaceable
2. **Observability**: Pipeline stages emit progress events; caches expose metrics
3. **Configurability**: Runtime tuning of retrieval parameters without code changes
4. **Performance**: Caching, batching, and connection pooling minimize latency
5. **Extensibility**: New entity types, retrieval modes, and rerankers can be added via configuration

## Learning Resources

After understanding core concepts:
- Study [RAG Pipeline](03-components/backend/rag-pipeline.md) implementation
- Review [Chat Query Flow](05-data-flows/chat-query-flow.md) end-to-end
- Explore [Entity Extraction](03-components/backend/entity-extraction.md) process
- Configure [Retrieval Parameters](07-configuration/rag-tuning.md)

## Related Documentation

- [Architecture Overview](01-getting-started/architecture-overview.md)
- [Backend Components](03-components/backend)
- [Data Flows](05-data-flows)
- [Configuration Reference](07-configuration)
