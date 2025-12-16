# Amber Documentation

Complete technical documentation for the Amber graph-enhanced RAG platform.

## Documentation Structure

### [01. Getting Started](01-getting-started)
Quick start guides, installation instructions, and initial configuration for developers and operators.

### [02. Core Concepts](02-core-concepts)
Fundamental architectural concepts including graph-enhanced RAG, the Neo4j data model, entity taxonomy, retrieval strategies, and the multi-layer caching system.

### [03. Components](03-components)
Detailed technical reference for all system components organized by layer: backend API and pipeline, ingestion processors, and frontend UI.

### [04. Features](04-features)
In-depth documentation for advanced features including entity clustering, reranking, chat tuning, classification, gleaning, and response caching.
- **[External User Integration](./04-features/external-user-integration.md)**: Details on the guest/external chat access feature.
- **[Graph Curation Workbench](./04-features/graph-curation-workbench.md)**: Manual and AI-assisted tools for maintaining the Knowledge Graph (Healing, Merging, Pruning).
- **[Search & RAG Pipeline](./04-features/search-rag.md)**: Understanding the retrieval mechanisms.

### [05. Data Flows](05-data-flows)
End-to-end traces of key operations showing how data moves through the system from input to output.

### [06. API Reference](06-api-reference)
Complete REST API documentation including endpoints, request/response schemas, and authentication.

### [07. Configuration](07-configuration)
Comprehensive configuration reference covering environment variables, tuning parameters, cache settings, and optimization tradeoffs.

### [08. Operations](08-operations)
Production deployment guides, monitoring strategies, performance tuning, troubleshooting procedures, and scaling considerations.

### [09. Development](09-development)
Developer workflows including code conventions, testing procedures, feature development guidelines, and debugging techniques.

### [10. Scripts](10-scripts)
Command-line tool documentation for ingestion scripts, clustering utilities, and maintenance operations.

## Quick Links

- [Architecture Overview](01-getting-started/architecture-overview.md)
- [Local Development Setup](01-getting-started/local-development.md)
- [RAG Pipeline Details](03-components/backend/rag-pipeline.md)
- [Chat Query Flow](05-data-flows/chat-query-flow.md)
- [API Endpoints](06-api-reference)
- [Environment Variables](07-configuration/environment-variables.md)
- [Testing Guide](09-development/testing-backend.md)

## Documentation Standards

Each document follows a consistent structure:
- **Overview**: Brief description and purpose
- **Architecture**: Technical design and key components
- **Usage**: Code examples and practical implementation
- **Configuration**: Relevant settings and parameters
- **Testing**: Validation and debugging approaches
- **Related Documentation**: Cross-references to connected topics

## Contributing to Documentation

When adding or modifying features:
1. Update relevant component documentation in `03-components/`
2. Add configuration details to `07-configuration/`
3. Update data flow diagrams if request/response patterns change
4. Add API reference entries for new endpoints
5. Include code examples and test cases

## Support

For questions or issues:
- Review [Troubleshooting Guide](08-operations/troubleshooting.md)
- Check [Common Issues](08-operations/troubleshooting.md#common-issues)
- Consult component-specific documentation
- Review source code comments and docstrings
