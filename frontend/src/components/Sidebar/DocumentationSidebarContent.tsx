'use client';

import { useState, useEffect } from 'react';
import { FileText, Folder, ChevronRight, Search, X } from 'lucide-react';

interface DocFile {
  name: string;
  path: string;
  type: 'file' | 'directory';
}

interface DocSection {
  id: string;
  title: string;
  files: DocFile[];
}

interface DocumentationSidebarContentProps {
  onFileSelect: (path: string) => void;
  selectedFile: string | null;
}

export default function DocumentationSidebarContent({ onFileSelect, selectedFile: initialSelectedFile }: DocumentationSidebarContentProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['features']));
  const [selectedFile, setSelectedFile] = useState<string | null>(initialSelectedFile);

  // Listen for file changes from DocumentationView
  useEffect(() => {
    const handleFileChanged = (event: CustomEvent<string>) => {
      setSelectedFile(event.detail);
    };

    window.addEventListener('documentation-file-changed', handleFileChanged as EventListener);
    return () => {
      window.removeEventListener('documentation-file-changed', handleFileChanged as EventListener);
    };
  }, []);

  // Documentation structure
  const docSections: DocSection[] = [
    {
      id: 'getting-started',
      title: 'Getting Started',
      files: [
        { name: 'Overview', path: '01-getting-started/README.md', type: 'file' },
        { name: 'Docker Setup', path: '01-getting-started/docker-setup.md', type: 'file' },
        { name: 'Local Development', path: '01-getting-started/local-development.md', type: 'file' },
        { name: 'Architecture', path: '01-getting-started/architecture-overview.md', type: 'file' },
        { name: 'Configuration', path: '01-getting-started/configuration.md', type: 'file' },
      ],
    },
    {
      id: 'core-concepts',
      title: 'Core Concepts',
      files: [
        { name: 'Overview', path: '02-core-concepts/README.md', type: 'file' },
        { name: 'Graph RAG Pipeline', path: '02-core-concepts/graph-rag-pipeline.md', type: 'file' },
        { name: 'Data Model', path: '02-core-concepts/data-model.md', type: 'file' },
        { name: 'Entity Types', path: '02-core-concepts/entity-types.md', type: 'file' },
        { name: 'Retrieval Strategies', path: '02-core-concepts/retrieval-strategies.md', type: 'file' },
        { name: 'Caching System', path: '02-core-concepts/caching-system.md', type: 'file' },
      ],
    },
    {
      id: 'components',
      title: 'Components',
      files: [
        { name: 'Overview', path: '03-components/README.md', type: 'file' },
        { name: 'Backend Overview', path: '03-components/backend/README.md', type: 'file' },
        { name: 'Frontend Overview', path: '03-components/frontend/README.md', type: 'file' },
        { name: 'Ingestion Overview', path: '03-components/ingestion/README.md', type: 'file' },
      ],
    },
    {
      id: 'features',
      title: 'Features',
      files: [
        { name: 'Overview', path: '04-features/README.md', type: 'file' },
        { name: 'Query Routing', path: '04-features/query-routing.md', type: 'file' },
        { name: 'Routing Metrics', path: '04-features/routing-metrics.md', type: 'file' },
        { name: 'Smart Consolidation', path: '04-features/smart-consolidation.md', type: 'file' },
        { name: 'Category Prompts', path: '04-features/category-prompts.md', type: 'file' },
        { name: 'Structured KG', path: '04-features/structured-kg.md', type: 'file' },
        { name: 'Adaptive Routing', path: '04-features/adaptive-routing.md', type: 'file' },
        { name: 'Hybrid Retrieval', path: '04-features/hybrid-retrieval.md', type: 'file' },
        { name: 'Entity Reasoning', path: '04-features/entity-reasoning.md', type: 'file' },
        { name: 'Community Detection', path: '04-features/community-detection.md', type: 'file' },
        { name: 'Chat Tuning', path: '04-features/chat-tuning.md', type: 'file' },
        { name: 'Document Upload', path: '04-features/document-upload.md', type: 'file' },
        { name: 'Conversation History', path: '04-features/conversation-history.md', type: 'file' },
      ],
    },
    {
      id: 'data-flows',
      title: 'Data Flows',
      files: [
        { name: 'Overview', path: '05-data-flows/README.md', type: 'file' },
        { name: 'Chat Query Flow', path: '05-data-flows/chat-query-flow.md', type: 'file' },
        { name: 'Document Ingestion', path: '05-data-flows/document-ingestion-flow.md', type: 'file' },
        { name: 'Entity Extraction', path: '05-data-flows/entity-extraction-flow.md', type: 'file' },
        { name: 'Graph Expansion', path: '05-data-flows/graph-expansion-flow.md', type: 'file' },
        { name: 'Reranking Flow', path: '05-data-flows/reranking-flow.md', type: 'file' },
        { name: 'SSE Streaming', path: '05-data-flows/streaming-sse-flow.md', type: 'file' },
      ],
    },
    {
      id: 'api',
      title: 'API Reference',
      files: [
        { name: 'Overview', path: '06-api-reference/README.md', type: 'file' },
        { name: 'Chat Endpoints', path: '06-api-reference/chat-endpoints.md', type: 'file' },
        { name: 'Document Endpoints', path: '06-api-reference/document-endpoints.md', type: 'file' },
        { name: 'Database Endpoints', path: '06-api-reference/database-endpoints.md', type: 'file' },
        { name: 'History Endpoints', path: '06-api-reference/history-endpoints.md', type: 'file' },
        { name: 'Jobs Endpoints', path: '06-api-reference/jobs-endpoints.md', type: 'file' },
        { name: 'Models & Schemas', path: '06-api-reference/models-schemas.md', type: 'file' },
      ],
    },
    {
      id: 'configuration',
      title: 'Configuration',
      files: [
        { name: 'Overview', path: '07-configuration/README.md', type: 'file' },
        { name: 'Environment Variables', path: '07-configuration/environment-variables.md', type: 'file' },
        { name: 'Optimal Defaults', path: '07-configuration/optimal-defaults.md', type: 'file' },
        { name: 'Feature Flags', path: '07-configuration/feature-flags.md', type: 'file' },
        { name: 'Caching Settings', path: '07-configuration/caching-settings.md', type: 'file' },
        { name: 'Clustering Settings', path: '07-configuration/clustering-settings.md', type: 'file' },
        { name: 'RAG Tuning', path: '07-configuration/rag-tuning.md', type: 'file' },
      ],
    },
    {
      id: 'operations',
      title: 'Operations',
      files: [
        { name: 'Overview', path: '08-operations/README.md', type: 'file' },
        { name: 'Docker Deployment', path: '08-operations/docker-deployment.md', type: 'file' },
        { name: 'Local Runtime', path: '08-operations/local-runtime.md', type: 'file' },
        { name: 'Monitoring', path: '08-operations/monitoring.md', type: 'file' },
        { name: 'Troubleshooting', path: '08-operations/troubleshooting.md', type: 'file' },
        { name: 'Maintenance', path: '08-operations/maintenance-reindexing.md', type: 'file' },
      ],
    },
    {
      id: 'development',
      title: 'Development',
      files: [
        { name: 'Overview', path: '09-development/README.md', type: 'file' },
        { name: 'Contributing', path: '09-development/contributing.md', type: 'file' },
        { name: 'Coding Standards', path: '09-development/coding-standards.md', type: 'file' },
        { name: 'Testing Backend', path: '09-development/testing-backend.md', type: 'file' },
        { name: 'Testing Frontend', path: '09-development/testing-frontend.md', type: 'file' },
        { name: 'Dev Scripts', path: '09-development/dev-scripts.md', type: 'file' },
        { name: 'Feature Flag Wiring', path: '09-development/feature-flag-wiring.md', type: 'file' },
      ],
    },
    {
      id: 'scripts',
      title: 'Scripts',
      files: [
        { name: 'Overview', path: '10-scripts/README.md', type: 'file' },
        { name: 'Setup Neo4j', path: '10-scripts/setup-neo4j.md', type: 'file' },
        { name: 'Ingest Documents', path: '10-scripts/ingest-documents.md', type: 'file' },
        { name: 'Run Clustering', path: '10-scripts/run-clustering.md', type: 'file' },
        { name: 'Create Similarities', path: '10-scripts/create-similarities.md', type: 'file' },
        { name: 'Inspect Entities', path: '10-scripts/inspect-entities.md', type: 'file' },
        { name: 'Reindex Classification', path: '10-scripts/reindex-classification.md', type: 'file' },
        { name: 'FlashRank Prewarm', path: '10-scripts/flashrank-prewarm-worker.md', type: 'file' },
        { name: 'Build Leiden Projection', path: '10-scripts/build-leiden-projection.md', type: 'file' },
      ],
    },
  ];

  const toggleFolder = (folderId: string) => {
    const newExpanded = new Set(expandedFolders);
    if (newExpanded.has(folderId)) {
      newExpanded.delete(folderId);
    } else {
      newExpanded.add(folderId);
    }
    setExpandedFolders(newExpanded);
  };

  const filteredSections = searchQuery
    ? docSections
      .map(section => {
        const query = searchQuery.toLowerCase();
        const sectionTitleMatch = section.title.toLowerCase().includes(query);

        // If section title matches, show all files in that section
        if (sectionTitleMatch) {
          return section;
        }

        // Otherwise, filter files by name or path
        const matchingFiles = section.files.filter(file =>
          file.name.toLowerCase().includes(query) ||
          file.path.toLowerCase().includes(query)
        );

        return {
          ...section,
          files: matchingFiles,
        };
      })
      .filter(section => section.files.length > 0)
    : docSections;

  return (
    <div className="flex flex-col h-full">
      {/* Search */}
      <div className="p-4 border-b" style={{ borderColor: 'var(--border)' }}>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search docs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-9 py-2 text-sm border rounded-lg focus:outline-none focus:ring-2 focus:border-transparent"
            style={{
              borderColor: 'var(--border)',
              background: 'var(--bg-primary)',
              color: 'var(--text-primary)',
            }}
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
              aria-label="Clear search"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Documentation Tree */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {filteredSections.map((section) => (
          <div key={section.id}>
            <button
              onClick={() => toggleFolder(section.id)}
              className="flex items-center gap-2 w-full text-left mb-2 text-sm font-semibold transition-colors"
              style={{ color: 'var(--text-secondary)' }}
              onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--accent-primary)')}
              onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-secondary)')}
            >
              <ChevronRight
                className={`w-4 h-4 transition-transform ${expandedFolders.has(section.id) ? 'rotate-90' : ''
                  }`}
              />
              <Folder className="w-4 h-4" />
              <span>{section.title}</span>
            </button>

            {expandedFolders.has(section.id) && (
              <div className="ml-6 space-y-1">
                {section.files.map((file) => (
                  <button
                    key={file.path}
                    onClick={() => onFileSelect(file.path)}
                    className={`flex items-center gap-2 w-full text-left px-3 py-2 text-sm rounded-lg transition-colors ${selectedFile === file.path
                      ? 'bg-orange-50 dark:bg-orange-900/20'
                      : 'hover:bg-gray-50 dark:hover:bg-neutral-800'
                      }`}
                    style={{
                      color: selectedFile === file.path ? 'var(--accent-primary)' : 'var(--text-secondary)',
                    }}
                  >
                    <FileText className="w-4 h-4" />
                    <span>{file.name}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
