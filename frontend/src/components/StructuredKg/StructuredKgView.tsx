'use client';

import { useState, useEffect } from 'react';
import { Database, Play, CheckCircle, XCircle, RefreshCw, Code, Link2, Network } from 'lucide-react';
import { Button } from '@mui/material';
import ExpandablePanel from '@/components/Utils/ExpandablePanel';
import { api } from '@/lib/api';

interface StructuredKGConfig {
  enabled: boolean;
  entity_threshold: number;
  max_corrections: number;
  timeout_ms: number;
  supported_query_types: string[];
}

interface StructuredKGSchema {
  node_labels: string[];
  relationship_types: string[];
  node_properties: Record<string, string[]>;
}

interface QueryResult {
  success: boolean;
  results: any[];
  cypher: string | null;
  query_type: string | null;
  entities: any[];
  corrections: number;
  duration_ms: number;
  error: string | null;
  fallback_recommended: boolean;
}

interface LinkedEntity {
  name: string;
  label: string;
  similarity: number;
}

export default function StructuredKgView() {
  const [query, setQuery] = useState('');
  const [config, setConfig] = useState<StructuredKGConfig | null>(null);
  const [schema, setSchema] = useState<StructuredKGSchema | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [configLoading, setConfigLoading] = useState(true);
  const [schemaLoading, setSchemaLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedPanels, setExpandedPanels] = useState<Set<string>>(new Set(['query']));

  const togglePanel = (panel: string) => {
    setExpandedPanels((prev) => {
      const next = new Set(prev);
      if (next.has(panel)) {
        next.delete(panel);
      } else {
        next.add(panel);
      }
      return next;
    });
  };

  useEffect(() => {
    loadConfig();
    loadSchema();
  }, []);

  const loadConfig = async () => {
    try {
      setConfigLoading(true);
      const data = await api.getStructuredKGConfig();
      setConfig(data);
    } catch (err: any) {
      console.error('Failed to load config:', err);
    } finally {
      setConfigLoading(false);
    }
  };

  const loadSchema = async () => {
    try {
      setSchemaLoading(true);
      const data = await api.getStructuredKGSchema();
      setSchema(data);
    } catch (err: any) {
      console.error('Failed to load schema:', err);
    } finally {
      setSchemaLoading(false);
    }
  };

  const executeQuery = async () => {
    if (!query.trim()) return;

    try {
      setLoading(true);
      setError(null);
      setResult(null);
      const data = await api.executeStructuredQuery(query);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to execute query');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      executeQuery();
    }
  };

  const exampleQueries = [
    'How many documents are there?',
    'Show me all entities related to Python',
    'What are the top 5 most connected entities?',
    'Find documents that mention Docker and Kubernetes',
    'Count chunks by category'
  ];

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--bg-primary)' }}>
      {/* Header */}
      <div style={{ borderBottom: '1px solid var(--border)', padding: 'var(--space-6)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: 'var(--space-2)' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '8px',
            backgroundColor: '#f27a0320',
            border: '1px solid #f27a03',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Network size={24} color="#f27a03" />
          </div>
          <div style={{ flex: 1 }}>
            <h1 className="font-display" style={{ fontSize: 'var(--text-2xl)', fontWeight: 700, color: 'var(--text-primary)' }}>
              Structured Knowledge Graph
            </h1>
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
              Natural language to Cypher translation with entity linking
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 min-h-0 overflow-y-auto" style={{ padding: 'var(--space-6)', display: 'flex', flexDirection: 'column', gap: '12px' }}>

        {/* Configuration Panel */}
        {config && (
          <ExpandablePanel
            title="Configuration"
            expanded={expandedPanels.has('config')}
            onToggle={() => togglePanel('config')}
          >
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '0.875rem' }}>
              <div>
                <span style={{ color: 'var(--text-secondary)' }}>Status:</span>
                <span style={{ marginLeft: '8px', fontWeight: 600, color: config.enabled ? '#10b981' : '#ef4444' }}>
                  {config.enabled ? 'Enabled' : 'Disabled'}
                </span>
              </div>
              <div>
                <span style={{ color: 'var(--text-secondary)' }}>Entity Threshold:</span>
                <span style={{ marginLeft: '8px', fontWeight: 600, color: 'var(--text-primary)' }}>{config.entity_threshold}</span>
              </div>
              <div>
                <span style={{ color: 'var(--text-secondary)' }}>Max Corrections:</span>
                <span style={{ marginLeft: '8px', fontWeight: 600, color: 'var(--text-primary)' }}>{config.max_corrections}</span>
              </div>
              <div>
                <span style={{ color: 'var(--text-secondary)' }}>Timeout:</span>
                <span style={{ marginLeft: '8px', fontWeight: 600, color: 'var(--text-primary)' }}>{config.timeout_ms}ms</span>
              </div>
              <div style={{ gridColumn: '1 / -1' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Supported Query Types:</span>
                <div style={{ marginTop: '6px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                  {config.supported_query_types.map((type) => (
                    <span key={type} style={{ padding: '4px 8px', fontSize: '0.75rem', backgroundColor: '#f27a0320', color: '#f27a03', borderRadius: '4px' }}>
                      {type}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </ExpandablePanel>
        )}

        {/* Schema Panel */}
        {schema && (
          <ExpandablePanel
            title="Graph Schema"
            expanded={expandedPanels.has('schema')}
            onToggle={() => togglePanel('schema')}
          >
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '0.875rem' }}>
              <div>
                <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Node Labels:</span>
                <div style={{ marginTop: '6px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                  {schema.node_labels.map((label) => (
                    <span key={label} style={{ padding: '4px 8px', fontSize: '0.75rem', backgroundColor: '#3b82f620', color: '#3b82f6', borderRadius: '4px' }}>
                      {label}
                    </span>
                  ))}
                </div>
              </div>
              <div>
                <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Relationships:</span>
                <div style={{ marginTop: '6px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                  {schema.relationship_types.map((rel) => (
                    <span key={rel} style={{ padding: '4px 8px', fontSize: '0.75rem', backgroundColor: '#8b5cf620', color: '#8b5cf6', borderRadius: '4px' }}>
                      {rel}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </ExpandablePanel>
        )}

        {/* Query Input Panel */}
        <ExpandablePanel
          title="Natural Language Query"
          expanded={expandedPanels.has('query')}
          onToggle={() => togglePanel('query')}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your knowledge graph... (⌘/Ctrl + Enter to execute)"
              className="input-field"
              style={{ height: '96px', resize: 'none' }}
              disabled={loading || !config?.enabled}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Button
                size="small"
                variant="contained"
                onClick={executeQuery}
                disabled={loading || !query.trim() || !config?.enabled}
                startIcon={loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                style={{
                  textTransform: 'none',
                  backgroundColor: 'var(--accent-primary)',
                  color: 'white',
                }}
              >
                {loading ? 'Executing...' : 'Execute Query'}
              </Button>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                ⌘/Ctrl + Enter to execute
              </span>
            </div>

            {/* Example Queries */}
            <div style={{ paddingTop: '12px', borderTop: '1px solid var(--border)' }}>
              <p style={{ fontSize: '0.75rem', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '8px' }}>Example Queries:</p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {exampleQueries.map((example, idx) => (
                  <button
                    key={idx}
                    onClick={() => setQuery(example)}
                    className="button-secondary"
                    style={{ padding: '4px 12px', fontSize: '0.75rem' }}
                    disabled={loading}
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </ExpandablePanel>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 flex items-start gap-3">
            <XCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-red-900 dark:text-red-100">Error</h3>
              <p className="text-sm text-red-700 dark:text-red-300 mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-4">
            {/* Query Metadata */}
            <div className="bg-white dark:bg-secondary-800 rounded-lg border border-secondary-200 dark:border-secondary-700 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100">Query Execution</h3>
                <div className="flex items-center gap-2">
                  {result.success ? (
                    <span className="flex items-center gap-1 text-sm text-green-600 dark:text-green-400">
                      <CheckCircle className="w-4 h-4" />
                      Success
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-sm text-red-600 dark:text-red-400">
                      <XCircle className="w-4 h-4" />
                      Failed
                    </span>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-secondary-600 dark:text-secondary-400">Query Type:</span>
                  <span className="ml-2 font-medium text-secondary-900 dark:text-secondary-100">
                    {result.query_type || 'N/A'}
                  </span>
                </div>
                <div>
                  <span className="text-secondary-600 dark:text-secondary-400">Duration:</span>
                  <span className="ml-2 font-medium text-secondary-900 dark:text-secondary-100">
                    {result.duration_ms}ms
                  </span>
                </div>
                <div>
                  <span className="text-secondary-600 dark:text-secondary-400">Corrections:</span>
                  <span className="ml-2 font-medium text-secondary-900 dark:text-secondary-100">
                    {result.corrections}
                  </span>
                </div>
                <div>
                  <span className="text-secondary-600 dark:text-secondary-400">Results:</span>
                  <span className="ml-2 font-medium text-secondary-900 dark:text-secondary-100">
                    {result.results.length} rows
                  </span>
                </div>
              </div>

              {result.fallback_recommended && (
                <div className="mt-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded text-sm text-yellow-700 dark:text-yellow-300">
                  Fallback to standard retrieval recommended
                </div>
              )}
            </div>

            {/* Linked Entities */}
            {result.entities && result.entities.length > 0 && (
              <div className="bg-white dark:bg-secondary-800 rounded-lg border border-secondary-200 dark:border-secondary-700 p-6">
                <h3 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100 mb-4 flex items-center gap-2">
                  <Link2 className="w-5 h-5" />
                  Linked Entities ({result.entities.length})
                </h3>
                <div className="space-y-2">
                  {result.entities.map((entity: LinkedEntity, idx: number) => (
                    <div key={idx} className="flex items-center justify-between p-3 bg-secondary-50 dark:bg-secondary-900 rounded-lg">
                      <div>
                        <span className="font-medium text-secondary-900 dark:text-secondary-100">{entity.name}</span>
                        <span className="ml-2 text-xs text-secondary-500 dark:text-secondary-400">({entity.label})</span>
                      </div>
                      <span className="text-sm text-secondary-600 dark:text-secondary-400">
                        {(entity.similarity * 100).toFixed(1)}% match
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Generated Cypher */}
            {result.cypher && (
              <div className="bg-white dark:bg-secondary-800 rounded-lg border border-secondary-200 dark:border-secondary-700 p-6">
                <h3 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100 mb-4 flex items-center gap-2">
                  <Code className="w-5 h-5" />
                  Generated Cypher Query
                </h3>
                <pre className="p-4 bg-secondary-900 dark:bg-black text-green-400 rounded-lg overflow-x-auto text-sm font-mono">
                  {result.cypher}
                </pre>
              </div>
            )}

            {/* Results Table */}
            {result.success && result.results.length > 0 && (
              <div className="bg-white dark:bg-secondary-800 rounded-lg border border-secondary-200 dark:border-secondary-700 p-6">
                <h3 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100 mb-4">
                  Results ({result.results.length})
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-secondary-200 dark:border-secondary-700">
                        {Object.keys(result.results[0]).map((key) => (
                          <th key={key} className="text-left p-3 font-medium text-secondary-700 dark:text-secondary-300">
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.results.map((row, idx) => (
                        <tr key={idx} className="border-b border-secondary-100 dark:border-secondary-800 hover:bg-secondary-50 dark:hover:bg-secondary-900">
                          {Object.values(row).map((value: any, cellIdx) => (
                            <td key={cellIdx} className="p-3 text-secondary-900 dark:text-secondary-100">
                              {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Error Message */}
            {!result.success && result.error && (
              <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-red-900 dark:text-red-100 mb-2">Execution Error</h3>
                <pre className="text-sm text-red-700 dark:text-red-300 whitespace-pre-wrap font-mono">
                  {result.error}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* Empty State */}
        {!result && !error && !loading && (
          <div className="bg-white dark:bg-secondary-800 rounded-lg border border-secondary-200 dark:border-secondary-700 p-12 text-center">
            <Database className="w-16 h-16 text-secondary-300 dark:text-secondary-600 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100 mb-2">
              Ready to Query
            </h3>
            <p className="text-secondary-600 dark:text-secondary-400 max-w-md mx-auto">
              Enter a natural language query above to translate it to Cypher and execute against the knowledge graph.
              The system will automatically link entities and correct errors.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
