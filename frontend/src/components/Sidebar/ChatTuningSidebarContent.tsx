'use client';

import { useState, useEffect, useMemo } from 'react';
import { Search, X } from 'lucide-react';
import { fuzzySearch } from '@/lib/searchUtils';

interface ChatTuningSidebarContentProps {
  onSectionClick: (sectionId: string) => void;
  activeSection?: string;
}

const sections = [
  { id: 'model-selection', label: 'Model Selection' },
  { id: 'retrieval-basics', label: 'Retrieval Basics' },
  { id: 'multi-hop-reasoning', label: 'Multi-Hop Reasoning' },
  { id: 'graph-expansion', label: 'Graph Expansion' },
  { id: 'reranking', label: 'Reranking' },
  { id: 'context-filtering', label: 'Context Filtering' },
  { id: 'performance-&-caching', label: 'Performance & Caching' },
  { id: 'api-keys', label: 'API Keys' },
];

// Parameter data for search (matches ChatTuningPanel categories)
const allParameters = [
  { key: 'llm_model', label: 'LLM Model', category: 'Model Selection', categoryId: 'model-selection' },
  { key: 'temperature', label: 'Temperature', category: 'Model Selection', categoryId: 'model-selection' },
  { key: 'top_k', label: 'Top K Results', category: 'Retrieval Basics', categoryId: 'retrieval-basics' },
  { key: 'similarity_threshold', label: 'Similarity Threshold', category: 'Retrieval Basics', categoryId: 'retrieval-basics' },
  { key: 'max_hops', label: 'Max Hops', category: 'Multi-Hop Reasoning', categoryId: 'multi-hop-reasoning' },
  { key: 'enable_multi_hop', label: 'Enable Multi-Hop', category: 'Multi-Hop Reasoning', categoryId: 'multi-hop-reasoning' },
  { key: 'enable_graph_expansion', label: 'Enable Graph Expansion', category: 'Graph Expansion', categoryId: 'graph-expansion' },
  { key: 'expansion_depth', label: 'Expansion Depth', category: 'Graph Expansion', categoryId: 'graph-expansion' },
  { key: 'enable_reranking', label: 'Enable Reranking', category: 'Reranking', categoryId: 'reranking' },
  { key: 'reranker_model', label: 'Reranker Model', category: 'Reranking', categoryId: 'reranking' },
  { key: 'min_relevance_score', label: 'Min Relevance Score', category: 'Context Filtering', categoryId: 'context-filtering' },
  { key: 'max_context_chunks', label: 'Max Context Chunks', category: 'Context Filtering', categoryId: 'context-filtering' },
];

export default function ChatTuningSidebarContent({ onSectionClick, activeSection: propActiveSection }: ChatTuningSidebarContentProps) {
  const [activeSection, setActiveSection] = useState<string>(propActiveSection || 'model-selection');
  const [searchTerm, setSearchTerm] = useState('');

  // Listen for active section changes from the panel
  useEffect(() => {
    const handleActiveSectionChanged = (event: CustomEvent<string>) => {
      setActiveSection(event.detail);
    };

    window.addEventListener('chat-tuning-active-section-changed', handleActiveSectionChanged as EventListener);
    return () => {
      window.removeEventListener('chat-tuning-active-section-changed', handleActiveSectionChanged as EventListener);
    };
  }, []);

  const searchResults = useMemo(() => {
    if (!searchTerm) return [];
    return fuzzySearch(searchTerm, allParameters, ['label', 'key', 'category']);
  }, [searchTerm]);

  const handleSearchSelect = (match: { item: typeof allParameters[0] }) => {
    const param = match.item;
    // Navigate to section
    onSectionClick(param.categoryId);
    // Dispatch event to highlight parameter in panel
    window.dispatchEvent(new CustomEvent('chat-tuning-highlight-param', { detail: param.key }));
    setSearchTerm('');
  };

  return (
    <div className="flex flex-col h-full">
      {/* Search */}
      <div className="p-4 border-b" style={{ borderColor: 'var(--border)' }}>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search settings..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-9 py-2 text-sm border rounded-lg focus:outline-none focus:ring-2 focus:border-transparent"
            style={{
              borderColor: 'var(--border)',
              background: 'var(--bg-primary)',
              color: 'var(--text-primary)',
            }}
          />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
              aria-label="Clear search"
            >
              <X className="w-4 h-4" />
            </button>
          )}

          {/* Search Results Dropdown */}
          {searchTerm && searchResults.length > 0 && (
            <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg max-h-60 overflow-y-auto">
              {searchResults.map((match) => (
                <button
                  key={match.item.key}
                  onClick={() => handleSearchSelect(match)}
                  className="w-full text-left px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 text-sm"
                  style={{ color: 'var(--text-primary)' }}
                >
                  <div className="font-medium">{match.item.label}</div>
                  <div className="text-xs text-gray-500">{match.item.category}</div>
                </button>
              ))}
            </div>
          )}
          {searchTerm && searchResults.length === 0 && (
            <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg px-4 py-2 text-sm text-gray-500">
              No results found
            </div>
          )}
        </div>
      </div>

      {/* Sections Navigation */}
      <div className="flex-1 overflow-y-auto p-4 pt-6">
        <div className="space-y-1">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => onSectionClick(section.id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${activeSection === section.id
                ? 'bg-orange-50 dark:bg-orange-900/20 font-medium'
                : 'hover:bg-gray-50 dark:hover:bg-neutral-800'
                }`}
              style={{ color: activeSection === section.id ? 'var(--accent-primary)' : 'var(--text-secondary)' }}
            >
              {section.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

