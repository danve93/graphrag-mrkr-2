'use client';

import { useState, useEffect, useMemo } from 'react';
import { Search, X } from 'lucide-react';
import { fuzzySearch } from '@/lib/searchUtils';

interface RAGTuningSidebarContentProps {
  onSectionClick: (sectionId: string) => void;
  activeSection?: string;
  sections?: Array<{ key: string; label: string }>;
}

// Organized sections by category
const defaultSections = [
  { key: 'default', label: 'Default LLM Model' },
];

const sectionCategories = [
  {
    category: 'Retrieval Settings',
    sections: [
      { key: 'content_filtering', label: 'Content Filtering' },
      { key: 'temporal_retrieval', label: 'Temporal Retrieval' },
      { key: 'multi_stage_retrieval', label: 'Multi-Stage Retrieval' },
      { key: 'fuzzy_matching', label: 'Fuzzy Matching' },
      { key: 'retrieval_fusion', label: 'Retrieval Fusion' },
    ],
  },
  {
    category: 'Query Processing',
    sections: [
      { key: 'query_analysis', label: 'Query Analysis & Expansion' },
    ],
  },
  {
    category: 'Reranking',
    sections: [
      { key: 'reranking', label: 'Reranking' },
    ],
  },
  {
    category: 'Quality & Monitoring',
    sections: [
      { key: 'quality_monitoring', label: 'Quality Monitoring' },
    ],
  },
  {
    category: 'Ingestion Settings',
    sections: [
      { key: 'chunking', label: 'Chunking' },
      { key: 'chunk_patterns', label: 'Chunk Patterns' },
      { key: 'pdf_processing', label: 'Document Conversion' },
      { key: 'ocr_processing', label: 'OCR & Image Processing' },
      { key: 'entity_extraction', label: 'Entity Extraction' },
      { key: 'description_enhancement', label: 'Description Enhancement' },
      { key: 'graph_persistence', label: 'Graph Persistence' },
      { key: 'performance', label: 'Performance & Limits' },
    ],
  },
  {
    category: 'Advanced Features',
    sections: [
      { key: 'client_side_vector_search', label: 'Client-Side Vector Search' },
      { key: 'layered_memory', label: 'Layered Memory System' },
      { key: 'sentence_window_retrieval', label: 'Sentence-Window Retrieval' },
    ],
  },
];

// Flatten all section data for search
const allSectionsForSearch = [
  ...defaultSections.map(s => ({ ...s, categoryName: 'Default' })),
  ...sectionCategories.flatMap(cat =>
    cat.sections.map(s => ({ ...s, categoryName: cat.category }))
  ),
];

export default function RAGTuningSidebarContent({
  onSectionClick,
  activeSection: propActiveSection,
  sections = defaultSections
}: RAGTuningSidebarContentProps) {
  const [activeSection, setActiveSection] = useState<string>(propActiveSection || 'default');
  const [searchTerm, setSearchTerm] = useState('');

  // Listen for active section changes from the panel
  useEffect(() => {
    const handleActiveSectionChanged = (event: CustomEvent<string>) => {
      setActiveSection(event.detail);
    };

    window.addEventListener('rag-tuning-active-section-changed', handleActiveSectionChanged as EventListener);
    return () => {
      window.removeEventListener('rag-tuning-active-section-changed', handleActiveSectionChanged as EventListener);
    };
  }, []);

  const searchResults = useMemo(() => {
    if (!searchTerm) return [];
    return fuzzySearch(searchTerm, allSectionsForSearch, ['label', 'key', 'categoryName']);
  }, [searchTerm]);

  const handleSearchSelect = (match: { item: typeof allSectionsForSearch[0] }) => {
    const section = match.item;
    onSectionClick(section.key);
    // Dispatch event to highlight section in panel (optional, for consistency)
    window.dispatchEvent(new CustomEvent('rag-tuning-highlight-section', { detail: section.key }));
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
                  <div className="text-xs text-gray-500">{match.item.categoryName}</div>
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
        <div className="space-y-4">
          {/* Default Section */}
          {sections.map((section) => (
            <button
              key={section.key}
              onClick={() => onSectionClick(section.key)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${activeSection === section.key
                ? 'bg-orange-50 dark:bg-orange-900/20 font-medium'
                : 'hover:bg-gray-50 dark:hover:bg-neutral-800'
                }`}
              style={{ color: activeSection === section.key ? 'var(--accent-primary)' : 'var(--text-secondary)' }}
            >
              {section.label}
            </button>
          ))}

          {/* Categorized Sections */}
          {sectionCategories.map((category, categoryIndex) => (
            <div key={categoryIndex}>
              <h3
                className="text-xs font-semibold uppercase tracking-wide mb-2 px-3"
                style={{ color: 'var(--text-tertiary)' }}
              >
                {category.category}
              </h3>
              <div className="space-y-1">
                {category.sections.map((section) => (
                  <button
                    key={section.key}
                    onClick={() => onSectionClick(section.key)}
                    className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${activeSection === section.key
                      ? 'bg-orange-50 dark:bg-orange-900/20 font-medium'
                      : 'hover:bg-gray-50 dark:hover:bg-neutral-800'
                      }`}
                    style={{ color: activeSection === section.key ? 'var(--accent-primary)' : 'var(--text-secondary)' }}
                  >
                    {section.label}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

