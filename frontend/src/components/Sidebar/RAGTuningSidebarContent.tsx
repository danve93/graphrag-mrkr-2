'use client';

import { useState, useEffect } from 'react';

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
      { key: 'pdf_processing', label: 'PDF Processing' },
      { key: 'entity_extraction', label: 'Entity Extraction' },
      { key: 'description_enhancement', label: 'Description Enhancement' },
      { key: 'graph_persistence', label: 'Graph Persistence' },
      { key: 'ocr_processing', label: 'OCR & Image Processing' },
      { key: 'performance', label: 'Performance & Limits' },
    ],
  },
  {
    category: 'Advanced Features',
    sections: [
      { key: 'client_side_vector_search', label: 'Client-Side Vector Search' },
      { key: 'layered_memory', label: 'Layered Memory System' },
    ],
  },
];

export default function RAGTuningSidebarContent({
  onSectionClick,
  activeSection: propActiveSection,
  sections = defaultSections
}: RAGTuningSidebarContentProps) {
  const [activeSection, setActiveSection] = useState<string>(propActiveSection || 'default');

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

  return (
    <div className="flex flex-col h-full">
      {/* Sections Navigation */}
      <div className="flex-1 overflow-y-auto p-4 pt-6">
        <div className="space-y-4">
          {/* Default Section */}
          {sections.map((section) => (
            <button
              key={section.key}
              onClick={() => onSectionClick(section.key)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                activeSection === section.key
                  ? 'bg-orange-50 dark:bg-orange-900/20 text-[#f27a03] font-medium'
                  : 'hover:bg-gray-50 dark:hover:bg-neutral-800'
              }`}
              style={activeSection !== section.key ? { color: 'var(--text-secondary)' } : {}}
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
                    className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                      activeSection === section.key
                        ? 'bg-orange-50 dark:bg-orange-900/20 text-[#f27a03] font-medium'
                        : 'hover:bg-gray-50 dark:hover:bg-neutral-800'
                    }`}
                    style={activeSection !== section.key ? { color: 'var(--text-secondary)' } : {}}
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
