'use client';

import { Sliders } from 'lucide-react';

interface RAGTuningSidebarContentProps {
  onSectionClick: (sectionId: string) => void;
  activeSection?: string;
  sections?: Array<{ key: string; label: string }>;
}

const defaultSections = [
  { key: 'default', label: 'Default LLM Model' },
  { key: 'retrieval', label: 'Retrieval' },
  { key: 'expansion', label: 'Graph Expansion' },
  { key: 'reranking', label: 'Reranking' },
  { key: 'query-routing', label: 'Query Routing' },
  { key: 'structured-kg', label: 'Structured KG' },
];

export default function RAGTuningSidebarContent({ 
  onSectionClick, 
  activeSection,
  sections = defaultSections 
}: RAGTuningSidebarContentProps) {
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b" style={{ borderColor: 'var(--border)' }}>
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-lg border flex items-center justify-center" style={{ borderColor: 'var(--border)', backgroundColor: 'var(--bg-secondary)' }}>
            <Sliders className="w-5 h-5" style={{ color: 'var(--text-primary)' }} />
          </div>
          <h2 className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>
            RAG Tuning
          </h2>
        </div>
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          Configure retrieval, expansion, and reranking
        </p>
      </div>

      {/* Sections Navigation */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-1">
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
        </div>
      </div>
    </div>
  );
}
