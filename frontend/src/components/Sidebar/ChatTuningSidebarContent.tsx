'use client';

import { Settings } from 'lucide-react';

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
];

export default function ChatTuningSidebarContent({ onSectionClick, activeSection }: ChatTuningSidebarContentProps) {
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b" style={{ borderColor: 'var(--border)' }}>
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-lg border flex items-center justify-center" style={{ borderColor: 'var(--border)', backgroundColor: 'var(--bg-secondary)' }}>
            <Settings className="w-5 h-5" style={{ color: 'var(--text-primary)' }} />
          </div>
          <h2 className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>
            Chat Tuning
          </h2>
        </div>
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          Adjust LLM and embedding model settings
        </p>
      </div>

      {/* Sections Navigation */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-1">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => onSectionClick(section.id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                activeSection === section.id
                  ? 'bg-orange-50 dark:bg-orange-900/20 text-[#f27a03] font-medium'
                  : 'hover:bg-gray-50 dark:hover:bg-neutral-800'
              }`}
              style={activeSection !== section.id ? { color: 'var(--text-secondary)' } : {}}
            >
              {section.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
