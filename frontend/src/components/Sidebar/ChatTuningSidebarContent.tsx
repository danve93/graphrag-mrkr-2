'use client';

import { useState, useEffect } from 'react';

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

export default function ChatTuningSidebarContent({ onSectionClick, activeSection: propActiveSection }: ChatTuningSidebarContentProps) {
  const [activeSection, setActiveSection] = useState<string>(propActiveSection || 'model-selection');

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

  return (
    <div className="flex flex-col h-full">
      {/* Sections Navigation */}
      <div className="flex-1 overflow-y-auto p-4 pt-6">
        <div className="space-y-1">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => onSectionClick(section.id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${activeSection === section.id
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
