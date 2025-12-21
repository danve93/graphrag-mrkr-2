'use client';

import { useState, useEffect } from 'react';

interface MetricsSidebarContentProps {
  onSectionClick: (sectionId: string) => void;
  activeSection?: string;
}

const sections = [
  { id: 'llmUsage', label: 'LLM Token Usage' },
  { id: 'trulens', label: 'TruLens' },
  { id: 'ragas', label: 'RAGAS' },
  { id: 'routing', label: 'Routing' },
  { id: 'opentelemetry', label: 'OpenTelemetry' },
];

export default function MetricsSidebarContent({
  onSectionClick,
  activeSection: propActiveSection,
}: MetricsSidebarContentProps) {
  const [activeSection, setActiveSection] = useState<string>(propActiveSection || 'llmUsage');

  useEffect(() => {
    const handleActiveSectionChanged = (event: CustomEvent<string>) => {
      setActiveSection(event.detail);
    };

    window.addEventListener('metrics-active-section-changed', handleActiveSectionChanged as EventListener);
    return () => {
      window.removeEventListener('metrics-active-section-changed', handleActiveSectionChanged as EventListener);
    };
  }, []);

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 pt-6">
        <div className="space-y-1">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => {
                setActiveSection(section.id);
                onSectionClick(section.id);
              }}
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
