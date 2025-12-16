'use client';

import { useState, useEffect } from 'react';

interface CategoriesSidebarContentProps {
    onSectionClick: (sectionId: string) => void;
    activeSection?: string;
}

const sections = [
    { id: 'categories', label: 'Categories' },
    { id: 'prompts', label: 'Prompts' },
];

export default function CategoriesSidebarContent({ onSectionClick, activeSection: propActiveSection }: CategoriesSidebarContentProps) {
    const [activeSection, setActiveSection] = useState<string>(propActiveSection || 'categories');

    // Listen for active section changes from the panel
    useEffect(() => {
        const handleActiveSectionChanged = (event: CustomEvent<string>) => {
            setActiveSection(event.detail);
        };

        window.addEventListener('categories-active-section-changed', handleActiveSectionChanged as EventListener);
        return () => {
            window.removeEventListener('categories-active-section-changed', handleActiveSectionChanged as EventListener);
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
