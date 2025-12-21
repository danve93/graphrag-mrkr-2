'use client';

import { useState, useEffect } from 'react';
import { Activity } from 'lucide-react';
import RagasSubPanel from './RagasSubPanel';
import TruLensSubPanel from './TruLensSubPanel';
import OpenTelemetrySubPanel from './OpenTelemetrySubPanel';
import LLMUsageSubPanel from './LLMUsageSubPanel';

import RoutingView from '../Routing/RoutingView';

export default function MetricsPanel() {
  const [activeSection, setActiveSection] = useState<string>('llmUsage');

  // Listen for section selection from sidebar
  useEffect(() => {
    const handleSectionSelect = (event: CustomEvent<string>) => {
      setActiveSection(event.detail);
    };

    window.addEventListener('metrics-section-select', handleSectionSelect as EventListener);
    return () => {
      window.removeEventListener('metrics-section-select', handleSectionSelect as EventListener);
    };
  }, []);

  // Broadcast active section changes to sidebar
  useEffect(() => {
    window.dispatchEvent(new CustomEvent('metrics-active-section-changed', { detail: activeSection }));
  }, [activeSection]);

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--bg-primary)' }}>
      {/* Main Header */}
      <div style={{ borderBottom: '1px solid var(--border)', padding: 'var(--space-6)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '8px',
            backgroundColor: 'var(--accent-subtle)',
            border: '1px solid var(--accent-primary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Activity style={{ color: 'var(--accent-primary)' }} size={24} />
          </div>
          <div style={{ flex: 1 }}>
            <h1 className="font-display" style={{ fontSize: 'var(--text-2xl)', fontWeight: 700, color: 'var(--text-primary)' }}>
              Metrics & Evaluation
            </h1>
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
              Monitor quality, run benchmarks, and analyze system performance
            </p>
          </div>
        </div>
      </div>

      {/* Sub-Panel Router */}
      <div className="flex-1 overflow-hidden">
        {activeSection === 'ragas' && <RagasSubPanel />}
        {activeSection === 'trulens' && <TruLensSubPanel />}
        {activeSection === 'routing' && <RoutingView />}
        {activeSection === 'opentelemetry' && <OpenTelemetrySubPanel />}
        {activeSection === 'llmUsage' && <LLMUsageSubPanel />}
      </div>
    </div>
  );
}
