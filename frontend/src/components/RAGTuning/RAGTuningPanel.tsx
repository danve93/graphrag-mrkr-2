
'use client'

import { useState, useEffect, useRef } from 'react'
import { SlidersHorizontal, Info } from 'lucide-react'
import { Button } from '@mui/material'
import ExpandablePanel from '@/components/Utils/ExpandablePanel'
import { AnimatedTooltip as Tooltip } from '@motion-primitives/animated-tooltip'
import Loader from '@/components/Utils/Loader'
import { API_URL } from '@/lib/api'

interface RAGParameter {
  key: string
  label: string
  value: number | boolean | string
  options?: string[]
  min?: number
  max?: number
  step?: number
  type: 'slider' | 'toggle' | 'select' | 'number'
  tooltip: string
}

interface RAGSection {
  key: string
  label: string
  description: string
  llm_override_enabled: boolean
  llm_override_value: string | null
  parameters: RAGParameter[]
}

interface RAGTuningConfig {
  default_llm_model: string
  sections: RAGSection[]
}

const LLM_MODEL_OPTIONS = [
  'gpt-4o-mini',
  'gpt-4o',
  'gpt-4-turbo',
  'claude-3-5-sonnet-20241022',
  'claude-3-5-haiku-20241022',
  'gemini-3-flash-preview',
  'gemini-3-pro-preview'
]

export default function RAGTuningPanel() {
  const [config, setConfig] = useState<RAGTuningConfig | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [activeSection, setActiveSection] = useState<string>('default')
  const [hasChanges, setHasChanges] = useState(false)
  const [highlightedParam, setHighlightedParam] = useState<string | null>(null)

  // Ref for scrolling to highlighted element
  const highlightedRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (highlightedParam && highlightedRef.current) {
      highlightedRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' })
      // Clear highlight after animation
      const timer = setTimeout(() => setHighlightedParam(null), 2000)
      return () => clearTimeout(timer)
    }
  }, [highlightedParam, activeSection])

  useEffect(() => {
    loadConfig()
  }, [])

  // Listen for section selection from sidebar
  useEffect(() => {
    const handleSectionSelect = (event: CustomEvent<string>) => {
      setActiveSection(event.detail);
    };

    window.addEventListener('rag-tuning-section-select', handleSectionSelect as EventListener);
    return () => {
      window.removeEventListener('rag-tuning-section-select', handleSectionSelect as EventListener);
    };
  }, [])

  // Broadcast active section changes to sidebar
  useEffect(() => {
    window.dispatchEvent(new CustomEvent('rag-tuning-active-section-changed', { detail: activeSection }));
  }, [activeSection])

  // Listen for highlight events from sidebar search
  useEffect(() => {
    const handleHighlightSection = (event: CustomEvent<string>) => {
      // The sidebar dispatches the section key, we just acknowledge it
      // The actual section navigation is handled by onSectionClick in sidebar
    };

    window.addEventListener('rag-tuning-highlight-section', handleHighlightSection as EventListener);
    return () => {
      window.removeEventListener('rag-tuning-highlight-section', handleHighlightSection as EventListener);
    };
  }, []);

  const loadConfig = async () => {
    try {
      setIsLoading(true)
      setError(null)
      const response = await fetch(`${API_URL}/api/rag-tuning/config`)
      if (!response.ok) {
        throw new Error(`Failed to load config: ${response.statusText}`)
      }
      const data = await response.json()
      setConfig(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load configuration')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSave = async () => {
    if (!config) return

    try {
      setIsSaving(true)
      setError(null)
      setSuccessMessage(null)

      const response = await fetch(`${API_URL}/api/rag-tuning/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })

      if (!response.ok) {
        throw new Error(`Failed to save config: ${response.statusText}`)
      }

      const { showToast } = require('@/components/Toast/ToastContainer')
      showToast('success', 'Configuration saved successfully', undefined, 3000)
      setHasChanges(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save configuration')
    } finally {
      setIsSaving(false)
    }
  }

  const handleDefaultModelChange = (value: string) => {
    if (!config) return
    if (!hasChanges) {
      setHasChanges(true)
      showWarningToast()
    }
    setConfig({
      ...config,
      default_llm_model: value,
    })
  }

  const handleSectionLLMOverride = (sectionKey: string, value: string | null) => {
    if (!config) return
    if (!hasChanges) {
      setHasChanges(true)
      showWarningToast()
    }
    setConfig({
      ...config,
      sections: config.sections.map((section) =>
        section.key === sectionKey
          ? { ...section, llm_override_value: value }
          : section
      ),
    })
  }

  const handleParameterChange = (sectionKey: string, paramKey: string, value: number | boolean | string) => {
    if (!config) return
    if (!hasChanges) {
      setHasChanges(true)
      showWarningToast()
    }
    setConfig({
      ...config,
      sections: config.sections.map((section) =>
        section.key === sectionKey
          ? {
            ...section,
            parameters: section.parameters.map((param) =>
              param.key === paramKey ? { ...param, value } : param
            ),
          }
          : section
      ),
    })
  }

  const handleReset = () => {
    loadConfig()
    setSuccessMessage(null)
    setError(null)
    setHasChanges(false)
  }

  const showWarningToast = () => {
    const { showToast } = require('@/components/Toast/ToastContainer')
    showToast(
      'warning',
      'Existing documents require re-ingestion',
      'Changes apply to newly ingested documents only. Use Database panel to re-index if needed.',
      8000,
      'top-center'
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <Loader size={48} label="Loading configuration..." />
        </div>
      </div>
    )
  }

  if (error && !config) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center max-w-md">
          <p style={{ fontSize: 'var(--text-base)', color: '#dc2626', marginBottom: 'var(--space-4)' }}>{error}</p>
          <button onClick={loadConfig} className="button-primary">
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!config) return null

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--bg-primary)' }}>
      {/* Header */}
      <div style={{ borderBottom: '1px solid var(--border)', padding: 'var(--space-6)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: 'var(--space-2)' }}>
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
            <SlidersHorizontal size={24} color="var(--accent-primary)" />
          </div>
          <h1 className="font-display" style={{ fontSize: 'var(--text-2xl)', fontWeight: 700, color: 'var(--text-primary)', flex: 1 }}>
            RAG Tuning
          </h1>
          <div style={{ display: 'flex', gap: '8px' }}>
            <Button
              size="small"
              onClick={handleReset}
              disabled={isSaving}
              style={{ textTransform: 'none', color: 'var(--text-secondary)' }}
            >
              Reset
            </Button>
            <Button
              size="small"
              variant="contained"
              onClick={handleSave}
              disabled={isSaving}
              style={{
                textTransform: 'none',
                backgroundColor: 'var(--accent-primary)',
                color: 'white',
              }}
            >
              {isSaving ? 'Saving...' : 'Save Changes'}
            </Button>
          </div>
        </div>
        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
          Configure document ingestion and entity extraction behavior. Changes apply to newly ingested documents only.
        </p>
      </div>

      {/* Messages */}
      {(error || successMessage) && (
        <div style={{ padding: 'var(--space-4) var(--space-6) 0' }}>
          {error && (
            <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: 'var(--radius-md)', padding: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
              <p style={{ fontSize: 'var(--text-sm)', color: '#dc2626' }}>{error}</p>
            </div>
          )}
          {successMessage && (
            <div style={{ background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.3)', borderRadius: 'var(--radius-md)', padding: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
              <p style={{ fontSize: 'var(--text-sm)', color: '#16a34a' }}>{successMessage}</p>
            </div>
          )}
        </div>
      )}

      {/* Content */}
      <div className="flex-1 min-h-0 overflow-y-auto pb-28 p-[var(--space-6)]">
        {/* Default LLM Model Section */}
        {activeSection === 'default' && (
          <div>
            <h2 style={{ fontSize: 'var(--text-xl)', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 'var(--space-2)' }}>
              Default Configuration
            </h2>
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginBottom: 'var(--space-6)' }}>
              Default LLM model used for all ingestion operations unless overridden in specific stages below.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <label style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-primary)' }}>
                  LLM Model
                </label>
                <select
                  value={config.default_llm_model}
                  onChange={(e) => handleDefaultModelChange(e.target.value)}
                  className="input-field"
                  style={{ width: '100%' }}
                >
                  {LLM_MODEL_OPTIONS.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Other Sections */}
        {config.sections
          .filter((section) => section.key === activeSection)
          .map((section) => {
            const effectiveModel = section.llm_override_value || config.default_llm_model

            return (
              <div key={section.key}>
                <h2 style={{ fontSize: 'var(--text-xl)', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 'var(--space-2)' }}>
                  {section.label}
                </h2>
                <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginBottom: 'var(--space-6)' }}>
                  {section.description}
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {/* LLM Override */}
                  {section.llm_override_enabled && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      <label style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-primary)' }}>
                        LLM Model Override
                      </label>
                      <select
                        value={section.llm_override_value || ''}
                        onChange={(e) => handleSectionLLMOverride(section.key, e.target.value || null)}
                        className="input-field"
                        style={{ width: '100%' }}
                      >
                        <option value="">(use default) — {config.default_llm_model}</option>
                        {LLM_MODEL_OPTIONS.map((model) => (
                          <option key={model} value={model}>
                            {model}
                          </option>
                        ))}
                      </select>
                      {section.llm_override_value && (
                        <p style={{ fontSize: '0.75rem', color: 'var(--accent-primary)' }}>
                          ✓ Override active: using {effectiveModel}
                        </p>
                      )}
                    </div>
                  )}

                  {/* Parameters */}
                  {section.parameters.map((param) => (
                    <div
                      key={param.key}
                      ref={param.key === highlightedParam ? highlightedRef : null}
                      style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '8px',
                        transition: 'background-color 0.3s ease',
                        backgroundColor: param.key === highlightedParam ? 'rgba(var(--accent-primary-rgb), 0.1)' : 'transparent',
                        borderRadius: '8px',
                        padding: param.key === highlightedParam ? '8px' : '0',
                        margin: param.key === highlightedParam ? '-8px' : '0'
                      }}
                    >
                      <div className="flex items-center" style={{ gap: '8px' }}>
                        <label style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-primary)' }}>
                          {param.label}
                        </label>
                        <Tooltip content={param.tooltip}>
                          <button style={{ color: 'var(--text-tertiary)', display: 'flex', alignItems: 'center' }} type="button" aria-label="Information">
                            <Info size={16} />
                          </button>
                        </Tooltip>
                      </div>

                      {param.type === 'slider' && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontWeight: 600, minWidth: '3rem' }}>
                            {param.value}
                          </span>
                          <input
                            type="range"
                            min={param.min}
                            max={param.max}
                            step={param.step}
                            value={param.value as number}
                            onChange={(e) => handleParameterChange(section.key, param.key, parseFloat(e.target.value))}
                            className="slider"
                            style={{ flex: 1 }}
                            title={String(param.value)}
                          />
                        </div>
                      )}

                      {param.type === 'number' && (
                        <input
                          type="number"
                          min={param.min}
                          max={param.max}
                          step={param.step}
                          value={param.value as number}
                          onChange={(e) => handleParameterChange(section.key, param.key, parseInt(e.target.value))}
                          className="input-field"
                          style={{ width: '100%' }}
                        />
                      )}

                      {param.type === 'select' && param.options && (
                        <select
                          value={String(param.value)}
                          onChange={(e) => handleParameterChange(section.key, param.key, e.target.value)}
                          className="input-field"
                          style={{ width: '100%' }}
                        >
                          {param.options.map((opt) => (
                            <option key={opt} value={opt}>
                              {opt}
                            </option>
                          ))}
                        </select>
                      )}

                      {param.type === 'toggle' && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <button
                            onClick={() => handleParameterChange(section.key, param.key, !param.value)}
                            className={`relative inline-flex h-6 w-11 items-center transition-colors ${param.value ? 'toggle-on' : 'toggle-off'
                              }`}
                            style={{ borderRadius: 'var(--radius-full)' }}
                            aria-label={`Toggle ${param.label}`}
                          >
                            <span
                              className={`inline-block h-4 w-4 transform transition-transform ${param.value ? 'translate-x-6' : 'translate-x-1'
                                }`}
                              style={{
                                borderRadius: 'var(--radius-full)',
                                background: 'var(--bg-primary)'
                              }}
                            />
                          </button>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                            {param.value ? 'Enabled' : 'Disabled'}
                          </span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
      </div>
    </div>
  )
}
