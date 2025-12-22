
'use client'

import { useState, useEffect, useRef } from 'react'
import { SlidersHorizontal, Info } from 'lucide-react'
import ExpandablePanel from '@/components/Utils/ExpandablePanel'
import { AnimatedTooltip as Tooltip } from '@motion-primitives/animated-tooltip'
import Loader from '@/components/Utils/Loader'
import { API_URL } from '@/lib/api'
import ChunkPatternsPanel from './ChunkPatternsPanel'

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
          <p className="text-base text-red-600 mb-4">{error}</p>
          <button onClick={loadConfig} className="button-primary">
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!config) return null

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)]">
      {/* Header */}
      <div className="border-b border-[var(--border)] p-6">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-lg bg-[var(--accent-subtle)] border border-[var(--accent-primary)] flex items-center justify-center">
            <SlidersHorizontal size={24} className="text-[var(--accent-primary)]" />
          </div>
          <h1 className="flex-1 font-display text-2xl font-bold text-[var(--text-primary)]">
            RAG Tuning
          </h1>
          <div className="flex gap-2">
            <button
              onClick={handleReset}
              disabled={isSaving}
              className="button-secondary text-sm py-2 px-4"
            >
              Reset
            </button>
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="button-primary text-sm py-2 px-4 whitespace-nowrap"
            >
              {isSaving ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </div>
        <p className="text-sm text-[var(--text-secondary)]">
          Configure document ingestion and entity extraction behavior. Changes apply to newly ingested documents only.
        </p>
      </div>

      {/* Messages */}
      {(error || successMessage) && (
        <div className="px-6 pt-4">
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-md p-4 mb-4">
              <p className="text-sm text-red-600">{error}</p>
            </div>
          )}
          {successMessage && (
            <div className="bg-green-500/10 border border-green-500/30 rounded-md p-4 mb-4">
              <p className="text-sm text-green-600">{successMessage}</p>
            </div>
          )}
        </div>
      )}

      {/* Content */}
      <div className="flex-1 min-h-0 overflow-y-auto pb-28 p-6">
        {/* Default LLM Model Section */}
        {activeSection === 'default' && (
          <div>
            <h2 className="text-xl font-semibold text-[var(--text-primary)] mb-2">
              Default Configuration
            </h2>
            <p className="text-sm text-[var(--text-secondary)] mb-6">
              Default LLM model used for all ingestion operations unless overridden in specific stages below.
            </p>
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-2">
                <label className="text-sm font-medium text-[var(--text-primary)]">
                  LLM Model
                </label>
                <select
                  value={config.default_llm_model}
                  onChange={(e) => handleDefaultModelChange(e.target.value)}
                  className="input-field w-full"
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

        {/* Chunk Patterns Section - Custom Component */}
        {activeSection === 'chunk_patterns' && (
          <ChunkPatternsPanel />
        )}

        {/* Other Sections */}
        {config.sections
          .filter((section) => section.key === activeSection)
          .map((section) => {
            const effectiveModel = section.llm_override_value || config.default_llm_model

            return (
              <div key={section.key}>
                <h2 className="text-xl font-semibold text-[var(--text-primary)] mb-2">
                  {section.label}
                </h2>
                <p className="text-sm text-[var(--text-secondary)] mb-6">
                  {section.description}
                </p>
                <div className="flex flex-col gap-4">
                  {/* LLM Override */}
                  {section.llm_override_enabled && (
                    <div className="flex flex-col gap-2">
                      <label className="text-sm font-medium text-[var(--text-primary)]">
                        LLM Model Override
                      </label>
                      <select
                        value={section.llm_override_value || ''}
                        onChange={(e) => handleSectionLLMOverride(section.key, e.target.value || null)}
                        className="input-field w-full"
                      >
                        <option value="">(use default) — {config.default_llm_model}</option>
                        {LLM_MODEL_OPTIONS.map((model) => (
                          <option key={model} value={model}>
                            {model}
                          </option>
                        ))}
                      </select>
                      {section.llm_override_value && (
                        <p className="text-xs text-[var(--accent-primary)]">
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
                      className={`flex flex-col gap-2 transition-colors duration-300 ${param.key === highlightedParam
                          ? 'bg-[var(--accent-primary)]/10 p-2 -m-2 rounded-lg'
                          : 'bg-transparent p-0 m-0'
                        }`}
                    >
                      <div className="flex items-center gap-2">
                        <label className="text-sm font-medium text-[var(--text-primary)]">
                          {param.label}
                        </label>
                        <Tooltip content={param.tooltip}>
                          <button className="text-[var(--text-tertiary)] flex items-center" type="button" aria-label="Information">
                            <Info size={16} />
                          </button>
                        </Tooltip>
                      </div>

                      {param.type === 'slider' && (
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-mono font-semibold text-[var(--text-secondary)] min-w-[3rem]">
                            {param.value}
                          </span>
                          <input
                            type="range"
                            min={param.min}
                            max={param.max}
                            step={param.step}
                            value={param.value as number}
                            onChange={(e) => handleParameterChange(section.key, param.key, parseFloat(e.target.value))}
                            className="slider flex-1"
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
                          className="input-field w-full"
                        />
                      )}

                      {param.type === 'select' && param.options && (
                        <select
                          value={String(param.value)}
                          onChange={(e) => handleParameterChange(section.key, param.key, e.target.value)}
                          className="input-field w-full"
                        >
                          {param.options.map((opt) => (
                            <option key={opt} value={opt}>
                              {opt}
                            </option>
                          ))}
                        </select>
                      )}

                      {param.type === 'toggle' && (
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => handleParameterChange(section.key, param.key, !param.value)}
                            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${param.value ? 'toggle-on' : 'toggle-off'
                              }`}
                            aria-label={`Toggle ${param.label}`}
                          >
                            <span
                              className={`inline-block h-4 w-4 transform rounded-full bg-[var(--bg-primary)] transition-transform ${param.value ? 'translate-x-6' : 'translate-x-1'
                                }`}
                            />
                          </button>
                          <span className="text-xs text-[var(--text-secondary)]">
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
