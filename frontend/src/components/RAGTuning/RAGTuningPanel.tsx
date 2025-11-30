
'use client'

import { useState, useEffect } from 'react'
import { InformationCircleIcon, ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline'
import Tooltip from '@/components/Utils/Tooltip'
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
  'claude-3-5-haiku-20241022'
]

export default function RAGTuningPanel() {
  const [config, setConfig] = useState<RAGTuningConfig | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set())
  const [hasChanges, setHasChanges] = useState(false)

  useEffect(() => {
    loadConfig()
  }, [])

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
      // Expand first section by default
      if (data.sections && data.sections.length > 0) {
        setExpandedSections(new Set([data.sections[0].key]))
      }
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

  const toggleSection = (sectionKey: string) => {
    setExpandedSections((prev) => {
      const newSet = new Set(prev)
      if (newSet.has(sectionKey)) {
        newSet.delete(sectionKey)
      } else {
        newSet.add(sectionKey)
      }
      return newSet
    })
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
          <div style={{ borderBottom: '1px solid var(--border)', padding: '0 var(--space-6) var(--space-6)' }}>
        <h1 className="font-display" style={{ fontSize: 'var(--text-2xl)', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 'var(--space-2)' }}>
          RAG Tuning
        </h1>
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
      <div className="flex-1 min-h-0 overflow-y-auto" style={{ padding: '0 var(--space-6) var(--space-6)' }}>
        {/* Default LLM Model */}
        <div style={{ marginBottom: 'var(--space-12)', paddingBottom: 'var(--space-8)', borderBottom: '1px solid var(--border)' }}>
          <h2 className="font-display" style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 'var(--space-6)' }}>
            Default Configuration
          </h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
            <label style={{ fontSize: 'var(--text-base)', fontWeight: 500, color: 'var(--text-primary)' }}>
              LLM Model
            </label>
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginBottom: 'var(--space-2)' }}>
              Used for all ingestion operations unless overridden in specific stages below
            </p>
            <select
              value={config.default_llm_model}
              onChange={(e) => handleDefaultModelChange(e.target.value)}
              className="input-field"
              style={{ maxWidth: '400px' }}
            >
              {LLM_MODEL_OPTIONS.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Sections */}
        {config.sections.map((section, idx) => {
          const effectiveModel = section.llm_override_value || config.default_llm_model

          return (
            <div key={section.key} style={{ marginBottom: 'var(--space-12)', paddingBottom: idx < config.sections.length - 1 ? 'var(--space-8)' : 0, borderBottom: idx < config.sections.length - 1 ? '1px solid var(--border)' : 'none' }}>
              {/* Section Header */}
              <div style={{ marginBottom: 'var(--space-6)' }}>
                <h2 className="font-display" style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                  {section.label}
                </h2>
                <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginTop: 'var(--space-2)' }}>
                  {section.description}
                </p>
              </div>

              {/* Section Content */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
                {/* LLM Override */}
                {section.llm_override_enabled && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
                    <label style={{ fontSize: 'var(--text-base)', fontWeight: 500, color: 'var(--text-primary)' }}>
                      LLM Model Override
                    </label>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                      <select
                        value={section.llm_override_value || ''}
                        onChange={(e) => handleSectionLLMOverride(section.key, e.target.value || null)}
                        className="input-field"
                        style={{ maxWidth: '400px' }}
                      >
                        <option value="">(use default) — {config.default_llm_model}</option>
                        {LLM_MODEL_OPTIONS.map((model) => (
                          <option key={model} value={model}>
                            {model}
                          </option>
                        ))}
                      </select>
                    </div>
                    {section.llm_override_value && (
                      <p style={{ fontSize: 'var(--text-sm)', color: 'var(--accent-primary)' }}>
                        ✓ Override active: using {effectiveModel}
                      </p>
                    )}
                  </div>
                )}

                {/* Parameters */}
                {section.parameters.map((param) => (
                  <div key={param.key} style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
                    <div className="flex items-center" style={{ gap: 'var(--space-2)' }}>
                      <label style={{ fontSize: 'var(--text-base)', fontWeight: 500, color: 'var(--text-primary)' }}>
                        {param.label}
                      </label>
                      <Tooltip content={param.tooltip}>
                        <button style={{ color: 'var(--text-tertiary)', display: 'flex', alignItems: 'center' }} type="button" aria-label="Information">
                          <InformationCircleIcon style={{ width: '18px', height: '18px' }} />
                        </button>
                      </Tooltip>
                    </div>

                    {param.type === 'slider' && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ fontSize: 'var(--text-sm)', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontWeight: 600, minWidth: '3rem' }}>
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
                          style={{ flex: 1, maxWidth: '500px' }}
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
                        style={{ maxWidth: '200px' }}
                      />
                    )}

                    {param.type === 'select' && param.options && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                        <select
                          value={String(param.value)}
                          onChange={(e) => handleParameterChange(section.key, param.key, e.target.value)}
                          className="input-field"
                          style={{ maxWidth: '400px' }}
                        >
                          {param.options.map((opt) => (
                            <option key={opt} value={opt}>
                              {opt}
                            </option>
                          ))}
                        </select>
                      </div>
                    )}

                    {param.type === 'toggle' && (
                      <div>
                        <button
                          onClick={() => handleParameterChange(section.key, param.key, !param.value)}
                          className={`relative inline-flex h-6 w-11 items-center transition-colors ${
                            param.value ? 'toggle-on' : 'toggle-off'
                          }`}
                          style={{ borderRadius: 'var(--radius-full)' }}
                          aria-label={`Toggle ${param.label}`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform transition-transform ${
                              param.value ? 'translate-x-6' : 'translate-x-1'
                            }`}
                            style={{ 
                              borderRadius: 'var(--radius-full)', 
                              background: 'var(--bg-primary)' 
                            }}
                          />
                        </button>
                        <span style={{ marginLeft: 'var(--space-3)', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
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

      {/* Footer Actions */}
      <div className="flex" style={{ borderTop: '1px solid var(--border)', padding: 'var(--space-6)', gap: 'var(--space-4)' }}>
        <button
          onClick={handleSave}
          disabled={isSaving}
          className="flex-1 button-primary"
        >
          {isSaving ? <Loader size={14} label="Saving..." /> : 'Save Configuration'}
        </button>
        <button
          onClick={handleReset}
          disabled={isSaving}
          className="button-secondary"
        >
          Reset
        </button>
      </div>
    </div>
  )
}
