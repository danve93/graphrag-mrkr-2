'use client'

import { useState, useEffect } from 'react'
import { InformationCircleIcon } from '@heroicons/react/24/outline'
import { Button } from '@mui/material'
import { Settings as SettingsIcon } from '@mui/icons-material'
import ExpandablePanel from '@/components/Utils/ExpandablePanel'
import Tooltip from '@/components/Utils/Tooltip'
import Loader from '@/components/Utils/Loader'
import { API_URL } from '@/lib/api'

interface ChatParameter {
  key: string
  label: string
  value: number | boolean | string
  options?: string[]
  min?: number
  max?: number
  step?: number
  type: 'slider' | 'toggle' | 'select'
  category: string
  tooltip: string
}

interface ChatTuningConfig {
  parameters: ChatParameter[]
}

const CATEGORY_ORDER = [
  'Model Selection',
  'Retrieval Basics',
  'Multi-Hop Reasoning',
  'Graph Expansion',
  'Reranking',
  'Context Filtering'
]

export default function ChatTuningPanel() {
  const [config, setConfig] = useState<ChatTuningConfig | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(CATEGORY_ORDER))

  useEffect(() => {
    loadConfig()
  }, [])

  // Listen for section selection from sidebar
  useEffect(() => {
    const handleSectionSelect = (event: CustomEvent<string>) => {
      const sectionId = event.detail;
      const element = document.getElementById(sectionId);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    };

    window.addEventListener('chat-tuning-section-select', handleSectionSelect as EventListener);
    return () => {
      window.removeEventListener('chat-tuning-section-select', handleSectionSelect as EventListener);
    };
  }, [])

  const loadConfig = async () => {
    try {
      setIsLoading(true)
      setError(null)
      const response = await fetch(`${API_URL}/api/chat-tuning/config`)
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

      const response = await fetch(`${API_URL}/api/chat-tuning/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })

      if (!response.ok) {
        throw new Error(`Failed to save config: ${response.statusText}`)
      }

      const { showToast } = require('@/components/Toast/ToastContainer')
      showToast('success', 'Configuration saved successfully', undefined, 3000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save configuration')
    } finally {
      setIsSaving(false)
    }
  }

  const handleValueChange = (key: string, value: number | boolean | string) => {
    if (!config) return

    setConfig({
      ...config,
      parameters: config.parameters.map((param) =>
        param.key === key ? { ...param, value } : param
      ),
    })
  }

  const handleReset = () => {
    loadConfig()
    setSuccessMessage(null)
    setError(null)
  }

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev)
      if (next.has(category)) {
        next.delete(category)
      } else {
        next.add(category)
      }
      return next
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
          <button
            onClick={loadConfig}
            className="button-primary"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!config) return null

  // Group parameters by category and sort by predefined order
  const groupedParams = config.parameters.reduce((acc, param) => {
    if (!acc[param.category]) {
      acc[param.category] = []
    }
    acc[param.category].push(param)
    return acc
  }, {} as Record<string, ChatParameter[]>)

  // Sort categories according to CATEGORY_ORDER
  const sortedCategories = CATEGORY_ORDER.filter(cat => groupedParams[cat])

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--bg-primary)' }}>
        {/* Header */}
          <div style={{ borderBottom: '1px solid var(--border)', padding: 'var(--space-6)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: 'var(--space-2)' }}>
          <div style={{ 
            width: '40px', 
            height: '40px', 
            borderRadius: '8px', 
            backgroundColor: '#f27a0320',
            border: '1px solid #f27a03',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <SettingsIcon style={{ fontSize: '24px', color: '#f27a03' }} />
          </div>
          <h1 className="font-display" style={{ fontSize: 'var(--text-2xl)', fontWeight: 700, color: 'var(--text-primary)', flex: 1 }}>
            Chat Tuning
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
          Adjust retrieval and generation parameters for chat responses. Changes apply instantly to new queries.
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
      <div className="flex-1 min-h-0 overflow-y-auto" style={{ padding: 'var(--space-6)', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {sortedCategories.map((category) => {
          const params = groupedParams[category]
          const isExpanded = expandedCategories.has(category)
          const sectionId = category.toLowerCase().replace(/\s+/g, '-');
          
          return (
            <div key={category} id={sectionId} style={{ scrollMarginTop: '24px' }}>
            <ExpandablePanel
              title={category}
              expanded={isExpanded}
              onToggle={() => toggleCategory(category)}
            >
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {params.map((param) => (
                  <div key={param.key} style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div className="flex items-center" style={{ gap: '8px' }}>
                      <label style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-primary)' }}>
                        {param.label}
                      </label>
                      <Tooltip content={param.tooltip}>
                        <button style={{ color: 'var(--text-tertiary)', display: 'flex', alignItems: 'center' }} type="button" aria-label="Information">
                          <InformationCircleIcon style={{ width: '16px', height: '16px' }} />
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
                          onChange={(e) => handleValueChange(param.key, parseFloat(e.target.value))}
                          className="slider"
                          style={{ flex: 1 }}
                          title={String(param.value)}
                        />
                      </div>
                    )}

                    {param.type === 'select' && param.options && (
                      <select
                        value={String(param.value)}
                        onChange={(e) => handleValueChange(param.key, e.target.value)}
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
                          onClick={() => handleValueChange(param.key, !param.value)}
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
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                          {param.value ? 'Enabled' : 'Disabled'}
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </ExpandablePanel>
            </div>
          )
        })}
      </div>

    </div>
  )
}
 
