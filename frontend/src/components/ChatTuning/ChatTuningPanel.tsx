'use client'

import { useState, useEffect } from 'react'
import { InformationCircleIcon } from '@heroicons/react/24/outline'

interface ChatParameter {
  key: string
  label: string
  value: number | boolean | string
  options?: string[]
  min?: number
  max?: number
  step?: number
  type: 'slider' | 'toggle'
  | 'select'
  category: string
  tooltip: string
}

interface ChatTuningConfig {
  parameters: ChatParameter[]
}

export default function ChatTuningPanel() {
  const [config, setConfig] = useState<ChatTuningConfig | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)

  useEffect(() => {
    loadConfig()
  }, [])

  const loadConfig = async () => {
    try {
      setIsLoading(true)
      setError(null)
      const response = await fetch('/api/chat-tuning/config')
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

      const response = await fetch('/api/chat-tuning/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })

      if (!response.ok) {
        throw new Error(`Failed to save config: ${response.statusText}`)
      }

      setSuccessMessage('Configuration saved successfully!')
      setTimeout(() => setSuccessMessage(null), 3000)
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

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-secondary-600 dark:text-secondary-400">Loading configuration...</p>
        </div>
      </div>
    )
  }

  if (error && !config) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center max-w-md">
          <div className="text-red-500 mb-4">⚠️</div>
          <p className="text-red-600 dark:text-red-400 mb-4">{error}</p>
          <button
            onClick={loadConfig}
            className="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!config) return null

  // Group parameters by category
  const groupedParams = config.parameters.reduce((acc, param) => {
    if (!acc[param.category]) {
      acc[param.category] = []
    }
    acc[param.category].push(param)
    return acc
  }, {} as Record<string, ChatParameter[]>)

  return (
    <div className="h-full flex flex-col bg-white dark:bg-secondary-900">
      {/* Header */}
      <div className="border-b border-secondary-200 dark:border-secondary-700 p-6">
        <h1 className="text-2xl font-bold text-secondary-900 dark:text-secondary-50 mb-2">
          Chat Tuning
        </h1>
        <p className="text-sm text-secondary-600 dark:text-secondary-400">
          Adjust retrieval and reasoning parameters for chat responses
        </p>
      </div>

      {/* Messages */}
      {(error || successMessage) && (
        <div className="px-6 pt-4">
          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-4">
              <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
            </div>
          )}
          {successMessage && (
            <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4 mb-4">
              <p className="text-sm text-green-600 dark:text-green-400">{successMessage}</p>
            </div>
          )}
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-8">
        {Object.entries(groupedParams).map(([category, params]) => (
          <div key={category} className="space-y-4">
            <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-50 border-b border-secondary-200 dark:border-secondary-700 pb-2">
              {category}
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {params.map((param) => (
                <div key={param.key} className="space-y-2 p-3 rounded-lg bg-white dark:bg-secondary-900 border border-secondary-100 dark:border-secondary-800">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <label className="text-sm font-medium text-secondary-700 dark:text-secondary-300">
                        {param.label}
                      </label>
                      <Tooltip content={param.tooltip} />
                    </div>
                    {param.type === 'slider' && (
                      <span className="text-sm font-mono text-secondary-600 dark:text-secondary-400">
                        {param.value}
                      </span>
                    )}
                  </div>

                  {param.type === 'slider' && (
                    <input
                      type="range"
                      min={param.min}
                      max={param.max}
                      step={param.step}
                      value={param.value as number}
                      onChange={(e) => handleValueChange(param.key, parseFloat(e.target.value))}
                      className="w-full h-2 bg-secondary-200 dark:bg-secondary-700 rounded-lg appearance-none cursor-pointer slider"
                    />
                  )}

                  {param.type === 'select' && param.options && (
                    <select
                      value={String(param.value)}
                      onChange={(e) => handleValueChange(param.key, e.target.value)}
                      className="w-full p-2 bg-white dark:bg-secondary-800 border border-secondary-200 dark:border-secondary-700 rounded"
                    >
                      {param.options.map((opt) => (
                        <option key={opt} value={opt}>
                          {opt}
                        </option>
                      ))}
                    </select>
                  )}

                  {param.type === 'toggle' && (
                    <button
                      onClick={() => handleValueChange(param.key, !param.value)}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                        param.value
                          ? 'bg-primary-600'
                          : 'bg-secondary-300 dark:bg-secondary-600'
                      }`}
                    >
                      <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                          param.value ? 'translate-x-6' : 'translate-x-1'
                        }`}
                      />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Footer Actions */}
      <div className="border-t border-secondary-200 dark:border-secondary-700 p-6 flex gap-4">
        <button
          onClick={handleSave}
          disabled={isSaving}
          className="flex-1 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
        >
          {isSaving ? 'Saving...' : 'Save Configuration'}
        </button>
        <button
          onClick={handleReset}
          disabled={isSaving}
          className="px-4 py-2 bg-secondary-200 dark:bg-secondary-700 text-secondary-700 dark:text-secondary-300 rounded-lg hover:bg-secondary-300 dark:hover:bg-secondary-600 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
        >
          Reset
        </button>
      </div>
    </div>
  )
}

function Tooltip({ content }: { content: string }) {
  const [isVisible, setIsVisible] = useState(false)

  return (
    <div className="relative inline-block">
      <button
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
        className="text-secondary-400 hover:text-secondary-600 dark:hover:text-secondary-300"
        type="button"
      >
        <InformationCircleIcon className="w-4 h-4" />
      </button>
      {isVisible && (
        <div className="absolute left-6 top-0 z-50 w-64 p-3 text-xs bg-secondary-800 dark:bg-secondary-700 text-white rounded-lg shadow-lg">
          {content}
          <div className="absolute left-0 top-2 -translate-x-1 w-2 h-2 bg-secondary-800 dark:bg-secondary-700 rotate-45"></div>
        </div>
      )}
    </div>
  )
}
