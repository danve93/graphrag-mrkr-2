'use client'

import { useState, useEffect } from 'react'
import {
  PlusIcon,
  TrashIcon,
  PencilIcon,
  MagnifyingGlassIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline'

interface ClassificationConfig {
  entity_types: string[]
  entity_type_overrides: Record<string, string>
  relationship_suggestions: string[]
  low_value_patterns: string[]
  leiden_parameters: {
    resolution: number
    min_edge_weight: number
    relationship_types: string[]
  }
}

type TabId = 'entity_types' | 'overrides' | 'relationships' | 'leiden'

export default function ClassificationPanel() {
  const [config, setConfig] = useState<ClassificationConfig | null>(null)
  const [activeTab, setActiveTab] = useState<TabId>('entity_types')
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [showReindexModal, setShowReindexModal] = useState(false)

  useEffect(() => {
    loadConfig()
  }, [])

  const loadConfig = async () => {
    try {
      setIsLoading(true)
      setError(null)
      const response = await fetch('/api/classification/config')
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

      const response = await fetch('/api/classification/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })

      if (!response.ok) {
        throw new Error(`Failed to save config: ${response.statusText}`)
      }

      setSuccessMessage('Configuration saved successfully! Changes will apply to new ingestion.')
      setTimeout(() => setSuccessMessage(null), 5000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save configuration')
    } finally {
      setIsSaving(false)
    }
  }

  const handleReindex = async () => {
    try {
      setIsSaving(true)
      setError(null)
      setSuccessMessage(null)
      setShowReindexModal(false)

      const response = await fetch('/api/classification/reindex', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm: true }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `Reindex failed: ${response.statusText}`)
      }

      const result = await response.json()
      setSuccessMessage(result.message || 'Reindex completed successfully!')
      setTimeout(() => setSuccessMessage(null), 5000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reindex')
    } finally {
      setIsSaving(false)
    }
  }

  const tabs = [
    { id: 'entity_types' as const, label: 'Entity Types' },
    { id: 'overrides' as const, label: 'Type Overrides' },
    { id: 'relationships' as const, label: 'Relationships' },
    { id: 'leiden' as const, label: 'Leiden Config' },
  ]

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
          <div className="text-red-500 mb-4 text-4xl">⚠️</div>
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

  return (
    <div className="h-full flex flex-col bg-white dark:bg-secondary-900">
      {/* Header */}
      <div className="border-b border-secondary-200 dark:border-secondary-700 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-secondary-900 dark:text-secondary-50 mb-2">
              Classification Configuration
            </h1>
            <p className="text-sm text-secondary-600 dark:text-secondary-400">
              Manage entity types, overrides, relationships, and clustering parameters
            </p>
          </div>
          <button
            onClick={() => setShowReindexModal(true)}
            disabled={isSaving}
            className="flex items-center gap-2 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
          >
            <ArrowPathIcon className="w-5 h-5" />
            Update & Reindex
          </button>
        </div>
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

      {/* Tabs */}
      <div className="flex border-b border-secondary-200 dark:border-secondary-700 px-6">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => {
              setActiveTab(tab.id)
              setSearchQuery('')
            }}
            className={`px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-primary-600 dark:text-primary-400 border-b-2 border-primary-600 dark:border-primary-400'
                : 'text-secondary-600 dark:text-secondary-400 hover:text-secondary-900 dark:hover:text-secondary-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {activeTab === 'entity_types' && (
          <EntityTypesEditor
            entityTypes={config.entity_types}
            searchQuery={searchQuery}
            onSearchChange={setSearchQuery}
            onChange={(types) => setConfig({ ...config, entity_types: types })}
          />
        )}
        {activeTab === 'overrides' && (
          <OverridesEditor
            overrides={config.entity_type_overrides}
            searchQuery={searchQuery}
            onSearchChange={setSearchQuery}
            onChange={(overrides) => setConfig({ ...config, entity_type_overrides: overrides })}
          />
        )}
        {activeTab === 'relationships' && (
          <RelationshipsEditor
            relationships={config.relationship_suggestions}
            searchQuery={searchQuery}
            onSearchChange={setSearchQuery}
            onChange={(rels) => setConfig({ ...config, relationship_suggestions: rels })}
          />
        )}
        {activeTab === 'leiden' && (
          <LeidenConfigEditor
            config={config.leiden_parameters}
            onChange={(leiden) => setConfig({ ...config, leiden_parameters: leiden })}
          />
        )}
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
          onClick={() => {
            loadConfig()
            setSuccessMessage(null)
            setError(null)
          }}
          disabled={isSaving}
          className="px-4 py-2 bg-secondary-200 dark:bg-secondary-700 text-secondary-700 dark:text-secondary-300 rounded-lg hover:bg-secondary-300 dark:hover:bg-secondary-600 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
        >
          Reset
        </button>
      </div>

      {/* Reindex Confirmation Modal */}
      {showReindexModal && (
        <ReindexModal
          onConfirm={handleReindex}
          onCancel={() => setShowReindexModal(false)}
        />
      )}
    </div>
  )
}

// Entity Types Editor
function EntityTypesEditor({
  entityTypes,
  searchQuery,
  onSearchChange,
  onChange,
}: {
  entityTypes: string[]
  searchQuery: string
  onSearchChange: (query: string) => void
  onChange: (types: string[]) => void
}) {
  const [newType, setNewType] = useState('')

  const filteredTypes = entityTypes.filter((type) =>
    type.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleAdd = () => {
    if (newType.trim() && !entityTypes.includes(newType.trim())) {
      onChange([...entityTypes, newType.trim()])
      setNewType('')
    }
  }

  const handleRemove = (type: string) => {
    onChange(entityTypes.filter((t) => t !== type))
  }

  return (
    <div className="space-y-4">
      {/* Search */}
      <div className="relative">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-secondary-400" />
        <input
          type="text"
          placeholder="Search entity types..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="w-full pl-10 pr-4 py-2 border border-secondary-300 dark:border-secondary-600 rounded-lg bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100"
        />
      </div>

      {/* Add new type */}
      <div className="flex gap-2">
        <input
          type="text"
          placeholder="New entity type..."
          value={newType}
          onChange={(e) => setNewType(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleAdd()}
          className="flex-1 px-4 py-2 border border-secondary-300 dark:border-secondary-600 rounded-lg bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100"
        />
        <button
          onClick={handleAdd}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 flex items-center gap-2"
        >
          <PlusIcon className="w-5 h-5" />
          Add
        </button>
      </div>

      {/* List */}
      <div className="space-y-2">
        <p className="text-sm text-secondary-600 dark:text-secondary-400">
          {filteredTypes.length} of {entityTypes.length} types
        </p>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredTypes.map((type) => (
            <div
              key={type}
              className="flex items-center justify-between p-3 bg-secondary-50 dark:bg-secondary-800 rounded-lg"
            >
              <span className="text-sm text-secondary-900 dark:text-secondary-100">{type}</span>
              <button
                onClick={() => handleRemove(type)}
                className="text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
              >
                <TrashIcon className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// Overrides Editor
function OverridesEditor({
  overrides,
  searchQuery,
  onSearchChange,
  onChange,
}: {
  overrides: Record<string, string>
  searchQuery: string
  onSearchChange: (query: string) => void
  onChange: (overrides: Record<string, string>) => void
}) {
  const [newKey, setNewKey] = useState('')
  const [newValue, setNewValue] = useState('')

  const entries = Object.entries(overrides)
  const filteredEntries = entries.filter(
    ([key, value]) =>
      key.toLowerCase().includes(searchQuery.toLowerCase()) ||
      value.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleAdd = () => {
    if (newKey.trim() && newValue.trim() && !overrides[newKey.trim()]) {
      onChange({ ...overrides, [newKey.trim()]: newValue.trim() })
      setNewKey('')
      setNewValue('')
    }
  }

  const handleRemove = (key: string) => {
    const newOverrides = { ...overrides }
    delete newOverrides[key]
    onChange(newOverrides)
  }

  return (
    <div className="space-y-4">
      {/* Search */}
      <div className="relative">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-secondary-400" />
        <input
          type="text"
          placeholder="Search overrides..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="w-full pl-10 pr-4 py-2 border border-secondary-300 dark:border-secondary-600 rounded-lg bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100"
        />
      </div>

      {/* Add new override */}
      <div className="flex gap-2">
        <input
          type="text"
          placeholder="From (e.g., email_service)"
          value={newKey}
          onChange={(e) => setNewKey(e.target.value)}
          className="flex-1 px-4 py-2 border border-secondary-300 dark:border-secondary-600 rounded-lg bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100"
        />
        <span className="flex items-center text-secondary-600 dark:text-secondary-400">→</span>
        <input
          type="text"
          placeholder="To (e.g., Service)"
          value={newValue}
          onChange={(e) => setNewValue(e.target.value)}
          className="flex-1 px-4 py-2 border border-secondary-300 dark:border-secondary-600 rounded-lg bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100"
        />
        <button
          onClick={handleAdd}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 flex items-center gap-2"
        >
          <PlusIcon className="w-5 h-5" />
          Add
        </button>
      </div>

      {/* List */}
      <div className="space-y-2">
        <p className="text-sm text-secondary-600 dark:text-secondary-400">
          {filteredEntries.length} of {entries.length} overrides
        </p>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredEntries.map(([key, value]) => (
            <div
              key={key}
              className="flex items-center justify-between p-3 bg-secondary-50 dark:bg-secondary-800 rounded-lg"
            >
              <div className="flex items-center gap-3 flex-1">
                <span className="text-sm font-mono text-secondary-700 dark:text-secondary-300">{key}</span>
                <span className="text-secondary-500">→</span>
                <span className="text-sm font-medium text-secondary-900 dark:text-secondary-100">{value}</span>
              </div>
              <button
                onClick={() => handleRemove(key)}
                className="text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
              >
                <TrashIcon className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// Relationships Editor
function RelationshipsEditor({
  relationships,
  searchQuery,
  onSearchChange,
  onChange,
}: {
  relationships: string[]
  searchQuery: string
  onSearchChange: (query: string) => void
  onChange: (rels: string[]) => void
}) {
  const [newRel, setNewRel] = useState('')

  const filteredRels = relationships.filter((rel) =>
    rel.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleAdd = () => {
    if (newRel.trim() && !relationships.includes(newRel.trim())) {
      onChange([...relationships, newRel.trim()])
      setNewRel('')
    }
  }

  const handleRemove = (rel: string) => {
    onChange(relationships.filter((r) => r !== rel))
  }

  return (
    <div className="space-y-4">
      {/* Search */}
      <div className="relative">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-secondary-400" />
        <input
          type="text"
          placeholder="Search relationships..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="w-full pl-10 pr-4 py-2 border border-secondary-300 dark:border-secondary-600 rounded-lg bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100"
        />
      </div>

      {/* Add new relationship */}
      <div className="flex gap-2">
        <input
          type="text"
          placeholder="New relationship type..."
          value={newRel}
          onChange={(e) => setNewRel(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleAdd()}
          className="flex-1 px-4 py-2 border border-secondary-300 dark:border-secondary-600 rounded-lg bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100"
        />
        <button
          onClick={handleAdd}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 flex items-center gap-2"
        >
          <PlusIcon className="w-5 h-5" />
          Add
        </button>
      </div>

      {/* List */}
      <div className="space-y-2">
        <p className="text-sm text-secondary-600 dark:text-secondary-400">
          {filteredRels.length} of {relationships.length} relationship types
        </p>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredRels.map((rel) => (
            <div
              key={rel}
              className="flex items-center justify-between p-3 bg-secondary-50 dark:bg-secondary-800 rounded-lg"
            >
              <span className="text-sm text-secondary-900 dark:text-secondary-100">{rel}</span>
              <button
                onClick={() => handleRemove(rel)}
                className="text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
              >
                <TrashIcon className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// Leiden Config Editor
function LeidenConfigEditor({
  config,
  onChange,
}: {
  config: { resolution: number; min_edge_weight: number; relationship_types: string[] }
  onChange: (config: { resolution: number; min_edge_weight: number; relationship_types: string[] }) => void
}) {
  return (
    <div className="space-y-6 max-w-2xl">
      <div className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-secondary-700 dark:text-secondary-300">
              Resolution
            </label>
            <span className="text-sm font-mono text-secondary-600 dark:text-secondary-400">
              {config.resolution.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="0.1"
            max="2.0"
            step="0.1"
            value={config.resolution}
            onChange={(e) =>
              onChange({ ...config, resolution: parseFloat(e.target.value) })
            }
            className="w-full h-2 bg-secondary-200 dark:bg-secondary-700 rounded-lg appearance-none cursor-pointer"
          />
          <p className="text-xs text-secondary-500 dark:text-secondary-400">
            Higher values create more clusters. Default: 1.0
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-secondary-700 dark:text-secondary-300">
              Minimum Edge Weight
            </label>
            <span className="text-sm font-mono text-secondary-600 dark:text-secondary-400">
              {config.min_edge_weight.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="0.0"
            max="1.0"
            step="0.05"
            value={config.min_edge_weight}
            onChange={(e) =>
              onChange({ ...config, min_edge_weight: parseFloat(e.target.value) })
            }
            className="w-full h-2 bg-secondary-200 dark:bg-secondary-700 rounded-lg appearance-none cursor-pointer"
          />
          <p className="text-xs text-secondary-500 dark:text-secondary-400">
            Minimum similarity threshold for edges. Default: 0.1
          </p>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-secondary-700 dark:text-secondary-300">
            Relationship Types
          </label>
          <div className="space-y-2">
            {config.relationship_types.map((type, idx) => (
              <div key={idx} className="flex items-center gap-2">
                <input
                  type="text"
                  value={type}
                  onChange={(e) => {
                    const newTypes = [...config.relationship_types]
                    newTypes[idx] = e.target.value
                    onChange({ ...config, relationship_types: newTypes })
                  }}
                  className="flex-1 px-3 py-2 border border-secondary-300 dark:border-secondary-600 rounded-lg bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100"
                />
                <button
                  onClick={() => {
                    const newTypes = config.relationship_types.filter((_, i) => i !== idx)
                    onChange({ ...config, relationship_types: newTypes })
                  }}
                  className="text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
                >
                  <TrashIcon className="w-4 h-4" />
                </button>
              </div>
            ))}
            <button
              onClick={() =>
                onChange({
                  ...config,
                  relationship_types: [...config.relationship_types, ''],
                })
              }
              className="px-3 py-2 text-sm bg-secondary-200 dark:bg-secondary-700 text-secondary-700 dark:text-secondary-300 rounded-lg hover:bg-secondary-300 dark:hover:bg-secondary-600 flex items-center gap-2"
            >
              <PlusIcon className="w-4 h-4" />
              Add Relationship Type
            </button>
          </div>
          <p className="text-xs text-secondary-500 dark:text-secondary-400">
            Relationship types to include in clustering graph
          </p>
        </div>
      </div>
    </div>
  )
}

// Reindex Confirmation Modal
function ReindexModal({
  onConfirm,
  onCancel,
}: {
  onConfirm: () => void
  onCancel: () => void
}) {
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel()
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [onCancel])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black bg-opacity-50">
      <div className="bg-white dark:bg-secondary-800 rounded-lg shadow-xl max-w-md w-full p-6 space-y-4">
        <div className="flex items-center gap-3">
          <div className="flex-shrink-0 w-12 h-12 bg-orange-100 dark:bg-orange-900/30 rounded-full flex items-center justify-center">
            <ArrowPathIcon className="w-6 h-6 text-orange-600 dark:text-orange-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-secondary-900 dark:text-secondary-50">
              Reindex All Documents?
            </h3>
          </div>
        </div>

        <div className="space-y-2 text-sm text-secondary-600 dark:text-secondary-400">
          <p>This will:</p>
          <ul className="list-disc list-inside space-y-1 ml-2">
            <li>Clear all existing entities and relationships</li>
            <li>Re-extract entities from all documents</li>
            <li>Re-run Leiden clustering with new parameters</li>
            <li>This operation may take several minutes</li>
          </ul>
          <p className="font-medium text-orange-600 dark:text-orange-400 mt-3">
            ⚠️ This is a destructive operation and cannot be undone.
          </p>
        </div>

        <div className="flex gap-3 pt-2">
          <button
            onClick={onCancel}
            className="flex-1 px-4 py-2 bg-secondary-200 dark:bg-secondary-700 text-secondary-700 dark:text-secondary-300 rounded-lg hover:bg-secondary-300 dark:hover:bg-secondary-600 font-medium"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="flex-1 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 font-medium"
          >
            Confirm Reindex
          </button>
        </div>
      </div>
    </div>
  )
}
