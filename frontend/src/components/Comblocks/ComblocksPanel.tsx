"use client"

import React, { useEffect, useState } from "react"

type Relation = { id: string; type: string; targetEntityId: string }
type Entity = { id: string; name: string; type?: string; relations: Relation[] }
type Community = { id: string | number; entities: Entity[] }

const STORAGE_KEY = "comblocks_communities_v1"

function uid(prefix = "id_") {
  return prefix + Math.random().toString(36).slice(2, 9)
}

export default function ComblocksPanel() {
  const [communities, setCommunities] = useState<Community[]>([])
  const [selectedCommunity, setSelectedCommunity] = useState<string | number | null>(null)
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null)
  const [status, setStatus] = useState("ready")

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (raw) setCommunities(JSON.parse(raw))
    } catch (e) {
      // ignore
    }
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(communities))
    } catch (e) {
      // ignore
    }
  }, [communities])

  const addCommunity = () => {
    const id = uid("c_")
    setCommunities((s) => [...s, { id, entities: [] }])
    setSelectedCommunity(id)
  }

  const removeCommunity = (id: string | number) => {
    setCommunities((s) => s.filter((c) => c.id !== id))
    if (selectedCommunity === id) setSelectedCommunity(null)
  }

  const addEntity = (communityId: string | number) => {
    const id = uid("e_")
    setCommunities((s) => s.map((c) => (c.id === communityId ? { ...c, entities: [...c.entities, { id, name: "New Entity", relations: [] }] } : c)))
    setSelectedEntity(id)
  }

  const updateEntity = (entityId: string, patch: Partial<Entity>) => {
    setCommunities((s) => s.map((c) => ({ ...c, entities: c.entities.map((e) => (e.id === entityId ? { ...e, ...patch } : e)) })))
  }

  const removeEntity = (entityId: string) => {
    setCommunities((s) => s.map((c) => ({ ...c, entities: c.entities.filter((e) => e.id !== entityId) })))
    if (selectedEntity === entityId) setSelectedEntity(null)
  }

  const addRelation = (fromEntityId: string, targetEntityId: string) => {
    const rel: Relation = { id: uid("r_"), type: "RELATED_TO", targetEntityId }
    setCommunities((s) => s.map((c) => ({ ...c, entities: c.entities.map((e) => (e.id === fromEntityId ? { ...e, relations: [...e.relations, rel] } : e)) })))
  }

  const removeRelation = (fromEntityId: string, relId: string) => {
    setCommunities((s) => s.map((c) => ({ ...c, entities: c.entities.map((e) => (e.id === fromEntityId ? { ...e, relations: e.relations.filter((r) => r.id !== relId) } : e)) })))
  }

  const exportJson = () => {
    const payload = JSON.stringify(communities, null, 2)
    navigator.clipboard?.writeText(payload)
    setStatus("exported to clipboard")
    setTimeout(() => setStatus("ready"), 1500)
  }

  const clearAll = () => {
    setCommunities([])
    setSelectedCommunity(null)
    setSelectedEntity(null)
    localStorage.removeItem(STORAGE_KEY)
  }

  const allEntities = communities.flatMap((c) => c.entities)
  const selectedEntityObj = allEntities.find((e) => e.id === selectedEntity) || null

  return (
    <div className="h-full flex flex-col bg-white dark:bg-secondary-900">
      {/* Header */}
      <div className="flex-shrink-0 border-b border-secondary-200 dark:border-secondary-700 px-6 py-4">
        <h2 className="text-xl font-bold text-secondary-900 dark:text-secondary-50 mb-2">Comblocks — Tree Builder</h2>
        <p className="text-sm text-secondary-600 dark:text-secondary-400">
          Visual editor for Community → Entity → Relation. Load communities from the backend,
          add entities and relations, and export the structure as JSON.
        </p>
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0 overflow-y-auto p-6">
        <div className="flex gap-4 h-full">
        <div className="w-1/4 border rounded p-3 bg-white dark:bg-secondary-800">
          <div className="flex justify-between items-center mb-2">
            <strong>Communities</strong>
            <div className="flex gap-2">
              <button className="px-3 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm font-medium" onClick={addCommunity}>Add</button>
              <button className="px-3 py-2 bg-secondary-200 dark:bg-secondary-700 text-secondary-700 dark:text-secondary-300 rounded-lg hover:bg-secondary-300 dark:hover:bg-secondary-600 text-sm font-medium" onClick={() => { setCommunities((s) => s); setStatus('ready') }}>Refresh</button>
            </div>
          </div>
          <div className="space-y-2 max-h-96 overflow-auto">
            {communities.length === 0 && <div className="text-sm text-secondary-500">No communities</div>}
            {communities.map((c) => (
              <div key={String(c.id)} className={`p-2 rounded cursor-pointer ${selectedCommunity === c.id ? 'bg-sky-100' : 'bg-transparent'}`} onClick={() => setSelectedCommunity(c.id)}>
                <div className="flex justify-between items-center">
                  <div className="text-sm">{String(c.id)}</div>
                  <button className="text-xs text-red-600" onClick={(e) => { e.stopPropagation(); removeCommunity(c.id) }}>Delete</button>
                </div>
                <div className="text-xs text-secondary-400">Entities: {c.entities.length}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="w-2/4 border rounded p-3 bg-white dark:bg-secondary-800">
          <div className="flex justify-between items-center mb-3">
            <strong>Tree</strong>
            <div className="flex gap-2">
              <button className="px-3 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm font-medium" onClick={() => exportJson()}>Export JSON</button>
              <button className="px-3 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 text-sm font-medium" onClick={clearAll}>Clear</button>
            </div>
          </div>

          {!selectedCommunity && <div className="text-sm text-secondary-500">Select a community to view its entities</div>}

          {selectedCommunity && (
            <div>
              <div className="mb-2">
                <button className="px-3 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm font-medium" onClick={() => addEntity(selectedCommunity)}>Add Entity</button>
              </div>
              <div className="space-y-3">
                {communities.find((c) => c.id === selectedCommunity)?.entities.map((e) => (
                  <div key={e.id} className="border rounded p-2">
                    <div className="flex justify-between items-center">
                      <div>
                        <div className="font-medium">{e.name}</div>
                        <div className="text-xs text-secondary-400">id: {e.id} • type: {e.type || '—'}</div>
                      </div>
                      <div className="flex gap-2">
                        <button className="px-3 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm font-medium" onClick={() => setSelectedEntity(e.id)}>Edit</button>
                        <button className="px-3 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 text-sm font-medium" onClick={() => removeEntity(e.id)}>Delete</button>
                      </div>
                    </div>

                    <div className="mt-2">
                      <div className="text-xs text-secondary-500 mb-1">Relations</div>
                      <div className="space-y-1">
                        {e.relations.length === 0 && <div className="text-xs text-secondary-400">No relations</div>}
                        {e.relations.map((r) => (
                          <div key={r.id} className="flex justify-between items-center text-xs">
                            <div>{r.type} → {r.targetEntityId}</div>
                            <button className="text-red-600 text-xs" onClick={() => removeRelation(e.id, r.id)}>Remove</button>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="w-1/4 border rounded p-3 bg-white dark:bg-secondary-800">
          <strong>Inspector</strong>
          <div className="mt-3">
            {selectedEntityObj ? (
              <EntityInspector
                entity={selectedEntityObj}
                allEntities={allEntities}
                onUpdate={(patch) => updateEntity(selectedEntityObj.id, patch)}
                onAddRelation={(targetId) => addRelation(selectedEntityObj.id, targetId)}
              />
            ) : (
              <div className="text-sm text-secondary-500">Select an entity to edit</div>
            )}
          </div>
        </div>
      </div>

        <div className="text-sm text-secondary-500 mt-3">Status: {status}</div>
      </div>
    </div>
  )
}

function EntityInspector({ entity, allEntities, onUpdate, onAddRelation }: { entity: Entity; allEntities: Entity[]; onUpdate: (p: Partial<Entity>) => void; onAddRelation: (targetId: string) => void }) {
  const [name, setName] = useState(entity.name)
  const [type, setType] = useState(entity.type || '')
  const [target, setTarget] = useState('')

  useEffect(() => {
    setName(entity.name)
    setType(entity.type || '')
  }, [entity.id])

  return (
    <div>
      <label className="block text-xs text-secondary-500">Name</label>
      <input className="w-full p-1 border rounded mb-2" value={name} onChange={(e) => setName(e.target.value)} />
      <label className="block text-xs text-secondary-500">Type</label>
      <input className="w-full p-1 border rounded mb-2" value={type} onChange={(e) => setType(e.target.value)} />
      <div className="flex gap-2 mb-2">
        <button className="px-3 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm font-medium" onClick={() => onUpdate({ name, type })}>Save</button>
      </div>

      <div className="mt-2">
        <label className="block text-xs text-secondary-500">Add Relation</label>
        <select className="w-full p-1 border rounded mb-2" value={target} onChange={(e) => setTarget(e.target.value)}>
          <option value="">— select target entity —</option>
          {allEntities.filter((e) => e.id !== entity.id).map((e) => (
            <option key={e.id} value={e.id}>{e.name} ({e.id})</option>
          ))}
        </select>
        <div className="flex gap-2">
          <button className="px-3 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm font-medium" onClick={() => { if (target) { onAddRelation(target); setTarget('') } }}>Add</button>
        </div>
      </div>
    </div>
  )
}