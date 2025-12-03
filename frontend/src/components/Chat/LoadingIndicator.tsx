'use client'

import { useEffect, useState } from 'react'

interface Stage {
  id: string
  label: string
  emoji: string
  color: string
}

const STAGES: Record<string, Stage> = {
  query_analysis: {
    id: 'query_analysis',
    label: 'Analyzing Query',
    emoji: '',
    color: '',
  },
  routing: {
    id: 'routing',
    label: 'Query Routing',
    emoji: '',
    color: '',
  },
  retrieval: {
    id: 'retrieval',
    label: 'Retrieving Documents',
    emoji: '',
    color: '',
  },
  consolidation: {
    id: 'consolidation',
    label: 'Smart Consolidation',
    emoji: '',
    color: '',
  },
  graph_reasoning: {
    id: 'graph_reasoning',
    label: 'Graph Reasoning',
    emoji: '',
    color: '',
  },
  generation: {
    id: 'generation',
    label: 'Generating Response',
    emoji: '',
    color: '',
  },
  quality_calculation: {
    id: 'quality_calculation',
    label: 'Quality Check',
    emoji: '',
    color: '',
  },
  suggestions: {
    id: 'suggestions',
    label: 'Preparing Suggestions',
    emoji: '',
    color: '',
  },
}

interface StageUpdate {
  name: string
  duration_ms?: number
  timestamp?: number
  metadata?: {
    chunks_retrieved?: number
    context_items?: number
    response_length?: number
    [key: string]: any
  }
}

interface LoadingIndicatorProps {
  currentStage?: string
  completedStages?: string[]
  stageUpdates?: StageUpdate[]
  isLoading?: boolean
  enableQualityScoring?: boolean
}

export default function LoadingIndicator({ 
  currentStage = 'query_analysis',
  completedStages = [],
  stageUpdates = [],
  isLoading = true,
  enableQualityScoring = true
}: LoadingIndicatorProps) {
  const [displayedStage, setDisplayedStage] = useState<string>(currentStage)
  const [stageHistory, setStageHistory] = useState<string[]>(completedStages)

  useEffect(() => {
    // Update when currentStage changes (during loading)
    if (isLoading && currentStage) {
      setDisplayedStage(currentStage)
      setStageHistory((prev) => {
        if (!prev.includes(currentStage)) {
          return [...prev, currentStage]
        }
        return prev
      })
    }
  }, [currentStage, isLoading])

  // Update stage history when completed stages change (after loading)
  useEffect(() => {
    if (!isLoading && completedStages.length > 0) {
      setStageHistory(completedStages)
      // Set displayed stage to the last completed stage
      setDisplayedStage(completedStages[completedStages.length - 1])
    }
  }, [completedStages, isLoading])

  const isStageCompleted = (stageId: string) => {
    if (stageId === 'quality_calculation' && !enableQualityScoring) {
      return false
    }
    return stageHistory.includes(stageId)
  }

  const stage = STAGES[displayedStage] || STAGES.query_analysis
  const completedStagesCount = isLoading ? stageHistory.slice(0, -1).length : stageHistory.length
  
  const completedStagesPercent = (stageHistory.length / Object.keys(STAGES).length) * 100
  
  // Get timing info for current or completed stages
  const getStageInfo = (stageId: string) => {
    return stageUpdates.find(s => s.name === stageId)
  }
  
  const currentStageInfo = getStageInfo(displayedStage)
  const totalDuration = stageUpdates.reduce((sum, s) => sum + (s.duration_ms || 0), 0)
  
  // If not loading, show minimal clean display with timing
  if (!isLoading) {
    return (
      <div className="w-full space-y-2">
        {/* Progress bar at 100% - solid color */}
        <div className="relative h-1 rounded-full overflow-hidden" style={{ background: 'var(--gray-200)' }}>
          <div
            className="h-full"
            style={{ width: '100%', background: 'var(--accent-primary)' }}
          ></div>
        </div>

        {/* Stage history dots with timing */}
        <div className="flex items-center justify-center" style={{ gap: 'var(--space-2)', padding: 'var(--space-2) 0' }}>
          {Object.values(STAGES).map((s) => {
            const isCompleted = isStageCompleted(s.id)
            const stageInfo = getStageInfo(s.id)
            const hasTimingInfo = stageInfo && stageInfo.duration_ms !== undefined
            
            // Format tooltip with timing info
            let tooltip = s.label
            if (hasTimingInfo && stageInfo.duration_ms !== undefined) {
              tooltip += ` (${stageInfo.duration_ms}ms)`
              if (stageInfo.metadata?.chunks_retrieved) {
                tooltip += ` - ${stageInfo.metadata.chunks_retrieved} chunks`
              }
              if (stageInfo.metadata?.context_items) {
                tooltip += ` - ${stageInfo.metadata.context_items} items`
              }
              if (stageInfo.metadata?.routing_categories && stageInfo.metadata.routing_categories.length > 0) {
                const conf = stageInfo.metadata?.routing_confidence
                const confPct = typeof conf === 'number' ? ` (${(conf * 100).toFixed(0)}% confidence)` : ''
                const categories = stageInfo.metadata.routing_categories.join(', ')
                tooltip += ` — Routed to: ${categories}${confPct}`
                if (stageInfo.metadata?.document_count) {
                  tooltip += ` — ${stageInfo.metadata.document_count} documents`
                }
              } else if (stageInfo.metadata?.routing_category_id) {
                const conf = stageInfo.metadata?.routing_confidence
                const confPct = typeof conf === 'number' ? ` ${(conf * 100).toFixed(0)}%` : ''
                tooltip += ` — Routed to category ${stageInfo.metadata.routing_category_id}${confPct}`
              }
            }
            
            return (
              <div
                key={s.id}
                className="cursor-help group relative"
                title={tooltip}
              >
                <div
                  style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: 'var(--radius-full)',
                    background: isCompleted ? 'var(--accent-primary)' : 'var(--gray-300)',
                    transition: 'all var(--timing-normal) var(--easing-standard)'
                  }}
                ></div>
                <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 text-xs rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" style={{ padding: 'var(--space-1) var(--space-2)', background: 'var(--gray-800)', color: 'white' }}>
                  {tooltip}
                </div>
              </div>
            )
          })}
        </div>
        
        {/* Total duration display */}
        {totalDuration > 0 && (
          <div className="flex items-center justify-center" style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>
            <span>Completed in {totalDuration}ms</span>
          </div>
        )}
      </div>
    )
  }

  // Loading state
  return (
    <div className="w-full">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
        {/* Current stage with spinner and timing info */}
        <div className="flex items-center justify-between">
          <div className="flex items-center" style={{ gap: 'var(--space-3)' }}>
            <div className="spinner" style={{ width: '12px', height: '12px' }} />
            <span style={{ fontSize: 'var(--text-sm)', fontWeight: 500, color: 'var(--text-secondary)' }}>{stage.label}</span>
          </div>
          {currentStageInfo?.metadata && (
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)' }}>
              {currentStageInfo.metadata.chunks_retrieved && `${currentStageInfo.metadata.chunks_retrieved} chunks`}
              {currentStageInfo.metadata.context_items && `${currentStageInfo.metadata.context_items} items`}
            </span>
          )}
        </div>

        {/* Progress bar */}
        <div className="relative h-1 rounded-full overflow-hidden" style={{ background: 'var(--gray-200)' }}>
          <div
            className="h-full"
            style={{
              width: `${((completedStagesCount + 1) / Object.keys(STAGES).length) * 100}%`,
              background: 'var(--accent-primary)',
              transition: 'width var(--timing-slow) var(--easing-standard)'
            }}
          ></div>
        </div>

        {/* Stage history dots with timing */}
        <div className="flex items-center justify-center" style={{ gap: 'var(--space-2)', padding: 'var(--space-2) 0' }}>
          {Object.values(STAGES).map((s) => {
            const isCompleted = isStageCompleted(s.id)
            const isCurrent = s.id === displayedStage
            const stageInfo = getStageInfo(s.id)
            
            // Format tooltip with timing info
            let tooltip = s.label
            if (stageInfo && stageInfo.duration_ms !== undefined) {
              tooltip += ` (${stageInfo.duration_ms}ms)`
              if (stageInfo.metadata?.routing_categories && stageInfo.metadata.routing_categories.length > 0) {
                const conf = stageInfo.metadata?.routing_confidence
                const confPct = typeof conf === 'number' ? ` (${(conf * 100).toFixed(0)}% confidence)` : ''
                const categories = stageInfo.metadata.routing_categories.join(', ')
                tooltip += ` — Routed to: ${categories}${confPct}`
                if (stageInfo.metadata?.document_count) {
                  tooltip += ` — ${stageInfo.metadata.document_count} documents`
                }
              } else if (stageInfo.metadata?.routing_category_id) {
                const conf = stageInfo.metadata?.routing_confidence
                const confPct = typeof conf === 'number' ? ` ${(conf * 100).toFixed(0)}%` : ''
                tooltip += ` — Routed to category ${stageInfo.metadata.routing_category_id}${confPct}`
              }
            }

            return (
              <div
                key={s.id}
                className="cursor-help group relative"
                title={tooltip}
              >
                <div
                  style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: 'var(--radius-full)',
                    background: isCompleted ? 'var(--accent-primary)' : isCurrent ? 'var(--accent-primary)' : 'var(--gray-300)',
                    opacity: isCurrent ? 0.5 : 1,
                    transform: isCurrent ? 'scale(1.25)' : 'scale(1)',
                    transition: 'all var(--timing-normal) var(--easing-standard)'
                  }}
                ></div>
                <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 text-xs rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" style={{ padding: 'var(--space-1) var(--space-2)', background: 'var(--gray-800)', color: 'white' }}>
                  {tooltip}
                </div>
              </div>
            )
          })}
        </div>

            {/* Processing indicator */}
        <div className="flex items-center justify-center" style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>
          <span>Processing</span>
        </div>
      </div>
    </div>
  )
}
