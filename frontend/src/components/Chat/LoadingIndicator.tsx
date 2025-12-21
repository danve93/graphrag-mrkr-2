'use client'

import { useEffect, useState } from 'react'

interface Stage {
  id: string
  label: string
}

const STAGES: Record<string, Stage> = {
  query_analysis: { id: 'query_analysis', label: 'Analyzing' },
  routing: { id: 'routing', label: 'Routing' },
  retrieval: { id: 'retrieval', label: 'Retrieving' },
  consolidation: { id: 'consolidation', label: 'Consolidating' },
  graph_reasoning: { id: 'graph_reasoning', label: 'Reasoning' },
  generation: { id: 'generation', label: 'Generating' },
  quality_calculation: { id: 'quality_calculation', label: 'Checking' },
  suggestions: { id: 'suggestions', label: 'Finalizing' },
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

  useEffect(() => {
    if (isLoading && currentStage) {
      setDisplayedStage(currentStage)
    }
  }, [currentStage, isLoading])

  useEffect(() => {
    if (!isLoading && completedStages.length > 0) {
      setDisplayedStage(completedStages[completedStages.length - 1])
    }
  }, [completedStages, isLoading])

  const stage = STAGES[displayedStage] || STAGES.query_analysis
  const totalDuration = stageUpdates.reduce((sum, s) => sum + (s.duration_ms || 0), 0)

  // Completion state - minimal summary
  if (!isLoading) {
    return (
      <div className="flex items-center gap-3" style={{ fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)' }}>
        <div className="flex items-center gap-1.5">
          <div style={{ width: '4px', height: '4px', borderRadius: '50%', background: 'var(--accent-primary)' }} />
          <div style={{ width: '4px', height: '4px', borderRadius: '50%', background: 'var(--accent-primary)' }} />
          <div style={{ width: '4px', height: '4px', borderRadius: '50%', background: 'var(--accent-primary)' }} />
        </div>
        {totalDuration > 0 && (
          <span>{(totalDuration / 1000).toFixed(1)}s</span>
        )}
      </div>
    )
  }

  // Loading state - clean typing indicator
  return (
    <div className="flex items-center gap-3">
      {/* Typing indicator - 3 bouncing dots */}
      <div
        className="flex items-center justify-center gap-1"
        style={{
          padding: '12px 16px',
          borderRadius: 'var(--radius-lg)',
          background: 'var(--bg-secondary)',
        }}
      >
        <div className="typing-dot" />
        <div className="typing-dot" />
        <div className="typing-dot" />
      </div>

      {/* Subtle stage indicator */}
      <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)' }}>
        {stage.label}
      </span>
    </div>
  )
}
