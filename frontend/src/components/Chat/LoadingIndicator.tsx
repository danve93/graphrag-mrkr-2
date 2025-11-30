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
  retrieval: {
    id: 'retrieval',
    label: 'Retrieving Documents',
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

interface LoadingIndicatorProps {
  currentStage?: string
  completedStages?: string[]
  isLoading?: boolean
  enableQualityScoring?: boolean
}

export default function LoadingIndicator({ 
  currentStage = 'query_analysis',
  completedStages = [],
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
  
  // If not loading, show minimal clean display
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

        {/* Stage history dots */}
        <div className="flex items-center justify-center" style={{ gap: 'var(--space-2)', padding: 'var(--space-2) 0' }}>
          {Object.values(STAGES).map((s) => {
            const isCompleted = isStageCompleted(s.id)
            return (
              <div
                key={s.id}
                className="cursor-help group relative"
                title={s.label}
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
                  {s.label}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    )
  }

  // Loading state
  return (
    <div className="w-full">
      <style>{`
        .animate-slide-in {
          animation: slideIn var(--timing-normal) var(--easing-standard);
        }

        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateX(-10px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
      `}</style>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
        {/* Current stage with spinner */}
        <div className="flex items-center animate-slide-in" style={{ gap: 'var(--space-3)' }}>
          <div className="spinner" style={{ width: '12px', height: '12px' }} />
          <span style={{ fontSize: 'var(--text-sm)', fontWeight: 500, color: 'var(--text-secondary)' }}>{stage.label}</span>
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

        {/* Stage history dots */}
        <div className="flex items-center justify-center" style={{ gap: 'var(--space-2)', padding: 'var(--space-2) 0' }}>
          {Object.values(STAGES).map((s) => {
            const isCompleted = isStageCompleted(s.id)
            const isCurrent = s.id === displayedStage

            return (
              <div
                key={s.id}
                className="cursor-help group relative"
                title={s.label}
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
                  {s.label}
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
