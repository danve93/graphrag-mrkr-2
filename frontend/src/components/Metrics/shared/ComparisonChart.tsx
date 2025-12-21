'use client';

import { motion } from 'framer-motion';

interface VariantScore {
  average_score: number;
  metrics: {
    context_precision?: number;
    context_recall?: number;
    faithfulness?: number;
    answer_relevancy?: number;
  };
}

interface BestVariant {
  name: string;
  average_score: number;
  metrics: Record<string, number>;
}

interface ComparisonChartProps {
  data: Record<string, VariantScore>;
  bestVariant?: BestVariant;
}

export default function ComparisonChart({ data, bestVariant }: ComparisonChartProps) {
  const variants = Object.entries(data).map(([name, scores]) => ({
    name,
    score: scores.average_score,
  }));

  // Sort by score descending
  variants.sort((a, b) => b.score - a.score);

  const maxScore = Math.max(...variants.map(v => v.score), 1);

  const getBarColor = (score: number, index: number) => {
    if (index === 0) return ''; // Best variant - use style prop for accent color
    if (score >= 0.75) return 'bg-green-500';
    if (score >= 0.60) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  if (variants.length === 0) {
    return (
      <div className="text-center p-8 text-gray-600 dark:text-gray-400">
        No data to compare
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Best Variant Banner */}
      {bestVariant && (
        <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Best Overall Variant</p>
              <p className="text-xl font-bold" style={{ color: 'var(--accent-primary)' }}>{bestVariant.name}</p>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-600 dark:text-gray-400">Average Score</p>
              <p className="text-2xl font-bold" style={{ color: 'var(--accent-primary)' }}>
                {(bestVariant.average_score * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Bar Chart */}
      <div className="space-y-4">
        {variants.map((variant, index) => (
          <div key={variant.name} className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium flex items-center gap-2">
                {variant.name}
                {index === 0 && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-orange-100 dark:bg-orange-900/30" style={{ color: 'var(--accent-primary)' }}>
                    Best
                  </span>
                )}
              </span>
              <span className="text-gray-600 dark:text-gray-400">
                {(variant.score * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-8 bg-gray-100 dark:bg-neutral-800 rounded-lg overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${(variant.score / maxScore) * 100}%` }}
                transition={{ duration: 0.8, ease: 'easeOut', delay: index * 0.1 }}
                className={`h-full ${getBarColor(variant.score, index)} rounded-lg flex items-center justify-end pr-3`}
                style={{ backgroundColor: index === 0 ? 'var(--accent-primary)' : undefined }}
              >
                {variant.score / maxScore > 0.3 && (
                  <span className="text-white text-xs font-semibold">
                    {(variant.score * 100).toFixed(1)}%
                  </span>
                )}
              </motion.div>
            </div>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-xs text-gray-600 dark:text-gray-400 pt-2 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: 'var(--accent-primary)' }}></div>
          <span>Best</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-green-500"></div>
          <span>≥75%</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-yellow-500"></div>
          <span>≥60%</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-red-500"></div>
          <span>&lt;60%</span>
        </div>
      </div>
    </div>
  );
}
