'use client';

import { useState, useEffect } from 'react';
import { Activity, RefreshCw } from 'lucide-react';
import BenchmarkRunner from './shared/BenchmarkRunner';
import MetricsCard from './shared/MetricsCard';
import ResultsTable from './shared/ResultsTable';
import ComparisonChart from './shared/ComparisonChart';

interface RagasStats {
  status: string;
  variants?: Record<string, {
    metrics: {
      context_precision?: number;
      context_recall?: number;
      faithfulness?: number;
      answer_relevancy?: number;
    };
    evaluation_count?: number;
    last_evaluation_timestamp?: number;
  }>;
}

interface RagasComparison {
  status: string;
  best_overall_variant?: {
    name: string;
    average_score: number;
    metrics: Record<string, number>;
  };
  all_variants?: Record<string, {
    average_score: number;
    metrics: Record<string, number>;
  }>;
}

export default function RagasSubPanel() {
  const [stats, setStats] = useState<RagasStats | null>(null);
  const [comparison, setComparison] = useState<RagasComparison | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Load stats and comparison on mount
  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      await Promise.all([loadStats(), loadComparison()]);
    } finally {
      setLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const response = await fetch('/api/ragas/stats');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Failed to load RAGAS stats:', error);
    }
  };

  const loadComparison = async () => {
    try {
      const response = await fetch('/api/ragas/comparison');
      if (response.ok) {
        const data = await response.json();
        setComparison(data);
      }
    } catch (error) {
      console.error('Failed to load RAGAS comparison:', error);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await loadData();
    } finally {
      setRefreshing(false);
    }
  };

  const handleBenchmarkComplete = () => {
    // Reload data after benchmark completes
    setTimeout(() => {
      loadData();
    }, 2000);
  };

  // Calculate overall stats from all variants
  const calculateOverallStats = () => {
    if (!stats?.variants) return null;

    const variants = Object.values(stats.variants);
    if (variants.length === 0) return null;

    const avgMetrics = {
      context_precision: 0,
      context_recall: 0,
      faithfulness: 0,
      answer_relevancy: 0,
    };

    let count = 0;
    variants.forEach(variant => {
      if (variant.metrics.context_precision !== undefined) {
        avgMetrics.context_precision += variant.metrics.context_precision;
        count++;
      }
      if (variant.metrics.context_recall !== undefined) {
        avgMetrics.context_recall += variant.metrics.context_recall;
      }
      if (variant.metrics.faithfulness !== undefined) {
        avgMetrics.faithfulness += variant.metrics.faithfulness;
      }
      if (variant.metrics.answer_relevancy !== undefined) {
        avgMetrics.answer_relevancy += variant.metrics.answer_relevancy;
      }
    });

    if (count === 0) return null;

    return {
      context_precision: avgMetrics.context_precision / count,
      context_recall: avgMetrics.context_recall / count,
      faithfulness: avgMetrics.faithfulness / count,
      answer_relevancy: avgMetrics.answer_relevancy / count,
    };
  };

  const overallStats = calculateOverallStats();

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2 text-gray-400" />
          <p className="text-sm text-gray-600 dark:text-gray-400">Loading RAGAS data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col overflow-hidden" style={{ background: 'var(--bg-primary)' }}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: 'var(--border)' }}>
        <h3 className="font-medium" style={{ color: 'var(--text-secondary)' }}>RAGAS Benchmarks</h3>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="px-3 py-1.5 border rounded-md text-sm transition-colors disabled:opacity-50 flex items-center gap-2"
          style={{
            borderColor: 'var(--border)',
            color: 'var(--text-primary)',
            background: 'var(--bg-secondary)'
          }}
        >
          <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh Results
        </button>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto p-6 pb-28 space-y-8">
        {/* Section 1: Run Benchmark */}
        <section>
          <h3 className="text-lg font-semibold mb-4">Run Benchmark</h3>
          <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-6 bg-white dark:bg-neutral-800">
            <BenchmarkRunner onComplete={handleBenchmarkComplete} />
          </div>
        </section>

        {/* Section 2: Overall Metrics */}
        {overallStats && (
          <section>
            <h3 className="text-lg font-semibold mb-4">Overall Metrics (All Variants)</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricsCard
                title="Context Precision"
                value={overallStats.context_precision}
                format="percentage"
                threshold={{ min: 0.65 }}
              />
              <MetricsCard
                title="Context Recall"
                value={overallStats.context_recall}
                format="percentage"
                threshold={{ min: 0.60 }}
              />
              <MetricsCard
                title="Faithfulness"
                value={overallStats.faithfulness}
                format="percentage"
                threshold={{ min: 0.85 }}
              />
              <MetricsCard
                title="Answer Relevancy"
                value={overallStats.answer_relevancy}
                format="percentage"
                threshold={{ min: 0.80 }}
              />
            </div>
          </section>
        )}

        {/* Section 3: Results Table */}
        {stats?.variants && Object.keys(stats.variants).length > 0 && (
          <section>
            <h3 className="text-lg font-semibold mb-4">Variant Results</h3>
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden bg-white dark:bg-neutral-800">
              <ResultsTable data={stats.variants} />
            </div>
          </section>
        )}

        {/* Section 4: Comparison Chart */}
        {comparison?.all_variants && Object.keys(comparison.all_variants).length > 0 && (
          <section>
            <h3 className="text-lg font-semibold mb-4">Variant Comparison</h3>
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-6 bg-white dark:bg-neutral-800">
              <ComparisonChart
                data={comparison.all_variants}
                bestVariant={comparison.best_overall_variant}
              />
            </div>
          </section>
        )}

        {/* No Data State */}
        {(!stats?.variants || Object.keys(stats.variants).length === 0) && (
          <div className="text-center py-12">
            <Activity className="w-16 h-16 mx-auto mb-4 text-gray-300 dark:text-gray-600" />
            <h3 className="text-lg font-semibold mb-2">No Benchmark Results Yet</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
              Run your first RAGAS benchmark to see quality metrics and variant comparisons
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
