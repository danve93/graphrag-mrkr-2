'use client';

import { useState, useEffect } from 'react';
import { Activity, TrendingUp, Clock, Database, Split, RefreshCw, AlertCircle } from 'lucide-react';
import { Button } from '@mui/material';
import ExpandablePanel from '@/components/Utils/ExpandablePanel';
import { FeedbackMetricsDashboard } from '../Chat/FeedbackMetricsDashboard';
import { API_URL } from '@/lib/api';

interface RoutingMetrics {
  total_queries: number;
  avg_routing_latency_ms: number;
  routing_accuracy: number | null;
  cache_hit_rate: number;
  fallback_rate: number;
  multi_category_rate: number;
  top_categories: [string, number][];
  failure_point_rates?: Record<string, number>;
  failure_point_counts?: Record<string, number>;
}

export default function RoutingView() {
  const [metrics, setMetrics] = useState<RoutingMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [expandedPanels, setExpandedPanels] = useState<Set<string>>(new Set(['metrics', 'feedback']));

  const togglePanel = (panel: string) => {
    setExpandedPanels((prev) => {
      const next = new Set(prev);
      if (next.has(panel)) {
        next.delete(panel);
      } else {
        next.add(panel);
      }
      return next;
    });
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${API_URL}/api/database/routing-metrics`);
      if (!response.ok) throw new Error('Failed to fetch metrics');
      const data = await response.json();
      setMetrics(data);
      setError(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(fetchMetrics, 3000); // Refresh every 3s
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;
  const formatLatency = (ms: number) => ms < 1000 ? `${ms.toFixed(0)}ms` : `${(ms / 1000).toFixed(2)}s`;

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-secondary-50 dark:bg-secondary-900 pb-28">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-secondary-400" />
          <p className="text-secondary-600 dark:text-secondary-400">Loading metrics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center bg-secondary-50 dark:bg-secondary-900 pb-28">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 mx-auto mb-4 text-red-500" />
          <h2 className="text-xl font-semibold mb-2 text-secondary-900 dark:text-secondary-100">
            Failed to Load Metrics
          </h2>
          <p className="text-secondary-600 dark:text-secondary-400 mb-4">{error}</p>
          <button
            onClick={fetchMetrics}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!metrics) return null;

  const hasQueries = metrics.total_queries > 0;

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--bg-primary)' }}>
      {/* Header */}
      <div style={{ borderBottom: '1px solid var(--border)', padding: 'var(--space-6)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 'var(--space-4)' }}>
          <div>
            <h2 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
              Routing Performance
            </h2>
            <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              Real-time query routing statistics and cache performance
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button
              onClick={async () => {
                if (!confirm('Are you sure you want to clear all routing metrics? This cannot be undone.')) return;
                try {
                  const res = await fetch(`${API_URL}/api/database/routing-metrics`, { method: 'DELETE' });
                  if (!res.ok) throw new Error('Failed to clear metrics');
                  fetchMetrics();
                } catch (err: any) {
                  alert(err.message);
                }
              }}
              className="px-3 py-1.5 text-xs font-medium text-red-600 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors border border-transparent hover:border-red-200 dark:hover:border-red-800"
            >
              Clear Metrics
            </button>
            <div className="h-4 w-px bg-gray-200 dark:bg-gray-700" />
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded border-gray-300"
              />
              Auto-refresh
            </label>
            <Button
              size="small"
              onClick={fetchMetrics}
              style={{ minWidth: 'auto', padding: '6px' }}
              title="Refresh metrics"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto pb-28 p-[var(--space-6)]">
        {/* Main Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" style={{ marginBottom: '24px' }}>
          {/* Total Queries */}
          <MetricCard
            icon={<Activity className="w-5 h-5" />}
            label="Total Queries"
            value={metrics.total_queries.toString()}
            iconColor="text-primary-500"
            bgColor="bg-primary-500/10 dark:bg-primary-500/20"
          />

          {/* Cache Hit Rate */}
          <MetricCard
            icon={<Database className="w-5 h-5" />}
            label="Cache Hit Rate"
            value={hasQueries ? formatPercentage(metrics.cache_hit_rate) : 'N/A'}
            subtitle={hasQueries ? 'Latency reduction: ~30%' : 'No queries yet'}
            iconColor="text-primary-500"
            bgColor="bg-primary-500/10 dark:bg-primary-500/20"
            status={hasQueries && metrics.cache_hit_rate > 0.4 ? 'good' : hasQueries ? 'warning' : undefined}
          />

          {/* Avg Routing Latency */}
          <MetricCard
            icon={<Clock className="w-5 h-5" />}
            label="Avg Routing Latency"
            value={hasQueries ? formatLatency(metrics.avg_routing_latency_ms) : 'N/A'}
            subtitle={hasQueries ? (metrics.cache_hit_rate > 0 ? 'With cache hits' : 'No cache hits yet') : 'No queries yet'}
            iconColor="text-primary-500"
            bgColor="bg-primary-500/10 dark:bg-primary-500/20"
            status={hasQueries && metrics.avg_routing_latency_ms < 500 ? 'good' : hasQueries && metrics.avg_routing_latency_ms < 2000 ? 'warning' : hasQueries ? 'error' : undefined}
          />

          {/* Fallback Rate */}
          <MetricCard
            icon={<TrendingUp className="w-5 h-5" />}
            label="Fallback Rate"
            value={hasQueries ? formatPercentage(metrics.fallback_rate) : 'N/A'}
            subtitle={hasQueries ? 'Expanded to all docs' : 'No queries yet'}
            iconColor="text-primary-500"
            bgColor="bg-primary-500/10 dark:bg-primary-500/20"
            status={hasQueries && metrics.fallback_rate < 0.1 ? 'good' : hasQueries && metrics.fallback_rate < 0.3 ? 'warning' : hasQueries ? 'error' : undefined}
          />

          {/* Multi-Category Rate */}
          <MetricCard
            icon={<Split className="w-5 h-5" />}
            label="Multi-Category Rate"
            value={hasQueries ? formatPercentage(metrics.multi_category_rate) : 'N/A'}
            subtitle={hasQueries ? 'Queries matching 2-3 categories' : 'No queries yet'}
            iconColor="text-primary-500"
            bgColor="bg-primary-500/10 dark:bg-primary-500/20"
          />

          {/* Routing Accuracy */}
          <MetricCard
            icon={<TrendingUp className="w-5 h-5" />}
            label="Routing Accuracy"
            value={metrics.routing_accuracy !== null ? formatPercentage(metrics.routing_accuracy) : 'N/A'}
            subtitle="Based on user feedback"
            iconColor="text-primary-500"
            bgColor="bg-primary-500/10 dark:bg-primary-500/20"
          />
        </div>

        {/* Top Categories Chart */}
        {hasQueries && metrics.top_categories.length > 0 && (
          <div
            className="rounded-lg p-6 border"
            style={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
          >
            <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
              Top Categories
            </h2>
            <div className="space-y-3">
              {metrics.top_categories.slice(0, 10).map(([category, count]) => {
                const percentage = (count / metrics.total_queries) * 100;
                return (
                  <div key={category} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                        {category}
                      </span>
                      <span style={{ color: 'var(--text-secondary)' }}>
                        {count} ({percentage.toFixed(1)}%)
                      </span>
                    </div>
                    <div className="h-2 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-tertiary)' }}>
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{ width: `${percentage}%`, backgroundColor: 'var(--accent-primary)' }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Empty State */}
        {!hasQueries && (
          <div
            className="rounded-lg p-12 text-center border"
            style={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border)', borderStyle: 'dashed' }}
          >
            <Activity className="w-12 h-12 mx-auto mb-3" style={{ color: 'var(--text-tertiary)' }} />
            <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
              No Routing Data Yet
            </h3>
            <p style={{ color: 'var(--text-secondary)' }}>
              Send some queries with routing enabled to see metrics here.
            </p>
            <p className="text-sm mt-2" style={{ color: 'var(--text-tertiary)' }}>
              Set <code className="px-2 py-1 rounded" style={{ backgroundColor: 'var(--bg-tertiary)' }}>ENABLE_QUERY_ROUTING=true</code> in your .env file
            </p>
          </div>
        )}

        {/* KPI Status Summary */}
        {hasQueries && (
          <div
            className="rounded-lg p-6 border"
            style={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
          >
            <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
              Cache & Routing Stats
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <KPIStatus
                label="Cache Hit Rate"
                target="> 40%"
                current={formatPercentage(metrics.cache_hit_rate)}
                status={metrics.cache_hit_rate > 0.4 ? 'good' : 'warning'}
              />
              <KPIStatus
                label="Routing Latency"
                target="< 500ms"
                current={formatLatency(metrics.avg_routing_latency_ms)}
                status={metrics.avg_routing_latency_ms < 500 ? 'good' : metrics.avg_routing_latency_ms < 2000 ? 'warning' : 'error'}
              />
              <KPIStatus
                label="Fallback Rate"
                target="< 10%"
                current={formatPercentage(metrics.fallback_rate)}
                status={metrics.fallback_rate < 0.1 ? 'good' : metrics.fallback_rate < 0.3 ? 'warning' : 'error'}
              />
              {metrics.routing_accuracy !== null && (
                <KPIStatus
                  label="Routing Accuracy"
                  target="> 85%"
                  current={formatPercentage(metrics.routing_accuracy)}
                  status={metrics.routing_accuracy > 0.85 ? 'good' : metrics.routing_accuracy > 0.7 ? 'warning' : 'error'}
                />
              )}
            </div>
          </div>
        )}

        {/* Feedback Metrics Dashboard */}
        <div style={{ marginTop: 'var(--space-6)' }}>
          <FeedbackMetricsDashboard />
        </div>
      </div>
    </div>
  );
}

interface MetricCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  subtitle?: string;
  iconColor: string;
  bgColor: string;
  status?: 'good' | 'warning' | 'error';
}

function MetricCard({ icon, label, value, subtitle, iconColor, bgColor, status }: MetricCardProps) {
  return (
    <div
      className="rounded-lg p-5 border relative overflow-hidden"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        borderColor: 'var(--border)'
      }}
    >
      <div className="flex items-start justify-between mb-3">
        <div className={`p-2 rounded-lg ${bgColor}`}>
          <div className={iconColor}>{icon}</div>
        </div>
        {status && (
          <div className={`w-2 h-2 rounded-full ${status === 'good' ? 'bg-green-500' :
            status === 'warning' ? 'bg-yellow-500' :
              'bg-red-500'
            }`} />
        )}
      </div>
      <div className="text-3xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>
        {value}
      </div>
      <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
        {label}
      </div>
      {subtitle && (
        <div className="text-xs mt-1" style={{ color: 'var(--text-tertiary)' }}>
          {subtitle}
        </div>
      )}
    </div>
  );
}

interface KPIStatusProps {
  label: string;
  target: string;
  current: string;
  status: 'good' | 'warning' | 'error';
}

function KPIStatus({ label, target, current, status }: KPIStatusProps) {
  // Use project standard colors for status indicators
  const statusColors = {
    good: { bg: 'rgba(50, 215, 75, 0.15)', color: '#32D74B' }, // Green
    warning: { bg: 'rgba(255, 214, 10, 0.15)', color: '#FFD60A' }, // Yellow
    error: { bg: 'rgba(255, 69, 58, 0.15)', color: '#FF453A' }, // Red
  };

  const statusLabels = {
    good: '✓ Meeting Target',
    warning: '⚠ Below Target',
    error: '✗ Critical',
  };

  return (
    <div
      className="flex items-center justify-between p-3 rounded-lg"
      style={{ backgroundColor: 'var(--bg-tertiary)' }}
    >
      <div>
        <div className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
          {label}
        </div>
        <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
          Target: {target} | Current: {current}
        </div>
      </div>
      <div
        className="px-3 py-1 rounded-full text-xs font-medium"
        style={{ backgroundColor: statusColors[status].bg, color: statusColors[status].color }}
      >
        {statusLabels[status]}
      </div>
    </div>
  );
}
