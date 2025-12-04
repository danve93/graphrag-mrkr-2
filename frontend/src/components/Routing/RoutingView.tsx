'use client';

import { useState, useEffect } from 'react';
import { Activity, TrendingUp, Clock, Database, Split, RefreshCw, AlertCircle } from 'lucide-react';
import { Button } from '@mui/material';
import { Route as RouteIcon } from '@mui/icons-material';
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
            <RouteIcon style={{ fontSize: '24px', color: '#f27a03' }} />
          </div>
          <div style={{ flex: 1 }}>
            <h1 className="font-display" style={{ fontSize: 'var(--text-2xl)', fontWeight: 700, color: 'var(--text-primary)' }}>
              Query Routing Dashboard
            </h1>
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
              Real-time metrics and performance monitoring
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded"
              />
              Auto-refresh (3s)
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

      <div className="flex-1 min-h-0 overflow-y-auto" style={{ padding: 'var(--space-6)' }}>
        {/* Main Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" style={{ marginBottom: '24px' }}>
          {/* Total Queries */}
          <MetricCard
            icon={<Activity className="w-5 h-5" />}
            label="Total Queries"
            value={metrics.total_queries.toString()}
            iconColor="text-blue-600"
            bgColor="bg-blue-50 dark:bg-blue-900/20"
          />

          {/* Cache Hit Rate */}
          <MetricCard
            icon={<Database className="w-5 h-5" />}
            label="Cache Hit Rate"
            value={hasQueries ? formatPercentage(metrics.cache_hit_rate) : 'N/A'}
            subtitle={hasQueries ? 'Latency reduction: ~30%' : 'No queries yet'}
            iconColor="text-green-600"
            bgColor="bg-green-50 dark:bg-green-900/20"
            status={hasQueries && metrics.cache_hit_rate > 0.4 ? 'good' : hasQueries ? 'warning' : undefined}
          />

          {/* Avg Routing Latency */}
          <MetricCard
            icon={<Clock className="w-5 h-5" />}
            label="Avg Routing Latency"
            value={hasQueries ? formatLatency(metrics.avg_routing_latency_ms) : 'N/A'}
            subtitle={hasQueries ? (metrics.cache_hit_rate > 0 ? 'With cache hits' : 'No cache hits yet') : 'No queries yet'}
            iconColor="text-purple-600"
            bgColor="bg-purple-50 dark:bg-purple-900/20"
            status={hasQueries && metrics.avg_routing_latency_ms < 500 ? 'good' : hasQueries && metrics.avg_routing_latency_ms < 2000 ? 'warning' : hasQueries ? 'error' : undefined}
          />

          {/* Fallback Rate */}
          <MetricCard
            icon={<TrendingUp className="w-5 h-5" />}
            label="Fallback Rate"
            value={hasQueries ? formatPercentage(metrics.fallback_rate) : 'N/A'}
            subtitle={hasQueries ? 'Expanded to all docs' : 'No queries yet'}
            iconColor="text-orange-600"
            bgColor="bg-orange-50 dark:bg-orange-900/20"
            status={hasQueries && metrics.fallback_rate < 0.1 ? 'good' : hasQueries && metrics.fallback_rate < 0.3 ? 'warning' : hasQueries ? 'error' : undefined}
          />

          {/* Multi-Category Rate */}
          <MetricCard
            icon={<Split className="w-5 h-5" />}
            label="Multi-Category Rate"
            value={hasQueries ? formatPercentage(metrics.multi_category_rate) : 'N/A'}
            subtitle={hasQueries ? 'Queries matching 2-3 categories' : 'No queries yet'}
            iconColor="text-indigo-600"
            bgColor="bg-indigo-50 dark:bg-indigo-900/20"
          />

          {/* Routing Accuracy */}
          <MetricCard
            icon={<TrendingUp className="w-5 h-5" />}
            label="Routing Accuracy"
            value={metrics.routing_accuracy !== null ? formatPercentage(metrics.routing_accuracy) : 'N/A'}
            subtitle="Based on user feedback"
            iconColor="text-teal-600"
            bgColor="bg-teal-50 dark:bg-teal-900/20"
          />
        </div>

        {/* Top Categories Chart */}
        {hasQueries && metrics.top_categories.length > 0 && (
          <div className="bg-white dark:bg-secondary-800 rounded-lg border border-secondary-200 dark:border-secondary-700 p-6">
            <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100 mb-4">
              Top Categories
            </h2>
            <div className="space-y-3">
              {metrics.top_categories.slice(0, 10).map(([category, count]) => {
                const percentage = (count / metrics.total_queries) * 100;
                return (
                  <div key={category} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium text-secondary-700 dark:text-secondary-300">
                        {category}
                      </span>
                      <span className="text-secondary-600 dark:text-secondary-400">
                        {count} ({percentage.toFixed(1)}%)
                      </span>
                    </div>
                    <div className="h-2 bg-secondary-100 dark:bg-secondary-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary-600 rounded-full transition-all duration-300"
                        style={{ width: `${percentage}%` }}
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
          <div className="bg-white dark:bg-secondary-800 rounded-lg border border-secondary-200 dark:border-secondary-700 p-12 text-center">
            <Activity className="w-16 h-16 mx-auto mb-4 text-secondary-300 dark:text-secondary-600" />
            <h3 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100 mb-2">
              No Routing Data Yet
            </h3>
            <p className="text-secondary-600 dark:text-secondary-400">
              Send some queries with routing enabled to see metrics here.
            </p>
            <p className="text-sm text-secondary-500 dark:text-secondary-500 mt-2">
              Set <code className="px-2 py-1 bg-secondary-100 dark:bg-secondary-700 rounded">ENABLE_QUERY_ROUTING=true</code> in your .env file
            </p>
          </div>
        )}

        {/* KPI Status Summary */}
        {hasQueries && (
          <div className="bg-white dark:bg-secondary-800 rounded-lg border border-secondary-200 dark:border-secondary-700 p-6">
            <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100 mb-4">
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
        <FeedbackMetricsDashboard />
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
    <div className="bg-white dark:bg-secondary-800 rounded-lg border border-secondary-200 dark:border-secondary-700 p-6 relative overflow-hidden">
      <div className="flex items-start justify-between mb-2">
        <div className={`p-2 rounded-lg ${bgColor}`}>
          <div className={iconColor}>{icon}</div>
        </div>
        {status && (
          <div className={`w-2 h-2 rounded-full ${
            status === 'good' ? 'bg-green-500' :
            status === 'warning' ? 'bg-yellow-500' :
            'bg-red-500'
          }`} />
        )}
      </div>
      <div className="text-2xl font-bold text-secondary-900 dark:text-secondary-100 mb-1">
        {value}
      </div>
      <div className="text-sm text-secondary-600 dark:text-secondary-400">
        {label}
      </div>
      {subtitle && (
        <div className="text-xs text-secondary-500 dark:text-secondary-500 mt-1">
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
  const statusColors = {
    good: 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20',
    warning: 'text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/20',
    error: 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20',
  };

  const statusLabels = {
    good: '✓ Meeting Target',
    warning: '⚠ Below Target',
    error: '✗ Critical',
  };

  return (
    <div className="flex items-center justify-between p-3 rounded-lg bg-secondary-50 dark:bg-secondary-900/50">
      <div>
        <div className="text-sm font-medium text-secondary-900 dark:text-secondary-100">
          {label}
        </div>
        <div className="text-xs text-secondary-600 dark:text-secondary-400">
          Target: {target} | Current: {current}
        </div>
      </div>
      <div className={`px-3 py-1 rounded-full text-xs font-medium ${statusColors[status]}`}>
        {statusLabels[status]}
      </div>
    </div>
  );
}
