'use client';

import { useState, useEffect } from 'react';
import { Activity, Power, PowerOff, ExternalLink, Download, Trash2, RefreshCw, AlertCircle } from 'lucide-react';
import HealthIndicator from './shared/HealthIndicator';
import MetricsCard from './shared/MetricsCard';

interface TruLensHealth {
  status: string;
  monitoring_enabled: boolean;
  database_connected?: boolean;
  database_status?: string;
  sampling_rate?: number;
  message?: string;
  error?: string;
}

interface TruLensStats {
  status: string;
  total_queries?: number;
  avg_answer_relevance?: number;
  avg_groundedness?: number;
  avg_context_relevance?: number;
  avg_latency_ms?: number;
  error_rate?: number;
  last_updated?: string;
}

export default function TruLensSubPanel() {
  const [health, setHealth] = useState<TruLensHealth | null>(null);
  const [stats, setStats] = useState<TruLensStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [samplingRate, setSamplingRate] = useState(100);
  const [dashboardStatus, setDashboardStatus] = useState<'stopped' | 'running' | 'unknown'>('unknown');

  useEffect(() => {
    loadData();
    loadDashboardStatus();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      await Promise.all([loadHealth(), loadStats()]);
    } finally {
      setLoading(false);
    }
  };

  const loadHealth = async () => {
    try {
      const response = await fetch('/api/trulens/health');
      if (response.ok) {
        const data = await response.json();
        setHealth(data);
        if (data.sampling_rate !== undefined) {
          setSamplingRate(data.sampling_rate * 100);
        }
      }
    } catch (error) {
      console.error('Failed to load TruLens health:', error);
    }
  };

  const loadStats = async () => {
    try {
      const response = await fetch('/api/trulens/stats');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Failed to load TruLens stats:', error);
    }
  };

  const loadDashboardStatus = async () => {
    try {
      const response = await fetch('/api/trulens/dashboard/status');
      if (response.ok) {
        const data = await response.json();
        setDashboardStatus(data.status);
      }
    } catch (error) {
      console.error('Failed to load dashboard status:', error);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await loadData();
      await loadDashboardStatus();
    } finally {
      setRefreshing(false);
    }
  };

  const toggleMonitoring = async () => {
    try {
      const action = health?.monitoring_enabled ? 'disable' : 'enable';
      const response = await fetch('/api/trulens/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });

      if (response.ok) {
        await loadHealth();
      } else {
        const error = await response.json();
        alert(`Failed to ${action} monitoring: ${error.detail}`);
      }
    } catch (error) {
      console.error('Failed to toggle monitoring:', error);
      alert('Failed to toggle monitoring');
    }
  };

  const updateSamplingRate = async (newRate: number) => {
    try {
      const response = await fetch('/api/trulens/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sampling_rate: newRate / 100 }),
      });

      if (response.ok) {
        setSamplingRate(newRate);
      } else {
        const error = await response.json();
        alert(`Failed to update sampling rate: ${error.detail}`);
      }
    } catch (error) {
      console.error('Failed to update sampling rate:', error);
      alert('Failed to update sampling rate');
    }
  };

  const launchDashboard = async () => {
    try {
      const response = await fetch('/api/trulens/dashboard/launch', {
        method: 'POST',
      });

      if (response.ok) {
        const data = await response.json();
        window.open(data.url, '_blank');
        setDashboardStatus('running');
      } else {
        const error = await response.json();
        alert(`Failed to launch dashboard: ${error.detail}`);
      }
    } catch (error) {
      console.error('Failed to launch dashboard:', error);
      alert('Failed to launch dashboard');
    }
  };

  const stopDashboard = async () => {
    try {
      const response = await fetch('/api/trulens/dashboard/stop', {
        method: 'POST',
      });

      if (response.ok) {
        setDashboardStatus('stopped');
      } else {
        const error = await response.json();
        alert(`Failed to stop dashboard: ${error.detail}`);
      }
    } catch (error) {
      console.error('Failed to stop dashboard:', error);
      alert('Failed to stop dashboard');
    }
  };

  const resetDatabase = async () => {
    if (!confirm('Are you sure you want to reset the TruLens database? This will delete all monitoring records.')) {
      return;
    }

    try {
      const response = await fetch('/api/trulens/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'reset' }),
      });

      if (response.ok) {
        alert('Database reset successfully');
        await loadData();
      } else {
        const error = await response.json();
        alert(`Failed to reset database: ${error.detail}`);
      }
    } catch (error) {
      console.error('Failed to reset database:', error);
      alert('Failed to reset database');
    }
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2 text-gray-400" />
          <p className="text-sm text-gray-600 dark:text-gray-400">Loading TruLens data...</p>
        </div>
      </div>
    );
  }

  // Not installed state
  if (health?.status === 'unavailable') {
    return (
      <div className="h-full flex flex-col overflow-hidden">
        <div className="border-b p-6">
          <h2 className="text-2xl font-bold mb-2">TruLens Monitoring</h2>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Continuous quality monitoring with LLM-based feedback
          </p>
        </div>
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="max-w-md text-center">
            <AlertCircle className="w-16 h-16 mx-auto mb-4 text-yellow-500" />
            <h3 className="text-lg font-semibold mb-2">TruLens Not Installed</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              {health.message || 'TruLens monitoring is not installed. Install it to enable continuous quality monitoring.'}
            </p>
            <code className="block bg-gray-100 dark:bg-neutral-800 p-3 rounded text-sm">
              uv pip install -r evals/trulens/requirements-trulens.txt
            </code>
          </div>
        </div>
      </div>
    );
  }

  const healthIndicatorData = health
    ? {
      monitoring_enabled: health.monitoring_enabled,
      database_connected: health.database_status === 'healthy',
      last_check: new Date().toISOString(),
    }
    : null;

  return (
    <div className="h-full flex flex-col overflow-hidden" style={{ background: 'var(--bg-primary)' }}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: 'var(--border)' }}>
        <h3 className="font-medium" style={{ color: 'var(--text-secondary)' }}>TruLens Integration</h3>
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
          Refresh Data
        </button>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Section 1: Controls */}
        <section>
          <h3 className="text-lg font-semibold mb-4">Monitoring Controls</h3>
          <div className="space-y-4">
            {/* Enable/Disable Toggle */}
            <div className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-neutral-800">
              <div className="flex-1">
                <p className="font-medium">Enable Monitoring</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Monitor all queries with TruLens feedback functions
                </p>
              </div>
              <button
                onClick={toggleMonitoring}
                className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${health?.monitoring_enabled
                  ? 'bg-[#f27a03]'
                  : 'bg-gray-200 dark:bg-gray-700'
                  }`}
              >
                <span
                  className={`inline-block h-6 w-6 transform rounded-full bg-white transition-transform ${health?.monitoring_enabled ? 'translate-x-7' : 'translate-x-1'
                    }`}
                />
              </button>
            </div>

            {/* Health Indicator */}
            {healthIndicatorData && <HealthIndicator status={healthIndicatorData} />}

            {/* Sampling Rate Slider */}
            <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-neutral-800">
              <label className="block text-sm font-medium mb-3">
                Sampling Rate: {samplingRate}%
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={samplingRate}
                onChange={(e) => setSamplingRate(Number(e.target.value))}
                onMouseUp={(e) => updateSamplingRate(Number((e.target as HTMLInputElement).value))}
                onTouchEnd={(e) => updateSamplingRate(Number((e.target as HTMLInputElement).value))}
                className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-[#f27a03]"
              />
              <div className="flex justify-between text-xs text-gray-600 dark:text-gray-400 mt-1">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: Live Statistics */}
        {stats && stats.status === 'active' && (
          <section>
            <h3 className="text-lg font-semibold mb-4">Live Statistics</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <MetricsCard
                title="Answer Relevance"
                value={stats.avg_answer_relevance || 0}
                format="percentage"
                threshold={{ min: 0.7 }}
              />
              <MetricsCard
                title="Groundedness"
                value={stats.avg_groundedness || 0}
                format="percentage"
                threshold={{ min: 0.7 }}
              />
              <MetricsCard
                title="Context Relevance"
                value={stats.avg_context_relevance || 0}
                format="percentage"
                threshold={{ min: 0.7 }}
              />
              <MetricsCard
                title="Total Queries"
                value={stats.total_queries || 0}
                format="number"
              />
              <MetricsCard
                title="Avg Latency"
                value={stats.avg_latency_ms || 0}
                format="milliseconds"
                threshold={{ max: 5000 }}
              />
              <MetricsCard
                title="Error Rate"
                value={stats.error_rate || 0}
                format="percentage"
                threshold={{ max: 0.05 }}
              />
            </div>
          </section>
        )}

        {/* Section 3: Dashboard */}
        <section className="flex-1 flex flex-col min-h-0">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Dashboard</h3>
            <button
              onClick={() => window.open('http://localhost:8501', '_blank')}
              disabled={dashboardStatus !== 'running'}
              className="px-4 py-2 bg-[#f27a03] text-white rounded-lg font-medium hover:bg-[#d96d03] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <ExternalLink className="w-4 h-4" />
              Open Dashboard
            </button>
          </div>
        </section>

        {/* Section 4: Advanced Actions */}
        <section>
          <h3 className="text-lg font-semibold mb-4">Advanced</h3>
          <div className="space-y-3">
            <button
              onClick={resetDatabase}
              className="px-4 py-2 border border-red-300 dark:border-red-600 text-red-600 dark:text-red-400 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors flex items-center gap-2"
            >
              <Trash2 className="w-4 h-4" />
              Reset Database
            </button>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              Warning: This will permanently delete all monitoring records and feedback data.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
