'use client';

import { useEffect, useState } from 'react';
import { Activity, AlertCircle, ExternalLink, CheckCircle2, Server, Globe } from 'lucide-react';

export default function OpenTelemetrySubPanel() {
  const [isEnabled, setIsEnabled] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/health`);
        const data = await response.json();
        setIsEnabled(data.opentelemetry_enabled);
      } catch (error) {
        console.error('Failed to check OTEL status:', error);
        setIsEnabled(false);
      } finally {
        setLoading(false);
      }
    };
    checkStatus();
  }, []);

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center p-6" style={{ background: 'var(--bg-primary)' }}>
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col overflow-hidden" style={{ background: 'var(--bg-primary)' }}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: 'var(--border)' }}>
        <h3 className="font-medium" style={{ color: 'var(--text-secondary)' }}>Distributed Tracing</h3>
        <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${isEnabled
          ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
          : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'
          }`}>
          <div className={`w-2 h-2 rounded-full ${isEnabled ? 'bg-green-500' : 'bg-gray-400'}`} />
          {isEnabled ? 'Active' : 'Disabled'}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 pb-28 space-y-6">

        {/* Status Card */}
        <div className={`border rounded-lg p-6 ${isEnabled
          ? 'bg-green-50/50 dark:bg-green-900/10 border-green-200 dark:border-green-800'
          : 'bg-gray-50 dark:bg-neutral-800/50 border-gray-200 dark:border-gray-700'
          }`}>
          <div className="flex items-start gap-3">
            {isEnabled ? (
              <CheckCircle2 className="text-green-600 dark:text-green-400 mt-1 flex-shrink-0" size={20} />
            ) : (
              <AlertCircle className="text-gray-500 dark:text-gray-400 mt-1 flex-shrink-0" size={20} />
            )}
            <div>
              <h3 className={`font-semibold mb-1 ${isEnabled ? 'text-green-900 dark:text-green-100' : 'text-gray-900 dark:text-gray-100'
                }`}>
                {isEnabled ? 'OpenTelemetry Tracing Enabled' : 'Tracing is Disabled'}
              </h3>
              <p className={`text-sm ${isEnabled ? 'text-green-800 dark:text-green-200' : 'text-gray-600 dark:text-gray-400'
                }`}>
                {isEnabled
                  ? 'Application is currently emitting OTLP traces to the configured collector.'
                  : 'Enable tracing by setting ENABLE_OPENTELEMETRY=1 in your environment configurations.'}
              </p>
            </div>
          </div>
        </div>

        {/* Configuration View */}
        <section>
          <h3 className="text-sm font-semibold mb-4 text-gray-500 dark:text-gray-400 uppercase tracking-wider">Configuration</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-neutral-800">
              <div className="flex items-center gap-2 mb-2">
                <Globe className="w-4 h-4 text-gray-500" />
                <label className="text-xs font-medium text-gray-500">OTLP Endpoint</label>
              </div>
              <div className="font-mono text-sm">http://tempo:4317</div>
            </div>

            <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-neutral-800">
              <div className="flex items-center gap-2 mb-2">
                <Server className="w-4 h-4 text-gray-500" />
                <label className="text-xs font-medium text-gray-500">Service Name</label>
              </div>
              <div className="font-mono text-sm">amber-graphrag</div>
            </div>

            <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-neutral-800">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-4 h-4 text-gray-500" />
                <label className="text-xs font-medium text-gray-500">Integration</label>
              </div>
              <div className="text-sm">GraphRAG + RAGAS + TruLens</div>
            </div>

            <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-neutral-800">
              <div className="flex items-center gap-2 mb-2">
                <ExternalLink className="w-4 h-4 text-gray-500" />
                <label className="text-xs font-medium text-gray-500">Backend</label>
              </div>
              <div className="text-sm">Grafana Tempo</div>
            </div>
          </div>
        </section>

        {/* Observability Links */}
        <section>
          <h3 className="text-sm font-semibold mb-4 text-gray-500 dark:text-gray-400 uppercase tracking-wider">Access Traces</h3>
          <div className="bg-white dark:bg-neutral-800 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-neutral-800/50">
              <p className="text-sm text-gray-600 dark:text-gray-300">
                To visualize traces, connect Grafana to the Tempo datasource at <code className="text-xs bg-gray-100 dark:bg-neutral-700 px-1 py-0.5 rounded">http://tempo:3200</code> (HTTP) or inspect raw traces directly.
              </p>
            </div>

            <a
              href="#"
              className="flex items-center justify-between p-4 hover:bg-gray-50 dark:hover:bg-neutral-700 transition-colors cursor-not-allowed opacity-60"
              title="Grafana UI not detected"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded bg-orange-100 flex items-center justify-center">
                  <Activity className="w-5 h-5 text-orange-600" />
                </div>
                <div>
                  <div className="font-medium">Grafana Dashboard</div>
                  <div className="text-xs text-gray-500">Visualization UI (Coming Soon)</div>
                </div>
              </div>
              <ExternalLink className="w-4 h-4 text-gray-400" />
            </a>
          </div>
        </section>

      </div>
    </div>
  );
}
