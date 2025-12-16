'use client';

interface HealthIndicatorProps {
  status: {
    monitoring_enabled: boolean;
    database_connected: boolean;
    last_check: string;
  } | null;
}

export default function HealthIndicator({ status }: HealthIndicatorProps) {
  if (!status) {
    return (
      <div className="flex items-center gap-2 p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
        <div className="w-3 h-3 rounded-full bg-gray-400"></div>
        <span className="text-sm text-gray-600 dark:text-gray-400">Loading...</span>
      </div>
    );
  }

  const isHealthy = status.monitoring_enabled && status.database_connected;

  return (
    <div className="flex items-center gap-2 p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
      <div className={`w-3 h-3 rounded-full ${isHealthy ? 'bg-green-500' : 'bg-red-500'}`}></div>
      <div className="flex-1">
        <p className="text-sm font-medium">
          {isHealthy ? 'Healthy' : 'Unhealthy'}
        </p>
        <p className="text-xs text-gray-600 dark:text-gray-400">
          Last check: {new Date(status.last_check).toLocaleString()}
        </p>
      </div>
    </div>
  );
}
