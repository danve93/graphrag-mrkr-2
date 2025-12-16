'use client';

interface MetricsCardProps {
  title: string;
  value: number;
  format: 'number' | 'percentage' | 'milliseconds';
  trend?: 'up' | 'down' | 'neutral';
  threshold?: { min?: number; max?: number };
}

export default function MetricsCard({ title, value, format, trend, threshold }: MetricsCardProps) {
  const formatValue = () => {
    switch (format) {
      case 'percentage':
        return `${(value * 100).toFixed(1)}%`;
      case 'milliseconds':
        return `${value.toFixed(0)}ms`;
      default:
        return value.toLocaleString();
    }
  };

  const getColor = () => {
    if (!threshold) return 'text-gray-900 dark:text-gray-100';
    if (threshold.min && value < threshold.min) return 'text-red-600 dark:text-red-400';
    if (threshold.max && value > threshold.max) return 'text-red-600 dark:text-red-400';
    return 'text-green-600 dark:text-green-400';
  };

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 bg-white dark:bg-neutral-800">
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">{title}</p>
      <div className="flex items-end gap-2">
        <p className={`text-2xl font-bold ${getColor()}`}>{formatValue()}</p>
        {trend && (
          <span className={`text-xs mb-1 ${trend === 'up' ? 'text-green-600' : trend === 'down' ? 'text-red-600' : 'text-gray-600'}`}>
            {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→'}
          </span>
        )}
      </div>
    </div>
  );
}
