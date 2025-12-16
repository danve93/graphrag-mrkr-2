
import React, { useEffect, useState } from "react";
import axios from "axios";
import { API_URL } from "../../lib/api";

export const FeedbackMetricsDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await axios.get(`${API_URL || ''}/api/feedback/metrics`, { withCredentials: true });
        setMetrics(res.data);
      } catch (err) {
        setMetrics(null);
      } finally {
        setLoading(false);
      }
    };
    fetchMetrics();
  }, []);

  if (loading) {
    return (
      <div className="w-full flex items-center justify-center py-8">
        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600" />
        <span className="ml-3 text-secondary-500">Loading feedback metrics...</span>
      </div>
    );
  }
  if (!metrics) {
    return (
      <div className="bg-white dark:bg-secondary-800 rounded-lg border border-secondary-200 dark:border-secondary-700 p-6 text-center">
        <span className="text-red-500 font-semibold">Failed to load feedback metrics.</span>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-secondary-800 rounded-lg border border-secondary-200 dark:border-secondary-700 p-6">
      <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100 mb-4">
        Feedback Metrics
      </h2>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
        <div>
          <div className="text-xs text-secondary-500">Total Feedback</div>
          <div className="text-xl font-bold text-secondary-900 dark:text-secondary-100">{metrics.total_feedback}</div>
        </div>
        <div>
          <div className="text-xs text-secondary-500">Positive</div>
          <div className="text-xl font-bold text-green-600">{metrics.positive_feedback}</div>
        </div>
        <div>
          <div className="text-xs text-secondary-500">Negative</div>
          <div className="text-xl font-bold text-red-500">{metrics.negative_feedback}</div>
        </div>
        <div>
          <div className="text-xs text-secondary-500">Accuracy</div>
          <div className="text-lg font-semibold">{metrics.accuracy?.toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-xs text-secondary-500">Convergence</div>
          <div className="text-lg font-semibold">{metrics.convergence?.toFixed(3)}</div>
        </div>
      </div>
      <div className="mt-4">
        <div className="text-xs text-secondary-500 mb-1">Current Weights</div>
        <div className="flex flex-wrap gap-4">
          <div className="bg-secondary-100 dark:bg-secondary-700 rounded px-3 py-2">
            <span className="font-semibold">Chunk:</span> {metrics.current_weights?.chunk_weight?.toFixed(2)}
          </div>
          <div className="bg-secondary-100 dark:bg-secondary-700 rounded px-3 py-2">
            <span className="font-semibold">Entity:</span> {metrics.current_weights?.entity_weight?.toFixed(2)}
          </div>
          <div className="bg-secondary-100 dark:bg-secondary-700 rounded px-3 py-2">
            <span className="font-semibold">Path:</span> {metrics.current_weights?.path_weight?.toFixed(2)}
          </div>
        </div>
      </div>
    </div>
  );
};
