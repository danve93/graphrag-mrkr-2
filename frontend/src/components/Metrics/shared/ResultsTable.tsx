'use client';

interface VariantMetrics {
  context_precision?: number;
  context_recall?: number;
  faithfulness?: number;
  answer_relevancy?: number;
}

interface VariantResult {
  variant: string;
  metrics: VariantMetrics;
  evaluation_count?: number;
  timestamp?: string;
}

interface VariantData {
  metrics: VariantMetrics;
  evaluation_count?: number;
  last_evaluation_timestamp?: number;
}

interface ResultsTableProps {
  data: Record<string, VariantData> | VariantResult[];
}

export default function ResultsTable({ data }: ResultsTableProps) {
  // Normalize data to array format
  const results: VariantResult[] = Array.isArray(data)
    ? data
    : Object.entries(data).map(([key, value]) => ({
        variant: key,
        ...value,
      }));

  if (results.length === 0) {
    return (
      <div className="text-center p-8 text-gray-600 dark:text-gray-400">
        No results available
      </div>
    );
  }

  const formatScore = (score: number | undefined) => {
    if (score === undefined) return '-';
    return (score * 100).toFixed(1) + '%';
  };

  const getScoreColor = (score: number | undefined, threshold: number) => {
    if (score === undefined) return 'text-gray-600 dark:text-gray-400';
    if (score >= threshold) return 'text-green-600 dark:text-green-400';
    if (score >= threshold - 0.1) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const calculateAverage = (metrics: VariantResult['metrics']) => {
    const scores = [
      metrics.context_precision,
      metrics.context_recall,
      metrics.faithfulness,
      metrics.answer_relevancy,
    ].filter((s): s is number => s !== undefined);

    if (scores.length === 0) return undefined;
    return scores.reduce((a, b) => a + b, 0) / scores.length;
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b border-gray-200 dark:border-gray-700">
            <th className="text-left p-3 text-sm font-semibold">Variant</th>
            <th className="text-center p-3 text-sm font-semibold">Context Precision</th>
            <th className="text-center p-3 text-sm font-semibold">Context Recall</th>
            <th className="text-center p-3 text-sm font-semibold">Faithfulness</th>
            <th className="text-center p-3 text-sm font-semibold">Answer Relevancy</th>
            <th className="text-center p-3 text-sm font-semibold">Average</th>
          </tr>
        </thead>
        <tbody>
          {results.map((result, index) => {
            const avg = calculateAverage(result.metrics);
            return (
              <tr
                key={result.variant || index}
                className="border-b border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-neutral-800"
              >
                <td className="p-3 font-medium">{result.variant}</td>
                <td className={`p-3 text-center ${getScoreColor(result.metrics.context_precision, 0.65)}`}>
                  {formatScore(result.metrics.context_precision)}
                </td>
                <td className={`p-3 text-center ${getScoreColor(result.metrics.context_recall, 0.60)}`}>
                  {formatScore(result.metrics.context_recall)}
                </td>
                <td className={`p-3 text-center ${getScoreColor(result.metrics.faithfulness, 0.85)}`}>
                  {formatScore(result.metrics.faithfulness)}
                </td>
                <td className={`p-3 text-center ${getScoreColor(result.metrics.answer_relevancy, 0.80)}`}>
                  {formatScore(result.metrics.answer_relevancy)}
                </td>
                <td className={`p-3 text-center font-semibold ${getScoreColor(avg, 0.70)}`}>
                  {formatScore(avg)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
