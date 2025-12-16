'use client';

import { useState, useEffect } from 'react';
import { Play, Loader2, CheckCircle, XCircle } from 'lucide-react';

interface Dataset {
  path: string;
  name: string;
  sample_count: number;
}

interface Variant {
  key: string;
  label: string;
  description: string;
}

interface BenchmarkRunnerProps {
  onComplete?: () => void;
}

export default function BenchmarkRunner({ onComplete }: BenchmarkRunnerProps) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [variants, setVariants] = useState<Variant[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [selectedVariants, setSelectedVariants] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<'idle' | 'running' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState<string>('');

  // Load datasets and variants on mount
  useEffect(() => {
    loadDatasets();
    loadVariants();
  }, []);

  const loadDatasets = async () => {
    try {
      const response = await fetch('/api/ragas/datasets');
      if (response.ok) {
        const data = await response.json();
        setDatasets(data.datasets || []);
        if (data.datasets?.length > 0) {
          setSelectedDataset(data.datasets[0].path);
        }
      }
    } catch (error) {
      console.error('Failed to load datasets:', error);
    }
  };

  const loadVariants = async () => {
    try {
      const response = await fetch('/api/ragas/variants');
      if (response.ok) {
        const data = await response.json();
        setVariants(data.variants || []);
      }
    } catch (error) {
      console.error('Failed to load variants:', error);
    }
  };

  const handleVariantToggle = (variantKey: string) => {
    setSelectedVariants(prev =>
      prev.includes(variantKey)
        ? prev.filter(v => v !== variantKey)
        : [...prev, variantKey]
    );
  };

  const runBenchmark = async () => {
    if (!selectedDataset || selectedVariants.length === 0) return;

    setIsRunning(true);
    setStatus('running');
    setProgress(0);
    setMessage('Starting benchmark...');

    try {
      // Start benchmark job
      const response = await fetch('/api/ragas/run-benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_path: selectedDataset,
          variants: selectedVariants,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to start benchmark');
      }

      const { job_id } = await response.json();

      // Poll job status
      const pollInterval = setInterval(async () => {
        try {
          const statusResponse = await fetch(`/api/ragas/job/${job_id}`);
          if (statusResponse.ok) {
            const statusData = await statusResponse.json();
            setProgress(statusData.progress || 0);

            if (statusData.status === 'completed') {
              clearInterval(pollInterval);
              setIsRunning(false);
              setStatus('success');
              setMessage('Benchmark completed successfully!');
              setProgress(100);
              if (onComplete) onComplete();
            } else if (statusData.status === 'failed') {
              clearInterval(pollInterval);
              setIsRunning(false);
              setStatus('error');
              setMessage(statusData.error || 'Benchmark failed');
            }
          }
        } catch (error) {
          console.error('Error polling job status:', error);
        }
      }, 2000);

      // Timeout after 10 minutes
      setTimeout(() => {
        clearInterval(pollInterval);
        if (isRunning) {
          setIsRunning(false);
          setStatus('error');
          setMessage('Benchmark timed out');
        }
      }, 600000);

    } catch (error) {
      setIsRunning(false);
      setStatus('error');
      setMessage(error instanceof Error ? error.message : 'Failed to run benchmark');
    }
  };

  return (
    <div className="space-y-4">
      {/* Dataset Selection */}
      <div>
        <label className="block text-sm font-medium mb-2">Dataset</label>
        <select
          value={selectedDataset}
          onChange={(e) => setSelectedDataset(e.target.value)}
          disabled={isRunning}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-neutral-800 disabled:opacity-50"
        >
          {datasets.map((dataset) => (
            <option key={dataset.path} value={dataset.path}>
              {dataset.name} ({dataset.sample_count} samples)
            </option>
          ))}
        </select>
      </div>

      {/* Variant Selection */}
      <div>
        <label className="block text-sm font-medium mb-2">Variants to Test</label>
        <div className="space-y-2">
          {variants.map((variant) => (
            <label
              key={variant.key}
              className="flex items-start gap-2 p-3 border border-gray-200 dark:border-gray-700 rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-neutral-800"
            >
              <input
                type="checkbox"
                checked={selectedVariants.includes(variant.key)}
                onChange={() => handleVariantToggle(variant.key)}
                disabled={isRunning}
                className="mt-1"
              />
              <div className="flex-1">
                <p className="font-medium text-sm">{variant.label}</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">{variant.description}</p>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Run Button */}
      <button
        onClick={runBenchmark}
        disabled={isRunning || selectedVariants.length === 0}
        className="w-full px-4 py-2 bg-[#f27a03] text-white rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#d96d03] transition-colors flex items-center justify-center gap-2"
      >
        {isRunning ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Running Benchmark...
          </>
        ) : (
          <>
            <Play className="w-4 h-4" />
            Run Benchmark
          </>
        )}
      </button>

      {/* Progress Bar */}
      {isRunning && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-400">{message}</span>
            <span className="font-medium">{progress.toFixed(0)}%</span>
          </div>
          <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-[#f27a03] transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Status Message */}
      {status !== 'idle' && !isRunning && (
        <div
          className={`flex items-center gap-2 p-3 rounded-lg ${
            status === 'success'
              ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400'
              : 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400'
          }`}
        >
          {status === 'success' ? (
            <CheckCircle className="w-5 h-5" />
          ) : (
            <XCircle className="w-5 h-5" />
          )}
          <span className="text-sm font-medium">{message}</span>
        </div>
      )}
    </div>
  );
}
