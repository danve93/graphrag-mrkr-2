'use client';

import { useState, useEffect, useCallback } from 'react';
import {
    Zap,
    RefreshCw,
    Database,
    TrendingUp,
    Clock,
    FileText,
    MessageSquare,
    Search,
    Download,
    DollarSign,
    CheckCircle,
    AlertTriangle,
    PieChart,
    BarChart3
} from 'lucide-react';
import MetricsCard from './shared/MetricsCard';
import QueryRecordsTable from './QueryRecordsTable';
import ConversationModal from '../shared/ConversationModal';

interface UsageSummary {
    total_calls: number;
    total_input_tokens: number;
    total_output_tokens: number;
    total_tokens: number;
    avg_latency_ms: number;
}

interface OperationStats {
    [key: string]: {
        calls: number;
        input: number;
        output: number;
    };
}

interface CostEstimate {
    total_cost_usd: number;
    total_cost_eur: number;
    usd_to_eur_rate: number;
    by_model: {
        [model: string]: {
            input_tokens: number;
            output_tokens: number;
            total_cost: number;
            total_cost_eur: number;
        };
    };
}

interface SuccessRate {
    total_calls: number;
    successful: number;
    failed: number;
    success_rate: number;
    recent_errors: Array<{
        timestamp: string;
        operation: string;
        model: string;
        error_message: string;
    }>;
}

interface TimeTrends {
    daily: Array<{
        date: string;
        calls: number;
        total_tokens: number;
    }>;
    hourly: Array<{
        hour: string;
        calls: number;
        total_tokens: number;
    }>;
}

interface Efficiency {
    io_ratio: number;
    avg_input_per_call: number;
    avg_output_per_call: number;
}

interface RecentRecord {
    id: number;
    timestamp: string;
    operation: string;
    provider: string;
    model: string;
    input_tokens: number;
    output_tokens: number;
    latency_ms: number | null;
}

interface ConversationUsage {
    conversation_id: string;
    calls: number;
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    cost_eur: number;
    first_call: string;
    last_call: string;
}

// Operation category mapping
const OPERATION_CATEGORIES: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
    'ingestion.classification': { icon: <FileText size={14} />, color: 'var(--systemOrange)', label: 'Classification' },
    'ingestion.summarization': { icon: <FileText size={14} />, color: 'var(--systemOrange)', label: 'Summarization' },
    'ingestion.entity_extraction': { icon: <FileText size={14} />, color: 'var(--systemOrange)', label: 'Entity Extraction' },
    'rag.generation': { icon: <MessageSquare size={14} />, color: 'var(--systemGreen)', label: 'RAG Generation' },
    'rag.query_routing': { icon: <Search size={14} />, color: 'var(--systemBlue)', label: 'Query Routing' },
    'rag.query_analysis': { icon: <Search size={14} />, color: 'var(--systemBlue)', label: 'Query Analysis' },
    'rag.query_expansion': { icon: <Search size={14} />, color: 'var(--systemBlue)', label: 'Query Expansion' },
    'rag.query_contextualization': { icon: <Search size={14} />, color: 'var(--systemBlue)', label: 'Contextualization' },
    'rag.follow_up': { icon: <Search size={14} />, color: 'var(--systemBlue)', label: 'Follow-up' },
    'rag.text_to_cypher': { icon: <Database size={14} />, color: 'var(--systemTeal)', label: 'Text-to-Cypher' },
    'rag.cypher_correction': { icon: <Database size={14} />, color: 'var(--systemTeal)', label: 'Cypher Correction' },
    'rag.structured_kg_entity_extraction': { icon: <Database size={14} />, color: 'var(--systemTeal)', label: 'KG Entity' },
    'rag.category_analysis': { icon: <FileText size={14} />, color: 'var(--systemPurple)', label: 'Category Analysis' },
    'rag.category_classification': { icon: <FileText size={14} />, color: 'var(--systemPurple)', label: 'Category Class' },
    'chat.title_generation': { icon: <MessageSquare size={14} />, color: 'var(--systemPurple)', label: 'Title Gen' },
    'chat.summarization': { icon: <MessageSquare size={14} />, color: 'var(--systemPurple)', label: 'Chat Summary' },
    'chat.preference_extraction': { icon: <MessageSquare size={14} />, color: 'var(--systemPurple)', label: 'Preferences' },
};

const getOperationInfo = (op: string) => {
    return OPERATION_CATEGORIES[op] || { icon: <Zap size={14} />, color: 'var(--systemGray)', label: op.split('.').pop() };
};

const formatNumber = (n: number | null | undefined): string => {
    if (n === null || n === undefined) return '0';
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
    return n.toLocaleString();
};

const formatCost = (n: number | null | undefined): string => {
    if (n === null || n === undefined) return '$0.00';
    if (n < 0.01) return `$${n.toFixed(4)}`;
    if (n < 1) return `$${n.toFixed(3)}`;
    return `$${n.toFixed(2)}`;
};

// Provider colors for pie chart
const PROVIDER_COLORS: Record<string, string> = {
    openai: '#10B981',
    anthropic: '#D97706',
    mistral: '#3B82F6',
    ollama: '#8B5CF6',
    lmstudio: '#EC4899',
};

export default function LLMUsageSubPanel() {
    const [summary, setSummary] = useState<UsageSummary | null>(null);
    const [byOperation, setByOperation] = useState<OperationStats | null>(null);
    const [byProvider, setByProvider] = useState<OperationStats | null>(null);
    const [byModel, setByModel] = useState<OperationStats | null>(null);
    const [costEstimate, setCostEstimate] = useState<CostEstimate | null>(null);
    const [successRate, setSuccessRate] = useState<SuccessRate | null>(null);
    const [timeTrends, setTimeTrends] = useState<TimeTrends | null>(null);
    const [efficiency, setEfficiency] = useState<Efficiency | null>(null);
    const [recentRecords, setRecentRecords] = useState<RecentRecord[]>([]);
    const [byConversation, setByConversation] = useState<ConversationUsage[]>([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null);

    const loadData = useCallback(async () => {
        try {
            const endpoints = [
                '/api/metrics/llm-usage/summary',
                '/api/metrics/llm-usage/by-operation',
                '/api/metrics/llm-usage/by-provider',
                '/api/metrics/llm-usage/by-model',
                '/api/metrics/llm-usage/cost-estimate',
                '/api/metrics/llm-usage/success-rate',
                '/api/metrics/llm-usage/time-trends',
                '/api/metrics/llm-usage/efficiency',
                '/api/metrics/llm-usage/recent?limit=20',
                '/api/metrics/llm-usage/by-conversation?limit=15',
            ];

            const results = await Promise.all(endpoints.map(url => fetch(url).then(r => r.ok ? r.json() : null).catch(() => null)));

            setSummary(results[0]);
            setByOperation(results[1]?.by_operation || results[1]);
            setByProvider(results[2]?.by_provider || results[2]);
            setByModel(results[3]?.by_model || results[3]);
            setCostEstimate(results[4]);
            setSuccessRate(results[5]);
            setTimeTrends(results[6]);
            setEfficiency(results[7]);
            setRecentRecords(results[8]?.records || []);
            setByConversation(results[9]?.conversations || []);
        } catch (error) {
            console.error('Failed to load LLM usage data:', error);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadData();
        const interval = setInterval(loadData, 30000);
        return () => clearInterval(interval);
    }, [loadData]);

    const handleRefresh = async () => {
        setRefreshing(true);
        await loadData();
        setRefreshing(false);
    };

    const exportData = async () => {
        try {
            const res = await fetch('/api/metrics/llm-usage/full-report');
            if (res.ok) {
                const data = await res.json();
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `llm-usage-report-${new Date().toISOString().split('T')[0]}.json`;
                a.click();
                URL.revokeObjectURL(url);
            }
        } catch (error) {
            console.error('Failed to export data:', error);
        }
    };

    if (loading) {
        return (
            <div className="h-full flex items-center justify-center" style={{ background: 'var(--bg-primary)' }}>
                <div className="text-center">
                    <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-3 text-gray-400" />
                    <p className="text-sm text-gray-600 dark:text-gray-400">Loading token usage data...</p>
                </div>
            </div>
        );
    }

    const sortedOperations = byOperation ? Object.entries(byOperation).sort((a, b) => b[1].calls - a[1].calls) : [];
    const maxOperationTokens = sortedOperations.length > 0 ? Math.max(...sortedOperations.map(([, v]) => v.input + v.output)) : 1;
    const providerEntries = byProvider ? Object.entries(byProvider) : [];
    const modelEntries = byModel ? Object.entries(byModel).sort((a, b) => (b[1].input + b[1].output) - (a[1].input + a[1].output)) : [];
    const totalProviderTokens = providerEntries.reduce((sum, [, v]) => sum + v.input + v.output, 0) || 1;

    return (
        <div className="h-full flex flex-col overflow-hidden" style={{ background: 'var(--bg-primary)' }}>
            {/* Toolbar */}
            <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: 'var(--border)' }}>
                <h3 className="font-medium" style={{ color: 'var(--text-secondary)' }}>LLM Token Usage</h3>
                <div className="flex items-center gap-2">
                    <button onClick={exportData} className="px-3 py-1.5 border rounded-md text-sm transition-colors flex items-center gap-2 hover:bg-gray-50 dark:hover:bg-neutral-800" style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}>
                        <Download size={14} /> Export
                    </button>
                    <button onClick={handleRefresh} disabled={refreshing} className="px-3 py-1.5 border rounded-md text-sm transition-colors disabled:opacity-50 flex items-center gap-2" style={{ borderColor: 'var(--border)', color: 'var(--text-primary)', background: 'var(--bg-secondary)' }}>
                        <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} /> Refresh
                    </button>
                </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6 pb-28 space-y-6">
                {/* Hero Row: Cost + Success + Totals */}
                <section>
                    <h3 className="text-sm font-semibold mb-4 text-gray-500 dark:text-gray-400 uppercase tracking-wider">Overview</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                        {/* Cost Card */}
                        <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-neutral-800">
                            <div className="flex items-center gap-2 mb-2">
                                <DollarSign size={16} className="text-green-600" />
                                <span className="text-xs font-medium text-gray-500 uppercase">Est. Cost</span>
                            </div>
                            <div className="text-2xl font-bold text-green-600">
                                €{(costEstimate?.total_cost_eur ?? 0).toFixed(2)}
                            </div>
                            <p className="text-xs text-gray-500 mt-1">
                                ${(costEstimate?.total_cost_usd ?? 0).toFixed(2)} USD
                            </p>
                        </div>

                        {/* Success Rate */}
                        <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-neutral-800">
                            <div className="flex items-center gap-2 mb-2">
                                {(successRate?.success_rate ?? 100) >= 95 ? (
                                    <CheckCircle size={16} className="text-green-600" />
                                ) : (
                                    <AlertTriangle size={16} className="text-yellow-500" />
                                )}
                                <span className="text-xs font-medium text-gray-500 uppercase">Success Rate</span>
                            </div>
                            <div className={`text-2xl font-bold ${(successRate?.success_rate ?? 100) >= 95 ? 'text-green-600' : 'text-yellow-500'}`}>
                                {successRate?.success_rate ?? 100}%
                            </div>
                            <p className="text-xs text-gray-500 mt-1">
                                {successRate?.successful ?? 0} OK / {successRate?.failed ?? 0} failed
                            </p>
                        </div>

                        {/* Total Tokens */}
                        <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-neutral-800">
                            <div className="flex items-center gap-2 mb-2">
                                <Zap size={16} style={{ color: 'var(--accent-primary)' }} />
                                <span className="text-xs font-medium text-gray-500 uppercase">Total Tokens</span>
                            </div>
                            <div className="text-2xl font-bold" style={{ color: 'var(--accent-primary)' }}>
                                {formatNumber(summary?.total_tokens)}
                            </div>
                            <p className="text-xs text-gray-500 mt-1">
                                {formatNumber(summary?.total_input_tokens)} in / {formatNumber(summary?.total_output_tokens)} out
                            </p>
                        </div>

                        {/* Efficiency */}
                        <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-neutral-800">
                            <div className="flex items-center gap-2 mb-2">
                                <BarChart3 size={16} className="text-blue-500" />
                                <span className="text-xs font-medium text-gray-500 uppercase">Efficiency</span>
                            </div>
                            <div className="text-2xl font-bold text-blue-500">
                                {efficiency?.io_ratio?.toFixed(2) ?? '0.00'}x
                            </div>
                            <p className="text-xs text-gray-500 mt-1">
                                Output/input ratio ({efficiency?.avg_output_per_call?.toFixed(0) ?? 0} avg out)
                            </p>
                        </div>
                    </div>
                </section>

                {/* Provider & Model Breakdown */}
                <section>
                    <h3 className="text-sm font-semibold mb-4 text-gray-500 dark:text-gray-400 uppercase tracking-wider">Provider & Model Breakdown</h3>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        {/* Provider Breakdown */}
                        <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-neutral-800">
                            <div className="flex items-center gap-2 mb-3">
                                <PieChart size={16} className="text-gray-500" />
                                <span className="text-sm font-medium">By Provider</span>
                            </div>
                            {providerEntries.length === 0 ? (
                                <p className="text-sm text-gray-500 italic">No data yet</p>
                            ) : (
                                <div className="space-y-2">
                                    {providerEntries.map(([provider, stats]) => {
                                        const pct = ((stats.input + stats.output) / totalProviderTokens * 100).toFixed(1);
                                        const color = PROVIDER_COLORS[provider] || 'var(--systemGray)';
                                        return (
                                            <div key={provider} className="flex items-center gap-3">
                                                <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                                                <span className="flex-1 text-sm capitalize">{provider}</span>
                                                <span className="text-xs text-gray-500">{stats.calls} calls</span>
                                                <span className="text-xs font-mono font-medium w-12 text-right">{pct}%</span>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>

                        {/* Model Breakdown */}
                        <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-neutral-800">
                            <div className="flex items-center gap-2 mb-3">
                                <Database size={16} className="text-gray-500" />
                                <span className="text-sm font-medium">By Model</span>
                            </div>
                            {modelEntries.length === 0 ? (
                                <p className="text-sm text-gray-500 italic">No data yet</p>
                            ) : (
                                <div className="space-y-2 max-h-40 overflow-y-auto">
                                    {modelEntries.slice(0, 5).map(([model, stats]) => (
                                        <div key={model} className="flex items-center justify-between">
                                            <span className="text-sm font-mono truncate flex-1">{model}</span>
                                            <div className="flex items-center gap-3 ml-2">
                                                <span className="text-xs text-gray-500">{stats.calls} calls</span>
                                                <span className="text-xs font-mono">{formatNumber(stats.input + stats.output)} tok</span>
                                                {costEstimate?.by_model?.[model] && (
                                                    <span className="text-xs text-green-600">€{costEstimate.by_model[model].total_cost_eur?.toFixed(3)}</span>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </section>

                {/* Time Trends */}
                {timeTrends && timeTrends.daily && timeTrends.daily.length > 0 && (
                    <section>
                        <h3 className="text-sm font-semibold mb-4 text-gray-500 dark:text-gray-400 uppercase tracking-wider">Usage Trend (Last 7 Days)</h3>
                        <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-neutral-800">
                            <div className="flex items-end gap-1 h-24">
                                {timeTrends.daily.map((day, i) => {
                                    const maxTokens = Math.max(...timeTrends.daily.map(d => d.total_tokens)) || 1;
                                    const height = (day.total_tokens / maxTokens) * 100;
                                    return (
                                        <div key={i} className="flex-1 flex flex-col items-center h-full justify-end">
                                            <div
                                                className="w-full rounded-t transition-all"
                                                style={{
                                                    height: `${Math.max(height, 4)}%`,
                                                    backgroundColor: 'var(--accent-primary)',
                                                    opacity: 0.7 + (i / timeTrends.daily.length) * 0.3
                                                }}
                                                title={`${day.date}: ${formatNumber(day.total_tokens)} tokens, ${day.calls} calls`}
                                            />
                                        </div>
                                    );
                                })}
                            </div>
                            <div className="flex justify-between mt-2 text-xs text-gray-500">
                                <span>{timeTrends.daily[0]?.date}</span>
                                <span>{timeTrends.daily[timeTrends.daily.length - 1]?.date}</span>
                            </div>
                        </div>
                    </section>
                )}

                {/* By Conversation */}
                {byConversation.length > 0 && (
                    <section>
                        <h3 className="text-sm font-semibold mb-4 text-gray-500 dark:text-gray-400 uppercase tracking-wider">By Conversation</h3>
                        <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-neutral-800">
                            <div className="space-y-2 max-h-64 overflow-y-auto">
                                {byConversation.map((conv, i) => {
                                    // Color code by cost: green < €0.10, yellow < €1, red > €1
                                    const costColor = conv.cost_eur < 0.1
                                        ? 'var(--systemGreen)'
                                        : conv.cost_eur < 1
                                            ? 'var(--systemOrange)'
                                            : 'var(--systemRed)';
                                    return (
                                        <div
                                            key={conv.conversation_id || i}
                                            onClick={() => conv.conversation_id && setSelectedConversationId(conv.conversation_id)}
                                            className="flex items-center justify-between py-2 px-2 hover:bg-gray-50 dark:hover:bg-neutral-800 rounded-md cursor-pointer transition-colors border-b border-gray-100 dark:border-gray-700 last:border-0"
                                        >
                                            <div className="flex items-center gap-2 flex-1 min-w-0">
                                                <MessageSquare size={14} className="text-gray-400 flex-shrink-0" />
                                                <span className="text-xs font-mono truncate" title={conv.conversation_id}>
                                                    {conv.conversation_id?.slice(0, 12)}...
                                                </span>
                                            </div>
                                            <div className="flex items-center gap-4 text-xs">
                                                <span className="text-gray-500">{conv.calls} calls</span>
                                                <span className="font-mono">{formatNumber(conv.total_tokens)} tok</span>
                                                <span className="font-medium w-16 text-right" style={{ color: costColor }}>
                                                    €{conv.cost_eur.toFixed(3)}
                                                </span>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    </section>
                )}

                <ConversationModal
                    isOpen={!!selectedConversationId}
                    onClose={() => setSelectedConversationId(null)}
                    sessionId={selectedConversationId}
                />

                {/* Operations Breakdown */}
                <section>
                    <h3 className="text-sm font-semibold mb-4 text-gray-500 dark:text-gray-400 uppercase tracking-wider">Usage by Operation</h3>
                    <div className="space-y-2">
                        {sortedOperations.length === 0 ? (
                            <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-neutral-800">
                                <p className="text-sm text-gray-600 dark:text-gray-400 italic">No operations tracked yet.</p>
                            </div>
                        ) : (
                            sortedOperations.slice(0, 10).map(([operation, stats]) => {
                                const info = getOperationInfo(operation);
                                const totalTokens = stats.input + stats.output;
                                const barWidth = (totalTokens / maxOperationTokens) * 100;
                                return (
                                    <div key={operation} className="relative overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-neutral-800">
                                        <div className="absolute inset-y-0 left-0 opacity-10" style={{ width: `${Math.max(barWidth, 2)}%`, backgroundColor: info.color }} />
                                        <div className="relative flex items-center justify-between px-4 py-3">
                                            <div className="flex items-center gap-3">
                                                <span style={{ color: info.color }}>{info.icon}</span>
                                                <span className="text-sm font-medium">{info.label}</span>
                                                <span className="text-xs px-2 py-0.5 rounded bg-gray-100 dark:bg-neutral-700 text-gray-600 dark:text-gray-400">{stats.calls} calls</span>
                                            </div>
                                            <div className="flex items-center gap-4 text-xs font-mono text-gray-600 dark:text-gray-400">
                                                <span>{formatNumber(stats.input)} in</span>
                                                <span>{formatNumber(stats.output)} out</span>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })
                        )}
                    </div>
                </section>

                {/* Recent Errors (if any) */}
                {successRate && successRate.recent_errors && successRate.recent_errors.length > 0 && (
                    <section>
                        <h3 className="text-sm font-semibold mb-4 text-gray-500 dark:text-gray-400 uppercase tracking-wider flex items-center gap-2">
                            <AlertTriangle size={14} className="text-red-500" /> Recent Errors
                        </h3>
                        <div className="border border-red-200 dark:border-red-800 rounded-lg bg-red-50 dark:bg-red-900/20 divide-y divide-red-200 dark:divide-red-800">
                            {successRate.recent_errors.slice(0, 5).map((err, i) => (
                                <div key={i} className="px-4 py-2">
                                    <div className="flex items-center justify-between">
                                        <span className="text-sm font-medium text-red-700 dark:text-red-400">{err.operation}</span>
                                        <span className="text-xs text-red-500">{new Date(err.timestamp).toLocaleString()}</span>
                                    </div>
                                    <p className="text-xs text-red-600 dark:text-red-400 mt-1 truncate">{err.error_message || 'Unknown error'}</p>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {/* Query Records - Trulens-style table */}
                <section>
                    <h3 className="text-sm font-semibold mb-4 text-gray-500 dark:text-gray-400 uppercase tracking-wider">Query Records</h3>
                    <QueryRecordsTable records={recentRecords as any} />
                </section>
            </div>
        </div>
    );
}
