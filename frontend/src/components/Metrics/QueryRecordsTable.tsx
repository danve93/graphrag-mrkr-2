'use client';

import { useState, useMemo } from 'react';
import {
    ChevronUp,
    ChevronDown,
    ChevronsUpDown,
    Filter,
    X,
    ChevronLeft,
    ChevronRight,
    FileText,
    MessageSquare,
    Search,
    Database,
    Zap,
} from 'lucide-react';

// Operation category mapping
const OPERATION_CATEGORIES: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
    'ingestion.classification': { icon: <FileText size={12} />, color: 'var(--systemOrange)', label: 'Classification' },
    'ingestion.summarization': { icon: <FileText size={12} />, color: 'var(--systemOrange)', label: 'Summarization' },
    'ingestion.entity_extraction': { icon: <FileText size={12} />, color: 'var(--systemOrange)', label: 'Entity Extraction' },
    'rag.generation': { icon: <MessageSquare size={12} />, color: 'var(--systemGreen)', label: 'RAG Generation' },
    'rag.query_routing': { icon: <Search size={12} />, color: 'var(--systemBlue)', label: 'Query Routing' },
    'rag.query_analysis': { icon: <Search size={12} />, color: 'var(--systemBlue)', label: 'Query Analysis' },
    'rag.query_expansion': { icon: <Search size={12} />, color: 'var(--systemBlue)', label: 'Query Expansion' },
    'rag.follow_up': { icon: <Search size={12} />, color: 'var(--systemBlue)', label: 'Follow-up' },
    'rag.text_to_cypher': { icon: <Database size={12} />, color: 'var(--systemTeal)', label: 'Text-to-Cypher' },
    'rag.cypher_correction': { icon: <Database size={12} />, color: 'var(--systemTeal)', label: 'Cypher Correction' },
    'chat.title_generation': { icon: <MessageSquare size={12} />, color: 'var(--systemPurple)', label: 'Title Gen' },
    'chat.summarization': { icon: <MessageSquare size={12} />, color: 'var(--systemPurple)', label: 'Chat Summary' },
};

const getOperationInfo = (op: string) => {
    return OPERATION_CATEGORIES[op] || { icon: <Zap size={12} />, color: 'var(--systemGray)', label: op.split('.').pop() };
};

const formatNumber = (n: number | null | undefined): string => {
    if (n === null || n === undefined) return '0';
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
    return n.toLocaleString();
};

interface QueryRecord {
    id: number;
    timestamp: string;
    operation: string;
    provider: string;
    model: string;
    input_tokens: number;
    output_tokens: number;
    latency_ms: number | null;
    conversation_id?: string;
    document_id?: string;
    success?: number;
    error_message?: string;
}

interface ColumnDef {
    key: string;
    label: string;
    sortable: boolean;
    filterable: boolean;
    hidden?: boolean;
    metadata?: boolean;
}

const CORE_COLUMNS: ColumnDef[] = [
    { key: 'timestamp', label: 'Time', sortable: true, filterable: false },
    { key: 'operation', label: 'Operation', sortable: true, filterable: true },
    { key: 'total_tokens', label: 'Tokens', sortable: true, filterable: false },
    { key: 'cost_eur', label: 'Cost (EUR)', sortable: true, filterable: false },
    { key: 'latency_ms', label: 'Latency', sortable: true, filterable: false },
];

const METADATA_COLUMNS: ColumnDef[] = [
    { key: 'model', label: 'LLM Model', sortable: true, filterable: true, metadata: true },
    { key: 'provider', label: 'Provider', sortable: true, filterable: true, metadata: true },
    { key: 'conversation_id', label: 'Session', sortable: true, filterable: false, metadata: true },
    { key: 'success', label: 'Status', sortable: true, filterable: true, metadata: true },
];

type SortDirection = 'asc' | 'desc' | null;

interface QueryRecordsTableProps {
    records: QueryRecord[];
}

export default function QueryRecordsTable({ records }: QueryRecordsTableProps) {
    const [sortColumn, setSortColumn] = useState<string>('timestamp');
    const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
    const [visibleMetadata, setVisibleMetadata] = useState<Set<string>>(new Set(['model', 'conversation_id']));
    const [filters, setFilters] = useState<Record<string, string>>({});
    const [currentPage, setCurrentPage] = useState(1);
    const pageSize = 15;

    // Toggle metadata column visibility
    const toggleMetadata = (key: string) => {
        setVisibleMetadata(prev => {
            const next = new Set(prev);
            if (next.has(key)) {
                next.delete(key);
            } else {
                next.add(key);
            }
            return next;
        });
    };

    // Handle sort
    const handleSort = (column: string) => {
        if (sortColumn === column) {
            setSortDirection(prev => prev === 'asc' ? 'desc' : prev === 'desc' ? null : 'asc');
        } else {
            setSortColumn(column);
            setSortDirection('desc');
        }
    };

    // Get unique values for filterable columns
    const getUniqueValues = (columnKey: string): string[] => {
        const values = new Set<string>();
        records.forEach(r => {
            const val = (r as any)[columnKey];
            if (val) values.add(String(val));
        });
        return Array.from(values).sort();
    };

    // Process records with computed fields
    const processedRecords = useMemo(() => {
        return records.map(r => ({
            ...r,
            total_tokens: r.input_tokens + r.output_tokens,
            cost_eur: ((r.input_tokens * 0.15 + r.output_tokens * 0.60) / 1_000_000) * 0.92,
        }));
    }, [records]);

    // Filter records
    const filteredRecords = useMemo(() => {
        return processedRecords.filter(record => {
            return Object.entries(filters).every(([key, value]) => {
                if (!value) return true;
                const recordValue = String((record as any)[key] || '').toLowerCase();
                return recordValue.includes(value.toLowerCase());
            });
        });
    }, [processedRecords, filters]);

    // Sort records
    const sortedRecords = useMemo(() => {
        if (!sortDirection || !sortColumn) return filteredRecords;

        return [...filteredRecords].sort((a, b) => {
            let aVal = (a as any)[sortColumn];
            let bVal = (b as any)[sortColumn];

            // Handle nulls
            if (aVal === null || aVal === undefined) return 1;
            if (bVal === null || bVal === undefined) return -1;

            // Handle dates
            if (sortColumn === 'timestamp') {
                aVal = new Date(aVal).getTime();
                bVal = new Date(bVal).getTime();
            }

            // Compare
            if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
            if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
            return 0;
        });
    }, [filteredRecords, sortColumn, sortDirection]);

    // Paginate
    const totalPages = Math.ceil(sortedRecords.length / pageSize);
    const paginatedRecords = sortedRecords.slice((currentPage - 1) * pageSize, currentPage * pageSize);

    // Get visible columns
    const visibleColumns = [
        ...CORE_COLUMNS,
        ...METADATA_COLUMNS.filter(c => visibleMetadata.has(c.key)),
    ];

    // Render sort icon
    const SortIcon = ({ column }: { column: string }) => {
        if (sortColumn !== column) return <ChevronsUpDown size={14} className="opacity-30" />;
        if (sortDirection === 'asc') return <ChevronUp size={14} />;
        if (sortDirection === 'desc') return <ChevronDown size={14} />;
        return <ChevronsUpDown size={14} className="opacity-30" />;
    };

    return (
        <div className="space-y-4">
            {/* Metadata Column Toggles */}
            <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Display Columns:
                </span>
                {METADATA_COLUMNS.map(col => (
                    <button
                        key={col.key}
                        onClick={() => toggleMetadata(col.key)}
                        className={`px-2 py-1 text-xs rounded-md border transition-all ${visibleMetadata.has(col.key)
                            ? 'bg-orange-500/20 border-orange-500/50 text-orange-600 dark:text-orange-400'
                            : 'bg-transparent border-gray-300 dark:border-gray-600 text-gray-500 hover:border-gray-400'
                            }`}
                    >
                        {col.label}
                        {visibleMetadata.has(col.key) && <X size={10} className="inline ml-1" />}
                    </button>
                ))}
            </div>

            {/* Active Filters */}
            {Object.entries(filters).some(([, v]) => v) && (
                <div className="flex items-center gap-2 text-xs">
                    <Filter size={12} className="text-gray-400" />
                    {Object.entries(filters).map(([key, value]) => value && (
                        <span key={key} className="flex items-center gap-1 px-2 py-0.5 bg-blue-500/20 text-blue-600 dark:text-blue-400 rounded">
                            {key}: {value}
                            <X size={10} className="cursor-pointer" onClick={() => setFilters(prev => ({ ...prev, [key]: '' }))} />
                        </span>
                    ))}
                    <button
                        onClick={() => setFilters({})}
                        className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                    >
                        Clear all
                    </button>
                </div>
            )}

            {/* Table */}
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                        <thead className="bg-neutral-900/80 border-b border-gray-700">
                            <tr>
                                {visibleColumns.map(col => (
                                    <th
                                        key={col.key}
                                        className={`px-3 py-2.5 text-left font-medium text-gray-400 whitespace-nowrap ${col.sortable ? 'cursor-pointer hover:text-gray-200 select-none' : ''
                                            }`}
                                        onClick={() => col.sortable && handleSort(col.key)}
                                    >
                                        <div className="flex items-center gap-1">
                                            {col.label}
                                            {col.sortable && <SortIcon column={col.key} />}
                                        </div>
                                        {col.filterable && (
                                            <select
                                                className="mt-1 w-full bg-neutral-800 border border-gray-600 rounded text-xs py-0.5 px-1"
                                                value={filters[col.key] || ''}
                                                onChange={(e) => setFilters(prev => ({ ...prev, [col.key]: e.target.value }))}
                                                onClick={(e) => e.stopPropagation()}
                                            >
                                                <option value="">All</option>
                                                {getUniqueValues(col.key).map(v => (
                                                    <option key={v} value={v}>{v}</option>
                                                ))}
                                            </select>
                                        )}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-700/50 bg-neutral-800/50">
                            {paginatedRecords.length === 0 ? (
                                <tr>
                                    <td colSpan={visibleColumns.length} className="px-4 py-8 text-center text-gray-500">
                                        No records found
                                    </td>
                                </tr>
                            ) : (
                                paginatedRecords.map((record) => {
                                    const info = getOperationInfo(record.operation);
                                    const time = new Date(record.timestamp);
                                    const costColor = record.cost_eur < 0.001 ? 'var(--systemGreen)' : record.cost_eur < 0.01 ? 'var(--systemOrange)' : 'var(--systemRed)';

                                    return (
                                        <tr key={record.id} className="hover:bg-neutral-700/30 transition-colors">
                                            {/* Time */}
                                            <td className="px-3 py-2 whitespace-nowrap text-gray-400">
                                                <div>{time.toLocaleDateString('en-GB')}</div>
                                                <div className="text-gray-500">{time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</div>
                                            </td>

                                            {/* Operation */}
                                            <td className="px-3 py-2">
                                                <div className="flex items-center gap-2">
                                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: info.color }} />
                                                    <span className="text-gray-200">{info.label}</span>
                                                </div>
                                            </td>

                                            {/* Tokens */}
                                            <td className="px-3 py-2 text-right font-mono">
                                                <span className="text-gray-200">{formatNumber(record.total_tokens)}</span>
                                                <div className="text-gray-500">{formatNumber(record.input_tokens)} / {formatNumber(record.output_tokens)}</div>
                                            </td>

                                            {/* Cost */}
                                            <td className="px-3 py-2 text-right font-mono" style={{ color: costColor }}>
                                                €{record.cost_eur.toFixed(4)}
                                            </td>

                                            {/* Latency */}
                                            <td className="px-3 py-2 text-right text-gray-400">
                                                {record.latency_ms ? `${record.latency_ms}ms` : '-'}
                                            </td>

                                            {/* Metadata columns */}
                                            {visibleMetadata.has('model') && (
                                                <td className="px-3 py-2 font-mono text-gray-400 text-[10px]">
                                                    {record.model}
                                                </td>
                                            )}
                                            {visibleMetadata.has('provider') && (
                                                <td className="px-3 py-2 text-gray-400">
                                                    {record.provider}
                                                </td>
                                            )}
                                            {visibleMetadata.has('conversation_id') && (
                                                <td className="px-3 py-2 font-mono text-gray-500" title={record.conversation_id || ''}>
                                                    {record.conversation_id ? record.conversation_id.slice(0, 8) + '...' : '-'}
                                                </td>
                                            )}
                                            {visibleMetadata.has('success') && (
                                                <td className="px-3 py-2">
                                                    <span className={`px-1.5 py-0.5 rounded text-[10px] ${record.success !== 0
                                                        ? 'bg-green-500/20 text-green-400'
                                                        : 'bg-red-500/20 text-red-400'
                                                        }`}>
                                                        {record.success !== 0 ? 'OK' : 'ERR'}
                                                    </span>
                                                </td>
                                            )}
                                        </tr>
                                    );
                                })
                            )}
                        </tbody>
                    </table>
                </div>

                {/* Pagination */}
                <div className="flex items-center justify-between px-4 py-2 border-t border-gray-700 bg-neutral-900/50">
                    <div className="text-xs text-gray-500">
                        {sortedRecords.length > 0 ? (
                            <>
                                {(currentPage - 1) * pageSize + 1} to {Math.min(currentPage * pageSize, sortedRecords.length)} of {sortedRecords.length}
                            </>
                        ) : (
                            'No records'
                        )}
                    </div>
                    <div className="flex items-center gap-1">
                        <button
                            onClick={() => setCurrentPage(1)}
                            disabled={currentPage === 1}
                            className="p-1 rounded hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed flex items-center"
                            title="First page"
                        >
                            <span className="inline-flex">«</span>
                        </button>
                        <button
                            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                            disabled={currentPage === 1}
                            className="p-1 rounded hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed"
                            title="Previous page"
                        >
                            <ChevronLeft size={14} />
                        </button>
                        <span className="px-2 text-xs text-gray-400">
                            Page {currentPage} of {totalPages || 1}
                        </span>
                        <button
                            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                            disabled={currentPage >= totalPages}
                            className="p-1 rounded hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed"
                            title="Next page"
                        >
                            <ChevronRight size={14} />
                        </button>
                        <button
                            onClick={() => setCurrentPage(totalPages)}
                            disabled={currentPage >= totalPages}
                            className="p-1 rounded hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed flex items-center"
                            title="Last page"
                        >
                            <span className="inline-flex">»</span>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
