import React, { useState, useEffect } from 'react';
import { X, Wand2, Link2, Check, AlertCircle, Loader2 } from 'lucide-react';
import { useGraphEditorStore } from './useGraphEditorStore';
import { API_URL } from '@/lib/api';

interface Suggestion {
    id: string;
    name: string;
    description: string;
    type: string;
    score: number;
}

interface HealingSuggestionsModalProps {
    nodeId: string;
    onClose: () => void;
}

export const HealingSuggestionsModal: React.FC<HealingSuggestionsModalProps> = ({ nodeId, onClose }) => {
    const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [connectingId, setConnectingId] = useState<string | null>(null);
    const [connectedIds, setConnectedIds] = useState<Set<string>>(new Set());

    useEffect(() => {
        const fetchSuggestions = async () => {
            setLoading(true);
            setError(null);
            try {
                const token = localStorage.getItem('authToken');
                const response = await fetch(`${API_URL}/api/graph/editor/heal`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                    },
                    credentials: 'include',
                    body: JSON.stringify({ node_id: nodeId }),
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Failed to fetch suggestions');
                }

                const data = await response.json();
                setSuggestions(data.suggestions || []);
            } catch (e: any) {
                console.error('Healing fetch failed', e);
                setError(e.message);
            } finally {
                setLoading(false);
            }
        };

        if (nodeId) {
            fetchSuggestions();
        }
    }, [nodeId]);

    const handleConnect = async (targetId: string) => {
        setConnectingId(targetId);
        try {
            const token = localStorage.getItem('authToken');
            const response = await fetch(`${API_URL}/api/graph/editor/edge`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                },
                credentials: 'include',
                body: JSON.stringify({
                    source_id: nodeId,
                    target_id: targetId,
                    relation_type: 'RELATED_TO',
                    properties: {
                        source: 'ai_healing',
                        confidence: suggestions.find(s => s.id === targetId)?.score || 0
                    }
                }),
            });

            if (!response.ok) throw new Error('Failed to create connection');

            setConnectedIds(prev => new Set(prev).add(targetId));

        } catch (e: any) {
            console.error('Connection failed', e);
        } finally {
            setConnectingId(null);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
            <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl shadow-2xl max-w-lg w-full overflow-hidden flex flex-col max-h-[85vh]">
                {/* Header */}
                <div className="p-5 border-b border-[var(--border)] flex items-center justify-between bg-[var(--bg-secondary)]">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-[var(--bg-tertiary)] rounded-lg text-accent-primary">
                            <Wand2 className="w-5 h-5 text-[var(--accent-primary)]" />
                        </div>
                        <div>
                            <h2 className="text-lg font-bold text-[var(--text-primary)]">AI Graph Healing</h2>
                            <p className="text-xs text-[var(--text-secondary)]">Suggestions for Node <span className="font-mono text-accent-primary">{nodeId}</span></p>
                        </div>
                    </div>
                    <button onClick={onClose} className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors p-1 hover:bg-[var(--bg-tertiary)] rounded-md">
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-5 space-y-4 bg-[var(--bg-primary)]">
                    {loading ? (
                        <div className="flex flex-col items-center justify-center py-10 text-[var(--text-secondary)]">
                            <Loader2 className="w-8 h-8 animate-spin mb-3 text-accent-primary" />
                            <p>Analyzing semantic connections...</p>
                        </div>
                    ) : error ? (
                        <div className="p-4 bg-[var(--systemRed)]/10 border border-[var(--systemRed)]/30 rounded-lg flex gap-3 text-[var(--systemRed)]">
                            <AlertCircle className="w-5 h-5 shrink-0" />
                            <p className="text-sm">{error}</p>
                        </div>
                    ) : suggestions.length === 0 ? (
                        <div className="text-center py-10 text-[var(--text-secondary)]">
                            <p>No high-confidence suggestions found for this node.</p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {suggestions.map((suggestion) => {
                                const isConnected = connectedIds.has(suggestion.id);
                                const isConnecting = connectingId === suggestion.id;

                                return (
                                    <div
                                        key={suggestion.id}
                                        className="bg-[var(--bg-secondary)] border border-[var(--border)] hover:border-accent-primary/50 rounded-lg p-3 transition-colors group card"
                                    >
                                        <div className="flex items-start justify-between gap-3">
                                            <div className="flex-1 min-w-0">
                                                <div className="flex items-center gap-2 mb-1">
                                                    <h4 className="text-sm font-medium text-[var(--text-primary)] truncate">{suggestion.name}</h4>
                                                    <span className="text-[10px] uppercase font-bold tracking-wider text-[var(--text-secondary)] bg-[var(--bg-tertiary)] px-1.5 py-0.5 rounded">
                                                        {suggestion.type}
                                                    </span>
                                                    <span className="text-[10px] font-mono text-[var(--systemGreen)] bg-[var(--systemGreen)]/10 px-1.5 py-0.5 rounded">
                                                        {(suggestion.score * 100).toFixed(0)}% Match
                                                    </span>
                                                </div>
                                                <p className="text-xs text-[var(--text-secondary)] line-clamp-2 leading-relaxed">
                                                    {suggestion.description}
                                                </p>
                                            </div>

                                            <button
                                                onClick={() => handleConnect(suggestion.id)}
                                                disabled={isConnected || isConnecting}
                                                className={`shrink-0 p-2 rounded-lg transition-all ${isConnected
                                                    ? 'bg-[var(--systemGreen)]/10 text-[var(--systemGreen)] cursor-default'
                                                    : 'bg-[var(--bg-tertiary)] hover:bg-accent-primary hover:text-white text-[var(--text-secondary)]'
                                                    }`}
                                                title={isConnected ? "Connected" : "Connect Edge"}
                                            >
                                                {isConnecting ? (
                                                    <Loader2 className="w-4 h-4 animate-spin" />
                                                ) : isConnected ? (
                                                    <Check className="w-4 h-4" />
                                                ) : (
                                                    <Link2 className="w-4 h-4" />
                                                )}
                                            </button>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
