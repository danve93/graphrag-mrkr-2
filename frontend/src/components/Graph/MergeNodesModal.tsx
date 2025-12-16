import React, { useState } from 'react';
import { GitMerge, ArrowRight, AlertTriangle, Loader2 } from 'lucide-react';

interface MergeNodesModalProps {
    selectedNodes: any[];
    onClose: () => void;
    onMergeSuccess: () => void;
}

export const MergeNodesModal: React.FC<MergeNodesModalProps> = ({ selectedNodes, onClose, onMergeSuccess }) => {
    const [targetId, setTargetId] = useState<string>(selectedNodes[0]?.id || '');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const sourceNodes = selectedNodes.filter(n => n.id !== targetId);
    const targetNode = selectedNodes.find(n => n.id === targetId);

    const handleMerge = async () => {
        if (!targetId || sourceNodes.length === 0) return;

        setLoading(true);
        setError(null);

        try {
            const token = localStorage.getItem('authToken');
            const response = await fetch('/api/graph/editor/nodes/merge', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                },
                credentials: 'include',
                body: JSON.stringify({
                    target_id: targetId,
                    source_ids: sourceNodes.map(n => n.id)
                }),
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Merge failed');
            }

            onMergeSuccess();
            onClose();
        } catch (e: any) {
            console.error("Merge failed", e);
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
            <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl shadow-2xl max-w-lg w-full overflow-hidden flex flex-col">
                <div className="p-5 border-b border-[var(--border)] flex items-center gap-3 bg-[var(--bg-secondary)]">
                    <div className="p-2 bg-[var(--bg-tertiary)] rounded-lg">
                        <GitMerge className="w-5 h-5 text-[var(--accent-primary)]" />
                    </div>
                    <div>
                        <h2 className="text-lg font-bold text-[var(--text-primary)]">Merge Nodes</h2>
                        <p className="text-xs text-[var(--text-secondary)]">Combine {selectedNodes.length} nodes into one</p>
                    </div>
                </div>

                <div className="p-5 space-y-4 bg-[var(--bg-primary)]">
                    <div className="bg-[var(--systemRed)]/10 border border-[var(--systemRed)]/20 p-3 rounded-lg flex gap-3">
                        <AlertTriangle className="w-5 h-5 text-[var(--systemRed)] shrink-0" />
                        <p className="text-xs text-[var(--text-primary)] leading-relaxed">
                            This action acts as a &quot;hard merge&quot;. All edges from source nodes will be moved to the target node.
                            Source nodes will be permanently deleted.
                        </p>
                    </div>

                    <div className="space-y-2">
                        <label className="text-sm font-medium text-[var(--text-secondary)]">Select Primary Node (Target)</label>
                        <div className="space-y-2 max-h-48 overflow-y-auto pr-2">
                            {selectedNodes.map(node => (
                                <button
                                    key={node.id}
                                    onClick={() => setTargetId(node.id)}
                                    className={`w-full text-left p-3 rounded-lg border transition-all flex items-center justify-between group ${targetId === node.id
                                        ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] shadow-sm'
                                        : 'bg-[var(--bg-tertiary)] border-[var(--border)] hover:border-[var(--text-secondary)]'
                                        }`}
                                >
                                    <div className="min-w-0">
                                        <div className="font-medium text-[var(--text-primary)] truncate">{node.label}</div>
                                        <div className="text-xs text-[var(--text-secondary)] font-mono truncate">{node.id}</div>
                                    </div>
                                    {targetId === node.id && (
                                        <div className="w-2 h-2 rounded-full bg-[var(--accent-primary)] shadow-sm" />
                                    )}
                                </button>
                            ))}
                        </div>
                    </div>

                    {error && (
                        <div className="text-[var(--systemRed)] text-sm bg-[var(--systemRed)]/10 p-2 rounded border border-[var(--systemRed)]/20">
                            {error}
                        </div>
                    )}
                </div>

                <div className="p-5 border-t border-[var(--border)] bg-[var(--bg-secondary)] flex justify-end gap-3">
                    <button
                        onClick={onClose}
                        className="button-secondary"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleMerge}
                        disabled={loading}
                        className="button-primary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {loading && <Loader2 className="w-4 h-4 animate-spin" />}
                        Merge into {targetNode?.label || 'Target'}
                    </button>
                </div>
            </div>
        </div>
    );
};
