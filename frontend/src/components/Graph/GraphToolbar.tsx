import React, { useState } from 'react';
import { useGraphEditorStore } from './useGraphEditorStore';
import { Download, Upload, MousePointer, Link2, Scissors, Wand2, Ghost, Loader2 } from 'lucide-react';
import { RestoreGraphModal } from './RestoreGraphModal';

interface GraphToolbarProps {
    onFit: () => void;
}

export const GraphToolbar: React.FC<GraphToolbarProps> = ({ onFit }) => {
    const [isRestoreModalOpen, setIsRestoreModalOpen] = useState(false);
    const [isBackingUp, setIsBackingUp] = useState(false);

    const { mode, setMode } = useGraphEditorStore();

    const handleBackup = async () => {
        try {
            setIsBackingUp(true);
            const token = localStorage.getItem('authToken');

            const response = await fetch('/api/graph/editor/snapshot', {
                method: 'GET',
                headers: {
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                },
                credentials: 'include',
            });

            if (!response.ok) throw new Error('Backup failed');

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `graph-snapshot-${new Date().toISOString().slice(0, 10)}.json`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error('Backup failed:', error);
        } finally {
            setIsBackingUp(false);
        }
    };

    const toolButtonClass = (isActive: boolean, activeColorClass: string, activeTextClass: string) => `
        flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-md transition-colors text-left w-full
        ${isActive
            ? `bg-[var(--bg-tertiary)] border border-[var(--accent-primary)] ${activeTextClass}`
            : 'text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] border border-transparent hover:text-[var(--text-primary)]'
        }
    `;

    return (
        <div className="absolute top-4 left-4 flex flex-col gap-2 z-10 pointer-events-auto">
            {/* Editor Tools */}
            <div className="bg-[var(--bg-secondary)] border border-[var(--border)] p-2 rounded-lg shadow-xl flex flex-col gap-1 w-36">
                <div className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider px-1 mb-1">
                    Tools
                </div>

                <button
                    onClick={() => setMode('select')}
                    className={toolButtonClass(mode === 'select', '', 'text-blue-400')}
                    title="Select & Explore Mode"
                >
                    <MousePointer className={`w-4 h-4 ${mode === 'select' ? 'text-blue-400' : 'text-[var(--text-secondary)]'}`} />
                    <span>Select</span>
                </button>

                <button
                    onClick={() => setMode('connect')}
                    className={toolButtonClass(mode === 'connect', '', 'text-purple-400')}
                    title="Connect / Draw Edges"
                >
                    <Link2 className={`w-4 h-4 ${mode === 'connect' ? 'text-purple-400' : 'text-[var(--text-secondary)]'}`} />
                    <span>Connect</span>
                </button>

                <button
                    onClick={() => setMode('prune')}
                    className={toolButtonClass(mode === 'prune', '', 'text-red-400')}
                    title="Prune / Delete Items"
                >
                    <Scissors className={`w-4 h-4 ${mode === 'prune' ? 'text-red-400' : 'text-[var(--text-secondary)]'}`} />
                    <span>Prune</span>
                </button>

                <button
                    onClick={() => setMode('heal')}
                    className={toolButtonClass(mode === 'heal', '', 'text-amber-400')}
                    title="Heal / Suggest Connections (AI)"
                >
                    <Wand2 className={`w-4 h-4 ${mode === 'heal' ? 'text-amber-400' : 'text-[var(--text-secondary)]'}`} />
                    <span>Heal</span>
                </button>

                <button
                    onClick={() => setMode(mode === 'orphan' ? 'select' : 'orphan')}
                    className={toolButtonClass(mode === 'orphan', '', 'text-cyan-400')}
                    title="Toggle Orphan Nodes Highlight"
                >
                    <Ghost className={`w-4 h-4 ${mode === 'orphan' ? 'text-cyan-400' : 'text-[var(--text-secondary)]'}`} />
                    <span>Orphans</span>
                </button>
            </div>

            {/* Safety Tools */}
            <div className="bg-[var(--bg-secondary)] border border-[var(--border)] p-2 rounded-lg shadow-xl flex flex-col gap-1 w-36">
                <div className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider px-1 mb-1">
                    Safety
                </div>

                <button
                    onClick={handleBackup}
                    disabled={isBackingUp}
                    className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded-md transition-colors text-left disabled:opacity-50 disabled:cursor-wait w-full"
                    title="Download Graph Snapshot (JSON)"
                >
                    {isBackingUp ? (
                        <Loader2 className="w-4 h-4 text-emerald-400 animate-spin" />
                    ) : (
                        <Download className="w-4 h-4 text-emerald-400" />
                    )}
                    <span>{isBackingUp ? 'Backing...' : 'Backup'}</span>
                </button>

                <button
                    onClick={() => setIsRestoreModalOpen(true)}
                    className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-[var(--text-secondary)] hover:text-red-400 hover:bg-red-500/10 rounded-md transition-colors text-left w-full"
                    title="Restore Graph from Snapshot"
                >
                    <Upload className="w-4 h-4 text-red-400" />
                    <span>Restore</span>
                </button>
            </div>

            {isRestoreModalOpen && (
                <RestoreGraphModal onClose={() => setIsRestoreModalOpen(false)} />
            )}
        </div>
    );
};
