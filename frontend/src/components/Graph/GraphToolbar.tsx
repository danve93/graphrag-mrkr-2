import React, { useState } from 'react';
import { useGraphEditorStore } from './useGraphEditorStore';
import { Download, Upload, MousePointer, Link2, Scissors, Wand2 } from 'lucide-react';
import { RestoreGraphModal } from './RestoreGraphModal';
import { API_URL } from '@/lib/api';

export const GraphToolbar: React.FC = () => {
    const [isRestoreModalOpen, setIsRestoreModalOpen] = useState(false);

    // Safety: Graph Editor Store might be used later for other tools
    const { mode, setMode } = useGraphEditorStore();

    const handleBackup = async () => {
        try {
            const token = localStorage.getItem('authToken');
            const response = await fetch(`${API_URL}/api/graph/editor/snapshot`, {
                method: 'GET',
                headers: {
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                },
                credentials: 'include',
            });

            if (!response.ok) {
                alert('Backup failed: ' + (await response.text()));
                return;
            }

            const data = await response.json();
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `graph-backup-${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (e) {
            console.error('Backup failed:', e);
            alert('Backup failed: ' + (e instanceof Error ? e.message : 'Unknown error'));
        }
    };

    const toolButtonClass = (isActive: boolean, activeTextClass: string) => `
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
                    className={toolButtonClass(mode === 'select', 'text-blue-400')}
                    title="Select & Explore Mode"
                >
                    <MousePointer className={`w-4 h-4 ${mode === 'select' ? 'text-blue-400' : 'text-[var(--text-secondary)]'}`} />
                    <span>Select</span>
                </button>

                <button
                    onClick={() => setMode('connect')}
                    className={toolButtonClass(mode === 'connect', 'text-purple-400')}
                    title="Connect / Draw Edges"
                >
                    <Link2 className={`w-4 h-4 ${mode === 'connect' ? 'text-purple-400' : 'text-[var(--text-secondary)]'}`} />
                    <span>Connect</span>
                </button>

                <button
                    onClick={() => setMode('heal')}
                    className={toolButtonClass(mode === 'heal', 'text-emerald-400')}
                    title="Heal / Suggest Connections (AI)"
                >
                    <Wand2 className={`w-4 h-4 ${mode === 'heal' ? 'text-emerald-400' : 'text-[var(--text-secondary)]'}`} />
                    <span>Heal</span>
                </button>

                <button
                    onClick={() => setMode('prune')}
                    className={toolButtonClass(mode === 'prune', 'text-red-400')}
                    title="Prune / Delete Items"
                >
                    <Scissors className={`w-4 h-4 ${mode === 'prune' ? 'text-red-400' : 'text-[var(--text-secondary)]'}`} />
                    <span>Prune</span>
                </button>
            </div>

            {/* Safety Tools */}
            <div className="bg-[var(--bg-secondary)] border border-[var(--border)] p-2 rounded-lg shadow-xl flex flex-col gap-1 w-36">
                <div className="text-xs font-semibold text-[var(--text-secondary)] uppercase tracking-wider px-1 mb-1">
                    Safety
                </div>

                <button
                    onClick={handleBackup}
                    className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded-md transition-colors text-left w-full"
                    title="Download Graph Snapshot (JSON)"
                >
                    <Download className="w-4 h-4 text-emerald-400" />
                    <span>Backup</span>
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
