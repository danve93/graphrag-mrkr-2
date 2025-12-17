import React, { useState } from 'react';
import { useGraphEditorStore } from './useGraphEditorStore';
import { Download, Upload, MousePointer, Link2, Scissors } from 'lucide-react';
import { RestoreGraphModal } from './RestoreGraphModal';

interface GraphToolbarProps {
    onFit: () => void;
}

export const GraphToolbar: React.FC<GraphToolbarProps> = ({ onFit }) => {
    const [isRestoreModalOpen, setIsRestoreModalOpen] = useState(false);

    // Safety: Graph Editor Store might be used later for other tools
    const { mode, setMode } = useGraphEditorStore();

    const handleBackup = async () => {
        // Trigger download of the JSON snapshot directly from the API
        window.open('/api/graph/editor/snapshot', '_blank');
    };

    return (
        <div className="absolute top-4 left-4 flex flex-col gap-2 z-10">
            {/* Editor Tools */}
            <div className="bg-slate-800/90 backdrop-blur border border-slate-700 p-2 rounded-lg shadow-xl flex flex-col gap-1">
                <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-1 mb-1">
                    Tools
                </div>

                <button
                    onClick={() => setMode('select')}
                    className={`flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-md transition-colors text-left ${mode === 'select'
                        ? 'bg-blue-600/20 text-blue-200 border border-blue-600/50'
                        : 'text-slate-200 hover:bg-slate-700/50 border border-transparent'
                        }`}
                    title="Select & Explore Mode"
                >
                    <MousePointer className={`w-4 h-4 ${mode === 'select' ? 'text-blue-400' : 'text-slate-400'}`} />
                    <span>Select</span>
                </button>

                <button
                    onClick={() => setMode('connect')}
                    className={`flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-md transition-colors text-left ${mode === 'connect'
                        ? 'bg-purple-600/20 text-purple-200 border border-purple-600/50'
                        : 'text-slate-200 hover:bg-slate-700/50 border border-transparent'
                        }`}
                    title="Connect / Draw Edges"
                >
                    <Link2 className={`w-4 h-4 ${mode === 'connect' ? 'text-purple-400' : 'text-slate-400'}`} />
                    <span>Connect</span>
                </button>

                <button
                    onClick={() => setMode('prune')}
                    className={`flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-md transition-colors text-left ${mode === 'prune'
                            ? 'bg-red-600/20 text-red-200 border border-red-600/50'
                            : 'text-slate-200 hover:bg-slate-700/50 border border-transparent'
                        }`}
                    title="Prune / Delete Items"
                >
                    <Scissors className={`w-4 h-4 ${mode === 'prune' ? 'text-red-400' : 'text-slate-400'}`} />
                    <span>Prune</span>
                </button>
            </div>

            {/* Safety Tools */}
            <div className="bg-slate-800/90 backdrop-blur border border-slate-700 p-2 rounded-lg shadow-xl flex flex-col gap-1">
                <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-1 mb-1">
                    Safety
                </div>

                <button
                    onClick={handleBackup}
                    className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-slate-200 hover:bg-slate-700/50 rounded-md transition-colors text-left"
                    title="Download Graph Snapshot (JSON)"
                >
                    <Download className="w-4 h-4 text-emerald-400" />
                    <span>Backup</span>
                </button>

                <button
                    onClick={() => setIsRestoreModalOpen(true)}
                    className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-slate-200 hover:bg-red-900/20 rounded-md transition-colors text-left"
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
