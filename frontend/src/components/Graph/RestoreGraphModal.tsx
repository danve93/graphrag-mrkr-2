import React, { useState, useRef } from 'react';
import { AlertTriangle, X, Upload } from 'lucide-react';

interface RestoreGraphModalProps {
    onClose: () => void;
}

export const RestoreGraphModal: React.FC<RestoreGraphModalProps> = ({ onClose }) => {
    const [confirmation, setConfirmation] = useState('');
    const [file, setFile] = useState<File | null>(null);
    const [isRestoring, setIsRestoring] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleRestore = async () => {
        if (confirmation !== 'DELETE') return;
        if (!file) {
            setError('Please select a snapshot file.');
            return;
        }

        setIsRestoring(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('confirmation', 'DELETE');

            const token = localStorage.getItem('authToken');
            const response = await fetch('/api/graph/editor/restore', {
                method: 'POST',
                headers: {
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                },
                credentials: 'include',
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Restore failed');
            }

            setSuccess(true);
            setTimeout(() => {
                window.location.reload();
            }, 2000);

        } catch (e: any) {
            console.error('Restore failed', e);
            setError(e.message);
            setIsRestoring(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
            <div className="bg-[var(--bg-secondary)] border border-[var(--systemRed)]/50 rounded-xl shadow-2xl max-w-md w-full overflow-hidden">
                <div className="p-6">
                    <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center gap-3 text-[var(--systemRed)]">
                            <div className="p-2 bg-[var(--systemRed)]/10 rounded-lg">
                                <AlertTriangle className="w-6 h-6" />
                            </div>
                            <h2 className="text-xl font-bold font-display text-[var(--text-primary)]">Disaster Recovery</h2>
                        </div>
                        <button onClick={onClose} className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors">
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    {!success ? (
                        <div className="space-y-6">
                            <div className="bg-[var(--systemRed)]/10 border border-[var(--systemRed)]/30 rounded-lg p-4">
                                <p className="text-[var(--systemRed)] text-sm leading-relaxed">
                                    <strong className="font-bold block mb-1">WARNING: DATA LOSS IMMINENT</strong>
                                    This action will <span className="underline decoration-2 underline-offset-2">WIPE THE ENTIRE DATABASE</span> and replace it with the uploaded snapshot.
                                    This process cannot be undone.
                                </p>
                            </div>

                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">
                                        1. Upload Snapshot (JSON)
                                    </label>
                                    <div
                                        onClick={() => fileInputRef.current?.click()}
                                        className="border-2 border-dashed border-[var(--border)] hover:border-[var(--text-secondary)] rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer transition-colors bg-[var(--bg-tertiary)]"
                                    >
                                        <Upload className="w-8 h-8 text-[var(--text-secondary)] mb-2" />
                                        <span className="text-sm text-[var(--text-secondary)]">
                                            {file ? file.name : "Click to select snapshot.json"}
                                        </span>
                                        <input
                                            type="file"
                                            ref={fileInputRef}
                                            onChange={(e) => setFile(e.target.files?.[0] || null)}
                                            accept=".json"
                                            className="hidden"
                                        />
                                    </div>
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">
                                        2. Confirm Deletion
                                    </label>
                                    <input
                                        type="text"
                                        value={confirmation}
                                        onChange={(e) => setConfirmation(e.target.value)}
                                        placeholder="Type DELETE to confirm"
                                        className="input-field w-full font-mono"
                                    />
                                </div>
                            </div>

                            {error && (
                                <div className="text-[var(--systemRed)] text-sm bg-[var(--systemRed)]/10 p-2 rounded border border-[var(--systemRed)]/20">
                                    {error}
                                </div>
                            )}

                            <div className="flex gap-3 pt-2">
                                <button
                                    onClick={onClose}
                                    className="button-secondary flex-1"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={handleRestore}
                                    disabled={confirmation !== 'DELETE' || !file || isRestoring}
                                    className="flex-1 px-4 py-2 bg-[var(--systemRed)] hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-bold transition-colors shadow-lg shadow-red-900/20"
                                >
                                    {isRestoring ? 'Restoring...' : 'WIPE & RESTORE'}
                                </button>
                            </div>
                        </div>
                    ) : (
                        <div className="text-center py-6 space-y-4">
                            <div className="w-16 h-16 bg-[var(--systemGreen)]/10 rounded-full flex items-center justify-center mx-auto">
                                <Upload className="w-8 h-8 text-[var(--systemGreen)]" />
                            </div>
                            <h3 className="text-lg font-bold text-[var(--text-primary)]">Restore Successful</h3>
                            <p className="text-[var(--text-secondary)]">The page will reload in a moment...</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
