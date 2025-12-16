import React from 'react';
import { AlertTriangle, X } from 'lucide-react';

interface ConfirmActionModalProps {
    title: string;
    message: string;
    onConfirm: () => void;
    onCancel: () => void;
    confirmText?: string;
    isDangerous?: boolean;
}

export const ConfirmActionModal: React.FC<ConfirmActionModalProps> = ({
    title,
    message,
    onConfirm,
    onCancel,
    confirmText = "Confirm",
    isDangerous = false
}) => {
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
            <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl shadow-2xl max-w-sm w-full overflow-hidden">
                <div className="p-6">
                    <div className="flex items-center justify-between mb-4">
                        <div className={`flex items-center gap-3 ${isDangerous ? 'text-[var(--systemRed)]' : 'text-[var(--text-primary)]'}`}>
                            {isDangerous && <AlertTriangle className="w-5 h-5" />}
                            <h2 className="text-lg font-bold font-display">{title}</h2>
                        </div>
                        <button onClick={onCancel} className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors">
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    <p className="text-[var(--text-secondary)] text-sm mb-6 leading-relaxed">
                        {message}
                    </p>

                    <div className="flex gap-3">
                        <button
                            onClick={onCancel}
                            className="button-secondary flex-1 justify-center"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={onConfirm}
                            className={`flex-1 px-4 py-2 text-white rounded-md font-medium transition-colors shadow-lg ${isDangerous
                                ? 'bg-[var(--systemRed)] hover:bg-red-600'
                                : 'button-primary justify-center'
                                }`}
                        >
                            {confirmText}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};
