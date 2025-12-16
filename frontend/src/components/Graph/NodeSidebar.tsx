import React, { useState } from 'react';
import { X, Edit2, Trash2, MessageSquare, FileText, Check, XCircle } from 'lucide-react';
import type { GraphNode } from '@/types/graph';

interface NodeSidebarProps {
    node: GraphNode;
    onClose: () => void;
    onEdit: (node: GraphNode) => void;
    onDelete: (nodeId: string) => void;
    onChat: (node: GraphNode) => void;
}

export default function NodeSidebar({ node, onClose, onEdit, onDelete, onChat }: NodeSidebarProps) {
    const [isEditing, setIsEditing] = useState(false);
    const [editedDescription, setEditedDescription] = useState(node.description || '');

    const handleSaveEdit = async () => {
        try {
            const token = localStorage.getItem('authToken');
            const response = await fetch('/api/graph/editor/node', {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                },
                credentials: 'include',
                body: JSON.stringify({
                    node_id: node.id,
                    updates: { description: editedDescription }
                }),
            });
            if (response.ok) {
                setIsEditing(false);
                // Trigger refresh via onEdit
                onEdit({ ...node, description: editedDescription });
            }
        } catch (e) {
            console.error('Failed to save edit', e);
        }
    };

    return (
        <div className="absolute top-0 right-0 h-full w-80 bg-[var(--bg-secondary)] border-l border-[var(--border)] shadow-xl z-20 flex flex-col transition-transform duration-300">
            {/* Header */}
            <div className="p-4 border-b border-[var(--border)] flex items-start justify-between bg-[var(--bg-secondary)]">
                <div className="flex items-center gap-3">
                    <div
                        className="w-10 h-10 rounded-lg flex items-center justify-center text-white font-bold text-lg shadow-sm"
                        style={{ backgroundColor: node.color || 'var(--accent-primary)' }}
                    >
                        {node.label.charAt(0).toUpperCase()}
                    </div>
                    <div>
                        <h2 className="font-bold text-[var(--text-primary)] line-clamp-1 break-all" title={node.label}>
                            {node.label}
                        </h2>
                        <div className="flex items-center gap-2 mt-1">
                            <span className="text-[10px] uppercase font-bold text-[var(--text-secondary)] bg-[var(--bg-tertiary)] px-1.5 py-0.5 rounded">
                                {node.type || 'Entity'}
                            </span>
                            <span className="text-xs text-[var(--text-secondary)] font-mono">ID: {node.id.slice(0, 6)}...</span>
                        </div>
                    </div>
                </div>
                <button
                    onClick={onClose}
                    className="p-1 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] transition-colors"
                >
                    <X size={20} />
                </button>
            </div>

            {/* Actions Toolbar */}
            <div className="p-3 grid grid-cols-3 gap-2 border-b border-[var(--border)] bg-[var(--bg-secondary)]">
                <button
                    onClick={() => onChat(node)}
                    className="flex flex-col items-center justify-center gap-1 py-2 px-1 rounded-md hover:bg-[var(--bg-tertiary)] text-[var(--text-primary)] transition-colors border border-transparent hover:border-[var(--border)]"
                >
                    <MessageSquare size={16} className="text-accent-primary" />
                    <span className="text-[10px] font-medium">Chat</span>
                </button>
                <button
                    onClick={() => setIsEditing(true)}
                    className="flex flex-col items-center justify-center gap-1 py-2 px-1 rounded-md hover:bg-[var(--bg-tertiary)] text-[var(--text-primary)] transition-colors border border-transparent hover:border-[var(--border)]"
                >
                    <Edit2 size={16} />
                    <span className="text-[10px] font-medium">Edit</span>
                </button>
                <button
                    onClick={() => onDelete(node.id)}
                    className="flex flex-col items-center justify-center gap-1 py-2 px-1 rounded-md hover:bg-red-500/10 text-red-500 transition-colors border border-transparent hover:border-red-500/20"
                >
                    <Trash2 size={16} />
                    <span className="text-[10px] font-medium">Delete</span>
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4 space-y-6">

                {/* Description (with inline edit) */}
                <div className="space-y-2">
                    <div className="flex items-center justify-between">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)]">Description</h3>
                        {isEditing && (
                            <div className="flex gap-1">
                                <button onClick={handleSaveEdit} className="p-1 text-green-500 hover:bg-green-500/10 rounded">
                                    <Check size={16} />
                                </button>
                                <button onClick={() => { setIsEditing(false); setEditedDescription(node.description || ''); }} className="p-1 text-red-400 hover:bg-red-500/10 rounded">
                                    <XCircle size={16} />
                                </button>
                            </div>
                        )}
                    </div>
                    {isEditing ? (
                        <textarea
                            className="input-field w-full min-h-[100px] text-sm"
                            value={editedDescription}
                            onChange={(e) => setEditedDescription(e.target.value)}
                        />
                    ) : (
                        <p className="text-sm text-[var(--text-primary)] leading-relaxed bg-[var(--bg-tertiary)] p-3 rounded-md border border-[var(--border)]">
                            {node.description || "No description available for this entity."}
                        </p>
                    )}
                </div>

                {/* Stats */}
                <div className="grid grid-cols-2 gap-3">
                    <div className="bg-[var(--bg-tertiary)] p-3 rounded-md border border-[var(--border)] text-center">
                        <div className="text-xs text-[var(--text-secondary)] mb-1">Community</div>
                        <div className="text-lg font-mono font-semibold text-[var(--text-primary)]">
                            {node.community_id ?? '-'}
                        </div>
                    </div>
                    <div className="bg-[var(--bg-tertiary)] p-3 rounded-md border border-[var(--border)] text-center">
                        <div className="text-xs text-[var(--text-secondary)] mb-1">Degree</div>
                        <div className="text-lg font-mono font-semibold text-[var(--text-primary)]">
                            {node.degree ?? '-'}
                        </div>
                    </div>
                </div>

                {/* Source Documents (Provenance) */}
                {node.documents && node.documents.length > 0 && (
                    <div className="space-y-2">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)]">Source Documents</h3>
                        <div className="space-y-1">
                            {node.documents.map((doc, idx) => (
                                <a
                                    key={doc.document_id || idx}
                                    href={`#`}
                                    className="flex items-center gap-2 p-2 bg-[var(--bg-tertiary)] rounded border border-[var(--border)] hover:border-accent-primary transition-colors group"
                                >
                                    <FileText size={14} className="text-accent-primary shrink-0" />
                                    <span className="text-xs text-[var(--text-primary)] truncate group-hover:text-accent-primary transition-colors">
                                        {doc.document_name || doc.document_id || 'Unknown Document'}
                                    </span>
                                </a>
                            ))}
                        </div>
                    </div>
                )}

                {/* Metadata */}
                {(node.metadata && Object.keys(node.metadata).length > 0) && (
                    <div className="space-y-2">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)]">Metadata</h3>
                        <div className="bg-[var(--bg-tertiary)] rounded-md border border-[var(--border)] overflow-hidden">
                            {Object.entries(node.metadata).map(([key, value], idx) => (
                                <div key={key} className={`flex text-xs p-2 ${idx !== 0 ? 'border-t border-[var(--border)]' : ''}`}>
                                    <span className="font-medium text-[var(--text-secondary)] w-1/3 shrink-0">{key}</span>
                                    <span className="text-[var(--text-primary)] font-mono break-all">{String(value)}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
