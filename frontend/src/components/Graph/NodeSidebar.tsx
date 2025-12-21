import React, { useEffect, useMemo, useState } from 'react';
import { X, Edit2, MessageSquare, FileText, Check, XCircle, Network, Users } from 'lucide-react';
import type { GraphNode, GraphEdge } from '@/types/graph';
import { API_URL } from '@/lib/api';

interface NodeSidebarProps {
    node: GraphNode;
    onClose: () => void;
    onEdit: (node: GraphNode) => void;
    onChat: (node: GraphNode) => void;
    allNodes?: GraphNode[];  // Optional: for finding connected nodes
    allEdges?: GraphEdge[];  // Optional: for finding connections
    onNodeSelect?: (node: GraphNode) => void;  // Optional: for navigating to connected node
}

export default function NodeSidebar({
    node,
    onClose,
    onEdit,
    onChat,
    allNodes = [],
    allEdges = [],
    onNodeSelect
}: NodeSidebarProps) {
    const [isEditing, setIsEditing] = useState(false);
    const [editedDescription, setEditedDescription] = useState(node.description || '');

    useEffect(() => {
        setIsEditing(false);
        setEditedDescription(node.description || '');
    }, [node.id, node.description]);

    // Find connected nodes from edges
    const connectedNodes = useMemo(() => {
        if (!allEdges.length || !allNodes.length) return [];

        const connectedIds = new Set<string>();
        allEdges.forEach(edge => {
            if (edge.source === node.id) connectedIds.add(edge.target);
            if (edge.target === node.id) connectedIds.add(edge.source);
        });

        return allNodes
            .filter(n => connectedIds.has(n.id))
            .sort((a, b) => (b.degree || 0) - (a.degree || 0))
            .slice(0, 10); // Limit to top 10 by degree
    }, [node.id, allNodes, allEdges]);

    // Find nodes in the same community
    const communityNodes = useMemo(() => {
        if (node.community_id === null || node.community_id === undefined) return [];
        return allNodes
            .filter(n => n.community_id === node.community_id && n.id !== node.id)
            .sort((a, b) => (b.degree || 0) - (a.degree || 0))
            .slice(0, 8); // Limit to top 8
    }, [node.id, node.community_id, allNodes]);

    const handleSaveEdit = async () => {
        try {
            const token = localStorage.getItem('authToken');
            const response = await fetch(`${API_URL}/api/graph/editor/node`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                },
                credentials: 'include',
                body: JSON.stringify({
                    node_id: node.id,
                    properties: { description: editedDescription }
                }),
            });
            if (response.ok) {
                setIsEditing(false);
                onEdit({ ...node, description: editedDescription });
            } else {
                console.error('Failed to save edit:', await response.text());
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
            <div className="p-3 grid grid-cols-2 gap-2 border-b border-[var(--border)] bg-[var(--bg-secondary)]">
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
                        <div className="text-xs text-[var(--text-secondary)] mb-1 flex items-center justify-center gap-1" title="Number of direct connections this entity has">
                            Degree
                        </div>
                        <div className="text-lg font-mono font-semibold text-[var(--text-primary)]">
                            {node.degree ?? '-'}
                        </div>
                    </div>
                </div>

                {/* Connected Entities */}
                {connectedNodes.length > 0 && (
                    <div className="space-y-2">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)] flex items-center gap-2">
                            <Network size={12} />
                            Connected Entities ({connectedNodes.length})
                        </h3>
                        <div className="space-y-1 max-h-40 overflow-y-auto">
                            {connectedNodes.map((connNode) => (
                                <button
                                    key={connNode.id}
                                    onClick={() => onNodeSelect?.(connNode)}
                                    className="w-full flex items-center gap-2 p-2 bg-[var(--bg-tertiary)] rounded border border-[var(--border)] hover:border-accent-primary transition-colors group text-left"
                                >
                                    <div
                                        className="w-6 h-6 rounded flex items-center justify-center text-white text-[10px] font-bold shrink-0"
                                        style={{ backgroundColor: connNode.color || 'var(--accent-primary)' }}
                                    >
                                        {connNode.label.charAt(0).toUpperCase()}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <span className="text-xs text-[var(--text-primary)] truncate block group-hover:text-accent-primary transition-colors">
                                            {connNode.label}
                                        </span>
                                        <span className="text-[10px] text-[var(--text-secondary)]">
                                            {connNode.type || 'Entity'} • deg: {connNode.degree ?? 0}
                                        </span>
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {/* Same Community Entities */}
                {communityNodes.length > 0 && (
                    <div className="space-y-2">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)] flex items-center gap-2">
                            <Users size={12} />
                            Same Community ({communityNodes.length})
                        </h3>
                        <div className="space-y-1 max-h-32 overflow-y-auto">
                            {communityNodes.map((commNode) => (
                                <button
                                    key={commNode.id}
                                    onClick={() => onNodeSelect?.(commNode)}
                                    className="w-full flex items-center gap-2 p-2 bg-[var(--bg-tertiary)] rounded border border-[var(--border)] hover:border-accent-primary transition-colors group text-left"
                                >
                                    <div
                                        className="w-6 h-6 rounded flex items-center justify-center text-white text-[10px] font-bold shrink-0"
                                        style={{ backgroundColor: commNode.color || 'var(--accent-primary)' }}
                                    >
                                        {commNode.label.charAt(0).toUpperCase()}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <span className="text-xs text-[var(--text-primary)] truncate block group-hover:text-accent-primary transition-colors">
                                            {commNode.label}
                                        </span>
                                        <span className="text-[10px] text-[var(--text-secondary)]">
                                            {commNode.type || 'Entity'} • deg: {commNode.degree ?? 0}
                                        </span>
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>
                )}

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
