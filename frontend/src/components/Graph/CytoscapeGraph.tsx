import React, { useEffect, useMemo, useRef } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import cytoscape from 'cytoscape'
import fcose from 'cytoscape-fcose'
import type { GraphEdge, GraphNode } from '@/types/graph'
import { useGraphEditorStore } from './useGraphEditorStore'
import { ConfirmActionModal } from './ConfirmActionModal';
import { HealingSuggestionsModal } from './HealingSuggestionsModal';
import { MergeNodesModal } from './MergeNodesModal';
import { GitMerge } from 'lucide-react';
import edgehandles from 'cytoscape-edgehandles'

// Register extensions
cytoscape.use(fcose)
cytoscape.use(edgehandles)

interface CytoscapeGraphProps {
    nodes: GraphNode[]
    edges: GraphEdge[]
    backgroundColor?: string
    onNodeClick?: (node: GraphNode) => void
    width?: number
    height?: number
    onGraphUpdate?: () => void
}

const COMMUNITY_COLORS = [
    '#22d3ee', // cyan
    '#a855f7', // purple
    '#f97316', // orange
    '#10b981', // emerald
    '#3b82f6', // blue
    '#f59e0b', // amber
    '#e11d48', // rose
    '#0ea5e9', // sky
    '#8b5cf6', // violet
    '#14b8a6', // teal
]

function getCommunityColor(communityId?: number | null) {
    if (communityId === undefined || communityId === null) return '#9ca3af' // gray-400
    const idNum = Number(communityId)
    if (!Number.isFinite(idNum)) return '#9ca3af'
    const idx = Math.abs(Math.trunc(idNum)) % COMMUNITY_COLORS.length
    return COMMUNITY_COLORS[idx]
}

export default function CytoscapeGraph({ nodes, edges, backgroundColor, width, height, onGraphUpdate, onNodeClick }: CytoscapeGraphProps) {
    const cyRef = useRef<cytoscape.Core | null>(null)
    const ehRef = useRef<any>(null)
    const { mode } = useGraphEditorStore()

    // Safety Modal State for Pruning
    const [pruneTarget, setPruneTarget] = React.useState<{ type: 'edge' | 'node', id: string, source?: string, target?: string } | null>(null);
    const [healingNodeId, setHealingNodeId] = React.useState<string | null>(null);

    // Phase 4: Multi-select & Merge
    const [selectedNodes, setSelectedNodes] = React.useState<any[]>([]);
    const [showMergeModal, setShowMergeModal] = React.useState(false);

    // Phase 3: Ghost Edges
    const [ghostEdges, setGhostEdges] = React.useState<any[]>([]);
    const [tappedGhostEdge, setTappedGhostEdge] = React.useState<any | null>(null);

    // Phase 7: Orphan Mode
    const [orphanIds, setOrphanIds] = React.useState<Set<string>>(new Set());

    // Notification for heal mode
    const [healNotification, setHealNotification] = React.useState<string | null>(null);
    const [healLoading, setHealLoading] = React.useState(false);

    // Clear ghost edges and selection when mode changes
    useEffect(() => {
        if (mode !== 'heal') {
            setGhostEdges([]);
            setTappedGhostEdge(null);
        }
        if (mode !== 'select') {
            setSelectedNodes([]);
        }
        // Fetch orphans when entering orphan mode
        if (mode === 'orphan') {
            (async () => {
                try {
                    const token = localStorage.getItem('authToken');
                    const res = await fetch('/api/graph/editor/orphans', {
                        credentials: 'include',
                        headers: {
                            ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                        }
                    });
                    if (res.ok) {
                        const data = await res.json();
                        setOrphanIds(new Set(data.orphan_ids || []));
                    }
                } catch (e) {
                    console.error('Failed to fetch orphans', e);
                }
            })();
        } else {
            setOrphanIds(new Set());
        }
    }, [mode]);

    const fetchHealingSuggestions = React.useCallback(async (nodeId: string) => {
        try {
            setHealLoading(true);
            setHealNotification("Analyzing connections with AI... This may take a moment.");

            const token = localStorage.getItem('authToken');
            const response = await fetch('/api/graph/editor/heal', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                },
                credentials: 'include',
                body: JSON.stringify({ node_id: nodeId }),
            });

            if (!response.ok) throw new Error('Failed to fetch suggestions');

            const data = await response.json();
            const suggestions = data.suggestions || [];

            // Map suggestions to ghost edges
            // Only create ghosts for nodes that are currently visible to avoid layout explosions
            // (Or we could add ghost nodes, but let's stick to edges between visible nodes for now)
            const visibleNodeIds = new Set(nodes.map(n => n.id));

            const newGhosts = suggestions
                .filter((s: any) => visibleNodeIds.has(s.id))
                .map((s: any) => ({
                    data: {
                        source: nodeId,
                        target: s.id,
                        id: `ghost-${nodeId}-${s.id}`,
                        weight: s.score,
                        type: 'GHOST',
                        color: '#f59e0b', // Amber for suggestions
                        reason: s.reason || '' // Include LLM reasoning
                    },
                    classes: 'ghost-edge'
                }));

            setGhostEdges(newGhosts);
            setHealLoading(false);

            if (newGhosts.length === 0) {
                console.log("No visible suggestions found.");
                const allSuggestions = suggestions.length;
                if (allSuggestions === 0) {
                    setHealNotification("AI found no valid connection suggestions for this node.");
                } else {
                    setHealNotification(`AI approved ${allSuggestions} suggestions, but none are visible in the current view. Try zooming out.`);
                }
                setTimeout(() => setHealNotification(null), 4000);
            } else {
                setHealNotification(`AI approved ${newGhosts.length} connection(s). Click on a dashed line to accept.`);
                setTimeout(() => setHealNotification(null), 4000);
            }

        } catch (error) {
            console.error("Healing failed", error);
            setHealLoading(false);
            setHealNotification("Healing failed. Please try again.");
            setTimeout(() => setHealNotification(null), 3000);
        }
    }, [nodes]);

    const handleAcceptGhostEdge = React.useCallback(async () => {
        if (!tappedGhostEdge) return;

        try {
            const token = localStorage.getItem('authToken');
            const response = await fetch('/api/graph/editor/edge', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                },
                credentials: 'include',
                body: JSON.stringify({
                    source_id: tappedGhostEdge.source,
                    target_id: tappedGhostEdge.target,
                    relation_type: 'RELATED_TO',
                    properties: {
                        source: 'ai_healing',
                        confidence: tappedGhostEdge.weight
                    }
                }),
            });

            if (!response.ok) throw new Error('Failed to create edge');

            // Success
            onGraphUpdate?.();
            setGhostEdges(prev => prev.filter(e => e.data.id !== tappedGhostEdge.id));
            setTappedGhostEdge(null);

        } catch (e) {
            console.error("Failed to accept ghost edge", e);
        }
    }, [tappedGhostEdge, onGraphUpdate]);

    // Transform graph data into Cytoscape elements
    const elements = useMemo(() => {
        const cyNodes = nodes.map((node) => ({
            data: {
                id: node.id,
                label: node.label,
                community: node.community_id,
                degree: node.degree || 1,
                type: node.type,
                color: getCommunityColor(node.community_id),
            },
            classes: orphanIds.has(node.id) ? 'orphan-node' : '',
        }))

        const cyEdges = edges.map((edge) => ({
            data: {
                source: edge.source,
                target: edge.target,
                weight: edge.weight || 1,
                type: edge.type,
            },
        }))

        // Merge real elements with ghost edges
        return [...cyNodes, ...cyEdges, ...ghostEdges]
    }, [nodes, edges, ghostEdges, orphanIds])


    // Define the stylesheet with a "Premium Dark" aesthetic aligned with UI Kit
    const stylesheet = useMemo(
        () => [
            // NODES
            {
                selector: 'node',
                style: {
                    'background-color': 'data(color)',
                    label: 'data(label)',
                    color: '#E5E7EB', // --text-primary
                    'font-size': '12px',
                    'font-family': "'Noto Sans', sans-serif", // --font-body
                    'text-valign': 'bottom',
                    'text-halign': 'center',
                    'text-margin-y': 6,
                    'text-outline-color': '#121212', // --bg-primary
                    'text-outline-width': 2,
                    width: 'mapData(degree, 0, 20, 10, 40)',
                    height: 'mapData(degree, 0, 20, 10, 40)',
                    'border-width': 1,
                    'border-color': 'rgba(255,255,255,0.2)',
                    'transition-property': 'background-color, border-color, width, height',
                    'transition-duration': 300,
                },
            },
            // EDGES
            {
                selector: 'edge',
                style: {
                    width: 'mapData(weight, 0, 1, 1, 3)',
                    'line-color': '#3D3D3D', // --border
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': '#3D3D3D',
                    opacity: 0.6,
                },
            },
            // EDGE HANDLES (GHOST)
            {
                selector: '.eh-handle',
                style: {
                    'background-color': '#FF453A', // --systemRed
                    width: 12,
                    height: 12,
                    shape: 'ellipse',
                    'overlay-opacity': 0,
                    'border-width': 12,
                    'border-opacity': 0
                }
            },
            {
                selector: '.eh-hover',
                style: {
                    'background-color': '#FF453A'
                }
            },
            {
                selector: '.eh-source',
                style: {
                    'border-width': 2,
                    'border-color': '#FF453A'
                }
            },
            {
                selector: '.eh-target',
                style: {
                    'border-width': 2,
                    'border-color': '#FF453A'
                }
            },
            {
                selector: '.ghost-edge',
                style: {
                    'line-style': 'dashed',
                    'line-dash-pattern': [6, 3],
                    'line-color': '#f27a03', // --accent-primary
                    'target-arrow-color': '#f27a03',
                    'target-arrow-shape': 'triangle',
                    width: 3,
                    opacity: 0.8
                }
            },
            // ORPHAN NODES
            {
                selector: '.orphan-node',
                style: {
                    'border-width': 4,
                    'border-color': '#64D2FF', // --systemTeal
                    'border-style': 'double',
                    'background-opacity': 0.9,
                }
            }
        ],
        []
    )

    const layout = useMemo(() => ({
        name: 'fcose',
        quality: 'default',
        randomize: false,
        animate: true,
        animationDuration: 1000,
        fit: true,
        padding: 30,
        nodeDimensionsIncludeLabels: true,
        idealEdgeLength: (edge: any) => 100,
        nodeRepulsion: (node: any) => 4500,
    }), []);

    // Prune (Delete) Logic
    const handlePruneConfirm = async () => {
        if (!pruneTarget) return;

        try {
            if (pruneTarget.type === 'edge') {
                const token = localStorage.getItem('authToken');
                const response = await fetch('/api/graph/editor/edge', {
                    method: 'DELETE',
                    headers: {
                        'Content-Type': 'application/json',
                        ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                    },
                    credentials: 'include',
                    body: JSON.stringify({
                        source_id: pruneTarget.source,
                        target_id: pruneTarget.target,
                        relation_type: 'RELATED_TO' // Simplified for now, ideally we get type from edge data
                    })
                });

                if (!response.ok) throw new Error('Failed to delete edge');
            }
            // Node deletion to be implemented

            // Refresh graph
            onGraphUpdate?.();
            setPruneTarget(null);
        } catch (error) {
            console.error("Prune failed", error);
            // Optionally show error toast
        }
    };

    // Initialize Event Listeners (including Tap for Prune)
    useEffect(() => {
        if (!cyRef.current) return;
        const cy = cyRef.current;

        const handleTap = (evt: any) => {
            const target = evt.target;
            if (target === cy) return; // Clicked background

            const nodeData = target.data();

            if (target.isNode()) {
                if (mode === 'select') {
                    // Check for shift key (multi-select)
                    // Cytoscape events have originalEvent
                    const isShift = evt.originalEvent.shiftKey;

                    if (isShift) {
                        const newNode = { id: nodeData.id, label: nodeData.label };
                        setSelectedNodes(prev => {
                            const exists = prev.find(n => n.id === newNode.id);
                            if (exists) return prev.filter(n => n.id !== newNode.id);
                            return [...prev, newNode];
                        });
                    } else {
                        // Normal click - if passing to parent handler, do that
                        // But also reset local multi-selection unless we want to keep it?
                        // Let's clear multi-select on normal click for now to match expected behavior
                        setSelectedNodes([{ id: nodeData.id, label: nodeData.label }]);
                        // Map Cytoscape data back to GraphNode format
                        if (onNodeClick) onNodeClick({
                            id: nodeData.id,
                            label: nodeData.label,
                            type: nodeData.type,
                            community_id: nodeData.community,  // Map 'community' back to 'community_id'
                            degree: nodeData.degree,
                            color: nodeData.color,
                        });
                    }
                } else if (mode === 'prune') {
                    setPruneTarget({ type: 'node', id: nodeData.id });
                } else if (mode === 'heal') {
                    // Replaced Modal with Ghost Edges
                    // setHealingNodeId(nodeData.id); 
                    fetchHealingSuggestions(nodeData.id);
                }
            } else if (target.isEdge()) {
                if (mode === 'prune') {
                    setPruneTarget({
                        type: 'edge',
                        id: target.id(),
                        source: target.source().id(),
                        target: target.target().id()
                    });
                } else if (mode === 'heal' && target.hasClass('ghost-edge')) {
                    // Handle click on ghost edge
                    setTappedGhostEdge(target.data());
                }
            }
        };

        cy.on('tap', handleTap);
        return () => {
            cy.off('tap', handleTap);
        };
    }, [mode, onNodeClick, onGraphUpdate, fetchHealingSuggestions]); // Added onNodeClick and onGraphUpdate to dependencies

    // Initialize EdgeHandles
    useEffect(() => {
        if (!cyRef.current) return
        const cy = cyRef.current

        // Only initialize once
        if (!ehRef.current) {
            // Initialize edgehandles without strict type-checking
            // @ts-ignore
            if (cy.edgehandles) {
                // cast to any to avoid EdgeHandlesOptions type mismatches
                ehRef.current = (cy as any).edgehandles({
                    snap: true,
                    handleNodes: 'node',
                    handlePosition: function (node: any) {
                        return 'middle top';
                    },
                    handleInDrawMode: false,
                    edgeType: function (sourceNode: any, targetNode: any) {
                        return 'flat';
                    },
                    loopAllowed: (node: any) => false,
                    nodeLoopOffset: -50,
                    nodeParams: (sourceNode: any, targetNode: any) => ({ data: {} }),
                    edgeParams: (sourceNode: any, targetNode: any) => ({ data: {} }),
                    ghostEdgeParams: () => ({ data: {} }),
                    show: (sourceNode: any) => { },
                    hide: (sourceNode: any) => { },
                    start: (sourceNode: any) => { },
                    complete: async (sourceNode: any, targetNode: any, addedEles: any) => {
                        // When edge is drawn by user
                        console.log("Edge created!", sourceNode.id(), targetNode.id())

                        try {
                            // Call API to create edge
                            const token = localStorage.getItem('authToken');
                            const response = await fetch('/api/graph/editor/edge', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                    ...(token ? { 'Authorization': `Bearer ${token}` } : {})
                                },
                                credentials: 'include',
                                body: JSON.stringify({
                                    source_id: sourceNode.id(),
                                    target_id: targetNode.id(),
                                    relation_type: 'RELATED_TO', // Default, prompts can be added here
                                    properties: {
                                        weight: 1.0,
                                        created_by: 'manual_curation'
                                    }
                                })
                            });

                            if (!response.ok) {
                                console.error("Failed to persist edge")
                                // Ideally remove the added edge if API fails
                                addedEles.remove()
                            } else {
                                // Success!
                                onGraphUpdate?.();
                            }
                        } catch (e) {
                            console.error("Error creating edge", e)
                            addedEles.remove()
                        }
                    },
                    cancel: (sourceNode: any, cancelledTargets: any) => { },
                    hoverover: (sourceNode: any, targetNode: any) => { },
                    hoverout: (sourceNode: any, targetNode: any) => { },
                    previewon: (sourceNode: any, targetNode: any, previewEles: any) => { },
                    previewoff: (sourceNode: any, targetNode: any, previewEles: any) => { },
                    drawon: () => { },
                    drawoff: () => { }
                } as any);
            }
        }

        // Toggle based on mode
        if (ehRef.current) {
            if (mode === 'connect') {
                ehRef.current.enableDrawMode()
            } else {
                ehRef.current.disableDrawMode()
                ehRef.current.disable()
            }
        }

    }, [mode, onGraphUpdate]) // Dependencies

    // Handle re-layout when data dramatically changes
    useEffect(() => {
        if (!cyRef.current) return
        const cy = cyRef.current
        const runLayout = () => {
            cy.layout(layout as any).run()
        }
        runLayout()
    }, [elements.length, layout])

    return (
        <div style={{ width: width ?? '100%', height: height ?? '100%', backgroundColor }}>
            <CytoscapeComponent
                elements={CytoscapeComponent.normalizeElements(elements)}
                style={{ width: '100%', height: '100%' }}
                stylesheet={stylesheet as any}
                layout={layout}
                cy={(cy) => {
                    cyRef.current = cy
                }}
            // wheelSensitivity removed - use browser default for natural scrolling
            />

            {pruneTarget && (
                <ConfirmActionModal
                    title={pruneTarget.type === 'edge' ? 'Prune Connection' : `Prune Node ${pruneTarget.id}`}
                    message={`Are you sure you want to permanently delete this ${pruneTarget.type === 'edge' ? 'connection' : 'node'}? This action cannot be undone.`}
                    confirmText="Prune"
                    isDangerous={true}
                    onConfirm={handlePruneConfirm}
                    onCancel={() => setPruneTarget(null)}
                />
            )}

            {healingNodeId && (
                <HealingSuggestionsModal
                    nodeId={healingNodeId}
                    onClose={() => setHealingNodeId(null)}
                />
            )}

            {tappedGhostEdge && (
                <ConfirmActionModal
                    title="Accept Suggestion?"
                    message={`Do you want to create this connection with ${(tappedGhostEdge.weight * 100).toFixed(0)}% confidence?`}
                    confirmText="Create Connection"
                    isDangerous={false}
                    onConfirm={handleAcceptGhostEdge}
                    onCancel={() => setTappedGhostEdge(null)}
                />
            )}

            {/* Merge Button Overlay */}
            {selectedNodes.length > 1 && mode === 'select' && (
                <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-10">
                    <button
                        onClick={() => setShowMergeModal(true)}
                        className="button-primary shadow-lg shadow-indigo-900/40 rounded-full flex items-center gap-2 hover:scale-105 active:scale-95 transition-all"
                    >
                        <GitMerge className="w-5 h-5" />
                        Merge {selectedNodes.length} Nodes
                    </button>
                </div>
            )}

            {showMergeModal && (
                <MergeNodesModal
                    selectedNodes={selectedNodes}
                    onClose={() => setShowMergeModal(false)}
                    onMergeSuccess={() => {
                        setShowMergeModal(false);
                        setSelectedNodes([]);
                        onGraphUpdate?.();
                    }}
                />
            )}

            {/* Heal Notification Banner */}
            {healNotification && (
                <div className={`absolute top-4 left-1/2 transform -translate-x-1/2 z-20 ${healLoading ? 'bg-accent-primary' : 'bg-accent-hover'} text-white px-4 py-2 rounded-lg shadow-lg text-sm font-medium flex items-center gap-2`}>
                    {healLoading && (
                        <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                    )}
                    {healNotification}
                </div>
            )}
        </div>
    )
}
