import React, { useEffect, useMemo, useRef } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import cytoscape from 'cytoscape'
import fcose from 'cytoscape-fcose'
import type { GraphEdge, GraphNode } from '@/types/graph'
import { useGraphEditorStore } from './useGraphEditorStore'
import { ConfirmActionModal } from './ConfirmActionModal'
import { API_URL } from '@/lib/api'
import { showToast } from '@/components/Toast/ToastContainer'
import { GitMerge } from 'lucide-react'
import { MergeNodesModal } from './MergeNodesModal'
import { HealingSuggestionsModal } from './HealingSuggestionsModal'
// @ts-ignore
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
    editable?: boolean
}

// Premium color palette synced with ThreeGraph for consistency
const COMMUNITY_COLORS = [
    '#06b6d4', // cyan-500 (deeper)
    '#8b5cf6', // violet-500
    '#f97316', // orange-500
    '#10b981', // emerald-500
    '#3b82f6', // blue-500
    '#f59e0b', // amber-500
    '#ec4899', // pink-500
    '#6366f1', // indigo-500
    '#14b8a6', // teal-500
    '#ef4444', // red-500
]

function getCommunityColor(communityId?: number | null) {
    if (communityId === undefined || communityId === null) return '#9ca3af' // gray-400
    const idNum = Number(communityId)
    if (!Number.isFinite(idNum)) return '#9ca3af'
    const idx = Math.abs(Math.trunc(idNum)) % COMMUNITY_COLORS.length
    return COMMUNITY_COLORS[idx]
}

export default function CytoscapeGraph({
    nodes,
    edges,
    backgroundColor,
    width,
    height,
    onGraphUpdate,
    onNodeClick,
    editable = true,
}: CytoscapeGraphProps) {
    const cyRef = useRef<cytoscape.Core | null>(null)
    const ehRef = useRef<any>(null)
    const { mode, setMode } = useGraphEditorStore()
    const effectiveMode = editable ? mode : 'select'
    const nodeLookup = useMemo(() => new Map(nodes.map((node) => [node.id, node])), [nodes])

    // Safety Modal State for Pruning
    // Safety Modal State for Pruning
    const [pruneTarget, setPruneTarget] = React.useState<{ type: 'edge' | 'node', id: string, source?: string, target?: string } | null>(null);

    // Interaction State
    const [healingNodeId, setHealingNodeId] = React.useState<string | null>(null);
    const [tappedGhostEdge, setTappedGhostEdge] = React.useState<any>(null);
    const [showMergeModal, setShowMergeModal] = React.useState(false);
    const [selectedNodes, setSelectedNodes] = React.useState<any[]>([]);
    const [healNotification, setHealNotification] = React.useState<string | null>(null);
    const [healLoading, setHealLoading] = React.useState(false);

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
        }))

        const cyEdges = edges.map((edge) => ({
            data: {
                source: edge.source,
                target: edge.target,
                weight: edge.weight || 1,
                type: edge.type,
            },
        }))

        return [...cyNodes, ...cyEdges]
    }, [nodes, edges])

    // Define the stylesheet with a "Premium Dark" aesthetic
    const stylesheet = useMemo(
        () => [
            // NODES
            {
                selector: 'node',
                style: {
                    'background-color': 'data(color)',
                    label: 'data(label)',
                    color: '#e2e8f0', // slate-200
                    'font-size': '12px',
                    'font-family': 'Inter, sans-serif, system-ui',
                    'text-valign': 'bottom',
                    'text-halign': 'center',
                    'text-margin-y': 6,
                    'text-outline-color': '#0f172a', // slate-900 (background)
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
                    width: 1.5,
                    'line-color': '#334155', // slate-700
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': '#334155',
                    opacity: 0.6,
                },
            },
            {
                selector: 'edge[weight]',
                style: {
                    width: 'mapData(weight, 0, 1, 1, 3)',
                },
            },
            // EDGE HANDLES (GHOST)
            {
                selector: '.eh-handle',
                style: {
                    'background-color': '#f87171', // red-400
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
                    'background-color': '#f87171'
                }
            },
            {
                selector: '.eh-source',
                style: {
                    'border-width': 2,
                    'border-color': '#f87171'
                }
            },
            {
                selector: '.eh-target',
                style: {
                    'border-width': 2,
                    'border-color': '#f87171'
                }
            },
            {
                selector: '.eh-preview, .eh-ghost-edge',
                style: {
                    'background-color': '#f87171',
                    'line-color': '#f87171',
                    'target-arrow-color': '#f87171',
                    'source-arrow-color': '#f87171'
                }
            },
            {
                selector: '.eh-ghost-edge.eh-preview-active',
                style: {
                    'opacity': 0
                }
            }
        ],
        []
    )

    const layout = {
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
    }

    // Prune (Delete) Logic
    const handlePruneConfirm = async () => {
        if (!pruneTarget) return;

        try {
            const token = localStorage.getItem('authToken');
            if (pruneTarget.type === 'edge') {
                const response = await fetch(`${API_URL}/api/graph/editor/edge`, {
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

                if (!response.ok) {
                    const payload = await response.text()
                    throw new Error(payload || 'Failed to delete edge');
                }
            }
            // Node deletion to be implemented

            // Refresh graph
            onGraphUpdate?.();
            setPruneTarget(null);
            setMode('select');
        } catch (error) {
            console.error("Prune failed", error);
            showToast('error', 'Prune failed', error instanceof Error ? error.message : 'Failed to delete edge');
        }
    };

    const handleAcceptGhostEdge = async () => {
        if (!tappedGhostEdge) return;
        // Logic to accept ghost edge (AI suggestion)
        // For now, treat same as manual edge creation
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
                    source_id: tappedGhostEdge.source,
                    target_id: tappedGhostEdge.target,
                    relation_type: 'RELATED_TO',
                    properties: {
                        weight: tappedGhostEdge.weight,
                        created_by: 'ai_suggestion'
                    }
                })
            });

            if (!response.ok) throw new Error('Failed to create edge');
            onGraphUpdate?.();
            setTappedGhostEdge(null);
            showToast('success', 'Connection created', 'AI suggestion accepted');
        } catch (e) {
            showToast('error', 'Failed to create connection', String(e));
        }
    };

    // Initialize Event Listeners (including Tap for Prune and Heal)
    useEffect(() => {
        if (!editable || !cyRef.current) return;
        const cy = cyRef.current;

        const handleTap = (evt: any) => {
            if (effectiveMode === 'prune') {
                const target = evt.target;
                if (target === cy) return; // Clicked background

                if (target.isEdge()) {
                    setPruneTarget({
                        type: 'edge',
                        id: target.id(),
                        source: target.source().id(),
                        target: target.target().id()
                    });
                }
                // Node pruning can be added here
            }
        };

        cy.on('tap', handleTap);
        return () => {
            cy.off('tap', handleTap);
        };
    }, [editable, effectiveMode]);

    // Handle Node Selection & Heal Mode
    useEffect(() => {
        if (!cyRef.current) return;
        const cy = cyRef.current;
        const handleNodeTap = (evt: any) => {
            const nodeId = evt.target.id();
            const selected = nodeLookup.get(nodeId);

            if (effectiveMode === 'heal') {
                setHealingNodeId(nodeId);
                return;
            }

            if (effectiveMode === 'select') {
                if (selected) {
                    onNodeClick?.(selected);
                }
                // Update internal selection state for Merge
                const currentSelected = cy.$(':selected').map((ele: any) => ele.id());
                // We need the actual node objects for the merge modal
                // Using a slight delay to let Cytoscape update its selection state
                setTimeout(() => {
                    const selectedEles = cy.$('node:selected');
                    const selectedData = selectedEles.map((ele: any) => ({
                        id: ele.id(),
                        label: ele.data('label'),
                        type: ele.data('type')
                    }));

                    // If multiple nodes selected, update state
                    if (selectedData.length > 0) {
                        setSelectedNodes(selectedData);
                    } else {
                        setSelectedNodes([]);
                    }
                }, 10);
            }
        };

        cy.on('tap', 'node', handleNodeTap);
        // Also listen for unselect to clear merge state
        const handleUnselect = () => {
            setTimeout(() => {
                const selectedEles = cy.$('node:selected');
                if (selectedEles.length === 0) {
                    setSelectedNodes([]);
                }
            }, 10);
        };
        cy.on('unselect', 'node', handleUnselect);

        return () => {
            cy.off('tap', 'node', handleNodeTap);
            cy.off('unselect', 'node', handleUnselect);
        };
    }, [effectiveMode, nodeLookup, onNodeClick]);

    // Initialize EdgeHandles
    useEffect(() => {
        if (!editable || !cyRef.current) return
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
                            const response = await fetch(`${API_URL}/api/graph/editor/edge`, {
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
                                const errorText = await response.text();
                                console.error("Failed to persist edge:", errorText);
                                showToast('error', 'Edge creation failed', errorText || 'Failed to create edge');
                                // Remove the added edge if API fails
                                addedEles.remove();
                            } else {
                                const data = await response.json();
                                console.log("Edge created successfully:", data);
                                showToast('success', 'Edge created', 'Relationship saved to the graph');
                                // Success! Refresh graph
                                onGraphUpdate?.();
                                setMode('select');
                            }
                        } catch (e) {
                            console.error("Error creating edge", e);
                            showToast('error', 'Edge creation failed', e instanceof Error ? e.message : String(e));
                            addedEles.remove();
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
            if (effectiveMode === 'connect') {
                ehRef.current.enable()  // Must enable first
                ehRef.current.enableDrawMode()
            } else {
                ehRef.current.disableDrawMode()
                ehRef.current.disable()
            }
        }

    }, [editable, effectiveMode]) // Dependencies

    // Handle re-layout when data dramatically changes
    useEffect(() => {
        if (!cyRef.current) return
        const cy = cyRef.current
        const runLayout = () => {
            cy.layout(layout as any).run()
        }
        runLayout()
    }, [elements.length])

    useEffect(() => {
        if (!cyRef.current) return
        cyRef.current.resize()
    }, [width, height])

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
