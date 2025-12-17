import React, { useEffect, useMemo, useRef } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import cytoscape from 'cytoscape'
import fcose from 'cytoscape-fcose'
import type { GraphEdge, GraphNode } from '@/types/graph'
import { useGraphEditorStore } from './useGraphEditorStore'
import { ConfirmActionModal } from './ConfirmActionModal'
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

export default function CytoscapeGraph({ nodes, edges, backgroundColor, width, height, onGraphUpdate }: CytoscapeGraphProps) {
    const cyRef = useRef<cytoscape.Core | null>(null)
    const ehRef = useRef<any>(null)
    const { mode } = useGraphEditorStore()

    // Safety Modal State for Pruning
    const [pruneTarget, setPruneTarget] = React.useState<{ type: 'edge' | 'node', id: string, source?: string, target?: string } | null>(null);

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
                    width: 'mapData(weight, 0, 1, 1, 3)',
                    'line-color': '#334155', // slate-700
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': '#334155',
                    opacity: 0.6,
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
            if (pruneTarget.type === 'edge') {
                const response = await fetch('/api/graph/editor/edge', {
                    method: 'DELETE',
                    headers: { 'Content-Type': 'application/json' },
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
            if (mode === 'prune') {
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
    }, [mode]);

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
                            const response = await fetch('/api/graph/editor/edge', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
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

    }, [mode]) // Dependencies

    // Handle re-layout when data dramatically changes
    useEffect(() => {
        if (!cyRef.current) return
        const cy = cyRef.current
        const runLayout = () => {
            cy.layout(layout as any).run()
        }
        runLayout()
    }, [elements.length])

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
                wheelSensitivity={0.3}
            />

            {pruneTarget && (
                <ConfirmActionModal
                    title="Prune Edge"
                    message="Are you sure you want to permanently delete this relationship?"
                    confirmText="Delete"
                    isDangerous={true}
                    onConfirm={handlePruneConfirm}
                    onCancel={() => setPruneTarget(null)}
                />
            )}
        </div>
    )
}
