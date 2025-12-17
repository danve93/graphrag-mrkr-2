'use client'

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'
import type { GraphEdge, GraphNode } from '@/types/graph'
import { useGestureControls } from './useGestureControls'

// Same community colors as CytoscapeGraph for consistency
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

function getCommunityColor(communityId?: number | null): string {
    if (communityId === undefined || communityId === null) return '#9ca3af'
    const idNum = Number(communityId)
    if (!Number.isFinite(idNum)) return '#9ca3af'
    const idx = Math.abs(Math.trunc(idNum)) % COMMUNITY_COLORS.length
    return COMMUNITY_COLORS[idx]
}

interface ThreeGraphProps {
    nodes: GraphNode[]
    edges: GraphEdge[]
    onNodeClick?: (node: GraphNode) => void
    gestureEnabled?: boolean
}

// Transform nodes/edges to force-graph format
interface GraphData {
    nodes: Array<{
        id: string
        label: string
        color: string
        community_id?: number | null
        type?: string | null
        degree?: number
        val: number
        x?: number // Injected by force-graph
        y?: number
        z?: number
    }>
    links: Array<{
        source: string
        target: string
        weight: number
        type?: string | null
    }>
}

export default function ThreeGraph({ nodes, edges, onNodeClick, gestureEnabled = true }: ThreeGraphProps) {
    const fgRef = useRef<any>()
    const { gestureState, getVideoElement } = useGestureControls(gestureEnabled)
    const [hoveredNode, setHoveredNode] = useState<string | null>(null)
    const [showWebcam, setShowWebcam] = useState(false)
    const webcamContainerRef = useRef<HTMLDivElement>(null)

    // Raycaster for custom cursor selection
    const raycaster = useMemo(() => new THREE.Raycaster(), [])
    const mouse = useMemo(() => new THREE.Vector2(), [])

    // Transform data for force-graph-3d
    const graphData = useMemo<GraphData>(() => {
        const nodeMap = new Map(nodes.map(n => [n.id, n]))

        return {
            nodes: nodes.map(node => ({
                id: node.id,
                label: node.label,
                color: getCommunityColor(node.community_id),
                community_id: node.community_id,
                type: node.type,
                degree: node.degree,
                val: Math.max(1, (node.degree || 1) * 2), // Size based on degree
            })),
            links: edges
                .filter(e => nodeMap.has(e.source) && nodeMap.has(e.target))
                .map(edge => ({
                    source: edge.source,
                    target: edge.target,
                    weight: edge.weight || 1,
                    type: edge.type,
                })),
        }
    }, [nodes, edges])

    // Handle node click
    const handleNodeClick = useCallback((node: any) => {
        if (onNodeClick) {
            onNodeClick({
                id: node.id,
                label: node.label,
                type: node.type,
                community_id: node.community_id,
                degree: node.degree,
            })
        }

        // Zoom to node with smooth animation
        if (fgRef.current) {
            const distance = 200
            const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z)

            fgRef.current.cameraPosition(
                { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
                node,
                1000
            )
        }
    }, [onNodeClick])

    // Custom node rendering with glow effect
    const nodeThreeObject = useCallback((node: any) => {
        const group = new THREE.Group()

        // Core sphere with glow
        const geometry = new THREE.SphereGeometry(node.val, 16, 16)

        // Create gradient material for premium look
        const color = new THREE.Color(node.color)
        const material = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.9,
        })

        const sphere = new THREE.Mesh(geometry, material)
        group.add(sphere)

        // Outer glow ring
        const glowGeometry = new THREE.SphereGeometry(node.val * 1.3, 16, 16)
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.15,
            side: THREE.BackSide,
        })
        const glow = new THREE.Mesh(glowGeometry, glowMaterial)
        group.add(glow)

        // Create text sprite for label
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')!
        const fontSize = 48
        const label = node.label || node.id

        canvas.width = 512
        canvas.height = 64
        ctx.font = `${fontSize}px 'Noto Sans', sans-serif`
        ctx.fillStyle = '#E5E7EB' // --text-primary
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(label.substring(0, 30), canvas.width / 2, canvas.height / 2)

        const texture = new THREE.CanvasTexture(canvas)
        const spriteMaterial = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            depthTest: false,
        })
        const sprite = new THREE.Sprite(spriteMaterial)
        sprite.scale.set(40, 5, 1)
        sprite.position.y = -node.val - 8

        group.add(sprite)

        return group
    }, [])

    // Handle gesture interactions
    const prevPinching = useRef(false)

    useEffect(() => {
        if (!fgRef.current || !gestureEnabled) return

        const fg = fgRef.current

        // --- 1. ROTATION (Two Fists) ---
        if (gestureState.mode === 'rotate') {
            // Manual Camera Orbiting using Spherical Coordinates
            // This bypasses OrbitControls state issues by calculating the new camera position directly.
            const cam = fg.camera()
            if (cam) {
                const rotSpeed = 2.5 // Tuned for direct spherical updates

                // 1. Get current position
                const pos = cam.position.clone()

                // 2. Convert to Spherical (Radius, Phi, Theta)
                const spherical = new THREE.Spherical()
                spherical.setFromVector3(pos)

                // 3. Apply rotation deltas
                // Azimuth (Theta): Horizontal rotation around Y axis
                // "Grab and Drag Left" (dx > 0) -> Object Left -> Camera Right -> Decrease Theta (Math convention)
                // actually OrbitControls "Left" increases Theta?
                // Let's stick to: dx * speed. If it's inverted, we flip sign.
                // Previous logic: Hand Left (dx > 0) -> Camera Right.
                // In standard spherical: Camera Right = Decrease Theta.
                // So: -dx. 
                // Let's try negative first for "Grab" feel.
                spherical.theta -= gestureState.rotation.x * rotSpeed

                // Polar (Phi): Vertical rotation (0 = Top, PI = Bottom)
                // Drag Up (dy < 0) -> Object Up -> Camera Down -> Increase Phi
                // dy is neg. So -dy is pos.
                // So: check sign.
                spherical.phi -= gestureState.rotation.y * rotSpeed

                // 4. Clamp Phi to avoid camera flipping over poles
                spherical.phi = Math.max(0.05, Math.min(Math.PI - 0.05, spherical.phi))

                // 5. Convert back to Cartesian
                const newPos = new THREE.Vector3()
                newPos.setFromSpherical(spherical)

                // 6. Update Camera Position
                // 0ms transition for immediate response
                // null for lookAt keeps looking at (0,0,0) or current target
                fg.cameraPosition(
                    { x: newPos.x, y: newPos.y, z: newPos.z },
                    { x: 0, y: 0, z: 0 },
                    0
                )
            }
        }

        // --- 2. ZOOM (Two Open Hands) ---
        if (gestureState.mode === 'zoom' && gestureState.zoomDelta !== 1) {
            try {
                const cam = fg.camera()
                if (cam && cam.position) {
                    const currentDistance = cam.position.length()
                    const newDistance = currentDistance / gestureState.zoomDelta
                    const ratio = newDistance / currentDistance

                    fg.cameraPosition({
                        x: cam.position.x * ratio,
                        y: cam.position.y * ratio,
                        z: cam.position.z * ratio
                    })
                }
            } catch (e) {
                // Ignore zoom errors 
            }
        }

        // --- 3. PINCH CLICK (Cursor) ---
        if (gestureState.isPinching && !prevPinching.current) {
            const cam = fg.camera()
            const scene = fg.scene()

            if (cam && scene) {
                // gestureState.cursor is norm coords 0-1
                mouse.x = (gestureState.cursor.x * 2) - 1
                mouse.y = -(gestureState.cursor.y * 2) + 1

                raycaster.setFromCamera(mouse, cam)

                let closestNode = null
                let minDist = 0.05

                const vector = new THREE.Vector3()

                // Helper to access current nodes with position
                // ForceGraph3D modifies the graphData object in place with x,y,z
                // We use the local graphData which contains the same update references
                const currentNodes = graphData.nodes

                if (currentNodes && currentNodes.length > 0) {
                    for (const node of currentNodes) {
                        if (node.x !== undefined && node.y !== undefined) {
                            vector.set(node.x, node.y, node.z as number)
                            vector.project(cam)

                            const dist = Math.sqrt((vector.x - mouse.x) ** 2 + (vector.y - mouse.y) ** 2)
                            if (dist < minDist) {
                                minDist = dist
                                closestNode = node
                            }
                        }
                    }
                }

                if (closestNode) {
                    handleNodeClick(closestNode)
                }
            }
        }

        prevPinching.current = gestureState.isPinching

    }, [gestureState, gestureEnabled, graphData, mouse, raycaster, handleNodeClick])

    // Attach webcam video to PiP container
    useEffect(() => {
        if (!showWebcam || !webcamContainerRef.current) return

        const video = getVideoElement()
        if (video) {
            video.style.display = 'block'
            video.style.width = '100%'
            video.style.height = '100%'
            video.style.objectFit = 'cover'
            video.style.borderRadius = '8px'
            video.style.transform = 'scaleX(-1)' // Mirror for natural feel
            webcamContainerRef.current.appendChild(video)

            return () => {
                video.style.display = 'none'
                video.remove()
                document.body.appendChild(video) // Put it back for MediaPipe
            }
        }
    }, [showWebcam, getVideoElement])

    return (
        <div className="relative w-full h-full"> {/* Removed cursor-none */}
            {/* 3D Graph */}
            <ForceGraph3D
                ref={fgRef}
                graphData={graphData}
                nodeLabel=""
                nodeThreeObject={nodeThreeObject}
                nodeThreeObjectExtend={false}
                linkOpacity={0.3}
                linkWidth={1}
                linkColor={() => '#3D3D3D'}
                backgroundColor="#121212"
                showNavInfo={false}
                onNodeClick={handleNodeClick}
                onNodeHover={(node) => setHoveredNode(node?.id || null)}
                d3AlphaDecay={0.02}
                d3VelocityDecay={0.3}
                warmupTicks={100}
                cooldownTicks={0}
            />

            {/* Visual Pointing Cursor */}
            {gestureEnabled && gestureState.isTracking && gestureState.handCount === 1 && (
                <div
                    className="absolute pointer-events-none z-50 transition-transform duration-75 ease-out will-change-transform"
                    style={{
                        left: 0,
                        top: 0,
                        transform: `translate3d(${gestureState.cursor.x * window.innerWidth}px, ${gestureState.cursor.y * window.innerHeight}px, 0) translate(-50%, -50%)`
                    }}
                >
                    {/* Ring */}
                    <div className={`
                        w-12 h-12 rounded-full border-2 
                        ${gestureState.isPinching ? 'border-[var(--systemGreen)] scale-75' : 'border-[var(--accent-primary)]'}
                        shadow-[0_0_15px_rgba(255,255,255,0.5)]
                        transition-all duration-150
                    `}></div>
                    {/* Dot */}
                    <div className={`
                        absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full
                        ${gestureState.isPinching ? 'w-full h-full bg-[var(--systemGreen)] opacity-50' : 'w-2 h-2 bg-white'}
                    `}></div>
                </div>
            )}

            {/* Gesture Status Overlay */}
            {gestureEnabled && (
                <div
                    className="absolute bottom-4 left-4 z-10"
                    onMouseEnter={() => setShowWebcam(true)}
                    onMouseLeave={() => setShowWebcam(false)}
                >
                    {/* Status indicator */}
                    <div className={`
                        flex items-center gap-2 px-3 py-2 rounded-lg backdrop-blur-md
                        ${gestureState.isTracking
                            ? 'bg-[var(--bg-secondary)]/80 border border-[var(--systemGreen)]/50'
                            : gestureState.permissionStatus === 'denied'
                                ? 'bg-[var(--bg-secondary)]/80 border border-[var(--systemRed)]/50'
                                : 'bg-[var(--bg-secondary)]/80 border border-[var(--border)]'
                        }
                    `}>
                        <div className={`w-2 h-2 rounded-full ${gestureState.isTracking
                            ? 'bg-[var(--systemGreen)] animate-pulse'
                            : gestureState.permissionStatus === 'denied'
                                ? 'bg-[var(--systemRed)]'
                                : 'bg-[var(--systemOrange)]'
                            }`} />
                        <span className="text-xs text-[var(--text-secondary)]">
                            {gestureState.gestureName ||
                                (gestureState.permissionStatus === 'pending' ? 'Waiting for camera...' :
                                    gestureState.permissionStatus === 'denied' ? 'Camera denied' :
                                        'Gesture control ready')}
                        </span>
                        {gestureState.handCount > 0 && (
                            <div className="flex flex-col items-start gap-1">
                                <span className="text-xs px-1.5 py-0.5 rounded bg-[var(--accent-primary)]/20 text-[var(--accent-primary)] font-mono">
                                    {gestureState.mode === 'cursor' ? 'CURSOR' :
                                        gestureState.mode === 'rotate' ? 'ROTATE' :
                                            gestureState.mode === 'zoom' ? 'ZOOM' : 'IDLE'}
                                </span>
                                {gestureState.mode === 'rotate' && (
                                    <span className="text-[10px] text-[var(--text-tertiary)] font-mono">
                                        dx: {gestureState.rotation.x.toFixed(3)} dy: {gestureState.rotation.y.toFixed(3)}
                                    </span>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Webcam PiP - only visible on hover */}
                    {showWebcam && gestureState.isTracking && (
                        <div
                            ref={webcamContainerRef}
                            className="mt-2 w-32 h-24 rounded-lg overflow-hidden border border-[var(--border)] shadow-xl"
                            style={{ boxShadow: '0 0 20px rgba(242, 122, 3, 0.3)' }}
                        />
                    )}
                </div>
            )}

            {/* Hovered node tooltip */}
            {hoveredNode && (
                <div className="absolute top-4 right-4 z-10 max-w-xs">
                    <div className="px-3 py-2 rounded-lg bg-[var(--bg-secondary)]/90 border border-[var(--accent-primary)]/50 backdrop-blur-sm">
                        <p className="text-sm font-medium text-[var(--text-primary)]">{hoveredNode}</p>
                    </div>
                </div>
            )}
        </div>
    )
}
