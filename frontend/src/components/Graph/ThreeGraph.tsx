'use client'

import React, { useCallback, useMemo, useRef, useState } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'
import type { GraphEdge, GraphNode } from '@/types/graph'

// Premium color palette with richer, more sophisticated hues
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
        x?: number
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

export default function ThreeGraph({ nodes, edges, onNodeClick }: ThreeGraphProps) {
    const fgRef = useRef<any>()
    const [hoveredNode, setHoveredNode] = useState<string | null>(null)

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
                val: Math.max(1, (node.degree || 1) * 2),
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

    // Custom node rendering with premium glass-like glow effect
    const nodeThreeObject = useCallback((node: any) => {
        const group = new THREE.Group()
        const baseSize = node.val || 4

        // Core sphere with premium metallic effect
        const geometry = new THREE.SphereGeometry(baseSize, 32, 32)
        const color = new THREE.Color(node.color)

        // Use MeshStandardMaterial for better lighting response
        const coreMaterial = new THREE.MeshStandardMaterial({
            color: color,
            metalness: 0.3,
            roughness: 0.4,
            emissive: color,
            emissiveIntensity: 0.15,
        })

        const sphere = new THREE.Mesh(geometry, coreMaterial)
        group.add(sphere)

        // Inner glow layer
        const innerGlowGeometry = new THREE.SphereGeometry(baseSize * 1.2, 24, 24)
        const innerGlowMaterial = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.25,
            side: THREE.BackSide,
        })
        const innerGlow = new THREE.Mesh(innerGlowGeometry, innerGlowMaterial)
        group.add(innerGlow)

        // Outer glow layer for depth
        const outerGlowGeometry = new THREE.SphereGeometry(baseSize * 1.6, 16, 16)
        const outerGlowMaterial = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.08,
            side: THREE.BackSide,
        })
        const outerGlow = new THREE.Mesh(outerGlowGeometry, outerGlowMaterial)
        group.add(outerGlow)

        // Create text sprite for label with improved styling
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')!
        const fontSize = 48
        const label = node.label || node.id

        canvas.width = 512
        canvas.height = 72

        // Text outline for better readability
        ctx.font = `600 ${fontSize}px 'Inter', 'Noto Sans', sans-serif`
        ctx.strokeStyle = '#000000'
        ctx.lineWidth = 4
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.strokeText(label.substring(0, 30), canvas.width / 2, canvas.height / 2)

        // Main text
        ctx.fillStyle = '#F1F5F9' // slate-100
        ctx.fillText(label.substring(0, 30), canvas.width / 2, canvas.height / 2)

        const texture = new THREE.CanvasTexture(canvas)
        const spriteMaterial = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            depthTest: false,
        })
        const sprite = new THREE.Sprite(spriteMaterial)
        sprite.scale.set(45, 6, 1)
        sprite.position.y = -baseSize - 10

        group.add(sprite)

        return group
    }, [])

    return (
        <div className="relative w-full h-full">
            {/* 3D Graph */}
            <ForceGraph3D
                ref={fgRef}
                graphData={graphData}
                nodeLabel=""
                nodeThreeObject={nodeThreeObject}
                nodeThreeObjectExtend={false}
                linkOpacity={0.4}
                linkWidth={1.5}
                linkColor={() => 'rgba(100, 116, 139, 0.7)'}
                backgroundColor="#0a0a12"
                showNavInfo={false}
                onNodeClick={handleNodeClick}
                onNodeHover={(node) => setHoveredNode(node?.id || null)}
                d3AlphaDecay={0.02}
                d3VelocityDecay={0.3}
                warmupTicks={100}
                cooldownTicks={0}
            />

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
