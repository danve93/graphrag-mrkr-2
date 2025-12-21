'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Database, FileText, Layers, Network, Link } from 'lucide-react'
import { api } from '@/lib/api'
import DocumentStatsCard from './DocumentStatsCard'

interface GlobalStats {
    total_documents: number
    total_chunks: number
    total_entities: number
    total_relationships: number
}

export default function DocumentEmptyState() {
    const [globalStats, setGlobalStats] = useState<GlobalStats | null>(null)

    useEffect(() => {
        const loadGlobalStats = () => {
            api.getStats().then((data) => {
                setGlobalStats({
                    total_documents: data.total_documents || 0,
                    total_chunks: data.total_chunks || 0,
                    total_entities: data.total_entities || 0,
                    total_relationships: data.total_relationships || 0,
                })
            }).catch(() => {
                // Silently fail
            })
        }

        loadGlobalStats()

        const handleStatsRefresh = () => {
            loadGlobalStats()
        }

        if (typeof window !== 'undefined') {
            window.addEventListener('documents:processed', handleStatsRefresh)
            window.addEventListener('documents:processing-updated', handleStatsRefresh)
            window.addEventListener('documents:uploaded', handleStatsRefresh)
            window.addEventListener('server:reconnected', handleStatsRefresh)
        }

        const pollInterval = window.setInterval(loadGlobalStats, 5000)

        return () => {
            if (typeof window !== 'undefined') {
                window.removeEventListener('documents:processed', handleStatsRefresh)
                window.removeEventListener('documents:processing-updated', handleStatsRefresh)
                window.removeEventListener('documents:uploaded', handleStatsRefresh)
                window.removeEventListener('server:reconnected', handleStatsRefresh)
            }
            window.clearInterval(pollInterval)
        }
    }, [])

    return (
        <div className="h-full flex flex-col" style={{ background: 'var(--bg-primary)' }}>
            {/* Header */}
            <div style={{ borderBottom: '1px solid var(--border)', padding: 'var(--space-6)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        style={{
                            width: '40px',
                            height: '40px',
                            borderRadius: '8px',
                            backgroundColor: 'var(--accent-subtle)',
                            border: '1px solid var(--accent-primary)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                        }}
                    >
                        <Database className="w-6 h-6" style={{ color: 'var(--accent-primary)' }} />
                    </motion.div>
                    <div style={{ flex: 1 }}>
                        <h1
                            className="font-display"
                            style={{ fontSize: 'var(--text-2xl)', fontWeight: 700, color: 'var(--text-primary)' }}
                        >
                            Database
                        </h1>
                        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
                            Document repository with chunks, entities, and relationships
                        </p>
                    </div>
                </div>
            </div>

            {/* Stats Cards */}
            <div className="flex-1 p-6">
                <h2 className="text-lg font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
                    Knowledge Base Overview
                </h2>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    <DocumentStatsCard
                        value={globalStats?.total_documents || 0}
                        label="Documents"
                        color="var(--accent-primary)"
                        icon={FileText}
                    />
                    <DocumentStatsCard
                        value={globalStats?.total_chunks || 0}
                        label="Chunks"
                        color="var(--text-primary)"
                        icon={Layers}
                    />
                    <DocumentStatsCard
                        value={globalStats?.total_entities || 0}
                        label="Entities"
                        color="#32D74B"
                        icon={Network}
                    />
                    <DocumentStatsCard
                        value={globalStats?.total_relationships || 0}
                        label="Relationships"
                        color="#BF5AF2"
                        icon={Link}
                    />
                </div>

                {/* Empty State */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="flex flex-col items-center justify-center py-16 mt-8 rounded-lg border"
                    style={{ borderColor: 'var(--border)', borderStyle: 'dashed' }}
                >
                    <motion.div
                        animate={{
                            scale: [1, 1.05, 1],
                            opacity: [0.7, 1, 0.7],
                        }}
                        transition={{
                            duration: 3,
                            repeat: Infinity,
                            ease: 'easeInOut',
                        }}
                    >
                        <FileText className="w-12 h-12 mb-3" style={{ color: 'var(--text-tertiary)' }} />
                    </motion.div>
                    <p className="text-base font-medium" style={{ color: 'var(--text-secondary)' }}>
                        Select a document from the sidebar to view its details
                    </p>
                    <p className="text-sm mt-1" style={{ color: 'var(--text-tertiary)' }}>
                        Or upload a new document to get started
                    </p>
                </motion.div>
            </div>
        </div>
    )
}
