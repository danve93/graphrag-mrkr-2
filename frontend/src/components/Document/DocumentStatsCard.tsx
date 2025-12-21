'use client'

import { motion } from 'framer-motion'
import { LucideIcon } from 'lucide-react'

export interface StatsCardProps {
    value: number
    label: string
    color?: string
    icon?: LucideIcon
}

export default function DocumentStatsCard({ value, label, color = 'var(--text-primary)', icon: Icon }: StatsCardProps) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-lg p-5 border shadow-sm"
            style={{
                backgroundColor: 'var(--bg-secondary)',
                borderColor: 'var(--border)',
            }}
        >
            <div className="flex items-start justify-between">
                <div>
                    <div className="text-3xl font-bold mb-1" style={{ color }}>
                        {value.toLocaleString()}
                    </div>
                    <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                        {label}
                    </div>
                </div>
                {Icon && (
                    <div
                        className="w-10 h-10 rounded-lg flex items-center justify-center"
                        style={{ backgroundColor: `${color}15` }}
                    >
                        <Icon className="w-5 h-5" style={{ color }} />
                    </div>
                )}
            </div>
        </motion.div>
    )
}
