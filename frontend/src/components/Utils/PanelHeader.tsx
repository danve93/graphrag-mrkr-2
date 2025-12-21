'use client'

import { LucideIcon } from 'lucide-react'

interface PanelHeaderProps {
    icon: LucideIcon
    title: string
    subtitle?: string
    actions?: React.ReactNode
}

/**
 * Standardized panel header component used across all main views.
 * Provides consistent styling for icon container, title, subtitle, and optional actions.
 */
export default function PanelHeader({ icon: Icon, title, subtitle, actions }: PanelHeaderProps) {
    return (
        <div style={{ borderBottom: '1px solid var(--border)', padding: 'var(--space-6)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div
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
                    <Icon size={24} style={{ color: 'var(--accent-primary)' }} />
                </div>
                <div style={{ flex: 1 }}>
                    <h1
                        className="font-display"
                        style={{
                            fontSize: 'var(--text-2xl)',
                            fontWeight: 700,
                            color: 'var(--text-primary)',
                            margin: 0,
                        }}
                    >
                        {title}
                    </h1>
                    {subtitle && (
                        <p
                            style={{
                                fontSize: 'var(--text-sm)',
                                color: 'var(--text-secondary)',
                                margin: 0,
                            }}
                        >
                            {subtitle}
                        </p>
                    )}
                </div>
                {actions && <div className="flex items-center gap-2">{actions}</div>}
            </div>
        </div>
    )
}
