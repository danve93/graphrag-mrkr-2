'use client'

import { useEffect, useState } from 'react'
import { UserCircleIcon, ArrowRightOnRectangleIcon } from '@heroicons/react/24/outline'
import { useChatStore } from '@/store/chatStore'
import { api } from '@/lib/api'
import Tooltip from '@/components/Utils/Tooltip'

export default function SidebarUser() {
    const { user, identifyUser, setActiveView, logout } = useChatStore()
    const [loading, setLoading] = useState(false)

    // Auto-identify if no user exists
    useEffect(() => {
        if (!user && !loading) {
            // Check if we have one in local storage not yet loaded? 
            // Store init handles that.
            // So if user is null here, we truly have no user.
            setLoading(true)
            identifyUser().finally(() => setLoading(false))
        }
    }, [user, identifyUser, loading])

    const handleLogout = async () => {
        if (confirm("Are you sure you want to logout?")) {
            await api.adminLogout() // Clear admin session cookie
            logout()
            // Redirect to admin login if needed, or just let page.tsx handle it (it will show login screen)
            window.location.href = '/admin'
        }
    }

    if (!user) return <div className="p-3 text-xs text-gray-400">Loading user...</div>

    return (
        <div className="flex flex-col p-3 border-t border-[var(--border)] bg-[var(--bg-secondary)]">
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2 overflow-hidden">
                    <UserCircleIcon className="w-5 h-5 text-[var(--accent)] flex-shrink-0" />
                    <div className="flex flex-col overflow-hidden">
                        <span className="text-xs font-medium text-[var(--text-primary)] truncate max-w-[120px]" title={user.id}>
                            {user.id}
                        </span>
                        <span className="text-[10px] text-[var(--text-secondary)]">
                            {user.token.substring(0, 8)}...
                        </span>
                    </div>
                </div>

                <Tooltip content="Logout">
                    <button
                        onClick={handleLogout}
                        className="p-1.5 hover:bg-[var(--bg-hover)] rounded-md transition-colors text-[var(--text-secondary)]"
                    >
                        <ArrowRightOnRectangleIcon className="w-4 h-4" />
                    </button>
                </Tooltip>
            </div>


        </div>
    )
}
