'use client'

import { useEffect, useState } from 'react'
import BottomDock from './BottomDock'

/**
 * Wrapper component that conditionally renders BottomDock.
 * Hides BottomDock completely when in external view (?view=external).
 */
export default function ConditionalBottomDock() {
    const [isExternalView, setIsExternalView] = useState(false)
    const [mounted, setMounted] = useState(false)

    useEffect(() => {
        setMounted(true)
        if (typeof window !== 'undefined') {
            const params = new URLSearchParams(window.location.search)
            setIsExternalView(params.get('view') === 'external')
        }
    }, [])

    // Don't render anything during SSR to avoid hydration mismatch
    if (!mounted) return null

    // In external view, render nothing (BottomDock completely hidden)
    if (isExternalView) return null

    return <BottomDock />
}
