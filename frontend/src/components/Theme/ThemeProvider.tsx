'use client'

import { useEffect } from 'react'
import { initializeTheme } from '@/store/themeStore'

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    const cleanup = initializeTheme()
    return cleanup
  }, [])

  return <>{children}</>
}
