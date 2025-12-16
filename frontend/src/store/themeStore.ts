import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type Theme = 'light' | 'dark' | 'auto'

interface ThemeStore {
  theme: Theme
  setTheme: (theme: Theme) => void
  isDark: boolean
  setIsDark: (isDark: boolean) => void
}

export const useThemeStore = create<ThemeStore>()(
  persist(
    (set, get) => ({
      theme: 'dark',
      isDark: true,
      setTheme: (theme: Theme) => {
        set({ theme: 'dark', isDark: true })
        applyTheme()
      },
      setIsDark: (isDark: boolean) => {
        set({ isDark: true })
        applyTheme()
      },
    }),
    {
      name: 'theme-store',
      partialize: (state) => ({ theme: state.theme }),
    }
  )
)

export function applyTheme() {
  if (typeof document === 'undefined') return
  const html = document.documentElement
  html.classList.add('dark')
  html.dataset.theme = 'dark'
}

export function initializeTheme() {
  // Always enforce dark mode
  applyTheme()
  return () => { }
}
