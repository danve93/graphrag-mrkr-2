import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface UIStore {
    showSuggestionIndicators: boolean
    setShowSuggestionIndicators: (show: boolean) => void
}

export const useUIStore = create<UIStore>()(
    persist(
        (set) => ({
            showSuggestionIndicators: false,
            setShowSuggestionIndicators: (show) => set({ showSuggestionIndicators: show }),
        }),
        {
            name: 'ui-store',
        }
    )
)
