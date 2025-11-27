'use client'

import { useState } from 'react'
import { InformationCircleIcon } from '@heroicons/react/24/outline'

/**
 * PANEL LAYOUT TEMPLATE
 * 
 * This template demonstrates the unified layout pattern for all panel components.
 * 
 * STRUCTURE:
 * - Root container: h-full flex flex-col (fills parent, vertical stacking)
 * - Header (optional): flex-shrink-0 (fixed height, never collapses)
 * - Content: flex-1 min-h-0 overflow-y-auto (flexible height, scrollable)
 * - Footer (optional): flex-shrink-0 (fixed height, never collapses)
 * 
 * KEY CONCEPTS:
 * 1. h-full: Component fills 100% of parent height (from grid-cell in page.tsx)
 * 2. flex flex-col: Children stack vertically
 * 3. flex-shrink-0: Element keeps its size and never shrinks
 * 4. flex-1: Element takes remaining available space
 * 5. min-h-0: Critical for flex children to enable proper scrolling
 * 6. overflow-y-auto: Enables vertical scrolling when content exceeds height
 * 
 * USAGE:
 * Copy this structure for all panel components. Adjust header/footer as needed.
 */

export default function PanelTemplate() {
  const [data, setData] = useState<string[]>([
    'Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5',
    'Item 6', 'Item 7', 'Item 8', 'Item 9', 'Item 10',
    'Item 11', 'Item 12', 'Item 13', 'Item 14', 'Item 15',
    'Item 16', 'Item 17', 'Item 18', 'Item 19', 'Item 20',
  ])

  return (
    <div className="h-full flex flex-col bg-white dark:bg-secondary-900">
      
      {/* ============================================
          STICKY HEADER (Optional)
          - flex-shrink-0: Stays visible, doesn't collapse
          - border-b: Visual separator
          ============================================ */}
      <div className="flex-shrink-0 border-b border-secondary-200 dark:border-secondary-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-secondary-900 dark:text-secondary-50 mb-2">
              Panel Title
            </h1>
            <p className="text-sm text-secondary-600 dark:text-secondary-400">
              This is the panel description. Headers stay fixed at the top.
            </p>
          </div>
          
          {/* Header actions (buttons, etc.) */}
          <div className="flex items-center gap-2">
            <button className="button-secondary text-sm">
              Cancel
            </button>
            <button className="button-primary text-sm">
              Save Changes
            </button>
          </div>
        </div>

        {/* Optional: Sub-header elements like tabs or filters */}
        <div className="mt-4 flex items-center gap-2">
          <div className="flex items-center gap-2 rounded-lg bg-secondary-100 dark:bg-secondary-800 px-3 py-2 text-xs text-secondary-700 dark:text-secondary-300">
            <InformationCircleIcon className="h-4 w-4" />
            <span>This is an info banner in the header</span>
          </div>
        </div>
      </div>

      {/* ============================================
          SCROLLABLE CONTENT AREA
          - flex-1: Takes all remaining vertical space
          - min-h-0: CRITICAL - allows flex child to scroll
          - overflow-y-auto: Enables scrolling when content overflows
          ============================================ */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        
        {/* Inner content wrapper with padding */}
        <div className="px-6 py-6 space-y-6">
          
          {/* Content sections */}
          <section className="bg-white dark:bg-secondary-800 rounded-lg shadow-sm border border-secondary-200 dark:border-secondary-700 p-5">
            <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-50 mb-4">
              Content Section 1
            </h2>
            <p className="text-sm text-secondary-600 dark:text-secondary-400">
              All content goes inside the scrollable area. The header and footer stay fixed.
            </p>
          </section>

          <section className="bg-white dark:bg-secondary-800 rounded-lg shadow-sm border border-secondary-200 dark:border-secondary-700 p-5">
            <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-50 mb-4">
              Content Section 2 - List Example
            </h2>
            <div className="space-y-2">
              {data.map((item, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-secondary-50 dark:bg-secondary-900 rounded-lg"
                >
                  <span className="text-sm text-secondary-900 dark:text-secondary-100">
                    {item}
                  </span>
                  <button className="text-xs text-red-600 hover:text-red-700">
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </section>

          <section className="bg-white dark:bg-secondary-800 rounded-lg shadow-sm border border-secondary-200 dark:border-secondary-700 p-5">
            <h2 className="text-lg font-semibold text-secondary-900 dark:text-secondary-50 mb-4">
              Content Section 3
            </h2>
            <p className="text-sm text-secondary-600 dark:text-secondary-400 mb-4">
              Add as many sections as needed. The content area will scroll automatically
              when the content height exceeds the available space.
            </p>
            
            {/* Form example */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">
                  Input Field
                </label>
                <input
                  type="text"
                  placeholder="Enter text..."
                  className="w-full px-4 py-2 border border-secondary-300 dark:border-secondary-600 rounded-lg bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">
                  Textarea Field
                </label>
                <textarea
                  rows={4}
                  placeholder="Enter longer text..."
                  className="w-full px-4 py-2 border border-secondary-300 dark:border-secondary-600 rounded-lg bg-white dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100"
                />
              </div>
            </div>
          </section>

        </div>
      </div>

      {/* ============================================
          STICKY FOOTER (Optional)
          - flex-shrink-0: Stays visible, doesn't collapse
          - border-t: Visual separator
          - Common use: Action buttons, save/cancel, pagination
          ============================================ */}
      <div className="flex-shrink-0 border-t border-secondary-200 dark:border-secondary-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="text-sm text-secondary-600 dark:text-secondary-400">
            {data.length} items total
          </div>
          
          <div className="flex gap-3">
            <button className="button-secondary">
              Reset
            </button>
            <button className="button-primary">
              Save All Changes
            </button>
          </div>
        </div>
      </div>

    </div>
  )
}


/**
 * VARIANTS & COMMON PATTERNS
 * ============================================
 */

/**
 * 1. SIMPLE PANEL (No header/footer)
 * 
 * <div className="h-full overflow-y-auto bg-white dark:bg-secondary-900">
 *   <div className="px-6 py-6">
 *     Content here
 *   </div>
 * </div>
 */

/**
 * 2. PANEL WITH HEADER ONLY
 * 
 * <div className="h-full flex flex-col bg-white dark:bg-secondary-900">
 *   <div className="flex-shrink-0 border-b px-6 py-4">
 *     Header content
 *   </div>
 *   <div className="flex-1 min-h-0 overflow-y-auto px-6 py-6">
 *     Scrollable content
 *   </div>
 * </div>
 */

/**
 * 3. PANEL WITH FOOTER ONLY
 * 
 * <div className="h-full flex flex-col bg-white dark:bg-secondary-900">
 *   <div className="flex-1 min-h-0 overflow-y-auto px-6 py-6">
 *     Scrollable content
 *   </div>
 *   <div className="flex-shrink-0 border-t px-6 py-4">
 *     Footer content
 *   </div>
 * </div>
 */

/**
 * 4. CHAT-SPECIFIC PATTERN (Messages + Input)
 * 
 * <div className="h-full flex flex-col">
 *   <ConnectionStatus />  // Optional banner at top
 *   <div className="flex-1 min-h-0 overflow-y-auto px-6 py-6">
 *     Messages list
 *   </div>
 *   <div className="flex-shrink-0 border-t bg-white px-6 py-4">
 *     <ChatInput />
 *   </div>
 * </div>
 */

/**
 * 5. SPLIT PANEL (Sidebar + Content)
 * 
 * <div className="h-full flex">
 *   <div className="w-64 flex-shrink-0 border-r overflow-y-auto">
 *     Sidebar navigation
 *   </div>
 *   <div className="flex-1 min-w-0 overflow-y-auto">
 *     Main content
 *   </div>
 * </div>
 */

/**
 * TROUBLESHOOTING
 * ============================================
 * 
 * Problem: Content doesn't scroll
 * Solution: Add `min-h-0` to the flex child with `overflow-y-auto`
 * 
 * Problem: Footer/header disappears
 * Solution: Add `flex-shrink-0` to prevent collapsing
 * 
 * Problem: Content overflows viewport
 * Solution: Ensure parent has `h-full` and uses `flex flex-col`
 * 
 * Problem: Scroll jumps or layout shifts
 * Solution: Use `flex-1 min-h-0` on scrollable area, avoid nested `h-full`
 */
