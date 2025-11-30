'use client'

import { useState, useCallback, useEffect } from 'react'
import { AnimatePresence } from 'framer-motion'
import Toast, { ToastProps } from './Toast'

interface ToastData {
  id: string
  type: 'success' | 'error' | 'warning'
  message: string
  description?: string
  duration?: number
  position?: 'bottom-right' | 'top-center'
}

let toastCounter = 0
const toastCallbacks: Set<(toast: ToastData) => void> = new Set()

export function showToast(
  type: 'success' | 'error' | 'warning',
  message: string,
  description?: string,
  duration = 5000,
  position: 'bottom-right' | 'top-center' = 'bottom-right'
) {
  const toast: ToastData = {
    id: `toast-${++toastCounter}`,
    type,
    message,
    description,
    duration,
    position,
  }
  toastCallbacks.forEach(cb => cb(toast))
}

export default function ToastContainer() {
  const [toasts, setToasts] = useState<ToastData[]>([])

  const dismissToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  const addToast = useCallback((toast: ToastData) => {
    setToasts(prev => [...prev, toast])
    
    // Auto-dismiss after duration
    if (toast.duration && toast.duration > 0) {
      setTimeout(() => {
        dismissToast(toast.id)
      }, toast.duration)
    }
  }, [dismissToast])

  useEffect(() => {
    toastCallbacks.add(addToast)
    return () => {
      toastCallbacks.delete(addToast)
    }
  }, [addToast])

  const topCenterToasts = toasts.filter(t => t.position === 'top-center')
  const bottomRightToasts = toasts.filter(t => !t.position || t.position === 'bottom-right')

  return (
    <>
      {/* Top Center Toasts */}
      <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 flex flex-col gap-2" style={{ minWidth: '400px', maxWidth: '600px' }}>
        <AnimatePresence>
          {topCenterToasts.map(toast => (
            <Toast
              key={toast.id}
              {...toast}
              onDismiss={dismissToast}
            />
          ))}
        </AnimatePresence>
      </div>
      
      {/* Bottom Right Toasts */}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
        <AnimatePresence>
          {bottomRightToasts.map(toast => (
            <Toast
              key={toast.id}
              {...toast}
              onDismiss={dismissToast}
            />
          ))}
        </AnimatePresence>
      </div>
    </>
  )
}
