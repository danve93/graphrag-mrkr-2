'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle, XCircle, AlertTriangle, X } from 'lucide-react'

export interface ToastProps {
  id: string
  type: 'success' | 'error' | 'warning'
  message: string
  description?: string
  onDismiss: (id: string) => void
}

export default function Toast({ id, type, message, description, onDismiss }: ToastProps) {
  const styles = {
    success: {
      container: 'border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/30',
      icon: <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />,
      text: 'text-green-900 dark:text-green-300',
      description: 'text-green-700 dark:text-green-400',
      button: 'text-green-600 dark:text-green-400 hover:text-green-800 dark:hover:text-green-300'
    },
    error: {
      container: 'border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/30',
      icon: <XCircle className="h-5 w-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />,
      text: 'text-red-900 dark:text-red-300',
      description: 'text-red-700 dark:text-red-400',
      button: 'text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300'
    },
    warning: {
      container: 'border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/30',
      icon: <AlertTriangle className="h-5 w-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />,
      text: 'text-yellow-900 dark:text-yellow-300',
      description: 'text-yellow-700 dark:text-yellow-400',
      button: 'text-yellow-600 dark:text-yellow-400 hover:text-yellow-800 dark:hover:text-yellow-300'
    }
  }

  const style = styles[type]

  return (
    <motion.div
      initial={{ opacity: 0, y: 50, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className={`flex items-start gap-3 rounded-lg border px-4 py-3 shadow-lg max-w-md ${style.container}`}
    >
      {style.icon}
      <div className="flex-1 min-w-0">
        <p className={`text-sm font-medium ${style.text}`}>
          {message}
        </p>
        {description && (
          <p className={`text-xs mt-1 ${style.description}`}>
            {description}
          </p>
        )}
      </div>
      <button
        onClick={() => onDismiss(id)}
        className={`flex-shrink-0 ${style.button}`}
      >
        <X className="h-4 w-4" />
      </button>
    </motion.div>
  )
}
