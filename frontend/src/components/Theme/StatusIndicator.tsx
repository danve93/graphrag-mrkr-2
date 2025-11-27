'use client'

import React from 'react'
import { useChatStore } from '@/store/chatStore'
import Loader from '@/components/Utils/Loader'

export default function StatusIndicator() {
  const isConnected = useChatStore((s) => s.isConnected)

  return (
    <div className="fixed top-4 right-4 z-50 flex items-center gap-3">
      {isConnected ? (
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: '#24c6e6' }} />
          <span className="text-sm text-secondary-200">Connected</span>
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full bg-secondary-700" />
          <span className="text-sm text-secondary-500">Disconnected</span>
        </div>
      )}
    </div>
  )
}
