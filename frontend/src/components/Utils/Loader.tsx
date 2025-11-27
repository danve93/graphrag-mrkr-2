'use client'

import React from 'react'

export default function Loader({ size = 6, label }: { size?: number; label?: string }) {
  const px = `${size}px`
  return (
    <div className="flex items-center gap-2">
      <div className="spinner" style={{ width: px, height: px }} aria-hidden />
      {label ? <span className="text-sm text-secondary-400">{label}</span> : null}
    </div>
  )
}
