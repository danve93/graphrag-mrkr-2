'use client'

import React from 'react'

export default function Tooltip({ children, content }: { children: React.ReactNode; content: string }) {
  return (
    <span className="tooltip">
      {children}
      <span className="tooltip-content">{content}</span>
    </span>
  )
}
