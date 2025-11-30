'use client'

import React, { useRef, useState, useEffect, isValidElement, cloneElement } from 'react'
import { createPortal } from 'react-dom'

export default function Tooltip({ children, content }: { children: React.ReactNode; content: string }) {
  const triggerRef = useRef<HTMLElement | null>(null)
  const tooltipRef = useRef<HTMLSpanElement>(null)
  const [position, setPosition] = useState({ top: 0, left: 0, transform: 'translateX(-50%) translateY(-4px)' })
  const [isVisible, setIsVisible] = useState(false)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  const computePosition = () => {
    const el = triggerRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const viewportWidth = window.innerWidth

    let top = rect.top - 8
    let left = rect.left + rect.width / 2
    let transform = 'translateX(-50%) translateY(-4px)'

    const estimatedWidth = Math.min(content.length * 7, 400)
    const tooltipLeft = left - estimatedWidth / 2
    const tooltipRight = left + estimatedWidth / 2

    if (tooltipLeft < 10) {
      left = 10
      transform = 'translateX(0) translateY(-4px)'
    } else if (tooltipRight > viewportWidth - 10) {
      left = viewportWidth - 10
      transform = 'translateX(-100%) translateY(-4px)'
    }

    if (top < 40) {
      top = rect.bottom + 8
      transform = transform.replace('translateY(-4px)', 'translateY(4px)')
    }

    setPosition({ top, left, transform })
  }

  const handleMouseEnter = () => {
    computePosition()
    setIsVisible(true)
  }

  const handleMouseLeave = () => {
    setIsVisible(false)
  }

  const tooltipContent = isVisible && mounted ? (
    <span 
      ref={tooltipRef}
      className="tooltip-content" 
      style={{ 
        top: `${position.top}px`, 
        left: `${position.left}px`,
        visibility: 'visible',
        opacity: 1,
        transform: position.transform
      }}
    >
      {content}
    </span>
  ) : null

  // If children is a single DOM element, clone it and attach handlers/ref
  if (isValidElement(children) && typeof children !== 'string') {
    try {
      const child = children as any
      const existingOnEnter = child.props?.onMouseEnter
      const existingOnLeave = child.props?.onMouseLeave

      const cloned = cloneElement(child, {
        ref: (node: HTMLElement | null) => {
          triggerRef.current = node
          const origRef = child.ref
          if (typeof origRef === 'function') origRef(node)
          else if (origRef && typeof origRef === 'object') origRef.current = node
        },
        onMouseEnter: (e: any) => {
          handleMouseEnter()
          if (existingOnEnter) existingOnEnter(e)
        },
        onMouseLeave: (e: any) => {
          handleMouseLeave()
          if (existingOnLeave) existingOnLeave(e)
        }
      })

      return (
        <>
          {cloned}
          {mounted && createPortal(tooltipContent, document.body)}
        </>
      )
    } catch (e) {
      // Fallback to wrapper approach if cloning fails
    }
  }

  // Fallback: wrap children in a span trigger
  return (
    <>
      <span 
        className="tooltip-trigger" 
        ref={triggerRef as any} 
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        style={{ position: 'relative', display: 'inline-block' }}
      >
        {children}
      </span>
      {mounted && createPortal(tooltipContent, document.body)}
    </>
  )
}
