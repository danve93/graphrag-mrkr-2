"use client"

/* eslint-disable @next/next/no-img-element */

import { useEffect, useState } from 'react'
import { XMarkIcon, ArrowTopRightOnSquareIcon } from '@heroicons/react/24/outline'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

type DocumentPreviewProps = {
  previewUrl?: string | null
  mimeType?: string
  content?: string | null
  onClose?: () => void
}

const isPdf = (mimeType?: string) => mimeType?.includes('pdf') ?? false
const isImage = (mimeType?: string) => mimeType?.startsWith('image/') ?? false
const isMarkdown = (mimeType?: string) =>
  (mimeType?.includes('markdown') || mimeType?.includes('text/markdown') || mimeType?.includes('text/x-markdown')) ?? false
const isPlainText = (mimeType?: string) => !mimeType ? false : mimeType === 'text/plain'
const isTextPreviewable = (mimeType?: string) => isMarkdown(mimeType) || isPlainText(mimeType)
const isOfficeDoc = (mimeType?: string) => {
  if (!mimeType) return false
  return (
    mimeType.includes('officedocument') || // Microsoft Office formats
    mimeType.includes('ms-word') ||
    mimeType.includes('ms-excel') ||
    mimeType.includes('ms-powerpoint') ||
    mimeType.includes('application/vnd.ms-') ||
    mimeType === 'application/msword' ||
    mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
    mimeType === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' ||
    mimeType === 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
  )
}

export default function DocumentPreview({ previewUrl, mimeType, content, onClose }: DocumentPreviewProps) {
  const [htmlContent, setHtmlContent] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    const handleEsc = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose?.()
      }
    }

    document.addEventListener('keydown', handleEsc)
    return () => document.removeEventListener('keydown', handleEsc)
  }, [onClose])

  // Convert markdown content to HTML and wrap in a complete HTML document
  useEffect(() => {
    if (!content && !previewUrl) return

    setIsLoading(true)

    const processMarkdown = async () => {
      try {
        let markdownText = content

        // If no content but we have a URL, fetch it
        if (!markdownText && previewUrl) {
          const res = await fetch(previewUrl)
          if (!res.ok) throw new Error('Failed to fetch content')
          markdownText = await res.text()
        }

        if (!markdownText) {
          setHtmlContent('<p>No content available</p>')
          setIsLoading(false)
          return
        }

        // Convert markdown to HTML
        const rawHtml = await marked(markdownText)

        // Sanitize HTML to prevent XSS
        const cleanHtml = DOMPurify.sanitize(rawHtml)

        // Wrap in a complete HTML document with styling
        const fullHtml = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Document Preview</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
      line-height: 1.6;
      background: #121212;
      color: #f3f4f6;
      padding: 2rem;
      max-width: 900px;
      margin: 0 auto;
    }
    
    a {
      color: #60a5fa;
      text-decoration: none;
    }
    
    a:visited {
      color: #a78bfa;
    }
    
    a:hover {
      text-decoration: underline;
    }
    
    code {
      background: #1e1e1e;
      color: #e5e7eb;
      padding: 0.2rem 0.4rem;
      border-radius: 4px;
      font-family: 'Monaco', 'Courier New', monospace;
      font-size: 0.9em;
    }
    
    pre {
      background: #1e1e1e;
      color: #e5e7eb;
      padding: 1rem;
      border-radius: 8px;
      overflow-x: auto;
      margin-bottom: 1rem;
      font-family: 'Monaco', 'Courier New', monospace;
    }
    
    pre code {
      background: none;
      padding: 0;
      border-radius: 0;
    }
    
    blockquote {
      border-left: 4px solid #4b5563;
      padding-left: 1rem;
      margin-left: 0;
      margin-bottom: 1rem;
      color: #d1d5db;
    }
    
    table {
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 1rem;
      border: 1px solid #4b5563;
    }
    
    th, td {
      border: 1px solid #4b5563;
      padding: 0.75rem;
      text-align: left;
    }
    
    th {
      background: #1e1e1e;
      font-weight: 600;
    }
    
    h1, h2, h3, h4, h5, h6 {
      margin-top: 1.5rem;
      margin-bottom: 1rem;
      font-weight: 600;
      color: #f3f4f6;
    }
    
    h1 {
      font-size: 2rem;
      border-bottom: 2px solid #4b5563;
      padding-bottom: 0.5rem;
    }
    
    h2 {
      font-size: 1.5rem;
    }
    
    h3 {
      font-size: 1.25rem;
    }
    
    p {
      margin-bottom: 1rem;
    }
    
    ul, ol {
      margin-left: 2rem;
      margin-bottom: 1rem;
    }
    
    li {
      margin-bottom: 0.5rem;
    }
    
    img {
      max-width: 100%;
      height: auto;
      margin: 1rem 0;
    }
    
    hr {
      border: none;
      border-top: 1px solid #4b5563;
      margin: 2rem 0;
    }
  </style>
</head>
<body>
${cleanHtml}
</body>
</html>`

        setHtmlContent(fullHtml)
      } catch (error) {
        console.error('Failed to process markdown:', error)
        setHtmlContent('<p>Failed to process markdown content</p>')
      } finally {
        setIsLoading(false)
      }
    }

    processMarkdown()
  }, [content, previewUrl])

  // Generate Office Online viewer URL for Office documents
  const getOfficeViewerUrl = (url: string) => {
    // Use Microsoft Office Online viewer
    return `https://view.officeapps.live.com/op/embed.aspx?src=${encodeURIComponent(url)}`
  }

  const renderPreviewContent = () => {
    if (isLoading) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="text-secondary-600 dark:text-secondary-400">Processing markdown...</div>
        </div>
      )
    }

    if (!htmlContent) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="text-secondary-600 dark:text-secondary-400">No content available</div>
        </div>
      )
    }

    // For markdown and text, render as HTML in iframe using srcdoc
    if (isMarkdown(mimeType) || isPlainText(mimeType)) {
      return (
        <iframe
          srcDoc={htmlContent}
          title="Markdown preview"
          className="w-full h-full border-none rounded-lg"
          sandbox="allow-same-origin allow-scripts"
        />
      )
    }

    // If we have a previewUrl for other file types
    if (previewUrl) {
      if (isPdf(mimeType)) {
        return (
          <iframe
            src={previewUrl}
            title="PDF preview"
            className="w-full h-full rounded-lg border border-secondary-200"
          />
        )
      }

      if (isImage(mimeType)) {
        return (
          <div className="flex items-center justify-center h-full">
            <img
              src={previewUrl}
              alt="Document preview"
              className="max-h-full max-w-full rounded-lg shadow"
            />
          </div>
        )
      }

      if (isOfficeDoc(mimeType)) {
        const isAbsoluteUrl = previewUrl.startsWith('http://') || previewUrl.startsWith('https://')
        if (isAbsoluteUrl) {
          return (
            <div className="w-full h-full space-y-4">
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-800">
                <p className="font-medium">Office Document Viewer</p>
                <p className="text-xs mt-1">
                  Using Microsoft Office Online viewer. If the document doesn&apos;t load,
                  try opening it in a new tab or downloading it.
                </p>
              </div>
              <iframe
                src={getOfficeViewerUrl(previewUrl)}
                title="Office document preview"
                className="w-full h-[calc(100%-5rem)] rounded-lg border border-secondary-200"
              />
            </div>
          )
        }
      }

      return (
        <iframe
          src={previewUrl}
          title="Document preview"
          className="w-full h-full rounded-lg border border-secondary-200 dark:border-secondary-600 bg-white dark:bg-secondary-800"
        />
      )
    }

    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-secondary-600 dark:text-secondary-400">Unable to preview this document</div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4">
      <div className="relative w-full max-w-5xl bg-white dark:bg-secondary-800 rounded-xl shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between border-b border-secondary-200 dark:border-secondary-700 px-4 py-3 bg-secondary-50 dark:bg-secondary-700">
          <div>
            <p className="text-sm font-medium text-secondary-900 dark:text-secondary-50">Document Preview</p>
            <p className="text-xs text-secondary-500 dark:text-secondary-400">Press Escape to close</p>
          </div>
          <div className="flex items-center gap-2">
            {previewUrl && (
              <a
                href={previewUrl}
                target="_blank"
                rel="noreferrer"
                className="button-ghost text-xs flex items-center gap-1"
              >
                <ArrowTopRightOnSquareIcon className="w-4 h-4" />
                Open in new tab
              </a>
            )}
            <button
              type="button"
              onClick={onClose}
              className="button-secondary text-xs flex items-center gap-1"
            >
              <XMarkIcon className="w-4 h-4" />
              Close
            </button>
          </div>
        </div>

        <div className="h-[70vh] bg-secondary-100 dark:bg-secondary-900 p-4 overflow-auto">
          {renderPreviewContent()}
        </div>
      </div>
    </div>
  )
}
