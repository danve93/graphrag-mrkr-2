"use client"

import React, { createContext, useContext } from 'react'

export type Branding = {
  title?: string
  description?: string
  heading?: string
  tagline?: string
  use_image?: boolean
  image_path?: string
  short_name?: string
}

const defaultBranding: Branding = {
  title: 'GraphRAG',
  description: '',
  heading: 'GraphRAG',
  tagline: '',
  use_image: false,
  image_path: '',
  short_name: 'GraphRAG',
}

const BrandingContext = createContext<Branding>(defaultBranding)

export function BrandingProvider({ branding, children }: { branding: Branding; children: React.ReactNode }) {
  return <BrandingContext.Provider value={branding || defaultBranding}>{children}</BrandingContext.Provider>
}

export function useBranding() {
  return useContext(BrandingContext)
}

export default BrandingProvider
