import type { Metadata } from 'next'
import './globals.css'
import ToastContainer from '@/components/Toast/ToastContainer'
import { ThemeProvider } from '@/components/Theme/ThemeProvider'
import fs from 'fs'
import path from 'path'
import BrandingProvider from '@/components/Branding/BrandingProvider'

// Read branding.json from the Next.js `public` folder at build/server time
const brandingPath = path.join(process.cwd(), 'frontend', 'public', 'branding.json')
let branding = { title: 'GraphRAG', description: '', use_image: false } as any
try {
  const raw = fs.readFileSync(brandingPath, 'utf8')
  branding = JSON.parse(raw)
} catch (err) {
  // fallback defaults are already set
}

export const metadata: Metadata = {
  title: branding.title,
  description: branding.description,
  icons: branding.use_image ? { icon: branding.image_path, shortcut: branding.image_path } : undefined,
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider>
          <BrandingProvider branding={branding}>
            {children}
            <ToastContainer />
          </BrandingProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}
