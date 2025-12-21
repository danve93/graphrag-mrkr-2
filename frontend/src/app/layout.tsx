import type { Metadata } from 'next'
import './globals.css'
import ToastContainer from '@/components/Toast/ToastContainer'
import { ThemeProvider } from '@/components/Theme/ThemeProvider'
import StatusIndicator from '@/components/Theme/StatusIndicator'
import ConditionalBottomDock from '@/components/Navigation/ConditionalBottomDock'
import KeyboardShortcutHint from '@/components/Utils/KeyboardShortcutHint'
import fs from 'fs'
import path from 'path'
import BrandingProvider from '@/components/Branding/BrandingProvider'

// Read branding.json from the Next.js `public` folder at build/server time
const brandingPath = path.join(process.cwd(), 'public', 'branding.json')
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
  icons: {
    icon: [
      {
        url: branding.use_image && branding.image_path ? branding.image_path : '/favicon.svg',
        type: 'image/svg+xml'
      }
    ]
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
      </head>
      <body className="min-h-screen font-sans">
        <ThemeProvider>
          <BrandingProvider branding={branding}>
            <StatusIndicator />
            {children}
            <ConditionalBottomDock />
            <KeyboardShortcutHint />
            <ToastContainer />
          </BrandingProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
