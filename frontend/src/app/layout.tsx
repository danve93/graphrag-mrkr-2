import type { Metadata } from 'next'
import './globals.css'
import ToastContainer from '@/components/Toast/ToastContainer'
import { ThemeProvider } from '@/components/Theme/ThemeProvider'
import branding from '../../../branding.json'

export const metadata: Metadata = {
  title: branding.title,
  description: branding.description,
  icons: branding.use_image ? { icon: branding.image_path, shortcut: branding.image_path } : undefined,
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider>
          {children}
          <ToastContainer />
        </ThemeProvider>
      </body>
    </html>
  )
}
