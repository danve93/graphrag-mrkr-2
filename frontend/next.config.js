/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    // Default to empty so client-side code uses relative paths by default.
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || process.env.API_URL || '',
  },
  // Rewrites are only used when NEXT_PUBLIC_API_URL is empty (relative requests).
  // When NEXT_PUBLIC_API_URL is set to an absolute URL, client-side code calls
  // the API directly and rewrites are not needed.
  async rewrites() {
    // Disable rewrites if using an absolute API URL (e.g., production with external hostname)
    if (process.env.NEXT_PUBLIC_API_URL) {
      return []
    }
    
    // If running in development on the host machine, default to localhost:8000
    const serverApiUrl =
      process.env.NEXT_PUBLIC_API_URL_SERVER ||
      process.env.API_URL ||
      (process.env.NODE_ENV === 'development' ? 'http://localhost:8000' : 'http://backend:8000')
    return [
      {
        source: '/api/:path*',
        destination: `${serverApiUrl}/api/:path*`,
      },
    ]
  },
}

module.exports = nextConfig
