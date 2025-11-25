/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    // Default to empty so client-side code uses relative paths by default.
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || process.env.API_URL || '',
  },
  // Rewrites are executed by the Next.js server at runtime; use a server-side
  // API URL (separate env var) as the destination so the server proxies
  // `/api/:path*` to the backend service inside Docker. Keep the client
  // env `NEXT_PUBLIC_API_URL` empty so browser code issues relative requests.
  async rewrites() {
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
