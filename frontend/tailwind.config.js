/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Satin / neon primary palette for elegant accents
        primary: {
          50: '#f7f9fb',
          100: '#eef6fb',
          200: '#d9f0fb',
          300: '#bfe9fb',
          400: '#92e0f8',
          500: '#4fd5f2',
          600: '#24c6e6',
          700: '#0aa3b8',
          800: '#077a86',
          900: '#044f55',
          950: '#01292d',
        },
        // Secondary palette tuned for an elegant dark grey UI
        secondary: {
          50: '#fbfbfc',
          100: '#f6f7f8',
          200: '#eceef0',
          300: '#dfe3e6',
          400: '#bfc9cf',
          500: '#9aa6af',
          600: '#6f7b84',
          700: '#4b5358',
          800: '#2b2f33',
          900: '#16181a',
          950: '#0b0c0d',
        },
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
