"use client"

import { useState } from 'react'

export default function AdminLoginPage() {
  const [password, setPassword] = useState('')
  const [message, setMessage] = useState<string | null>(null)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setMessage(null)
    try {
      const res = await fetch('/api/admin/user-management/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ password }),
      })
      if (res.ok) {
        setMessage('Login successful — admin session cookie set.')
      } else {
        const j = await res.json().catch(() => ({}))
        setMessage(j.detail || 'Login failed')
      }
    } catch (err) {
      setMessage('Network error')
    }
  }

  return (
    <div className="p-8 max-w-md">
      <h1 className="text-2xl font-semibold">Admin Login</h1>
      <form onSubmit={submit} className="mt-4 flex flex-col gap-3">
        <label className="flex flex-col">
          <span className="text-sm text-gray-600">Admin token</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="mt-1 p-2 border rounded"
            placeholder="Enter admin token"
          />
        </label>
        <div className="flex gap-2">
          <button type="submit" className="px-4 py-2 bg-blue-600 text-white rounded">Log in</button>
        </div>
        {message && <div className="mt-2 text-sm">{message}</div>}
      </form>
    </div>
  )
}
