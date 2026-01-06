import { useState } from 'react'
import * as api from './api'

export default function Login({ onAuthenticated, onBack, initialMode = 'login' }) {
  const [mode, setMode] = useState(initialMode)   // 'login' | 'signup'
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const isSignup = mode === 'signup'

  const submit = async (e) => {
    e.preventDefault()
    if (busy) return
    setError(null)
    setBusy(true)
    try {
      const fn = isSignup ? api.signup : api.login
      const res = await fn(username.trim(), password)
      api.setSession(res.token, res.username, res.refresh_token)
      onAuthenticated(res.username)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const swap = () => {
    setMode(isSignup ? 'login' : 'signup')
    setError(null)
  }

  return (
    <div className="auth-screen">
      <form className="auth-card" onSubmit={submit}>
        {onBack && (
          <button type="button" className="auth-back" onClick={onBack}>← Back</button>
        )}
        <div className="auth-brand">
          <span className="auth-logo">📄</span>
          <h1>{isSignup ? 'Create your account' : 'Welcome back'}</h1>
          <p>Ask questions about a document. One conversation per file.</p>
        </div>

        <label className="auth-field">
          <span>Username</span>
          <input
            type="email"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            placeholder="you@gmail.com"
            pattern="[^@\s]+@gmail\.com"
            title="Must be a Gmail address ending in @gmail.com"
            autoFocus
            required
          />
        </label>

        <label className="auth-field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={isSignup ? 'new-password' : 'current-password'}
            required
          />
          {isSignup && <span className="auth-hint">At least 8 characters.</span>}
        </label>

        {error && <div className="auth-error">{error}</div>}

        <button className="btn btn-primary auth-submit" disabled={busy || !username || !password}>
          {busy ? <><span className="spinner" /> Please wait…</> : (isSignup ? 'Create account' : 'Log in')}
        </button>

        <button type="button" className="auth-swap" onClick={swap}>
          {isSignup ? 'Already have an account? Log in' : "New here? Create an account"}
        </button>
      </form>
    </div>
  )
}
