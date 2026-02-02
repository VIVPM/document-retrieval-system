// API client. Owns the tokens and every call to the backend.
//
// VITE_API_URL comes from frontend/.env (gitignored, local-only) → localhost.
// On Render that file is absent, so this falls back to the deployed backend.
const API = import.meta.env.VITE_API_URL || 'https://document-retrieval-system-5gqx.onrender.com'

const TOKEN_KEY = 'drs_token'
const REFRESH_KEY = 'drs_refresh'
const USER_KEY = 'drs_user'

export const getToken = () => localStorage.getItem(TOKEN_KEY)
export const getUser = () => localStorage.getItem(USER_KEY)

// True when there is no access token or it has passed its exp. Reading the JWT
// exp lets the app notice an expired session while idle, instead of only when
// the next request happens to 401.
export function accessTokenExpired() {
  const t = getToken()
  if (!t) return true
  try {
    const payload = JSON.parse(atob(t.split('.')[1]))
    return !payload.exp || payload.exp * 1000 <= Date.now()
  } catch {
    return true
  }
}

export function setSession(token, username, refresh) {
  localStorage.setItem(TOKEN_KEY, token)
  localStorage.setItem(USER_KEY, username)
  if (refresh) localStorage.setItem(REFRESH_KEY, refresh)
}

function setAccessToken(token) {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearSession() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
  localStorage.removeItem(REFRESH_KEY)
}

// Set by App so an expired session drops straight to the login screen instead
// of surfacing as a confusing error on whatever call happened to notice first.
let onUnauthorized = () => {}
export const setUnauthorizedHandler = (fn) => { onUnauthorized = fn }

// The access token is short-lived; the refresh token silently mints a new one.
// A single in-flight refresh is shared so a burst of 401s doesn't fire N
// refreshes (and race each other into logging the user out).
let refreshInFlight = null

async function tryRefresh() {
  const refresh = localStorage.getItem(REFRESH_KEY)
  if (!refresh) return false
  if (!refreshInFlight) {
    refreshInFlight = fetch(`${API}/api/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refresh }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (d?.token) { setAccessToken(d.token); return true } return false })
      .catch(() => false)
      .finally(() => { refreshInFlight = null })
  }
  return refreshInFlight
}

// Called proactively (on focus / on a timer) so an idle user whose session has
// died is sent to the landing page instead of sitting on a stale app view. If
// the access token is still good, does nothing. If it has expired, tries one
// refresh; if that fails the session is dead — clear it and signal the app.
export async function ensureFreshSession() {
  if (!getToken()) return
  if (!accessTokenExpired()) return
  const ok = await tryRefresh()
  if (!ok) {
    clearSession()
    onUnauthorized()
  }
}

async function authFetch(path, opts, retry = true) {
  const token = getToken()
  const headers = { ...(opts.headers || {}) }
  if (token) headers.Authorization = `Bearer ${token}`
  const res = await fetch(`${API}${path}`, { ...opts, headers })
  if (res.status === 401 && retry && (await tryRefresh())) {
    return authFetch(path, opts, false)  // one retry with the fresh token
  }
  return res
}

async function request(path, { method = 'GET', body, form, auth = true } = {}) {
  const headers = {}
  if (body) headers['Content-Type'] = 'application/json'
  const opts = { method, headers, body: form || (body ? JSON.stringify(body) : undefined) }

  const res = auth
    ? await authFetch(path, opts)
    : await fetch(`${API}${path}`, opts)

  if (res.status === 401) {
    clearSession()
    onUnauthorized()
    throw new Error('Your session expired. Please log in again.')
  }

  if (!res.ok) {
    let detail
    try {
      const e = await res.json()
      // FastAPI validation errors arrive as a list of field errors.
      detail = Array.isArray(e.detail) ? e.detail[0]?.msg : e.detail
    } catch {
      detail = null
    }
    throw new Error(detail || `Request failed (${res.status})`)
  }

  return res.status === 204 ? null : res.json()
}

// ── Auth ────────────────────────────────────────────────────────────────
export const signup = (username, password) =>
  request('/api/auth/signup', { method: 'POST', body: { username, password }, auth: false })

export const login = (username, password) =>
  request('/api/auth/login', { method: 'POST', body: { username, password }, auth: false })

export async function logout() {
  const refresh = localStorage.getItem(REFRESH_KEY)
  if (refresh) {
    // Best effort — revoke the refresh token server-side. Clearing local state
    // is what actually logs the user out, so a failed request must not block it.
    try {
      await fetch(`${API}/api/auth/logout`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refresh }),
      })
    } catch { /* ignore */ }
  }
  clearSession()
}

// ── Chats ───────────────────────────────────────────────────────────────
export const listChats = () => request('/api/chats')
export const newChat = () => request('/api/chats/new', { method: 'POST' })
export const getChat = (id) => request(`/api/chats/${id}`)
export const chatStatus = (id) => request(`/api/chats/${id}/status`)
export const renameChat = (id, title) =>
  request(`/api/chats/${id}`, { method: 'PATCH', body: { title } })
export const deleteChat = (id) => request(`/api/chats/${id}`, { method: 'DELETE' })

export function uploadDocument(id, file) {
  const form = new FormData()
  form.append('file', file)
  return request(`/api/chats/${id}/document`, { method: 'POST', form })
}

// Streamed answer. EventSource can't POST or send an Authorization header, so
// this reads the SSE body off a fetch() stream by hand. Calls the handlers as
// events arrive: onMeta(sources+query), onToken(text), onDone(), onError(msg).
export async function streamMessage(id, payload, { onMeta, onToken, onDone, onError }) {
  const res = await authFetch(`/api/chats/${id}/message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (res.status === 401) {
    clearSession()
    onUnauthorized()
    throw new Error('Your session expired. Please log in again.')
  }
  if (!res.ok || !res.body) {
    let detail
    try { const e = await res.json(); detail = Array.isArray(e.detail) ? e.detail[0]?.msg : e.detail } catch { detail = null }
    throw new Error(detail || `Request failed (${res.status})`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const line = buffer.slice(0, sep).trim()
      buffer = buffer.slice(sep + 2)
      if (!line.startsWith('data:')) continue
      let evt
      try { evt = JSON.parse(line.slice(5).trim()) } catch { continue }
      if (evt.type === 'meta') onMeta?.(evt.data)
      else if (evt.type === 'token') onToken?.(evt.data)
      else if (evt.type === 'done') onDone?.(evt.data)
      else if (evt.type === 'error') onError?.(evt.data)
    }
  }
}
