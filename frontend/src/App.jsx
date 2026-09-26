import { useCallback, useEffect, useState } from 'react'
import './App.css'
import * as api from './api'
import ChatPanel from './ChatPanel'
import Landing from './Landing'
import Login from './Login'

const SELECTED_KEY = 'drs_selected_chat'

function useToasts() {
  const [toasts, setToasts] = useState([])
  const addToast = useCallback((msg, type = 'info') => {
    const id = Date.now() + Math.random()
    let added = false
    setToasts((prev) => {
      if (prev.some((t) => t.msg === msg)) return prev
      added = true
      return [...prev, { id, msg, type }]
    })
    if (added) setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4500)
  }, [])
  return { toasts, addToast }
}

const STATUS_LABEL = {
  awaiting_document: 'No document',
  processing: 'Processing',
  ready: 'Ready',
  failed: 'Failed',
}

// One-glance review state for a file in the list: issues, or clear.
function ReviewBadge({ rs }) {
  if (!rs) return null
  if (rs.status === 'running') return <span className="file-badge">reviewing…</span>
  if (rs.status !== 'done' || !rs.summary) return null
  const { mismatch = 0, review = 0 } = rs.summary
  if (!mismatch && !review) return <span className="file-badge file-badge-ok">✓ clear</span>
  if (rs.resolved) {
    return rs.open
      ? <span className="file-badge file-badge-warn">{rs.open} open · {rs.resolved} resolved</span>
      : <span className="file-badge file-badge-ok">✓ reviewed</span>
  }
  const parts = []
  if (mismatch) parts.push(`${mismatch} issue${mismatch > 1 ? 's' : ''}`)
  if (review) parts.push(`${review} review`)
  return <span className={`file-badge ${mismatch ? 'file-badge-bad' : 'file-badge-warn'}`}>{parts.join(' · ')}</span>
}

function ChatRow({ chat, active, onSelect, onRename, onDelete }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(chat.title)
  const busy = chat.status === 'processing'

  const commit = () => {
    setEditing(false)
    const t = draft.trim()
    if (t && t !== chat.title) onRename(chat.id, t)
    else setDraft(chat.title)
  }

  return (
    <div className={`chat-row${active ? ' active' : ''}`} onClick={() => onSelect(chat.id)}>
      <span className={`status-dot ${chat.status}`} title={STATUS_LABEL[chat.status]} />
      {editing ? (
        <input
          className="chat-row-input"
          value={draft}
          autoFocus
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === 'Enter') commit()
            if (e.key === 'Escape') { setDraft(chat.title); setEditing(false) }
          }}
          onClick={(e) => e.stopPropagation()}
        />
      ) : (
        <span className="chat-row-text">
          <span className="chat-row-title" title={chat.title}>
            {chat.review_summary?.borrower || chat.title}
          </span>
          <ReviewBadge rs={chat.review_summary} />
        </span>
      )}
      <span className="chat-row-actions">
        <button title="Rename" onClick={(e) => { e.stopPropagation(); setDraft(chat.title); setEditing(true) }}>✎</button>
        <button
          title={busy ? 'Cannot delete while the document is processing' : 'Delete'}
          disabled={busy}
          onClick={(e) => { e.stopPropagation(); if (!busy) onDelete(chat) }}
        >🗑</button>
      </span>
    </div>
  )
}

export default function App() {
  const { toasts, addToast } = useToasts()
  const [username, setUsername] = useState(api.getUser())
  const [authed, setAuthed] = useState(Boolean(api.getToken()))
  const [authMode, setAuthMode] = useState(null)
  const [chats, setChats] = useState([])
  const [selectedId, setSelectedId] = useState(localStorage.getItem(SELECTED_KEY))
  const [pendingDelete, setPendingDelete] = useState(null)
  const [deleting, setDeleting] = useState(false)
  const [loading, setLoading] = useState(true)
  const [credits, setCredits] = useState(null)

  useEffect(() => {
    api.setUnauthorizedHandler(() => {
      setAuthed(false)
      setAuthMode(null)
      setChats([])
      setSelectedId(null)
    })
  }, [])

  useEffect(() => {
    if (!authed) return
    const check = () => api.ensureFreshSession()
    check()
    window.addEventListener('focus', check)
    const id = setInterval(check, 30000)
    return () => { window.removeEventListener('focus', check); clearInterval(id) }
  }, [authed])

  const refresh = useCallback(async () => {
    try {
      const d = await api.listChats()
      setChats(d.chats)
      api.getCredits().then(setCredits).catch(() => {})
      return d.chats
    } catch (err) {
      addToast(err.message, 'error')
      return []
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    if (!authed) { setLoading(false); return }
    refresh().then((list) => {
      setSelectedId((cur) => (cur && list.some((c) => c.id === cur) ? cur : list[0]?.id ?? null))
    })
  }, [authed, refresh])

  useEffect(() => {
    if (selectedId) localStorage.setItem(SELECTED_KEY, selectedId)
    else localStorage.removeItem(SELECTED_KEY)
  }, [selectedId])

  useEffect(() => {
    if (!pendingDelete) return
    const onKey = (e) => { if (e.key === 'Escape' && !deleting) setPendingDelete(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [pendingDelete, deleting])

  const createChat = async () => {
    try {
      const { chat } = await api.newChat()
      await refresh()
      setSelectedId(chat.id)
    } catch (err) {
      addToast(err.message, 'error')
    }
  }

  const rename = async (id, title) => {
    try {
      await api.renameChat(id, title)
      refresh()
    } catch (err) {
      addToast(err.message, 'error')
    }
  }

  const remove = (chat) => setPendingDelete(chat)

  const confirmDelete = async () => {
    const chat = pendingDelete
    if (!chat) return
    setDeleting(true)
    try {
      await api.deleteChat(chat.id)
      const list = await refresh()
      if (selectedId === chat.id) setSelectedId(list[0]?.id ?? null)
      addToast('Chat deleted.', 'success')
      setPendingDelete(null)
    } catch (err) {
      addToast(err.message, 'error')
    } finally {
      setDeleting(false)
    }
  }

  const logout = () => {
    api.logout()
    localStorage.removeItem(SELECTED_KEY)
    setAuthed(false)
    setAuthMode(null)
    setChats([])
    setSelectedId(null)
  }

  if (!authed) {
    return (
      <>
        {authMode === null ? (
          <Landing
            onSignIn={() => setAuthMode('login')}
            onGetStarted={() => setAuthMode('signup')}
          />
        ) : (
          <Login
            initialMode={authMode}
            onBack={() => setAuthMode(null)}
            onAuthenticated={(u) => { setUsername(u); setAuthed(true); setLoading(true) }}
          />
        )}
        <div className="toast-container">
          {toasts.map((t) => <div key={t.id} className={`toast ${t.type}`}>{t.msg}</div>)}
        </div>
      </>
    )
  }

  const selected = chats.find((c) => c.id === selectedId) || null

  return (
    <div className="app">
      <header className="lp-nav header">
        <span className="lp-wordmark"><span className="lp-logo">📄</span> DocQA</span>
        <div className="header-user">
          {credits && (
            <span
              className={`credits${credits.remaining === 0 ? ' credits-out' : ''}`}
              title={`${credits.used} of ${credits.cap} used today. Resets at midnight IST.`}
            >
              {credits.remaining} / {credits.cap} credits
            </span>
          )}
          <span className="username">{username}</span>
          <button className="btn btn-secondary btn-sm" onClick={logout}>Log out</button>
        </div>
      </header>

      <div className="main-grid">
        <aside className="chat-rail">
          <button className="btn btn-primary new-chat-btn" onClick={createChat}>+ New loan file</button>

          <div className="chat-list">
            {loading ? (
              <div className="rail-empty"><span className="spinner" /></div>
            ) : chats.length === 0 ? (
              <div className="rail-empty">No loan files yet.<br />Start one to upload a borrower&rsquo;s packet.</div>
            ) : (
              chats.map((c) => (
                <ChatRow
                  key={c.id}
                  chat={c}
                  active={c.id === selectedId}
                  onSelect={setSelectedId}
                  onRename={rename}
                  onDelete={remove}
                />
              ))
            )}
          </div>
        </aside>

        {selected ? (
          <ChatPanel
            key={selected.id}
            chat={selected}
            onChatChanged={refresh}
            addToast={addToast}
          />
        ) : (
          <section className="chat-panel lp-mesh">
            <div className="empty-chat">
              <div className="empty-icon">📄</div>
              <h3>No loan file open</h3>
              <p>Start a new file and upload a borrower&rsquo;s packet to get the review.</p>
            </div>
          </section>
        )}
      </div>

      {pendingDelete && (
        <div
          className="modal-backdrop"
          onMouseDown={(e) => { if (e.target === e.currentTarget && !deleting) setPendingDelete(null) }}
        >
          <div className="modal" role="dialog" aria-modal="true" aria-labelledby="del-title">
            <h2 id="del-title">Delete this chat?</h2>
            <p>
              <strong>{pendingDelete.title}</strong> and all of its messages will be
              removed, along with the document&rsquo;s vectors. This cannot be undone.
            </p>
            <div className="modal-actions">
              <button
                className="btn btn-secondary"
                onClick={() => setPendingDelete(null)}
                disabled={deleting}
                autoFocus
              >
                No, keep it
              </button>
              <button className="btn btn-danger" onClick={confirmDelete} disabled={deleting}>
                {deleting ? 'Deleting…' : 'Yes, delete'}
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="toast-container">
        {toasts.map((t) => <div key={t.id} className={`toast ${t.type}`}>{t.msg}</div>)}
      </div>
    </div>
  )
}
