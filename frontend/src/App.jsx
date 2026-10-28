import { useCallback, useEffect, useState } from 'react'
import './App.css'
import * as api from './api'
import ChatPanel, { SettingsModal } from './ChatPanel'
import Landing from './Landing'
import Login from './Login'

const SELECTED_KEY = 'drs_selected_chat'
const PAGE_SIZE = 10

const RAIL_PATHS = {
  plus: <path d="M12 5v14M5 12h14" />,
  close: <><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M9 3v18m7-12-3 3 3 3" /></>,
  open: <><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M9 3v18m4-12 3 3-3 3" /></>,
  search: <><circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" /></>,
  x: <path d="M18 6 6 18M6 6l12 12" />,
  rename: <><path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" /></>,
  delete: <><path d="M3 6h18m-2 0-.8 14H5.8L5 6m3 0V4h8v2m-6 4v6m4-6v6" /></>,
  settings: <><circle cx="12" cy="12" r="3" /><path d="M10 2h4l.5 2.1 1.7.7L18 3.7l2.8 2.8-1.1 1.8.7 1.7L22.5 10v4l-2.1.5-.7 1.7 1.1 1.8-2.8 2.8-1.8-1.1-1.7.7L14 22.5h-4l-.5-2.1-1.7-.7-1.8 1.1-2.8-2.8 1.1-1.8-.7-1.7L1.5 14v-4l2.1-.5.7-1.7-1.1-1.8L6 3.2l1.8 1.1 1.7-.7Z" /></>,
  logout: <><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4m7 14 5-5-5-5m5 5H9" /></>,
}

function RailIcon({ name, size = 16 }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{RAIL_PATHS[name]}</svg>
}

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
  const parts = []
  if (mismatch) parts.push(`${mismatch} issue${mismatch > 1 ? 's' : ''}`)
  if (review) parts.push(`${review} to review`)
  return <>
    <span className={`file-badge ${mismatch ? 'file-badge-bad' : 'file-badge-warn'}`}>{parts.join(' · ')}</span>
    <span className="file-badge">{rs.resolved || 0}/{mismatch + review} reviewed</span>
  </>
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
        <button title="Rename" aria-label={`Rename ${chat.title}`} onClick={(e) => { e.stopPropagation(); setDraft(chat.title); setEditing(true) }}><RailIcon name="rename" size={14} /></button>
        <button
          title={busy ? 'Cannot delete while the document is processing' : 'Delete'}
          aria-label={`Delete ${chat.title}`}
          disabled={busy}
          onClick={(e) => { e.stopPropagation(); if (!busy) onDelete(chat) }}
        ><RailIcon name="delete" size={14} /></button>
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
  const hasOwnKey = Boolean(credits?.own_key)
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 860)
  const [searchQuery, setSearchQuery] = useState('')
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE)
  const [showSettings, setShowSettings] = useState(false)
  const [docFilter, setDocFilter] = useState('All')
  const [numChunks, setNumChunks] = useState(6)
  const [alpha, setAlpha] = useState(0.5)

  useEffect(() => { setDocFilter('All') }, [selectedId])

  useEffect(() => {
    api.setUnauthorizedHandler(() => {
      setAuthed(false)
      setAuthMode(null)
      setChats([])
      setSelectedId(null)
      setCredits(null)
      setShowSettings(false)
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

  useEffect(() => {
    if (!authed || hasOwnKey || credits?.remaining !== 0) return
    const now = Date.now()
    const untilMidnight = 86400000 - ((now + 19800000) % 86400000)
    const timer = setTimeout(() => api.getCredits().then(setCredits).catch(() => {}), untilMidnight + 1000)
    return () => clearTimeout(timer)
  }, [authed, hasOwnKey, credits?.remaining])

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
      if (window.innerWidth <= 860) setSidebarOpen(false)
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
    setCredits(null)
    setAuthMode(null)
    setChats([])
    setSelectedId(null)
    setShowSettings(false)
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
  const filteredChats = chats.filter((c) =>
    (c.review_summary?.borrower || c.title).toLowerCase().includes(searchQuery.toLowerCase()) ||
    c.title.toLowerCase().includes(searchQuery.toLowerCase()))
  const visibleChats = filteredChats.slice(0, visibleCount)
  const selectChat = (id) => {
    setSelectedId(id)
    if (window.innerWidth <= 860) setSidebarOpen(false)
  }

  return (
    <div className="app">
      <header className="lp-nav header">
        <span className="lp-wordmark"><span className="lp-logo">📄</span> FinDoc</span>
        <div className="header-user">
          {!hasOwnKey && credits && (
            <span
              className={`credits${credits.remaining === 0 ? ' credits-out' : ''}`}
              title={`${credits.remaining} of ${credits.cap} daily message credits left. 1 credit = 1 question and its answer. Resets at midnight IST.`}
            >
              {credits.remaining} / {credits.cap} credits left today
            </span>
          )}
        </div>
      </header>

      <div className={`main-grid${sidebarOpen ? '' : ' rail-closed'}`}>
        {sidebarOpen && <div className="rail-backdrop" onClick={() => setSidebarOpen(false)} />}
        <aside className={`chat-rail${sidebarOpen ? '' : ' collapsed'}`} aria-label="Loan files">
          <div className="rail-header">
            {sidebarOpen && <button className="new-chat-btn" onClick={createChat}><RailIcon name="plus" size={18} /> New loan file</button>}
            <button className="rail-toggle" onClick={() => setSidebarOpen((open) => !open)}
              title={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
              aria-label={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'} aria-expanded={sidebarOpen}>
              <RailIcon name={sidebarOpen ? 'close' : 'open'} />
            </button>
          </div>

          {sidebarOpen && <>
            <div className="rail-search">
              <input type="search" placeholder="Search loan files..." aria-label="Search loan files"
                value={searchQuery} onChange={(e) => { setSearchQuery(e.target.value); setVisibleCount(PAGE_SIZE) }} />
              {searchQuery ? (
                <button onClick={() => { setSearchQuery(''); setVisibleCount(PAGE_SIZE) }} aria-label="Clear search"><RailIcon name="x" size={15} /></button>
              ) : <RailIcon name="search" size={16} />}
            </div>

            <div className="chat-list">
              {loading ? (
                <div className="rail-empty"><span className="spinner" /></div>
              ) : chats.length === 0 ? (
                <div className="rail-empty">No loan files yet.<br />Start one to upload a borrower&rsquo;s packet.</div>
              ) : filteredChats.length === 0 ? (
                <div className="rail-empty">No loan files found.</div>
              ) : <>
                {visibleChats.map((c) => (
                  <ChatRow key={c.id} chat={c} active={c.id === selectedId}
                    onSelect={selectChat} onRename={rename} onDelete={remove} />
                ))}
                {filteredChats.length > visibleCount && (
                  <button className="rail-more" onClick={() => setVisibleCount((n) => n + PAGE_SIZE)}>
                    Load {Math.min(PAGE_SIZE, filteredChats.length - visibleCount)} more files
                  </button>
                )}
              </>}
            </div>
          </>}
          <div className="rail-footer">
            {sidebarOpen && <span className="rail-user"><span className="rail-user-dot" />{username}</span>}
            <button onClick={() => setShowSettings(true)} title="Settings" aria-label="Settings"><RailIcon name="settings" /></button>
            <button onClick={logout} title="Log out" aria-label="Log out"><RailIcon name="logout" /></button>
          </div>
        </aside>

        {selected ? (
          <ChatPanel
            key={selected.id}
            chat={selected}
            docFilter={docFilter}
            numChunks={numChunks}
            alpha={alpha}
            onChatChanged={refresh}
            credits={credits}
            hasOwnKey={hasOwnKey}
            onCreditsRefresh={() => api.getCredits().then(setCredits).catch(() => {})}
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

      {showSettings && <SettingsModal chat={selected} onClose={() => setShowSettings(false)}
        hasOwnKey={hasOwnKey} credits={credits} onOwnKeyChange={() => refresh()}
        docFilter={docFilter} setDocFilter={setDocFilter} numChunks={numChunks} setNumChunks={setNumChunks}
        alpha={alpha} setAlpha={setAlpha} />}

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
