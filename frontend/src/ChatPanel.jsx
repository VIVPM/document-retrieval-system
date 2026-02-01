import { useCallback, useEffect, useRef, useState } from 'react'
import * as api from './api'

const POLL_MS = 5000

// Ingestion runs for minutes (GPU extraction, one LLM call per page, one
// embedding call per chunk), so the upload response only says "processing".
// This component owns the polling that turns that into a usable UI.
function ProcessingView({ filename, since }) {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setElapsed(Math.floor((Date.now() - since) / 1000)), 1000)
    return () => clearInterval(t)
  }, [since])

  return (
    <div className="empty-chat">
      <div className="empty-icon"><span className="spinner spinner-lg" /></div>
      <h3>Reading {filename}</h3>
      <p>
        Extracting pages, identifying documents and building the search index.
        This takes a few minutes for a large file — you can open another chat
        and come back.
      </p>
      <div className="processing-clock">{Math.floor(elapsed / 60)}m {elapsed % 60}s</div>
    </div>
  )
}

function UploadView({ chat, onUploaded, addToast }) {
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  const inputRef = useRef(null)

  // Mirror the backend cap (main.py MAX_UPLOAD_MB) so an oversized file is
  // rejected instantly instead of after a full upload and a 413.
  const MAX_MB = 3

  const pick = (f) => {
    if (!f) return
    if (!f.name.toLowerCase().endsWith('.pdf')) return addToast('Only PDF files are accepted.', 'error')
    if (f.size > MAX_MB * 1024 * 1024) return addToast(`File is larger than the ${MAX_MB} MB limit.`, 'error')
    setFile(f)
  }

  const start = async () => {
    if (!file || busy) return
    setBusy(true)
    try {
      await api.uploadDocument(chat.id, file)
      onUploaded()
    } catch (err) {
      addToast(err.message, 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="upload-view">
      {chat.status === 'failed' && (
        <div className="status-box error upload-failed">
          <strong>That document could not be processed.</strong>
          <span>{chat.error || 'Unknown error.'}</span>
        </div>
      )}

      <h3>Add a document to this conversation</h3>
      <p className="upload-view-sub">Every chat is about a single PDF. Upload one to begin.</p>

      <div
        className={`upload-zone${dragging ? ' drag-over' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files[0]) }}
      >
        <input ref={inputRef} type="file" accept=".pdf" onChange={(e) => pick(e.target.files[0])} />
        <div className="upload-icon">📂</div>
        <p>{dragging ? 'Drop it' : 'Click or drag a PDF here'}</p>
        <p className="upload-hint">PDF · up to {MAX_MB} MB</p>
        {file && <div className="filename">📄 {file.name}</div>}
      </div>

      <button className="btn btn-primary" onClick={start} disabled={!file || busy}>
        {busy ? <><span className="spinner" /> Uploading…</> : 'Process document'}
      </button>
    </div>
  )
}

function Sources({ sources }) {
  const [open, setOpen] = useState(false)
  if (!sources?.length) return null
  return (
    <div className="sources">
      <button className="sources-toggle" onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} {sources.length} source{sources.length > 1 ? 's' : ''}
      </button>
      {open && (
        <ul className="sources-list">
          {sources.map((s, i) => (
            <li key={i}>
              <span className="src-type">{s.doc_type}</span>
              <span className="src-meta">pages {s.pages} · {s.relevance}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default function ChatPanel({ chat, onChatChanged, addToast }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [querying, setQuerying] = useState(false)
  const [loading, setLoading] = useState(true)
  const [showSettings, setShowSettings] = useState(false)
  const [uploadStartedAt, setUploadStartedAt] = useState(Date.now())

  const [docFilter, setDocFilter] = useState('All')
  const [numChunks, setNumChunks] = useState(6)
  const [alpha, setAlpha] = useState(0.5)

  const endRef = useRef(null)
  const chatId = chat.id

  // Load history whenever the selected chat changes.
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setMessages([])
    setDocFilter('All')
    api.getChat(chatId)
      .then((d) => { if (!cancelled) setMessages(d.messages) })
      .catch((e) => { if (!cancelled) addToast(e.message, 'error') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [chatId, addToast])

  // Poll only while this chat is actually ingesting.
  useEffect(() => {
    if (chat.status !== 'processing') return
    const t = setInterval(async () => {
      try {
        const s = await api.chatStatus(chatId)
        if (s.status !== 'processing') onChatChanged()
      } catch { /* transient — keep polling */ }
    }, POLL_MS)
    return () => clearInterval(t)
  }, [chatId, chat.status, onChatChanged])

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, querying])

  const send = useCallback(async () => {
    const q = input.trim()
    if (!q || querying) return
    setInput('')
    const aid = `tmp-a-${Date.now()}`
    // Add the user turn and an empty assistant bubble the tokens will fill in.
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: q, id: `tmp-${Date.now()}` },
      { role: 'assistant', content: '', sources: null, id: aid, streaming: true },
    ])
    const patch = (fn) => setMessages((prev) => prev.map((x) => (x.id === aid ? fn(x) : x)))
    setQuerying(true)
    try {
      await api.streamMessage(chatId, {
        question: q,
        filter_type: docFilter === 'All' ? null : docFilter,
        num_chunks: numChunks,
        alpha,
      }, {
        onMeta: (m) => patch((x) => ({ ...x, sources: m.sources })),
        onToken: (t) => patch((x) => ({ ...x, content: x.content + t })),
        onError: (msg) => patch((x) => ({ ...x, content: `⚠️ ${msg}`, streaming: false })),
        onDone: () => { patch((x) => ({ ...x, streaming: false })); onChatChanged() },
      })
    } catch (err) {
      patch((x) => ({ ...x, content: x.content || `⚠️ ${err.message}`, streaming: false }))
    } finally {
      setQuerying(false)
      patch((x) => (x.streaming ? { ...x, streaming: false } : x))
    }
  }, [input, querying, chatId, docFilter, numChunks, alpha, onChatChanged])

  if (chat.status === 'processing') {
    return (
      <section className="chat-panel">
        <ProcessingView filename={chat.filename || 'your document'} since={uploadStartedAt} />
      </section>
    )
  }

  if (chat.status !== 'ready') {
    return (
      <section className="chat-panel">
        <UploadView
          chat={chat}
          addToast={addToast}
          onUploaded={() => { setUploadStartedAt(Date.now()); onChatChanged() }}
        />
      </section>
    )
  }

  const stats = chat.doc_stats || {}
  const docTypes = ['All', ...(stats.document_types || [])]

  return (
    <section className="chat-panel">
      {/* Document strip — what this conversation is about */}
      <div className="doc-strip">
        <div className="doc-strip-main">
          <span className="doc-name">📄 {chat.filename}</span>
          <span className="doc-meta">
            {stats.total_pages} pages · {stats.documents_found} documents · {stats.total_chunks} chunks
          </span>
        </div>
        <button className="doc-strip-btn" onClick={() => setShowSettings(!showSettings)}>
          {showSettings ? 'Hide search settings' : 'Search settings'}
        </button>
      </div>

      {showSettings && (
        <div className="settings-strip">
          <label>
            <span className="setting-label">Document type</span>
            <select value={docFilter} onChange={(e) => setDocFilter(e.target.value)}>
              {docTypes.map((t) => <option key={t}>{t}</option>)}
            </select>
          </label>

          <label>
            <span className="setting-label">Balance: {alpha === 1 ? 'semantic' : alpha === 0 ? 'keyword' : alpha}</span>
            <input type="range" min={0} max={1} step={0.1} value={alpha}
                   onChange={(e) => setAlpha(Number(e.target.value))} />
            <span className="setting-hint">0 = keyword · 1 = semantic</span>
          </label>

          <label>
            <span className="setting-label">Context chunks: {numChunks}</span>
            <input type="range" min={1} max={10} step={1} value={numChunks}
                   onChange={(e) => setNumChunks(Number(e.target.value))} />
          </label>
        </div>
      )}

      <div className="chat-messages">
        {loading ? (
          <div className="empty-chat"><span className="spinner spinner-lg" /></div>
        ) : messages.length === 0 ? (
          <div className="empty-chat">
            <div className="empty-icon">💬</div>
            <h3>Ask about {chat.filename}</h3>
            <p>Try a specific lookup — a figure, a date, a field name.</p>
          </div>
        ) : (
          messages.map((m) => (
            <div key={m.id} className={`message-bubble ${m.role}`}>
              <div className={`avatar ${m.role}`}>{m.role === 'user' ? '👤' : '🤖'}</div>
              <div className={`bubble-content ${m.role}`}>
                {m.role === 'assistant' && m.streaming && !m.content ? (
                  <span style={{ color: 'var(--text-muted)' }}>
                    <span className="spinner" style={{ marginRight: 8 }} /> Searching the document…
                  </span>
                ) : (
                  <>{m.content}{m.streaming && <span className="stream-caret" />}</>
                )}
                {m.role === 'assistant' && <Sources sources={m.sources} />}
              </div>
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>

      <div className="chat-input-bar">
        <textarea
          rows={1}
          placeholder="Ask a question about this document…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
          disabled={querying}
        />
        <button className="send-btn" onClick={send} disabled={!input.trim() || querying}>
          {querying ? <span className="spinner" /> : '➤'}
        </button>
      </div>
    </section>
  )
}
