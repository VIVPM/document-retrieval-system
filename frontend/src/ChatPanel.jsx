import { useCallback, useEffect, useRef, useState } from 'react'
import * as api from './api'

const POLL_MS = 2500   // snappy enough to track ingest sub-steps as they change

// The real ingest pipeline, in order. `at` is the elapsed second each stage
// lights up — a plausible timeline (ingest ~45-55s), not a live signal: the
// backend exposes only processing/ready, so the last stage holds until the poll
// flips to ready and this view swaps for the chat. Windows mirror where time
// ACTUALLY goes — extraction (Docling GPU) is the bulk, embeddings second;
// chunking is sub-second, so it flashes by rather than looking like the holdup.
const STAGES = [
  { key: 'extract', label: 'Extracting text & tables', sub: 'Layout-aware OCR reads every page', at: 0 },
  { key: 'split', label: 'Splitting into documents', sub: 'Classifying pages, detecting boundaries', at: 26 },
  { key: 'chunk', label: 'Chunking', sub: 'Structure-aware — tables kept whole', at: 32 },
  { key: 'embed', label: 'Embedding chunks', sub: 'Vectorizing each chunk with Gemini', at: 35 },
  { key: 'store', label: 'Storing in vector DB', sub: 'Indexing in Pinecone for hybrid search', at: 47 },
]

// Ingestion runs for ~a minute (GPU extraction, per-page LLM calls, per-chunk
// embeddings), so the upload response only says "processing". This walks the
// user through what's actually happening rather than parking on a spinner.
function ProcessingView({ filename, since, stage }) {
  const [elapsed, setElapsed] = useState(() => Math.max(0, Math.floor((Date.now() - since) / 1000)))
  useEffect(() => {
    const t = setInterval(() => setElapsed(Math.max(0, Math.floor((Date.now() - since) / 1000))), 500)
    return () => clearInterval(t)
  }, [since])

  // Prefer the real backend stage; fall back to the elapsed timeline for the
  // brief moment before the first stage lands (or an ingest predating this).
  const stageIdx = STAGES.findIndex((s) => s.key === stage)
  let active = stageIdx
  if (active < 0) {
    active = 0
    for (let i = 0; i < STAGES.length; i++) if (elapsed >= STAGES[i].at) active = i
  }
  // Bar is a smooth time-based estimate (the steps carry the exact stage —
  // real stages are too uneven, ~70% is extraction, to map onto a bar). Never
  // 100% here: hitting ready unmounts this view.
  const pct = Math.min(94, Math.round((elapsed / 52) * 94))

  return (
    <div className="proc-wrap">
      <div className="proc-card">
        <div className="proc-head">
          <span className="proc-file">📄 {filename}</span>
          <span className="proc-clock">{Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, '0')}</span>
        </div>

        <div className="proc-bar"><div className="proc-bar-fill" style={{ width: `${pct}%` }} /></div>

        <ul className="proc-steps">
          {STAGES.map((s, i) => {
            const state = i < active ? 'done' : i === active ? 'active' : 'pending'
            return (
              <li key={s.key} className={`proc-step ${state}`}>
                <span className="proc-ico">
                  {state === 'done' ? '✓' : state === 'active' ? <span className="spinner" /> : <span className="proc-pip" />}
                </span>
                <span className="proc-text">
                  <span className="proc-label">{s.label}</span>
                  <span className="proc-sub">{s.sub}</span>
                </span>
              </li>
            )
          })}
        </ul>

        <p className="proc-note">This runs in the background — open another chat and come back, it&rsquo;ll be ready to ask.</p>
      </div>
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
  const [stage, setStage] = useState(null)   // live ingest sub-step from /status

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

  // Poll only while this chat is actually ingesting; track the live sub-step.
  useEffect(() => {
    if (chat.status !== 'processing') { setStage(null); return }
    const t = setInterval(async () => {
      try {
        const s = await api.chatStatus(chatId)
        setStage(s.stage)
        if (s.status !== 'processing') onChatChanged()
      } catch { /* transient — keep polling */ }
    }, POLL_MS)
    return () => clearInterval(t)
  }, [chatId, chat.status, onChatChanged])

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, querying])

  const runMessage = useCallback(async (question, extra = {}) => {
    if (querying) return
    const aid = `tmp-a-${Date.now()}`
    // Add the user turn and an empty assistant bubble the tokens will fill in.
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: question, id: `tmp-${Date.now()}` },
      { role: 'assistant', content: '', sources: null, id: aid, streaming: true },
    ])
    const patch = (fn) => setMessages((prev) => prev.map((x) => (x.id === aid ? fn(x) : x)))
    setQuerying(true)
    try {
      await api.streamMessage(chatId, {
        question,
        filter_type: docFilter === 'All' ? null : docFilter,
        num_chunks: numChunks,
        alpha,
        ...extra,
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
  }, [querying, chatId, docFilter, numChunks, alpha, onChatChanged])

  const send = useCallback(() => {
    const q = input.trim()
    if (!q) return
    setInput('')
    runMessage(q)
  }, [input, runMessage])

  // Whole-document summary — feeds every chunk, not the top-k a query retrieves.
  const summarize = useCallback(
    () => runMessage('Summarize this document.', { summarize: true }), [runMessage])

  if (chat.status === 'processing') {
    return (
      <section className="chat-panel">
        <ProcessingView filename={chat.filename || 'your document'} since={uploadStartedAt} stage={stage} />
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
        <div className="doc-strip-actions">
          <button className="doc-strip-btn" onClick={summarize} disabled={querying}
                  title="Summarize the whole document (reads every page, not just the top matches)">
            📝 Summarize
          </button>
          <button className="doc-strip-btn" onClick={() => setShowSettings(!showSettings)}>
            {showSettings ? 'Hide search settings' : 'Search settings'}
          </button>
        </div>
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
