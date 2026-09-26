import { useCallback, useEffect, useRef, useState } from 'react'
import * as api from './api'

const POLL_MS = 2500

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

  const stageIdx = STAGES.findIndex((s) => s.key === stage)
  let active = stageIdx
  if (active < 0) {
    active = 0
    for (let i = 0; i < STAGES.length; i++) if (elapsed >= STAGES[i].at) active = i
  }
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
  const [files, setFiles] = useState([])
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  const inputRef = useRef(null)

  const MAX_MB = 3

  const pick = (list) => {
    const incoming = [...(list || [])]
    if (!incoming.length) return
    if (incoming.some((f) => !f.name.toLowerCase().endsWith('.pdf'))) {
      return addToast('Only PDF files are accepted.', 'error')
    }
    const next = [...files, ...incoming.filter((f) => !files.some((g) => g.name === f.name && g.size === f.size))]
    const total = next.reduce((n, f) => n + f.size, 0)
    if (total > MAX_MB * 1024 * 1024) return addToast(`Files add up to more than the ${MAX_MB} MB limit.`, 'error')
    setFiles(next)
  }
  const removeAt = (i) => setFiles(files.filter((_, j) => j !== i))

  const start = async () => {
    if (!files.length || busy) return
    setBusy(true)
    try {
      await api.uploadDocument(chat.id, files)
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

      <h3>Upload the borrower&rsquo;s packet</h3>
      <p className="upload-view-sub">
        One packet PDF, or the documents as separate files — they&rsquo;re merged into one, in this order.
      </p>

      <div
        className={`upload-zone${dragging ? ' drag-over' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files) }}
      >
        <input ref={inputRef} type="file" accept=".pdf" multiple
          onChange={(e) => { pick(e.target.files); e.target.value = '' }} />
        <div className="upload-icon">📂</div>
        <p>{dragging ? 'Drop them' : 'Click or drag PDFs here'}</p>
        <p className="upload-hint">PDF · up to {MAX_MB} MB in total</p>
      </div>

      {files.length > 0 && (
        <ol className="upload-files">
          {files.map((f, i) => (
            <li key={`${f.name}-${f.size}`}>
              <span className="upload-file-name">📄 {f.name}</span>
              <button type="button" onClick={() => removeAt(i)} disabled={busy}
                aria-label={`Remove ${f.name}`}>✕</button>
            </li>
          ))}
        </ol>
      )}

      <button className="btn btn-primary" onClick={start} disabled={!files.length || busy}>
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
              <div className="src-head">
                <span className="src-type">{s.doc_type}</span>
                <span className="src-meta">pages {s.pages} · {s.relevance}</span>
              </div>
              {s.preview && <blockquote className="src-preview">{s.preview}</blockquote>}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

const REVIEW_ICON = { mismatch: '❌', review: '⚠️', missing: '❔', match: '✅', info: 'ℹ️' }

const DECISION_LABEL = { accepted: 'Accepted', confirmed: 'Confirmed issue' }

// Accept (explained — note required) or Confirm (real issue) one flag, and
// show who decided it. Decisions are append-only; changing one adds a row.
function FlagDecision({ flagKey, decision, history, onDecide }) {
  const [mode, setMode] = useState(null)
  const [note, setNote] = useState('')
  const [saving, setSaving] = useState(false)
  const [showHist, setShowHist] = useState(false)
  const needsNote = mode === 'accepted'

  const save = async () => {
    setSaving(true)
    const ok = await onDecide(flagKey, mode, note.trim())
    setSaving(false)
    if (ok) { setMode(null); setNote('') }
  }

  return (
    <div className="flag-decision">
      {decision && (
        <div className={`flag-status flag-${decision.decision}`}>
          {DECISION_LABEL[decision.decision]} by {decision.by} · {new Date(decision.at).toLocaleString()}
          {decision.note && <span className="flag-note">“{decision.note}”</span>}
        </div>
      )}
      {mode ? (
        <div className="flag-form">
          <input className="flag-input" autoFocus value={note} maxLength={1000}
                 placeholder={needsNote ? 'Why is this acceptable? (required)' : 'Note (optional)'}
                 onChange={(e) => setNote(e.target.value)}
                 onKeyDown={(e) => { if (e.key === 'Enter' && !(needsNote && !note.trim())) save() }} />
          <button className="flag-btn flag-btn-primary" onClick={save}
                  disabled={saving || (needsNote && !note.trim())}>
            {saving ? 'Saving…' : DECISION_LABEL[mode]}
          </button>
          <button className="flag-btn" onClick={() => { setMode(null); setNote('') }} disabled={saving}>Cancel</button>
        </div>
      ) : (
        <div className="flag-actions">
          <button className="flag-btn" onClick={() => setMode('accepted')}>{decision ? 'Change: accept' : 'Accept'}</button>
          <button className="flag-btn" onClick={() => setMode('confirmed')}>{decision ? 'Change: confirm' : 'Confirm issue'}</button>
          {history.length > 0 && (
            <button className="flag-link" onClick={() => setShowHist(!showHist)}>
              {showHist ? 'Hide history' : `History (${history.length})`}
            </button>
          )}
        </div>
      )}
      {showHist && (
        <ul className="flag-hist">
          {history.map((h, i) => (
            <li key={i}>{new Date(h.at).toLocaleString()} — {DECISION_LABEL[h.decision]} by {h.by}{h.note ? `: “${h.note}”` : ''}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

function ReviewPanel({ review, onAsk, full = false, decisions = {}, history = [], onDecide }) {
  const issues = review?.summary ? review.summary.mismatch + review.summary.review : 0
  const [open, setOpen] = useState(true)
  if (!review) return null

  if (review.status === 'running') {
    return <div className={`review review-note${full ? ' review-full' : ''}`}><span className="spinner" /> Reviewing the file — checking the application against the supporting documents…</div>
  }
  if (review.status !== 'done') {
    return <div className={`review review-note${full ? ' review-full' : ''}`}>File review unavailable{review.error ? `: ${review.error}` : '.'}</div>
  }

  const s = review.summary
  const flagKeys = review.checks.map((c, i) => `${c.id}:${i}`)
    .filter((k, i) => ['mismatch', 'review'].includes(review.checks[i].status))
  const resolved = flagKeys.filter((k) => decisions[k]).length
  const headline = [
    s.mismatch && `${s.mismatch} mismatch${s.mismatch > 1 ? 'es' : ''}`,
    s.review && `${s.review} to review`,
    s.missing && `${s.missing} not found`,
    `${s.match} matched`,
    resolved && `${resolved}/${flagKeys.length} decided`,
  ].filter(Boolean).join(' · ')

  return (
    <div className={`review${issues ? ' review-has-issues' : ''}${full ? ' review-full' : ''}`}>
      <button className="review-head" onClick={() => setOpen(!open)}>
        <span className="review-title">{open ? '▾' : '▸'} File review{review.borrower ? ` — ${review.borrower}` : ''}</span>
        <span className="review-sum">{headline}</span>
      </button>
      {open && (
        <ul className="review-list">
          {review.checks.map((c, i) => (
            <li key={`${c.id}-${i}`} className={`review-row review-${c.status}`}>
              <span className="review-ico" title={c.status}>{REVIEW_ICON[c.status] || '•'}</span>
              <div className="review-body">
                <div className="review-line">
                  <span className="review-label">{c.label}</span>
                  <span className="review-detail">{c.detail}</span>
                </div>
                {c.evidence?.some((e) => e.value) && (
                  <div className="review-ev">
                    {c.evidence.filter((e) => e.value).map((e, j) => (
                      <span key={j} className="review-chip" title={e.doc_type}>
                        {e.label}: <b>{String(e.value)}</b>{e.page ? ` · p.${e.page}` : ''}
                      </span>
                    ))}
                  </div>
                )}
                {(c.status === 'mismatch' || c.status === 'review') && onDecide && review.run_id && (
                  <FlagDecision flagKey={`${c.id}:${i}`} decision={decisions[`${c.id}:${i}`]}
                                history={history.filter((h) => h.run_id === review.run_id && h.check_key === `${c.id}:${i}`)}
                                onDecide={onDecide} />
                )}
              </div>
              {(c.status === 'mismatch' || c.status === 'review') && (
                <button className="review-ask" onClick={() => onAsk(`Explain this ${c.label.toLowerCase()} issue: ${c.detail}`)}>
                  Ask
                </button>
              )}
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
  const [stage, setStage] = useState(null)
  const [review, setReview] = useState(null)
  const [decisions, setDecisions] = useState({})
  const [history, setHistory] = useState([])
  const [tab, setTab] = useState('review')

  const [docFilter, setDocFilter] = useState('All')
  const [numChunks, setNumChunks] = useState(6)
  const [alpha, setAlpha] = useState(0.5)

  const endRef = useRef(null)
  const chatId = chat.id

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setMessages([])
    setDocFilter('All')
    setReview(null)
    setDecisions({})
    setHistory([])
    setTab('review')
    api.getChat(chatId)
      .then((d) => {
        if (cancelled) return
        setMessages(d.messages)
        setReview(d.chat?.review ?? null)
        setDecisions(d.chat?.decisions ?? {})
        setHistory(d.chat?.decision_history ?? [])
      })
      .catch((e) => { if (!cancelled) addToast(e.message, 'error') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [chatId, chat.status, addToast])

  useEffect(() => {
    if (chat.status !== 'ready' || review?.status !== 'running') return
    const t = setInterval(() => {
      api.getChat(chatId).then((d) => {
        const r = d.chat?.review ?? null
        setReview(r)
        setDecisions(d.chat?.decisions ?? {})
        setHistory(d.chat?.decision_history ?? [])
        if (r?.status !== 'running') onChatChanged()
      }).catch(() => {})
    }, 4000)
    return () => clearInterval(t)
  }, [chatId, chat.status, review?.status, onChatChanged])

  useEffect(() => {
    if (chat.status !== 'processing') { setStage(null); return }
    const t = setInterval(() => {
      api.chatStatus(chatId).then((s) => {
        setStage(s.stage)
        if (s.status !== 'processing') onChatChanged()
      }).catch(() => {})
    }, POLL_MS)
    return () => clearInterval(t)
  }, [chatId, chat.status, onChatChanged])

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, querying])

  const runMessage = useCallback(async (question, extra = {}) => {
    if (querying) return
    const aid = `tmp-a-${Date.now()}`
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

  const summarize = useCallback(
    () => runMessage('Summarize this document.', { summarize: true }), [runMessage])

  if (chat.status === 'processing') {
    return (
      <section className="chat-panel lp-mesh">
        <ProcessingView filename={chat.filename || 'your document'} since={uploadStartedAt} stage={stage} />
      </section>
    )
  }

  if (chat.status !== 'ready') {
    return (
      <section className="chat-panel lp-mesh">
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
  const view = review ? tab : 'ask'
  const askAbout = (text) => { setInput(text); setTab('ask') }
  const decide = async (key, decision, note) => {
    try {
      const { decision: d } = await api.decideFlag(chatId, key, decision, note)
      setDecisions((prev) => ({ ...prev, [key]: d }))
      setHistory((prev) => [...prev, d])
      onChatChanged()
      return true
    } catch (err) {
      addToast(err.message, 'error')
      return false
    }
  }
  const s = review?.summary
  const reviewTabLabel = review?.status === 'running' ? 'Review…'
    : s ? `Review${s.mismatch + s.review ? ` (${s.mismatch + s.review})` : ' ✓'}` : 'Review'

  return (
    <section className="chat-panel">
      <div className="doc-strip">
        <div className="doc-strip-main">
          <span className="doc-name">
            {review?.borrower ? <>{review.borrower} <span className="doc-file">· {chat.filename}</span></> : <>📄 {chat.filename}</>}
          </span>
          <span className="doc-meta">
            {stats.total_pages} pages · {stats.documents_found} documents · {stats.total_chunks} chunks
          </span>
        </div>
        {review && (
          <div className="file-tabs" role="tablist">
            <button role="tab" aria-selected={view === 'review'}
                    className={`file-tab${view === 'review' ? ' active' : ''}`}
                    onClick={() => setTab('review')}>{reviewTabLabel}</button>
            <button role="tab" aria-selected={view === 'ask'}
                    className={`file-tab${view === 'ask' ? ' active' : ''}`}
                    onClick={() => setTab('ask')}>Ask</button>
          </div>
        )}
        <div className="doc-strip-actions">
          <button className="doc-strip-btn" onClick={() => { setTab('ask'); summarize() }} disabled={querying}
                  title="Summarize the whole document (reads every page, not just the top matches)">
            📝 Summarize
          </button>
          {view === 'ask' && (
            <button className="doc-strip-btn" onClick={() => setShowSettings(!showSettings)}>
              {showSettings ? 'Hide search settings' : 'Search settings'}
            </button>
          )}
        </div>
      </div>

      {view === 'review' && <ReviewPanel review={review} onAsk={askAbout} full
        decisions={decisions} history={history} onDecide={decide} />}

      {view === 'ask' && showSettings && (
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

      {view === 'ask' && <>
      <div className="chat-messages">
        {loading ? (
          <div className="empty-chat"><span className="spinner spinner-lg" /></div>
        ) : messages.length === 0 ? (
          <div className="empty-chat">
            <div className="empty-icon">💬</div>
            <h3>Ask about this file</h3>
            <p>Anything the review doesn&rsquo;t cover — a figure, a date, a lien, a contract term. Answers cite the page.</p>
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
          placeholder="Ask about this loan file…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
          disabled={querying}
        />
        <button className="send-btn" onClick={send} disabled={!input.trim() || querying}>
          {querying ? <span className="spinner" /> : '➤'}
        </button>
      </div>
      </>}
    </section>
  )
}
