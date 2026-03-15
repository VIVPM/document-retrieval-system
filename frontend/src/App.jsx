import { useState, useRef, useEffect, useCallback } from 'react'
import './App.css'

const API = 'http://localhost:8000'

// ── Toast utility ──────────────────────────────────────────────────────
function useToasts() {
  const [toasts, setToasts] = useState([])
  const addToast = useCallback((msg, type = 'info') => {
    const id = Date.now()
    setToasts(prev => [...prev, { id, msg, type }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4000)
  }, [])
  return { toasts, addToast }
}

// ── API helpers ────────────────────────────────────────────────────────
async function apiUpload(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API}/upload`, { method: 'POST', body: form })
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Upload failed') }
  return res.json()
}

async function apiQuery(payload) {
  const res = await fetch(`${API}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Query failed') }
  return res.json()
}

async function apiClear() {
  const res = await fetch(`${API}/clear`, { method: 'POST' })
  if (!res.ok) throw new Error('Clear failed')
  return res.json()
}

async function apiSetRerank(enabled) {
  const res = await fetch(`${API}/settings/rerank`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  })
  if (!res.ok) throw new Error('Rerank setting failed')
  return res.json()
}

async function apiStructure() {
  const res = await fetch(`${API}/structure`)
  if (!res.ok) return []
  const d = await res.json()
  return d.structure || []
}

// ── Format response text ──────────────────────────────────────────────
function buildAnswerText(data) {
  let txt = data.answer + '\n\n'

    if (data.sources?.length) {
    txt += '📍 Sources:\n'
    data.sources.forEach(s => {
      txt += `• ${s.filename} | ${s.doc_type} (Pages ${s.pages}) - Relevance: ${s.relevance}\n`
    })
  }

  const rd = data.retrieval_details || {}
  const cfg = rd.rrf_config || {}
  const stats = rd.retrieval_stats || {}
  if (cfg.k !== undefined) {
    txt += `\n🔍 Search: Hybrid (FAISS + BM25 with RRF, k=${cfg.k})`
    txt += `\n📊 Searched ${stats.total_chunks ?? 0} chunks`
    if (cfg.rerank_enabled) {
      txt += '\n🎯 Reranking: ✅ Applied'
    }
  }

  const ri = rd.routing_info || {}
  if (ri.method === 'auto_route')
    txt += `\n🎯 Routed to: ${ri.predicted_type} (conf: ${(ri.confidence * 100).toFixed(0)}%)`
  else if (ri.method === 'filter')
    txt += `\n🏷️ Filtered to: ${ri.type}`

  txt += `\n\nConfidence: ${(data.confidence * 100).toFixed(1)}% | Filter: ${data.filter_used}`
  return txt
}

// ── Main App ───────────────────────────────────────────────────────────
export default function App() {
  const { toasts, addToast } = useToasts()

  // Upload state
  const [pdfFile, setPdfFile]     = useState(null)
  const [dragging, setDragging]   = useState(false)
  const [uploading, setUploading] = useState(false)
  const [stats, setStats]         = useState(null)
  const [structure, setStructure] = useState([])
  const fileInputRef = useRef(null)

  // Chat state
  const [messages, setMessages]   = useState([])
  const [input, setInput]         = useState('')
  const [querying, setQuerying]   = useState(false)
  const messagesEndRef = useRef(null)

  // Settings
  const [docFilter, setDocFilter]     = useState('All')
  const [autoRoute, setAutoRoute]     = useState(true)
  const [useRerank, setUseRerank]     = useState(false)
  const [numChunks, setNumChunks]     = useState(4)
  const [docTypes, setDocTypes]       = useState(['All'])

  // Scroll to bottom
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  // ── Drag & drop handlers ────────────────────────────────────────────
  const onDrop = useCallback(e => {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f?.name.toLowerCase().endsWith('.pdf')) setPdfFile(f)
    else addToast('Only PDF files are accepted.', 'error')
  }, [addToast])

  const onFileChange = e => {
    const f = e.target.files[0]
    if (f) setPdfFile(f)
  }

  // ── Upload / Process ────────────────────────────────────────────────
  const handleProcess = async () => {
    if (!pdfFile) return addToast('Please select a PDF first.', 'error')
    setUploading(true)
    try {
      const res = await apiUpload(pdfFile)
      setStats(res.stats)
      const types = ['All', ...(res.stats.document_types || [])]
      setDocTypes(types)
      setDocFilter('All')
      const s = await apiStructure()
      setStructure(s)
      addToast('✅ Document processed successfully!', 'success')
    } catch (err) {
      addToast(`❌ ${err.message}`, 'error')
    } finally {
      setUploading(false)
    }
  }

  // ── Rerank toggle ────────────────────────────────────────────────────
  const handleRerankToggle = async val => {
    setUseRerank(val)
    try {
      if (val) addToast('⏳ Connecting to MiniLM Reranker on Modal…')
      await apiSetRerank(val)
      addToast(val ? '✅ Reranker enabled!' : '🔄 Reranking disabled.', 'success')
    } catch (err) {
      setUseRerank(!val)
      addToast(`❌ ${err.message}`, 'error')
    }
  }

  // ── Clear ────────────────────────────────────────────────────────────
  const handleClear = async () => {
    try {
      await apiClear()
      setPdfFile(null); setStats(null); setStructure([])
      setMessages([]); setDocTypes(['All']); setDocFilter('All')
      if (fileInputRef.current) fileInputRef.current.value = ''
      addToast('🗑️ Cleared all documents and chat.', 'success')
    } catch (err) {
      addToast(`❌ ${err.message}`, 'error')
    }
  }

  // ── Download chat ────────────────────────────────────────────────────
  const handleDownloadChat = () => {
    if (!messages.length) return addToast('No chat history to download.', 'error')
    let content = '=' .repeat(60) + '\nCHAT HISTORY - Document Q&A System\n' + '='.repeat(60) + '\n\n'
    const pairs = []
    let cur = ['', '']
    messages.forEach(m => {
      if (m.role === 'user') { cur[0] = m.content }
      else { cur[1] = m.content; pairs.push(cur); cur = ['', ''] }
    })
    pairs.forEach(([ u, a ], i) => {
      content += `--- Message ${i + 1} ---\nUSER: ${u}\n\nASSISTANT: ${a}\n\n${'-'.repeat(40)}\n\n`
    })
    const blob = new Blob([content], { type: 'text/plain' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href = url; a.download = 'chat_history.txt'; a.click()
    URL.revokeObjectURL(url)
  }

  // ── Send message ─────────────────────────────────────────────────────
  const handleSend = async () => {
    const q = input.trim()
    if (!q || querying) return
    if (!stats) return addToast('Please upload and process a PDF first.', 'error')

    setMessages(prev => [...prev, { role: 'user', content: q }])
    setInput('')
    setQuerying(true)

    try {
      const res = await apiQuery({
        question: q,
        filter_type: docFilter === 'All' ? null : docFilter,
        auto_route: autoRoute,
        num_chunks: numChunks,
        use_rerank: useRerank,
      })
      setMessages(prev => [...prev, { role: 'assistant', content: buildAnswerText(res) }])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: `❌ Error: ${err.message}` }])
    } finally {
      setQuerying(false)
    }
  }

  const onKeyDown = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <span className="header-icon">🚀</span>
        <div>
          <h1>Document Q&amp;A System</h1>
          <p>Intelligent Multi-Document Analysis with Hybrid RAG Pipeline</p>
        </div>
      </header>

      <div className="main-grid">
        {/* ── Left sidebar ── */}
        <aside className="sidebar">

          {/* Upload */}
          <div className="card">
            <div className="card-title">📄 Upload Document</div>
            <div
              className={`upload-zone${dragging ? ' drag-over' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <input ref={fileInputRef} type="file" accept=".pdf" onChange={onFileChange} />
              <div className="upload-icon">📂</div>
              <p>{dragging ? 'Drop it!' : 'Click or drag a PDF here'}</p>
              {pdfFile && <div className="filename">📄 {pdfFile.name}</div>}
            </div>

            <div className="btn-row" style={{ marginTop: 12 }}>
              <button className="btn btn-primary" onClick={handleProcess} disabled={!pdfFile || uploading}>
                {uploading ? <><span className="spinner" /> Processing…</> : '🔄 Process'}
              </button>
              <button className="btn btn-danger" onClick={handleClear}>🗑️ Clear</button>
            </div>
          </div>

          {/* Status */}
          <div className="card">
            <div className="card-title">📊 Status</div>
            {!stats ? (
              <div className="status-box">⏳ Waiting for PDF upload…</div>
            ) : (
              <>
                <div className="status-box success">✅ Processed: {stats.filename}</div>
                <div className="stat-grid">
                  <div className="stat-item"><div className="stat-label">Pages</div><div className="stat-value">{stats.total_pages}</div></div>
                  <div className="stat-item"><div className="stat-label">Documents</div><div className="stat-value">{stats.documents_found}</div></div>
                  <div className="stat-item"><div className="stat-label">Chunks</div><div className="stat-value">{stats.total_chunks}</div></div>
                  <div className="stat-item"><div className="stat-label">Search</div><div className="stat-value" style={{ fontSize: '0.72rem' }}>{stats.search_type}</div></div>
                </div>
              </>
            )}
          </div>

          {/* Document Structure */}
          {structure.length > 0 && (
            <div className="card">
              <div className="card-title">🗂️ Document Structure</div>
              <ul className="structure-list">
                {structure.map((doc, i) => (
                  <li key={i}>
                    <span className="doc-type">{doc.type}</span>
                    <span className="doc-meta">Pages {doc.pages} · {doc.chunks} chunks</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Search Settings */}
          <div className="card">
            <div className="card-title">⚙️ Search Settings</div>

            <div className="setting-row">
              <label className="setting-label">📑 Filter by Document Type</label>
              <select value={docFilter} onChange={e => setDocFilter(e.target.value)}>
                {docTypes.map(t => <option key={t}>{t}</option>)}
              </select>
            </div>

            <div className="setting-row">
              <div className="toggle-wrapper">
                <span className="toggle-info">🎯 Auto-route Queries</span>
                <label className="toggle">
                  <input type="checkbox" checked={autoRoute} onChange={e => setAutoRoute(e.target.checked)} />
                  <span className="toggle-slider" />
                </label>
              </div>
              <span className="setting-hint">Automatically detect document type from query</span>
            </div>

            <div className="setting-row">
              <div className="toggle-wrapper">
                <span className="toggle-info">🧠 Enable Reranking</span>
                <label className="toggle">
                  <input type="checkbox" checked={useRerank} onChange={e => handleRerankToggle(e.target.checked)} />
                  <span className="toggle-slider" />
                </label>
              </div>
              <span className="setting-hint">MiniLM-L-6 via Modal (fast & efficient reranking)</span>
            </div>

            <div className="setting-row">
              <div className="slider-row">
                <label className="setting-label">🔢 Context Chunks</label>
                <span className="slider-value">{numChunks}</span>
              </div>
              <input type="range" min={1} max={10} step={1} value={numChunks} onChange={e => setNumChunks(Number(e.target.value))} />
            </div>
          </div>

          {/* Download chat */}
          <button className="btn btn-secondary" onClick={handleDownloadChat} disabled={!messages.length}>
            💾 Download Chat History
          </button>
        </aside>

        {/* ── Chat panel ── */}
        <section className="chat-panel">
          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="empty-chat">
                <div className="empty-icon">💬</div>
                <h3>Ask a question about your document</h3>
                <p>Upload and process a PDF on the left, then type your question below.</p>
              </div>
            ) : (
              messages.map((m, i) => (
                <div key={i} className={`message-bubble ${m.role}`}>
                  <div className={`avatar ${m.role}`}>{m.role === 'user' ? '👤' : '🤖'}</div>
                  <div className={`bubble-content ${m.role}`}>{m.content}</div>
                </div>
              ))
            )}
            {querying && (
              <div className="message-bubble assistant">
                <div className="avatar assistant">🤖</div>
                <div className="bubble-content assistant" style={{ color: 'var(--text-muted)' }}>
                  <span className="spinner" style={{ display: 'inline-block', marginRight: 8 }} />
                  Thinking…
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-bar">
            <textarea
              rows={1}
              placeholder="💬 Ask a question about your document…"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              disabled={querying}
            />
            <button className="send-btn" onClick={handleSend} disabled={!input.trim() || querying}>
              {querying ? <span className="spinner" /> : '➤'}
            </button>
          </div>
        </section>
      </div>

      {/* Toast notifications */}
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className={`toast ${t.type}`}>{t.msg}</div>
        ))}
      </div>
    </div>
  )
}
