import { useEffect, useRef, useState } from 'react'
import './Landing.css'

const prefersReduced = () =>
  window.matchMedia('(prefers-reduced-motion: reduce)').matches

// The three-word streamed answer in the hero mock. Rendered as staggered spans
// so it reads as if the model is writing it, without a JS typing loop.
const ANSWER = ['The', 'Total', 'Loan', 'Amount', 'is']

const STEPS = [
  {
    n: '1',
    title: 'Upload a document',
    body: 'One PDF per conversation. It is extracted, split into its logical documents, and indexed for hybrid search — usually in under a minute.',
  },
  {
    n: '2',
    title: 'Ask in plain English',
    body: 'Type a question. Follow-ups are rewritten to stand on their own, so “and when does it lock?” still finds the right chunk.',
  },
  {
    n: '3',
    title: 'Get a cited answer',
    body: 'Answers stream in token by token, every figure quoted verbatim and linked to its page. A wrong number is worse than none — so it refuses when it is not sure.',
  },
]

const FEATURES = [
  {
    title: 'Hybrid retrieval',
    body: 'Dense semantic search and BM25 keyword search, fused in a single Pinecone query — so an exact figure lookup and a fuzzy concept both land.',
  },
  {
    title: 'Page-accurate citations',
    body: 'Every answer names the document type and the page it came from. The citation is the deliverable: a value you cannot trace is a value you cannot trust.',
  },
  {
    title: 'Grounded, never invented',
    body: 'Answers are drawn only from your document and quoted, not computed. No plausible-looking total the packet never actually stated.',
  },
]

// Count a number up once, ease-out cubic. Reduced motion → straight to target.
function useCountUp(target, durationMs, delayMs) {
  const [value, setValue] = useState(() => (prefersReduced() ? target : 0))
  useEffect(() => {
    if (prefersReduced()) return
    let raf
    const timer = setTimeout(() => {
      const start = performance.now()
      const tick = (now) => {
        const t = Math.min((now - start) / durationMs, 1)
        setValue(Math.round(target * (1 - Math.pow(1 - t, 3))))
        if (t < 1) raf = requestAnimationFrame(tick)
      }
      raf = requestAnimationFrame(tick)
    }, delayMs)
    return () => { clearTimeout(timer); cancelAnimationFrame(raf) }
  }, [target, durationMs, delayMs])
  return value
}

// Reveal children on scroll — adds .lp-in when the block enters the viewport.
function Reveal({ children, className = '' }) {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) { el.classList.add('lp-in'); io.disconnect() }
      },
      { threshold: 0.15 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])
  return <div ref={ref} className={`lp-reveal ${className}`}>{children}</div>
}

export default function Landing({ onSignIn, onGetStarted }) {
  const amount = useCountUp(380000, 1200, 2200)
  const dollars = `$${amount.toLocaleString('en-US')}`

  return (
    <div className="lp">
      <nav className="lp-nav">
        <span className="lp-wordmark"><span className="lp-logo">📄</span> DocQA</span>
        <div className="lp-nav-actions">
          <button className="btn btn-ghost" onClick={onSignIn}>Sign in</button>
          <button className="btn btn-primary" onClick={onGetStarted}>Get started</button>
        </div>
      </nav>

      {/* Hero */}
      <header className="lp-hero">
        <div className="lp-hero-copy">
          <p className="lp-eyebrow lp-fade" style={{ '--d': '0ms' }}>Hybrid retrieval · cited answers</p>
          <h1 className="lp-headline lp-fade" style={{ '--d': '110ms' }}>
            Ask your document.<br />Every figure, sourced&nbsp;to&nbsp;the&nbsp;page.
          </h1>
          <p className="lp-sub lp-fade" style={{ '--d': '230ms' }}>
            Upload a PDF, ask in plain English, and get exact values back — each one
            quoted from the document and linked to the page it came from. Hybrid search,
            streamed answers, no hallucinated numbers.
          </p>
          <div className="lp-cta-row lp-fade" style={{ '--d': '350ms' }}>
            <button className="btn btn-primary btn-lg" onClick={onGetStarted}>Get started — it’s free</button>
            <a className="btn btn-ghost btn-lg" href="#how">See how it works</a>
          </div>
        </div>

        {/* Animated product mock — a real cited, streamed answer */}
        <div className="lp-mock lp-fade" style={{ '--d': '470ms' }}>
          <div className="lp-mock-bar">
            <span className="lp-dot r" /><span className="lp-dot y" /><span className="lp-dot g" />
            <span className="lp-mock-file">Test Blob File.pdf</span>
          </div>
          <div className="lp-mock-body">
            <div className="lp-msg user">What is the total loan amount?</div>
            <div className="lp-msg bot">
              <span className="lp-answer">
                {ANSWER.map((w, i) => (
                  <span key={i} className="lp-word" style={{ '--i': i }}>{w} </span>
                ))}
                <span className="lp-value">{dollars}</span>.
                <span className="lp-caret" />
              </span>
              <div className="lp-cite">
                <span className="lp-chip">Lender Fee Sheet · p.1</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* How it works */}
      <section className="lp-section" id="how">
        <Reveal><h2 className="lp-h2">From PDF to cited answer in three steps</h2></Reveal>
        <div className="lp-steps">
          {STEPS.map((s, i) => (
            <Reveal key={s.n} className={`lp-stagger-${i}`}>
              <div className="lp-step">
                <div className="lp-step-n">{s.n}</div>
                <h3 className="lp-h3">{s.title}</h3>
                <p className="lp-body">{s.body}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="lp-section lp-band">
        <div className="lp-features">
          {FEATURES.map((f, i) => (
            <Reveal key={f.title} className={`lp-stagger-${i}`}>
              <div className="lp-feature">
                <div className="lp-feature-rule" />
                <h3 className="lp-h3">{f.title}</h3>
                <p className="lp-body">{f.body}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* Closing CTA */}
      <section className="lp-section lp-section-tight">
        <Reveal>
          <div className="lp-cta-band">
            <h2 className="lp-h2">Ask your first document in minutes</h2>
            <p className="lp-body lp-cta-sub">Create an account, upload a PDF, and start asking.</p>
            <button className="btn btn-primary btn-lg" onClick={onGetStarted}>Get started</button>
          </div>
        </Reveal>
      </section>

      <footer className="lp-footer">
        <span className="lp-wordmark"><span className="lp-logo">📄</span> DocQA</span>
        <span className="lp-copyright">© {new Date().getFullYear()} · Mortgage-packet Q&amp;A with page-accurate citations</span>
      </footer>
    </div>
  )
}
