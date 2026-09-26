import { useEffect, useRef } from 'react'
import './Landing.css'

const ANSWER = ['No', '—', 'the', 'pay', 'slips', 'support']

const STEPS = [
  {
    n: '1',
    title: 'Upload the borrower’s file',
    body: 'One packet PDF, or the documents as separate files — they are merged in order. Each page is read and sorted into application, pay slips, bank statements, loan estimate and title report.',
  },
  {
    n: '2',
    title: 'Get the review',
    body: 'The application’s claims are checked against the evidence automatically: stated income against pay slips, balances and deposits against bank statements, loan terms against the loan estimate, liens on the title report.',
  },
  {
    n: '3',
    title: 'Ask about anything else',
    body: 'Chat with the file for what a checklist doesn’t cover. Every answer is quoted from the documents with its page — and it says so when the file doesn’t contain the answer.',
  },
]

const FEATURES = [
  {
    title: 'Mismatches flagged with proof',
    body: 'Each check shows both sides — what the applicant stated and what the documents show — with the page for each. You make the call; the review makes sure nothing is skipped.',
  },
  {
    title: 'Page-accurate citations',
    body: 'Every answer names the document and the page it came from, with the quoted passage one click away. A value you cannot trace is a value you cannot trust.',
  },
  {
    title: 'Grounded, never invented',
    body: 'Figures are quoted, not computed. Extracted values are checked against the document text, and a value the file does not contain is reported as not found — never guessed.',
  },
]

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
  return (
    <div className="lp">
      <nav className="lp-nav">
        <span className="lp-wordmark"><span className="lp-logo">📄</span> DocQA</span>
        <div className="lp-nav-actions">
          <button className="btn btn-ghost" onClick={onSignIn}>Sign in</button>
          <button className="btn btn-primary" onClick={onGetStarted}>Get started</button>
        </div>
      </nav>

      <header className="lp-hero">
        <div className="lp-hero-copy">
          <p className="lp-eyebrow lp-fade" style={{ '--d': '0ms' }}>Loan file review · cited answers</p>
          <h1 className="lp-headline lp-fade" style={{ '--d': '110ms' }}>
            Review a loan file in minutes.<br />Every flag, sourced&nbsp;to&nbsp;the&nbsp;page.
          </h1>
          <p className="lp-sub lp-fade" style={{ '--d': '230ms' }}>
            Upload a borrower&rsquo;s packet. The application is checked against the pay
            slips, bank statements, loan estimate and title report automatically — income,
            deposits, balances, loan terms, liens — and anything else is one cited question away.
          </p>
          <div className="lp-cta-row lp-fade" style={{ '--d': '350ms' }}>
            <button className="btn btn-primary btn-lg" onClick={onGetStarted}>Get started</button>
            <a className="btn btn-ghost btn-lg" href="#how">See how it works</a>
          </div>
        </div>

        <div className="lp-mock lp-fade" style={{ '--d': '470ms' }}>
          <div className="lp-mock-bar">
            <span className="lp-dot r" /><span className="lp-dot y" /><span className="lp-dot g" />
            <span className="lp-mock-file">whitfield_loan_packet.pdf</span>
          </div>
          <div className="lp-mock-body">
            <div className="lp-msg user">Does the stated income match the pay slips?</div>
            <div className="lp-msg bot">
              <span className="lp-typing" aria-hidden="true">
                <span className="lp-tdot" /><span className="lp-tdot" /><span className="lp-tdot" />
              </span>
              <span className="lp-answer">
                {ANSWER.map((w, i) => (
                  <span key={i} className="lp-word" style={{ '--i': i }}>{w}</span>
                ))}
                <span className="lp-word lp-value" style={{ '--i': ANSWER.length }}>$8,750/month</span>
                <span className="lp-word lp-period" style={{ '--i': ANSWER.length }}>.</span>
                <span className="lp-caret" />
              </span>
              <div className="lp-cite">
                <span className="lp-chip">Pay Slip · p.4</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <section className="lp-section" id="how">
        <Reveal><h2 className="lp-h2">From packet to reviewed file in three steps</h2></Reveal>
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

      <section className="lp-section">
        <Reveal>
          <div className="lp-cta-band">
            <h2 className="lp-h2 lp-cta-title">Review your next loan file in minutes.</h2>
            <p className="lp-body lp-cta-sub">Create an account, upload a borrower’s packet, and get the review.</p>
            <button className="btn btn-primary btn-lg" onClick={onGetStarted}>Get started</button>
          </div>
        </Reveal>
      </section>

      <footer className="lp-footer">
        <span className="lp-copyright">© {new Date().getFullYear()} DocQA. All rights reserved.</span>
      </footer>
    </div>
  )
}
