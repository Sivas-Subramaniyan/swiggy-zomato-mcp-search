import { useState, useRef, useEffect, useCallback } from 'react'

const API_BASE = '/api'

function generateSessionId() {
    return 'sess_' + Math.random().toString(36).substring(2, 10)
}

// ─── Markdown renderer ───────────────────────────────────────────────────────
function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
}

function processInline(text) {
    return escapeHtml(text)
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
}

function renderTable(lines) {
    if (lines.length < 3) return lines.map(l => `<p>${processInline(l)}</p>`).join('')
    const parseRow = line =>
        line.replace(/^\||\|$/g, '').split('|').map(c => c.trim())
    const headers = parseRow(lines[0])
    const rows = lines.slice(2).filter(l => l.trim() && l.includes('|')).map(parseRow)
    const ths = headers.map(h => `<th>${processInline(h)}</th>`).join('')
    const trs = rows.map(r =>
        `<tr>${r.map(c => `<td>${processInline(c)}</td>`).join('')}</tr>`
    ).join('')
    return `<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`
}

function renderMarkdown(text) {
    if (typeof text !== 'string') return escapeHtml(String(text))
    const lines = text.split('\n')
    const out = []
    let i = 0

    while (i < lines.length) {
        const line = lines[i]
        const trimmed = line.trim()

        // Table: header row followed by separator (---|---)
        if (
            trimmed.startsWith('|') &&
            i + 1 < lines.length &&
            /^\|[\s\-:| ]+\|/.test(lines[i + 1].trim())
        ) {
            const tableLines = []
            while (i < lines.length && lines[i].trim().startsWith('|')) {
                tableLines.push(lines[i])
                i++
            }
            out.push(renderTable(tableLines))
            continue
        }

        // Inline image: ![alt](src)
        // Payment QR images (/api/payment/qr/…) are handled exclusively by the
        // payment panel — suppress them here so they don't appear in the chat bubble.
        if (/^!\[.*?\]\(.+?\)$/.test(trimmed)) {
            const m = trimmed.match(/^!\[(.*?)\]\((.+?)\)$/)
            if (m) {
                if (!m[2].includes('/api/payment/qr/')) {
                    out.push(`<div class="qr-block"><img src="${m[2]}" alt="${escapeHtml(m[1])}" class="qr-img"/></div>`)
                }
                i++
                continue
            }
        }

        // Horizontal rule
        if (/^-{3,}$/.test(trimmed) || /^\*{3,}$/.test(trimmed)) {
            out.push('<hr/>')
        }
        // H3
        else if (line.startsWith('### ')) {
            out.push(`<h3>${processInline(line.slice(4))}</h3>`)
        }
        // H2
        else if (line.startsWith('## ')) {
            out.push(`<h2>${processInline(line.slice(3))}</h2>`)
        }
        // H1
        else if (line.startsWith('# ')) {
            out.push(`<h1>${processInline(line.slice(2))}</h1>`)
        }
        // Unordered list
        else if (/^[-*] /.test(line)) {
            out.push(`<li>${processInline(line.slice(2))}</li>`)
        }
        // Ordered list
        else if (/^\d+\. /.test(line)) {
            out.push(`<li>${processInline(line.replace(/^\d+\.\s+/, ''))}</li>`)
        }
        // Empty line
        else if (trimmed === '') {
            out.push('<br/>')
        }
        // Regular line
        else {
            out.push(`<p>${processInline(line)}</p>`)
        }

        i++
    }

    // Wrap consecutive <li> in <ul>
    const html = out.join('')
    return html.replace(/(<li>.*?<\/li>(\s*<br\/>)?)+/g, match => {
        const items = match.replace(/<br\/>/g, '')
        return `<ul>${items}</ul>`
    })
}

// ─── Platform Panel ──────────────────────────────────────────────────────────
function PlatformPanel({ platform, status, onConnect }) {
    const isSwiggy = platform === 'swiggy'
    const name = isSwiggy ? 'Swiggy' : 'Zomato'
    const isConnected = status?.status === 'ok'
    const isError = status?.status === 'error'
    const isWaiting = status?.status === 'waiting'

    let cardClass = 'platform-card'
    if (isConnected) cardClass += ` connected-${platform}`

    return (
        <div className={cardClass}>
            <div className="platform-card-top">
                <span className="platform-name">
                    <span className={`platform-dot ${platform}`} />
                    {name}
                    {isConnected && (
                        <span className="tool-count">({status.tools} tools)</span>
                    )}
                </span>
                {isConnected && (
                    <span className="status-pill ok">
                        <span className="status-pill-dot" /> Connected
                    </span>
                )}
                {isError && (
                    <span className="status-pill error">
                        <span className="status-pill-dot" /> Failed
                    </span>
                )}
                {isWaiting && (
                    <span className="status-pill waiting">
                        <span className="status-pill-dot" /> Waiting
                    </span>
                )}
            </div>

            <div className="platform-card-body">
                <button
                    className={`connect-btn ${isConnected ? 'connected' : platform}`}
                    onClick={onConnect}
                    disabled={isWaiting}
                >
                    {isWaiting
                        ? '⏳ Waiting for login…'
                        : isConnected
                            ? `✓ Connected — Re-login`
                            : `🔐 Login with ${name}`}
                </button>

                {isWaiting && (
                    <p className="waiting-msg">🔐 Complete sign-in in the popup window.</p>
                )}
                {isError && (
                    <p className="error-msg">⚠ {status.detail}</p>
                )}
                {!isConnected && !isError && !isWaiting && (
                    <p className="hint-msg">Opens {name} login in a popup.</p>
                )}
            </div>
        </div>
    )
}

// ─── App ─────────────────────────────────────────────────────────────────────
export default function App() {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [sessionId, setSessionId] = useState(generateSessionId)
    const [swiggyStatus, setSwiggyStatus] = useState(null)
    const [zomatoStatus, setZomatoStatus] = useState(null)
    const [pendingPayment, setPendingPayment] = useState(null) // {orderId, platform}

    const messagesEndRef = useRef(null)
    const inputRef = useRef(null)
    const popupRefs = useRef({ swiggy: null, zomato: null })

    const agentReady = swiggyStatus?.status === 'ok' || zomatoStatus?.status === 'ok'

    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [])

    useEffect(() => { scrollToBottom() }, [messages, loading, scrollToBottom])
    useEffect(() => {
        if (agentReady) inputRef.current?.focus()
    }, [agentReady])

    // Fetch status for a specific platform (or 'both')
    const fetchPlatformStatus = useCallback(async (targetPlatform) => {
        try {
            const res = await fetch(`${API_BASE}/status`)
            const data = await res.json()
            if (targetPlatform === 'swiggy' || targetPlatform === 'both') {
                setSwiggyStatus(data.swiggy?.status ? data.swiggy : { status: 'error', detail: 'Session expired — please re-login.' })
            }
            if (targetPlatform === 'zomato' || targetPlatform === 'both') {
                setZomatoStatus(data.zomato?.status ? data.zomato : { status: 'error', detail: 'Session expired — please re-login.' })
            }
        } catch (_) { }
    }, [])

    // Listen for postMessage from OAuth popups
    useEffect(() => {
        function onMessage(e) {
            if (e.data?.type === 'swiggy_connected') {
                fetchPlatformStatus('swiggy')
            } else if (e.data?.type === 'swiggy_error') {
                setSwiggyStatus({ status: 'error', detail: e.data.message })
            } else if (e.data?.type === 'zomato_connected') {
                fetchPlatformStatus('zomato')
            } else if (e.data?.type === 'zomato_error') {
                setZomatoStatus({ status: 'error', detail: e.data.message })
            }
        }
        window.addEventListener('message', onMessage)
        return () => window.removeEventListener('message', onMessage)
    }, [fetchPlatformStatus])

    function openAuthPopup(platform, setStatus) {
        const existing = popupRefs.current[platform]
        if (existing && !existing.closed) existing.close()

        const popup = window.open(
            `${API_BASE}/auth/${platform}/start`,
            `${platform}_auth`,
            'width=520,height=680,left=200,top=80'
        )

        if (!popup) {
            setStatus({ status: 'error', detail: 'Popup blocked — allow popups for this site and try again.' })
            return
        }

        popupRefs.current[platform] = popup
        setStatus({ status: 'waiting' })

        const timer = setInterval(() => {
            if (popup.closed) {
                clearInterval(timer)
                setStatus(prev => prev?.status === 'waiting'
                    ? { status: 'error', detail: 'Login window closed before completing sign-in.' }
                    : prev
                )
            }
        }, 1000)
    }

    async function handleSend(e) {
        e.preventDefault()
        const query = input.trim()
        if (!query || loading) return

        setMessages(prev => [...prev, { role: 'user', content: query }])
        setInput('')
        setLoading(true)

        try {
            const res = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, session_id: sessionId }),
            })
            const data = await res.json()
            setMessages(prev => [...prev, { role: 'assistant', content: data.response }])
            // Re-fetch status for platforms whose session expired this turn
            if (data.status_changed && data.expired_platforms?.length > 0) {
                data.expired_platforms.forEach(p => fetchPlatformStatus(p))
            }
            // Use qr_order_id from API (authoritative) — fall back to text regex
            const orderId = data.qr_order_id
                || data.response?.match(/\/api\/payment\/qr\/([^\s)\]"]+)/)?.[1]
            if (orderId) {
                setPendingPayment({ orderId })
            }
        } catch (err) {
            setMessages(prev => [...prev, { role: 'assistant', content: `❌ Network error: ${err.message}` }])
        }
        setLoading(false)
        inputRef.current?.focus()
    }

    const suggestions = [
        'Find biryani options near me',
        'Compare pizza prices on Swiggy vs Zomato',
        'Show me my saved addresses',
        'Best deals for butter chicken today',
    ]

    const connectedPlatforms = [
        swiggyStatus?.status === 'ok' && 'swiggy',
        zomatoStatus?.status === 'ok' && 'zomato',
    ].filter(Boolean)

    return (
        <div className="app-container">
            {/* ─── Sidebar ─── */}
            <aside className="sidebar">
                <div className="sidebar-header">
                    <div className="brand">
                        <div className="brand-icon">🍽️</div>
                        <h2>FoodCompare</h2>
                    </div>
                    <p className="brand-tagline">Compare Swiggy & Zomato — find the best deals</p>
                </div>

                <div className="sidebar-body">
                    <PlatformPanel
                        platform="swiggy"
                        status={swiggyStatus}
                        onConnect={() => openAuthPopup('swiggy', setSwiggyStatus)}
                    />
                    <PlatformPanel
                        platform="zomato"
                        status={zomatoStatus}
                        onConnect={() => openAuthPopup('zomato', setZomatoStatus)}
                    />

                    <p className="section-label">Try asking</p>
                    <div className="suggestions">
                        {suggestions.map((s, i) => (
                            <button
                                key={i}
                                className="suggestion-chip"
                                onClick={() => { setInput(s); inputRef.current?.focus() }}
                            >
                                {s}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="sidebar-footer">
                    <button className="clear-btn" onClick={() => {
                        setMessages([])
                        setSessionId(generateSessionId())
                    }}>
                        🗑 Clear chat
                    </button>
                    <span className="session-label">{sessionId}</span>
                </div>
            </aside>

            {/* ─── Main chat ─── */}
            <main className="chat-area">
                <header className="chat-header">
                    <div className="chat-header-left">
                        <h1>🍔 Food Price Comparator</h1>
                        <p>AI-powered Swiggy &amp; Zomato comparison</p>
                    </div>
                    <div className="platform-indicators">
                        {['swiggy', 'zomato'].map(p => {
                            const active = connectedPlatforms.includes(p)
                            return (
                                <span
                                    key={p}
                                    className={`platform-indicator ${active ? `active-${p}` : ''}`}
                                >
                                    <span className="indicator-dot" />
                                    {p.charAt(0).toUpperCase() + p.slice(1)}
                                </span>
                            )
                        })}
                    </div>
                </header>

                <div className="messages-container">
                    {messages.length === 0 && !loading && (
                        <div className="empty-state">
                            <div className="empty-icon">🍽️</div>
                            <h3>What would you like to eat today?</h3>
                            <p>
                                {agentReady
                                    ? "I'll compare prices across Swiggy and Zomato and find you the best deal."
                                    : 'Log in to Swiggy or Zomato (or both) in the sidebar to get started.'}
                            </p>
                            {agentReady && (
                                <div className="empty-chips">
                                    {suggestions.map((s, i) => (
                                        <button
                                            key={i}
                                            className="empty-chip"
                                            onClick={() => {
                                                setInput(s)
                                                inputRef.current?.focus()
                                            }}
                                        >
                                            {s}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {messages.map((msg, i) => (
                        <div key={i} className={`message ${msg.role}`}>
                            <div className="avatar">
                                {msg.role === 'user' ? '👤' : '🤖'}
                            </div>
                            {msg.role === 'assistant' ? (
                                <div
                                    className="bubble"
                                    dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
                                />
                            ) : (
                                <div className="bubble">{msg.content}</div>
                            )}
                        </div>
                    ))}

                    {loading && (
                        <div className="typing-indicator">
                            <div className="avatar">🤖</div>
                            <div className="typing-dots">
                                <span /><span /><span />
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {pendingPayment && (
                    <div className="payment-panel">
                        <div className="payment-panel-header">
                            <div className="payment-panel-title">
                                <span className="payment-pulse" />
                                UPI Payment Required
                            </div>
                            <button className="dismiss-btn" onClick={() => setPendingPayment(null)}>✕</button>
                        </div>
                        <div className="payment-panel-body">
                            <div className="payment-qr-wrap">
                                <img
                                    src={`${API_BASE}/payment/qr/${pendingPayment.orderId}`}
                                    alt="UPI QR Code"
                                    className="payment-qr-img"
                                    onError={e => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'block' }}
                                />
                                <p className="qr-fallback" style={{display:'none'}}>
                                    QR not available — pay via Zomato app.
                                </p>
                                <a
                                    className="qr-download-btn"
                                    href={`${API_BASE}/payment/qr/${pendingPayment.orderId}?download=1`}
                                    download={`upi-qr-${pendingPayment.orderId}.png`}
                                >
                                    ⬇ Download QR
                                </a>
                            </div>
                            <div className="payment-panel-info">
                                <p className="payment-order-id">Order ID: <strong>{pendingPayment.orderId}</strong></p>
                                <p className="payment-instruction">Scan the QR code with any UPI app to complete payment. Your order will be confirmed automatically once payment is received.</p>
                                <button
                                    className="check-payment-btn"
                                    onClick={() => {
                                        setInput(`Has order ${pendingPayment.orderId} been paid and confirmed? Check the order tracking status.`)
                                        setPendingPayment(null)
                                        inputRef.current?.focus()
                                    }}
                                >
                                    Check payment status
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                <div className="input-bar">
                    <form className="input-form" onSubmit={handleSend}>
                        <input
                            ref={inputRef}
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            placeholder={
                                agentReady
                                    ? 'Ask me to compare food prices…'
                                    : 'Login to Swiggy or Zomato first…'
                            }
                            disabled={!agentReady || loading}
                        />
                        <button
                            type="submit"
                            className="send-btn"
                            disabled={!agentReady || loading || !input.trim()}
                            aria-label="Send"
                        >
                            ↑
                        </button>
                    </form>
                    {agentReady && (
                        <p className="input-hint">
                            Connected to: {connectedPlatforms.map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(' & ')}
                        </p>
                    )}
                </div>
            </main>
        </div>
    )
}
