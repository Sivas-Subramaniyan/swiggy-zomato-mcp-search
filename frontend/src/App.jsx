import { useState, useRef, useEffect, useCallback } from 'react'

const API_BASE = '/api'

function generateSessionId() {
    return 'sess_' + Math.random().toString(36).substring(2, 10)
}

function formatMessage(text) {
    if (typeof text !== 'string') return String(text)
    let formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    formatted = formatted.replace(/\n/g, '<br/>')
    return formatted
}

function PlatformPanel({ name, platform, status, onConnect }) {
    const isSwiggy = platform === 'swiggy'
    const isConnected = status?.status === 'ok'
    const isError = status?.status === 'error'
    const isWaiting = status?.status === 'waiting'

    return (
        <div className="connect-panel">
            <div className="connect-panel-header">
                <span className="platform-label">{isSwiggy ? '🟠' : '🔴'} {name}</span>
                {isConnected && <span className="status-badge ok">✓ {status.tools} tools</span>}
                {isError && <span className="status-badge error">✗ Failed</span>}
            </div>

            {isWaiting && <p className="auth-msg">🔐 Login window opened — complete sign-in to connect.</p>}

            <button
                className={`btn btn-full ${isConnected ? 'btn-connected' : isSwiggy ? 'btn-swiggy' : 'btn-zomato'}`}
                onClick={onConnect}
                disabled={isWaiting}
            >
                {isWaiting
                    ? '⏳ Waiting for login...'
                    : isConnected
                        ? `✅ Connected — Reconnect`
                        : `🔐 Login with ${name}`}
            </button>

            {isError && <p className="connect-error">{status.detail}</p>}
            {!isConnected && !isError && !isWaiting && (
                <p className="token-hint">Opens a {name} login window in your browser.</p>
            )}
        </div>
    )
}

export default function App() {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [sessionId] = useState(generateSessionId)
    const [swiggyStatus, setSwiggyStatus] = useState(null)
    const [zomatoStatus, setZomatoStatus] = useState(null)

    const messagesEndRef = useRef(null)
    const inputRef = useRef(null)
    const popupRefs = useRef({ swiggy: null, zomato: null })

    const agentReady = swiggyStatus?.status === 'ok' || zomatoStatus?.status === 'ok'

    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [])

    useEffect(() => { scrollToBottom() }, [messages, loading, scrollToBottom])
    useEffect(() => { if (agentReady) inputRef.current?.focus() }, [agentReady])

    // Sync status from backend after OAuth completes
    const fetchStatus = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/status`)
            const data = await res.json()
            if (data.swiggy?.status) setSwiggyStatus(data.swiggy)
            if (data.zomato?.status) setZomatoStatus(data.zomato)
        } catch (_) { }
    }, [])

    // Listen for postMessage from OAuth popups
    useEffect(() => {
        function onMessage(e) {
            if (e.data?.type === 'swiggy_connected') {
                setSwiggyStatus(null)     // will be overwritten by fetchStatus
                fetchStatus()
            } else if (e.data?.type === 'swiggy_error') {
                setSwiggyStatus({ status: 'error', detail: e.data.message })
            } else if (e.data?.type === 'zomato_connected') {
                setZomatoStatus(null)
                fetchStatus()
            } else if (e.data?.type === 'zomato_error') {
                setZomatoStatus({ status: 'error', detail: e.data.message })
            }
        }
        window.addEventListener('message', onMessage)
        return () => window.removeEventListener('message', onMessage)
    }, [fetchStatus])

    function openAuthPopup(platform, setStatus) {
        const existing = popupRefs.current[platform]
        if (existing && !existing.closed) existing.close()

        const popup = window.open(
            `${API_BASE}/auth/${platform}/start`,
            `${platform}_auth`,
            'width=520,height=680,left=200,top=80'
        )

        if (!popup) {
            setStatus({ status: 'error', detail: 'Popup blocked — please allow popups for this site and try again.' })
            return
        }

        popupRefs.current[platform] = popup
        setStatus({ status: 'waiting' })

        // Detect manual close without completing auth
        const timer = setInterval(() => {
            if (popup.closed) {
                clearInterval(timer)
                setStatus(prev => prev?.status === 'waiting'
                    ? { status: 'error', detail: 'Login window closed before completing authentication.' }
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
        } catch (err) {
            setMessages(prev => [...prev, { role: 'assistant', content: `❌ Error: ${err.message}` }])
        }
        setLoading(false)
        inputRef.current?.focus()
    }

    const suggestions = [
        'I want to order Narmada Chicken Biriyani',
        'Find me the cheapest pizza near me',
        'Show me my saved addresses',
    ]

    return (
        <div className="app-container">
            <aside className="sidebar">
                <div className="sidebar-brand">
                    <h2>🍔 Food Aggregator</h2>
                    <p>Compare Swiggy &amp; Zomato prices</p>
                </div>

                <PlatformPanel
                    name="Swiggy"
                    platform="swiggy"
                    status={swiggyStatus}
                    onConnect={() => openAuthPopup('swiggy', setSwiggyStatus)}
                />

                <PlatformPanel
                    name="Zomato"
                    platform="zomato"
                    status={zomatoStatus}
                    onConnect={() => openAuthPopup('zomato', setZomatoStatus)}
                />

                <button className="btn btn-danger btn-full" onClick={() => setMessages([])}>
                    🗑️ Clear Chat
                </button>

                <div className="sidebar-section">
                    <h4>💡 Try asking</h4>
                    <ul>
                        {suggestions.map((s, i) => (
                            <li key={i} onClick={() => { setInput(s); inputRef.current?.focus() }}>{s}</li>
                        ))}
                    </ul>
                </div>

                <div className="sidebar-footer">
                    <span className="session-id">Session: {sessionId}</span>
                </div>
            </aside>

            <main className="chat-area">
                <header className="chat-header">
                    <h1>🍔 Food Aggregator Agent</h1>
                    <p>Compare Swiggy &amp; Zomato — find the best deals on your favourite food</p>
                </header>

                <div className="messages-container">
                    {messages.length === 0 && !loading && (
                        <div className="empty-state">
                            <span className="emoji">🍽️</span>
                            <h3>What would you like to eat today?</h3>
                            <p>{agentReady
                                ? "Type your food query below and I'll find the best deals!"
                                : 'Log in to Swiggy or Zomato in the sidebar to get started.'}
                            </p>
                        </div>
                    )}

                    {messages.map((msg, i) => (
                        <div key={i} className={`message ${msg.role}`}>
                            <div className="message-avatar">{msg.role === 'user' ? '👤' : '🤖'}</div>
                            <div className="message-content"
                                dangerouslySetInnerHTML={{ __html: formatMessage(msg.content) }} />
                        </div>
                    ))}

                    {loading && (
                        <div className="typing-indicator">
                            <div className="message-avatar" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>🤖</div>
                            <div className="typing-dots"><span /><span /><span /></div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                <div className="input-bar">
                    <form className="input-form" onSubmit={handleSend}>
                        <input
                            ref={inputRef}
                            value={input}
                            onChange={e => setInput(e.target.value)}
                            placeholder={agentReady ? 'What would you like to eat today?' : 'Log in to a platform first...'}
                            disabled={!agentReady || loading}
                        />
                        <button type="submit" className="send-btn" disabled={!agentReady || loading || !input.trim()}>↑</button>
                    </form>
                </div>
            </main>
        </div>
    )
}
