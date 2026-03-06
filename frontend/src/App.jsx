import { useState, useRef, useEffect, useCallback } from 'react'

const API_BASE = '/api'

function generateSessionId() {
    return 'sess_' + Math.random().toString(36).substring(2, 10)
}

function formatMessage(text) {
    if (typeof text !== 'string') return String(text)
    // Bold **text**
    let formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    // Newlines
    formatted = formatted.replace(/\n/g, '<br/>')
    return formatted
}

export default function App() {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [connected, setConnected] = useState(false)
    const [connecting, setConnecting] = useState(false)
    const [connectionStatus, setConnectionStatus] = useState([])
    const [sessionId] = useState(generateSessionId)
    const messagesEndRef = useRef(null)
    const inputRef = useRef(null)

    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [])

    useEffect(() => {
        scrollToBottom()
    }, [messages, loading, scrollToBottom])

    useEffect(() => {
        inputRef.current?.focus()
    }, [connected])

    async function handleConnect() {
        setConnecting(true)
        setConnectionStatus([])
        try {
            const res = await fetch(`${API_BASE}/connect`, { method: 'POST' })
            const data = await res.json()
            setConnected(data.connected)
            setConnectionStatus(data.servers || [])
        } catch (err) {
            setConnectionStatus([{ name: 'Error', status: 'error', detail: err.message }])
        }
        setConnecting(false)
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

    function handleClear() {
        setMessages([])
    }

    function handleSuggestion(text) {
        setInput(text)
        inputRef.current?.focus()
    }

    const suggestions = [
        'I want to order Narmada Chicken Biriyani',
        'Find me the cheapest pizza near me',
        'Show me my saved addresses',
    ]

    return (
        <div className="app-container">
            {/* ─── Sidebar ─── */}
            <aside className="sidebar">
                <div className="sidebar-brand">
                    <h2>🍔 Food Aggregator</h2>
                    <p>Compare Swiggy &amp; Zomato prices</p>
                </div>

                <button
                    className={`btn ${connected ? 'btn' : 'btn-primary'}`}
                    onClick={handleConnect}
                    disabled={connecting}
                >
                    {connecting ? '⏳ Connecting...' : connected ? '✅ Connected' : '🔌 Connect to MCP Servers'}
                </button>

                {connectionStatus.length > 0 && (
                    <div className="connection-list">
                        {connectionStatus.map((s, i) => (
                            <div key={i} className="connection-item">
                                <span className={`status-dot ${s.status === 'ok' ? 'connected' : 'error'}`} />
                                {s.name} {s.status === 'ok' ? `(${s.tools} tools)` : `— ${s.detail || 'failed'}`}
                            </div>
                        ))}
                    </div>
                )}

                <button className="btn btn-danger" onClick={handleClear}>
                    🗑️ Clear Chat
                </button>

                <div className="sidebar-section">
                    <h4>💡 Try asking</h4>
                    <ul>
                        {suggestions.map((s, i) => (
                            <li key={i} onClick={() => handleSuggestion(s)}>{s}</li>
                        ))}
                    </ul>
                </div>

                <div className="sidebar-footer">
                    <span className="session-id">Session: {sessionId}</span>
                </div>
            </aside>

            {/* ─── Main ─── */}
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
                            <p>
                                {connected
                                    ? 'Type your food query below and I\'ll search both Swiggy and Zomato for the best deals!'
                                    : 'Click "Connect to MCP Servers" in the sidebar to get started.'}
                            </p>
                        </div>
                    )}

                    {messages.map((msg, i) => (
                        <div key={i} className={`message ${msg.role}`}>
                            <div className="message-avatar">
                                {msg.role === 'user' ? '👤' : '🤖'}
                            </div>
                            <div
                                className="message-content"
                                dangerouslySetInnerHTML={{ __html: formatMessage(msg.content) }}
                            />
                        </div>
                    ))}

                    {loading && (
                        <div className="typing-indicator">
                            <div className="message-avatar" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                                🤖
                            </div>
                            <div className="typing-dots">
                                <span /><span /><span />
                            </div>
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
                            placeholder={connected ? 'What would you like to eat today?' : 'Connect to MCP servers first...'}
                            disabled={!connected || loading}
                        />
                        <button type="submit" className="send-btn" disabled={!connected || loading || !input.trim()}>
                            ↑
                        </button>
                    </form>
                </div>
            </main>
        </div>
    )
}
