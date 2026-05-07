import { useState, useEffect, useRef } from 'react';

// ─── Default V1-V28 feature values ───────────────────────────────────────────
function defaultFeatures() {
  const features = {};
  for (let i = 1; i <= 28; i++) {
    features[`V${i}`] = '0.0';
  }
  return features;
}

const INITIAL_FORM = {
  Time: '9800',
  Amount: '850',
  ...defaultFeatures(),
  V14: '-6.5', // override default — most important feature
};

const STARTER_QUERIES = [
  'Esta transação de R$850 às 3h da manhã com V14=-6.5 é suspeita?',
  'Quais são os principais fatores de risco de fraude neste sistema?',
  'Como funciona o monitoramento de drift?',
];

// ─── Inline styles ────────────────────────────────────────────────────────────
const styles = `
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f7fa; color: #212121; }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  @keyframes dots {
    0%, 20%  { content: '.'; }
    40%      { content: '..'; }
    60%, 100%{ content: '...'; }
  }
  .spinner {
    width: 24px; height: 24px;
    border: 3px solid #e3e8f0;
    border-top-color: #1565c0;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    display: inline-block;
  }
  .dots::after {
    content: '';
    animation: dots 1.5s steps(1, end) infinite;
  }
`;

export default function App() {
  const [activeTab, setActiveTab] = useState('detector');
  const [healthStatus, setHealthStatus] = useState(null); // null | 'ok' | 'error'

  // Fetch health on mount
  useEffect(() => {
    fetch('/health')
      .then((r) => r.json())
      .then((data) => setHealthStatus(data.model_loaded ? 'ok' : 'error'))
      .catch(() => setHealthStatus('error'));
  }, []);

  return (
    <>
      <style>{styles}</style>
      <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <header
          style={{
            background: '#0d47a1',
            color: '#fff',
            padding: '14px 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            boxShadow: '0 2px 8px rgba(0,0,0,0.25)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 22, fontWeight: 700, letterSpacing: 0.5 }}>
              Fraud Detection
            </span>
          </div>
          <HealthBadge status={healthStatus} />
        </header>

        {/* Tab bar */}
        <nav
          style={{
            background: '#1565c0',
            display: 'flex',
            gap: 4,
            padding: '0 24px',
          }}
        >
          {['detector', 'agent'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                padding: '12px 20px',
                color: activeTab === tab ? '#fff' : 'rgba(255,255,255,0.65)',
                fontWeight: activeTab === tab ? 700 : 400,
                fontSize: 15,
                borderBottom: activeTab === tab ? '3px solid #fff' : '3px solid transparent',
                transition: 'all 0.15s',
              }}
            >
              {tab === 'detector' ? 'Detector de Fraude' : 'Agente IA'}
            </button>
          ))}
        </nav>

        {/* Main content */}
        <main style={{ flex: 1, padding: '28px 24px', maxWidth: 900, width: '100%', margin: '0 auto' }}>
          {activeTab === 'detector' ? <DetectorTab /> : <AgentTab />}
        </main>
      </div>
    </>
  );
}

// ─── Health Badge ─────────────────────────────────────────────────────────────
function HealthBadge({ status }) {
  const label =
    status === null ? 'Verificando...' : status === 'ok' ? 'Modelo carregado' : 'Modelo indisponível';
  const dotColor = status === 'ok' ? '#69f0ae' : status === 'error' ? '#ff5252' : '#bdbdbd';

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: 'rgba(255,255,255,0.9)' }}>
      <span
        style={{
          width: 10,
          height: 10,
          borderRadius: '50%',
          background: dotColor,
          display: 'inline-block',
          boxShadow: status === 'ok' ? `0 0 6px ${dotColor}` : 'none',
        }}
      />
      {label}
    </div>
  );
}

// ─── Detector Tab ─────────────────────────────────────────────────────────────
function DetectorTab() {
  const [form, setForm] = useState(INITIAL_FORM);
  const [showOthers, setShowOthers] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  function handleChange(e) {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    // Build numeric payload
    const payload = {};
    for (const key of Object.keys(form)) {
      payload[key] = parseFloat(form[key]) || 0;
    }

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.text();
        throw new Error(`HTTP ${res.status}: ${err}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Erro ao contatar a API.');
    } finally {
      setLoading(false);
    }
  }

  // V1-V28 except V14
  const otherFeatures = [];
  for (let i = 1; i <= 28; i++) {
    if (i !== 14) otherFeatures.push(`V${i}`);
  }

  return (
    <div>
      <h2 style={{ fontSize: 20, fontWeight: 600, marginBottom: 20, color: '#1565c0' }}>
        Analisar Transacao
      </h2>

      {error && (
        <div
          style={{
            background: '#ffebee',
            border: '1px solid #ef9a9a',
            color: '#c62828',
            padding: '12px 16px',
            borderRadius: 8,
            marginBottom: 16,
            fontSize: 14,
          }}
        >
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        {/* Primary fields */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: 16,
            marginBottom: 16,
          }}
        >
          <FieldGroup label="Tempo (segundos)" name="Time" value={form.Time} onChange={handleChange} />
          <FieldGroup label="Valor (BRL)" name="Amount" value={form.Amount} onChange={handleChange} />
        </div>

        {/* V14 highlighted */}
        <div
          style={{
            background: '#e3f2fd',
            border: '2px solid #1565c0',
            borderRadius: 10,
            padding: '14px 16px',
            marginBottom: 16,
          }}
        >
          <FieldGroup
            label="V14 — feature mais importante para o modelo"
            name="V14"
            value={form.V14}
            onChange={handleChange}
            highlight
          />
        </div>

        {/* Collapsible others */}
        <div
          style={{
            border: '1px solid #e0e0e0',
            borderRadius: 10,
            overflow: 'hidden',
            marginBottom: 20,
          }}
        >
          <button
            type="button"
            onClick={() => setShowOthers((v) => !v)}
            style={{
              width: '100%',
              background: '#f5f7fa',
              border: 'none',
              cursor: 'pointer',
              padding: '12px 16px',
              textAlign: 'left',
              fontSize: 14,
              fontWeight: 600,
              color: '#424242',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <span>Outras features (V1-V28)</span>
            <span style={{ fontSize: 18, lineHeight: 1 }}>{showOthers ? '▲' : '▼'}</span>
          </button>

          {showOthers && (
            <div
              style={{
                padding: 16,
                display: 'grid',
                gridTemplateColumns: 'repeat(4, 1fr)',
                gap: 10,
                background: '#fff',
              }}
            >
              {otherFeatures.map((key) => (
                <SmallField key={key} name={key} value={form[key]} onChange={handleChange} />
              ))}
            </div>
          )}
        </div>

        <button
          type="submit"
          disabled={loading}
          style={{
            background: loading ? '#90a4ae' : '#1565c0',
            color: '#fff',
            border: 'none',
            borderRadius: 8,
            padding: '13px 32px',
            fontSize: 15,
            fontWeight: 600,
            cursor: loading ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            transition: 'background 0.2s',
          }}
        >
          {loading && <span className="spinner" />}
          {loading ? 'Analisando...' : 'Analisar Transacao'}
        </button>
      </form>

      {result && <ResultCard result={result} />}
    </div>
  );
}

// ─── Field Group ──────────────────────────────────────────────────────────────
function FieldGroup({ label, name, value, onChange, highlight }) {
  return (
    <div>
      <label
        htmlFor={name}
        style={{
          display: 'block',
          fontSize: 13,
          fontWeight: 600,
          color: highlight ? '#0d47a1' : '#616161',
          marginBottom: 6,
        }}
      >
        {label}
      </label>
      <input
        id={name}
        name={name}
        type="number"
        step="any"
        value={value}
        onChange={onChange}
        style={{
          width: '100%',
          padding: '10px 12px',
          border: highlight ? '2px solid #1565c0' : '1px solid #bdbdbd',
          borderRadius: 7,
          fontSize: 15,
          outline: 'none',
          background: '#fff',
        }}
      />
    </div>
  );
}

// ─── Small Field (for V1-V28 grid) ────────────────────────────────────────────
function SmallField({ name, value, onChange }) {
  return (
    <div>
      <label
        htmlFor={name}
        style={{ display: 'block', fontSize: 11, fontWeight: 600, color: '#757575', marginBottom: 3 }}
      >
        {name}
      </label>
      <input
        id={name}
        name={name}
        type="number"
        step="any"
        value={value}
        onChange={onChange}
        style={{
          width: '100%',
          padding: '6px 8px',
          border: '1px solid #e0e0e0',
          borderRadius: 5,
          fontSize: 12,
          outline: 'none',
          background: '#fafafa',
        }}
      />
    </div>
  );
}

// ─── Result Card ──────────────────────────────────────────────────────────────
function ResultCard({ result }) {
  const { fraud_score, label, threshold } = result;
  const isFraud = label === 1 || label === 'fraud' || label === true;
  const scoreNum = parseFloat(fraud_score) || 0;
  const thresholdNum = parseFloat(threshold) || 0;

  return (
    <div
      style={{
        marginTop: 28,
        background: '#fff',
        border: `2px solid ${isFraud ? '#ef5350' : '#66bb6a'}`,
        borderRadius: 12,
        padding: 24,
        boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 20 }}>
        <span
          style={{
            background: isFraud ? '#ef5350' : '#66bb6a',
            color: '#fff',
            fontWeight: 700,
            fontSize: 18,
            padding: '8px 22px',
            borderRadius: 30,
            letterSpacing: 1,
          }}
        >
          {isFraud ? 'FRAUDE' : 'LEGITIMA'}
        </span>
        <div>
          <div style={{ fontSize: 14, color: '#757575' }}>Score de fraude</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: isFraud ? '#c62828' : '#2e7d32' }}>
            {(scoreNum * 100).toFixed(2)}%
          </div>
        </div>
        <div>
          <div style={{ fontSize: 14, color: '#757575' }}>Threshold</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#424242' }}>
            {(thresholdNum * 100).toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Score bar */}
      <div>
        <div style={{ fontSize: 12, color: '#9e9e9e', marginBottom: 6 }}>Score vs Threshold</div>
        <div
          style={{
            position: 'relative',
            height: 20,
            background: '#e8f5e9',
            borderRadius: 10,
            overflow: 'visible',
          }}
        >
          {/* Score fill */}
          <div
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              height: '100%',
              width: `${Math.min(scoreNum * 100, 100)}%`,
              background: isFraud ? '#ef5350' : '#66bb6a',
              borderRadius: 10,
              transition: 'width 0.5s ease',
            }}
          />
          {/* Threshold marker */}
          <div
            style={{
              position: 'absolute',
              top: -4,
              bottom: -4,
              left: `${Math.min(thresholdNum * 100, 100)}%`,
              width: 3,
              background: '#1565c0',
              borderRadius: 2,
            }}
          />
          {/* Threshold label */}
          <div
            style={{
              position: 'absolute',
              top: 24,
              left: `${Math.min(thresholdNum * 100, 100)}%`,
              transform: 'translateX(-50%)',
              fontSize: 10,
              color: '#1565c0',
              fontWeight: 600,
              whiteSpace: 'nowrap',
            }}
          >
            Threshold
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Agent Tab ────────────────────────────────────────────────────────────────
function AgentTab() {
  const [model, setModel] = useState('llama3.2:3b');
  const [messages, setMessages] = useState([]); // { role: 'user'|'agent', text, steps }
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  async function sendMessage(query) {
    if (!query.trim()) return;
    setError(null);
    setMessages((prev) => [...prev, { role: 'user', text: query }]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('/agent/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), model_name: model }),
      });
      if (!res.ok) {
        const err = await res.text();
        throw new Error(`HTTP ${res.status}: ${err}`);
      }
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: 'agent', text: data.answer, steps: data.steps },
      ]);
    } catch (err) {
      setError(err.message || 'Erro ao contatar o agente.');
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <h2 style={{ fontSize: 20, fontWeight: 600, color: '#1565c0' }}>Agente IA</h2>

        {/* Model selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <label htmlFor="model-select" style={{ fontSize: 13, fontWeight: 600, color: '#616161' }}>
            Modelo:
          </label>
          <select
            id="model-select"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            style={{
              padding: '7px 12px',
              border: '1px solid #bdbdbd',
              borderRadius: 7,
              fontSize: 14,
              background: '#fff',
              cursor: 'pointer',
              outline: 'none',
            }}
          >
            <option value="llama3.2:3b">llama3.2:3b</option>
            <option value="gpt-4o-mini">gpt-4o-mini</option>
          </select>
        </div>
      </div>

      {error && (
        <div
          style={{
            background: '#ffebee',
            border: '1px solid #ef9a9a',
            color: '#c62828',
            padding: '12px 16px',
            borderRadius: 8,
            marginBottom: 16,
            fontSize: 14,
          }}
        >
          {error}
        </div>
      )}

      {/* Chat window */}
      <div
        style={{
          background: '#fff',
          border: '1px solid #e0e0e0',
          borderRadius: 12,
          minHeight: 300,
          maxHeight: 400,
          overflowY: 'auto',
          padding: 16,
          marginBottom: 12,
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
        }}
      >
        {messages.length === 0 && !loading && (
          <StarterChips onSelect={sendMessage} />
        )}

        {messages.map((msg, idx) => (
          <ChatBubble key={idx} msg={msg} />
        ))}

        {loading && (
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
            <div
              style={{
                background: '#f5f5f5',
                border: '1px solid #e0e0e0',
                borderRadius: '4px 12px 12px 12px',
                padding: '10px 14px',
                fontSize: 14,
                color: '#616161',
              }}
            >
              Agente pensando<span className="dots" />
            </div>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Input area */}
      <div style={{ display: 'flex', gap: 10 }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Digite sua pergunta... (Enter para enviar)"
          rows={2}
          style={{
            flex: 1,
            padding: '10px 14px',
            border: '1px solid #bdbdbd',
            borderRadius: 8,
            fontSize: 14,
            resize: 'none',
            outline: 'none',
            fontFamily: 'inherit',
          }}
        />
        <button
          onClick={() => sendMessage(input)}
          disabled={loading || !input.trim()}
          style={{
            background: loading || !input.trim() ? '#90a4ae' : '#1565c0',
            color: '#fff',
            border: 'none',
            borderRadius: 8,
            padding: '0 22px',
            fontWeight: 600,
            fontSize: 15,
            cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
            transition: 'background 0.2s',
            alignSelf: 'stretch',
          }}
        >
          Enviar
        </button>
      </div>
    </div>
  );
}

// ─── Chat Bubble ──────────────────────────────────────────────────────────────
function ChatBubble({ msg }) {
  const isUser = msg.role === 'user';
  const stepsCount = Array.isArray(msg.steps) ? msg.steps.length : typeof msg.steps === 'number' ? msg.steps : null;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        gap: 4,
      }}
    >
      <div
        style={{
          maxWidth: '80%',
          background: isUser ? '#1565c0' : '#f5f5f5',
          color: isUser ? '#fff' : '#212121',
          padding: '10px 14px',
          borderRadius: isUser ? '12px 4px 12px 12px' : '4px 12px 12px 12px',
          fontSize: 14,
          lineHeight: 1.5,
          border: isUser ? 'none' : '1px solid #e0e0e0',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {msg.text}
      </div>
      {!isUser && stepsCount !== null && (
        <span
          style={{
            fontSize: 11,
            fontWeight: 600,
            background: '#e8eaf6',
            color: '#3949ab',
            padding: '2px 8px',
            borderRadius: 10,
          }}
        >
          Steps: {stepsCount}
        </span>
      )}
    </div>
  );
}

// ─── Starter Chips ────────────────────────────────────────────────────────────
function StarterChips({ onSelect }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center', justifyContent: 'center', flex: 1, padding: '20px 0' }}>
      <p style={{ fontSize: 13, color: '#9e9e9e', marginBottom: 8 }}>
        Sugestoes de perguntas:
      </p>
      {STARTER_QUERIES.map((q, i) => (
        <button
          key={i}
          onClick={() => onSelect(q)}
          style={{
            background: '#e3f2fd',
            border: '1px solid #90caf9',
            color: '#1565c0',
            borderRadius: 20,
            padding: '8px 16px',
            fontSize: 13,
            cursor: 'pointer',
            maxWidth: 540,
            textAlign: 'center',
            transition: 'background 0.15s',
          }}
          onMouseOver={(e) => (e.currentTarget.style.background = '#bbdefb')}
          onMouseOut={(e) => (e.currentTarget.style.background = '#e3f2fd')}
        >
          {q}
        </button>
      ))}
    </div>
  );
}
