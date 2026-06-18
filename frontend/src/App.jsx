import React, { useState, useEffect, useRef } from 'react';

// Sensible defaults for models
const DEFAULT_MODELS = {
  groq: 'llama-3.3-70b-versatile',
  ollama: 'llama3',
  gemini: 'gemini-1.5-flash',
  claude: 'claude-3-5-sonnet-latest'
};

const DEFAULT_URLS = {
  ollama: 'http://localhost:11434'
};

const BACKEND_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  // Settings State
  const [provider, setProvider] = useState(() => localStorage.getItem('qm_provider') || 'groq');
  const [apiKey, setApiKey] = useState(() => localStorage.getItem(`qm_key_${provider}`) || '');
  const [baseUrl, setBaseUrl] = useState(() => localStorage.getItem(`qm_url_${provider}`) || DEFAULT_URLS[provider] || '');
  const [model, setModel] = useState(() => localStorage.getItem(`qm_model_${provider}`) || DEFAULT_MODELS[provider] || '');

  // Workspace / Upload State
  const [activeMode, setActiveMode] = useState('pdf'); // 'pdf' or 'code'
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [docStatus, setDocStatus] = useState({
    filename: null,
    file_type: null,
    page_count: 0,
    chunk_count: 0
  });

  // Chat State
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [querying, setQuerying] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // Inspector Drawer State
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [activeSources, setActiveSources] = useState([]);
  const [selectedSource, setSelectedSource] = useState(null);

  const messagesEndRef = useRef(null);

  // Sync API Key / Base URL / Model when provider changes
  useEffect(() => {
    localStorage.setItem('qm_provider', provider);
    const key = localStorage.getItem(`qm_key_${provider}`) || '';
    const url = localStorage.getItem(`qm_url_${provider}`) || DEFAULT_URLS[provider] || '';
    const mdl = localStorage.getItem(`qm_model_${provider}`) || DEFAULT_MODELS[provider] || '';
    setApiKey(key);
    setBaseUrl(url);
    setModel(mdl);
  }, [provider]);

  // Save config values to localStorage
  const handleConfigChange = (type, val) => {
    if (type === 'key') {
      setApiKey(val);
      localStorage.setItem(`qm_key_${provider}`, val);
    } else if (type === 'url') {
      setBaseUrl(val);
      localStorage.setItem(`qm_url_${provider}`, val);
    } else if (type === 'model') {
      setModel(val);
      localStorage.setItem(`qm_model_${provider}`, val);
    }
  };

  // Check backend document status on mount
  useEffect(() => {
    fetch(`${BACKEND_URL}/api/status`)
      .then(res => res.json())
      .then(data => {
        if (data && data.filename) {
          setDocStatus(data);
          // Auto set mode from file type
          if (data.file_type === 'pdf') {
            setActiveMode('pdf');
          } else {
            setActiveMode('code');
          }
        }
      })
      .catch(err => console.error('Error fetching backend status:', err));
  }, []);

  // Auto-scroll chat to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, querying]);

  // Drag and Drop Handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      uploadFile(e.target.files[0]);
    }
  };

  // Upload File API Call
  const uploadFile = (file) => {
    setUploading(true);
    setErrorMsg(null);
    const formData = new FormData();
    formData.append('file', file);

    fetch(`${BACKEND_URL}/api/upload`, {
      method: 'POST',
      body: formData
    })
      .then(async (res) => {
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || 'Upload failed');
        }
        return data;
      })
      .then((data) => {
        setDocStatus(data);
        setMessages([
          {
            role: 'assistant',
            text: `Successfully indexed **${data.filename}** in **${data.file_type === 'pdf' ? 'DocMode' : 'CodeMode'}**.\n\nStats: **${data.page_count}** ${data.file_type === 'pdf' ? 'pages' : 'files'}, **${data.chunk_count}** chunks.\n\nGo ahead and ask me questions about it!`,
            sources: []
          }
        ]);
        setInspectorOpen(false);
        setSelectedSource(null);
      })
      .catch((err) => {
        setErrorMsg(err.message);
        console.error('Upload error:', err);
      })
      .finally(() => {
        setUploading(false);
      });
  };

  // Submit Query API Call
  const handleQuery = (e) => {
    e.preventDefault();
    if (!query.trim() || querying) return;

    const userMsg = { role: 'user', text: query };
    setMessages(prev => [...prev, userMsg]);
    setQuery('');
    setQuerying(true);
    setErrorMsg(null);

    fetch(`${BACKEND_URL}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: userMsg.text,
        provider,
        apiKey: provider !== 'ollama' ? apiKey : undefined,
        baseUrl: provider === 'ollama' ? baseUrl : undefined,
        model
      })
    })
      .then(async (res) => {
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || 'Failed to query model');
        }
        return data;
      })
      .then((data) => {
        setMessages(prev => [...prev, {
          role: 'assistant',
          text: data.answer,
          sources: data.sources || []
        }]);
      })
      .catch((err) => {
        setErrorMsg(err.message);
        console.error('Query error:', err);
      })
      .finally(() => {
        setQuerying(false);
      });
  };

  // Reset workspace
  const handleClear = () => {
    if (!window.confirm('Are you sure you want to reset the current workspace and clear the database?')) return;
    
    fetch(`${BACKEND_URL}/api/clear`, { method: 'POST' })
      .then(res => res.json())
      .then(() => {
        setDocStatus({
          filename: null,
          file_type: null,
          page_count: 0,
          chunk_count: 0
        });
        setMessages([]);
        setInspectorOpen(false);
        setSelectedSource(null);
        setErrorMsg(null);
      })
      .catch(err => {
        console.error('Error resetting:', err);
        setErrorMsg('Failed to clear database context.');
      });
  };

  const openSourceInspector = (sources) => {
    setActiveSources(sources);
    if (sources && sources.length > 0) {
      setSelectedSource(sources[0]);
      setInspectorOpen(true);
    }
  };

  return (
    <div className="app-container">
      {/* 1. Sidebar Config */}
      <aside className="sidebar">
        <div className="logo-container">
          <div className="logo-icon">QM</div>
          <span className="logo-text">QueryMind</span>
        </div>

        {/* Mode switcher tabs */}
        <div className="sidebar-section">
          <h3 className="section-title">Analysis Mode</h3>
          <div className="mode-tabs">
            <div 
              className={`mode-tab ${activeMode === 'pdf' ? 'active' : ''}`}
              onClick={() => { if (!docStatus.filename) setActiveMode('pdf'); }}
              style={{ cursor: docStatus.filename ? 'not-allowed' : 'pointer', opacity: docStatus.filename && activeMode !== 'pdf' ? 0.4 : 1 }}
              title={docStatus.filename ? "Reset workspace to switch mode" : ""}
            >
              DocMode (PDF)
            </div>
            <div 
              className={`mode-tab ${activeMode === 'code' ? 'active' : ''}`}
              onClick={() => { if (!docStatus.filename) setActiveMode('code'); }}
              style={{ cursor: docStatus.filename ? 'not-allowed' : 'pointer', opacity: docStatus.filename && activeMode !== 'code' ? 0.4 : 1 }}
              title={docStatus.filename ? "Reset workspace to switch mode" : ""}
            >
              CodeMode (ZIP)
            </div>
          </div>
        </div>

        {/* API Configurations */}
        <div className="sidebar-section">
          <h3 className="section-title">LLM Provider</h3>
          <div className="settings-card">
            <div className="form-group">
              <label>Provider</label>
              <select 
                className="form-select"
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
              >
                <option value="groq">Groq (Free)</option>
                <option value="ollama">Ollama (Local)</option>
                <option value="gemini">Gemini (Free Tier)</option>
                <option value="claude">Claude</option>
              </select>
            </div>

            {provider !== 'ollama' && (
              <div className="form-group">
                <label>API Key</label>
                <input 
                  type="password"
                  className="form-input"
                  placeholder="Paste key here..."
                  value={apiKey}
                  onChange={(e) => handleConfigChange('key', e.target.value)}
                />
              </div>
            )}

            {provider === 'ollama' && (
              <div className="form-group">
                <label>Base URL</label>
                <input 
                  type="text"
                  className="form-input"
                  placeholder="http://localhost:11434"
                  value={baseUrl}
                  onChange={(e) => handleConfigChange('url', e.target.value)}
                />
              </div>
            )}

            <div className="form-group">
              <label>Model</label>
              <input 
                type="text"
                className="form-input"
                placeholder={DEFAULT_MODELS[provider]}
                value={model}
                onChange={(e) => handleConfigChange('model', e.target.value)}
              />
            </div>
          </div>
        </div>

        {/* Reset Workspace Action */}
        {docStatus.filename && (
          <button className="btn btn-danger" onClick={handleClear}>
            <svg width="13" height="13" fill="currentColor" viewBox="0 0 16 16">
              <path d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6z"/>
              <path fillRule="evenodd" d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1v1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4H4.118zM2.5 3V2h11v1h-11z"/>
            </svg>
            Reset Workspace
          </button>
        )}
      </aside>

      {/* 2. Workspace Viewport */}
      <main className="workspace">
        {/* Workspace Header Status */}
        <header className="workspace-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <h2>Active Workspace</h2>
            {docStatus.filename ? (
              <div className="status-badge">
                <span className="dot"></span>
                {docStatus.filename}
              </div>
            ) : (
              <div className="status-badge empty">No document loaded</div>
            )}
          </div>
          {errorMsg && (
            <div className="error-banner">
              <svg width="13" height="13" fill="currentColor" viewBox="0 0 16 16">
                <path d="M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767L8.982 1.566zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5zm.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2z"/>
              </svg>
              {errorMsg}
            </div>
          )}
        </header>

        {/* Content Columns split */}
        <div className={`main-content ${docStatus.filename ? '' : 'full-chat'}`}>
          
          {/* UPLOAD PANEL: Shown when no doc is loaded OR on the left when doc is loaded */}
          {!docStatus.filename ? (
            <div className="upload-panel" style={{ gridColumn: 'span 2' }}>
              <div 
                className={`dropzone ${dragActive ? 'drag-active' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-upload-input').click()}
              >
                <input 
                  id="file-upload-input"
                  type="file"
                  style={{ display: 'none' }}
                  accept={activeMode === 'pdf' ? '.pdf' : '.zip,.py,.js,.jsx,.ts,.tsx,.go,.cpp,.cc,.h,.java,.html,.css,.rb,.rs'}
                  onChange={handleFileChange}
                />
                <div className="upload-icon">
                  {activeMode === 'pdf' ? '📁' : '⚙️'}
                </div>
                {uploading ? (
                  <div>
                    <h3 style={{ marginBottom: '0.5rem' }}>Indexing Knowledge...</h3>
                    <div style={{ display: 'flex', justifyContent: 'center', marginTop: '1rem' }}>
                      <div className="loader-spinner"></div>
                    </div>
                  </div>
                ) : (
                  <div>
                    <h3 style={{ marginBottom: '0.5rem', fontWeight: 600 }}>
                      Upload {activeMode === 'pdf' ? 'PDF Document' : 'Codebase (ZIP / File)'}
                    </h3>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                      Drag and drop your file here, or click to browse.
                    </p>
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* DUAL SPLIT CHAT PANEL */
            <div className="chat-panel" style={{ gridColumn: inspectorOpen ? 'span 1' : 'span 2' }}>
              {/* Messages list */}
              <div className="messages-list">
                {messages.length === 0 ? (
                  <div className="chat-welcome">
                    <div className="welcome-icon">💬</div>
                    <h3>Start Sourced QA</h3>
                    <p style={{ fontSize: '0.875rem' }}>
                      Ask questions about the uploaded file. QueryMind will retrieve matching content from the vector store and generate answers strictly sourced from it.
                    </p>
                  </div>
                ) : (
                  messages.map((msg, i) => (
                    <div key={i} className={`message-bubble ${msg.role}`}>
                      <div className="avatar">
                        {msg.role === 'user' ? 'U' : 'AI'}
                      </div>
                      <div className="message-content-wrapper">
                        <div className="message-text">{msg.text}</div>
                        {msg.sources && msg.sources.length > 0 && (
                          <div className="sources-container">
                            <span className="sources-label">Sources:</span>
                            {msg.sources.map((src, sIdx) => {
                              const label = src.page 
                                ? `p.${src.page}` 
                                : `${src.source.split('/').pop()}:${src.start_line}–${src.end_line}`;
                              return (
                                <button 
                                  key={sIdx} 
                                  className="source-pill"
                                  onClick={() => openSourceInspector(msg.sources)}
                                >
                                  {label}
                                </button>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                )}
                {querying && (
                  <div className="message-bubble assistant">
                    <div className="avatar">AI</div>
                    <div className="message-content-wrapper">
                      <div className="message-text">
                        <div className="chat-loading-dots">
                          <div className="loading-dot"></div>
                          <div className="loading-dot"></div>
                          <div className="loading-dot"></div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Form */}
              <div className="chat-input-container">
                <form className="chat-input-form" onSubmit={handleQuery}>
                  <input 
                    type="text"
                    className="chat-input"
                    placeholder="Ask a question about the document/code..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    disabled={querying}
                  />
                  <button type="submit" className="btn btn-primary" disabled={querying || !query.trim()}>
                    Send
                  </button>
                </form>
              </div>
            </div>
          )}

          {/* 3. Source Inspector Drawer */}
          {docStatus.filename && inspectorOpen && selectedSource && (
            <div className="inspector-panel">
              <div className="inspector-header">
                <h3 style={{ fontSize: '0.95rem', fontWeight: 600 }}>Source Context Inspector</h3>
                <button 
                  className="btn btn-secondary" 
                  style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                  onClick={() => setInspectorOpen(false)}
                >
                  Close
                </button>
              </div>
              <div className="inspector-body">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <label style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>Retrieved Matches</label>
                  <select 
                    className="form-select"
                    value={activeSources.indexOf(selectedSource)}
                    onChange={(e) => setSelectedSource(activeSources[parseInt(e.target.value)])}
                  >
                    {activeSources.map((src, sIdx) => {
                      const label = src.page ? `Match ${sIdx+1}: Page ${src.page}` : `Match ${sIdx+1}: ${src.source.split('/').pop()}:${src.start_line}-${src.end_line}`;
                      return (
                        <option key={sIdx} value={sIdx}>{label}</option>
                      );
                    })}
                  </select>
                </div>

                <div className="source-chunk-card">
                  <div className="chunk-header">
                    <span>File: {selectedSource.source}</span>
                    <span>{selectedSource.page ? `Page ${selectedSource.page}` : `Lines ${selectedSource.start_line}-${selectedSource.end_line}`}</span>
                  </div>
                  <pre className="chunk-code">
                    <code>{selectedSource.content}</code>
                  </pre>
                </div>
              </div>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}

export default App;
