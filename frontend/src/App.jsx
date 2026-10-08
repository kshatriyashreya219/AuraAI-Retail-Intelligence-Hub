import { useEffect, useRef, useState } from 'react';
import {
  ArrowDownRight,
  ArrowRight,
  AudioLines,
  BadgeCheck,
  BarChart3,
  Camera,
  Check,
  ChevronDown,
  CircleHelp,
  ExternalLink,
  FileImage,
  ImagePlus,
  LoaderCircle,
  Mic,
  MicOff,
  Search,
  Shirt,
  Sparkles,
  Upload,
  WandSparkles,
  X,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const EXAMPLE_PROMPT = 'Generate a dashboard for wedding-season product performance';
const MAX_UPLOAD_BYTES = 12 * 1024 * 1024;

async function postJson(path, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || 'The request could not be completed.');
  return data;
}

async function postFile(path, field, file) {
  const body = new FormData();
  body.append(field, file);
  const response = await fetch(`${API_BASE}${path}`, { method: 'POST', body });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || 'The search could not be completed.');
  return data;
}

function ProductResults({ products, metricKey = 'match_score' }) {
  if (!products?.length) return <div className="empty-results">No close matches yet. Try another query.</div>;
  return (
    <div className="product-list">
      {products.map((item, index) => {
        const score = Number(item[metricKey] ?? item.similarity ?? 0);
        return (
          <article className="product-row" key={`${item.product}-${index}`}>
            <span className={`product-rank rank-${index + 1}`}>{String(index + 1).padStart(2, '0')}</span>
            <div className="product-meta">
              <strong>{item.product}</strong>
              <span>{item.category || 'Retail catalog'}{item.sales != null ? ` · $${Number(item.sales).toLocaleString()}` : ''}</span>
            </div>
            <div className="match-score">
              <span>{Math.round(score * 100)}%</span>
              <i><b style={{ width: `${Math.max(5, Math.min(100, score * 100))}%` }} /></i>
            </div>
          </article>
        );
      })}
    </div>
  );
}

function GeneratedDashboard({ data }) {
  const metrics = data.metrics || [];
  const chart = data.chart || { labels: [], values: [] };
  const peak = Math.max(1, ...chart.values.map(Number));
  return (
    <div className="generated-dashboard">
      <div className="generated-heading">
        <div>
          <span className="eyebrow"><span className="live-dot" /> GENERATED VIEW · ILLUSTRATIVE</span>
          <h2>{data.title}</h2>
          <p>{data.subtitle}</p>
        </div>
        <span className="generated-stamp"><Sparkles size={14} /> Built for your brief</span>
      </div>
      <div className="metric-grid">
        {metrics.map((metric, index) => (
          <article className={`metric-tile metric-${index}`} key={`${metric.label}-${index}`}>
            <span>{metric.label}</span>
            <strong>{metric.value}</strong>
            <small><ArrowDownRight size={13} /> {metric.change}</small>
          </article>
        ))}
      </div>
      <div className="chart-panel">
        <div className="chart-heading"><h3>{chart.title}</h3><span>Sample data</span></div>
        <div className="chart-bars" role="img" aria-label={chart.title}>
          {chart.labels.map((label, index) => (
            <div className="bar-column" key={`${label}-${index}`}>
              <span className="bar-value">{Number(chart.values[index] || 0).toLocaleString()}</span>
              <div className={`bar-fill bar-${index % 5}`} style={{ height: `${Math.max(5, (Number(chart.values[index] || 0) / peak) * 100)}%` }} />
              <span className="bar-label">{label}</span>
            </div>
          ))}
        </div>
      </div>
      {data.insights?.length > 0 && (
        <div className="insight-row">
          {data.insights.map((insight, index) => <p key={index}><BadgeCheck size={16} />{insight}</p>)}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState('generate');
  const [prompt, setPrompt] = useState('');
  const [designPrompt, setDesignPrompt] = useState('Blue kurta with golden border for haldi');
  const [designColor, setDesignColor] = useState('#315CBE');
  const [dashboard, setDashboard] = useState(null);
  const [designResult, setDesignResult] = useState(null);
  const [jsxSource, setJsxSource] = useState('');
  const [showSource, setShowSource] = useState(false);
  const [products, setProducts] = useState([]);
  const [transcript, setTranscript] = useState('');
  const [imagePreview, setImagePreview] = useState('');
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const fileRef = useRef(null);

  useEffect(() => () => {
    recorderRef.current?.stream?.getTracks().forEach((track) => track.stop());
    streamRef.current?.getTracks().forEach((track) => track.stop());
    if (imagePreview) URL.revokeObjectURL(imagePreview);
  }, [imagePreview]);

  async function generateDashboard(value = prompt) {
    if (!value.trim()) return;
    setLoading(true);
    setError('');
    setNotice('');
    try {
      const result = await postJson('/api/generative-ui', { prompt: value.trim() });
      setDashboard(result.dashboard);
      setJsxSource(result.jsx || '');
      setShowSource(false);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function generateDesigns() {
    if (!designPrompt.trim() || !designColor) return;
    setLoading(true);
    setError('');
    setNotice('');
    try {
      const result = await postJson('/api/generate-design', {
        prompt: designPrompt.trim(),
        color: designColor,
      });
      setDesignResult(result);
      setNotice('Two design concepts are ready.');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function submitVoice(blob) {
    if (blob.size > MAX_UPLOAD_BYTES) {
      setError('Audio uploads must be 12 MB or smaller. Try a shorter recording.');
      return;
    }
    setLoading(true);
    setError('');
    setNotice('');
    try {
      const audio = new File([blob], `aura-query.${blob.type.includes('ogg') ? 'ogg' : 'webm'}`, { type: blob.type || 'audio/webm' });
      const result = await postFile('/api/voice-search', 'audio', audio);
      setTranscript(result.transcript || '');
      setProducts(result.products || []);
      setNotice('Voice query transcribed and matched against the catalog.');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function startRecording() {
    setError('');
    setNotice('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const options = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? { mimeType: 'audio/webm;codecs=opus' } : undefined;
      const recorder = new MediaRecorder(stream, options);
      const chunks = [];
      recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
        if (blob.size) submitVoice(blob);
      };
      recorderRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch (err) {
      setError(err.name === 'NotAllowedError' ? 'Microphone access was blocked. Allow it in browser settings and try again.' : 'Microphone is unavailable in this browser.');
    }
  }

  function stopRecording() {
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
    setRecording(false);
  }

  async function findByImage(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Choose a JPG, PNG, or other image file.');
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setError('Image uploads must be 12 MB or smaller.');
      return;
    }
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImagePreview(URL.createObjectURL(file));
    setLoading(true);
    setError('');
    setNotice('');
    try {
      const result = await postFile('/api/image-search', 'image', file);
      setProducts(result.products || []);
      setNotice('Visual matches ranked with CLIP.');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const modes = [
    { id: 'generate', label: 'Make a dashboard', mobileLabel: 'Dashboard', Icon: WandSparkles, number: '01' },
    { id: 'voice', label: 'Speak to search', mobileLabel: 'Voice', Icon: AudioLines, number: '02' },
    { id: 'image', label: 'Find by photo', mobileLabel: 'Photo', Icon: Camera, number: '03' },
    { id: 'design', label: 'Design studio', mobileLabel: 'Design', Icon: Shirt, number: '04' },
  ];

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#home" aria-label="AURA OS home">
          <span className="brand-mark"><span>A</span></span>
          <span className="brand-name">AURA<span>OS</span></span>
        </a>
        <div className="topbar-center"><span className="status-pill"><span className="status-dot" /> DISCOVERY LAB</span><span className="topbar-divider" /><span className="topbar-caption">Retail, reimagined.</span></div>
        <button className="icon-button help-button" title="About Discovery Lab" onClick={() => setNotice('Generative dashboards use illustrative values. Voice and image matches use the uploaded query against the local product catalog.')}><CircleHelp size={18} /></button>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <div className="sidebar-kicker">YOUR TOOLKIT</div>
          <nav className="mode-nav" aria-label="Discovery tools">
            {modes.map(({ id, label, Icon, number }) => (
              <button className={`mode-link ${mode === id ? 'active' : ''}`} key={id} onClick={() => { setMode(id); setError(''); setNotice(''); }}>
                <span className="mode-icon"><Icon size={17} /></span><span className="mode-label">{label}</span><span className="mode-number">{number}</span>
              </button>
            ))}
          </nav>
          <div className="sidebar-bottom">
            <div className="catalog-card">
              <div className="catalog-icon"><BarChart3 size={17} /></div>
              <div><strong>Product catalog</strong><span>Superstore dataset</span></div>
              <span className="catalog-state" title="Local catalog configured"><Check size={13} /></span>
            </div>
            <div className="sidebar-foot"><span>BUILT FOR THE CURIOUS</span><span>v1.0</span></div>
          </div>
        </aside>

        <section className="main-content">
          <div className="page-heading">
            <div>
              <div className="breadcrumb"><span>Workspace</span><ChevronDown size={12} /><span>Discovery Lab</span></div>
              <h1>Ideas in.<br /><em>Insight out.</em></h1>
              <p>Make a dashboard, say what you need, or show us what you mean.</p>
            </div>
            <div className="heading-art" aria-hidden="true"><span className="orbit orbit-one" /><span className="orbit orbit-two" /><span className="orbit-core"><Sparkles size={26} /></span><span className="art-dot dot-one" /><span className="art-dot dot-two" /></div>
          </div>

          <div className="mobile-mode-select" aria-label="Discovery tools">
            {modes.map(({ id, label, mobileLabel, Icon }) => <button key={id} className={mode === id ? 'selected' : ''} onClick={() => { setMode(id); setError(''); setNotice(''); }}><Icon size={16} /><span>{mobileLabel || label}</span></button>)}
          </div>

          <div className="tool-panel">
            <div className="tool-panel-head">
              <div className="tool-label"><span className="tool-step">{modes.find((item) => item.id === mode)?.number}</span><span>{modes.find((item) => item.id === mode)?.label}</span></div>
              <span className="powered-label"><Sparkles size={13} /> AURA INTELLIGENCE</span>
            </div>

            {mode === 'generate' && (
              <div className="generate-tools">
                <label className="input-label" htmlFor="dashboard-prompt">What should your dashboard show?</label>
                <div className="prompt-box">
                  <textarea id="dashboard-prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) generateDashboard(); }} placeholder="Try “Generate Dashboard for Wedding”" rows={3} maxLength={500} />
                  <div className="prompt-footer"><span>{prompt.length}/500 <span className="key-hint">· Ctrl + Enter to run</span></span><button className="primary-button" onClick={() => generateDashboard()} disabled={loading || !prompt.trim()}>{loading ? <LoaderCircle className="spin" size={16} /> : <Sparkles size={16} />}<span>{loading ? 'Creating' : 'Generate'}</span><ArrowRight size={15} /></button></div>
                </div>
                <div className="suggestions"><span>QUICK START</span><button onClick={() => { setPrompt(EXAMPLE_PROMPT); generateDashboard(EXAMPLE_PROMPT); }}>Wedding sales snapshot <ArrowRight size={13} /></button><button onClick={() => { const text = 'Compare top product categories by monthly sales'; setPrompt(text); generateDashboard(text); }}>Category pulse <ArrowRight size={13} /></button><button onClick={() => { const text = 'Create a returns and profitability overview'; setPrompt(text); generateDashboard(text); }}>Returns overview <ArrowRight size={13} /></button></div>
              </div>
            )}

            {mode === 'design' && (
              <div className="design-tools">
                <label className="input-label" htmlFor="design-prompt">Describe the garment</label>
                <textarea
                  id="design-prompt"
                  className="design-prompt"
                  value={designPrompt}
                  onChange={(event) => setDesignPrompt(event.target.value)}
                  maxLength={500}
                  rows={3}
                />
                <div className="design-controls">
                  <label className="input-label" htmlFor="design-color">Lead color</label>
                  <div className="design-color-control">
                    <input
                      id="design-color"
                      type="color"
                      value={designColor}
                      onChange={(event) => setDesignColor(event.target.value)}
                      aria-label="Choose the lead color"
                    />
                    <span>{designColor.toUpperCase()}</span>
                  </div>
                  <div className="design-style-tags" aria-label="Applied style">
                    <span>INDIAN ETHNIC WEAR</span>
                    <span>STUDIO LIGHTING</span>
                    <span>4K</span>
                  </div>
                  <button className="primary-button design-submit" onClick={generateDesigns} disabled={loading || !designPrompt.trim()}>
                    {loading ? <LoaderCircle className="spin" size={16} /> : <Sparkles size={16} />}
                    <span>{loading ? 'Creating' : 'Generate 2 designs'}</span>
                    <ArrowRight size={15} />
                  </button>
                </div>
              </div>
            )}

            {mode === 'voice' && (
              <div className="voice-tools">
                <div className={`voice-orb ${recording ? 'recording' : ''}`}><AudioLines size={29} /></div>
                <p className="voice-title">Say it like you mean it.</p>
                <p className="voice-description">Ask for a product by name, color, or category. Your words become a catalog search.</p>
                <button className={`record-button ${recording ? 'is-recording' : ''}`} onClick={recording ? stopRecording : startRecording} disabled={loading}>
                  {recording ? <><span className="record-pulse" /><MicOff size={17} /> Stop recording</> : loading ? <><LoaderCircle className="spin" size={17} /> Searching</> : <><Mic size={17} /> Start voice search</>}
                </button>
                <span className="privacy-note">{recording ? 'Recording on this device · tap to finish' : 'Audio is sent to Whisper for transcription'}</span>
                {transcript && <div className="transcript"><span>YOU SAID</span><strong>“{transcript}”</strong></div>}
                {products.length > 0 && <div className="results-block"><div className="results-heading"><h3>Catalog matches</h3><span>{products.length} found</span></div><ProductResults products={products} /> </div>}
              </div>
            )}

            {mode === 'image' && (
              <div className="image-tools">
                <div className="image-intro"><span className="image-symbol"><ImagePlus size={20} /></span><div><p className="input-label">Show us the thing.</p><p>Drop in a product photo. CLIP finds the closest catalog matches.</p></div></div>
                <button className={`drop-zone ${dragOver ? 'drag-over' : ''} ${imagePreview ? 'has-preview' : ''}`} onClick={() => fileRef.current?.click()} onDragOver={(event) => { event.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)} onDrop={(event) => { event.preventDefault(); setDragOver(false); findByImage(event.dataTransfer.files?.[0]); }}>
                  {imagePreview ? <img src={imagePreview} alt="Selected product for visual search" /> : <><span className="upload-icon"><Upload size={20} /></span><strong>Drop a photo here, or browse</strong><span>PNG, JPG or WEBP · max 12 MB</span></>}
                  <input ref={fileRef} type="file" accept="image/*" hidden onChange={(event) => findByImage(event.target.files?.[0])} />
                  {imagePreview && <span className="replace-image"><FileImage size={14} /> Choose another image</span>}
                </button>
                <div className="image-hint"><span><Check size={14} /> Visual similarity</span><span><Check size={14} /> Product + category</span><span><Check size={14} /> Top 8 results</span></div>
                {loading && <div className="loading-line"><LoaderCircle className="spin" size={16} /> Comparing against the catalog…</div>}
                {products.length > 0 && <div className="results-block"><div className="results-heading"><h3>Closest lookalikes</h3><span>Ranked by CLIP similarity</span></div><ProductResults products={products} metricKey="similarity" /></div>}
              </div>
            )}
          </div>

          {error && <div className="feedback error-feedback" role="alert"><X size={16} /><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error"><X size={15} /></button></div>}
          {notice && <div className="feedback notice-feedback" role="status"><BadgeCheck size={16} /><span>{notice}</span><button onClick={() => setNotice('')} aria-label="Dismiss message"><X size={15} /></button></div>}

          {mode === 'design' && designResult && (
            <section className="design-results" aria-label="Generated design concepts">
              <div className="design-results-heading">
                <div>
                  <span className="result-kicker">DESIGN CONCEPTS</span>
                  <h2>{designPrompt}</h2>
                </div>
                <span className="design-estimate">Est. ${Number(designResult.estimated_price).toFixed(2)} USD</span>
              </div>
              <div className="design-grid">
                {[designResult.image_url_1, designResult.image_url_2].map((imageUrl, index) => (
                  <article className="design-card" key={`${imageUrl}-${index}`}>
                    <a href={imageUrl} target="_blank" rel="noreferrer" aria-label={`Open design concept ${index + 1}`}>
                      <img src={imageUrl} alt={`${designPrompt}, design concept ${index + 1}`} loading="lazy" />
                    </a>
                    <div className="design-card-footer">
                      <span>CONCEPT {String(index + 1).padStart(2, '0')}</span>
                      <a href={imageUrl} target="_blank" rel="noreferrer" aria-label={`Open full-size concept ${index + 1}`}><ExternalLink size={15} /></a>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          )}

          {mode === 'generate' && dashboard && (
            <div className="result-area">
              <div className="result-bar"><div><span className="result-kicker">YOUR GENERATED DASHBOARD</span><span className="result-dot" /></div><button className="text-button" onClick={() => setShowSource((value) => !value)}>{showSource ? 'Hide JSX' : 'View JSX'}<ArrowRight size={14} /></button></div>
              <GeneratedDashboard data={dashboard} />
              {showSource && <pre className="code-output"><code>{jsxSource}</code></pre>}
            </div>
          )}
        </section>

        <aside className="right-rail">
          <div className="rail-topline"><span>AT A GLANCE</span><span className="rail-spark"><Sparkles size={14} /></span></div>
          <div className="rail-section"><span className="rail-number">A</span><div><h3>Build a view</h3><p>Give your idea a shape. Get a dashboard spec back in seconds.</p></div></div>
          <div className="rail-section"><span className="rail-number rail-number-coral">B</span><div><h3>Find by voice</h3><p>Talk naturally. Whisper transcribes, catalog search does the rest.</p></div></div>
          <div className="rail-section"><span className="rail-number rail-number-blue">C</span><div><h3>Search visually</h3><p>One photo, eight lookalikes. CLIP understands what it sees.</p></div></div>
          <div className="rail-section"><span className="rail-number">D</span><div><h3>Design a look</h3><p>Shape an ethnic-wear concept around a prompt and color.</p></div></div>
          <div className="rail-divider" />
          <div className="rail-tip"><div className="tip-icon"><Search size={16} /></div><div><span>MAKE IT SPECIFIC</span><p>“Show festive wear demand by category” gives your dashboard more to work with.</p></div></div>
          <div className="rail-footer"><span>YOUR RETAIL, YOUR RULES.</span><ArrowDownRight size={15} /></div>
        </aside>
      </div>
    </main>
  );
}
