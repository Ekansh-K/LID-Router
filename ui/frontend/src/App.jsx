import { useState, useEffect } from 'react';
import axios from 'axios';
import AudioControl from './components/AudioControl';
import PipelineVisualizer from './components/PipelineVisualizer';
import { Layers, Moon, Sun, Dices } from 'lucide-react';
import './index.css';

function App() {
  const [pipelineData, setPipelineData] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState('dark');
  const [expectedText, setExpectedText] = useState(null);
  const [replayKey, setReplayKey] = useState(0);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const handleAudioSubmit = async (audioBlob, forcedExpectedText = null) => {
    setIsProcessing(true);
    setError(null);
    setPipelineData(null);
    setExpectedText(forcedExpectedText);

    const formData = new FormData();
    formData.append('file', audioBlob, 'audio.wav');

    try {
      const response = await axios.post('http://localhost:8000/process', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPipelineData(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || err.message || "An error occurred during processing.");
    } finally {
      setIsProcessing(false);
    }
  };

  const fetchRandomAudio = async () => {
    setIsProcessing(true);
    setError(null);
    setPipelineData(null);
    setExpectedText(null);

    try {
      const res = await axios.get('http://localhost:8000/random_audio', {
        responseType: 'blob'
      });

      const expectedRaw = res.headers['x-expected-text'];
      const strippedRaw = res.headers['x-stripped-text'];
      const refText = expectedRaw ? decodeURIComponent(expectedRaw) : null;
      const stripText = strippedRaw ? decodeURIComponent(strippedRaw) : null;

      const textInfo = refText ? { reference: refText, stripped: stripText } : null;

      // Submit the blob to process
      await handleAudioSubmit(res.data, textInfo);

    } catch (err) {
      console.error(err);
      // Show real backend error detail if available
      const detail = err?.response?.data?.detail
        || err?.response?.data
        || err?.message
        || 'Unknown error';
      const status = err?.response?.status ? ` (HTTP ${err.response.status})` : '';
      setError(`Random audio fetch failed${status}: ${typeof detail === 'string' ? detail.slice(0, 300) : JSON.stringify(detail)}`);
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="title-group">
          <h1>LID Router</h1>
          <p>End-to-End Multilingual Speech Routing Pipeline</p>
        </div>
        <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle Theme">
          {theme === 'dark' ? <Sun size={24} /> : <Moon size={24} />}
        </button>
      </header>

      <main className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'row', overflow: 'hidden' }}>

        {/* Sidebar Controls */}
        <div className="sidebar-controls" style={{
          width: '280px',
          borderRight: '1px solid var(--border-color)',
          padding: '2rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.5rem',
          flexShrink: 0,
          background: 'rgba(0,0,0,0.1)'
        }}>
          <button
            className="btn btn-accent"
            onClick={fetchRandomAudio}
            disabled={isProcessing}
            style={{ width: '100%' }}
          >
            <Dices size={20} /> Pick Random Audio
          </button>

          <AudioControl
            onSubmit={handleAudioSubmit}
            disabled={isProcessing}
          />
          
          <button 
            className="btn btn-secondary"
            onClick={() => setReplayKey(k => k + 1)}
            disabled={!pipelineData || isProcessing}
            style={{ width: '100%', marginTop: 'auto' }}
          >
            Replay Animation
          </button>
        </div>

        {/* Main Pipeline Area */}
        <div className="main-content no-scrollbar" style={{ flex: 1, overflowY: 'auto', padding: '0', position: 'relative' }}>

          {error && (
            <div style={{ color: '#ff3366', textAlign: 'center', padding: '1rem', background: 'rgba(255,51,102,0.1)', margin: '2rem', borderRadius: '8px' }}>
              <strong>Error:</strong> {error}
            </div>
          )}

          <div className="pipeline-container">
            {isProcessing && (
              <div className="processing-overlay">
                <div className="spinner"></div>
                <p style={{ fontWeight: 600 }}>Processing Audio through LID Router...</p>
              </div>
            )}

            <PipelineVisualizer
              data={pipelineData}
              expectedText={expectedText}
              replayKey={replayKey}
            />
          </div>

        </div>
      </main>
    </div>
  );
}

export default App;
