import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from 'recharts';
import { Shield, Activity, FileText, Zap, Upload, Terminal, Play, Pause, Server, AlertTriangle, TrendingUp, Clock, Cpu } from 'lucide-react';
import './App.css';

const API_URL = "http://localhost:8000/api";

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isRunning, setIsRunning] = useState(true);
  const [alertPulse, setAlertPulse] = useState(false);
  const fileInputRef = useRef(null);
  const prevAnomalyRef = useRef(false);

  // Poll Backend
  useEffect(() => {
    const interval = setInterval(() => {
      if (!isRunning) return;
      fetchData();
    }, 1000);
    
    fetchData();
    
    return () => clearInterval(interval);
  }, [isRunning]);

  // Trigger alert animation when anomaly state changes
  useEffect(() => {
    if (data && data.is_anomaly && !prevAnomalyRef.current) {
      setAlertPulse(true);
      setTimeout(() => setAlertPulse(false), 2000);
    }
    if (data) {
      prevAnomalyRef.current = data.is_anomaly;
    }
  }, [data?.is_anomaly]);

  const fetchData = async () => {
    try {
      const res = await axios.get(`${API_URL}/live-data`);
      setData(res.data);
      setLoading(false);
    } catch (error) {
      console.error("Connection lost", error);
    }
  };

  const toggleMonitor = async () => {
    const newState = !isRunning;
    setIsRunning(newState);
  };

  const handleInjectAttack = async () => {
    await axios.post(`${API_URL}/control/inject-attack`);
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await axios.post(`${API_URL}/ingest/upload`, formData);
      alert(`✅ Upload Complete: ${res.data.lines_processed} lines processed.`);
    } catch (error) {
      alert("Upload failed.");
    }
  };

  if (loading) return (
    <div className="loading-screen">
      <div className="loading-content">
        <div className="loading-spinner"></div>
        <div className="loading-text">INITIALIZING SENTINEL</div>
        <div className="loading-bar">
          <div className="loading-progress"></div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="app-container">
      {/* Background effects */}
      <div className="bg-grid"></div>
      <div className="bg-gradient"></div>
      
      {/* === SIDEBAR === */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-title">
            <Activity className="sidebar-icon" />
            <div>
              <h2>Control Panel</h2>
              <span className="sidebar-subtitle">System Controls</span>
            </div>
          </div>
        </div>

        {/* Monitor Toggle */}
        <div className="control-section">
          <div className="control-card">
            <div className="control-header">
              <span className="control-label">Monitor Status</span>
              <div className={`status-indicator ${isRunning ? 'active' : 'inactive'}`}>
                <span className="status-dot"></span>
                {isRunning ? 'ACTIVE' : 'PAUSED'}
              </div>
            </div>
            <button 
              onClick={toggleMonitor}
              className={`toggle-button ${isRunning ? 'running' : 'paused'}`}
            >
              {isRunning ? <Pause size={18} /> : <Play size={18} />}
              <span>{isRunning ? 'Pause Monitor' : 'Start Monitor'}</span>
            </button>
          </div>
        </div>

        {/* Attack Injection */}
        <div className="control-section">
          <div className="section-label">SIMULATION CONTROLS</div>
          <button 
            onClick={handleInjectAttack}
            className="attack-button"
          >
            <Zap className="attack-icon" />
            <span>Inject DDoS Attack</span>
          </button>
          <div className="help-box">
            <div className="help-icon">💡</div>
            <div className="help-text">
              Simulate a distributed denial-of-service attack to test anomaly detection
            </div>
          </div>
        </div>

        {/* File Upload */}
        <div className="control-section upload-section">
          <div className="section-label">LOG INGESTION</div>
          <div className="upload-area" onClick={() => fileInputRef.current?.click()}>
            <Upload className="upload-icon" />
            <div className="upload-text">
              <div className="upload-title">Upload Log Files</div>
              <div className="upload-subtitle">Click or drag files here</div>
            </div>
            <input 
              type="file" 
              ref={fileInputRef}
              onChange={handleFileUpload}
              className="file-input"
            />
          </div>
        </div>

        {/* System Info */}
        <div className="sidebar-footer">
          <div className="footer-stat">
            <Clock size={14} />
            <span>{new Date().toLocaleTimeString()}</span>
          </div>
          <div className="footer-stat">
            <Cpu size={14} />
            <span>SENTINEL v2.1</span>
          </div>
        </div>
      </aside>

      {/* === MAIN CONTENT === */}
      <main className="main-content">
        {/* Header */}
        <header className="header">
          <div className="header-content">
            <div className="header-title-group">
              <Shield className="header-shield" />
              <div>
                <h1 className="header-title">AIOps Sentinel</h1>
                <p className="header-subtitle">Real-time Infrastructure Anomaly Detection</p>
              </div>
            </div>
            <div className="header-badge">
              <span className="badge-dot"></span>
              <span>Isolation Forest ML</span>
            </div>
          </div>
        </header>

        {/* Critical Alert Banner */}
        {data.is_anomaly && (
          <div className={`alert-banner ${alertPulse ? 'pulse' : ''}`}>
            <div className="alert-content">
              <AlertTriangle className="alert-icon" />
              <div className="alert-text">
                <div className="alert-title">ANOMALY DETECTED</div>
                <div className="alert-message">
                  Abnormal traffic pattern identified • Isolation Forest score: {data.anomaly_score.toFixed(4)} • Immediate investigation required
                </div>
              </div>
            </div>
            <div className="alert-line"></div>
          </div>
        )}

        {/* KPI Grid */}
        <div className="kpi-grid">
          <KpiCard 
            title="Server Status" 
            value={data.server_status} 
            icon={<Server size={24}/>} 
            trend="+2.3%"
            colorClass="blue"
          />
          <KpiCard 
            title="AI Diagnosis" 
            value={data.ai_diagnosis} 
            icon={<Activity size={24}/>} 
            trend={data.is_anomaly ? "CRITICAL" : "NORMAL"}
            colorClass={data.is_anomaly ? "red" : "green"}
            pulsate={data.is_anomaly}
          />
          <KpiCard 
            title="Anomaly Score" 
            value={data.anomaly_score.toFixed(4)} 
            icon={<TrendingUp size={24}/>}
            trend="Real-time"
            colorClass="yellow"
          />
          <KpiCard 
            title="Log Volume" 
            value={data.log_volume} 
            icon={<FileText size={24}/>}
            trend="Events/sec"
            colorClass="purple"
          />
        </div>

        {/* Charts & Logs Layout */}
        <div className="dashboard-grid">
          
          {/* Main Chart */}
          <div className="chart-container">
            <div className="chart-header">
              <div className="chart-title">
                <Activity size={20} />
                <span>Live Anomaly Score Stream</span>
              </div>
              <div className="chart-legend">
                <div className={`legend-item ${data.is_anomaly ? 'red' : 'blue'}`}>
                  <span className="legend-dot"></span>
                  <span>Anomaly Score</span>
                </div>
              </div>
            </div>
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.history} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={data.is_anomaly ? "#ff3366" : "#00d4ff"} stopOpacity={0.6}/>
                      <stop offset="95%" stopColor={data.is_anomaly ? "#ff3366" : "#00d4ff"} stopOpacity={0}/>
                    </linearGradient>
                    <filter id="glow">
                      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                      <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                      </feMerge>
                    </filter>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="time" hide />
                  <YAxis domain={[-0.8, 0.8]} stroke="rgba(255,255,255,0.2)" tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11 }} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'rgba(15, 23, 42, 0.95)', 
                      border: '1px solid rgba(255,255,255,0.1)', 
                      borderRadius: '8px',
                      backdropFilter: 'blur(10px)',
                      boxShadow: '0 8px 32px rgba(0,0,0,0.3)'
                    }}
                    labelStyle={{ color: '#94a3b8' }}
                  />
                  <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" strokeDasharray="3 3" />
                  <Area 
                    type="monotone" 
                    dataKey="score" 
                    stroke={data.is_anomaly ? "#ff3366" : "#00d4ff"} 
                    strokeWidth={3}
                    fill="url(#colorScore)" 
                    isAnimationActive={false}
                    filter="url(#glow)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-footer">
              <div className="chart-stat">
                <span className="stat-label">Peak Score</span>
                <span className="stat-value">{Math.max(...data.history.map(h => h.score)).toFixed(3)}</span>
              </div>
              <div className="chart-stat">
                <span className="stat-label">Samples</span>
                <span className="stat-value">{data.history.length}</span>
              </div>
              <div className="chart-stat">
                <span className="stat-label">Threshold</span>
                <span className="stat-value">±0.5</span>
              </div>
            </div>
          </div>

          {/* Terminal Logs */}
          <div className="terminal-container">
            <div className="terminal-header">
              <div className="terminal-title">
                <Terminal size={16} />
                <span>SYSTEM LOGS</span>
              </div>
              <div className="terminal-controls">
                <div className="terminal-dot red"></div>
                <div className="terminal-dot yellow"></div>
                <div className="terminal-dot green"></div>
              </div>
            </div>
            <div className="terminal-content">
              {data.logs.slice().reverse().map((log, idx) => (
                <div key={idx} className={`log-line ${log.includes("ERROR") || log.includes("Attack") ? "error" : ""}`}>
                  <span className="log-time">[{new Date().toLocaleTimeString()}]</span>
                  <span className="log-message">{log}</span>
                </div>
              ))}
            </div>
            <div className="terminal-footer">
              <span className="terminal-cursor">▊</span>
              <span className="terminal-prompt">Monitoring {data.logs.length} events...</span>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}

// Enhanced KPI Component
const KpiCard = ({ title, value, icon, trend, colorClass, pulsate }) => (
  <div className={`kpi-card ${colorClass} ${pulsate ? 'pulsate' : ''}`}>
    <div className="kpi-header">
      <div className="kpi-icon-wrapper">
        {icon}
      </div>
      <div className="kpi-trend">{trend}</div>
    </div>
    <div className="kpi-body">
      <div className="kpi-title">{title}</div>
      <div className="kpi-value">{value}</div>
    </div>
    <div className="kpi-shine"></div>
  </div>
);

export default App;
