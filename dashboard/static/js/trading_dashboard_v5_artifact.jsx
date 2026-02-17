// ==================== TRADING DASHBOARD V5 - ARTIFACT VERSION ====================
// Standalone React component with embedded styles
// Can be used in React projects or converted to HTML

import React, { useState, useEffect } from 'react';

const TradingDashboardV5Artifact = () => {
  const [expertMode, setExpertMode] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [activeStrategyFilter, setActiveStrategyFilter] = useState('all');
  const [activeEdgeTab, setActiveEdgeTab] = useState('weather');

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (date) => {
    return date.toLocaleTimeString('en-US', { hour12: false });
  };

  // Embedded CSS (converted to inline styles object for artifact)
  const styles = `
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif;
      background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #1e1b4b 100%);
      color: #e2e8f0;
      min-height: 100vh;
    }

    .dashboard-root {
      padding: 80px 20px 40px;
      background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #1e1b4b 100%);
      min-height: 100vh;
    }

    .training-banner {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
      color: #000;
      padding: 14px 20px;
      font-weight: 700;
      font-size: 14px;
      z-index: 99999;
      display: flex;
      justify-content: center;
      align-items: center;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .grid-container {
      max-width: 1920px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 20px;
    }

    .col-12 { grid-column: span 12; }
    .col-6 { grid-column: span 6; }
    .col-3 { grid-column: span 3; }

    @media (max-width: 1200px) {
      .col-3 { grid-column: span 6; }
      .col-6 { grid-column: span 12; }
    }

    @media (max-width: 768px) {
      .grid-container { grid-template-columns: 1fr; gap: 16px; }
      .col-3, .col-6, .col-12 { grid-column: span 1; }
    }

    .card {
      background: rgba(30, 41, 59, 0.4);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid rgba(148, 163, 184, 0.1);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
      overflow: hidden;
    }

    .card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    }

    .card:hover {
      transform: translateY(-2px);
      box-shadow: 0 30px 60px -12px rgba(0, 0, 0, 0.6);
      border-color: rgba(148, 163, 184, 0.2);
    }

    .card-ai {
      background: linear-gradient(135deg, rgba(49, 46, 129, 0.3) 0%, rgba(30, 41, 59, 0.4) 100%);
      border-color: rgba(139, 92, 246, 0.2);
    }

    .card-edge {
      background: linear-gradient(135deg, rgba(6, 78, 59, 0.2) 0%, rgba(30, 41, 59, 0.4) 100%);
      border-color: rgba(16, 185, 129, 0.2);
    }

    .kpi-card {
      cursor: default;
      position: relative;
      overflow: hidden;
    }

    .kpi-gradient {
      position: absolute;
      inset: 0;
      opacity: 0;
      transition: opacity 0.3s ease;
    }

    .kpi-card:hover .kpi-gradient {
      opacity: 1;
    }

    .kpi-label {
      font-size: 12px;
      color: #94a3b8;
      margin-bottom: 8px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .kpi-value {
      font-size: 42px;
      font-weight: 900;
      line-height: 1;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .kpi-subtitle {
      font-size: 11px;
      color: #64748b;
      margin-top: 8px;
    }

    .trend-arrow {
      font-size: 32px;
      opacity: 0.9;
    }

    .badge {
      display: inline-block;
      padding: 4px 12px;
      font-size: 11px;
      font-weight: 700;
      border-radius: 6px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      transition: transform 0.2s ease;
    }

    .badge:hover { transform: scale(1.05); }

    .badge-green {
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(16, 185, 129, 0.15));
      color: #4ade80;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-amber {
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.25), rgba(245, 158, 11, 0.15));
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .badge-blue {
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(59, 130, 246, 0.15));
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .badge-purple {
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(139, 92, 246, 0.15));
      color: #a78bfa;
      border: 1px solid rgba(139, 92, 246, 0.3);
    }

    .kill-btn {
      padding: 12px 24px;
      background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
      color: white;
      border: none;
      border-radius: 12px;
      font-weight: 700;
      font-size: 14px;
      cursor: pointer;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
      overflow: hidden;
    }

    .kill-btn:hover {
      transform: translateY(-3px) scale(1.02);
      box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
    }

    .kill-btn:active {
      transform: translateY(-1px) scale(0.98);
    }

    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }

    .header-left {
      display: flex;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }

    .header-item {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      font-weight: 600;
    }

    .divider {
      width: 1px;
      height: 24px;
      background: rgba(148, 163, 184, 0.2);
    }

    .status-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #4ade80;
    }

    .section-header {
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .sparkline {
      height: 40px;
      display: flex;
      align-items: flex-end;
      gap: 3px;
      margin-top: 12px;
    }

    .sparkline-bar {
      flex: 1;
      background: linear-gradient(to top, rgba(16, 185, 129, 0.3) 0%, rgba(16, 185, 129, 0.6) 50%, rgba(74, 222, 128, 0.9) 100%);
      border-radius: 4px 4px 0 0;
      transition: all 0.3s ease;
      cursor: pointer;
    }

    .sparkline-bar:hover {
      background: linear-gradient(to top, rgba(16, 185, 129, 0.5) 0%, rgba(16, 185, 129, 0.8) 50%, rgba(74, 222, 128, 1) 100%);
      transform: scaleY(1.1);
    }

    .strategy-card {
      margin-bottom: 16px;
      padding: 16px;
      background: rgba(15, 23, 42, 0.5);
      backdrop-filter: blur(10px);
      border-radius: 12px;
      border: 1px solid rgba(148, 163, 184, 0.05);
      transition: all 0.3s ease;
    }

    .strategy-card:hover {
      background: rgba(15, 23, 42, 0.7);
      transform: translateY(-2px);
    }

    .strategy-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }

    .strategy-name {
      font-weight: 700;
      font-size: 14px;
    }

    .strategy-metrics {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 16px;
    }

    .metric-label {
      font-size: 10px;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
    }

    .metric-value {
      font-size: 20px;
      font-weight: 800;
    }

    .btn-secondary {
      padding: 8px 16px;
      background: rgba(51, 65, 85, 0.8);
      color: #e2e8f0;
      border: none;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .btn-secondary:hover {
      background: rgba(51, 65, 85, 1);
      transform: translateY(-1px);
    }

    .table-responsive {
      overflow-x: auto;
      border-radius: 12px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    th {
      text-align: left;
      font-size: 10px;
      font-weight: 700;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      padding: 12px 8px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.1);
    }

    td {
      padding: 12px 8px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.05);
    }

    tbody tr:hover {
      background: rgba(148, 163, 184, 0.03);
    }

    .text-green { color: #4ade80; }
    .text-amber { color: #fbbf24; }
    .text-red { color: #f87171; }

    .mb-4 { margin-bottom: 16px; }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 20px;
    }

    @media (max-width: 1200px) {
      .kpi-grid { grid-template-columns: repeat(3, 1fr); }
    }

    @media (max-width: 768px) {
      .kpi-grid { grid-template-columns: repeat(2, 1fr); }
      .kpi-value { font-size: 32px; }
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.5); border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #8b5cf6 0%, #3b82f6 100%); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #a78bfa 0%, #60a5fa 100%); }
  `;

  return (
    <>
      <style>{styles}</style>

      <div className="dashboard-root">
        {/* Training Banner */}
        <div className="training-banner">
          🎓 TRAINING MODE — Hover over any element to see what it means
        </div>

        <div className="grid-container">

          {/* Header */}
          <div className="card col-12">
            <div className="header">
              <div className="header-left">
                <div className="header-item">
                  <span>📶</span>
                  <div className="status-dot" />
                  <span className="badge badge-green">ONLINE</span>
                </div>
                <div className="divider" />
                <div className="header-item">
                  <span>🕐</span>
                  <span>{formatTime(currentTime)}</span>
                </div>
                <div className="divider" />
                <span className="badge badge-blue">PAPER</span>
                <div className="divider" />
                <span className="badge badge-purple">V5.0</span>
              </div>
              <button className="kill-btn">⚡ KILL SWITCH</button>
            </div>
          </div>

          {/* KPI Row */}
          <div className="col-12">
            <div className="kpi-grid">

              <div className="card kpi-card">
                <div className="kpi-gradient" style={{background: 'linear-gradient(to br, rgba(16, 185, 129, 0.05), transparent)'}} />
                <div style={{position: 'relative', zIndex: 10}}>
                  <div className="kpi-label">💰 Total P&L</div>
                  <div className="kpi-value text-green">
                    <span className="trend-arrow">↑</span>
                    +$127.45
                  </div>
                </div>
              </div>

              <div className="card kpi-card">
                <div className="kpi-gradient" style={{background: 'linear-gradient(to br, rgba(245, 158, 11, 0.05), transparent)'}} />
                <div style={{position: 'relative', zIndex: 10}}>
                  <div className="kpi-label">🛡️ Daily Limit</div>
                  <div className="kpi-value">$11.25</div>
                  <div className="kpi-subtitle">of $22.50 (50%)</div>
                </div>
              </div>

              <div className="card kpi-card">
                <div className="kpi-gradient" style={{background: 'linear-gradient(to br, rgba(245, 158, 11, 0.05), transparent)'}} />
                <div style={{position: 'relative', zIndex: 10}}>
                  <div className="kpi-label">📉 Drawdown</div>
                  <div className="kpi-value text-amber">
                    <span className="trend-arrow">↓</span>
                    $23.50
                  </div>
                  <div className="kpi-subtitle">Max: $67.50 (15%)</div>
                </div>
              </div>

              <div className="card kpi-card">
                <div className="kpi-gradient" style={{background: 'linear-gradient(to br, rgba(16, 185, 129, 0.05), transparent)'}} />
                <div style={{position: 'relative', zIndex: 10}}>
                  <div className="kpi-label">📊 Win Rate</div>
                  <div className="kpi-value">67%</div>
                </div>
              </div>

              <div className="card kpi-card">
                <div className="kpi-gradient" style={{background: 'linear-gradient(to br, rgba(59, 130, 246, 0.05), transparent)'}} />
                <div style={{position: 'relative', zIndex: 10}}>
                  <div className="kpi-label">📈 Exposure</div>
                  <div className="kpi-value">45%</div>
                </div>
              </div>

            </div>
          </div>

          {/* Left Column - Strategies */}
          <div className="col-3">
            <div className="card">
              <div className="section-header">📊 Strategies (2/4)</div>

              <div className="strategy-card">
                <div className="strategy-header">
                  <span className="strategy-name">☀️ Weather Edge</span>
                  <span className="badge badge-green">LIVE</span>
                </div>
                <div className="strategy-metrics">
                  <div>
                    <div className="metric-label">P&L</div>
                    <div className="metric-value text-green">↑ +$52.30</div>
                  </div>
                  <div>
                    <div className="metric-label">Trades</div>
                    <div className="metric-value">24</div>
                  </div>
                </div>
                <div className="sparkline">
                  {[40, 60, 45, 80, 70, 90, 85].map((h, i) => (
                    <div key={i} className="sparkline-bar" style={{height: `${h}%`}} />
                  ))}
                </div>
              </div>

              <div className="strategy-card">
                <div className="strategy-header">
                  <span className="strategy-name">🏛️ Fed Watcher</span>
                  <span className="badge badge-green">LIVE</span>
                </div>
                <div className="strategy-metrics">
                  <div>
                    <div className="metric-label">P&L</div>
                    <div className="metric-value text-green">↑ +$45.15</div>
                  </div>
                  <div>
                    <div className="metric-label">Trades</div>
                    <div className="metric-value">12</div>
                  </div>
                </div>
                <div className="sparkline">
                  {[50, 55, 65, 70, 75].map((h, i) => (
                    <div key={i} className="sparkline-bar" style={{height: `${h}%`}} />
                  ))}
                </div>
              </div>

            </div>
          </div>

          {/* Center Column */}
          <div className="col-6">
            <div className="card mb-4">
              <div className="section-header">🎯 Positions (3)</div>
              <div className="table-responsive">
                <table>
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Broker</th>
                      <th>Side</th>
                      <th>Entry</th>
                      <th>Current</th>
                      <th>P&L</th>
                      <th>Age</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><strong>KXHIGHNY-26JAN31</strong></td>
                      <td><span className="badge badge-blue">Kalshi</span></td>
                      <td className="text-green">YES</td>
                      <td>$0.62</td>
                      <td>$0.71</td>
                      <td className="text-green" style={{fontWeight: 700}}>↑ +$9.00 (+14.5%)</td>
                      <td>2h 15m</td>
                    </tr>
                    <tr>
                      <td><strong>SPY</strong></td>
                      <td><span className="badge badge-green">Alpaca</span></td>
                      <td className="text-green">LONG</td>
                      <td>$485.20</td>
                      <td>$486.50</td>
                      <td className="text-green" style={{fontWeight: 700}}>↑ +$13.00 (+0.27%)</td>
                      <td>45m</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="col-3">
            <div className="card card-ai">
              <div className="section-header">🧠 AI Status</div>
              <div style={{display: 'flex', justifyContent: 'space-between', padding: '12px 0', borderBottom: '1px solid rgba(148,163,184,0.05)'}}>
                <span style={{fontSize: '13px', color: '#94a3b8'}}>Regime</span>
                <span className="badge badge-amber">SIDEWAYS</span>
              </div>
              <div style={{display: 'flex', justifyContent: 'space-between', padding: '12px 0'}}>
                <span style={{fontSize: '13px', color: '#94a3b8'}}>Sentiment</span>
                <span className="text-green" style={{fontWeight: 700}}>+0.35</span>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="col-12" style={{textAlign: 'center', padding: '32px 0', color: '#64748b', fontSize: '12px'}}>
            Trading Dashboard V5.0 • Enhanced Visual Design • All data is simulated
          </div>

        </div>
      </div>
    </>
  );
};

export default TradingDashboardV5Artifact;
