import React, { useState, useEffect } from 'react';
import './dashboard_v5.css';

// ==================== KPI CARD COMPONENT ====================
const KPICard = ({ label, value, trend, subtitle, gradientType = 'neutral' }) => {
  const gradients = {
    profit: 'from-green-500/5 to-transparent',
    loss: 'from-red-500/5 to-transparent',
    warning: 'from-amber-500/5 to-transparent',
    info: 'from-blue-500/5 to-transparent',
    neutral: 'from-slate-500/5 to-transparent'
  };

  const trendColors = {
    up: 'text-green-400',
    down: 'text-red-400',
    neutral: 'text-slate-400'
  };

  const trendArrows = {
    up: '↑',
    down: '↓',
    neutral: '→'
  };

  return (
    <div className={`card-v5 kpi-card relative overflow-hidden group`}>
      {/* Gradient overlay */}
      <div className={`absolute inset-0 bg-gradient-to-br ${gradients[gradientType]} opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />

      <div className="relative z-10">
        <div className="kpi-label">{label}</div>
        <div className={`kpi-value ${trend ? trendColors[trend] : ''} flex items-center gap-2`}>
          {trend && <span className="trend-arrow text-3xl">{trendArrows[trend]}</span>}
          <span>{value}</span>
        </div>
        {subtitle && <div className="kpi-subtitle">{subtitle}</div>}
      </div>
    </div>
  );
};

// ==================== SPARKLINE COMPONENT ====================
const Sparkline = ({ data = [40, 60, 45, 80, 70, 90, 85] }) => {
  return (
    <div className="sparkline-v5">
      {data.map((height, idx) => (
        <div
          key={idx}
          className="sparkline-bar-v5"
          style={{ height: `${height}%` }}
        />
      ))}
    </div>
  );
};

// ==================== STATUS BADGE COMPONENT ====================
const StatusBadge = ({ status, label }) => {
  const statusStyles = {
    online: 'badge-green',
    warning: 'badge-amber',
    error: 'badge-red',
    info: 'badge-blue',
    active: 'badge-purple'
  };

  return (
    <span className={`badge-v5 ${statusStyles[status] || 'badge-blue'}`}>
      {label}
    </span>
  );
};

// ==================== MAIN DASHBOARD COMPONENT ====================
const TradingDashboardV5 = () => {
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

  return (
    <div className="dashboard-v5-container">

      {/* Training Banner */}
      <div className="training-banner">
        🎓 TRAINING MODE — Hover over any element to see what it means
      </div>

      <div className="container-v5">

        {/* ==================== HEADER ==================== */}
        <div className="card-v5 col-span-12">
          <div className="header-v5">
            <div className="header-left">

              {/* System Status */}
              <div className="header-item">
                <span>📶</span>
                <span>System</span>
                <div className="status-indicator">
                  <div className="status-dot status-dot-green" />
                  <StatusBadge status="online" label="ONLINE" />
                </div>
              </div>

              <div className="divider" />

              {/* Clock */}
              <div className="header-item">
                <span>🕐</span>
                <span className="font-bold">{formatTime(currentTime)}</span>
              </div>

              <div className="divider" />

              {/* Mode */}
              <StatusBadge status="info" label="PAPER" />

              <div className="divider" />

              {/* Version */}
              <StatusBadge status="active" label="V5.0" />

              <div className="divider" />

              {/* Expert Mode Toggle */}
              <button
                className={`expert-toggle ${expertMode ? 'active' : ''}`}
                onClick={() => setExpertMode(!expertMode)}
              >
                🔧 Expert Mode {expertMode ? 'ON' : ''}
              </button>

            </div>

            {/* Kill Switch */}
            <button className="kill-btn-v5">
              ⚡ KILL SWITCH
            </button>
          </div>
        </div>

        {/* ==================== KPI ROW ==================== */}
        <div className="col-span-12">
          <div className="kpi-grid-v5">

            <KPICard
              label="💰 Total P&L"
              value="+$127.45"
              trend="up"
              gradientType="profit"
            />

            <KPICard
              label="🛡️ Daily Limit"
              value="$11.25"
              subtitle="of $22.50 (50%)"
              gradientType="warning"
            />

            <KPICard
              label="📉 Drawdown"
              value="↓ $23.50"
              subtitle="Max: $67.50 (15%)"
              gradientType="warning"
            />

            <KPICard
              label="📊 Win Rate"
              value="67%"
              gradientType="profit"
            />

            <KPICard
              label="📈 Exposure"
              value="45%"
              gradientType="info"
            />

          </div>
        </div>

        {/* ==================== LEFT COLUMN: STRATEGIES ==================== */}
        <div className="col-span-12 lg:col-span-3">
          <div className="card-v5">

            <div className="section-header-v5">
              📊 Strategies (<span>2</span>/4)
            </div>

            {/* Filters */}
            <div className="filters-v5">
              <button className={`filter-btn-v5 ${activeStrategyFilter === 'all' ? 'active' : ''}`} onClick={() => setActiveStrategyFilter('all')}>All</button>
              <button className={`filter-btn-v5 ${activeStrategyFilter === 'live' ? 'active' : ''}`} onClick={() => setActiveStrategyFilter('live')}>Live</button>
              <button className={`filter-btn-v5 ${activeStrategyFilter === 'paused' ? 'active' : ''}`} onClick={() => setActiveStrategyFilter('paused')}>Paused</button>
              <select className="sort-select-v5">
                <option>Sort: P&L</option>
                <option>Win Rate</option>
                <option>Drawdown</option>
              </select>
            </div>

            {/* Strategy Cards */}
            <div className="strategy-card-v5">
              <div className="strategy-header">
                <span className="strategy-name">☀️ Weather Edge</span>
                <StatusBadge status="online" label="LIVE" />
              </div>
              <div className="strategy-metrics">
                <div>
                  <div className="metric-label">P&L</div>
                  <div className="metric-value text-green-400">↑ +$52.30</div>
                </div>
                <div>
                  <div className="metric-label">Trades</div>
                  <div className="metric-value">24</div>
                </div>
              </div>
              <Sparkline data={[40, 60, 45, 80, 70, 90, 85]} />
            </div>

            <div className="strategy-card-v5">
              <div className="strategy-header">
                <span className="strategy-name">🏛️ Fed Watcher</span>
                <StatusBadge status="online" label="LIVE" />
              </div>
              <div className="strategy-metrics">
                <div>
                  <div className="metric-label">P&L</div>
                  <div className="metric-value text-green-400">↑ +$45.15</div>
                </div>
                <div>
                  <div className="metric-label">Trades</div>
                  <div className="metric-value">12</div>
                </div>
              </div>
              <Sparkline data={[50, 55, 65, 70, 75]} />
            </div>

            <div className="strategy-card-v5">
              <div className="strategy-header">
                <span className="strategy-name">📈 Momentum</span>
                <StatusBadge status="warning" label="PAUSED" />
              </div>
              <div className="strategy-metrics">
                <div>
                  <div className="metric-label">P&L</div>
                  <div className="metric-value text-amber-400">+$12.00</div>
                </div>
                <div>
                  <div className="metric-label">Trades</div>
                  <div className="metric-value">8</div>
                </div>
              </div>
            </div>

          </div>
        </div>

        {/* ==================== CENTER COLUMN ==================== */}
        <div className="col-span-12 lg:col-span-6">

          {/* Equity Curve */}
          <div className="card-v5 mb-4">
            <div className="section-header-v5 justify-between">
              <span>📈 Equity Curve</span>
              <div className="flex gap-3 text-xs">
                <label className="flex items-center gap-1 cursor-pointer">
                  <input type="checkbox" defaultChecked className="accent-blue-500" /> Trades
                </label>
                <label className="flex items-center gap-1 cursor-pointer">
                  <input type="checkbox" defaultChecked className="accent-blue-500" /> Regime
                </label>
              </div>
            </div>

            <div className="chart-controls-v5">
              <button className="chart-btn-v5 active">1W</button>
              <button className="chart-btn-v5">1M</button>
              <button className="chart-btn-v5">3M</button>
              <button className="chart-btn-v5">ALL</button>
            </div>

            <div className="chart-container-v5">
              [Equity Chart - Shows account value over time]
            </div>
          </div>

          {/* Positions Table */}
          <div className="card-v5 mb-4">
            <div className="section-header-v5 justify-between">
              <span>🎯 Positions (<span>3</span>)</span>
              <button className="btn-danger-v5">Close All</button>
            </div>

            <div className="table-container-v5">
              <table className="table-v5">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Broker</th>
                    <th>Strategy</th>
                    <th>Side</th>
                    <th>Entry</th>
                    <th>Current</th>
                    <th>P&L</th>
                    <th>Age</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>KXHIGHNY-26JAN31</strong></td>
                    <td><StatusBadge status="info" label="Kalshi" /></td>
                    <td>Weather</td>
                    <td className="text-green-400">YES</td>
                    <td>$0.62</td>
                    <td>$0.71</td>
                    <td className="text-green-400 font-bold">↑ +$9.00 (+14.5%)</td>
                    <td>2h 15m</td>
                    <td><button className="btn-secondary-v5">Close</button></td>
                  </tr>
                  <tr>
                    <td><strong>SPY</strong></td>
                    <td><StatusBadge status="online" label="Alpaca" /></td>
                    <td>Momentum</td>
                    <td className="text-green-400">LONG</td>
                    <td>$485.20</td>
                    <td>$486.50</td>
                    <td className="text-green-400 font-bold">↑ +$13.00 (+0.27%)</td>
                    <td>45m</td>
                    <td><button className="btn-secondary-v5">Close</button></td>
                  </tr>
                  <tr>
                    <td><strong>FED-26FEB-HOLD</strong></td>
                    <td><StatusBadge status="info" label="Kalshi" /></td>
                    <td>Fed</td>
                    <td className="text-green-400">YES</td>
                    <td>$0.78</td>
                    <td>$0.82</td>
                    <td className="text-green-400 font-bold">↑ +$4.00 (+5.1%)</td>
                    <td>1d 4h</td>
                    <td><button className="btn-secondary-v5">Close</button></td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="positions-footer-v5">
              <div>Invested: <span className="font-bold">$202.50</span></div>
              <div>Available: <span className="font-bold text-green-400">$247.50</span></div>
              <div>Exposure: <span className="font-bold">45%</span></div>
            </div>
          </div>

          {/* Recent Trades */}
          <div className="card-v5">
            <div className="section-header-v5">
              📝 Recent Trades
            </div>

            <div className="table-container-v5">
              <table className="table-v5">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Signal</th>
                    <th>Filled</th>
                    <th>Slip</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>14:28</td>
                    <td><strong>KXHIGHNY-26JAN31</strong></td>
                    <td className="text-green-400">BUY YES</td>
                    <td>$0.62</td>
                    <td>$0.62</td>
                    <td className="text-green-400">$0.00</td>
                  </tr>
                  <tr>
                    <td>13:45</td>
                    <td><strong>SPY</strong></td>
                    <td className="text-green-400">BUY</td>
                    <td>$485.15</td>
                    <td>$485.20</td>
                    <td className="text-amber-400">-$0.05</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

        </div>

        {/* ==================== RIGHT COLUMN ==================== */}
        <div className="col-span-12 lg:col-span-3">

          {/* Broker Health */}
          <div className="card-v5 mb-4">
            <div className="section-header-v5">
              🏦 Broker Health
            </div>

            <div className="broker-row-v5">
              <div className="broker-info">
                <div className="status-dot status-dot-green" />
                <span className="broker-name">Alpaca</span>
              </div>
              <div className="broker-stats">
                <div className="broker-stat">
                  <span className="broker-stat-label">Balance</span>
                  <span className="broker-stat-value">$247.50</span>
                </div>
                <div className="broker-stat">
                  <span className="broker-stat-label">Latency</span>
                  <span className="broker-stat-value text-green-400">45ms</span>
                </div>
              </div>
            </div>

            <div className="broker-row-v5">
              <div className="broker-info">
                <div className="status-dot status-dot-green" />
                <span className="broker-name">Kalshi</span>
              </div>
              <div className="broker-stats">
                <div className="broker-stat">
                  <span className="broker-stat-label">Balance</span>
                  <span className="broker-stat-value">$185.00</span>
                </div>
                <div className="broker-stat">
                  <span className="broker-stat-label">Latency</span>
                  <span className="broker-stat-value text-green-400">62ms</span>
                </div>
              </div>
            </div>
          </div>

          {/* Active Edges */}
          <div className="card-v5 card-edge mb-4">
            <div className="section-header-v5">
              🎯 Active Edges
            </div>

            <div className="edge-tabs-v5">
              <button className={`edge-tab-v5 ${activeEdgeTab === 'weather' ? 'active' : ''}`} onClick={() => setActiveEdgeTab('weather')}>☀️ Weather</button>
              <button className={`edge-tab-v5 ${activeEdgeTab === 'fed' ? 'active' : ''}`} onClick={() => setActiveEdgeTab('fed')}>🏛️ Fed</button>
              <button className={`edge-tab-v5 ${activeEdgeTab === 'sports' ? 'active' : ''}`} onClick={() => setActiveEdgeTab('sports')}>⚽ Sports</button>
            </div>

            <div className="edge-item-v5">
              <div className="edge-header">
                <span className="edge-title">NYC High Temp 1/31</span>
                <StatusBadge status="online" label="TRADE" />
              </div>
              <div className="edge-value text-green-400">+9.2%</div>
              <div className="edge-details">
                Our: 72% | Market: 63% | Conf: HIGH
              </div>
            </div>

            <div className="edge-item-v5">
              <div className="edge-header">
                <span className="edge-title">Chicago Low Temp 2/1</span>
                <StatusBadge status="warning" label="MONITOR" />
              </div>
              <div className="edge-value text-amber-400">+4.1%</div>
              <div className="edge-details">
                Our: 58% | Market: 54% | Conf: MED
              </div>
            </div>
          </div>

          {/* AI Status */}
          <div className="card-v5 card-ai mb-4">
            <div className="section-header-v5">
              🧠 AI Status
            </div>

            <div className="info-row-v5">
              <span className="info-label">Regime</span>
              <StatusBadge status="warning" label="SIDEWAYS" />
            </div>

            <div className="info-row-v5">
              <span className="info-label">Sentiment</span>
              <span className="info-value text-green-400">+0.35</span>
            </div>

            <div className="info-row-v5">
              <span className="info-label">Veto Rate</span>
              <span className="info-value">12%</span>
            </div>

            <div className="info-row-v5">
              <span className="info-label">AI Cost</span>
              <span className="info-value">$0.42</span>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="card-v5 mb-4">
            <div className="section-header-v5">
              ⚡ Actions
            </div>

            <div className="actions-grid-v5">
              <button className="btn-warning-v5">⏸️ Pause All</button>
              <button className="btn-secondary-v5">▶️ Resume All</button>
            </div>
          </div>

          {/* Alerts */}
          <div className="card-v5">
            <div className="section-header-v5">
              👁️ Alerts
            </div>

            <div className="filters-v5 mb-2">
              <button className="filter-btn-v5 active">All</button>
              <button className="filter-btn-v5">🔴</button>
              <button className="filter-btn-v5">🟡</button>
              <button className="filter-btn-v5">🔵</button>
            </div>

            <div className="alert-v5 alert-info">
              <div className="alert-message">Weather Edge opened position KXHIGHNY-26JAN31 @ $0.62</div>
              <div className="alert-time">2 min ago</div>
            </div>

            <div className="alert-v5 alert-warning">
              <div className="alert-message">Daily loss limit 50% consumed ($11.25 / $22.50)</div>
              <div className="alert-time">15 min ago</div>
            </div>

            <div className="alert-v5 alert-critical">
              <div className="alert-message">OANDA connection lost - forex trading paused</div>
              <div className="alert-time">1 hour ago</div>
            </div>
          </div>

        </div>

        {/* ==================== FOOTER ==================== */}
        <div className="col-span-12 text-center py-8 text-slate-500 text-xs">
          Trading Dashboard V5.0 • Enhanced Visual Design • All data is simulated for demonstration
        </div>

      </div>
    </div>
  );
};

export default TradingDashboardV5;
