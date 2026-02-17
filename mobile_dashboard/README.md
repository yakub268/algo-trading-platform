# 📱 Mobile Trading Dashboard

A comprehensive, production-ready mobile trading dashboard for real-time portfolio management and trading oversight. Built with Flask, SocketIO, and modern web technologies for optimal mobile experience.

## ✨ Features

### 🔧 Core Functionality
- **Real-time Portfolio Monitoring** - Live P&L updates with WebSocket integration
- **Interactive Performance Charts** - Chart.js powered visualizations
- **Bot Control Panel** - Start/stop/configure individual trading bots
- **Risk Management Dashboard** - Live risk metrics and alerts
- **Trade Execution Interface** - Manual trade overrides with validation
- **Alert Management System** - Acknowledge and configure notifications

### 📱 Mobile-First Design
- **Progressive Web App (PWA)** - Install as native app
- **Responsive Design** - Optimized for mobile, tablet, and desktop
- **Offline Capability** - View cached data when connection is poor
- **Touch Optimized** - Gesture navigation and touch-friendly interactions
- **Bottom Navigation** - Easy thumb access on mobile devices

### 🎯 Advanced Features
- **Push Notifications** - Real-time alerts for critical events
- **Voice Alerts** - Spoken notifications for important updates
- **Dark/Light Mode** - Theme switching with persistence
- **Portfolio Heatmap** - Visual performance across exchanges
- **Emergency Stop Button** - Instantly halt all trading activities
- **Real-time News Feed** - Market updates and analysis

### 🔒 Security & Performance
- **JWT Authentication** - Secure API access
- **Rate Limiting** - Protection against abuse
- **Input Validation** - Comprehensive trade validation
- **Risk Management** - Position sizing and exposure limits
- **WebSocket Encryption** - Secure real-time communication
- **Offline Security** - Encrypted local storage

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Redis Server** (for caching and sessions)
3. **Existing Trading Bot System** (from parent directory)
4. **API Keys** configured in `.env` file

### Installation

1. **Navigate to the mobile dashboard directory:**
   ```bash
   cd mobile_dashboard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp ../env.example .env
   # Edit .env with your API keys and settings
   ```

4. **Start the server:**
   ```bash
   python run_mobile_dashboard.py
   ```

5. **Access the dashboard:**
   - Open browser to `http://localhost:5001`
   - On mobile: Add to home screen for PWA experience

### Environment Variables

```bash
# Flask Configuration
FLASK_ENV=development
FLASK_SECRET_KEY=your-secret-key
MOBILE_DASHBOARD_PORT=5001

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Trading API Keys
ALPACA_API_KEY=your-alpaca-key
ALPACA_SECRET_KEY=your-alpaca-secret
FREQTRADE_API_URL=http://localhost:8080

# Push Notifications (Optional)
VAPID_PUBLIC_KEY=your-vapid-public-key
VAPID_PRIVATE_KEY=your-vapid-private-key

# Telegram Alerts (Optional)
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_CHAT_ID=your-chat-id

# Feature Flags
ENABLE_TRADE_EXECUTION=true
ENABLE_BOT_CONTROL=true
ENABLE_PUSH_NOTIFICATIONS=true
ENABLE_VOICE_ALERTS=false
ENABLE_EMERGENCY_STOP=true
```

## 🏗️ Architecture

### Project Structure

```
mobile_dashboard/
├── app.py                     # Main Flask application
├── run_mobile_dashboard.py    # Production startup script
├── manifest.json              # PWA manifest
├── service-worker.js          # PWA service worker
├── requirements.txt           # Python dependencies
├── api/
│   └── trade_executor.py      # Advanced trade execution
├── config/
│   └── config.py             # Configuration management
├── static/
│   ├── css/
│   │   └── mobile-dashboard.css # Responsive styles
│   ├── js/
│   │   └── mobile-dashboard.js  # Frontend logic
│   ├── icons/                 # PWA icons
│   └── img/                   # Images and graphics
└── templates/
    └── mobile_dashboard.html  # Main HTML template
```

### Technology Stack

- **Backend:** Flask, SocketIO, SQLite/Redis
- **Frontend:** Vanilla JS, Chart.js, CSS Grid/Flexbox
- **Real-time:** WebSocket with automatic reconnection
- **PWA:** Service Worker, Web App Manifest
- **Charts:** Chart.js with responsive design
- **Styling:** CSS Custom Properties, Mobile-first design

## 📊 Dashboard Sections

### 1. Portfolio Overview
- **Portfolio Summary Widget** - Total value, daily P&L, percentage change
- **Performance Chart** - Real-time portfolio value visualization
- **Asset Allocation** - Pie chart of current holdings
- **Exchange Heatmap** - Performance across different exchanges

### 2. Positions Management
- **Active Positions Table** - Current holdings with P&L
- **Quick Trade Form** - Rapid order entry
- **Position Analytics** - Risk metrics per position

### 3. Alerts & Notifications
- **Active Alerts List** - Real-time risk and trading alerts
- **Alert Configuration** - Customize notification preferences
- **Push Notification Settings** - Browser and mobile notifications

### 4. Bot Control
- **Bot Status Dashboard** - Current status of all trading bots
- **Control Buttons** - Start/stop/restart individual bots
- **Bot Performance Charts** - Individual bot performance metrics

### 5. Performance Analytics
- **Performance Metrics** - Sharpe ratio, max drawdown, win rate
- **Daily P&L Chart** - Historical profit/loss visualization
- **Risk Analysis** - Comprehensive risk metrics

## 🎛️ API Endpoints

### Portfolio & Positions
```
GET /api/portfolio          # Portfolio summary
GET /api/positions          # Active positions
GET /api/performance        # Performance metrics
```

### Trading & Execution
```
POST /api/trade/execute     # Execute manual trade
GET /api/trade/history      # Trade history
GET /api/trade/status/{id}  # Trade status
```

### Alerts & Notifications
```
GET /api/alerts             # Active alerts
POST /api/alerts/{id}/acknowledge  # Acknowledge alert
POST /api/alerts/configure  # Configure alert settings
```

### Bot Management
```
GET /api/bots/status        # Bot status
POST /api/bots/{bot}/control # Control bot (start/stop/restart)
```

## 🔧 Configuration

### Risk Management Settings

```python
RISK_SETTINGS = {
    'max_position_size': 0.05,         # 5% max position size
    'max_portfolio_exposure': 0.95,     # 95% max exposure
    'max_daily_loss': 0.02,            # 2% max daily loss
    'emergency_stop_loss': 0.10        # 10% emergency stop
}
```

### Update Intervals

```python
UPDATE_INTERVALS = {
    'portfolio': 30,      # Portfolio updates every 30 seconds
    'positions': 15,      # Position updates every 15 seconds
    'alerts': 10,         # Alert checks every 10 seconds
    'performance': 60,    # Performance updates every minute
    'bot_status': 30      # Bot status every 30 seconds
}
```

### Theme Configuration

The dashboard supports both dark and light themes with automatic persistence:

```css
/* Dark theme (default) */
--bg-primary: #1a1a2e;
--text-primary: #eee;
--color-success: #00d4aa;
--color-danger: #ff6b6b;

/* Light theme */
--bg-primary: #ffffff;
--text-primary: #1a202c;
```

## 📱 Progressive Web App (PWA)

### Installation
1. **Desktop:** Chrome/Edge will show install prompt
2. **iOS Safari:** Share → Add to Home Screen
3. **Android Chrome:** Menu → Add to Home Screen

### PWA Features
- **Offline Access** - View cached portfolio data
- **Push Notifications** - Background alerts
- **Home Screen Icon** - Quick access
- **Standalone Mode** - Runs like native app
- **Background Sync** - Sync trades when back online

## 🔒 Security Features

### Authentication & Authorization
- JWT tokens for API access
- Rate limiting on all endpoints
- Input validation and sanitization
- CORS protection

### Trade Execution Security
- Risk checks before execution
- Position size validation
- Emergency stop functionality
- Audit trail for all trades

### Data Protection
- Encrypted WebSocket connections
- Secure local storage
- No sensitive data in client logs
- Regular security audits

## 🚀 Production Deployment

### Using Gunicorn (Recommended)

```bash
# Install Gunicorn
pip install gunicorn

# Start with Gunicorn
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5001 mobile_dashboard.app:app
```

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5001

CMD ["python", "run_mobile_dashboard.py", "--host", "0.0.0.0", "--port", "5001", "--env", "production"]
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket support
    location /socket.io/ {
        proxy_pass http://127.0.0.1:5001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## 📈 Performance Optimization

### Frontend Optimizations
- Lazy loading of charts
- Virtual scrolling for large lists
- Debounced API calls
- Optimized bundle size
- Service Worker caching

### Backend Optimizations
- Redis caching for frequent data
- Database connection pooling
- Async request handling
- WebSocket connection management
- Background task processing

## 🛠️ Development

### Local Development

```bash
# Start with hot reload
python run_mobile_dashboard.py --env development

# Run tests
pytest tests/

# Code formatting
black mobile_dashboard/
flake8 mobile_dashboard/
```

### Adding New Features

1. **Backend:** Add API endpoints in `app.py`
2. **Frontend:** Add UI components in `templates/mobile_dashboard.html`
3. **Styling:** Update `static/css/mobile-dashboard.css`
4. **Logic:** Extend `static/js/mobile-dashboard.js`
5. **Configuration:** Update `config/config.py`

### Testing

```bash
# Run all tests
pytest

# Test specific component
pytest tests/test_trade_executor.py

# Test with coverage
pytest --cov=mobile_dashboard tests/
```

## 🔧 Troubleshooting

### Common Issues

**WebSocket Connection Failed**
- Check Redis server is running
- Verify firewall settings
- Confirm CORS configuration

**Trade Execution Failed**
- Verify API keys in `.env`
- Check exchange API status
- Review risk management settings

**PWA Not Installing**
- Ensure HTTPS in production
- Verify manifest.json syntax
- Check service worker registration

**Performance Issues**
- Enable Redis caching
- Optimize chart data points
- Check update intervals

### Logs

```bash
# View application logs
tail -f mobile_dashboard.log

# Check Redis logs (if using Redis)
redis-cli monitor

# WebSocket debug
# Enable debug mode in config
```

## 📞 Support & Contributing

### Getting Help
- Check existing issues in the repository
- Review troubleshooting section
- Enable debug logging for detailed errors

### Contributing
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure code passes linting
5. Submit pull request with clear description

## 📄 License

This mobile trading dashboard is part of the comprehensive trading bot system. See the main project LICENSE file for details.

---

## 🎯 Next Steps

1. **Customize Themes** - Modify CSS variables for your brand
2. **Add Exchanges** - Integrate additional trading platforms
3. **Extend Analytics** - Add more performance metrics
4. **Mobile App** - Consider React Native wrapper
5. **Voice Commands** - Add voice control for trades
6. **AI Integration** - Add trading insights and recommendations

---

*Built with ❤️ for traders who need mobile access to their portfolios*