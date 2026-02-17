/**
 * Mobile Trading Dashboard JavaScript
 * ===================================
 *
 * Handles real-time updates, user interactions, and chart rendering
 * for the mobile trading dashboard.
 *
 * Features:
 * - WebSocket connection for real-time data
 * - Interactive charts with Chart.js
 * - Push notifications
 * - Voice alerts
 * - Responsive interactions
 * - PWA functionality
 *
 * Author: Trading Bot System
 * Created: February 2026
 */

class MobileTradingDashboard {
    constructor() {
        this.socket = null;
        this.charts = {};
        this.theme = localStorage.getItem('dashboard-theme') || 'dark';
        this.voiceAlertsEnabled = localStorage.getItem('voice-alerts') === 'true';
        this.lastUpdateTime = null;
        this.refreshInterval = 30000; // 30 seconds
        this.portfolioData = null;
        this.isOnline = navigator.onLine;

        this.init();
    }

    async init() {
        console.log('Initializing Mobile Trading Dashboard...');

        // Set initial theme
        this.setTheme(this.theme);

        // Initialize components
        this.setupEventListeners();
        this.setupWebSocket();
        this.setupCharts();
        this.setupPWA();
        this.setupNotifications();
        this.setupVoiceAlerts();
        this.setupOfflineSupport();

        // Load initial data
        await this.loadInitialData();

        // Start periodic updates
        this.startPeriodicUpdates();

        console.log('Mobile Trading Dashboard initialized successfully');
    }

    setupEventListeners() {
        // Theme toggle
        const themeToggle = document.querySelector('.theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }

        // Emergency stop button
        const emergencyStop = document.querySelector('.emergency-stop');
        if (emergencyStop) {
            emergencyStop.addEventListener('click', () => this.emergencyStopAll());
        }

        // Bottom navigation
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                this.navigateToSection(item.dataset.section);
            });
        });

        // Alert acknowledgment
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('acknowledge-alert')) {
                this.acknowledgeAlert(e.target.dataset.alertId);
            }
        });

        // Bot controls
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('bot-control')) {
                this.controlBot(
                    e.target.dataset.bot,
                    e.target.dataset.action
                );
            }
        });

        // Online/offline detection
        window.addEventListener('online', () => {
            this.isOnline = true;
            this.updateConnectionStatus();
            this.reconnectWebSocket();
        });

        window.addEventListener('offline', () => {
            this.isOnline = false;
            this.updateConnectionStatus();
        });

        // Refresh on page visibility change
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.isOnline) {
                this.refreshAllData();
            }
        });

        // Swipe gestures for mobile
        this.setupSwipeGestures();
    }

    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const socketUrl = `${protocol}//${window.location.host}/socket.io`;

        try {
            this.socket = io(socketUrl, {
                transports: ['websocket', 'polling'],
                upgrade: true,
                rememberUpgrade: true
            });

            this.socket.on('connect', () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus(true);
                this.socket.emit('subscribe_updates', {
                    types: ['portfolio', 'alerts', 'performance', 'positions']
                });
            });

            this.socket.on('disconnect', () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
            });

            this.socket.on('portfolio_update', (data) => {
                this.updatePortfolioDisplay(data);
            });

            this.socket.on('new_alert', (alert) => {
                this.handleNewAlert(alert);
            });

            this.socket.on('bot_status_update', (status) => {
                this.updateBotStatus(status);
            });

            this.socket.on('performance_update', (data) => {
                this.updatePerformanceCharts(data);
            });

        } catch (error) {
            console.error('Failed to setup WebSocket:', error);
            this.updateConnectionStatus(false);
        }
    }

    setupCharts() {
        // Portfolio Performance Chart
        const perfCtx = document.getElementById('performanceChart');
        if (perfCtx) {
            this.charts.performance = new Chart(perfCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Portfolio Value',
                        data: [],
                        borderColor: '#00d4aa',
                        backgroundColor: 'rgba(0, 212, 170, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: getComputedStyle(document.documentElement)
                                    .getPropertyValue('--text-primary')
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: getComputedStyle(document.documentElement)
                                    .getPropertyValue('--text-secondary')
                            }
                        },
                        y: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: getComputedStyle(document.documentElement)
                                    .getPropertyValue('--text-secondary'),
                                callback: function(value) {
                                    return '$' + value.toLocaleString();
                                }
                            }
                        }
                    },
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    }
                }
            });
        }

        // Asset Allocation Pie Chart
        const allocCtx = document.getElementById('allocationChart');
        if (allocCtx) {
            this.charts.allocation = new Chart(allocCtx, {
                type: 'doughnut',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        backgroundColor: [
                            '#00d4aa',
                            '#ff6b6b',
                            '#feca57',
                            '#48cae4',
                            '#a29bfe',
                            '#fd79a8'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: getComputedStyle(document.documentElement)
                                    .getPropertyValue('--text-primary'),
                                padding: 20
                            }
                        }
                    }
                }
            });
        }

        // P&L Histogram
        const pnlCtx = document.getElementById('pnlChart');
        if (pnlCtx) {
            this.charts.pnl = new Chart(pnlCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Daily P&L',
                        data: [],
                        backgroundColor: function(context) {
                            const value = context.parsed.y;
                            return value >= 0 ? '#00d4aa' : '#ff6b6b';
                        }
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: getComputedStyle(document.documentElement)
                                    .getPropertyValue('--text-secondary')
                            }
                        },
                        y: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: getComputedStyle(document.documentElement)
                                    .getPropertyValue('--text-secondary'),
                                callback: function(value) {
                                    return '$' + value.toFixed(2);
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    setupPWA() {
        // Register service worker
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/service-worker.js')
                .then(registration => {
                    console.log('Service Worker registered:', registration);
                })
                .catch(error => {
                    console.error('Service Worker registration failed:', error);
                });
        }

        // Handle PWA install prompt
        let deferredPrompt;
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;
            this.showInstallPrompt();
        });

        window.addEventListener('appinstalled', () => {
            console.log('PWA was installed');
            this.hideInstallPrompt();
        });
    }

    setupNotifications() {
        if ('Notification' in window) {
            if (Notification.permission === 'default') {
                this.requestNotificationPermission();
            }
        }

        // Setup push notifications
        if ('serviceWorker' in navigator && 'PushManager' in window) {
            navigator.serviceWorker.ready.then(registration => {
                registration.pushManager.getSubscription()
                    .then(subscription => {
                        if (!subscription) {
                            this.subscribeToPush(registration);
                        }
                    });
            });
        }
    }

    setupVoiceAlerts() {
        if ('speechSynthesis' in window) {
            this.speechSynthesis = window.speechSynthesis;
            this.voiceEnabled = true;
        } else {
            console.warn('Speech synthesis not supported');
            this.voiceEnabled = false;
        }
    }

    setupOfflineSupport() {
        // Cache initial data for offline viewing
        this.cacheData = {
            portfolio: null,
            positions: [],
            alerts: [],
            performance: {}
        };
    }

    setupSwipeGestures() {
        let startX = null;
        let startY = null;

        document.addEventListener('touchstart', (e) => {
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
        });

        document.addEventListener('touchmove', (e) => {
            if (!startX || !startY) return;

            const currentX = e.touches[0].clientX;
            const currentY = e.touches[0].clientY;

            const diffX = startX - currentX;
            const diffY = startY - currentY;

            if (Math.abs(diffX) > Math.abs(diffY)) {
                if (diffX > 50) {
                    // Swipe left - next section
                    this.navigateNext();
                } else if (diffX < -50) {
                    // Swipe right - previous section
                    this.navigatePrevious();
                }
            } else {
                if (diffY > 50) {
                    // Swipe up - refresh
                    this.refreshAllData();
                }
            }

            startX = null;
            startY = null;
        });
    }

    async loadInitialData() {
        try {
            // Load portfolio data
            const portfolioResponse = await fetch('/api/portfolio');
            if (portfolioResponse.ok) {
                const portfolioData = await portfolioResponse.json();
                this.updatePortfolioDisplay(portfolioData);
            }

            // Load alerts
            const alertsResponse = await fetch('/api/alerts');
            if (alertsResponse.ok) {
                const alertsData = await alertsResponse.json();
                this.updateAlertsDisplay(alertsData.alerts);
            }

            // Load performance data
            const performanceResponse = await fetch('/api/performance');
            if (performanceResponse.ok) {
                const performanceData = await performanceResponse.json();
                this.updatePerformanceCharts(performanceData);
            }

            // Load bot status
            const botsResponse = await fetch('/api/bots/status');
            if (botsResponse.ok) {
                const botsData = await botsResponse.json();
                this.updateBotStatus(botsData);
            }

        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showToast('Failed to load data', 'error');
        }
    }

    updatePortfolioDisplay(data) {
        this.portfolioData = data;
        this.cacheData.portfolio = data;

        // Update portfolio metrics
        const totalValue = document.querySelector('.total-value');
        const dayPnl = document.querySelector('.day-pnl');
        const dayPnlPercent = document.querySelector('.day-pnl-percent');

        if (totalValue) {
            totalValue.textContent = this.formatCurrency(data.total_value);
        }

        if (dayPnl) {
            dayPnl.textContent = this.formatCurrency(data.day_pnl);
            dayPnl.className = `metric-value ${data.day_pnl >= 0 ? 'positive' : 'negative'}`;
        }

        if (dayPnlPercent) {
            dayPnlPercent.textContent = this.formatPercent(data.day_pnl_percent);
            dayPnlPercent.className = `metric-value ${data.day_pnl_percent >= 0 ? 'positive' : 'negative'}`;
        }

        // Update last updated time
        this.lastUpdateTime = new Date().toLocaleTimeString();
        const lastUpdated = document.querySelector('.last-updated');
        if (lastUpdated) {
            lastUpdated.textContent = `Updated: ${this.lastUpdateTime}`;
        }

        // Update allocation chart if data available
        if (data.positions && data.positions.length > 0) {
            this.updateAllocationChart(data.positions);
        }

        // Update portfolio heatmap
        this.updatePortfolioHeatmap(data.exchanges);
    }

    updateAlertsDisplay(alerts) {
        this.cacheData.alerts = alerts;

        const alertsContainer = document.querySelector('.alerts-container');
        if (!alertsContainer) return;

        alertsContainer.innerHTML = '';

        if (alerts.length === 0) {
            alertsContainer.innerHTML = '<p class="text-muted">No active alerts</p>';
            return;
        }

        alerts.forEach(alert => {
            const alertElement = this.createAlertElement(alert);
            alertsContainer.appendChild(alertElement);
        });

        // Update alerts count
        const alertsCount = document.querySelector('.alerts-count');
        if (alertsCount) {
            alertsCount.textContent = alerts.length;
        }
    }

    updatePerformanceCharts(data) {
        this.cacheData.performance = data;

        // Update performance chart
        if (this.charts.performance && data.daily_returns) {
            const chart = this.charts.performance;
            chart.data.labels = data.daily_returns.map(item => item.date);
            chart.data.datasets[0].data = data.daily_returns.map(item => item.value);
            chart.update('none');
        }

        // Update P&L chart
        if (this.charts.pnl && data.daily_pnl) {
            const chart = this.charts.pnl;
            chart.data.labels = data.daily_pnl.map(item => item.date);
            chart.data.datasets[0].data = data.daily_pnl.map(item => item.pnl);
            chart.update('none');
        }

        // Update performance metrics
        const sharpeRatio = document.querySelector('.sharpe-ratio');
        const maxDrawdown = document.querySelector('.max-drawdown');
        const winRate = document.querySelector('.win-rate');

        if (sharpeRatio) sharpeRatio.textContent = data.sharpe_ratio?.toFixed(2) || '0.00';
        if (maxDrawdown) maxDrawdown.textContent = this.formatPercent(data.max_drawdown);
        if (winRate) winRate.textContent = this.formatPercent(data.win_rate);
    }

    updateAllocationChart(positions) {
        if (!this.charts.allocation) return;

        const chart = this.charts.allocation;
        const labels = positions.map(pos => pos.symbol);
        const data = positions.map(pos => Math.abs(pos.market_value || 0));

        chart.data.labels = labels;
        chart.data.datasets[0].data = data;
        chart.update('none');
    }

    updatePortfolioHeatmap(exchanges) {
        const heatmapContainer = document.querySelector('.heatmap-container');
        if (!heatmapContainer) return;

        heatmapContainer.innerHTML = '';

        Object.entries(exchanges || {}).forEach(([exchange, data]) => {
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';

            const dayPnl = data.day_pnl || 0;
            if (dayPnl > 0) {
                cell.classList.add('positive');
            } else if (dayPnl < 0) {
                cell.classList.add('negative');
            } else {
                cell.classList.add('neutral');
            }

            cell.innerHTML = `
                <div class="exchange-name">${exchange.toUpperCase()}</div>
                <div class="exchange-pnl">${this.formatCurrency(dayPnl)}</div>
            `;

            heatmapContainer.appendChild(cell);
        });
    }

    updateBotStatus(status) {
        Object.entries(status).forEach(([botName, botStatus]) => {
            const statusElement = document.querySelector(`.bot-status[data-bot="${botName}"]`);
            if (statusElement) {
                const isActive = botStatus.active || botStatus.status === 'running';
                statusElement.className = `bot-status ${isActive ? 'active' : 'inactive'}`;
                statusElement.textContent = isActive ? 'Running' : 'Stopped';
            }

            const controlButtons = document.querySelectorAll(`[data-bot="${botName}"]`);
            controlButtons.forEach(button => {
                const action = button.dataset.action;
                button.disabled = false;

                if (action === 'start') {
                    button.disabled = isActive;
                } else if (action === 'stop') {
                    button.disabled = !isActive;
                }
            });
        });
    }

    handleNewAlert(alert) {
        // Add to alerts display
        const alertsContainer = document.querySelector('.alerts-container');
        if (alertsContainer) {
            const alertElement = this.createAlertElement(alert);
            alertsContainer.insertBefore(alertElement, alertsContainer.firstChild);
        }

        // Show browser notification
        this.showNotification(alert.title, alert.message, alert.severity);

        // Play voice alert if enabled
        if (this.voiceAlertsEnabled && alert.severity === 'high') {
            this.playVoiceAlert(alert.title);
        }

        // Show toast
        this.showToast(alert.message, alert.severity);
    }

    createAlertElement(alert) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert-item severity-${alert.severity}`;
        alertDiv.innerHTML = `
            <div class="alert-icon">⚠️</div>
            <div class="alert-content">
                <div class="alert-title">${alert.title}</div>
                <div class="alert-message">${alert.message}</div>
                <div class="alert-time">${new Date(alert.created_at).toLocaleTimeString()}</div>
                <div class="alert-actions">
                    <button class="btn btn-sm acknowledge-alert" data-alert-id="${alert.id}">
                        Acknowledge
                    </button>
                </div>
            </div>
        `;
        return alertDiv;
    }

    async acknowledgeAlert(alertId) {
        try {
            const response = await fetch(`/api/alerts/${alertId}/acknowledge`, {
                method: 'POST'
            });

            if (response.ok) {
                // Remove alert from display
                const alertElement = document.querySelector(`[data-alert-id="${alertId}"]`).closest('.alert-item');
                if (alertElement) {
                    alertElement.remove();
                }
                this.showToast('Alert acknowledged', 'success');
            }
        } catch (error) {
            console.error('Failed to acknowledge alert:', error);
            this.showToast('Failed to acknowledge alert', 'error');
        }
    }

    async controlBot(botName, action) {
        try {
            const response = await fetch(`/api/bots/${botName}/control`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action })
            });

            if (response.ok) {
                this.showToast(`Bot ${action} command sent`, 'success');
                // Refresh bot status
                setTimeout(() => this.refreshBotStatus(), 2000);
            } else {
                this.showToast(`Failed to ${action} bot`, 'error');
            }
        } catch (error) {
            console.error(`Failed to ${action} bot:`, error);
            this.showToast(`Failed to ${action} bot`, 'error');
        }
    }

    async emergencyStopAll() {
        if (!confirm('Emergency stop all trading bots? This action cannot be undone.')) {
            return;
        }

        try {
            const bots = ['freqtrade', 'strategy'];
            const promises = bots.map(bot =>
                fetch(`/api/bots/${bot}/control`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action: 'stop' })
                })
            );

            await Promise.all(promises);
            this.showToast('Emergency stop executed', 'warning');
            this.playVoiceAlert('Emergency stop executed');
        } catch (error) {
            console.error('Emergency stop failed:', error);
            this.showToast('Emergency stop failed', 'error');
        }
    }

    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        this.setTheme(this.theme);
        localStorage.setItem('dashboard-theme', this.theme);
    }

    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);

        // Update chart colors
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                const textColor = getComputedStyle(document.documentElement)
                    .getPropertyValue('--text-primary');

                chart.options.plugins.legend.labels.color = textColor;
                chart.options.scales.x.ticks.color = textColor;
                chart.options.scales.y.ticks.color = textColor;
                chart.update('none');
            }
        });
    }

    showNotification(title, message, severity) {
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification(title, {
                body: message,
                icon: '/static/icons/icon-192x192.png',
                tag: 'trading-alert'
            });
        }
    }

    playVoiceAlert(message) {
        if (this.voiceEnabled && this.voiceAlertsEnabled) {
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 0.8;
            this.speechSynthesis.speak(utterance);
        }
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <strong>${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
                <p>${message}</p>
            </div>
        `;

        document.body.appendChild(toast);

        // Show toast
        setTimeout(() => toast.classList.add('show'), 100);

        // Hide toast
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => document.body.removeChild(toast), 300);
        }, 5000);
    }

    navigateToSection(section) {
        // Update active navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-section="${section}"]`).classList.add('active');

        // Show/hide sections
        document.querySelectorAll('.dashboard-section').forEach(section => {
            section.style.display = 'none';
        });
        document.getElementById(section).style.display = 'block';
    }

    updateConnectionStatus(connected = this.isOnline) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.connection-status span');

        if (statusDot) {
            statusDot.className = `status-dot ${connected ? 'connected' : 'disconnected'}`;
        }

        if (statusText) {
            statusText.textContent = connected ? 'Online' : 'Offline';
        }
    }

    reconnectWebSocket() {
        if (!this.socket || this.socket.disconnected) {
            this.setupWebSocket();
        }
    }

    async refreshAllData() {
        await this.loadInitialData();
    }

    startPeriodicUpdates() {
        setInterval(() => {
            if (this.isOnline && !document.hidden) {
                this.refreshAllData();
            }
        }, this.refreshInterval);
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }

    formatPercent(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'percent',
            minimumFractionDigits: 2
        }).format(value / 100);
    }

    // Additional utility methods
    requestNotificationPermission() {
        Notification.requestPermission().then(permission => {
            if (permission === 'granted') {
                console.log('Notifications enabled');
            }
        });
    }

    subscribeToPush(registration) {
        // Implementation for push subscription
        // Would need VAPID keys configuration
    }

    showInstallPrompt() {
        // Show custom install prompt
    }

    hideInstallPrompt() {
        // Hide install prompt
    }

    navigateNext() {
        // Navigate to next section
    }

    navigatePrevious() {
        // Navigate to previous section
    }

    async refreshBotStatus() {
        try {
            const response = await fetch('/api/bots/status');
            if (response.ok) {
                const status = await response.json();
                this.updateBotStatus(status);
            }
        } catch (error) {
            console.error('Failed to refresh bot status:', error);
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new MobileTradingDashboard();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MobileTradingDashboard;
}