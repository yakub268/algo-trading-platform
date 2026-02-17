"""
MASTER ORCHESTRATOR - ALL 23 BOTS
=================================

Unified system to run ALL trading bots in paper mode.

Markets Covered:
- Stocks/ETFs (7 strategies)
- Prediction Markets (8 bots)
- Forex (2 bots)
- Crypto (2 bots)
- Other (4 bots)

Features:
- Paper trading simulation for all
- Unified trade logging
- Centralized risk management
- Real-time status dashboard
- Telegram alerts

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import logging
import time
import json
import sqlite3
import signal
import threading
import asyncio
import inspect
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum
import schedule
import concurrent.futures

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Lazy imports for slow modules (deferred until first use)
# DoNothingFilter: ~0.9s import (numpy+pandas), disabled by default (V4)
DO_NOTHING_FILTER_AVAILABLE = True  # assume available; checked on first use
DoNothingFilter = None  # lazy-loaded in _init_do_nothing_filter()

# RegimeDetector: fast import (numpy+pandas already cached by bots)
try:
    from filters.regime_detector import RegimeDetector, MarketRegime as RegimeType
    REGIME_DETECTOR_AVAILABLE = True
except ImportError:
    REGIME_DETECTOR_AVAILABLE = False

# PositionSizer: ~1.3s import (scipy), broken import anyway (no RiskLevel export)
POSITION_SIZER_AVAILABLE = False
PositionSizer = None

# Import Advanced Risk Management
try:
    from risk_management.integration.trading_integration import (
        TradingSystemIntegration, initialize_risk_management
    )
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError:
    RISK_MANAGEMENT_AVAILABLE = False

# Configure logging (explicit setup — basicConfig is a no-op if imports already touched root logger)
os.makedirs('logs', exist_ok=True)
_root = logging.getLogger()
_root.setLevel(logging.INFO)
_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
if not any(isinstance(h, logging.FileHandler) for h in _root.handlers):
    _fh = logging.FileHandler('logs/master_orchestrator.log')
    _fh.setFormatter(_fmt)
    _root.addHandler(_fh)
if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in _root.handlers):
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    _root.addHandler(_sh)
logger = logging.getLogger('MasterOrchestrator')

# Suppress noisy yfinance "possibly delisted" errors - our code handles empty data gracefully
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class Market(Enum):
    STOCKS = "stocks"
    CRYPTO = "crypto"
    FOREX = "forex"
    PREDICTION = "prediction"
    OTHER = "other"


class BotStatus(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"
    WAITING = "waiting"


@dataclass
class BotConfig:
    """Configuration for a single bot"""
    name: str
    module_path: str
    class_name: str
    market: Market
    enabled: bool = True
    schedule_type: str = "interval"  # interval, daily, hourly, custom
    schedule_value: Any = 300  # seconds for interval, time for daily
    allocation_pct: float = 0.0  # % of capital
    description: str = ""


@dataclass
class PaperTrade:
    """Paper trade record"""
    trade_id: str
    bot_name: str
    market: str
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    entry_time: datetime
    exit_time: Optional[datetime]
    pnl: float
    status: str


@dataclass
class BotState:
    """Runtime state of a bot"""
    config: BotConfig
    instance: Any = None
    status: BotStatus = BotStatus.STOPPED
    last_run: Optional[datetime] = None
    last_signal: Optional[Dict] = None
    error: Optional[str] = None
    trades_today: int = 0
    pnl_today: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


# =============================================================================
# BOT REGISTRY - ALL 23 BOTS
# =============================================================================

BOT_REGISTRY: List[BotConfig] = [
    # =========================================================================
    # STOCK/ETF STRATEGIES (7) - PDT RULE: Need $25k+ to enable
    # =========================================================================
    BotConfig(
        name="RSI2-MeanReversion",
        module_path="strategies.rsi2_mean_reversion",
        class_name="RSI2MeanReversion",
        market=Market.STOCKS,
        enabled=False,  # PDT rule - need $25k+
        schedule_type="interval",
        schedule_value=300,
        allocation_pct=0.0,
        description="Larry Connors RSI-2 (PDT rule - need $25k)"
    ),
    BotConfig(
        name="CumulativeRSI",
        module_path="strategies.cumulative_rsi",
        class_name="CumulativeRSIStrategy",
        market=Market.STOCKS,
        enabled=False,  # PDT rule
        schedule_type="interval",
        schedule_value=300,
        allocation_pct=0.0,
        description="ConnorsRSI cumulative strategy (PDT rule)"
    ),
    BotConfig(
        name="MACD-RSI-Combo",
        module_path="strategies.macd_rsi_combo",
        class_name="MACDRSIStrategy",
        market=Market.STOCKS,
        enabled=False,  # PDT rule
        schedule_type="interval",
        schedule_value=300,
        allocation_pct=0.0,
        description="MACD + RSI combination (PDT rule)"
    ),
    BotConfig(
        name="BollingerSqueeze",
        module_path="strategies.bollinger_squeeze",
        class_name="BollingerSqueezeStrategy",
        market=Market.STOCKS,
        enabled=False,  # PDT rule
        schedule_type="interval",
        schedule_value=300,
        allocation_pct=0.0,
        description="Bollinger Band squeeze breakout (PDT rule)"
    ),
    BotConfig(
        name="MTF-RSI",
        module_path="strategies.mtf_rsi",
        class_name="MultiTimeframeRSI",
        market=Market.STOCKS,
        enabled=False,  # PDT rule
        schedule_type="interval",
        schedule_value=300,
        allocation_pct=0.0,
        description="Multi-timeframe RSI (PDT rule)"
    ),
    BotConfig(
        name="DualMomentum",
        module_path="strategies.dual_momentum",
        class_name="DualMomentumStrategy",
        market=Market.STOCKS,
        enabled=False,  # PDT rule
        schedule_type="daily",
        schedule_value="09:00",
        allocation_pct=0.0,
        description="Gary Antonacci dual momentum (PDT rule)"
    ),
    BotConfig(
        name="SectorRotation",
        module_path="bots.sector_rotation_bot",
        class_name="SectorRotationBot",
        market=Market.STOCKS,
        enabled=False,  # PDT rule
        schedule_type="daily",
        schedule_value="09:05",
        allocation_pct=0.0,
        description="Monthly sector rotation (PDT rule)"
    ),

    # =========================================================================
    # PREDICTION MARKETS - ALL ENABLED
    # =========================================================================
    BotConfig(
        name="Kalshi-Fed",
        module_path="bots.kalshi_strategy",  # FIX: Use strategy module with CME FedWatch
        class_name="KalshiFedStrategy",  # FIX: Use strategy class
        market=Market.PREDICTION,
        enabled=True,
        schedule_type="interval",
        schedule_value=600,
        allocation_pct=0.15,
        description="Kalshi Fed/CPI/NFP events (CME FedWatch probabilities)"
    ),
    BotConfig(
        name="Weather-Edge",
        module_path="bots.weather_edge_finder",
        class_name="WeatherEdgeFinder",
        market=Market.PREDICTION,
        enabled=True,  # ENABLED
        schedule_type="interval",
        schedule_value=1800,
        allocation_pct=0.03,
        description="NWS weather prediction markets"
    ),
    BotConfig(
        name="Sports-Edge",
        module_path="bots.sports_edge_finder",
        class_name="SportsEdgeFinder",
        market=Market.PREDICTION,
        enabled=True,  # ENABLED
        schedule_type="interval",
        schedule_value=900,
        allocation_pct=0.03,
        description="Sports game outcome predictions"
    ),
    BotConfig(
        name="Sports-AI",
        module_path="bots.sports_ai_bot",
        class_name="SportsAIBot",
        market=Market.PREDICTION,
        enabled=True,
        schedule_type="interval",
        schedule_value=1800,
        allocation_pct=0.05,
        description="AI-powered sports betting"
    ),
    BotConfig(
        name="BoxOffice-Edge",
        module_path="bots.boxoffice_edge_finder",
        class_name="BoxOfficeEdgeFinder",
        market=Market.PREDICTION,
        enabled=True,  # ENABLED
        schedule_type="daily",
        schedule_value="08:00",
        allocation_pct=0.02,
        description="Movie box office predictions"
    ),
    BotConfig(
        name="Awards-Edge",
        module_path="bots.awards_edge_finder",
        class_name="AwardsEdgeFinder",
        market=Market.PREDICTION,
        enabled=True,  # ENABLED
        schedule_type="daily",
        schedule_value="08:30",
        allocation_pct=0.02,
        description="Awards show predictions"
    ),
    BotConfig(
        name="Climate-Edge",
        module_path="bots.climate_edge_finder",
        class_name="ClimateEdgeFinder",
        market=Market.PREDICTION,
        enabled=True,  # ENABLED
        schedule_type="daily",
        schedule_value="07:00",
        allocation_pct=0.02,
        description="Climate event predictions"
    ),
    BotConfig(
        name="Economic-Edge",
        module_path="bots.economic_releases_edge_finder",
        class_name="EconomicReleasesEdgeFinder",
        market=Market.PREDICTION,
        enabled=True,  # ENABLED
        schedule_type="daily",
        schedule_value="07:30",
        allocation_pct=0.03,
        description="Economic releases edge finder"
    ),
    BotConfig(
        name="Cross-Market-Arbitrage",
        module_path="bots.arbitrage_bot",
        class_name="CrossMarketArbitrageBot",
        market=Market.PREDICTION,
        enabled=True,
        schedule_type="interval",
        schedule_value=600,
        allocation_pct=0.03,
        description="Cross-market arbitrage detection"
    ),
    BotConfig(
        name="Event-Edge",
        module_path="bots.event_trading.event_edge_bot",
        class_name="EventEdgeBot",
        market=Market.PREDICTION,
        enabled=False,  # DISABLED: 0 wins / 49 trades (-$380). Manual bets 6/6. Needs rebuild before re-enable.
        schedule_type="interval",
        schedule_value=60,
        allocation_pct=0.0,
        description="Autonomous Kalshi edge detection (8 sources + ML + consensus)"
    ),

    # =========================================================================
    # FOREX - ALL ENABLED
    # =========================================================================
    BotConfig(
        name="OANDA-Forex",
        module_path="bots.oanda_bot",
        class_name="OANDABot",
        market=Market.FOREX,
        enabled=True,
        schedule_type="interval",
        schedule_value=300,
        allocation_pct=0.15,
        description="OANDA forex MA crossover + RSI"
    ),
    BotConfig(
        name="London-Breakout",
        module_path="bots.oanda_london_breakout",
        class_name="OANDALondonBreakout",
        market=Market.FOREX,
        enabled=True,  # ENABLED
        schedule_type="daily",
        schedule_value="03:00",
        allocation_pct=0.05,
        description="London session breakout strategy"
    ),

    # =========================================================================
    # CRYPTO - ENABLED WHERE POSSIBLE
    # =========================================================================
    BotConfig(
        name="FundingRate-Arb",
        module_path="bots.funding_rate_arbitrage",
        class_name="FundingRateScanner",
        market=Market.CRYPTO,
        enabled=False,  # Binance Futures blocked in US
        schedule_type="interval",
        schedule_value=300,
        allocation_pct=0.0,
        description="Funding rate arb (Binance blocked in US)"
    ),
    BotConfig(
        name="Crypto-Arb",
        module_path="bots.crypto_arb_scanner",
        class_name="CryptoArbScanner",
        market=Market.CRYPTO,
        enabled=False,  # Needs $100k+ capital
        schedule_type="interval",
        schedule_value=60,
        allocation_pct=0.0,
        description="Cross-exchange arb (needs $100k+)"
    ),
    BotConfig(
        name="Kalshi-Hourly-Crypto",
        module_path="bots.kalshi_hourly_crypto",
        class_name="KalshiHourlyCryptoBot",
        market=Market.CRYPTO,
        enabled=False,  # NEGATIVE EV proven
        schedule_type="interval",
        schedule_value=300,
        allocation_pct=0.0,
        description="DISABLED: Negative EV proven"
    ),
    BotConfig(
        name="Alpaca-Crypto-Momentum",
        module_path="bots.alpaca_crypto_momentum",
        class_name="AlpacaCryptoMomentumBot",
        market=Market.CRYPTO,
        enabled=True,  # V4: EMA crossover replaces RSI-2
        schedule_type="interval",
        schedule_value=900,  # 15-min timeframe
        allocation_pct=0.10,
        description="V4 EMA(12/26) Crossover + Volume BTC/ETH"
    ),

    # =========================================================================
    # OTHER - ENABLED
    # =========================================================================
    BotConfig(
        name="Earnings-PEAD",
        module_path="bots.earnings_bot",
        class_name="EarningsBot",
        market=Market.OTHER,
        enabled=True,  # ENABLED
        schedule_type="daily",
        schedule_value="16:30",
        allocation_pct=0.03,
        description="Post-earnings announcement drift"
    ),
    BotConfig(
        name="Sentiment-Bot",
        module_path="bots.sentiment_bot",
        class_name="SentimentBot",
        market=Market.OTHER,
        enabled=True,  # ENABLED
        schedule_type="interval",
        schedule_value=600,
        allocation_pct=0.02,
        description="News sentiment trading"
    ),
    BotConfig(
        name="FOMC-Trader",
        module_path="bots.alpaca_fomc_trader",
        class_name="AlpacaFOMCTrader",
        market=Market.OTHER,
        enabled=True,  # ENABLED
        schedule_type="interval",
        schedule_value=3600,
        allocation_pct=0.02,
        description="Fed meeting plays"
    ),
    BotConfig(
        name="Market-Scanner",
        module_path="bots.market_scanner",
        class_name="MarketScanner",
        market=Market.OTHER,
        enabled=True,  # ENABLED
        schedule_type="interval",
        schedule_value=300,
        allocation_pct=0.0,
        description="Market-wide opportunity scanner"
    ),

    # =========================================================================
    # COINBASE & DCA CRYPTO
    # =========================================================================
    BotConfig(
        name="Coinbase-Arb",
        module_path="bots.coinbase_arb_bot",
        class_name="CoinbaseArbBot",
        market=Market.CRYPTO,
        enabled=True,
        schedule_type="interval",
        schedule_value=60,
        allocation_pct=0.03,
        description="Cross-exchange crypto arb"
    ),
    BotConfig(
        name="Crypto-DCA-Accumulation",
        module_path="bots.crypto_dca_bot",
        class_name="CryptoDCABot",
        market=Market.CRYPTO,
        enabled=False,  # DISABLED: Coinbase account has $0 balance
        schedule_type="interval",
        schedule_value=900,
        allocation_pct=0.10,
        description="DISABLED: Coinbase $0 balance - DCA for XRP/HBAR/XLM"
    ),
    BotConfig(
        name="AI-Crypto-Analyzer",
        module_path="bots.ai_crypto_analyzer",
        class_name="AICryptoAnalyzer",
        market=Market.CRYPTO,
        enabled=True,  # NEW - AI analysis before trades
        schedule_type="interval",
        schedule_value=600,  # Every 10 minutes
        allocation_pct=0.0,  # Advisory only - no direct trades
        description="AI analysis of XRP/HBAR/XLM/BTC/ETH before buys"
    ),
    BotConfig(
        name="Colorado-Sports-Arb",
        module_path="bots.colorado_sports_arb",
        class_name="ColoradoSportsArbBot",
        market=Market.OTHER,
        enabled=True,  # ENABLED
        schedule_type="interval",
        schedule_value=300,
        allocation_pct=0.02,
        description="CO sportsbook arbitrage scanner"
    ),
    BotConfig(
        name="Fidelity-Alerts",
        module_path="bots.fidelity_alerts",
        class_name="FidelityAlertsBot",
        market=Market.STOCKS,
        enabled=True,
        schedule_type="interval",
        schedule_value=600,
        allocation_pct=0.0,
        description="ETF premium/discount alerts"
    ),

    # =========================================================================
    # AI & ML TRADING
    # =========================================================================
    BotConfig(
        name="Computer-Vision-Bot",
        module_path="bots.computer_vision_bot",
        class_name="ComputerVisionBot",
        market=Market.OTHER,
        enabled=True,  # ENABLED
        schedule_type="interval",
        schedule_value=1800,
        allocation_pct=0.02,
        description="AI computer vision trading"
    ),
    BotConfig(
        name="ML-Prediction-Bot",
        module_path="bots.ml_prediction_bot",
        class_name="MLPredictionBot",
        market=Market.CRYPTO,
        enabled=True,
        schedule_type="interval",
        schedule_value=600,
        allocation_pct=0.05,
        description="ML price prediction (crypto only: BTC, ETH, SOL)"
    ),
    BotConfig(
        name="AI-Prediction-Analyzer",
        module_path="bots.ai_prediction_analyzer",
        class_name="AIPredictionAnalyzer",
        market=Market.OTHER,
        enabled=True,  # NEW
        schedule_type="interval",
        schedule_value=1200,
        allocation_pct=0.02,
        description="AI news analysis for predictions"
    ),

    # =========================================================================
    # KALSHI ADVANCED STRATEGIES - NEW
    # =========================================================================
    BotConfig(
        name="Kalshi-Probability-Arb",
        module_path="bots.kalshi_probability_arb",
        class_name="KalshiProbabilityArbitrage",
        market=Market.PREDICTION,
        enabled=True,  # NEW - Risk-free probability arb
        schedule_type="interval",
        schedule_value=30,  # Fast - arbs close quickly
        allocation_pct=0.05,
        description="YES+NO < $1 risk-free arbitrage"
    ),
    BotConfig(
        name="Kalshi-Sum-To-One",
        module_path="bots.kalshi_sum_to_one_arb",
        class_name="KalshiSumToOneArbitrage",
        market=Market.PREDICTION,
        enabled=True,  # NEW - Multi-outcome arb
        schedule_type="interval",
        schedule_value=60,
        allocation_pct=0.05,
        description="Multi-outcome sum-to-one arbitrage"
    ),
    BotConfig(
        name="Kalshi-Market-Maker",
        module_path="bots.kalshi_market_maker",
        class_name="KalshiMarketMaker",
        market=Market.PREDICTION,
        enabled=True,  # NEW - Zero maker fees!
        schedule_type="interval",
        schedule_value=30,
        allocation_pct=0.05,
        description="Market making (zero maker fees)"
    ),
    BotConfig(
        name="Kalshi-Cross-Platform",
        module_path="bots.kalshi_cross_platform_arb",
        class_name="KalshiCrossPlatformArbitrage",
        market=Market.PREDICTION,
        enabled=True,  # NEW - Kalshi vs others
        schedule_type="interval",
        schedule_value=120,
        allocation_pct=0.03,
        description="Kalshi vs other platforms arb"
    ),

    # =========================================================================
    # POLYMARKET STRATEGIES - NEW
    # =========================================================================
    BotConfig(
        name="Polymarket-Cross-Platform",
        module_path="bots.polymarket.cross_platform_arb",
        class_name="CrossPlatformArbitrage",
        market=Market.PREDICTION,
        enabled=True,  # NEW - Polymarket vs Kalshi
        schedule_type="interval",
        schedule_value=120,
        allocation_pct=0.03,
        description="Polymarket vs Kalshi arbitrage"
    ),
    BotConfig(
        name="Polymarket-Sum-To-One",
        module_path="bots.polymarket.sum_to_one_arb",
        class_name="SumToOneArbitrage",
        market=Market.PREDICTION,
        enabled=True,  # NEW - NegRisk arb
        schedule_type="interval",
        schedule_value=60,
        allocation_pct=0.03,
        description="Polymarket sum-to-one NegRisk arb"
    ),
    BotConfig(
        name="Polymarket-Sports-Delay",
        module_path="bots.polymarket.sports_delay_bot",
        class_name="SportsDelayBot",
        market=Market.PREDICTION,
        enabled=True,  # NEW - Sports delay exploitation
        schedule_type="interval",
        schedule_value=10,  # Very fast for live sports
        allocation_pct=0.02,
        description="Sports market delay exploitation"
    ),
    BotConfig(
        name="Polymarket-Market-Maker",
        module_path="bots.polymarket.market_maker",
        class_name="MarketMakerBot",
        market=Market.PREDICTION,
        enabled=True,  # NEW - Market making
        schedule_type="interval",
        schedule_value=30,
        allocation_pct=0.02,
        description="Polymarket liquidity provision"
    ),

    # =========================================================================
    # DEFI & ADVANCED - NEW
    # =========================================================================
    BotConfig(
        name="DeFi-Yield-Optimizer",
        module_path="bots.defi_yield_optimizer",
        class_name="DeFiYieldOptimizer",
        market=Market.CRYPTO,
        enabled=True,  # NEW - DeFi yield farming
        schedule_type="interval",
        schedule_value=3600,  # Hourly checks
        allocation_pct=0.02,
        description="DeFi yield optimization (Aave/Compound/Curve)"
    ),
    BotConfig(
        name="Cross-Exchange-Arb",
        module_path="bots.cross_exchange_arb",
        class_name="CrossExchangeArbitrage",
        market=Market.CRYPTO,
        enabled=True,  # NEW - CEX arbitrage
        schedule_type="interval",
        schedule_value=30,
        allocation_pct=0.02,
        description="Coinbase/Kraken cross-exchange arb"
    ),
    BotConfig(
        name="Multi-Market-Strategy",
        module_path="bots.multi_market_strategy",
        class_name="MultiMarketStrategy",
        market=Market.OTHER,
        enabled=True,  # NEW - Unified ranking
        schedule_type="interval",
        schedule_value=300,
        allocation_pct=0.0,  # Coordinator, no direct allocation
        description="Multi-market opportunity ranker"
    ),

    # =========================================================================
    # AGGRESSIVE / HIGH-RISK CRYPTO BOTS (Toggle via dashboard)
    # =========================================================================
    BotConfig(
        name="Momentum-Scalper",
        module_path="bots.aggressive.momentum_scalper",
        class_name="MomentumScalper",
        market=Market.CRYPTO,
        enabled=False,  # Controlled by high_risk_enabled toggle
        schedule_type="interval",
        schedule_value=120,  # Every 2 minutes
        allocation_pct=0.05,
        description="HIGH RISK: Buys coins up 5%+ in 1hr, 3% trailing stop"
    ),
    BotConfig(
        name="Breakout-Hunter",
        module_path="bots.aggressive.breakout_hunter",
        class_name="BreakoutHunter",
        market=Market.CRYPTO,
        enabled=False,  # Controlled by high_risk_enabled toggle
        schedule_type="interval",
        schedule_value=180,  # Every 3 minutes
        allocation_pct=0.05,
        description="HIGH RISK: Volume spike + price breakout detection"
    ),
    BotConfig(
        name="Meme-Sniper",
        module_path="bots.aggressive.meme_sniper",
        class_name="MemeCoinSniper",
        market=Market.CRYPTO,
        enabled=False,  # Controlled by high_risk_enabled toggle
        schedule_type="interval",
        schedule_value=60,  # Every minute - memes move fast
        allocation_pct=0.03,
        description="HIGH RISK: PEPE/BONK/WIF/SHIB meme coin scalping"
    ),
    BotConfig(
        name="RSI-Extremes",
        module_path="bots.aggressive.rsi_extremes",
        class_name="RSIExtremesBot",
        market=Market.CRYPTO,
        enabled=False,  # Controlled by high_risk_enabled toggle
        schedule_type="interval",
        schedule_value=300,  # Every 5 minutes
        allocation_pct=0.05,
        description="HIGH RISK: Buy RSI<20, sell RSI>80 mean reversion"
    ),
    BotConfig(
        name="Multi-Momentum",
        module_path="bots.aggressive.multi_momentum",
        class_name="MultiMomentumBot",
        market=Market.CRYPTO,
        enabled=True,  # Controlled by high_risk_enabled toggle
        schedule_type="interval",
        schedule_value=14400,  # Every 4 hours
        allocation_pct=0.10,
        description="HIGH RISK: Rotate into top 3 daily performers"
    ),

    # =========================================================================
    # FLEET ORCHESTRATOR — 19 sub-bots managed by single adapter
    # =========================================================================
    BotConfig(
        name="Fleet-Orchestrator",
        module_path="bots.fleet.fleet_adapter",
        class_name="FleetAdapter",
        market=Market.OTHER,
        enabled=True,
        schedule_type="interval",
        schedule_value=60,  # Fleet checks sub-bot schedules internally
        allocation_pct=0.0,  # Fleet manages its own capital allocation
        description="Fleet: 19 sub-bots (Kalshi/crypto/forex/prediction)"
    ),
]


# =============================================================================
# TRADING DATABASE (Always data/live/ - shared with dashboard)
# =============================================================================

# Import data path utility for mode-specific storage
try:
    from utils.data_paths import get_db_path, get_mode_string, is_live_mode
    DATA_PATHS_AVAILABLE = True
except ImportError:
    DATA_PATHS_AVAILABLE = False
    def get_db_path(name):
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, 'data', 'live', name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    def get_mode_string(): return "paper"
    def is_live_mode(): return False


class TradingDB:
    """Unified trading database - ALWAYS uses data/live/ for central DB.
    Paper/live distinction affects execution mode, not database location."""

    def __init__(self, db_name: str = "trading_master.db"):
        # ALWAYS use data/live/ for the central trading DB so dashboard and orchestrator share one DB
        base = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(base, 'data', 'live', db_name)
        self.mode = get_mode_string()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        logger.info(f"TradingDB initialized: {self.db_path} (mode={self.mode})")
        self._lock = threading.Lock()
        self._init_db()
        # Enable WAL mode and set busy timeout for concurrent access
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.close()
    
    def _init_db(self):
        """Initialize database tables"""
        with self._lock:
            with self._connect() as conn:
                cursor = conn.cursor()

                # Trades table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        trade_id TEXT PRIMARY KEY,
                        bot_name TEXT,
                        market TEXT,
                        symbol TEXT,
                        side TEXT,
                        entry_price REAL,
                        exit_price REAL,
                        quantity REAL,
                        entry_time TEXT,
                        exit_time TEXT,
                        pnl REAL,
                        pnl_pct REAL,
                        status TEXT
                    )
                ''')

                # Add close_retry_count column if not present (schema migration)
                try:
                    cursor.execute('ALTER TABLE trades ADD COLUMN close_retry_count INTEGER DEFAULT 0')
                except sqlite3.OperationalError:
                    pass  # Column already exists

                # Add PnL breakdown columns (schema migration)
                for col in ['gross_pnl REAL', 'net_pnl REAL', 'estimated_fees REAL']:
                    try:
                        cursor.execute(f'ALTER TABLE trades ADD COLUMN {col}')
                    except sqlite3.OperationalError:
                        pass  # Column already exists

                # Bot status table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS bot_status (
                        bot_name TEXT PRIMARY KEY,
                        status TEXT,
                        last_run TEXT,
                        last_signal TEXT,
                        trades_today INTEGER,
                        pnl_today REAL,
                        error TEXT
                    )
                ''')

                # Daily summary table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS daily_summary (
                        date TEXT,
                        bot_name TEXT,
                        trades INTEGER,
                        wins INTEGER,
                        losses INTEGER,
                        pnl REAL,
                        PRIMARY KEY (date, bot_name)
                    )
                ''')

                # Portfolio value table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio_value (
                        timestamp TEXT PRIMARY KEY,
                        total_value REAL,
                        cash REAL,
                        positions_value REAL
                    )
                ''')

                # Performance indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_bot_symbol_status ON trades(bot_name, symbol, status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)')

                conn.commit()

    def _connect(self):
        """Create a connection with proper pragmas."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def log_trade(self, trade: PaperTrade):
        """Log a paper trade"""
        pnl_pct = 0
        if trade.entry_price > 0 and trade.exit_price:
            pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price

        with self._lock:
            with self._connect() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO trades
                    (trade_id, bot_name, market, symbol, side, entry_price, exit_price,
                     quantity, entry_time, exit_time, pnl, pnl_pct, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.trade_id, trade.bot_name, trade.market, trade.symbol,
                    trade.side, trade.entry_price, trade.exit_price, trade.quantity,
                    trade.entry_time.isoformat() if trade.entry_time else None,
                    trade.exit_time.isoformat() if trade.exit_time else None,
                    trade.pnl, pnl_pct, trade.status
                ))

                conn.commit()

        logger.info(f"Trade logged: {trade.bot_name} {trade.symbol} {trade.side}")
    
    def update_bot_status(self, bot_name: str, status: Dict):
        """Update bot status"""
        # Safely serialize last_signal - handle non-JSON-serializable objects
        last_signal_json = None
        if status.get('last_signal'):
            try:
                last_signal_json = json.dumps(status.get('last_signal'))
            except (TypeError, ValueError) as e:
                # Fallback: convert to summary string
                signal = status.get('last_signal')
                if isinstance(signal, list):
                    last_signal_json = json.dumps({'type': 'list', 'count': len(signal), 'summary': f'{len(signal)} items'})
                else:
                    last_signal_json = json.dumps({'type': str(type(signal).__name__), 'summary': str(signal)[:200]})
                logger.debug(f"Serialization fallback for {bot_name}: {e}")

        with self._lock:
            with self._connect() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO bot_status
                    (bot_name, status, last_run, last_signal, trades_today, pnl_today, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    bot_name,
                    status.get('status', 'unknown'),
                    status.get('last_run'),
                    last_signal_json,
                    status.get('trades_today', 0),
                    status.get('pnl_today', 0),
                    status.get('error')
                ))

                conn.commit()
    
    def get_all_bot_status(self) -> List[Dict]:
        """Get status of all bots"""
        with self._lock:
            with self._connect() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM bot_status")
                columns = [desc[0] for desc in cursor.description]
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return rows
    
    def close_trade(self, bot_name: str, symbol: str, exit_price: float, pnl: float,
                    status: str = 'closed', gross_pnl: float = None,
                    net_pnl: float = None, estimated_fees: float = None):
        """Close an open trade by updating exit info in the database.

        Args:
            status: Target status - 'closed' (default) or 'close_pending' if exchange close failed.
            gross_pnl: Gross PnL before fees.
            net_pnl: Net PnL after estimated fees.
            estimated_fees: Estimated round-trip fees.
        """
        now = datetime.now().isoformat()
        pnl_pct = 0

        with self._lock:
            with self._connect() as conn:
                cursor = conn.cursor()

                # Find the most recent open trade for this bot+symbol
                cursor.execute('''
                    SELECT trade_id, entry_price, side, quantity FROM trades
                    WHERE bot_name = ? AND symbol = ? AND status IN ('open', 'close_pending')
                    ORDER BY entry_time DESC LIMIT 1
                ''', (bot_name, symbol))

                row = cursor.fetchone()
                if row:
                    trade_id, entry_price, side, quantity = row

                    # Calculate pnl_pct based on position direction
                    if entry_price and entry_price > 0 and exit_price and exit_price > 0:
                        if side in ('short', 'sell_short'):
                            # For SHORT: profit when exit < entry
                            pnl_pct = (entry_price - exit_price) / entry_price
                        else:
                            # For LONG: profit when exit > entry
                            pnl_pct = (exit_price - entry_price) / entry_price

                    # If pnl wasn't calculated by caller, compute it now
                    if pnl == 0 and quantity and quantity > 0:
                        if side in ('short', 'sell_short'):
                            pnl = quantity * (entry_price - exit_price)
                        else:
                            pnl = quantity * (exit_price - entry_price)
                        logger.info(f"[PNL FIX] Computed missing PnL for {symbol}: ${pnl:.2f}")

                    cursor.execute('''
                        UPDATE trades
                        SET exit_price = ?, exit_time = ?, pnl = ?, pnl_pct = ?, status = ?,
                            gross_pnl = ?, net_pnl = ?, estimated_fees = ?
                        WHERE trade_id = ?
                    ''', (exit_price, now, pnl, pnl_pct, status,
                          gross_pnl, net_pnl, estimated_fees, trade_id))

                    conn.commit()
                    logger.info(
                        f"[TRADE LIFECYCLE] Closing trade {trade_id}: {symbol} | "
                        f"Entry: ${entry_price:.4f} -> Exit: ${exit_price:.4f} | "
                        f"P&L: ${pnl:.2f} ({pnl_pct:.1%}) | Status: {status}"
                    )
                else:
                    logger.warning(f"[TRADE LIFECYCLE] No open trade found to close for {bot_name} {symbol}")

    def get_open_trades(self) -> List[Dict]:
        """Get all open trades"""
        with self._lock:
            with self._connect() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM trades WHERE status = 'open'")
                columns = [desc[0] for desc in cursor.description]
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return rows

    def get_close_pending_trades(self) -> List[Dict]:
        """Get all trades stuck in close_pending status (exchange close failed)."""
        with self._lock:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM trades WHERE status = 'close_pending'")
                columns = [desc[0] for desc in cursor.description]
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return rows

    def increment_close_retry(self, trade_id: str):
        """Increment the close retry counter for a trade."""
        with self._lock:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE trades SET close_retry_count = COALESCE(close_retry_count, 0) + 1
                    WHERE trade_id = ?
                ''', (trade_id,))
                conn.commit()

    def force_close_trade_in_db(self, trade_id: str, exit_price: float = 0, pnl: float = 0):
        """Force a close_pending trade to closed status (after max retries)."""
        now = datetime.now().isoformat()
        with self._lock:
            with self._connect() as conn:
                cursor = conn.cursor()

                # Get entry price and quantity to calculate pnl_pct
                cursor.execute('SELECT entry_price, quantity, side FROM trades WHERE trade_id = ?', (trade_id,))
                row = cursor.fetchone()
                pnl_pct = 0
                if row:
                    entry_price, quantity, side = row
                    if entry_price and entry_price > 0 and exit_price and exit_price > 0:
                        if side in ('short', 'sell_short'):
                            pnl_pct = (entry_price - exit_price) / entry_price
                        else:
                            pnl_pct = (exit_price - entry_price) / entry_price

                cursor.execute('''
                    UPDATE trades SET status = 'closed', exit_time = ?, exit_price = ?, pnl = ?, pnl_pct = ?
                    WHERE trade_id = ?
                ''', (now, exit_price, pnl, pnl_pct, trade_id))
                conn.commit()
                logger.warning(
                    f"[TRADE LIFECYCLE] Force-closed trade {trade_id} in DB after max retries "
                    f"(exit_price=${exit_price:.4f}, pnl=${pnl:.2f}, pnl_pct={pnl_pct:.2%})"
                )

    def get_today_summary(self) -> Dict:
        """Get today's trading summary"""
        today = datetime.now().strftime('%Y-%m-%d')

        with self._lock:
            with self._connect() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT COUNT(*) as trades,
                           SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                           SUM(pnl) as total_pnl
                    FROM trades
                    WHERE date(entry_time) = ?
                ''', (today,))

                row = cursor.fetchone()

        return {
            'date': today,
            'trades': row[0] or 0,
            'wins': row[1] or 0,
            'total_pnl': row[2] or 0
        }


# =============================================================================
# MASTER ORCHESTRATOR
# =============================================================================

class MasterOrchestrator:
    """
    Master orchestrator that runs ALL 23 bots in paper mode.
    
    Features:
    - Unified initialization
    - Scheduled execution
    - Risk management
    - Trade logging
    - Status monitoring
    """
    
    def __init__(
        self,
        starting_capital: float = 500.0,
        paper_mode: bool = True,
        use_do_nothing_filter: bool = False,  # V4: Disabled for execution test
        enable_risk_management: bool = True  # V4: Re-enabled after post-mortem analysis
    ):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.paper_mode = paper_mode
        self.running = False

        # Initialize components
        self.db = TradingDB()
        self.bots: Dict[str, BotState] = {}
        self.alerts = None

        # Initialize Advanced Risk Management
        self.enable_risk_management = enable_risk_management and RISK_MANAGEMENT_AVAILABLE
        self.risk_integration: Optional[TradingSystemIntegration] = None

        # Initialize DoNothingFilter (lazy import — saves ~0.9s startup when disabled)
        self.use_do_nothing_filter = use_do_nothing_filter
        if self.use_do_nothing_filter:
            try:
                from filters.do_nothing_filter import DoNothingFilter as _DNF
                self.do_nothing_filter = _DNF(
                    entropy_threshold=0.95,  # Much higher to allow live trading
                    volatility_percentile=0.92,
                    tired_hours=(3, 5),  # Narrow window
                    drawdown_threshold=0.07
                )
                self.do_nothing_active = False
                self.do_nothing_reason = None
            except ImportError:
                logger.warning("DoNothingFilter import failed — disabling")
                self.use_do_nothing_filter = False
                self.do_nothing_filter = None
        else:
            self.do_nothing_filter = None

        # Initialize AI Veto Filter (V6: LIVE mode - actually filters trades)
        self.ai_filter = None
        self.use_ai_filter = True
        self._init_ai_filter(dry_run=False)  # V6: LIVE - AI actively filters trades

        # V7: Initialize Data Hub (central data aggregator)
        self.data_hub = None
        self._init_data_hub()

        # Initialize AI Market Analyst (V6: LLM-powered market analysis + predictions)
        self.market_analyst = None
        self._init_market_analyst()

        # V6+/V7: Initialize advanced AI components (MTF, Signal Injector, Ensemble, Adaptive Params, MetaSignal)
        self.mtf_engine = None
        self.signal_injector = None
        self.ensemble = None
        self.adaptive_params = None
        self.meta_signal = None
        self.prediction_engine = None
        self._init_v6plus_ai()

        # Load high-risk config
        self.high_risk_enabled = self._load_high_risk_config()

        # Telegram command listener (initialized in _init_telegram_command_listener)
        self.telegram_command_listener = None

        # Kalshi shared infrastructure (positions, fills, settlements, risk)
        self.kalshi_infrastructure = None

        # Cached AlpacaCryptoClient instance (lazy-initialized)
        self._alpaca_client = None

        # Circuit breaker: per-bot consecutive loss tracking
        # {bot_name: {'consecutive_losses': int, 'paused_until': datetime|None}}
        self._circuit_breaker: Dict[str, Dict] = {}
        self.CIRCUIT_BREAKER_THRESHOLD = 10     # Pause after N consecutive losses
        self.CIRCUIT_BREAKER_COOLDOWN = 3600    # Cooldown in seconds (1 hour)

        # Thread pool for concurrent bot execution (prevents schedule blocking)
        self._bot_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="bot"
        )
        self._bot_futures: Dict[str, concurrent.futures.Future] = {}
        self._bot_lock = threading.Lock()

        # Load bots
        self._init_alerts()
        self._init_risk_management()
        self._init_telegram_command_listener()
        self._init_bots()

        # Validate DB file exists and is not empty
        self._validate_db_file()

        logger.info("=" * 70)
        logger.info("MASTER ORCHESTRATOR INITIALIZED")
        logger.info(f"Capital: ${starting_capital:,.0f}")
        logger.info(f"Paper Mode: {paper_mode}")
        logger.info(f"Bots Loaded: {len(self.bots)}")
        logger.info(f"DoNothing Filter: {'Enabled' if self.use_do_nothing_filter else 'Disabled'}")
        logger.info(f"AI Veto Layer: {'LIVE' if self.ai_filter else 'Disabled'}")
        logger.info(f"V7 Data Hub: {'Enabled' if self.data_hub else 'Disabled'}")
        logger.info(f"AI Market Analyst: {'Enabled' if self.market_analyst else 'Disabled'}")
        logger.info(f"AI MTF Engine: {'Enabled (parallel)' if self.mtf_engine else 'Disabled'}")
        logger.info(f"AI Signal Injector: {'Enabled' if self.signal_injector else 'Disabled'}")
        logger.info(f"V7 Ensemble: {'Enabled (8 sources)' if self.ensemble else 'Disabled'}")
        logger.info(f"V7 MetaSignal: {'Enabled' if self.meta_signal else 'Disabled'}")
        logger.info(f"V7 ML Engine: {'Enabled' if self.prediction_engine else 'Disabled'}")
        logger.info(f"AI Adaptive Params: {'Enabled' if self.adaptive_params else 'Disabled'}")
        logger.info(f"Risk Management: {'Enabled' if self.enable_risk_management else 'Disabled'}")
        logger.info(f"High Risk Trading: {'ACTIVE' if self.high_risk_enabled else 'Disabled'}")
        logger.info(f"Telegram Commands: {'Enabled' if self.telegram_command_listener else 'Disabled'}")
        logger.info(f"Kalshi Infrastructure: {'Enabled' if self.kalshi_infrastructure else 'Disabled'}")
        logger.info("=" * 70)
    
    def _validate_db_file(self):
        """Verify the configured DB file exists and is not empty (0 bytes)."""
        db_path = self.db.db_path
        if not os.path.exists(db_path):
            logger.warning(f"[DB VALIDATION] Database file does not exist: {db_path}")
            logger.warning("[DB VALIDATION] TradingDB._init_db() should create it on first use.")
        elif os.path.getsize(db_path) == 0:
            logger.warning(f"[DB VALIDATION] Database file is 0 bytes (empty): {db_path}")
            logger.warning("[DB VALIDATION] This may indicate a corrupted or uninitialized DB. "
                           "Re-initializing tables...")
            self.db._init_db()
        else:
            size_kb = os.path.getsize(db_path) / 1024
            logger.info(f"[DB VALIDATION] Database OK: {db_path} ({size_kb:.1f} KB)")

    def _get_alpaca_client(self):
        """Get or create cached AlpacaCryptoClient instance."""
        if self._alpaca_client is None:
            try:
                from bots.alpaca_crypto_client import AlpacaCryptoClient
                self._alpaca_client = AlpacaCryptoClient()
                if not self._alpaca_client._initialized:
                    logger.warning("AlpacaCryptoClient failed to initialize")
                    self._alpaca_client = None
            except ImportError:
                logger.error("Cannot import AlpacaCryptoClient")
        return self._alpaca_client

    def _load_high_risk_config(self) -> bool:
        """Load high-risk trading configuration from config file"""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'high_risk_config.json')
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    enabled = config.get('high_risk_enabled', False)
                    if enabled:
                        logger.info("[HIGH RISK] Aggressive trading ENABLED via dashboard toggle")
                    return enabled
        except Exception as e:
            logger.warning(f"Failed to load high_risk_config: {e}")
        return False

    def _is_aggressive_bot(self, bot_name: str) -> bool:
        """Check if a bot is an aggressive/high-risk bot"""
        aggressive_bots = [
            'Momentum-Scalper', 'Breakout-Hunter', 'Meme-Sniper',
            'RSI-Extremes', 'Multi-Momentum'
        ]
        return bot_name in aggressive_bots

    def _init_alerts(self):
        """Initialize Telegram alerts"""
        try:
            from utils.trading_alerts import TradingAlerts
            self.alerts = TradingAlerts()
            logger.info("[OK] Telegram alerts initialized")
        except Exception as e:
            logger.warning(f"Alerts failed: {e}")

    def _init_telegram_command_listener(self):
        """Initialize Telegram command listener for remote /emergency_stop."""
        try:
            from utils.telegram_bot import TelegramCommandListener

            self.telegram_command_listener = TelegramCommandListener()

            if not self.telegram_command_listener.is_configured:
                logger.warning("Telegram command listener not configured (missing token/chat_id)")
                self.telegram_command_listener = None
                return

            # Register emergency stop: calls our own emergency_stop method
            self.telegram_command_listener.register_emergency_callback(self._telegram_emergency_stop)

            # Register status callback
            self.telegram_command_listener.register_status_callback(self.get_status)

            logger.info("[OK] Telegram command listener initialized")

        except Exception as e:
            logger.warning(f"Telegram command listener init failed: {e}")
            self.telegram_command_listener = None

    def _telegram_emergency_stop(self, reason: str):
        """
        Handle emergency stop triggered via Telegram command.

        Sequence:
        1. If risk_integration is available, use its emergency_stop_all (closes positions)
        2. Stop the orchestrator loop
        """
        logger.critical(f"TELEGRAM EMERGENCY STOP: {reason}")

        # Step 1: Close all positions via risk management
        if self.risk_integration:
            try:
                self.risk_integration.emergency_stop_all(reason)
                logger.critical("Risk integration emergency stop completed")
            except Exception as e:
                logger.error(f"Risk integration emergency stop failed: {e}")

        # Step 2: Stop the orchestrator main loop
        self.running = False

        # Step 3: Send alert via trading alerts
        if self.alerts:
            self.alerts.send(
                f"\xf0\x9f\x9a\xa8 EMERGENCY STOP via Telegram\n"
                f"Reason: {reason}\n"
                f"All bots disabled. Orchestrator halted."
            )

    def _init_ai_filter(self, dry_run: bool = False):
        """Initialize AI Veto Filter"""
        try:
            from filters.ai_filter import get_ai_filter
            self.ai_filter = get_ai_filter(enabled=self.use_ai_filter, dry_run=dry_run)
            logger.info(f"[OK] AI Veto Layer initialized (dry_run={dry_run})")
        except Exception as e:
            logger.warning(f"AI Filter init failed: {e}")
            self.ai_filter = None

    def _init_data_hub(self):
        """V7: Initialize central data aggregator"""
        try:
            from ai.data_hub import DataHub
            self.data_hub = DataHub()
            logger.info("[OK] V7 Data Hub initialized (F&G, FRED, FedWatch, cross-asset, funding rates, on-chain)")
        except Exception as e:
            logger.warning(f"V7 Data Hub init failed: {e}")
            self.data_hub = None

    def _init_market_analyst(self):
        """Initialize AI Market Analyst for predictions and analysis"""
        try:
            from ai.market_analyst import AIMarketAnalyst
            llm = self.ai_filter.llm_client if self.ai_filter else None
            self.market_analyst = AIMarketAnalyst(llm_client=llm, data_hub=self.data_hub)
            logger.info("[OK] AI Market Analyst initialized (predictions + analysis + data_hub)")
        except Exception as e:
            logger.warning(f"AI Market Analyst init failed: {e}")
            self.market_analyst = None

    def _init_v6plus_ai(self):
        """Initialize V6+/V7 AI components: MTF, Signal Injector, Ensemble, Adaptive Params, MetaSignal, ML"""
        # Share LLM client across all components (cost optimization)
        llm = None
        if self.ai_filter and hasattr(self.ai_filter, 'llm_client'):
            llm = self.ai_filter.llm_client
        elif self.market_analyst and hasattr(self.market_analyst, 'llm'):
            llm = self.market_analyst.llm

        # 1. Multi-Timeframe Engine (V7: parallelized with ThreadPoolExecutor)
        try:
            from ai.multi_timeframe import MultiTimeframeEngine
            self.mtf_engine = MultiTimeframeEngine(llm_client=llm)
            logger.info("[OK] V7 Multi-Timeframe Engine initialized (parallel 1h/4h/24h confluence)")
        except Exception as e:
            logger.warning(f"V6+ MTF Engine init failed: {e}")
            self.mtf_engine = None

        # V7 2. Meta-Signal Aggregator (bot consensus)
        try:
            from ai.meta_signal import MetaSignalAggregator
            self.meta_signal = MetaSignalAggregator()
            logger.info("[OK] V7 MetaSignal Aggregator initialized (bot consensus detection)")
        except Exception as e:
            logger.warning(f"V7 MetaSignal init failed: {e}")
            self.meta_signal = None

        # V7 3. ML Prediction Engine
        try:
            from ml_models.inference.prediction_engine import PredictionEngine
            self.prediction_engine = PredictionEngine(max_workers=2, enable_caching=True)
            logger.info("[OK] V7 ML PredictionEngine initialized (LSTM + volatility models)")
        except Exception as e:
            logger.warning(f"V7 PredictionEngine init failed: {e}")
            self.prediction_engine = None

        # 4. Ensemble Combiner (V7: 8 sources with data_hub, ML, bot consensus)
        try:
            from ai.ensemble import EnsembleCombiner
            self.ensemble = EnsembleCombiner(
                market_analyst=self.market_analyst,
                mtf_engine=self.mtf_engine,
                data_hub=self.data_hub,
                prediction_engine=self.prediction_engine,
                meta_signal=self.meta_signal
            )
            logger.info("[OK] V7 Ensemble Combiner initialized (8-source weighted fusion + auto-calibration)")
        except Exception as e:
            logger.warning(f"V7 Ensemble init failed: {e}")
            self.ensemble = None

        # 5. Signal Injector
        try:
            from ai.signal_injector import SignalInjector
            self.signal_injector = SignalInjector(
                market_analyst=self.market_analyst,
                mtf_engine=self.mtf_engine
            )
            logger.info("[OK] V6+ Signal Injector initialized (AI-driven signal modification)")
        except Exception as e:
            logger.warning(f"V6+ Signal Injector init failed: {e}")
            self.signal_injector = None

        # 6. Adaptive Parameters
        try:
            from ai.adaptive_params import AdaptiveParams
            self.adaptive_params = AdaptiveParams(
                market_analyst=self.market_analyst,
                ensemble=self.ensemble
            )
            logger.info("[OK] V6+ Adaptive Params initialized (regime-based auto-tuning)")
        except Exception as e:
            logger.warning(f"V6+ Adaptive Params init failed: {e}")
            self.adaptive_params = None

    def _run_market_analysis(self):
        """Run periodic AI market analysis (called every 30 min by schedule)"""
        if not self.market_analyst:
            return

        try:
            # V7: Refresh data hub before analysis
            if self.data_hub:
                try:
                    self.data_hub.refresh()
                except Exception as e:
                    logger.warning(f"Data hub refresh failed: {e}")

            # V7: Collect bot signals into meta-signal aggregator
            if self.meta_signal:
                try:
                    self._collect_bot_signals_for_meta()
                except Exception as e:
                    logger.debug(f"Meta-signal collection failed: {e}")

            # Build portfolio state from current data
            today = self.db.get_today_summary()
            open_trades = self.db.get_open_trades()
            trades_count = today.get('trades', 0)
            wins_count = today.get('wins', 0)
            portfolio_state = {
                'total_capital': self.starting_capital,
                'current_pnl': today.get('total_pnl', 0),
                'open_positions': len(open_trades),
                'trades_today': trades_count,
                'win_rate': (wins_count / trades_count * 100) if trades_count > 0 else 0,
                'active_bots': len([s for s in self.bots.values() if s.status != BotStatus.ERROR]),
            }

            analysis = self.market_analyst.run_analysis(portfolio_state)

            # V7: Run ensemble accuracy feedback loop
            if self.ensemble and analysis:
                try:
                    self._run_ensemble_accuracy_feedback(analysis)
                except Exception as e:
                    logger.debug(f"Ensemble accuracy feedback failed: {e}")

            # V6+: Run MTF analysis (V7: parallelized)
            mtf_summary = ""
            if self.mtf_engine:
                try:
                    confluence_signals = self.mtf_engine.run_multi_timeframe_analysis()
                    if confluence_signals:
                        strong = [(s, c) for s, c in confluence_signals.items() if c.strength == 'strong']
                        moderate = [(s, c) for s, c in confluence_signals.items() if c.strength == 'moderate']
                        if strong or moderate:
                            mtf_summary = "\nMTF Confluence:"
                            for sym, conf in (strong + moderate)[:5]:
                                arrow = "↑" if conf.direction == "bullish" else "↓"
                                mtf_summary += f"\n  {arrow} {sym}: {conf.direction} ({conf.strength}) [{','.join(conf.agreeing_timeframes)}]"
                except Exception as e:
                    logger.warning(f"V6+ MTF analysis failed: {e}")

            # V7: Run ensemble predictions (now with 8 sources)
            ensemble_summary = ""
            if self.ensemble:
                try:
                    ensemble_preds = self.ensemble.predict_all()
                    high_conf = [(s, p) for s, p in ensemble_preds.items() if p.confidence >= 60]
                    if high_conf:
                        active_sources = self.ensemble.get_stats().get('active_sources', 0)
                        ensemble_summary = f"\nEnsemble ({active_sources} sources, 60%+):"
                        for sym, pred in sorted(high_conf, key=lambda x: -x[1].confidence)[:5]:
                            arrow = "↑" if pred.direction == "bullish" else "↓"
                            ensemble_summary += f"\n  {arrow} {sym}: {pred.direction} ({pred.confidence}%) size={pred.position_size_pct:.0%}"
                except Exception as e:
                    logger.warning(f"V6+ Ensemble prediction failed: {e}")

            if analysis and self.alerts:
                # Send analysis summary to Telegram
                pred_summary = ""
                for p in analysis.predictions[:5]:
                    arrow = "↑" if p.direction == "bullish" else "↓" if p.direction == "bearish" else "→"
                    pred_summary += f"  {arrow} {p.symbol}: {p.direction} ({p.confidence}%)\n"

                conviction = analysis.conviction_trade
                conv_line = ""
                if conviction.get('direction', 'none') != 'none':
                    conv_line = f"\nConviction: {conviction.get('direction', '').upper()} {conviction.get('symbol', '')}\n{conviction.get('reasoning', '')}"

                self.alerts.send(
                    f"[AI ANALYST] Market Report\n"
                    f"Regime: {analysis.market_regime} | Risk: {analysis.risk_level}\n"
                    f"\nPredictions:\n{pred_summary}"
                    f"{conv_line}"
                    f"{mtf_summary}"
                    f"{ensemble_summary}\n"
                    f"\n{analysis.summary}"
                )

        except Exception as e:
            logger.error(f"Market analysis failed: {e}")

    def _collect_bot_signals_for_meta(self):
        """V7: Collect latest bot signals into MetaSignal aggregator for consensus detection."""
        if not self.meta_signal:
            return

        for bot_name, bot_state in self.bots.items():
            if bot_state.status == BotStatus.ERROR:
                continue

            # Get latest signal from each bot (if it has one cached)
            try:
                bot = bot_state.instance if hasattr(bot_state, 'instance') else None
                if bot and hasattr(bot, 'last_signal') and bot.last_signal:
                    sig = bot.last_signal
                    symbol = sig.get('symbol', '')
                    if symbol:
                        direction = sig.get('direction', sig.get('action', 'neutral'))
                        if direction in ('buy', 'long'):
                            direction = 'bullish'
                        elif direction in ('sell', 'short'):
                            direction = 'bearish'
                        confidence = sig.get('strategy_confidence', sig.get('confidence', 0.5))
                        if isinstance(confidence, (int, float)):
                            self.meta_signal.record_signal(bot_name, symbol, direction, confidence)
            except Exception:
                pass

    def _run_ensemble_accuracy_feedback(self, analysis):
        """V7: Check predictions from 4+ hours ago and feed accuracy back to ensemble."""
        if not self.ensemble:
            return

        try:
            import yfinance as yf
            import sqlite3

            # Check analyst predictions
            if self.market_analyst:
                analyst_db = self.market_analyst.db_path
                conn = sqlite3.connect(analyst_db)
                cursor = conn.cursor()

                from datetime import timedelta
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()

                cursor.execute('''
                    SELECT symbol, direction, price_at_prediction, key_factors
                    FROM predictions
                    WHERE checked = 0 AND timestamp < ?
                    LIMIT 20
                ''', (cutoff,))

                rows = cursor.fetchall()
                for row in rows:
                    symbol, direction, price_at, _ = row
                    try:
                        ticker = symbol.replace('/', '-')
                        stock = yf.Ticker(ticker)
                        hist = stock.history(period="1d", interval="1h")
                        if len(hist) > 0:
                            current_price = float(hist['Close'].iloc[-1])
                            change_pct = ((current_price - price_at) / price_at) * 100

                            # V7: 1.0% threshold for crypto
                            if change_pct > 1.0:
                                actual = 'bullish'
                            elif change_pct < -1.0:
                                actual = 'bearish'
                            else:
                                actual = 'neutral'

                            was_correct = (direction == actual) or \
                                         (direction == 'neutral' and abs(change_pct) < 1.5)

                            # Feed back to ensemble for each source
                            self.ensemble.update_accuracy(symbol, 'analyst', was_correct)

                    except Exception:
                        pass

                conn.close()

        except Exception as e:
            logger.debug(f"Ensemble accuracy feedback error: {e}")

    def _run_adaptive_params(self):
        """Run adaptive parameter adjustment (called every 15 min by schedule)"""
        if not self.adaptive_params:
            return
        try:
            adjustments = self.adaptive_params.adjust_bot_params(self.bots)
            if adjustments and self.alerts:
                regime = self.adaptive_params.get_current_regime()
                adj_lines = [f"  {a.bot_name}: {a.param_name} {a.original_value:.4f} -> {a.adjusted_value:.4f}"
                             for a in adjustments[:8]]
                self.alerts.send(
                    f"[ADAPTIVE] Params adjusted for regime: {regime}\n"
                    + "\n".join(adj_lines)
                )
        except Exception as e:
            logger.error(f"Adaptive params failed: {e}")

    def _init_risk_management(self):
        """Initialize Advanced Risk Management System"""
        if not self.enable_risk_management:
            logger.info("Risk management disabled")
            return

        try:
            # Initialize risk management with portfolio value
            from risk_management.config.risk_config import RiskManagementConfig

            config = RiskManagementConfig()
            config.portfolio_value = self.starting_capital

            self.risk_integration = initialize_risk_management(config)

            # Register alert callback
            if self.alerts:
                self.risk_integration.add_alert_callback(self._handle_risk_alert)

            # Register trade callback
            self.risk_integration.add_trade_callback(self._handle_risk_trade)

            logger.info("[OK] Advanced Risk Management initialized")

        except Exception as e:
            logger.warning(f"Risk Management init failed: {e}")
            self.enable_risk_management = False
            self.risk_integration = None
    
    def _handle_risk_alert(self, message: str):
        """Handle risk management alerts"""
        logger.warning(f"RISK MANAGEMENT ALERT: {message}")

        # Send Telegram alert
        if self.alerts:
            self.alerts.send(f"⚠️ RISK: {message}")

    def _handle_risk_trade(self, trade_request, executed_size: float, executed_price: float, success: bool):
        """Handle risk management trade notifications"""
        if success:
            logger.info(f"Risk-managed trade executed: {trade_request.bot_name} "
                       f"{trade_request.action} {trade_request.symbol} "
                       f"${executed_size:,.0f} @ ${executed_price:.2f}")
        else:
            logger.warning(f"Risk-managed trade failed: {trade_request.bot_name} "
                          f"{trade_request.action} {trade_request.symbol}")

    def _init_bots(self):
        """Initialize all bots from registry"""
        # V4 ACTIVE BOTS: Momentum crypto + Kalshi prediction markets
        V4_ACTIVE_BOTS = [
            # Crypto momentum bots
            'Momentum-Scalper',
            'Multi-Momentum',
            'RSI-Extremes',
            # Kalshi prediction market bots
            'Kalshi-Fed',
            'Kalshi-Probability-Arb',
            'Kalshi-Sum-To-One',
            'Kalshi-Market-Maker',
            'Kalshi-Cross-Platform',
            # V6: AI-powered bots
            'Alpaca-Crypto-Momentum',
            'AI-Crypto-Analyzer',
            # V7: ML prediction (crypto only)
            'ML-Prediction-Bot',
            # Event trading — DISABLED: 0/49 win rate (-$380), needs rebuild
            # 'Event-Edge',
            # V8: Fleet system — 19 sub-bots (Kalshi/crypto/forex/prediction)
            'Fleet-Orchestrator',
        ]

        active_configs = [c for c in BOT_REGISTRY if c.name in V4_ACTIVE_BOTS]

        # Step 1: Import modules serially (fast ~0.4s, avoids Python import lock deadlocks)
        bot_classes = {}
        for config in active_configs:
            try:
                module = __import__(config.module_path, fromlist=[config.class_name])
                bot_classes[config.name] = (config, getattr(module, config.class_name))
            except Exception as e:
                self.bots[config.name] = BotState(
                    config=config, instance=None,
                    status=BotStatus.ERROR, error=str(e)
                )
                logger.error(f"[FAIL] {config.name}: {e}")

        # Step 2: Instantiate constructors in parallel (I/O: Kalshi auth, Alpaca connect)
        def _instantiate(name, config, bot_class):
            try:
                try:
                    return config, bot_class(paper_mode=self.paper_mode), None
                except TypeError:
                    return config, bot_class(), None
            except Exception as e:
                return config, None, str(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(bot_classes)) as pool:
            futures = {
                pool.submit(_instantiate, name, cfg, cls): name
                for name, (cfg, cls) in bot_classes.items()
            }
            for future in concurrent.futures.as_completed(futures):
                config, bot_instance, error = future.result()

                if error:
                    self.bots[config.name] = BotState(
                        config=config, instance=None,
                        status=BotStatus.ERROR, error=error
                    )
                    logger.error(f"[FAIL] {config.name}: {error}")
                    continue

                self.bots[config.name] = BotState(
                    config=config,
                    instance=bot_instance,
                    status=BotStatus.WAITING if bot_instance else BotStatus.ERROR,
                    error=None if bot_instance else "Failed to load"
                )

                if bot_instance:
                    logger.info(f"[OK] {config.name} ({config.market.value})")

                    # Register bot with risk management
                    if self.risk_integration:
                        strategy_type = self._determine_strategy_type(config.name)
                        self.risk_integration.register_bot(
                            bot_name=config.name,
                            strategy_type=strategy_type,
                            max_position_size=config.allocation_pct * self.starting_capital
                        )
                        logger.debug(f"Registered {config.name} with risk management")
                else:
                    logger.warning(f"[WARN] {config.name} - instance is None")

        # Step 3: Initialize shared Kalshi infrastructure and inject into Kalshi bots
        self._init_kalshi_infrastructure()

    def _init_kalshi_infrastructure(self):
        """Initialize shared Kalshi infrastructure and inject into all Kalshi bots."""
        # Find any Kalshi bot with a working client to share
        kalshi_client = None
        kalshi_bot_names = []

        for name, state in self.bots.items():
            if state.instance and hasattr(state.instance, 'client') and state.config.market == Market.PREDICTION:
                if hasattr(state.instance.client, '_initialized') and state.instance.client._initialized:
                    kalshi_client = state.instance.client
                    kalshi_bot_names.append(name)
                elif not kalshi_client and hasattr(state.instance, 'client') and state.instance.client:
                    kalshi_client = state.instance.client
                    kalshi_bot_names.append(name)
                else:
                    kalshi_bot_names.append(name)

        if not kalshi_client:
            logger.info("[Kalshi] No authenticated Kalshi client found — infrastructure skipped")
            return

        try:
            from bots.kalshi_infrastructure import KalshiInfrastructure
            self.kalshi_infrastructure = KalshiInfrastructure(
                client=kalshi_client,
                config={
                    'daily_loss_limit_cents': 2000,   # $20
                    'max_open_positions': 12,
                    'max_contracts_per_market': 25,
                }
            )

            # Startup reconciliation: sync positions from Kalshi API
            self.kalshi_infrastructure.startup_reconciliation()

            # Inject infrastructure into all Kalshi bots
            for name in kalshi_bot_names:
                state = self.bots.get(name)
                if state and state.instance:
                    state.instance.infrastructure = self.kalshi_infrastructure
                    # Share the same client to avoid duplicate auth
                    if hasattr(state.instance, 'client'):
                        state.instance.client = kalshi_client
                        state.instance._connected = True
                    logger.debug(f"[Kalshi] Injected infrastructure into {name}")

            logger.info(f"[Kalshi] Infrastructure initialized for {len(kalshi_bot_names)} bots: {kalshi_bot_names}")

        except Exception as e:
            logger.error(f"[Kalshi] Infrastructure init failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _kalshi_poll_fills(self):
        """Background task: poll Kalshi fills every 60s."""
        if self.kalshi_infrastructure:
            try:
                new_fills = self.kalshi_infrastructure.poll_fills()
                if new_fills:
                    logger.info(f"[Kalshi] Fill poll: {len(new_fills)} new fills")
            except Exception as e:
                logger.error(f"[Kalshi] Fill poll error: {e}")

    def _kalshi_collect_settlements(self):
        """Background task: collect Kalshi settlements every 30 min."""
        if self.kalshi_infrastructure:
            try:
                stats = self.kalshi_infrastructure.collect_settlements()
                if stats.get('settled', 0) > 0:
                    logger.info(f"[Kalshi] Settlement collection: {stats}")
            except Exception as e:
                logger.error(f"[Kalshi] Settlement collection error: {e}")

    def _check_event_exits(self):
        """Background task: check event bot positions for exit conditions every 60s."""
        state = self.bots.get('Event-Edge')
        if state and state.instance and hasattr(state.instance, 'check_exits'):
            try:
                state.instance.check_exits()
            except Exception as e:
                logger.error(f"[Event-Edge] Exit check error: {e}")

    def _check_fleet_exits(self):
        """Background task: check fleet bot positions for exit conditions every 60s."""
        state = self.bots.get('Fleet-Orchestrator')
        if state and state.instance and hasattr(state.instance, 'check_exits'):
            try:
                exits = state.instance.check_exits()
                if exits:
                    logger.info(f"[Fleet] Closed {len(exits)} positions")
            except Exception as e:
                logger.error(f"[Fleet] Exit check error: {e}")

    def _determine_strategy_type(self, bot_name: str) -> str:
        """Determine strategy type based on bot name"""
        strategy_map = {
            'RSI2-MeanReversion': 'mean_reversion',
            'CumulativeRSI': 'mean_reversion',
            'MACD-RSI-Combo': 'technical',
            'BollingerSqueeze': 'volatility',
            'MTF-RSI': 'mean_reversion',
            'DualMomentum': 'momentum',
            'SectorRotation': 'momentum',
            'Kalshi-Fed': 'arbitrage',
            'Kalshi-Probability-Arb': 'arbitrage',
            'Kalshi-Sum-To-One': 'arbitrage',
            'Kalshi-Market-Maker': 'market_making',
            'Kalshi-Cross-Platform': 'arbitrage',
            'Weather-Edge': 'arbitrage',
            'Sports-Edge': 'arbitrage',
            'Sports-AI': 'ai_prediction',
            'BoxOffice-Edge': 'arbitrage',
            'Awards-Edge': 'arbitrage',
            'Climate-Edge': 'arbitrage',
            'Economic-Edge': 'arbitrage',
            'Cross-Market-Arbitrage': 'arbitrage',
            'OANDA-Forex': 'forex',
            'London-Breakout': 'forex',
            'FundingRate-Arb': 'arbitrage',
            'Crypto-Arb': 'arbitrage',
            'Kalshi-Hourly-Crypto': 'arbitrage',
            'Alpaca-Crypto-Momentum': 'momentum',
            'Earnings-PEAD': 'earnings',
            'Sentiment-Bot': 'sentiment',
            'FOMC-Trader': 'event_driven',
            'Market-Scanner': 'scanner',
            'Coinbase-Arb': 'arbitrage',
            'Colorado-Sports-Arb': 'arbitrage',
            'Fidelity-Alerts': 'scanner',
            'Computer-Vision-Bot': 'ai_prediction',
            'ML-Prediction-Bot': 'ai_prediction',
            'Event-Edge': 'edge_detection'
        }

        return strategy_map.get(bot_name, 'unknown')

    def _evaluate_trade_with_risk_management(self, bot_name: str, signal: Dict):
        """Evaluate trade through risk management system"""
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            action = signal.get('action', 'unknown')
            size = signal.get('quantity', signal.get('shares', 1))
            price = signal.get('price', 0.0)
            confidence = signal.get('confidence', signal.get('ai_confidence', 1.0))

            # Convert size to dollar amount if needed
            if isinstance(size, (int, float)) and size > 0:
                if price > 0:
                    dollar_size = abs(size * price)
                else:
                    # Assume size is already in dollars
                    dollar_size = abs(size)
            else:
                dollar_size = 1000  # Default position size

            return self.risk_integration.request_trade(
                bot_name=bot_name,
                symbol=symbol,
                action=action,
                size=dollar_size,
                price=price,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Risk evaluation failed for {bot_name}: {e}")
            # Return a rejection response
            from risk_management.integration.trading_integration import TradeResponse
            return TradeResponse(
                request_id=f"ERROR_{int(time.time())}",
                approved=False,
                recommended_size=0.0,
                rejection_reasons=[f"Risk evaluation error: {str(e)}"],
                risk_adjustments={},
                risk_score=100.0,
                max_allowed_size=0.0
            )

    def _apply_ai_filter(self, signal: Dict, bot_name: str, market: Market) -> Optional[Dict]:
        """
        Apply AI confidence scoring to a signal and adjust position size.

        Args:
            signal: Trade signal dictionary
            bot_name: Name of the bot generating the signal
            market: Market type (for quantity rounding)

        Returns:
            Dict with veto info if rejected, None if approved
        """
        if not self.ai_filter:
            return None

        context = self._build_ai_context()
        filter_result = self.ai_filter.evaluate(signal, context)

        # CONFIDENCE-BASED: Only block if confidence < 0.15 (15% threshold)
        if filter_result.confidence < 0.15:
            logger.info(f"AI REJECT: {bot_name} {signal.get('symbol')} - confidence {filter_result.confidence:.2f} below 0.15 - {filter_result.reasoning}")
            return {
                'ai_vetoed': True,
                'ai_reasoning': filter_result.reasoning,
                'ai_confidence': filter_result.confidence
            }

        # Adjust position size based on AI confidence (continuous scaling)
        original_qty = signal.get('quantity', signal.get('shares', 1))
        signal['original_quantity'] = original_qty
        raw_qty = original_qty * filter_result.size_multiplier
        if market == Market.CRYPTO:
            new_qty = max(0.001, round(raw_qty, 6))
        else:
            new_qty = max(1, int(raw_qty))
        signal['quantity'] = new_qty
        if 'shares' in signal:
            signal['shares'] = new_qty
        signal['ai_size_adjusted'] = True if filter_result.size_multiplier < 1.0 else False
        signal['ai_confidence'] = filter_result.confidence
        if filter_result.size_multiplier < 1.0:
            logger.info(f"AI CONFIDENCE: {bot_name} {original_qty} -> {new_qty} (conf={filter_result.confidence:.0%}, mult={filter_result.size_multiplier:.2f})")
        else:
            logger.info(f"AI APPROVE: {bot_name} full size (conf={filter_result.confidence:.0%})")

        return None  # No veto

    def _apply_risk_check(self, signal: Dict, bot_name: str) -> Optional[Dict]:
        """
        Apply risk management checks to a signal and adjust position size.

        Args:
            signal: Trade signal dictionary
            bot_name: Name of the bot generating the signal

        Returns:
            Dict with veto info if rejected, None if approved
        """
        if not self.risk_integration:
            return None

        trade_response = self._evaluate_trade_with_risk_management(bot_name, signal)

        if not trade_response.approved:
            # Log detailed heat breakdown so we always know WHY trades are rejected
            heat_detail = ""
            if self.risk_integration and hasattr(self.risk_integration, 'risk_manager'):
                rm = self.risk_integration.risk_manager
                if hasattr(rm, 'last_assessment') and rm.last_assessment:
                    la = rm.last_assessment
                    heat_detail = (f" heat={la.heat_score:.0f} "
                                   f"drawdown={la.current_drawdown:.1%} "
                                   f"emergency={getattr(rm, 'emergency_mode', False)}")
            logger.warning(f"RISK VETO: {bot_name} {signal.get('symbol')}{heat_detail} - {trade_response.rejection_reasons}")
            return {
                'risk_vetoed': True,
                'risk_reasons': trade_response.rejection_reasons,
                'risk_score': trade_response.risk_score
            }

        # Adjust size based on risk management
        # Convert recommended_size (dollars) to shares (supports fractional)
        price = signal.get('price', 0)
        if price > 0:
            # Use fractional shares, rounded to 3 decimal places
            recommended_shares = round(trade_response.recommended_size / price, 3)
            # Minimum 0.001 shares (Alpaca minimum)
            recommended_shares = max(0.001, recommended_shares)
        else:
            recommended_shares = 1  # Fallback if no price

        original_qty = signal.get('quantity', signal.get('shares', 1))
        if recommended_shares != original_qty:
            signal['original_quantity'] = original_qty
            signal['quantity'] = recommended_shares
            if 'shares' in signal:
                signal['shares'] = recommended_shares
            signal['risk_size_adjusted'] = True
            signal['risk_score'] = trade_response.risk_score
            signal['risk_dollar_size'] = trade_response.recommended_size  # Store original dollar amount
            logger.info(f"RISK SIZE: {bot_name} ${trade_response.recommended_size:.0f} -> {recommended_shares:.3f} shares @ ${price:.2f} (Risk: {trade_response.risk_score:.0f})")

        # Store request ID for execution reporting
        signal['risk_request_id'] = trade_response.request_id

        return None  # No veto

    def _load_bot(self, config: BotConfig) -> Optional[Any]:
        """Load a bot module and create instance"""
        try:
            # Import module
            module = __import__(config.module_path, fromlist=[config.class_name])
            bot_class = getattr(module, config.class_name)

            # Create instance with paper mode
            try:
                instance = bot_class(paper_mode=self.paper_mode)
            except TypeError:
                # Some bots don't take paper_mode
                instance = bot_class()

            # Verify instance has callable methods
            logger.debug(f"[{config.name}] Loaded, available methods: {[m for m in dir(instance) if not m.startswith('_') and callable(getattr(instance, m, None))]}")
            return instance

        except ImportError as e:
            logger.warning(f"[{config.name}] Import failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"[{config.name}] Load failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _fetch_market_data(self, symbol: str = "SPY", period: str = "60d") -> Optional[Any]:
        """Fetch market data for stock strategies"""
        try:
            import yfinance as yf
            df = yf.download(symbol, period=period, progress=False)
            if len(df) > 0:
                return df
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch market data for {symbol}: {e}")
            return None

    def _build_ai_context(self) -> Dict[str, Any]:
        """Build enriched market context for AI veto evaluation.
        V6: includes market analyst predictions, portfolio state, news sentiment."""
        context = {
            'time_of_day': datetime.now().strftime('%H:%M'),
            'day_of_week': datetime.now().strftime('%A'),
            'market_regime': 'unknown',
            'vix': 20.0,
            'spy_trend': 'unknown'
        }

        # V6: Use market analyst data if available (avoids redundant yfinance calls)
        if self.market_analyst and self.market_analyst.latest_analysis:
            analysis = self.market_analyst.latest_analysis
            context['market_regime'] = analysis.market_regime
            context['risk_level'] = analysis.risk_level
            context['key_risks'] = ', '.join(analysis.key_risks[:3]) if analysis.key_risks else 'none'
            context['ai_summary'] = analysis.summary[:150]

            # Populate VIX and SPY from analyst cache (already fetched)
            # Still try yfinance as fallback below

        try:
            import yfinance as yf

            # Get VIX
            try:
                vix = yf.Ticker("^VIX")
                vix_hist = vix.history(period="1d")
                if len(vix_hist) > 0:
                    context['vix'] = float(vix_hist['Close'].iloc[-1])
            except Exception as e:
                logger.debug(f"Failed to fetch VIX: {e}")

            # Get SPY trend and regime (only if analyst didn't provide)
            if context['market_regime'] == 'unknown':
                try:
                    spy = yf.Ticker("SPY")
                    hist = spy.history(period="5d")
                    if len(hist) > 0:
                        trend = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
                        if trend > 0.01:
                            context['spy_trend'] = 'up'
                        elif trend < -0.01:
                            context['spy_trend'] = 'down'
                        else:
                            context['spy_trend'] = 'sideways'

                        returns = hist['Close'].pct_change().dropna()
                        volatility = returns.std() * (252 ** 0.5)

                        if volatility > 0.25 or context['vix'] > 25:
                            context['market_regime'] = 'volatile'
                        elif trend > 0.02:
                            context['market_regime'] = 'bullish'
                        elif trend < -0.02:
                            context['market_regime'] = 'bearish'
                        else:
                            context['market_regime'] = 'neutral'
                except Exception as e:
                    logger.debug(f"Failed to fetch SPY trend: {e}")

        except Exception as e:
            logger.debug(f"Failed to build AI context: {e}")

        # V6: Add portfolio context
        try:
            today = self.db.get_today_summary()
            context['portfolio_pnl'] = today.get('total_pnl', 0)
            context['trades_today'] = today.get('total_trades', 0)
            context['open_positions'] = today.get('open_positions', 0)
        except Exception:
            pass

        return context

    def _serialize_result(self, result: Any) -> Any:
        """Convert dataclass/enum results to JSON-serializable dict"""
        if result is None:
            return None

        # Handle lists
        if isinstance(result, list):
            return [self._serialize_result(item) for item in result]

        # Handle dataclasses using built-in function (more reliable)
        if is_dataclass(result) and not isinstance(result, type):
            try:
                return asdict(result)
            except Exception:
                # Fallback: manual conversion
                return {k: self._serialize_result(v) for k, v in result.__dict__.items()}

        # Handle enums
        if hasattr(result, 'value') and hasattr(result, 'name'):
            return result.value

        # Handle datetime
        if isinstance(result, datetime):
            return result.isoformat()

        # Handle dicts
        if isinstance(result, dict):
            return {k: self._serialize_result(v) for k, v in result.items()}

        return result

    def _run_bot_threaded(self, bot_name: str) -> None:
        """Submit bot run to thread pool so schedule doesn't block."""
        with self._bot_lock:
            # Skip if this bot is already running in a thread
            future = self._bot_futures.get(bot_name)
            if future and not future.done():
                logger.debug(f"[{bot_name}] Still running from previous cycle, skipping")
                return

        future = self._bot_executor.submit(self._run_bot, bot_name)
        with self._bot_lock:
            self._bot_futures[bot_name] = future

        def _on_done(f, name=bot_name):
            try:
                f.result()  # propagate exceptions to log
            except Exception as e:
                logger.error(f"[{name}] Thread error: {e}")
            # Clean up completed future to prevent memory leak
            with self._bot_lock:
                if self._bot_futures.get(name) is f:
                    del self._bot_futures[name]

        future.add_done_callback(_on_done)

    def _run_bot(self, bot_name: str) -> Optional[Dict]:
        """Run a single bot and return results"""
        if bot_name not in self.bots:
            return None

        state = self.bots[bot_name]
        logger.info(f"Attempting to run bot: {bot_name} (market={state.config.market.value})")

        # Check DoNothingFilter for stock/forex bots only
        # Crypto and prediction markets run 24/7 (including weekends)
        if (self.use_do_nothing_filter and self.do_nothing_filter and
            state.config.market in [Market.STOCKS, Market.FOREX]):
            df = self._fetch_market_data('SPY')
            result = self.do_nothing_filter.analyze(df)
            if result.do_nothing:
                logger.info(f"DoNothingFilter: Skipping {bot_name} - {result.reasoning}")
                return {'status': 'skipped', 'reason': 'do_nothing_filter', 'detail': result.reasoning}

        if state.status == BotStatus.ERROR or state.instance is None:
            return {'status': 'error', 'error': state.error}

        try:
            with state.lock:
                state.status = BotStatus.RUNNING
                state.last_run = datetime.now()

            result = None

            # Stock strategies may need DataFrame passed to generate_signal
            if state.config.market == Market.STOCKS and hasattr(state.instance, 'generate_signal'):
                # inspect is imported globally at top of file
                method = getattr(state.instance, 'generate_signal')
                sig = inspect.signature(method)

                # Check if method requires a 'df' parameter
                params = list(sig.parameters.keys())
                needs_df = 'df' in params or (len(params) > 0 and params[0] not in ['self'])

                if needs_df:
                    # Fetch market data
                    symbol = getattr(state.instance, 'symbol', 'SPY')
                    df = self._fetch_market_data(symbol)
                    if df is not None:
                        result = method(df)
                    else:
                        result = {'status': 'error', 'error': 'Failed to fetch market data'}
                else:
                    # Strategy fetches its own data
                    result = method()
            else:
                # Try different method names bots might use
                # Order matters: action methods first, status methods last
                method_priority = [
                    'run_strategy',       # AI bots: SportsAIBot, CrossMarketArbitrageBot, etc.
                    'run_scan',           # AlpacaCryptoRSIBot, KalshiHourlyCrypto, MLPredictionBot
                    'scan_all_pairs',     # CryptoArbScanner
                    'scan_all_sports',    # ColoradoSportsArbBot
                    'scan_all_events',    # KalshiSumToOneArbitrage
                    'find_opportunities', # Sports-Edge, Sports-Props, Weather-Edge, etc.
                    'scan_opportunities', # Cross-platform arb bots (Kalshi, Polymarket)
                    'scan_all_categories',# MarketScanner
                    'run',                # Generic run method
                    'scan',               # Generic scan method
                    'scan_all',           # Generic scan_all method
                    'check',              # Generic check method
                    # Note: 'execute' removed - requires args, use scan methods instead
                    'get_status',         # Fallback - just returns status (no action)
                ]

                method_found = None
                # Debug: Log what we're checking
                logger.debug(f"[{bot_name}] Checking methods: {method_priority}")
                logger.debug(f"[{bot_name}] Instance type: {type(state.instance)}")

                for method_name in method_priority:
                    has_method = hasattr(state.instance, method_name)
                    if has_method:
                        # Verify it's actually callable
                        attr = getattr(state.instance, method_name, None)
                        is_callable = callable(attr)
                        logger.debug(f"[{bot_name}] Found '{method_name}': callable={is_callable}")
                        if is_callable:
                            method_found = method_name
                            break
                    else:
                        logger.debug(f"[{bot_name}] No '{method_name}'")

                if method_found:
                    logger.info(f"[{bot_name}] Calling method: {method_found}()")
                    bot_timeout = getattr(state.config, 'scan_timeout', 120) if hasattr(state, 'config') else 120
                    try:
                        method = getattr(state.instance, method_found)
                        result_container = [None]
                        error_container = [None]
                        def _run_method():
                            try:
                                result_container[0] = method()
                            except Exception as e:
                                error_container[0] = e

                        t = threading.Thread(target=_run_method, daemon=True)
                        t.start()
                        t.join(timeout=bot_timeout)
                        if t.is_alive():
                            logger.error(f"[{bot_name}] Method {method_found}() TIMED OUT after {bot_timeout}s")
                            result = []
                        elif error_container[0]:
                            raise error_container[0]
                        else:
                            result = result_container[0]
                        # Handle async methods: if result is a coroutine, run it
                        if inspect.iscoroutine(result):
                            logger.info(f"[{bot_name}] Awaiting async method {method_found}()")
                            result = asyncio.run(result)
                        logger.info(f"[{bot_name}] Method {method_found}() returned: {type(result).__name__}")
                        if result:
                            # Log summary of result
                            if isinstance(result, list):
                                logger.info(f"[{bot_name}] Result: {len(result)} items")
                            elif isinstance(result, dict):
                                logger.info(f"[{bot_name}] Result keys: {list(result.keys())[:5]}")
                    except Exception as method_error:
                        logger.error(f"[{bot_name}] Method {method_found}() FAILED: {method_error}")
                        import traceback
                        logger.error(traceback.format_exc())
                        raise
                else:
                    logger.warning(f"[{bot_name}] No runnable method found! Available: {dir(state.instance)}")
                    result = {'status': 'error', 'error': 'No runnable method found'}

            # Serialize result to handle dataclasses/enums
            result = self._serialize_result(result)

            # V4: Normalize aggressive bot signals (TradeSignal -> orchestrator format)
            if result and isinstance(result, list):
                for i, sig in enumerate(result):
                    if isinstance(sig, dict):
                        # Map 'side' -> 'action' (BUY->buy, SELL->sell)
                        if 'side' in sig and 'action' not in sig:
                            result[i]['action'] = sig['side'].lower()
                        # Map 'entry_price' -> 'price'
                        if 'entry_price' in sig and 'price' not in sig:
                            result[i]['price'] = sig['entry_price']
                        # V4: Use bot's calculated quantity; fallback to $50 position sizing
                        if 'quantity' not in sig and 'position_size_usd' in sig:
                            price = sig.get('price', sig.get('entry_price', 0))
                            if price > 0:
                                result[i]['quantity'] = sig['position_size_usd'] / price
                            else:
                                result[i]['quantity'] = 50.0

            with state.lock:
                state.status = BotStatus.WAITING
                state.last_signal = result

            # Update database
            self.db.update_bot_status(bot_name, {
                'status': 'running',
                'last_run': state.last_run.isoformat(),
                'last_signal': result,
                'trades_today': state.trades_today,
                'pnl_today': state.pnl_today
            })

            # Log any trades (after AI veto check and risk management)
            # Handle list of signals (e.g., from Kalshi bot)
            if result and isinstance(result, list):
                processed_signals = []
                seen_symbols = set()  # Dedup: track symbols already processed in this batch
                for signal in result:
                    if isinstance(signal, dict) and signal.get('action') in ['buy', 'sell', 'long', 'short']:
                        # Dedup: skip duplicate symbols within the same batch
                        sig_symbol = signal.get('symbol', signal.get('ticker', ''))
                        if sig_symbol and sig_symbol in seen_symbols:
                            logger.info(f"DEDUP: Skipping duplicate signal for {sig_symbol} from {bot_name}")
                            continue
                        if sig_symbol:
                            seen_symbols.add(sig_symbol)

                        # V6+: Run through Signal Injector (AI-driven size modification)
                        if self.signal_injector:
                            try:
                                self.signal_injector.apply_injection(signal, bot_name)
                            except Exception as e:
                                logger.debug(f"Signal injection error: {e}")

                        # Run through AI Confidence Scoring (not binary veto)
                        ai_veto = self._apply_ai_filter(signal, bot_name, state.config.market)
                        if ai_veto:
                            signal.update(ai_veto)
                            processed_signals.append(signal)
                            continue  # Skip to next signal

                        # Run through Risk Management Layer
                        risk_veto = self._apply_risk_check(signal, bot_name)
                        if risk_veto:
                            signal.update(risk_veto)
                            processed_signals.append(signal)
                            continue  # Skip to next signal

                        self._log_trade_from_signal(bot_name, signal)
                        processed_signals.append(signal)
                    else:
                        # Non-trade signals (e.g., status updates) - pass through
                        processed_signals.append(signal)
                return processed_signals

            # Handle single dict signal (original logic)
            if result and isinstance(result, dict):
                if result.get('action') in ['buy', 'sell', 'long', 'short']:
                    # V6+: Run through Signal Injector (AI-driven size modification)
                    if self.signal_injector:
                        try:
                            self.signal_injector.apply_injection(result, bot_name)
                        except Exception as e:
                            logger.debug(f"Signal injection error: {e}")

                    # Run through AI Confidence Scoring (not binary veto)
                    market = self.bots[bot_name].config.market if bot_name in self.bots else Market.OTHER
                    ai_veto = self._apply_ai_filter(result, bot_name, market)
                    if ai_veto:
                        result.update(ai_veto)
                        return result

                    # Run through Risk Management Layer
                    risk_veto = self._apply_risk_check(result, bot_name)
                    if risk_veto:
                        result.update(risk_veto)
                        return result

                    self._log_trade_from_signal(bot_name, result)

            return result

        except Exception as e:
            state.status = BotStatus.ERROR
            state.error = str(e)
            logger.error(f"Bot {bot_name} error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _execute_trade_via_bot(self, bot_name: str, signal: Dict) -> Optional[Dict]:
        """
        Execute a trade through the bot's API.

        Different bots have different execution methods:
        - Kalshi: place_order(ticker, side, quantity, price)
        - OANDA: market_order(instrument, units, stop_loss, take_profit)
        - Alpaca Crypto: execute_trade(signal)
        - Generic: execute_trade() or place_order()

        Args:
            bot_name: Name of the bot to execute through
            signal: Trade signal dictionary

        Returns:
            Order result dictionary or None if execution failed
        """
        if bot_name not in self.bots:
            logger.error(f"Bot {bot_name} not found for trade execution")
            return None

        bot_instance = self.bots[bot_name].instance
        if bot_instance is None:
            logger.error(f"Bot {bot_name} has no instance for trade execution")
            return None

        market = self.bots[bot_name].config.market

        try:
            # Kalshi prediction market bots - use place_order()
            if hasattr(bot_instance, 'place_order') and market == Market.PREDICTION:
                ticker = signal.get('ticker', signal.get('symbol', ''))
                side = signal.get('side', 'yes' if signal.get('action') in ['buy', 'long'] else 'no')
                quantity = int(signal.get('quantity', signal.get('shares', 1)))
                # Kalshi prices are in cents (1-99)
                # Check for price_cents first (new format), then price (legacy)
                if 'price_cents' in signal:
                    price = int(signal['price_cents'])
                else:
                    raw_price = signal.get('price', 50)
                    # If price is decimal (0-1), convert to cents
                    if isinstance(raw_price, float) and 0 < raw_price < 1:
                        price = int(raw_price * 100)
                    else:
                        price = int(raw_price)
                # Ensure valid range
                price = max(1, min(99, price))

                logger.info(f"[EXECUTE] {bot_name}: place_order({ticker}, {side}, {quantity}, {price})")
                result = bot_instance.place_order(
                    ticker=ticker,
                    side=side,
                    quantity=quantity,
                    price=price
                )
                return result

            # OANDA forex bots - use market_order()
            elif hasattr(bot_instance, 'market_order') and market == Market.FOREX:
                instrument = signal.get('symbol', signal.get('instrument', signal.get('pair', '')))
                # Units: positive for buy, negative for sell
                units = int(signal.get('quantity', signal.get('units', 1)))
                if signal.get('action') in ['sell', 'short']:
                    units = -abs(units)
                else:
                    units = abs(units)
                stop_loss = signal.get('stop_loss')
                take_profit = signal.get('take_profit')

                logger.info(f"[EXECUTE] {bot_name}: market_order({instrument}, {units}, sl={stop_loss}, tp={take_profit})")
                result = bot_instance.market_order(
                    instrument=instrument,
                    units=units,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                return result

            # Alpaca crypto/stock bots - use execute_trade() with Signal object
            elif hasattr(bot_instance, 'execute_trade'):
                # For stock-market bots, round to whole shares (fractionals unreliable)
                # Note: Market.CRYPTO and Market.OTHER are excluded — crypto uses fractional quantities
                if market == Market.STOCKS:
                    qty = signal.get('quantity', signal.get('shares', 1))
                    if isinstance(qty, float) and qty < 1:
                        signal['quantity'] = max(1, int(round(qty)))
                        if 'shares' in signal:
                            signal['shares'] = signal['quantity']
                        logger.info(f"[ROUND] {bot_name}: fractional {qty} -> {signal['quantity']} whole shares")
                logger.info(f"[EXECUTE] {bot_name}: execute_trade({signal})")
                result = bot_instance.execute_trade(signal)
                return result

            # Generic place_order for other bots
            elif hasattr(bot_instance, 'place_order'):
                symbol = signal.get('symbol', signal.get('ticker', ''))
                side = signal.get('action', signal.get('side', 'buy'))
                quantity = signal.get('quantity', signal.get('shares', 1))
                price = signal.get('price', 0)
                # Round to whole shares for stocks only (crypto uses fractional)
                if market == Market.STOCKS and isinstance(quantity, float) and quantity < 1:
                    quantity = max(1, int(round(quantity)))

                logger.info(f"[EXECUTE] {bot_name}: place_order({symbol}, {side}, {quantity}, {price})")
                result = bot_instance.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price
                )
                return result

            # No execution method found - log warning
            else:
                logger.warning(f"Bot {bot_name} has no execution method (place_order/market_order/execute_trade)")
                logger.warning(f"Available methods: {[m for m in dir(bot_instance) if not m.startswith('_')]}")
                return None

        except Exception as e:
            logger.error(f"Trade execution failed for {bot_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _log_trade_from_signal(self, bot_name: str, signal: Dict):
        """Execute and log a trade from a signal"""

        action = signal.get('action', '').lower()

        # Signal-level tracing for every signal processed
        logger.info(
            f"[SIGNAL] {bot_name} | action={action} | symbol={signal.get('symbol', 'UNKNOWN')} | "
            f"price={signal.get('price', 0)} | pnl={signal.get('pnl', 'N/A')} | "
            f"status={signal.get('status', 'new')} | reason={signal.get('reason', '')[:80]}"
        )

        # GUARD: Reject micro-priced assets (avoids millions-of-shares positions on tokens like SHIB)
        if action in ('buy', 'long'):
            trade_price = float(signal.get('price', 0))
            if 0 < trade_price < 0.01:
                logger.warning(f"MICRO PRICE GUARD: {bot_name} {signal.get('symbol')} price ${trade_price:.8f} < $0.01 — trade REJECTED")
                return

        # GUARD: Reject trades where notional value exceeds 50% of capital
        if action in ('buy', 'long'):
            trade_price = float(signal.get('price', 0))
            trade_qty = float(signal.get('quantity', 0))
            notional = trade_qty * trade_price if trade_price > 0 and trade_qty > 0 else 0
            max_notional = self.starting_capital * 0.50
            if notional > max_notional:
                logger.error(
                    f"NOTIONAL GUARD: {bot_name} {signal.get('symbol')} "
                    f"notional ${notional:,.2f} (qty={trade_qty} x ${trade_price:,.2f}) "
                    f"exceeds 50% of capital (${max_notional:,.2f}) — trade REJECTED"
                )
                return
        symbol = signal.get('symbol', signal.get('ticker', 'UNKNOWN'))
        is_close = action in ('sell', 'close')

        # GUARD: Reject duplicate entries — block if bot already has an open position on this symbol
        if action in ('buy', 'long'):
            open_trades = self.db.get_open_trades()
            existing = [t for t in open_trades if t.get('bot_name') == bot_name and t.get('symbol') == symbol]
            if existing:
                logger.warning(f"DUPLICATE GUARD: {bot_name} already has open position on {symbol} — trade REJECTED")
                return
        already_executed = signal.get('status') in ('filled', 'submitted')

        # V4: Daily loss limit check (3% of capital) - only block new entries, not exits
        if not is_close:
            today = self.db.get_today_summary()
            if today['total_pnl'] <= -(self.starting_capital * 0.03):
                logger.warning(f"DAILY LOSS LIMIT: ${today['total_pnl']:.2f} exceeds ${self.starting_capital * 0.03:.0f} (3%) limit - blocking {bot_name} {symbol}")
                return

        # Circuit breaker: block new entries if bot has too many consecutive losses
        if not is_close:
            cb = self._circuit_breaker.get(bot_name, {})
            paused_until = cb.get('paused_until')
            if paused_until:
                from datetime import datetime as _dt
                if _dt.now() < paused_until:
                    remaining = (paused_until - _dt.now()).total_seconds() / 60
                    logger.warning(
                        f"CIRCUIT BREAKER: {bot_name} paused ({cb.get('consecutive_losses', 0)} consecutive losses). "
                        f"Resumes in {remaining:.0f}min. Blocking {symbol}."
                    )
                    return
                else:
                    # Cooldown expired — reset
                    logger.info(f"CIRCUIT BREAKER: {bot_name} cooldown expired, resuming trading")
                    self._circuit_breaker[bot_name] = {'consecutive_losses': 0, 'paused_until': None}

        # If the bot already executed the trade (e.g. alpaca_crypto_rsi runs
        # execute_trade internally in run_scan), skip re-execution.
        if already_executed:
            execution_result = signal
            execution_success = True
        else:
            # EXECUTE THE TRADE VIA BOT'S API
            execution_result = self._execute_trade_via_bot(bot_name, signal)

            if execution_result is None:
                logger.warning(f"Trade execution returned None for {bot_name} - logging as failed")
                execution_success = False
            elif isinstance(execution_result, dict) and not execution_result.get('success'):
                logger.warning(f"Trade execution returned failure for {bot_name}: {execution_result.get('error', 'Unknown error')}")
                execution_success = False
            elif isinstance(execution_result, dict) and 'error' in execution_result:
                logger.warning(f"Trade execution returned error for {bot_name}: {execution_result['error']}")
                execution_success = False
            else:
                execution_success = True
                logger.info(f"Trade executed successfully for {bot_name}: {execution_result}")

        # Store execution result in signal for tracking
        signal['execution_result'] = execution_result
        signal['execution_success'] = execution_success

        # CLOSE existing open trade if this is a sell/exit signal
        if is_close and execution_success:
            # CRITICAL: Try multiple keys for exit price: price -> exit_price -> fill_price
            exit_price = signal.get('price') or signal.get('exit_price') or signal.get('fill_price')

            # If still no exit price, attempt to fetch live market price
            if not exit_price or exit_price <= 0:
                logger.warning(f"Exit signal for {bot_name} {symbol} missing price, attempting to fetch live price")
                try:
                    bot = self.bots.get(bot_name, {})
                    if hasattr(bot, 'instance'):
                        # Try different price fetching methods
                        if hasattr(bot.instance, 'get_price'):
                            exit_price = bot.instance.get_price(symbol)
                        elif hasattr(bot.instance, 'alpaca') and hasattr(bot.instance.alpaca, 'get_price'):
                            exit_price = bot.instance.alpaca.get_price(symbol)
                        elif hasattr(bot.instance, 'client') and hasattr(bot.instance.client, 'get_latest_price'):
                            exit_price = bot.instance.client.get_latest_price(symbol)

                        if exit_price and exit_price > 0:
                            logger.info(f"Fetched live exit price for {symbol}: ${exit_price:.6f}")
                        else:
                            logger.error(f"CRITICAL: Cannot fetch live exit price for {bot_name} {symbol} - trade exit may be inaccurate")
                            exit_price = 0
                    else:
                        logger.error(f"No bot instance available to fetch price for {bot_name} {symbol}")
                        exit_price = 0
                except Exception as e:
                    logger.error(f"Failed to fetch live exit price for {bot_name} {symbol}: {e}")
                    exit_price = 0

            # Validate exit price
            if not exit_price or exit_price <= 0:
                logger.error(f"Invalid exit_price={exit_price} for {bot_name} {symbol} - cannot close trade accurately")

            # Compute PnL from entry vs exit prices if not provided in signal
            pnl = signal.get('pnl', None)
            if pnl is None:
                # Only recompute when PnL is genuinely missing
                open_trades = self.db.get_open_trades()
                matching = [t for t in open_trades if t.get('bot_name') == bot_name and t.get('symbol') == symbol]
                if matching and exit_price and exit_price > 0:
                    entry_price = float(matching[0].get('entry_price', 0))
                    quantity = float(matching[0].get('quantity', 0))
                    if entry_price > 0 and quantity > 0:
                        side = matching[0].get('side', 'buy')
                        if side in ('short', 'sell_short', 'sell'):
                            pnl = quantity * (entry_price - exit_price)
                        else:
                            pnl = quantity * (exit_price - entry_price)
                        logger.info(f"Computed PnL for {bot_name} {symbol}: side={side}, entry={entry_price}, exit={exit_price}, qty={quantity}, pnl=${pnl:.2f}")
                if pnl is None:
                    pnl = 0

            # Ghost trade detection: warn if no matching open trade exists in DB
            open_trades_check = self.db.get_open_trades()
            matching_open = [t for t in open_trades_check if t.get('bot_name') == bot_name and t.get('symbol') == symbol]
            if not matching_open:
                logger.warning(
                    f"[GHOST TRADE] {bot_name} {symbol}: close signal received but no matching "
                    f"open trade in DB. PnL=${pnl:.2f}, exit_price=${exit_price:.4f}"
                )

            # Estimate fees with market-specific rates
            FEE_RATES = {'prediction': 0.0, 'crypto': 0.003, 'stocks': 0.0, 'forex': 0.0001}
            market_str = self.bots[bot_name].config.market.value if bot_name in self.bots else 'other'
            fee_rate = FEE_RATES.get(market_str, 0.005)
            estimated_fees = abs(pnl) * fee_rate if pnl != 0 else (exit_price * signal.get('quantity', 0) * fee_rate)
            net_pnl = pnl - estimated_fees

            self.db.close_trade(bot_name, symbol, exit_price, pnl,
                                gross_pnl=pnl, net_pnl=net_pnl, estimated_fees=estimated_fees)

            # PnL discrepancy check: compare bot-reported vs DB-computed
            if matching_open:
                db_entry_price = float(matching_open[0].get('entry_price', 0))
                db_quantity = float(matching_open[0].get('quantity', 0))
                if db_entry_price > 0 and db_quantity > 0 and exit_price > 0:
                    db_side = matching_open[0].get('side', 'buy')
                    if db_side in ('short', 'sell_short', 'sell'):
                        db_computed_pnl = db_quantity * (db_entry_price - exit_price)
                    else:
                        db_computed_pnl = db_quantity * (exit_price - db_entry_price)
                    if abs(pnl - db_computed_pnl) > 0.01:
                        logger.warning(
                            f"[PNL DISCREPANCY] {bot_name} {symbol}: bot reported PnL=${pnl:.4f}, "
                            f"DB computed PnL=${db_computed_pnl:.4f}, diff=${abs(pnl - db_computed_pnl):.4f}"
                        )

            # Circuit breaker: track consecutive losses per bot
            if bot_name not in self._circuit_breaker:
                self._circuit_breaker[bot_name] = {'consecutive_losses': 0, 'paused_until': None}
            cb = self._circuit_breaker[bot_name]
            if pnl < 0:
                cb['consecutive_losses'] = cb.get('consecutive_losses', 0) + 1
                if cb['consecutive_losses'] >= self.CIRCUIT_BREAKER_THRESHOLD:
                    from datetime import datetime as _dt, timedelta as _td
                    cb['paused_until'] = _dt.now() + _td(seconds=self.CIRCUIT_BREAKER_COOLDOWN)
                    logger.warning(
                        f"CIRCUIT BREAKER TRIGGERED: {bot_name} hit {cb['consecutive_losses']} consecutive losses. "
                        f"Pausing new entries for {self.CIRCUIT_BREAKER_COOLDOWN // 60} minutes."
                    )
                    if self.alerts:
                        try:
                            self.alerts.send(
                                f"CIRCUIT BREAKER: {bot_name}\n"
                                f"{cb['consecutive_losses']} consecutive losses\n"
                                f"Paused for {self.CIRCUIT_BREAKER_COOLDOWN // 60}min"
                            )
                        except Exception:
                            pass
            elif pnl > 0:
                cb['consecutive_losses'] = 0
                cb['paused_until'] = None

            # Update PnL tracking
            state = self.bots[bot_name]
            with state.lock:
                state.pnl_today += pnl
                state.trades_today += 1
        else:
            # OPEN a new trade entry
            # CRITICAL: Prefer broker fill price over signal estimate to avoid price corruption.
            # The execution result may contain filled_avg_price from the broker (actual fill).
            # Signal 'price' is only a pre-execution estimate (bid-ask midpoint or yfinance).
            signal_price = signal.get('price', 0)
            broker_fill_price = None
            if execution_result and isinstance(execution_result, dict):
                # Check multiple keys where broker fill price might be stored
                raw_fill = (execution_result.get('filled_avg_price')
                            or execution_result.get('fill_price')
                            or execution_result.get('entry_price'))
                if raw_fill:
                    try:
                        broker_fill_price = float(raw_fill)
                        if broker_fill_price <= 0:
                            broker_fill_price = None
                    except (TypeError, ValueError):
                        broker_fill_price = None

            if broker_fill_price and broker_fill_price > 0:
                entry_price = broker_fill_price
                if signal_price and signal_price > 0:
                    drift_pct = abs(broker_fill_price - signal_price) / signal_price
                    if drift_pct > 0.02:
                        logger.warning(
                            f"[PRICE DRIFT] {bot_name} {symbol}: signal=${signal_price:.4f}, "
                            f"broker fill=${broker_fill_price:.4f} (drift: {drift_pct:.2%})"
                        )
                logger.info(f"[ENTRY PRICE] {bot_name} {symbol}: using broker fill ${entry_price:.4f}")
            else:
                entry_price = signal_price
                if entry_price > 0:
                    logger.info(f"[ENTRY PRICE] {bot_name} {symbol}: using signal price ${entry_price:.4f} (no broker fill available)")
                else:
                    logger.warning(f"[ENTRY PRICE] {bot_name} {symbol}: no price available from broker or signal")

            trade = PaperTrade(
                trade_id=f"{bot_name}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                bot_name=bot_name,
                market=self.bots[bot_name].config.market.value,
                symbol=symbol,
                side=action,
                entry_price=entry_price,
                exit_price=None,
                quantity=signal.get('quantity', signal.get('shares', 1)),
                entry_time=datetime.now(),
                exit_time=None,
                pnl=0,
                status='open' if execution_success else 'failed'
            )

            self.db.log_trade(trade)
            with self.bots[bot_name].lock:
                self.bots[bot_name].trades_today += 1

        # Report execution to risk management
        if self.risk_integration and signal.get('risk_request_id'):
            try:
                self.risk_integration.execute_trade(
                    request_id=signal['risk_request_id'],
                    executed_size=signal.get('quantity', signal.get('shares', 1)),
                    executed_price=signal.get('price', 0),
                    success=execution_success
                )
            except Exception as e:
                logger.error(f"Failed to report trade execution to risk management: {e}")

        # Send alert with execution status + AI reasoning
        if self.alerts:
            risk_info = ""
            if signal.get('risk_size_adjusted'):
                risk_info = f"\nRisk-adjusted size (Risk: {signal.get('risk_score', 0):.0f})"
            elif signal.get('ai_size_adjusted'):
                risk_info = f"\nAI-adjusted size ({signal.get('ai_confidence', 0):.0%} confidence)"

            # V6: Generate AI reasoning for trade
            ai_reasoning = ""
            if self.market_analyst:
                try:
                    reasoning = self.market_analyst.generate_trade_reasoning(signal)
                    if reasoning:
                        ai_reasoning = f"\nAI: {reasoning}"
                except Exception:
                    pass

            # V6: Add prediction context
            prediction_info = ""
            if self.market_analyst:
                symbol = signal.get('symbol', '')
                pred = self.market_analyst.get_prediction_for_symbol(symbol)
                if pred:
                    arrow = "↑" if pred['direction'] == 'bullish' else "↓" if pred['direction'] == 'bearish' else "→"
                    prediction_info = f"\nPrediction: {arrow} {pred['direction']} ({pred['confidence']}%)"

            # V6+: Add ensemble + signal injection context
            ensemble_info = ""
            if self.ensemble:
                symbol = signal.get('symbol', '')
                epred = self.ensemble.get_prediction(symbol)
                if epred:
                    arrow = "↑" if epred['direction'] == 'bullish' else "↓"
                    ensemble_info = f"\nEnsemble: {arrow} {epred['direction']} ({epred['confidence']}%) size={epred['position_size_pct']:.0%}"

            injection_info = ""
            if signal.get('si_action') and signal['si_action'] != 'pass':
                injection_info = f"\nSignal Inject: {signal['si_action']} (x{signal.get('si_multiplier', 1):.2f})"

            exec_status = "EXECUTED" if execution_success else "FAILED"

            self.alerts.send(
                f"[{exec_status}] {bot_name}\n"
                f"Action: {signal.get('action', 'SIGNAL').upper()}: {signal.get('symbol', signal.get('ticker', 'N/A'))}\n"
                f"Price: ${signal.get('price', 0):.2f}\n"
                f"Size: {signal.get('quantity', signal.get('shares', 1))}"
                f"{risk_info}{prediction_info}{ensemble_info}{injection_info}{ai_reasoning}"
            )
    
    def _close_on_exchange(self, bot_name: str, symbol: str, market: str, quantity: float) -> bool:
        """Close a position on the actual exchange/broker, not just in DB.

        Returns True if closed successfully, False on failure.
        Wrapped in try/except so exchange failure never prevents DB cleanup.
        """
        try:
            if market == 'prediction':
                # Kalshi/Polymarket settle automatically — nothing to close
                logger.debug(f"[EXCHANGE CLOSE] {symbol}: prediction market, skipping (auto-settle)")
                return True

            if market == 'forex':
                # OANDA bot has close_position() already
                if bot_name in self.bots and self.bots[bot_name].instance is not None:
                    bot_inst = self.bots[bot_name].instance
                    if hasattr(bot_inst, 'close_position'):
                        result = bot_inst.close_position(symbol)
                        success = result is not None
                        if success:
                            logger.info(f"[EXCHANGE CLOSE] {bot_name} {symbol}: forex position closed")
                        else:
                            logger.warning(f"[EXCHANGE CLOSE] {bot_name} {symbol}: forex close returned None")
                        return success
                logger.warning(f"[EXCHANGE CLOSE] {bot_name} {symbol}: no forex bot instance available")
                return False

            if market in ('crypto', 'stocks', 'other', ''):
                # Alpaca handles both crypto and stocks
                _alpaca = self._get_alpaca_client()
                if _alpaca is not None:
                    success = _alpaca.close_position(symbol)
                    if success:
                        logger.info(f"[EXCHANGE CLOSE] {bot_name} {symbol}: closed on Alpaca")
                    else:
                        logger.warning(f"[EXCHANGE CLOSE] {bot_name} {symbol}: Alpaca close failed")
                    return success
                else:
                    logger.warning(f"[EXCHANGE CLOSE] {bot_name} {symbol}: Alpaca client not initialized")
                    return False

            logger.debug(f"[EXCHANGE CLOSE] {bot_name} {symbol}: unknown market '{market}', skipping")
            return False

        except Exception as e:
            logger.error(f"[EXCHANGE CLOSE] {bot_name} {symbol}: exception during exchange close: {e}")
            return False

    def _check_momentum_exits(self):
        """
        V4: Check exit conditions for ALL open positions (not just momentum).
        Runs every 60 seconds to ensure positions are closed properly.

        Exit conditions:
        - Take profit: +8% (or signal's take_profit if available)
        - Stop loss: -3% (or -7% for non-momentum bots)
        - Max hold time: 24 hours (force close)
        """
        try:
            open_trades = self.db.get_open_trades()

            if not open_trades:
                return

            logger.info(f"[EXIT CHECK] Checking {len(open_trades)} open positions...")

            for trade in open_trades:
                bot_name = trade.get('bot_name', '')
                symbol = trade.get('symbol', '')
                entry_price = float(trade.get('entry_price', 0))

                if entry_price <= 0 or not symbol:
                    continue

                # RSI-Extremes manages its own exits via run_scan() every 300s
                # using 1-hour candle data — skip the 60s exit checker
                if bot_name == 'RSI-Extremes':
                    continue

                # Get current price — try bot instance first, then fallbacks
                current_price = None
                price_source = None
                if bot_name in self.bots and self.bots[bot_name].instance is not None:
                    bot_instance = self.bots[bot_name].instance
                    try:
                        current_price = bot_instance.get_price(symbol)
                        if current_price is not None:
                            price_source = 'bot_instance'
                    except Exception as e:
                        logger.warning(f"[EXIT CHECK] {bot_name} get_price({symbol}) failed: {e}")

                # Fallback #1: Alpaca for crypto symbols (contain "/" or "-USDC")
                if current_price is None and ('/' in symbol or '-USDC' in symbol or '-USD' in symbol):
                    try:
                        _alpaca = self._get_alpaca_client()
                        if _alpaca is not None:
                            current_price = _alpaca.get_price(symbol)
                            if current_price is not None:
                                price_source = 'alpaca'
                    except Exception as e:
                        logger.warning(f"[EXIT CHECK] Alpaca get_price({symbol}) failed: {e}")

                # Fallback #2: yfinance for stocks
                if current_price is None:
                    try:
                        import yfinance as yf
                        yf_symbol = symbol.replace('/USD', '-USD').replace('-USDC', '-USD')
                        ticker = yf.Ticker(yf_symbol)
                        hist = ticker.history(period='1d')
                        if len(hist) > 0:
                            current_price = float(hist['Close'].iloc[-1])
                            price_source = 'yfinance'
                    except Exception as e:
                        logger.warning(f"[EXIT CHECK] yfinance get_price({symbol}) failed: {e}")

                if current_price is None:
                    market = trade.get('market', '')
                    logger.warning(
                        f"[EXIT CHECK] Could not get price for {symbol} "
                        f"(bot={bot_name}, market={market}) — skipping exit check"
                    )
                    continue

                # Parse entry time
                entry_time_str = trade.get('entry_time', '')
                try:
                    entry_time = datetime.fromisoformat(str(entry_time_str))
                except (ValueError, TypeError):
                    entry_time = datetime.now()

                hold_seconds = (datetime.now() - entry_time).total_seconds()
                side = trade.get('side', 'buy')

                # Calculate P&L percentage based on position direction
                if side in ('short', 'sell_short'):
                    # For SHORT: profit when current < entry
                    pnl_pct = (entry_price - current_price) / entry_price if entry_price > 0 else 0
                else:
                    # For LONG: profit when current > entry
                    pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0

                exit_reason = None

                # ML-Prediction-Bot: tight exits based on trade data analysis
                # Avg win +0.4% in 10min, avg loss -1.1% in 43min — cut losers fast,
                # take modest profits, don't hold overnight
                if bot_name == 'ML-Prediction-Bot':
                    tp_threshold = 0.02    # +2% take profit (data shows +0.4% avg win)
                    sl_threshold = -0.015  # -1.5% stop loss (cut losers before -1.1% avg)
                    max_hold = 14400       # 4 hours max hold (not 24h)

                    # Trailing break-even: once +0.75%, move stop to entry (lock in gains)
                    if pnl_pct >= 0.0075:
                        # At +0.75%, any pullback below entry = exit
                        if pnl_pct < 0.001:  # Pulled back to near-flat
                            exit_reason = 'trailing_breakeven'
                else:
                    # Other bots: safety net thresholds (bots manage own exits)
                    tp_threshold = 0.15   # +15% take profit (safety net)
                    sl_threshold = -0.10  # -10% stop loss (safety net)
                    max_hold = 86400      # 24 hours

                if not exit_reason:
                    # Take profit
                    if pnl_pct >= tp_threshold:
                        exit_reason = 'take_profit'
                    # Stop loss
                    elif pnl_pct <= sl_threshold:
                        exit_reason = 'stop_loss'
                    # Max hold time
                    elif hold_seconds > max_hold:
                        exit_reason = f'max_hold_time_{max_hold//3600}h'

                if exit_reason:
                    quantity = float(trade.get('quantity', 0))
                    # Calculate actual USD P&L based on price difference
                    if side in ('short', 'sell_short'):
                        pnl_usd = quantity * (entry_price - current_price)
                    else:
                        pnl_usd = quantity * (current_price - entry_price)

                    logger.info(
                        f"[EXIT] {bot_name} {symbol}: {exit_reason} | "
                        f"Entry: ${entry_price:.4f} -> Exit: ${current_price:.4f} | "
                        f"PnL: {pnl_pct:+.2%} (${pnl_usd:+.2f}) | "
                        f"Hold: {hold_seconds/3600:.1f}h"
                    )

                    # Close on exchange first, then in DB
                    market = trade.get('market', '')
                    exchange_closed = self._close_on_exchange(bot_name, symbol, market, quantity)

                    if exchange_closed:
                        logger.info(f"[EXIT] {bot_name} {symbol}: exchange close SUCCESS, closing in DB")
                        self.db.close_trade(bot_name, symbol, current_price, pnl_usd, status='closed')
                    else:
                        logger.warning(
                            f"[EXIT] {bot_name} {symbol}: exchange close FAILED — "
                            f"marking as close_pending for retry"
                        )
                        self.db.close_trade(bot_name, symbol, current_price, pnl_usd, status='close_pending')

                    # Update bot state PnL tracking
                    if bot_name in self.bots:
                        self.bots[bot_name].pnl_today += pnl_usd
                        self.bots[bot_name].trades_today += 1

                    # Remove from bot's internal tracking if present
                    bot_inst = self.bots[bot_name].instance if bot_name in self.bots else None
                    if bot_inst and hasattr(bot_inst, 'active_positions') and symbol in bot_inst.active_positions:
                        del bot_inst.active_positions[symbol]
                    if bot_inst and hasattr(bot_inst, 'trailing_highs') and symbol in getattr(bot_inst, 'trailing_highs', {}):
                        del bot_inst.trailing_highs[symbol]
                    if bot_inst and hasattr(bot_inst, 'rsi_positions') and symbol in getattr(bot_inst, 'rsi_positions', {}):
                        del bot_inst.rsi_positions[symbol]
                    if bot_inst and hasattr(bot_inst, 'holdings') and symbol in getattr(bot_inst, 'holdings', {}):
                        del bot_inst.holdings[symbol]

                    # Send alert
                    close_status = "CLOSED" if exchange_closed else "CLOSE_PENDING"
                    if self.alerts:
                        self.alerts.send(
                            f"[{close_status}] {bot_name}\n"
                            f"Symbol: {symbol}\n"
                            f"Reason: {exit_reason}\n"
                            f"PnL: {pnl_pct:+.2%} (${pnl_usd:+.2f})\n"
                            f"Hold: {hold_seconds/3600:.1f}h"
                        )
                else:
                    logger.debug(
                        f"[HOLD] {bot_name} {symbol}: "
                        f"PnL {pnl_pct:+.2%}, Hold {hold_seconds/3600:.1f}h"
                    )

        except Exception as e:
            logger.error(f"Exit check error: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _force_close_all_stale_positions(self):
        """
        V4: Force close ANY position older than 48 hours regardless of bot type.
        Emergency cleanup for stuck positions.
        """
        try:
            open_trades = self.db.get_open_trades()

            for trade in open_trades:
                entry_time_str = trade.get('entry_time', '')
                try:
                    entry_time = datetime.fromisoformat(str(entry_time_str))
                except (ValueError, TypeError):
                    continue

                hold_seconds = (datetime.now() - entry_time).total_seconds()

                # Force close after 48 hours
                if hold_seconds > 172800:
                    bot_name = trade.get('bot_name', 'unknown')
                    symbol = trade.get('symbol', 'unknown')
                    entry_price = float(trade.get('entry_price', 0))

            # Also check positions from errored bots
            for trade in open_trades:
                bot_name = trade.get('bot_name', '')
                if bot_name in self.bots and self.bots[bot_name].status == BotStatus.ERROR:
                    logger.warning(f"[ORPHANED] {bot_name} {trade.get('symbol')} has errored bot with open position")

                    # Try to fetch current market price for accurate PnL
                    current_price = None
                    market = trade.get('market', '')

                    # Try bot's get_price method first
                    if bot_name in self.bots and self.bots[bot_name].instance is not None:
                        try:
                            current_price = self.bots[bot_name].instance.get_price(symbol)
                        except Exception as e:
                            logger.warning(f"[FORCE CLOSE] {bot_name} get_price({symbol}) failed: {e}")

                    # Fallback: try yfinance for stocks/crypto
                    if current_price is None and market in ('stocks', 'crypto', 'other'):
                        try:
                            import yfinance as yf
                            # Convert symbol format for yfinance
                            yf_symbol = symbol.replace('-USDC', '-USD')
                            ticker = yf.Ticker(yf_symbol)
                            hist = ticker.history(period='1d')
                            if len(hist) > 0:
                                yf_price = float(hist['Close'].iloc[-1])
                                # Sanity check: reject if >50% off from entry price
                                if entry_price > 0 and yf_price > 0:
                                    drift = abs(yf_price - entry_price) / entry_price
                                    if drift > 0.50:
                                        logger.warning(
                                            f"[FORCE CLOSE] yfinance price ${yf_price:.4f} for {symbol} "
                                            f"is {drift:.0%} off from entry ${entry_price:.4f} — rejecting"
                                        )
                                    else:
                                        current_price = yf_price
                                else:
                                    current_price = yf_price
                        except Exception as e:
                            logger.warning(f"[FORCE CLOSE] yfinance get_price({symbol}) failed: {e}")

                    if current_price is None:
                        logger.warning(
                            f"[FORCE CLOSE] {bot_name} {symbol}: could not get live price, "
                            f"using entry price ${entry_price:.4f} for close"
                        )

                    # Use current price if available, otherwise fall back to entry price
                    close_price = current_price if current_price and current_price > 0 else entry_price
                    quantity = float(trade.get('quantity', 0))
                    side = trade.get('side', 'buy')
                    if quantity > 0 and entry_price > 0:
                        if side in ('short', 'sell_short', 'sell'):
                            pnl = quantity * (entry_price - close_price)
                        else:
                            pnl = quantity * (close_price - entry_price)
                    else:
                        pnl = 0

                    logger.warning(
                        f"[FORCE CLOSE] {bot_name} {symbol}: "
                        f"Position held {hold_seconds/3600:.0f}h (>48h limit) | "
                        f"Entry: ${entry_price:.4f} Close: ${close_price:.4f} PnL: ${pnl:+.2f}"
                    )

                    # Close on exchange first
                    exchange_closed = self._close_on_exchange(bot_name, symbol, market, quantity)

                    if exchange_closed:
                        logger.info(f"[FORCE CLOSE] {bot_name} {symbol}: exchange close SUCCESS")
                        self.db.close_trade(bot_name, symbol, close_price, pnl, status='closed')
                    else:
                        # For stale positions >48h, force close in DB even if exchange fails
                        # (position may already be gone from exchange)
                        logger.warning(
                            f"[FORCE CLOSE] {bot_name} {symbol}: exchange close FAILED — "
                            f"force-closing in DB anyway (stale >48h)"
                        )
                        self.db.close_trade(bot_name, symbol, close_price, pnl, status='closed')

                    if self.alerts:
                        price_source = "live price" if current_price else "entry price (no live data)"
                        self.alerts.send(
                            f"[FORCE CLOSE] {bot_name}\n"
                            f"Symbol: {symbol}\n"
                            f"Reason: Stale position >48h\n"
                            f"Close: ${close_price:.4f} ({price_source})\n"
                            f"PnL: ${pnl:+.2f}"
                        )

        except Exception as e:
            logger.error(f"Force close error: {e}")

    def _check_drawdown_emergency_close(self):
        """
        V4: Emergency close ALL open positions if daily drawdown exceeds 15%.
        Runs every 60 seconds. Shuts down the orchestrator after closing.
        """
        try:
            today = self.db.get_today_summary()
            total_pnl = today.get('total_pnl', 0)

            if total_pnl >= 0:
                return  # No drawdown

            drawdown = abs(total_pnl) / self.starting_capital
            if drawdown < 0.15:
                return  # Below critical threshold

            logger.critical(
                f"[DRAWDOWN EMERGENCY] Daily PnL: ${total_pnl:+.2f} | "
                f"Drawdown: {drawdown:.1%} exceeds 15% of ${self.starting_capital:,.0f} capital"
            )

            open_trades = self.db.get_open_trades()
            if not open_trades:
                logger.warning("[DRAWDOWN EMERGENCY] No open positions to close")
                self.running = False
                return

            closed_count = 0
            total_close_pnl = 0

            for trade in open_trades:
                bot_name = trade.get('bot_name', 'unknown')
                symbol = trade.get('symbol', 'unknown')
                entry_price = float(trade.get('entry_price', 0))
                quantity = float(trade.get('quantity', 0))
                market = trade.get('market', '')

                # Try to fetch current market price for accurate PnL
                current_price = None

                # Try bot's get_price method first
                if bot_name in self.bots and self.bots[bot_name].instance is not None:
                    try:
                        current_price = self.bots[bot_name].instance.get_price(symbol)
                    except Exception as e:
                        logger.warning(f"[DRAWDOWN CLOSE] {bot_name} get_price({symbol}) failed: {e}")

                # Fallback: try yfinance for stocks/crypto
                if current_price is None and market in ('stocks', 'crypto', 'other'):
                    try:
                        import yfinance as yf
                        yf_symbol = symbol.replace('-USDC', '-USD')
                        ticker = yf.Ticker(yf_symbol)
                        hist = ticker.history(period='1d')
                        if len(hist) > 0:
                            current_price = float(hist['Close'].iloc[-1])
                    except Exception as e:
                        logger.warning(f"[DRAWDOWN CLOSE] yfinance get_price({symbol}) failed: {e}")

                if current_price is None:
                    logger.warning(
                        f"[DRAWDOWN CLOSE] {bot_name} {symbol}: no live price, using entry price"
                    )

                close_price = current_price if current_price and current_price > 0 else entry_price
                pnl = quantity * (close_price - entry_price) if quantity > 0 and entry_price > 0 else 0

                logger.critical(
                    f"[DRAWDOWN CLOSE] {bot_name} {symbol}: "
                    f"Entry: ${entry_price:.4f} Close: ${close_price:.4f} PnL: ${pnl:+.2f}"
                )

                # Close on exchange first — emergency so force DB close regardless
                exchange_closed = self._close_on_exchange(bot_name, symbol, market, quantity)
                if exchange_closed:
                    logger.info(f"[DRAWDOWN CLOSE] {bot_name} {symbol}: exchange close SUCCESS")
                else:
                    logger.warning(
                        f"[DRAWDOWN CLOSE] {bot_name} {symbol}: exchange close FAILED — "
                        f"force-closing in DB anyway (emergency drawdown)"
                    )

                # Emergency: always close in DB regardless of exchange result
                self.db.close_trade(bot_name, symbol, close_price, pnl, status='closed')
                closed_count += 1
                total_close_pnl += pnl

            # Send alert
            if self.alerts:
                self.alerts.send(
                    f"🚨 [DRAWDOWN EMERGENCY CLOSE]\n"
                    f"Daily PnL: ${total_pnl:+.2f} ({drawdown:.1%} drawdown)\n"
                    f"Capital: ${self.starting_capital:,.0f}\n"
                    f"Positions closed: {closed_count}\n"
                    f"Close PnL: ${total_close_pnl:+.2f}\n"
                    f"Orchestrator SHUTTING DOWN"
                )

            logger.critical(
                f"[DRAWDOWN EMERGENCY] Closed {closed_count} positions (PnL: ${total_close_pnl:+.2f}). "
                f"Shutting down orchestrator."
            )
            self.running = False

        except Exception as e:
            logger.error(f"Drawdown emergency close error: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _reconcile_exchange_positions(self):
        """
        V4: Reconcile exchange positions against DB records every 30 minutes.
        Catches orphaned exchange positions and stale DB ghost records.
        """
        try:
            logger.info("[RECONCILE] Starting position reconciliation...")

            # 1. Fetch all Alpaca positions
            try:
                alpaca = self._get_alpaca_client()
                if alpaca is None:
                    logger.warning("[RECONCILE] Alpaca client not initialized, skipping")
                    return
                exchange_positions = alpaca.get_positions()
            except Exception as e:
                logger.error(f"[RECONCILE] Failed to fetch Alpaca positions: {e}")
                return

            # 2. Fetch all DB open trades
            db_open_trades = self.db.get_open_trades()

            # 3. Build lookup maps with normalized symbols
            def normalize_symbol(sym: str) -> str:
                """Normalize to uppercase no-separator form: BTCUSD"""
                return sym.upper().replace('/', '').replace('-', '').replace('USDC', 'USD')

            exchange_map = {}  # normalized_symbol -> position dict
            for pos in exchange_positions:
                norm = normalize_symbol(pos['symbol'])
                exchange_map[norm] = pos

            db_map = {}  # normalized_symbol -> trade dict
            for trade in db_open_trades:
                norm = normalize_symbol(trade.get('symbol', ''))
                db_map[norm] = trade

            orphans_closed = 0
            ghosts_closed = 0
            dust_closed = 0
            issues = []

            # 4. Orphans: on exchange but no DB record
            for norm_sym, pos in exchange_map.items():
                if norm_sym not in db_map:
                    raw_symbol = pos['symbol']
                    market_value = abs(pos.get('market_value', 0))

                    if market_value < 0.50:
                        # Dust — close regardless
                        logger.warning(f"[RECONCILE] Dust on exchange: {raw_symbol} (${market_value:.4f}) — closing")
                        alpaca.close_position(raw_symbol)
                        dust_closed += 1
                        issues.append(f"Dust closed: {raw_symbol} (${market_value:.4f})")
                    else:
                        logger.warning(f"[RECONCILE] Orphan on exchange: {raw_symbol} (${market_value:.2f}) — closing")
                        success = alpaca.close_position(raw_symbol)
                        if success:
                            orphans_closed += 1
                            issues.append(f"Orphan closed: {raw_symbol} (${market_value:.2f})")
                        else:
                            issues.append(f"Orphan close FAILED: {raw_symbol} (${market_value:.2f})")

            # 5. Ghosts: DB says open but no exchange position (for crypto/stocks only)
            for norm_sym, trade in db_map.items():
                market = trade.get('market', '')
                if market in ('prediction', 'forex'):
                    continue  # Don't reconcile non-Alpaca markets here
                if norm_sym not in exchange_map:
                    bot_name = trade.get('bot_name', 'unknown')

                    # FIX: Skip paper mode bots - they don't have exchange positions by design
                    bot_instance = self.bots.get(bot_name, {})
                    if hasattr(bot_instance, 'config') and getattr(bot_instance.config, 'paper_mode', False):
                        logger.debug(f"[RECONCILE] Skipping {bot_name} {norm_sym} - paper mode bot")
                        continue

                    symbol = trade.get('symbol', 'unknown')
                    entry_price = float(trade.get('entry_price', 0))
                    quantity = float(trade.get('quantity', 0))
                    side = trade.get('side', 'buy')

                    # Get current price to calculate actual P&L
                    # Try multiple price sources before giving up
                    current_price = None
                    price_source = None
                    try:
                        bot_state = self.bots.get(bot_name, {})
                        if hasattr(bot_state, 'instance') and hasattr(bot_state.instance, 'client'):
                            current_price = bot_state.instance.client.get_latest_price(symbol)
                            price_source = 'bot_client'
                    except Exception as e:
                        logger.debug(f"[RECONCILE] Bot client price fetch failed for {symbol}: {e}")

                    # Fallback: try yfinance
                    if current_price is None:
                        try:
                            import yfinance as yf
                            yf_symbol = symbol.replace('/', '-')
                            ticker = yf.Ticker(yf_symbol)
                            hist = ticker.history(period='1d')
                            if not hist.empty:
                                yf_price = float(hist['Close'].iloc[-1])
                                # Sanity check: reject yfinance price if >50% off from entry
                                # (protects against stale/wrong data from yfinance)
                                if entry_price > 0 and yf_price > 0:
                                    drift = abs(yf_price - entry_price) / entry_price
                                    if drift > 0.50:
                                        logger.warning(
                                            f"[RECONCILE] yfinance price ${yf_price:.4f} for {symbol} is "
                                            f"{drift:.0%} off from entry ${entry_price:.4f} — rejecting as stale/corrupt"
                                        )
                                    else:
                                        current_price = yf_price
                                        price_source = 'yfinance'
                                else:
                                    current_price = yf_price
                                    price_source = 'yfinance'
                        except Exception as e:
                            logger.debug(f"[RECONCILE] yfinance price fetch failed for {symbol}: {e}")

                    # Fallback: try Alpaca live quote directly
                    if current_price is None:
                        try:
                            bot_state = self.bots.get(bot_name, {})
                            if hasattr(bot_state, 'instance') and hasattr(bot_state.instance, 'alpaca'):
                                alpaca_price = bot_state.instance.alpaca.get_price(symbol)
                                if alpaca_price and alpaca_price > 0:
                                    current_price = float(alpaca_price)
                                    price_source = 'alpaca_quote'
                            elif hasattr(bot_state, 'instance') and hasattr(bot_state.instance, 'get_price'):
                                alpaca_price = bot_state.instance.get_price(symbol)
                                if alpaca_price and alpaca_price > 0:
                                    current_price = float(alpaca_price)
                                    price_source = 'bot_get_price'
                        except Exception as e:
                            logger.debug(f"[RECONCILE] Alpaca quote price fetch failed for {symbol}: {e}")

                    # If ALL price sources failed, skip this trade — retry next cycle
                    if current_price is None:
                        logger.warning(
                            f"[RECONCILE] Ghost DB record: {bot_name} {symbol} — "
                            f"cannot fetch price from any source, skipping (will retry next cycle)"
                        )
                        continue

                    # Calculate P&L based on position direction
                    if side in ('short', 'sell_short'):
                        pnl = quantity * (entry_price - current_price)
                    else:
                        pnl = quantity * (current_price - entry_price)

                    status = 'closed'
                    logger.warning(
                        f"[RECONCILE] Ghost DB record: {bot_name} {symbol} — no exchange position, "
                        f"closing with PnL=${pnl:.2f} (price via {price_source})"
                    )
                    self.db.close_trade(bot_name, symbol, current_price, pnl, status=status)
                    ghosts_closed += 1
                    issues.append(f"Ghost closed in DB: {bot_name} {symbol} PnL=${pnl:.2f}")

            # 6. Dust check on positions that ARE in both (matched)
            for norm_sym, pos in exchange_map.items():
                if norm_sym in db_map and abs(pos.get('market_value', 0)) < 0.50 and abs(pos.get('qty', 0)) > 0:
                    raw_symbol = pos['symbol']
                    logger.warning(f"[RECONCILE] Dust position (matched): {raw_symbol} (${pos['market_value']:.4f}) — closing")
                    alpaca.close_position(raw_symbol)
                    dust_closed += 1
                    issues.append(f"Dust closed (matched): {raw_symbol}")

            # 7. Summary log
            total_issues = orphans_closed + ghosts_closed + dust_closed
            if total_issues > 0:
                summary = (
                    f"[RECONCILE] Done — {total_issues} issue(s) fixed: "
                    f"{orphans_closed} orphans, {ghosts_closed} ghosts, {dust_closed} dust"
                )
                logger.warning(summary)

                # Telegram alert
                if self.alerts:
                    detail_text = "\n".join(f"  - {i}" for i in issues[:10])
                    self.alerts.send(
                        f"[RECONCILE] Position reconciliation\n"
                        f"Orphans closed: {orphans_closed}\n"
                        f"Ghost DB records closed: {ghosts_closed}\n"
                        f"Dust cleaned: {dust_closed}\n"
                        f"Details:\n{detail_text}"
                    )
            else:
                logger.info("[RECONCILE] Done — no discrepancies found")

        except Exception as e:
            logger.error(f"[RECONCILE] Reconciliation error: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _retry_close_pending_trades(self):
        """
        V4: Retry closing trades stuck in 'close_pending' status.
        Runs every 5 minutes. After 5 failed retries, force-close in DB.
        """
        MAX_RETRIES = 5
        try:
            pending_trades = self.db.get_close_pending_trades()

            if not pending_trades:
                return

            logger.info(f"[CLOSE RETRY] Found {len(pending_trades)} close_pending trades to retry")

            for trade in pending_trades:
                trade_id = trade.get('trade_id', '')
                bot_name = trade.get('bot_name', 'unknown')
                symbol = trade.get('symbol', 'unknown')
                market = trade.get('market', '')
                quantity = float(trade.get('quantity', 0))
                entry_price = float(trade.get('entry_price', 0))
                side = trade.get('side', 'buy')
                retry_count = int(trade.get('close_retry_count', 0) or 0)

                # CRITICAL: Fetch CURRENT market price, NEVER use entry_price as fallback
                exit_price = 0
                pnl = 0
                try:
                    bot = self.bots.get(bot_name, {})
                    if hasattr(bot, 'instance'):
                        # Try multiple price fetching methods
                        if hasattr(bot.instance, 'client') and hasattr(bot.instance.client, 'get_latest_price'):
                            exit_price = bot.instance.client.get_latest_price(symbol)
                        elif hasattr(bot.instance, 'get_price'):
                            exit_price = bot.instance.get_price(symbol)
                        elif hasattr(bot.instance, 'alpaca') and hasattr(bot.instance.alpaca, 'get_price'):
                            exit_price = bot.instance.alpaca.get_price(symbol)

                        if not exit_price or exit_price <= 0:
                            logger.error(
                                f"[CLOSE RETRY] {symbol}: Failed to fetch live price from bot. "
                                f"CANNOT use entry_price as fallback - closing with exit_price=0 (INACCURATE)"
                            )
                            exit_price = 0
                        else:
                            logger.info(f"[CLOSE RETRY] {symbol}: Fetched live exit price ${exit_price:.6f}")
                    else:
                        logger.error(f"[CLOSE RETRY] {symbol}: No bot instance available to fetch price - exit_price=0")
                        exit_price = 0

                    # Calculate P&L based on direction (only if we have valid exit price)
                    if entry_price > 0 and exit_price > 0 and quantity > 0:
                        if side in ('short', 'sell_short'):
                            pnl = quantity * (entry_price - exit_price)
                        else:
                            pnl = quantity * (exit_price - entry_price)
                    else:
                        logger.warning(f"[CLOSE RETRY] {symbol}: Invalid prices (entry={entry_price}, exit={exit_price}) - PnL=0")
                        pnl = 0

                except Exception as e:
                    logger.error(f"[CLOSE RETRY] Error fetching price for {symbol}: {e}")
                    # DO NOT use entry_price as fallback - this causes the bug
                    exit_price = 0
                    pnl = 0

                if retry_count >= MAX_RETRIES:
                    logger.warning(
                        f"[CLOSE RETRY] {bot_name} {symbol}: max retries ({MAX_RETRIES}) reached — "
                        f"force-closing in DB with exit_price=${exit_price:.4f}, pnl=${pnl:.2f}"
                    )
                    self.db.force_close_trade_in_db(trade_id, exit_price, pnl)

                    if self.alerts:
                        self.alerts.send(
                            f"[FORCE DB CLOSE] {bot_name}\n"
                            f"Symbol: {symbol}\n"
                            f"Reason: Max close retries ({MAX_RETRIES}) exceeded\n"
                            f"Exit Price: ${exit_price:.4f}\n"
                            f"P&L: ${pnl:.2f}\n"
                            f"WARNING: Exchange position may still be open!"
                        )
                    continue

                # Attempt exchange close again
                logger.info(
                    f"[CLOSE RETRY] {bot_name} {symbol}: retry #{retry_count + 1}/{MAX_RETRIES}, "
                    f"exit_price=${exit_price:.4f}, pnl=${pnl:.2f}"
                )
                exchange_closed = self._close_on_exchange(bot_name, symbol, market, quantity)
                self.db.increment_close_retry(trade_id)

                if exchange_closed:
                    # Success — update to 'closed' with fresh price and P&L
                    self.db.close_trade(bot_name, symbol, exit_price, pnl, status='closed')
                    logger.info(
                        f"[CLOSE RETRY] {bot_name} {symbol}: exchange close SUCCESS on retry #{retry_count + 1}, "
                        f"P&L=${pnl:.2f}"
                    )
                else:
                    logger.warning(
                        f"[CLOSE RETRY] {bot_name} {symbol}: exchange close FAILED "
                        f"(retry #{retry_count + 1}/{MAX_RETRIES})"
                    )

        except Exception as e:
            logger.error(f"[CLOSE RETRY] Error retrying close_pending trades: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _daily_reconciliation_report(self):
        """
        V4: Daily reconciliation report comparing DB open trades vs actual exchange positions.
        Runs once daily. Logs discrepancies between DB state and exchange state for
        Alpaca (crypto/stocks) and OANDA (forex).
        """
        try:
            logger.info("[DAILY RECONCILE] Starting daily reconciliation report...")

            db_open_trades = self.db.get_open_trades()
            db_pending_trades = self.db.get_close_pending_trades()
            all_db_trades = db_open_trades + db_pending_trades

            discrepancies = []

            # --- Alpaca (crypto/stocks) ---
            alpaca_positions = []
            try:
                alpaca = self._get_alpaca_client()
                if alpaca is not None:
                    alpaca_positions = alpaca.get_positions()
            except Exception as e:
                logger.warning(f"[DAILY RECONCILE] Failed to fetch Alpaca positions: {e}")

            def normalize_symbol(sym: str) -> str:
                return sym.upper().replace('/', '').replace('-', '').replace('USDC', 'USD')

            alpaca_map = {normalize_symbol(p['symbol']): p for p in alpaca_positions}

            # DB trades for Alpaca-tracked markets
            db_alpaca_trades = [
                t for t in all_db_trades
                if t.get('market', '') in ('crypto', 'stocks', 'other', '')
            ]
            db_alpaca_map = {}
            for t in db_alpaca_trades:
                norm = normalize_symbol(t.get('symbol', ''))
                db_alpaca_map[norm] = t

            # Exchange has position but DB doesn't
            for norm_sym, pos in alpaca_map.items():
                if norm_sym not in db_alpaca_map:
                    msg = (
                        f"ALPACA ORPHAN: {pos['symbol']} on exchange "
                        f"(qty={pos.get('qty', '?')}, value=${pos.get('market_value', 0):.2f}) "
                        f"but NO matching open/pending trade in DB"
                    )
                    discrepancies.append(msg)
                    logger.warning(f"[DAILY RECONCILE] {msg}")

            # DB says open but exchange doesn't have position
            for norm_sym, trade in db_alpaca_map.items():
                if norm_sym not in alpaca_map:
                    msg = (
                        f"DB GHOST: {trade.get('bot_name', '?')} {trade.get('symbol', '?')} "
                        f"status={trade.get('status', '?')} in DB "
                        f"but NO position on Alpaca exchange"
                    )
                    discrepancies.append(msg)
                    logger.warning(f"[DAILY RECONCILE] {msg}")

            # --- OANDA (forex) ---
            db_forex_trades = [
                t for t in all_db_trades
                if t.get('market', '') == 'forex'
            ]

            oanda_positions = []
            # Try to get OANDA positions from the forex bot instance
            for bot_name_key in ('OANDA-Forex', 'London-Breakout'):
                if bot_name_key in self.bots and self.bots[bot_name_key].instance is not None:
                    bot_inst = self.bots[bot_name_key].instance
                    if hasattr(bot_inst, 'get_positions'):
                        try:
                            positions = bot_inst.get_positions()
                            if positions:
                                oanda_positions.extend(positions)
                        except Exception as e:
                            logger.warning(
                                f"[DAILY RECONCILE] Failed to fetch OANDA positions "
                                f"from {bot_name_key}: {e}"
                            )

            oanda_map = {}
            for pos in oanda_positions:
                sym = pos.get('instrument', pos.get('symbol', ''))
                norm = normalize_symbol(sym)
                oanda_map[norm] = pos

            db_forex_map = {}
            for t in db_forex_trades:
                norm = normalize_symbol(t.get('symbol', ''))
                db_forex_map[norm] = t

            for norm_sym, pos in oanda_map.items():
                if norm_sym not in db_forex_map:
                    raw_sym = pos.get('instrument', pos.get('symbol', norm_sym))
                    msg = (
                        f"OANDA ORPHAN: {raw_sym} on exchange "
                        f"but NO matching open/pending trade in DB"
                    )
                    discrepancies.append(msg)
                    logger.warning(f"[DAILY RECONCILE] {msg}")

            for norm_sym, trade in db_forex_map.items():
                if norm_sym not in oanda_map:
                    msg = (
                        f"DB GHOST (forex): {trade.get('bot_name', '?')} {trade.get('symbol', '?')} "
                        f"status={trade.get('status', '?')} in DB "
                        f"but NO position on OANDA"
                    )
                    discrepancies.append(msg)
                    logger.warning(f"[DAILY RECONCILE] {msg}")

            # --- Summary ---
            total_db_open = len(db_open_trades)
            total_db_pending = len(db_pending_trades)
            total_exchange = len(alpaca_positions) + len(oanda_positions)

            summary = (
                f"[DAILY RECONCILE] Report complete:\n"
                f"  DB open trades: {total_db_open}\n"
                f"  DB close_pending trades: {total_db_pending}\n"
                f"  Exchange positions (Alpaca): {len(alpaca_positions)}\n"
                f"  Exchange positions (OANDA): {len(oanda_positions)}\n"
                f"  Discrepancies found: {len(discrepancies)}"
            )

            if discrepancies:
                logger.warning(summary)
                if self.alerts:
                    detail_text = "\n".join(f"  - {d}" for d in discrepancies[:15])
                    self.alerts.send(
                        f"[DAILY RECONCILE] Discrepancy Report\n"
                        f"DB open: {total_db_open} | DB pending: {total_db_pending}\n"
                        f"Exchange: {total_exchange}\n"
                        f"Issues ({len(discrepancies)}):\n{detail_text}"
                    )
            else:
                logger.info(summary)

        except Exception as e:
            logger.error(f"[DAILY RECONCILE] Error generating reconciliation report: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _reset_daily_stats(self):
        """Reset per-bot daily counters at midnight."""
        for name, state in self.bots.items():
            state.trades_today = 0
            state.pnl_today = 0.0
        logger.info(f"[DAILY RESET] Reset trades_today and pnl_today for {len(self.bots)} bots")

    def _send_daily_telegram_summary(self):
        """Send daily P&L summary via Telegram at 8 PM MT for vacation monitoring."""
        try:
            if not self.alerts:
                logger.warning("[DAILY SUMMARY] Telegram alerts not initialized, skipping")
                return

            # Today's summary from DB
            today = self.db.get_today_summary()
            trades_today = today.get('trades', 0)
            wins_today = today.get('wins', 0)
            pnl_today = today.get('total_pnl', 0) or 0

            # Open positions
            open_trades = self.db.get_open_trades()
            open_count = len(open_trades)

            # All-time P&L from DB
            try:
                with self.db._lock:
                    with self.db._connect() as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            SELECT COUNT(*) as total_trades,
                                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as total_wins,
                                   SUM(pnl) as all_time_pnl
                            FROM trades WHERE status = 'closed'
                        ''')
                        row = cursor.fetchone()
                        total_trades = row[0] or 0
                        total_wins = row[1] or 0
                        all_time_pnl = row[2] or 0
            except Exception as e:
                logger.warning(f"[DAILY SUMMARY] Failed to query all-time stats: {e}")
                total_trades = 0
                total_wins = 0
                all_time_pnl = 0

            # Portfolio value from Alpaca
            portfolio_value = 0
            try:
                alpaca = self._get_alpaca_client()
                if alpaca is not None:
                    account = alpaca.get_account()
                    portfolio_value = float(account.get('equity', 0) or account.get('portfolio_value', 0))
            except Exception as e:
                logger.warning(f"[DAILY SUMMARY] Failed to fetch Alpaca portfolio: {e}")

            # Bot status
            active_bots = sum(1 for s in self.bots.values()
                              if s.status in (BotStatus.RUNNING, BotStatus.WAITING))
            error_bots = sum(1 for s in self.bots.values() if s.status == BotStatus.ERROR)

            # Win rates
            win_rate_today = (wins_today / trades_today * 100) if trades_today > 0 else 0
            win_rate_all = (total_wins / total_trades * 100) if total_trades > 0 else 0

            pnl_emoji = "\U0001f4b0" if pnl_today >= 0 else "\U0001f4c9"
            all_time_emoji = "\U0001f4b0" if all_time_pnl >= 0 else "\U0001f4c9"

            text = (
                f"\U0001f4ca <b>CRYPTO BOT DAILY SUMMARY</b>\n"
                f"\n"
                f"<b>Today:</b>\n"
                f"\U0001f4c8 Trades: {trades_today} ({wins_today} wins, {win_rate_today:.0f}%)\n"
                f"{pnl_emoji} Day P&L: ${pnl_today:+.2f}\n"
                f"\n"
                f"<b>Portfolio:</b>\n"
                f"\U0001f4b5 Value: ${portfolio_value:,.2f}\n"
                f"\U0001f4c2 Open Positions: {open_count}\n"
                f"\n"
                f"<b>All-Time:</b>\n"
                f"{all_time_emoji} Total P&L: ${all_time_pnl:+.2f}\n"
                f"\U0001f4ca Total Trades: {total_trades} ({win_rate_all:.0f}% win rate)\n"
                f"\n"
                f"<b>System:</b>\n"
                f"\U00002705 Active Bots: {active_bots}\n"
                f"{f'\U0000274c Error Bots: {error_bots}' if error_bots > 0 else '\U00002705 No errors'}\n"
                f"\n"
                f"<i>{'[PAPER]' if self.paper_mode else '[LIVE]'} "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M MT')}</i>"
            )

            self.alerts._send(text, silent=True)
            logger.info(f"[DAILY SUMMARY] Telegram summary sent: {trades_today} trades, ${pnl_today:+.2f} today, ${all_time_pnl:+.2f} all-time")

        except Exception as e:
            logger.error(f"[DAILY SUMMARY] Failed to send daily summary: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def setup_schedule(self):
        """Setup schedules for all bots"""
        for name, state in self.bots.items():
            if state.status == BotStatus.ERROR:
                continue
            
            config = state.config
            
            if config.schedule_type == "interval":
                schedule.every(config.schedule_value).seconds.do(
                    self._run_bot_threaded, bot_name=name
                )
                logger.info(f"  {name}: every {config.schedule_value}s (threaded)")

            elif config.schedule_type == "daily":
                schedule.every().day.at(config.schedule_value).do(
                    self._run_bot_threaded, bot_name=name
                )
                logger.info(f"  {name}: daily at {config.schedule_value} (threaded)")

            elif config.schedule_type == "hourly":
                schedule.every().hour.at(f":{config.schedule_value:02d}").do(
                    self._run_bot_threaded, bot_name=name
                )
                logger.info(f"  {name}: hourly at :{config.schedule_value:02d} (threaded)")

        # V7: AI Market Analyst + Data Hub + Ensemble + MTF - every 30 minutes
        if self.market_analyst:
            schedule.every(30).minutes.do(
                lambda: self._bot_executor.submit(self._run_market_analysis)
            )
            logger.info("  [V7] AI Market Analyst: every 30min (data_hub + 8-source ensemble + parallel MTF)")

        # V6+: Adaptive Parameter Tuning - every 15 minutes
        if self.adaptive_params:
            schedule.every(15).minutes.do(
                lambda: self._bot_executor.submit(self._run_adaptive_params)
            )
            logger.info("  [V6+] Adaptive Params: every 15min (regime-based auto-tuning)")

        # Kalshi infrastructure background tasks
        if self.kalshi_infrastructure:
            schedule.every(60).seconds.do(
                lambda: self._bot_executor.submit(self._kalshi_poll_fills)
            )
            logger.info("  [Kalshi] Fill tracker: every 60s")

            schedule.every(30).minutes.do(
                lambda: self._bot_executor.submit(self._kalshi_collect_settlements)
            )
            logger.info("  [Kalshi] Settlement collector: every 30min")

        # Event-Edge exit checker (runs alongside bot scan cycle)
        if 'Event-Edge' in self.bots:
            schedule.every(60).seconds.do(
                lambda: self._bot_executor.submit(self._check_event_exits)
            )
            logger.info("  [Event-Edge] Exit checker: every 60s")

        # Fleet exit checker (runs alongside bot scan cycle)
        if 'Fleet-Orchestrator' in self.bots:
            schedule.every(60).seconds.do(
                lambda: self._bot_executor.submit(self._check_fleet_exits)
            )
            logger.info("  [Fleet] Exit checker: every 60s")

        # V4: Schedule exit checking every 60 seconds
        schedule.every(60).seconds.do(self._check_momentum_exits)
        logger.info("  [V4] Exit checker: every 60s (TP/SL/time-based)")

        # V4: Force-close stale positions every hour
        schedule.every(1).hours.do(self._force_close_all_stale_positions)
        logger.info("  [V4] Stale position cleanup: every 1h (>48h force close)")

        # V4: Drawdown emergency close every 60 seconds
        schedule.every(60).seconds.do(self._check_drawdown_emergency_close)
        logger.info("  [V4] Drawdown emergency close: every 60s (>15% drawdown)")

        # V4: Position reconciliation every 30 minutes
        schedule.every(30).minutes.do(self._reconcile_exchange_positions)
        logger.info("  [V4] Position reconciliation: every 30min (orphan/ghost/dust cleanup)")

        # V4: Retry close_pending trades every 5 minutes
        schedule.every(5).minutes.do(self._retry_close_pending_trades)
        logger.info("  [V4] Close-pending retry: every 5min (max 5 retries then force-close)")

        # V4: Daily reconciliation report at 06:00 UTC
        schedule.every().day.at("06:00").do(self._daily_reconciliation_report)
        logger.info("  [V4] Daily reconciliation report: 06:00 (DB vs exchange audit)")

        # V4: Reset daily stats at midnight
        schedule.every().day.at("00:00").do(self._reset_daily_stats)
        logger.info("  [V4] Daily stats reset: midnight")

        # Daily Telegram summary at 8 PM MT (vacation monitoring)
        schedule.every().day.at("20:00").do(self._send_daily_telegram_summary)
        logger.info("  [DAILY] Telegram P&L summary: 8 PM MT")

    def get_status(self) -> Dict:
        """Get comprehensive system status"""
        by_market = {}
        for market in Market:
            by_market[market.value] = {
                'total': 0,
                'running': 0,
                'error': 0,
                'bots': []
            }
        
        for name, state in self.bots.items():
            market = state.config.market.value
            by_market[market]['total'] += 1
            by_market[market]['bots'].append({
                'name': name,
                'status': state.status.value,
                'last_run': state.last_run.isoformat() if state.last_run else None,
                'trades_today': state.trades_today,
                'pnl_today': state.pnl_today,
                'error': state.error
            })
            
            if state.status == BotStatus.RUNNING or state.status == BotStatus.WAITING:
                by_market[market]['running'] += 1
            elif state.status == BotStatus.ERROR:
                by_market[market]['error'] += 1
        
        # Today's summary
        today = self.db.get_today_summary()

        # Risk management status
        risk_status = {}
        if self.risk_integration:
            try:
                risk_status = self.risk_integration.get_integration_status()
            except Exception as e:
                logger.warning(f"Failed to get risk status: {e}")

        # V6+ AI status
        ai_v6plus = {}
        if self.mtf_engine:
            try:
                ai_v6plus['mtf_accuracy'] = self.mtf_engine.get_accuracy_stats()
                ai_v6plus['mtf_confluence'] = self.mtf_engine.get_all_confluence()
            except Exception:
                pass
        if self.ensemble:
            try:
                ai_v6plus['ensemble'] = self.ensemble.get_stats()
            except Exception:
                pass
        if self.signal_injector:
            try:
                ai_v6plus['signal_injector'] = self.signal_injector.get_stats()
            except Exception:
                pass
        if self.adaptive_params:
            try:
                ai_v6plus['adaptive_params'] = self.adaptive_params.get_stats()
            except Exception:
                pass
        if self.data_hub:
            try:
                hub_data = self.data_hub.get_data()
                ai_v6plus['data_hub'] = {
                    'sources_loaded': len([k for k, v in hub_data.items() if v is not None and v != 0]),
                    'fear_greed': hub_data.get('fear_greed'),
                    'fed_funds_rate': hub_data.get('fed_funds_rate'),
                    'btc_dominance': hub_data.get('btc_dominance'),
                    'dxy': hub_data.get('dxy'),
                    'funding_rate_btc': hub_data.get('funding_rate_btc'),
                    'last_refresh': getattr(self.data_hub, '_last_refresh', None),
                }
            except Exception:
                pass
        if self.meta_signal:
            try:
                consensus_all = {}
                for sym in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
                    c = self.meta_signal.get_consensus(sym)
                    if c:
                        consensus_all[sym] = c
                ai_v6plus['meta_signal'] = {
                    'consensus': consensus_all,
                    'total_signals': sum(len(sigs) for sigs in getattr(self.meta_signal, '_signals', {}).values()),
                }
            except Exception:
                pass
        if self.prediction_engine:
            try:
                ai_v6plus['prediction_engine'] = {
                    'status': 'loaded',
                    'models_available': list(getattr(self.prediction_engine, 'models', {}).keys()) if hasattr(self.prediction_engine, 'models') else [],
                }
            except Exception:
                pass

        return {
            'timestamp': datetime.now().isoformat(),
            'paper_mode': self.paper_mode,
            'starting_capital': self.starting_capital,
            'current_capital': self.current_capital,
            'total_bots': len(self.bots),
            'running': self.running,
            'by_market': by_market,
            'today': today,
            'open_trades': len(self.db.get_open_trades()),
            'risk_management': risk_status,
            'ai_v6plus': ai_v6plus
        }
    
    def check_do_nothing_filter(self) -> bool:
        """
        Check if DoNothingFilter recommends skipping trading.

        Returns True if we should skip trading this cycle.
        """
        if not self.use_do_nothing_filter or not self.do_nothing_filter:
            return False

        # Fetch market data for analysis
        df = self._fetch_market_data('SPY')

        # Analyze conditions
        result = self.do_nothing_filter.analyze(df)

        self.do_nothing_active = result.do_nothing
        self.do_nothing_reason = result.reasoning if result.do_nothing else None

        if result.do_nothing:
            logger.warning(f"DoNothingFilter ACTIVE: {result.reasoning}")
            if self.alerts:
                self.alerts.send(
                    f"🚫 DoNothingFilter Active\n"
                    f"Reason: {result.reason.value}\n"
                    f"Entropy: {result.entropy:.2f}\n"
                    f"Volatility: {result.volatility_percentile:.0%} pctl\n"
                    f"Skipping all trades this cycle."
                )
        return result.do_nothing

    def run_all_once(self) -> Dict:
        """Run all bots once (for testing)"""
        results = {}

        # Check DoNothingFilter (only affects stocks/forex, not crypto/prediction)
        do_nothing_active = self.check_do_nothing_filter()
        if do_nothing_active:
            logger.info(f"DoNothingFilter active: {self.do_nothing_reason}")
            logger.info("Crypto and prediction bots will still run (24/7 markets)")

        for name, state in self.bots.items():
            # Skip stocks/forex when DoNothingFilter is active (e.g., weekends)
            # But ALWAYS run crypto and prediction bots (24/7 markets)
            if do_nothing_active and state.config.market in [Market.STOCKS, Market.FOREX]:
                logger.info(f"Skipping {name} (DoNothingFilter active for {state.config.market.value})")
                results[name] = {'status': 'skipped', 'reason': 'do_nothing_filter'}
                continue

            logger.info(f"Running {name}...")
            results[name] = self._run_bot(name)
        return results
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.warning(f"\nReceived {sig_name} signal - initiating graceful shutdown...")
            self.running = False  # Signal main loop to exit; stop() called after loop

        # Handle SIGINT (Ctrl+C) and SIGTERM (kill)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # On Windows, also handle SIGBREAK if available
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)

        logger.info("Signal handlers configured for graceful shutdown")

    def start(self):
        """Start the orchestrator"""
        self.running = True

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        logger.info("\n" + "=" * 70)
        logger.info("STARTING MASTER ORCHESTRATOR")
        logger.info("=" * 70)

        # Setup schedules
        logger.info("\nConfiguring schedules:")
        self.setup_schedule()

        # Send startup alert
        if self.alerts:
            status = self.get_status()
            self.alerts.system_started(mode="Paper" if self.paper_mode else "LIVE")
            self.alerts.send(
                f"[START] Master Orchestrator Started\n"
                f"[BOTS] Bots: {status['total_bots']}\n"
                f"[CAPITAL] Capital: ${self.starting_capital:,.0f}"
            )

        # Start risk management monitoring
        if self.risk_integration:
            self.risk_integration.start_monitoring()
            logger.info("Risk management monitoring started")

        # Start Telegram command listener for remote /emergency_stop
        if self.telegram_command_listener:
            self.telegram_command_listener.start()
            logger.info("Telegram command listener started (listening for /emergency_stop)")

        # Initial run
        logger.info("\nRunning initial check...")
        initial_status = self.get_status()
        logger.info(f"Status: {json.dumps(initial_status, indent=2, default=str)}")

        # V6: Run initial AI market analysis (threaded, non-blocking)
        if self.market_analyst:
            logger.info("Running initial AI market analysis (includes V6+ MTF + Ensemble)...")
            self._bot_executor.submit(self._run_market_analysis)

        # V6+: Run initial adaptive parameter adjustment
        if self.adaptive_params:
            logger.info("Running initial adaptive parameter adjustment...")
            self._bot_executor.submit(self._run_adaptive_params)

        # Main loop
        logger.info("\n[RUNNING] Orchestrator running. Press Ctrl+C to stop.\n")

        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(5)

        self.stop()
    
    def stop(self):
        """Stop the orchestrator"""
        self.running = False

        # Shut down bot thread pool (wait up to 30s for running bots to finish)
        if hasattr(self, '_bot_executor'):
            logger.info("Shutting down bot thread pool...")
            self._bot_executor.shutdown(wait=True, cancel_futures=True)
            logger.info("Bot thread pool shut down")

        # Stop Telegram command listener
        if self.telegram_command_listener:
            self.telegram_command_listener.stop()
            logger.info("Telegram command listener stopped")

        # Stop risk management monitoring
        if self.risk_integration:
            self.risk_integration.stop_monitoring()
            logger.info("Risk management monitoring stopped")

        # Cancel all Kalshi open orders on shutdown
        if self.kalshi_infrastructure:
            try:
                self.kalshi_infrastructure.cancel_all_orders()
                logger.info("Kalshi orders cancelled on shutdown")
            except Exception as e:
                logger.error(f"Failed to cancel Kalshi orders on shutdown: {e}")

        logger.info("\n🛑 Master Orchestrator stopped")

        if self.alerts:
            self.alerts.system_stopped()


# =============================================================================
# MAIN
# =============================================================================

def confirm_live_trading() -> bool:
    """
    Require explicit confirmation before enabling live trading.

    Returns:
        True if user confirms, False otherwise
    """
    print("\n" + "=" * 70)
    print("WARNING: LIVE TRADING MODE REQUESTED")
    print("=" * 70)
    print("\nThis will execute REAL trades with REAL money.")
    print("All positions and orders will affect your actual account balances.")
    print("\nTo confirm, type exactly: CONFIRM LIVE")
    print("To cancel, type anything else or press Ctrl+C\n")

    try:
        confirmation = input("Confirmation: ").strip()
        if confirmation == "CONFIRM LIVE":
            print("\n[CONFIRMED] Live trading confirmed.\n")
            return True
        else:
            print("\n[CANCELLED] Live trading NOT confirmed. Exiting.")
            return False
    except (KeyboardInterrupt, EOFError):
        print("\n❌ Cancelled. Exiting.")
        return False


def main():
    import argparse
    from dotenv import load_dotenv

    # Load .env file
    load_dotenv()

    # Get capital from environment, default to 500 if not set
    default_capital = float(os.getenv('TOTAL_CAPITAL', 500))
    paper_mode_env = os.getenv('PAPER_MODE', 'true').lower() == 'true'

    parser = argparse.ArgumentParser(description="Master Trading Orchestrator")
    parser.add_argument('--capital', type=float, default=default_capital, help="Starting capital")
    parser.add_argument('--live', action='store_true', help="Live mode (default: paper)")
    parser.add_argument('--confirm-live', action='store_true', help="Skip interactive confirmation for live mode")
    parser.add_argument('--status', action='store_true', help="Show status and exit")
    parser.add_argument('--test', action='store_true', help="Run all bots once and exit")
    parser.add_argument('--list', action='store_true', help="List all bots")
    parser.add_argument('--no-risk-management', action='store_true', help="Disable advanced risk management")

    args = parser.parse_args()

    if args.list:
        print("\n" + "=" * 70)
        print("REGISTERED BOTS")
        print("=" * 70)
        for config in BOT_REGISTRY:
            status = "[ON]" if config.enabled else "[OFF]"
            print(f"{status} {config.name:<25} {config.market.value:<12} {config.allocation_pct*100:>5.1f}%")
        print("=" * 70)
        print(f"Total: {len(BOT_REGISTRY)} bots")
        return

    # Use --live flag if passed, otherwise use PAPER_MODE from environment
    paper_mode = False if args.live else paper_mode_env

    # Require confirmation for live trading
    if not paper_mode:
        if not args.confirm_live:
            if not confirm_live_trading():
                return
        else:
            print("\nLive trading mode enabled via --confirm-live flag\n")

    orchestrator = MasterOrchestrator(
        starting_capital=args.capital,
        paper_mode=paper_mode,
        enable_risk_management=not args.no_risk_management
    )
    
    if args.status:
        status = orchestrator.get_status()
        print(json.dumps(status, indent=2, default=str))
        return
    
    if args.test:
        print("\n🧪 Running all bots once (test mode)...\n")
        results = orchestrator.run_all_once()
        for name, result in results.items():
            # Handle both dict results and string results (from filters)
            if isinstance(result, dict):
                status = "[OK]" if result.get('status') != 'error' else "[ERR]"
            elif isinstance(result, str):
                status = "[INFO]"  # Info/filter message
            else:
                status = "❓"
            print(f"{status} {name}: {result}")
        return
    
    try:
        orchestrator.start()
    except KeyboardInterrupt:
        orchestrator.stop()


if __name__ == "__main__":
    main()
