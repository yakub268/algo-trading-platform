"""
Extended Scheduler Integration

Adds all new strategies to unified_scheduler.py
Copy this code into unified_scheduler.py or import as module.

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import logging
import schedule
from datetime import datetime, timezone
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger('ExtendedScheduler')


class ExtendedStrategyRunner:
    """
    Runs all extended strategies and integrates with UnifiedEventScheduler.
    
    Strategies included:
    - RSI-2 Mean Reversion
    - Cumulative RSI
    - MACD+RSI Combo
    - Bollinger Squeeze
    - Multi-Timeframe RSI
    - Earnings Bot (PEAD)
    - Sentiment Bot
    - Crypto Arb Scanner
    """
    
    # Default watchlist for technical strategies
    STOCK_WATCHLIST = [
        'SPY', 'QQQ', 'IWM', 'DIA',  # ETFs
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',  # Tech mega-caps
        'META', 'TSLA', 'AMD', 'NFLX', 'CRM',  # Growth
    ]
    
    CRYPTO_PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT']
    
    def __init__(self, paper_mode: bool = True, telegram_enabled: bool = True):
        self.paper_mode = paper_mode
        self.telegram_enabled = telegram_enabled
        self.strategies = {}
        self.active_signals = []
        self.trade_log = []
        
        self._init_strategies()
        logger.info(f"ExtendedStrategyRunner initialized (paper={paper_mode})")
    
    def _init_strategies(self):
        """Initialize all strategy instances"""
        
        # RSI-2 Mean Reversion
        try:
            from strategies.rsi2_mean_reversion import RSI2MeanReversion
            self.strategies['rsi2'] = RSI2MeanReversion(paper_mode=self.paper_mode)
            logger.info("[OK] RSI-2 Mean Reversion")
        except Exception as e:
            logger.warning(f"RSI-2 init failed: {e}")
        
        # Cumulative RSI
        try:
            from strategies.cumulative_rsi import CumulativeRSIStrategy
            self.strategies['cumulative_rsi'] = CumulativeRSIStrategy(paper_mode=self.paper_mode)
            logger.info("[OK] Cumulative RSI")
        except Exception as e:
            logger.warning(f"Cumulative RSI init failed: {e}")
        
        # MACD+RSI Combo
        try:
            from strategies.macd_rsi_combo import MACDRSIStrategy
            self.strategies['macd_rsi'] = MACDRSIStrategy(paper_mode=self.paper_mode)
            logger.info("[OK] MACD+RSI Combo")
        except Exception as e:
            logger.warning(f"MACD+RSI init failed: {e}")
        
        # Bollinger Squeeze
        try:
            from strategies.bollinger_squeeze import BollingerSqueezeStrategy
            self.strategies['bollinger_squeeze'] = BollingerSqueezeStrategy(paper_mode=self.paper_mode)
            logger.info("[OK] Bollinger Squeeze")
        except Exception as e:
            logger.warning(f"Bollinger Squeeze init failed: {e}")
        
        # Multi-Timeframe RSI
        try:
            from strategies.mtf_rsi import MultiTimeframeRSI
            self.strategies['mtf_rsi'] = MultiTimeframeRSI(paper_mode=self.paper_mode)
            logger.info("[OK] MTF RSI")
        except Exception as e:
            logger.warning(f"MTF RSI init failed: {e}")
        
        # Earnings Bot
        try:
            from bots.earnings_bot import EarningsBot
            self.strategies['earnings'] = EarningsBot(paper_mode=self.paper_mode)
            logger.info("[OK] Earnings Bot")
        except Exception as e:
            logger.warning(f"Earnings Bot init failed: {e}")
        
        # Sentiment Bot
        try:
            from bots.sentiment_bot import SentimentBot
            self.strategies['sentiment'] = SentimentBot(paper_mode=self.paper_mode)
            logger.info("[OK] Sentiment Bot")
        except Exception as e:
            logger.warning(f"Sentiment Bot init failed: {e}")
        
        # Crypto Arb Scanner
        try:
            from bots.crypto_arb_scanner import CryptoArbScanner
            self.strategies['crypto_arb'] = CryptoArbScanner()
            logger.info("[OK] Crypto Arb Scanner")
        except Exception as e:
            logger.warning(f"Crypto Arb init failed: {e}")
    
    def run_rsi2_scan(self) -> List[Dict]:
        """Run RSI-2 strategy scan"""
        if 'rsi2' not in self.strategies:
            return []
        
        import yfinance as yf
        signals = []
        
        logger.info("[RSI2] Running scan...")
        
        for symbol in self.STOCK_WATCHLIST:
            try:
                df = yf.download(symbol, period="6mo", progress=False)
                if len(df) > 0:
                    signal = self.strategies['rsi2'].generate_signal(df)
                    if signal and signal.signal_type.value in ['buy', 'sell']:
                        sig_dict = {
                            'strategy': 'rsi2',
                            'symbol': symbol,
                            'signal': signal.signal_type.value,
                            'price': signal.price,
                            'rsi2': signal.rsi2,
                            'confidence': signal.confidence,
                            'stop_loss': signal.stop_loss,
                            'reasoning': signal.reasoning,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                        signals.append(sig_dict)
                        logger.info(f"[RSI2] {signal.signal_type.value.upper()} {symbol} @ ${signal.price:.2f}")
                        
                        if self.telegram_enabled:
                            self._send_signal_alert(sig_dict)
            except Exception as e:
                logger.debug(f"[RSI2] {symbol} error: {e}")
        
        return signals
    
    def run_cumulative_rsi_scan(self) -> List[Dict]:
        """Run Cumulative RSI strategy scan"""
        if 'cumulative_rsi' not in self.strategies:
            return []
        
        import yfinance as yf
        signals = []
        
        logger.info("[CUM_RSI] Running scan...")
        
        for symbol in self.STOCK_WATCHLIST:
            try:
                df = yf.download(symbol, period="6mo", progress=False)
                if len(df) > 0:
                    signal = self.strategies['cumulative_rsi'].generate_signal(df)
                    if signal and signal.signal_type.value in ['buy', 'sell']:
                        sig_dict = {
                            'strategy': 'cumulative_rsi',
                            'symbol': symbol,
                            'signal': signal.signal_type.value,
                            'price': signal.price,
                            'cumulative_rsi': signal.cumulative_rsi,
                            'confidence': signal.confidence,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                        signals.append(sig_dict)
                        logger.info(f"[CUM_RSI] {signal.signal_type.value.upper()} {symbol}")
                        
                        if self.telegram_enabled:
                            self._send_signal_alert(sig_dict)
            except Exception as e:
                logger.debug(f"[CUM_RSI] {symbol} error: {e}")
        
        return signals
    
    def run_macd_rsi_scan(self) -> List[Dict]:
        """Run MACD+RSI combo strategy scan"""
        if 'macd_rsi' not in self.strategies:
            return []
        
        import yfinance as yf
        signals = []
        
        logger.info("[MACD_RSI] Running scan...")
        
        for symbol in self.STOCK_WATCHLIST:
            try:
                df = yf.download(symbol, period="6mo", progress=False)
                if len(df) > 0:
                    signal = self.strategies['macd_rsi'].generate_signal(df)
                    if signal and signal.signal_type.value in ['buy', 'sell']:
                        sig_dict = {
                            'strategy': 'macd_rsi',
                            'symbol': symbol,
                            'signal': signal.signal_type.value,
                            'price': signal.price,
                            'confidence': signal.confidence,
                            'reasoning': signal.reasoning,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                        signals.append(sig_dict)
                        logger.info(f"[MACD_RSI] {signal.signal_type.value.upper()} {symbol}")
                        
                        if self.telegram_enabled:
                            self._send_signal_alert(sig_dict)
            except Exception as e:
                logger.debug(f"[MACD_RSI] {symbol} error: {e}")
        
        return signals
    
    def run_bollinger_scan(self) -> List[Dict]:
        """Run Bollinger Squeeze strategy scan"""
        if 'bollinger_squeeze' not in self.strategies:
            return []
        
        import yfinance as yf
        signals = []
        
        logger.info("[BOLLINGER] Running scan...")
        
        for symbol in self.STOCK_WATCHLIST:
            try:
                df = yf.download(symbol, period="1y", progress=False)
                if len(df) > 0:
                    signal = self.strategies['bollinger_squeeze'].generate_signal(df)
                    if signal and signal.signal_type.value in ['buy', 'sell']:
                        sig_dict = {
                            'strategy': 'bollinger_squeeze',
                            'symbol': symbol,
                            'signal': signal.signal_type.value,
                            'price': signal.price,
                            'bb_width_pctl': signal.bb_width_percentile,
                            'confidence': signal.confidence,
                            'reasoning': signal.reasoning,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                        signals.append(sig_dict)
                        logger.info(f"[BOLLINGER] {signal.signal_type.value.upper()} {symbol}")
                        
                        if self.telegram_enabled:
                            self._send_signal_alert(sig_dict)
            except Exception as e:
                logger.debug(f"[BOLLINGER] {symbol} error: {e}")
        
        return signals
    
    def run_mtf_rsi_scan(self) -> List[Dict]:
        """Run Multi-Timeframe RSI strategy scan"""
        if 'mtf_rsi' not in self.strategies:
            return []
        
        import yfinance as yf
        signals = []
        
        logger.info("[MTF_RSI] Running scan...")
        
        for symbol in self.STOCK_WATCHLIST:
            try:
                df = yf.download(symbol, period="6mo", progress=False)
                if len(df) > 0:
                    signal = self.strategies['mtf_rsi'].generate_signal(df)
                    if signal and signal.signal_type.value in ['buy', 'sell']:
                        sig_dict = {
                            'strategy': 'mtf_rsi',
                            'symbol': symbol,
                            'signal': signal.signal_type.value,
                            'price': signal.price,
                            'daily_rsi': signal.daily_rsi,
                            'h4_rsi': signal.h4_rsi,
                            'h1_rsi': signal.h1_rsi,
                            'confidence': signal.confidence,
                            'reasoning': signal.reasoning,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                        signals.append(sig_dict)
                        logger.info(f"[MTF_RSI] {signal.signal_type.value.upper()} {symbol}")
                        
                        if self.telegram_enabled:
                            self._send_signal_alert(sig_dict)
            except Exception as e:
                logger.debug(f"[MTF_RSI] {symbol} error: {e}")
        
        return signals
    
    def run_earnings_scan(self) -> List[Dict]:
        """Run Earnings Bot (PEAD) scan"""
        if 'earnings' not in self.strategies:
            return []
        
        signals = []
        logger.info("[EARNINGS] Running scan...")
        
        try:
            opportunities = self.strategies['earnings'].scan_for_surprises()
            
            for opp in opportunities:
                if abs(opp.get('surprise_pct', 0)) > 0.05:  # 5% threshold
                    sig_dict = {
                        'strategy': 'earnings',
                        'symbol': opp['symbol'],
                        'signal': 'buy' if opp['surprise_pct'] > 0 else 'avoid',
                        'surprise_pct': opp['surprise_pct'],
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    signals.append(sig_dict)
                    logger.info(f"[EARNINGS] {opp['symbol']} surprise: {opp['surprise_pct']:.1%}")
                    
                    if self.telegram_enabled:
                        self._send_signal_alert(sig_dict)
        except Exception as e:
            logger.error(f"[EARNINGS] Error: {e}")
        
        return signals
    
    def run_sentiment_scan(self) -> List[Dict]:
        """Run Sentiment Bot scan"""
        if 'sentiment' not in self.strategies:
            return []
        
        signals = []
        logger.info("[SENTIMENT] Running scan...")
        
        try:
            sentiment_signals = self.strategies['sentiment'].scan_watchlist(
                ['AAPL', 'TSLA', 'NVDA', 'AMD', 'SPY']
            )
            
            for sig in sentiment_signals:
                sig_dict = {
                    'strategy': 'sentiment',
                    'symbol': sig.symbol,
                    'signal': sig.direction,
                    'z_score': sig.z_score,
                    'confidence': sig.confidence,
                    'reasoning': sig.reasoning,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                signals.append(sig_dict)
                logger.info(f"[SENTIMENT] {sig.direction.upper()} {sig.symbol} (z={sig.z_score:.2f})")
                
                if self.telegram_enabled:
                    self._send_signal_alert(sig_dict)
        except Exception as e:
            logger.error(f"[SENTIMENT] Error: {e}")
        
        return signals
    
    def run_crypto_arb_scan(self) -> List[Dict]:
        """Run Crypto Arbitrage scanner"""
        if 'crypto_arb' not in self.strategies:
            return []
        
        signals = []
        logger.info("[CRYPTO_ARB] Running scan...")
        
        try:
            opportunities = self.strategies['crypto_arb'].scan_all_pairs(self.CRYPTO_PAIRS)
            
            for opp in opportunities:
                sig_dict = {
                    'strategy': 'crypto_arb',
                    'symbol': opp.symbol,
                    'buy_exchange': opp.buy_exchange,
                    'sell_exchange': opp.sell_exchange,
                    'spread_pct': opp.spread_pct,
                    'profit_pct': opp.potential_profit_pct,
                    'confidence': opp.confidence,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                signals.append(sig_dict)
                logger.info(f"[CRYPTO_ARB] {opp.symbol}: {opp.spread_pct:.2%} spread")
                
                if self.telegram_enabled and opp.spread_pct >= 0.01:  # Alert on 1%+ spread
                    self._send_arb_alert(sig_dict)
        except Exception as e:
            logger.error(f"[CRYPTO_ARB] Error: {e}")
        
        return signals
    
    def run_all_technical_strategies(self) -> Dict[str, List]:
        """Run all technical analysis strategies"""
        logger.info("=" * 50)
        logger.info("RUNNING ALL TECHNICAL STRATEGIES")
        logger.info("=" * 50)
        
        results = {
            'rsi2': self.run_rsi2_scan(),
            'cumulative_rsi': self.run_cumulative_rsi_scan(),
            'macd_rsi': self.run_macd_rsi_scan(),
            'bollinger': self.run_bollinger_scan(),
            'mtf_rsi': self.run_mtf_rsi_scan(),
        }
        
        total_signals = sum(len(v) for v in results.values())
        logger.info(f"TOTAL SIGNALS: {total_signals}")
        
        return results
    
    def run_all_event_strategies(self) -> Dict[str, List]:
        """Run all event-driven strategies"""
        logger.info("=" * 50)
        logger.info("RUNNING ALL EVENT STRATEGIES")
        logger.info("=" * 50)
        
        results = {
            'earnings': self.run_earnings_scan(),
            'sentiment': self.run_sentiment_scan(),
        }
        
        return results
    
    def run_full_scan(self) -> Dict[str, List]:
        """Run complete scan of all strategies"""
        results = {}
        results.update(self.run_all_technical_strategies())
        results.update(self.run_all_event_strategies())
        results['crypto_arb'] = self.run_crypto_arb_scan()
        
        # Store active signals
        self.active_signals = []
        for strategy, signals in results.items():
            self.active_signals.extend(signals)
        
        return results
    
    def _send_signal_alert(self, signal: Dict):
        """Send Telegram alert for trading signal"""
        try:
            from utils.telegram_alerts import send_strategy_alert
            send_strategy_alert(signal)
        except Exception as e:
            logger.debug(f"Telegram alert failed: {e}")
    
    def _send_arb_alert(self, signal: Dict):
        """Send Telegram alert for arbitrage opportunity"""
        try:
            from utils.telegram_alerts import send_arb_alert
            send_arb_alert(signal)
        except Exception as e:
            logger.debug(f"Telegram alert failed: {e}")
    
    def get_status(self) -> Dict:
        """Get current status of all strategies"""
        return {
            'strategies_loaded': list(self.strategies.keys()),
            'paper_mode': self.paper_mode,
            'telegram_enabled': self.telegram_enabled,
            'active_signals': len(self.active_signals),
            'watchlist_stocks': len(self.STOCK_WATCHLIST),
            'watchlist_crypto': len(self.CRYPTO_PAIRS),
        }


def add_to_scheduler(scheduler):
    """
    Add extended strategies to an existing UnifiedEventScheduler instance.
    
    Usage:
        scheduler = UnifiedEventScheduler(...)
        add_to_scheduler(scheduler)
        scheduler.run()
    """
    runner = ExtendedStrategyRunner(
        paper_mode=scheduler.paper_mode,
        telegram_enabled=True
    )
    
    # Add to scheduler instance
    scheduler.extended_runner = runner
    
    # Technical scans at market open and mid-day
    schedule.every().day.at("09:35").do(runner.run_all_technical_strategies)
    schedule.every().day.at("14:00").do(runner.run_all_technical_strategies)
    
    # Event strategies
    schedule.every().day.at("08:00").do(runner.run_earnings_scan)
    schedule.every().day.at("12:00").do(runner.run_sentiment_scan)
    schedule.every().day.at("16:00").do(runner.run_sentiment_scan)
    
    # Crypto arb every 5 minutes
    schedule.every(5).minutes.do(runner.run_crypto_arb_scan)
    
    logger.info("Extended strategies added to scheduler")
    logger.info(f"  - Technical scans: 09:35, 14:00")
    logger.info(f"  - Earnings scan: 08:00")
    logger.info(f"  - Sentiment scan: 12:00, 16:00")
    logger.info(f"  - Crypto arb: every 5 min")
    
    return runner


if __name__ == "__main__":
    # Test run
    print("=" * 60)
    print("EXTENDED STRATEGY RUNNER - TEST")
    print("=" * 60)
    
    runner = ExtendedStrategyRunner(paper_mode=True, telegram_enabled=False)
    
    print(f"\nLoaded strategies: {list(runner.strategies.keys())}")
    print(f"Stock watchlist: {len(runner.STOCK_WATCHLIST)} symbols")
    print(f"Crypto pairs: {len(runner.CRYPTO_PAIRS)} pairs")
    
    # Run a quick scan
    print("\nRunning RSI-2 scan on SPY only...")
    runner.STOCK_WATCHLIST = ['SPY']  # Limit for test
    signals = runner.run_rsi2_scan()
    
    print(f"\nSignals found: {len(signals)}")
    for sig in signals:
        print(f"  {sig['strategy']}: {sig['signal']} {sig['symbol']}")
