"""
Extended Strategy Integration Module

Integrates all new strategies into the unified scheduler:
- Earnings Bot (PEAD)
- Sentiment Bot
- Crypto Arb Scanner
- MACD+RSI Combo
- Cumulative RSI
- Bollinger Squeeze
- Multi-Timeframe RSI

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('StrategyIntegration')


class StrategyManager:
    """
    Manages all trading strategies and coordinates their execution.
    
    Strategies are organized by type:
    - Event-driven: Earnings, FOMC, Fed decisions
    - Technical: RSI-2, MACD+RSI, Bollinger, MTF RSI
    - Sentiment: Social media analysis
    - Arbitrage: Crypto cross-exchange
    """
    
    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.strategies = {}
        self.active_signals = []
        
        self._init_strategies()
        logger.info(f"StrategyManager initialized with {len(self.strategies)} strategies")
    
    def _init_strategies(self):
        """Initialize all available strategies"""
        
        # Technical Strategies
        try:
            from strategies.rsi2_mean_reversion import RSI2MeanReversion
            self.strategies['rsi2'] = {
                'class': RSI2MeanReversion,
                'instance': None,
                'enabled': True,
                'schedule': ['09:35', '10:30', '14:00', '15:30'],
                'category': 'technical'
            }
            logger.info("[OK] RSI-2 Mean Reversion loaded")
        except ImportError as e:
            logger.warning(f"RSI-2 not available: {e}")
        
        try:
            from strategies.cumulative_rsi import CumulativeRSIStrategy
            self.strategies['cumulative_rsi'] = {
                'class': CumulativeRSIStrategy,
                'instance': None,
                'enabled': True,
                'schedule': ['09:35', '10:30', '14:00', '15:30'],
                'category': 'technical'
            }
            logger.info("[OK] Cumulative RSI loaded")
        except ImportError as e:
            logger.warning(f"Cumulative RSI not available: {e}")
        
        try:
            from strategies.macd_rsi_combo import MACDRSIStrategy
            self.strategies['macd_rsi'] = {
                'class': MACDRSIStrategy,
                'instance': None,
                'enabled': True,
                'schedule': ['09:35', '14:00'],
                'category': 'technical'
            }
            logger.info("[OK] MACD+RSI Combo loaded")
        except ImportError as e:
            logger.warning(f"MACD+RSI not available: {e}")
        
        try:
            from strategies.bollinger_squeeze import BollingerSqueezeStrategy
            self.strategies['bollinger_squeeze'] = {
                'class': BollingerSqueezeStrategy,
                'instance': None,
                'enabled': True,
                'schedule': ['09:35', '14:00'],
                'category': 'technical'
            }
            logger.info("[OK] Bollinger Squeeze loaded")
        except ImportError as e:
            logger.warning(f"Bollinger Squeeze not available: {e}")
        
        try:
            from strategies.mtf_rsi import MultiTimeframeRSI
            self.strategies['mtf_rsi'] = {
                'class': MultiTimeframeRSI,
                'instance': None,
                'enabled': True,
                'schedule': ['09:35', '14:00'],
                'category': 'technical'
            }
            logger.info("[OK] Multi-Timeframe RSI loaded")
        except ImportError as e:
            logger.warning(f"MTF RSI not available: {e}")
        
        # Event-Driven Strategies
        try:
            from bots.earnings_bot import EarningsBot
            self.strategies['earnings'] = {
                'class': EarningsBot,
                'instance': None,
                'enabled': True,
                'schedule': ['08:00', '16:30'],  # Pre-market and after-hours
                'category': 'event'
            }
            logger.info("[OK] Earnings Bot (PEAD) loaded")
        except ImportError as e:
            logger.warning(f"Earnings Bot not available: {e}")
        
        # Sentiment Strategies
        try:
            from bots.sentiment_bot import SentimentBot
            self.strategies['sentiment'] = {
                'class': SentimentBot,
                'instance': None,
                'enabled': True,
                'schedule': ['08:00', '12:00', '16:00'],
                'category': 'sentiment'
            }
            logger.info("[OK] Sentiment Bot loaded")
        except ImportError as e:
            logger.warning(f"Sentiment Bot not available: {e}")
        
        # Arbitrage Strategies
        try:
            from bots.crypto_arb_scanner import CryptoArbScanner
            self.strategies['crypto_arb'] = {
                'class': CryptoArbScanner,
                'instance': None,
                'enabled': True,
                'schedule': ['every_5_min'],
                'category': 'arbitrage'
            }
            logger.info("[OK] Crypto Arb Scanner loaded")
        except ImportError as e:
            logger.warning(f"Crypto Arb Scanner not available: {e}")
    
    def get_strategy(self, name: str):
        """Get or create strategy instance"""
        if name not in self.strategies:
            return None
        
        strat = self.strategies[name]
        if strat['instance'] is None:
            try:
                strat['instance'] = strat['class'](paper_mode=self.paper_mode)
            except TypeError:
                # Some strategies don't take paper_mode
                strat['instance'] = strat['class']()
        
        return strat['instance']
    
    def run_technical_scan(self, symbols: List[str] = None) -> List[Dict]:
        """
        Run all technical strategies on watchlist.
        
        Args:
            symbols: List of symbols to scan
            
        Returns:
            List of signals from all strategies
        """
        symbols = symbols or [
            'SPY', 'QQQ', 'IWM', 'DIA',
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'AMD', 'NFLX'
        ]
        
        all_signals = []
        
        for name, config in self.strategies.items():
            if config['category'] != 'technical' or not config['enabled']:
                continue
            
            strategy = self.get_strategy(name)
            if not strategy:
                continue
            
            logger.info(f"Running {name} scan...")
            
            for symbol in symbols:
                try:
                    # Most strategies have generate_signal method
                    if hasattr(strategy, 'generate_signal'):
                        signal = strategy.generate_signal(symbol)
                        if signal and hasattr(signal, 'signal_type'):
                            if signal.signal_type.value in ['buy', 'sell']:
                                all_signals.append({
                                    'strategy': name,
                                    'symbol': symbol,
                                    'signal': signal.signal_type.value,
                                    'price': getattr(signal, 'price', None),
                                    'confidence': getattr(signal, 'confidence', 0.5),
                                    'reasoning': getattr(signal, 'reasoning', ''),
                                    'stop_loss': getattr(signal, 'stop_loss', None),
                                    'timestamp': datetime.now(timezone.utc).isoformat()
                                })
                except Exception as e:
                    logger.debug(f"{name} error for {symbol}: {e}")
        
        logger.info(f"Technical scan complete: {len(all_signals)} signals")
        return all_signals
    
    def run_earnings_scan(self) -> List[Dict]:
        """Run earnings bot scan"""
        strategy = self.get_strategy('earnings')
        if not strategy:
            return []
        
        try:
            logger.info("Running earnings scan...")
            surprises = strategy.scan_for_surprises()
            
            signals = []
            for surprise in surprises:
                if surprise.get('surprise_pct', 0) > 0.05:  # 5% threshold
                    signals.append({
                        'strategy': 'earnings',
                        'symbol': surprise['symbol'],
                        'signal': 'buy' if surprise['surprise_pct'] > 0 else 'sell',
                        'surprise_pct': surprise['surprise_pct'],
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
            
            logger.info(f"Earnings scan complete: {len(signals)} opportunities")
            return signals
            
        except Exception as e:
            logger.error(f"Earnings scan error: {e}")
            return []
    
    def run_sentiment_scan(self, symbols: List[str] = None) -> List[Dict]:
        """Run sentiment bot scan"""
        strategy = self.get_strategy('sentiment')
        if not strategy:
            return []
        
        symbols = symbols or ['AAPL', 'TSLA', 'SPY', 'NVDA', 'AMD']
        
        try:
            logger.info("Running sentiment scan...")
            signals = strategy.scan_watchlist(symbols)
            
            results = []
            for sig in signals:
                results.append({
                    'strategy': 'sentiment',
                    'symbol': sig.symbol,
                    'signal': sig.direction,
                    'z_score': sig.z_score,
                    'confidence': sig.confidence,
                    'reasoning': sig.reasoning,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            
            logger.info(f"Sentiment scan complete: {len(results)} signals")
            return results
            
        except Exception as e:
            logger.error(f"Sentiment scan error: {e}")
            return []
    
    def run_arb_scan(self) -> List[Dict]:
        """Run crypto arbitrage scan"""
        strategy = self.get_strategy('crypto_arb')
        if not strategy:
            return []
        
        try:
            logger.info("Running crypto arb scan...")
            opportunities = strategy.scan_all_pairs()
            
            results = []
            for opp in opportunities:
                results.append({
                    'strategy': 'crypto_arb',
                    'symbol': opp.symbol,
                    'buy_exchange': opp.buy_exchange,
                    'sell_exchange': opp.sell_exchange,
                    'spread_pct': opp.spread_pct,
                    'profit_pct': opp.potential_profit_pct,
                    'confidence': opp.confidence,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            
            logger.info(f"Arb scan complete: {len(results)} opportunities")
            return results
            
        except Exception as e:
            logger.error(f"Arb scan error: {e}")
            return []
    
    def run_all_scans(self) -> Dict[str, List]:
        """Run all strategy scans"""
        return {
            'technical': self.run_technical_scan(),
            'earnings': self.run_earnings_scan(),
            'sentiment': self.run_sentiment_scan(),
            'arbitrage': self.run_arb_scan()
        }
    
    def get_status(self) -> Dict:
        """Get status of all strategies"""
        status = {
            'total_strategies': len(self.strategies),
            'enabled': sum(1 for s in self.strategies.values() if s['enabled']),
            'paper_mode': self.paper_mode,
            'strategies': {}
        }
        
        for name, config in self.strategies.items():
            status['strategies'][name] = {
                'enabled': config['enabled'],
                'category': config['category'],
                'schedule': config['schedule'],
                'loaded': config['instance'] is not None
            }
        
        return status


def integrate_with_scheduler():
    """
    Integration code for unified_scheduler.py
    
    Add these methods to UnifiedEventScheduler class:
    """
    
    integration_code = '''
    # ==================== EXTENDED STRATEGY INTEGRATION ====================
    
    def init_extended_strategies(self):
        """Initialize extended strategy manager"""
        from config.strategy_integration import StrategyManager
        self.strategy_manager = StrategyManager(paper_mode=self.paper_mode)
        logger.info(f"Extended strategies initialized: {self.strategy_manager.get_status()['enabled']} enabled")
    
    def run_extended_technical_scan(self):
        """Run extended technical analysis scan"""
        if not hasattr(self, 'strategy_manager'):
            self.init_extended_strategies()
        
        signals = self.strategy_manager.run_technical_scan()
        
        # Log high-confidence signals
        for sig in signals:
            if sig.get('confidence', 0) >= 0.7:
                logger.info(f"[SIGNAL] {sig['strategy']}: {sig['signal'].upper()} {sig['symbol']} "
                           f"(conf={sig['confidence']:.0%})")
                
                # Send Telegram alert
                self.send_telegram_message(
                    f"<b>[{sig['strategy'].upper()}]</b> {sig['signal'].upper()} {sig['symbol']}\\n"
                    f"Confidence: {sig['confidence']:.0%}\\n"
                    f"Reasoning: {sig.get('reasoning', 'N/A')}"
                )
        
        return signals
    
    def run_crypto_arb_scan(self):
        """Run crypto arbitrage scan"""
        if not hasattr(self, 'strategy_manager'):
            self.init_extended_strategies()
        
        opportunities = self.strategy_manager.run_arb_scan()
        
        for opp in opportunities:
            if opp.get('spread_pct', 0) >= 0.01:  # 1% spread
                logger.info(f"[ARB] {opp['symbol']}: {opp['spread_pct']:.2%} spread "
                           f"({opp['buy_exchange']} -> {opp['sell_exchange']})")
        
        return opportunities
    
    # Add to setup_schedule():
    # schedule.every().day.at("09:35").do(self.run_extended_technical_scan)
    # schedule.every().day.at("14:00").do(self.run_extended_technical_scan)
    # schedule.every(5).minutes.do(self.run_crypto_arb_scan)
    '''
    
    return integration_code


if __name__ == "__main__":
    print("=" * 60)
    print("STRATEGY INTEGRATION MODULE")
    print("=" * 60)
    
    manager = StrategyManager(paper_mode=True)
    
    print("\nLoaded Strategies:")
    status = manager.get_status()
    for name, info in status['strategies'].items():
        emoji = "✅" if info['enabled'] else "❌"
        print(f"  {emoji} {name}: {info['category']} - {info['schedule']}")
    
    print(f"\nTotal: {status['enabled']}/{status['total_strategies']} enabled")
    
    # Run a quick scan
    print("\n" + "=" * 60)
    print("RUNNING TEST SCAN")
    print("=" * 60)
    
    results = manager.run_technical_scan(['SPY', 'QQQ'])
    
    if results:
        print(f"\nFound {len(results)} signals:")
        for sig in results:
            print(f"  [{sig['strategy']}] {sig['signal'].upper()} {sig['symbol']} "
                  f"(conf={sig.get('confidence', 0):.0%})")
    else:
        print("\nNo signals at current time (markets may be closed)")
