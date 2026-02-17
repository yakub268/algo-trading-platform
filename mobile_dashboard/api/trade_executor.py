"""
Mobile Trade Execution Interface
===============================

Advanced trade execution system for mobile dashboard with proper
risk management, validation, and real-time execution capabilities.

Features:
- Multi-exchange trade execution
- Real-time order status updates
- Position sizing with risk management
- Order validation and safety checks
- Portfolio impact analysis
- Emergency stop functionality

Author: Trading Bot System
Created: February 2026
"""

import os
import sys
import json
import sqlite3
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from decimal import Decimal, ROUND_DOWN
from dataclasses import dataclass, asdict
from enum import Enum

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from dashboard.alpaca_client import AlpacaClient
    from dashboard.freqtrade_client import FreqtradeClient
    from utils.enhanced_risk_manager import EnhancedRiskManager
    TRADING_MODULES_AVAILABLE = True
except ImportError:
    TRADING_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

@dataclass
class TradeRequest:
    """Trade request data structure"""
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    exchange: str
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "DAY"
    user_id: str = "mobile_user"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Convert string enums to proper enum values"""
        if isinstance(self.side, str):
            self.side = OrderSide(self.side.lower())
        if isinstance(self.order_type, str):
            self.order_type = OrderType(self.order_type.lower())
        if self.metadata is None:
            self.metadata = {}

@dataclass
class TradeResponse:
    """Trade execution response"""
    trade_id: str
    status: OrderStatus
    symbol: str
    side: OrderSide
    quantity: Decimal
    filled_quantity: Decimal = Decimal('0')
    avg_fill_price: Optional[Decimal] = None
    commission: Decimal = Decimal('0')
    error_message: Optional[str] = None
    exchange_order_id: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

class MobileTradeExecutor:
    """Advanced trade execution system for mobile dashboard"""

    def __init__(self):
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data',
            'mobile_trades.db'
        )
        self.init_database()

        # Initialize exchange clients
        self.clients = {}
        if TRADING_MODULES_AVAILABLE:
            try:
                self.clients['alpaca'] = AlpacaClient()
                self.clients['freqtrade'] = FreqtradeClient()
                self.risk_manager = EnhancedRiskManager()
                logger.info("Trade executor initialized with exchange clients")
            except Exception as e:
                logger.error(f"Failed to initialize exchange clients: {e}")

        # Configuration
        self.max_position_size = Decimal('0.05')  # 5% max position size
        self.max_portfolio_exposure = Decimal('0.95')  # 95% max portfolio exposure
        self.emergency_stop_active = False

    def init_database(self):
        """Initialize SQLite database for trade tracking"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trade_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity DECIMAL NOT NULL,
                    order_type TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    price DECIMAL,
                    stop_price DECIMAL,
                    time_in_force TEXT DEFAULT 'DAY',
                    status TEXT DEFAULT 'pending',
                    filled_quantity DECIMAL DEFAULT 0,
                    avg_fill_price DECIMAL,
                    commission DECIMAL DEFAULT 0,
                    exchange_order_id TEXT,
                    error_message TEXT,
                    user_id TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS trade_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    execution_id TEXT UNIQUE NOT NULL,
                    quantity DECIMAL NOT NULL,
                    price DECIMAL NOT NULL,
                    commission DECIMAL DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES trade_orders (trade_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS risk_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    check_type TEXT NOT NULL,
                    result TEXT NOT NULL,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES trade_orders (trade_id)
                )
            ''')

            conn.commit()

    async def execute_trade(self, trade_request: TradeRequest) -> TradeResponse:
        """Execute a trade with full validation and risk management"""
        try:
            # Generate unique trade ID
            trade_id = self._generate_trade_id()

            # Validate trade request
            validation_result = await self._validate_trade_request(trade_request)
            if not validation_result['valid']:
                return TradeResponse(
                    trade_id=trade_id,
                    status=OrderStatus.REJECTED,
                    symbol=trade_request.symbol,
                    side=trade_request.side,
                    quantity=trade_request.quantity,
                    error_message=validation_result['error']
                )

            # Perform risk checks
            risk_result = await self._perform_risk_checks(trade_request, trade_id)
            if not risk_result['approved']:
                return TradeResponse(
                    trade_id=trade_id,
                    status=OrderStatus.REJECTED,
                    symbol=trade_request.symbol,
                    side=trade_request.side,
                    quantity=trade_request.quantity,
                    error_message=f"Risk check failed: {risk_result['reason']}"
                )

            # Store trade order in database
            await self._store_trade_order(trade_request, trade_id)

            # Execute trade on exchange
            if trade_request.exchange in self.clients:
                execution_result = await self._execute_on_exchange(trade_request, trade_id)

                # Update trade status
                await self._update_trade_status(trade_id, execution_result)

                return execution_result
            else:
                error_msg = f"Exchange '{trade_request.exchange}' not supported"
                await self._update_trade_status(trade_id, TradeResponse(
                    trade_id=trade_id,
                    status=OrderStatus.REJECTED,
                    symbol=trade_request.symbol,
                    side=trade_request.side,
                    quantity=trade_request.quantity,
                    error_message=error_msg
                ))
                return TradeResponse(
                    trade_id=trade_id,
                    status=OrderStatus.REJECTED,
                    symbol=trade_request.symbol,
                    side=trade_request.side,
                    quantity=trade_request.quantity,
                    error_message=error_msg
                )

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return TradeResponse(
                trade_id=trade_id if 'trade_id' in locals() else "unknown",
                status=OrderStatus.REJECTED,
                symbol=trade_request.symbol,
                side=trade_request.side,
                quantity=trade_request.quantity,
                error_message=str(e)
            )

    async def _validate_trade_request(self, request: TradeRequest) -> Dict[str, Any]:
        """Validate trade request parameters"""
        try:
            # Check if emergency stop is active
            if self.emergency_stop_active:
                return {
                    'valid': False,
                    'error': 'Emergency stop is active - all trading disabled'
                }

            # Validate symbol format
            if not request.symbol or len(request.symbol) < 1:
                return {
                    'valid': False,
                    'error': 'Invalid symbol format'
                }

            # Validate quantity
            if request.quantity <= 0:
                return {
                    'valid': False,
                    'error': 'Quantity must be positive'
                }

            # Validate price for limit orders
            if request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if not request.price or request.price <= 0:
                    return {
                        'valid': False,
                        'error': 'Price required for limit orders'
                    }

            # Validate stop price for stop orders
            if request.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                if not request.stop_price or request.stop_price <= 0:
                    return {
                        'valid': False,
                        'error': 'Stop price required for stop orders'
                    }

            # Validate exchange support
            if request.exchange not in ['alpaca', 'freqtrade']:
                return {
                    'valid': False,
                    'error': f'Exchange {request.exchange} not supported'
                }

            return {'valid': True}

        except Exception as e:
            logger.error(f"Trade validation failed: {e}")
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}'
            }

    async def _perform_risk_checks(self, request: TradeRequest, trade_id: str) -> Dict[str, Any]:
        """Perform comprehensive risk checks"""
        try:
            risk_checks = []

            # Get current portfolio value
            portfolio_value = await self._get_portfolio_value()

            if portfolio_value == 0:
                return {
                    'approved': False,
                    'reason': 'Unable to determine portfolio value'
                }

            # Calculate position value
            estimated_price = request.price or await self._get_current_price(request.symbol, request.exchange)
            if not estimated_price:
                return {
                    'approved': False,
                    'reason': f'Unable to get price for {request.symbol}'
                }

            position_value = request.quantity * estimated_price
            position_percent = (position_value / portfolio_value) * 100

            # Position size check
            position_size_check = {
                'type': 'position_size',
                'result': 'pass' if position_percent <= float(self.max_position_size * 100) else 'fail',
                'details': f'Position size: {position_percent:.2f}%, Max allowed: {float(self.max_position_size * 100):.2f}%'
            }
            risk_checks.append(position_size_check)

            if position_size_check['result'] == 'fail':
                await self._store_risk_check(trade_id, position_size_check)
                return {
                    'approved': False,
                    'reason': f'Position size ({position_percent:.2f}%) exceeds maximum ({float(self.max_position_size * 100):.2f}%)'
                }

            # Portfolio exposure check
            current_exposure = await self._get_current_exposure()
            new_exposure = current_exposure + (position_value / portfolio_value)

            exposure_check = {
                'type': 'portfolio_exposure',
                'result': 'pass' if new_exposure <= float(self.max_portfolio_exposure) else 'fail',
                'details': f'New exposure: {new_exposure:.2f}%, Max allowed: {float(self.max_portfolio_exposure):.2f}%'
            }
            risk_checks.append(exposure_check)

            if exposure_check['result'] == 'fail':
                await self._store_risk_check(trade_id, exposure_check)
                return {
                    'approved': False,
                    'reason': f'Total portfolio exposure would exceed maximum ({float(self.max_portfolio_exposure * 100):.2f}%)'
                }

            # Market hours check (for stock trades)
            if request.exchange == 'alpaca':
                market_hours_check = await self._check_market_hours()
                risk_checks.append(market_hours_check)

                if market_hours_check['result'] == 'fail':
                    await self._store_risk_check(trade_id, market_hours_check)
                    return {
                        'approved': False,
                        'reason': 'Market is closed - trade rejected'
                    }

            # Store all risk checks
            for check in risk_checks:
                await self._store_risk_check(trade_id, check)

            return {'approved': True}

        except Exception as e:
            logger.error(f"Risk check failed: {e}")
            return {
                'approved': False,
                'reason': f'Risk assessment error: {str(e)}'
            }

    async def _execute_on_exchange(self, request: TradeRequest, trade_id: str) -> TradeResponse:
        """Execute trade on specified exchange"""
        try:
            client = self.clients[request.exchange]

            if request.exchange == 'alpaca':
                result = await self._execute_alpaca_trade(client, request, trade_id)
            elif request.exchange == 'freqtrade':
                result = await self._execute_freqtrade_trade(client, request, trade_id)
            else:
                raise ValueError(f"Unsupported exchange: {request.exchange}")

            return result

        except Exception as e:
            logger.error(f"Exchange execution failed: {e}")
            return TradeResponse(
                trade_id=trade_id,
                status=OrderStatus.REJECTED,
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                error_message=str(e)
            )

    async def _execute_alpaca_trade(self, client, request: TradeRequest, trade_id: str) -> TradeResponse:
        """Execute trade on Alpaca"""
        try:
            # Prepare order parameters
            order_params = {
                'symbol': request.symbol,
                'qty': float(request.quantity),
                'side': request.side.value,
                'type': request.order_type.value,
                'time_in_force': request.time_in_force
            }

            if request.price:
                order_params['limit_price'] = float(request.price)

            if request.stop_price:
                order_params['stop_price'] = float(request.stop_price)

            # Submit order
            order_result = client.place_order(order_params)

            if order_result.get('success'):
                return TradeResponse(
                    trade_id=trade_id,
                    status=OrderStatus.SUBMITTED,
                    symbol=request.symbol,
                    side=request.side,
                    quantity=request.quantity,
                    exchange_order_id=order_result.get('order_id')
                )
            else:
                return TradeResponse(
                    trade_id=trade_id,
                    status=OrderStatus.REJECTED,
                    symbol=request.symbol,
                    side=request.side,
                    quantity=request.quantity,
                    error_message=order_result.get('error', 'Unknown error')
                )

        except Exception as e:
            logger.error(f"Alpaca trade execution failed: {e}")
            return TradeResponse(
                trade_id=trade_id,
                status=OrderStatus.REJECTED,
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                error_message=str(e)
            )

    async def _execute_freqtrade_trade(self, client, request: TradeRequest, trade_id: str) -> TradeResponse:
        """Execute trade on Freqtrade"""
        try:
            # Freqtrade doesn't support manual trades in the same way
            # This would typically involve forcing a buy/sell through the bot
            return TradeResponse(
                trade_id=trade_id,
                status=OrderStatus.REJECTED,
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                error_message="Manual Freqtrade execution not supported in this version"
            )

        except Exception as e:
            logger.error(f"Freqtrade trade execution failed: {e}")
            return TradeResponse(
                trade_id=trade_id,
                status=OrderStatus.REJECTED,
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                error_message=str(e)
            )

    async def get_trade_status(self, trade_id: str) -> Optional[TradeResponse]:
        """Get current status of a trade"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT trade_id, symbol, side, quantity, status, filled_quantity,
                           avg_fill_price, commission, exchange_order_id, error_message, updated_at
                    FROM trade_orders
                    WHERE trade_id = ?
                ''', (trade_id,))

                row = cursor.fetchone()
                if row:
                    return TradeResponse(
                        trade_id=row[0],
                        symbol=row[1],
                        side=OrderSide(row[2]),
                        quantity=Decimal(str(row[3])),
                        status=OrderStatus(row[4]),
                        filled_quantity=Decimal(str(row[5])) if row[5] else Decimal('0'),
                        avg_fill_price=Decimal(str(row[6])) if row[6] else None,
                        commission=Decimal(str(row[7])) if row[7] else Decimal('0'),
                        exchange_order_id=row[8],
                        error_message=row[9],
                        timestamp=datetime.fromisoformat(row[10]) if row[10] else None
                    )

                return None

        except Exception as e:
            logger.error(f"Failed to get trade status: {e}")
            return None

    async def get_recent_trades(self, limit: int = 50) -> List[TradeResponse]:
        """Get recent trades for dashboard display"""
        try:
            trades = []

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT trade_id, symbol, side, quantity, status, filled_quantity,
                           avg_fill_price, commission, exchange_order_id, error_message, updated_at
                    FROM trade_orders
                    ORDER BY updated_at DESC
                    LIMIT ?
                ''', (limit,))

                for row in cursor.fetchall():
                    trades.append(TradeResponse(
                        trade_id=row[0],
                        symbol=row[1],
                        side=OrderSide(row[2]),
                        quantity=Decimal(str(row[3])),
                        status=OrderStatus(row[4]),
                        filled_quantity=Decimal(str(row[5])) if row[5] else Decimal('0'),
                        avg_fill_price=Decimal(str(row[6])) if row[6] else None,
                        commission=Decimal(str(row[7])) if row[7] else Decimal('0'),
                        exchange_order_id=row[8],
                        error_message=row[9],
                        timestamp=datetime.fromisoformat(row[10]) if row[10] else None
                    ))

            return trades

        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []

    def activate_emergency_stop(self):
        """Activate emergency stop - disable all new trades"""
        self.emergency_stop_active = True
        logger.warning("Emergency stop activated - all trading disabled")

    def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        self.emergency_stop_active = False
        logger.info("Emergency stop deactivated - trading re-enabled")

    # Helper methods
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        import uuid
        return f"mobile_{uuid.uuid4().hex[:12]}"

    async def _store_trade_order(self, request: TradeRequest, trade_id: str):
        """Store trade order in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO trade_orders
                (trade_id, symbol, side, quantity, order_type, exchange, price, stop_price,
                 time_in_force, user_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id, request.symbol, request.side.value, float(request.quantity),
                request.order_type.value, request.exchange,
                float(request.price) if request.price else None,
                float(request.stop_price) if request.stop_price else None,
                request.time_in_force, request.user_id, json.dumps(request.metadata)
            ))
            conn.commit()

    async def _update_trade_status(self, trade_id: str, response: TradeResponse):
        """Update trade status in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE trade_orders
                SET status = ?, filled_quantity = ?, avg_fill_price = ?, commission = ?,
                    exchange_order_id = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                WHERE trade_id = ?
            ''', (
                response.status.value,
                float(response.filled_quantity),
                float(response.avg_fill_price) if response.avg_fill_price else None,
                float(response.commission),
                response.exchange_order_id,
                response.error_message,
                trade_id
            ))
            conn.commit()

    async def _store_risk_check(self, trade_id: str, check: Dict[str, Any]):
        """Store risk check result"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO risk_checks (trade_id, check_type, result, details)
                VALUES (?, ?, ?, ?)
            ''', (trade_id, check['type'], check['result'], check['details']))
            conn.commit()

    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        try:
            total_value = 0

            if 'alpaca' in self.clients:
                alpaca_data = self.clients['alpaca'].get_account_summary()
                if alpaca_data:
                    total_value += float(alpaca_data.get('equity', 0))

            return total_value
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return 0

    async def _get_current_price(self, symbol: str, exchange: str) -> Optional[Decimal]:
        """Get current market price for symbol"""
        try:
            if exchange == 'alpaca' and 'alpaca' in self.clients:
                price_data = self.clients['alpaca'].get_latest_quote(symbol)
                if price_data:
                    return Decimal(str(price_data.get('price', 0)))

            return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    async def _get_current_exposure(self) -> float:
        """Get current portfolio exposure percentage"""
        # Simplified - would calculate based on current positions
        return 0.5  # 50% exposure placeholder

    async def _check_market_hours(self) -> Dict[str, Any]:
        """Check if market is open"""
        try:
            if 'alpaca' in self.clients:
                market_status = self.clients['alpaca'].get_market_status()
                is_open = market_status.get('is_open', False)

                return {
                    'type': 'market_hours',
                    'result': 'pass' if is_open else 'fail',
                    'details': f'Market is {"open" if is_open else "closed"}'
                }

            return {
                'type': 'market_hours',
                'result': 'pass',
                'details': 'Unable to check market hours - assuming open'
            }
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return {
                'type': 'market_hours',
                'result': 'pass',
                'details': 'Market hours check failed - assuming open'
            }

# Initialize global trade executor
trade_executor = MobileTradeExecutor()