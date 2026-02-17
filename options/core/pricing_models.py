"""
Options Pricing Models

Implementation of various options pricing models including:
- Black-Scholes-Merton model
- Binomial/Cox-Ross-Rubinstein model
- Monte Carlo simulation
- American option pricing (Binomial)
- Implied volatility calculation with multiple methods
"""

import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import brentq, newton
from typing import Union, Optional, Tuple
from decimal import Decimal, getcontext
from dataclasses import dataclass
import logging

# Set decimal precision for financial calculations
getcontext().prec = 28

logger = logging.getLogger(__name__)


@dataclass
class OptionPrice:
    """Container for option pricing results"""
    price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

    # Model-specific information
    model_type: str = "unknown"
    convergence_steps: Optional[int] = None
    calculation_time: Optional[float] = None


class BlackScholes:
    """
    Black-Scholes-Merton option pricing model

    Handles European options with dividends.
    """

    @staticmethod
    def price(
        spot_price: Union[float, Decimal],
        strike_price: Union[float, Decimal],
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str = 'call',
        dividend_yield: float = 0.0
    ) -> float:
        """
        Calculate Black-Scholes option price

        Args:
            spot_price: Current underlying price
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate
            volatility: Volatility (annualized)
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield (annualized)

        Returns:
            Option price
        """
        try:
            S = float(spot_price)
            K = float(strike_price)
            T = time_to_expiry
            r = risk_free_rate
            sigma = volatility
            q = dividend_yield

            # Handle edge cases
            if T <= 0:
                if option_type.lower() == 'call':
                    return max(S - K, 0.0)
                else:
                    return max(K - S, 0.0)

            if sigma <= 0:
                sigma = 1e-6

            # Calculate d1 and d2
            d1, d2 = BlackScholes._calculate_d_values(S, K, T, r, sigma, q)

            # Calculate option price
            if option_type.lower() == 'call':
                price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

            return max(price, 0.0)

        except Exception as e:
            logger.error(f"Error in Black-Scholes pricing: {e}")
            return 0.0

    @staticmethod
    def _calculate_d_values(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> Tuple[float, float]:
        """Calculate d1 and d2 values"""
        sqrt_T = math.sqrt(T)

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        return d1, d2

    @staticmethod
    def delta(
        spot_price: Union[float, Decimal],
        strike_price: Union[float, Decimal],
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str = 'call',
        dividend_yield: float = 0.0
    ) -> float:
        """Calculate option Delta"""
        try:
            S = float(spot_price)
            K = float(strike_price)
            T = time_to_expiry
            r = risk_free_rate
            sigma = volatility
            q = dividend_yield

            if T <= 0:
                if option_type.lower() == 'call':
                    return 1.0 if S > K else 0.0
                else:
                    return -1.0 if S < K else 0.0

            d1, _ = BlackScholes._calculate_d_values(S, K, T, r, sigma, q)

            if option_type.lower() == 'call':
                return math.exp(-q * T) * norm.cdf(d1)
            else:
                return -math.exp(-q * T) * norm.cdf(-d1)

        except Exception as e:
            logger.error(f"Error calculating Delta: {e}")
            return 0.0


class BinomialModel:
    """
    Binomial (Cox-Ross-Rubinstein) option pricing model

    Can handle American and European options.
    """

    def __init__(self, steps: int = 100):
        """
        Initialize binomial model

        Args:
            steps: Number of time steps
        """
        self.steps = steps

    def price(
        self,
        spot_price: Union[float, Decimal],
        strike_price: Union[float, Decimal],
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str = 'call',
        exercise_style: str = 'european',
        dividend_yield: float = 0.0
    ) -> OptionPrice:
        """
        Price option using binomial model

        Args:
            spot_price: Current underlying price
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate
            volatility: Volatility (annualized)
            option_type: 'call' or 'put'
            exercise_style: 'european' or 'american'
            dividend_yield: Dividend yield (annualized)

        Returns:
            OptionPrice object with price and Greeks
        """
        try:
            S = float(spot_price)
            K = float(strike_price)
            T = time_to_expiry
            r = risk_free_rate
            sigma = volatility
            q = dividend_yield

            if T <= 0:
                intrinsic = self._intrinsic_value(S, K, option_type)
                return OptionPrice(price=intrinsic, model_type="binomial", convergence_steps=0)

            # Calculate parameters
            dt = T / self.steps
            u = math.exp(sigma * math.sqrt(dt))  # Up factor
            d = 1 / u                            # Down factor
            p = (math.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability
            discount = math.exp(-r * dt)

            # Initialize price tree at expiry
            prices = np.zeros(self.steps + 1)
            option_values = np.zeros(self.steps + 1)

            # Calculate underlying prices at expiry
            for i in range(self.steps + 1):
                prices[i] = S * (u ** (self.steps - i)) * (d ** i)
                option_values[i] = self._intrinsic_value(prices[i], K, option_type)

            # Work backwards through the tree
            for j in range(self.steps - 1, -1, -1):
                for i in range(j + 1):
                    # Expected option value
                    expected_value = discount * (p * option_values[i] + (1 - p) * option_values[i + 1])

                    if exercise_style.lower() == 'american':
                        # For American options, check early exercise
                        current_price = S * (u ** (j - i)) * (d ** i)
                        intrinsic = self._intrinsic_value(current_price, K, option_type)
                        option_values[i] = max(expected_value, intrinsic)
                    else:
                        # European option
                        option_values[i] = expected_value

            price = option_values[0]

            # Calculate Greeks using finite differences
            delta = self._calculate_delta_binomial(S, K, T, r, sigma, q, option_type, exercise_style)
            gamma = self._calculate_gamma_binomial(S, K, T, r, sigma, q, option_type, exercise_style)

            return OptionPrice(
                price=price,
                delta=delta,
                gamma=gamma,
                model_type="binomial",
                convergence_steps=self.steps
            )

        except Exception as e:
            logger.error(f"Error in binomial pricing: {e}")
            return OptionPrice(price=0.0, model_type="binomial")

    def _intrinsic_value(self, spot: float, strike: float, option_type: str) -> float:
        """Calculate intrinsic value"""
        if option_type.lower() == 'call':
            return max(spot - strike, 0.0)
        else:
            return max(strike - spot, 0.0)

    def _calculate_delta_binomial(self, S: float, K: float, T: float, r: float,
                                 sigma: float, q: float, option_type: str, exercise_style: str) -> float:
        """Calculate Delta using finite differences"""
        try:
            h = S * 0.01  # 1% price change
            price_up = self.price(S + h, K, T, r, sigma, option_type, exercise_style, q).price
            price_down = self.price(S - h, K, T, r, sigma, option_type, exercise_style, q).price
            return (price_up - price_down) / (2 * h)
        except Exception as e:
            logger.debug(f"Error calculating delta: {e}")
            return 0.0

    def _calculate_gamma_binomial(self, S: float, K: float, T: float, r: float,
                                 sigma: float, q: float, option_type: str, exercise_style: str) -> float:
        """Calculate Gamma using finite differences"""
        try:
            h = S * 0.01  # 1% price change
            price_current = self.price(S, K, T, r, sigma, option_type, exercise_style, q).price
            price_up = self.price(S + h, K, T, r, sigma, option_type, exercise_style, q).price
            price_down = self.price(S - h, K, T, r, sigma, option_type, exercise_style, q).price
            return (price_up - 2 * price_current + price_down) / (h ** 2)
        except Exception as e:
            logger.debug(f"Error calculating gamma: {e}")
            return 0.0


class MonteCarloModel:
    """
    Monte Carlo option pricing model

    Flexible pricing method that can handle complex payoffs and path-dependent options.
    """

    def __init__(self, simulations: int = 100000, time_steps: int = 252):
        """
        Initialize Monte Carlo model

        Args:
            simulations: Number of Monte Carlo simulations
            time_steps: Number of time steps per simulation
        """
        self.simulations = simulations
        self.time_steps = time_steps
        np.random.seed(42)  # For reproducible results

    def price(
        self,
        spot_price: Union[float, Decimal],
        strike_price: Union[float, Decimal],
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str = 'call',
        dividend_yield: float = 0.0,
        barrier: Optional[float] = None,
        barrier_type: str = 'none'
    ) -> OptionPrice:
        """
        Price option using Monte Carlo simulation

        Args:
            spot_price: Current underlying price
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate
            volatility: Volatility (annualized)
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield (annualized)
            barrier: Barrier level for barrier options
            barrier_type: 'knock_in', 'knock_out', or 'none'

        Returns:
            OptionPrice object
        """
        try:
            S = float(spot_price)
            K = float(strike_price)
            T = time_to_expiry
            r = risk_free_rate
            sigma = volatility
            q = dividend_yield

            if T <= 0:
                intrinsic = max(S - K, 0.0) if option_type.lower() == 'call' else max(K - S, 0.0)
                return OptionPrice(price=intrinsic, model_type="monte_carlo")

            dt = T / self.time_steps
            discount_factor = math.exp(-r * T)

            payoffs = []

            for _ in range(self.simulations):
                # Generate price path
                path = self._generate_price_path(S, r, q, sigma, T, dt)

                # Calculate payoff
                final_price = path[-1]

                # Check barrier conditions
                if barrier_type.lower() == 'knock_out' and barrier:
                    if option_type.lower() == 'call' and any(p >= barrier for p in path):
                        payoff = 0.0
                    elif option_type.lower() == 'put' and any(p <= barrier for p in path):
                        payoff = 0.0
                    else:
                        payoff = self._calculate_payoff(final_price, K, option_type)
                elif barrier_type.lower() == 'knock_in' and barrier:
                    if option_type.lower() == 'call' and any(p >= barrier for p in path):
                        payoff = self._calculate_payoff(final_price, K, option_type)
                    elif option_type.lower() == 'put' and any(p <= barrier for p in path):
                        payoff = self._calculate_payoff(final_price, K, option_type)
                    else:
                        payoff = 0.0
                else:
                    # Standard European option
                    payoff = self._calculate_payoff(final_price, K, option_type)

                payoffs.append(payoff)

            # Calculate option price
            average_payoff = np.mean(payoffs)
            option_price = discount_factor * average_payoff

            # Calculate standard error
            std_error = np.std(payoffs) / math.sqrt(self.simulations)
            confidence_interval = 1.96 * std_error * discount_factor

            return OptionPrice(
                price=option_price,
                model_type="monte_carlo",
                convergence_steps=self.simulations
            )

        except Exception as e:
            logger.error(f"Error in Monte Carlo pricing: {e}")
            return OptionPrice(price=0.0, model_type="monte_carlo")

    def _generate_price_path(self, S0: float, r: float, q: float, sigma: float, T: float, dt: float) -> np.ndarray:
        """Generate stock price path using geometric Brownian motion"""
        n_steps = int(T / dt)
        path = np.zeros(n_steps + 1)
        path[0] = S0

        # Generate random numbers
        random_numbers = np.random.normal(0, 1, n_steps)

        for i in range(1, n_steps + 1):
            drift = (r - q - 0.5 * sigma**2) * dt
            diffusion = sigma * math.sqrt(dt) * random_numbers[i - 1]
            path[i] = path[i - 1] * math.exp(drift + diffusion)

        return path

    def _calculate_payoff(self, final_price: float, strike: float, option_type: str) -> float:
        """Calculate option payoff"""
        if option_type.lower() == 'call':
            return max(final_price - strike, 0.0)
        else:
            return max(strike - final_price, 0.0)


class ImpliedVolatilityCalculator:
    """
    Advanced implied volatility calculator with multiple methods
    """

    @staticmethod
    def calculate_iv_brent(
        market_price: float,
        spot_price: Union[float, Decimal],
        strike_price: Union[float, Decimal],
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: str = 'call',
        dividend_yield: float = 0.0,
        vol_bounds: Tuple[float, float] = (0.01, 5.0)
    ) -> Optional[float]:
        """
        Calculate implied volatility using Brent's method

        More robust than Newton-Raphson for extreme cases.
        """
        try:
            def objective(vol):
                bs_price = BlackScholes.price(
                    spot_price, strike_price, time_to_expiry,
                    risk_free_rate, vol, option_type, dividend_yield
                )
                return bs_price - market_price

            # Check bounds
            price_low = BlackScholes.price(
                spot_price, strike_price, time_to_expiry,
                risk_free_rate, vol_bounds[0], option_type, dividend_yield
            )
            price_high = BlackScholes.price(
                spot_price, strike_price, time_to_expiry,
                risk_free_rate, vol_bounds[1], option_type, dividend_yield
            )

            if (price_low - market_price) * (price_high - market_price) > 0:
                logger.warning(f"Market price {market_price} outside theoretical bounds [{price_low}, {price_high}]")
                return None

            iv = brentq(objective, vol_bounds[0], vol_bounds[1], xtol=1e-6)
            return iv

        except Exception as e:
            logger.error(f"Error calculating IV with Brent method: {e}")
            return None

    @staticmethod
    def calculate_iv_newton(
        market_price: float,
        spot_price: Union[float, Decimal],
        strike_price: Union[float, Decimal],
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: str = 'call',
        dividend_yield: float = 0.0,
        initial_guess: float = 0.3,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method
        """
        try:
            S = float(spot_price)
            K = float(strike_price)
            T = time_to_expiry
            r = risk_free_rate
            q = dividend_yield

            vol = initial_guess

            for i in range(max_iterations):
                # Calculate price and vega
                price = BlackScholes.price(S, K, T, r, vol, option_type, q)
                price_diff = price - market_price

                if abs(price_diff) < tolerance:
                    return vol

                # Calculate vega
                d1, _ = BlackScholes._calculate_d_values(S, K, T, r, vol, q)
                vega = S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)

                if abs(vega) < 1e-10:
                    break

                # Newton-Raphson update
                vol_new = vol - price_diff / vega

                # Keep vol positive and reasonable
                vol_new = max(0.001, min(5.0, vol_new))

                if abs(vol_new - vol) < tolerance:
                    return vol_new

                vol = vol_new

            logger.warning(f"IV Newton method did not converge after {max_iterations} iterations")
            return None

        except Exception as e:
            logger.error(f"Error calculating IV with Newton method: {e}")
            return None

    @staticmethod
    def calculate_iv_adaptive(
        market_price: float,
        spot_price: Union[float, Decimal],
        strike_price: Union[float, Decimal],
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: str = 'call',
        dividend_yield: float = 0.0
    ) -> Optional[float]:
        """
        Adaptive IV calculation that tries multiple methods
        """
        # Try Newton-Raphson first (fastest)
        iv = ImpliedVolatilityCalculator.calculate_iv_newton(
            market_price, spot_price, strike_price, time_to_expiry,
            risk_free_rate, option_type, dividend_yield
        )

        if iv is not None:
            return iv

        # Fall back to Brent's method (more robust)
        logger.info("Newton-Raphson failed, trying Brent's method")
        iv = ImpliedVolatilityCalculator.calculate_iv_brent(
            market_price, spot_price, strike_price, time_to_expiry,
            risk_free_rate, option_type, dividend_yield
        )

        return iv


class VolatilitySmile:
    """
    Volatility smile modeling and analysis
    """

    def __init__(self):
        self.strikes = []
        self.implied_vols = []
        self.spot_price = None
        self.expiry = None

    def fit_smile(self, option_chain_data: list, spot_price: float, expiry_date: str):
        """
        Fit volatility smile from option chain data

        Args:
            option_chain_data: List of option contracts with IV data
            spot_price: Current spot price
            expiry_date: Expiry date string
        """
        self.spot_price = spot_price
        self.expiry = expiry_date

        strikes = []
        ivs = []

        for contract in option_chain_data:
            if (contract.expiry.strftime('%Y-%m-%d') == expiry_date and
                contract.implied_volatility is not None and
                contract.implied_volatility > 0):
                strikes.append(float(contract.strike))
                ivs.append(contract.implied_volatility)

        # Sort by strike
        sorted_data = sorted(zip(strikes, ivs))
        self.strikes = [x[0] for x in sorted_data]
        self.implied_vols = [x[1] for x in sorted_data]

    def interpolate_iv(self, strike: float) -> Optional[float]:
        """Interpolate implied volatility for a given strike"""
        if len(self.strikes) < 2:
            return None

        if strike <= self.strikes[0]:
            return self.implied_vols[0]
        elif strike >= self.strikes[-1]:
            return self.implied_vols[-1]
        else:
            # Linear interpolation
            for i in range(len(self.strikes) - 1):
                if self.strikes[i] <= strike <= self.strikes[i + 1]:
                    t = (strike - self.strikes[i]) / (self.strikes[i + 1] - self.strikes[i])
                    return self.implied_vols[i] * (1 - t) + self.implied_vols[i + 1] * t

        return None

    def get_atm_iv(self) -> Optional[float]:
        """Get at-the-money implied volatility"""
        if not self.spot_price or not self.strikes:
            return None

        return self.interpolate_iv(self.spot_price)

    def calculate_skew(self) -> Optional[float]:
        """Calculate volatility skew (25-delta put IV - 25-delta call IV)"""
        # Simplified skew calculation using strikes relative to ATM
        if len(self.strikes) < 3 or not self.spot_price:
            return None

        atm_iv = self.get_atm_iv()
        if atm_iv is None:
            return None

        # Find strikes approximately 25 delta
        otm_put_strike = self.spot_price * 0.9  # Approximate
        otm_call_strike = self.spot_price * 1.1  # Approximate

        otm_put_iv = self.interpolate_iv(otm_put_strike)
        otm_call_iv = self.interpolate_iv(otm_call_strike)

        if otm_put_iv and otm_call_iv:
            return otm_put_iv - otm_call_iv

        return None

    def calculate_term_structure_slope(self, other_smile: 'VolatilitySmile') -> Optional[float]:
        """Calculate term structure slope between two expiries"""
        atm_iv_1 = self.get_atm_iv()
        atm_iv_2 = other_smile.get_atm_iv()

        if atm_iv_1 and atm_iv_2:
            return atm_iv_2 - atm_iv_1

        return None