"""
Greeks Calculator for Options

Provides real-time calculation of option Greeks:
- Delta: Price sensitivity to underlying price changes
- Gamma: Delta sensitivity to underlying price changes
- Theta: Time decay
- Vega: Volatility sensitivity
- Rho: Interest rate sensitivity

Uses analytical formulas and numerical methods for accurate calculations.
"""

import numpy as np
import math
from scipy.stats import norm
from dataclasses import dataclass
from typing import Union, Optional
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


@dataclass
class Greeks:
    """Container for option Greeks"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    # Additional Greeks
    lambda_: Optional[float] = None  # Leverage/elasticity
    vanna: Optional[float] = None    # Delta sensitivity to volatility
    charm: Optional[float] = None    # Delta decay (theta of delta)
    vomma: Optional[float] = None    # Vega sensitivity to volatility

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho,
            'lambda': self.lambda_,
            'vanna': self.vanna,
            'charm': self.charm,
            'vomma': self.vomma
        }

    def __repr__(self) -> str:
        return f"Greeks(Δ={self.delta:.4f}, Γ={self.gamma:.4f}, Θ={self.theta:.4f}, ν={self.vega:.4f}, ρ={self.rho:.4f})"


class GreeksCalculator:
    """
    Advanced Greeks calculator using Black-Scholes model
    and numerical methods for enhanced accuracy.
    """

    def __init__(self, use_numerical_methods: bool = False):
        """
        Initialize Greeks calculator

        Args:
            use_numerical_methods: Use numerical differentiation for enhanced accuracy
        """
        self.use_numerical_methods = use_numerical_methods
        self.epsilon = 1e-6  # Small value for numerical differentiation

    def calculate_greeks(
        self,
        spot_price: Union[float, Decimal],
        strike_price: Union[float, Decimal],
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str = 'call',
        dividend_yield: float = 0.0
    ) -> Greeks:
        """
        Calculate all Greeks for an option

        Args:
            spot_price: Current underlying price
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate
            volatility: Implied volatility (annualized)
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield (annualized)

        Returns:
            Greeks object with all calculated values
        """
        try:
            S = float(spot_price)
            K = float(strike_price)
            T = time_to_expiry
            r = risk_free_rate
            sigma = volatility
            q = dividend_yield

            # Validate inputs
            if T <= 0:
                logger.warning(f"Time to expiry {T} <= 0, returning zero Greeks")
                return Greeks(0.0, 0.0, 0.0, 0.0, 0.0)

            if sigma <= 0:
                logger.warning(f"Volatility {sigma} <= 0, using minimum 0.01")
                sigma = 0.01

            # Calculate d1 and d2
            d1, d2 = self._calculate_d_values(S, K, T, r, sigma, q)

            # Standard normal CDF and PDF values
            N_d1 = norm.cdf(d1)
            N_d2 = norm.cdf(d2)
            n_d1 = norm.pdf(d1)

            # For puts
            N_neg_d1 = norm.cdf(-d1)
            N_neg_d2 = norm.cdf(-d2)

            is_call = option_type.lower() == 'call'

            # Calculate Greeks using analytical formulas
            if self.use_numerical_methods:
                delta = self._calculate_delta_numerical(S, K, T, r, sigma, q, option_type)
                gamma = self._calculate_gamma_numerical(S, K, T, r, sigma, q, option_type)
                theta = self._calculate_theta_numerical(S, K, T, r, sigma, q, option_type)
                vega = self._calculate_vega_numerical(S, K, T, r, sigma, q, option_type)
                rho = self._calculate_rho_numerical(S, K, T, r, sigma, q, option_type)
            else:
                delta = self._calculate_delta_analytical(S, K, T, r, sigma, q, d1, is_call)
                gamma = self._calculate_gamma_analytical(S, K, T, r, sigma, q, d1, n_d1)
                theta = self._calculate_theta_analytical(S, K, T, r, sigma, q, d1, d2, n_d1, N_d1, N_d2, N_neg_d1, N_neg_d2, is_call)
                vega = self._calculate_vega_analytical(S, K, T, r, sigma, q, n_d1)
                rho = self._calculate_rho_analytical(S, K, T, r, sigma, q, N_d2, N_neg_d2, is_call)

            # Calculate additional Greeks
            lambda_ = self._calculate_lambda(delta, S, self._black_scholes_price(S, K, T, r, sigma, q, option_type))
            vanna = self._calculate_vanna(S, K, T, r, sigma, q, d1, d2, n_d1)
            charm = self._calculate_charm(S, K, T, r, sigma, q, d1, d2, n_d1, is_call)
            vomma = self._calculate_vomma(S, K, T, r, sigma, q, d1, d2, n_d1)

            return Greeks(
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                lambda_=lambda_,
                vanna=vanna,
                charm=charm,
                vomma=vomma
            )

        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return Greeks(0.0, 0.0, 0.0, 0.0, 0.0)

    def _calculate_d_values(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> tuple:
        """Calculate d1 and d2 values for Black-Scholes"""
        sqrt_T = math.sqrt(T)

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        return d1, d2

    def _black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: str) -> float:
        """Calculate Black-Scholes option price"""
        d1, d2 = self._calculate_d_values(S, K, T, r, sigma, q)

        if option_type.lower() == 'call':
            price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

        return max(price, 0.0)

    # Analytical Greek calculations
    def _calculate_delta_analytical(self, S: float, K: float, T: float, r: float, sigma: float, q: float, d1: float, is_call: bool) -> float:
        """Calculate Delta analytically"""
        if is_call:
            return math.exp(-q * T) * norm.cdf(d1)
        else:
            return -math.exp(-q * T) * norm.cdf(-d1)

    def _calculate_gamma_analytical(self, S: float, K: float, T: float, r: float, sigma: float, q: float, d1: float, n_d1: float) -> float:
        """Calculate Gamma analytically"""
        return math.exp(-q * T) * n_d1 / (S * sigma * math.sqrt(T))

    def _calculate_theta_analytical(self, S: float, K: float, T: float, r: float, sigma: float, q: float,
                                   d1: float, d2: float, n_d1: float, N_d1: float, N_d2: float, N_neg_d1: float, N_neg_d2: float, is_call: bool) -> float:
        """Calculate Theta analytically"""
        term1 = -S * math.exp(-q * T) * n_d1 * sigma / (2 * math.sqrt(T))

        if is_call:
            term2 = -r * K * math.exp(-r * T) * N_d2
            term3 = q * S * math.exp(-q * T) * N_d1
            theta = term1 + term2 + term3
        else:
            term2 = r * K * math.exp(-r * T) * N_neg_d2
            term3 = -q * S * math.exp(-q * T) * N_neg_d1
            theta = term1 + term2 + term3

        # Convert from per year to per day
        return theta / 365.0

    def _calculate_vega_analytical(self, S: float, K: float, T: float, r: float, sigma: float, q: float, n_d1: float) -> float:
        """Calculate Vega analytically"""
        # Return vega per 1% change in volatility
        return S * math.exp(-q * T) * n_d1 * math.sqrt(T) / 100.0

    def _calculate_rho_analytical(self, S: float, K: float, T: float, r: float, sigma: float, q: float, N_d2: float, N_neg_d2: float, is_call: bool) -> float:
        """Calculate Rho analytically"""
        if is_call:
            rho = K * T * math.exp(-r * T) * N_d2
        else:
            rho = -K * T * math.exp(-r * T) * N_neg_d2

        # Return rho per 1% change in interest rate
        return rho / 100.0

    # Numerical Greek calculations (for validation/enhanced accuracy)
    def _calculate_delta_numerical(self, S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: str) -> float:
        """Calculate Delta using numerical differentiation"""
        h = S * self.epsilon
        price_up = self._black_scholes_price(S + h, K, T, r, sigma, q, option_type)
        price_down = self._black_scholes_price(S - h, K, T, r, sigma, q, option_type)
        return (price_up - price_down) / (2 * h)

    def _calculate_gamma_numerical(self, S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: str) -> float:
        """Calculate Gamma using numerical differentiation"""
        h = S * self.epsilon
        delta_up = self._calculate_delta_numerical(S + h, K, T, r, sigma, q, option_type)
        delta_down = self._calculate_delta_numerical(S - h, K, T, r, sigma, q, option_type)
        return (delta_up - delta_down) / (2 * h)

    def _calculate_theta_numerical(self, S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: str) -> float:
        """Calculate Theta using numerical differentiation"""
        if T <= 1/365:  # Less than 1 day to expiry
            return 0.0

        h = 1/365  # One day
        price_now = self._black_scholes_price(S, K, T, r, sigma, q, option_type)
        price_tomorrow = self._black_scholes_price(S, K, T - h, r, sigma, q, option_type)
        return price_tomorrow - price_now

    def _calculate_vega_numerical(self, S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: str) -> float:
        """Calculate Vega using numerical differentiation"""
        h = 0.01  # 1% volatility change
        price_up = self._black_scholes_price(S, K, T, r, sigma + h, q, option_type)
        price_down = self._black_scholes_price(S, K, T, r, sigma - h, q, option_type)
        return (price_up - price_down) / (2 * h)

    def _calculate_rho_numerical(self, S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: str) -> float:
        """Calculate Rho using numerical differentiation"""
        h = 0.01  # 1% interest rate change
        price_up = self._black_scholes_price(S, K, T, r + h, sigma, q, option_type)
        price_down = self._black_scholes_price(S, K, T, r - h, sigma, q, option_type)
        return (price_up - price_down) / (2 * h)

    # Second-order Greeks
    def _calculate_lambda(self, delta: float, S: float, option_price: float) -> float:
        """Calculate Lambda (elasticity)"""
        if option_price == 0:
            return 0.0
        return delta * S / option_price

    def _calculate_vanna(self, S: float, K: float, T: float, r: float, sigma: float, q: float, d1: float, d2: float, n_d1: float) -> float:
        """Calculate Vanna (delta sensitivity to volatility)"""
        return -math.exp(-q * T) * n_d1 * d2 / sigma

    def _calculate_charm(self, S: float, K: float, T: float, r: float, sigma: float, q: float, d1: float, d2: float, n_d1: float, is_call: bool) -> float:
        """Calculate Charm (delta decay)"""
        sqrt_T = math.sqrt(T)

        term1 = q * math.exp(-q * T) * norm.cdf(d1 if is_call else -d1)
        term2 = math.exp(-q * T) * n_d1 * (2 * (r - q) * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)

        if is_call:
            return -term1 - term2
        else:
            return -term1 + term2

    def _calculate_vomma(self, S: float, K: float, T: float, r: float, sigma: float, q: float, d1: float, d2: float, n_d1: float) -> float:
        """Calculate Vomma (vega sensitivity to volatility)"""
        return S * math.exp(-q * T) * n_d1 * math.sqrt(T) * d1 * d2 / sigma

    def calculate_portfolio_greeks(self, positions: list) -> Greeks:
        """
        Calculate aggregate Greeks for a portfolio of options

        Args:
            positions: List of dicts with keys: quantity, spot_price, strike_price,
                      time_to_expiry, risk_free_rate, volatility, option_type

        Returns:
            Aggregate Greeks for the portfolio
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0

        for position in positions:
            quantity = position.get('quantity', 0)
            if quantity == 0:
                continue

            greeks = self.calculate_greeks(
                spot_price=position['spot_price'],
                strike_price=position['strike_price'],
                time_to_expiry=position['time_to_expiry'],
                risk_free_rate=position['risk_free_rate'],
                volatility=position['volatility'],
                option_type=position['option_type'],
                dividend_yield=position.get('dividend_yield', 0.0)
            )

            # Weight by position size
            total_delta += greeks.delta * quantity
            total_gamma += greeks.gamma * quantity
            total_theta += greeks.theta * quantity
            total_vega += greeks.vega * quantity
            total_rho += greeks.rho * quantity

        return Greeks(
            delta=total_delta,
            gamma=total_gamma,
            theta=total_theta,
            vega=total_vega,
            rho=total_rho
        )

    def calculate_implied_volatility(
        self,
        market_price: float,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: str = 'call',
        dividend_yield: float = 0.0,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method

        Args:
            market_price: Market price of the option
            spot_price: Current underlying price
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance

        Returns:
            Implied volatility or None if convergence fails
        """
        try:
            # Initial guess
            vol = 0.3

            for i in range(max_iterations):
                price = self._black_scholes_price(spot_price, strike_price, time_to_expiry,
                                                 risk_free_rate, vol, dividend_yield, option_type)

                price_diff = price - market_price

                if abs(price_diff) < tolerance:
                    return vol

                # Calculate vega for Newton-Raphson
                d1, _ = self._calculate_d_values(spot_price, strike_price, time_to_expiry,
                                               risk_free_rate, vol, dividend_yield)
                n_d1 = norm.pdf(d1)
                vega = spot_price * math.exp(-dividend_yield * time_to_expiry) * n_d1 * math.sqrt(time_to_expiry)

                if abs(vega) < 1e-10:  # Avoid division by zero
                    return None

                # Newton-Raphson update
                vol = vol - price_diff / vega

                # Keep volatility positive and reasonable
                vol = max(0.01, min(5.0, vol))

            logger.warning(f"IV calculation did not converge after {max_iterations} iterations")
            return None

        except Exception as e:
            logger.error(f"Error calculating implied volatility: {e}")
            return None