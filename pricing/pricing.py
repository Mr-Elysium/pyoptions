from dataclasses import InitVar
from datetime import datetime, timedelta

from scipy.stats import norm

# from pricing.constants import *
from constants import *

import matplotlib.pyplot as plt
import numpy as np

def riskfree():
    return FALLBACK_RISK_FREE_RATE

# Calculate options price with Black Scholes formula
def blackscholes_price(option, K=None, S=None, T=None, sigma=None, r=None, div=None):
    if not K:
        K=option.K
    if not S:
        S=option.S
    if not T:
        T=option.T
    if not sigma:
        sigma=option.sigma
    if not r:
        r=option.r
    if not div:
        div=option.div

    d1 = (np.log(S / K) + (r - div + np.power(sigma, 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option.side == "call":
        return max(S * np.exp(-div * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0)
    if option.side == "put":
        return max(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-div * T) * norm.cdf(-d1), 0)

# Calculate options price using Cox-Ross-Rubinstein binomial tree model
def binomial_tree_price(option, K=None, S=None, T=None, sigma=None, r=None, div=None, periods=30):
    if not K:
        K=option.K
    if not S:
        S=option.S
    if not T:
        T=option.T
    if not sigma:
        sigma=option.sigma
    if not r:
        r=option.r
    if not div:
        div=option.div

    # Set parameters
    dt = T / periods
    u = np.exp(sigma * np.sqrt(dt))  # Expected value in the up state
    d = 1 / u  # Expected value in the down state
    R = np.exp((r - div) * dt)
    up = (R - d) / (u - d)  # Probability of up move
    dp = 1 - up
    df = np.exp(-(r - div) * dt)

    # Initialize terminal price nodes to zeros
    STs = [np.array([option.S])]

    # Calculate expected stock prices for each node
    for i in range(periods):
        prev_branches = STs[-1]
        st = np.concatenate((prev_branches * u, [prev_branches[-1] * d]))
        STs.append(st)

    # Get payoffs when the option expires at terminal nodes
    payoffs = np.maximum(0, (STs[periods] - K) if option.side == "call" else (K - STs[periods]))

    def __check_early_exercise__(payoffs, node):
        early_ex_payoff = (STs[node] - K) if option.side == "call" else (K - STs[node])
        return np.maximum(payoffs, early_ex_payoff)

    # Starting from the time the option expires, traverse
    # backwards and calculate discounted payoffs at each node
    for i in reversed(range(periods)):
        payoffs = (payoffs[:-1] * up + payoffs[1:] * dp) * df

        if not option.is_european:
            payoffs = __check_early_exercise__(payoffs, i)

    return payoffs[0]

# Calculate Implied Volatility
def calculate_implied_volatility(option, option_price=None):
    if not option_price:
        option_price=option.spot_price

    sigma = SOLVER_STARTING_VALUE
    epsilon = IMPLIED_VOLATILITY_TOLERANCE

    def newton_raphson(option, sigma, epsilon):
        diff = abs(blackscholes_price(option, sigma=sigma) - option_price)
        while diff > epsilon:
            sigma = (sigma - (blackscholes_price(option, sigma=sigma) - option_price) / max((option.vega(sigma=sigma) * 100), 100))
            diff = abs(blackscholes_price(option, sigma=sigma) - option_price)
        return sigma

    iv = newton_raphson(option, sigma, epsilon)
    return iv

# S = Stock price (Input: Stock symbol string or arbitrary number)
# K = Strike price
# T = Time to experation in years (Input: Number of days or string: 'dd-mm-yyyy')
# r = Risk free rate over lifetime option
#
# Either sigma or option price ought to be given, one will be derived from the other
# sigma = (Implied) Volatility over lifetime option in decimal percent (30% --> 0.3)
# spot_price = Option spot price
class option:
    def __init__(self, side: str, is_european: bool, K:float, S: float = None, T: float = None, expiration: str = None, sigma: float = None, spot_price: float = None, r: float = None, div: float = None, pricing_model: str = None):
        self.side = side
        self.is_european = is_european
        self.K = K

        if not S:
            raise Exception("Either stock or spot price must be given.")
        self.S = S

        if not expiration and not T:
            raise Exception("Either expiration date or time to expiration in years must be given.")
        if expiration:
            self.T = (datetime.strptime(expiration, DATE_FORMAT) - datetime.today()).days / 365
        elif not T:
            raise Exception("Either expiration date or time to expiration in years must be given.")
        else:
            self.expiration = datetime.strftime(datetime.today() + timedelta(days=T*365), DATE_FORMAT)
            self.T = T

        if not r:
            self.r = float(riskfree())
        else:
            self.r = r
        if not div:
            self.div = 0
        else:
            self.div = div

        if not sigma and not spot_price:
            raise Exception("Either option price or volatility sigma must be given.")
        elif not sigma and spot_price:
            self.spot_price = spot_price
            self.sigma = calculate_implied_volatility(self)
        elif not spot_price and sigma:
            self.sigma = sigma
            self.spot_price=blackscholes_price(self)
        else:
            raise Exception("You cannot set both spot_price and volatility sigma, this will create a discrepancy in the pricing.")

        if not pricing_model:
            self.pricing_model = "blackscholes"
        else:
            self.pricing_model = pricing_model

    def option_price(self, K=None, S=None, T=None, sigma=None, r=None, div=None, periods=30):
        if not K:
            K = self.K
        if not S:
            S = self.S
        if not T:
            T = self.T
        if not sigma:
            sigma = self.sigma
        if not r:
            r = self.r
        if not div:
            div = self.div

        if self.spot_price:
            return self.spot_price
        if self.pricing_model == "blackscholes":
            return blackscholes_price(self, K, S, T, sigma, r, div)
        if self.pricing_model == "binomial":
            return binomial_tree_price(self, K, S, T, sigma, r, div, periods)
        raise Exception("Invalid pricing model, must be 'blackscholes' or 'binomial.'")

    def blackscholes_option_price(self, K=None, S=None, T=None, sigma=None, r=None, div=None):
        if not K:
            K = self.K
        if not S:
            S = self.S
        if not T:
            T = self.T
        if not sigma:
            sigma = self.sigma
        if not r:
            r = self.r
        if not div:
            div = self.div

        if not self.is_european:
            raise Exception("You cannot price options using the blackscholes pricing model")
        return blackscholes_price(self, K, S, T, sigma, r, div)

    def binomial_option_price(self, K=None, S=None, T=None, sigma=None, r=None, div=None, periods=30):
        if not K:
            K = self.K
        if not S:
            S = self.S
        if not T:
            T = self.T
        if not sigma:
            sigma = self.sigma
        if not r:
            r = self.r
        if not div:
            div = self.div

        return binomial_tree_price(self, K, S, T, sigma, r, div, periods)

    def implied_volatility(self):
        if self.sigma:
            return self.sigma
        return calculate_implied_volatility(self)

    def delta(self, S=None):
        if not S:
            S=self.S
        h = DELTA_DIFFERENTIAL
        p1 = self.option_price(self, S=S + h)
        p2 = self.option_price(self, S=S - h)
        return (p1 - p2) / (2 * h)

    def gamma(self, S=None):
        if not S:
            S=self.S
        h = GAMMA_DIFFERENTIAL
        p1 = self.option_price(self, S=S + h)
        p2 = self.option_price(self)
        p3 = self.option_price(self, S=S - h)
        return (p1 - 2 * p2 + p3) / (h ** 2)

    def vega(self, sigma=None):
        if not sigma:
            sigma=self.sigma
        h = VEGA_DIFFERENTIAL
        p1 = self.option_price(self, sigma=sigma + h)
        p2 = self.option_price(self, sigma=sigma - h)
        return (p1 - p2) / (2 * h * 100)

    def theta(self):
        h = THETA_DIFFERENTIAL
        p1 = self.option_price(self, T=self.T + h)
        p2 = self.option_price(self, T=self.T - h)
        return (p1 - p2) / (2 * h * 365)

    def rho(self):
        h = RHO_DIFFERENTIAL
        p1 = self.option_price(self, r=self.r + h)
        p2 = self.option_price(self, r=self.r - h)
        return (p1 - p2) / (2 * h * 100)
