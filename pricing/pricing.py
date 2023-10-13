from dataclasses import InitVar
from datetime import datetime, timedelta
from math import sqrt, log, exp

from scipy.stats import norm

from constants import *

def riskfree():
    return FALLBACK_RISK_FREE_RATE

# Calculate options price with Black Scholes formula
def price(option, K=None, S=None, T=None, sigma=None, r=None):
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

    d1 = (log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option.side == "call":
        return max(S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2), 0)
    if option.side == "put":
        return max(K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0)

# Calculate Implied Volatility
def IV(option, option_price=None):
    if not option_price:
        option_price=option.spot_price

    sigma = SOLVER_STARTING_VALUE
    epsilon = IMPLIED_VOLATILITY_TOLERANCE

    def newton_raphson(option, sigma, epsilon):
        diff = abs(price(option, sigma=sigma) - option_price)
        while diff > epsilon:
            sigma = (sigma - (price(option, sigma=sigma) - option_price) / max((option.vega(sigma=sigma) * 100), 100))
            diff = abs(price(option, sigma=sigma) - option_price)
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
    def __init__(self, side: str, K:float, S: float = None, T: float = None, expiration: str = None, sigma: float = None, spot_price: float = None, r: float = None, stock: InitVar[str] = None):
        self.side = side
        self.K = K

        if stock:
            # TODO Import from yfinance
            self.stock = stock
        if not S:
            raise Exception("Either stock or spot price must be given")
        self.S = S

        if expiration:
            self.T = (datetime.strptime(expiration, DATE_FORMAT) - datetime.today()).days / 365
        elif not T:
            raise Exception("Either expiration date or time to expiration in years must be given")
        else:
            self.expiration = datetime.strftime(datetime.today() + timedelta(days=int(T)), DATE_FORMAT)
            self.T = T / 365

        self.r = float(riskfree())

        if not sigma and not spot_price:
            raise Exception("Either option price or volatility sigma must be given")
        if not sigma:
            self.spot_price = spot_price
            self.sigma = IV(self)
        if not spot_price:
            self.sigma = sigma
            self.spot_price=price(self)

    def option_price(self):
        if self.spot_price:
            return self.spot_price
        return price(self)

    def iv(self):
        if self.sigma:
            return self.sigma
        return IV(self)

    def delta(self):
        h = DELTA_DIFFERENTIAL
        p1 = price(self, S=self.S + h)
        p2 = price(self, S=self.S - h)
        return (p1 - p2) / (2 * h)

    def gamma(self):
        h = GAMMA_DIFFERENTIAL
        p1 = price(self, S=self.S + h)
        p2 = price(self)
        p3 = price(self, S=self.S - h)
        return (p1 - 2 * p2 + p3) / (h ** 2)

    def vega(self, sigma=None):
        if not sigma:
            sigma=option.sigma
        h = VEGA_DIFFERENTIAL
        p1 = price(self, sigma=sigma + h)
        p2 = price(self, sigma=sigma - h)
        return (p1 - p2) / (2 * h * 100)

    def theta(self):
        h = THETA_DIFFERENTIAL
        p1 = price(self, T=self.T + h)
        p2 = price(self, T=self.T - h)
        return (p1 - p2) / (2 * h * 365)

    def rho(self):
        h = RHO_DIFFERENTIAL
        p1 = price(self, r=self.r + h)
        p2 = price(self, r=self.r - h)
        return (p1 - p2) / (2 * h * 100)