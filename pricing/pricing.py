import requests
import xml.etree.ElementTree as ET

from dataclasses import dataclass, InitVar, field
from datetime import datetime, timedelta
from math import sqrt, log, exp

# import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.stats import norm

from constants import *

def riskfree():
    return FALLBACK_RISK_FREE_RATE

# S = Stock price (Input: Stock symbol string or arbitrary number)
# K = Strike price
# T = Time to experation in years (Input: Number of days or string: 'dd-mm-yyyy')
# r = Risk free rate over lifetime option
#
# Either sigma or option price ought to be given, one will be derived from the other
# sigma = (Implied) Volatility over lifetime option in decimal percent (30% --> 0.3)
# price = Option spot price
# @dataclass
class option:
    S = None

    def __init__(self, side: str, K:float, S: float = None, T: float = None, expiration: str = None, sigma: float = None, option_price: float = None, r: float = None, stock: InitVar[str] = None):
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

        if sigma and not option_price:
            self.option_price = None
            self.sigma = sigma
        elif option_price and not sigma:
            self.option_price = option_price
            self.sigma = None
        elif not sigma and not option_price:
            raise Exception("Either option price or volatility sigma must be given")
        self.r = float(riskfree())

    # Calculate options price with Black & Scholes formula
    def price(self, S=self.S, T=self.T, sigma=self.sigma, r=self.r):
        d1 = (log(S / self.K) + (r + (sigma ** 2) / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        if self.side == "call":
            return max(S * norm.cdf(d1) - self.K * exp(-r * T) * norm.cdf(d2), 0)
        if self.side == "put":
            return max(self.K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0)

    # Calculate Implied Volatility
    def IV(self, option_price=None):
        if not option_price:
            option_price = self.option_price

        def _fprime(self, sigma):
            logSolverK = log(self.S / self.K)
            n12 = ((self.r + sigma ** 2 / 2) * self.T)
            number1 = logSolverK + n12
            d1 = number1 / (sigma * sqrt(self.T))
            return self.S * sqrt(self.T) * norm.pdf(d1) * exp(-self.r * self.T)

        impvol = lambda x: price(self, sigma=x) - option_price
        iv = fsolve(impvol, SOLVER_STARTING_VALUE, fprime=_fprime, xtol=IMPLIED_VOLATILITY_TOLERANCE)
        return iv[0]

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

    def vega(self):
        h = VEGA_DIFFERENTIAL
        p1 = price(self, sigma=self.sigma + h)
        p2 = price(self, sigma=self.sigma - h)
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

a = option(side='call', K=28000, S=28275, T=47.75, option_price=1575)