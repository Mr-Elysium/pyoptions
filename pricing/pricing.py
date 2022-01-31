import requests
from bs4 import BeautifulSoup
# import numpy as np
# import matplotlib.pyplot as plt

from constants import *

from scipy.interpolate import interp1d
from datetime import datetime, date, timedelta
from time import mktime, sleep
from math import sqrt, log, exp
from scipy.stats import norm
from scipy.optimize import fsolve

# Imports current treasury rates and extrapolates to continuous time
# Returns riskfree rate function
def riskfree():
    try:
        r = requests.get(TREASURY_URL)
        soup = BeautifulSoup(r.text, 'html.parser')

        table = soup.find("table", attrs={'class' : 't-chart'})
        rows = table.find_all('tr')
        lastrow = len(rows)-1
        cells = rows[lastrow].find_all("td")
        date = cells[0].get_text()
        m1 = float(cells[1].get_text())
        m2 = float(cells[2].get_text())
        m3 = float(cells[3].get_text())
        m6 = float(cells[4].get_text())
        y1 = float(cells[5].get_text())
        y2 = float(cells[6].get_text())
        y3 = float(cells[7].get_text())
        y5 = float(cells[8].get_text())
        y7 = float(cells[9].get_text())
        y10 = float(cells[10].get_text())
        y20 = float(cells[11].get_text())
        y30 = float(cells[12].get_text())

        years = (0, 1/12, 3/12, 6/12, 12/12, 24/12, 36/12, 60/12, 84/12, 120/12, 240/12, 360/12)
        rates = (OVERNIGHT_RATE, m1/100, m3/100, m6/100, y1/100, y2/100, y3/100, y5/100, y7/100, y10/100, y20/100, y30/100)
        return interp1d(years, rates)
    # If scraping treasury data fails use the constant fallback risk free rate
    except Exception:
        return lambda x: FALLBACK_RISK_FREE_RATE

# S = Stock price (Input: Stock symbol string or arbitrary number)
# K = Strike price
# T = Time to experation in years (Input: Number of days or string: 'dd-mm-yyyy')
# r = Risk free rate over lifetime option
# 
# Either sigma or optprice ought to be given, one will be derived from the other
# sigma = (Implied) Volatility over lifetime option
# price = Option spot price
class Call:
    def __init__(self, spot, strike, experation, sigma = None, calcprice = None, optprice = None):
        if (type(spot) == str):
            # TODO Import from yfinance
            pass
        else:
            self.S = spot
        self.K = strike
        if (type(experation) == str):
            self.T = (datetime.strptime(experation, DATE_FORMAT) - datetime.today()).days/365
        else:
            self.T = experation/365
        self.r = riskfree()(self.T)
        if not sigma and not optprice:
            raise Exception("Either sigma or optprice ought to be given.")
        if optprice:
            self.optprice = optprice
            if not sigma:
                self.sigma = self.IV()
            self.calcprice = self.price()
        if sigma:
            self.sigma = sigma
            self.calcprice = self.price()
            if not optprice:
                self.optprice = self.calcprice

    def price(self, S=None, K=None, T=None, r=None, sigma=None):
        if not S:
            S = self.S
        if not K:
            K = self.K
        if not T:
            T = self.T
        if not r:
            r = self.r
        if not sigma:
            sigma = self.sigma
        d1 = (log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S * norm.cdf(d1) - K * norm.cdf(d2)

    def _fprime(self, sigma):
        logSoverK = log(self.S/self.K)
        n12 = ((self.r + sigma**2/2)*self.T)
        numerd1 = logSoverK + n12
        d1 = numerd1/(sigma*sqrt(self.T))
        return self.S*sqrt(self.T)*norm.pdf(d1)*exp(-self.r*self.T)

    def IV(self, optprice = None):
        if not optprice:
            optprice = self.optprice
        impvol = lambda x: self.price(sigma=x) - optprice
        iv = fsolve(impvol, SOLVER_STARTING_VALUE, fprime=self._fprime, xtol=IMPLIED_VOLATILITY_TOLERANCE)
        return iv[0]
