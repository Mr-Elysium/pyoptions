from src.stockoption._constants import *

from datetime import datetime, timedelta

import numpy as np
from scipy.stats import norm

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

    # Initialize terminal price nodes
    STs = [np.array([S])]

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

# Calculate Implied Volatility using newton raphson
def calculate_implied_volatility(option, option_price=None):
    if not option_price:
        option_price=option.spot_price

    sigma = SOLVER_STARTING_VALUE
    epsilon = IMPLIED_VOLATILITY_TOLERANCE

    def newton_raphson(option, sigma, epsilon):
        diff = abs(option.option_price(sigma=sigma) - option_price)
        while diff > epsilon:
            sigma = (sigma - (option.option_price(sigma=sigma) - option_price) / max((option.vega(sigma=sigma) * 100), 100))
            diff = abs(option.option_price(sigma=sigma) - option_price)
        return sigma

    iv = newton_raphson(option, sigma, epsilon)
    return iv

# side = Option side/type either "call" or "put"
# is_european = Option type, either european (True) or american (False)
# S = Stock price (Input: Stock symbol string or arbitrary number)
# K = Strike price
# r = Annualised risk-free rate
# div = Annualised dividend of underlying
# pricing_model = Method of calculating option price, either "binomial" or "blackscholes"
#
# Either expiration or T must be given, one will be derived from the other, both cannot be given
# expiration = Expiration date, input as string: 'dd-mm-yyyy'
# T = Time to expiration in years, input number of years
#
# Either sigma or option price ought to be given, one will be derived from the other, both cannot be given
# sigma = (Implied) Volatility over lifetime option in decimal percent (30% --> 0.3)
# spot_price = Option spot price
class option:
    # Initialise option variables
    def __init__(self, side: str, is_european: bool, K:float, S: float, T: float = None, expiration: str = None, sigma: float = None, spot_price: float = None, r: float = None, div: float = None, pricing_model: str = None, periods: int = None):
        if not side:
            raise Exception("Option side must be given, either 'call' or 'put'")
        self.side = side
        if not is_european:
            raise Exception("Must mention whether option is european (Ture) or American (False)")
        self.is_european = is_european
        if not K:
            raise Exception("Strike price of option must be given")
        self.K = K
        if not S:
            raise Exception("Spot price of stock must be given")
        self.S = S

        if not expiration and not T:
            raise Exception("Either expiration date or time to expiration in years must be given.")
        elif expiration and not T:
            time_difference = datetime.strptime(expiration, DATE_FORMAT) - datetime.today()
            self.T = (time_difference.days + time_difference.seconds / 86400) / 365
        elif T and not expiration:
            self.expiration = datetime.strftime(datetime.today() + timedelta(days=T*365), DATE_FORMAT)
            self.T = T
        else:
            raise Exception("You cannot set both T and expiration, this will create a discrepancy in the calculation of T")

        if not r:
            self.r = float(riskfree())
        else:
            self.r = r
        if not div:
            self.div = 0
        else:
            self.div = div

        if not periods:
            self.periods = 30
        else:
            self.periods = periods

        if not pricing_model:
            self.pricing_model = "blackscholes"
        elif pricing_model != "blackscholes" and pricing_model != "binomial":
            raise Exception("Invalid stockoption model, must be 'blackscholes' or 'binomial'")
        else:
            self.pricing_model = pricing_model

        if not sigma and not spot_price:
            raise Exception("Either option price or volatility sigma must be given.")
        elif not sigma and spot_price:
            self.spot_price = spot_price
            self.sigma = calculate_implied_volatility(self)
        elif not spot_price and sigma:
            self.sigma = sigma
            if self.pricing_model == "blackscholes":
                self.spot_price = blackscholes_price(self, K, S, T, sigma, r, div)
            if self.pricing_model == "binomial":
                self.spot_price = binomial_tree_price(self, K, S, T, sigma, r, div, periods=periods)
        else:
            raise Exception("You cannot set both spot_price and volatility sigma, this will create a discrepancy in the stockoption.")

    # Basic option price function, uses pricing_model to calculate price
    def option_price(self, K=None, S=None, T=None, sigma=None, r=None, div=None, periods=None):
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
        if not periods:
            periods = self.periods

        if self.pricing_model == "blackscholes" and self.is_european == False:
            raise Exception("You cannot price american options using the blackscholes stockoption model")
        if self.pricing_model == "blackscholes":
            return blackscholes_price(self, K, S, T, sigma, r, div)
        if self.pricing_model == "binomial":
            return binomial_tree_price(self, K=K, S=S, T=T, sigma=sigma, r=r, div=div, periods=periods)
        raise Exception("Invalid stockoption model, must be 'blackscholes' or 'binomial'")

    # Pricing function using blackscholes
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
            raise Exception("You cannot price american options using the blackscholes stockoption model")
        return blackscholes_price(self, K, S, T, sigma, r, div)

    # Pricing function using binomial tree
    def binomial_option_price(self, K=None, S=None, T=None, sigma=None, r=None, div=None, periods=None):
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
        if not periods:
            periods = self.periods

        return binomial_tree_price(self, K, S, T, sigma, r, div, periods)

    # Calculate implied volatility
    def implied_volatility(self):
        return calculate_implied_volatility(self)

    # Calculate delta
    def delta(self, S=None):
        if not S:
            S=self.S
        h = DELTA_DIFFERENTIAL
        p1 = self.option_price(S=S + h)
        p2 = self.option_price(S=S - h)
        return (p1 - p2) / (2 * h)

    # Calculate gamma
    def gamma(self, S=None):
        if not S:
            S=self.S
        h = GAMMA_DIFFERENTIAL
        p1 = self.option_price(S=S + h)
        p2 = self.option_price()
        p3 = self.option_price(S=S - h)
        return (p1 - 2 * p2 + p3) / np.power(h, 2)

    # Calculate vega
    def vega(self, sigma=None):
        if not sigma:
            sigma=self.sigma
        h = VEGA_DIFFERENTIAL
        p1 = self.option_price(sigma=sigma + h)
        p2 = self.option_price(sigma=sigma - h)
        return (p1 - p2) / (2 * h * 100)

    # Calculate theta
    def theta(self):
        h = THETA_DIFFERENTIAL
        p1 = self.option_price(T=self.T + h)
        p2 = self.option_price(T=self.T - h)
        return (p1 - p2) / (2 * h * 365)

    # Calculate rho
    def rho(self):
        h = RHO_DIFFERENTIAL
        p1 = self.option_price(r=self.r + h)
        p2 = self.option_price(r=self.r - h)
        return (p1 - p2) / (2 * h * 100)
