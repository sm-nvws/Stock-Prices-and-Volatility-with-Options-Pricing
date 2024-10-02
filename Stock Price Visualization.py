import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

class Evaluation:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    
    def _d1_d2(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def black_scholes_call(self):
        d1, d2 = self._d1_d2()
        call_price = (self.S * stats.norm.cdf(d1)) - (self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2))
        return call_price

    def black_scholes_put(self):
        d1, d2 = self._d1_d2()
        put_price = (self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2)) - (self.S * stats.norm.cdf(-d1))
        return put_price

    # Greeks
    def delta(self, option_type='call'):
        d1, _ = self._d1_d2()
        if option_type == 'call':
            return stats.norm.cdf(d1)  # Delta for call
        elif option_type == 'put':
            return stats.norm.cdf(d1) - 1  # Delta for put

    def gamma(self):
        d1, _ = self._d1_d2()
        return stats.norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        d1, _ = self._d1_d2()
        return self.S * stats.norm.pdf(d1) * np.sqrt(self.T)

    def theta(self, option_type='call'):
        d1, d2 = self._d1_d2()
        term1 = -(self.S * stats.norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        if option_type == 'call':
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2)
            return term1 + term2  # Theta for call
        elif option_type == 'put':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2)
            return term1 + term2  # Theta for put

    def rho(self, option_type='call'):
        _, d2 = self._d1_d2()
        if option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * stats.norm.cdf(d2)  # Rho for call
        elif option_type == 'put':
            return -self.K * self.T * np.exp(-self.r * self.T) * stats.norm.cdf(-d2)  # Rho for put


def main():
    # Fetch two stocks
    stock = yf.Ticker(input("Enter the first stock ticker: "))
    stock1 = yf.Ticker(input("Enter the second stock ticker: "))

    # Fetch historical data for both stocks (3 months)
    hist = stock.history(period='3mo')
    hist1 = stock1.history(period='3mo')

    # Calculate 3-month moving averages
    hist['3-Month MA'] = hist['Close'].rolling(window=5).mean()
    hist1['3-Month MA'] = hist1['Close'].rolling(window=5).mean()
        # Calculate daily returns and volatility for both stocks
    hist['Daily Return'] = hist['Close'].pct_change().dropna()
    hist1['Daily Return'] = hist1['Close'].pct_change().dropna()

    hist['Volatility'] = hist['Daily Return'].rolling(window=5).std() * (252 ** 0.5)  # Annualized volatility
    hist1['Volatility'] = hist1['Daily Return'].rolling(window=5).std() * (252 ** 0.5)

    # Create the plot for both stock prices and volatility
    fig, ax1 = plt.subplots(figsize=(10,6))

    # Plot stock 1 prices and moving averages on the first y-axis (left)
    line1, = ax1.plot(hist.index, hist['Close'], label=f"{stock.ticker} Close", color='blue')
    line2, = ax1.plot(hist.index, hist['3-Month MA'], label=f"{stock.ticker} 3-Month MA", linestyle='--', color='blue')
    ax1.set_ylabel(f"{stock.ticker} Price", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis for stock 2 prices (right)
    ax2 = ax1.twinx()
    line3, = ax2.plot(hist1.index, hist1['Close'], label=f"{stock1.ticker} Close", color='red')
    line4, = ax2.plot(hist1.index, hist1['3-Month MA'], label=f"{stock1.ticker} 3-Month MA", linestyle='--', color='red')
    ax2.set_ylabel(f"{stock1.ticker} Price", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Create a third y-axis for volatility (on the right, shared with the second y-axis)
    ax3 = ax2.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Move this third y-axis outward
    line5, = ax3.plot(hist.index, hist['Volatility'], label=f"{stock.ticker} Volatility", color='blue', linestyle=':', linewidth=1.5)
    line6, = ax3.plot(hist1.index, hist1['Volatility'], label=f"{stock1.ticker} Volatility", color='red', linestyle=':', linewidth=1.5)
    ax3.set_ylabel("Volatility")

    # Combine the legends from all axes
    lines = [line1, line2, line3, line4, line5, line6]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")

    # Add title and grid
    plt.title(f"Stock Price and Volatility Comparison: {stock.ticker} vs {stock1.ticker}")
    ax1.grid(True)

    # Show the plot
    plt.ion()
    plt.show()

    # Option trading part
    if input("Would you like to view the stocks and possibly buy an option (Yes/No)? ").lower() == "yes":
        which = input("Okay, which stock? ")
        type_option = input("Great. Call or Put? ").lower()
        strike = float(input("What is the strike price? "))
        time = float(input("How long till maturity (in years)? "))
        interest = float(input("What is the interest rate (as a decimal)? "))
        pr = yf.Ticker(which)
        
        # Fetch current price of the stock
        try:
            current_price = pr.info['regularMarketPrice']
        except KeyError:
            # Handle missing price
            current_price = pr.info.get('previousClose', pr.info.get('open', None))
            if current_price is None:
                print("Failed to retrieve current price for the stock.")
                return

        # Calculate historical volatility
        hi = pr.history(period='10y')
        sigma = hi['Close'].pct_change().dropna().std() * np.sqrt(252)

        # Create the evaluation object
        option_eval = Evaluation(current_price, strike, time, interest, sigma)

        # Calculate option price and Greeks
        if type_option == "call":
            price = option_eval.black_scholes_call()
            print(f"Call Option Price: {price:.2f}")
            print(f"Delta (Call): {option_eval.delta(option_type='call'):.4f}")
            print(f"Gamma: {option_eval.gamma():.4f}")
            print(f"Vega: {option_eval.vega():.4f}")
            print(f"Theta (Call): {option_eval.theta(option_type='call'):.4f}")
            print(f"Rho (Call): {option_eval.rho(option_type='call'):.4f}")
        else:
            price = option_eval.black_scholes_put()
            print(f"Put Option Price: {price:.2f}")
            print(f"Delta (Put): {option_eval.delta(option_type='put'):.4f}")
            print(f"Gamma: {option_eval.gamma():.4f}")
            print(f"Vega: {option_eval.vega():.4f}")
            print(f"Theta (Put): {option_eval.theta(option_type='put'):.4f}")
            print(f"Rho (Put): {option_eval.rho(option_type='put'):.4f}")
            
if __name__ == "__main__":
    main()
