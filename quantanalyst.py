## Baseline code for Financial analysis from FinViz data
## S.M. Laignel - 8 Oct 24
## No validation carried out yet

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List, Optional


class Universe:
    """
    Base class representing a universe of stocks loaded from a CSV file.
    Provides common preprocessing and utility methods.
    """
    def __init__(self, data_path: str):
        """
        Initialize the Universe with data from the specified CSV file.
        
        Args:
            data_path (str): Path to the CSV file containing stock data.
        """
        self.data = pd.read_csv(data_path)
        self.preprocess_data()
    
    def preprocess_data(self):
        """
        Preprocess the loaded data by cleaning column names, removing unwanted characters,
        converting columns to numeric types, and dropping rows with missing essential data.
        """
        # Strip whitespace from column headers
        self.data.columns = self.data.columns.str.strip()
        
        # Remove commas, percentage signs, and other non-numeric characters from numeric columns
        self.data.replace({',': '', '%': ''}, regex=True, inplace=True)
        
        # Define numeric columns to convert
        numeric_columns = [
            'Price', '20-Day Simple Moving Average', '50-Day Simple Moving Average',
            '200-Day Simple Moving Average', 'Volume', 'Average Volume', 'Market Cap',
            'Current Ratio', 'Profit Margin', 'Return on Assets', 'Return on Equity',
            'LT Debt/Equity', 'P/E', 'P/B', 'EPS (ttm)'
        ]
        
        # Convert columns to numeric types, coercing errors to NaN
        for col in numeric_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Define essential columns that must not have NaN values
        essential_columns = [
            'Ticker', 'Price', '20-Day Simple Moving Average',
            '50-Day Simple Moving Average', '200-Day Simple Moving Average',
            'Volume', 'Average Volume', 'Market Cap'
        ]
        
        # Drop rows with NaN in essential columns
        self.data.dropna(subset=essential_columns, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def describe(self):
        """
        Print a statistical summary and the first few rows of the dataset.
        """
        print(self.data.describe())
        print(self.data.head())

    def score_and_get_top(self, scoring_criteria: str, ascending: bool = True, top_n: int = 10, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Score the stocks based on the specified criteria and return the top N stocks.
        
        Args:
            scoring_criteria (str): The column name to base the scoring on.
            ascending (bool): Whether a lower score is better. True for ascending, False for descending.
            top_n (int): Number of top stocks to return.
            data (Optional[pd.DataFrame]): DataFrame to score. If None, uses the entire dataset.
        
        Returns:
            pd.DataFrame: Top N stocks based on the scoring criteria.
        """
        if data is None:
            data = self.data
        data = data.copy()  # Avoid SettingWithCopyWarning
        data['Score'] = data[scoring_criteria].rank(ascending=ascending, method='first')
        top_stocks = data.nsmallest(top_n, 'Score') if ascending else data.nlargest(top_n, 'Score')
        return top_stocks

    def components(self) -> List[str]:
        """
        Get the list of stock tickers in the universe.
        
        Returns:
            List[str]: List of ticker symbols.
        """
        return self.data['Ticker'].tolist()

    def profile(self, tickers: Optional[List[str]] = None, universe_name: Optional[str] = "Universe Components"):
        """
        Plot a scatter plot of volatility vs. cumulative return for the specified tickers.
        
        Args:
            tickers (Optional[List[str]]): List of ticker symbols to profile. 
                                           If None, profiles all components.
            universe_name (Optional[str]): Name of the sub-universe for the plot title.
        """
        if tickers is None:
            tickers = self.components()
        
        volatilities = []
        returns = []
        valid_tickers = []

        for ticker in tickers:
            try:
                # Download historical data for the last quarter (3 months)
                stock_data = yf.download(ticker, period='3mo', progress=False)
                if not stock_data.empty:
                    # Calculate daily returns
                    stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
                    # Calculate volatility (standard deviation of daily returns)
                    volatility = stock_data['Daily_Return'].std()
                    # Calculate cumulative return over the period
                    cumulative_return = (stock_data['Daily_Return'] + 1).prod() - 1

                    volatilities.append(volatility)
                    returns.append(cumulative_return)
                    valid_tickers.append(ticker)
            except Exception as e:
                print(f"Error downloading data for {ticker}: {e}")

        if not valid_tickers:
            print(f"No valid tickers to profile for {universe_name}.")
            return

        # Plot the scatter plot
        plt.figure(figsize=(12, 8))
        for i, ticker in enumerate(valid_tickers):
            plt.scatter(volatilities[i], returns[i], alpha=0.5)
            plt.text(volatilities[i], returns[i], ticker, fontsize=9)
        plt.xlabel('Volatility (Std Dev of Daily Returns)')
        plt.ylabel('Cumulative Return (Last 3 Months)')
        plt.title(f'Volatility vs. Return for {universe_name}')
        plt.grid(True)
        plt.show()


# High Momentum Sub-Class
class HighMomentumUniverse(Universe):
    """
    Subclass of Universe to identify high momentum stocks based on specific criteria.
    """
    def __init__(self, data_path: str):
        super().__init__(data_path)
    
    def calculate_high_momentum(self) -> pd.DataFrame:
        """
        Calculate high momentum scores and return the top 10 high momentum stocks.
        
        Returns:
            pd.DataFrame: Top 10 high momentum stocks.
        """
        required_columns = [
            'Price', '20-Day Simple Moving Average', '50-Day Simple Moving Average',
            '200-Day Simple Moving Average', 'Volume', 'Average Volume', 'Market Cap'
        ]
        for column in required_columns:
            if column not in self.data.columns:
                raise KeyError(f"Column '{column}' not found in data")
        
        # Define high momentum criteria
        self.data['Above_SMA20'] = self.data['Price'] > self.data['20-Day Simple Moving Average']
        self.data['MA_Hierarchy'] = (
            (self.data['20-Day Simple Moving Average'] > self.data['50-Day Simple Moving Average']) & 
            (self.data['50-Day Simple Moving Average'] > self.data['200-Day Simple Moving Average'])
        )
        self.data['Relative_Volume'] = self.data['Volume'] / self.data['Average Volume']
        self.data['High_Momentum_Score'] = (
            self.data['Above_SMA20'].astype(int) + 
            self.data['MA_Hierarchy'].astype(int) + 
            (self.data['Relative_Volume'] > 1).astype(int)
        )
        
        # Get top 10 high momentum stocks
        high_momentum_stocks = self.score_and_get_top(
            scoring_criteria='High_Momentum_Score', 
            ascending=False, 
            top_n=10
        )
        return high_momentum_stocks


# Mean Reversion Sub-Class
class MeanReversionUniverse(Universe):
    """
    Subclass of Universe to identify mean-reverting stocks based on specific criteria.
    """
    def __init__(self, data_path: str):
        super().__init__(data_path)
    
    def calculate_mean_reversion(self, window: int = 20) -> pd.DataFrame:
        """
        Calculate mean reversion scores and return the top 10 mean-reverting stocks.
        Limits yfinance calls to 10 stocks.
        
        Args:
            window (int): Rolling window size for moving average calculation.
        
        Returns:
            pd.DataFrame: Top 10 mean-reverting stocks.
        """
        if 'Price' not in self.data.columns:
            raise KeyError("Column 'Price' not found in data")
        
        # Initial candidate selection based on existing data
        # For mean reversion, select stocks with recent negative price change
        # Assuming 'Price' represents recent closing price; using relative metrics
        # Here, we'll use 'Price' and '20-Day Simple Moving Average' to infer recent performance
        candidates = self.data.copy()
        candidates['Price_Change'] = candidates['Price'] / candidates['20-Day Simple Moving Average'] - 1
        # Select top 10 candidates with lowest Price_Change (i.e., most below SMA20)
        candidates = candidates.sort_values('Price_Change').head(10)
        tickers = candidates['Ticker'].tolist()

        mean_reversion_scores = []
        for ticker in tickers:
            try:
                # Download historical data for a valid period
                stock_data = yf.download(ticker, period='3mo', progress=False)
                if not stock_data.empty and len(stock_data) >= window:
                    # Calculate moving average and deviation
                    stock_data['Moving_Avg'] = stock_data['Adj Close'].rolling(window=window).mean()
                    stock_data['Deviation'] = stock_data['Adj Close'] - stock_data['Moving_Avg']
                    latest_deviation = stock_data['Deviation'].iloc[-1]
                    mean_reversion_scores.append({
                        'Ticker': ticker, 
                        'Mean_Reversion_Score': -abs(latest_deviation)
                    })
                else:
                    mean_reversion_scores.append({
                        'Ticker': ticker, 
                        'Mean_Reversion_Score': np.nan
                    })
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                mean_reversion_scores.append({
                    'Ticker': ticker, 
                    'Mean_Reversion_Score': np.nan
                })

        scores_df = pd.DataFrame(mean_reversion_scores).dropna()
        if scores_df.empty:
            print("No valid mean reversion scores calculated.")
            return pd.DataFrame()

        merged_df = pd.merge(self.data, scores_df, on='Ticker')
        mean_reverting_stocks = merged_df.nsmallest(10, 'Mean_Reversion_Score')
        return mean_reverting_stocks


# Undervalued Stocks Sub-Class
class UndervaluedUniverse(Universe):
    """
    Subclass of Universe to identify undervalued stocks based on specific financial metrics.
    """
    def __init__(self, data_path: str):
        super().__init__(data_path)
    
    def calculate_undervalued_stocks(self) -> pd.DataFrame:
        """
        Identify undervalued stocks based on predefined financial thresholds
        and return the top 10 undervalued stocks.
        
        Returns:
            pd.DataFrame: Top 10 undervalued stocks.
        """
        thresholds = {
            'Current Ratio': 1.5,          # Should be greater than 1.5
            'Profit Margin': 0,            # Should be positive
            'Return on Assets': 0.05,      # Should be greater than 5%
            'Return on Equity': 0.10,      # Should be greater than 10%
            'LT Debt/Equity': 0.5,         # Should be less than 0.5
            'P/E': 0,                       # Should be positive
            'P/B': 1,                       # Should be less than 1
            'EPS (ttm)': 0                  # Should be positive
        }

        # Verify all required columns are present
        for column in thresholds.keys():
            if column not in self.data.columns:
                raise KeyError(f"Column '{column}' not found in data")
        
        # Apply filtering conditions based on thresholds
        self.data_filtered = self.data[
            (self.data['Current Ratio'] > thresholds['Current Ratio']) &
            (self.data['Profit Margin'] > thresholds['Profit Margin']) &
            (self.data['Return on Assets'] > thresholds['Return on Assets']) &
            (self.data['Return on Equity'] > thresholds['Return on Equity']) &
            (self.data['LT Debt/Equity'] < thresholds['LT Debt/Equity']) &
            (self.data['P/E'] > thresholds['P/E']) &
            (self.data['P/B'] < thresholds['P/B']) &
            (self.data['EPS (ttm)'] > thresholds['EPS (ttm)'])
        ]

        if self.data_filtered.empty:
            print("No undervalued stocks found based on the given thresholds.")
            return pd.DataFrame()

        # Score undervalued stocks based on average of P/E and P/B ratios
        self.data_filtered = self.data_filtered.copy()
        self.data_filtered.loc[:, 'Undervalued_Score'] = self.data_filtered[['P/E', 'P/B']].mean(axis=1)
        undervalued_stocks = self.score_and_get_top(
            scoring_criteria='Undervalued_Score', 
            ascending=True, 
            top_n=10, 
            data=self.data_filtered
        )
        return undervalued_stocks


# Breakout Potential Sub-Class
class BreakoutPotentialUniverse(Universe):
    """
    Subclass of Universe to identify stocks with breakout potential based on ATR.
    """
    def __init__(self, data_path: str):
        super().__init__(data_path)
    
    def calculate_breakout_potential(self, window: int = 20) -> pd.DataFrame:
        """
        Calculate breakout potential scores and return the top 10 breakout potential stocks.
        Limits yfinance calls to 10 stocks.
        
        Args:
            window (int): Rolling window size for ATR calculation.
        
        Returns:
            pd.DataFrame: Top 10 breakout potential stocks.
        """
        if 'Price' not in self.data.columns:
            raise KeyError("Column 'Price' not found in data")
        
        # Initial candidate selection based on existing data
        # For breakout potential, select stocks with high relative volume
        candidates = self.data.copy()
        candidates['Relative_Volume'] = candidates['Volume'] / self.data['Average Volume']
        # Select top 10 candidates with highest Relative Volume
        candidates = candidates.sort_values('Relative_Volume', ascending=False).head(10)
        tickers = candidates['Ticker'].tolist()

        breakout_scores = []
        for ticker in tickers:
            try:
                # Download historical data for a valid period
                stock_data = yf.download(ticker, period='3mo', progress=False)
                if not stock_data.empty and len(stock_data) >= window:
                    # Calculate True Range
                    stock_data['High-Low'] = stock_data['High'] - stock_data['Low']
                    stock_data['High-Close'] = abs(stock_data['High'] - stock_data['Close'].shift(1))
                    stock_data['Low-Close'] = abs(stock_data['Low'] - stock_data['Close'].shift(1))
                    stock_data['True_Range'] = stock_data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
                    # Calculate ATR
                    stock_data['ATR'] = stock_data['True_Range'].rolling(window=window).mean()
                    latest_atr = stock_data['ATR'].iloc[-1]
                    latest_price = stock_data['Close'].iloc[-1]
                    breakout_score = latest_atr / latest_price
                    breakout_scores.append({
                        'Ticker': ticker, 
                        'Breakout_Score': breakout_score
                    })
                else:
                    breakout_scores.append({
                        'Ticker': ticker, 
                        'Breakout_Score': np.nan
                    })
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                breakout_scores.append({
                    'Ticker': ticker, 
                    'Breakout_Score': np.nan
                })

        scores_df = pd.DataFrame(breakout_scores).dropna()
        if scores_df.empty:
            print("No breakout potential scores calculated.")
            return pd.DataFrame()

        merged_df = pd.merge(self.data, scores_df, on='Ticker')
        breakout_stocks = merged_df.nlargest(10, 'Breakout_Score')
        return breakout_stocks


# Example Usage
if __name__ == "__main__":
    data_path = 'finviz(1).csv'  # Path to your CSV file

    # Instantiate and calculate High Momentum Universe
    high_momentum_universe = HighMomentumUniverse(data_path)
    high_momentum_stocks = high_momentum_universe.calculate_high_momentum()
    print("High Momentum Stocks:")
    print(high_momentum_stocks)
    # Profile only the top 10 high momentum stocks with appropriate title
    high_momentum_universe.profile(tickers=high_momentum_stocks['Ticker'].tolist(), universe_name="High Momentum")

    # Instantiate and calculate Mean Reversion Universe
    mean_reversion_universe = MeanReversionUniverse(data_path)
    mean_reverting_stocks = mean_reversion_universe.calculate_mean_reversion()
    print("\nMean Reverting Stocks:")
    print(mean_reverting_stocks)
    if not mean_reverting_stocks.empty:
        # Profile only the top 10 mean reverting stocks with appropriate title
        mean_reversion_universe.profile(tickers=mean_reverting_stocks['Ticker'].tolist(), universe_name="Mean Reversion")

    # Instantiate and calculate Undervalued Universe
    undervalued_universe = UndervaluedUniverse(data_path)
    undervalued_stocks = undervalued_universe.calculate_undervalued_stocks()
    print("\nUndervalued Stocks:")
    print(undervalued_stocks)
    if not undervalued_stocks.empty:
        # Optionally, profile undervalued stocks with appropriate title
        undervalued_universe.profile(tickers=undervalued_stocks['Ticker'].tolist(), universe_name="Undervalued")

    # Instantiate and calculate Breakout Potential Universe
    breakout_potential_universe = BreakoutPotentialUniverse(data_path)
    breakout_stocks = breakout_potential_universe.calculate_breakout_potential()
    print("\nBreakout Potential Stocks:")
    print(breakout_stocks)
    if not breakout_stocks.empty:
        # Profile only the top 10 breakout potential stocks with appropriate title
        breakout_potential_universe.profile(tickers=breakout_stocks['Ticker'].tolist(), universe_name="Breakout Potential")
