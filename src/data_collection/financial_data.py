"""
Financial data collection module for retrieving stock prices,
financial metrics, and market data from various sources.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import yaml
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataCollector:
    """Collector for financial data from Yahoo Finance and other sources."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize financial data collector with configuration."""
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
    
    def get_stock_data(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Retrieve stock price data for given tickers.
        
        Args:
            tickers: List of stock ticker symbols
            period: Time period for data retrieval (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with stock price data
        """
        logger.info(f"Retrieving stock data for {len(tickers)} tickers over {period}")
        
        stock_data = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if not hist.empty:
                    # Calculate key metrics
                    latest_price = hist['Close'].iloc[-1]
                    price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
                    volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized volatility
                    avg_volume = hist['Volume'].mean()
                    
                    stock_info = {
                        'ticker': ticker,
                        'latest_price': latest_price,
                        'price_change_pct': price_change,
                        'volatility': volatility,
                        'avg_volume': avg_volume,
                        'period': period,
                        'data_points': len(hist),
                        'last_updated': datetime.now().isoformat()
                    }
                    stock_data.append(stock_info)
                else:
                    logger.warning(f"No data available for ticker {ticker}")
                    
            except Exception as e:
                logger.error(f"Error retrieving data for {ticker}: {str(e)}")
                
        return pd.DataFrame(stock_data)
    
    def get_financial_metrics(self, tickers: List[str]) -> pd.DataFrame:
        """
        Retrieve key financial metrics for given tickers.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            DataFrame with financial metrics
        """
        logger.info(f"Retrieving financial metrics for {len(tickers)} tickers")
        
        financial_data = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                metrics = {
                    'ticker': ticker,
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'pb_ratio': info.get('priceToBook'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'roe': info.get('returnOnEquity'),
                    'roa': info.get('returnOnAssets'),
                    'profit_margin': info.get('profitMargins'),
                    'revenue_growth': info.get('revenueGrowth'),
                    'earnings_growth': info.get('earningsGrowth'),
                    'dividend_yield': info.get('dividendYield'),
                    'beta': info.get('beta'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    'last_updated': datetime.now().isoformat()
                }
                financial_data.append(metrics)
                
            except Exception as e:
                logger.error(f"Error retrieving financial metrics for {ticker}: {str(e)}")
                
        return pd.DataFrame(financial_data)
    
    def calculate_risk_metrics(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Calculate risk metrics for given tickers.
        
        Args:
            tickers: List of stock ticker symbols
            period: Time period for calculation
            
        Returns:
            DataFrame with risk metrics
        """
        logger.info(f"Calculating risk metrics for {len(tickers)} tickers")
        
        risk_data = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if len(hist) > 30:  # Ensure sufficient data
                    returns = hist['Close'].pct_change().dropna()
                    
                    # Calculate VaR (Value at Risk) at 95% confidence level
                    var_95 = returns.quantile(0.05)
                    
                    # Calculate maximum drawdown
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = drawdown.min()
                    
                    # Calculate Sharpe ratio (assuming risk-free rate of 2%)
                    excess_returns = returns.mean() - 0.02/252  # Daily risk-free rate
                    sharpe_ratio = excess_returns / returns.std() * np.sqrt(252)
                    
                    risk_metrics = {
                        'ticker': ticker,
                        'var_95': var_95,
                        'max_drawdown': max_drawdown,
                        'sharpe_ratio': sharpe_ratio,
                        'skewness': returns.skew(),
                        'kurtosis': returns.kurtosis(),
                        'downside_deviation': returns[returns < 0].std(),
                        'last_updated': datetime.now().isoformat()
                    }
                    risk_data.append(risk_metrics)
                    
            except Exception as e:
                logger.error(f"Error calculating risk metrics for {ticker}: {str(e)}")
                
        return pd.DataFrame(risk_data)
    
    def save_financial_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save financial data to processed data directory."""
        filepath = os.path.join("data", "processed", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, index=False)
        logger.info(f"Financial data saved to {filepath}")

def main():
    """Example usage of FinancialDataCollector."""
    collector = FinancialDataCollector()
    
    # Example tickers
    tickers = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN']
    
    # Collect stock data
    stock_data = collector.get_stock_data(tickers)
    collector.save_financial_data(stock_data, "stock_data.csv")
    
    # Collect financial metrics
    financial_metrics = collector.get_financial_metrics(tickers)
    collector.save_financial_data(financial_metrics, "financial_metrics.csv")
    
    # Calculate risk metrics
    risk_metrics = collector.calculate_risk_metrics(tickers)
    collector.save_financial_data(risk_metrics, "risk_metrics.csv")
    
    logger.info("Financial data collection completed")

if __name__ == "__main__":
    main()