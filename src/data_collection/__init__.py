"""
Data collection module for ESG scoring dashboard.
Includes web scraping and financial data collection components.
"""

from .scraper import ESGScraper
from .financial_data import FinancialDataCollector

__all__ = ['ESGScraper', 'FinancialDataCollector']