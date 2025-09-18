"""
ESG Scoring Dashboard - Main package initialization
"""

__version__ = "0.1.0"
__author__ = "ESG Dashboard Team"
__email__ = "team@esg-dashboard.com"

# Package imports for easier access
from .data_collection import scraper, financial_data
from .scoring import environmental, social, governance
from .analysis import correlation_analysis
from .dashboard import app

__all__ = [
    'scraper',
    'financial_data', 
    'environmental',
    'social',
    'governance',
    'correlation_analysis',
    'app'
]