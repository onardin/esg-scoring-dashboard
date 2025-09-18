"""
Test suite for ESG Scoring Dashboard.

This module contains unit tests for all components of the ESG scoring system.
Run with: pytest tests/
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scoring.environmental import EnvironmentalScorer
from scoring.social import SocialScorer
from scoring.governance import GovernanceScorer
from data_collection.financial_data import FinancialDataCollector
from analysis.correlation_analysis import CorrelationAnalyzer

class TestEnvironmentalScorer(unittest.TestCase):
    """Test cases for EnvironmentalScorer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = EnvironmentalScorer()
        self.sample_data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'carbon_emissions': [1000, 1500, 800],
            'revenue': [100000, 120000, 95000],
            'renewable_energy_pct': [80, 60, 90]
        })
    
    def test_scorer_initialization(self):
        """Test scorer initializes correctly."""
        self.assertIsInstance(self.scorer, EnvironmentalScorer)
        self.assertIn('carbon_weight', self.scorer.weights)
    
    def test_carbon_intensity_calculation(self):
        """Test carbon intensity score calculation."""
        result = self.scorer.calculate_carbon_intensity_score(self.sample_data)
        self.assertIn('carbon_intensity_score', result.columns)
        self.assertTrue(all(result['carbon_intensity_score'] >= 0))
        self.assertTrue(all(result['carbon_intensity_score'] <= 100))

class TestSocialScorer(unittest.TestCase):
    """Test cases for SocialScorer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = SocialScorer()
        self.sample_data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'gender_diversity_pct': [45, 40, 48],
            'charitable_giving_pct_revenue': [1.2, 0.8, 1.5]
        })
    
    def test_scorer_initialization(self):
        """Test scorer initializes correctly."""
        self.assertIsInstance(self.scorer, SocialScorer)
        self.assertIn('diversity_weight', self.scorer.weights)
    
    def test_diversity_score_calculation(self):
        """Test diversity score calculation."""
        result = self.scorer.calculate_diversity_score(self.sample_data)
        self.assertIn('diversity_score', result.columns)
        self.assertTrue(all(result['diversity_score'] >= 0))

class TestGovernanceScorer(unittest.TestCase):
    """Test cases for GovernanceScorer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = GovernanceScorer()
        self.sample_data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'independent_directors_pct': [75, 80, 70],
            'board_size': [8, 9, 10]
        })
    
    def test_scorer_initialization(self):
        """Test scorer initializes correctly."""
        self.assertIsInstance(self.scorer, GovernanceScorer)
        self.assertIn('board_composition_weight', self.scorer.weights)
    
    def test_board_composition_calculation(self):
        """Test board composition score calculation."""
        result = self.scorer.calculate_board_composition_score(self.sample_data)
        self.assertIn('board_composition_score', result.columns)
        self.assertTrue(all(result['board_composition_score'] >= 0))

class TestFinancialDataCollector(unittest.TestCase):
    """Test cases for FinancialDataCollector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = FinancialDataCollector()
    
    def test_collector_initialization(self):
        """Test collector initializes correctly."""
        self.assertIsInstance(self.collector, FinancialDataCollector)

class TestCorrelationAnalyzer(unittest.TestCase):
    """Test cases for CorrelationAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = CorrelationAnalyzer()
        
        # Create sample data for testing
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'esg_score': [75, 70, 80, 65, 85],
            'environmental_score': [80, 75, 85, 60, 90],
            'social_score': [70, 65, 75, 70, 80],
            'governance_score': [75, 70, 80, 65, 85],
            'price_change_pct': [15, 12, 18, 8, 25],
            'volatility': [25, 20, 22, 30, 45],
            'pe_ratio': [28, 25, 30, 22, 40]
        })
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        self.assertIsInstance(self.analyzer, CorrelationAnalyzer)
    
    def test_correlation_calculation(self):
        """Test correlation calculation."""
        correlations = self.analyzer.calculate_correlations(self.sample_data)
        self.assertIsInstance(correlations, dict)
        
        if correlations:  # If any correlations were calculated
            # Check structure
            first_key = list(correlations.keys())[0]
            self.assertIsInstance(correlations[first_key], dict)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete ESG scoring pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'sector': ['Technology', 'Technology', 'Technology'],
            'carbon_emissions': [1000, 1500, 800],
            'revenue': [100000, 120000, 95000],
            'renewable_energy_pct': [80, 60, 90],
            'gender_diversity_pct': [45, 40, 48],
            'independent_directors_pct': [75, 80, 70],
            'board_size': [8, 9, 10]
        })
    
    def test_complete_scoring_pipeline(self):
        """Test complete ESG scoring pipeline."""
        # Initialize scorers
        env_scorer = EnvironmentalScorer()
        social_scorer = SocialScorer()
        gov_scorer = GovernanceScorer()
        
        # Calculate scores
        result = env_scorer.calculate_environmental_score(self.sample_data)
        result = social_scorer.calculate_social_score(result)
        result = gov_scorer.calculate_governance_score(result)
        
        # Check all scores are present
        self.assertIn('environmental_score', result.columns)
        self.assertIn('social_score', result.columns)
        self.assertIn('governance_score', result.columns)
        
        # Check score ranges
        for score_col in ['environmental_score', 'social_score', 'governance_score']:
            self.assertTrue(all(result[score_col] >= 0))
            self.assertTrue(all(result[score_col] <= 100))

if __name__ == '__main__':
    unittest.main()