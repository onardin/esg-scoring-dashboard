"""
Environmental scoring module for ESG analysis.
Calculates environmental scores based on carbon intensity, renewable energy adoption,
environmental risk disclosures, and resource efficiency metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yaml
import argparse
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentalScorer:
    """Environmental scoring component of ESG analysis."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize environmental scorer with configuration."""
        self.config = self._load_config(config_path)
        self.weights = self.config.get('scoring', {}).get('environmental', {})
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {
                'scoring': {
                    'environmental': {
                        'carbon_weight': 0.3,
                        'renewable_weight': 0.25,
                        'resource_efficiency_weight': 0.25,
                        'disclosure_quality_weight': 0.2
                    }
                }
            }
    
    def calculate_carbon_intensity_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate carbon intensity scores for companies.
        
        Args:
            data: DataFrame with carbon emissions and revenue data
            
        Returns:
            DataFrame with carbon intensity scores
        """
        logger.info("Calculating carbon intensity scores")
        
        result_data = data.copy()
        
        # Calculate carbon intensity (emissions per revenue)
        if 'carbon_emissions' in data.columns and 'revenue' in data.columns:
            result_data['carbon_intensity'] = data['carbon_emissions'] / data['revenue']
            
            # Normalize to 0-100 scale (lower intensity = higher score)
            max_intensity = result_data['carbon_intensity'].quantile(0.95)  # Use 95th percentile to handle outliers
            result_data['carbon_intensity_score'] = np.maximum(
                0, 100 * (1 - result_data['carbon_intensity'] / max_intensity)
            )
        else:
            logger.warning("Missing carbon emissions or revenue data. Using placeholder scores.")
            result_data['carbon_intensity_score'] = 50  # Neutral score
            
        return result_data
    
    def calculate_renewable_energy_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate renewable energy adoption scores.
        
        Args:
            data: DataFrame with renewable energy usage data
            
        Returns:
            DataFrame with renewable energy scores
        """
        logger.info("Calculating renewable energy scores")
        
        result_data = data.copy()
        
        if 'renewable_energy_pct' in data.columns:
            # Direct percentage to score mapping
            result_data['renewable_energy_score'] = np.minimum(100, data['renewable_energy_pct'])
        else:
            logger.warning("Missing renewable energy data. Using placeholder scores.")
            result_data['renewable_energy_score'] = 40  # Below average score
            
        return result_data
    
    def calculate_resource_efficiency_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate resource efficiency scores based on water usage, waste management, etc.
        
        Args:
            data: DataFrame with resource usage data
            
        Returns:
            DataFrame with resource efficiency scores
        """
        logger.info("Calculating resource efficiency scores")
        
        result_data = data.copy()
        
        # Placeholder implementation - in real scenario, this would use actual resource metrics
        if 'water_usage_intensity' in data.columns and 'waste_recycling_rate' in data.columns:
            # Normalize water usage (lower = better)
            max_water = result_data['water_usage_intensity'].quantile(0.95)
            water_score = 100 * (1 - result_data['water_usage_intensity'] / max_water)
            
            # Waste recycling rate (higher = better)
            recycling_score = result_data['waste_recycling_rate']
            
            # Combined resource efficiency score
            result_data['resource_efficiency_score'] = (water_score + recycling_score) / 2
        else:
            logger.warning("Missing resource efficiency data. Using placeholder scores.")
            result_data['resource_efficiency_score'] = 45  # Below average score
            
        return result_data
    
    def calculate_disclosure_quality_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate environmental disclosure quality scores based on reporting completeness.
        
        Args:
            data: DataFrame with disclosure quality indicators
            
        Returns:
            DataFrame with disclosure quality scores
        """
        logger.info("Calculating environmental disclosure quality scores")
        
        result_data = data.copy()
        
        # Assess disclosure quality based on available environmental metrics
        disclosure_indicators = [
            'carbon_emissions', 'renewable_energy_pct', 'water_usage_intensity',
            'waste_recycling_rate', 'environmental_investments', 'green_revenue_pct'
        ]
        
        # Count available indicators
        available_indicators = sum(1 for col in disclosure_indicators if col in data.columns and not data[col].isna().all())
        total_indicators = len(disclosure_indicators)
        
        # Base disclosure score on completeness
        base_score = (available_indicators / total_indicators) * 100
        
        result_data['disclosure_quality_score'] = base_score
        
        return result_data
    
    def calculate_environmental_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate overall environmental scores using weighted components.
        
        Args:
            data: DataFrame with company environmental data
            
        Returns:
            DataFrame with environmental scores
        """
        logger.info("Calculating overall environmental scores")
        
        # Calculate component scores
        result_data = self.calculate_carbon_intensity_score(data)
        result_data = self.calculate_renewable_energy_score(result_data)
        result_data = self.calculate_resource_efficiency_score(result_data)
        result_data = self.calculate_disclosure_quality_score(result_data)
        
        # Calculate weighted overall score
        result_data['environmental_score'] = (
            result_data['carbon_intensity_score'] * self.weights.get('carbon_weight', 0.3) +
            result_data['renewable_energy_score'] * self.weights.get('renewable_weight', 0.25) +
            result_data['resource_efficiency_score'] * self.weights.get('resource_efficiency_weight', 0.25) +
            result_data['disclosure_quality_score'] * self.weights.get('disclosure_quality_weight', 0.2)
        )
        
        # Add scoring metadata
        result_data['environmental_score_date'] = datetime.now().isoformat()
        result_data['environmental_score_components'] = 'carbon_intensity,renewable_energy,resource_efficiency,disclosure_quality'
        
        logger.info(f"Environmental scores calculated for {len(result_data)} companies")
        
        return result_data
    
    def save_scores(self, data: pd.DataFrame, filename: str = "environmental_scores.csv") -> None:
        """Save environmental scores to processed data directory."""
        filepath = os.path.join("data", "processed", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, index=False)
        logger.info(f"Environmental scores saved to {filepath}")

def main():
    """Command line interface for environmental scoring."""
    parser = argparse.ArgumentParser(description='Calculate environmental ESG scores')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file with company data')
    parser.add_argument('--output', type=str, default='environmental_scores.csv',
                       help='Output filename for environmental scores')
    
    args = parser.parse_args()
    
    try:
        # Load input data
        data = pd.read_csv(args.input)
        logger.info(f"Loaded data for {len(data)} companies from {args.input}")
        
        # Initialize scorer and calculate scores
        scorer = EnvironmentalScorer()
        scored_data = scorer.calculate_environmental_score(data)
        
        # Save results
        scorer.save_scores(scored_data, args.output)
        
        # Print summary statistics
        if 'environmental_score' in scored_data.columns:
            avg_score = scored_data['environmental_score'].mean()
            min_score = scored_data['environmental_score'].min()
            max_score = scored_data['environmental_score'].max()
            
            logger.info(f"Environmental Scoring Summary:")
            logger.info(f"  Average Score: {avg_score:.2f}")
            logger.info(f"  Score Range: {min_score:.2f} - {max_score:.2f}")
        
    except Exception as e:
        logger.error(f"Error in environmental scoring: {str(e)}")
        raise

if __name__ == "__main__":
    main()