"""
Social scoring module for ESG analysis.
Calculates social scores based on employee diversity and inclusion,
community impact initiatives, supply chain responsibility, and product safety.
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

class SocialScorer:
    """Social scoring component of ESG analysis."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize social scorer with configuration."""
        self.config = self._load_config(config_path)
        self.weights = self.config.get('scoring', {}).get('social', {})
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {
                'scoring': {
                    'social': {
                        'diversity_weight': 0.3,
                        'community_impact_weight': 0.25,
                        'supply_chain_weight': 0.25,
                        'product_safety_weight': 0.2
                    }
                }
            }
    
    def calculate_diversity_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate diversity and inclusion scores for companies.
        
        Args:
            data: DataFrame with diversity metrics data
            
        Returns:
            DataFrame with diversity scores
        """
        logger.info("Calculating diversity and inclusion scores")
        
        result_data = data.copy()
        
        # Calculate diversity metrics
        diversity_components = []
        
        if 'gender_diversity_pct' in data.columns:
            # Score based on gender diversity (closer to 50% = higher score)
            gender_score = 100 - 2 * np.abs(data['gender_diversity_pct'] - 50)
            diversity_components.append(gender_score)
        
        if 'ethnic_diversity_pct' in data.columns:
            # Higher ethnic diversity generally scores better
            ethnic_score = np.minimum(100, data['ethnic_diversity_pct'] * 2)
            diversity_components.append(ethnic_score)
        
        if 'leadership_diversity_pct' in data.columns:
            # Leadership diversity is particularly important
            leadership_score = np.minimum(100, data['leadership_diversity_pct'] * 2.5)
            diversity_components.append(leadership_score)
        
        if 'pay_equity_score' in data.columns:
            # Pay equity score (assumed to be already 0-100 scale)
            diversity_components.append(data['pay_equity_score'])
        
        # Calculate overall diversity score
        if diversity_components:
            result_data['diversity_score'] = np.mean(diversity_components, axis=0)
        else:
            logger.warning("Missing diversity data. Using placeholder scores.")
            result_data['diversity_score'] = 45  # Below average score
            
        return result_data
    
    def calculate_community_impact_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate community impact scores based on charitable giving, local hiring, etc.
        
        Args:
            data: DataFrame with community impact data
            
        Returns:
            DataFrame with community impact scores
        """
        logger.info("Calculating community impact scores")
        
        result_data = data.copy()
        
        community_components = []
        
        if 'charitable_giving_pct_revenue' in data.columns:
            # Score based on charitable giving as percentage of revenue
            # Top performers typically give 1-2% of revenue
            giving_score = np.minimum(100, data['charitable_giving_pct_revenue'] * 50)
            community_components.append(giving_score)
        
        if 'volunteer_hours_per_employee' in data.columns:
            # Score based on employee volunteer hours (normalize to 0-100)
            max_hours = 40  # Assume 40 hours per year is excellent
            volunteer_score = np.minimum(100, (data['volunteer_hours_per_employee'] / max_hours) * 100)
            community_components.append(volunteer_score)
        
        if 'local_hiring_pct' in data.columns:
            # Score based on local hiring percentage
            local_score = np.minimum(100, data['local_hiring_pct'])
            community_components.append(local_score)
        
        if 'community_investment_score' in data.columns:
            # Direct community investment score (assumed 0-100)
            community_components.append(data['community_investment_score'])
        
        # Calculate overall community impact score
        if community_components:
            result_data['community_impact_score'] = np.mean(community_components, axis=0)
        else:
            logger.warning("Missing community impact data. Using placeholder scores.")
            result_data['community_impact_score'] = 40  # Below average score
            
        return result_data
    
    def calculate_supply_chain_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate supply chain responsibility scores.
        
        Args:
            data: DataFrame with supply chain metrics
            
        Returns:
            DataFrame with supply chain scores
        """
        logger.info("Calculating supply chain responsibility scores")
        
        result_data = data.copy()
        
        supply_chain_components = []
        
        if 'supplier_diversity_pct' in data.columns:
            # Score based on supplier diversity
            supplier_score = np.minimum(100, data['supplier_diversity_pct'] * 2)
            supply_chain_components.append(supplier_score)
        
        if 'supply_chain_audit_coverage' in data.columns:
            # Score based on supply chain audit coverage
            audit_score = np.minimum(100, data['supply_chain_audit_coverage'])
            supply_chain_components.append(audit_score)
        
        if 'labor_standards_compliance' in data.columns:
            # Labor standards compliance score (assumed 0-100)
            supply_chain_components.append(data['labor_standards_compliance'])
        
        if 'sustainable_sourcing_pct' in data.columns:
            # Sustainable sourcing percentage
            sourcing_score = np.minimum(100, data['sustainable_sourcing_pct'])
            supply_chain_components.append(sourcing_score)
        
        # Calculate overall supply chain score
        if supply_chain_components:
            result_data['supply_chain_score'] = np.mean(supply_chain_components, axis=0)
        else:
            logger.warning("Missing supply chain data. Using placeholder scores.")
            result_data['supply_chain_score'] = 50  # Neutral score
            
        return result_data
    
    def calculate_product_safety_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate product safety and quality scores.
        
        Args:
            data: DataFrame with product safety metrics
            
        Returns:
            DataFrame with product safety scores
        """
        logger.info("Calculating product safety scores")
        
        result_data = data.copy()
        
        safety_components = []
        
        if 'product_recall_rate' in data.columns:
            # Lower recall rate = higher score
            max_recall_rate = result_data['product_recall_rate'].quantile(0.95)
            recall_score = 100 * (1 - result_data['product_recall_rate'] / max_recall_rate)
            safety_components.append(np.maximum(0, recall_score))
        
        if 'safety_incidents_per_year' in data.columns:
            # Lower incident rate = higher score
            max_incidents = result_data['safety_incidents_per_year'].quantile(0.95)
            incident_score = 100 * (1 - result_data['safety_incidents_per_year'] / max_incidents)
            safety_components.append(np.maximum(0, incident_score))
        
        if 'customer_satisfaction_score' in data.columns:
            # Direct customer satisfaction score (assumed 0-100)
            safety_components.append(data['customer_satisfaction_score'])
        
        if 'quality_certifications' in data.columns:
            # Number of quality certifications (normalize to 0-100)
            max_certs = 10  # Assume 10 certifications is excellent
            cert_score = np.minimum(100, (data['quality_certifications'] / max_certs) * 100)
            safety_components.append(cert_score)
        
        # Calculate overall product safety score
        if safety_components:
            result_data['product_safety_score'] = np.mean(safety_components, axis=0)
        else:
            logger.warning("Missing product safety data. Using placeholder scores.")
            result_data['product_safety_score'] = 60  # Above average placeholder
            
        return result_data
    
    def calculate_social_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate overall social scores using weighted components.
        
        Args:
            data: DataFrame with company social data
            
        Returns:
            DataFrame with social scores
        """
        logger.info("Calculating overall social scores")
        
        # Calculate component scores
        result_data = self.calculate_diversity_score(data)
        result_data = self.calculate_community_impact_score(result_data)
        result_data = self.calculate_supply_chain_score(result_data)
        result_data = self.calculate_product_safety_score(result_data)
        
        # Calculate weighted overall score
        result_data['social_score'] = (
            result_data['diversity_score'] * self.weights.get('diversity_weight', 0.3) +
            result_data['community_impact_score'] * self.weights.get('community_impact_weight', 0.25) +
            result_data['supply_chain_score'] * self.weights.get('supply_chain_weight', 0.25) +
            result_data['product_safety_score'] * self.weights.get('product_safety_weight', 0.2)
        )
        
        # Add scoring metadata
        result_data['social_score_date'] = datetime.now().isoformat()
        result_data['social_score_components'] = 'diversity,community_impact,supply_chain,product_safety'
        
        logger.info(f"Social scores calculated for {len(result_data)} companies")
        
        return result_data
    
    def save_scores(self, data: pd.DataFrame, filename: str = "social_scores.csv") -> None:
        """Save social scores to processed data directory."""
        filepath = os.path.join("data", "processed", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, index=False)
        logger.info(f"Social scores saved to {filepath}")

def main():
    """Command line interface for social scoring."""
    parser = argparse.ArgumentParser(description='Calculate social ESG scores')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file with company data')
    parser.add_argument('--output', type=str, default='social_scores.csv',
                       help='Output filename for social scores')
    
    args = parser.parse_args()
    
    try:
        # Load input data
        data = pd.read_csv(args.input)
        logger.info(f"Loaded data for {len(data)} companies from {args.input}")
        
        # Initialize scorer and calculate scores
        scorer = SocialScorer()
        scored_data = scorer.calculate_social_score(data)
        
        # Save results
        scorer.save_scores(scored_data, args.output)
        
        # Print summary statistics
        if 'social_score' in scored_data.columns:
            avg_score = scored_data['social_score'].mean()
            min_score = scored_data['social_score'].min()
            max_score = scored_data['social_score'].max()
            
            logger.info(f"Social Scoring Summary:")
            logger.info(f"  Average Score: {avg_score:.2f}")
            logger.info(f"  Score Range: {min_score:.2f} - {max_score:.2f}")
        
    except Exception as e:
        logger.error(f"Error in social scoring: {str(e)}")
        raise

if __name__ == "__main__":
    main()