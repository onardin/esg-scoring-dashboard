"""
Governance scoring module for ESG analysis.
Calculates governance scores based on board composition and independence,
executive compensation alignment, transparency and disclosure quality, 
and risk management practices.
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

class GovernanceScorer:
    """Governance scoring component of ESG analysis."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize governance scorer with configuration."""
        self.config = self._load_config(config_path)
        self.weights = self.config.get('scoring', {}).get('governance', {})
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {
                'scoring': {
                    'governance': {
                        'board_composition_weight': 0.3,
                        'executive_compensation_weight': 0.25,
                        'transparency_weight': 0.25,
                        'risk_management_weight': 0.2
                    }
                }
            }
    
    def calculate_board_composition_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate board composition and independence scores.
        
        Args:
            data: DataFrame with board composition data
            
        Returns:
            DataFrame with board composition scores
        """
        logger.info("Calculating board composition scores")
        
        result_data = data.copy()
        
        board_components = []
        
        if 'independent_directors_pct' in data.columns:
            # Higher percentage of independent directors = better score
            # Target is typically 75-80%+
            independence_score = np.minimum(100, (data['independent_directors_pct'] / 80) * 100)
            board_components.append(independence_score)
        
        if 'board_diversity_pct' in data.columns:
            # Board diversity score
            diversity_score = np.minimum(100, data['board_diversity_pct'] * 2)
            board_components.append(diversity_score)
        
        if 'board_size' in data.columns:
            # Optimal board size is typically 7-12 members
            size_score = np.where(
                (data['board_size'] >= 7) & (data['board_size'] <= 12),
                100,
                np.maximum(0, 100 - 10 * np.abs(data['board_size'] - 9.5))
            )
            board_components.append(size_score)
        
        if 'ceo_chair_separation' in data.columns:
            # CEO and Chairman roles should be separate (1=separated, 0=combined)
            separation_score = data['ceo_chair_separation'] * 100
            board_components.append(separation_score)
        
        if 'audit_committee_independence' in data.columns:
            # Audit committee should be fully independent
            audit_score = data['audit_committee_independence'] * 100
            board_components.append(audit_score)
        
        # Calculate overall board composition score
        if board_components:
            result_data['board_composition_score'] = np.mean(board_components, axis=0)
        else:
            logger.warning("Missing board composition data. Using placeholder scores.")
            result_data['board_composition_score'] = 55  # Slightly above average
            
        return result_data
    
    def calculate_executive_compensation_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate executive compensation alignment scores.
        
        Args:
            data: DataFrame with executive compensation data
            
        Returns:
            DataFrame with compensation scores
        """
        logger.info("Calculating executive compensation scores")
        
        result_data = data.copy()
        
        compensation_components = []
        
        if 'pay_for_performance_alignment' in data.columns:
            # Direct pay-for-performance score (assumed 0-100)
            compensation_components.append(data['pay_for_performance_alignment'])
        
        if 'ceo_to_median_pay_ratio' in data.columns:
            # Lower ratios are generally better (score inversely)
            # Typical "good" ratios are 20-50x, "concerning" ratios are >300x
            ratio_score = np.where(
                data['ceo_to_median_pay_ratio'] <= 50,
                100,
                np.maximum(0, 100 - (data['ceo_to_median_pay_ratio'] - 50) / 5)
            )
            compensation_components.append(ratio_score)
        
        if 'long_term_incentive_pct' in data.columns:
            # Higher percentage of long-term incentives = better alignment
            lti_score = np.minimum(100, data['long_term_incentive_pct'])
            compensation_components.append(lti_score)
        
        if 'clawback_policy' in data.columns:
            # Presence of clawback policy (1=yes, 0=no)
            clawback_score = data['clawback_policy'] * 100
            compensation_components.append(clawback_score)
        
        if 'say_on_pay_support' in data.columns:
            # Shareholder support for executive compensation
            support_score = np.minimum(100, data['say_on_pay_support'])
            compensation_components.append(support_score)
        
        # Calculate overall compensation score
        if compensation_components:
            result_data['executive_compensation_score'] = np.mean(compensation_components, axis=0)
        else:
            logger.warning("Missing compensation data. Using placeholder scores.")
            result_data['executive_compensation_score'] = 50  # Neutral score
            
        return result_data
    
    def calculate_transparency_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate transparency and disclosure quality scores.
        
        Args:
            data: DataFrame with transparency metrics
            
        Returns:
            DataFrame with transparency scores
        """
        logger.info("Calculating transparency scores")
        
        result_data = data.copy()
        
        transparency_components = []
        
        if 'financial_reporting_quality' in data.columns:
            # Financial reporting quality score (assumed 0-100)
            transparency_components.append(data['financial_reporting_quality'])
        
        if 'esg_disclosure_completeness' in data.columns:
            # ESG disclosure completeness percentage
            disclosure_score = np.minimum(100, data['esg_disclosure_completeness'])
            transparency_components.append(disclosure_score)
        
        if 'stakeholder_engagement_score' in data.columns:
            # Stakeholder engagement quality (assumed 0-100)
            transparency_components.append(data['stakeholder_engagement_score'])
        
        if 'third_party_assurance' in data.columns:
            # Third-party assurance of reports (1=yes, 0=no)
            assurance_score = data['third_party_assurance'] * 100
            transparency_components.append(assurance_score)
        
        if 'timely_disclosure_score' in data.columns:
            # Timeliness of regulatory disclosures (assumed 0-100)
            transparency_components.append(data['timely_disclosure_score'])
        
        # Check for presence of various governance documents
        governance_docs = ['code_of_conduct', 'whistleblower_policy', 'insider_trading_policy']
        doc_scores = []
        for doc in governance_docs:
            if doc in data.columns:
                doc_scores.append(data[doc] * 100)
        
        if doc_scores:
            policy_score = np.mean(doc_scores, axis=0)
            transparency_components.append(policy_score)
        
        # Calculate overall transparency score
        if transparency_components:
            result_data['transparency_score'] = np.mean(transparency_components, axis=0)
        else:
            logger.warning("Missing transparency data. Using placeholder scores.")
            result_data['transparency_score'] = 48  # Below average score
            
        return result_data
    
    def calculate_risk_management_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk management practices scores.
        
        Args:
            data: DataFrame with risk management metrics
            
        Returns:
            DataFrame with risk management scores
        """
        logger.info("Calculating risk management scores")
        
        result_data = data.copy()
        
        risk_components = []
        
        if 'risk_committee_effectiveness' in data.columns:
            # Risk committee effectiveness score (assumed 0-100)
            risk_components.append(data['risk_committee_effectiveness'])
        
        if 'cybersecurity_preparedness' in data.columns:
            # Cybersecurity preparedness score (assumed 0-100)
            risk_components.append(data['cybersecurity_preparedness'])
        
        if 'compliance_violations' in data.columns:
            # Lower number of violations = higher score
            max_violations = result_data['compliance_violations'].quantile(0.95)
            violation_score = 100 * (1 - result_data['compliance_violations'] / max_violations)
            risk_components.append(np.maximum(0, violation_score))
        
        if 'business_continuity_score' in data.columns:
            # Business continuity planning score (assumed 0-100)
            risk_components.append(data['business_continuity_score'])
        
        if 'regulatory_fines_amount' in data.columns:
            # Lower fines = better score (normalize by revenue if available)
            if 'revenue' in data.columns:
                fines_ratio = data['regulatory_fines_amount'] / data['revenue']
                max_ratio = fines_ratio.quantile(0.95)
                fines_score = 100 * (1 - fines_ratio / max_ratio)
            else:
                max_fines = data['regulatory_fines_amount'].quantile(0.95)
                fines_score = 100 * (1 - data['regulatory_fines_amount'] / max_fines)
            
            risk_components.append(np.maximum(0, fines_score))
        
        # Calculate overall risk management score
        if risk_components:
            result_data['risk_management_score'] = np.mean(risk_components, axis=0)
        else:
            logger.warning("Missing risk management data. Using placeholder scores.")
            result_data['risk_management_score'] = 52  # Slightly above average
            
        return result_data
    
    def calculate_governance_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate overall governance scores using weighted components.
        
        Args:
            data: DataFrame with company governance data
            
        Returns:
            DataFrame with governance scores
        """
        logger.info("Calculating overall governance scores")
        
        # Calculate component scores
        result_data = self.calculate_board_composition_score(data)
        result_data = self.calculate_executive_compensation_score(result_data)
        result_data = self.calculate_transparency_score(result_data)
        result_data = self.calculate_risk_management_score(result_data)
        
        # Calculate weighted overall score
        result_data['governance_score'] = (
            result_data['board_composition_score'] * self.weights.get('board_composition_weight', 0.3) +
            result_data['executive_compensation_score'] * self.weights.get('executive_compensation_weight', 0.25) +
            result_data['transparency_score'] * self.weights.get('transparency_weight', 0.25) +
            result_data['risk_management_score'] * self.weights.get('risk_management_weight', 0.2)
        )
        
        # Add scoring metadata
        result_data['governance_score_date'] = datetime.now().isoformat()
        result_data['governance_score_components'] = 'board_composition,executive_compensation,transparency,risk_management'
        
        logger.info(f"Governance scores calculated for {len(result_data)} companies")
        
        return result_data
    
    def save_scores(self, data: pd.DataFrame, filename: str = "governance_scores.csv") -> None:
        """Save governance scores to processed data directory."""
        filepath = os.path.join("data", "processed", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, index=False)
        logger.info(f"Governance scores saved to {filepath}")

def main():
    """Command line interface for governance scoring."""
    parser = argparse.ArgumentParser(description='Calculate governance ESG scores')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file with company data')
    parser.add_argument('--output', type=str, default='governance_scores.csv',
                       help='Output filename for governance scores')
    
    args = parser.parse_args()
    
    try:
        # Load input data
        data = pd.read_csv(args.input)
        logger.info(f"Loaded data for {len(data)} companies from {args.input}")
        
        # Initialize scorer and calculate scores
        scorer = GovernanceScorer()
        scored_data = scorer.calculate_governance_score(data)
        
        # Save results
        scorer.save_scores(scored_data, args.output)
        
        # Print summary statistics
        if 'governance_score' in scored_data.columns:
            avg_score = scored_data['governance_score'].mean()
            min_score = scored_data['governance_score'].min()
            max_score = scored_data['governance_score'].max()
            
            logger.info(f"Governance Scoring Summary:")
            logger.info(f"  Average Score: {avg_score:.2f}")
            logger.info(f"  Score Range: {min_score:.2f} - {max_score:.2f}")
        
    except Exception as e:
        logger.error(f"Error in governance scoring: {str(e)}")
        raise

if __name__ == "__main__":
    main()