"""
Correlation analysis module for examining relationships between ESG scores 
and financial performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """Analyzer for ESG-financial performance correlations."""
    
    def __init__(self):
        """Initialize correlation analyzer."""
        self.results = {}
        
    def load_data(self, esg_data_path: str, financial_data_path: str) -> pd.DataFrame:
        """
        Load and merge ESG and financial data.
        
        Args:
            esg_data_path: Path to ESG scores CSV file
            financial_data_path: Path to financial metrics CSV file
            
        Returns:
            Merged DataFrame with ESG and financial data
        """
        logger.info("Loading ESG and financial data")
        
        try:
            esg_df = pd.read_csv(esg_data_path)
            financial_df = pd.read_csv(financial_data_path)
            
            # Merge on ticker/company identifier
            if 'ticker' in esg_df.columns and 'ticker' in financial_df.columns:
                merged_df = pd.merge(esg_df, financial_df, on='ticker', how='inner')
            elif 'company' in esg_df.columns and 'company' in financial_df.columns:
                merged_df = pd.merge(esg_df, financial_df, on='company', how='inner')
            else:
                logger.error("No common identifier found for merging datasets")
                return pd.DataFrame()
            
            logger.info(f"Merged dataset contains {len(merged_df)} companies")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_correlations(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate correlations between ESG scores and financial metrics.
        
        Args:
            data: Merged DataFrame with ESG and financial data
            
        Returns:
            Dictionary containing correlation results
        """
        logger.info("Calculating ESG-financial correlations")
        
        # Define ESG score columns
        esg_columns = [col for col in data.columns if 'score' in col.lower() and 
                      any(keyword in col.lower() for keyword in ['esg', 'environmental', 'social', 'governance'])]
        
        # Define financial metric columns
        financial_columns = [
            'latest_price', 'price_change_pct', 'volatility', 'market_cap',
            'pe_ratio', 'pb_ratio', 'roe', 'roa', 'profit_margin',
            'revenue_growth', 'earnings_growth', 'dividend_yield', 'beta'
        ]
        
        # Filter to available columns
        available_esg = [col for col in esg_columns if col in data.columns]
        available_financial = [col for col in financial_columns if col in data.columns]
        
        correlation_results = {}
        
        for esg_col in available_esg:
            correlation_results[esg_col] = {}
            
            for fin_col in available_financial:
                # Calculate Pearson correlation
                if data[esg_col].notna().sum() > 10 and data[fin_col].notna().sum() > 10:
                    correlation, p_value = stats.pearsonr(
                        data[esg_col].dropna(), 
                        data[fin_col].dropna()
                    )
                    
                    correlation_results[esg_col][fin_col] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'significance': 'significant' if p_value < 0.05 else 'not_significant',
                        'sample_size': min(data[esg_col].notna().sum(), data[fin_col].notna().sum())
                    }
        
        self.results['correlations'] = correlation_results
        logger.info(f"Calculated correlations for {len(available_esg)} ESG metrics and {len(available_financial)} financial metrics")
        
        return correlation_results
    
    def sector_analysis(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Perform sector-specific correlation analysis.
        
        Args:
            data: Merged DataFrame with sector information
            
        Returns:
            Dictionary containing sector-specific results
        """
        logger.info("Performing sector-specific analysis")
        
        if 'sector' not in data.columns:
            logger.warning("Sector information not available for sector analysis")
            return {}
        
        sector_results = {}
        
        for sector in data['sector'].unique():
            if pd.isna(sector):
                continue
                
            sector_data = data[data['sector'] == sector]
            
            if len(sector_data) < 5:  # Skip sectors with too few companies
                continue
            
            sector_correlations = self.calculate_correlations(sector_data)
            sector_results[sector] = {
                'correlations': sector_correlations,
                'company_count': len(sector_data),
                'avg_esg_score': sector_data.get('esg_score', pd.Series()).mean() if 'esg_score' in sector_data.columns else None
            }
        
        self.results['sector_analysis'] = sector_results
        return sector_results
    
    def risk_return_analysis(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze relationship between ESG scores and risk-return characteristics.
        
        Args:
            data: DataFrame with ESG scores and financial data
            
        Returns:
            Dictionary containing risk-return analysis results
        """
        logger.info("Performing ESG risk-return analysis")
        
        risk_return_results = {}
        
        # Define ESG score column (use overall ESG score if available, otherwise combine components)
        esg_col = None
        if 'esg_score' in data.columns:
            esg_col = 'esg_score'
        elif all(col in data.columns for col in ['environmental_score', 'social_score', 'governance_score']):
            data['combined_esg_score'] = (
                data['environmental_score'] + data['social_score'] + data['governance_score']
            ) / 3
            esg_col = 'combined_esg_score'
        
        if esg_col and 'price_change_pct' in data.columns and 'volatility' in data.columns:
            # Create ESG quintiles
            data['esg_quintile'] = pd.qcut(data[esg_col], q=5, labels=['Q1_Low', 'Q2', 'Q3', 'Q4', 'Q5_High'])
            
            # Calculate risk-return metrics by quintile
            quintile_analysis = {}
            for quintile in data['esg_quintile'].unique():
                if pd.isna(quintile):
                    continue
                    
                quintile_data = data[data['esg_quintile'] == quintile]
                
                quintile_analysis[str(quintile)] = {
                    'avg_return': quintile_data['price_change_pct'].mean(),
                    'avg_volatility': quintile_data['volatility'].mean(),
                    'company_count': len(quintile_data),
                    'avg_esg_score': quintile_data[esg_col].mean(),
                    'sharpe_ratio': quintile_data.get('sharpe_ratio', pd.Series()).mean() if 'sharpe_ratio' in quintile_data.columns else None
                }
            
            risk_return_results['quintile_analysis'] = quintile_analysis
            
            # Calculate overall ESG-performance correlations
            risk_return_results['esg_return_correlation'] = stats.pearsonr(
                data[esg_col].dropna(), data['price_change_pct'].dropna()
            )[0] if len(data[esg_col].dropna()) > 10 else None
            
            risk_return_results['esg_risk_correlation'] = stats.pearsonr(
                data[esg_col].dropna(), data['volatility'].dropna()
            )[0] if len(data[esg_col].dropna()) > 10 else None
        
        self.results['risk_return_analysis'] = risk_return_results
        return risk_return_results
    
    def generate_correlation_heatmap(self, data: pd.DataFrame, output_path: str = "data/processed/correlation_heatmap.png") -> None:
        """
        Generate correlation heatmap visualization.
        
        Args:
            data: DataFrame with ESG and financial data
            output_path: Path to save the heatmap image
        """
        logger.info("Generating correlation heatmap")
        
        # Select relevant columns for heatmap
        esg_columns = [col for col in data.columns if 'score' in col.lower() and 
                      any(keyword in col.lower() for keyword in ['environmental', 'social', 'governance'])]
        
        financial_columns = [
            'price_change_pct', 'volatility', 'pe_ratio', 'roe', 'roa', 
            'profit_margin', 'revenue_growth', 'dividend_yield'
        ]
        
        # Filter to available columns
        available_cols = [col for col in esg_columns + financial_columns if col in data.columns]
        
        if len(available_cols) > 3:
            # Calculate correlation matrix
            corr_matrix = data[available_cols].corr()
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                       square=True, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
            plt.title('ESG Scores vs Financial Metrics Correlation Heatmap')
            plt.tight_layout()
            
            # Save plot
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Correlation heatmap saved to {output_path}")
    
    def save_analysis_results(self, output_path: str = "data/processed/correlation_analysis_results.csv") -> None:
        """
        Save correlation analysis results to CSV.
        
        Args:
            output_path: Path to save the results CSV file
        """
        logger.info("Saving correlation analysis results")
        
        if 'correlations' not in self.results:
            logger.warning("No correlation results to save")
            return
        
        # Flatten correlation results for CSV output
        flattened_results = []
        
        for esg_metric, financial_correlations in self.results['correlations'].items():
            for financial_metric, corr_data in financial_correlations.items():
                flattened_results.append({
                    'esg_metric': esg_metric,
                    'financial_metric': financial_metric,
                    'correlation': corr_data['correlation'],
                    'p_value': corr_data['p_value'],
                    'significance': corr_data['significance'],
                    'sample_size': corr_data['sample_size'],
                    'analysis_date': datetime.now().isoformat()
                })
        
        results_df = pd.DataFrame(flattened_results)
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"Correlation analysis results saved to {output_path}")
    
    def generate_summary_report(self) -> str:
        """
        Generate a text summary of key findings.
        
        Returns:
            String containing summary report
        """
        logger.info("Generating correlation analysis summary report")
        
        report = ["ESG-Financial Performance Correlation Analysis Summary", "=" * 55, ""]
        
        if 'correlations' in self.results:
            # Find strongest correlations
            strongest_correlations = []
            for esg_metric, financial_correlations in self.results['correlations'].items():
                for financial_metric, corr_data in financial_correlations.items():
                    if corr_data['significance'] == 'significant':
                        strongest_correlations.append((
                            esg_metric, financial_metric, 
                            corr_data['correlation'], corr_data['p_value']
                        ))
            
            # Sort by absolute correlation strength
            strongest_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            report.append("Top 10 Significant Correlations:")
            report.append("-" * 35)
            
            for i, (esg, financial, corr, p_val) in enumerate(strongest_correlations[:10]):
                direction = "positive" if corr > 0 else "negative"
                report.append(f"{i+1:2d}. {esg} vs {financial}")
                report.append(f"    Correlation: {corr:.3f} ({direction}), p-value: {p_val:.4f}")
                report.append("")
        
        if 'risk_return_analysis' in self.results and 'quintile_analysis' in self.results['risk_return_analysis']:
            report.append("ESG Quintile Performance Analysis:")
            report.append("-" * 35)
            
            quintile_data = self.results['risk_return_analysis']['quintile_analysis']
            for quintile, metrics in quintile_data.items():
                report.append(f"{quintile}: Avg Return {metrics['avg_return']:.2f}%, "
                            f"Avg Volatility {metrics['avg_volatility']:.2f}%")
        
        report.append(f"\nAnalysis completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(report)

def main():
    """Example usage of CorrelationAnalyzer."""
    analyzer = CorrelationAnalyzer()
    
    # Example file paths - these would be actual data files
    esg_data_path = "data/processed/esg_scores_combined.csv"
    financial_data_path = "data/processed/financial_metrics.csv"
    
    # Check if data files exist
    if not (os.path.exists(esg_data_path) and os.path.exists(financial_data_path)):
        logger.warning("Required data files not found. Creating placeholder analysis.")
        return
    
    # Load and analyze data
    data = analyzer.load_data(esg_data_path, financial_data_path)
    
    if not data.empty:
        # Perform correlation analysis
        correlations = analyzer.calculate_correlations(data)
        
        # Sector analysis
        sector_results = analyzer.sector_analysis(data)
        
        # Risk-return analysis
        risk_return_results = analyzer.risk_return_analysis(data)
        
        # Generate visualizations
        analyzer.generate_correlation_heatmap(data)
        
        # Save results
        analyzer.save_analysis_results()
        
        # Generate summary report
        summary = analyzer.generate_summary_report()
        print(summary)
        
        logger.info("Correlation analysis completed successfully")
    else:
        logger.error("No data available for analysis")

if __name__ == "__main__":
    main()