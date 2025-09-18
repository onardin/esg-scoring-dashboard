"""
Streamlit dashboard application for ESG scoring visualization and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import os
import sys
from datetime import datetime, timedelta
import logging

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.scraper import ESGScraper
from data_collection.financial_data import FinancialDataCollector
from scoring.environmental import EnvironmentalScorer
from scoring.social import SocialScorer
from scoring.governance import GovernanceScorer
from analysis.correlation_analysis import CorrelationAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ESG Scoring Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ESGDashboard:
    """Main dashboard class for ESG scoring application."""
    
    def __init__(self):
        """Initialize dashboard with configuration."""
        self.config = self.load_config()
        
    def load_config(self) -> dict:
        """Load configuration from YAML file."""
        config_path = "config/config.yaml"
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            st.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
    
    def load_sample_data(self) -> pd.DataFrame:
        """Load or generate sample data for demonstration."""
        # Check for existing processed data
        data_files = [
            "data/processed/environmental_scores.csv",
            "data/processed/social_scores.csv", 
            "data/processed/governance_scores.csv",
            "data/processed/financial_metrics.csv"
        ]
        
        # For demo purposes, create sample data if files don't exist
        if not all(os.path.exists(f) for f in data_files):
            return self.generate_sample_data()
        
        # Load actual data if available
        try:
            env_data = pd.read_csv("data/processed/environmental_scores.csv")
            social_data = pd.read_csv("data/processed/social_scores.csv")
            gov_data = pd.read_csv("data/processed/governance_scores.csv")
            financial_data = pd.read_csv("data/processed/financial_metrics.csv")
            
            # Merge data
            merged_data = env_data
            if 'ticker' in social_data.columns:
                merged_data = merged_data.merge(social_data, on='ticker', how='outer')
                merged_data = merged_data.merge(gov_data, on='ticker', how='outer')
                merged_data = merged_data.merge(financial_data, on='ticker', how='outer')
            
            return merged_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return self.generate_sample_data()
    
    def generate_sample_data(self) -> pd.DataFrame:
        """Generate sample data for demonstration."""
        np.random.seed(42)  # For reproducible demo data
        
        companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'PG']
        sectors = ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 'Automotive',
                  'Technology', 'Technology', 'Financial', 'Healthcare', 'Consumer Staples']
        
        sample_data = []
        
        for i, (ticker, sector) in enumerate(zip(companies, sectors)):
            # Generate correlated ESG and financial metrics
            base_esg = np.random.normal(60, 15)
            
            data_point = {
                'ticker': ticker,
                'sector': sector,
                'environmental_score': np.clip(base_esg + np.random.normal(0, 5), 0, 100),
                'social_score': np.clip(base_esg + np.random.normal(0, 5), 0, 100),
                'governance_score': np.clip(base_esg + np.random.normal(0, 5), 0, 100),
                'latest_price': np.random.uniform(50, 500),
                'price_change_pct': np.random.normal(8, 20),
                'volatility': np.random.uniform(15, 40),
                'market_cap': np.random.uniform(100e9, 3000e9),
                'pe_ratio': np.random.uniform(10, 35),
                'roe': np.random.uniform(5, 25),
                'profit_margin': np.random.uniform(5, 30),
                'dividend_yield': np.random.uniform(0, 5),
                'beta': np.random.uniform(0.5, 2.0),
            }
            
            # Calculate overall ESG score
            data_point['esg_score'] = (
                data_point['environmental_score'] + 
                data_point['social_score'] + 
                data_point['governance_score']
            ) / 3
            
            sample_data.append(data_point)
        
        return pd.DataFrame(sample_data)
    
    def render_sidebar(self, data: pd.DataFrame):
        """Render sidebar with filters and controls."""
        st.sidebar.header("🌱 ESG Dashboard Controls")
        
        # Company selection
        companies = ['All'] + sorted(data['ticker'].unique().tolist())
        selected_companies = st.sidebar.multiselect(
            "Select Companies",
            companies,
            default=['All']
        )
        
        if 'All' in selected_companies:
            filtered_data = data
        else:
            filtered_data = data[data['ticker'].isin(selected_companies)]
        
        # Sector filter
        sectors = ['All'] + sorted(data['sector'].unique().tolist())
        selected_sector = st.sidebar.selectbox(
            "Select Sector",
            sectors
        )
        
        if selected_sector != 'All':
            filtered_data = filtered_data[filtered_data['sector'] == selected_sector]
        
        # ESG Score range
        if 'esg_score' in data.columns:
            min_score, max_score = st.sidebar.slider(
                "ESG Score Range",
                min_value=float(data['esg_score'].min()),
                max_value=float(data['esg_score'].max()),
                value=(float(data['esg_score'].min()), float(data['esg_score'].max())),
                step=1.0
            )
            
            filtered_data = filtered_data[
                (filtered_data['esg_score'] >= min_score) & 
                (filtered_data['esg_score'] <= max_score)
            ]
        
        return filtered_data
    
    def render_overview(self, data: pd.DataFrame):
        """Render overview section with key metrics."""
        st.header("📊 ESG Overview")
        
        if len(data) == 0:
            st.warning("No data available for selected filters.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_esg = data['esg_score'].mean() if 'esg_score' in data.columns else 0
            st.metric("Average ESG Score", f"{avg_esg:.1f}", delta=None)
        
        with col2:
            num_companies = len(data)
            st.metric("Companies Analyzed", f"{num_companies}", delta=None)
        
        with col3:
            top_performer = data.loc[data['esg_score'].idxmax()]['ticker'] if 'esg_score' in data.columns else 'N/A'
            st.metric("Top ESG Performer", top_performer, delta=None)
        
        with col4:
            avg_return = data['price_change_pct'].mean() if 'price_change_pct' in data.columns else 0
            st.metric("Average Return", f"{avg_return:.1f}%", delta=None)
        
        # ESG Score Distribution
        if 'esg_score' in data.columns:
            fig = px.histogram(
                data, x='esg_score', nbins=20,
                title="ESG Score Distribution",
                labels={'esg_score': 'ESG Score', 'count': 'Number of Companies'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_esg_breakdown(self, data: pd.DataFrame):
        """Render ESG component breakdown charts."""
        st.header("🔍 ESG Component Analysis")
        
        if len(data) == 0:
            st.warning("No data available for selected filters.")
            return
        
        # ESG Components Radar Chart
        if all(col in data.columns for col in ['environmental_score', 'social_score', 'governance_score']):
            avg_scores = {
                'Environmental': data['environmental_score'].mean(),
                'Social': data['social_score'].mean(),
                'Governance': data['governance_score'].mean()
            }
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=list(avg_scores.values()),
                theta=list(avg_scores.keys()),
                fill='toself',
                name='Average ESG Scores'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                title="Average ESG Component Scores",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Sector Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            if 'sector' in data.columns and 'esg_score' in data.columns:
                sector_avg = data.groupby('sector')['esg_score'].mean().sort_values(ascending=True)
                
                fig = px.bar(
                    x=sector_avg.values,
                    y=sector_avg.index,
                    orientation='h',
                    title="Average ESG Score by Sector",
                    labels={'x': 'Average ESG Score', 'y': 'Sector'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'ticker' in data.columns and 'esg_score' in data.columns:
                top_10 = data.nlargest(10, 'esg_score')
                
                fig = px.bar(
                    top_10,
                    x='esg_score',
                    y='ticker',
                    orientation='h',
                    title="Top 10 ESG Performers",
                    labels={'esg_score': 'ESG Score', 'ticker': 'Company'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_financial_correlation(self, data: pd.DataFrame):
        """Render financial correlation analysis."""
        st.header("💹 ESG-Financial Performance Correlation")
        
        if len(data) == 0:
            st.warning("No data available for selected filters.")
            return
        
        # Scatter plots
        col1, col2 = st.columns(2)
        
        with col1:
            if 'esg_score' in data.columns and 'price_change_pct' in data.columns:
                fig = px.scatter(
                    data, x='esg_score', y='price_change_pct',
                    color='sector' if 'sector' in data.columns else None,
                    hover_data=['ticker'] if 'ticker' in data.columns else None,
                    title="ESG Score vs Stock Return",
                    labels={'esg_score': 'ESG Score', 'price_change_pct': 'Price Change (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'esg_score' in data.columns and 'volatility' in data.columns:
                fig = px.scatter(
                    data, x='esg_score', y='volatility',
                    color='sector' if 'sector' in data.columns else None,
                    hover_data=['ticker'] if 'ticker' in data.columns else None,
                    title="ESG Score vs Volatility",
                    labels={'esg_score': 'ESG Score', 'volatility': 'Volatility (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        esg_cols = [col for col in numeric_cols if 'score' in col]
        financial_cols = [col for col in numeric_cols if col not in esg_cols and col != 'latest_price']
        
        if len(esg_cols) > 0 and len(financial_cols) > 0:
            corr_data = data[esg_cols + financial_cols].corr()
            
            fig = px.imshow(
                corr_data,
                title="ESG-Financial Metrics Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_company_comparison(self, data: pd.DataFrame):
        """Render company comparison section."""
        st.header("🏢 Company Comparison")
        
        if len(data) < 2:
            st.warning("Select at least 2 companies for comparison.")
            return
        
        # Company selection for detailed comparison
        selected_for_comparison = st.multiselect(
            "Select companies for detailed comparison:",
            data['ticker'].tolist(),
            default=data['ticker'].tolist()[:5] if len(data) >= 5 else data['ticker'].tolist()
        )
        
        if len(selected_for_comparison) < 2:
            st.warning("Please select at least 2 companies for comparison.")
            return
        
        comparison_data = data[data['ticker'].isin(selected_for_comparison)]
        
        # ESG Components Comparison
        if all(col in comparison_data.columns for col in ['environmental_score', 'social_score', 'governance_score']):
            fig = go.Figure()
            
            for _, company in comparison_data.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[company['environmental_score'], company['social_score'], company['governance_score']],
                    theta=['Environmental', 'Social', 'Governance'],
                    fill='toself',
                    name=company['ticker']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                title="ESG Components Comparison",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.subheader("Detailed Metrics Comparison")
        
        display_cols = ['ticker', 'sector', 'esg_score', 'environmental_score', 'social_score', 
                       'governance_score', 'price_change_pct', 'volatility', 'pe_ratio']
        available_cols = [col for col in display_cols if col in comparison_data.columns]
        
        st.dataframe(
            comparison_data[available_cols].round(2),
            use_container_width=True
        )
    
    def render_data_collection_tools(self):
        """Render data collection and scoring tools section."""
        st.header("🔧 Data Collection & Scoring Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Collection")
            
            companies_input = st.text_input(
                "Enter company tickers (comma-separated):",
                value="AAPL,MSFT,TSLA"
            )
            
            year_input = st.number_input(
                "Select year for data collection:",
                min_value=2020,
                max_value=datetime.now().year,
                value=datetime.now().year - 1
            )
            
            if st.button("Collect ESG Data"):
                with st.spinner("Collecting data..."):
                    # This would trigger actual data collection
                    st.success(f"Data collection initiated for {companies_input} (Year: {year_input})")
                    st.info("Note: This is a demo. Actual implementation would scrape real data.")
        
        with col2:
            st.subheader("Score Calculation")
            
            uploaded_file = st.file_uploader(
                "Upload company data CSV for scoring:",
                type=['csv']
            )
            
            scoring_type = st.selectbox(
                "Select scoring type:",
                ["All ESG Components", "Environmental Only", "Social Only", "Governance Only"]
            )
            
            if st.button("Calculate ESG Scores"):
                if uploaded_file is not None:
                    with st.spinner("Calculating scores..."):
                        # This would trigger actual scoring
                        st.success(f"ESG scoring completed for {scoring_type}")
                        st.info("Note: This is a demo. Actual implementation would process uploaded data.")
                else:
                    st.warning("Please upload a CSV file first.")
    
    def run(self):
        """Run the main dashboard application."""
        st.title("🌱 ESG Scoring Dashboard")
        st.markdown("*Environmental, Social, and Governance Performance Analytics*")
        
        # Load data
        try:
            data = self.load_sample_data()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return
        
        # Sidebar filters
        filtered_data = self.render_sidebar(data)
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🔍 ESG Breakdown", 
            "💹 Financial Correlation",
            "🏢 Company Comparison",
            "🔧 Tools"
        ])
        
        with tab1:
            self.render_overview(filtered_data)
        
        with tab2:
            self.render_esg_breakdown(filtered_data)
        
        with tab3:
            self.render_financial_correlation(filtered_data)
        
        with tab4:
            self.render_company_comparison(filtered_data)
        
        with tab5:
            self.render_data_collection_tools()
        
        # Footer
        st.markdown("---")
        st.markdown("*ESG Scoring Dashboard - Built with Streamlit*")

def main():
    """Main function to run the dashboard."""
    dashboard = ESGDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()