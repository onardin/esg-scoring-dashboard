# esg-scoring-dashboard
# ESG Scoring Dashboard

A machine learning-powered dashboard that analyzes corporate Environmental, Social, and Governance (ESG) performance and correlates it with financial metrics.

## 🎯 Project Objectives

- **Automate ESG Scoring**: Build algorithmic ESG scores using corporate filings and sustainability reports
- **Financial Correlation**: Analyze relationships between ESG performance and financial returns
- **Interactive Visualization**: Provide stakeholders with an intuitive dashboard for ESG insights
- **Sector Benchmarking**: Enable peer comparisons across industries

## 🚀 Features

- **Automated Data Collection**: Scrapes SEC filings, financial data, and sustainability reports
- **Multi-dimensional Scoring**: Environmental, Social, and Governance scores with sub-metrics
- **Financial Integration**: Links ESG scores with stock performance, volatility, and risk metrics
- **Interactive Dashboard**: Streamlit-based interface with company comparisons and trend analysis
- **Sector Analysis**: Industry benchmarking and peer group comparisons

## 📊 Data Sources

- **SEC EDGAR**: 10-K/10-Q filings for corporate disclosures
- **Yahoo Finance**: Stock prices and financial metrics
- **Company Websites**: Sustainability and ESG reports
- **SASB Standards**: Sector-specific ESG materiality framework

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/esg-scoring-dashboard.git
   cd esg-scoring-dashboard
   ```

2. **Create virtual environment**
   ```bash
   python -m venv esg_env
   source esg_env/bin/activate  # On Windows: esg_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (first-time setup)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('vader_lexicon')
   ```

## 📈 Usage

### 1. Data Collection
```bash
python src/data_collection/scraper.py --companies "AAPL,MSFT,TSLA" --year 2023
```

### 2. Generate ESG Scores
```bash
python -m src.scoring.environmental --input data/processed/companies.csv
python -m src.scoring.social --input data/processed/companies.csv  
python -m src.scoring.governance --input data/processed/companies.csv
```

### 3. Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```

Navigate to `http://localhost:8501` to view the dashboard.

## 📁 Project Structure

```
esg-scoring-dashboard/
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── config/
│   └── config.yaml       # Configuration settings
├── data/
│   ├── raw/              # Raw scraped data
│   ├── processed/        # Cleaned datasets
│   └── external/         # Reference data (SASB standards, etc.)
├── src/
│   ├── data_collection/  # Web scraping and data gathering
│   ├── scoring/          # ESG scoring algorithms
│   ├── analysis/         # Statistical analysis and correlations
│   └── dashboard/        # Streamlit dashboard components
├── notebooks/            # Jupyter notebooks for exploration
├── tests/               # Unit tests
└── docs/                # Additional documentation
```

## 🔬 ESG Scoring Methodology

### Environmental (E)
- Carbon intensity analysis
- Renewable energy adoption
- Environmental risk disclosures
- Resource efficiency metrics

### Social (S)
- Employee diversity and inclusion
- Community impact initiatives
- Supply chain responsibility
- Product safety and quality

### Governance (G)
- Board composition and independence
- Executive compensation alignment
- Transparency and disclosure quality
- Risk management practices

## 📊 Sample Results

| Company | E Score | S Score | G Score | Overall ESG | Stock Return (YoY) | Volatility |
|---------|---------|---------|---------|-------------|-------------------|------------|
| AAPL    | 85      | 78      | 92      | 85          | +12.3%            | 0.24       |
| MSFT    | 82      | 85      | 88      | 85          | +8.7%             | 0.22       |
| TSLA    | 92      | 65      | 45      | 67          | +25.1%            | 0.45       |

## 🚧 Development Roadmap

### Phase 1: MVP (Weeks 1-3) ✅
- [x] Basic data collection pipeline
- [x] Simple ESG scoring algorithm
- [x] Interactive dashboard prototype

### Phase 2: Enhancement (Weeks 4-6)
- [ ] Advanced NLP for text analysis
- [ ] Machine learning model improvements
- [ ] Additional data sources integration

### Phase 3: Production (Weeks 7-8)
- [ ] API development
- [ ] Database optimization
- [ ] Performance monitoring

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/esg-scoring-dashboard

## 🙏 Acknowledgments

- SASB Foundation for ESG materiality standards
- SEC for providing open access to corporate filings
- Climate fintech community for inspiration and feedback
