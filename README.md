# Stock Price Predictor with Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modern stock prediction dashboard using LSTM neural networks enhanced with financial news sentiment analysis for improved prediction accuracy.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ajaygm18/xrs.git
cd xrs

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_stock_predictor.py
```

Open your browser and navigate to `http://localhost:8501` to access the dashboard.

## Features

### Core Prediction Engine
- **LSTM Neural Networks**: Advanced deep learning model for time series prediction
- **Technical Indicators**: RSI, moving averages, and price pattern analysis
- **Multi-currency Support**: Automatic currency detection for international stocks
- **Comprehensive Metrics**: RMSE, directional accuracy, and confidence scores

### 🆕 Sentiment Analysis Integration
- **Dual Sentiment Analysis**: Combines TextBlob and VADER sentiment analyzers
- **Financial News Processing**: Analyzes market sentiment from financial news
- **Sentiment-Adjusted Predictions**: Integrates sentiment scores with LSTM predictions
- **Configurable Impact**: Adjustable sentiment weight and maximum adjustment parameters
- **Smart Fallback**: Uses price trend analysis when real news data is unavailable

### Interactive Dashboard
- **Real-time Data**: Live stock data via Yahoo Finance API
- **Interactive Charts**: Price history, volume, sentiment distribution
- **Prediction Comparison**: Visual comparison of different model outputs
- **Responsive Design**: Modern dark theme with gradient styling

## How Sentiment Analysis Improves Predictions

1. **News Sentiment Scoring**: Analyzes financial news headlines and content
2. **Multi-method Analysis**: Combines TextBlob (linguistic) and VADER (social media optimized) sentiment
3. **Confidence Weighting**: Weights sentiment impact by analysis confidence
4. **Bounded Adjustment**: Limits sentiment adjustment to prevent over-correction (default: ±3%)
5. **Accuracy Enhancement**: Boosts prediction accuracy when high-confidence sentiment is available

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: At least 4GB RAM (8GB recommended for training)
- **Storage**: 2GB free space for dependencies and models
- **OS**: Windows, macOS, or Linux

## Installation

### Option 1: Direct Installation
```bash
pip install -r requirements.txt
```

### Option 2: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using Conda
```bash
conda create -n stock-predictor python=3.9
conda activate stock-predictor
pip install -r requirements.txt
```

## Usage

### Running the Dashboard
```bash
streamlit run streamlit_stock_predictor.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Basic Workflow
1. **Enter Stock Symbol**: Type a stock symbol (e.g., AAPL, GOOGL, TSLA)
2. **Configure Settings**: Adjust LSTM and sentiment analysis parameters in the sidebar
3. **Generate Predictions**: Click "Predict" to run the analysis
4. **Review Results**: Examine predictions, charts, and sentiment analysis
5. **Export Data**: Download predictions and charts for further analysis

### Configuration Options

#### LSTM Settings
- **Sequence Length**: Number of historical days to analyze (20-60)
- **Training Epochs**: Model training iterations (20-100)
- **Batch Size**: Training batch size (16-64)

#### Sentiment Analysis Settings
- **Enable/Disable**: Toggle sentiment analysis integration
- **Sentiment Weight**: Control sentiment impact strength (0.1-1.0)
- **Max Adjustment**: Maximum percentage adjustment from sentiment (1-5%)

## Example Use Cases

### 📈 Bullish Sentiment Example
```
Stock: AAPL
Positive earnings news → Sentiment score: +0.6
High confidence → Adjustment: +2.1%
LSTM prediction: $150.00 → Final: $153.15
Recommendation: BUY (High Confidence)
```

### 📉 Bearish Sentiment Example  
```
Stock: TSLA
Regulatory concerns → Sentiment score: -0.4
Medium confidence → Adjustment: -1.2%
LSTM prediction: $150.00 → Final: $148.20
Recommendation: SELL (Medium Confidence)
```

### ➡️ Neutral Sentiment Example
```
Stock: MSFT
Mixed market signals → Sentiment score: 0.1
Low impact → Adjustment: +0.1%
LSTM prediction: $150.00 → Final: $150.15
Recommendation: HOLD (Low Confidence)
```

## Technical Architecture

### Sentiment Pipeline
1. **News Fetching**: Attempts to fetch financial news (with fallback)
2. **Text Processing**: Cleans and preprocesses news content
3. **Sentiment Analysis**: Applies TextBlob + VADER analysis
4. **Score Aggregation**: Weighted average by confidence
5. **Prediction Adjustment**: Applies bounded sentiment modification

### LSTM Integration
```python
# Sentiment adjustment formula
sentiment_adjustment = (
    sentiment_score * 
    sentiment_strength * 
    confidence * 
    max_adjustment * 
    user_weight
)

adjusted_prediction = base_prediction * (1 + sentiment_adjustment)
```

## Dependencies

### Core Requirements
| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.28.0 | Web dashboard framework |
| `tensorflow` | ≥2.13.0 | LSTM neural networks |
| `yfinance` | ≥0.2.18 | Stock data API |
| `plotly` | ≥5.15.0 | Interactive charts |
| `pandas` | ≥2.0.0 | Data manipulation |
| `numpy` | ≥1.24.0 | Numerical computing |

### Sentiment Analysis
| Package | Version | Purpose |
|---------|---------|---------|
| `textblob` | ≥0.18.0 | Natural language processing |
| `vaderSentiment` | ≥3.3.2 | Social media sentiment analysis |
| `feedparser` | ≥6.0.10 | RSS/news feed parsing |

### Machine Learning
| Package | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | ≥1.3.0 | Data preprocessing and metrics |
| `statsmodels` | ≥0.14.0 | Statistical modeling |
| `matplotlib` | ≥3.6.0 | Static plotting |
| `seaborn` | ≥0.12.0 | Statistical visualization |

## Performance Metrics

- **Base LSTM Accuracy**: 35-75% directional prediction
- **Sentiment Enhancement**: Up to 5% accuracy improvement
- **Processing Speed**: ~30 seconds for full analysis
- **Memory Usage**: ~2GB for model training

## Troubleshooting

### Common Issues

#### 1. TensorFlow Installation Issues
```bash
# For GPU support (optional)
pip install tensorflow[and-cuda]

# For CPU only (default)
pip install tensorflow
```

#### 2. Streamlit Port Already in Use
```bash
# Use a different port
streamlit run streamlit_stock_predictor.py --server.port 8502
```

#### 3. Memory Errors During Training
- Reduce sequence length in settings (try 20-30 instead of 60)
- Decrease batch size (try 16 instead of 32)
- Close other applications to free memory

#### 4. No News Data Available
- The app includes fallback mechanisms when news APIs are unavailable
- Sentiment analysis will use price trend analysis as backup
- Check internet connection for news feed access

#### 5. Stock Symbol Not Found
- Verify the stock symbol is correct (e.g., AAPL for Apple)
- Use Yahoo Finance format for international stocks (e.g., NESN.SW for Nestlé)
- Check if the market is active and trading

### Performance Tips
- **For faster predictions**: Reduce training epochs to 20-50
- **For better accuracy**: Increase sequence length to 60 days
- **For memory optimization**: Use smaller batch sizes (16-32)

## Supported Stock Exchanges

| Exchange | Currency | Example Symbols | Notes |
|----------|----------|-----------------|-------|
| NYSE/NASDAQ | USD ($) | AAPL, GOOGL, MSFT | Primary US markets |
| London Stock Exchange | GBP (£) | LLOY.L, BP.L | Add .L suffix |
| Tokyo Stock Exchange | JPY (¥) | 7203.T, 9984.T | Add .T suffix |
| Euronext | EUR (€) | ASML.AS, MC.PA | Various suffixes |
| TSX (Canada) | CAD ($) | SHOP.TO, RY.TO | Add .TO suffix |
| ASX (Australia) | AUD ($) | CBA.AX, BHP.AX | Add .AX suffix |

*Currency symbols are automatically detected based on the exchange.*

## Contributing

We welcome contributions! Here's how you can help:

### 🐛 Bug Reports
- Use the [GitHub Issues](https://github.com/ajaygm18/xrs/issues) page
- Include system info, error messages, and steps to reproduce
- Check existing issues before creating new ones

### 💡 Feature Requests
- Describe the problem you're trying to solve
- Provide examples of how the feature would work
- Consider contributing the implementation

### 🔧 Development Workflow
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**:
   - Add sentiment analysis improvements
   - Enhance prediction models
   - Improve UI/UX
4. **Test thoroughly**:
   - Test with various stocks and market conditions
   - Verify sentiment analysis accuracy
   - Check performance with different settings
5. **Submit a pull request**:
   - Include a clear description of changes
   - Reference any related issues
   - Ensure all tests pass

### 📋 Code Guidelines
- Follow PEP 8 style guidelines
- Add docstrings for new functions
- Include type hints where appropriate
- Test with multiple stock symbols

## Roadmap

### 🚧 In Development
- [ ] Real-time news API integration
- [ ] Additional technical indicators (MACD, Bollinger Bands)
- [ ] Portfolio analysis and optimization
- [ ] Mobile-responsive design improvements

### 🎯 Future Features
- [ ] Multiple timeframe analysis (1min, 5min, 1hour)
- [ ] Backtesting framework
- [ ] Alert system for price targets
- [ ] Integration with additional news sources
- [ ] Advanced sentiment models (BERT, FinBERT)
- [ ] Multi-language news support

## License

MIT License - See [LICENSE](LICENSE) file for details.

### What this means:
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ❌ No warranty provided
- ❌ No liability accepted

## Disclaimer

⚠️ **Important Notice**

This tool is for **educational and research purposes only**. Stock market predictions are inherently uncertain, and sentiment analysis adds additional complexity. 

**Key Points:**
- Past performance does not guarantee future results
- All predictions are probabilistic, not guaranteed
- Market conditions can change rapidly
- External factors may not be captured in the analysis

**Always:**
- Conduct your own research
- Consult financial advisors before making investment decisions
- Never invest more than you can afford to lose
- Consider this tool as one of many analysis methods

---

*Made with ❤️ for the open source community*