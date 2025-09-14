# Stock Price Predictor with Sentiment Analysis

A modern stock prediction dashboard using LSTM neural networks enhanced with financial news sentiment analysis for improved prediction accuracy.

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

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running the Dashboard
```bash
streamlit run streamlit_stock_predictor.py
```

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

### Bullish Sentiment
- Positive earnings news → Sentiment score: +0.6
- High confidence → Adjustment: +2.1%
- LSTM prediction: $150 → Final: $153.15

### Bearish Sentiment  
- Regulatory concerns → Sentiment score: -0.4
- Medium confidence → Adjustment: -1.2%
- LSTM prediction: $150 → Final: $148.20

### Neutral Sentiment
- Mixed market signals → Sentiment score: 0.1
- Low impact → Adjustment: +0.1%
- LSTM prediction: $150 → Final: $150.15

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
- `streamlit>=1.28.0` - Web dashboard framework
- `tensorflow>=2.13.0` - LSTM neural networks
- `yfinance>=0.2.18` - Stock data API
- `plotly>=5.15.0` - Interactive charts

### Sentiment Analysis
- `textblob>=0.18.0` - Natural language processing
- `vaderSentiment>=3.3.2` - Social media sentiment analysis
- `feedparser>=6.0.10` - RSS/news feed parsing

## Performance Metrics

- **Base LSTM Accuracy**: 35-75% directional prediction
- **Sentiment Enhancement**: Up to 5% accuracy improvement
- **Processing Speed**: ~30 seconds for full analysis
- **Memory Usage**: ~2GB for model training

## Supported Stock Exchanges

- NYSE, NASDAQ (USD $)
- London Stock Exchange (GBP £)
- Tokyo Stock Exchange (JPY ¥)
- Euronext (EUR €)
- And many more with automatic currency detection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add sentiment analysis improvements
4. Test with various stocks and market conditions
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Disclaimer

This tool is for educational and research purposes only. Stock market predictions are inherently uncertain, and sentiment analysis adds additional complexity. Always conduct your own research and consult financial advisors before making investment decisions.