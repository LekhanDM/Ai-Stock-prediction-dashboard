# 📈 AI Stock Market Dashboard

A professional **AI-powered stock market analysis dashboard** built with Python and Streamlit. The application combines live Yahoo Finance market data, technical indicators, machine learning, financial news sentiment, company fundamentals, interactive charts, and voice commands in one dashboard.

## 🚀 Key Features

### 📡 Live Stock Market Data
- Fetches historical stock prices using `yfinance`
- Supports configurable periods:
  - 1 Month
  - 3 Months
  - 6 Months
  - 1 Year
  - 2 Years
  - 5 Years
- Includes retry and fallback handling for Yahoo Finance rate limits
- Displays company information and financial metrics when available

### 📊 Advanced Technical Analysis
The dashboard calculates and visualizes multiple indicators:

- SMA 20, 50, 200
- EMA 12 and 26
- MACD
- MACD Signal
- MACD Histogram
- RSI
- Bollinger Bands
- ATR
- Stochastic Oscillator
- Volume SMA
- Volume Ratio
- Price Change
- Price Change %
- ADX-style trend strength
- Rate of Change (ROC)
- Volume Price Trend (VPT)

Interactive charts are created using **Plotly**.

### 🤖 Random Forest Machine Learning
The project uses a `RandomForestRegressor` to predict the next trading day's closing price.

The model includes engineered features such as:

- Historical returns
- Lagged prices
- Lagged volume
- Rolling averages
- Rolling standard deviations
- Moving-average relationships
- Volatility
- Bollinger Band features
- RSI and MACD features
- ATR and ADX-style features
- Stochastic indicators
- Seasonality features
- Volume-price features

The model uses up to **5 years of historical data** when available and performs a **time-aware 80/20 train-test split**.

### 🔮 7-Day Price Forecast
After predicting the next trading day, the application generates an iterative 7-day forecast and displays it using an interactive Plotly chart.

### 🧠 AI-Powered Market Analysis
The dashboard generates market insights from:

- Price movement
- RSI
- Moving averages
- Bollinger Bands
- MACD
- Trading volume
- Market capitalization

The analysis highlights possible bullish, bearish, overbought, oversold, or consolidation conditions based on the calculated indicators.

### 📰 News Sentiment Analysis
The application can fetch recent financial news through **NewsAPI**.

Each headline is classified using a lightweight financial sentiment word list as:

- 🟢 Positive
- 🔴 Negative
- 🟡 Neutral

The dashboard also provides an overall sentiment summary and links to the original articles.

### 🎙️ Voice Commands
The dashboard includes browser-based voice control using the Web Speech API.

Example commands:

```text
Show Apple
Show Tesla
Show NVIDIA
Set 1 year
Set 3 months
Refresh
```

Supported browsers include **Chrome and Edge**.

### 🏢 Company & Financial Information
The dashboard displays available Yahoo Finance information such as:

- Company name
- Sector
- Industry
- Country
- Website
- Number of employees
- P/E ratio
- Forward P/E
- PEG ratio
- Price-to-book ratio
- Dividend yield
- Beta
- 52-week high
- 52-week low

### 📥 Data Export
The dashboard allows users to download recent stock data as a CSV file.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web dashboard |
| yfinance | Stock market data |
| Pandas | Data processing |
| NumPy | Numerical calculations |
| Plotly | Interactive charts |
| Scikit-learn | Machine learning |
| Random Forest | Stock price prediction |
| Requests | NewsAPI requests |
| Web Speech API | Voice commands |

---

## 📁 Project Structure

```text
AI-Stock-Market-Dashboard/
│
├── stock_dashboard.py
├── requirements.txt
└── README.md
```

If the project is divided into multiple modules later, they can be added to this structure as separate files.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd AI-Stock-Market-Dashboard
```

Or download the project ZIP and open the project folder.

### 2. Create a virtual environment

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit yfinance pandas numpy plotly scikit-learn requests
```

Or, if `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

---

## 🔑 NewsAPI Setup

News sentiment requires a NewsAPI key.

Create an account at:

https://newsapi.org/

Then start the application and enter your key in the **NewsAPI Key** field in the sidebar.

The application does not require the NewsAPI feature to run. If no key is provided, the news section remains disabled.

---

## ▶️ Run the Dashboard

Run:

```bash
streamlit run stock_dashboard.py
```

Then open the Streamlit URL shown in the terminal, normally:

```text
http://localhost:8501
```

---

## 🖥️ How to Use

1. Start the Streamlit application.
2. Enter a stock symbol such as:
   ```text
   AAPL
   ```
3. Select the required historical period.
4. Enable or disable:
   - Technical Analysis
   - Performance Analysis
   - ML Prediction
   - AI Market Analysis
   - News Sentiment
5. View the current price and market metrics.
6. Explore the advanced technical charts.
7. Train the Random Forest model and view the next-day prediction.
8. Check the 7-day forecast.
9. Review feature importance.
10. Read AI-generated market insights.
11. Enter a NewsAPI key to enable news sentiment.
12. Use voice commands if supported by the browser.
13. Download recent stock data as CSV.

---

## 🔄 System Workflow

```text
              ┌────────────────────┐
              │   User / Stock     │
              │      Symbol        │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │    Yahoo Finance   │
              │    Live / History  │
              └─────────┬──────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
      ┌───────────────┐   ┌─────────────────┐
      │ Price & Volume│   │ Company Details │
      └───────┬───────┘   └─────────────────┘
              │
              ▼
      ┌───────────────────┐
      │ Technical         │
      │ Indicators        │
      └─────────┬─────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
┌───────────────┐  ┌─────────────────┐
│ Interactive   │  │ Feature         │
│ Charts        │  │ Engineering     │
└───────────────┘  └────────┬────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ Random Forest ML  │
                  │     Model        │
                  └────────┬─────────┘
                           │
                    ┌──────┴───────┐
                    ▼              ▼
             Next-Day Price    7-Day Forecast

       NewsAPI ──► Sentiment Analysis
                          │
                          ▼
                 Market Sentiment

       Voice ──► Speech Recognition
                          │
                          ▼
                 Dashboard Control
```

---

## 🧮 Machine Learning Methodology

### Feature Engineering

The model creates a large set of features from historical market data, including:

```text
Returns
Lag Features
Rolling Mean
Rolling Standard Deviation
Moving Average Distance
Volatility
Bollinger Bands
RSI
MACD
ATR
ADX-style Trend Features
ROC
VPT
Day of Week
Month
Quarter
```

### Training Process

```text
Historical Data
      ↓
Technical Indicators
      ↓
Feature Engineering
      ↓
Remove Missing Values
      ↓
Time-Based 80/20 Split
      ↓
StandardScaler
      ↓
Random Forest Regressor
      ↓
Next-Day Prediction
      ↓
7-Day Iterative Forecast
```

The dashboard reports:

- Training score
- Test score
- Number of training rows
- Feature importance

---

## 📊 Dashboard Metrics

The main dashboard displays:

### Current Price
Latest available closing price and change from the previous trading session.

### Volume
Latest trading volume compared with the 20-day average.

### RSI
14-period Relative Strength Index with basic overbought/oversold interpretation.

### SMA 20
Percentage distance between the current price and 20-day simple moving average.

### Market Cap
Company market capitalization when Yahoo Finance provides it.

---

## 📰 News Sentiment

NewsAPI returns recent articles related to the selected company.

The application calculates:

```text
Positive Articles
Negative Articles
Neutral Articles
Overall Sentiment
```

The sentiment classifier uses a financial keyword-based approach rather than a large NLP model, keeping the application lightweight.

---

## 🎙️ Voice Command Architecture

The voice feature runs inside the browser using the Web Speech API.

```text
Microphone
    ↓
Speech Recognition
    ↓
Command Parser
    ↓
Stock / Period / Refresh Detection
    ↓
URL Parameters
    ↓
Streamlit Dashboard Update
```

Examples:

```text
"Show Apple"       → AAPL
"Show Tesla"       → TSLA
"Show NVIDIA"      → NVDA
"Set 1 year"       → 1y
"Set 5 years"      → 5y
"Refresh"          → Refresh dashboard
```

---

## ⚠️ Error Handling

The application includes handling for common data-fetching problems:

- Yahoo Finance request failures
- Empty stock datasets
- Yahoo Finance rate limiting
- Invalid stock symbols
- Missing company information
- NewsAPI authentication errors
- NewsAPI connection errors
- Missing news articles
- Insufficient historical data for machine learning

---

## 🚧 Limitations

- Yahoo Finance data availability depends on the external service.
- Yahoo Finance may temporarily rate-limit requests.
- Company information may be unavailable during rate limiting.
- Machine learning predictions are estimates and are not guaranteed to be accurate.
- The 7-day forecast is an iterative model-based projection.
- News sentiment uses a simplified financial keyword approach.
- NewsAPI requires an API key for news analysis.
- Voice commands depend on browser support and microphone permissions.
- Market analysis is based on predefined technical rules.
- The project does not execute real stock trades.

---

## 🔮 Future Enhancements

Possible improvements include:

- LSTM-based time-series prediction
- XGBoost comparison
- Ensemble ML models
- More advanced NLP sentiment models
- Real-time stock alerts
- Portfolio tracking
- Watchlists
- Price target alerts
- Technical indicator alerts
- Candlestick pattern detection
- Backtesting
- Model performance comparison
- Authentication and user accounts
- Database integration
- Secure Streamlit Secrets for API keys
- Deployment to Streamlit Cloud or another cloud platform

---

## 🔐 Security

Never commit API keys to GitHub.

Instead of writing secrets directly into source code, use:

```text
Streamlit Secrets
Environment Variables
```

Also add sensitive files to `.gitignore`:

```text
.env
.streamlit/secrets.toml
venv/
__pycache__/
```

---

## 📌 Disclaimer

This project is intended for **educational and demonstration purposes only**.

The machine learning predictions, technical indicators, sentiment analysis, and market insights are computational outputs and should not be treated as professional financial advice or guaranteed forecasts.

---

## 👨‍💻 Project Objective

The objective of this project is to demonstrate the integration of:

**Financial Data + Technical Analysis + Feature Engineering + Machine Learning + News Sentiment + AI-Based Market Analysis + Voice Interaction**

into a single interactive stock market dashboard.

It provides a practical example of how Python, Streamlit, data science, and machine learning can be combined to build a modern financial analytics application.

---

## ⭐ GitHub

If this project helped you learn or explore stock market analytics, consider giving the repository a ⭐.

**Built with Python, Streamlit, Machine Learning, and Data Science.**
