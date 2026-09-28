import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import requests
import json
import re
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="AI Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StockAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def fetch_stock_data(self, symbol, period="1y"):
        """Fetch stock data with Mac/rate-limit resilient error handling"""
        import time

        # --- price history with retry + back-off ---
        data  = None
        stock = None
        for attempt in range(3):
            try:
                stock = yf.Ticker(symbol)
                data  = stock.history(period=period, timeout=30)
                if data is not None and not data.empty:
                    break
            except Exception:
                pass
            time.sleep(2 + attempt * 2)

        if data is None or data.empty:
            # Last-resort: yf.download uses a different endpoint
            try:
                data = yf.download(symbol, period=period, progress=False, timeout=30)
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {str(e)}")
                return None, None

        if data is None or data.empty:
            st.error(
                f"Could not fetch data for {symbol}. "
                "Yahoo Finance may be rate-limiting your IP — wait 1–2 min and retry, "
                "or try a VPN / different network."
            )
            return None, None

        # --- company info (non-fatal: 429 here is common on Mac, just skip) ---
        info = {}
        try:
            time.sleep(1)
            if stock is None:
                stock = yf.Ticker(symbol)
            info = stock.info or {}
            if len(info) < 5:   # yfinance returns near-empty dict on rate-limit
                info = {}
        except Exception:
            info = {}

        return data, info
    
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators using pure pandas/numpy"""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price-based indicators
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Volatility (Average True Range approximation)
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
        df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['True_Range'].rolling(window=14).mean()
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        return df
    
    def prepare_ml_features(self, data):
        """Prepare enriched features for machine learning"""
        df = data.copy()

        # ── Returns & momentum ────────────────────────────────────────
        df['Returns']      = df['Close'].pct_change()
        df['Returns_3d']   = df['Close'].pct_change(3)
        df['Returns_5d']   = df['Close'].pct_change(5)
        df['Returns_10d']  = df['Close'].pct_change(10)
        df['Returns_20d']  = df['Close'].pct_change(20)
        df['Returns_60d']  = df['Close'].pct_change(60)

        # ── Extended lag features ─────────────────────────────────────
        for lag in [1, 2, 3, 5, 10, 20, 30]:
            df[f'Close_lag_{lag}']   = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}']  = df['Volume'].shift(lag)
            df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)

        # ── Rolling statistics (more windows) ─────────────────────────
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'Close_mean_{window}']   = df['Close'].rolling(window).mean()
            df[f'Close_std_{window}']    = df['Close'].rolling(window).std()
            df[f'Volume_mean_{window}']  = df['Volume'].rolling(window).mean()
            df[f'High_mean_{window}']    = df['High'].rolling(window).mean()
            df[f'Low_mean_{window}']     = df['Low'].rolling(window).mean()

        # ── Price vs moving averages ───────────────────────────────────
        df['Price_vs_SMA20']  = (df['Close'] - df['SMA_20'])  / df['SMA_20']  * 100
        df['Price_vs_SMA50']  = (df['Close'] - df['SMA_50'])  / df['SMA_50']  * 100
        df['Price_vs_SMA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200'] * 100

        # ── Volatility ────────────────────────────────────────────────
        df['Price_volatility_5d']  = df['Returns'].rolling(5).std()
        df['Price_volatility_10d'] = df['Returns'].rolling(10).std()
        df['Price_volatility_20d'] = df['Returns'].rolling(20).std()
        df['Price_volatility_60d'] = df['Returns'].rolling(60).std()

        # ── Bollinger Band width & %B ─────────────────────────────────
        bb_width = df['BB_upper'] - df['BB_lower']
        df['BB_width']   = bb_width / df['BB_middle']
        df['BB_pct']     = (df['Close'] - df['BB_lower']) / (bb_width + 1e-9)

        # ── Trend strength (ADX proxy) ────────────────────────────────
        df['Up_move']   = df['High'].diff()
        df['Down_move'] = -df['Low'].diff()
        df['DM_plus']   = np.where((df['Up_move'] > df['Down_move']) & (df['Up_move'] > 0), df['Up_move'], 0)
        df['DM_minus']  = np.where((df['Down_move'] > df['Up_move']) & (df['Down_move'] > 0), df['Down_move'], 0)
        df['DI_plus']   = df['DM_plus'].rolling(14).mean()  / (df['ATR'] + 1e-9)
        df['DI_minus']  = df['DM_minus'].rolling(14).mean() / (df['ATR'] + 1e-9)
        dx              = np.abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus'] + 1e-9) * 100
        df['ADX']       = dx.rolling(14).mean()

        # ── Rate of change ────────────────────────────────────────────
        for n in [5, 10, 20]:
            df[f'ROC_{n}'] = df['Close'].pct_change(n) * 100

        # ── Volume-price trend ────────────────────────────────────────
        df['VPT'] = (df['Volume'] * df['Returns']).cumsum()
        df['VPT_lag5'] = df['VPT'].shift(5)

        # ── Seasonality (day-of-week, month) ─────────────────────────
        df['DayOfWeek'] = df.index.dayofweek
        df['Month']     = df.index.month
        df['Quarter']   = df.index.quarter

        # ── MACD histogram slope ──────────────────────────────────────
        df['MACD_hist_slope'] = df['MACD_histogram'].diff()

        # ── RSI slope ────────────────────────────────────────────────
        df['RSI_slope'] = df['RSI'].diff(3)

        # ── Stochastic slope ─────────────────────────────────────────
        df['Stoch_slope'] = df['Stoch_K'].diff(3)

        return df
    
    def train_prediction_model(self, data, symbol=None):
        """Train ML model using max available historical data for better accuracy"""
        import time

        # ── Always fetch 5 years for training regardless of chart period ──
        training_data = data.copy()
        if symbol and len(data) < 500:
            try:
                time.sleep(0.5)
                import yfinance as _yf
                hist = _yf.Ticker(symbol).history(period="5y", timeout=20)
                if hist is not None and len(hist) > len(data):
                    _tmp = StockAnalyzer()
                    training_data = _tmp.calculate_technical_indicators(hist)
            except Exception:
                pass  # fall back to chart data silently

        df = self.prepare_ml_features(training_data)
        df = df.dropna()

        if len(df) < 100:
            return None

        # ── Feature selection ─────────────────────────────────────────
        exclude_raw = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends',
                       'Stock Splits', 'Returns', 'Returns_3d', 'Returns_5d',
                       'Returns_10d', 'Returns_20d', 'Returns_60d',
                       'Up_move', 'Down_move', 'DM_plus', 'DM_minus',
                       'High_Low', 'High_Close', 'Low_Close', 'True_Range',
                       'BB_middle', 'EMA_12', 'EMA_26']
        keep_patterns = ['lag', 'mean', 'std', 'volatility', 'vs_SMA', 'ROC',
                         'slope', 'width', 'pct', 'VPT']
        keep_exact    = ['RSI', 'MACD', 'ATR', 'ADX', 'Stoch_K', 'Stoch_D',
                         'MACD_histogram', 'Volume_ratio', 'High_Low_Pct',
                         'Price_Change_Pct', 'BB_pct', 'BB_width',
                         'DI_plus', 'DI_minus', 'DayOfWeek', 'Month', 'Quarter',
                         'MACD_signal', 'RSI_slope', 'Stoch_slope']

        feature_cols = [
            col for col in df.columns
            if col not in exclude_raw
            and (any(p in col for p in keep_patterns) or col in keep_exact)
        ]

        if len(feature_cols) < 5:
            return None

        X = df[feature_cols].ffill().bfill()
        y = df['Close'].shift(-1)

        X = X[:-1]
        y = y[:-1]
        mask = ~(X.isna().any(axis=1) | y.isna())
        X, y = X[mask], y[mask]

        if len(X) < 50:
            return None

        # ── Time-aware split (last 20% of time = test, not random) ───
        split_idx    = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        train_score = self.model.score(X_train_scaled, y_train)
        test_score  = self.model.score(X_test_scaled,  y_test)

        return {
            'train_score':       train_score,
            'test_score':        test_score,
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_)),
            'last_features':     X.iloc[-1:],
            'feature_cols':      feature_cols,
            'training_rows':     len(X),
            'data':              df,          # kept for 7-day forecast
            'scaler':            self.scaler,
        }
    
    def predict_next_price(self, model_info):
        """Predict next trading day price"""
        if model_info is None:
            return None
        last_features_scaled = self.scaler.transform(model_info['last_features'])
        return self.model.predict(last_features_scaled)[0]

    def predict_7day_forecast(self, model_info, current_price):
        """Roll forward 7 trading days of predictions iteratively"""
        if model_info is None:
            return []
        predictions = [current_price]
        features = model_info['last_features'].copy()
        scaler  = model_info['scaler']
        for _ in range(7):
            try:
                scaled = scaler.transform(features)
                pred   = self.model.predict(scaled)[0]
                predictions.append(pred)
                # Shift lag columns forward by one step (best-effort)
                for col in features.columns:
                    if 'lag_1' in col and 'Close' in col:
                        features[col] = pred
                    elif 'lag_2' in col and 'Close' in col:
                        features[col] = features.get(col.replace('lag_2', 'lag_1'), features[col]).values[0]
                    elif 'Returns_lag_1' in col:
                        features[col] = (pred - predictions[-2]) / (predictions[-2] + 1e-9)
            except Exception:
                break
        return predictions[1:]  # exclude current price
    
    def generate_market_analysis(self, data, info, symbol):
        """Generate AI-powered market analysis"""
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Price movement
        price_change = latest['Close'] - prev['Close']
        price_change_pct = (price_change / prev['Close']) * 100
        
        # Technical analysis
        rsi = latest.get('RSI', 50)
        sma_20 = latest.get('SMA_20', latest['Close'])
        sma_50 = latest.get('SMA_50', latest['Close'])
        bb_upper = latest.get('BB_upper', latest['Close'])
        bb_lower = latest.get('BB_lower', latest['Close'])
        
        # Volume analysis
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 1
        
        # MACD analysis
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_signal', 0)
        
        # Generate analysis
        analysis = []
        
        # Price trend
        if price_change_pct > 3:
            analysis.append(f"🚀 {symbol} shows exceptional bullish momentum with a {price_change_pct:.2f}% surge")
        elif price_change_pct > 1:
            analysis.append(f"🟢 {symbol} demonstrates strong upward movement (+{price_change_pct:.2f}%)")
        elif price_change_pct > 0:
            analysis.append(f"🟡 {symbol} shows modest gains (+{price_change_pct:.2f}%)")
        elif price_change_pct > -1:
            analysis.append(f"🟡 {symbol} experiences slight decline ({price_change_pct:.2f}%)")
        elif price_change_pct > -3:
            analysis.append(f"🔴 {symbol} shows moderate bearish pressure ({price_change_pct:.2f}%)")
        else:
            analysis.append(f"🔻 {symbol} faces significant selling pressure ({price_change_pct:.2f}%)")
        
        # RSI analysis
        if rsi > 80:
            analysis.append(f"🚨 RSI at {rsi:.1f} indicates severely overbought conditions - potential reversal ahead")
        elif rsi > 70:
            analysis.append(f"⚠️ RSI at {rsi:.1f} shows overbought territory - exercise caution")
        elif rsi < 20:
            analysis.append(f"🛒 RSI at {rsi:.1f} signals severely oversold - strong buying opportunity")
        elif rsi < 30:
            analysis.append(f"💡 RSI at {rsi:.1f} suggests oversold conditions - potential buying opportunity")
        elif 40 <= rsi <= 60:
            analysis.append(f"⚖️ RSI at {rsi:.1f} indicates balanced momentum")
        else:
            analysis.append(f"📊 RSI at {rsi:.1f} shows {('bullish' if rsi > 50 else 'bearish')} bias")
        
        # Moving average analysis
        if latest['Close'] > sma_20 > sma_50:
            analysis.append("📈 Strong bullish alignment - price above both 20 and 50-day MAs")
        elif latest['Close'] < sma_20 < sma_50:
            analysis.append("📉 Bearish trend confirmed - price below key moving averages")
        elif latest['Close'] > sma_20 and sma_20 < sma_50:
            analysis.append("🔄 Mixed signals - short-term bullish but longer-term bearish")
        else:
            analysis.append("➡️ Consolidation phase - awaiting directional breakout")
        
        # Bollinger Bands analysis
        if latest['Close'] > bb_upper:
            analysis.append("📊 Price trading above upper Bollinger Band - potential overbought")
        elif latest['Close'] < bb_lower:
            analysis.append("📊 Price near lower Bollinger Band - potential oversold bounce")
        
        # MACD analysis
        if macd > macd_signal and macd > 0:
            analysis.append("⚡ MACD shows strong bullish momentum")
        elif macd < macd_signal and macd < 0:
            analysis.append("⚡ MACD indicates bearish momentum")
        elif macd > macd_signal:
            analysis.append("⚡ MACD bullish crossover - momentum improving")
        else:
            analysis.append("⚡ MACD bearish crossover - momentum weakening")
        
        # Volume analysis
        if volume_ratio > 2:
            analysis.append("🔥 Exceptional volume surge confirms strong conviction")
        elif volume_ratio > 1.5:
            analysis.append("📊 High volume validates price movement")
        elif volume_ratio < 0.5:
            analysis.append("📊 Below-average volume suggests weak conviction")
        else:
            analysis.append("📊 Normal volume levels")
        
        # Market cap context
        market_cap = info.get('marketCap', 0)
        if market_cap:
            if market_cap > 200e9:  # > 200B
                analysis.append("🏢 Large-cap stability with lower volatility expected")
            elif market_cap > 10e9:  # > 10B
                analysis.append("🏢 Mid-cap stock with balanced growth-stability profile")
            else:
                analysis.append("🏢 Small-cap stock with higher growth potential and volatility")
        
        return analysis

    def fetch_news_sentiment(self, symbol, company_name, api_key):
        """Fetch recent news and compute sentiment using NewsAPI"""
        try:
            query = company_name if company_name and company_name != 'N/A' else symbol
            url = (
                f"https://newsapi.org/v2/everything"
                f"?q={requests.utils.quote(query)}"
                f"&language=en"
                f"&sortBy=publishedAt"
                f"&pageSize=10"
                f"&apiKey={api_key}"
            )
            resp = requests.get(url, timeout=10)
            if resp.status_code == 401:
                return None, "invalid_key"
            if resp.status_code != 200:
                return None, f"api_error_{resp.status_code}"

            articles = resp.json().get("articles", [])
            if not articles:
                return [], "no_articles"

            # Simple lexicon-based sentiment (no extra deps)
            POSITIVE = {"surge","soars","gains","bullish","beats","record","growth",
                        "profit","rises","rally","upgraded","buy","strong","positive",
                        "up","higher","boost","breakthrough","win","expands","milestone"}
            NEGATIVE = {"fall","drops","plunges","bearish","misses","loss","decline",
                        "lawsuit","downgrade","sell","weak","negative","down","lower",
                        "cut","risk","concern","warning","slump","crash","layoff","probe"}

            results = []
            for a in articles[:8]:
                title = (a.get("title") or "")
                desc  = (a.get("description") or "")
                text  = (title + " " + desc).lower()
                words = set(re.findall(r'[a-z]+', text))
                pos   = len(words & POSITIVE)
                neg   = len(words & NEGATIVE)
                if pos > neg:
                    sentiment, emoji = "Positive", "🟢"
                elif neg > pos:
                    sentiment, emoji = "Negative", "🔴"
                else:
                    sentiment, emoji = "Neutral",  "🟡"
                results.append({
                    "title":     title,
                    "url":       a.get("url", "#"),
                    "source":    (a.get("source") or {}).get("name", "Unknown"),
                    "published": (a.get("publishedAt") or "")[:10],
                    "sentiment": sentiment,
                    "emoji":     emoji,
                    "pos_score": pos,
                    "neg_score": neg,
                })
            return results, "ok"
        except requests.exceptions.ConnectionError:
            return None, "connection_error"
        except Exception as e:
            return None, f"error: {e}"

def create_advanced_chart(data, symbol):
    """Create advanced candlestick chart with technical indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price Action & Moving Averages', 'Volume', 'MACD', 'RSI & Stochastic'),
        row_heights=[0.5, 0.15, 0.2, 0.15]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Moving averages
    colors = ['#ff9500', '#007aff', '#5856d6']
    mas = [('SMA_20', 'SMA 20'), ('SMA_50', 'SMA 50'), ('SMA_200', 'SMA 200')]
    
    for i, (ma_col, ma_name) in enumerate(mas):
        if ma_col in data.columns and not data[ma_col].isna().all():
            fig.add_trace(
                go.Scatter(x=data.index, y=data[ma_col], 
                          line=dict(color=colors[i], width=1.5), name=ma_name),
                row=1, col=1
            )
    
    # Bollinger Bands
    if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_upper'], 
                      line=dict(color='rgba(128,128,128,0.5)', width=1), name='BB Upper',
                      showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_lower'], 
                      line=dict(color='rgba(128,128,128,0.5)', width=1), name='BB Lower',
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                      showlegend=False),
            row=1, col=1
        )
    
    # Volume
    volume_colors = ['#00ff88' if data['Close'].iloc[i] >= data['Open'].iloc[i] else '#ff4444' 
                    for i in range(len(data))]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], 
              marker_color=volume_colors, name='Volume', opacity=0.7),
        row=2, col=1
    )
    
    if 'Volume_SMA' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Volume_SMA'], 
                      line=dict(color='white', width=1), name='Vol SMA'),
            row=2, col=1
        )
    
    # MACD
    if all(col in data.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], 
                      line=dict(color='#007aff', width=2), name='MACD'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_signal'], 
                      line=dict(color='#ff9500', width=2), name='Signal'),
            row=3, col=1
        )
        
        histogram_colors = ['#00ff88' if val >= 0 else '#ff4444' for val in data['MACD_histogram']]
        fig.add_trace(
            go.Bar(x=data.index, y=data['MACD_histogram'], 
                  marker_color=histogram_colors, name='Histogram', opacity=0.6),
            row=3, col=1
        )
    
    # RSI and Stochastic
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], 
                      line=dict(color='#af52de', width=2), name='RSI'),
            row=4, col=1
        )
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=4, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=4, col=1)
    
    if 'Stoch_K' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Stoch_K'], 
                      line=dict(color='#ffcc00', width=1.5), name='Stoch %K'),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Stoch_D'], 
                      line=dict(color='#ff6600', width=1.5), name='Stoch %D'),
            row=4, col=1
        )
    
    fig.update_layout(
        title=f'{symbol} - Complete Technical Analysis Dashboard',
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        template='plotly_dark',
        font=dict(size=10)
    )
    
    # Remove x-axis labels from all but bottom subplot
    for i in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=i, col=1)
    
    return fig

def create_performance_metrics(data, symbol):
    """Create performance metrics visualization"""
    # Calculate returns
    data['Daily_Returns'] = data['Close'].pct_change()
    data['Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod() - 1
    
    # Performance metrics
    total_return = data['Cumulative_Returns'].iloc[-1] * 100
    volatility = data['Daily_Returns'].std() * np.sqrt(252) * 100  # Annualized
    sharpe_ratio = (data['Daily_Returns'].mean() * 252) / (data['Daily_Returns'].std() * np.sqrt(252))
    
    max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{total_return:.1f}%")
    with col2:
        st.metric("Volatility (Ann.)", f"{volatility:.1f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
    
    # Cumulative returns chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Cumulative_Returns'] * 100,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='#00ff88', width=2)
        )
    )
    
    fig.update_layout(
        title=f'{symbol} Cumulative Returns (%)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Streamlit App
def main():
    # ── Sidebar (must come first so variables are defined) ───────────
    st.sidebar.header("📊 Dashboard Controls")
    st.sidebar.markdown("---")

    popular_stocks = {
        'Apple': 'AAPL', 'Microsoft': 'MSFT', 'Google': 'GOOGL',
        'Amazon': 'AMZN', 'Tesla': 'TSLA', 'NVIDIA': 'NVDA',
        'Meta': 'META', 'Netflix': 'NFLX', 'AMD': 'AMD', 'Intel': 'INTC'
    }
    sym_to_name = {v: k for k, v in popular_stocks.items()}
    period_options = ['1mo', '3mo', '6mo', '1y', '2y', '5y']

    # Read voice-command query params
    qp = st.query_params
    vc_symbol = qp.get("symbol", "").upper()
    vc_period  = qp.get("period", "")
    vc_refresh = qp.get("refresh", "")

    # Resolve default stock index from voice command
    if vc_symbol and vc_symbol in sym_to_name:
        default_stock_idx = list(popular_stocks.keys()).index(sym_to_name[vc_symbol])
    else:
        default_stock_idx = 0

    stock_choice = st.sidebar.selectbox(
        "🏢 Select Stock:",
        options=list(popular_stocks.keys()) + ['Custom'],
        index=default_stock_idx
    )

    if stock_choice == 'Custom':
        default_custom = vc_symbol if (vc_symbol and vc_symbol not in sym_to_name) else "AAPL"
        symbol = st.sidebar.text_input("Enter Stock Symbol:", value=default_custom, max_chars=10).upper()
    else:
        symbol = popular_stocks[stock_choice]

    # Resolve default period from voice command
    if vc_period and vc_period in period_options:
        default_period_idx = period_options.index(vc_period)
    else:
        default_period_idx = 3

    period = st.sidebar.selectbox(
        "📅 Analysis Period:",
        options=period_options,
        index=default_period_idx
    )

    # Auto-clear query params after applying so next load is clean
    if vc_symbol or vc_period or vc_refresh:
        st.query_params.clear()
    if vc_refresh:
        st.cache_data.clear()

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Analysis Options")
    show_prediction = st.sidebar.checkbox("🔮 ML Price Prediction", value=True)
    show_technical  = st.sidebar.checkbox("📈 Technical Charts", value=True)
    show_performance= st.sidebar.checkbox("📊 Performance Metrics", value=True)
    show_analysis   = st.sidebar.checkbox("🧠 AI Market Analysis", value=True)

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()

    # ── News Sentiment ──────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("📰 News Sentiment")
    news_api_key = st.sidebar.text_input(
        "NewsAPI Key:",
        type="password",
        placeholder="Enter your NewsAPI key",
        help="Get a free key at newsapi.org"
    )
    show_news = st.sidebar.checkbox("📰 News Sentiment", value=bool(news_api_key))
    if not news_api_key:
        st.sidebar.caption("🔑 [Get free key at newsapi.org](https://newsapi.org/register)")

    # ── Voice Commands ───────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎙️ Voice Commands")
    voice_enabled = st.sidebar.checkbox("Enable Voice Commands", value=False)
    if voice_enabled:
        st.sidebar.caption(
            "**Say:** *'Show Apple'* · *'Show Tesla'* · "
            "*'Set period 1 year'* · *'Refresh'*"
        )

    # ── Title ────────────────────────────────────────────────────────
    st.title("🚀 Professional AI Stock Market Dashboard")
    st.markdown("*Advanced technical analysis with machine learning predictions*")

    # ── Voice Command Component ──────────────────────────────────────
    if voice_enabled:
        import streamlit.components.v1 as components
        voice_html = """
<style>
  #vc-bar{display:flex;align-items:center;gap:12px;padding:10px 16px;
          background:linear-gradient(135deg,rgba(52,199,89,0.08),rgba(0,122,255,0.08));
          border:1px solid rgba(52,199,89,0.25);border-radius:12px;
          font-family:-apple-system,BlinkMacSystemFont,sans-serif;}
  #vc-btn{background:#ff3b30;border:none;color:#fff;padding:7px 18px;
          border-radius:8px;cursor:pointer;font-size:13px;font-weight:700;
          transition:background .2s;}
  #vc-btn.listening{background:#34c759;animation:pulse 1.2s infinite;}
  #vc-status{font-size:13px;color:#888;flex:1;}
  #vc-last{font-size:12px;color:#5ac8fa;font-style:italic;max-width:340px;
           white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
</style>
<div id="vc-bar">
  <button id="vc-btn" onclick="toggleListen()">🎙️ Start</button>
  <span id="vc-status">Click Start and speak a command</span>
  <span id="vc-last"></span>
</div>
<script>
// ── Mappings ────────────────────────────────────────────────────────
const STOCK_MAP = {
  apple:'AAPL', microsoft:'MSFT', google:'GOOGL', amazon:'AMZN',
  tesla:'TSLA', nvidia:'NVDA',  meta:'META',     netflix:'NFLX',
  amd:'AMD',    intel:'INTC'
};
const PERIOD_MAP = {
  'one month':'1mo',   '1 month':'1mo',
  'three month':'3mo', '3 month':'3mo',
  'six month':'6mo',   '6 month':'6mo',
  'one year':'1y',     '1 year':'1y',
  'two year':'2y',     '2 year':'2y',
  'five year':'5y',    '5 year':'5y'
};

let recog = null, listening = false;
const btn    = document.getElementById('vc-btn');
const status = document.getElementById('vc-status');
const last   = document.getElementById('vc-last');

function setStatus(msg, col){ status.textContent=msg; status.style.color=col||'#888'; }

// ── Core: navigate parent window with query param ────────────────────
function navigate(params){
  // Build query string on the parent page URL (strips iframe path)
  const base = window.parent.location.href.split('?')[0];
  const qs   = Object.entries(params).map(([k,v])=>k+'='+encodeURIComponent(v)).join('&');
  window.parent.location.href = base + '?' + qs;
}

// ── Command parser ───────────────────────────────────────────────────
function handleCommand(raw){
  const t = raw.toLowerCase().trim();
  last.textContent = '🗣 "' + raw + '"';

  // Stock
  for(const [name, sym] of Object.entries(STOCK_MAP)){
    if(t.includes('show '+name)||t.includes('load '+name)||
       t.includes('open '+name)||t.includes(name+' stock')||t===name){
      setStatus('Loading '+sym+' …','#34c759');
      navigate({symbol: sym});
      return;
    }
  }

  // Period
  for(const [phrase, code] of Object.entries(PERIOD_MAP)){
    if(t.includes(phrase)){
      setStatus('Setting period to '+code+' …','#34c759');
      navigate({period: code});
      return;
    }
  }

  // Refresh
  if(t.includes('refresh')||t.includes('reload')||t.includes('update')){
    setStatus('Refreshing …','#34c759');
    navigate({refresh: '1'});
    return;
  }

  setStatus('Not recognised — try: "Show Tesla" or "Set 1 year"','#ff9f0a');
}

// ── Speech recognition ───────────────────────────────────────────────
function toggleListen(){
  if(!('webkitSpeechRecognition' in window||'SpeechRecognition' in window)){
    setStatus('Speech API not supported — use Chrome or Edge','#ff3b30');
    return;
  }
  if(listening){ recog && recog.stop(); return; }

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recog = new SR();
  recog.lang = 'en-US';
  recog.interimResults = false;
  recog.maxAlternatives = 1;
  recog.continuous = false;

  recog.onstart  = ()=>{ listening=true;  btn.textContent='⏹ Stop'; btn.classList.add('listening');    setStatus('Listening …','#34c759'); };
  recog.onend    = ()=>{ listening=false; btn.textContent='🎙️ Start'; btn.classList.remove('listening'); setStatus('Done — click Start to speak again','#888'); };
  recog.onerror  = e =>{ setStatus('Mic error: '+e.error,'#ff3b30'); };
  recog.onresult = e =>{ handleCommand(e.results[0][0].transcript); };
  recog.start();
}
</script>
"""
        components.html(voice_html, height=60)
        st.caption(
            "🎙️ Say: **'Show Apple'** · **'Show Tesla'** · **'Show NVIDIA'** · "
            "**'Set 1 year'** · **'Set 3 months'** · **'Refresh'** — "
            "works in Chrome / Edge only"
        )
    
    # Initialize analyzer
    analyzer = StockAnalyzer()
    
    # Fetch and display data
    with st.spinner(f"📡 Fetching live data for {symbol}..."):
        data, info = analyzer.fetch_stock_data(symbol, period)
    
    if data is None or data.empty:
        st.error(f"❌ Could not fetch data for {symbol}. Please verify the symbol and try again.")
        st.info("💡 Try popular symbols like AAPL, MSFT, GOOGL, TSLA, etc.")
        return
    
    # Calculate technical indicators
    with st.spinner("⚙️ Calculating technical indicators..."):
        data = analyzer.calculate_technical_indicators(data)
    
    # Main dashboard header
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    latest_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = latest_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    with col1:
        st.metric(
            label="💰 Current Price",
            value=f"${latest_price:.2f}",
            delta=f"{price_change:.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_change = ((volume - avg_volume) / avg_volume) * 100 if avg_volume > 0 else 0
        st.metric(
            label="📊 Volume",
            value=f"{volume:,.0f}",
            delta=f"{volume_change:+.1f}% vs 20d avg"
        )
    
    with col3:
        if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]):
            rsi = data['RSI'].iloc[-1]
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            st.metric(
                label="⚡ RSI (14)",
                value=f"{rsi:.1f}",
                delta=rsi_status
            )
        else:
            st.metric(label="⚡ RSI (14)", value="N/A")
    
    with col4:
        if 'SMA_20' in data.columns and not pd.isna(data['SMA_20'].iloc[-1]):
            sma_20 = data['SMA_20'].iloc[-1]
            sma_distance = ((latest_price - sma_20) / sma_20) * 100
            st.metric(
                label="📈 vs SMA 20",
                value=f"{sma_distance:+.1f}%",
                delta="Above" if sma_distance > 0 else "Below"
            )
        else:
            st.metric(label="📈 vs SMA 20", value="N/A")
    
    with col5:
        market_cap = info.get('marketCap', 0)
        if market_cap:
            if market_cap > 1e12:
                cap_display = f"${market_cap/1e12:.2f}T"
            elif market_cap > 1e9:
                cap_display = f"${market_cap/1e9:.1f}B"
            else:
                cap_display = f"${market_cap/1e6:.0f}M"
            st.metric(label="🏢 Market Cap", value=cap_display)
        else:
            st.metric(label="🏢 Market Cap", value="N/A")
    
    st.markdown("---")
    
    # Advanced Chart
    if show_technical:
        st.subheader("📈 Advanced Technical Analysis")
        with st.spinner("Creating advanced charts..."):
            chart = create_advanced_chart(data, symbol)
            st.plotly_chart(chart, use_container_width=True)
    
    # Performance Metrics
    if show_performance:
        st.subheader("📊 Performance Analysis")
        create_performance_metrics(data, symbol)
    
    # ML Prediction
    if show_prediction:
        st.subheader("🔮 Machine Learning Price Prediction")

        col1, col2 = st.columns([1, 1])

        with col1:
            with st.spinner("🤖 Fetching 5Y history & training model…"):
                model_info = analyzer.train_prediction_model(data, symbol)

            if model_info:
                prediction    = analyzer.predict_next_price(model_info)
                current_price = data['Close'].iloc[-1]
                predicted_change = ((prediction - current_price) / current_price) * 100

                st.success("✅ Model trained successfully!")

                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    st.metric(
                        label="🎯 Next Day Prediction",
                        value=f"${prediction:.2f}",
                        delta=f"{predicted_change:+.2f}%"
                    )
                with pred_col2:
                    confidence = model_info['test_score']
                    confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                    st.metric(
                        label="🎲 Model Confidence",
                        value=f"{confidence:.1%}",
                        delta=confidence_level
                    )

                st.info(
                    f"📈 **Training Accuracy:** {model_info['train_score']:.1%} | "
                    f"**Test Accuracy:** {model_info['test_score']:.1%} | "
                    f"**Training Rows:** {model_info['training_rows']:,} trading days"
                )

                # ── 7-day forecast chart ──────────────────────────────
                st.markdown("**📅 7-Day Price Forecast**")
                forecasts = analyzer.predict_7day_forecast(model_info, current_price)
                if forecasts:
                    from datetime import timedelta
                    import pandas as _pd
                    last_date  = data.index[-1]
                    bdays      = _pd.bdate_range(start=last_date + timedelta(days=1), periods=7)
                    labels     = [d.strftime("%b %d") for d in bdays]
                    all_prices = [current_price] + forecasts
                    colors     = ["#34c759" if p >= current_price else "#ff3b30" for p in forecasts]

                    import plotly.graph_objects as _go
                    fig_fc = _go.Figure()
                    fig_fc.add_trace(_go.Scatter(
                        x=["Today"] + labels, y=all_prices,
                        mode='lines+markers+text',
                        text=[f"${p:.2f}" for p in all_prices],
                        textposition="top center",
                        line=dict(color="#5ac8fa", width=2),
                        marker=dict(size=10,
                                    color=["#888"] + colors,
                                    line=dict(width=2, color="#fff")),
                        name="Forecast"
                    ))
                    fig_fc.add_hline(y=current_price, line_dash="dash",
                                     line_color="#888", annotation_text="Current Price")
                    fig_fc.update_layout(
                        template="plotly_dark", height=280,
                        margin=dict(t=20, b=20, l=10, r=10),
                        showlegend=False,
                        yaxis_title="Price (USD)",
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)
            else:
                st.warning("⚠️ Insufficient data for reliable ML prediction. Need more historical data.")
        
        with col2:
            if model_info:
                # Feature importance
                importance_df = pd.DataFrame(
                    list(model_info['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False).head(10)
                
                fig_importance = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="🔍 Top 10 Most Important Features",
                    template='plotly_dark'
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
    
    # AI Market Analysis
    if show_analysis:
        st.subheader("🧠 AI-Powered Market Analysis")
        
        with st.spinner("🤖 Generating intelligent market insights..."):
            analysis = analyzer.generate_market_analysis(data, info, symbol)
        
        # Display analysis in an attractive format
        for i, insight in enumerate(analysis):
            if i == 0:  # First insight (price movement) gets special treatment
                if "🚀" in insight or "🟢" in insight:
                    st.success(insight)
                elif "🔴" in insight or "🔻" in insight:
                    st.error(insight)
                else:
                    st.warning(insight)
            else:
                st.info(insight)
    
    # ── News Sentiment Section ───────────────────────────────────────
    if show_news:
        st.subheader("📰 News Sentiment Analysis")
        if not news_api_key:
            st.warning("🔑 Enter your NewsAPI key in the sidebar to enable news sentiment.")
        else:
            company_name = info.get('longName', symbol) if info else symbol
            with st.spinner("🔍 Fetching latest news…"):
                articles, status_code = analyzer.fetch_news_sentiment(symbol, company_name, news_api_key)

            if status_code == "invalid_key":
                st.error("❌ Invalid NewsAPI key. Please check your key at newsapi.org.")
            elif status_code == "connection_error":
                st.error("❌ Could not connect to NewsAPI. Check your internet connection.")
            elif status_code == "no_articles" or (articles is not None and len(articles) == 0):
                st.info(f"ℹ️ No recent news found for {symbol}.")
            elif articles is None:
                st.error(f"❌ NewsAPI error: {status_code}")
            else:
                # Sentiment summary bar
                pos_count  = sum(1 for a in articles if a['sentiment'] == 'Positive')
                neg_count  = sum(1 for a in articles if a['sentiment'] == 'Negative')
                neu_count  = sum(1 for a in articles if a['sentiment'] == 'Neutral')
                total      = len(articles)

                s_col1, s_col2, s_col3, s_col4 = st.columns(4)
                with s_col1:
                    overall = "🟢 Bullish" if pos_count > neg_count else ("🔴 Bearish" if neg_count > pos_count else "🟡 Mixed")
                    st.metric("Overall Sentiment", overall)
                with s_col2:
                    st.metric("🟢 Positive", f"{pos_count}/{total}")
                with s_col3:
                    st.metric("🔴 Negative", f"{neg_count}/{total}")
                with s_col4:
                    st.metric("🟡 Neutral", f"{neu_count}/{total}")

                st.markdown("**Recent Headlines:**")
                for a in articles:
                    col_e, col_t = st.columns([0.05, 0.95])
                    with col_e:
                        st.write(a['emoji'])
                    with col_t:
                        st.markdown(
                            f"[{a['title']}]({a['url']})  \n"
                            f"<span style='font-size:11px;color:#888;'>{a['source']} · {a['published']}</span>",
                            unsafe_allow_html=True
                        )

    # ────────────────────────────────────────────────────────────────
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📋 Company Info", "📊 Raw Data", "🔧 Technical Indicators"])
    
    with tab1:
        if info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🏢 Company Details")
                company_info = {
                    "Company Name": info.get('longName', 'N/A'),
                    "Sector": info.get('sector', 'N/A'),
                    "Industry": info.get('industry', 'N/A'),
                    "Country": info.get('country', 'N/A'),
                    "Website": info.get('website', 'N/A'),
                    "Employees": f"{info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else 'N/A'
                }
                
                for key, value in company_info.items():
                    st.write(f"**{key}:** {value}")
            
            with col2:
                st.write("### 📈 Financial Metrics")
                financial_info = {
                    "P/E Ratio": f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else 'N/A',
                    "Forward P/E": f"{info.get('forwardPE', 'N/A'):.2f}" if info.get('forwardPE') else 'N/A',
                    "PEG Ratio": f"{info.get('pegRatio', 'N/A'):.2f}" if info.get('pegRatio') else 'N/A',
                    "Price to Book": f"{info.get('priceToBook', 'N/A'):.2f}" if info.get('priceToBook') else 'N/A',
                    "Dividend Yield": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A',
                    "Beta": f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else 'N/A',
                    "52W High": f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if info.get('fiftyTwoWeekHigh') else 'N/A',
                    "52W Low": f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f}" if info.get('fiftyTwoWeekLow') else 'N/A'
                }
                
                for key, value in financial_info.items():
                    st.write(f"**{key}:** {value}")
        else:
            st.warning("Company information not available")
    
    with tab2:
        st.write("### 📊 Recent Price Data")
        display_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(20)
        display_data.index = display_data.index.strftime('%Y-%m-%d')
        st.dataframe(display_data, use_container_width=True)
        
        # Download option
        csv = display_data.to_csv()
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f'{symbol}_stock_data.csv',
            mime='text/csv'
        )
    
    with tab3:
        st.write("### 🔧 Technical Indicators (Last 10 Days)")
        
        tech_columns = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'ATR']
        available_columns = [col for col in tech_columns if col in data.columns]
        
        if available_columns:
            tech_data = data[available_columns].tail(10)
            tech_data.index = tech_data.index.strftime('%Y-%m-%d')
            st.dataframe(tech_data.round(3), use_container_width=True)
        else:
            st.warning("Technical indicators not available")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🚀 <strong>AI Stock Dashboard</strong> - Professional technical analysis with machine learning</p>
            <p><em>⚠️ This is for educational purposes only. Not financial advice.</em></p>
            <p>Built with ❤️ by <a href='https://erikthiart.com' target='_blank'>Erik Thiart</a></p>
            <p>📊 Powered by <a href='https://plotly.com' target='_blank'>Plotly</a> and <a href='https://streamlit.io' target='_blank'>Streamlit</a></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()