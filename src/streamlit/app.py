import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import joblib
import os

# API URL configuration - default for local dev, overridden by environment variable in Docker
API_URL = os.environ.get('API_URL', 'http://localhost:8000')

# Configure Streamlit page
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Title and introduction
st.title("📈 Bitcoin Price Prediction")
st.markdown("""
This is a simple model to predict whether the Bitcoin price will rise or fall tomorrow.
Note: This is for educational purposes only and should not be used as financial advice.
""")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Monitoring"])

if page == "Monitoring":
    # Import og vis monitoring-siden
    import src.streamlit.monitoring as monitoring
    # Denne import vil køre monitoring-siden
    st.stop()  # Stop denne side for at undgå dobbelt rendering
    
# Resten af koden køres kun hvis vi er på Prediction-siden

# Load scaler at startup
try:
    scaler = joblib.load('models/scaler.joblib')
except Exception as e:
    st.error(f"Kunne ikke indlæse scaler: {e}")
    scaler = None

# Function to get latest data
def get_latest_data():
    try:
        # Opdateret filsti til features
        df = pd.read_csv('data/features/bitcoin_features.csv')
        return df.iloc[-1]
    except Exception as e:
        st.error(f"Could not fetch data: {e}")
        return None

# Function to make prediction
def make_prediction(data):
    """Make prediction using the API."""
    try:
        # Valider input data
        required_features = [
            'price', 'market_cap', 'total_volume',
            'price_lag_1', 'price_lag_3', 'price_lag_7',
            'price_sma_7', 'price_sma_30',
            'price_volatility_14', 'day_of_week', 'month', 'year',
            'rsi_14', 'rsi_7', 'rsi_21',
            'macd', 'macd_signal', 'macd_histogram',
            'macd_fast', 'macd_signal_fast',
            'bb_upper', 'bb_middle', 'bb_lower',
            'bb_width', 'bb_position',
            'volume_sma_7', 'volume_sma_30',
            'volume_ratio', 'volume_ratio_30',
            'volume_momentum',
            'price_momentum_1', 'price_momentum_7',
            'price_momentum_30', 'price_momentum_90',
            'volatility_7', 'volatility_14', 'volatility_30',
            'market_cap_to_volume',
            'market_cap_momentum_1', 'market_cap_momentum_7',
            'market_cap_momentum_30',
            'volume_to_market_cap',
            'price_volatility_ratio',
            'momentum_volatility_ratio',
            'volume_price_ratio',
            'day_of_month',
            'is_weekend'
        ]
        
        # Check for missing features
        missing_features = [feat for feat in required_features if feat not in data]
        if missing_features:
            st.error(f"Manglende features: {', '.join(missing_features)}")
            return None
            
        # Forbered data til API kald
        prediction_data = {
            'price': float(data['price']),
            'market_cap': float(data['market_cap']),
            'total_volume': float(data['total_volume']),
            'price_lag_1': float(data['price_lag_1']),
            'price_lag_3': float(data['price_lag_3']),
            'price_lag_7': float(data['price_lag_7']),
            'price_sma_7': float(data['price_sma_7']),
            'price_sma_30': float(data['price_sma_30']),
            'price_volatility_14': float(data['price_volatility_14']),
            'day_of_week': int(data['day_of_week']),
            'month': int(data['month']),
            'year': int(data['year']),
            'rsi_14': float(data['rsi_14']),
            'rsi_7': float(data['rsi_7']),
            'rsi_21': float(data['rsi_21']),
            'macd': float(data['macd']),
            'macd_signal': float(data['macd_signal']),
            'macd_histogram': float(data['macd_histogram']),
            'macd_fast': float(data['macd_fast']),
            'macd_signal_fast': float(data['macd_signal_fast']),
            'bb_upper': float(data['bb_upper']),
            'bb_middle': float(data['bb_middle']),
            'bb_lower': float(data['bb_lower']),
            'bb_width': float(data['bb_width']),
            'bb_position': float(data['bb_position']),
            'volume_sma_7': float(data['volume_sma_7']),
            'volume_sma_30': float(data['volume_sma_30']),
            'volume_ratio': float(data['volume_ratio']),
            'volume_ratio_30': float(data['volume_ratio_30']),
            'volume_momentum': float(data['volume_momentum']),
            'price_momentum_1': float(data['price_momentum_1']),
            'price_momentum_7': float(data['price_momentum_7']),
            'price_momentum_30': float(data['price_momentum_30']),
            'price_momentum_90': float(data['price_momentum_90']),
            'volatility_7': float(data['volatility_7']),
            'volatility_14': float(data['volatility_14']),
            'volatility_30': float(data['volatility_30']),
            'market_cap_to_volume': float(data['market_cap_to_volume']),
            'market_cap_momentum_1': float(data['market_cap_momentum_1']),
            'market_cap_momentum_7': float(data['market_cap_momentum_7']),
            'market_cap_momentum_30': float(data['market_cap_momentum_30']),
            'volume_to_market_cap': float(data['volume_to_market_cap']),
            'price_volatility_ratio': float(data['price_volatility_ratio']),
            'momentum_volatility_ratio': float(data['momentum_volatility_ratio']),
            'volume_price_ratio': float(data['volume_price_ratio']),
            'day_of_month': int(data['day_of_month']),
            'is_weekend': int(data['is_weekend'])
        }
        
        # Send request til API - brug API_URL miljøvariabel
        api_endpoint = f"{API_URL}/predict"
        st.info(f"Sender anmodning til API: {api_endpoint}")
        
        response = requests.post(
            api_endpoint,
            json=prediction_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error(f"Fejl ved forespørgsel til API: {response.status_code} {response.reason}")
            if response.text:
                try:
                    error_detail = response.json().get('detail', '')
                    st.error(f"API fejlbesked: {error_detail}")
                except:
                    st.error(f"API fejlbesked: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Netværksfejl: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Uventet fejl: {str(e)}")
        return None

# Function to display price history
def plot_price_history():
    try:
        # Opdateret filsti til rådata
        df = pd.read_csv('data/raw/crypto/bitcoin_usd_365d.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['price'],
            mode='lines',
            name='Bitcoin Price'
        ))
        
        fig.update_layout(
            title='Bitcoin Price History (365 days)',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark'
        )
        
        return fig
    except Exception as e:
        st.error(f"Could not display price history: {e}")
        return None

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Price History")
    fig = plot_price_history()
    if fig:
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Today's Prediction")
    
    # Get latest data
    latest_data = get_latest_data()
    if latest_data is not None:
        # Make prediction
        prediction = make_prediction(latest_data)
        
        if prediction is not None:
            try:
                # Display prediction
                st.markdown("### Prediction for Tomorrow:")
                
                # Display prediction with color
                if prediction.get('prediction') == 1:
                    st.success("📈 Price is expected to RISE")
                else:
                    st.error("📉 Price is expected to FALL")
                
                # Display probability
                probability = prediction.get('probability', 0)
                st.markdown(f"### Probability: {probability*100:.1f}%")
                
                # Show prediction timestamp if available
                if 'timestamp' in prediction:
                    timestamp = prediction.get('timestamp')
                    st.info(f"Prediction generated at: {timestamp}")
                
                # Display latest price data
                st.markdown("### Latest Price Data:")
                st.write(f"Price: ${float(latest_data['price']):,.2f}")
                st.write(f"Market Cap: ${float(latest_data['market_cap']):,.2f}")
                st.write(f"24h Volume: ${float(latest_data['total_volume']):,.2f}")
            except Exception as e:
                st.error(f"Error displaying prediction: {e}")
        else:
            st.error("Could not get prediction from API")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>This is an educational project. Not for real trading.</p>
    <p>Data is updated daily from CoinGecko API</p>
</div>
""", unsafe_allow_html=True) 