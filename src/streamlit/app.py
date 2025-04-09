import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import joblib
import os
import sys
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Tilføj import sti-justeringer for at kunne køre både fra rod og fra src/streamlit
current_dir = Path(__file__).parent.absolute()
if current_dir.name == 'streamlit':
    # Vi er i src/streamlit, så importer monitoring direkte
    import_path = "monitoring"
else:
    # Vi er et andet sted, så brug absolut import
    import_path = "src.streamlit.monitoring"

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
    try:
        if import_path == "monitoring":
            import monitoring as monitoring
        else:
            import src.streamlit.monitoring as monitoring
        # Denne import vil køre monitoring-siden
        st.stop()  # Stop denne side for at undgå dobbelt rendering
    except ImportError as e:
        st.error(f"Kunne ikke importere monitoring modul: {e}")
        st.error("Streamlit skal køres fra projektets rodmappe med: streamlit run src/streamlit/app.py")
        st.stop()
    
# Resten af koden køres kun hvis vi er på Prediction-siden

# Load feature names and scaler at startup
try:
    scaler = joblib.load('models/scaler.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    st.success(f"Indlæste {len(feature_names)} features fra model")
except Exception as e:
    st.error(f"Kunne ikke indlæse scaler eller feature_names: {e}")
    scaler = None
    feature_names = []

# Function to get latest data
def get_latest_data():
    try:
        # Bruger trading days filen i stedet
        df = pd.read_csv('data/features/bitcoin_features_trading_days.csv')
        return df.iloc[-1]
    except Exception as e:
        st.error(f"Could not fetch data: {e}")
        return None

# Function to make prediction
def make_prediction(data):
    """Make prediction using the API."""
    try:
        # Check for missing features
        missing_features = [feat for feat in feature_names if feat not in data]
        if missing_features:
            st.error(f"Manglende features: {', '.join(missing_features)}")
            return None
            
        # Forbered data til API kald - kun features som modellen bruger
        prediction_data = {}
        for feature in feature_names:
            if feature in data:
                # Konverter værdien til korrekt type (int eller float)
                if feature in ['day_of_week', 'month', 'year', 'day_of_month', 'is_weekend']:
                    prediction_data[feature] = int(data[feature])
                else:
                    prediction_data[feature] = float(data[feature])
        
        # Send request til API - brug API_URL miljøvariabel
        api_endpoint = f"{API_URL}/predict"
        st.info(f"Sender anmodning til API: {api_endpoint}")
        
        # Tilføj timeout og retry_strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Brug session med timeout og retry
        response = session.post(
            api_endpoint,
            json=prediction_data,
            headers={"Content-Type": "application/json"},
            timeout=10  # 10 sekunder timeout
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
        st.error(f"API URL: {API_URL} - Check venligst om API'en kører på denne adresse")
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