import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json

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

# Function to get latest data
def get_latest_data():
    try:
        df = pd.read_csv('data/features/bitcoin_usd_365d_features.csv')
        return df.iloc[-1]
    except Exception as e:
        st.error(f"Could not fetch data: {e}")
        return None

# Function to make prediction
def make_prediction(data):
    try:
        # Prepare data for API call
        prediction_data = {
            "price": float(data['price']),
            "market_cap": float(data['market_cap']),
            "total_volume": float(data['total_volume']),
            "price_lag_1": float(data['price_lag_1']),
            "price_lag_3": float(data['price_lag_3']),
            "price_lag_7": float(data['price_lag_7']),
            "price_sma_7": float(data['price_sma_7']),
            "price_sma_30": float(data['price_sma_30']),
            "price_volatility_14": float(data['price_volatility_14']),
            "day_of_week": int(data['day_of_week']),
            "month": int(data['month']),
            "year": int(data['year'])
        }
        
        # Call the API
        response = requests.post(
            'http://localhost:8000/predict',
            json=prediction_data
        )
        return response.json()
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Function to display price history
def plot_price_history():
    try:
        df = pd.read_csv('data/raw/bitcoin_usd_365d_raw.csv')
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
        
        if prediction:
            # Display prediction
            st.markdown("### Prediction for Tomorrow:")
            
            # Display prediction with color
            if prediction['prediction'] == 1:
                st.success("📈 Price is expected to RISE")
            else:
                st.error("📉 Price is expected to FALL")
            
            # Display probability
            st.markdown(f"### Probability: {prediction['probability']*100:.1f}%")
            
            # Display latest price data
            st.markdown("### Latest Price Data:")
            st.write(f"Price: ${float(latest_data['price']):,.2f}")
            st.write(f"Market Cap: ${float(latest_data['market_cap']):,.2f}")
            st.write(f"24h Volume: ${float(latest_data['total_volume']):,.2f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>This is an educational project. Not for real trading.</p>
    <p>Data is updated daily from CoinGecko API</p>
</div>
""", unsafe_allow_html=True) 