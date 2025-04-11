import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Any, Optional

# API URL (can also be retrieved from environment variable)
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Streamlit page configuration
st.set_page_config(
    page_title="Vestas Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.title("Vestas Stock Dashboard 🌬️")
st.markdown("""
    This dashboard displays historical Vestas stock prices and predictions for future prices.
    Data is updated daily, and predictions are based on machine learning models.
""")

# Add a session state to store data between interactions
if 'price_history' not in st.session_state:
    st.session_state.price_history = None

if 'latest_prediction' not in st.session_state:
    st.session_state.latest_prediction = None

if 'features' not in st.session_state:
    st.session_state.features = None

# Function to check API health status
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            # Log API response for debugging
            print(f"API health response: {health_data}")
            return health_data
        return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Function to fetch price history
def get_price_history(days=365):
    try:
        response = requests.get(f"{API_URL}/price/history?days={days}", timeout=10)
        if response.status_code == 200:
            return response.json()['data']
        st.error(f"Error retrieving price history: {response.status_code} {response.reason}")
        return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

# Function to make predictions with LSTM model
def make_lstm_prediction(days_ahead=None):
    try:
        url = f"{API_URL}/predict/lstm"
        if days_ahead:
            url += f"?days_ahead={days_ahead}"
            
        response = requests.post(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        st.error(f"Error executing prediction: {response.status_code} {response.reason}")
        return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

# Function to plot price history
def plot_price_history(price_data):
    if not price_data:
        return None
    
    df = pd.DataFrame(price_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Add moving averages
    df['sma_7'] = df['price'].rolling(window=7).mean()
    df['sma_30'] = df['price'].rolling(window=30).mean()
    
    fig = go.Figure()
    
    # Add area under price line for better visual effect
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price'],
        mode='lines',
        name='Vestas stock price (EUR)',
        line=dict(color='#00573F', width=2),  # Vestas green
        fill='tozeroy',
        fillcolor='rgba(0, 87, 63, 0.1)'
    ))
    
    # Add 7-day moving average
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sma_7'],
        mode='lines',
        name='7-day moving avg.',
        line=dict(color='#2196F3', width=1.5, dash='dot')
    ))
    
    # Add 30-day moving average if there's enough data
    if len(df) > 30:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['sma_30'],
            mode='lines',
            name='30-day moving avg.',
            line=dict(color='#4CAF50', width=1.5)
        ))
    
    # Add volume as bar chart if available
    if 'volume' in df.columns and df['volume'].notna().any():
        fig.add_trace(go.Bar(
            x=df['date'],
            y=df['volume'],
            name='Trading volume',
            marker=dict(color='rgba(158, 158, 158, 0.3)'),
            yaxis='y2'
        ))
    
    # Layout configuration
    fig.update_layout(
        title='Vestas Stock Price History 📊',
        xaxis_title='Date',
        yaxis_title='Price (EUR)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            side='left',
            showgrid=True,
            gridcolor='rgba(240, 240, 240, 0.5)'
        ),
        yaxis2=dict(
            title='Volume',
            side='right',
            overlaying='y',
            showgrid=False
        ),
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(255, 255, 255, 0.9)'
    )
    
    # Add watermark
    fig.add_annotation(
        text="Vestas Dashboard",
        x=0.01,
        y=0.01,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=10, color="rgba(150, 150, 150, 0.5)")
    )
    
    return fig

# Function to display predictions
def plot_predictions(price_data, prediction_data):
    if not price_data or not prediction_data or 'vestas_predictions' not in prediction_data:
        return None
    
    df = pd.DataFrame(price_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Get the latest date and price
    latest_date = pd.to_datetime(prediction_data.get('last_price_date', df['date'].max()))
    latest_price = prediction_data.get('last_price', df['price'].iloc[-1])
    
    # Create prediction points
    predictions = prediction_data['vestas_predictions']
    future_dates = []
    future_prices = []
    horizon_labels = []
    
    for horizon, pred_info in predictions.items():
        future_dates.append(pd.to_datetime(pred_info['prediction_date']))
        future_prices.append(pred_info['predicted_price'])
        horizon_labels.append(f"{pred_info['horizon_days']} days")
    
    # Create DataFrame for plotting
    pred_df = pd.DataFrame({
        'date': [latest_date] + future_dates,
        'price': [latest_price] + future_prices,
        'label': ['Latest'] + horizon_labels
    })
    
    # Sort by date
    pred_df = pred_df.sort_values('date')
    
    # Calculate estimated confidence interval (simulated for visualization)
    pred_df['upper_bound'] = pred_df['price'] * (1 + 0.02 * np.arange(len(pred_df)))
    pred_df['lower_bound'] = pred_df['price'] * (1 - 0.015 * np.arange(len(pred_df)))
    
    # Create figure
    fig = go.Figure()
    
    # Add confidence area
    fig.add_trace(go.Scatter(
        x=pred_df['date'].tolist() + pred_df['date'].tolist()[::-1],
        y=pred_df['upper_bound'].tolist() + pred_df['lower_bound'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0, 87, 63, 0.2)',
        line=dict(color='rgba(0, 87, 63, 0)'),
        name='Forecast interval'
    ))
    
    # Add historical prices
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price'],
        mode='lines',
        name='Historical price',
        line=dict(color='#00573F', width=2)
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=pred_df['date'],
        y=pred_df['price'],
        mode='lines+markers',
        name='Predictions',
        line=dict(color='#4CAF50', width=2, dash='dash'),
        marker=dict(size=8, symbol='circle')
    ))
    
    # Highlight the latest price
    fig.add_trace(go.Scatter(
        x=[latest_date],
        y=[latest_price],
        mode='markers',
        name='Latest price',
        marker=dict(color='#F44336', size=10, symbol='circle')
    ))
    
    # Add labels to the predictions
    for i, row in pred_df.iterrows():
        if i > 0:  # Skip latest price
            fig.add_annotation(
                x=row['date'],
                y=row['price'],
                text=f"{row['price']:.2f} EUR",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#4CAF50",
                ax=0,
                ay=-30
            )
    
    # Layout configuration
    fig.update_layout(
        title='Vestas Stock Price Predictions 🔮',
        xaxis_title='Date',
        yaxis_title='Price (EUR)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(255, 255, 255, 0.9)'
    )
    
    # Add note about uncertainty of predictions
    fig.add_annotation(
        text="Note: The confidence interval is illustrative and does not reflect actual model uncertainty",
        x=0.5,
        y=0,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=10, color="rgba(0, 0, 0, 0.5)"),
        xanchor='center',
        yanchor='top',
        yshift=-20
    )
    
    return fig

# Function to calculate and display volatility
def plot_volatility(price_data, window=14):
    if not price_data:
        return None
    
    df = pd.DataFrame(price_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate daily percent change
    df['pct_change'] = df['price'].pct_change() * 100
    
    # Calculate rolling volatility (standard deviation of percent changes)
    df['volatility'] = df['pct_change'].rolling(window=window).std()
    
    # Clean data for NaN values
    df = df.dropna()
    
    # Create figure
    fig = go.Figure()
    
    # Add volatility line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['volatility'],
        mode='lines',
        name=f'{window}-day volatility',
        line=dict(color='#9C27B0', width=2)
    ))
    
    # Layout configuration
    fig.update_layout(
        title=f'Vestas {window}-day Volatility 📉',
        xaxis_title='Date',
        yaxis_title='Volatility (%)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

# Sidebar navigation
st.sidebar.title("Navigation 🧭")
page = st.sidebar.radio("Select page", ["Dashboard", "Predictions"])

# API status check
with st.sidebar.expander("API Status 🔌", expanded=False):
    health_status = check_api_health()
    if health_status["status"] == "healthy":
        st.sidebar.success("API is online and functioning")
    else:
        st.sidebar.error(f"API is offline or has issues: {health_status.get('error', '')}")
    
    st.sidebar.text(f"API URL: {API_URL}")
    for key, value in health_status.items():
        if key != "status" and key != "error":
            st.sidebar.text(f"{key}: {value}")

# Update data button
if st.sidebar.button("Update data 🔄"):
    with st.spinner("Fetching data..."):
        st.session_state.price_history = get_price_history()
        st.success("Data updated!")

# Select time period
days_to_show = st.sidebar.slider(
    "Number of days to display", 
    min_value=7, 
    max_value=365, 
    value=90, 
    step=1
)

# Load data if not already loaded
if st.session_state.price_history is None:
    with st.spinner("Fetching price history..."):
        st.session_state.price_history = get_price_history()

# Dashboard page
if page == "Dashboard":
    # Display dashboard content
    st.header("Vestas Stock Price Overview 💰")
    
    # Display price data, if available
    if st.session_state.price_history:
        # Filter data based on slider
        filtered_data = st.session_state.price_history[-days_to_show:]
        
        # Create DataFrame for key figures
        df = pd.DataFrame(filtered_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Display key figures
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            latest_price = df['price'].iloc[-1]
            st.metric("Latest price", f"{latest_price:.2f} EUR")
        
        with col2:
            price_change = df['price'].iloc[-1] - df['price'].iloc[0]
            price_change_pct = (price_change / df['price'].iloc[0]) * 100
            st.metric("Change", f"{price_change:.2f} EUR", f"{price_change_pct:.2f}%")
        
        with col3:
            high_price = df['price'].max()
            st.metric("Highest price", f"{high_price:.2f} EUR")
        
        with col4:
            low_price = df['price'].min()
            st.metric("Lowest price", f"{low_price:.2f} EUR")
        
        # Add more metrics
        st.subheader("Market Statistics 📊")
        col1, col2 = st.columns(2)
        
        with col1:
            avg_price = df['price'].mean()
            st.metric("Average price", f"{avg_price:.2f} EUR")
            
            if 'volume' in df.columns and df['volume'].notna().any():
                latest_volume = df['volume'].iloc[-1]
                st.metric("Trading volume", f"{latest_volume:,.0f}")
        
        with col2:
            # 30-day return
            if len(df) >= 30:
                return_30d = (df['price'].iloc[-1] / df['price'].iloc[-min(30, len(df))] - 1) * 100
                st.metric("30-day return", f"{return_30d:.2f}%", delta_color="normal" if return_30d >= 0 else "inverse")
            
            # Calculate volatility (last 14 days)
            if len(df) >= 14:
                volatility = df['price'].pct_change().rolling(14).std().iloc[-1] * 100
                st.metric("14-day volatility", f"{volatility:.2f}%")
        
        # Display tabs with different graphs
        tab1, tab2 = st.tabs(["Price Development", "Volatility"])
        
        with tab1:
            # Display price graph
            price_chart = plot_price_history(filtered_data)
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)
            else:
                st.error("Could not generate price chart.")
        
        with tab2:
            # Display volatility graph
            volatility_chart = plot_volatility(filtered_data)
            if volatility_chart:
                st.plotly_chart(volatility_chart, use_container_width=True)
            else:
                st.error("Could not generate volatility chart.")
    else:
        st.warning("No price data available. Try updating data via the button in the sidebar.")

# Predictions page
elif page == "Predictions":
    st.header("Vestas Stock Price Predictions 🔮")
    
    # Display info about predictions
    st.info("""
    **Note**: The predictions are based on the latest Vestas stock price data combined with an LSTM model.
    Results should be interpreted with caution and not as investment advice.
    """)
    
    if st.session_state.price_history:
        st.subheader("Predictions based on latest data ✨")
        
        # UI for horizons
        col1, col2 = st.columns([1, 2])
        with col1:
            horizon_choice = st.radio(
                "Select prediction horizon",
                options=["All horizons", "1 day", "3 days", "7 days"]
            )
            
            # Convert choice to days_ahead parameter
            days_ahead = None
            if horizon_choice == "1 day":
                days_ahead = 1
            elif horizon_choice == "3 days":
                days_ahead = 3
            elif horizon_choice == "7 days":
                days_ahead = 7
        
        # Make prediction
        if st.button("Make prediction 🚀"):
            with st.spinner("Making prediction..."):
                st.session_state.latest_prediction = make_lstm_prediction(days_ahead)
        
        # Display prediction, if available
        if st.session_state.latest_prediction and 'vestas_predictions' in st.session_state.latest_prediction:
            predictions = st.session_state.latest_prediction['vestas_predictions']
            current_price = st.session_state.latest_prediction.get('last_price', 0)
            
            # Display predictions in columns
            cols = st.columns(len(predictions) + 1)
            
            # Display current price
            with cols[0]:
                st.metric("Current price", f"{current_price:.2f} EUR")
            
            # Display predictions for each horizon
            for i, (horizon, pred_info) in enumerate(sorted(predictions.items(), key=lambda x: int(x[0]))):
                with cols[i+1]:
                    price = pred_info['predicted_price']
                    change = price - current_price
                    change_pct = (change / current_price) * 100
                    days = pred_info['horizon_days']
                    horizon_display = f"{days} day{'s' if days > 1 else ''}"
                    st.metric(
                        horizon_display,
                        f"{price:.2f} EUR",
                        f"{change_pct:+.2f}%",
                        delta_color="normal" if change >= 0 else "inverse"
                    )
            
            # Display prediction graph
            prediction_chart = plot_predictions(
                st.session_state.price_history[-days_to_show:],
                st.session_state.latest_prediction
            )
            if prediction_chart:
                st.plotly_chart(prediction_chart, use_container_width=True)
            else:
                st.error("Could not generate prediction chart.")
            
            # Display model type
            st.caption(f"Model: {st.session_state.latest_prediction.get('model_type', 'LSTM')}")
        else:
            st.info("Click on 'Make prediction' to get predictions for the coming days.")
    else:
        st.warning("No price data available. Try updating data via the button in the sidebar.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 Vestas Dashboard 🌬️") 