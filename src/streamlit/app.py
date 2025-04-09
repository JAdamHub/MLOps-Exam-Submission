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
    page_title="Bitcoin Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.title("Bitcoin Dashboard 🚀")
st.markdown("""
    This dashboard shows historical Bitcoin prices and predictions for future prices.
    Data is updated daily and predictions are based on machine learning models.
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

# Function to fetch features
def get_features():
    try:
        response = requests.get(f"{API_URL}/features", timeout=5)
        if response.status_code == 200:
            return response.json()['features']
        st.error(f"Error retrieving features: {response.status_code} {response.reason}")
        return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

# Function to make predictions
def make_prediction(features):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=features,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        st.error(f"Error making prediction: {response.status_code} {response.reason}")
        if response.status_code == 400:
            st.error(f"Details: {response.json().get('detail', '')}")
        return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

# Function to get training metrics
def get_metrics():
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
        st.error(f"Error retrieving metrics: {response.status_code} {response.reason}")
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
        name='Bitcoin price (USD)',
        line=dict(color='#F7931A', width=2),  # Bitcoin orange
        fill='tozeroy',
        fillcolor='rgba(247, 147, 26, 0.1)'
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
    if 'total_volume' in df.columns and df['total_volume'].notna().any():
        fig.add_trace(go.Bar(
            x=df['date'],
            y=df['total_volume'],
            name='Trading volume',
            marker=dict(color='rgba(158, 158, 158, 0.3)'),
            yaxis='y2'
        ))
    
    # Layout configuration
    fig.update_layout(
        title='Bitcoin Price History 📊',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
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
        text="Bitcoin Dashboard",
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
    if not price_data or not prediction_data or 'predictions' not in prediction_data:
        return None
    
    df = pd.DataFrame(price_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Get the latest date and price
    latest_date = df['date'].max()
    latest_price = df['price'].iloc[-1]
    
    # Create prediction points
    predictions = prediction_data['predictions']
    future_dates = {}
    future_prices = {}
    
    for horizon, price in predictions.items():
        if horizon != 'current_price':
            days = int(horizon.replace('d', ''))
            future_dates[horizon] = latest_date + timedelta(days=days)
            future_prices[horizon] = price
    
    # Sort dates to create a continuous line
    pred_df = pd.DataFrame({
        'date': [latest_date] + [future_dates[h] for h in sorted(future_dates.keys(), key=lambda x: int(x.replace('d', '')))],
        'price': [latest_price] + [future_prices[h] for h in sorted(future_dates.keys(), key=lambda x: int(x.replace('d', '')))]
    })
    
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
        fillcolor='rgba(76, 175, 80, 0.2)',
        line=dict(color='rgba(76, 175, 80, 0)'),
        name='Forecast interval'
    ))
    
    # Add historical prices
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price'],
        mode='lines',
        name='Historical price',
        line=dict(color='#F7931A', width=2)
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
                text=f"${row['price']:,.0f}",
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
        title='Bitcoin Price Predictions 🔮',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
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
        text="Note: Confidence interval is illustrative and does not reflect actual model uncertainty",
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

# Function to create latest feature data
def get_latest_features(price_data):
    if not price_data:
        return None
    
    if not st.session_state.features:
        st.session_state.features = get_features()
    
    if not st.session_state.features:
        return None
    
    # Get the latest data record as a starting point
    latest_data = price_data[-1]
    
    # Convert to features format
    features = {}
    
    # Set default values for all features (based on averages or reasonable values)
    default_features = {
        'price': latest_data.get('price', 60000),
        'market_cap': latest_data.get('market_cap', 1200000000000),
        'fed_rate': 5.0,
        'sp500_pct_change': 0.001,
        'eurusd_pct_change': 0.0005,
        'price_lag_7': latest_data.get('price', 60000) * 0.95,  # approx. 5% less than current price
        'price_sma_7': latest_data.get('price', 60000) * 1.02,  # approx. 2% more than current price
        'market_cap_momentum_30': 0.02,
        'treasury_10y_volatility': 0.03,
        'treasury_5y_ma7': 3.5,
        'vix_ma30': 18.0,
        'dxy_volatility': 0.05,
        'sp500_ma30': 4500,
        'btc_nasdaq_corr_30d': 0.65,
        'btc_dow_corr_30d': 0.45,
        'gold_ma7': 2000,
        'oil_ma7': 80,
        'eurusd_rsi_14': 50,
        'gold_dxy_corr_30d': -0.3,
        'oil_dxy_corr_30d': -0.2
    }
    
    # For each feature, try to find it in our data or set it to a default value
    for feature in st.session_state.features:
        if feature in latest_data:
            features[feature] = latest_data[feature]
        else:
            # If the feature doesn't exist in our data, use the default value
            features[feature] = default_features.get(feature, 0.0)
    
    return features

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
        name=f'{window}-Day Volatility',
        line=dict(color='#9C27B0', width=2)
    ))
    
    # Layout configuration
    fig.update_layout(
        title=f'Bitcoin {window}-Day Volatility 📉',
        xaxis_title='Date',
        yaxis_title='Volatility (%)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

# Sidebar navigation
st.sidebar.title("Navigation 🧭")
page = st.sidebar.radio("Select page", ["Dashboard", "Predictions", "Model Metrics"])

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
        st.session_state.features = get_features()
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
    st.header("Bitcoin Price Overview 💰")
    
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
            st.metric("Latest price", f"${latest_price:,.2f}")
        
        with col2:
            price_change = df['price'].iloc[-1] - df['price'].iloc[0]
            price_change_pct = (price_change / df['price'].iloc[0]) * 100
            st.metric("Change", f"${price_change:,.2f}", f"{price_change_pct:.2f}%")
        
        with col3:
            high_price = df['price'].max()
            st.metric("Highest price", f"${high_price:,.2f}")
        
        with col4:
            low_price = df['price'].min()
            st.metric("Lowest price", f"${low_price:,.2f}")
        
        # Add more metrics
        st.subheader("Market Statistics 📊")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_price = df['price'].mean()
            st.metric("Average price", f"${avg_price:,.2f}")
            
            if 'market_cap' in df.columns:
                latest_market_cap = df['market_cap'].iloc[-1] / 1_000_000_000
                st.metric("Market Cap", f"${latest_market_cap:,.2f} B")
        
        with col2:
            # 30-day return
            if len(df) >= 30:
                return_30d = (df['price'].iloc[-1] / df['price'].iloc[-min(30, len(df))] - 1) * 100
                st.metric("30-day return", f"{return_30d:.2f}%", delta_color="normal" if return_30d >= 0 else "inverse")
            
            # Calculate volatility (last 14 days)
            if len(df) >= 14:
                volatility = df['price'].pct_change().rolling(14).std().iloc[-1] * 100
                st.metric("14-day volatility", f"{volatility:.2f}%")
        
        with col3:
            # Volume if available
            if 'total_volume' in df.columns and df['total_volume'].notna().any():
                avg_volume = df['total_volume'].mean() / 1_000_000_000
                latest_volume = df['total_volume'].iloc[-1] / 1_000_000_000
                st.metric("Trading volume (24h)", f"${latest_volume:.2f} B", 
                        f"{(latest_volume/avg_volume - 1)*100:.1f}% vs. avg.")
        
        # Display tabs with different graphs
        tab1, tab2, tab3 = st.tabs(["Price Development", "Volatility", "Price Distribution"])
        
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
        
        with tab3:
            # Display histogram of price distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['price'],
                nbinsx=20,
                marker_color='#F7931A'
            ))
            
            fig.update_layout(
                title='Distribution of Bitcoin Prices 📊',
                xaxis_title='Price (USD)',
                yaxis_title='Number of days',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No price data available. Try updating data via the button in the sidebar.")

# Predictions page
elif page == "Predictions":
    st.header("Bitcoin Price Predictions 🔮")
    
    # Display info about simulated features
    st.info("""
    **Note**: The predictions are based on the latest Bitcoin price data combined with simulated values 
    for other economic indicators that the model requires (e.g., correlations, technical indicators, etc.).
    This is for demonstration purposes only, and the results therefore do not necessarily reflect the actual market situation.
    """)
    
    if st.session_state.price_history:
        # Get latest features
        latest_features = get_latest_features(st.session_state.price_history)
        
        if latest_features:
            st.subheader("Predictions based on latest data ✨")
            
            # Make prediction
            if st.button("Make prediction 🚀"):
                with st.spinner("Making prediction..."):
                    st.session_state.latest_prediction = make_prediction(latest_features)
            
            # Display prediction, if available
            if st.session_state.latest_prediction and 'predictions' in st.session_state.latest_prediction:
                predictions = st.session_state.latest_prediction['predictions']
                
                # Display predictions in columns
                cols = st.columns(4)
                
                # Display current price
                with cols[0]:
                    current_price = predictions.get('current_price', 0)
                    st.metric("Current price", f"${current_price:,.2f}")
                
                # Display predictions for each horizon
                horizons = [h for h in predictions.keys() if h != 'current_price']
                for i, horizon in enumerate(sorted(horizons, key=lambda x: int(x.replace('d', '')))):
                    with cols[i+1]:
                        price = predictions[horizon]
                        change = price - current_price
                        change_pct = (change / current_price) * 100
                        days_text = horizon.replace('d', '')
                        horizon_display = f"{days_text} day{'s' if days_text != '1' else ''}"
                        st.metric(
                            horizon_display,
                            f"${price:,.2f}",
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
                
                # Display prediction timestamp
                st.caption(f"Prediction made: {st.session_state.latest_prediction['timestamp']}")
            
            else:
                st.info("Click 'Make prediction' to get predictions for the coming days.")
        else:
            st.warning("Could not construct features for predictions. Try updating data.")
    else:
        st.warning("No price data available. Try updating data via the button in the sidebar.")

# Model metrics page
elif page == "Model Metrics":
    st.header("Model Metrics 📏")
    
    # Get metrics
    metrics_data = get_metrics()
    
    if metrics_data and 'metrics' in metrics_data:
        metrics = metrics_data['metrics']
        
        # Display metrics for each model
        for horizon, metric_values in metrics.items():
            days_text = horizon.replace('price_target_', '').replace('d', '')
            st.subheader(f"Model for {days_text} day{'s' if days_text != '1' else ''} ⏱️")
            
            cols = st.columns(3)
            with cols[0]:
                st.metric("RMSE", f"{metric_values['rmse']:.4f}")
            with cols[1]:
                st.metric("MAE", f"{metric_values['mae']:.4f}")
            with cols[2]:
                st.metric("R²", f"{metric_values['r2']:.4f}")
        
        # Feature importance
        if 'feature_importance' in metrics_data:
            st.subheader("Feature Importance 🌟")
            
            # Tab for each horizon
            tab_labels = []
            for h in metrics_data['feature_importance'].keys():
                days_text = h.replace('price_target_', '').replace('d', '')
                tab_labels.append(f"{days_text} day{'s' if days_text != '1' else ''}")
            
            tabs = st.tabs(tab_labels)
            
            for i, (horizon, importance) in enumerate(metrics_data['feature_importance'].items()):
                with tabs[i]:
                    # Sort features by importance
                    sorted_features = sorted(
                        importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    # Create DataFrame
                    df = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])
                    
                    # Display importance chart
                    fig = go.Figure(go.Bar(
                        x=df['Importance'],
                        y=df['Feature'],
                        orientation='h',
                        marker=dict(color='#F7931A')
                    ))
                    
                    days_text = horizon.replace('price_target_', '').replace('d', '')
                    horizon_display = f"{days_text} day{'s' if days_text != '1' else ''}"
                    
                    fig.update_layout(
                        title=f"Feature Importance for {horizon_display} 📊",
                        xaxis_title='Importance',
                        yaxis_title='Feature',
                        height=600,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not retrieve model metrics from the API.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2024 Bitcoin Dashboard 🌐") 