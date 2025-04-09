import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import os
import json
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# API URL configuration - default for local dev, overridden by environment variable in Docker
API_URL = os.environ.get('API_URL', 'http://localhost:8000')

# Konfigurer retry strategi
retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("http://", adapter)
session.mount("https://", adapter)

# Configure page
st.set_page_config(
    page_title="Model Monitoring",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Model Monitoring Dashboard")

def get_api_data(endpoint):
    """Helper function to get data from API"""
    try:
        response = session.get(f"{API_URL}/{endpoint}", timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching data from {endpoint}: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error communicating with API: {e}")
        st.error(f"API URL: {API_URL} - Check venligst om API'en kører på denne adresse")
        return None

# Sidebar actions
st.sidebar.header("Model Actions")

if st.sidebar.button("Trigger Model Retraining"):
    try:
        response = session.post(f"{API_URL}/monitoring/retrain", timeout=10)
        if response.status_code == 200:
            st.sidebar.success("Retraining initiated!")
        else:
            st.sidebar.error(f"Error triggering retraining: {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"Error communicating with API: {e}")
        st.sidebar.error(f"API URL: {API_URL} - Check venligst om API'en kører på denne adresse")

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Health Checks")
refresh = st.sidebar.button("Refresh Data")

# Main content in tabs
tab1, tab2, tab3 = st.tabs(["Model Performance", "Data Drift", "Predictions"])

with tab1:
    st.header("Model Performance Metrics")
    
    # Get performance metrics
    performance = get_api_data("monitoring/performance")
    
    if performance:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{performance.get('latest_accuracy', 0):.2%}")
            st.metric("Precision", f"{performance.get('latest_precision', 0):.2%}")
        
        with col2:
            st.metric("Recall", f"{performance.get('latest_recall', 0):.2%}")
            st.metric("F1 Score", f"{performance.get('latest_f1', 0):.2%}")
        
        with col3:
            st.metric("Accuracy Trend", performance.get('accuracy_trend', 'N/A'))
            if performance.get('profit_loss') is not None:
                st.metric("Profit/Loss", f"{performance.get('profit_loss', 0):.2f}")
        
        # Load and display performance charts if available
        st.subheader("Performance Over Time")
        st.info("Charts would be loaded from models/figures/ directory")
    else:
        st.warning("No performance data available")

with tab2:
    st.header("Data Drift Detection")
    
    # Get drift detection results
    drift_results = get_api_data("monitoring/drift")
    
    if drift_results and "has_significant_drift" in drift_results:
        # Display drift status
        if drift_results.get("has_significant_drift", False):
            st.error("⚠️ Significant drift detected in data!")
        else:
            st.success("✅ No significant drift detected")
        
        # Display recommendation
        recommendation = drift_results.get("recommendation", {})
        if recommendation.get("needs_retraining", False):
            st.warning(f"Recommendation: Retrain model - {recommendation.get('reason')}")
        else:
            st.info(f"Recommendation: {recommendation.get('reason', 'No action needed')}")
        
        # Show top drifted features
        top_drifted = drift_results.get("top_drifted_features", [])
        if top_drifted:
            st.subheader("Top Features with Drift")
            
            # Convert to DataFrame for display
            drift_df = pd.DataFrame(top_drifted)
            
            # Display as bar chart
            fig = px.bar(
                drift_df, 
                x='drift_magnitude', 
                y='feature', 
                orientation='h',
                title="Feature Drift Magnitude"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No drift detection data available")

with tab3:
    st.header("Recent Predictions")
    
    # Get prediction history
    predictions_data = get_api_data("monitoring/predictions")
    
    if predictions_data and "predictions" in predictions_data:
        predictions = predictions_data["predictions"]
        
        if predictions:
            # Convert to DataFrame
            pred_df = pd.DataFrame(predictions)
            
            # Convert timestamp to datetime
            pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
            
            # Display recent predictions
            st.dataframe(
                pred_df[['timestamp', 'prediction', 'actual_value', 'model_version']]
                .sort_values('timestamp', ascending=False)
            )
            
            # Show prediction distribution
            st.subheader("Prediction Distribution")
            
            # Count predictions by class
            prediction_counts = pred_df['prediction'].value_counts().reset_index()
            prediction_counts.columns = ['Prediction', 'Count']
            
            # Create pie chart
            fig = px.pie(
                prediction_counts, 
                values='Count', 
                names='Prediction',
                hole=0.4,
                color_discrete_map={0: 'red', 1: 'green'}
            )
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No predictions available")
    else:
        st.warning("Failed to load prediction data")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Model monitoring dashboard displays real-time information about model performance, 
        data drift, and predictions. This helps ensure the model remains accurate over time.</p>
    </div>
    """, 
    unsafe_allow_html=True
) 