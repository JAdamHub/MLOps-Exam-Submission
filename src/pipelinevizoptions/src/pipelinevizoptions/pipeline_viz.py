import graphviz
from pathlib import Path

def create_pipeline_visualization():
    """Creates a visualization of the complete MLOps pipeline architecture."""
    
    # Create a new digraph
    dot = graphviz.Digraph(comment='MLOps Pipeline Architecture')
    dot.attr(rankdir='LR')  # Left to right layout
    
    # Add styling
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    dot.attr('edge', color='gray50')
    
    # Data Pipeline Nodes
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Data Pipeline')
        c.attr('node', fillcolor='lightgreen')
        c.node('vestas_data', 'Vestas Stock Data\n(VWSB.DEX)')
        c.node('macro_data', 'Macroeconomic Data\n(Alpha Vantage API)')
        c.node('combined_data', 'Combined Data\n(combined_data_processor.py)')
        c.node('processed_data', 'Processed Data\n(preprocessing.py)')
        c.node('feature_data', 'Feature Data\n(feature_engineering.py)')
        
        # Data flow
        c.edge('vestas_data', 'combined_data')
        c.edge('macro_data', 'combined_data')
        c.edge('combined_data', 'processed_data')
        c.edge('processed_data', 'feature_data')
    
    # Model Training Nodes
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Model Training & Deployment')
        c.attr('node', fillcolor='lightyellow')
        c.node('model_training', 'LSTM Model Training\n(lstm_model.py)')
        c.node('lstm_model', 'LSTM Model\n(lstm_multi_horizon_model.keras)')
        c.node('feature_scaler', 'Feature Scaler\n(lstm_feature_scaler.joblib)')
        c.node('target_scalers', 'Target Scalers\n(lstm_target_scalers.joblib)')
        c.node('model_metrics', 'Model Metrics\n(model_metrics.json)')
        
        # Model flow
        c.edge('feature_data', 'model_training')
        c.edge('model_training', 'lstm_model')
        c.edge('model_training', 'feature_scaler')
        c.edge('model_training', 'target_scalers')
        c.edge('model_training', 'model_metrics')
    
    # API & Monitoring Nodes
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='API & Monitoring')
        c.attr('node', fillcolor='lightpink')
        c.node('api', 'FastAPI\n(stock_api.py)')
        c.node('prediction_store', 'Prediction Store\n(prediction_store.py)')
        c.node('scheduler', 'Model Update Scheduler\n(scheduler.py)')
        c.node('metrics_viz', 'Model Metrics Viz\n(model_metrics_viz.py)')
        
        # API flow
        c.edge('lstm_model', 'api')
        c.edge('feature_scaler', 'api')
        c.edge('target_scalers', 'api')
        c.edge('api', 'prediction_store')
        c.edge('prediction_store', 'scheduler')
        c.edge('model_metrics', 'metrics_viz')
        c.edge('scheduler', 'model_training', style='dashed')
    
    # Save visualization
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    dot.render(str(output_dir / 'pipeline_architecture'), format='png', cleanup=True)
    print(f"Visualization saved in {output_dir}/pipeline_architecture.png")

if __name__ == '__main__':
    create_pipeline_visualization() 