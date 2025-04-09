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
        c.node('raw_data', 'Raw Crypto Data\n(bitcoin_usd_365d_raw.csv)')
        c.node('macro_data', 'Macroeconomic Data\n(Yahoo Finance API)')
        c.node('combined_data', 'Combined Data\n(combined_data_processor.py)')
        c.node('processed_data', 'Processed Data\n(preprocessing.py)')
        c.node('feature_data', 'Feature Data\n(feature_engineering.py)')
        
        # Data flow
        c.edge('raw_data', 'combined_data')
        c.edge('macro_data', 'combined_data')
        c.edge('combined_data', 'processed_data')
        c.edge('processed_data', 'feature_data')
    
    # Model Training Nodes
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Model Training & Deployment')
        c.attr('node', fillcolor='lightyellow')
        c.node('model_training', 'Model Training\n(training.py)')
        c.node('xgboost_model', 'XGBoost Model\n(xgboost_model.joblib)')
        c.node('scaler', 'Feature Scaler\n(scaler.joblib)')
        c.node('feature_names', 'Feature Names\n(feature_names.joblib)')
        c.node('model_metrics', 'Model Metrics\n(training_metrics.json)')
        
        # Model flow
        c.edge('feature_data', 'model_training')
        c.edge('model_training', 'xgboost_model')
        c.edge('model_training', 'scaler')
        c.edge('model_training', 'feature_names')
        c.edge('model_training', 'model_metrics')
    
    # API & Frontend Nodes
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='API & Frontend')
        c.attr('node', fillcolor='lightpink')
        c.node('api', 'FastAPI\n(api/main.py)')
        c.node('streamlit', 'Streamlit Frontend\n(streamlit/app.py)')
        c.node('prediction_store', 'Prediction Store\n(prediction_store.py)')
        
        # API flow
        c.edge('xgboost_model', 'api')
        c.edge('scaler', 'api')
        c.edge('feature_names', 'api')
        c.edge('api', 'streamlit')
        c.edge('api', 'prediction_store')
    
    # Monitoring Nodes
    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='Monitoring')
        c.attr('node', fillcolor='lightgray')
        c.node('drift_detector', 'Drift Detection\n(drift_detector.py)')
        c.node('evaluation', 'Model Evaluation\n(evaluation.py)')
        c.node('scheduler', 'Model Update Scheduler\n(scheduler.py)')
        
        # Monitoring flow
        c.edge('feature_data', 'drift_detector')
        c.edge('xgboost_model', 'evaluation')
        c.edge('drift_detector', 'scheduler')
        c.edge('evaluation', 'scheduler')
        c.edge('scheduler', 'model_training', style='dashed')
        c.edge('prediction_store', 'evaluation')
    
    # Visualization Nodes
    with dot.subgraph(name='cluster_4') as c:
        c.attr(label='Visualization')
        c.attr('node', fillcolor='lightblue')
        c.node('pipeline_viz', 'Pipeline Visualization\n(pipeline_viz.py)')
        c.node('performance_viz', 'Performance Visualization\n(performance_viz.py)')
        
        # Visualization flow
        c.edge('model_metrics', 'performance_viz')
        c.edge('evaluation', 'performance_viz')
    
    # Save visualization
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    dot.render(str(output_dir / 'pipeline_architecture'), format='png', cleanup=True)
    print(f"Visualization saved in {output_dir}/pipeline_architecture.png")

if __name__ == '__main__':
    create_pipeline_visualization() 