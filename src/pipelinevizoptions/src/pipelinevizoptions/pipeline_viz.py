import graphviz
from pathlib import Path

def create_pipeline_visualization():
    """Opretter en visualisering af pipeline arkitekturen med Graphviz"""
    
    # Opret en ny digraph
    dot = graphviz.Digraph(comment='MLOps Pipeline Architecture')
    dot.attr(rankdir='LR')  # Left to right layout
    
    # Tilføj styling
    dot.attr('node', shape='box', style='rounded,filled')
    
    # Data Pipeline Nodes
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Data Pipeline')
        c.node('vestas_data', 'Vestas Stock Data\n(VWSB.DEX)', fillcolor='lightgreen')
        c.node('macro_data', 'Macroeconomic Data\n(Alpha Vantage API)', fillcolor='lightgreen')
        c.node('combined_data', 'Combined Data\n(combined_data_processor.py)', fillcolor='lightgreen')
        c.node('processed_data', 'Processed Data\n(preprocessing.py)', fillcolor='lightgreen')
        c.node('feature_data', 'Feature Data\n(feature_engineering.py)', fillcolor='lightgreen')
        
        # Data flow edges
        c.edge('vestas_data', 'combined_data', color='green')
        c.edge('macro_data', 'combined_data', color='green')
        c.edge('combined_data', 'processed_data', color='green')
        c.edge('processed_data', 'feature_data', color='green')
    
    # Model Training Nodes
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Model Training & Deployment')
        c.node('model_training', 'LSTM Model Training\n(lstm_model.py)', fillcolor='lightyellow')
        c.node('lstm_model', 'LSTM Model\n(lstm_multi_horizon_model.keras)', fillcolor='lightyellow')
        c.node('feature_scaler', 'Feature Scaler\n(lstm_feature_scaler.joblib)', fillcolor='lightyellow')
        c.node('target_scalers', 'Target Scalers\n(lstm_target_scalers.joblib)', fillcolor='lightyellow')
        c.node('model_metrics', 'Model Metrics\n(seq2seq_evaluation_results.csv)', fillcolor='lightyellow')
        
        # Model flow edges
        c.edge('feature_data', 'model_training', color='orange')
        c.edge('model_training', 'lstm_model', color='orange')
        c.edge('model_training', 'feature_scaler', color='orange')
        c.edge('model_training', 'target_scalers', color='orange')
        c.edge('model_training', 'model_metrics', color='orange')
    
    # API & Frontend Nodes
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='API & Frontend')
        c.node('api', 'FastAPI\n(stock_api.py)', fillcolor='lightpink')
        c.node('streamlit', 'Streamlit Frontend\n(streamlit/app.py)', fillcolor='lightpink')
        c.node('scheduler', 'Model Update Scheduler\n(scheduler.py)', fillcolor='lightpink')
        c.node('metrics_viz', 'Model Metrics Viz\n(model_metrics_viz.py)', fillcolor='lightpink')
        
        # API flow edges
        c.edge('lstm_model', 'api', color='red')
        c.edge('feature_scaler', 'api', color='red')
        c.edge('target_scalers', 'api', color='red')
        c.edge('api', 'streamlit', color='red')
        c.edge('model_metrics', 'metrics_viz', color='red')
        c.edge('metrics_viz', 'streamlit', color='red')
        c.edge('scheduler', 'model_training', style='dashed', color='red')
    
    # Opret output directory
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # Gem visualisering
    dot.render(str(output_dir / 'pipeline_architecture'), format='png', cleanup=True)
    print(f"Visualization saved in {output_dir}/pipeline_architecture.png")

if __name__ == '__main__':
    create_pipeline_visualization() 