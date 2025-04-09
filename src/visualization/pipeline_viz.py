import graphviz
from pathlib import Path

def create_pipeline_visualization():
    """Opretter en visualisering af pipeline arkitekturen."""
    
    # Opret en ny digraph
    dot = graphviz.Digraph(comment='MLOps Pipeline Arkitektur')
    dot.attr(rankdir='LR')  # Left to right layout
    
    # Tilføj styling
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    dot.attr('edge', color='gray50')
    
    # Data Processing Nodes
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Data Processing Pipeline')
        c.attr('node', fillcolor='lightgreen')
        c.node('raw_data', 'Rå Data\n(bitcoin_usd_365d_raw.csv)')
        c.node('macro_data', 'Makroøkonomisk Data\n(FRED API)')
        c.node('combined_data', 'Kombineret Data\n(combined_data_processor.py)')
        c.node('processed_data', 'Forarbejdet Data\n(preprocessing.py)')
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
        c.node('model_training', 'Model Træning\n(training.py)')
        c.node('xgboost_model', 'XGBoost Model\n(xgboost_model.joblib)')
        c.node('scaler', 'Feature Scaler\n(scaler.joblib)')
        c.node('feature_names', 'Feature Names\n(feature_names.joblib)')
        
        # Model flow
        c.edge('feature_data', 'model_training')
        c.edge('model_training', 'xgboost_model')
        c.edge('model_training', 'scaler')
        c.edge('model_training', 'feature_names')
    
    # API & Frontend Nodes
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='API & Frontend')
        c.attr('node', fillcolor='lightpink')
        c.node('api', 'FastAPI\n(api/main.py)')
        c.node('streamlit', 'Streamlit Frontend\n(streamlit/app.py)')
        
        # API flow
        c.edge('xgboost_model', 'api')
        c.edge('scaler', 'api')
        c.edge('feature_names', 'api')
        c.edge('api', 'streamlit')
    
    # Monitoring Nodes
    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='Monitoring')
        c.attr('node', fillcolor='lightgray')
        c.node('drift_detector', 'Drift Detection\n(drift_detector.py)')
        c.node('evaluation', 'Model Evaluation\n(evaluation.py)')
        
        # Monitoring flow
        c.edge('feature_data', 'drift_detector')
        c.edge('xgboost_model', 'evaluation')
    
    # Gem visualiseringen
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    dot.render(str(output_dir / 'pipeline_architecture'), format='png', cleanup=True)
    print(f"Visualisering gemt i {output_dir}/pipeline_architecture.png")

if __name__ == '__main__':
    create_pipeline_visualization() 