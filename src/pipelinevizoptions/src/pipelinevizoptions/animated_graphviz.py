import graphviz
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
import os
import time
import numpy as np

def create_animated_pipeline():
    """Opretter en animeret version af pipeline arkitekturen med Graphviz"""
    
    # Opret en ny digraph for hvert step
    def create_graph(active_step, dot_position=0):
        dot = graphviz.Digraph(comment='MLOps Pipeline Architecture')
        dot.attr(rankdir='LR')  # Left to right layout
        
        # Tilføj styling
        dot.attr('node', shape='box', style='rounded,filled')
        
        # Data Pipeline Nodes
        with dot.subgraph(name='cluster_0') as c:
            c.attr(label='Data Pipeline')
            
            # Node farver baseret på active step
            data_color = 'lightgreen' if active_step >= 0 else 'lightgray'
            preproc_color = 'lightgreen' if active_step >= 1 else 'lightgray'
            feature_color = 'lightgreen' if active_step >= 2 else 'lightgray'
            
            # Edge farver baseret på active step
            data_edge_color = 'green' if active_step >= 0 else 'gray50'
            preproc_edge_color = 'green' if active_step >= 1 else 'gray50'
            feature_edge_color = 'green' if active_step >= 2 else 'gray50'
            
            c.node('vestas_data', 'Vestas Stock Data\n(VWSB.DEX)', fillcolor=data_color)
            c.node('macro_data', 'Macroeconomic Data\n(Alpha Vantage API)', fillcolor=data_color)
            c.node('combined_data', 'Combined Data\n(combined_data_processor.py)', fillcolor=preproc_color)
            c.node('processed_data', 'Processed Data\n(preprocessing.py)', fillcolor=preproc_color)
            c.node('feature_data', 'Feature Data\n(feature_engineering.py)', fillcolor=feature_color)
            
            # Data flow edges med farver og moving dots
            if active_step == 0:
                c.edge('vestas_data', 'combined_data', color=data_edge_color, 
                      label='●' if dot_position < 0.5 else '')
                c.edge('macro_data', 'combined_data', color=data_edge_color,
                      label='●' if dot_position >= 0.5 else '')
            else:
                c.edge('vestas_data', 'combined_data', color=data_edge_color)
                c.edge('macro_data', 'combined_data', color=data_edge_color)
            
            if active_step == 1:
                c.edge('combined_data', 'processed_data', color=preproc_edge_color,
                      label='●' if dot_position < 0.5 else '')
            else:
                c.edge('combined_data', 'processed_data', color=preproc_edge_color)
            
            if active_step == 2:
                c.edge('processed_data', 'feature_data', color=feature_edge_color,
                      label='●' if dot_position < 0.5 else '')
            else:
                c.edge('processed_data', 'feature_data', color=feature_edge_color)
        
        # Model Training Nodes
        with dot.subgraph(name='cluster_1') as c:
            c.attr(label='Model Training & Deployment')
            
            # Node farver baseret på active step
            model_color = 'lightyellow' if active_step >= 3 else 'lightgray'
            model_edge_color = 'orange' if active_step >= 3 else 'gray50'
            
            c.node('model_training', 'LSTM Model Training\n(lstm_model.py)', fillcolor=model_color)
            c.node('lstm_model', 'LSTM Model\n(lstm_multi_horizon_model.keras)', fillcolor=model_color)
            c.node('feature_scaler', 'Feature Scaler\n(lstm_feature_scaler.joblib)', fillcolor=model_color)
            c.node('target_scalers', 'Target Scalers\n(lstm_target_scalers.joblib)', fillcolor=model_color)
            c.node('model_metrics', 'Model Metrics\n(seq2seq_evaluation_results.csv)', fillcolor=model_color)
            
            # Model flow edges med farver og moving dots
            if active_step == 3:
                c.edge('feature_data', 'model_training', color=model_edge_color,
                      label='●' if dot_position < 0.5 else '')
            else:
                c.edge('feature_data', 'model_training', color=model_edge_color)
            
            if active_step == 4:
                c.edge('model_training', 'lstm_model', color=model_edge_color,
                      label='●' if dot_position < 0.25 else '')
                c.edge('model_training', 'feature_scaler', color=model_edge_color,
                      label='●' if 0.25 <= dot_position < 0.5 else '')
                c.edge('model_training', 'target_scalers', color=model_edge_color,
                      label='●' if 0.5 <= dot_position < 0.75 else '')
                c.edge('model_training', 'model_metrics', color=model_edge_color,
                      label='●' if dot_position >= 0.75 else '')
            else:
                c.edge('model_training', 'lstm_model', color=model_edge_color)
                c.edge('model_training', 'feature_scaler', color=model_edge_color)
                c.edge('model_training', 'target_scalers', color=model_edge_color)
                c.edge('model_training', 'model_metrics', color=model_edge_color)
        
        # API & Frontend Nodes
        with dot.subgraph(name='cluster_2') as c:
            c.attr(label='API & Frontend')
            
            # Node farver baseret på active step
            api_color = 'lightpink' if active_step >= 5 else 'lightgray'
            api_edge_color = 'red' if active_step >= 5 else 'gray50'
            
            c.node('api', 'FastAPI\n(stock_api.py)', fillcolor=api_color)
            c.node('streamlit', 'Streamlit Frontend\n(streamlit/app.py)', fillcolor=api_color)
            c.node('scheduler', 'Model Update Scheduler\n(scheduler.py)', fillcolor=api_color)
            c.node('metrics_viz', 'Model Metrics Viz\n(model_metrics_viz.py)', fillcolor=api_color)
            
            # API flow edges med farver og moving dots
            if active_step == 5:
                c.edge('lstm_model', 'api', color=api_edge_color,
                      label='●' if dot_position < 0.33 else '')
                c.edge('feature_scaler', 'api', color=api_edge_color,
                      label='●' if 0.33 <= dot_position < 0.66 else '')
                c.edge('target_scalers', 'api', color=api_edge_color,
                      label='●' if dot_position >= 0.66 else '')
            else:
                c.edge('lstm_model', 'api', color=api_edge_color)
                c.edge('feature_scaler', 'api', color=api_edge_color)
                c.edge('target_scalers', 'api', color=api_edge_color)
            
            if active_step == 6:
                c.edge('api', 'streamlit', color=api_edge_color,
                      label='●' if dot_position < 0.33 else '')
                c.edge('model_metrics', 'metrics_viz', color=api_edge_color,
                      label='●' if 0.33 <= dot_position < 0.66 else '')
                c.edge('metrics_viz', 'streamlit', color=api_edge_color,
                      label='●' if dot_position >= 0.66 else '')
            else:
                c.edge('api', 'streamlit', color=api_edge_color)
                c.edge('model_metrics', 'metrics_viz', color=api_edge_color)
                c.edge('metrics_viz', 'streamlit', color=api_edge_color)
            
            c.edge('scheduler', 'model_training', style='dashed', color=api_edge_color)
        
        return dot
    
    # Opret output directory
    output_dir = Path('src/pipelinevizoptions/animations/frames')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generer frames for hvert step og dot position
    frames = []
    steps = 7  # Antal steps i pipeline
    dot_positions = np.linspace(0, 1, 10)  # 10 positioner for dot animation
    
    for step in range(steps):
        for dot_pos in dot_positions:
            dot = create_graph(step, dot_pos)
            frame_path = output_dir / f'frame_{step}_{dot_pos:.2f}.png'
            dot.render(str(frame_path).replace('.png', ''), format='png', cleanup=True)
            frames.append(str(frame_path))
    
    # Opret animation med matplotlib
    fig, ax = plt.subplots(figsize=(15, 10))
    
    def update(frame):
        ax.clear()
        img = mpimg.imread(frames[frame])
        ax.imshow(img)
        ax.axis('off')
    
    # Opret animation
    ani = FuncAnimation(fig, update,
                       frames=len(frames),
                       interval=100,  # Hurtigere animation
                       repeat=True)
    
    # Gem animation
    ani.save('src/pipelinevizoptions/animations/pipeline_flow.gif',
             writer='pillow',
             fps=10)  # 10 FPS for smooth animation
    
    # Ryd op
    plt.close()
    for frame in frames:
        if os.path.exists(frame):
            os.remove(frame)

if __name__ == '__main__':
    create_animated_pipeline() 