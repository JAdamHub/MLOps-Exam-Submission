import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime

def load_metrics():
    """Indlæser metrics fra JSON filen"""
    metrics_file = Path("src/monitoring/model_metrics.json")
    if not metrics_file.exists():
        raise FileNotFoundError("Metrics fil ikke fundet")
    
    with open(metrics_file, 'r') as f:
        return json.load(f)

def create_accuracy_plot():
    """Opretter plot af accuracy over tid for både LSTM og XGBoost modeller"""
    metrics = load_metrics()
    
    # Konverter accuracy historik til DataFrame
    df_lstm = pd.DataFrame(metrics['lstm_accuracy_history'])
    df_xgboost = pd.DataFrame(metrics['xgboost_accuracy_history'])
    
    df_lstm['date'] = pd.to_datetime(df_lstm['date'])
    df_xgboost['date'] = pd.to_datetime(df_xgboost['date'])
    
    # Opret plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Plot accuracy linjer for begge modeller
    sns.lineplot(data=df_lstm, x='date', y='accuracy', marker='o', label='LSTM Model')
    sns.lineplot(data=df_xgboost, x='date', y='accuracy', marker='s', label='XGBoost Model')
    
    # Tilføj labels og titel
    plt.title('Model Accuracy Over Tid', fontsize=14, pad=20)
    plt.xlabel('Dato', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    
    # Roter x-aksis labels for bedre læsbarhed
    plt.xticks(rotation=45)
    
    # Tilføj grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Tilføj sidste accuracy værdier som tekst
    last_lstm_accuracy = df_lstm['accuracy'].iloc[-1]
    last_xgboost_accuracy = df_xgboost['accuracy'].iloc[-1]
    
    plt.annotate(f'LSTM: {last_lstm_accuracy:.4f}',
                xy=(df_lstm['date'].iloc[-1], last_lstm_accuracy),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    
    plt.annotate(f'XGBoost: {last_xgboost_accuracy:.4f}',
                xy=(df_xgboost['date'].iloc[-1], last_xgboost_accuracy),
                xytext=(10, -10), textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    
    # Tilføj legend
    plt.legend()
    
    # Tilpas layout
    plt.tight_layout()
    
    # Gem plot
    output_dir = Path("src/monitoring/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'accuracy_over_time.png')
    plt.close()

def create_version_comparison_plot():
    """Opretter plot der sammenligner model versioner for både LSTM og XGBoost"""
    metrics = load_metrics()
    
    # Konverter version data til DataFrame
    df_lstm = pd.DataFrame(metrics['lstm_model_versions'])
    df_xgboost = pd.DataFrame(metrics['xgboost_model_versions'])
    
    df_lstm['date'] = pd.to_datetime(df_lstm['date'])
    df_xgboost['date'] = pd.to_datetime(df_xgboost['date'])
    
    # Opret plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Plot version accuracy for begge modeller
    x = range(len(df_lstm))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], df_lstm['accuracy'], width, label='LSTM')
    plt.bar([i + width/2 for i in x], df_xgboost['accuracy'], width, label='XGBoost')
    
    # Tilføj labels og titel
    plt.title('Model Version Accuracy Sammenligning', fontsize=14, pad=20)
    plt.xlabel('Model Version', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    
    # Tilføj accuracy værdier over bars
    for i, v in enumerate(df_lstm['accuracy']):
        plt.text(i - width/2, v, f'{v:.4f}', ha='center', va='bottom')
    for i, v in enumerate(df_xgboost['accuracy']):
        plt.text(i + width/2, v, f'{v:.4f}', ha='center', va='bottom')
    
    # Tilføj grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Tilføj legend
    plt.legend()
    
    # Tilpas layout
    plt.tight_layout()
    
    # Gem plot
    output_dir = Path("src/monitoring/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'version_comparison.png')
    plt.close()

def main():
    """Hovedfunktion til at generere alle visualiseringer"""
    try:
        create_accuracy_plot()
        create_version_comparison_plot()
        print("Visualiseringer genereret succesfuldt i 'src/monitoring/figures/'")
    except Exception as e:
        print(f"Fejl under generering af visualiseringer: {e}")

if __name__ == "__main__":
    main() 