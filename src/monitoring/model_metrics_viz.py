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
    """Opretter plot af accuracy over tid"""
    metrics = load_metrics()
    
    # Konverter accuracy historik til DataFrame
    df = pd.DataFrame(metrics['accuracy_history'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Opret plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Plot accuracy linje
    sns.lineplot(data=df, x='date', y='accuracy', marker='o')
    
    # Tilføj labels og titel
    plt.title('Model Accuracy Over Tid', fontsize=14, pad=20)
    plt.xlabel('Dato', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    
    # Roter x-aksis labels for bedre læsbarhed
    plt.xticks(rotation=45)
    
    # Tilføj grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Tilføj sidste accuracy værdi som tekst
    last_accuracy = df['accuracy'].iloc[-1]
    plt.annotate(f'Sidste accuracy: {last_accuracy:.4f}',
                xy=(df['date'].iloc[-1], last_accuracy),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    
    # Tilpas layout
    plt.tight_layout()
    
    # Gem plot
    output_dir = Path("src/monitoring/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'accuracy_over_time.png')
    plt.close()

def create_version_comparison_plot():
    """Opretter plot der sammenligner model versioner"""
    metrics = load_metrics()
    
    # Konverter version data til DataFrame
    df = pd.DataFrame(metrics['model_versions'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Opret plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Plot version accuracy
    sns.barplot(data=df, x='version', y='accuracy')
    
    # Tilføj labels og titel
    plt.title('Model Version Accuracy Sammenligning', fontsize=14, pad=20)
    plt.xlabel('Model Version', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    
    # Tilføj accuracy værdier over bars
    for i, v in enumerate(df['accuracy']):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # Tilføj grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
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