import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelResultsVisualizer:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.results_files = self._get_results_files()
        self.all_results = self._load_all_results()
        
    def _get_results_files(self):
        """Find all result CSV files in the results directory"""
        pattern = os.path.join(self.results_dir, 'seq2seq_evaluation_results_*.csv')
        files = glob.glob(pattern)
        files.sort()  # Sort by filename (which includes timestamp)
        logger.info(f"Found {len(files)} results files")
        return files
    
    def _load_all_results(self):
        """Load all results into a single DataFrame with version information"""
        all_results = []
        
        for file_path in self.results_files:
            try:
                # Extract timestamp from filename
                filename = os.path.basename(file_path)
                timestamp_str = filename.replace('seq2seq_evaluation_results_', '').replace('.csv', '')
                
                # Parse timestamp
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                except ValueError:
                    timestamp = None
                
                # Load results
                df = pd.read_csv(file_path)
                
                # Add metadata
                df['timestamp'] = timestamp
                df['version'] = timestamp_str
                df['file'] = file_path
                
                all_results.append(df)
                
            except Exception as e:
                logger.error(f"Error loading results file {file_path}: {str(e)}")
                
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            logger.info(f"Loaded {len(combined_results)} result entries from {len(all_results)} files")
            return combined_results
        else:
            logger.warning("No results loaded")
            return pd.DataFrame()
    
    def plot_metric_history(self, metric='R2', save_path=None):
        """Plot history of a specific metric for all horizons"""
        if self.all_results.empty:
            logger.error("No results available for plotting")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Group by horizon and timestamp, then plot
        for horizon in self.all_results['Horizon'].unique():
            horizon_data = self.all_results[self.all_results['Horizon'] == horizon]
            if not horizon_data.empty:
                plt.plot(
                    horizon_data['timestamp'], 
                    horizon_data[metric], 
                    marker='o', 
                    linestyle='-', 
                    label=f'Horizon {horizon}'
                )
        
        plt.title(f'LSTM Model {metric} History')
        plt.xlabel('Timestamp')
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend()
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Metric history plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_horizon_comparison(self, metric='R2', latest_only=False, save_path=None):
        """Plot comparison of horizons for a specific metric"""
        if self.all_results.empty:
            logger.error("No results available for plotting")
            return
        
        # Filter for latest version only if requested
        df = self.all_results
        if latest_only and not df.empty and 'timestamp' in df.columns:
            latest_timestamp = df['timestamp'].max()
            df = df[df['timestamp'] == latest_timestamp]
            title_suffix = "(Latest Version)"
        else:
            title_suffix = "(All Versions)"
        
        plt.figure(figsize=(12, 6))
        
        # Create barplot
        sns.barplot(data=df, x='Horizon', y=metric, palette='viridis')
        
        plt.title(f'LSTM Model {metric} by Horizon {title_suffix}')
        plt.xlabel('Forecast Horizon')
        plt.ylabel(metric)
        plt.grid(True, axis='y')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Horizon comparison plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_metrics_radar(self, latest_only=True, save_path=None):
        """Create a radar chart comparing different metrics across horizons"""
        if self.all_results.empty:
            logger.error("No results available for plotting")
            return
        
        # Filter for latest version only if requested
        df = self.all_results
        if latest_only and not df.empty and 'timestamp' in df.columns:
            latest_timestamp = df['timestamp'].max()
            df = df[df['timestamp'] == latest_timestamp]
            title_suffix = "(Latest Version)"
        else:
            title_suffix = "(All Versions)"
        
        # Get available metrics (excluding non-metric columns)
        metrics = [col for col in df.columns if col not in ['Horizon', 'timestamp', 'version', 'file']]
        
        # If MSE and RMSE are too large compared to other metrics, we can normalize
        df_radar = df.copy()
        for metric in metrics:
            if metric in ['MSE', 'RMSE']:
                # Apply inverse to make lower values "better" on the radar chart
                max_val = df_radar[metric].max()
                df_radar[f'{metric}_inv'] = 1 - (df_radar[metric] / max_val)
                metrics[metrics.index(metric)] = f'{metric}_inv'
        
        # Create radar chart for each horizon
        from matplotlib.path import Path
        from matplotlib.spines import Spine
        from matplotlib.projections.polar import PolarAxes
        from matplotlib.projections import register_projection
        
        def radar_factory(num_vars, frame='polygon'):
            theta = 2*3.1415926 * np.linspace(0, 1-1./num_vars, num_vars)
            
            class RadarAxes(PolarAxes):
                name = 'radar'
                
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.set_theta_zero_location('N')
                
                def fill(self, *args, **kwargs):
                    return super().fill_between(*args, **kwargs)
                
                def plot(self, *args, **kwargs):
                    lines = super().plot(*args, **kwargs)
                    self._close_polygon(lines[0])
                    return lines
                
                def _close_polygon(self, line):
                    x, y = line.get_data()
                    if x[0] != x[-1]:
                        x = np.concatenate((x, [x[0]]))
                        y = np.concatenate((y, [y[0]]))
                        line.set_data(x, y)
                
                def set_varlabels(self, labels):
                    self.set_thetagrids(np.degrees(theta), labels)
                
                def _gen_axes_patch(self):
                    if frame == 'circle':
                        return Circle((0.5, 0.5), 0.5)
                    elif frame == 'polygon':
                        return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, orientation=np.pi/2)
                    else:
                        raise ValueError("unknown value for 'frame': %s" % frame)
                
                def draw(self, renderer):
                    if frame == 'circle':
                        patch = Circle((0.5, 0.5), 0.5)
                        patch.set_transform(self.transAxes)
                        patch.set_clip_path(self.patch)
                        patch.set_facecolor('white')
                        self.add_patch(patch)
                    elif frame == 'polygon':
                        patch = RegularPolygon((0.5, 0.5), num_vars, radius=0.5, orientation=np.pi/2)
                        patch.set_transform(self.transAxes)
                        patch.set_clip_path(self.patch)
                        patch.set_facecolor('white')
                        self.add_patch(patch)
                    else:
                        raise ValueError("unknown value for 'frame': %s" % frame)
                    super().draw(renderer)
                    
            register_projection(RadarAxes)
            return theta

        # We need numpy for the radar chart
        import numpy as np
        from matplotlib.patches import Circle, RegularPolygon
        
        # Create the radar chart
        num_metrics = len(metrics)
        theta = radar_factory(num_metrics)
        
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        
        for i, horizon in enumerate(df_radar['Horizon'].unique()):
            color = colors[i % len(colors)]
            horizon_data = df_radar[df_radar['Horizon'] == horizon]
            
            values = [horizon_data[metric].iloc[0] for metric in metrics]
            ax.plot(theta, values, color=color, label=f'Horizon {horizon}')
            ax.fill(theta, values, color=color, alpha=0.25)
        
        ax.set_varlabels(metrics)
        plt.title(f'LSTM Model Metrics Comparison {title_suffix}')
        plt.legend(loc='upper right')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Radar plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def generate_report(self, save_dir='reports'):
        """Generate comprehensive visualization report"""
        try:
            # Create reports directory if it doesn't exist
            report_dir = Path(save_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate plots
            self.plot_metric_history('R2', f"{save_dir}/lstm_r2_history_{timestamp}.png")
            self.plot_metric_history('RMSE', f"{save_dir}/lstm_rmse_history_{timestamp}.png")
            self.plot_horizon_comparison('R2', True, f"{save_dir}/lstm_r2_comparison_{timestamp}.png")
            self.plot_metrics_radar(True, f"{save_dir}/lstm_metrics_radar_{timestamp}.png")
            
            # Generate summary table
            if not self.all_results.empty:
                latest_results = self.all_results.copy()
                if 'timestamp' in latest_results.columns:
                    latest_timestamp = latest_results['timestamp'].max()
                    latest_results = latest_results[latest_results['timestamp'] == latest_timestamp]
                
                # Save summary to CSV
                latest_results.to_csv(f"{save_dir}/lstm_latest_metrics_{timestamp}.csv", index=False)
                
                logger.info(f"Latest metrics:\n{latest_results[['Horizon', 'RMSE', 'MAE', 'R2']]}")
            
            logger.info(f"Report generated successfully in {save_dir}")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}", exc_info=True)

# Tilføj den manglende main() funktion, der bliver kaldt fra main.py
def main():
    """Entry point for the pipeline integration"""
    logger.info("Starting model metrics visualization...")
    try:
        visualizer = ModelResultsVisualizer()
        visualizer.generate_report()
        logger.info("Model metrics visualization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in model metrics visualization: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    visualizer = ModelResultsVisualizer()
    visualizer.generate_report()
