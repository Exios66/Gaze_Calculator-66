#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaze Data Analysis Pipeline

This script provides a comprehensive pipeline for processing, analyzing, 
and visualizing eye gaze data from experiments.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Use try-except blocks for plotly imports to handle potential import errors
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: Plotly not fully available. Interactive visualizations will be disabled.")
    PLOTLY_AVAILABLE = False
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
from tqdm import tqdm
import statsmodels.api as sm
import json
from datetime import datetime

class GazeDataAnalyzer:
    """
    A class for analyzing eye-tracking gaze data.
    Handles data loading, preprocessing, visualization, and analysis.
    """
    
    def __init__(self, raw_data_dir="raw data", processed_data_dir="processed_data", output_dir="results"):
        """
        Initialize the GazeDataAnalyzer with directory paths.
        
        Parameters:
        -----------
        raw_data_dir : str
            Directory containing raw gaze data files
        processed_data_dir : str
            Directory for storing processed data
        output_dir : str
            Directory for storing results and visualizations
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.output_dir = output_dir
        
        # Create directories if they don't exist
        for directory in [processed_data_dir, output_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize data containers
        self.raw_data = {}
        self.processed_data = {}
        self.metrics = {}
        
        # Configuration parameters
        self.config = {
            'fixation_threshold': 50,  # pixel distance
            'min_fixation_duration': 100,  # ms
            'smoothing_window': 3,  # data points 
            'outlier_threshold': 3,  # standard deviations
            'heatmap_resolution': 100,  # grid size
        }
    
    def load_experimental_data(self, file_path=None, participant_id=None):
        """
        Load experimental gaze data from a CSV file.
        
        Parameters:
        -----------
        file_path : str, optional
            Path to the CSV file. If None, will try to find files for participant_id.
        participant_id : str, optional
            Participant ID to search for matching files.
            
        Returns:
        --------
        pd.DataFrame
            Loaded gaze data
        """
        if file_path is None and participant_id is None:
            raise ValueError("Either file_path or participant_id must be provided")
            
        if file_path is None:
            # Find files for this participant
            for root, _, files in os.walk(self.raw_data_dir):
                for file in files:
                    if file.startswith(participant_id) and file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        break
        
        if file_path is None or not os.path.exists(file_path):
            raise FileNotFoundError(f"No data file found for participant {participant_id}")
            
        print(f"Loading experimental data from {file_path}")
        data = pd.read_csv(file_path)
        
        # Extract participant ID from filename if not provided
        if participant_id is None:
            filename = os.path.basename(file_path)
            participant_id = filename.split('_')[0]
        
        # Store the data
        self.raw_data[participant_id] = data
        
        return data
    
    def load_fallback_data(self, participant_id=None):
        """
        Load fallback gaze data with simple x, y, timestamp format.
        
        Parameters:
        -----------
        participant_id : str, optional
            Participant ID to associate with this data.
            
        Returns:
        --------
        pd.DataFrame
            Loaded fallback gaze data
        """
        fallback_dir = os.path.join(self.raw_data_dir, "fallback-data")
        
        if not os.path.exists(fallback_dir):
            raise FileNotFoundError(f"Fallback data directory not found: {fallback_dir}")
            
        # List all fallback data files
        fallback_files = [f for f in os.listdir(fallback_dir) if f.endswith('.csv')]
        
        if not fallback_files:
            raise FileNotFoundError(f"No fallback data files found in {fallback_dir}")
        
        # Use the first file (or we could implement selection logic here)
        file_path = os.path.join(fallback_dir, fallback_files[0])
        
        print(f"Loading fallback data from {file_path}")
        data = pd.read_csv(file_path)
        
        # If no participant ID provided, use a default
        if participant_id is None:
            participant_id = "fallback_participant"
        
        # Store the data
        self.raw_data[participant_id] = data
        
        return data
    
    def preprocess_data(self, participant_id):
        """
        Preprocess raw gaze data for analysis.
        
        Parameters:
        -----------
        participant_id : str
            Participant ID whose data to preprocess
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed gaze data
        """
        if participant_id not in self.raw_data:
            raise ValueError(f"No data loaded for participant {participant_id}")
            
        data = self.raw_data[participant_id]
        
        # Check data format (experimental vs. fallback)
        if 'timestamp' in data.columns and 'x' in data.columns and 'y' in data.columns:
            # Simple fallback format
            processed = self._preprocess_fallback_data(data)
        else:
            # Experimental format
            processed = self._preprocess_experimental_data(data)
        
        # Store processed data
        self.processed_data[participant_id] = processed
        
        # Save processed data to file
        output_file = os.path.join(self.processed_data_dir, f"{participant_id}_processed.csv")
        processed.to_csv(output_file, index=False)
        print(f"Preprocessed data saved to {output_file}")
        
        return processed
    
    def _preprocess_fallback_data(self, data):
        """Process simple x,y,timestamp format data"""
        # Make a copy to avoid modifying the original
        processed = data.copy()
        
        # Convert timestamp to datetime if needed
        if isinstance(processed['timestamp'].iloc[0], (int, float)):
            # Assume milliseconds since epoch
            processed['datetime'] = pd.to_datetime(processed['timestamp'], unit='ms')
        
        # Calculate time difference between points (in milliseconds)
        processed['time_diff'] = processed['timestamp'].diff().fillna(0)
        
        # Calculate distance between consecutive points
        processed['distance'] = np.sqrt(
            (processed['x'].diff().fillna(0))**2 + 
            (processed['y'].diff().fillna(0))**2
        )
        
        # Apply smoothing to x, y coordinates
        window = self.config['smoothing_window']
        processed['x_smooth'] = processed['x'].rolling(window=window, center=True).mean().fillna(processed['x'])
        processed['y_smooth'] = processed['y'].rolling(window=window, center=True).mean().fillna(processed['y'])
        
        # Remove outliers
        z_scores_x = np.abs((processed['x'] - processed['x'].mean()) / processed['x'].std())
        z_scores_y = np.abs((processed['y'] - processed['y'].mean()) / processed['y'].std())
        threshold = self.config['outlier_threshold']
        processed['is_outlier'] = (z_scores_x > threshold) | (z_scores_y > threshold)
        
        # Identify fixations (points where distance moved is below threshold)
        processed['is_fixation'] = processed['distance'] < self.config['fixation_threshold']
        
        # Group consecutive fixation points
        processed['fixation_group'] = (processed['is_fixation'] != processed['is_fixation'].shift()).cumsum()
        
        return processed
    
    def _preprocess_experimental_data(self, data):
        """Process data from experimental format with more complex columns"""
        # Extract the key gaze coordinates based on column names
        # This might need customization based on the exact format
        
        # Identify columns containing x and y coordinates
        x_cols = [col for col in data.columns if 'x' in col.lower() and not 'calibration' in col.lower()]
        y_cols = [col for col in data.columns if 'y' in col.lower() and not 'calibration' in col.lower()]
        
        # If we have clear gaze data columns, use those
        if 'Calibrate_X' in data.columns and 'Calibrate_Y' in data.columns:
            processed = pd.DataFrame({
                'x': data['Calibrate_X'],
                'y': data['Calibrate_Y'],
                'trial': data['trials.thisN'],
                'duration': data['Duration'],
                'target_size': data['TargetSize']
            })
        # Otherwise try our best to extract gaze data
        elif x_cols and y_cols:
            # Use the first columns that look like they contain x,y data
            processed = pd.DataFrame({
                'x': data[x_cols[0]],
                'y': data[y_cols[0]],
                'trial': data.index
            })
        else:
            raise ValueError("Could not identify gaze coordinate columns in the data")
        
        # Add a timestamp column if not present
        if 'timestamp' not in processed.columns:
            processed['timestamp'] = np.arange(len(processed))
        
        # Apply the same processing as fallback data
        return self._preprocess_fallback_data(processed)
    
    def identify_fixations(self, participant_id):
        """
        Identify fixation points and periods in the gaze data.
        
        Parameters:
        -----------
        participant_id : str
            Participant ID whose data to analyze
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing fixation information
        """
        if participant_id not in self.processed_data:
            raise ValueError(f"No processed data for participant {participant_id}")
            
        data = self.processed_data[participant_id]
        
        # Filter for fixation points only
        fixation_data = data[data['is_fixation']]
        
        # Group by fixation_group
        fixation_groups = fixation_data.groupby('fixation_group')
        
        # Create a summary of each fixation
        fixations = []
        
        for group_id, group in fixation_groups:
            # Skip non-fixation groups
            if not group['is_fixation'].all():
                continue
                
            # Calculate fixation duration
            if 'time_diff' in group.columns:
                duration = group['time_diff'].sum()
            else:
                duration = len(group)  # Use point count as a proxy if no timing information
            
            # Skip fixations that are too short
            if duration < self.config['min_fixation_duration']:
                continue
                
            # Calculate mean position
            mean_x = group['x_smooth'].mean() if 'x_smooth' in group.columns else group['x'].mean()
            mean_y = group['y_smooth'].mean() if 'y_smooth' in group.columns else group['y'].mean()
            
            fixations.append({
                'fixation_id': group_id,
                'start_time': group['timestamp'].min(),
                'end_time': group['timestamp'].max(),
                'duration': duration,
                'mean_x': mean_x,
                'mean_y': mean_y,
                'std_x': group['x'].std(),
                'std_y': group['y'].std(),
                'point_count': len(group)
            })
        
        fixation_df = pd.DataFrame(fixations)
        
        # Save fixation data
        output_file = os.path.join(self.processed_data_dir, f"{participant_id}_fixations.csv")
        fixation_df.to_csv(output_file, index=False)
        print(f"Fixation data saved to {output_file}")
        
        return fixation_df
    
    def calculate_metrics(self, participant_id):
        """
        Calculate key eye-tracking metrics.
        
        Parameters:
        -----------
        participant_id : str
            Participant ID whose data to analyze
            
        Returns:
        --------
        dict
            Dictionary of calculated metrics
        """
        if participant_id not in self.processed_data:
            self.preprocess_data(participant_id)
            
        # Get fixations
        try:
            fixations = self.identify_fixations(participant_id)
        except ValueError:
            print(f"Warning: Could not identify fixations for participant {participant_id}")
            return {}
            
        # Get raw data
        data = self.processed_data[participant_id]
        
        # Calculate metrics
        metrics = {}
        
        # Basic metrics
        metrics['total_points'] = len(data)
        metrics['valid_points'] = len(data[~data['is_outlier']])
        metrics['outlier_percentage'] = (len(data[data['is_outlier']]) / len(data)) * 100
        
        # Fixation metrics
        if not fixations.empty:
            metrics['fixation_count'] = len(fixations)
            metrics['mean_fixation_duration'] = fixations['duration'].mean()
            metrics['total_fixation_time'] = fixations['duration'].sum()
            metrics['fixation_percentage'] = (metrics['total_fixation_time'] / 
                                             (data['timestamp'].max() - data['timestamp'].min())) * 100
            
            # Calculate spatial distribution
            x_range = data['x'].max() - data['x'].min()
            y_range = data['y'].max() - data['y'].min()
            metrics['spatial_density'] = metrics['fixation_count'] / (x_range * y_range)
        
        # Save metrics
        self.metrics[participant_id] = metrics
        
        # Save to file
        output_file = os.path.join(self.processed_data_dir, f"{participant_id}_metrics.json")
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics
    
    def plot_gaze_path(self, participant_id, output_file=None, show=True):
        """
        Plot the gaze path with fixations highlighted.
        
        Parameters:
        -----------
        participant_id : str
            Participant ID whose data to visualize
        output_file : str, optional
            Path to save the visualization
        show : bool, default=True
            Whether to display the plot
        
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if participant_id not in self.processed_data:
            self.preprocess_data(participant_id)
        
        data = self.processed_data[participant_id]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot gaze path
        ax.plot(data['x'], data['y'], 'b-', alpha=0.3, linewidth=0.5)
        
        # Plot points (non-fixation in blue, fixation in red)
        non_fixation = data[~data['is_fixation']]
        fixation = data[data['is_fixation']]
        
        ax.scatter(non_fixation['x'], non_fixation['y'], 
                  c='blue', alpha=0.2, s=10, label='Saccades')
        ax.scatter(fixation['x'], fixation['y'], 
                  c='red', alpha=0.5, s=20, label='Fixations')
        
        # Add labels and legend
        ax.set_title(f'Gaze Path for Participant {participant_id}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()
        
        # Keep axis equal for proper spatial representation
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(alpha=0.3)
        
        # Save if requested
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"{participant_id}_gaze_path.png")
            
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Gaze path plot saved to {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def plot_heatmap(self, participant_id, output_file=None, show=True):
        """
        Create a heatmap of gaze density.
        
        Parameters:
        -----------
        participant_id : str
            Participant ID whose data to visualize
        output_file : str, optional
            Path to save the visualization
        show : bool, default=True
            Whether to display the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if participant_id not in self.processed_data:
            self.preprocess_data(participant_id)
            
        data = self.processed_data[participant_id]
        
        # Filter out outliers
        filtered_data = data[~data['is_outlier']]
        
        # Create a 2D histogram
        x = filtered_data['x']
        y = filtered_data['y']
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        heatmap, xedges, yedges = np.histogram2d(
            x, y, bins=self.config['heatmap_resolution']
        )
        
        # Smooth the heatmap
        heatmap = gaussian_kde(np.vstack([x, y]))(np.vstack([
            np.repeat(np.linspace(x.min(), x.max(), 100), 100),
            np.tile(np.linspace(y.min(), y.max(), 100), 100)
        ]))
        
        heatmap = heatmap.reshape(100, 100)
        
        # Plot heatmap
        im = ax.imshow(
            heatmap.T,
            cmap='hot',
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin='lower',
            interpolation='gaussian'
        )
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Gaze Density')
        
        # Add labels
        ax.set_title(f'Gaze Heatmap for Participant {participant_id}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Keep aspect ratio
        ax.set_aspect('equal')
        
        # Save if requested
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"{participant_id}_heatmap.png")
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def generate_interactive_visualization(self, participant_id, output_file=None):
        """
        Generate an interactive visualization of gaze data using Plotly.
        
        Parameters:
        -----------
        participant_id : str
            Participant ID whose data to visualize
        output_file : str, optional
            Path to save the HTML visualization
            
        Returns:
        --------
        plotly.graph_objects.Figure or None
            The interactive figure if Plotly is available, None otherwise
        """
        if not PLOTLY_AVAILABLE:
            print("Interactive visualization not available: Plotly package not fully installed.")
            print("Install with: pip install plotly==5.15.0 --upgrade")
            return None
            
        if participant_id not in self.processed_data:
            self.preprocess_data(participant_id)
            
        data = self.processed_data[participant_id]
        
        # Create subplot with shared axes
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=("Gaze Path", "Fixation Duration"),
            horizontal_spacing=0.1
        )
        
        # Add gaze path trace
        fig.add_trace(
            go.Scatter(
                x=data['x'], 
                y=data['y'],
                mode='lines',
                line=dict(color='blue', width=1),
                opacity=0.3,
                name='Gaze Path'
            ),
            row=1, col=1
        )
        
        # Add fixation points
        fixation_data = data[data['is_fixation']]
        fig.add_trace(
            go.Scatter(
                x=fixation_data['x'],
                y=fixation_data['y'],
                mode='markers',
                marker=dict(
                    color='red',
                    size=5,
                    opacity=0.5
                ),
                name='Fixations'
            ),
            row=1, col=1
        )
        
        # Try to get fixation data if available
        try:
            fixations = self.identify_fixations(participant_id)
            
            # Add a scatter plot of fixations with duration as size
            fig.add_trace(
                go.Scatter(
                    x=fixations['mean_x'],
                    y=fixations['mean_y'],
                    mode='markers',
                    marker=dict(
                        color='orange',
                        size=fixations['duration'] / 50,  # Scale size by duration
                        opacity=0.6,
                        line=dict(color='black', width=1)
                    ),
                    text=fixations['duration'].apply(lambda x: f"Duration: {x:.0f}ms"),
                    hoverinfo='text',
                    name='Fixation Duration'
                ),
                row=1, col=2
            )
        except:
            # If fixation identification fails, just show a message
            fig.add_annotation(
                text="No fixation data available",
                xref="x2 domain", yref="y2 domain",
                x=0.5, y=0.5,
                showarrow=False,
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Interactive Gaze Visualization for Participant {participant_id}",
            height=600,
            width=1200,
            showlegend=True
        )
        
        # Update x and y axes to be equal for proper representation
        fig.update_xaxes(title_text="X Coordinate", row=1, col=1)
        fig.update_yaxes(title_text="Y Coordinate", row=1, col=1)
        fig.update_xaxes(title_text="X Coordinate", row=1, col=2)
        fig.update_yaxes(title_text="Y Coordinate", row=1, col=2)
        
        # Make sure both subplots have the same scale
        if len(data) > 0:
            x_range = [data['x'].min(), data['x'].max()]
            y_range = [data['y'].min(), data['y'].max()]
            
            fig.update_xaxes(range=x_range, row=1, col=1)
            fig.update_yaxes(range=y_range, row=1, col=1)
            fig.update_xaxes(range=x_range, row=1, col=2)
            fig.update_yaxes(range=y_range, row=1, col=2)
        
        # Save to HTML if output file provided
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"{participant_id}_interactive.html")
        
        fig.write_html(output_file)
        print(f"Interactive visualization saved to {output_file}")
        
        return fig
    
    def generate_report(self, participant_id):
        """
        Generate a comprehensive report with all analyses.
        
        Parameters:
        -----------
        participant_id : str
            Participant ID to generate the report for
            
        Returns:
        --------
        str
            Path to the generated report
        """
        if participant_id not in self.processed_data:
            self.preprocess_data(participant_id)
            
        # Make sure we have metrics
        if participant_id not in self.metrics:
            self.calculate_metrics(participant_id)
        
        # Generate all visualizations
        self.plot_gaze_path(participant_id, show=False)
        self.plot_heatmap(participant_id, show=False)
        
        # Only generate interactive visualization if Plotly is available
        has_interactive = False
        if PLOTLY_AVAILABLE:
            self.generate_interactive_visualization(participant_id)
            has_interactive = True
        
        # Create report filename
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"{participant_id}_report_{now}.md")
        
        # Write report
        with open(report_file, 'w') as f:
            f.write(f"# Gaze Data Analysis Report: Participant {participant_id}\n\n")
            f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            # Summary section
            f.write("## Summary\n\n")
            for key, value in self.metrics[participant_id].items():
                if isinstance(value, float):
                    f.write(f"- **{key}**: {value:.2f}\n")
                else:
                    f.write(f"- **{key}**: {value}\n")
            
            # Visualizations section
            f.write("\n## Visualizations\n\n")
            f.write("### Gaze Path\n\n")
            f.write(f"![Gaze Path]({participant_id}_gaze_path.png)\n\n")
            f.write("### Heatmap\n\n")
            f.write(f"![Heatmap]({participant_id}_heatmap.png)\n\n")
            
            # Only include interactive visualization link if available
            if has_interactive:
                f.write("### Interactive Visualization\n\n")
                f.write(f"[Open Interactive Visualization]({participant_id}_interactive.html)\n\n")
            else:
                f.write("### Interactive Visualization\n\n")
                f.write("*Interactive visualization not available (Plotly package not fully installed)*\n\n")
            
        print(f"Report generated at {report_file}")
        return report_file


def main():
    """Main function to demonstrate the Gaze Data Analyzer"""
    # Create analyzer instance
    analyzer = GazeDataAnalyzer()
    
    # Find all data files
    data_files = []
    for root, _, files in os.walk(analyzer.raw_data_dir):
        for file in files:
            if file.endswith('.csv'):
                data_files.append(os.path.join(root, file))
    
    # Check if we found any data files
    if not data_files:
        print("No data files found. Using fallback data.")
        analyzer.load_fallback_data()
        participant_ids = list(analyzer.raw_data.keys())
    else:
        # Load each data file
        participant_ids = []
        for file_path in data_files:
            try:
                # Try to extract participant ID from filename
                filename = os.path.basename(file_path)
                participant_id = filename.split('_')[0]
                
                analyzer.load_experimental_data(file_path, participant_id)
                participant_ids.append(participant_id)
                print(f"Loaded data for participant {participant_id}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Process each participant
    for participant_id in tqdm(participant_ids, desc="Processing participants"):
        try:
            # Preprocess data
            analyzer.preprocess_data(participant_id)
            
            # Analyze fixations
            analyzer.identify_fixations(participant_id)
            
            # Calculate metrics
            analyzer.calculate_metrics(participant_id)
            
            # Generate visualizations
            analyzer.plot_gaze_path(participant_id, show=False)
            analyzer.plot_heatmap(participant_id, show=False)
            
            # Only attempt interactive visualization if Plotly is available
            if PLOTLY_AVAILABLE:
                analyzer.generate_interactive_visualization(participant_id)
            
            # Generate report
            analyzer.generate_report(participant_id)
            
            print(f"Analysis complete for participant {participant_id}")
        except Exception as e:
            print(f"Error processing participant {participant_id}: {e}")
    
    print("Analysis pipeline completed successfully!")


if __name__ == "__main__":
    main() 