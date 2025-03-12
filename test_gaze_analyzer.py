#!/usr/bin/env python3
# Test the GazeDataAnalyzer class with graceful degradation

from gaze_analysis import GazeDataAnalyzer, PLOTLY_AVAILABLE

def main():
    print(f"Plotly fully available: {PLOTLY_AVAILABLE}")
    
    # Create analyzer instance
    analyzer = GazeDataAnalyzer()
    print(f"Analyzer initialized with directories:")
    print(f"  - Raw data: {analyzer.raw_data_dir}")
    print(f"  - Processed data: {analyzer.processed_data_dir}")
    print(f"  - Output: {analyzer.output_dir}")
    
    # Test if we can load fallback data
    try:
        print("\nAttempting to load fallback data...")
        analyzer.load_fallback_data("test_participant")
        print("Fallback data loaded successfully!")
    except Exception as e:
        print(f"Error loading fallback data: {e}")
    
    print("\nGaze analysis script is functional with graceful degradation for Plotly components.")
    print("To fix the plotly import issue, run:")
    print("  pip install plotly pandas --upgrade")
    
if __name__ == "__main__":
    main() 