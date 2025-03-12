#!/usr/bin/env python3
# Minimal test to verify import of gaze_analysis.py

# This should succeed regardless of whether dependencies are installed
try:
    print("Attempting to import from gaze_analysis...")
    from gaze_analysis import (
        NUMPY_AVAILABLE, 
        PANDAS_AVAILABLE, 
        MATPLOTLIB_AVAILABLE,
        PLOTLY_AVAILABLE, 
        SCIENTIFIC_PACKAGES_AVAILABLE,
        TQDM_AVAILABLE
    )
    
    # Report on available functionality
    print("\nDependency Status:")
    print(f"NumPy available: {NUMPY_AVAILABLE}")
    print(f"Pandas available: {PANDAS_AVAILABLE}")
    print(f"Matplotlib available: {MATPLOTLIB_AVAILABLE}")
    print(f"Plotly available: {PLOTLY_AVAILABLE}")
    print(f"Scientific packages available: {SCIENTIFIC_PACKAGES_AVAILABLE}")
    print(f"TQDM available: {TQDM_AVAILABLE}")
    
    print("\nModule imported successfully with graceful degradation!")
    print("\nTo fix all dependency issues, install the required packages:")
    print("  pip install -r requirements.txt")
    
except Exception as e:
    print(f"Error importing gaze_analysis: {e}") 