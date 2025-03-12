#!/usr/bin/env python3
# Test script to verify plotly imports
try:
    import plotly
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    print(f"Plotly version: {plotly.__version__}")
    print("All plotly imports successful!")
except ImportError as e:
    print(f"Import error: {e}") 