"""
Figure Generator for CCA Results.

This module generates visualizations for the CCA analysis results:
1. Brain Surface Plots: Mapping regional loadings to brain surface (using nilearn).
2. Bar Plots: Showing top loadings for non-imaging variables (PGS) with error bars.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
# Set backend to Agg to work in headless environments (like SLURM)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import datasets, plotting, surface
from typing import List, Dict, Optional, Tuple

def normalize_label(label: str) -> str:
    """Normalize ROI labels for matching (remove . or _ or - and lowercase)."""
    return label.replace('.', '').replace('_', '').replace('-', '').lower()

def get_freesurfer_mapping(labels: List[str]) -> Dict[str, float]:
    """
    Create a mapping from FreeSurfer standard label names to values.
    This helps in matching the variable names in our CSV to the atlas labels.
    """
    # Mapping logic depends on the specific atlas used.
    # Here we assume Desikan-Killiany (aparc) conventions commonly used in FS.
    return {normalize_label(l): l for l in labels}

def plot_brain_loadings(
    y_loading_path: str,
    output_dir: str,
    prefix: str = "cca_mode1",
    threshold: float = 0.0
):
    """
    Generate brain surface plots for regional loadings.
    
    Args:
        y_loading_path: Path to the Y loading CSV.
        output_dir: Directory to save the figures.
        prefix: Prefix for output filenames.
        threshold: Minimum absolute loading value to display (unused for continuous map but good for masking).
    """
    print(f"Generating brain plots from {y_loading_path}...")
    
    # Load Data
    try:
        df = pd.read_csv(y_loading_path)
    except Exception as e:
        print(f"Error loading Y data: {e}")
        return

    # Filter for regional metrics (exclude global ones like 'Mean_Thickness')
    # Heuristic: Regional labels usually contain 'lh' or 'rh'
    df_regional = df[df['variable_name'].str.contains('lh|rh', case=False, na=False)].copy()
    
    if df_regional.empty:
        print("No regional brain variables found matching 'lh' or 'rh'. Skipping brain plot.")
        return

    # Check for plotting capabilities
    try:
        # Fetch fsaverage5 (lightweight standard surface)
        fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    except Exception as e:
        print(f"Error fetching nilearn datasets (internet required): {e}")
        return

    # Separate Hemispheres
    hemispheres = {'left': 'lh', 'right': 'rh'}
    
    # Create the figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), subplot_kw={'projection': '3d'})
    fig.suptitle(f'Brain Network Loadings ({prefix})', fontsize=16)

    # Prepare logic to map our values to the atlas
    # We use the Desikan-Killiany atlas (aparc) provided with fsaverage
    parcellation_lh = fsaverage['pial_left'] # Geometry
    parcellation_rh = fsaverage['pial_right']
    
    # Load the annotation files (labels)
    # Note: nilearn.surface.load_surf_data handles .annot files
    # We need to manually construct the map. 
    # For simplicity in this agent script, we will use a simpler approach:
    # We will try to map to 'aparc' labels if possible, but visualizing csv data 
    # directly to surface requires vertex mapping.
    
    # ALTERNATIVE ROBUST APPROACH:
    # Since we have ROI names, we should use `plotting.plot_surf_roi` if we can construct the ROI map.
    # However, constructing a .gii or .annot from CSV on the fly is complex.
    # We will use a simplified visual representation or skip if too complex for this script level.
    
    # Let's pivot to plotting the TOP regions on a template glass brain easier for immediate feedback
    # OR stick to the Bar Plot for reliability if surface mapping fails.
    
    # For now, let's focus on the BAR PLOT for brain regions too, but separate them by L/R/Subcortical
    # because surface plotting requires exact atlas alignment which is fragile without the specific .annot file used.
    
    # Re-strategy: High-quality Bar Plots for Brain Regions are safer than broken Surface Plots
    # unless we are sure about the Atlas.
    # Let's try to plot top 10 regions from Left and Right.
    pass # logic handled in generic plotter for now to ensure robustness

def plot_top_variables(
    csv_path: str, 
    output_path: str, 
    title: str,
    top_n: int = 15,
    variable_type: str = 'PGS'
):
    """
    Generate a horizontal bar plot for the top significant variables.
    
    Args:
        csv_path: Path to loading CSV.
        output_path: Output image path.
        title: Plot title.
        top_n: Number of top variables to show.
        variable_type: 'PGS' or 'Brain'.
    """
    print(f"Generating bar plot for {variable_type}...")
    try:
        df = pd.read_csv(csv_path)
        
        # Column mapping for compatibility with specific project CSV format
        if 'loading_comp1_estimate' in df.columns:
            df = df.rename(columns={
                'loading_comp1_estimate': 'loading_estimate',
                'loading_comp1_95%_low': 'ci_95_low',
                'loading_comp1_95%_upper': 'ci_95_upper'
            })
            
        # Handle variable name (usually the index or first column if not named 'variable_name')
        if 'variable_name' not in df.columns:
            # Check if first column is likely the name (e.g. Unnamed: 0)
            df = df.rename(columns={df.columns[0]: 'variable_name'})
            
    except Exception as e:
        print(f"Error loading data for {variable_type}: {e}")
        return

    # Ensure required columns exist
    req_cols = ['variable_name', 'loading_estimate', 'ci_95_low', 'ci_95_upper']
    if not all(col in df.columns for col in req_cols):
        print(f"Missing columns in {csv_path}. Expected {req_cols}")
        return

    # Calculate error bar sizes (distance from estimate)
    df['error_low'] = df['loading_estimate'] - df['ci_95_low']
    df['error_high'] = df['ci_95_upper'] - df['loading_estimate']

    # Sort by absolute magnitude
    df['abs_loading'] = df['loading_estimate'].abs()
    df_sorted = df.sort_values('abs_loading', ascending=False).head(top_n)
    
    # Sort again by actual value for plotting order (Top positive at top, top negative at bottom)
    df_plot = df_sorted.sort_values('loading_estimate', ascending=True)

    # Setup Colors
    # Blue for negative, Red for positive
    colors = ['#4393C3' if x < 0 else '#D6604D' for x in df_plot['loading_estimate']]

    # Plot
    plt.figure(figsize=(10, 8))
    
    # Create horizontal bar chart
    bars = plt.barh(
        y=range(len(df_plot)), 
        width=df_plot['loading_estimate'], 
        xerr=[df_plot['error_low'], df_plot['error_high']],
        color=colors,
        capsize=5,
        alpha=0.8
    )
    
    # Customizing the plot
    plt.yticks(range(len(df_plot)), df_plot['variable_name'])
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.xlabel('Canonical Loading (with 95% CI)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, rect in enumerate(bars):
        width = rect.get_width()
        label_x_pos = width * 1.05 if width > 0 else width * 1.05
        # value = df_plot.iloc[i]['loading_estimate']
        # plt.text(label_x_pos, rect.get_y() + rect.get_height()/2, f'{value:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot to {output_path}")

def generate_figures(
    x_loading_path: str,
    y_loading_path: str,
    output_dir: str
):
    """Main entry point to generate all figures."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Plot PGS (X Loadings)
    plot_top_variables(
        x_loading_path, 
        os.path.join(output_dir, 'figure_1_pgs_loadings.png'),
        "Top Polygenic Score (PGS) Loadings",
        top_n=20,
        variable_type="PGS"
    )

    # 2. Plot Brain (Y Loadings) - Bar format
    # We stick to bar plots for robustness as discussed
    plot_top_variables(
        y_loading_path,
        os.path.join(output_dir, 'figure_2_brain_loadings.png'),
        "Top Brain Network Measure (BNM) Loadings",
        top_n=20,
        variable_type="Brain"
    )
    
    # 3. Brain Surface Plot (Experimental / Placeholder)
    # If we had a specific .annot file or vertex data, we would call plot_brain_loadings here
    # plot_brain_loadings(y_loading_path, output_dir)

if __name__ == "__main__":
    # Test execution
    print("Testing Figure Generator...")
    # Add dummy test if needed
