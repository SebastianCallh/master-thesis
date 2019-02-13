"""This module contains useful plotting functionality.
"""
import matplotlib.pyplot as plt

def plot_grid(n_rows, n_cols):
    fig_size = 8
    return plt.subplots(
        nrows=n_rows, 
        ncols=n_cols, 
        figsize=(
            fig_size*n_cols, 
            fig_size*n_rows
        )
    )
    
