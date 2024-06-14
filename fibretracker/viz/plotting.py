''' 
This module contains functions for plotting the tracks of fibres detected in the volume
'''
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive
import ipywidgets as widgets


def plot_tracks(
        tracks: List[np.ndarray],
        grid: bool = False,):
    
    '''Plot tracks of fibres detected in the volume

    Args:
        tracks: List of arrays of shape (n_points, 3)
        grid: Whether to show grid in the plot
    
    Returns:
        fig (matplotlib.figure.Figure): matplotlib figure object
    
    Example:
        ```python
        import fibretracker as ft

        # Load the volume and detected coordinates
        vol = ft.io.load("path/to/volume.txm")
        vol = ft.io.normalize(vol)
        vol = vol[100:350] # 250 slices along the z-axis
        detect_coords = ft.models.get_fibre_coords(vol)
        tracks_gauss = ft.models.track_fibres(coords=detect_coords, smoothtrack_gaussian=True)
        ft.viz.plot_tracks(tracks_gauss)
        ```

        ![viz tracks](figures/tracks_gauss.gif)


    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    for track in tracks:
        ax.plot(track[:,0], track[:,1], track[:,2])
    ax.grid(grid)
    ax.set_aspect('equal')
    
    plt.show()
    return fig