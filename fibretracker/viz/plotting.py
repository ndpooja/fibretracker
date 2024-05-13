''' 
This module contains functions for plotting the tracks of fibres detected in the volume
'''
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_tracks(
        tracks: List[np.ndarray]):
    
    '''
    Plot tracks of fibres detected in the volume

    Args:
    tracks: List of np.ndarrays of shape (n_points, 3)
    show: whether to display the plot, default is True
    kwargs: additional arguments to pass to the plot function

    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    for track in tracks:
        ax.plot(track[:,0], track[:,1], track[:,2])
    ax.set_aspect('equal')
    
    plt.show()
