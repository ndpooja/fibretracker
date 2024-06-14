""" 
Provides a visualization functions for detected fibre centre on slices or 3D volume slices.

"""

from typing import List, Optional

import ipywidgets as widgets
from ipywidgets import interactive
import matplotlib.pyplot as plt
import numpy as np

def slicer(
    vol: np.ndarray,
    detect_coords: Optional[List[np.ndarray]] = None,
    mark_size: Optional[int] = None,
    axis: int = 0,
    cmap: str = "gray",
    img_height: int = 5,
    img_width: int = 5, 
) -> widgets.interactive:
    """Interactive widget for visualizing slices of a 3D volume and fibres centre if provided.

    Args:
        vol (np.ndarray): The 3D volume to be sliced.
        detect_coords (list, optional): List of coordinates of detected fibres. Defaults to None.
        mark_size (int, optional): Size of the marker for detected fibres. Defaults to None.
        axis (int, optional): Specifies the axis, or dimension, along which to slice. Defaults to 0.
        cmap (str, optional): Specifies the color map for the image. Defaults to "gray".
        img_height (int, optional): Height of the figure. Defaults to 5.
        img_width (int, optional): Width of the figure. Defaults to 5.

    Returns:
        slicer_obj (widgets.interactive): The interactive widget for visualizing slices of a 3D volume.

    Example:
        ```python
        import fibretracker as ft
        
        # Load the volume and visualize the slices
        vol = ft.io.load("path/to/volume.txm")
        ft.viz.slicer(vol)
        ```
        ![viz slicer](figures/viz-slicer.gif)

        ```python
        import fibretracker as ft

        # Load the volume and detected coordinates
        vol = ft.io.load("path/to/volume.txm")
        vol = ft.io.normalize(vol)
        vol = vol[100:350] # 250 slices along the z-axis
        detect_coords = ft.models.get_fibre_coords(vol)
        ft.viz.slicer(vol, detect_coords=detect_coords, mark_size=4)
        ```

        ![viz slicer](figures/viz-slicer_detector.gif)
        
    """

    if detect_coords is None:
        fig, ax = plt.subplots(figsize=(img_width, img_height))
        ax.axis("off")
    else:
        fig, ax = plt.subplots(1, 2, figsize=(2*img_width, img_height), sharex=True, sharey=True, gridspec_kw={'wspace': 0})
        ax[0].axis("off")
        ax[1].axis("off")
    
    def _slice(slice_idx):
        slice_img = vol.take(slice_idx, axis=axis)
        if detect_coords is not None:
            [l.remove() for l in ax[1].lines]
            ax[0].imshow(slice_img, cmap=cmap)
            ax[1].imshow(slice_img, cmap=cmap)
            if mark_size is not None:
                ax[1].plot(detect_coords[slice_idx][:, 0], detect_coords[slice_idx][:, 1], 'rx', markersize=mark_size)
            else:
                ax[1].plot(detect_coords[slice_idx][:, 0], detect_coords[slice_idx][:, 1], 'rx', markersize=3)
        else:
            ax.imshow(slice_img, cmap=cmap)

    def on_release(event):
        if detect_coords is not None:
            xlim = ax[0].get_xlim()
            ylim = ax[0].get_ylim()
            ax[1].set_xlim(xlim)
            ax[1].set_ylim(ylim)
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

    slice_slider = widgets.IntSlider(
        value=vol.shape[axis] // 2,
        min=0,
        max=vol.shape[axis] - 1,
        description="Slice",
        continuous_update=True,
    )

    slice_slider.style.description_width = 'middle'  # Set the width of the description to fit the larger font size
    slice_slider.style.handle_color = 'blue'  # Optional: Change the handle color
    slice_slider.style.font_size = '150px'  
    
    fig.canvas.mpl_connect('button_release_event', on_release)
    slicer_obj = interactive(_slice, slice_idx = slice_slider)

    return slicer_obj

def orthogonal(
    vol: np.ndarray,
    cmap: str = "gray",
    img_height: int = 5,
    img_width: int = 5
)-> widgets.HBox:
    """Interactive widget for visualizing orthogonal slices of a 3D volume.

    Args:
        vol (np.ndarray or torch.Tensor): The 3D volume to be sliced.
        cmap (str, optional): Specifies the color map for the image. Defaults to "gray".
        img_height(int, optional): Height of the figure.
        img_width(int, optional): Width of the figure.

    Returns:
        orthogonal_obj (widgets.HBox): The interactive widget for visualizing orthogonal slices of a 3D volume.

    Example:
        ```python
        import fibretracker as ft

        vol = ft.io.load("path/to/volume.txm")
        ft.viz.orthogonal(vol, cmap="gray")
        ```
        ![viz orthogonal](figures/viz-orthogonal.gif)
    """

    fig, ax = plt.subplots(1, 3, figsize=(3*img_width, img_height))
    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    
    def _slice(slice_idx_z, slice_idx_y, slice_idx_x):
        slice_img_z = vol.take(slice_idx_z, axis=0)
        slice_img_y = vol.take(slice_idx_y, axis=1)
        slice_img_x = vol.take(slice_idx_x, axis=2)

        ax[0].imshow(slice_img_z, cmap=cmap)
        ax[0].set_aspect('equal')
        ax[1].imshow(slice_img_y, cmap=cmap)
        ax[1].set_aspect('equal')
        ax[2].imshow(slice_img_x, cmap=cmap)
        ax[2].set_aspect('equal')

    slice_slider_z = widgets.IntSlider(
        value=vol.shape[0] // 2,
        min=0,
        max=vol.shape[0] - 1,
        description="Z",
        continuous_update=True,
    )

    slice_slider_y = widgets.IntSlider(
        value=vol.shape[1] // 2,
        min=0,
        max=vol.shape[1] - 1,
        description="Y",
        continuous_update=True,
    )

    slice_slider_x = widgets.IntSlider(
        value=vol.shape[2] // 2,
        min=0,
        max=vol.shape[2] - 1,
        description="X",
        continuous_update=True,
    )

    slicer_obj = interactive(_slice, slice_idx_z=slice_slider_z, slice_idx_y=slice_slider_y, slice_idx_x=slice_slider_x)

    # Create a horizontal box for the sliders
    hbox = widgets.HBox([slicer_obj.children[0], slicer_obj.children[1], slicer_obj.children[2]], layout=widgets.Layout(align_items="stretch", justify_content="center", align_content="center", justify_items="center", justify_self="center", align_self="center", width="100%"))

    # Replace the sliders in the interactive widget with the horizontal box
    slicer_obj.children = (hbox,) + slicer_obj.children[3:]

    return slicer_obj