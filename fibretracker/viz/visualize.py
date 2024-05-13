""" 
Provides a visualization functions for detected fibre centre on slices or 3D volume slices.

#Adapted from: qim3d.viz - https://platform.qim.dk/qim3d/

"""

import math
from typing import List, Optional, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np


def slices(
    vol: np.ndarray,
    axis: int = 0,
    position: Optional[Union[str, int, List[int]]] = None,
    detect_coords: Optional[List[np.ndarray]] = None,
    mark_size: Optional[int] = None,
    n_slices: int = 5,
    max_cols: int = 5,
    cmap: str = "gray",
    img_height: int = 2,
    img_width: int = 2,
    show: bool = False,
    show_position: bool = True,
    interpolation: Optional[str] = "none",
    **imshow_kwargs,
):
    """Displays slices of a 3D volume along a specified axis and detected fibres centre if provided.

    By default if `position` is None, slices plots `n_slices` linearly spaced slices.
    If `position` is given as a string or integer, slices will plot an overview with `n_slices` figures around that position.
    If `position` is given as a list, `n_slices` will be ignored and the slices from `position` will be plotted.

    Args:
        vol (np.ndarray): The 3D volume to be sliced.
        axis (int, optional): Specifies the axis, or dimension, along which to slice. Defaults to 0.
        position (str, int, list, optional): One or several slicing levels. If None, linearly spaced slices will be displayed. Defaults to None.
        detect_coords (list, optional): List of coordinates of detected fibres. Defaults to None.
        mark_size (int, optional): Size of the marker for detected fibres. Defaults to None.
        n_slices (int, optional): Defines how many slices the user wants to be displayed. Defaults to 5.
        max_cols (int, optional): The maximum number of columns to be plotted. Defaults to 5.
        cmap (str, optional): Specifies the color map for the image. Defaults to "viridis".
        img_height(int, optional): Height of the figure.
        img_width(int, optional): Width of the figure.
        show (bool, optional): If True, displays the plot (i.e. calls plt.show()). Defaults to False.
        show_position (bool, optional): If True, displays the position of the slices. Defaults to True.
        interpolation (str, optional): Specifies the interpolation method for the image. Defaults to None.

    Returns:
        fig (matplotlib.figure.Figure): The figure with the slices from the 3d array.

    Raises:
        ValueError: If the input is not a numpy.ndarray.
        ValueError: If the axis to slice along is not a valid choice, i.e. not an integer between 0 and the number of dimensions of the volume minus 1.
        ValueError: If the file or array is not a volume with at least 3 dimensions.
        ValueError: If the `position` keyword argument is not a integer, list of integers or one of the following strings: "start", "mid" or "end".

    Example:
    
    """

    # Numpy array or Torch tensor input
    if not isinstance(vol, np.ndarray):
        raise ValueError("Input must be a numpy.ndarray")

    if vol.ndim < 3:
        raise ValueError(
            "The provided object is not a volume as it has less than 3 dimensions."
        )

    # Ensure axis is a valid choice
    if not (0 <= axis < vol.ndim):
        raise ValueError(
            f"Invalid value for 'axis'. It should be an integer between 0 and {vol.ndim - 1}."
        )

    # Get total number of slices in the specified dimension
    n_total = vol.shape[axis]

    # Position is not provided - will use linearly spaced slices
    if position is None:
        slice_idxs = np.linspace(0, n_total - 1, n_slices, dtype=int)
    # Position is a string
    elif isinstance(position, str) and position.lower() in ["start", "mid", "end"]:
        if position.lower() == "start":
            slice_idxs = _get_slice_range(0, n_slices, n_total)
        elif position.lower() == "mid":
            slice_idxs = _get_slice_range(n_total // 2, n_slices, n_total)
        elif position.lower() == "end":
            slice_idxs = _get_slice_range(n_total - 1, n_slices, n_total)
    #  Position is an integer
    elif isinstance(position, int):
        slice_idxs = _get_slice_range(position, n_slices, n_total)
    # Position is a list of integers
    elif isinstance(position, list) and all(isinstance(idx, int) for idx in position):
        slice_idxs = position
    else:
        raise ValueError(
            'Position not recognized. Choose an integer, list of integers or one of the following strings: "start", "mid" or "end".'
        )

    # Make grid
    nrows = math.ceil(n_slices / max_cols)
    ncols = min(n_slices, max_cols)

    # Generate figure
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * img_height, nrows * img_width),
        constrained_layout=True,
    )
    if nrows == 1:
        axs = [axs]  # Convert to a list for uniformity

    # Run through each ax of the grid
    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(np.atleast_1d(ax_row)):
            slice_idx = i * max_cols + j
            try:
                slice_img = vol.take(slice_idxs[slice_idx], axis=axis)
                if detect_coords is None:
                    ax.imshow(
                        slice_img, cmap=cmap, interpolation=interpolation, **imshow_kwargs
                    )
                elif detect_coords is not None:
                    id_coord = slice_idxs[slice_idx]
                    ax.imshow(
                        slice_img, cmap=cmap, interpolation=interpolation, **imshow_kwargs
                    )
                    if mark_size is not None:
                        ax.plot(detect_coords[id_coord][:, 0], detect_coords[id_coord][:, 1], 'rx', markersize=mark_size)
                    else:
                        ax.plot(detect_coords[id_coord][:, 0], detect_coords[id_coord][:, 1], 'rx')

                if show_position:
                    ax.text(
                        0.0,
                        1.0,
                        f"slice {slice_idxs[slice_idx]} ",
                        transform=ax.transAxes,
                        color="white",
                        fontsize=8,
                        va="top",
                        ha="left",
                        bbox=dict(facecolor="#303030", linewidth=0, pad=0),
                    )

                    ax.text(
                        1.0,
                        0.0,
                        f"axis {axis} ",
                        transform=ax.transAxes,
                        color="white",
                        fontsize=8,
                        va="bottom",
                        ha="right",
                        bbox=dict(facecolor="#303030", linewidth=0, pad=0),
                    )

            except IndexError:
                # Not a problem, because we simply do not have a slice to show
                pass

            # Hide the axis, so that we have a nice grid
            ax.axis("off")

    if show:
        plt.show()

    plt.close()

    return fig


def _get_slice_range(position: int, n_slices: int, n_total):
    """Helper function for `slices`. Returns the range of slices to be displayed around the given position."""
    start_idx = position - n_slices // 2
    end_idx = (
        position + n_slices // 2 if n_slices % 2 == 0 else position + n_slices // 2 + 1
    )
    slice_idxs = np.arange(start_idx, end_idx)

    if slice_idxs[0] < 0:
        slice_idxs = np.arange(0, n_slices)
    elif slice_idxs[-1] > n_total:
        slice_idxs = np.arange(n_total - n_slices, n_total)

    return slice_idxs


def slicer(
    vol: np.ndarray,
    detect_coords: Optional[List[np.ndarray]] = None,
    mark_size: Optional[int] = None,
    axis: int = 0,
    cmap: str = "gray",
    img_height: int = 3,
    img_width: int = 3,
    show_position: bool = False,
    interpolation: Optional[str] = "none",
    **imshow_kwargs,
) -> widgets.interactive:
    """Interactive widget for visualizing slices of a 3D volume and fibres centre if provided.

    Args:
        vol (np.ndarray): The 3D volume to be sliced.
        detect_coords (list, optional): List of coordinates of detected fibres. Defaults to None.
        mark_size (int, optional): Size of the marker for detected fibres. Defaults to None.
        axis (int, optional): Specifies the axis, or dimension, along which to slice. Defaults to 0.
        cmap (str, optional): Specifies the color map for the image. Defaults to "viridis".
        img_height (int, optional): Height of the figure. Defaults to 3.
        img_width (int, optional): Width of the figure. Defaults to 3.
        show_position (bool, optional): If True, displays the position of the slices. Defaults to False.
        interpolation (str, optional): Specifies the interpolation method for the image. Defaults to None.

    Returns:
        slicer_obj (widgets.interactive): The interactive widget for visualizing slices of a 3D volume.

    Example:
        
    """

    # Create the interactive widget
    def _slicer(position):
        fig = slices(
            vol,
            detect_coords=detect_coords,
            mark_size=mark_size,
            axis=axis,
            cmap=cmap,
            img_height=img_height,
            img_width=img_width,
            show_position=show_position,
            interpolation=interpolation,
            position=position,
            n_slices=1,
            show=True,
            **imshow_kwargs,
        )
        return fig

    position_slider = widgets.IntSlider(
        value=vol.shape[axis] // 2,
        min=0,
        max=vol.shape[axis] - 1,
        description="Slice",
        continuous_update=True,
    )
    slicer_obj = widgets.interactive(_slicer, position=position_slider)
    slicer_obj.layout = widgets.Layout(align_items="flex-start")

    return slicer_obj

def orthogonal(
    vol: np.ndarray,
    cmap: str = "viridis",
    img_height: int = 3,
    img_width: int = 3,
    show_position: bool = False,
    interpolation: Optional[str] = None,
):
    """Interactive widget for visualizing orthogonal slices of a 3D volume.

    Args:
        vol (np.ndarray or torch.Tensor): The 3D volume to be sliced.
        cmap (str, optional): Specifies the color map for the image. Defaults to "viridis".
        img_height(int, optional): Height of the figure.
        img_width(int, optional): Width of the figure.
        show_position (bool, optional): If True, displays the position of the slices. Defaults to False.
        interpolation (str, optional): Specifies the interpolation method for the image. Defaults to None.

    Returns:
        orthogonal_obj (widgets.HBox): The interactive widget for visualizing orthogonal slices of a 3D volume.

    Example:
        ```python
        import qim3d

        vol = qim3d.examples.fly_150x256x256
        qim3d.viz.orthogonal(vol, cmap="magma")
        ```
        ![viz orthogonal](assets/screenshots/viz-orthogonal.gif)
    """

    z_slicer = slicer(
        vol,
        axis=0,
        cmap=cmap,
        img_height=img_height,
        img_width=img_width,
        show_position=show_position,
        interpolation=interpolation,
    )
    y_slicer = slicer(
        vol,
        axis=1,
        cmap=cmap,
        img_height=img_height,
        img_width=img_width,
        show_position=show_position,
        interpolation=interpolation,
    )
    x_slicer = slicer(
        vol,
        axis=2,
        cmap=cmap,
        img_height=img_height,
        img_width=img_width,
        show_position=show_position,
        interpolation=interpolation,
    )

    z_slicer.children[0].description = "Z"
    y_slicer.children[0].description = "Y"
    x_slicer.children[0].description = "X"

    return widgets.HBox([z_slicer, y_slicer, x_slicer])
