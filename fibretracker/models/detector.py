
import scipy
import numpy as np 
import skimage

from typing import Optional

def gauss_filter(
        std: float
        ):
    ''' Generate a 1D Gaussian filter and its derivatives

    Args: 
        std: standard deviation of the Gaussian filter

    Returns:
        g (np.ndarray): 1D Gaussian filter
        dg (np.ndarray): derivative of the Gaussian filter
        ddg (np.ndarray): second derivative of the Gaussian filter

    Example:
        ```python
        import fibretracker as ft

        vol = ft.detector.gauss_filter(std=2.5)
        ```
        
    '''
    x = np.arange(-np.ceil(5*std), np.ceil(5*std) + 1)[:,None]
    g = np.exp(-x**2/(2*std**2))
    g /= np.sum(g)
    dg = -x/std**2 * g
    ddg = -g/std**2 -x/std**2 * dg
    return g, dg, ddg

def blob_centre_detector(
        im: np.ndarray, 
        std: float=2.5, 
        min_distance: int=3, 
        threshold_abs: float=0.4
        ):
    ''' Predict coordinates of fibres centre in a volume slice using blob detector

    Args: 
        im: input image
        std: standard deviation of the Gaussian filter
        min_distance: minimum distance between peaks
        threshold_abs: threshold value for the peak from the background

    Returns: 
        pred_coords (np.ndarray): predicted coordinates of the fibre centre

    Example:
        ```python
        import fibretracker as ft

        vol = ft.detector.blob_centre_detector(im, std=2.5, min_distance=3, threshold_abs=0.4)
        ```

    '''
    g = gauss_filter(std)[0]
    im_g = scipy.ndimage.convolve(scipy.ndimage.convolve(im, g), g.T)
    pred_coords = skimage.feature.peak_local_max(im_g, min_distance=min_distance, threshold_abs=threshold_abs)
    return pred_coords
    
def avg_fibre_coord(
        pred_coord: np.ndarray, 
        im: np.ndarray, 
        window_size: int = 10,
        apply_filter: bool = False,
        std: Optional[float] = None
        ):
    ''' Recompute the fibre centre in a slice using weighted average of peak neighbourhood

    Args: 
        pred_coord: predicted coordinates of the peaks
        im: input image
        window_size: size of the neighbourhood window around the peak
        apply_filter: whether to apply Gaussian filter to the window
        std: standard deviation of the Gaussian filter

    Returns:
        coords (np.ndarray): recomputed fibre centre coordinates in the slice with weighted average
    
    Example:
        ```python
        import fibretracker as ft

        avg_coord = ft.detector.avg_fib_coord(pred_coord, im, window_size)
        ```

    '''
    coords = []
    for coord in pred_coord:
        x, y = coord
        window = im[x-window_size//2:x+window_size//2+1, y-window_size//2:y+window_size//2+1]
        # Apply Gaussian filter to the window
        if apply_filter:
            if std is not None:
                g = gauss_filter(std)[0]
            else:
                g = gauss_filter(std=2.5)[0]
            window = scipy.ndimage.convolve(scipy.ndimage.convolve(window, g), g.T)
        x_coords, y_coords = np.meshgrid(range(x-window_size//2, x+window_size//2+1), range(y-window_size//2, y+window_size//2+1))
        weighted_x = np.sum(window * x_coords) / np.sum(window)
        weighted_y = np.sum(window * y_coords) / np.sum(window)
        coords.append([weighted_x, weighted_y])
    return np.array(coords)

def get_fibre_coords(
        vol: np.ndarray, 
        std: float=2.5, 
        min_distance: int=3, 
        threshold_abs: float=0.4,
        weighted_avg: bool=False,
        window_size: int=10,
        apply_filter: bool=False,
        ):
    ''' Get list of fibres centre coordinates in a volume using blob detector

    Args:
        vol: input volume
        std: standard deviation of the Gaussian filter
        min_distance: minimum distance between fibres
        threshold_abs: threshold value for the peak from the background
        weighted_avg: whether to apply weighted average to the detected coordinates
        window_size: size of the neighbourhood window around the peak
        apply_filter: whether to apply Gaussian filter to the window

    Returns:
        coords (List(nd.array)): List of fibres centre coordinates in the volume
    
    Example:
        ```python
        import fibretracker as ft

        vol = ft.detector.get_fib_coords(vol, std=2.5, min_distance=3, threshold_abs=0.4)
        ```

    '''
    coords = []
    for i, im in enumerate(vol):
        coord = blob_centre_detector(im, std=std, min_distance=min_distance, threshold_abs=threshold_abs)
        if weighted_avg:
            coord = avg_fibre_coord(coord, im, window_size=window_size, apply_filter=apply_filter, std=std)
        coords.append(np.stack([coord[:,1], coord[:,0], np.ones(len(coord)) * i], axis=1))
        print(f'Detecting coordinates - slice: {i+1}/{len(vol)}', end='\r')
    print(' ' * len(f'Detecting coordinates - slice: {i+1}/{len(vol)}'), end='\r')
    return coords